//! Factor the unit token-margin gate VJP into transported site inputs and the factual readout adjoint.
use super::*;
use crate::geometry_artifact::{GeometryAdapterPlacement, GeometryAdapterRecipe, PrefixRotation};

/// Total numerical payload in the returned vectors, checked before evaluating a token.
pub const MAX_PREFIX_SIGNAL_VALUES: usize = 1 << 18;

#[derive(Debug, Serialize)]
pub struct CausalLmPrefixMarginSignal {
    pub schema: &'static str,
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub frozen_lora_artifact_id: Option<String>,
    pub geometry_artifact_id: String,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub site: String,
    pub rotation: PrefixRotation,
    pub strength: f32,
    pub shape: [usize; 2],
    pub contrast: CausalLmTokenContrast,
    /// Actual input to the residual branch, flattened in [position, channel] order.
    pub site_input: Vec<f32>,
    /// R^(last-position) x[position] / steps, in row-major order, before f32 mean rounding.
    pub source_features: Vec<f64>,
    /// Unit logit-margin derivative at the final site's output, before residual scaling.
    pub upstream: Vec<f32>,
    /// Factual unmasked parameter VJPs, even when a parameter was frozen.
    pub gate_gradient: Vec<f32>,
    pub bias_gradient: Vec<f32>,
}

fn payload(steps: usize, width: usize) -> CausalLmResult<()> {
    let rows = steps
        .checked_mul(2)
        .and_then(|x| x.checked_add(3))
        .ok_or("prefix signal size overflow")?;
    product(
        &[rows, width],
        MAX_PREFIX_SIGNAL_VALUES,
        "prefix signal return",
    )?;
    Ok(())
}

fn features(input: &Tensor, turn: PrefixRotation) -> Vec<f64> {
    let (steps, width) = input.shape();
    let mut rows: Vec<_> = input.data().iter().copied().map(f64::from).collect();
    for (i, row) in rows.chunks_exact_mut(width).enumerate() {
        // Quarter turns have order four. No prefix is scanned a second time.
        for _ in 0..((steps - 1 - i) % 4) {
            turn.apply(row);
        }
        for x in row {
            *x /= steps as f64;
        }
    }
    rows
}

impl NativeCausalLm {
    fn signal_site(&self) -> CausalLmResult<&GeometryAdapterPlacement> {
        let artifact = self
            .geometry_artifact()
            .ok_or("prefix signal needs geometry")?;
        let block = format!("transformer.h.{}", self.config.layers - 1);
        let placements = &artifact.record().placements;
        if placements.iter().any(|p| p.path == block) {
            return Err("prefix signal does not cross downstream block geometry".into());
        }
        let path = format!("{block}.mlp");
        placements
            .iter()
            .find(|p| {
                p.path == path && matches!(p.recipe, GeometryAdapterRecipe::PrefixTransport { .. })
            })
            .ok_or_else(|| "prefix signal needs prefix transport on the final MLP".into())
    }

    /// Validate candidates, return and activation budgets before clients copy the token IDs.
    pub fn prefix_margin_signal_input_count(
        &self,
        steps: usize,
        target: usize,
        rival: usize,
    ) -> CausalLmResult<usize> {
        self.next_token_contrast_input_count(steps, target, rival)?;
        let site = self.signal_site()?;
        let GeometryAdapterRecipe::PrefixTransport { max_values, .. } = site.recipe else {
            unreachable!()
        };
        product(
            &[steps, self.config.width],
            max_values,
            "prefix transport site",
        )?;
        payload(steps, self.config.width)?;
        Ok(steps)
    }

    /// One factual prefix; no optimizer step, suffix, objective scaling or derivatives of discrete IDs.
    pub fn last_mlp_prefix_margin_signal(
        &self,
        prompt: &[usize],
        target: usize,
        rival: usize,
    ) -> CausalLmResult<CausalLmPrefixMarginSignal> {
        let steps = self.prefix_margin_signal_input_count(prompt.len(), target, rival)?;
        let site = self.signal_site()?;
        let GeometryAdapterRecipe::PrefixTransport {
            rotation, strength, ..
        } = site.recipe
        else {
            unreachable!()
        };
        let shape = [1, steps];
        let layer = self.config.layers - 1;
        let hidden = self.forward_blocks(self.embed(prompt, shape)?, shape, 0..layer)?;
        let mut snapshot = None;
        let pass = decoder::BlockPass::CaptureGeometry {
            path: &site.path,
            snapshot: &mut snapshot,
        };
        let result = self.run_block(layer, hidden, shape, pass)?;
        let (input, batch) = snapshot.ok_or("missing factual prefix transport capture")?;
        let last = result.output.gather_rows(&[steps - 1])?;
        let logits =
            self.project_readout(&self.final_norm.forward(&last, self.config.norm_epsilon)?)?;
        let mut seed = vec![0.; self.config.vocabulary];
        seed[target] = 1.;
        seed[rival] = -1.;
        let head = self.vocabulary_head_pullback_as(
            &Tensor::from_vec(1, self.config.vocabulary, seed)?,
            self.readout_accumulator.backward_precision(),
        )?;
        let upstream = last.layer_norm_input_backward(
            &self.final_norm.gamma,
            &head,
            self.config.norm_epsilon,
        )?;
        if input
            .data()
            .iter()
            .chain(logits.data())
            .chain(upstream.data())
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite prefix signal".into());
        }
        let width = self.config.width;
        let mut dy = vec![0.; steps * width];
        dy[(steps - 1) * width..].copy_from_slice(upstream.data());
        let mut grads = batch.vjp_parameters(&dy)?;
        Ok(CausalLmPrefixMarginSignal {
            schema: "spiraltorch.last_mlp_prefix_margin_signal.v1",
            base_model_identity: self.base_id.clone(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            frozen_lora_artifact_id: self.frozen_lora_artifact_id().map(str::to_owned),
            geometry_artifact_id: batch.source_artifact_id().to_owned(),
            readout_accumulator: self.readout_accumulator,
            site: site.path.clone(),
            rotation,
            strength: strength as f32,
            shape: [steps, width],
            contrast: Self::contrast_row(logits.data(), steps, target, rival),
            source_features: features(&input, rotation),
            site_input: input.data().to_vec(),
            upstream: upstream.data().to_vec(),
            gate_gradient: grads.remove("gate").ok_or("missing gate VJP")?,
            bias_gradient: grads.remove("bias").ok_or("missing bias VJP")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_counts_all_return_vectors_and_refuses_overflow() {
        assert!(payload(1, MAX_PREFIX_SIGNAL_VALUES / 5).is_ok());
        assert!(payload(1, MAX_PREFIX_SIGNAL_VALUES / 5 + 1).is_err());
        assert!(payload(usize::MAX, 1).is_err());
    }

    #[test]
    fn closed_lag_rows_keep_odd_tail_and_route_past_markers() {
        let mut x = vec![0.; 18];
        x[3] = 6.;
        x[12] = -12.;
        x[2] = 18.;
        let input = Tensor::from_vec(6, 3, x).unwrap();
        let positive = features(&input, PrefixRotation::QuarterTurn);
        assert_eq!(&positive[3..6], &[1., 0., 0.]);
        assert_eq!(&positive[12..15], &[0., -2., 0.]);
        assert_eq!(&positive[0..3], &[0., 0., 3.]);
        assert_eq!(
            &features(&input, PrefixRotation::InverseQuarterTurn)[12..15],
            &[0., 2., 0.]
        );
    }
}
