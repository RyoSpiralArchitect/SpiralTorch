//! A side connection reads the decoder's own attention values, without replacing its attention.
use super::*;
use crate::geometry_artifact::AttentionValueRoute as Route;

impl NativeCausalLm {
    pub(super) fn mlp_value_source(
        &self,
        path: &str,
        attended: &Tensor,
        values: &[f32],
        shape: [usize; 2],
        keys: usize,
    ) -> CausalLmResult<Option<Tensor>> {
        let Some((route, _)) = self
            .geometry_artifact()
            .and_then(|a| a.record().placements.iter().find(|p| p.path == path))
            .and_then(|p| p.recipe.value_gate())
        else {
            return Ok(None);
        };
        let [batch, added] = shape;
        let offset = keys
            .checked_sub(added)
            .ok_or("invalid value gate query offset")?;
        let (width, heads) = (self.config.width, self.config.heads);
        if batch.checked_mul(keys).and_then(|n| n.checked_mul(width)) != Some(values.len())
            || attended.shape() != (batch * added, width)
        {
            return Err("invalid decoder value shape".into());
        }
        if route == Route::Attention {
            return Ok(Some(attended.clone()));
        }
        let dim = width / heads;
        let mut out = vec![0.; attended.len()];
        for (src, dst) in values
            .chunks_exact(keys * dim)
            .zip(out.chunks_exact_mut(added * dim))
        {
            match route {
                Route::Attention => unreachable!(),
                Route::Fixed { position } => {
                    for (i, row) in dst.chunks_exact_mut(dim).enumerate() {
                        let j = (position as usize).min(offset + i) * dim;
                        row.copy_from_slice(&src[j..j + dim]);
                    }
                }
                Route::Uniform => {
                    let mut sum = vec![0f64; dim];
                    for (t, row) in src.chunks_exact(dim).enumerate() {
                        for (s, v) in sum.iter_mut().zip(row) {
                            *s += f64::from(*v);
                        }
                        if t >= offset {
                            for (x, s) in dst[(t - offset) * dim..].iter_mut().zip(&sum) {
                                *x = (*s / (t + 1) as f64) as f32;
                            }
                        }
                    }
                }
            }
        }
        Ok(Some(self.merge_heads([out.as_slice()], shape)?))
    }
}

/// Add the transpose of the causal uniform/fixed read to the ordinary value derivative.
/// The attention route instead shares the existing Q/K/V pullback with its ordinary consumer.
pub(super) fn add_control_vjp(
    route: Route,
    delta: &[f32],
    dv: &mut [f32],
    steps: usize,
    dim: usize,
) -> CausalLmResult<()> {
    let n = steps
        .checked_mul(dim)
        .filter(|&n| n > 0)
        .ok_or("invalid value source adjoint shape")?;
    if dv.len() != delta.len() || dv.len() % n != 0 {
        return Err("value source adjoint length mismatch".into());
    }
    for (extra, out) in delta.chunks_exact(n).zip(dv.chunks_exact_mut(n)) {
        let mut sum = vec![0f64; dim];
        for t in (0..steps).rev() {
            let row = &extra[t * dim..(t + 1) * dim];
            let dest = &mut out[t * dim..(t + 1) * dim];
            match route {
                Route::Uniform => {
                    for ((s, x), g) in sum.iter_mut().zip(dest).zip(row) {
                        *s += f64::from(*g) / (t + 1) as f64;
                        *x = (f64::from(*x) + *s) as f32;
                    }
                }
                Route::Fixed { position } => {
                    let j = (position as usize).min(steps - 1);
                    if t >= j {
                        for (s, g) in sum.iter_mut().zip(row) {
                            *s += f64::from(*g);
                        }
                    }
                    if t == j {
                        for (x, s) in dest.iter_mut().zip(&sum) {
                            *x = (f64::from(*x) + *s) as f32;
                        }
                    }
                    if t < j {
                        for (x, g) in dest.iter_mut().zip(row) {
                            *x += *g;
                        }
                    }
                }
                Route::Attention => return Err("attention reads require the Q/K/V adjoint".into()),
            }
        }
    }
    if dv.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite value route adjoint".into());
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct CausalLmValueReadSignal {
    pub schema: &'static str,
    pub site: String,
    pub base_model_identity: String,
    pub geometry_artifact_id: String,
    pub frozen_lora_artifact_id: Option<String>,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub route: Route,
    pub strength: f32,
    pub heads: usize,
    pub shape: [usize; 2],
    pub scale: f32,
    pub key_lag: usize,
    pub contrast: CausalLmTokenContrast,
    pub query: Vec<f32>,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub site_input: Vec<f32>,
    pub source: Vec<f32>,
    pub upstream: Vec<f32>,
    pub gate_gradient: Vec<f32>,
    pub bias_gradient: Vec<f32>,
}

impl NativeCausalLm {
    pub fn value_read_signal_input_count(
        &self,
        steps: usize,
        target: usize,
        rival: usize,
    ) -> CausalLmResult<usize> {
        self.next_token_contrast_input_count(steps, target, rival)?;
        let p = self
            .geometry_artifact()
            .and_then(|a| match a.record().placements.as_slice() {
                [p] => Some(p),
                _ => None,
            })
            .ok_or("value signal requires a single geometry placement")?;
        if p.path != format!("transformer.h.{}.mlp", self.config.layers - 1)
            || !p.recipe.value_gate().is_some_and(|(_, s)| s > 0.)
        {
            return Err("value signal requires an active final MLP value gate".into());
        }
        let rows = steps
            .checked_mul(2)
            .and_then(|n| n.checked_add(6))
            .ok_or("value signal overflow")?;
        product(
            &[rows, self.config.width],
            prefix_signal::MAX_PREFIX_SIGNAL_VALUES,
            "value signal return",
        )?;
        Ok(steps)
    }
    /// One factual forward and a unit token-margin derivative; no model or training state is advanced.
    pub fn last_mlp_value_read_signal(
        &self,
        ids: &[usize],
        target: usize,
        rival: usize,
    ) -> CausalLmResult<CausalLmValueReadSignal> {
        let n = self.value_read_signal_input_count(ids.len(), target, rival)?;
        let i = self.config.layers - 1;
        let shape = [1, n];
        let w = self.config.width;
        let hidden = self.forward_blocks(self.embed(ids, shape)?, shape, 0..i)?;
        let r = self.run_block(i, hidden, shape, decoder::BlockPass::BackwardReplay)?;
        let tape = r.cache.ok_or("missing value signal tape")?;
        let b = tape.mlp_geometry.ok_or("missing value signal geometry")?;
        let last = r.output.gather_rows(&[n - 1])?;
        let logits =
            self.project_readout(&self.final_norm.forward(&last, self.config.norm_epsilon)?)?;
        if logits.data().iter().any(|v| !v.is_finite()) {
            return Err("nonfinite value signal logits".into());
        }
        let mut seed = vec![0.; self.config.vocabulary];
        seed[target] = 1.;
        seed[rival] = -1.;
        let head =
            self.vocabulary_head_pullback(&Tensor::from_vec(1, self.config.vocabulary, seed)?)?;
        let dy = last.layer_norm_input_backward(
            &self.final_norm.gamma,
            &head,
            self.config.norm_epsilon,
        )?;
        let mut upstream = vec![0.; n * w];
        upstream[(n - 1) * w..].copy_from_slice(dy.data());
        let mut g = b.vjp_parameters(&upstream)?;
        let route = b.value_read_route().ok_or("missing value route")?;
        let dim = w / self.config.heads;
        let query = tape.qkv[0]
            .chunks_exact(n * dim)
            .flat_map(|r| r[(n - 1) * dim..].iter().copied())
            .collect();
        Ok(CausalLmValueReadSignal {
            schema: "spiraltorch.last_mlp_value_read_signal.v1",
            site: format!("transformer.h.{i}.mlp"),
            base_model_identity: self.base_id.clone(),
            geometry_artifact_id: b.source_artifact_id().into(),
            frozen_lora_artifact_id: self.frozen_lora_artifact_id().map(str::to_owned),
            readout_accumulator: self.readout_accumulator,
            route,
            strength: b.residual_strength(),
            heads: self.config.heads,
            shape: [n, w],
            scale: self.attention_spec(i, shape)?.scale(),
            key_lag: self.key_lag_at(i),
            contrast: Self::contrast_row(logits.data(), n, target, rival),
            query,
            keys: self.merge_heads([&tape.qkv[1]], shape)?.data().to_vec(),
            values: self.merge_heads([&tape.qkv[2]], shape)?.data().to_vec(),
            site_input: b.last_value_site_input()?.to_vec(),
            source: b.single_prefix_last_mean()?.to_vec(),
            upstream: dy.data().to_vec(),
            gate_gradient: g.remove("gate").ok_or("missing gate gradient")?,
            bias_gradient: g.remove("bias").ok_or("missing bias gradient")?,
        })
    }
}
