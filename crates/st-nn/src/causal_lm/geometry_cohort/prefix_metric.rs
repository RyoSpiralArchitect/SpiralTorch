//! Match motion in the calibrated final-row residual branch, not distance in parameter space.
use super::*;
use crate::geometry_artifact::GeometryAdapterPlacement;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixMetricRule {
    Uniform,
    Diagonal,
    Cyclic,
}

/// Only produced by native calibration; the current value of the gate is deliberately not an owner.
#[derive(Debug, Serialize)]
pub struct CausalLmPrefixResidualMetric {
    schema: &'static str,
    id: String,
    anchor: String,
    base_model_identity: String,
    frozen_lora_artifact_id: Option<String>,
    readout_accumulator: CausalLmReadoutAccumulator,
    samples: usize,
    site: String,
    second_moments: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct PrefixResidualStep {
    pub schema: &'static str,
    pub metric_id: String,
    pub rule: PrefixMetricRule,
    pub pre_geometry_id: String,
    pub post_geometry_id: String,
    pub members: usize,
    pub total_target_tokens: usize,
    pub mean_input_loss: f64,
    pub target_rms: f64,
    pub realized_rms: f64,
    pub requested_parameter_l2: f64,
    pub realized_parameter_l2: f64,
    pub member_loss_dots: Vec<f64>,
}

fn rms(q: &[f64], dx: impl Iterator<Item = f64>) -> f64 {
    q.iter()
        .zip(dx)
        .fold(0.0_f64, |n, (&v, d)| n.hypot(v.sqrt() * d))
        / (q.len() as f64).sqrt()
}

fn scale_direction(
    q: &[f64],
    g: &mut [f64],
    rule: PrefixMetricRule,
    target: f64,
    cap: f64,
) -> CausalLmResult<f64> {
    let largest = q.iter().copied().fold(0.0_f64, f64::max);
    if largest <= 0. || !largest.is_finite() {
        return Err("prefix calibration has no supported channels".into());
    }
    let damping = q.iter().map(|x| x / largest).sum::<f64>() / (16. * q.len() as f64);
    for (j, x) in g.iter_mut().enumerate() {
        if q[j] == 0. {
            *x = 0.;
        } else {
            *x /= match rule {
                PrefixMetricRule::Uniform => 1.,
                PrefixMetricRule::Diagonal => q[j] / largest + damping,
                PrefixMetricRule::Cyclic => q[(j + 1) % q.len()] / largest + damping,
            };
        }
    }
    let len = g.iter().fold(0.0_f64, |n, &v| n.hypot(v));
    if !len.is_finite() || len == 0. {
        return Err("no nonzero supported prefix descent direction".into());
    }
    for x in g.iter_mut() {
        *x /= len;
    }
    let l2 = target / rms(q, g.iter().copied());
    if !l2.is_finite() || l2 <= 0. || l2 > cap {
        return Err("residual RMS requires a step outside the parameter cap".into());
    }
    Ok(l2)
}

impl NativeCausalLm {
    fn metric_site(&self) -> CausalLmResult<(&GeometryAdapterPlacement, f32)> {
        let artifact = self
            .geometry_artifact()
            .ok_or("prefix calibration needs geometry")?;
        let [p] = artifact.record().placements.as_slice() else {
            return Err("prefix calibration requires a single placement".into());
        };
        // Orthogonal output routing preserves the squared norm of every gated row.
        let strength = p
            .recipe
            .prefix_transport()
            .ok_or("prefix calibration requires prefix transport")?
            .strength;
        if p.path != format!("transformer.h.{}.mlp", self.config.layers - 1)
            || strength == 0.
            || self.lora.is_some()
            || !p.training
            || !p
                .tensors
                .iter()
                .any(|t| t.name == "gate" && t.requires_grad)
            || p.tensors
                .iter()
                .any(|t| t.name != "gate" && t.requires_grad)
        {
            return Err(
                "prefix calibration needs only a trainable final MLP gate and a frozen host".into(),
            );
        }
        Ok((p, strength))
    }

    fn metric_anchor(&self) -> CausalLmResult<String> {
        self.metric_site()?;
        let mut record = self.geometry_artifact().unwrap().record().clone();
        record.placements[0]
            .tensors
            .iter_mut()
            .find(|t| t.name == "gate")
            .unwrap()
            .values
            .fill(0.);
        Ok(GeometryAdapterArtifact::create(record)?
            .artifact_id()
            .to_owned())
    }

    /// Bound aggregate token materialization, MLP/QKV work and attention pairs before copying IDs.
    pub fn validate_prefix_calibration(&self, lengths: &[usize]) -> CausalLmResult<()> {
        self.metric_site()?;
        if !(1..=MAX_GEOMETRY_COHORT_MEMBERS).contains(&lengths.len()) {
            return Err("prefix calibration requires 1 to 256 samples".into());
        }
        let mut cells = 0usize;
        let mut pairs = 0usize;
        for &n in lengths {
            self.context_input_count([1, n])?;
            let act = product(
                &[
                    n,
                    self.config.width.max(self.config.intermediate),
                    self.config.layers,
                ],
                MAX_GEOMETRY_COHORT_ELEMENTS,
                "prefix calibration activations",
            )?;
            let att = product(
                &[self.config.layers, self.config.heads, n, n],
                MAX_GEOMETRY_COHORT_ELEMENTS,
                "prefix calibration attention",
            )?;
            cells = cells
                .checked_add(act)
                .ok_or("prefix calibration overflow")?;
            pairs = pairs
                .checked_add(att)
                .ok_or("prefix calibration overflow")?;
            if cells > MAX_GEOMETRY_COHORT_ELEMENTS || pairs > MAX_GEOMETRY_COHORT_ELEMENTS {
                return Err("prefix calibration cumulative budget exceeded".into());
            }
        }
        Ok(())
    }

    pub fn calibrate_prefix_residual_metric(
        &self,
        prompts: &[Vec<usize>],
    ) -> CausalLmResult<CausalLmPrefixResidualMetric> {
        if prompts.len() > MAX_GEOMETRY_COHORT_MEMBERS {
            return Err("too many calibration samples".into());
        }
        self.validate_prefix_calibration(&prompts.iter().map(Vec::len).collect::<Vec<_>>())?;
        let (p, strength) = self.metric_site()?;
        let layer = self.config.layers - 1;
        let anchor = self.metric_anchor()?;
        let schema = "spiraltorch.prefix_residual_metric.v1";
        let mut hash = Sha256::new();
        hash.update(schema.as_bytes());
        hash.update(anchor.as_bytes());
        hash.update(self.readout_accumulator.as_str().as_bytes());
        let mut moments: Vec<_> = (0..self.config.width).map(|_| Sum::default()).collect();
        for prompt in prompts {
            let shape = [1, prompt.len()];
            let hidden = self.forward_blocks(self.embed(prompt, shape)?, shape, 0..layer)?;
            let mut snap = None;
            self.run_block(
                layer,
                hidden,
                shape,
                decoder::BlockPass::CaptureGeometry {
                    path: &p.path,
                    snapshot: &mut snap,
                },
            )?;
            let (_, b) = snap.ok_or("missing prefix calibration site")?;
            for (s, &x) in moments.iter_mut().zip(b.single_prefix_last_mean()?) {
                s.add((f64::from(strength) * f64::from(x)).powi(2));
            }
            hash.update((prompt.len() as u64).to_le_bytes());
            for &id in prompt {
                hash.update((id as u64).to_le_bytes());
            }
        }
        let q: Vec<_> = moments
            .into_iter()
            .map(|x| x.mean(prompts.len() as u64))
            .collect();
        if !q.iter().all(|v| v.is_finite()) || q.iter().all(|&v| v == 0.) {
            return Err("degenerate prefix calibration".into());
        }
        for v in &q {
            hash.update(v.to_le_bytes());
        }
        Ok(CausalLmPrefixResidualMetric {
            schema,
            id: format!("sha256:{:x}", hash.finalize()),
            anchor,
            base_model_identity: self.base_id.clone(),
            frozen_lora_artifact_id: self.frozen_lora_artifact_id().map(str::to_owned),
            readout_accumulator: self.readout_accumulator,
            samples: prompts.len(),
            site: p.path.clone(),
            second_moments: q,
        })
    }

    pub fn step_prefix_residual_rms(
        &self,
        metric: &CausalLmPrefixResidualMetric,
        gradients: &[&NativeCausalLmAdaptationGradients],
        rule: PrefixMetricRule,
        target_rms: f64,
        max_parameter_l2: f64,
    ) -> CausalLmResult<(Self, PrefixResidualStep)> {
        self.validate_geometry_cohort_size(gradients.len())?;
        if !target_rms.is_finite()
            || target_rms <= 0.
            || !max_parameter_l2.is_finite()
            || max_parameter_l2 <= 0.
        {
            return Err("prefix step needs positive finite RMS and parameter cap".into());
        }
        if self.base_id != metric.base_model_identity
            || self.readout_accumulator != metric.readout_accumulator
            || self.metric_anchor()? != metric.anchor
        {
            return Err("prefix calibration owner mismatch".into());
        }
        let mut mean = self.reduce_geometry_cohort(gradients, std::convert::identity)?;
        let g = mean
            .parameters
            .get_mut(&metric.site)
            .unwrap()
            .get_mut("gate")
            .unwrap();
        let l2 = scale_direction(
            &metric.second_moments,
            g,
            rule,
            target_rms,
            max_parameter_l2,
        )?;
        let origin = self.geometry_artifact().unwrap();
        let (end, motion) = origin.match_validated_cohort_ray(&mean.parameters, l2)?;
        let idx = mean.blocks[0].tensor;
        let before = &origin.record().placements[0].tensors[idx].values;
        let after = &end.record().placements[0].tensors[idx].values;
        let actual = rms(
            &metric.second_moments,
            after
                .iter()
                .zip(before)
                .map(|(&b, &a)| f64::from(b) - f64::from(a)),
        );
        let (physical, dots) = mean.realized(origin, &end);
        let mut predicted = Sum::default();
        for &v in &dots {
            predicted.add(v);
        }
        if !actual.is_finite()
            || (actual / target_rms - 1.).abs() > 1e-4
            || physical > max_parameter_l2
            || !predicted.value().is_finite()
            || predicted.value() >= 0.
        {
            return Err("stored prefix step violates RMS, cap or descent".into());
        }
        let receipt = PrefixResidualStep {
            schema: "spiraltorch.prefix_residual_step.v1",
            metric_id: metric.id.clone(),
            rule,
            pre_geometry_id: motion.origin_id,
            post_geometry_id: motion.result_id,
            members: mean.members,
            total_target_tokens: mean.targets,
            mean_input_loss: mean.loss,
            target_rms,
            realized_rms: actual,
            requested_parameter_l2: l2,
            realized_parameter_l2: physical,
            member_loss_dots: dots,
        };
        Ok((self.with_geometry(end)?, receipt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spectrum_control_and_null_axes() {
        for r in [
            PrefixMetricRule::Uniform,
            PrefixMetricRule::Diagonal,
            PrefixMetricRule::Cyclic,
        ] {
            let mut g = [3., 4., 7.];
            let len = scale_direction(&[1., 4., 0.], &mut g, r, 0.1, 2.).unwrap();
            assert_eq!(g[2], 0.);
            assert!((rms(&[1., 4., 0.], g.iter().map(|v| v * len)) - 0.1).abs() < 1e-14);
            assert!(scale_direction(&[1., 4., 0.], &mut [0., 0., 1.], r, 0.1, 2.).is_err());
            assert!(scale_direction(&[1., 4., 0.], &mut g, r, 0.1, 0.001).is_err());
        }
    }
    #[test]
    fn scalar_metric_changes_no_direction() {
        for r in [
            PrefixMetricRule::Uniform,
            PrefixMetricRule::Diagonal,
            PrefixMetricRule::Cyclic,
        ] {
            let mut g = [3., 4.];
            let len = scale_direction(&[4., 4.], &mut g, r, 1., 2.).unwrap();
            assert!((g[0] - 0.6).abs() < 1e-14 && (g[1] - 0.8).abs() < 1e-14);
            assert!((len - 0.5 * 2f64.sqrt()).abs() < 1e-14);
        }
    }
}
