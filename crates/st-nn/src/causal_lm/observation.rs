//! Scalar observations of an actual native corpus update; no activation export.
use super::*;
use crate::geometry_artifact::GeometryAdapterTensorRole;
use crate::geometry_runtime::ParameterGradients;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct CausalLmObservedTrainingStep {
    pub schema: &'static str,
    pub base_model_identity: String,
    pub data_id: String,
    pub step: CausalLmTrainingStep,
    pub sites: BTreeMap<String, CausalLmSiteObservation>,
}

#[derive(Debug, Serialize)]
pub struct CausalLmSiteObservation {
    pub elements: u64,
    pub input_rms: f64,
    pub residual_rms: f64,
    /// None for an all-zero input, including when the residual is also zero.
    pub relative_residual_rms: Option<f64>,
    /// L2 of the reduced objective gradient over trainable parameters, before clipping.
    pub trainable_gradient_l2: f64,
    pub pre_log_radius: Option<f32>,
    /// d(loss)/d(log_radius), not d(loss)/d(radius); None for a fixed/absent radius.
    pub log_radius_gradient: Option<f32>,
}

#[derive(Default)]
struct Squares {
    elements: u64,
    input: f64,
    residual: f64,
}

#[derive(Default)]
pub(super) struct ResidualSums(BTreeMap<String, Squares>);

impl ResidualSums {
    pub fn record(&mut self, path: &str, input: &Tensor, output: &Tensor) -> CausalLmResult<()> {
        if input.shape() != output.shape() || input.data().is_empty() {
            return Err("geometry observation shape mismatch or empty site".into());
        }
        let sums = self.0.entry(path.to_owned()).or_default();
        sums.elements = sums
            .elements
            .checked_add(input.data().len() as u64)
            .ok_or("geometry observation element count overflow")?;
        for (&x, &y) in input.data().iter().zip(output.data()) {
            let x = f64::from(x);
            let delta = f64::from(y) - x;
            sums.input += x * x;
            sums.residual += delta * delta;
        }
        Ok(())
    }

    pub fn finish(
        mut self,
        artifact: &GeometryAdapterArtifact,
        gradients: &ParameterGradients,
    ) -> CausalLmResult<BTreeMap<String, CausalLmSiteObservation>> {
        let mut sites = BTreeMap::new();
        for site in &artifact.record().placements {
            let sums = self
                .0
                .remove(&site.path)
                .ok_or("missing forward geometry observation")?;
            let group = gradients
                .get(&site.path)
                .ok_or("missing observed gradient placement")?;
            let mut squared_gradient = 0.0;
            let mut pre_log_radius = None;
            let mut log_radius_gradient = None;
            for parameter in &site.tensors {
                let radius = parameter.name == "log_radius";
                if radius {
                    pre_log_radius = Some(parameter.values[0]);
                }
                if parameter.role != GeometryAdapterTensorRole::Parameter
                    || !parameter.requires_grad
                {
                    continue;
                }
                let values = group
                    .get(&parameter.name)
                    .ok_or("missing observed trainable gradient")?;
                for &g in values {
                    squared_gradient += f64::from(g).powi(2);
                }
                if radius {
                    log_radius_gradient = Some(values[0]);
                }
            }
            if !sums.input.is_finite()
                || !sums.residual.is_finite()
                || !squared_gradient.is_finite()
            {
                return Err("nonfinite geometry observation".into());
            }
            let n = sums.elements as f64;
            sites.insert(
                site.path.clone(),
                CausalLmSiteObservation {
                    elements: sums.elements,
                    input_rms: (sums.input / n).sqrt(),
                    residual_rms: (sums.residual / n).sqrt(),
                    relative_residual_rms: (sums.input > 0.)
                        .then(|| (sums.residual / sums.input).sqrt()),
                    trainable_gradient_l2: squared_gradient.sqrt(),
                    pre_log_radius,
                    log_radius_gradient,
                },
            );
        }
        if !self.0.is_empty() {
            return Err("unexpected observed placement".into());
        }
        Ok(sites)
    }
}
