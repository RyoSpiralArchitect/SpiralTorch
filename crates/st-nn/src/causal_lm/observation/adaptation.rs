//! Optional scalar tap of one real adaptation window. Forward measurement is not repeated on replay.
use super::*;
use crate::geometry_runtime::NativeGeometryBatch;
use crate::geometry_runtime::{WaveFactorConfig, WaveFactorObservation};
use crate::WaveGateConditioning;

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmAdaptationObservation {
    pub schema: &'static str,
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub geometry_artifact_id: Option<String>,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub mode: CausalLmAdaptationMode,
    pub window_shape: [usize; 2],
    pub target_tokens: usize,
    pub loss: f32,
    pub sites: BTreeMap<String, CausalLmAdaptationSite>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmAdaptationSite {
    pub elements: u64,
    pub input_rms: f64,
    /// RMS of the actual rounded output minus input, including placement strength.
    pub residual_rms: f64,
    pub relative_residual_rms: Option<f64>,
    pub pullback: CausalLmSitePullback,
    /// Only parameters selected by this mode and marked trainable, before SGD.
    pub parameter_gradient_l2: BTreeMap<String, f64>,
    pub log_radius_gradient: Option<f32>,
    /// None for non-Wave sites and strength-zero bypasses.
    pub wave: Option<WaveGateConditioning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wave_factorial: Option<WaveFactorObservation>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmSitePullback {
    pub upstream_rms: f64,
    /// Null if the normal backward traversal stops at this site's parameter VJP.
    pub input_adjoint_rms: Option<f64>,
    /// RMS(dx - dy) isolates the effect on the adjoint of the residual branch from the identity.
    pub correction_rms: Option<f64>,
    pub relative_correction_rms: Option<f64>,
}

#[derive(Default)]
pub(in crate::causal_lm) struct AdaptationTrace {
    factor_config: Option<WaveFactorConfig>,
    pub(in crate::causal_lm) forward: ResidualSums,
    sites: BTreeMap<String, CausalLmAdaptationSite>,
}

impl AdaptationTrace {
    pub(in crate::causal_lm) fn factorial(config: WaveFactorConfig) -> Self {
        Self {
            factor_config: Some(config),
            ..Self::default()
        }
    }
    pub(in crate::causal_lm) fn pullback(
        &mut self,
        path: &str,
        batch: &NativeGeometryBatch,
        upstream: &Tensor,
        input_adjoint: Option<&Tensor>,
    ) -> CausalLmResult<()> {
        let primal = self
            .forward
            .0
            .remove(path)
            .ok_or("missing adaptation primal")?;
        if primal.elements != upstream.data().len() as u64
            || primal.elements == 0
            || upstream.shape() != batch.output().shape()
            || input_adjoint.is_some_and(|dx| dx.shape() != upstream.shape())
        {
            return Err("adaptation observation shape or multiplicity mismatch".into());
        }
        let n = primal.elements as f64;
        let norm_sq = |values: &[f32]| values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        let up = norm_sq(upstream.data());
        let input = input_adjoint.map(|x| norm_sq(x.data()));
        let correction = input_adjoint.map(|dx| {
            dx.data()
                .iter()
                .zip(upstream.data())
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                .sum::<f64>()
        });
        if [
            Some(primal.input),
            Some(primal.residual),
            Some(up),
            input,
            correction,
        ]
        .into_iter()
        .flatten()
        .any(|x| !x.is_finite())
        {
            return Err("nonfinite adaptation observation".into());
        }
        let rms = |sq: f64| (sq / n).sqrt();
        let site = CausalLmAdaptationSite {
            elements: primal.elements,
            input_rms: rms(primal.input),
            residual_rms: rms(primal.residual),
            relative_residual_rms: (primal.input > 0.)
                .then(|| (primal.residual / primal.input).sqrt()),
            pullback: CausalLmSitePullback {
                upstream_rms: rms(up),
                input_adjoint_rms: input.map(rms),
                correction_rms: correction.map(rms),
                relative_correction_rms: correction.filter(|_| up > 0.).map(|v| (v / up).sqrt()),
            },
            parameter_gradient_l2: BTreeMap::new(),
            log_radius_gradient: None,
            wave: batch.wave_conditioning(),
            wave_factorial: self
                .factor_config
                .map(|c| {
                    batch.wave_factorial(
                        c,
                        upstream,
                        input_adjoint.ok_or("wave factorial requires the factual input adjoint")?,
                    )
                })
                .transpose()?,
        };
        if self.sites.insert(path.to_owned(), site).is_some() {
            return Err("duplicate adaptation pullback observation".into());
        }
        Ok(())
    }

    pub(in crate::causal_lm) fn finish(
        mut self,
        model: &NativeCausalLm,
        mode: CausalLmAdaptationMode,
        gradients: &NativeCausalLmAdaptationGradients,
    ) -> CausalLmResult<CausalLmAdaptationObservation> {
        let placements = model
            .geometry
            .as_ref()
            .map_or(&[][..], |g| g.artifact().record().placements.as_slice());
        if !self.forward.0.is_empty() || self.sites.len() != placements.len() {
            return Err("incomplete adaptation observations".into());
        }
        for placement in placements {
            let site = self
                .sites
                .get_mut(&placement.path)
                .ok_or("missing adaptation site")?;
            if mode != CausalLmAdaptationMode::Lora {
                for t in placement
                    .tensors
                    .iter()
                    .filter(|t| t.role == GeometryAdapterTensorRole::Parameter && t.requires_grad)
                {
                    let values = gradients
                        .parameters()
                        .get(&placement.path)
                        .and_then(|v| v.get(&t.name))
                        .ok_or("missing observed adaptation parameter")?;
                    site.parameter_gradient_l2.insert(
                        t.name.clone(),
                        values
                            .iter()
                            .map(|&v| f64::from(v).powi(2))
                            .sum::<f64>()
                            .sqrt(),
                    );
                    if t.name == "log_radius" {
                        site.log_radius_gradient = Some(values[0]);
                    }
                }
            }
        }
        Ok(CausalLmAdaptationObservation {
            schema: if self.factor_config.is_some() {
                "spiraltorch.native_causal_lm_adaptation_observation.v2"
            } else {
                "spiraltorch.native_causal_lm_adaptation_observation.v1"
            },
            base_model_identity: gradients.base_model_identity().to_owned(),
            lora_artifact_id: gradients.lora_artifact_id().map(str::to_owned),
            geometry_artifact_id: gradients.geometry_artifact_id().map(str::to_owned),
            readout_accumulator: gradients.readout_accumulator(),
            mode,
            window_shape: gradients.window_shape(),
            target_tokens: gradients.target_tokens(),
            loss: gradients.loss(),
            sites: self.sites,
        })
    }
}
