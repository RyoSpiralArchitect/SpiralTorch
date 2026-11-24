//! Unit gold-minus-rival adjoint through the factual decoder; alternative channel maps stay local.
use super::*;
use crate::geometry_runtime::{ChannelParameterVjp, WaveFactorConfig};
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct CausalLmWaveMarginPullbacks {
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub geometry_artifact_id: String,
    pub site: String,
    pub factual: &'static str,
    pub config: WaveFactorConfig,
    pub strength: f32,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub contrast: CausalLmTokenContrast,
    /// Only the last row has a nonzero arriving adjoint for this placement.
    pub site_input: Vec<f32>,
    pub upstream: Vec<f32>,
    pub arms: BTreeMap<&'static str, ChannelParameterVjp>,
}

impl NativeCausalLm {
    pub fn wave_margin_input_count(
        &self,
        steps: usize,
        gold: usize,
        rival: usize,
        config: WaveFactorConfig,
    ) -> CausalLmResult<usize> {
        self.next_token_contrast_input_count(steps, gold, rival)?;
        self.validate_adaptation_mode(CausalLmAdaptationMode::Geometry)?;
        let artifact = self
            .geometry_artifact()
            .ok_or("missing margin probe geometry")?;
        let [p] = artifact.record().placements.as_slice() else {
            return Err("margin probe requires one final attention geometry".into());
        };
        if p.path != format!("transformer.h.{}.attn", self.config.layers - 1)
            || p.tensors
                .iter()
                .any(|t| t.requires_grad != matches!(t.name.as_str(), "gate" | "bias"))
        {
            return Err(
                "margin probe requires trainable gate/bias on final attention and a fixed radius"
                    .into(),
            );
        }
        let (_, max) = config.admit(p)?;
        if self.config.width as u128 * 10 > MAX_MIXTURE_PROBE_ELEMENTS as u128
            || steps as u128 * self.config.width as u128 > max as u128
        {
            return Err("wave margin probe exceeds local or return element budget".into());
        }
        self.adaptation_labeled_input_count(
            [1, steps.checked_add(1).ok_or("prompt length overflow")?],
            true,
        )?;
        Ok(steps)
    }

    pub fn last_attention_wave_margin_pullbacks(
        &self,
        prompt: &[usize],
        gold: usize,
        rival: usize,
        config: WaveFactorConfig,
    ) -> CausalLmResult<CausalLmWaveMarginPullbacks> {
        let steps = self.wave_margin_input_count(prompt.len(), gold, rival, config)?;
        let primal = self.capture_last_attention(prompt)?;
        let geom = primal
            .cache
            .attention_geometry
            .as_ref()
            .ok_or("missing factual geometry snapshot")?;
        let artifact = self.geometry_artifact().ok_or("missing factual geometry")?;
        let p = &artifact.record().placements[0];
        let contrast = Self::contrast_row(primal.logits.data(), steps, gold, rival);
        let mut seed = vec![0.; self.config.vocabulary];
        seed[gold] = 1.;
        seed[rival] = -1.;
        let incoming = self.last_attention_upstream(
            &primal,
            &Tensor::from_vec(1, self.config.vocabulary, seed)?,
            self.readout_accumulator,
            |_, _| Ok(()),
        )?;
        let upstream = mixture_probe::active_row(&incoming, steps)?;
        let x = geom.wave_factor_input()?;
        let width = self.config.width;
        if x.shape() != (steps, width) || x.data().iter().any(|v| !v.is_finite()) {
            return Err("invalid margin probe site input".into());
        }
        Ok(CausalLmWaveMarginPullbacks {
            base_model_identity: self.base_id.clone(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            geometry_artifact_id: artifact.artifact_id().to_owned(),
            site: p.path.clone(),
            factual: config.admit(p)?.0,
            config,
            strength: geom.residual_strength(),
            readout_accumulator: self.readout_accumulator,
            contrast,
            site_input: x.data()[(steps - 1) * width..].to_vec(),
            upstream,
            arms: geom.wave_parameter_pullbacks(config, incoming.data())?,
        })
    }
}
