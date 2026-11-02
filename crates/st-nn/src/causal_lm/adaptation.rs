//! Shared native adaptation contract. One snapshot owns both parameter groups;
//! bindings never implement a second optimizer or backward pass.
use super::*;
use crate::geometry_artifact::GeometryAdapterTensorRole;
use std::collections::BTreeMap;

/// Which attached group(s) an atomic SGD step updates. Unselected groups remain
/// in the forward function and its derivative, rather than being removed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmAdaptationMode {
    Lora,
    Geometry,
    Joint,
}

#[derive(Debug)]
pub struct NativeCausalLmAdaptationGradients {
    base_id: String,
    lora_id: Option<String>,
    geometry_id: Option<String>,
    readout_accumulator: CausalLmReadoutAccumulator,
    loss: f32,
    target_tokens: usize,
    shape: [usize; 2],
    parameters: BTreeMap<String, BTreeMap<String, Vec<f32>>>,
    pub(super) observation: Option<Box<CausalLmAdaptationObservation>>,
}

impl NativeCausalLmAdaptationGradients {
    pub fn observation(&self) -> Option<&CausalLmAdaptationObservation> {
        self.observation.as_deref()
    }

    pub fn loss(&self) -> f32 {
        self.loss
    }
    pub fn target_tokens(&self) -> usize {
        self.target_tokens
    }
    pub fn window_shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn base_model_identity(&self) -> &str {
        &self.base_id
    }
    pub fn lora_artifact_id(&self) -> Option<&str> {
        self.lora_id.as_deref()
    }
    pub fn geometry_artifact_id(&self) -> Option<&str> {
        self.geometry_id.as_deref()
    }
    pub fn readout_accumulator(&self) -> CausalLmReadoutAccumulator {
        self.readout_accumulator
    }
    pub fn parameters(&self) -> &BTreeMap<String, BTreeMap<String, Vec<f32>>> {
        &self.parameters
    }

    pub(super) fn into_parameters(self) -> BTreeMap<String, BTreeMap<String, Vec<f32>>> {
        self.parameters
    }
}

impl NativeCausalLm {
    /// This composition is intentionally one LoRA projection followed by one
    /// native geometry at the same attention output, before the residual add.
    pub(super) fn validate_adaptation_placement(&self) -> CausalLmResult<()> {
        if let (Some(lora), Some(geometry)) = (&self.lora, &self.geometry) {
            let placements = &geometry.artifact().record().placements;
            let expected = format!("transformer.h.{}.attn", lora.layer());
            if placements.len() != 1 || placements[0].path != expected {
                return Err("composed adaptation requires one geometry at the LoRA layer's attention output".into());
            }
        }
        Ok(())
    }

    pub fn validate_adaptation_mode(&self, mode: CausalLmAdaptationMode) -> CausalLmResult<()> {
        self.validate_adaptation_placement()?;
        if matches!(
            mode,
            CausalLmAdaptationMode::Lora | CausalLmAdaptationMode::Joint
        ) && self.lora.is_none()
        {
            return Err("selected adaptation mode requires a LoRA adapter".into());
        }
        if matches!(
            mode,
            CausalLmAdaptationMode::Geometry | CausalLmAdaptationMode::Joint
        ) {
            let runtime = self
                .geometry
                .as_ref()
                .ok_or("selected adaptation mode requires geometry")?;
            if !runtime.artifact().record().placements.iter().any(|p| {
                p.tensors
                    .iter()
                    .any(|t| t.role == GeometryAdapterTensorRole::Parameter && t.requires_grad)
            }) {
                return Err("selected geometry update has no trainable parameters".into());
            }
        }
        Ok(())
    }

    pub fn adaptation_gradient_input_count(&self, shape: [usize; 2]) -> CausalLmResult<usize> {
        self.adaptation_labeled_input_count(shape, false)
    }

    pub fn adaptation_labeled_input_count(
        &self,
        shape: [usize; 2],
        has_labels: bool,
    ) -> CausalLmResult<usize> {
        self.validate_adaptation_placement()?;
        if self.lora.is_some() {
            self.lora_pullback_input_count(shape, has_labels)
        } else {
            self.geometry_objective_input_count(shape, has_labels)
        }
    }

    /// Mean next-token CE and sealed gradients of every attached group. For a
    /// composed model a single decoder pass and reverse traversal produces both.
    pub fn adaptation_loss_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        self.adaptation_loss_gradients_with_labels(windows, shape, None, None)
    }

    /// Differentiate only the selected group(s), without removing unselected
    /// adapters from the forward or the pullback into earlier trainable weights.
    pub fn adaptation_loss_gradients_for(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        mode: CausalLmAdaptationMode,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        self.adaptation_loss_gradients_with_labels(windows, shape, None, Some(mode))
    }

    /// Mean CE of the selected shifted labels. Ignored positions remain in
    /// causal attention; only the readout and its reverse seed are selected.
    pub fn adaptation_loss_gradients_with_labels(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        mode: Option<CausalLmAdaptationMode>,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        self.adaptation_window_gradients(windows, shape, labels, mode, false)
    }

    /// Optionally measure the actual residual and adjoint while computing the same sealed gradients.
    pub fn adaptation_window_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        mode: Option<CausalLmAdaptationMode>,
        observe_geometry: bool,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        if let Some(mode) = mode {
            self.validate_adaptation_mode(mode)?;
        }
        self.adaptation_labeled_input_count(shape, labels.is_some())?;
        let prepared =
            self.prepare_objective(windows, shape, labels, CausalLmObjective::default())?;
        let hidden = self.embed(&prepared.input, prepared.shape)?;
        self.adaptation_gradients_from_prefix(
            &prepared,
            hidden,
            0,
            mode,
            observe_geometry.then(observation::AdaptationTrace::default),
        )
    }

    pub fn admit_wave_factorial(
        &self,
        config: crate::geometry_runtime::WaveFactorConfig,
    ) -> CausalLmResult<()> {
        self.validate_adaptation_mode(CausalLmAdaptationMode::Joint)?;
        let site = &self
            .geometry_artifact()
            .ok_or("missing factorial geometry")?
            .record()
            .placements[0];
        config.admit(site)?;
        Ok(())
    }

    /// Hold the input, gate, bias, residual strength and arriving CE gradient fixed.
    /// Only the factual map participates in the decoder; alternatives are local VJPs.
    pub fn wave_factorial_observation(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        config: crate::geometry_runtime::WaveFactorConfig,
    ) -> CausalLmResult<Box<CausalLmAdaptationObservation>> {
        self.admit_wave_factorial(config)?;
        self.adaptation_labeled_input_count(shape, labels.is_some())?;
        let prepared =
            self.prepare_objective(windows, shape, labels, CausalLmObjective::default())?;
        let hidden = self.embed(&prepared.input, prepared.shape)?;
        self.adaptation_gradients_from_prefix(
            &prepared,
            hidden,
            0,
            Some(CausalLmAdaptationMode::Joint),
            Some(observation::AdaptationTrace::factorial(config)),
        )?
        .observation
        .ok_or_else(|| "missing wave factorial receipt".into())
    }

    /// Caller has admitted this owner, selected mode, complete context and
    /// selected logits. Hidden is the boundary before any participating adapter.
    pub(super) fn adaptation_gradients_from_prefix(
        &self,
        prepared: &PreparedCausalWindow,
        hidden: Tensor,
        depth: usize,
        mode: Option<CausalLmAdaptationMode>,
        mut trace: Option<observation::AdaptationTrace>,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        let (first, site) = match &self.lora {
            Some(a) => (a.layer(), format!("transformer.h.{}.attn", a.layer())),
            None => self.first_geometry_site()?,
        };
        let mode = mode.unwrap_or(match (&self.lora, &self.geometry) {
            (Some(_), Some(_)) => CausalLmAdaptationMode::Joint,
            (Some(_), None) => CausalLmAdaptationMode::Lora,
            (None, Some(_)) => CausalLmAdaptationMode::Geometry,
            (None, None) => return Err("native LM has no adaptation parameters".into()),
        });
        let target = match mode {
            CausalLmAdaptationMode::Lora => learning::DecoderTarget::Lora,
            CausalLmAdaptationMode::Geometry => learning::DecoderTarget::GeometryFrom(&site),
            CausalLmAdaptationMode::Joint => learning::DecoderTarget::Joint,
        };
        let learning::TrainingReadout {
            checkpoints,
            hidden_rows,
            selected,
            d_normalized,
            loss,
            ..
        } = self.training_readout(
            prepared,
            hidden,
            depth,
            first,
            trace.as_mut().map(|t| &mut t.forward),
        )?;
        let parameters = self.decoder_parameter_pullback(
            checkpoints,
            hidden_rows,
            selected,
            prepared.shape,
            prepared.selected_rows.as_deref(),
            d_normalized,
            first,
            target,
            trace.as_mut(),
        )?;
        let want_lora = mode != CausalLmAdaptationMode::Geometry;
        let placements = self
            .geometry
            .as_ref()
            .filter(|_| mode != CausalLmAdaptationMode::Lora)
            .map_or(&[][..], |g| g.artifact().record().placements.as_slice());
        if parameters.len() != usize::from(want_lora) + placements.len()
            || self
                .lora
                .as_ref()
                .is_some_and(|a| parameters.contains_key(&a.path()) != want_lora)
            || placements.iter().any(|p| !parameters.contains_key(&p.path))
        {
            return Err("incorrect selected adaptation gradient groups".into());
        }
        let target_tokens = prepared.target_tokens;
        let shape = prepared.window_shape;
        if !loss.is_finite()
            || parameters
                .values()
                .flat_map(|p| p.values())
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err("nonfinite adaptation loss or gradient".into());
        }
        let mut gradients = NativeCausalLmAdaptationGradients {
            base_id: self.base_id.clone(),
            lora_id: self.lora_artifact_id().map(str::to_owned),
            geometry_id: self.geometry_artifact_id().map(str::to_owned),
            readout_accumulator: self.readout_accumulator,
            loss,
            target_tokens,
            shape,
            parameters,
            observation: None,
        };
        if let Some(trace) = trace {
            gradients.observation = Some(Box::new(trace.finish(self, mode, &gradients)?));
        }
        Ok(gradients)
    }

    /// Construct both updates from the same immutable source, committing only
    /// after every requested group validates. No intermediate model is exposed.
    pub fn step_adaptation_sgd(
        &self,
        gradients: &NativeCausalLmAdaptationGradients,
        mode: CausalLmAdaptationMode,
        learning_rate: f32,
    ) -> CausalLmResult<Self> {
        self.validate_adaptation_mode(mode)?;
        if gradients.base_id != self.base_id
            || gradients.lora_id.as_deref() != self.lora_artifact_id()
            || gradients.geometry_id.as_deref() != self.geometry_artifact_id()
            || gradients.readout_accumulator != self.readout_accumulator
            || !learning_rate.is_finite()
            || learning_rate <= 0.
        {
            return Err("adaptation gradients belong to a different snapshot/policy or learning rate is invalid".into());
        }
        let mut next = self.clone();
        if matches!(
            mode,
            CausalLmAdaptationMode::Lora | CausalLmAdaptationMode::Joint
        ) {
            let adapter = self.lora.as_ref().unwrap();
            let values = gradients
                .parameters
                .get(&adapter.path())
                .ok_or("missing LoRA parameters")?;
            let a = values.get("a").ok_or("missing LoRA A gradient")?;
            let b = values.get("b").ok_or("missing LoRA B gradient")?;
            next.lora = self.with_lora_parameter_step(a, b, learning_rate)?.lora;
        }
        if matches!(
            mode,
            CausalLmAdaptationMode::Geometry | CausalLmAdaptationMode::Joint
        ) {
            let runtime = self.geometry.as_ref().unwrap();
            let id = runtime.artifact().artifact_id();
            let candidate = if let Some(lora) = &self.lora {
                // Pass geometry's placement map by reference without copying LoRA A/B or any channel vectors.
                let external = lora.path();
                let view = gradients
                    .parameters
                    .iter()
                    .filter(|(path, _)| *path != &external)
                    .map(|(path, g)| (path.clone(), g))
                    .collect();
                runtime.with_gradients_sgd(id, &view, learning_rate)?
            } else {
                runtime.with_gradients_sgd(id, &gradients.parameters, learning_rate)?
            };
            next.geometry = Some(Arc::new(candidate));
        }
        Ok(next)
    }
}
