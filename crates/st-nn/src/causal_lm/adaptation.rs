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
}

impl NativeCausalLmAdaptationGradients {
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
        self.validate_adaptation_placement()?;
        if self.lora.is_some() {
            self.lora_pullback_input_count(shape)
        } else {
            self.geometry_gradient_input_count(shape)
        }
    }

    /// Mean next-token CE and sealed gradients of every attached group. For a
    /// composed model a single decoder pass and reverse traversal produces both.
    pub fn adaptation_loss_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        self.adaptation_gradient_input_count(shape)?;
        let (loss, target_tokens, parameters) = match (&self.lora, &self.geometry) {
            (None, Some(_)) => {
                let grads = self.geometry_loss_gradients(windows, shape)?;
                (grads.loss(), grads.target_tokens(), grads.into_parameters())
            }
            (Some(adapter), None) => {
                let grads = self.lora_loss_gradients(windows, shape)?;
                let (loss, targets) = (grads.loss(), grads.target_tokens());
                let (a, b) = grads.into_parameters();
                (
                    loss,
                    targets,
                    BTreeMap::from([(
                        adapter.path(),
                        BTreeMap::from([("a".into(), a), ("b".into(), b)]),
                    )]),
                )
            }
            (Some(adapter), Some(_)) => {
                let prepared =
                    self.prepare_objective(windows, shape, None, CausalLmObjective::default())?;
                let mut hidden = self.embed(&prepared.input, prepared.shape)?;
                let mut checkpoints = Vec::with_capacity(self.config.layers - adapter.layer() + 1);
                for index in 0..self.config.layers {
                    if index >= adapter.layer() {
                        checkpoints.push(hidden.clone());
                    }
                    hidden = self
                        .run_block(index, hidden, prepared.shape, decoder::BlockPass::Forward)?
                        .output;
                }
                checkpoints.push(hidden.clone());
                let normalized = self.final_norm.forward(&hidden, self.config.norm_epsilon)?;
                let logits = self.project_readout(&normalized)?;
                let config = prepared.objective.kernel_config();
                let loss = logits
                    .cross_entropy_with_logits(&prepared.labels, config)?
                    .data()[0];
                let seed = Tensor::from_vec(1, 1, vec![1.])?;
                let d_logits =
                    logits.cross_entropy_with_logits_backward(&prepared.labels, config, &seed)?;
                let d_normalized = learning::frozen_pullback(
                    &self.head,
                    &self.backward_head,
                    &d_logits,
                    self.readout_accumulator,
                )?;
                // Empty first geometry site forces its input VJP, then the
                // terminal LoRA parameter VJP stops before the frozen base.
                let parameters = self.decoder_parameter_pullback(
                    checkpoints,
                    hidden.shape().0,
                    hidden,
                    prepared.shape,
                    None,
                    d_normalized,
                    adapter.layer(),
                    "",
                    true,
                )?;
                if parameters.len() != 2
                    || !parameters.contains_key(&adapter.path())
                    || !parameters.contains_key(&format!("transformer.h.{}.attn", adapter.layer()))
                {
                    return Err("incomplete composed adaptation gradients".into());
                }
                (loss, prepared.target_tokens, parameters)
            }
            (None, None) => return Err("native LM has no adaptation parameters".into()),
        };
        if !loss.is_finite()
            || parameters
                .values()
                .flat_map(|p| p.values())
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err("nonfinite adaptation loss or gradient".into());
        }
        Ok(NativeCausalLmAdaptationGradients {
            base_id: self.base_id.clone(),
            lora_id: self.lora_artifact_id().map(str::to_owned),
            geometry_id: self.geometry_artifact_id().map(str::to_owned),
            readout_accumulator: self.readout_accumulator,
            loss,
            target_tokens,
            shape,
            parameters,
        })
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
