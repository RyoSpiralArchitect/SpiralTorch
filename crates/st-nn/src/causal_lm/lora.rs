//! One immutable attention-output LoRA adapter. Frozen GPT-2 arithmetic, loss
//! preparation and decoder pullback are shared with native geometry learning.
use super::*;
use crate::layers::lora_linear::{low_rank_forward, low_rank_parameter_vjp};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};
use st_tensor::Tile;

const MAX_LORA_JSON_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LoraRecord {
    schema: String,
    base_model_identity: String,
    layer: usize,
    width: usize,
    rank: usize,
    alpha: f32,
    a: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(super) struct NativeLoraAdapter {
    record: LoraRecord,
    id: String,
    a: PackedB,
    b: PackedB,
    b_transpose: PackedB,
    zero_delta: bool,
}

impl NativeLoraAdapter {
    fn create(record: LoraRecord, model: &NativeCausalLm) -> CausalLmResult<Self> {
        if record.schema != "spiraltorch.native_lora.attention_output.v1"
            || record.base_model_identity != model.base_id
            || record.layer >= model.config.layers
            || record.width != model.config.width
            || record.rank == 0
            || record.rank > record.width
            || !record.alpha.is_finite()
            || record.alpha <= 0.
        {
            return Err("invalid native LoRA configuration or base identity".into());
        }
        let count = product(
            &[2, record.width, record.rank],
            model.limits.max_parameter_elements,
            "LoRA parameters",
        )? / 2;
        if record.a.len() != count
            || record.b.len() != count
            || record.a.iter().chain(&record.b).any(|v| !v.is_finite())
        {
            return Err("invalid native LoRA tensor shape or nonfinite value".into());
        }
        let bytes = serde_json::to_vec(&record)?;
        if bytes.len() > MAX_LORA_JSON_BYTES {
            return Err("native LoRA artifact exceeds byte budget".into());
        }
        let mut hash = Sha256::new();
        hash.update(b"spiraltorch.native_lora.v1\0");
        hash.update(&bytes);
        let a = Tensor::from_vec(record.width, record.rank, record.a.clone())?;
        let b = Tensor::from_vec(record.rank, record.width, record.b.clone())?;
        Ok(Self {
            id: format!("sha256:{:x}", hash.finalize()),
            a: PackedB::from_tensor(&a, Tile::col_major())?,
            b: PackedB::from_tensor(&b, Tile::col_major())?,
            b_transpose: PackedB::from_tensor_transpose(&b, Tile::col_major())?,
            zero_delta: record.b.iter().all(|&v| v == 0.),
            record,
        })
    }

    pub(super) fn layer(&self) -> usize {
        self.record.layer
    }

    pub(super) fn path(&self) -> String {
        format!("transformer.h.{}.attn.c_proj", self.record.layer)
    }

    pub(super) fn vjp_parameters(
        &self,
        input: &Tensor,
        upstream: &Tensor,
    ) -> CausalLmResult<(Tensor, Tensor)> {
        Ok(low_rank_parameter_vjp(
            input,
            upstream,
            &self.a,
            &self.b_transpose,
            self.record.alpha / self.record.rank as f32,
            MatmulBackend::CpuSimd,
            MatmulBackend::CpuSimd,
            |_| TensorUtilBackend::Cpu,
        )?)
    }
}

/// Sealed language-loss gradients for exactly one immutable adapter snapshot.
#[derive(Debug)]
pub struct NativeCausalLmLoraGradients {
    base_id: String,
    adapter_id: String,
    readout_accumulator: CausalLmReadoutAccumulator,
    loss: f32,
    target_tokens: usize,
    shape: [usize; 2],
    a: Vec<f32>,
    b: Vec<f32>,
}

impl NativeCausalLmLoraGradients {
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
    pub fn lora_artifact_id(&self) -> &str {
        &self.adapter_id
    }
    pub fn a(&self) -> &[f32] {
        &self.a
    }
    pub fn b(&self) -> &[f32] {
        &self.b
    }
}

impl NativeCausalLm {
    /// Initialize one attention output adapter. A is deterministic and B is zero.
    /// An attached geometry must be the same layer's single attention output site.
    pub fn with_lora(&self, layer: usize, rank: usize, alpha: f32) -> CausalLmResult<Self> {
        if self.lora.is_some() {
            return Err(
                "native LoRA initialization requires a model without an existing LoRA adapter"
                    .into(),
            );
        }
        if layer >= self.config.layers
            || rank == 0
            || rank > self.config.width
            || !alpha.is_finite()
            || alpha <= 0.
        {
            return Err("invalid native LoRA layer, rank or alpha".into());
        }
        let count = product(
            &[2, self.config.width, rank],
            self.limits.max_parameter_elements,
            "LoRA parameters",
        )? / 2;
        if count > MAX_LORA_JSON_BYTES / 32 {
            return Err("native LoRA initialization exceeds artifact budget".into());
        }
        // A fixed seeded stream is reproducible without restricting A to a
        // rank-two row/column pattern. B remains zero for exact base behavior.
        let mut rng = ChaCha8Rng::seed_from_u64(0x5354_4c4f_5241_0001);
        let record = LoraRecord {
            schema: "spiraltorch.native_lora.attention_output.v1".into(),
            base_model_identity: self.base_id.clone(),
            layer,
            width: self.config.width,
            rank,
            alpha,
            a: (0..count)
                .map(|_| rng.gen_range(-0.01f32..0.01f32))
                .collect(),
            b: vec![0.; count],
        };
        let mut next = self.clone();
        next.lora = Some(Arc::new(NativeLoraAdapter::create(record, self)?));
        next.validate_adaptation_placement()?;
        Ok(next)
    }

    pub fn lora_artifact_id(&self) -> Option<&str> {
        self.lora.as_ref().map(|adapter| adapter.id.as_str())
    }

    /// Adapter-only JSON; base tensors, token cursor, learning-rate schedule
    /// and readout arithmetic policy are not embedded. Reconstruct the same
    /// policy, windows and schedule for exact continuation. Stateless SGD has
    /// no hidden optimizer moments.
    pub fn lora_to_json(&self) -> CausalLmResult<String> {
        Ok(serde_json::to_string(
            &self
                .lora
                .as_ref()
                .ok_or("native LM has no LoRA adapter")?
                .record,
        )?)
    }

    /// Restore or replace an adapter, checking the frozen checkpoint identity.
    pub fn with_lora_json(&self, json: &str) -> CausalLmResult<Self> {
        if json.len() > MAX_LORA_JSON_BYTES {
            return Err("native LoRA artifact exceeds byte budget".into());
        }
        let record = serde_json::from_str(json)?;
        let mut next = self.clone();
        next.lora = Some(Arc::new(NativeLoraAdapter::create(record, self)?));
        next.validate_adaptation_placement()?;
        Ok(next)
    }

    pub(super) fn project_attention_output(
        &self,
        index: usize,
        input: &Tensor,
    ) -> CausalLmResult<Tensor> {
        let mut output = self.blocks[index].output.forward(input)?;
        if let Some(adapter) = self
            .lora
            .as_ref()
            .filter(|a| a.record.layer == index && !a.zero_delta)
        {
            let delta = low_rank_forward(input, &adapter.a, &adapter.b, MatmulBackend::CpuSimd)?;
            output.add_scaled_with_backend(
                &delta,
                adapter.record.alpha / adapter.record.rank as f32,
                TensorUtilBackend::Cpu,
            )?;
        }
        Ok(output)
    }

    /// Bound checkpoints before clients copy token IDs.
    pub fn lora_gradient_input_count(&self, shape: [usize; 2]) -> CausalLmResult<usize> {
        let adapter = self.lora.as_ref().ok_or("native LM has no LoRA adapter")?;
        let count = self.loss_input_count(shape)?;
        product(
            &[
                self.config.layers - adapter.record.layer + 1,
                shape[0],
                shape[1] - 1,
                self.config.width,
            ],
            self.limits.max_training_checkpoint_elements,
            "LoRA training checkpoints",
        )?;
        Ok(count)
    }

    /// Mean next-token CE through the existing native decoder. No auxiliary
    /// repetition objective, masked labels or geometry co-training in this slice.
    pub fn lora_loss_gradients(
        &self,
        windows: &[usize],
        window_shape: [usize; 2],
    ) -> CausalLmResult<NativeCausalLmLoraGradients> {
        if self.geometry.is_some() {
            return Err("use adaptation_loss_gradients for a composed model".into());
        }
        self.lora_gradient_input_count(window_shape)?;
        let adapter = self.lora.as_ref().unwrap();
        let prepared =
            self.prepare_objective(windows, window_shape, None, CausalLmObjective::default())?;
        let shape = prepared.shape;
        let mut hidden = self.embed(&prepared.input, shape)?;
        let mut checkpoints = Vec::with_capacity(self.config.layers - adapter.record.layer + 1);
        for index in 0..self.config.layers {
            if index >= adapter.record.layer {
                checkpoints.push(hidden.clone());
            }
            hidden = self.run_block(index, hidden, shape, false)?.output;
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
        let d_normalized = super::learning::frozen_pullback(
            &self.head,
            &self.backward_head,
            &d_logits,
            self.readout_accumulator,
        )?;
        let parameters = self.decoder_parameter_pullback(
            checkpoints,
            hidden.shape().0,
            hidden,
            shape,
            None,
            d_normalized,
            adapter.record.layer,
            "",
            true,
        )?;
        let values = parameters
            .get(&adapter.path())
            .ok_or("missing LoRA loss gradient")?;
        if !loss.is_finite() || values.values().flatten().any(|v| !v.is_finite()) {
            return Err("nonfinite native LoRA loss or gradients".into());
        }
        Ok(NativeCausalLmLoraGradients {
            base_id: self.base_id.clone(),
            adapter_id: adapter.id.clone(),
            readout_accumulator: self.readout_accumulator,
            loss,
            target_tokens: prepared.target_tokens,
            shape: window_shape,
            a: values["a"].clone(),
            b: values["b"].clone(),
        })
    }

    /// Atomic stateless SGD: returns a new model; original base and adapter stay
    /// unchanged. A stale gradient or different readout policy is rejected.
    pub fn step_lora_sgd(
        &self,
        gradients: &NativeCausalLmLoraGradients,
        learning_rate: f32,
    ) -> CausalLmResult<Self> {
        if self.geometry.is_some() {
            return Err("use step_adaptation_sgd for a composed model".into());
        }
        let adapter = self.lora.as_ref().ok_or("native LM has no LoRA adapter")?;
        if gradients.base_id != self.base_id
            || gradients.adapter_id != adapter.id
            || gradients.readout_accumulator != self.readout_accumulator
            || !learning_rate.is_finite()
            || learning_rate <= 0.
        {
            return Err("invalid LoRA SGD learning rate or gradient owner".into());
        }
        self.with_lora_parameter_step(&gradients.a, &gradients.b, learning_rate)
    }

    pub(super) fn with_lora_parameter_step(
        &self,
        a: &[f32],
        b: &[f32],
        learning_rate: f32,
    ) -> CausalLmResult<Self> {
        let adapter = self.lora.as_ref().ok_or("native LM has no LoRA adapter")?;
        let mut record = adapter.record.clone();
        for (values, gradient) in [(&mut record.a, a), (&mut record.b, b)] {
            if values.len() != gradient.len() {
                return Err("LoRA gradient shape differs".into());
            }
            for (value, &delta) in values.iter_mut().zip(gradient) {
                *value -= learning_rate * delta;
                if !value.is_finite() {
                    return Err("nonfinite LoRA SGD update".into());
                }
            }
        }
        let mut next = self.clone();
        next.lora = Some(Arc::new(NativeLoraAdapter::create(record, self)?));
        next.validate_adaptation_placement()?;
        Ok(next)
    }
}
