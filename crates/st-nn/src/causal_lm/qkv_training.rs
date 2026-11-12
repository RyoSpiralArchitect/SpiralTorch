//! One unfrozen QKV matrix. Prefix, output projection, decoder head and attached adapters stay frozen.
use super::*;
use crate::language_windows::CausalDocumentWindows;
use sha2::{Digest, Sha256};
use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep};
use st_kernel_contracts::gradient_clip::GlobalNormClip;
use st_tensor::PackedLayout;

mod batching;
mod causal;
pub use causal::CausalLmQkvWindowStep;
mod documents;
pub use documents::CausalLmQkvDocumentStep;
mod session;
pub use session::{
    CausalLmQkvTrainingConfig, CausalLmQkvTrainingResume, CausalLmQkvTrainingStep,
    NativeCausalLmQkvTrainer,
};

const MAX_VALUES: usize = 1 << 22;
const MAX_HEADER: usize = 16384;

#[derive(Clone, Debug)]
pub struct NativeCausalLmQkvAdamW {
    model: NativeCausalLm,
    layer: usize,
    config: AdamWConfig,
    clip: Option<GlobalNormClip>,
    steps: u32,
    first: Vec<f32>,
    second: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmQkvStep {
    pub steps: u32,
    pub gold_nll: f32,
    /// Q, K, V weight-gradient norms, in that order.
    pub gradient_l2: [f64; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_clip: Option<CausalLmQkvGradientClip>,
}

/// Effective gradient before AdamW, not a bound on the final parameter update.
#[derive(Clone, Debug, Serialize)]
pub struct CausalLmQkvGradientClip {
    pub max_norm: f32,
    pub before_l2: f64,
    pub after_l2: f64,
    pub factors: Vec<f32>,
}

/// Smoothed training CE is not the gold-token NLL used in evaluation.
#[derive(Clone, Debug, Serialize)]
pub struct CausalLmQkvSmoothedStep {
    pub steps: u32,
    pub objective_loss: f32,
    pub label_smoothing: f64,
    pub gradient_l2: [f64; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_clip: Option<CausalLmQkvGradientClip>,
}

/// Read-only corpus score bound to the exact trained QKV projection.
#[derive(Clone, Debug, Serialize)]
pub struct CausalLmQkvEvaluation {
    pub schema: &'static str,
    pub serving_id: String,
    pub layer: usize,
    pub steps: u32,
    pub evaluation: CausalLmEvaluation,
}

/// Ordered source/QKV scores; no layer at or after the trained QKV is shared.
#[derive(Debug, Serialize)]
pub struct CausalLmQkvComparison {
    pub schema: &'static str,
    pub serving_id: String,
    pub layer: usize,
    pub steps: u32,
    pub comparison: CausalLmEvaluationComparison,
}

/// Answer-level source/QKV scores bound to the same trained serving projection.
#[derive(Debug, Serialize)]
pub struct CausalLmQkvAnswerComparison {
    pub schema: &'static str,
    pub serving_id: String,
    pub layer: usize,
    pub steps: u32,
    pub comparison: CausalLmAnswerComparison,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Header {
    schema: String,
    base: String,
    lora: Option<String>,
    geometry: Option<String>,
    readout: CausalLmReadoutAccumulator,
    layer: usize,
    width: usize,
    steps: u32,
    adam_bits: [u32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_gradient_norm_bits: Option<u32>,
}

impl NativeCausalLmQkvAdamW {
    pub const MAX_CHECKPOINT_BYTES: usize = 4 + MAX_HEADER + 12 * MAX_VALUES;
    /// Publish a new optimizer snapshot without overwriting an existing name.
    pub fn save_checkpoint_new(&self, path: impl AsRef<std::path::Path>) -> CausalLmResult<String> {
        super::checkpoint::admit_new(path.as_ref())?;
        super::checkpoint::save(path.as_ref(), self.checkpoint()?)
    }
    fn size(model: &NativeCausalLm, layer: usize) -> CausalLmResult<usize> {
        let w = model.config.width;
        let n = product(
            &[3, w, w],
            MAX_VALUES.min(model.limits.max_parameter_elements),
            "trainable QKV weights",
        )?;
        let weight = &model
            .blocks
            .get(layer)
            .ok_or("QKV layer out of range")?
            .qkv
            .weight;
        if weight.layout() != PackedLayout::ColMajor
            || (weight.inner(), weight.cols()) != (w, 3 * w)
            || weight.as_slice().len() != n
        {
            return Err("unexpected packed QKV layout".into());
        }
        Ok(n)
    }
    pub fn new(base: &NativeCausalLm, layer: usize, config: AdamWConfig) -> CausalLmResult<Self> {
        Self::new_with_gradient_clip(base, layer, config, None)
    }

    /// Apply the shared global norm rule once, after the logical batch mean.
    pub fn new_with_gradient_clip(
        base: &NativeCausalLm,
        layer: usize,
        config: AdamWConfig,
        clip: Option<GlobalNormClip>,
    ) -> CausalLmResult<Self> {
        let n = Self::size(base, layer)?;
        Ok(Self {
            model: base.clone(),
            layer,
            config,
            clip,
            steps: 0,
            first: vec![0.; n],
            second: vec![0.; n],
        })
    }
    pub fn steps(&self) -> u32 {
        self.steps
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.clip.map(GlobalNormClip::max_norm)
    }
    /// Output-major [3 * width, width], not the transposed HF Conv1D archive orientation.
    pub fn weights(&self) -> &[f32] {
        self.model.blocks[self.layer].qkv.weight.as_slice()
    }
    /// Bound a serving prefix before a binding materializes its token IDs.
    pub fn last_logits_input_count(&self, steps: usize) -> CausalLmResult<usize> {
        self.model.input_count([1, steps], true)
    }
    pub fn last_logits(&self, prompt: &[usize]) -> CausalLmResult<Vec<f32>> {
        Ok(self
            .model
            .logits(prompt, [1, prompt.len()], true)?
            .data()
            .to_vec())
    }
    pub fn serving_id(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"spiraltorch.qkv_projection.f32.v1\0");
        for s in [
            self.model.base_model_identity(),
            self.model.lora_artifact_id().unwrap_or(""),
            self.model.geometry_artifact_id().unwrap_or(""),
            self.model.readout_accumulator.as_str(),
        ] {
            h.update((s.len() as u64).to_le_bytes());
            h.update(s.as_bytes());
        }
        h.update((self.layer as u64).to_le_bytes());
        for &v in self.weights() {
            h.update(v.to_le_bytes());
        }
        format!("sha256:{:x}", h.finalize())
    }

    /// Admit a read-only held-out corpus against the current projection before
    /// a foreign client copies token IDs or labels.
    pub fn evaluation_plan(
        &self,
        lengths: &[usize],
        config: CausalLmEvaluationConfig,
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        self.model.evaluation_plan(lengths, config, has_labels)
    }

    /// Score the current immutable projection with the shared target-weighted
    /// evaluator. This never updates weights or Adam state.
    pub fn evaluate_documents(
        &self,
        documents: &[Vec<usize>],
        labels: Option<&[Vec<i64>]>,
        config: CausalLmEvaluationConfig,
    ) -> CausalLmResult<CausalLmQkvEvaluation> {
        let evaluation = self.model.evaluate_documents(documents, labels, config)?;
        Ok(CausalLmQkvEvaluation {
            schema: "spiraltorch.native_causal_lm_qkv_evaluation.v1",
            serving_id: self.serving_id(),
            layer: self.layer,
            steps: self.steps,
            evaluation,
        })
    }

    fn ensure_source(&self, source: &NativeCausalLm) -> CausalLmResult<()> {
        if source.base_model_identity() != self.model.base_model_identity()
            || source.lora_artifact_id() != self.model.lora_artifact_id()
            || source.geometry_artifact_id() != self.model.geometry_artifact_id()
            || source.readout_accumulator() != self.model.readout_accumulator()
        {
            return Err("QKV comparison requires the exact frozen source owner".into());
        }
        Ok(())
    }

    pub fn source_comparison_plan(
        &self,
        source: &NativeCausalLm,
        lengths: &[usize],
        config: CausalLmEvaluationConfig,
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        self.ensure_source(source)?;
        NativeCausalLm::comparison_plan(&[source, &self.model], lengths, config, has_labels)
    }

    pub fn source_answer_comparison_plan(
        &self,
        source: &NativeCausalLm,
        lengths: &[usize],
        config: CausalLmEvaluationConfig,
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        self.ensure_source(source)?;
        NativeCausalLm::answer_comparison_plan(&[source, &self.model], lengths, config, has_labels)
    }

    /// Evaluate the frozen source and this snapshot in one pass. The QKV layer
    /// itself and every later block execute separately even without adapters.
    pub fn evaluate_against_source(
        &self,
        source: &NativeCausalLm,
        documents: &[Vec<usize>],
        labels: Option<&[Vec<i64>]>,
        config: CausalLmEvaluationConfig,
    ) -> CausalLmResult<CausalLmQkvComparison> {
        self.ensure_source(source)?;
        let comparison = NativeCausalLm::evaluate_variants_with_prefix_cap(
            &[source, &self.model],
            documents,
            labels,
            config,
            self.layer,
        )?;
        Ok(CausalLmQkvComparison {
            schema: "spiraltorch.native_causal_lm_qkv_comparison.v1",
            serving_id: self.serving_id(),
            layer: self.layer,
            steps: self.steps,
            comparison,
        })
    }

    /// Retain contiguous supervised-answer scores without sharing the trained layer.
    pub fn evaluate_against_source_by_answer(
        &self,
        source: &NativeCausalLm,
        documents: &[Vec<usize>],
        labels: Option<&[Vec<i64>]>,
        config: CausalLmEvaluationConfig,
    ) -> CausalLmResult<CausalLmQkvAnswerComparison> {
        self.ensure_source(source)?;
        let comparison = NativeCausalLm::evaluate_variants_by_answer_with_prefix_cap(
            &[source, &self.model],
            documents,
            labels,
            config,
            self.layer,
        )?;
        Ok(CausalLmQkvAnswerComparison {
            schema: "spiraltorch.native_causal_lm_qkv_answer_comparison.v1",
            serving_id: self.serving_id(),
            layer: self.layer,
            steps: self.steps,
            comparison,
        })
    }
    /// Admit a snapshot's cache before a binding copies its prefix IDs.
    pub fn prefill_input_count(
        &self,
        steps: usize,
        max_cache_values: usize,
    ) -> CausalLmResult<usize> {
        self.model.context_prefill_count(steps, max_cache_values)
    }
    pub fn prefill(
        &self,
        prompt: Vec<usize>,
        budget: usize,
    ) -> CausalLmResult<NativeCausalLmProjectionPrefix> {
        self.prefill_input_count(prompt.len(), budget)?;
        NativeCausalLmProjectionPrefix::from_snapshot(
            &self.model,
            self.serving_id(),
            prompt,
            budget,
        )
    }

    /// Admit the complete generation budget before a binding reads prompt IDs.
    pub fn generation_input_count(
        &self,
        steps: usize,
        config: &CausalLmGenerationConfig,
    ) -> CausalLmResult<usize> {
        config.validate(&self.model, steps)?;
        Ok(steps)
    }

    /// Capture this projection's weights; later optimizer updates cannot change the generator.
    pub fn generator(
        &self,
        prompt: Vec<usize>,
        config: CausalLmGenerationConfig,
    ) -> CausalLmResult<NativeCausalLmGenerator> {
        self.generation_input_count(prompt.len(), &config)?;
        NativeCausalLmGenerator::new_owned(
            self.model.clone(),
            prompt,
            config,
            Some(self.serving_id()),
        )
    }

    /// Restore tokens/RNG only when they belong to these exact serving weights.
    /// Restore the optimizer checkpoint first to select an older projection.
    pub fn restore_generator(
        &self,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<NativeCausalLmGenerator> {
        NativeCausalLmGenerator::from_json_owned(
            self.model.clone(),
            bytes,
            expected_id,
            Some(self.serving_id()),
        )
    }
    /// Full-vocabulary CE on the last row of each unpadded prefix; labels are never model inputs.
    /// Returns the mean loss and its [3 * width, width] QKV weight derivative.
    pub fn answer_gradients(
        &self,
        ids: &[usize],
        shape: [usize; 2],
        gold: &[usize],
    ) -> CausalLmResult<(f32, Tensor)> {
        self.answer_gradients_smoothed(ids, shape, gold, 0.)
    }

    fn answer_gradients_smoothed(
        &self,
        ids: &[usize],
        shape: [usize; 2],
        gold: &[usize],
        label_smoothing: f64,
    ) -> CausalLmResult<(f32, Tensor)> {
        let config = CrossEntropyConfig {
            label_smoothing,
            ..CrossEntropyConfig::default()
        };
        config.validate()?;
        let m = &self.model;
        let c = &m.config;
        let [batch, steps] = shape;
        let rows = self.answer_gradient_rows(shape)?;
        if ids.len() != rows || gold.len() != batch || gold.iter().any(|&v| v >= c.vocabulary) {
            return Err("invalid QKV answer roster".into());
        }
        let (hidden, checkpoints) =
            m.checkpointed_hidden(m.embed(ids, shape)?, shape, 0, self.layer, None)?;
        let last = (0..batch).map(|b| (b + 1) * steps - 1).collect::<Vec<_>>();
        let selected = hidden.gather_rows(&last)?;
        let logits = m.project_readout(&m.final_norm.forward(&selected, c.norm_epsilon)?)?;
        let labels = gold.iter().map(|&t| t as i64).collect::<Vec<_>>();
        let (loss, seed) = learning::ordinary_loss_seed(&logits, &labels, config)?;
        let mut gradient = None;
        m.decoder_parameter_pullback(
            checkpoints,
            rows,
            selected,
            shape,
            Some(&last),
            m.vocabulary_head_pullback(&seed)?,
            self.layer,
            learning::DecoderTarget::QkvWeight(&mut gradient),
            None,
        )?;
        let gradient = gradient.ok_or("unreached QKV derivative")?;
        self.validate_derivative(loss, &gradient)?;
        Ok((loss, gradient))
    }

    fn validate_derivative(&self, loss: f32, gradient: &Tensor) -> CausalLmResult<()> {
        let c = &self.model.config;
        if !loss.is_finite()
            || gradient.shape() != (3 * c.width, c.width)
            || gradient.data().iter().any(|v| !v.is_finite())
        {
            return Err("nonfinite or misshaped QKV derivative".into());
        }
        Ok(())
    }

    fn answer_gradient_rows(&self, shape: [usize; 2]) -> CausalLmResult<usize> {
        let m = &self.model;
        let rows = m.input_count(shape, true)?;
        self.admit_gradient_rows(rows)?;
        Ok(rows)
    }

    fn admit_gradient_rows(&self, rows: usize) -> CausalLmResult<()> {
        let m = &self.model;
        product(
            &[m.config.layers - self.layer, rows, m.config.width],
            m.limits.max_training_checkpoint_elements,
            "QKV training checkpoints",
        )?;
        product(
            &[rows, 3, m.config.width, m.config.width],
            1 << 30,
            "QKV gradient work",
        )?;
        Ok(())
    }
    fn install(&mut self, weights: Vec<f32>) -> CausalLmResult<()> {
        let w = self.model.config.width;
        let dense = &mut self.model.blocks[self.layer].qkv;
        let packed = PackedB::from_tensor_transpose(
            &Tensor::from_vec(3 * w, w, weights)?,
            dense.weight.tile(),
        )?;
        dense.weight = packed;
        dense.backward_weight = Default::default();
        Ok(())
    }

    /// Keep entire sequences intact. Each microbatch yields a mean gradient;
    /// weight it by its sample count before applying a single optimizer update.
    pub fn answer_gradients_microbatched(
        &self,
        ids: &[usize],
        shape: [usize; 2],
        gold: &[usize],
        max_rows: usize,
    ) -> CausalLmResult<(f32, Tensor)> {
        let [batch, steps] = shape;
        let c = &self.model.config;
        if batch == 0
            || steps == 0
            || max_rows == 0
            || batch.div_ceil(max_rows) > 1024
            || batch.checked_mul(steps) != Some(ids.len())
            || gold.len() != batch
            || ids.iter().chain(gold).any(|&id| id >= c.vocabulary)
        {
            return Err("invalid logical QKV batch or microbatch limit".into());
        }
        product(
            &[ids.len(), 3, c.width, c.width],
            1 << 30,
            "logical QKV gradient work",
        )?;
        if max_rows >= batch {
            return self.answer_gradients(ids, shape, gold);
        }
        let mut mean = batching::WeightedGradientMean::new(c.width);
        for first in (0..batch).step_by(max_rows) {
            let end = first + max_rows.min(batch - first);
            let n = end - first;
            let (v, g) = self.answer_gradients(
                &ids[first * steps..end * steps],
                [n, steps],
                &gold[first..end],
            )?;
            mean.add(v, &g, n);
        }
        mean.finish()
    }
    /// Atomic checked AdamW, reusing the kernel contract including the zero-rate freeze policy.
    pub fn step_answers(
        &mut self,
        ids: &[usize],
        shape: [usize; 2],
        gold: &[usize],
        step: AdamWStep,
    ) -> CausalLmResult<CausalLmQkvStep> {
        let (loss, gradient) = self.answer_gradients(ids, shape, gold)?;
        self.apply_answer_gradient(loss, gradient, step)
    }

    /// Update once per logical batch, only after every microbatch completes successfully.
    pub fn step_answers_microbatched(
        &mut self,
        ids: &[usize],
        shape: [usize; 2],
        gold: &[usize],
        max_rows: usize,
        step: AdamWStep,
    ) -> CausalLmResult<CausalLmQkvStep> {
        let (loss, gradient) = self.answer_gradients_microbatched(ids, shape, gold, max_rows)?;
        self.apply_answer_gradient(loss, gradient, step)
    }

    fn apply_answer_gradient(
        &mut self,
        loss: f32,
        gradient: Tensor,
        step: AdamWStep,
    ) -> CausalLmResult<CausalLmQkvStep> {
        let (steps, gradient_l2, gradient_clip) = self.apply_gradient(gradient, step)?;
        Ok(CausalLmQkvStep {
            steps,
            gold_nll: loss,
            gradient_l2,
            gradient_clip,
        })
    }

    fn apply_gradient(
        &mut self,
        gradient: Tensor,
        step: AdamWStep,
    ) -> CausalLmResult<(u32, [f64; 3], Option<CausalLmQkvGradientClip>)> {
        let w = self.model.config.width;
        let gradient_l2 = std::array::from_fn(|part| {
            gradient.data()[part * w * w..(part + 1) * w * w]
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                .sqrt()
        });
        let mut clipped = None;
        let gradient_clip = self
            .clip
            .map(|clip| -> CausalLmResult<_> {
                let squared = gradient.data().iter().map(|&g| f64::from(g).powi(2)).sum();
                let factors = clip.factors(squared)?;
                let mut after_l2 = squared.sqrt();
                if !factors.as_slice().is_empty() {
                    let mut values = gradient.data().to_vec();
                    for &factor in factors.as_slice() {
                        for g in &mut values {
                            *g *= factor;
                        }
                    }
                    after_l2 = values
                        .iter()
                        .map(|&g| f64::from(g).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    clipped = Some(values);
                }
                Ok(CausalLmQkvGradientClip {
                    max_norm: clip.max_norm(),
                    before_l2: squared.sqrt(),
                    after_l2,
                    factors: factors.as_slice().to_vec(),
                })
            })
            .transpose()?;
        let effective = clipped.as_deref().unwrap_or_else(|| gradient.data());
        if step.rate() != 0. {
            let n = self.weights().len();
            let (mut weights, mut first, mut second) = (
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            );
            let clock = self
                .steps
                .checked_add(1)
                .ok_or("QKV AdamW clock exhausted")?;
            for (i, &g) in effective.iter().enumerate() {
                let v = step.candidate(
                    self.config,
                    self.weights()[i],
                    g,
                    self.first[i],
                    self.second[i],
                    self.steps,
                )?;
                weights.push(v.value);
                first.push(v.first);
                second.push(v.second);
            }
            self.install(weights)?;
            self.first = first;
            self.second = second;
            self.steps = clock;
        }
        Ok((self.steps, gradient_l2, gradient_clip))
    }

    /// Bounded little-endian weights and moments with a source-bound header and separately pinned digest.
    pub fn checkpoint(&self) -> CausalLmResult<(Vec<u8>, String)> {
        let h = serde_json::to_vec(&Header {
            schema: "spiraltorch.qkv_adamw.v1".into(),
            base: self.model.base_id.clone(),
            lora: self.model.lora_artifact_id().map(str::to_owned),
            geometry: self.model.geometry_artifact_id().map(str::to_owned),
            readout: self.model.readout_accumulator,
            layer: self.layer,
            width: self.model.config.width,
            steps: self.steps,
            adam_bits: [
                self.config.beta1(),
                self.config.beta2(),
                self.config.epsilon(),
            ]
            .map(f32::to_bits),
            max_gradient_norm_bits: self.max_gradient_norm().map(f32::to_bits),
        })?;
        if h.len() > MAX_HEADER {
            return Err("QKV checkpoint header exceeds limit".into());
        }
        let mut out = Vec::with_capacity(4 + h.len() + 12 * self.weights().len());
        out.extend_from_slice(&(h.len() as u32).to_le_bytes());
        out.extend(h);
        for &v in self.weights().iter().chain(&self.first).chain(&self.second) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let pin = format!("sha256:{:x}", Sha256::digest(&out));
        Ok((out, pin))
    }
    pub fn restore(base: &NativeCausalLm, bytes: &[u8], pin: &str) -> CausalLmResult<Self> {
        if !(4..=Self::MAX_CHECKPOINT_BYTES).contains(&bytes.len())
            || format!("sha256:{:x}", Sha256::digest(bytes)) != pin
        {
            return Err("QKV checkpoint size or digest mismatch".into());
        }
        let n = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
        if n > MAX_HEADER || bytes.len() < 4 + n {
            return Err("invalid QKV header size".into());
        }
        let h: Header = serde_json::from_slice(&bytes[4..4 + n])?;
        if h.schema != "spiraltorch.qkv_adamw.v1"
            || h.base != base.base_id
            || h.readout != base.readout_accumulator
            || h.lora.as_deref() != base.lora_artifact_id()
            || h.geometry.as_deref() != base.geometry_artifact_id()
            || h.width != base.config.width
        {
            return Err("QKV checkpoint owner mismatch".into());
        }
        let count = Self::size(base, h.layer)?;
        let data = &bytes[4 + n..];
        if data.len() != 12 * count {
            return Err("wrong QKV checkpoint payload length".into());
        }
        let [b1, b2, eps] = h.adam_bits.map(f32::from_bits);
        let config = AdamWConfig::new(b1, b2, eps)?;
        let clip = h
            .max_gradient_norm_bits
            .map(|bits| GlobalNormClip::new(f32::from_bits(bits)))
            .transpose()?;
        let decode = |part: usize| {
            data[4 * part * count..4 * (part + 1) * count]
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect::<Vec<_>>()
        };
        let (weights, first, second) = (decode(0), decode(1), decode(2));
        if weights.iter().chain(&first).any(|v| !v.is_finite())
            || second.iter().any(|v| !v.is_finite() || *v < 0.)
            || (h.steps == 0
                && (first.iter().chain(&second).any(|v| *v != 0.)
                    || weights != base.blocks[h.layer].qkv.weight.as_slice()))
        {
            return Err("invalid QKV checkpoint state".into());
        }
        let mut state = Self {
            model: base.clone(),
            layer: h.layer,
            config,
            clip,
            steps: h.steps,
            first,
            second,
        };
        state.install(weights)?;
        Ok(state)
    }
}
