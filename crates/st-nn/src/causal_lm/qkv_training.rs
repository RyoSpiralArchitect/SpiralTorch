//! One unfrozen QKV matrix. Prefix, output projection, decoder head and attached adapters stay frozen.
use super::*;
use sha2::{Digest, Sha256};
use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep};
use st_tensor::PackedLayout;

const MAX_VALUES: usize = 1 << 22;
const MAX_HEADER: usize = 16384;

#[derive(Clone, Debug)]
pub struct NativeCausalLmQkvAdamW {
    model: NativeCausalLm,
    layer: usize,
    config: AdamWConfig,
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
}

impl NativeCausalLmQkvAdamW {
    pub const MAX_CHECKPOINT_BYTES: usize = 4 + MAX_HEADER + 12 * MAX_VALUES;
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
        let n = Self::size(base, layer)?;
        Ok(Self {
            model: base.clone(),
            layer,
            config,
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
    /// Output-major [3 * width, width], not the transposed HF Conv1D archive orientation.
    pub fn weights(&self) -> &[f32] {
        self.model.blocks[self.layer].qkv.weight.as_slice()
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
    pub fn prefill(
        &self,
        prompt: Vec<usize>,
        budget: usize,
    ) -> CausalLmResult<NativeCausalLmProjectionPrefix> {
        NativeCausalLmProjectionPrefix::from_snapshot(
            &self.model,
            self.serving_id(),
            prompt,
            budget,
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
        let m = &self.model;
        let c = &m.config;
        let [batch, steps] = shape;
        let rows = m.input_count(shape, true)?;
        if ids.len() != rows || gold.len() != batch || gold.iter().any(|&v| v >= c.vocabulary) {
            return Err("invalid QKV answer roster".into());
        }
        product(
            &[c.layers - self.layer, rows, c.width],
            m.limits.max_training_checkpoint_elements,
            "QKV training checkpoints",
        )?;
        product(&[rows, 3, c.width, c.width], 1 << 30, "QKV gradient work")?;
        let (hidden, checkpoints) =
            m.checkpointed_hidden(m.embed(ids, shape)?, shape, 0, self.layer, None)?;
        let last = (0..batch).map(|b| (b + 1) * steps - 1).collect::<Vec<_>>();
        let selected = hidden.gather_rows(&last)?;
        let logits = m.project_readout(&m.final_norm.forward(&selected, c.norm_epsilon)?)?;
        let labels = gold.iter().map(|&t| t as i64).collect::<Vec<_>>();
        let (loss, seed) =
            learning::ordinary_loss_seed(&logits, &labels, CrossEntropyConfig::default())?;
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
        if !loss.is_finite()
            || gradient.shape() != (3 * c.width, c.width)
            || gradient.data().iter().any(|v| !v.is_finite())
        {
            return Err("nonfinite or misshaped QKV derivative".into());
        }
        Ok((loss, gradient))
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
        let mut loss = sum::Sum::default();
        let mut accum: Vec<_> = (0..self.weights().len())
            .map(|_| sum::Sum::default())
            .collect();
        for first in (0..batch).step_by(max_rows) {
            let end = first + max_rows.min(batch - first);
            let n = end - first;
            let (v, g) = self.answer_gradients(
                &ids[first * steps..end * steps],
                [n, steps],
                &gold[first..end],
            )?;
            loss.add(f64::from(v) * n as f64);
            for (a, &x) in accum.iter_mut().zip(g.data()) {
                a.add(f64::from(x) * n as f64);
            }
        }
        let grad = accum
            .into_iter()
            .map(|s| s.mean(batch as u64) as f32)
            .collect();
        Ok((
            loss.mean(batch as u64) as f32,
            Tensor::from_vec(3 * c.width, c.width, grad)?,
        ))
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
        let w = self.model.config.width;
        let gradient_l2 = std::array::from_fn(|part| {
            gradient.data()[part * w * w..(part + 1) * w * w]
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                .sqrt()
        });
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
            for (i, &g) in gradient.data().iter().enumerate() {
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
        Ok(CausalLmQkvStep {
            steps: self.steps,
            gold_nll: loss,
            gradient_l2,
        })
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
            steps: h.steps,
            first,
            second,
        };
        state.install(weights)?;
        Ok(state)
    }
}
