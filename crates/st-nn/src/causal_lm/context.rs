//! Owned single-sequence KV state. Frozen model/geometry travel with the cache.
//! Updates are built separately and published only after successful execution.
use super::*;
use crate::geometry_runtime::GeometrySequenceDependency;
use st_kernel_contracts::attention::{attention_reference, AttentionMask, AttentionSpec};

#[derive(Clone, Debug)]
pub(super) struct InferenceBlockCache {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub history: [Option<Tensor>; 3],
}

/// Non-serializable acceleration state, never an authority for learned weights.
/// Causal sequence geometry currently replays its retained input prefix; row-local
/// geometry only evaluates new rows. Old K/V values are not recomputed.
#[derive(Clone, Debug)]
pub struct NativeCausalLmContext {
    model: NativeCausalLm,
    max_cache_values: usize,
    tokens: Vec<usize>,
    blocks: Vec<Arc<InferenceBlockCache>>,
    logits: Tensor,
}

impl NativeCausalLm {
    pub(super) fn geometry_needs_history(&self, path: &str) -> CausalLmResult<bool> {
        let Some(runtime) = &self.geometry else {
            return Ok(false);
        };
        if !runtime
            .artifact()
            .record()
            .placements
            .iter()
            .any(|p| p.path == path)
        {
            return Ok(false);
        }
        Ok(runtime.sequence_dependency(path)? == GeometrySequenceDependency::CausalPrefix)
    }

    fn incremental_geometry(
        &self,
        path: &str,
        row: Tensor,
        previous: Option<&Tensor>,
        steps: usize,
    ) -> CausalLmResult<(Tensor, Option<Tensor>)> {
        if !self.geometry_needs_history(path)? {
            return Ok((self.geometry_output(path, row, [1, 1], false)?.0, None));
        }
        let previous = previous.ok_or("missing causal geometry history")?;
        if previous.shape() != (steps - 1, self.config.width) {
            return Err("invalid causal geometry history shape".into());
        }
        let mut values = Vec::new();
        values.try_reserve_exact(steps * self.config.width)?;
        values.extend_from_slice(previous.data());
        values.extend_from_slice(row.data());
        let input = Tensor::from_vec(steps, self.config.width, values)?;
        let output = self
            .geometry_output(path, input.clone(), [1, steps], false)?
            .0;
        Ok((output.gather_rows(&[steps - 1])?, Some(input)))
    }

    /// Persistent f32 K/V and geometry input history, excluding frozen weights,
    /// readout and transient old/new transaction buffers. Checked before copying IDs.
    pub fn context_cache_values(
        &self,
        steps: usize,
        max_cache_values: usize,
    ) -> CausalLmResult<usize> {
        if steps == 0
            || steps > self.config.positions
            || max_cache_values == 0
            || max_cache_values > isize::MAX as usize / 4
        {
            return Err("invalid cached context length or value budget".into());
        }
        let histories = if let Some(runtime) = &self.geometry {
            runtime
                .artifact()
                .record()
                .placements
                .iter()
                .try_fold(0usize, |count, p| {
                    Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
                        count
                            + usize::from(
                                runtime.sequence_dependency(&p.path)?
                                    == GeometrySequenceDependency::CausalPrefix,
                            ),
                    )
                })?
        } else {
            0
        };
        product(
            &[2 * self.config.layers + histories, steps, self.config.width],
            max_cache_values,
            "KV and geometry cache",
        )
    }

    /// One-query cached work, not the old square full-prefix attention budget.
    /// Sequence geometry retains its own full-prefix admission and work limits.
    pub fn cached_input_count(
        &self,
        steps: usize,
        max_cache_values: usize,
    ) -> CausalLmResult<usize> {
        self.context_cache_values(steps, max_cache_values)?;
        self.input_count([1, 1], true)?;
        product(
            &[self.config.layers, self.config.heads, steps],
            self.limits.max_attention_pairs,
            "cached attention pairs",
        )?;
        if let Some(runtime) = &self.geometry {
            for placement in &runtime.artifact().record().placements {
                if runtime.sequence_dependency(&placement.path)?
                    == GeometrySequenceDependency::CausalPrefix
                {
                    runtime.sequence_dimensions(&placement.path, [1, steps])?;
                }
            }
        }
        Ok(steps)
    }

    pub fn context_prefill_count(
        &self,
        steps: usize,
        max_cache_values: usize,
    ) -> CausalLmResult<usize> {
        self.context_cache_values(steps, max_cache_values)?;
        self.input_count([1, steps], true)
    }

    fn context_readout(&self, hidden: &Tensor) -> CausalLmResult<Tensor> {
        let normalized = self.final_norm.forward(hidden, self.config.norm_epsilon)?;
        let logits =
            normalized.matmul_prepacked_with_backend(&self.head, MatmulBackend::CpuSimd)?;
        if logits.data().iter().any(|v| !v.is_finite()) {
            return Err("cached readout produced non-finite logits".into());
        }
        Ok(logits)
    }
}

fn append_head_rows(previous: &[f32], row: &[f32], heads: usize) -> CausalLmResult<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(previous.len() + row.len())?;
    let old_stride = previous.len() / heads;
    let width = row.len() / heads;
    for h in 0..heads {
        values.extend_from_slice(&previous[h * old_stride..(h + 1) * old_stride]);
        values.extend_from_slice(&row[h * width..(h + 1) * width]);
    }
    Ok(values)
}

impl NativeCausalLmContext {
    pub const DEFAULT_MAX_CACHE_VALUES: usize = 16_777_216;

    pub fn new(
        model: NativeCausalLm,
        prompt: Vec<usize>,
        max_cache_values: usize,
    ) -> CausalLmResult<Self> {
        model.context_prefill_count(prompt.len(), max_cache_values)?;
        let mut hidden = model.embed(&prompt, [1, prompt.len()])?;
        let mut blocks = Vec::with_capacity(model.config.layers);
        for index in 0..model.config.layers {
            let result = model.prefill_block(index, hidden, prompt.len())?;
            hidden = result.output;
            blocks.push(Arc::new(
                result.inference_cache.ok_or("missing prefill KV state")?,
            ));
        }
        let logits = model.context_readout(&hidden.gather_rows(&[prompt.len() - 1])?)?;
        Ok(Self {
            model,
            max_cache_values,
            tokens: prompt,
            blocks,
            logits,
        })
    }
    pub fn token_ids(&self) -> &[usize] {
        &self.tokens
    }
    pub fn base_model_identity(&self) -> &str {
        self.model.base_model_identity()
    }
    pub fn geometry_artifact_id(&self) -> Option<&str> {
        self.model.geometry_artifact_id()
    }
    pub fn max_cache_values(&self) -> usize {
        self.max_cache_values
    }
    pub fn cache_values(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| {
                block.key.len()
                    + block.value.len()
                    + block
                        .history
                        .iter()
                        .flatten()
                        .map(|t| t.data().len())
                        .sum::<usize>()
            })
            .sum()
    }
    pub fn logits(&self) -> CausalLmResult<NdTensor> {
        Ok(self
            .logits
            .clone()
            .into_nd()?
            .reshape(&[1, 1, self.model.config.vocabulary])?)
    }
    pub(super) fn logits_tensor(&self) -> &Tensor {
        &self.logits
    }

    /// Appending a token does not sample. Failed admission/execution leaves this context untouched.
    pub fn append(&mut self, token: usize) -> CausalLmResult<NdTensor> {
        let next = self.preview(token)?;
        let logits = next.logits()?;
        *self = next;
        Ok(logits)
    }

    pub(super) fn preview(&self, token: usize) -> CausalLmResult<Self> {
        let model = &self.model;
        let c = &model.config;
        let steps = self
            .tokens
            .len()
            .checked_add(1)
            .ok_or("context length overflow")?;
        model.cached_input_count(steps, self.max_cache_values)?;
        if token >= c.vocabulary {
            return Err("cached token is outside vocabulary".into());
        }
        let mut hidden = model.token.gather_rows(&[token])?.add_with_backend(
            &model.position.gather_rows(&[steps - 1])?,
            TensorUtilBackend::Cpu,
        )?;
        let mut blocks = Vec::with_capacity(c.layers);
        for (index, previous) in self.blocks.iter().enumerate() {
            let block = &model.blocks[index];
            let norm = block.first_norm.forward(&hidden, c.norm_epsilon)?;
            let qkv = model.split_heads::<3>(&block.qkv.forward(&norm)?, [1, 1])?;
            let key = append_head_rows(&previous.key, &qkv[1], c.heads)?;
            let value = append_head_rows(&previous.value, &qkv[2], c.heads)?;
            let qshape = [1, c.heads, 1, c.width / c.heads];
            let kshape = [1, c.heads, steps, c.width / c.heads];
            let spec = AttentionSpec::new(
                &qshape,
                &kshape,
                &kshape,
                model.attention_spec(index, [1, 1])?.scale(),
                AttentionMask::Causal {
                    query_offset: steps - 1,
                },
            )?;
            let attended = attention_reference(spec, &qkv[0], &key, &value, None, None)?;
            let attention = block
                .output
                .forward(&model.merge_heads([attended.as_slice()], [1, 1])?)?;
            let path = format!("transformer.h.{index}");
            let (attention, attention_history) = model.incremental_geometry(
                &format!("{path}.attn"),
                attention,
                previous.history[0].as_ref(),
                steps,
            )?;
            let after_attention = attention.add_with_backend(&hidden, TensorUtilBackend::Cpu)?;
            let norm = block
                .second_norm
                .forward(&after_attention, c.norm_epsilon)?;
            let mlp = block
                .contract
                .forward(&block.expand.forward(&norm)?.try_into_gelu()?)?;
            let (mlp, mlp_history) = model.incremental_geometry(
                &format!("{path}.mlp"),
                mlp,
                previous.history[1].as_ref(),
                steps,
            )?;
            let output = after_attention.add_with_backend(&mlp, TensorUtilBackend::Cpu)?;
            let (output, block_history) =
                model.incremental_geometry(&path, output, previous.history[2].as_ref(), steps)?;
            hidden = output;
            blocks.push(Arc::new(InferenceBlockCache {
                key,
                value,
                history: [attention_history, mlp_history, block_history],
            }));
        }
        let logits = model.context_readout(&hidden)?;
        let mut tokens = self.tokens.clone();
        tokens.try_reserve(1)?;
        tokens.push(token);
        Ok(Self {
            model: model.clone(),
            max_cache_values: self.max_cache_values,
            tokens,
            blocks,
            logits,
        })
    }
}
