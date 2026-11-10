//! Owned single-sequence KV state. Frozen model and adapters travel with the cache.
//! Updates are built separately and published only after successful execution.
use super::*;
use crate::geometry_runtime::GeometrySequenceDependency;
use st_kernel_contracts::attention::{attention_key_lag_reference, AttentionMask, AttentionSpec};

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
            return Ok((
                self.geometry_output(path, row, [1, 1], &mut decoder::BlockPass::Forward)?
                    .0,
                None,
            ));
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
            .geometry_output(
                path,
                input.clone(),
                [1, steps],
                &mut decoder::BlockPass::Forward,
            )?
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
        let logits = self.project_readout(&normalized)?;
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

#[cfg(all(test, not(target_arch = "wasm32")))]
#[test]
fn key_substitution_matches_manual_cache_edit_before_any_append() {
    let m = super::tests::load();
    let (heads, d) = (m.config.heads, m.config.width / m.config.heads);
    let x = NativeCausalLmContext::new(m.clone(), vec![4, 8, 2], 4096).unwrap();
    let mut donor = NativeCausalLmContext::new(m, vec![4, 6, 2], 4096).unwrap();
    let mut manual = x.clone();
    let replacement = [17., -23., 41., -53.];
    for (j, &value) in replacement.iter().enumerate() {
        let offset = ((j / d) * 3 + 2) * d + j % d;
        Arc::make_mut(&mut manual.blocks[1]).key[offset] = value;
        Arc::make_mut(&mut donor.blocks[1]).key[offset] = value;
    }
    let actual = x.preview_with_edit(3, Some((1, 2, &replacement))).unwrap();
    let expected = manual.preview(3).unwrap();
    for layer in 0..2 {
        assert_eq!(actual.blocks[layer].key, expected.blocks[layer].key);
        assert_eq!(actual.blocks[layer].value, expected.blocks[layer].value);
    }
    assert_eq!(actual.key_row(1, 2).unwrap(), replacement);
    assert_eq!(actual.key_row(0, 2).unwrap(), x.key_row(0, 2).unwrap());
    assert_ne!(actual.key_row(1, 2).unwrap(), x.key_row(1, 2).unwrap());
    assert_eq!(
        actual.logits_tensor().data(),
        expected.logits_tensor().data()
    );
    assert_eq!(
        x.probe_key_from(&donor, 1, 2, &[3, 1]).unwrap(),
        expected.preview(1).unwrap().logits_tensor().data()
    );
    assert_eq!(heads * d, replacement.len());
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
            let result = model.run_block(
                index,
                hidden,
                [1, prompt.len()],
                decoder::BlockPass::Prefill,
            )?;
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
    pub fn lora_artifact_id(&self) -> Option<&str> {
        self.model.lora_artifact_id()
    }
    pub fn readout_accumulator(&self) -> CausalLmReadoutAccumulator {
        self.model.readout_accumulator()
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
        self.preview_with_edit(token, None)
    }

    pub(super) fn key_row(&self, layer: usize, position: usize) -> CausalLmResult<Vec<f32>> {
        let key = &self
            .blocks
            .get(layer)
            .ok_or("key layer outside cached prefix")?
            .key;
        let steps = self.tokens.len();
        if position >= steps {
            return Err("key position outside cached prefix".into());
        }
        let dim = self.model.config.width / self.model.config.heads;
        let mut row = Vec::with_capacity(self.model.config.width);
        for head in key.chunks_exact(steps * dim) {
            row.extend_from_slice(&head[position * dim..(position + 1) * dim]);
        }
        Ok(row)
    }

    // The caller has verified the two projection owners. Only future reads of
    // the specified cached key change. No transplanted cache escapes this call.
    pub(super) fn probe_key_from(
        &self,
        donor: &Self,
        layer: usize,
        position: usize,
        suffix: &[usize],
    ) -> CausalLmResult<Vec<f32>> {
        if self.tokens.len() != donor.tokens.len()
            || position >= self.tokens.len()
            || self.tokens[position] != donor.tokens[position]
            || !(1..=64).contains(&suffix.len())
            || suffix.iter().any(|&t| t >= self.model.config.vocabulary)
        {
            return Err(
                "key substitution requires aligned token, length and admitted suffix".into(),
            );
        }
        self.model.cached_input_count(
            self.tokens
                .len()
                .checked_add(suffix.len())
                .ok_or("prefix length overflow")?,
            self.max_cache_values,
        )?;
        let key = donor.key_row(layer, position)?;
        let mut branch = self.preview_with_edit(suffix[0], Some((layer, position, &key)))?;
        for &token in &suffix[1..] {
            branch = branch.preview(token)?;
        }
        Ok(branch.logits_tensor().data().to_vec())
    }

    fn preview_with_edit(
        &self,
        token: usize,
        edit: Option<(usize, usize, &[f32])>,
    ) -> CausalLmResult<Self> {
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
            let mut key = append_head_rows(&previous.key, &qkv[1], c.heads)?;
            if let Some((_, pos, row)) = edit.filter(|(layer, _, _)| *layer == index) {
                let dim = c.width / c.heads;
                for (head, replacement) in
                    key.chunks_exact_mut(steps * dim).zip(row.chunks_exact(dim))
                {
                    head[pos * dim..(pos + 1) * dim].copy_from_slice(replacement);
                }
            }
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
            let attended = attention_key_lag_reference(
                spec,
                &qkv[0],
                &key,
                &value,
                None,
                None,
                model.key_lag_at(index),
            )?;
            let attention = model.project_attention_output(
                index,
                &model.merge_heads([attended.as_slice()], [1, 1])?,
            )?;
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
