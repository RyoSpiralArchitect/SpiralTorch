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

#[derive(Clone, Copy)]
pub(super) struct KvPrefix<'a> {
    steps: usize,
    blocks: &'a [Arc<InferenceBlockCache>],
}

// A scoring branch needs only K/V and causal geometry history, not a new full
// vocabulary row, copied token history, or a second owner of the model.
pub(super) struct KvBranch {
    steps: usize,
    blocks: Vec<Arc<InferenceBlockCache>>,
}

impl KvBranch {
    pub(super) fn prefix(&self) -> KvPrefix<'_> {
        KvPrefix {
            steps: self.steps,
            blocks: &self.blocks,
        }
    }
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
        rows: Tensor,
        previous: Option<&Tensor>,
        steps: usize,
    ) -> CausalLmResult<(Tensor, Option<Tensor>)> {
        let added = rows.shape().0;
        if !self.geometry_needs_history(path)? {
            return Ok((
                self.geometry_output(path, rows, [1, added], &mut decoder::BlockPass::Forward)?
                    .0,
                None,
            ));
        }
        let previous = previous.ok_or("missing causal geometry history")?;
        let start = steps - added;
        if previous.shape() != (start, self.config.width) {
            return Err("invalid causal geometry history shape".into());
        }
        let mut values = Vec::new();
        values.try_reserve_exact(steps * self.config.width)?;
        values.extend_from_slice(previous.data());
        values.extend_from_slice(rows.data());
        let input = Tensor::from_vec(steps, self.config.width, values)?;
        let output = self
            .geometry_output(
                path,
                input.clone(),
                [1, steps],
                &mut decoder::BlockPass::Forward,
            )?
            .0;
        Ok((
            output.gather_rows(&(start..steps).collect::<Vec<_>>())?,
            Some(input),
        ))
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
        self.cached_query_count(steps, 1, max_cache_values)
    }

    pub(super) fn cached_query_count(
        &self,
        steps: usize,
        added: usize,
        max_cache_values: usize,
    ) -> CausalLmResult<usize> {
        self.context_cache_values(steps, max_cache_values)?;
        self.input_count([1, added], true)?;
        product(
            &[self.config.layers, self.config.heads, added, steps],
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

fn extend_head_rows(previous: &[f32], suffix: &[f32], heads: usize) -> CausalLmResult<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(previous.len() + suffix.len())?;
    let old_stride = previous.len() / heads;
    let added_stride = suffix.len() / heads;
    for h in 0..heads {
        values.extend_from_slice(&previous[h * old_stride..(h + 1) * old_stride]);
        values.extend_from_slice(&suffix[h * added_stride..(h + 1) * added_stride]);
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

    /// The already computed host row; borrowing it does not run the decoder or read a device.
    pub fn logit_values(&self) -> &[f32] {
        self.logits.data()
    }

    pub(super) fn decoder(&self) -> &NativeCausalLm {
        &self.model
    }

    pub(super) fn kv_prefix(&self) -> KvPrefix<'_> {
        KvPrefix {
            steps: self.tokens.len(),
            blocks: &self.blocks,
        }
    }

    /// Admit suffix work before a client extracts IDs. This is rectangular cached attention,
    /// not full-prefix admission and not a promise that many one-token calls fit as one chunk.
    pub fn suffix_input_count(&self, added: usize) -> CausalLmResult<usize> {
        self.suffix_end(self.tokens.len(), added)?;
        Ok(added)
    }

    fn suffix_end(&self, start: usize, added: usize) -> CausalLmResult<usize> {
        let steps = start.checked_add(added).ok_or("context length overflow")?;
        self.model
            .cached_query_count(steps, added, self.max_cache_values)
    }

    /// Fork with a known, nonempty suffix. Only its last position is projected into vocabulary
    /// space. Neither success nor failure changes the source cache or learned parameters.
    pub fn with_suffix(&self, ids: &[usize]) -> CausalLmResult<Self> {
        self.preview_suffix_with_edit(ids, None)
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
        self.preview_suffix_with_edit(&[token], edit)
    }

    fn preview_suffix_with_edit(
        &self,
        suffix: &[usize],
        edit: Option<(usize, usize, &[f32])>,
    ) -> CausalLmResult<Self> {
        let (branch, hidden) = self.suffix_hidden(self.kv_prefix(), suffix, edit)?;
        let last = if suffix.len() == 1 {
            hidden
        } else {
            hidden.gather_rows(&[suffix.len() - 1])?
        };
        self.finish_suffix(suffix, branch.blocks, self.model.context_readout(&last)?)
    }

    /// Decode attention once, then consume every teacher-forced vocabulary row in bounded tiles.
    pub(super) fn visit_scored_suffix(
        &self,
        prefix: KvPrefix<'_>,
        suffix: &[usize],
        tile_rows: usize,
        mut visit: impl FnMut(usize, &Tensor) -> CausalLmResult<()>,
    ) -> CausalLmResult<KvBranch> {
        let (branch, hidden) = self.suffix_hidden(prefix, suffix, None)?;
        for start in (0..suffix.len()).step_by(tile_rows) {
            let n = tile_rows.min(suffix.len() - start);
            let slice;
            let input = if n == suffix.len() {
                &hidden
            } else {
                slice = hidden.gather_rows(&(start..start + n).collect::<Vec<_>>())?;
                &slice
            };
            let rows = self.model.context_readout(input)?;
            if rows.layout() == st_tensor::Layout::RowMajor {
                visit(start, &rows)?;
            } else {
                visit(start, &rows.to_layout(st_tensor::Layout::RowMajor)?)?;
            }
        }
        Ok(branch)
    }

    fn suffix_hidden(
        &self,
        prefix: KvPrefix<'_>,
        suffix: &[usize],
        edit: Option<(usize, usize, &[f32])>,
    ) -> CausalLmResult<(KvBranch, Tensor)> {
        let model = &self.model;
        let c = &model.config;
        let added = suffix.len();
        let start = prefix.steps;
        let steps = self.suffix_end(start, added)?;
        let shape = [1, added];
        let mut hidden = model.embed_at(suffix, shape, start)?;
        #[cfg(all(test, not(target_arch = "wasm32")))]
        decoder::work_counts::suffix(added);
        let mut blocks = Vec::with_capacity(c.layers);
        for (index, previous) in prefix.blocks.iter().enumerate() {
            let block = &model.blocks[index];
            let norm = block.first_norm.forward(&hidden, c.norm_epsilon)?;
            let qkv = model.split_heads::<3>(&block.qkv.forward(&norm)?, shape)?;
            let mut key = extend_head_rows(&previous.key, &qkv[1], c.heads)?;
            if let Some((_, pos, row)) = edit.filter(|(layer, _, _)| *layer == index) {
                let dim = c.width / c.heads;
                for (head, replacement) in
                    key.chunks_exact_mut(steps * dim).zip(row.chunks_exact(dim))
                {
                    head[pos * dim..(pos + 1) * dim].copy_from_slice(replacement);
                }
            }
            let value = extend_head_rows(&previous.value, &qkv[2], c.heads)?;
            let qshape = [1, c.heads, added, c.width / c.heads];
            let kshape = [1, c.heads, steps, c.width / c.heads];
            let spec = AttentionSpec::new(
                &qshape,
                &kshape,
                &kshape,
                model.attention_spec(index, shape)?.scale(),
                AttentionMask::Causal {
                    query_offset: start,
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
                &model.merge_heads([attended.as_slice()], shape)?,
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
        Ok((KvBranch { steps, blocks }, hidden))
    }

    fn finish_suffix(
        &self,
        suffix: &[usize],
        blocks: Vec<Arc<InferenceBlockCache>>,
        logits: Tensor,
    ) -> CausalLmResult<Self> {
        let mut tokens = Vec::new();
        tokens.try_reserve_exact(self.tokens.len() + suffix.len())?;
        tokens.extend_from_slice(&self.tokens);
        tokens.extend_from_slice(suffix);
        Ok(Self {
            model: self.model.clone(),
            max_cache_values: self.max_cache_values,
            tokens,
            blocks,
            logits,
        })
    }
}
