use super::*;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

/// Work limits for admission of a single chunked prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalLmChunkedPrefillLimits {
    pub max_chunk_steps: usize,
    pub max_total_attention_pairs: usize,
    /// One unit per elliptic score cell, fractional tap product, or scanned value.
    pub max_total_geometry_work: usize,
}

/// Schedule and conservative work accounting for a single prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalLmChunkedPrefillPlan {
    pub chunk_steps: usize,
    pub chunks: usize,
    pub attention_pairs: usize,
    /// Work of prefix replay or full-prefix value scans, summed over all placements.
    pub geometry_prefix_work: usize,
}

impl NativeCausalLm {
    /// Admit the entire cache and every chunk before executing any decoder work.
    /// Attention charges rectangular pairs across layers, including masked pairs.
    /// Replayed geometry and full-prefix value scans have a separate work limit.
    /// These bounds are not elapsed time or transient allocation estimates.
    /// An oversized schedule is refused rather than changing chunk boundaries.
    pub fn context_chunked_prefill_plan(
        &self,
        steps: usize,
        max_cache_values: usize,
        limits: CausalLmChunkedPrefillLimits,
    ) -> CausalLmResult<CausalLmChunkedPrefillPlan> {
        self.context_cache_values(steps, max_cache_values)?;
        let c = &self.config;
        let heads = product(&[c.layers, c.heads], usize::MAX, "chunked attention heads")?;
        let last_query_cost = product(&[heads, steps], usize::MAX, "chunked attention extent")?;
        let row_cost = c
            .width
            .checked_mul(3)
            .ok_or("chunked activation width overflow")?
            .max(c.intermediate);
        let chunk_steps = steps
            .min(limits.max_chunk_steps)
            .min(self.limits.max_attention_pairs / last_query_cost)
            .min(self.limits.max_activation_elements / row_cost);
        if chunk_steps == 0 || limits.max_total_attention_pairs == 0 {
            return Err("invalid chunked prefill work budget".into());
        }

        let work = self
            .geometry
            .as_deref()
            .map(|r| r.cached_prefix_work(c.width))
            .transpose()?
            .unwrap_or_default();
        let mut attention_pairs = 0usize;
        let mut geometry_prefix_work = 0usize;
        for start in (0..steps).step_by(chunk_steps) {
            let count = (steps - start).min(chunk_steps);
            let end = start + count;
            let pairs = product(
                &[heads, count, end],
                limits.max_total_attention_pairs,
                "chunked prefill total attention pairs",
            )?;
            attention_pairs = attention_pairs
                .checked_add(pairs)
                .filter(|&n| n <= limits.max_total_attention_pairs)
                .ok_or("chunked prefill total attention pairs exceeds budget")?;
            geometry_prefix_work = geometry_prefix_work
                .checked_add(work.units_at(end)?)
                .filter(|&n| n <= limits.max_total_geometry_work)
                .ok_or("chunked prefill geometry prefix work exceeds budget")?;
            if start == 0 {
                self.context_prefill_count(end, max_cache_values)?;
            } else {
                self.cached_query_count(end, count, max_cache_values)?;
            }
        }
        Ok(CausalLmChunkedPrefillPlan {
            chunk_steps,
            chunks: steps.div_ceil(chunk_steps),
            attention_pairs,
            geometry_prefix_work,
        })
    }

    fn last_block_has_geometry(&self) -> bool {
        self.geometry_artifact().is_some_and(|artifact| {
            let path = format!("transformer.h.{}", self.config.layers - 1);
            artifact.record().placements.iter().any(|p| {
                p.path
                    .strip_prefix(&path)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
            })
        })
    }
}

impl NativeCausalLmContext {
    /// Process a known prompt in bounded chunks without intermediate vocabulary
    /// projections. Keep its complete K/V and geometry state for continuation.
    pub fn new_chunked_prefill(
        model: NativeCausalLm,
        prompt: Vec<usize>,
        max_cache_values: usize,
        limits: CausalLmChunkedPrefillLimits,
    ) -> CausalLmResult<Self> {
        Self::chunked_prefill(model, prompt, max_cache_values, limits, false)
    }

    /// Opt in to omitting unused final-block outputs while retaining all K/V.
    /// This uses the same admission and chunk boundaries as `new_chunked_prefill`.
    /// The final block runs normally if it has a geometry placement. Otherwise,
    /// only the terminal query needs attention, output projection, and MLP work.
    /// Skipped activations and their checks are not evaluated; projected Q/K/V
    /// must still be finite.
    pub fn new_chunked_prefill_last_output(
        model: NativeCausalLm,
        prompt: Vec<usize>,
        max_cache_values: usize,
        limits: CausalLmChunkedPrefillLimits,
    ) -> CausalLmResult<Self> {
        Self::chunked_prefill(model, prompt, max_cache_values, limits, true)
    }

    fn chunked_prefill(
        model: NativeCausalLm,
        prompt: Vec<usize>,
        max_cache_values: usize,
        limits: CausalLmChunkedPrefillLimits,
        last_output: bool,
    ) -> CausalLmResult<Self> {
        let plan = model.context_chunked_prefill_plan(prompt.len(), max_cache_values, limits)?;
        if prompt.iter().any(|&id| id >= model.config.vocabulary) {
            return Err("chunked prefill contains an out-of-range token ID".into());
        }
        let prune = last_output && !model.last_block_has_geometry();
        let mut chunks = prompt.chunks(plan.chunk_steps).peekable();
        let first = chunks.next().ok_or("missing first prefill chunk")?;
        let work = |rows, terminal| {
            if !prune || (terminal && rows == 1) {
                LastBlockWork::Full
            } else if terminal {
                LastBlockWork::LastQuery
            } else {
                LastBlockWork::KvOnly
            }
        };
        let (mut branch, mut hidden) = Self::prefill_hidden(
            &model,
            first,
            max_cache_values,
            work(first.len(), chunks.peek().is_none()),
        )?;
        while let Some(chunk) = chunks.next() {
            (branch, hidden) = Self::suffix_hidden_for(
                &model,
                max_cache_values,
                branch.prefix(),
                chunk,
                None,
                work(chunk.len(), chunks.peek().is_none()),
            )?;
        }
        Self::finish_prefill(model, prompt, max_cache_values, branch, hidden)
    }
}
