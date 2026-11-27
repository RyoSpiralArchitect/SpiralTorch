//! One decoder implementation for inference and checkpointed learning replay.
use super::context::InferenceBlockCache;
use super::*;
use crate::geometry_runtime::NativeGeometryBatch;
use st_kernel_contracts::attention::{attention_key_lag_reference, AttentionMask, AttentionSpec};

pub(super) enum BlockPass<'a> {
    Forward,
    Observe(&'a mut observation::ResidualSums),
    CaptureGeometry {
        path: &'a str,
        snapshot: &'a mut Option<(Tensor, NativeGeometryBatch)>,
    },
    BackwardReplay,
    Prefill,
    PrefillLastQuery,
    Projection(projection_probe::ProjectionDelta<'a>),
}

impl BlockPass<'_> {
    fn is_prefill(&self) -> bool {
        matches!(self, Self::Prefill | Self::PrefillLastQuery)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) mod work_counts {
    use std::cell::Cell;
    thread_local! { static COUNTS: Cell<[usize; 3]> = const { Cell::new([0; 3]) }; }
    thread_local! { static SUFFIXES: Cell<(usize,usize)> = const { Cell::new((0,0)) }; }
    thread_local! { static CACHE_ROWS: Cell<[usize; 2]> = const { Cell::new([0; 2]) }; }

    pub(in crate::causal_lm) fn cached_attention(rows: usize) {
        let [q, mlp] = CACHE_ROWS.get();
        CACHE_ROWS.set([q + rows, mlp]);
    }

    pub(in crate::causal_lm) fn cached_mlp(rows: usize) {
        let [q, mlp] = CACHE_ROWS.get();
        CACHE_ROWS.set([q, mlp + rows]);
    }

    pub(in crate::causal_lm) fn take_cache_rows() -> [usize; 2] {
        CACHE_ROWS.with(Cell::take)
    }

    pub(crate) fn suffix(queries: usize) {
        SUFFIXES.with(|c| {
            let (calls, n) = c.get();
            c.set((calls + 1, n + queries));
        });
    }
    pub(crate) fn take_suffixes() -> (usize, usize) {
        SUFFIXES.with(Cell::take)
    }

    pub(crate) fn block(pass: &super::BlockPass<'_>) {
        match pass {
            super::BlockPass::Forward
            | super::BlockPass::Observe(_)
            | super::BlockPass::CaptureGeometry { .. } => bump(0),
            super::BlockPass::BackwardReplay => bump(1),
            _ => {}
        }
    }
    pub(crate) fn head() {
        bump(2);
    }
    pub(crate) fn take() -> [usize; 3] {
        COUNTS.with(Cell::take)
    }
    fn bump(i: usize) {
        COUNTS.with(|c| {
            let mut n = c.get();
            n[i] += 1;
            c.set(n);
        });
    }
}

pub(super) struct BlockCache {
    pub qkv: [Vec<f32>; 3],
    pub attention_input: Tensor,
    pub after_attention: Tensor,
    pub pre_gelu: Tensor,
    pub attention_geometry: Option<NativeGeometryBatch>,
    pub mlp_geometry: Option<NativeGeometryBatch>,
    pub block_geometry: Option<NativeGeometryBatch>,
}

pub(super) struct BlockOutput {
    pub output: Tensor,
    pub cache: Option<BlockCache>,
    pub inference_cache: Option<InferenceBlockCache>,
}

pub(super) fn keep_last_query(query: &mut Vec<f32>, rows: usize, heads: usize, dim: usize) {
    for head in 0..heads {
        let end = (head + 1) * rows * dim;
        query.copy_within(end - dim..end, head * dim);
    }
    query.truncate(heads * dim);
}

impl NativeCausalLm {
    pub(super) fn embed(&self, ids: &[usize], shape: [usize; 2]) -> CausalLmResult<Tensor> {
        self.embed_at(ids, shape, 0)
    }

    pub(super) fn embed_at(
        &self,
        ids: &[usize],
        shape: [usize; 2],
        offset: usize,
    ) -> CausalLmResult<Tensor> {
        if shape[0].checked_mul(shape[1]) != Some(ids.len())
            || offset
                .checked_add(shape[1])
                .is_none_or(|end| end > self.config.positions)
            || ids.iter().any(|&id| id >= self.config.vocabulary)
        {
            return Err("native causal LM token count or ID is invalid".into());
        }
        let positions: Vec<_> = (0..ids.len()).map(|row| offset + row % shape[1]).collect();
        Ok(self.token.gather_rows(ids)?.add_with_backend(
            &self.position.gather_rows(&positions)?,
            TensorUtilBackend::Cpu,
        )?)
    }

    fn attention_scale(&self, layer: usize) -> f32 {
        let c = &self.config;
        let head_dim = c.width / c.heads;
        let mut scale = if c.scale_attention {
            1. / (head_dim as f32).sqrt()
        } else {
            1.
        };
        if c.inverse_layer_scale {
            scale /= (layer + 1) as f32;
        }
        scale
    }

    pub(super) fn attention_spec(
        &self,
        layer: usize,
        shape: [usize; 2],
    ) -> CausalLmResult<AttentionSpec> {
        let c = &self.config;
        let heads = [shape[0], c.heads, shape[1], c.width / c.heads];
        Ok(AttentionSpec::new(
            &heads,
            &heads,
            &heads,
            self.attention_scale(layer),
            AttentionMask::Causal { query_offset: 0 },
        )?)
    }

    pub(super) fn attention_suffix_spec(
        &self,
        layer: usize,
        queries: usize,
        steps: usize,
    ) -> CausalLmResult<AttentionSpec> {
        let c = &self.config;
        let dim = c.width / c.heads;
        let query = [1, c.heads, queries, dim];
        let kv = [1, c.heads, steps, dim];
        let query_offset = steps
            .checked_sub(queries)
            .ok_or("invalid attention suffix")?;
        Ok(AttentionSpec::new(
            &query,
            &kv,
            &kv,
            self.attention_scale(layer),
            AttentionMask::Causal { query_offset },
        )?)
    }

    pub(super) fn prefill_last_block_kv(
        &self,
        input: &Tensor,
        steps: usize,
    ) -> CausalLmResult<InferenceBlockCache> {
        let block = &self.blocks[self.config.layers - 1];
        let norm = block.first_norm.forward(input, self.config.norm_epsilon)?;
        // Even discarded Q rows pass through the checked dense projection.
        let [_, key, value] = self.split_heads::<3>(&block.qkv.forward(&norm)?, [1, steps])?;
        Ok(InferenceBlockCache {
            key,
            value,
            history: [None, None, None],
        })
    }

    // These permutations are exact inverses in both the primal and adjoint.
    pub(super) fn split_heads<const N: usize>(
        &self,
        input: &Tensor,
        shape: [usize; 2],
    ) -> CausalLmResult<[Vec<f32>; N]> {
        let [batch, steps] = shape;
        let c = &self.config;
        if input.shape() != (batch * steps, N * c.width) {
            return Err("invalid fused head shape".into());
        }
        let d = c.width / c.heads;
        let mut output: [Vec<f32>; N] = std::array::from_fn(|_| vec![0.; batch * steps * c.width]);
        for b in 0..batch {
            for h in 0..c.heads {
                for t in 0..steps {
                    let target = ((b * c.heads + h) * steps + t) * d;
                    for (n, values) in output.iter_mut().enumerate() {
                        let source = (b * steps + t) * (N * c.width) + n * c.width + h * d;
                        values[target..target + d]
                            .copy_from_slice(&input.data()[source..source + d]);
                    }
                }
            }
        }
        Ok(output)
    }

    pub(super) fn merge_heads<const N: usize>(
        &self,
        heads: [&[f32]; N],
        shape: [usize; 2],
    ) -> CausalLmResult<Tensor> {
        let [batch, steps] = shape;
        let c = &self.config;
        let count = batch * steps * c.width;
        if heads.iter().any(|values| values.len() != count) {
            return Err("invalid separated head shape".into());
        }
        let d = c.width / c.heads;
        let mut output = vec![0.; N * count];
        for b in 0..batch {
            for t in 0..steps {
                for h in 0..c.heads {
                    let source = ((b * c.heads + h) * steps + t) * d;
                    for (n, values) in heads.iter().enumerate() {
                        let target = (b * steps + t) * (N * c.width) + n * c.width + h * d;
                        output[target..target + d].copy_from_slice(&values[source..source + d]);
                    }
                }
            }
        }
        Ok(Tensor::from_vec(batch * steps, N * c.width, output)?)
    }

    pub(super) fn geometry_output(
        &self,
        path: &str,
        input: Tensor,
        shape: [usize; 2],
        source: Option<&Tensor>,
        pass: &mut BlockPass<'_>,
    ) -> CausalLmResult<(Tensor, Option<NativeGeometryBatch>)> {
        if let Some(runtime) = self.runtime_at(path) {
            let batch = match source {
                Some(v) => {
                    runtime.forward_value_read_sequence(path, input.data(), v.data(), shape)?
                }
                None => runtime.forward_sequence(path, input.data(), shape)?,
            };
            if let BlockPass::CaptureGeometry {
                path: wanted,
                snapshot,
            } = pass
            {
                if path == *wanted {
                    let output = batch.output().clone();
                    if snapshot.replace((input, batch)).is_some() {
                        return Err("duplicate geometry capture".into());
                    }
                    return Ok((output, None));
                }
            }
            if let BlockPass::Observe(sums) = pass {
                sums.record(path, &input, batch.output())?;
            }
            return Ok((
                batch.output().clone(),
                matches!(pass, BlockPass::BackwardReplay).then_some(batch),
            ));
        }
        Ok((input, None))
    }

    fn block_geometry_output(
        &self,
        path: &str,
        input: Tensor,
        shape: [usize; 2],
        pass: &mut BlockPass<'_>,
        history: &mut Option<context::GeometryHistory>,
        source: Option<&Tensor>,
    ) -> CausalLmResult<(Tensor, Option<NativeGeometryBatch>)> {
        if pass.is_prefill() && source.is_none() {
            if shape[0] != 1 {
                return Err("prefill requires exactly one sequence".into());
            }
            let (out, state) = self.incremental_geometry(path, input, None, shape[1])?;
            *history = state;
            return Ok((out, None));
        }
        self.geometry_output(path, input, shape, source, pass)
    }

    pub(super) fn run_block(
        &self,
        index: usize,
        input: Tensor,
        shape: [usize; 2],
        mut pass: BlockPass<'_>,
    ) -> CausalLmResult<BlockOutput> {
        #[cfg(all(test, not(target_arch = "wasm32")))]
        work_counts::block(&pass);
        let capture = matches!(pass, BlockPass::BackwardReplay);
        let inference = pass.is_prefill();
        let block = &self.blocks[index];
        let norm = block.first_norm.forward(&input, self.config.norm_epsilon)?;
        let mut fused = block.qkv.forward(&norm)?;
        if let BlockPass::Projection(d) = &pass {
            if let Some(part) = d.spec.site.qkv_part() {
                fused = d.apply(&norm, &fused, part)?;
            }
        }
        let mut qkv = self.split_heads::<3>(&fused, shape)?;
        drop(fused);
        let kv_steps = shape[1];
        let (spec, input, shape) = if matches!(pass, BlockPass::PrefillLastQuery) {
            debug_assert_eq!((shape[0], index + 1), (1, self.config.layers));
            let c = &self.config;
            keep_last_query(&mut qkv[0], kv_steps, c.heads, c.width / c.heads);
            (
                self.attention_suffix_spec(index, 1, kv_steps)?,
                input.gather_rows(&[kv_steps - 1])?,
                [1, 1],
            )
        } else {
            (self.attention_spec(index, shape)?, input, shape)
        };
        #[cfg(all(test, not(target_arch = "wasm32")))]
        if inference {
            work_counts::cached_attention(shape[1]);
        }
        let attended = attention_key_lag_reference(
            spec,
            &qkv[0],
            &qkv[1],
            &qkv[2],
            None,
            None,
            self.key_lag_at(index),
        )?;
        let attention_input = self.merge_heads([attended.as_slice()], shape)?;
        let mut attention = self.project_attention_output(index, &attention_input)?;
        if let BlockPass::Projection(d) = &pass {
            if d.spec.site == CausalLmProjectionSite::Output {
                attention = d.apply(&attention_input, &attention, 0)?;
            }
        }
        let path = format!("transformer.h.{index}");
        let mlp_path = format!("{path}.mlp");
        let value_read =
            self.mlp_value_source(&mlp_path, &attention_input, &qkv[2], shape, kv_steps)?;
        let attention_input = capture.then_some(attention_input);
        let mut history = std::array::from_fn(|_| None);
        let (attention, attention_geometry) = self.block_geometry_output(
            &format!("{path}.attn"),
            attention,
            shape,
            &mut pass,
            &mut history[0],
            None,
        )?;
        let after_attention = attention.add_with_backend(&input, TensorUtilBackend::Cpu)?;
        let norm = block
            .second_norm
            .forward(&after_attention, self.config.norm_epsilon)?;
        #[cfg(all(test, not(target_arch = "wasm32")))]
        if inference {
            work_counts::cached_mlp(norm.shape().0);
        }
        let expanded = block.expand.forward(&norm)?;
        let pre_gelu = capture.then(|| expanded.clone());
        let mlp = block.contract.forward(&expanded.try_into_gelu()?)?;
        let (mlp, mlp_geometry) = self.block_geometry_output(
            &mlp_path,
            mlp,
            shape,
            &mut pass,
            &mut history[1],
            value_read.as_ref(),
        )?;
        let output = after_attention.add_with_backend(&mlp, TensorUtilBackend::Cpu)?;
        let (output, block_geometry) =
            self.block_geometry_output(&path, output, shape, &mut pass, &mut history[2], None)?;
        let inference_cache = inference.then(|| InferenceBlockCache {
            key: std::mem::take(&mut qkv[1]),
            value: std::mem::take(&mut qkv[2]),
            history,
        });
        let cache = capture.then(|| BlockCache {
            qkv,
            attention_input: attention_input.unwrap(),
            after_attention,
            pre_gelu: pre_gelu.unwrap(),
            attention_geometry,
            mlp_geometry,
            block_geometry,
        });
        Ok(BlockOutput {
            output,
            cache,
            inference_cache,
        })
    }
}
