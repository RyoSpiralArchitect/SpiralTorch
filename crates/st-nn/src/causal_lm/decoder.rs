//! One decoder implementation for inference and checkpointed learning replay.
use super::context::InferenceBlockCache;
use super::*;
use crate::geometry_runtime::NativeGeometryBatch;
use st_kernel_contracts::attention::{attention_reference, AttentionMask, AttentionSpec};

pub(super) struct BlockCache {
    pub qkv: [Vec<f32>; 3],
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

impl NativeCausalLm {
    pub(super) fn embed(&self, ids: &[usize], shape: [usize; 2]) -> CausalLmResult<Tensor> {
        let rows = shape[0] * shape[1];
        if ids.len() != rows || ids.iter().any(|&id| id >= self.config.vocabulary) {
            return Err("native causal LM token count or ID is invalid".into());
        }
        let positions: Vec<_> = (0..rows).map(|row| row % shape[1]).collect();
        Ok(self.token.gather_rows(ids)?.add_with_backend(
            &self.position.gather_rows(&positions)?,
            TensorUtilBackend::Cpu,
        )?)
    }

    pub(super) fn attention_spec(
        &self,
        layer: usize,
        shape: [usize; 2],
    ) -> CausalLmResult<AttentionSpec> {
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
        let heads = [shape[0], c.heads, shape[1], head_dim];
        Ok(AttentionSpec::new(
            &heads,
            &heads,
            &heads,
            scale,
            AttentionMask::Causal { query_offset: 0 },
        )?)
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
        capture: bool,
    ) -> CausalLmResult<(Tensor, Option<NativeGeometryBatch>)> {
        if let Some(runtime) = &self.geometry {
            if runtime
                .artifact()
                .record()
                .placements
                .iter()
                .any(|p| p.path == path)
            {
                let batch = runtime.forward_sequence(path, input.data(), shape)?;
                return Ok((batch.output().clone(), capture.then_some(batch)));
            }
        }
        Ok((input, None))
    }

    pub(super) fn run_block(
        &self,
        index: usize,
        input: Tensor,
        shape: [usize; 2],
        capture: bool,
    ) -> CausalLmResult<BlockOutput> {
        self.run_block_capture(index, input, shape, capture, false)
    }

    pub(super) fn prefill_block(
        &self,
        index: usize,
        input: Tensor,
        steps: usize,
    ) -> CausalLmResult<BlockOutput> {
        self.run_block_capture(index, input, [1, steps], false, true)
    }

    fn run_block_capture(
        &self,
        index: usize,
        input: Tensor,
        shape: [usize; 2],
        capture: bool,
        inference: bool,
    ) -> CausalLmResult<BlockOutput> {
        let block = &self.blocks[index];
        let norm = block.first_norm.forward(&input, self.config.norm_epsilon)?;
        let qkv = self.split_heads::<3>(&block.qkv.forward(&norm)?, shape)?;
        let attended = attention_reference(
            self.attention_spec(index, shape)?,
            &qkv[0],
            &qkv[1],
            &qkv[2],
            None,
            None,
        )?;
        let attention = block
            .output
            .forward(&self.merge_heads([attended.as_slice()], shape)?)?;
        let path = format!("transformer.h.{index}");
        let mut history = std::array::from_fn(|_| None);
        if inference && self.geometry_needs_history(&format!("{path}.attn"))? {
            history[0] = Some(attention.clone());
        }
        let (attention, attention_geometry) =
            self.geometry_output(&format!("{path}.attn"), attention, shape, capture)?;
        let after_attention = attention.add_with_backend(&input, TensorUtilBackend::Cpu)?;
        let norm = block
            .second_norm
            .forward(&after_attention, self.config.norm_epsilon)?;
        let expanded = block.expand.forward(&norm)?;
        let pre_gelu = capture.then(|| expanded.clone());
        let mlp = block.contract.forward(&expanded.try_into_gelu()?)?;
        if inference && self.geometry_needs_history(&format!("{path}.mlp"))? {
            history[1] = Some(mlp.clone());
        }
        let (mlp, mlp_geometry) =
            self.geometry_output(&format!("{path}.mlp"), mlp, shape, capture)?;
        let output = after_attention.add_with_backend(&mlp, TensorUtilBackend::Cpu)?;
        if inference && self.geometry_needs_history(&path)? {
            history[2] = Some(output.clone());
        }
        let (output, block_geometry) = self.geometry_output(&path, output, shape, capture)?;
        let inference_cache = inference.then(|| InferenceBlockCache {
            key: qkv[1].clone(),
            value: qkv[2].clone(),
            history,
        });
        let cache = capture.then(|| BlockCache {
            qkv,
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
