//! Frozen native causal language-model execution from explicit checkpoint importers.
//! The first importer is HF GPT-2; Python performs no transformer arithmetic.

use crate::geometry_artifact::GeometryAdapterArtifact;
use crate::geometry_runtime::NativeGeometryRuntime;
use crate::weights::SafeTensorArchive;
use serde::{Deserialize, Serialize};
use st_kernel_contracts::attention::{attention_reference, AttentionMask, AttentionSpec};
use st_tensor::{
    CrossEntropyConfig, LayerNormBackend, MatmulBackend, NdTensor, PackedB, Tensor,
    TensorUtilBackend,
};
use std::sync::Arc;

mod gpt2;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub type CausalLmResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalLmConfig {
    pub vocabulary: usize,
    pub positions: usize,
    pub width: usize,
    pub heads: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub norm_epsilon: f32,
    pub scale_attention: bool,
    pub inverse_layer_scale: bool,
    pub tied_head: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct CausalLmLimits {
    pub max_parameter_elements: usize,
    pub max_activation_elements: usize,
    pub max_attention_pairs: usize,
    pub max_logit_elements: usize,
}

impl Default for CausalLmLimits {
    fn default() -> Self {
        Self {
            max_parameter_elements: 200_000_000,
            max_activation_elements: 16_777_216,
            max_attention_pairs: 4_194_304,
            max_logit_elements: 16_777_216,
        }
    }
}

impl CausalLmLimits {
    pub fn validate(self) -> CausalLmResult<()> {
        for limit in [
            self.max_parameter_elements,
            self.max_activation_elements,
            self.max_attention_pairs,
            self.max_logit_elements,
        ] {
            if limit == 0 || limit > isize::MAX as usize / 4 {
                return Err(
                    "causal LM budgets must be positive addressable f32 element counts".into(),
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Dense {
    weight: PackedB,
    bias: Tensor,
}

impl Dense {
    fn forward(&self, input: &Tensor) -> CausalLmResult<Tensor> {
        Ok(input.matmul_prepacked_bias_with_backend(
            &self.weight,
            self.bias.data(),
            MatmulBackend::CpuSimd,
        )?)
    }
}

#[derive(Clone, Debug)]
struct Norm {
    gamma: Tensor,
    beta: Tensor,
}

impl Norm {
    fn forward(&self, input: &Tensor, epsilon: f32) -> CausalLmResult<Tensor> {
        Ok(input.layer_norm_affine_with_backend(
            &self.gamma,
            &self.beta,
            epsilon,
            LayerNormBackend::Cpu,
        )?)
    }
}

#[derive(Clone, Debug)]
struct DecoderBlock {
    first_norm: Norm,
    qkv: Dense,
    output: Dense,
    second_norm: Norm,
    expand: Dense,
    contract: Dense,
}

/// Full unpadded prefixes with learned absolute positions, pre-norm residual
/// attention/MLP blocks and a vocabulary head. No dropout, KV cache or update
/// methods. Clones share frozen parameters, not a foreign framework/module.
#[derive(Clone, Debug)]
pub struct NativeCausalLm {
    config: CausalLmConfig,
    limits: CausalLmLimits,
    base_id: String,
    token: Tensor,
    position: Tensor,
    blocks: Vec<DecoderBlock>,
    final_norm: Norm,
    head: PackedB,
    geometry: Option<Arc<NativeGeometryRuntime>>,
}

fn product(values: &[usize], budget: usize, label: &'static str) -> CausalLmResult<usize> {
    values
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .filter(|&n| n <= budget)
        .ok_or_else(|| format!("causal LM {label} exceeds budget").into())
}

impl NativeCausalLm {
    /// Import one local GPT-2 safetensors snapshot; not arbitrary remote code or
    /// a generic HF architecture loader. Shape/config admission precedes packing.
    pub fn from_gpt2_json(
        config: &[u8],
        weights: &SafeTensorArchive,
        limits: CausalLmLimits,
    ) -> CausalLmResult<Self> {
        gpt2::load(config, weights, limits)
    }

    pub fn config(&self) -> &CausalLmConfig {
        &self.config
    }

    pub fn base_model_identity(&self) -> &str {
        &self.base_id
    }

    pub fn geometry_artifact_id(&self) -> Option<&str> {
        self.geometry
            .as_ref()
            .map(|runtime| runtime.artifact().artifact_id())
    }

    /// Attach existing native recipes at GPT-2-compatible branch output paths:
    /// transformer.h.N.attn, transformer.h.N.mlp, or transformer.h.N.
    /// No placements are ignored and the frozen checkpoint identity must match.
    pub fn with_geometry(&self, artifact: GeometryAdapterArtifact) -> CausalLmResult<Self> {
        if artifact.record().base_model_identity != self.base_id {
            return Err("geometry belongs to a different native base model".into());
        }
        let runtime = NativeGeometryRuntime::new(artifact);
        for placement in &runtime.artifact().record().placements {
            let valid = (0..self.config.layers).any(|index| {
                let block = format!("transformer.h.{index}");
                placement.path == block
                    || placement.path == format!("{block}.attn")
                    || placement.path == format!("{block}.mlp")
            });
            if !valid {
                return Err(
                    format!("unsupported native geometry placement: {}", placement.path).into(),
                );
            }
            let (width, _) = runtime.sequence_dimensions(&placement.path, [1, 1])?;
            if width != self.config.width {
                return Err("geometry feature width differs from the native model".into());
            }
        }
        let mut next = self.clone();
        next.geometry = Some(Arc::new(runtime));
        Ok(next)
    }

    /// Validate work before a binding extracts token IDs or any activation is allocated.
    pub fn input_count(&self, shape: [usize; 2], last_only: bool) -> CausalLmResult<usize> {
        let [batch, steps] = shape;
        let c = &self.config;
        if batch == 0 || steps == 0 || steps > c.positions {
            return Err(
                "native causal LM requires nonempty full prefixes within position capacity".into(),
            );
        }
        let rows = product(&shape, self.limits.max_activation_elements, "token rows")?;
        product(
            &[rows, c.width, 3],
            self.limits.max_activation_elements,
            "QKV activations",
        )?;
        product(
            &[rows, c.intermediate],
            self.limits.max_activation_elements,
            "MLP activations",
        )?;
        product(
            &[c.layers, batch, c.heads, steps, steps],
            self.limits.max_attention_pairs,
            "total attention pairs",
        )?;
        product(
            &[if last_only { batch } else { rows }, c.vocabulary],
            self.limits.max_logit_elements,
            "logits",
        )?;
        if let Some(runtime) = &self.geometry {
            for placement in &runtime.artifact().record().placements {
                runtime.sequence_dimensions(&placement.path, shape)?;
            }
        }
        Ok(rows)
    }

    pub fn forward(
        &self,
        input_ids: &[usize],
        shape: [usize; 2],
        last_only: bool,
    ) -> CausalLmResult<NdTensor> {
        let logits = self.logits(input_ids, shape, last_only)?;
        Ok(logits.into_nd()?.reshape(&[
            shape[0],
            if last_only { 1 } else { shape[1] },
            self.config.vocabulary,
        ])?)
    }

    pub fn next_token_ids(
        &self,
        input_ids: &[usize],
        shape: [usize; 2],
    ) -> CausalLmResult<Vec<usize>> {
        let logits = self.logits(input_ids, shape, true)?;
        Ok(logits
            .data()
            .chunks_exact(self.config.vocabulary)
            .map(|row| {
                let mut best = 0;
                for index in 1..row.len() {
                    if row[index] > row[best] {
                        best = index;
                    }
                }
                best
            })
            .collect())
    }

    /// Mean teacher-forced next-token NLL for B independent windows of T+1 IDs.
    /// Labels are shifted within each row only; there is no padding or ignore mask.
    pub fn next_token_loss(&self, windows: &[usize], shape: [usize; 2]) -> CausalLmResult<f32> {
        if shape[1] < 2 {
            return Err("next-token loss needs at least two IDs per window".into());
        }
        let steps = shape[1] - 1;
        let count = self.input_count([shape[0], steps], false)?;
        if windows.len() != product(&shape, usize::MAX, "window IDs")?
            || windows.iter().any(|&id| id >= self.config.vocabulary)
        {
            return Err("invalid next-token window length or token ID".into());
        }
        let mut input = Vec::with_capacity(count);
        let mut labels = Vec::with_capacity(count);
        for window in windows.chunks_exact(shape[1]) {
            input.extend_from_slice(&window[..steps]);
            labels.extend(window[1..].iter().map(|&id| id as i64));
        }
        Ok(self
            .logits(&input, [shape[0], steps], false)?
            .cross_entropy_with_logits(&labels, CrossEntropyConfig::default())?
            .data()[0])
    }

    fn intervene(&self, path: &str, input: Tensor, shape: [usize; 2]) -> CausalLmResult<Tensor> {
        if let Some(runtime) = &self.geometry {
            if runtime
                .artifact()
                .record()
                .placements
                .iter()
                .any(|p| p.path == path)
            {
                return Ok(runtime
                    .forward_sequence(path, input.data(), shape)?
                    .output()
                    .clone());
            }
        }
        Ok(input)
    }

    fn logits(&self, ids: &[usize], shape: [usize; 2], last_only: bool) -> CausalLmResult<Tensor> {
        let rows = self.input_count(shape, last_only)?;
        let c = &self.config;
        if ids.len() != rows || ids.iter().any(|&id| id >= c.vocabulary) {
            return Err("native causal LM token count or ID is invalid".into());
        }
        let [batch, steps] = shape;
        let positions: Vec<_> = (0..rows).map(|row| row % steps).collect();
        let mut hidden = self.token.gather_rows(ids)?.add_with_backend(
            &self.position.gather_rows(&positions)?,
            TensorUtilBackend::Cpu,
        )?;
        let head_dim = c.width / c.heads;
        for (index, block) in self.blocks.iter().enumerate() {
            let norm = block.first_norm.forward(&hidden, c.norm_epsilon)?;
            let fused = block.qkv.forward(&norm)?;
            let count = rows * c.width;
            let mut qkv = [vec![0.; count], vec![0.; count], vec![0.; count]];
            // Existing attention contracts consume explicit [B,H,T,D], not a
            // flattened row stream; unpack GPT-2's [B,T,3*H*D] projection once.
            for b in 0..batch {
                for h in 0..c.heads {
                    for t in 0..steps {
                        let destination = ((b * c.heads + h) * steps + t) * head_dim;
                        for (component, output) in qkv.iter_mut().enumerate() {
                            let source = (b * steps + t) * (3 * c.width)
                                + component * c.width
                                + h * head_dim;
                            output[destination..destination + head_dim]
                                .copy_from_slice(&fused.data()[source..source + head_dim]);
                        }
                    }
                }
            }
            let mut scale = if c.scale_attention {
                1. / (head_dim as f32).sqrt()
            } else {
                1.
            };
            if c.inverse_layer_scale {
                scale /= (index + 1) as f32;
            }
            let heads = [batch, c.heads, steps, head_dim];
            let spec = AttentionSpec::new(
                &heads,
                &heads,
                &heads,
                scale,
                AttentionMask::Causal { query_offset: 0 },
            )?;
            let attended = attention_reference(spec, &qkv[0], &qkv[1], &qkv[2], None, None)?;
            let mut merged = vec![0.; count];
            for b in 0..batch {
                for t in 0..steps {
                    for h in 0..c.heads {
                        let source = ((b * c.heads + h) * steps + t) * head_dim;
                        let destination = (b * steps + t) * c.width + h * head_dim;
                        merged[destination..destination + head_dim]
                            .copy_from_slice(&attended[source..source + head_dim]);
                    }
                }
            }
            let path = format!("transformer.h.{index}");
            let attention = block
                .output
                .forward(&Tensor::from_vec(rows, c.width, merged)?)?;
            let attention = self.intervene(&format!("{path}.attn"), attention, shape)?;
            hidden = attention.add_with_backend(&hidden, TensorUtilBackend::Cpu)?;
            let norm = block.second_norm.forward(&hidden, c.norm_epsilon)?;
            let expanded = block.expand.forward(&norm)?.try_into_gelu()?;
            let mlp = self.intervene(
                &format!("{path}.mlp"),
                block.contract.forward(&expanded)?,
                shape,
            )?;
            hidden = self.intervene(
                &path,
                hidden.add_with_backend(&mlp, TensorUtilBackend::Cpu)?,
                shape,
            )?;
        }
        let hidden = if last_only {
            hidden.gather_rows(&(0..batch).map(|b| (b + 1) * steps - 1).collect::<Vec<_>>())?
        } else {
            hidden
        };
        let normalized = self.final_norm.forward(&hidden, c.norm_epsilon)?;
        Ok(normalized.matmul_prepacked_with_backend(&self.head, MatmulBackend::CpuSimd)?)
    }
}
