//! Native causal language models with frozen base weights and trainable geometry.
//! The first importer is HF GPT-2; Python performs no transformer arithmetic.

use crate::geometry_artifact::GeometryAdapterArtifact;
use crate::geometry_runtime::NativeGeometryRuntime;
use crate::weights::SafeTensorArchive;
use serde::{Deserialize, Serialize};
use st_tensor::{
    CrossEntropyConfig, LayerNormBackend, MatmulBackend, NdTensor, PackedB, Tensor,
    TensorUtilBackend,
};
use std::sync::{Arc, OnceLock};

mod context;
mod decoder;
mod evaluation;
mod generation;
mod gpt2;
mod learning;
mod objective;
mod repetition;
mod training;
pub use context::NativeCausalLmContext;
pub use evaluation::{CausalLmEvaluation, CausalLmEvaluationConfig, MAX_EVALUATION_TOKENS};
pub use generation::{
    CausalLmGenerationConfig, CausalLmGenerationExecution, CausalLmGenerationStep,
    NativeCausalLmGenerator,
};
pub use learning::{NativeCausalLmAccumulator, NativeCausalLmGradients};
pub use objective::CausalLmObjective;
use objective::PreparedCausalWindow;
pub use repetition::CausalLmRepetition;
pub use training::{
    CausalLmTrainingConfig, CausalLmTrainingResume, CausalLmTrainingStep, NativeCausalLmTrainer,
};
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub type CausalLmResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
type TransposeCache = Arc<OnceLock<Result<PackedB, String>>>;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CausalLmReadoutAccumulator {
    #[default]
    F32,
    F64,
}

impl CausalLmReadoutAccumulator {
    fn is_f32(&self) -> bool {
        *self == Self::F32
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

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
    pub max_training_checkpoint_elements: usize,
}

impl Default for CausalLmLimits {
    fn default() -> Self {
        Self {
            max_parameter_elements: 200_000_000,
            max_activation_elements: 16_777_216,
            max_attention_pairs: 4_194_304,
            max_logit_elements: 16_777_216,
            max_training_checkpoint_elements: 16_777_216,
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
            self.max_training_checkpoint_elements,
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
    backward_weight: TransposeCache,
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
/// attention/MLP blocks and a vocabulary head. No dropout or base-weight updates.
/// The separate context owns K/V. Clones share frozen parameters, not a foreign framework.
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
    backward_head: TransposeCache,
    geometry: Option<Arc<NativeGeometryRuntime>>,
    readout_accumulator: CausalLmReadoutAccumulator,
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

    pub fn readout_accumulator(&self) -> CausalLmReadoutAccumulator {
        self.readout_accumulator
    }

    /// Preserve frozen/geometry storage while explicitly choosing readout arithmetic.
    /// Training and generation checkpoints bind this policy independently of weight identity.
    pub fn with_readout_accumulator(&self, accumulator: CausalLmReadoutAccumulator) -> Self {
        let mut model = self.clone();
        model.readout_accumulator = accumulator;
        model
    }

    fn project_readout(&self, normalized: &Tensor) -> CausalLmResult<Tensor> {
        Ok(match self.readout_accumulator {
            CausalLmReadoutAccumulator::F32 => {
                normalized.matmul_prepacked_with_backend(&self.head, MatmulBackend::CpuSimd)?
            }
            CausalLmReadoutAccumulator::F64 => normalized.matmul_prepacked_wide(&self.head)?,
        })
    }

    pub fn base_model_identity(&self) -> &str {
        &self.base_id
    }

    pub fn geometry_artifact_id(&self) -> Option<&str> {
        self.geometry
            .as_ref()
            .map(|runtime| runtime.artifact().artifact_id())
    }

    pub fn geometry_artifact(&self) -> Option<&GeometryAdapterArtifact> {
        self.geometry.as_ref().map(|runtime| runtime.artifact())
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
        let rows = self.context_input_count(shape)?;
        self.logit_count(if last_only { shape[0] } else { rows })?;
        Ok(rows)
    }

    fn logit_count(&self, rows: usize) -> CausalLmResult<usize> {
        product(
            &[rows, self.config.vocabulary],
            self.limits.max_logit_elements,
            "logits",
        )
    }

    fn context_input_count(&self, shape: [usize; 2]) -> CausalLmResult<usize> {
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
    /// Labels are shifted within each row only; all shifted targets participate.
    pub fn next_token_loss(&self, windows: &[usize], shape: [usize; 2]) -> CausalLmResult<f32> {
        self.next_token_loss_with_objective(windows, shape, None, CausalLmObjective::default())
    }

    fn logits(&self, ids: &[usize], shape: [usize; 2], last_only: bool) -> CausalLmResult<Tensor> {
        self.input_count(shape, last_only)?;
        let c = &self.config;
        let [batch, steps] = shape;
        let hidden = self.decode_hidden(ids, shape)?;
        let hidden = if last_only {
            hidden.gather_rows(&(0..batch).map(|b| (b + 1) * steps - 1).collect::<Vec<_>>())?
        } else {
            hidden
        };
        let normalized = self.final_norm.forward(&hidden, c.norm_epsilon)?;
        self.project_readout(&normalized)
    }

    fn decode_hidden(&self, ids: &[usize], shape: [usize; 2]) -> CausalLmResult<Tensor> {
        let mut hidden = self.embed(ids, shape)?;
        for index in 0..self.blocks.len() {
            hidden = self.run_block(index, hidden, shape, false)?.output;
        }
        Ok(hidden)
    }
}
