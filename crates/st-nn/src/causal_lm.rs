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

mod adaptation;
mod checkpoint;
mod common_geometry;
mod competition_pair;
mod context;
mod context_preference;
mod decoder;
mod evaluation;
mod generation;
mod geometry_cohort;
mod gpt2;
mod gradient_alignment;
mod key_lag;
mod learning;
mod line_menu;
mod line_set;
mod lora;
mod mixture_probe;
mod objective;
mod observation;
mod output_margin;
mod paired_geometry;
mod projection_probe;
mod qkv_training;
mod quartet_projection;
mod readout_policy;
mod repetition;
mod scalar_output;
mod session;
mod sum;
mod training;
pub use adaptation::{CausalLmAdaptationMode, NativeCausalLmAdaptationGradients};
pub use common_geometry::hull::CommonDirectionRefusal;
pub use common_geometry::{
    CausalLmCommonGeometryStep, CausalLmGeometrySteeringStep, CausalLmGeometryStepGuard,
    MAX_COMMON_GEOMETRY_GROUPS,
};
pub use competition_pair::{CausalLmGeometryCompetitionPair, CausalLmGeometryCompetitionStep};
pub use context::NativeCausalLmContext;
pub use evaluation::{
    CausalLmAnswerComparison, CausalLmAnswerContrast, CausalLmAnswerScore, CausalLmEvaluation,
    CausalLmEvaluationComparison, CausalLmEvaluationConfig, CausalLmScoredAnswer,
    MAX_ANSWER_SCORE_CELLS, MAX_EVALUATION_TOKENS, MAX_EVALUATION_VARIANTS,
};
pub use generation::{
    CausalLmGenerationConfig, CausalLmGenerationExecution, CausalLmGenerationStep,
    NativeCausalLmGenerator,
};
pub use geometry_cohort::{
    CausalLmGeometryCohortStep, MAX_GEOMETRY_COHORT_ELEMENTS, MAX_GEOMETRY_COHORT_MEMBERS,
};
pub use gradient_alignment::{CausalLmGradientAlignment, CausalLmGradientPairStats};
pub use key_lag::CausalLmKeyLag;
pub use learning::{NativeCausalLmAccumulator, NativeCausalLmGradients};
pub use line_menu::{CausalLmLineMenuReport, MAX_LINE_MENU_CANDIDATES, MAX_LINE_MENU_TARGETS};
pub use line_set::{CausalLmLineScores, CausalLmLineSetReport, CausalLmLineSubsetReport};
pub use lora::NativeCausalLmLoraGradients;
pub use mixture_probe::{
    CausalLmMixtureObjectiveTrace, CausalLmMixturePath, CausalLmMixtureProbe,
    MAX_MIXTURE_PROBE_ELEMENTS,
};
pub use objective::CausalLmObjective;
use objective::PreparedCausalWindow;
pub use observation::{
    CausalLmAdaptationObservation, CausalLmAdaptationSite, CausalLmObservedTrainingStep,
    CausalLmSiteObservation, CausalLmSitePullback,
};
pub use output_margin::{
    CausalLmOutputContrastBank, CausalLmOutputMarginBank, CausalLmOutputMarginDirection,
    CausalLmOutputMarginRule, MAX_OUTPUT_CONTRAST_ROWS,
};
pub use paired_geometry::{
    CausalLmPairDisplacement, CausalLmPairWeighting, CausalLmPairedGeometryStep,
};
pub use projection_probe::{
    CausalLmProjectionBasis, CausalLmProjectionPullback, CausalLmProjectionSite,
    CausalLmProjectionSpec,
};
pub use qkv_training::{
    CausalLmQkvDocumentStep, CausalLmQkvGradientClip, CausalLmQkvSmoothedStep, CausalLmQkvStep,
    CausalLmQkvWindowStep, NativeCausalLmQkvAdamW,
};
pub use quartet_projection::{
    CausalLmAnswerBinding, CausalLmAnswerCorrection, CausalLmAnswerGuard,
    CausalLmAnswerGuardStatus, CausalLmAnswerGuardStep, CausalLmAnswerPoint,
    CausalLmAnswerProjectionStep, CausalLmAnswerProtection, CausalLmAnswerTrial,
    CausalLmLossGradient, CausalLmQuartetObjective, CausalLmQuartetStep,
    NativeCausalLmProjectionPrefix, NativeCausalLmRankOneProjection,
};
pub use readout_policy::{CausalLmHeadPrecision, CausalLmReadoutAccumulator};
pub use repetition::CausalLmRepetition;
pub use scalar_output::{
    CausalLmScalarPairLoss, CausalLmScalarPairStep, CausalLmScalarSearch,
    CausalLmScalarSearchReport, CausalLmScalarSearchStatus, CausalLmScalarTrial,
    NativeCausalLmScalarOutput, MAX_SCALAR_OUTPUT_PAIRS, MAX_SCALAR_SEARCH_TRIALS,
};
pub use session::{
    CausalLmMatchedTrainingStep, CausalLmPrefixCacheAction, CausalLmPrefixCacheStats,
    CausalLmSessionWindowPolicy, CausalLmTrainingSessionConfig, CausalLmTrainingSessionResume,
    CausalLmTrainingSessionStep, NativeCausalLmTrainingPrefixCache, NativeCausalLmTrainingSession,
};
pub use st_tensor::classification::{
    ContextTokenPreference, DeclaredRivalReport, DeclaredRivalRow, RivalCrossEntropy,
    RivalCrossEntropyReport, TargetClassScore, MAX_DECLARED_RIVAL_IDS, MAX_RIVAL_CLASSES,
};
use st_tensor::classification::{RivalCoverageGuard, RivalCoverageReference};
pub use training::{
    CausalLmObservedCheckpointFiles, CausalLmTrainingConfig, CausalLmTrainingResume,
    CausalLmTrainingStep, NativeCausalLmTrainer,
};
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub type CausalLmResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
type TransposeCache = Arc<OnceLock<Result<PackedB, String>>>;

/// Raw last-position logits for two named IDs, plus the unrestricted greedy choice.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CausalLmTokenContrast {
    pub prompt_tokens: usize,
    pub target_id: usize,
    pub alternative_id: usize,
    pub greedy_id: usize,
    pub target_logit: f64,
    pub alternative_logit: f64,
    /// Difference after widening both f32 logits, with no sampling or temperature transform.
    pub target_minus_alternative: f64,
}

fn greedy_id(row: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..row.len() {
        if row[index] > row[best] {
            best = index;
        }
    }
    best
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
    key_lag: Option<CausalLmKeyLag>,
    limits: CausalLmLimits,
    base_id: String,
    token: Tensor,
    position: Tensor,
    blocks: Vec<DecoderBlock>,
    final_norm: Norm,
    head: PackedB,
    backward_head: TransposeCache,
    geometry: Option<Arc<NativeGeometryRuntime>>,
    lora: Option<Arc<lora::NativeLoraAdapter>>,
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
        self.readout_accumulator
            .forward_precision()
            .multiply(normalized, &self.head)
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
        next.validate_adaptation_placement()?;
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
            .map(greedy_id)
            .collect())
    }

    fn check_contrast_candidates(
        &self,
        target_id: usize,
        alternative_id: usize,
    ) -> CausalLmResult<()> {
        if target_id == alternative_id || target_id.max(alternative_id) >= self.config.vocabulary {
            return Err("next-token contrast requires two distinct in-vocabulary IDs".into());
        }
        Ok(())
    }

    /// Check the candidate IDs and shape before a client copies the prompt.
    pub fn next_token_contrast_input_count(
        &self,
        prompt_tokens: usize,
        target_id: usize,
        alternative_id: usize,
    ) -> CausalLmResult<usize> {
        self.check_contrast_candidates(target_id, alternative_id)?;
        self.input_count([1, prompt_tokens], true)
    }

    pub fn next_token_score_input_count(
        &self,
        prompt_tokens: usize,
        target_id: usize,
    ) -> CausalLmResult<usize> {
        TargetClassScore::validate_domain(self.config.vocabulary, target_id)?;
        self.input_count([1, prompt_tokens], true)
    }

    /// One readout of the prompt's final position, without feeding the target or copying the
    /// full vocabulary to the caller. Ties, rank, margin and NLL use tensor classification.
    pub fn next_token_score(
        &self,
        prompt_ids: &[usize],
        target_id: usize,
    ) -> CausalLmResult<TargetClassScore> {
        TargetClassScore::validate_domain(self.config.vocabulary, target_id)?;
        let logits = self.logits(prompt_ids, [1, prompt_ids.len()], true)?;
        Ok(TargetClassScore::from_logits(logits.data(), target_id)?)
    }

    /// Inspect one unpadded prompt. Neither candidate is appended to the input.
    /// The greedy choice is over the entire vocabulary; ties keep the lowest ID.
    pub fn next_token_contrast(
        &self,
        prompt_ids: &[usize],
        target_id: usize,
        alternative_id: usize,
    ) -> CausalLmResult<CausalLmTokenContrast> {
        self.check_contrast_candidates(target_id, alternative_id)?;
        let logits = self.logits(prompt_ids, [1, prompt_ids.len()], true)?;
        Ok(Self::contrast_row(
            logits.data(),
            prompt_ids.len(),
            target_id,
            alternative_id,
        ))
    }

    fn contrast_row(
        row: &[f32],
        prompt_tokens: usize,
        target_id: usize,
        alternative_id: usize,
    ) -> CausalLmTokenContrast {
        let target_logit = f64::from(row[target_id]);
        let alternative_logit = f64::from(row[alternative_id]);
        CausalLmTokenContrast {
            prompt_tokens,
            target_id,
            alternative_id,
            greedy_id: greedy_id(row),
            target_logit,
            alternative_logit,
            target_minus_alternative: target_logit - alternative_logit,
        }
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
        self.forward_blocks(self.embed(ids, shape)?, shape, 0..self.blocks.len())
    }

    fn forward_blocks(
        &self,
        mut hidden: Tensor,
        shape: [usize; 2],
        blocks: std::ops::Range<usize>,
    ) -> CausalLmResult<Tensor> {
        for index in blocks {
            hidden = self
                .run_block(index, hidden, shape, decoder::BlockPass::Forward)?
                .output;
        }
        Ok(hidden)
    }
}
