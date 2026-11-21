//! Paired evaluation of pinned serving snapshots; training data is not used.
use super::*;
use crate::language_windows::CausalDocumentWindows;

#[derive(Debug, Serialize)]
pub struct CausalLmConditionalScoredSnapshot {
    /// Index in the comparison; the frozen source is always variant zero.
    pub variant: usize,
    pub checkpoint_id: String,
    pub projection_id: String,
    pub layer: usize,
    pub site: &'static str,
    pub position: u64,
    pub total_batches: u64,
}

#[derive(Debug, Serialize)]
pub struct CausalLmConditionalAnswerComparison {
    pub schema: &'static str,
    pub base_model_identity: String,
    pub snapshots: Vec<CausalLmConditionalScoredSnapshot>,
    pub comparison: CausalLmAnswerComparison,
}

impl NativeCausalLmConditionalServingSnapshot {
    pub const MAX_COMPARISON_SNAPSHOTS: usize = MAX_EVALUATION_VARIANTS - 1;

    fn comparison_models<'a>(
        source: &'a NativeCausalLm,
        snapshots: &[&'a Self],
    ) -> CausalLmResult<Vec<&'a NativeCausalLm>> {
        if !(1..=Self::MAX_COMPARISON_SNAPSHOTS).contains(&snapshots.len()) {
            return Err("conditional comparison requires 1 to 15 snapshots".into());
        }
        let mut models = Vec::with_capacity(snapshots.len() + 1);
        models.push(source);
        for &snapshot in snapshots {
            let model = &snapshot.model;
            if model.base_model_identity() != source.base_model_identity()
                || model.lora_artifact_id() != source.lora_artifact_id()
                || model.geometry_artifact_id() != source.geometry_artifact_id()
                || model.readout_accumulator() != source.readout_accumulator()
            {
                return Err("conditional comparison requires the exact frozen source owner".into());
            }
            models.push(model);
        }
        Ok(models)
    }

    /// Check source and roster identity before a binding reads evaluation inputs.
    pub fn check_comparison_sources(
        source: &NativeCausalLm,
        snapshots: &[&Self],
    ) -> CausalLmResult<()> {
        Self::comparison_models(source, snapshots).map(|_| ())
    }

    pub fn answer_comparison_plan(
        source: &NativeCausalLm,
        snapshots: &[&Self],
        lengths: &[usize],
        config: CausalLmEvaluationConfig,
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        let models = Self::comparison_models(source, snapshots)?;
        NativeCausalLm::answer_comparison_plan(&models, lengths, config, has_labels)
    }

    /// Score the same supervised spans on the source and each saved projection.
    /// A packed projection is not reflected in the frozen base ID, so shared
    /// layers must also stop before the earliest rank-one override.
    pub fn evaluate_variants_by_answer(
        source: &NativeCausalLm,
        snapshots: &[&Self],
        documents: &[Vec<usize>],
        labels: Option<&[Vec<i64>]>,
        config: CausalLmEvaluationConfig,
    ) -> CausalLmResult<CausalLmConditionalAnswerComparison> {
        let models = Self::comparison_models(source, snapshots)?;
        let cap = snapshots
            .iter()
            .fold(source.config.layers, |depth, s| depth.min(s.layer));
        let comparison = NativeCausalLm::evaluate_variants_by_answer_with_prefix_cap(
            &models, documents, labels, config, cap,
        )?;
        Ok(CausalLmConditionalAnswerComparison {
            schema: "spiraltorch.native_causal_lm_conditional_answer_comparison.v1",
            base_model_identity: source.base_model_identity().to_owned(),
            snapshots: snapshots
                .iter()
                .enumerate()
                .map(|(i, s)| CausalLmConditionalScoredSnapshot {
                    variant: i + 1,
                    checkpoint_id: s.checkpoint_id.clone(),
                    projection_id: s.projection_id.clone(),
                    layer: s.layer,
                    site: s.site.as_str(),
                    position: s.position,
                    total_batches: s.total_batches,
                })
                .collect(),
            comparison,
        })
    }
}
