//! Geometry-only SGD controls. The base and any attached LoRA remain in both forwards.
use super::*;
use crate::geometry_artifact::GeometryDeltaMatch;

fn geometry(model: &NativeCausalLm) -> CausalLmResult<&GeometryAdapterArtifact> {
    model
        .geometry_artifact()
        .ok_or_else(|| "paired SGD requires attached geometry".into())
}

/// Two immutable snapshots for one controlled SGD update. Both have the same frozen owners;
/// their trainable geometry values may already differ as a result of previous steps.
pub struct CausalLmGeometryCompetitionPair<'a> {
    reference: &'a NativeCausalLm,
    mixed: &'a NativeCausalLm,
    recipe: RivalCrossEntropy,
    learning_rate: f32,
    depth: usize,
    initial_separation_l2: f64,
    guard: Option<RivalCoverageGuard>,
}

#[derive(Debug, Serialize)]
pub struct CausalLmGeometryCompetitionStep {
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub window_shape: [usize; 2],
    pub target_tokens: usize,
    pub shared_prefix_layers: usize,
    pub learning_rate: f32,
    pub pre_arm_separation_l2: f64,
    pub reference_pre_geometry_id: String,
    pub reference_post_geometry_id: String,
    pub reference_loss: f32,
    /// Pre-update scores on the mixed snapshot, not a post-step evaluation.
    pub competition: RivalCrossEntropyReport,
    /// requested_l2 is the realized ordinary CE step in rounded parameter coordinates.
    pub mixed_step: GeometryDeltaMatch,
}

impl NativeCausalLm {
    /// Apply a sealed geometry gradient at a specified global distance. First form the same
    /// rounded proposal as ordinary SGD; do not normalize the unrounded gradient instead.
    pub fn step_geometry_sgd_at_distance(
        &self,
        gradients: &NativeCausalLmAdaptationGradients,
        learning_rate: f32,
        target_l2: f64,
    ) -> CausalLmResult<(Self, GeometryDeltaMatch)> {
        GeometryDeltaMatch::validate_requested_l2(target_l2)?;
        let origin = geometry(self)?;
        let proposal =
            self.step_adaptation_sgd(gradients, CausalLmAdaptationMode::Geometry, learning_rate)?;
        let endpoint = geometry(&proposal)?;
        let (artifact, receipt) = origin.match_trainable_delta(endpoint, target_l2)?;
        let next = if receipt.result_id == receipt.origin_id {
            self.clone()
        } else if receipt.result_id == receipt.direction_id {
            proposal
        } else {
            self.with_geometry(artifact)?
        };
        Ok((next, receipt))
    }

    /// Prepare a paired update with this model as the ordinary CE control. No cursor, cached
    /// activation or optimizer state is advanced by creating or consuming the pair.
    pub fn geometry_competition_pair<'a>(
        &'a self,
        mixed: &'a Self,
        recipe: RivalCrossEntropy,
        learning_rate: f32,
    ) -> CausalLmResult<CausalLmGeometryCompetitionPair<'a>> {
        if !learning_rate.is_finite() || learning_rate <= 0. {
            return Err("learning rate must be finite and positive".into());
        }
        recipe.validate(self.config.vocabulary)?;
        for model in [self, mixed] {
            model.validate_adaptation_mode(CausalLmAdaptationMode::Geometry)?;
        }
        let depth = Self::comparison_prefix_layers(&[self, mixed])?;
        if self.lora_artifact_id() != mixed.lora_artifact_id() {
            return Err("paired geometry SGD requires identical frozen LoRA owners".into());
        }
        let initial_separation_l2 = geometry(self)?.trainable_delta_l2(geometry(mixed)?)?;
        Ok(CausalLmGeometryCompetitionPair {
            reference: self,
            mixed,
            recipe,
            learning_rate,
            depth,
            initial_separation_l2,
            guard: None,
        })
    }
}

impl CausalLmGeometryCompetitionPair<'_> {
    /// Penalize only losses of probability mass on the *pre-update CE reference's* target
    /// and rivals. Zero is the original algorithm, including its receipt and memory use.
    pub fn with_coverage_guard(mut self, strength: f64) -> CausalLmResult<Self> {
        let guard = RivalCoverageGuard::new(strength)?;
        self.guard = guard.enabled().then_some(guard);
        Ok(self)
    }

    /// Admit the full context and checkpoint work on both variants before copying foreign IDs.
    pub fn input_count(&self, shape: [usize; 2], has_labels: bool) -> CausalLmResult<usize> {
        self.mixed
            .adaptation_labeled_input_count(shape, has_labels)?;
        self.reference
            .adaptation_labeled_input_count(shape, has_labels)
    }

    /// Return both updated models or neither. Share only blocks strictly before both adapters;
    /// compute each objective and backward pass on its own current snapshot.
    pub fn step(
        self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
    ) -> CausalLmResult<(
        NativeCausalLm,
        NativeCausalLm,
        CausalLmGeometryCompetitionStep,
    )> {
        self.input_count(shape, labels.is_some())?;
        let ce = self.reference;
        let mode = CausalLmAdaptationMode::Geometry;
        let mut batch =
            ce.prepare_objective(windows, shape, labels, CausalLmObjective::default())?;
        self.mixed.logit_count(batch.target_tokens)?;
        let prefix = ce.forward_blocks(
            ce.embed(&batch.input, batch.shape)?,
            batch.shape,
            0..self.depth,
        )?;
        let (next_ce, reference_loss, coverage) = {
            let (grad, reference) = ce.adaptation_gradients_capturing_reference(
                &batch,
                prefix.clone(),
                self.depth,
                Some(mode),
                None,
                self.guard.map(|_| self.recipe.rivals),
            )?;
            (
                ce.step_adaptation_sgd(&grad, mode, self.learning_rate)?,
                grad.loss(),
                reference,
            )
        };
        let before = geometry(ce)?;
        let after = geometry(&next_ce)?;
        let distance = before.trainable_delta_l2(after)?;

        batch.competition = Some(objective::PreparedCompetition::Top {
            recipe: self.recipe,
            guard: match (coverage, self.guard) {
                (None, None) => None,
                (Some(reference), Some(guard)) => Some((reference, guard)),
                _ => {
                    return Err(
                        "coverage reference does not match the requested training policy".into(),
                    )
                }
            },
        });
        let grad = self.mixed.adaptation_gradients_from_prefix(
            &batch,
            prefix,
            self.depth,
            Some(mode),
            None,
        )?;
        let competition = *grad.competition().ok_or("missing competition objective")?;
        let (next_mixed, mixed_step) =
            self.mixed
                .step_geometry_sgd_at_distance(&grad, self.learning_rate, distance)?;
        let report = CausalLmGeometryCompetitionStep {
            base_model_identity: ce.base_id.clone(),
            lora_artifact_id: ce.lora_artifact_id().map(str::to_owned),
            readout_accumulator: ce.readout_accumulator,
            window_shape: shape,
            target_tokens: batch.target_tokens,
            shared_prefix_layers: self.depth,
            learning_rate: self.learning_rate,
            pre_arm_separation_l2: self.initial_separation_l2,
            reference_pre_geometry_id: before.artifact_id().to_owned(),
            reference_post_geometry_id: after.artifact_id().to_owned(),
            reference_loss,
            competition,
            mixed_step,
        };
        Ok((next_ce, next_mixed, report))
    }
}
