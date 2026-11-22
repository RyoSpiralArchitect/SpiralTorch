//! Shared native adaptation contract. One snapshot owns both parameter groups;
//! bindings never implement a second optimizer or backward pass.
use super::*;
use crate::geometry_artifact::GeometryAdapterTensorRole;
use std::collections::BTreeMap;

/// Which attached group(s) an atomic SGD step updates. Unselected groups remain
/// in the forward function and its derivative, rather than being removed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmAdaptationMode {
    Lora,
    Geometry,
    Joint,
}

impl std::str::FromStr for CausalLmAdaptationMode {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "lora" => Ok(Self::Lora),
            "geometry" => Ok(Self::Geometry),
            "joint" => Ok(Self::Joint),
            _ => Err("adaptation mode must be lora, geometry or joint"),
        }
    }
}

impl CausalLmAdaptationMode {
    pub(super) fn decoder_target(self, site: &str) -> learning::DecoderTarget<'_> {
        match self {
            Self::Lora => learning::DecoderTarget::Lora,
            Self::Geometry => learning::DecoderTarget::GeometryFrom(site),
            Self::Joint => learning::DecoderTarget::Joint,
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "paired_geometry_tests.rs"]
mod paired_geometry_tests;

#[derive(Debug)]
pub struct NativeCausalLmAdaptationGradients {
    base_id: String,
    lora_id: Option<String>,
    geometry_id: Option<String>,
    readout_accumulator: CausalLmReadoutAccumulator,
    loss: f32,
    target_tokens: usize,
    shape: [usize; 2],
    parameters: BTreeMap<String, BTreeMap<String, Vec<f32>>>,
    pub(super) observation: Option<Box<CausalLmAdaptationObservation>>,
    competition: Option<objective::CompetitionReport>,
    pub(super) line_work: Option<CausalLmLineTrainingWork>,
}

/// Block input before the earliest attached geometry, never an output of that geometry.
struct GeometryBoundary {
    layer: usize,
    site: String,
    hidden: Tensor,
}

/// Private to one training call: keep the exact batch layout, selected labels and prefix together.
pub(super) struct DeclaredSetTraining<'a> {
    model: &'a NativeCausalLm,
    window: PreparedCausalWindow,
    rows: Vec<DeclaredRivalRow>,
    boundary: GeometryBoundary,
}

impl DeclaredSetTraining<'_> {
    pub(super) fn gradients(
        &self,
    ) -> CausalLmResult<(
        NativeCausalLmAdaptationGradients,
        NativeCausalLmAdaptationGradients,
        DeclaredSetNll,
    )> {
        let (scores, [within, coverage]) =
            self.model
                .geometry_pair_pullbacks(&self.window, &self.boundary, |logits| {
                    logits.declared_set_nll_capture_within(
                        &self.window.labels,
                        &self.rows,
                        |seed| self.model.vocabulary_head_pullback(seed),
                    )
                })?;
        let mode = CausalLmAdaptationMode::Geometry;
        Ok((
            self.model.seal_adaptation_gradients(
                &self.window,
                mode,
                scores.within_nll as f32,
                None,
                within,
                None,
            )?,
            self.model.seal_adaptation_gradients(
                &self.window,
                mode,
                scores.coverage_nll as f32,
                None,
                coverage,
                None,
            )?,
            scores,
        ))
    }

    /// Only for the geometry-only candidate constructed from this call's immutable origin.
    pub(super) fn score_candidate(
        self,
        candidate: &NativeCausalLm,
    ) -> CausalLmResult<DeclaredSetNll> {
        if candidate.base_id != self.model.base_id
            || candidate.lora_artifact_id() != self.model.lora_artifact_id()
            || candidate.readout_accumulator != self.model.readout_accumulator
            || candidate.forward_storage != self.model.forward_storage
            || candidate.frozen_prefix_layers()? < self.boundary.layer
        {
            return Err("candidate does not share the declared-set batch's frozen prefix".into());
        }
        let hidden = candidate.forward_blocks(
            self.boundary.hidden,
            self.window.shape,
            self.boundary.layer..candidate.blocks.len(),
        )?;
        candidate.declared_set_scores(hidden, &self.window, &self.rows)
    }
}

impl NativeCausalLmAdaptationGradients {
    pub fn context_token_preference_report(&self) -> Option<&ContextTokenPreference> {
        match &self.competition {
            Some(objective::CompetitionReport::ContextToken(r)) => Some(r),
            _ => None,
        }
    }

    pub fn line_training_work(&self) -> Option<&CausalLmLineTrainingWork> {
        self.line_work.as_ref()
    }

    pub fn line_subset_report(&self) -> Option<&CausalLmLineSubsetReport> {
        match &self.competition {
            Some(objective::CompetitionReport::LineSubset(r)) => Some(r),
            _ => None,
        }
    }

    pub fn line_set_report(&self) -> Option<&CausalLmLineSetReport> {
        match &self.competition {
            Some(objective::CompetitionReport::LineSet(r)) => Some(r),
            _ => None,
        }
    }

    pub fn line_menu_report(&self) -> Option<&CausalLmLineMenuReport> {
        match &self.competition {
            Some(objective::CompetitionReport::LineMenu(r)) => Some(r),
            _ => None,
        }
    }
    pub fn competition(&self) -> Option<&RivalCrossEntropyReport> {
        match &self.competition {
            Some(objective::CompetitionReport::Top(r)) => Some(r),
            _ => None,
        }
    }

    pub fn declared_rival_report(&self) -> Option<&DeclaredRivalReport> {
        match &self.competition {
            Some(objective::CompetitionReport::Declared(r)) => Some(r),
            _ => None,
        }
    }

    pub fn observation(&self) -> Option<&CausalLmAdaptationObservation> {
        self.observation.as_deref()
    }

    pub fn loss(&self) -> f32 {
        self.loss
    }
    pub fn target_tokens(&self) -> usize {
        self.target_tokens
    }
    pub fn window_shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn base_model_identity(&self) -> &str {
        &self.base_id
    }
    pub fn lora_artifact_id(&self) -> Option<&str> {
        self.lora_id.as_deref()
    }
    pub fn geometry_artifact_id(&self) -> Option<&str> {
        self.geometry_id.as_deref()
    }
    pub fn readout_accumulator(&self) -> CausalLmReadoutAccumulator {
        self.readout_accumulator
    }
    pub fn parameters(&self) -> &BTreeMap<String, BTreeMap<String, Vec<f32>>> {
        &self.parameters
    }

    pub(super) fn into_parameters(self) -> BTreeMap<String, BTreeMap<String, Vec<f32>>> {
        self.parameters
    }
}

impl NativeCausalLm {
    /// This composition is intentionally one LoRA projection followed by one
    /// native geometry at the same attention output, before the residual add.
    pub(super) fn validate_adaptation_placement(&self) -> CausalLmResult<()> {
        if let (Some(lora), Some(geometry)) = (&self.lora, &self.geometry) {
            let placements = &geometry.artifact().record().placements;
            let expected = format!("transformer.h.{}.attn", lora.layer());
            if placements.len() != 1 || placements[0].path != expected {
                return Err("composed adaptation requires one geometry at the LoRA layer's attention output".into());
            }
        }
        Ok(())
    }

    pub fn validate_adaptation_mode(&self, mode: CausalLmAdaptationMode) -> CausalLmResult<()> {
        self.validate_adaptation_placement()?;
        if matches!(
            mode,
            CausalLmAdaptationMode::Lora | CausalLmAdaptationMode::Joint
        ) && self.lora.is_none()
        {
            return Err("selected adaptation mode requires a LoRA adapter".into());
        }
        if matches!(
            mode,
            CausalLmAdaptationMode::Geometry | CausalLmAdaptationMode::Joint
        ) {
            let runtime = self
                .geometry
                .as_ref()
                .ok_or("selected adaptation mode requires geometry")?;
            if !runtime.artifact().record().placements.iter().any(|p| {
                p.tensors
                    .iter()
                    .any(|t| t.role == GeometryAdapterTensorRole::Parameter && t.requires_grad)
            }) {
                return Err("selected geometry update has no trainable parameters".into());
            }
        }
        Ok(())
    }

    pub fn adaptation_gradient_input_count(&self, shape: [usize; 2]) -> CausalLmResult<usize> {
        self.adaptation_labeled_input_count(shape, false)
    }

    pub fn adaptation_labeled_input_count(
        &self,
        shape: [usize; 2],
        has_labels: bool,
    ) -> CausalLmResult<usize> {
        self.validate_adaptation_placement()?;
        if self.lora.is_some() {
            self.lora_pullback_input_count(shape, has_labels)
        } else {
            self.geometry_objective_input_count(shape, has_labels)
        }
    }

    /// Mean next-token CE and sealed gradients of every attached group. For a
    /// composed model a single decoder pass and reverse traversal produces both.
    pub fn adaptation_loss_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        self.adaptation_loss_gradients_with_labels(windows, shape, None, None)
    }

    /// Differentiate only the selected group(s), without removing unselected
    /// adapters from the forward or the pullback into earlier trainable weights.
    pub fn adaptation_loss_gradients_for(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        mode: CausalLmAdaptationMode,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        self.adaptation_loss_gradients_with_labels(windows, shape, None, Some(mode))
    }

    /// Mean CE of the selected shifted labels. Ignored positions remain in
    /// causal attention; only the readout and its reverse seed are selected.
    pub fn adaptation_loss_gradients_with_labels(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        mode: Option<CausalLmAdaptationMode>,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        self.adaptation_window_gradients(windows, shape, labels, mode, false)
    }

    /// Optionally measure the actual residual and adjoint while computing the same sealed gradients.
    pub fn adaptation_window_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        mode: Option<CausalLmAdaptationMode>,
        observe_geometry: bool,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        if let Some(mode) = mode {
            self.validate_adaptation_mode(mode)?;
        }
        self.adaptation_labeled_input_count(shape, labels.is_some())?;
        let prepared =
            self.prepare_objective(windows, shape, labels, CausalLmObjective::default())?;
        let hidden = self.embed(&prepared.input, prepared.shape)?;
        self.adaptation_gradients_from_prefix(
            &prepared,
            hidden,
            0,
            mode,
            observe_geometry.then(observation::AdaptationTrace::default),
        )
    }

    /// Admit the bounded candidate pool and geometry owner before a client copies IDs.
    pub fn geometry_competition_input_count(
        &self,
        shape: [usize; 2],
        has_labels: bool,
        recipe: RivalCrossEntropy,
    ) -> CausalLmResult<usize> {
        recipe.validate(self.config.vocabulary)?;
        self.validate_adaptation_mode(CausalLmAdaptationMode::Geometry)?;
        self.adaptation_labeled_input_count(shape, has_labels)
    }

    /// Backpropagate a mixture of ordinary and top-rival CE into geometry only.
    /// Attached LoRA remains in the forward computation and is not an update target.
    pub fn geometry_competition_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        recipe: RivalCrossEntropy,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        self.geometry_competition_input_count(shape, labels.is_some(), recipe)?;
        let mut prepared =
            self.prepare_objective(windows, shape, labels, CausalLmObjective::default())?;
        prepared.competition = Some(recipe.into());
        self.geometry_gradients_for_competition(&prepared)
    }

    /// The plan uses flat, unshifted `[batch,time]` target positions. The first
    /// token of each independent window has no predictor. Omitted active labels use full CE.
    pub fn geometry_declared_rival_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        plan: &[DeclaredRivalRow],
        mix: f64,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        self.validate_adaptation_mode(CausalLmAdaptationMode::Geometry)?;
        let count = self.adaptation_labeled_input_count(shape, labels.is_some())?;
        let (mut prepared, selected) =
            self.prepare_declared_rows(windows, shape, labels, plan, mix, count)?;
        prepared.competition = Some(objective::PreparedCompetition::Declared { selected, mix });
        self.geometry_gradients_for_competition(&prepared)
    }

    fn prepare_declared_rows(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        plan: &[DeclaredRivalRow],
        mix: f64,
        count: usize,
    ) -> CausalLmResult<(PreparedCausalWindow, Vec<DeclaredRivalRow>)> {
        DeclaredRivalRow::validate_plan(plan, count, self.config.vocabulary, mix)?;
        let prepared =
            self.prepare_objective(windows, shape, labels, CausalLmObjective::default())?;
        let mut selected = Vec::with_capacity(plan.len());
        for r in plan {
            let column = r.position % shape[1];
            if column == 0 {
                return Err("the first position in a causal window has no predictor".into());
            }
            let shifted = (r.position / shape[1]) * (shape[1] - 1) + column - 1;
            let row = match &prepared.selected_rows {
                Some(rows) => rows
                    .binary_search(&shifted)
                    .map_err(|_| "declared rival at an ignored target")?,
                None => shifted,
            };
            if usize::try_from(prepared.labels[row])? != r.target {
                return Err("declared target differs from the supervised label".into());
            }
            selected.push(DeclaredRivalRow {
                position: row,
                target: r.target,
                rivals: r.rivals.clone(),
            });
        }
        Ok((prepared, selected))
    }

    /// Admit both sealed gradient groups and both dense decoder adjoints before extracting IDs.
    pub fn geometry_gradient_pair_input_count(
        &self,
        shape: [usize; 2],
        has_labels: bool,
    ) -> CausalLmResult<usize> {
        self.validate_geometry_cohort_size(2)?;
        let count = self.adaptation_labeled_input_count(shape, has_labels)?;
        product(
            &[2, shape[0], shape[1] - 1, self.config.width],
            self.limits.max_activation_elements,
            "paired geometry adjoints",
        )?;
        Ok(count)
    }

    /// Return (full-vocabulary CE, declared-rival mixture) on exactly the same logits.
    /// Each is a separately usable geometry-only gradient of this immutable snapshot.
    /// Share the decoder forward, vocabulary forward and each block's replay, not the VJPs.
    pub fn geometry_full_and_declared_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        plan: &[DeclaredRivalRow],
        mix: f64,
    ) -> CausalLmResult<(
        NativeCausalLmAdaptationGradients,
        NativeCausalLmAdaptationGradients,
    )> {
        let count = self.geometry_gradient_pair_input_count(shape, labels.is_some())?;
        let (prepared, rows) =
            self.prepare_declared_rows(windows, shape, labels, plan, mix, count)?;
        let boundary = self.geometry_boundary(&prepared)?;
        let (report, [full, declared]) =
            self.geometry_pair_pullbacks(&prepared, &boundary, |logits| {
                logits.declared_rival_cross_entropy_capture_full(
                    &prepared.labels,
                    &rows,
                    mix,
                    |seed| self.vocabulary_head_pullback(seed),
                )
            })?;
        let mode = CausalLmAdaptationMode::Geometry;
        Ok((
            self.seal_adaptation_gradients(&prepared, mode, report.full_ce, None, full, None)?,
            self.seal_adaptation_gradients(
                &prepared,
                mode,
                report.loss,
                Some(objective::CompetitionReport::Declared(report)),
                declared,
                None,
            )?,
        ))
    }

    /// Score -log P(target | set) and -log P(set) with no backward pass or adapter requirement.
    pub fn declared_set_nll(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        plan: &[DeclaredRivalRow],
    ) -> CausalLmResult<DeclaredSetNll> {
        let count = self.loss_objective_input_count(shape, labels.is_some())?;
        let (prepared, rows) =
            self.prepare_declared_rows(windows, shape, labels, plan, 1., count)?;
        let hidden = self.decode_hidden(&prepared.input, prepared.shape)?;
        self.declared_set_scores(hidden, &prepared, &rows)
    }

    fn declared_set_scores(
        &self,
        hidden: Tensor,
        prepared: &PreparedCausalWindow,
        rows: &[DeclaredRivalRow],
    ) -> CausalLmResult<DeclaredSetNll> {
        Ok(self
            .causal_logits(hidden, prepared)?
            .declared_set_nll(&prepared.labels, rows)?)
    }

    /// Return (within-set gradient, set-coverage gradient, scores) on the same snapshot.
    /// Both handles can be used with the ordinary geometry optimizers or common-descent guard.
    pub fn geometry_declared_set_gradients(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        plan: &[DeclaredRivalRow],
    ) -> CausalLmResult<(
        NativeCausalLmAdaptationGradients,
        NativeCausalLmAdaptationGradients,
        DeclaredSetNll,
    )> {
        self.declared_set_training(windows, shape, labels, plan)?
            .gradients()
    }

    pub(super) fn declared_set_training(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        plan: &[DeclaredRivalRow],
    ) -> CausalLmResult<DeclaredSetTraining<'_>> {
        let count = self.geometry_gradient_pair_input_count(shape, labels.is_some())?;
        let (window, rows) = self.prepare_declared_rows(windows, shape, labels, plan, 1., count)?;
        let boundary = self.geometry_boundary(&window)?;
        Ok(DeclaredSetTraining {
            model: self,
            window,
            rows,
            boundary,
        })
    }

    fn geometry_boundary(&self, window: &PreparedCausalWindow) -> CausalLmResult<GeometryBoundary> {
        let (layer, site) = self.first_geometry_site()?;
        let hidden = self.forward_blocks(
            self.embed(&window.input, window.shape)?,
            window.shape,
            0..layer,
        )?;
        Ok(GeometryBoundary {
            layer,
            site,
            hidden,
        })
    }

    fn geometry_pair_pullbacks<R>(
        &self,
        prepared: &PreparedCausalWindow,
        boundary: &GeometryBoundary,
        seeds: impl FnOnce(&Tensor) -> CausalLmResult<(R, Tensor, Tensor)>,
    ) -> CausalLmResult<(R, [crate::geometry_runtime::ParameterGradients; 2])> {
        let learning::TrainingPrimal {
            checkpoints,
            hidden_rows,
            selected,
            logits,
        } = self.training_primal(
            prepared,
            boundary.hidden.clone(),
            boundary.layer,
            boundary.layer,
            None,
        )?;
        let (report, second_seed, first_normalized) = seeds(&logits)?;
        drop(logits);
        let second_normalized = self.vocabulary_head_pullback(&second_seed)?;
        drop(second_seed);
        let norm_vjp = |d| {
            selected.layer_norm_input_backward(&self.final_norm.gamma, d, self.config.norm_epsilon)
        };
        let adjoints = [norm_vjp(&first_normalized)?, norm_vjp(&second_normalized)?];
        drop((selected, first_normalized, second_normalized));
        let parameters = self.decoder_pullback_lanes_from_selected(
            checkpoints,
            hidden_rows,
            prepared.shape,
            prepared.selected_rows.as_deref(),
            adjoints,
            boundary.layer,
            learning::DecoderTarget::GeometryFrom(&boundary.site),
            None,
        )?;
        Ok((report, parameters))
    }

    pub(super) fn first_adaptation_site(&self) -> CausalLmResult<(usize, String)> {
        match &self.lora {
            Some(a) => Ok((a.layer(), format!("transformer.h.{}.attn", a.layer()))),
            None => self.first_geometry_site(),
        }
    }

    pub(super) fn geometry_gradients_for_competition(
        &self,
        prepared: &PreparedCausalWindow,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        let hidden = self.embed(&prepared.input, prepared.shape)?;
        self.adaptation_gradients_from_prefix(
            prepared,
            hidden,
            0,
            Some(CausalLmAdaptationMode::Geometry),
            None,
        )
    }

    pub fn admit_wave_factorial(
        &self,
        config: crate::geometry_runtime::WaveFactorConfig,
    ) -> CausalLmResult<()> {
        self.validate_adaptation_mode(CausalLmAdaptationMode::Joint)?;
        let site = &self
            .geometry_artifact()
            .ok_or("missing factorial geometry")?
            .record()
            .placements[0];
        config.admit(site)?;
        Ok(())
    }

    /// Hold the input, gate, bias, residual strength and arriving CE gradient fixed.
    /// Only the factual map participates in the decoder; alternatives are local VJPs.
    pub fn wave_factorial_observation(
        &self,
        windows: &[usize],
        shape: [usize; 2],
        labels: Option<&[i64]>,
        config: crate::geometry_runtime::WaveFactorConfig,
    ) -> CausalLmResult<Box<CausalLmAdaptationObservation>> {
        self.admit_wave_factorial(config)?;
        self.adaptation_labeled_input_count(shape, labels.is_some())?;
        let prepared =
            self.prepare_objective(windows, shape, labels, CausalLmObjective::default())?;
        let hidden = self.embed(&prepared.input, prepared.shape)?;
        self.adaptation_gradients_from_prefix(
            &prepared,
            hidden,
            0,
            Some(CausalLmAdaptationMode::Joint),
            Some(observation::AdaptationTrace::factorial(config)),
        )?
        .observation
        .ok_or_else(|| "missing wave factorial receipt".into())
    }

    /// Caller has admitted this owner, selected mode, complete context and
    /// selected logits. Hidden is the boundary before any participating adapter.
    pub(super) fn adaptation_gradients_from_prefix(
        &self,
        prepared: &PreparedCausalWindow,
        hidden: Tensor,
        depth: usize,
        mode: Option<CausalLmAdaptationMode>,
        trace: Option<observation::AdaptationTrace>,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        Ok(self
            .adaptation_gradients_capturing_reference(prepared, hidden, depth, mode, trace, None)?
            .0)
    }

    pub(super) fn adaptation_gradients_capturing_reference(
        &self,
        prepared: &PreparedCausalWindow,
        hidden: Tensor,
        depth: usize,
        mode: Option<CausalLmAdaptationMode>,
        mut trace: Option<observation::AdaptationTrace>,
        capture_rivals: Option<usize>,
    ) -> CausalLmResult<(
        NativeCausalLmAdaptationGradients,
        Option<RivalCoverageReference>,
    )> {
        let (first, site) = self.first_adaptation_site()?;
        let mode = mode.unwrap_or(match (&self.lora, &self.geometry) {
            (Some(_), Some(_)) => CausalLmAdaptationMode::Joint,
            (Some(_), None) => CausalLmAdaptationMode::Lora,
            (None, Some(_)) => CausalLmAdaptationMode::Geometry,
            (None, None) => return Err("native LM has no adaptation parameters".into()),
        });
        let target = mode.decoder_target(&site);
        let learning::TrainingReadout {
            checkpoints,
            hidden_rows,
            selected,
            d_normalized,
            loss,
            competition,
            logits,
        } = self.training_readout(
            prepared,
            hidden,
            depth,
            first,
            trace.as_mut().map(|t| &mut t.forward),
        )?;
        let reference = capture_rivals
            .map(|k| logits.rival_coverage_reference(&prepared.labels, k))
            .transpose()?;
        drop(logits);
        let parameters = self.decoder_parameter_pullback(
            checkpoints,
            hidden_rows,
            selected,
            prepared.shape,
            prepared.selected_rows.as_deref(),
            d_normalized,
            first,
            target,
            trace.as_mut(),
        )?;
        let gradients =
            self.seal_adaptation_gradients(prepared, mode, loss, competition, parameters, trace)?;
        Ok((gradients, reference))
    }

    pub(super) fn seal_adaptation_gradients(
        &self,
        prepared: &PreparedCausalWindow,
        mode: CausalLmAdaptationMode,
        loss: f32,
        competition: Option<objective::CompetitionReport>,
        parameters: crate::geometry_runtime::ParameterGradients,
        trace: Option<observation::AdaptationTrace>,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        let want_lora = mode != CausalLmAdaptationMode::Geometry;
        let placements = self
            .geometry
            .as_ref()
            .filter(|_| mode != CausalLmAdaptationMode::Lora)
            .map_or(&[][..], |g| g.artifact().record().placements.as_slice());
        if parameters.len() != usize::from(want_lora) + placements.len()
            || self
                .lora
                .as_ref()
                .is_some_and(|a| parameters.contains_key(&a.path()) != want_lora)
            || placements.iter().any(|p| !parameters.contains_key(&p.path))
        {
            return Err("incorrect selected adaptation gradient groups".into());
        }
        let target_tokens = prepared.target_tokens;
        let shape = prepared.window_shape;
        if !loss.is_finite()
            || parameters
                .values()
                .flat_map(|p| p.values())
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err("nonfinite adaptation loss or gradient".into());
        }
        let mut gradients = NativeCausalLmAdaptationGradients {
            base_id: self.base_id.clone(),
            lora_id: self.lora_artifact_id().map(str::to_owned),
            geometry_id: self.geometry_artifact_id().map(str::to_owned),
            readout_accumulator: self.readout_accumulator,
            loss,
            target_tokens,
            shape,
            parameters,
            observation: None,
            line_work: None,
            competition,
        };
        if let Some(trace) = trace {
            gradients.observation = Some(Box::new(trace.finish(self, mode, &gradients)?));
        }
        Ok(gradients)
    }

    /// Construct both updates from the same immutable source, committing only
    /// after every requested group validates. No intermediate model is exposed.
    pub fn step_adaptation_sgd(
        &self,
        gradients: &NativeCausalLmAdaptationGradients,
        mode: CausalLmAdaptationMode,
        learning_rate: f32,
    ) -> CausalLmResult<Self> {
        self.validate_adaptation_mode(mode)?;
        if gradients.base_id != self.base_id
            || gradients.lora_id.as_deref() != self.lora_artifact_id()
            || gradients.geometry_id.as_deref() != self.geometry_artifact_id()
            || gradients.readout_accumulator != self.readout_accumulator
            || !learning_rate.is_finite()
            || learning_rate <= 0.
        {
            return Err("adaptation gradients belong to a different snapshot/policy or learning rate is invalid".into());
        }
        let mut next = self.clone();
        if matches!(
            mode,
            CausalLmAdaptationMode::Lora | CausalLmAdaptationMode::Joint
        ) {
            let adapter = self.lora.as_ref().unwrap();
            let values = gradients
                .parameters
                .get(&adapter.path())
                .ok_or("missing LoRA parameters")?;
            let a = values.get("a").ok_or("missing LoRA A gradient")?;
            let b = values.get("b").ok_or("missing LoRA B gradient")?;
            next.lora = self.with_lora_parameter_step(a, b, learning_rate)?.lora;
        }
        if matches!(
            mode,
            CausalLmAdaptationMode::Geometry | CausalLmAdaptationMode::Joint
        ) {
            let runtime = self.geometry.as_ref().unwrap();
            let id = runtime.artifact().artifact_id();
            let candidate = if let Some(lora) = &self.lora {
                // Pass geometry's placement map by reference without copying LoRA A/B or any channel vectors.
                let external = lora.path();
                let view = gradients
                    .parameters
                    .iter()
                    .filter(|(path, _)| *path != &external)
                    .map(|(path, g)| (path.clone(), g))
                    .collect();
                runtime.with_gradients_sgd(id, &view, learning_rate)?
            } else {
                runtime.with_gradients_sgd(id, &gradients.parameters, learning_rate)?
            };
            next.geometry = Some(Arc::new(candidate));
        }
        Ok(next)
    }
}
