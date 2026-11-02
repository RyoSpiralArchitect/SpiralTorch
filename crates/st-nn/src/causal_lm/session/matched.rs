//! One bounded lockstep update across independently owned adaptation states.
use super::*;

#[derive(Debug, Serialize)]
pub struct CausalLmMatchedTrainingStep {
    /// Full frozen decoder blocks executed once, not a wall-time speedup.
    pub shared_prefix_layers: usize,
    pub steps: Vec<CausalLmTrainingSessionStep>,
}

impl NativeCausalLmTrainingSession {
    pub const MAX_MATCHED_VARIANTS: usize = MAX_EVALUATION_VARIANTS;

    /// Share immutable corpus storage and its admitted schedule, never an
    /// advanced cursor. Each variant must admit every shape under its own limits.
    pub fn fork_variant(
        &self,
        model: NativeCausalLm,
        mode: CausalLmAdaptationMode,
    ) -> CausalLmResult<Self> {
        if self.updates != 0 {
            return Err("fork_variant requires an unadvanced training session".into());
        }
        if model.base_id != self.model.base_id
            || model.readout_accumulator != self.model.readout_accumulator
        {
            return Err("training variants require the same frozen base and readout policy".into());
        }
        let config = CausalLmTrainingSessionConfig {
            mode,
            ..self.config
        };
        Self::validate_policy(&model, config, self.batch_windows)?;
        Self::admit_plan_shapes(&model, self.plan(), self.batch_windows, self.data_policy)?;
        model.logit_count(self.max_batch_targets)?;
        Ok(Self {
            model,
            config,
            corpus: Arc::clone(&self.corpus),
            updates: 0,
            data_policy: self.data_policy,
            batch_windows: self.batch_windows,
            live_ranges: self.live_ranges.clone(),
            max_batch_targets: self.max_batch_targets,
        })
    }

    /// Advance 2..=16 distinct sessions on one identical batch, or none on
    /// failure. Only the embedding and blocks before *every* attached adapter
    /// are shared. No activations are cached across steps or serialized.
    pub fn step_matched(
        sessions: &mut [&mut Self],
    ) -> CausalLmResult<Option<CausalLmMatchedTrainingStep>> {
        Self::step_matched_with_observation(sessions, false)
    }

    pub fn step_matched_with_observation(
        sessions: &mut [&mut Self],
        observe_geometry: bool,
    ) -> CausalLmResult<Option<CausalLmMatchedTrainingStep>> {
        if !(2..=Self::MAX_MATCHED_VARIANTS).contains(&sessions.len()) {
            return Err("matched training requires 2 to 16 sessions".into());
        }
        let models: Vec<_> = sessions.iter().map(|s| &s.model).collect();
        let depth = NativeCausalLm::comparison_prefix_layers(&models)?;
        let first = &*sessions[0];
        for s in &sessions[1..] {
            if s.data_id() != first.data_id()
                || s.data_policy != first.data_policy
                || s.batch_windows != first.batch_windows
                || s.updates != first.updates
                || s.config.context_length != first.config.context_length
                || s.config.epochs != first.config.epochs
                || s.config.learning_rate != first.config.learning_rate
                || s.total_updates() != first.total_updates()
            {
                return Err(
                    "matched training requires identical data, schedule, learning rate and cursor"
                        .into(),
                );
            }
        }
        if first.exhausted() {
            return Ok(None);
        }
        let range = first.next_range();
        let windows = &first.plan().windows()[range.clone()];
        let shape = [range.len(), windows[0].length as usize];
        let (ids, labels) = first.corpus.batch(windows);
        let prepared = first.model.prepare_objective(
            &ids,
            shape,
            labels.as_deref(),
            CausalLmObjective::default(),
        )?;
        for model in &models {
            model.adaptation_labeled_input_count(shape, labels.is_some())?;
            model.logit_count(prepared.target_tokens)?;
        }
        let prefix = first.model.forward_blocks(
            first.model.embed(&prepared.input, prepared.shape)?,
            prepared.shape,
            0..depth,
        )?;
        let mut candidates = Vec::with_capacity(sessions.len());
        for (s, hidden) in sessions
            .iter()
            .zip(std::iter::repeat_n(prefix, sessions.len()))
        {
            let gradients = s.model.adaptation_gradients_from_prefix(
                &prepared,
                hidden,
                depth,
                Some(s.config.mode),
                observe_geometry.then(observation::AdaptationTrace::default),
            )?;
            candidates.push(s.sgd_candidate(range.clone(), gradients)?);
        }
        let steps = sessions
            .iter_mut()
            .zip(candidates)
            .map(|(s, (model, report))| s.commit_candidate(model, report))
            .collect();
        Ok(Some(CausalLmMatchedTrainingStep {
            shared_prefix_layers: depth,
            steps,
        }))
    }
}
