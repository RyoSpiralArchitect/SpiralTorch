//! One target, two independent causal prefixes; the ordering is part of supervision.
use super::*;

impl NativeCausalLm {
    /// Bound both packed lanes and both selected vocabulary rows before a client copies any tokens.
    pub fn context_token_preference_shape(
        &self,
        preferred_len: usize,
        other_len: usize,
        target: usize,
        margin: f64,
        training: bool,
    ) -> CausalLmResult<[usize; 2]> {
        ContextTokenPreference::validate_domain(self.config.vocabulary, target, margin)?;
        if preferred_len == 0 || other_len == 0 {
            return Err("context preference requires two nonempty prefixes".into());
        }
        let width = preferred_len
            .max(other_len)
            .checked_add(1)
            .ok_or("context preference width overflow")?;
        let shape = [2, width];
        if training {
            self.validate_adaptation_mode(CausalLmAdaptationMode::Geometry)?;
            self.adaptation_labeled_input_count(shape, true)?;
        } else {
            self.loss_objective_input_count(shape, true)?;
        }
        self.logit_count(2)?;
        Ok(shape)
    }

    fn prepare_context_token_preference(
        &self,
        preferred: &[usize],
        other: &[usize],
        target: usize,
        margin: f64,
        training: bool,
    ) -> CausalLmResult<PreparedCausalWindow> {
        let shape = self.context_token_preference_shape(
            preferred.len(),
            other.len(),
            target,
            margin,
            training,
        )?;
        if preferred == other {
            return Err("context preference needs distinct tokenized prefixes".into());
        }
        let width = shape[1];
        let mut windows = vec![0; 2 * width];
        let mut labels = vec![-100; windows.len()];
        // Right padding is strictly after the last supervised predictor of the shorter lane.
        // Only that lane's own last real prefix position may seed its decision.
        for ((ids, ys), ctx) in windows
            .chunks_exact_mut(width)
            .zip(labels.chunks_exact_mut(width))
            .zip([preferred, other])
        {
            ids[..ctx.len()].copy_from_slice(ctx);
            ys[ctx.len()] = target as i64;
        }
        let mut p =
            self.prepare_objective(&windows, shape, Some(&labels), CausalLmObjective::default())?;
        p.competition = Some(objective::PreparedCompetition::ContextToken { target, margin });
        Ok(p)
    }

    /// softplus(margin - log P(target|preferred) + log P(target|other)), without updating weights.
    pub fn context_token_preference(
        &self,
        preferred: &[usize],
        other: &[usize],
        target: usize,
        margin: f64,
    ) -> CausalLmResult<ContextTokenPreference> {
        let p = self.prepare_context_token_preference(preferred, other, target, margin, false)?;
        let hidden = self.decode_hidden(&p.input, p.shape)?;
        Ok(self
            .causal_logits(hidden, &p)?
            .context_token_preference(target, margin)?)
    }

    /// Sealed geometry-only pullback of one ordered pair. The frozen LoRA stays in the graph.
    pub fn geometry_context_token_preference_gradients(
        &self,
        preferred: &[usize],
        other: &[usize],
        target: usize,
        margin: f64,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        let p = self.prepare_context_token_preference(preferred, other, target, margin, true)?;
        self.geometry_gradients_for_competition(&p)
    }
}
