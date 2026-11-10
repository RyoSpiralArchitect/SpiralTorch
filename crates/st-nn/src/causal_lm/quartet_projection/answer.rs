use super::*;

mod guard;
pub use guard::{
    CausalLmAnswerCorrection, CausalLmAnswerGuard, CausalLmAnswerGuardStatus,
    CausalLmAnswerGuardStep, CausalLmAnswerPoint, CausalLmAnswerProtection, CausalLmAnswerTrial,
};

/// Optional supervision on the relationship between four indexed prompts.
/// Gold CE labels are independent of the quartet orientation and contrast IDs.
#[derive(Clone, Copy, Debug)]
pub struct CausalLmAnswerBinding<'a> {
    pub ids: [usize; 2],
    pub quartets: &'a [[usize; 4]],
    /// Convex loss weight, held fixed for this differentiation.
    pub mix: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmLossGradient {
    pub loss: f64,
    pub gradient: Vec<f64>,
    pub norm: f64,
}

impl CausalLmLossGradient {
    fn new(loss: f64, gradient: Vec<f64>) -> CausalLmResult<Self> {
        let norm = gradient.iter().fold(0.0_f64, |a, &v| a.hypot(v));
        if !loss.is_finite() || !norm.is_finite() {
            return Err("nonfinite projection objective".into());
        }
        Ok(Self {
            loss,
            gradient,
            norm,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmAnswerProjectionStep {
    pub answer: CausalLmLossGradient,
    pub interaction: Option<CausalLmLossGradient>,
    pub combined: CausalLmLossGradient,
    pub mix: f64,
    pub prompt_forwards: usize,
    pub projection_pullbacks: usize,
}

impl NativeCausalLmRankOneProjection {
    /// Mean next-token cross entropy over the entire vocabulary. Optional binding
    /// adds a separate, explicitly weighted interaction loss. The two adjoints
    /// share one factual forward per prompt, but not a prematurely rounded f32
    /// logit seed. Supplying the binding at mix=0 or 1 still measures both terms.
    pub fn answer_gradients(
        &self,
        prompts: &[Vec<usize>],
        gold: &[usize],
        binding: Option<CausalLmAnswerBinding<'_>>,
    ) -> CausalLmResult<CausalLmAnswerProjectionStep> {
        self.admit_answers(prompts, gold, binding, true)?;
        let m = &self.current;
        let n = prompts.len();
        let w = m.config.width;
        let seeds = 1 + usize::from(binding.is_some());
        let contrast_seed = if let Some(b) = binding {
            let mut v = vec![0.; m.config.vocabulary];
            v[b.ids[0]] = 1.;
            v[b.ids[1]] = -1.;
            Some(Tensor::from_vec(1, v.len(), v)?)
        } else {
            None
        };

        let mut ce_total = sum::Sum::default();
        let mut ce_grad: Vec<_> = (0..w).map(|_| sum::Sum::default()).collect();
        let mut gaps = Vec::new();
        let mut gap_grad = Vec::new();
        for (p, &y) in prompts.iter().zip(gold) {
            let row = m.projection_last_row(p, self.spec)?;
            let (loss, seed) = learning::ordinary_loss_seed(
                &row.logits,
                &[y as i64],
                CrossEntropyConfig::default(),
            )?;
            if let (Some(b), Some(d)) = (binding, contrast_seed.as_ref()) {
                let z = row.logits.data();
                gaps.push(f64::from(z[b.ids[0]]) - f64::from(z[b.ids[1]]));
                gap_grad.push(m.fixed_left_seed(row.clone(), d, self.spec, &self.left)?);
            }
            ce_total.add(f64::from(loss));
            for (s, g) in ce_grad
                .iter_mut()
                .zip(m.fixed_left_seed(row, &seed, self.spec, &self.left)?)
            {
                s.add(g);
            }
        }
        let answer = CausalLmLossGradient::new(
            ce_total.value() / n as f64,
            ce_grad.into_iter().map(|s| s.value() / n as f64).collect(),
        )?;
        let (interaction, combined, mix) = if let Some(b) = binding {
            let (loss, alpha) = CausalLmQuartetObjective::Interaction.mean_seed(&gaps, b.quartets);
            let mut v: Vec<_> = (0..w).map(|_| sum::Sum::default()).collect();
            for (a, gs) in alpha.into_iter().zip(gap_grad) {
                for (s, g) in v.iter_mut().zip(gs) {
                    s.add(a * g);
                }
            }
            let term = CausalLmLossGradient::new(loss, v.into_iter().map(|s| s.value()).collect())?;
            let mix = |x, y| (1. - b.mix) * x + b.mix * y;
            let combined = CausalLmLossGradient::new(
                mix(answer.loss, term.loss),
                answer
                    .gradient
                    .iter()
                    .zip(&term.gradient)
                    .map(|(&x, &y)| mix(x, y))
                    .collect(),
            )?;
            (Some(term), combined, b.mix)
        } else {
            (None, answer.clone(), 0.)
        };
        Ok(CausalLmAnswerProjectionStep {
            answer,
            interaction,
            combined,
            mix,
            prompt_forwards: n,
            projection_pullbacks: n * seeds,
        })
    }

    fn admit_answers(
        &self,
        prompts: &[Vec<usize>],
        gold: &[usize],
        binding: Option<CausalLmAnswerBinding<'_>>,
        backward: bool,
    ) -> CausalLmResult<()> {
        let m = &self.current;
        let n = prompts.len();
        let w = m.config.width;
        if !(1..=MAX_OUTPUT_CONTRAST_ROWS).contains(&n)
            || n != gold.len()
            || gold.iter().any(|&id| id >= m.config.vocabulary)
        {
            return Err("answer supervision requires a vocabulary label per prompt".into());
        }
        product(
            &[n, w],
            output_margin::MAX_BANK_VALUES,
            "answer projection roster",
        )?;
        let mut tokens = 0usize;
        for p in prompts {
            m.projection_work_dimensions(p.len(), self.spec, backward)?;
            tokens = tokens.checked_add(p.len()).ok_or("answer token overflow")?;
            if p.iter().any(|&id| id >= m.config.vocabulary) {
                return Err("invalid answer prompt token".into());
            }
        }
        product(
            &[tokens, w, 1 + usize::from(backward && binding.is_some())],
            output_margin::MAX_CAPTURE_TERMS,
            "answer projection capture work",
        )?;
        if let Some(b) = binding {
            if !(0.0..=1.).contains(&b.mix) {
                return Err("answer mixture outside [0,1]".into());
            }
            m.check_contrast_candidates(b.ids[0], b.ids[1])?;
            validate_groups(prompts, b.quartets)?;
        }
        Ok(())
    }

    /// Apply one normalized descent step to the right factor. A refused batch
    /// or nonfinite proposal leaves the prior weights and serving recipe intact.
    pub fn step_answers(
        &mut self,
        prompts: &[Vec<usize>],
        gold: &[usize],
        binding: Option<CausalLmAnswerBinding<'_>>,
        length: f64,
    ) -> CausalLmResult<CausalLmAnswerProjectionStep> {
        if !length.is_finite() || length <= 0. {
            return Err("invalid answer projection step length".into());
        }
        let r = self.answer_gradients(prompts, gold, binding)?;
        self.descend(&r.combined.gradient, r.combined.norm, length)?;
        Ok(r)
    }
}
