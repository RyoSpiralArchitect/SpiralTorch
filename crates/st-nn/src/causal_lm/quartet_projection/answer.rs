use super::*;

mod guard;
mod observation;
pub use guard::{
    CausalLmAnswerCorrection, CausalLmAnswerGuard, CausalLmAnswerGuardStatus,
    CausalLmAnswerGuardStep, CausalLmAnswerPoint, CausalLmAnswerProtection, CausalLmAnswerTrial,
};
pub use observation::CausalLmBindingGradientObservation;
use observation::{candidate_mass_seed, PairObservation};

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

/// Two candidates scored from the same full-vocabulary next-token distribution.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct CausalLmPairedAnswerScore {
    pub gold_nll: f64,
    pub alternative_nll: f64,
    /// Computed directly from logits, without subtracting rounded NLLs.
    pub gold_minus_alternative: f64,
    pub top_token: usize,
}

impl CausalLmPairedAnswerScore {
    fn from_logits(z: &[f32], [gold, alt]: [usize; 2]) -> CausalLmResult<Self> {
        let mut top_token = 0;
        for (i, &logit) in z.iter().enumerate() {
            if !logit.is_finite() {
                return Err("nonfinite paired answer logits".into());
            }
            if logit > z[top_token] {
                top_token = i;
            }
        }
        let max = f64::from(z[top_token]);
        let mut tail = sum::Sum::default();
        for (i, &logit) in z.iter().enumerate() {
            if i != top_token {
                tail.add((f64::from(logit) - max).exp());
            }
        }
        // Adding the maximum's unit term before ln would round away tiny tails.
        let denom = tail.value().ln_1p();
        let (g, a) = (f64::from(z[gold]), f64::from(z[alt]));
        Ok(Self {
            gold_nll: (max - g) + denom,
            alternative_nll: (max - a) + denom,
            gold_minus_alternative: g - a,
            top_token,
        })
    }
}

fn answer_capture_work(
    tokens: usize,
    width: usize,
    seeds: usize,
    label: &'static str,
) -> CausalLmResult<usize> {
    product(
        &[tokens, width, seeds],
        output_margin::MAX_CAPTURE_TERMS,
        label,
    )
}

impl NativeCausalLmRankOneProjection {
    /// Read-only next-token scoring with no gradient, mutation or logits export.
    /// Each pair is ordered [gold, alternative]; equal candidates are rejected.
    pub fn score_answer_pairs(
        &self,
        prompts: &[Vec<usize>],
        candidates: &[[usize; 2]],
    ) -> CausalLmResult<Vec<CausalLmPairedAnswerScore>> {
        let m = &self.current;
        if prompts.len() != candidates.len() || candidates.len() > MAX_OUTPUT_CONTRAST_ROWS {
            return Err("invalid paired answer roster".into());
        }
        for &[gold, alt] in candidates {
            m.check_contrast_candidates(gold, alt)?;
        }
        let gold: Vec<_> = candidates.iter().map(|p| p[0]).collect();
        self.admit_answers(prompts, &gold, None, false)?;
        prompts
            .iter()
            .zip(candidates)
            .map(|(prompt, &pair)| {
                CausalLmPairedAnswerScore::from_logits(&self.last_logits(prompt)?, pair)
            })
            .collect()
    }

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
        self.answer_gradients_parts(prompts, gold, binding, false)
            .map(|(training, _)| training)
    }

    /// Observe candidate-mass descent and conditional-answer descent at
    /// the same sealed state and on the same forward pass as the training
    /// objective. Every gold ID must be one of the two binding candidates.
    pub fn observe_binding_gradients(
        &self,
        prompts: &[Vec<usize>],
        gold: &[usize],
        binding: CausalLmAnswerBinding<'_>,
    ) -> CausalLmResult<CausalLmBindingGradientObservation> {
        let (training, split) = self.answer_gradients_parts(prompts, gold, Some(binding), true)?;
        split
            .ok_or("missing paired gradient decomposition")?
            .finish(self.projection_id(), training)
    }

    fn answer_gradients_parts(
        &self,
        prompts: &[Vec<usize>],
        gold: &[usize],
        binding: Option<CausalLmAnswerBinding<'_>>,
        observe_pair: bool,
    ) -> CausalLmResult<(CausalLmAnswerProjectionStep, Option<PairObservation>)> {
        self.admit_answers(prompts, gold, binding, true)?;
        let m = &self.current;
        let n = prompts.len();
        let w = m.config.width;
        let mut split = if observe_pair {
            let b = binding.ok_or("paired gradient observation requires binding candidates")?;
            if gold.iter().any(|y| !b.ids.contains(y)) {
                return Err(
                    "paired gradient observation requires gold inside the candidate pair".into(),
                );
            }
            answer_capture_work(
                prompts.iter().map(Vec::len).sum(),
                w,
                3,
                "binding observation capture work",
            )?;
            Some(PairObservation::new(w))
        } else {
            None
        };
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
                let gap = f64::from(z[b.ids[0]]) - f64::from(z[b.ids[1]]);
                let g = m.fixed_left_seed(row.clone(), d, self.spec, &self.left)?;
                if let Some(observer) = &mut split {
                    let mass_seed = candidate_mass_seed(z, b.ids)?;
                    let upstream = mass_seed.readout_upstream(m)?;
                    let mut mass =
                        m.fixed_left_upstream(row.clone(), upstream, self.spec, &self.left)?;
                    for g in &mut mass {
                        *g *= mass_seed.outside_share;
                    }
                    observer.add(
                        mass_seed.loss,
                        mass,
                        gap,
                        if y == b.ids[0] { 1. } else { -1. },
                        &g,
                    );
                }
                gaps.push(gap);
                gap_grad.push(g);
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
        Ok((
            CausalLmAnswerProjectionStep {
                answer,
                interaction,
                combined,
                mix,
                prompt_forwards: n,
                projection_pullbacks: n * seeds,
            },
            split,
        ))
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
        answer_capture_work(
            tokens,
            w,
            1 + usize::from(backward && binding.is_some()),
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

#[cfg(test)]
mod scoring_tests {
    use super::*;

    #[test]
    fn observation_capture_cap_counts_all_three_pullbacks() {
        let width = 768;
        let cap = output_margin::MAX_CAPTURE_TERMS;
        let max_tokens = cap / (width * 3);
        assert_eq!(
            answer_capture_work(max_tokens, width, 3, "observation").unwrap(),
            max_tokens * width * 3
        );
        assert!(answer_capture_work(max_tokens + 1, width, 2, "training").is_ok());
        assert!(answer_capture_work(max_tokens + 1, width, 3, "observation").is_err());
        assert!(answer_capture_work(usize::MAX, width, 3, "observation").is_err());
    }

    #[test]
    fn paired_scores_keep_small_positive_nll_and_first_top_token() {
        let score = CausalLmPairedAnswerScore::from_logits(&[0., -45., -46.], [0, 1]).unwrap();
        let expected = ((-45.0_f64).exp() + (-46.0_f64).exp()).ln_1p();
        assert_eq!(score.gold_nll, expected);
        assert!(score.gold_nll > 0. && score.gold_nll < f64::EPSILON);
        assert_eq!(score.gold_minus_alternative, 45.);
        let reversed = CausalLmPairedAnswerScore::from_logits(&[0., -45., -46.], [1, 0]).unwrap();
        assert_eq!(reversed.alternative_nll, score.gold_nll);
        assert_eq!(reversed.gold_minus_alternative, -45.);

        let tie = CausalLmPairedAnswerScore::from_logits(&[1., 1.], [1, 0]).unwrap();
        assert_eq!(tie.top_token, 0);
        assert_eq!(tie.gold_nll, 2.0_f64.ln());
        assert_eq!(tie.gold_nll, tie.alternative_nll);
        assert_eq!(tie.gold_minus_alternative, 0.);
        assert!(CausalLmPairedAnswerScore::from_logits(&[0., 1., f32::NAN], [0, 1]).is_err());
    }
}
