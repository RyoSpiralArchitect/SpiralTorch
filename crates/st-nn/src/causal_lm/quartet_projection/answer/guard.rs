//! Finite, forward-checked correction around a feasible next-token CE step.
use super::*;

#[derive(Clone, Copy, Debug)]
pub struct CausalLmAnswerProtection<'a> {
    pub prompts: &'a [Vec<usize>],
    pub gold: &'a [usize],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmAnswerCorrection {
    AnchorOnly,
    Mixed,
    Interaction,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalLmAnswerGuard {
    pub length: f64,
    /// At most this many ordinary forward trials in each of the two phases.
    pub max_trials: usize,
    /// Required fraction of the *measured* CE improvement at this step's anchor.
    pub retain_answer: f64,
    pub min_answer_drop: f64,
    pub min_interaction_gain: f64,
    pub correction: CausalLmAnswerCorrection,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CausalLmAnswerPoint {
    pub answer: f64,
    pub interaction: f64,
    /// Optional, separate training roster. No gradients or quartet labels are used here.
    pub protected_answer: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmAnswerGuardStatus {
    Held,
    Anchor,
    Corrected,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmAnswerTrial {
    pub correction: bool,
    pub fraction: f64,
    pub owner: String,
    pub distance: f64,
    pub point: CausalLmAnswerPoint,
    pub required_answer_drop: f64,
    /// An eligible CE anchor may subsequently be replaced by an eligible correction.
    pub eligible: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmAnswerGuardStep {
    pub status: CausalLmAnswerGuardStatus,
    pub derivatives: CausalLmAnswerProjectionStep,
    pub before: CausalLmAnswerPoint,
    pub after: CausalLmAnswerPoint,
    pub trials: Vec<CausalLmAnswerTrial>,
    /// Actual ordinary forwards, including baseline and both rosters; excludes the gradient forward.
    pub objective_forwards: usize,
}

impl CausalLmAnswerGuard {
    pub fn validate(self) -> CausalLmResult<()> {
        if !self.length.is_finite()
            || self.length <= 0.
            || !(1..=8).contains(&self.max_trials)
            || !(self.retain_answer > 0. && self.retain_answer <= 1.)
            || [self.min_answer_drop, self.min_interaction_gain]
                .iter()
                .any(|x| !x.is_finite() || *x < 0.)
        {
            return Err("invalid answer guard radius, retention, floor or trial budget".into());
        }
        Ok(())
    }
}

struct Roster<'a> {
    active: CausalLmAnswerProtection<'a>,
    binding: CausalLmAnswerBinding<'a>,
    protected: Option<CausalLmAnswerProtection<'a>>,
}

impl Roster<'_> {
    fn point(&self, a: &NativeCausalLmRankOneProjection) -> CausalLmResult<CausalLmAnswerPoint> {
        let observe = |batch: CausalLmAnswerProtection<'_>,
                       mut gaps: Option<&mut Vec<f64>>|
         -> CausalLmResult<f64> {
            let mut total = sum::Sum::default();
            for (q, &y) in batch.prompts.iter().zip(batch.gold) {
                let logits = a.current.logits(q, [1, q.len()], true)?;
                let z = logits.data();
                total.add(TargetClassScore::from_logits(z, y)?.nll);
                if let Some(g) = gaps.as_mut() {
                    let [t, r] = self.binding.ids;
                    g.push(f64::from(z[t]) - f64::from(z[r]));
                }
            }
            Ok(total.mean(batch.gold.len() as u64))
        };
        let mut gaps = Vec::with_capacity(self.active.gold.len());
        let answer = observe(self.active, Some(&mut gaps))?;
        let interaction = CausalLmQuartetObjective::Interaction
            .mean_seed(&gaps, self.binding.quartets)
            .0;
        let protected_answer = self.protected.map(|b| observe(b, None)).transpose()?;
        if !answer.is_finite()
            || !interaction.is_finite()
            || protected_answer.is_some_and(|x| !x.is_finite())
        {
            return Err("nonfinite answer-guard forward objective".into());
        }
        Ok(CausalLmAnswerPoint {
            answer,
            interaction,
            protected_answer,
        })
    }
}

fn endpoint(origin: &[f64], gradient: &CausalLmLossGradient, length: f64) -> Option<Vec<f64>> {
    (gradient.norm > 0.).then(|| {
        origin
            .iter()
            .zip(&gradient.gradient)
            .map(|(&x, &g)| x - length * (g / gradient.norm))
            .collect()
    })
}

struct Search<'a, 'b> {
    origin: &'a NativeCausalLmRankOneProjection,
    roster: &'a Roster<'b>,
    derivative: &'a CausalLmLossGradient,
    before: CausalLmAnswerPoint,
    cfg: CausalLmAnswerGuard,
}

fn admissible(
    before: CausalLmAnswerPoint,
    point: CausalLmAnswerPoint,
    required: f64,
    interaction_limit: Option<(f64, f64)>,
) -> bool {
    let gain = before.answer - point.answer;
    let safe = match (before.protected_answer, point.protected_answer) {
        (Some(start), Some(end)) => end <= start,
        (None, None) => true,
        _ => false,
    };
    gain > 0.
        && gain >= required
        && safe
        && interaction_limit.is_none_or(|(start, floor)| start - point.interaction > floor)
}

impl Search<'_, '_> {
    fn trial(
        &self,
        right: Vec<f64>,
        anchor: CausalLmAnswerPoint,
        fraction: f64,
        correction: bool,
    ) -> CausalLmResult<(NativeCausalLmRankOneProjection, CausalLmAnswerTrial)> {
        let Self {
            origin,
            roster,
            derivative,
            before,
            cfg,
        } = *self;
        let mut pred = sum::Sum::default();
        let mut distance = 0.0_f64;
        for ((&x, &y), &g) in origin.right().iter().zip(&right).zip(&derivative.gradient) {
            pred.add(g * (x - y));
            distance = distance.hypot(x - y);
        }
        if !pred.value().is_finite() || !distance.is_finite() {
            return Err("nonfinite guard displacement".into());
        }
        let mut candidate = origin.clone();
        candidate.install(right)?;
        let point = roster.point(&candidate)?;
        let required_answer_drop =
            cfg.min_answer_drop
                .max(1e-4 * pred.value().max(0.))
                .max(if correction {
                    cfg.retain_answer * (before.answer - anchor.answer)
                } else {
                    0.
                });
        let eligible = admissible(
            before,
            point,
            required_answer_drop,
            correction.then_some((
                before.interaction.min(anchor.interaction),
                cfg.min_interaction_gain,
            )),
        );
        let record = CausalLmAnswerTrial {
            owner: candidate.projection_id(),
            correction,
            fraction,
            distance,
            point,
            required_answer_drop,
            eligible,
        };
        Ok((candidate, record))
    }
}

impl NativeCausalLmRankOneProjection {
    /// First seek a strict ordinary CE decrease, halving the normalized CE step.
    /// Then shrink a correction *toward that safe endpoint*, not toward an unsafe direction
    /// at the origin. If no anchor exists, a correction must earn both decreases from hold.
    /// The optional second roster must never increase in measured CE. Guarantees are
    /// local to the supplied examples, not to held-out data or a separate CE-only trajectory.
    pub fn step_answers_guarded(
        &mut self,
        prompts: &[Vec<usize>],
        gold: &[usize],
        binding: CausalLmAnswerBinding<'_>,
        protected: Option<CausalLmAnswerProtection<'_>>,
        guard: CausalLmAnswerGuard,
    ) -> CausalLmResult<CausalLmAnswerGuardStep> {
        use CausalLmAnswerGuardStatus::*;
        guard.validate()?;
        if let Some(b) = protected {
            self.admit_answers(b.prompts, b.gold, None, false)?;
        }
        let d = self.answer_gradients(prompts, gold, Some(binding))?;
        let roster = Roster {
            active: CausalLmAnswerProtection { prompts, gold },
            binding,
            protected,
        };
        let before = roster.point(self)?;
        let search = Search {
            origin: self,
            roster: &roster,
            derivative: &d.answer,
            before,
            cfg: guard,
        };
        let mut trials = Vec::with_capacity(2 * guard.max_trials);
        let mut chosen = None;
        let mut after = before;
        let mut status = Held;
        for i in 0..guard.max_trials {
            let f = 0.5_f64.powi(i as i32);
            let Some(right) = endpoint(self.right(), &d.answer, guard.length * f) else {
                break;
            };
            if self.right() == right {
                break;
            }
            let (candidate, row) = search.trial(right, before, f, false)?;
            let eligible = row.eligible;
            let point = row.point;
            trials.push(row);
            if eligible {
                chosen = Some(candidate);
                after = point;
                status = Anchor;
                break;
            }
        }
        let dir = match guard.correction {
            CausalLmAnswerCorrection::AnchorOnly => None,
            CausalLmAnswerCorrection::Mixed => Some(&d.combined),
            CausalLmAnswerCorrection::Interaction => d.interaction.as_ref(),
        };
        if let Some(target) = dir.and_then(|g| endpoint(self.right(), g, guard.length)) {
            let anchor = chosen.as_ref().unwrap_or(self).right();
            let mut correction = None;
            for i in 0..guard.max_trials {
                let f = 0.5_f64.powi(i as i32);
                let right = if i == 0 {
                    target.clone()
                } else {
                    anchor
                        .iter()
                        .zip(&target)
                        .map(|(&a, &t)| (1. - f) * a + f * t)
                        .collect()
                };
                if right == anchor {
                    break;
                }
                let (candidate, row) = search.trial(right, after, f, true)?;
                let eligible = row.eligible;
                let point = row.point;
                trials.push(row);
                if eligible {
                    correction = Some((candidate, point));
                    break;
                }
            }
            if let Some((v, p)) = correction {
                chosen = Some(v);
                after = p;
                status = Corrected;
            }
        }
        // Do not publish even the anchor until every required evaluation has succeeded.
        if let Some(v) = chosen {
            *self = v;
        }
        let objective_forwards =
            (gold.len() + protected.map_or(0, |b| b.gold.len())) * (1 + trials.len());
        Ok(CausalLmAnswerGuardStep {
            status,
            derivatives: d,
            before,
            after,
            trials,
            objective_forwards,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn correction_cannot_spend_the_reserved_answer_drop_or_other_batch() {
        let pt = |answer, interaction, protected_answer| CausalLmAnswerPoint {
            answer,
            interaction,
            protected_answer: Some(protected_answer),
        };
        let base = pt(2., 1., 0.5);
        let limit = Some((0.75, 0.125));
        assert!(admissible(base, pt(1.625, 0.5, 0.5), 0.375, limit));
        assert!(!admissible(base, pt(1.626, 0.4, 0.4), 0.375, limit));
        assert!(!admissible(base, pt(1.5, 0.625, 0.4), 0.375, limit));
        assert!(!admissible(base, pt(1.5, 0.4, 0.501), 0.375, limit));
        assert!(!admissible(base, base, 0., None));
    }
}
