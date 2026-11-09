//! Same-batch sufficient-decrease search. Trials are forward-only and never advance a checkpoint.
use super::*;

pub const MAX_SCALAR_SEARCH_TRIALS: usize = 16;
const ARMIJO: f64 = 1e-4;

#[derive(Clone, Copy, Debug)]
pub struct CausalLmScalarSearch {
    pub learning_rate: f64,
    pub max_step: f64,
    pub max_abs: f64,
    pub max_trials: usize,
    /// Minimum absolute decrease in mean pair loss, in nats. A tie always refuses.
    pub min_decrease: f64,
}

impl CausalLmScalarSearch {
    pub fn validate(self, model: &NativeCausalLmScalarOutput) -> CausalLmResult<()> {
        model.validate_sgd(self.learning_rate, self.max_step, self.max_abs)?;
        if !(1..=MAX_SCALAR_SEARCH_TRIALS).contains(&self.max_trials)
            || !self.min_decrease.is_finite()
            || self.min_decrease < 0.
        {
            return Err("invalid scalar search trial budget or decrease floor".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmScalarPairLoss {
    pub loss: f64,
    pub prompt_evaluations: usize,
    /// Positive and negative prompt contrasts in the caller's pair order.
    pub pairs: Vec<[f64; 2]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmScalarSearchStatus {
    Accepted,
    ZeroGradient,
    NoDisplacement,
    TrialsExhausted,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmScalarTrial {
    pub coefficient: f64,
    pub required_decrease: f64,
    pub objective: CausalLmScalarPairLoss,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmScalarSearchReport {
    pub status: CausalLmScalarSearchStatus,
    pub step: u64,
    pub before: f64,
    pub after: f64,
    pub baseline: CausalLmScalarPairLoss,
    pub gradient_loss: f64,
    pub gradient: f64,
    pub gradient_prompt_evaluations: usize,
    pub derivatives: Vec<[f64; 4]>,
    pub trials: Vec<CausalLmScalarTrial>,
}

impl NativeCausalLmScalarOutput {
    /// Ordinary full-prefix forward on both sides of every pair, without a backward pass or update.
    pub fn pair_objective(
        &self,
        positive: &[Vec<usize>],
        negative: &[Vec<usize>],
        target: usize,
        alternative: usize,
    ) -> CausalLmResult<CausalLmScalarPairLoss> {
        self.admit_pairs(positive, negative, target, alternative)?;
        let score = |ids: &[usize]| -> CausalLmResult<f64> {
            Ok(self
                .current
                .next_token_contrast(ids, target, alternative)?
                .target_minus_alternative)
        };
        let mut total = sum::Sum::default();
        let mut pairs = Vec::with_capacity(positive.len());
        let (prompt_evaluations, observations) = observe_pairs(positive, negative, score)?;
        for [a, b] in observations {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite scalar forward contrast".into());
            }
            total.add(logistic(a - b).0);
            pairs.push([a, b]);
        }
        let loss = total.mean(pairs.len() as u64);
        if !loss.is_finite() {
            return Err("nonfinite scalar forward objective".into());
        }
        Ok(CausalLmScalarPairLoss {
            loss,
            pairs,
            prompt_evaluations,
        })
    }

    /// Starting from the ordinary clipped SGD proposal, halve its *feasible displacement*.
    /// Accept only the first measured strict decrease satisfying the floor and Armijo bound.
    /// A normal refusal returns the same state and cursor; a numerical or decoder error propagates.
    pub fn step_pairs_backtracking(
        &self,
        positive: &[Vec<usize>],
        negative: &[Vec<usize>],
        target: usize,
        alternative: usize,
        search: CausalLmScalarSearch,
    ) -> CausalLmResult<(Self, CausalLmScalarSearchReport)> {
        use CausalLmScalarSearchStatus::*;
        search.validate(self)?;
        let d = self.batch_gradient(positive, negative, target, alternative)?;
        let baseline = self.pair_objective(positive, negative, target, alternative)?;
        let first = self.proposal(
            d.gradient,
            search.learning_rate,
            search.max_step,
            search.max_abs,
        )?;
        let mut report = CausalLmScalarSearchReport {
            status: TrialsExhausted,
            step: self.steps,
            before: self.coefficient,
            after: self.coefficient,
            baseline,
            gradient_loss: d.loss,
            gradient: d.gradient,
            gradient_prompt_evaluations: d.prompt_evaluations,
            derivatives: d.pairs,
            trials: Vec::with_capacity(search.max_trials),
        };
        let mut candidate = first;
        for _ in 0..search.max_trials {
            let delta = candidate - self.coefficient;
            if !delta.is_finite() {
                return Err("nonfinite feasible scalar displacement".into());
            }
            if delta == 0. {
                report.status = if d.gradient == 0. {
                    ZeroGradient
                } else {
                    NoDisplacement
                };
                break;
            }
            let predicted = -d.gradient * delta;
            if !predicted.is_finite() || predicted < 0. {
                return Err("invalid scalar descent prediction".into());
            }
            let required_decrease = (ARMIJO * predicted).max(search.min_decrease);
            let mut trial = self.clone();
            trial.set_coefficient(candidate)?;
            let objective = trial.pair_objective(positive, negative, target, alternative)?;
            let drop = report.baseline.loss - objective.loss;
            report.trials.push(CausalLmScalarTrial {
                coefficient: candidate,
                required_decrease,
                objective,
            });
            if drop > 0. && drop >= required_decrease {
                trial.advance()?;
                report.step = trial.steps;
                report.after = trial.coefficient;
                report.status = Accepted;
                return Ok((trial, report));
            }
            // IEEE midpoint avoids overflow, lies in the feasible segment even at a bound.
            let smaller = self.coefficient.midpoint(candidate);
            if smaller == candidate || smaller == self.coefficient {
                report.status = NoDisplacement;
                break;
            }
            candidate = smaller;
        }
        Ok((self.clone(), report))
    }
}
