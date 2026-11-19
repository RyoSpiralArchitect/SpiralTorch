//! Optimize two-candidate choice itself, optionally retaining candidate mass.
use super::observation::candidate_mass_loss;
use super::*;
use crate::causal_lm::scalar_output::logistic;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalLmConditionalAnswerConfig {
    /// Initial normalized right-factor step length; trials halve it.
    pub length: f64,
    pub max_trials: usize,
    /// None: direct conditional descent. Some: require finite-batch candidate
    /// mass non-regression and aim this far into its first-order descent cone.
    pub mass_inward_cosine: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CausalLmConditionalAnswerPoint {
    pub conditional_ce: f64,
    /// Mean -log(P(gold)+P(alternative)), not minus log of the batch mean.
    pub candidate_mass_nll: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmConditionalAnswerTrial {
    pub point: CausalLmConditionalAnswerPoint,
    pub projection_id: String,
    pub distance: f64,
    pub required_conditional_drop: f64,
    pub accepted: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmConditionalAnswerStep {
    pub before: CausalLmConditionalAnswerPoint,
    pub after: CausalLmConditionalAnswerPoint,
    pub trials: Vec<CausalLmConditionalAnswerTrial>,
    pub committed: bool,
    pub projected: bool,
    pub conditional_gradient_l2: f64,
    pub mass_gradient_l2: Option<f64>,
    pub gradient_forwards: usize,
    pub projection_pullbacks: usize,
    pub trial_forwards: usize,
    /// Frozen block calls skipped by replaying only the trial suffix.
    pub reused_prefix_block_calls: usize,
}

impl CausalLmConditionalAnswerConfig {
    fn validate(self) -> CausalLmResult<()> {
        if !self.length.is_finite()
            || self.length <= 0.
            || !(1..=8).contains(&self.max_trials)
            || self
                .mass_inward_cosine
                .is_some_and(|x| !(0.0..1.0).contains(&x))
        {
            return Err("invalid conditional answer step configuration".into());
        }
        Ok(())
    }
}

fn norm(xs: &[f64]) -> f64 {
    xs.iter().fold(0.0_f64, |n, &x| n.hypot(x))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut result = sum::Sum::default();
    for (&x, &y) in a.iter().zip(b) {
        result.add(x * y);
    }
    result.value()
}

/// Return a unit gradient-space direction (the actual step subtracts it).
fn direction(choice: &[f64], mass: Option<(&[f64], f64)>) -> Option<(Vec<f64>, bool)> {
    let n = norm(choice);
    if n == 0. || !n.is_finite() {
        return None;
    }
    let choice: Vec<_> = choice.iter().map(|v| v / n).collect();
    let Some((mass, inward)) = mass else {
        return Some((choice, false));
    };
    let n = norm(mass);
    if n == 0. {
        return Some((choice, false));
    }
    let mass: Vec<_> = mass.iter().map(|v| v / n).collect();
    let cos = dot(&choice, &mass);
    if cos >= inward {
        return Some((choice, false));
    }
    let tangent: Vec<_> = choice.iter().zip(&mass).map(|(c, m)| c - cos * m).collect();
    let n = norm(&tangent);
    if n <= 1e-12 {
        return None;
    }
    let scale = (1. - inward * inward).sqrt() / n;
    let projected: Vec<_> = tangent
        .iter()
        .zip(mass)
        .map(|(t, m)| scale * t + inward * m)
        .collect();
    (dot(&choice, &projected) > 0.).then_some((projected, true))
}

#[derive(Default)]
struct PointSum {
    ce: sum::Sum,
    mass: sum::Sum,
}

impl PointSum {
    fn add(&mut self, ce: f64, mass: f64) {
        self.ce.add(ce);
        self.mass.add(mass);
    }
    fn finish(self, n: usize) -> CausalLmConditionalAnswerPoint {
        CausalLmConditionalAnswerPoint {
            conditional_ce: self.ce.mean(n as u64),
            candidate_mass_nll: self.mass.mean(n as u64),
        }
    }
}

impl NativeCausalLmRankOneProjection {
    /// Fit conditional CE on explicit [gold, alternative] pairs. Commit only a
    /// measured decrease in conditional CE and, when guarded, non-increase in
    /// mass loss. Guarantees cover the supplied batch, not other prompts.
    /// An error or exhausted search never changes the serving identity.
    pub fn step_conditional_answers(
        &mut self,
        prompts: &[Vec<usize>],
        pairs: &[[usize; 2]],
        config: CausalLmConditionalAnswerConfig,
    ) -> CausalLmResult<CausalLmConditionalAnswerStep> {
        self.conditional_step(prompts, pairs, config, None)
            .map(|(r, _)| r)
    }

    /// As for ordinary conditional learning, with an additional finite
    /// sign check on every anchored prompt and an optional mean-mass check
    /// on the finite roster before a proposal may commit. The roster limits
    /// acceptance, not the construction of the mini-batch search direction.
    pub fn step_conditional_answers_retaining(
        &mut self,
        prompts: &[Vec<usize>],
        pairs: &[[usize; 2]],
        config: CausalLmConditionalAnswerConfig,
        retention: &mut CausalLmConditionalMarginRetention,
    ) -> CausalLmResult<CausalLmRetainedConditionalAnswerStep> {
        retention.verify_owner(self)?;
        let (step, checks) = self.conditional_step(prompts, pairs, config, Some(retention))?;
        if step.committed {
            retention.advance(self.projection_id());
        }
        Ok(CausalLmRetainedConditionalAnswerStep {
            step,
            retention: checks,
        })
    }

    fn conditional_step(
        &mut self,
        prompts: &[Vec<usize>],
        pairs: &[[usize; 2]],
        config: CausalLmConditionalAnswerConfig,
        retention: Option<&CausalLmConditionalMarginRetention>,
    ) -> CausalLmResult<(
        CausalLmConditionalAnswerStep,
        Vec<Option<CausalLmConditionalRetentionCheck>>,
    )> {
        config.validate()?;
        let m = &self.current;
        if prompts.len() != pairs.len() || pairs.len() > MAX_OUTPUT_CONTRAST_ROWS {
            return Err("invalid conditional answer roster".into());
        }
        for &[y, a] in pairs {
            m.check_contrast_candidates(y, a)?;
        }
        let gold: Vec<_> = pairs.iter().map(|p| p[0]).collect();
        self.admit_answers(prompts, &gold, None, true)?;
        let guarded = config.mass_inward_cosine.is_some();
        let n = prompts.len();
        let w = m.config.width;
        product(
            &[
                prompts.iter().map(Vec::len).sum(),
                w,
                1 + usize::from(guarded),
            ],
            output_margin::MAX_CAPTURE_TERMS,
            "conditional answer capture work",
        )?;

        let mut point = PointSum::default();
        let mut choice: Vec<_> = (0..w).map(|_| sum::Sum::default()).collect();
        let mut mass = guarded.then(|| (0..w).map(|_| sum::Sum::default()).collect::<Vec<_>>());
        let first = self.spec.layer;
        let mut prefixes = Vec::with_capacity(n);
        for (prompt, &[y, a]) in prompts.iter().zip(pairs) {
            let shape = [1, prompt.len()];
            let prefix = m.forward_blocks(m.embed(prompt, shape)?, shape, 0..first)?;
            let row =
                m.projection_last_row_from_hidden(prefix.clone(), prompt.len(), first, self.spec)?;
            prefixes.push(prefix);
            let z = row.logits.data();
            let (ce, slope) = logistic(f64::from(z[y]) - f64::from(z[a]));
            let mass_loss = if let Some(out) = &mut mass {
                let (loss, gradient) = self.candidate_mass_gradient(row.clone(), [y, a])?;
                for (s, g) in out.iter_mut().zip(gradient) {
                    s.add(g);
                }
                loss
            } else {
                candidate_mass_loss(z, [y, a])?
            };
            point.add(ce, mass_loss);
            let mut v = vec![0.; z.len()];
            v[y] = 1.;
            v[a] = -1.;
            let seed = Tensor::from_vec(1, z.len(), v)?;
            for (s, g) in choice
                .iter_mut()
                .zip(m.fixed_left_seed(row, &seed, self.spec, &self.left)?)
            {
                s.add(slope * g);
            }
        }
        let mean = |s: Vec<sum::Sum>| s.into_iter().map(|v| v.mean(n as u64)).collect::<Vec<_>>();
        let choice = mean(choice);
        let mass = mass.map(mean);
        let before = point.finish(n);
        let mut result = CausalLmConditionalAnswerStep {
            before,
            after: before,
            trials: Vec::new(),
            committed: false,
            projected: false,
            conditional_gradient_l2: norm(&choice),
            mass_gradient_l2: mass.as_ref().map(|v| norm(v)),
            gradient_forwards: n,
            projection_pullbacks: n * (1 + usize::from(guarded)),
            trial_forwards: 0,
            reused_prefix_block_calls: 0,
        };
        if !result.conditional_gradient_l2.is_finite()
            || result.mass_gradient_l2.is_some_and(|x| !x.is_finite())
            || !before.conditional_ce.is_finite()
            || !before.candidate_mass_nll.is_finite()
        {
            return Err("nonfinite conditional answer objective".into());
        }
        let constraint = mass.as_deref().zip(config.mass_inward_cosine);
        let mut checks = Vec::new();
        let Some((direction, projected)) = direction(&choice, constraint) else {
            return Ok((result, checks));
        };
        result.projected = projected;
        for i in 0..config.max_trials {
            let length = config.length * 0.5_f64.powi(i as i32);
            let right: Vec<_> = self
                .right()
                .iter()
                .zip(&direction)
                .map(|(&x, &d)| x - length * d)
                .collect();
            let displacement: Vec<_> = self
                .right()
                .iter()
                .zip(&right)
                .map(|(&x, &y)| x - y)
                .collect();
            let distance = norm(&displacement);
            if distance == 0. {
                break;
            }
            let required = 1e-4 * dot(&choice, &displacement).max(0.);
            if !distance.is_finite() || !required.is_finite() {
                return Err("nonfinite conditional displacement".into());
            }
            let mut candidate = self.clone();
            candidate.install(right)?;
            let mut point = PointSum::default();
            // Only this projection changes, so the prefix strictly before it
            // is identical for every rejected or committed proposal.
            let m = &candidate.current;
            for ((prompt, &[y, a]), prefix) in prompts.iter().zip(pairs).zip(&prefixes) {
                let shape = [1, prompt.len()];
                let h = m.forward_blocks(prefix.clone(), shape, first..m.config.layers)?;
                let last = h.gather_rows(&[prompt.len() - 1])?;
                let normalized = m.final_norm.forward(&last, m.config.norm_epsilon)?;
                let logits = m.project_readout(&normalized)?;
                let z = logits.data();
                point.add(
                    logistic(f64::from(z[y]) - f64::from(z[a])).0,
                    candidate_mass_loss(z, [y, a])?,
                );
            }
            let after = point.finish(n);
            let drop = before.conditional_ce - after.conditional_ce;
            let ordinary_accepted = drop > 0.
                && drop >= required
                && (!guarded || after.candidate_mass_nll <= before.candidate_mass_nll);
            let mut accepted = ordinary_accepted;
            if let Some(deck) = retention {
                let check = ordinary_accepted
                    .then(|| deck.check(&candidate))
                    .transpose()?;
                accepted &= check
                    .as_ref()
                    .is_none_or(CausalLmConditionalRetentionCheck::permits_commit);
                checks.push(check);
            }
            result.trial_forwards += n;
            result.reused_prefix_block_calls += n * first;
            result.trials.push(CausalLmConditionalAnswerTrial {
                point: after,
                projection_id: candidate.projection_id(),
                distance,
                required_conditional_drop: required,
                accepted,
            });
            if accepted {
                *self = candidate;
                result.after = after;
                result.committed = true;
                break;
            }
        }
        Ok((result, checks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_direction_gains_on_both_objectives_or_holds() {
        let pair = [-0.6, 0.8];
        let mass = [1., 0.];
        let (g, projected) = direction(&pair, Some((&mass, 0.05))).unwrap();
        assert!(projected && (g[0] - 0.05).abs() < 1e-15);
        assert!(dot(&pair, &g) > 0. && (norm(&g) - 1.).abs() < 1e-15);
        assert!(direction(&[-1., 0.], Some((&mass, 0.05))).is_none());
        assert_eq!(
            direction(&mass, Some((&mass, 0.05))),
            Some((mass.to_vec(), false))
        );
    }
}
