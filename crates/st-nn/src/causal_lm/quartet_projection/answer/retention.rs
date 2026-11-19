//! Finite protection of individually correct two-candidate margins.
use super::{observation::candidate_mass_loss, *};

mod checkpoint;
mod session;
pub use checkpoint::CausalLmConditionalRetentionRestore;
pub use session::{
    CausalLmConditionalAnswerBatch, CausalLmConditionalAnswerComparison,
    CausalLmConditionalAnswerSessionRestore, CausalLmConditionalAnswerSessionStep,
    CausalLmConditionalScoredSnapshot, ConditionalCheckpointPublicationUncertain,
    NativeCausalLmConditionalAnswerSession, NativeCausalLmConditionalServingSnapshot,
};

struct Anchor {
    prefix: Tensor,
    steps: usize,
    pair: [usize; 2],
    protects_sign: bool,
}

/// Prepared protection roster for one projection trajectory. Initially
/// correct rows are sign-protected; optional mass protection uses all rows.
#[derive(Clone)]
pub struct CausalLmConditionalMarginRetention {
    anchors: Arc<[Anchor]>,
    layer: usize,
    owner: String,
    baseline: Vec<f64>,
    roster_steps: Arc<[usize]>,
    protected_count: usize,
    mass: Option<MassFloor>,
    initial_projection_id: String,
    initial_right: Vec<f64>,
    roster_id: String,
    allowed_drop: Option<f64>,
}

#[derive(Clone)]
struct MassFloor {
    baseline: f64,
    minimum: f64,
    maximum_outside: f64,
    minimum_log_pair: Option<f64>,
}

#[derive(Default)]
struct PairMassMean {
    pair: sum::Sum,
    outside: sum::Sum,
    log_pair_sum: Option<f64>,
}

impl PairMassMean {
    fn add_loss(&mut self, loss: f64) {
        self.pair.add((-loss).exp());
        self.outside.add(-(-loss).exp_m1());
        let log_p = -loss;
        self.log_pair_sum = Some(self.log_pair_sum.map_or(log_p, |sum| {
            log_p.max(sum) + (-(log_p - sum).abs()).exp().ln_1p()
        }));
    }

    fn log_mean_pair(&self, rows: usize) -> Option<f64> {
        self.log_pair_sum.map(|sum| sum - (rows as f64).ln())
    }
}

impl MassFloor {
    fn log_pair_limit(&self) -> Option<f64> {
        self.minimum_log_pair
            .filter(|&floor| floor < f64::MIN_POSITIVE.ln())
    }

    fn from_baseline(mass: PairMassMean, rows: usize, drop: f64) -> Self {
        let minimum_log_pair = mass.log_mean_pair(rows).and_then(|log_mean| {
            if drop == 0. {
                return Some(log_mean);
            }
            let log_ratio = drop.ln() - log_mean;
            (log_ratio < 0.).then(|| log_mean + (-log_ratio.exp_m1()).ln())
        });
        let baseline = mass.pair.mean(rows as u64);
        Self {
            baseline,
            minimum: (baseline - drop).max(0.),
            maximum_outside: (mass.outside.mean(rows as u64) + drop).min(1.),
            minimum_log_pair,
        }
    }

    fn check(&self, mass: PairMassMean, rows: usize) -> CausalLmConditionalRetentionMassCheck {
        let log_pair = self.log_pair_limit().zip(mass.log_mean_pair(rows)).map(
            |(minimum_log_mean_pair_mass, log_mean_pair_mass)| {
                CausalLmConditionalRetentionLogMassCheck {
                    log_mean_pair_mass,
                    minimum_log_mean_pair_mass,
                }
            },
        );
        let mean_pair_mass = mass.pair.mean(rows as u64);
        let mean_outside_mass = mass.outside.mean(rows as u64);
        let rounded_pass = mean_pair_mass >= self.minimum;
        let outside_pass = mean_outside_mass <= self.maximum_outside;
        // Compare the smaller side of the floor, which can be far below one ulp
        // of the reported pair mass. Record when the rounded check is ambiguous.
        let outside = (self.minimum >= 0.5 && (self.minimum == 1. || rounded_pass != outside_pass))
            .then_some(CausalLmConditionalRetentionOutsideCheck {
                mean_outside_mass,
                maximum_mean_outside_mass: self.maximum_outside,
            });
        CausalLmConditionalRetentionMassCheck {
            rows,
            mean_pair_mass,
            minimum_mean_pair_mass: self.minimum,
            outside,
            log_pair,
        }
    }
}

/// Equivalent ceiling on probability outside the pair, computed without subtracting from 1.
#[derive(Clone, Debug, Serialize)]
pub struct CausalLmConditionalRetentionOutsideCheck {
    pub mean_outside_mass: f64,
    pub maximum_mean_outside_mass: f64,
}

/// Natural logarithms of the arithmetic mean pair mass and its positive floor.
#[derive(Clone, Debug, Serialize)]
pub struct CausalLmConditionalRetentionLogMassCheck {
    pub log_mean_pair_mass: f64,
    pub minimum_log_mean_pair_mass: f64,
}

/// Finite arithmetic mean of P(gold)+P(alternative) on the whole roster.
#[derive(Clone, Debug, Serialize)]
pub struct CausalLmConditionalRetentionMassCheck {
    pub rows: usize,
    pub mean_pair_mass: f64,
    pub minimum_mean_pair_mass: f64,
    /// When present, this is the authoritative comparison rather than the
    /// rounded pair-mass fields (notably when the floor rounds to 1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outside: Option<CausalLmConditionalRetentionOutsideCheck>,
    /// Authoritative when the positive floor is too small for normal f64 probability arithmetic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_pair: Option<CausalLmConditionalRetentionLogMassCheck>,
}

impl CausalLmConditionalRetentionMassCheck {
    pub fn permits_commit(&self) -> bool {
        if let Some(log_pair) = &self.log_pair {
            return log_pair.log_mean_pair_mass >= log_pair.minimum_log_mean_pair_mass;
        }
        match &self.outside {
            Some(outside) => outside.mean_outside_mass <= outside.maximum_mean_outside_mass,
            None => self.mean_pair_mass >= self.minimum_mean_pair_mass,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmConditionalRetentionCheck {
    pub checked: usize,
    pub minimum_margin: f64,
    pub nonpositive: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mass: Option<CausalLmConditionalRetentionMassCheck>,
}

impl CausalLmConditionalRetentionCheck {
    pub(super) fn permits_commit(&self) -> bool {
        self.nonpositive == 0
            && self
                .mass
                .as_ref()
                .is_none_or(CausalLmConditionalRetentionMassCheck::permits_commit)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmRetainedConditionalAnswerStep {
    pub step: CausalLmConditionalAnswerStep,
    /// One entry per ordinary trial; None means it failed the ordinary
    /// acceptance rule and the anchors did not need a forward pass.
    pub retention: Vec<Option<CausalLmConditionalRetentionCheck>>,
}

fn evaluate(
    m: &NativeCausalLm,
    prefix: &Tensor,
    layer: usize,
    steps: usize,
    [gold, rival]: [usize; 2],
    measure_mass: bool,
) -> CausalLmResult<(f64, Option<f64>)> {
    let hidden = m.forward_blocks(prefix.clone(), [1, steps], layer..m.config.layers)?;
    let last = hidden.gather_rows(&[steps - 1])?;
    let normalized = m.final_norm.forward(&last, m.config.norm_epsilon)?;
    let logits = m.project_readout(&normalized)?;
    let z = logits.data();
    let value = f64::from(z[gold]) - f64::from(z[rival]);
    if !value.is_finite() {
        return Err("nonfinite protected conditional margin".into());
    }
    let loss = measure_mass
        .then(|| candidate_mass_loss(z, [gold, rival]))
        .transpose()?;
    Ok((value, loss))
}

impl CausalLmConditionalMarginRetention {
    pub fn protected_count(&self) -> usize {
        self.protected_count
    }

    /// All preparation margins, in the original input order.
    pub fn baseline_margins(&self) -> &[f64] {
        &self.baseline
    }

    /// Baseline arithmetic mean of the two candidates' summed probability.
    pub fn baseline_mean_pair_mass(&self) -> Option<f64> {
        self.mass.as_ref().map(|mass| mass.baseline)
    }

    /// Minimum permitted mean on every committed step, fixed at preparation.
    pub fn minimum_mean_pair_mass(&self) -> Option<f64> {
        self.mass.as_ref().map(|mass| mass.minimum)
    }

    /// Authoritative positive floor when ordinary f64 probability is too small.
    pub fn minimum_log_mean_pair_mass(&self) -> Option<f64> {
        self.mass.as_ref().and_then(MassFloor::log_pair_limit)
    }

    pub(super) fn verify_owner(
        &self,
        projection: &NativeCausalLmRankOneProjection,
    ) -> CausalLmResult<()> {
        if self.owner != projection.projection_id() {
            return Err(
                "conditional margin retention belongs to a different projection state".into(),
            );
        }
        // Projection identity binds the weights, not the receiving model's work budget.
        for &steps in self.roster_steps.iter() {
            projection
                .current
                .projection_work_dimensions(steps, projection.spec, false)?;
        }
        Ok(())
    }

    pub(super) fn check(
        &self,
        candidate: &NativeCausalLmRankOneProjection,
    ) -> CausalLmResult<CausalLmConditionalRetentionCheck> {
        let mut minimum = f64::INFINITY;
        let mut nonpositive = 0;
        let mut mean_mass = PairMassMean::default();
        for a in self.anchors.iter() {
            let (v, loss) = evaluate(
                &candidate.current,
                &a.prefix,
                self.layer,
                a.steps,
                a.pair,
                self.mass.is_some(),
            )?;
            if a.protects_sign {
                minimum = minimum.min(v);
                nonpositive += usize::from(v <= 0.);
            }
            if let Some(loss) = loss {
                mean_mass.add_loss(loss);
            }
        }
        Ok(CausalLmConditionalRetentionCheck {
            checked: self.protected_count,
            minimum_margin: minimum,
            nonpositive,
            mass: self
                .mass
                .as_ref()
                .map(|floor| floor.check(mean_mass, self.anchors.len())),
        })
    }

    pub(super) fn advance(&mut self, owner: String) {
        self.owner = owner;
    }
}

impl NativeCausalLmRankOneProjection {
    /// Admit the retention roster's prefix storage before copying token IDs.
    pub fn conditional_retention_input_count(
        &self,
        steps: &[usize],
        allowed_drop: Option<f64>,
    ) -> CausalLmResult<usize> {
        Self::validate_mass_allowance(allowed_drop)?;
        let tokens = self.conditional_answer_input_count(steps, false)?;
        self.admit_conditional_retention_tokens(tokens)?;
        Ok(tokens)
    }

    fn admit_conditional_retention_tokens(&self, tokens: usize) -> CausalLmResult<()> {
        product(
            &[tokens, self.current.config.width],
            1 << 24,
            "retained margin-prefix elements",
        )?;
        Ok(())
    }

    fn validate_mass_allowance(allowed_drop: Option<f64>) -> CausalLmResult<()> {
        if allowed_drop.is_some_and(|drop| !(0.0..=1.0).contains(&drop)) {
            return Err("invalid conditional retention mass allowance".into());
        }
        Ok(())
    }

    /// Snapshot the signs of explicitly supplied training margins. A retained
    /// margin must stay strictly positive in every committed guarded step.
    /// This does not preserve its original size, the full-vocabulary winner,
    /// or any judgment outside the provided roster.
    pub fn prepare_conditional_margin_retention(
        &self,
        prompts: &[Vec<usize>],
        pairs: &[[usize; 2]],
    ) -> CausalLmResult<CausalLmConditionalMarginRetention> {
        self.prepare_conditional_retention(prompts, pairs, None)
    }

    /// Preserve the same finite sign guarantees and, independently, the
    /// arithmetic mean of the summed candidate probabilities over every row
    /// (including initially incorrect rows). The permitted absolute mean
    /// drop is relative to preparation, not to the previous step.
    pub fn prepare_conditional_margin_and_mass_retention(
        &self,
        prompts: &[Vec<usize>],
        pairs: &[[usize; 2]],
        max_mean_mass_drop: f64,
    ) -> CausalLmResult<CausalLmConditionalMarginRetention> {
        self.prepare_conditional_retention(prompts, pairs, Some(max_mean_mass_drop))
    }

    fn prepare_conditional_retention(
        &self,
        prompts: &[Vec<usize>],
        pairs: &[[usize; 2]],
        allowed_drop: Option<f64>,
    ) -> CausalLmResult<CausalLmConditionalMarginRetention> {
        Self::validate_mass_allowance(allowed_drop)?;
        if prompts.len() != pairs.len() {
            return Err("invalid conditional retention roster".into());
        }
        self.admit_conditional_answer_pairs(pairs)?;
        let gold = pairs.iter().map(|p| p[0]).collect::<Vec<_>>();
        self.admit_answers(prompts, &gold, None, false)?;
        let layer = self.spec.layer;
        let m = &self.current;
        self.admit_conditional_retention_tokens(prompts.iter().map(Vec::len).sum())?;
        let mut baseline = Vec::with_capacity(prompts.len());
        let mut anchors = Vec::new();
        let mut protected_count = 0;
        let mut mean_mass = PairMassMean::default();
        for (p, &pair) in prompts.iter().zip(pairs) {
            let steps = p.len();
            let shape = [1, steps];
            let prefix = m.forward_blocks(m.embed(p, shape)?, shape, 0..layer)?;
            let (v, loss) = evaluate(m, &prefix, layer, steps, pair, allowed_drop.is_some())?;
            baseline.push(v);
            let protects_sign = v > 0.;
            protected_count += usize::from(protects_sign);
            if let Some(loss) = loss {
                mean_mass.add_loss(loss);
            }
            if protects_sign || allowed_drop.is_some() {
                anchors.push(Anchor {
                    prefix,
                    steps,
                    pair,
                    protects_sign,
                });
            }
        }
        if protected_count == 0 {
            return Err("conditional retention found no initially correct margins".into());
        }
        let owner = self.projection_id();
        Ok(CausalLmConditionalMarginRetention {
            anchors: anchors.into(),
            baseline,
            roster_steps: prompts.iter().map(Vec::len).collect::<Vec<_>>().into(),
            layer,
            initial_projection_id: owner.clone(),
            owner,
            protected_count,
            initial_right: self.right().to_vec(),
            roster_id: checkpoint::roster_id(prompts, pairs),
            allowed_drop,
            mass: allowed_drop.map(|drop| MassFloor::from_baseline(mean_mass, prompts.len(), drop)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mass_for(rows: &[[f32; 3]]) -> PairMassMean {
        let mut mean = PairMassMean::default();
        for row in rows {
            mean.add_loss(candidate_mass_loss(row, [0, 1]).unwrap());
        }
        mean
    }

    #[test]
    fn saturated_pair_floor_detects_outside_probability_growth() {
        let before = [[45., 44., 0.], [45., 44., 0.]];
        let after = [[45., 44., 0.], [45., 44., 1.]];
        let floor = MassFloor::from_baseline(mass_for(&before), 2, 0.);
        assert_eq!(floor.minimum, 1.);
        assert!(floor.check(mass_for(&before), 2).permits_commit());

        let check = floor.check(mass_for(&after), 2);
        assert_eq!(check.mean_pair_mass, check.minimum_mean_pair_mass);
        let outside = check.outside.as_ref().unwrap();
        assert!(outside.maximum_mean_outside_mass > 0.);
        assert!(outside.mean_outside_mass > outside.maximum_mean_outside_mass);
        assert!(!check.permits_commit());

        let within_allowance = MassFloor::from_baseline(
            mass_for(&before),
            2,
            2. * (outside.mean_outside_mass - outside.maximum_mean_outside_mass),
        );
        assert_eq!(within_allowance.minimum, 1.);
        assert!(within_allowance.check(mass_for(&after), 2).permits_commit());
    }

    #[test]
    fn tiny_pair_floor_still_uses_pair_mass() {
        let before = [[0., -1., 100.]];
        let after = [[0., -1., 101.]];
        let floor = MassFloor::from_baseline(mass_for(&before), 1, 0.);
        assert!(floor.minimum > 0.);
        assert!(floor.check(mass_for(&before), 1).permits_commit());
        let check = floor.check(mass_for(&after), 1);
        assert!(check.outside.is_none());
        assert!(check.mean_pair_mass < check.minimum_mean_pair_mass);
        assert!(!check.permits_commit());
    }

    #[test]
    fn zero_drop_rejects_regression_below_the_probability_range() {
        let before = [[0., -1., 800.], [0., -1., 802.]];
        let after = [[0., -1., 801.], [0., -1., 802.]];
        let floor = MassFloor::from_baseline(mass_for(&before), 2, 0.);
        assert_eq!(floor.minimum, 0.);
        assert!(floor.check(mass_for(&before), 2).permits_commit());
        let check = floor.check(mass_for(&after), 2);
        assert_eq!(check.mean_pair_mass, 0.);
        let log_pair = check.log_pair.as_ref().unwrap();
        assert!(log_pair.log_mean_pair_mass < log_pair.minimum_log_mean_pair_mass);
        assert!(!check.permits_commit());

        let released = MassFloor::from_baseline(mass_for(&before), 2, f64::from_bits(1));
        assert!(released.check(mass_for(&after), 2).permits_commit());
    }

    #[test]
    fn subnormal_positive_allowance_is_subtracted_in_log_space() {
        let before = [[0., -1., 743.]];
        let floor = MassFloor::from_baseline(mass_for(&before), 1, f64::from_bits(2));
        let within = floor.check(mass_for(&[[0., -1., 743.3]]), 1);
        let below = floor.check(mass_for(&[[0., -1., 743.6]]), 1);
        assert!(within.log_pair.is_some() && within.permits_commit());
        assert!(below.log_pair.is_some() && !below.permits_commit());
    }
}
