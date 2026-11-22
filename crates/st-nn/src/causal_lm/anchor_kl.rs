//! Measure displacement in predictive distributions, not in adapter coordinates.
use super::*;

mod criterion;
mod distribution;
pub use criterion::CausalLmAnchorKlCriterion;
use criterion::Position;
use distribution::Reference;

#[derive(Debug, Copy, Clone, Serialize)]
pub struct CausalLmAnchorKlPolicy {
    /// Target forward KL in nats on last-position distributions (mean unless opted into peak).
    pub target_kl: f64,
    /// Maximum requested distance on the sealed joint parameter ray.
    pub max_parameter_l2: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CausalLmAnchorKlTrial {
    pub requested_l2: f64,
    pub mean_kl: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak: Option<CausalLmAnchorKlPeak>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CausalLmAnchorKlPeak {
    /// Zero-based index in the ordered anchor bank, not in the schedule. Ties choose first.
    pub anchor_index: usize,
    pub kl: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CausalLmAnchorKlMatch {
    pub method: &'static str,
    pub policy: CausalLmAnchorKlPolicy,
    #[serde(skip_serializing_if = "CausalLmAnchorKlCriterion::is_mean")]
    pub criterion: CausalLmAnchorKlCriterion,
    pub mean_kl: f64,
    pub per_anchor_kl: Vec<f64>,
    /// Counts individual last-position vocabulary readouts, including the origin.
    pub anchor_readouts: usize,
    pub trials: Vec<CausalLmAnchorKlTrial>,
}

/// A bounded search did not establish a match; this does not prove none exists.
#[derive(Debug, Serialize)]
pub struct CausalLmAnchorKlRefusal {
    pub reason: &'static str,
    pub policy: CausalLmAnchorKlPolicy,
    #[serde(skip_serializing_if = "CausalLmAnchorKlCriterion::is_mean")]
    pub criterion: CausalLmAnchorKlCriterion,
    pub anchor_readouts: usize,
    pub trials: Vec<CausalLmAnchorKlTrial>,
}
impl std::fmt::Display for CausalLmAnchorKlRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "anchor KL not matched: {} (target {}, last trial {:?})",
            self.reason,
            self.policy.target_kl,
            self.trials.last()
        )
    }
}
impl std::error::Error for CausalLmAnchorKlRefusal {}

impl CausalLmAnchorKlPolicy {
    pub const MAX_ANCHORS: usize = 16;
    pub const MAX_ANCHOR_IDS: usize = 1024;
    pub const MAX_ANCHOR_LOGITS: usize = 1 << 20;
    pub const MAX_TRIALS: usize = 20;
    pub const MIN_TARGET_RATIO: f64 = 0.98;

    pub(super) fn admit(
        self,
        model: &NativeCausalLm,
        mode: CausalLmAdaptationMode,
        prompts: &[&[usize]],
        criterion: CausalLmAnchorKlCriterion,
    ) -> CausalLmResult<()> {
        if !self.target_kl.is_finite() || self.target_kl <= 0. || self.max_parameter_l2 <= 0. {
            return Err("anchor KL needs a positive finite target and parameter ceiling".into());
        }
        criterion.admit(self.target_kl)?;
        model.validate_adaptation_mode(mode)?;
        model.admit_adaptation_distance(mode, self.max_parameter_l2)?;
        if !(1..=Self::MAX_ANCHORS).contains(&prompts.len())
            || model.config.vocabulary > Self::MAX_ANCHOR_LOGITS / prompts.len()
        {
            return Err("anchor KL reference bank exceeds count or vocabulary budget".into());
        }
        let mut remaining = Self::MAX_ANCHOR_IDS;
        for (i, &p) in prompts.iter().enumerate() {
            remaining = remaining
                .checked_sub(p.len())
                .ok_or("anchor KL token budget exceeded")?;
            model.input_count([1, p.len()], true)?;
            if prompts[..i].contains(&p) || p.iter().any(|&id| id >= model.config.vocabulary) {
                return Err("anchor KL requires distinct, in-vocabulary prompts".into());
            }
        }
        Ok(())
    }
}

impl NativeCausalLm {
    /// Find a measured finite update from this one origin and gradient. Never
    /// accept an undersized step merely because it satisfies an upper bound.
    /// Bisection maintains a measured bracket, not a global monotonicity claim.
    pub fn step_adaptation_at_anchor_kl(
        &self,
        grads: &NativeCausalLmAdaptationGradients,
        mode: CausalLmAdaptationMode,
        prompts: &[&[usize]],
        policy: CausalLmAnchorKlPolicy,
    ) -> CausalLmResult<(Self, CausalLmAdaptationRayStep, CausalLmAnchorKlMatch)> {
        self.step_adaptation_at_anchor_kl_with_criterion(
            grads,
            mode,
            prompts,
            policy,
            Default::default(),
        )
    }

    /// Change only the acceptance test, not the gradient, anchor weighting or ray.
    pub fn step_adaptation_at_anchor_kl_with_criterion(
        &self,
        grads: &NativeCausalLmAdaptationGradients,
        mode: CausalLmAdaptationMode,
        prompts: &[&[usize]],
        policy: CausalLmAnchorKlPolicy,
        criterion: CausalLmAnchorKlCriterion,
    ) -> CausalLmResult<(Self, CausalLmAdaptationRayStep, CausalLmAnchorKlMatch)> {
        policy.admit(self, mode, prompts, criterion)?;
        let (mut lo, mut hi) = (0., policy.max_parameter_l2);
        let mut distance = hi;
        let mut refs = Vec::with_capacity(prompts.len());
        let mut trials = Vec::new();
        for i in 0..CausalLmAnchorKlPolicy::MAX_TRIALS {
            let (candidate, ray) =
                self.step_adaptation_gradient_at_distance(grads, mode, distance)?;
            if i == 0 {
                for p in prompts {
                    refs.push(Reference::new(self.next_token_logits(p)?.data())?);
                }
            }
            let mut sum = sum::Sum::default();
            let mut per_anchor_kl = Vec::with_capacity(prompts.len());
            let mut peak = CausalLmAnchorKlPeak {
                anchor_index: 0,
                kl: 0.,
            };
            for (anchor_index, (p, source)) in prompts.iter().zip(&refs).enumerate() {
                let kl = source.forward_kl(candidate.next_token_logits(p)?.data())?;
                sum.add(kl);
                if kl > peak.kl {
                    peak = CausalLmAnchorKlPeak { anchor_index, kl };
                }
                per_anchor_kl.push(kl);
            }
            let mean_kl = sum.mean(prompts.len() as u64);
            trials.push(CausalLmAnchorKlTrial {
                requested_l2: distance,
                mean_kl,
                peak: (!criterion.is_mean()).then_some(peak),
            });
            let anchor_readouts = (1 + trials.len()) * prompts.len();
            match criterion.position(policy.target_kl, mean_kl, peak.kl) {
                Position::Matched => {
                    let report = CausalLmAnchorKlMatch {
                        method: "last_token_anchor_forward_kl.v1",
                        policy,
                        criterion,
                        mean_kl,
                        per_anchor_kl,
                        anchor_readouts,
                        trials,
                    };
                    return Ok((candidate, ray, report));
                }
                Position::Below if i == 0 => {
                    return Err(CausalLmAnchorKlRefusal {
                        reason: "ceiling below matching interval",
                        policy,
                        criterion,
                        anchor_readouts,
                        trials,
                    }
                    .into());
                }
                Position::Below => lo = distance,
                Position::Above => hi = distance,
            }
            distance = lo + 0.5 * (hi - lo);
            if distance <= lo || distance >= hi {
                break;
            }
        }
        Err(CausalLmAnchorKlRefusal {
            reason: "bounded bracket exhausted",
            policy,
            criterion,
            anchor_readouts: (1 + trials.len()) * prompts.len(),
            trials,
        }
        .into())
    }
}
