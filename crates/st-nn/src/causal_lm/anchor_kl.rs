//! Measure displacement in predictive distributions, not in adapter coordinates.
use super::*;

mod criterion;
mod direction;
mod distribution;
mod search;
mod sensitivity;
pub use criterion::CausalLmAnchorKlCriterion;
use criterion::Position;
pub use direction::{
    CausalLmAnchorDirection, CausalLmSecantAnchor, CausalLmSecantMidpoint, CausalLmSecantReport,
};
use distribution::Reference;
use search::{AnchorBank, RaySearch, SearchRay};
pub use sensitivity::{
    CausalLmSensitivityAxis, CausalLmSensitivityBasis, CausalLmSensitivityPrediction,
    CausalLmSensitivityReport, CausalLmSensitivitySelect,
};

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
pub struct CausalLmRawRayMiss {
    pub reason: &'static str,
    pub trials: Vec<CausalLmAnchorKlTrial>,
    /// Work performed before the optional rescue, including the one shared origin bank.
    pub anchor_readouts: usize,
    pub rescue_policy: CausalLmAnchorDirection,
}

#[derive(Debug, Clone, Serialize)]
pub struct CausalLmAnchorKlMatch {
    pub method: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<CausalLmSecantReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sensitivity: Option<Box<CausalLmSensitivityReport>>,
    /// Present only if the first bounded loss-gradient search missed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_miss: Option<Box<CausalLmRawRayMiss>>,
    pub policy: CausalLmAnchorKlPolicy,
    #[serde(skip_serializing_if = "CausalLmAnchorKlCriterion::is_mean")]
    pub criterion: CausalLmAnchorKlCriterion,
    pub mean_kl: f64,
    pub per_anchor_kl: Vec<f64>,
    /// Counts last-position vocabulary readouts at the origin, optional probe and candidates.
    pub anchor_readouts: usize,
    /// Only the terminal ray. Earlier raw trials, if any, are in raw_miss.
    pub trials: Vec<CausalLmAnchorKlTrial>,
}

/// A bounded search did not establish a match; this does not prove none exists.
#[derive(Debug, Serialize)]
pub struct CausalLmAnchorKlRefusal {
    pub reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<CausalLmSecantReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sensitivity: Option<Box<CausalLmSensitivityReport>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_miss: Option<Box<CausalLmRawRayMiss>>,
    pub policy: CausalLmAnchorKlPolicy,
    #[serde(skip_serializing_if = "CausalLmAnchorKlCriterion::is_mean")]
    pub criterion: CausalLmAnchorKlCriterion,
    pub anchor_readouts: usize,
    /// Empty if a raw miss triggered a probe but no midpoint direction could be formed.
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
        self.step_adaptation_at_anchor_kl_with_direction(
            grads,
            mode,
            prompts,
            policy,
            criterion,
            Default::default(),
        )
    }

    /// Optionally reshape the selected loss ray before the unchanged finite KL search.
    pub fn step_adaptation_at_anchor_kl_with_direction(
        &self,
        grads: &NativeCausalLmAdaptationGradients,
        mode: CausalLmAdaptationMode,
        prompts: &[&[usize]],
        policy: CausalLmAnchorKlPolicy,
        criterion: CausalLmAnchorKlCriterion,
        direction: CausalLmAnchorDirection,
    ) -> CausalLmResult<(Self, CausalLmAdaptationRayStep, CausalLmAnchorKlMatch)> {
        policy.admit(self, mode, prompts, criterion)?;
        self.validate_adaptation_gradient_owner(grads)?;
        direction.admit(self, mode, prompts, policy.max_parameter_l2, criterion)?;
        // Missing provenance is an admission error, even if the raw ray might fit.
        let spec = direction.resolve(grads, prompts)?;
        let mut search = RaySearch::new(self, grads.parameters(), mode, prompts, policy, criterion);
        let raw_miss = if !matches!(direction, CausalLmAnchorDirection::SecantProjection { .. }) {
            match search.run(SearchRay::Raw, None) {
                Ok(step) => return Ok(step),
                Err(err) if direction.is_rescue() => {
                    // Only a measured bracket miss is a license to change directions.
                    let miss = *err.downcast::<CausalLmAnchorKlRefusal>()?;
                    Some(Box::new(CausalLmRawRayMiss {
                        reason: miss.reason,
                        trials: miss.trials,
                        anchor_readouts: miss.anchor_readouts,
                        rescue_policy: direction,
                    }))
                }
                Err(err) => return Err(err),
            }
        } else {
            None
        };
        if let CausalLmAnchorDirection::RawThenSensitivityRay {
            probe_l2,
            basis,
            select,
        } = direction
        {
            let (ray, report) = self.sensitivity_ray(
                grads.parameters(),
                mode,
                &mut search.bank,
                sensitivity::Settings::new(policy, criterion, probe_l2, basis, select)?,
            )?;
            return if let Some(h) = ray {
                search.run(SearchRay::Sensitivity(&h, Box::new(report)), raw_miss)
            } else {
                Err(CausalLmAnchorKlRefusal {
                    reason: "no measurable sensitivity direction",
                    direction: None,
                    sensitivity: Some(Box::new(report)),
                    raw_miss,
                    policy,
                    criterion,
                    anchor_readouts: search.bank.readouts,
                    trials: Vec::new(),
                }
                .into())
            };
        }
        let shaped = self.secant_direction(
            grads.parameters(),
            mode,
            spec.ok_or("expected a secant direction")?,
            &mut search.bank,
        );
        let (h, report) = match shaped {
            Ok(result) => result,
            Err(err) if raw_miss.is_some() => {
                // A vanishing normal or residual is a documented, non-advancing
                // mathematical outcome. Other numeric/ownership errors stay errors.
                let unresolved = err.downcast::<direction::UnresolvedSecant>()?;
                return Err(CausalLmAnchorKlRefusal {
                    reason: unresolved.0,
                    policy,
                    criterion,
                    raw_miss,
                    direction: None,
                    sensitivity: None,
                    anchor_readouts: search.bank.readouts,
                    trials: Vec::new(),
                }
                .into());
            }
            Err(err) => return Err(err),
        };
        search.run(SearchRay::Secant(&h, report), raw_miss)
    }
}
