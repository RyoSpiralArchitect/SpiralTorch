use super::{CausalLmAnchorKlPolicy, CausalLmResult, Deserialize, Serialize};

/// Which measured statistic must use 98..=100% of target_kl. All KLs are nats.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CausalLmAnchorKlCriterion {
    Mean {},
    MeanWithCap { max_anchor_kl: f64 },
    Peak {},
}

impl Default for CausalLmAnchorKlCriterion {
    fn default() -> Self {
        Self::Mean {}
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Position {
    Below,
    Matched,
    Above,
}

impl CausalLmAnchorKlCriterion {
    pub(super) fn is_mean(&self) -> bool {
        matches!(self, Self::Mean {})
    }

    pub(super) fn admit(self, target: f64) -> CausalLmResult<()> {
        if let Self::MeanWithCap { max_anchor_kl: cap } = self {
            if !cap.is_finite()
                || cap <= 0.
                || cap / target < CausalLmAnchorKlPolicy::MIN_TARGET_RATIO
            {
                return Err(
                    "per-anchor KL cap must be finite, positive and permit the required mean floor"
                        .into(),
                );
            }
        }
        Ok(())
    }

    pub(super) fn position(self, target: f64, mean: f64, peak: f64) -> Position {
        let metric = match self {
            Self::MeanWithCap { max_anchor_kl } if peak > max_anchor_kl => return Position::Above,
            Self::Peak {} => peak,
            _ => mean,
        };
        if metric > target {
            Position::Above
        } else if metric / target >= CausalLmAnchorKlPolicy::MIN_TARGET_RATIO {
            Position::Matched
        } else {
            Position::Below
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use CausalLmAnchorKlCriterion::*;
    use Position::*;

    #[test]
    fn upper_bounds_are_inclusive_without_loosening_the_mean_floor() {
        let rule = MeanWithCap { max_anchor_kl: 2. };
        assert_eq!(rule.position(1., 0.98, 2.), Matched);
        assert_eq!(rule.position(1., 1., 2.), Matched);
        assert_eq!(rule.position(1., 0.98_f64.next_down(), 2.), Below);
        assert_eq!(rule.position(1., 1_f64.next_up(), 1.5), Above);
        assert_eq!(rule.position(1., 0.9, 2_f64.next_up()), Above);
        assert_eq!(Peak {}.position(2., 0.8, 1.96), Matched);
        assert_eq!(Peak {}.position(2., 0.8, 2_f64.next_up()), Above);
        assert_eq!(Mean {}.position(1., 1., 2.9), Matched);
        assert!(MeanWithCap {
            max_anchor_kl: 0.98
        }
        .admit(1.)
        .is_ok());
        for cap in [0.98_f64.next_down(), -1., f64::INFINITY, f64::NAN] {
            assert!(MeanWithCap { max_anchor_kl: cap }.admit(1.).is_err());
        }
    }
}
