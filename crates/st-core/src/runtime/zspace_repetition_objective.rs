// SPDX-License-Identifier: AGPL-3.0-or-later

//! Optional objective scaling, independent of the frozen v3 candidate planner.
//! Clients differentiate the active-position mean; Rust owns its coefficient.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::zspace_repetition_unlikelihood::{
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER, ZSPACE_REPETITION_UNLIKELIHOOD_MAX_STRENGTH,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS,
};

pub const CONTRACT_VERSION: &str = "spiraltorch.zspace_repetition_objective.v1";
pub const SEMANTIC_OWNER: &str = "st-core::runtime::zspace_repetition_objective";
pub const CLOCK_RULE: &str =
    "completed trainer update slots before this microbatch; all microbatches in an accumulation group share a slot; an AMP-skipped optimizer update may still advance the trainer slot; not a successful-update counter";
pub const OBJECTIVE_RULE: &str =
    "training_loss=causal_lm_loss+effective_strength*active_position_mean; effective_strength=base_strength*schedule_scale*normalization_scale; eligible_targets uses active_position_count/eligible_target_count within each microbatch, not a global token mean across accumulation or ranks; evaluation remains causal_lm_loss";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceRepetitionObjectiveConfig {
    pub normalization: ZSpaceRepetitionObjectiveNormalization,
    pub schedule: ZSpaceRepetitionObjectiveSchedule,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ZSpaceRepetitionObjectiveNormalization {
    ActivePositions,
    /// Eligibility follows the candidate source's mask/prefix rule, not all LM labels.
    EligibleTargets,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ZSpaceRepetitionObjectiveSchedule {
    Constant {},
    /// Full scale through `start_update`, final scale from `end_update` onward.
    LinearDecay {
        start_update: u64,
        end_update: u64,
        final_scale: f64,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceRepetitionObjectiveRequest {
    pub config: ZSpaceRepetitionObjectiveConfig,
    pub base_strength: f64,
    pub completed_update_slots: u64,
    pub active_position_count: u64,
    pub eligible_target_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceRepetitionObjectivePolicy {
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub clock_rule: &'static str,
    pub objective_rule: &'static str,
    pub config: ZSpaceRepetitionObjectiveConfig,
    pub base_strength: f64,
    pub policy_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceRepetitionObjectiveControl {
    pub policy: ZSpaceRepetitionObjectivePolicy,
    pub completed_update_slots: u64,
    pub active_position_count: u64,
    pub eligible_target_count: u64,
    pub schedule_scale: f64,
    pub normalization_scale: f64,
    pub effective_strength: f64,
}

#[derive(Debug, Error, PartialEq)]
pub enum ZSpaceRepetitionObjectiveError {
    #[error("invalid repetition objective: {0}")]
    Invalid(&'static str),
}

pub fn zspace_repetition_objective_control(
    request: ZSpaceRepetitionObjectiveRequest,
) -> Result<ZSpaceRepetitionObjectiveControl, ZSpaceRepetitionObjectiveError> {
    use ZSpaceRepetitionObjectiveError::Invalid;
    if !request.base_strength.is_finite()
        || !(0.0..=ZSPACE_REPETITION_UNLIKELIHOOD_MAX_STRENGTH).contains(&request.base_strength)
    {
        return Err(Invalid("base_strength must be finite and in [0, 10]"));
    }
    if request.completed_update_slots > ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER {
        return Err(Invalid(
            "completed_update_slots exceeds the portable integer limit",
        ));
    }
    if request.active_position_count > request.eligible_target_count
        || request.eligible_target_count > ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS as u64
    {
        return Err(Invalid(
            "counts must satisfy active <= eligible <= maximum plan tokens",
        ));
    }
    let schedule_scale = match request.config.schedule {
        ZSpaceRepetitionObjectiveSchedule::Constant {} => 1.0,
        ZSpaceRepetitionObjectiveSchedule::LinearDecay {
            start_update,
            end_update,
            final_scale,
        } => {
            if start_update >= end_update
                || end_update > ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER
                || !final_scale.is_finite()
                || !(0.0..=1.0).contains(&final_scale)
            {
                return Err(Invalid("linear decay requires 0 <= start < end <= MAX_SAFE_INTEGER and finite final_scale in [0, 1]"));
            }
            if request.completed_update_slots <= start_update {
                1.0
            } else if request.completed_update_slots >= end_update {
                final_scale
            } else {
                let progress = (request.completed_update_slots - start_update) as f64
                    / (end_update - start_update) as f64;
                1.0 - progress * (1.0 - final_scale)
            }
        }
    };
    let normalization_scale = if request.active_position_count == 0 {
        0.0
    } else {
        match request.config.normalization {
            ZSpaceRepetitionObjectiveNormalization::ActivePositions => 1.0,
            ZSpaceRepetitionObjectiveNormalization::EligibleTargets => {
                request.active_position_count as f64 / request.eligible_target_count as f64
            }
        }
    };
    let identity = serde_json::to_vec(&(
        CONTRACT_VERSION,
        CLOCK_RULE,
        OBJECTIVE_RULE,
        &request.config,
        request.base_strength,
    ))
    .expect("validated finite repetition objective is serializable");
    Ok(ZSpaceRepetitionObjectiveControl {
        effective_strength: request.base_strength * schedule_scale * normalization_scale,
        policy: ZSpaceRepetitionObjectivePolicy {
            contract_version: CONTRACT_VERSION,
            semantic_owner: SEMANTIC_OWNER,
            clock_rule: CLOCK_RULE,
            objective_rule: OBJECTIVE_RULE,
            config: request.config,
            base_strength: request.base_strength,
            policy_id: format!("sha256:{:x}", Sha256::digest(identity)),
        },
        completed_update_slots: request.completed_update_slots,
        active_position_count: request.active_position_count,
        eligible_target_count: request.eligible_target_count,
        schedule_scale,
        normalization_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ZSpaceRepetitionObjectiveRequest {
        ZSpaceRepetitionObjectiveRequest {
            config: ZSpaceRepetitionObjectiveConfig {
                normalization: ZSpaceRepetitionObjectiveNormalization::EligibleTargets,
                schedule: ZSpaceRepetitionObjectiveSchedule::LinearDecay {
                    start_update: 2,
                    end_update: 6,
                    final_scale: 0.0,
                },
            },
            base_strength: 0.2,
            completed_update_slots: 4,
            active_position_count: 2,
            eligible_target_count: 8,
        }
    }

    #[test]
    fn scales_the_differentiable_mean_without_changing_candidate_semantics() {
        let control = zspace_repetition_objective_control(request()).unwrap();
        assert_eq!(control.schedule_scale, 0.5);
        assert_eq!(control.normalization_scale, 0.25);
        assert_eq!(control.effective_strength, 0.025);
    }

    #[test]
    fn schedule_is_absolute_and_has_explicit_endpoints() {
        for (slot, expected) in [(0, 1.0), (2, 1.0), (3, 0.75), (6, 0.0), (9, 0.0)] {
            let mut req = request();
            req.completed_update_slots = slot;
            assert_eq!(
                zspace_repetition_objective_control(req)
                    .unwrap()
                    .schedule_scale,
                expected
            );
        }
        let mut req = request();
        req.config.schedule = ZSpaceRepetitionObjectiveSchedule::LinearDecay {
            start_update: 0,
            end_update: 4,
            final_scale: 0.2,
        };
        assert_eq!(
            zspace_repetition_objective_control(req)
                .unwrap()
                .schedule_scale,
            0.2
        );
    }

    #[test]
    fn constant_active_mean_preserves_legacy_coefficient() {
        let mut req = request();
        req.config.normalization = ZSpaceRepetitionObjectiveNormalization::ActivePositions;
        req.config.schedule = ZSpaceRepetitionObjectiveSchedule::Constant {};
        assert_eq!(
            zspace_repetition_objective_control(req)
                .unwrap()
                .effective_strength,
            0.2
        );
    }

    #[test]
    fn empty_and_inactive_batches_have_zero_coefficient() {
        for eligible in [0, 8] {
            let mut req = request();
            req.active_position_count = 0;
            req.eligible_target_count = eligible;
            assert_eq!(
                zspace_repetition_objective_control(req)
                    .unwrap()
                    .effective_strength,
                0.0
            );
        }
    }

    #[test]
    fn policy_identity_survives_restart_but_binds_recipe() {
        let req = request();
        let id = zspace_repetition_objective_control(req.clone())
            .unwrap()
            .policy
            .policy_id;
        let mut resumed: ZSpaceRepetitionObjectiveRequest =
            serde_json::from_slice(&serde_json::to_vec(&req).unwrap()).unwrap();
        resumed.completed_update_slots = 6;
        resumed.active_position_count = 1;
        assert_eq!(
            id,
            zspace_repetition_objective_control(resumed.clone())
                .unwrap()
                .policy
                .policy_id
        );
        resumed.base_strength = 0.1;
        assert_ne!(
            id,
            zspace_repetition_objective_control(resumed)
                .unwrap()
                .policy
                .policy_id
        );
    }

    #[test]
    fn rejects_nonfinite_strength_counts_and_nonportable_clock() {
        for strength in [f64::NAN, f64::INFINITY, -0.1, 10.1] {
            let mut req = request();
            req.base_strength = strength;
            assert!(zspace_repetition_objective_control(req).is_err());
        }
        for (active, eligible) in [(1, 0), (9, 8), (1, 1_000_001)] {
            let mut req = request();
            req.active_position_count = active;
            req.eligible_target_count = eligible;
            assert!(zspace_repetition_objective_control(req).is_err());
        }
        let mut req = request();
        req.completed_update_slots = u64::MAX;
        assert!(zspace_repetition_objective_control(req).is_err());
    }

    #[test]
    fn rejects_invalid_schedule_and_unknown_fields() {
        for (start_update, end_update, final_scale) in [
            (2, 2, 0.0),
            (3, 2, 0.0),
            (0, u64::MAX, 0.0),
            (0, 2, f64::NAN),
            (0, 2, 1.1),
            (0, 2, -0.1),
        ] {
            let mut req = request();
            req.config.schedule = ZSpaceRepetitionObjectiveSchedule::LinearDecay {
                start_update,
                end_update,
                final_scale,
            };
            assert!(zspace_repetition_objective_control(req).is_err());
        }
        let mut value = serde_json::to_value(request()).unwrap();
        value["config"]["schedule"]["typo"] = true.into();
        assert!(serde_json::from_value::<ZSpaceRepetitionObjectiveRequest>(value).is_err());
        let mut value = serde_json::to_value(request()).unwrap();
        value["config"]["schedule"] = serde_json::json!({"kind": "constant", "typo": true});
        assert!(serde_json::from_value::<ZSpaceRepetitionObjectiveRequest>(value).is_err());
    }
}
