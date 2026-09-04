// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Rust-owned SpiralK candidate planning and Black Cat rank adaptation.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::ops::rank_entry::{RankPlan, RankPlanError, RankPlanSnapshot};
use crate::runtime::blackcat::{
    bandit::{BanditDecisionWitness, SoftBanditMode},
    BlackCatError, ChoiceGroups, MultiBandit,
};

pub const RANK_ADAPTATION_CONTRACT_VERSION: &str = "spiraltorch.rank_adaptation.v1";
pub const RANK_ADAPTATION_KIND: &str = "spiraltorch.rank_adaptation";
pub const RANK_ADAPTATION_SEMANTIC_OWNER: &str = "st-core::runtime::rank_adaptation";
pub const RANK_ADAPTATION_SEMANTIC_BACKEND: &str = "rust";
pub const RANK_ADAPTATION_REWARD_FORMULA: &str = "1 / (1 + elapsed_ms)";
pub const RANK_ADAPTATION_MAX_CANDIDATES: usize = 32;
pub const RANK_ADAPTATION_MAX_SCRIPT_BYTES: usize = 16 * 1024;
pub const RANK_ADAPTATION_MAX_TOTAL_SCRIPT_BYTES: usize = 128 * 1024;
pub const RANK_ADAPTATION_MAX_SAFE_SELECTION_ID: u64 = 9_007_199_254_740_991;

const VARIANT_GROUP: &str = "rank_plan_variant";
const BANDIT_CONTEXT: [f64; 1] = [1.0];

#[derive(Clone, Debug, Error, PartialEq)]
pub enum RankAdaptationError {
    #[error("rank adaptation requires at least one SpiralK candidate")]
    NoCandidates,
    #[error("rank adaptation accepts at most {maximum} candidates, got {actual}")]
    TooManyCandidates { maximum: usize, actual: usize },
    #[error("SpiralK candidate {index} must not be empty")]
    EmptyScript { index: usize },
    #[error("SpiralK candidate {index} exceeds {maximum} bytes, got {actual}")]
    ScriptTooLarge {
        index: usize,
        maximum: usize,
        actual: usize,
    },
    #[error("aggregate SpiralK source exceeds {maximum} bytes, got {actual}")]
    TotalScriptBytesExceeded { maximum: usize, actual: usize },
    #[error("SpiralK candidate {duplicate} duplicates source from candidate {first}")]
    DuplicateScript { first: usize, duplicate: usize },
    #[error("SpiralK candidate {duplicate} resolves to the same rank choice as candidate {first}")]
    DuplicatePlan { first: usize, duplicate: usize },
    #[error("SpiralK candidate {index} failed: {detail}")]
    SpiralK { index: usize, detail: String },
    #[error("SpiralK candidate {index} produced an invalid rank plan: {detail}")]
    CandidateRankPlan { index: usize, detail: String },
    #[error(transparent)]
    RankPlan(#[from] RankPlanError),
    #[error(transparent)]
    BlackCat(#[from] BlackCatError),
    #[error(
        "rank adaptation selection {selection_id} is still awaiting observation or abandonment"
    )]
    PendingSelection { selection_id: u64 },
    #[error("rank adaptation has no pending selection")]
    MissingSelection,
    #[error("rank adaptation expected selection {expected}, got {actual}")]
    SelectionMismatch { expected: u64, actual: u64 },
    #[error("rank adaptation exhausted its transport-safe selection IDs")]
    SelectionIdExhausted,
    #[error("rank adaptation elapsed_ms must be finite, got {value}")]
    NonFiniteElapsed { value: f64 },
    #[error("rank adaptation elapsed_ms must be non-negative, got {value}")]
    NegativeElapsed { value: f64 },
    #[error("rank adaptation state is invalid at '{field}'")]
    InvalidState { field: &'static str },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RankAdaptationCandidateSnapshot {
    pub index: usize,
    pub spiralk_source_sha256: String,
    pub plan: RankPlanSnapshot,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankAdaptationSelectionReceipt {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub policy: &'static str,
    pub selection_id: u64,
    pub candidate_index: usize,
    pub spiralk_source_sha256: String,
    pub plan: RankPlanSnapshot,
    pub decision: BanditDecisionWitness,
}

#[derive(Clone, Debug)]
pub struct RankAdaptationSelection {
    plan: RankPlan,
    receipt: RankAdaptationSelectionReceipt,
}

impl RankAdaptationSelection {
    pub fn plan(&self) -> &RankPlan {
        &self.plan
    }

    pub fn receipt(&self) -> &RankAdaptationSelectionReceipt {
        &self.receipt
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankAdaptationObservationReceipt {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub selection_id: u64,
    pub candidate_index: usize,
    pub elapsed_ms: f64,
    pub correctness_passed: bool,
    pub credited: bool,
    pub reward: Option<f64>,
    pub reward_formula: &'static str,
    pub observation_counts: BTreeMap<String, BTreeMap<String, u64>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankAdaptationAbandonmentReceipt {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub selection_id: u64,
    pub candidate_index: usize,
    pub credited: bool,
    pub observation_counts: BTreeMap<String, BTreeMap<String, u64>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankAdaptationSessionSnapshot {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub policy: &'static str,
    pub seed: u64,
    pub bandit_context: [f64; 1],
    pub reward_formula: &'static str,
    pub candidates: Vec<RankAdaptationCandidateSnapshot>,
    pub choice_domains: BTreeMap<String, Vec<String>>,
    pub observation_counts: BTreeMap<String, BTreeMap<String, u64>>,
    pub pending_selection_id: Option<u64>,
}

#[derive(Clone, Debug)]
struct RankAdaptationCandidate {
    source_sha256: String,
    plan: RankPlan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingSelection {
    selection_id: u64,
    candidate_index: usize,
}

/// One-workload controller whose candidate and reward semantics stay in Rust.
pub struct RankAdaptationSession {
    candidates: Vec<RankAdaptationCandidate>,
    bandit: MultiBandit,
    policy: SoftBanditMode,
    seed: u64,
    next_selection_id: u64,
    pending: Option<PendingSelection>,
}

impl RankAdaptationSession {
    /// Evaluates bounded SpiralK sources against one base plan and initializes Black Cat.
    pub fn try_from_spiralk(
        base: &RankPlan,
        scripts: &[String],
        policy: SoftBanditMode,
        seed: u64,
    ) -> Result<Self, RankAdaptationError> {
        base.validate()?;
        validate_scripts(scripts)?;
        let context = base.spiralk_context()?;
        let mut source_digests = HashMap::<String, usize>::new();
        let mut plan_choices = HashMap::new();
        let mut candidates = Vec::with_capacity(scripts.len());

        for (index, script) in scripts.iter().enumerate() {
            let canonical = script.trim();
            let source_sha256 = sha256_hex(canonical.as_bytes());
            if let Some(first) = source_digests.insert(source_sha256.clone(), index) {
                return Err(RankAdaptationError::DuplicateScript {
                    first,
                    duplicate: index,
                });
            }
            let output = st_kdsl::eval_program(canonical, &context).map_err(|error| {
                RankAdaptationError::SpiralK {
                    index,
                    detail: error.to_string(),
                }
            })?;
            let plan = base.try_with_spiralk_hard(&output.hard).map_err(|error| {
                RankAdaptationError::CandidateRankPlan {
                    index,
                    detail: error.to_string(),
                }
            })?;
            let choice_key = serde_json::to_string(&plan.snapshot().choice).map_err(|_| {
                RankAdaptationError::InvalidState {
                    field: "candidate.choice_serialization",
                }
            })?;
            if let Some(first) = plan_choices.insert(choice_key, index) {
                return Err(RankAdaptationError::DuplicatePlan {
                    first,
                    duplicate: index,
                });
            }
            candidates.push(RankAdaptationCandidate {
                source_sha256,
                plan,
            });
        }

        let groups = ChoiceGroups {
            groups: HashMap::from([(
                VARIANT_GROUP.to_owned(),
                (0..candidates.len())
                    .map(|index| index.to_string())
                    .collect(),
            )]),
        };
        let bandit = MultiBandit::try_new_seeded(&groups, BANDIT_CONTEXT.len(), policy, seed)?;
        Ok(Self {
            candidates,
            bandit,
            policy,
            seed,
            next_selection_id: 1,
            pending: None,
        })
    }

    /// Selects one executable plan and opens exactly one observation slot.
    pub fn try_choose(&mut self) -> Result<RankAdaptationSelection, RankAdaptationError> {
        self.validate_pending_state()?;
        if let Some(pending) = self.pending {
            return Err(RankAdaptationError::PendingSelection {
                selection_id: pending.selection_id,
            });
        }
        if self.next_selection_id > RANK_ADAPTATION_MAX_SAFE_SELECTION_ID {
            return Err(RankAdaptationError::SelectionIdExhausted);
        }

        let mut next_bandit = self.bandit.clone();
        let (picks, decisions) = next_bandit.try_select_all(&BANDIT_CONTEXT)?;
        let candidate_index = picks
            .get(VARIANT_GROUP)
            .ok_or(RankAdaptationError::InvalidState {
                field: "selection.variant_pick",
            })?
            .parse::<usize>()
            .map_err(|_| RankAdaptationError::InvalidState {
                field: "selection.variant_index",
            })?;
        let decision =
            decisions
                .get(VARIANT_GROUP)
                .cloned()
                .ok_or(RankAdaptationError::InvalidState {
                    field: "selection.variant_decision",
                })?;
        let candidate =
            self.candidates
                .get(candidate_index)
                .ok_or(RankAdaptationError::InvalidState {
                    field: "selection.candidate",
                })?;
        let selection_id = self.next_selection_id;
        let next_selection_id = selection_id
            .checked_add(1)
            .ok_or(RankAdaptationError::SelectionIdExhausted)?;
        let receipt = RankAdaptationSelectionReceipt {
            kind: RANK_ADAPTATION_KIND,
            contract_version: RANK_ADAPTATION_CONTRACT_VERSION,
            semantic_owner: RANK_ADAPTATION_SEMANTIC_OWNER,
            semantic_backend: RANK_ADAPTATION_SEMANTIC_BACKEND,
            policy: self.policy.as_str(),
            selection_id,
            candidate_index,
            spiralk_source_sha256: candidate.source_sha256.clone(),
            plan: candidate.plan.snapshot(),
            decision,
        };

        self.bandit = next_bandit;
        self.next_selection_id = next_selection_id;
        self.pending = Some(PendingSelection {
            selection_id,
            candidate_index,
        });
        Ok(RankAdaptationSelection {
            plan: candidate.plan.clone(),
            receipt,
        })
    }

    /// Closes a selection, crediting latency only when correctness passed.
    pub fn try_observe(
        &mut self,
        selection_id: u64,
        elapsed_ms: f64,
        correctness_passed: bool,
    ) -> Result<RankAdaptationObservationReceipt, RankAdaptationError> {
        validate_elapsed(elapsed_ms)?;
        let pending = self.require_pending(selection_id)?;
        let mut next_bandit = self.bandit.clone();
        let reward = if correctness_passed {
            let reward = 1.0 / (1.0 + elapsed_ms);
            next_bandit
                .try_update_all(&BANDIT_CONTEXT, reward)
                .map_err(|field| RankAdaptationError::InvalidState { field })?;
            Some(reward)
        } else {
            next_bandit
                .try_abandon_all()
                .map_err(|field| RankAdaptationError::InvalidState { field })?;
            None
        };
        let observation_counts = next_bandit.observation_counts();
        self.bandit = next_bandit;
        self.pending = None;
        Ok(RankAdaptationObservationReceipt {
            kind: RANK_ADAPTATION_KIND,
            contract_version: RANK_ADAPTATION_CONTRACT_VERSION,
            semantic_owner: RANK_ADAPTATION_SEMANTIC_OWNER,
            semantic_backend: RANK_ADAPTATION_SEMANTIC_BACKEND,
            selection_id,
            candidate_index: pending.candidate_index,
            elapsed_ms,
            correctness_passed,
            credited: correctness_passed,
            reward,
            reward_formula: RANK_ADAPTATION_REWARD_FORMULA,
            observation_counts,
        })
    }

    /// Releases a failed or cancelled execution without posterior credit.
    pub fn try_abandon(
        &mut self,
        selection_id: u64,
    ) -> Result<RankAdaptationAbandonmentReceipt, RankAdaptationError> {
        let pending = self.require_pending(selection_id)?;
        let mut next_bandit = self.bandit.clone();
        next_bandit
            .try_abandon_all()
            .map_err(|field| RankAdaptationError::InvalidState { field })?;
        let observation_counts = next_bandit.observation_counts();
        self.bandit = next_bandit;
        self.pending = None;
        Ok(RankAdaptationAbandonmentReceipt {
            kind: RANK_ADAPTATION_KIND,
            contract_version: RANK_ADAPTATION_CONTRACT_VERSION,
            semantic_owner: RANK_ADAPTATION_SEMANTIC_OWNER,
            semantic_backend: RANK_ADAPTATION_SEMANTIC_BACKEND,
            selection_id,
            candidate_index: pending.candidate_index,
            credited: false,
            observation_counts,
        })
    }

    pub fn snapshot(&self) -> RankAdaptationSessionSnapshot {
        RankAdaptationSessionSnapshot {
            kind: RANK_ADAPTATION_KIND,
            contract_version: RANK_ADAPTATION_CONTRACT_VERSION,
            semantic_owner: RANK_ADAPTATION_SEMANTIC_OWNER,
            semantic_backend: RANK_ADAPTATION_SEMANTIC_BACKEND,
            policy: self.policy.as_str(),
            seed: self.seed,
            bandit_context: BANDIT_CONTEXT,
            reward_formula: RANK_ADAPTATION_REWARD_FORMULA,
            candidates: self
                .candidates
                .iter()
                .enumerate()
                .map(|(index, candidate)| RankAdaptationCandidateSnapshot {
                    index,
                    spiralk_source_sha256: candidate.source_sha256.clone(),
                    plan: candidate.plan.snapshot(),
                })
                .collect(),
            choice_domains: self.bandit.choice_domains(),
            observation_counts: self.bandit.observation_counts(),
            pending_selection_id: self.pending.map(|pending| pending.selection_id),
        }
    }

    pub fn pending_selection_id(&self) -> Option<u64> {
        self.pending.map(|pending| pending.selection_id)
    }

    /// Exposes validated candidates to an executing backend for admission checks.
    pub fn candidate_plans(&self) -> impl ExactSizeIterator<Item = &RankPlan> {
        self.candidates.iter().map(|candidate| &candidate.plan)
    }

    fn require_pending(&self, selection_id: u64) -> Result<PendingSelection, RankAdaptationError> {
        self.validate_pending_state()?;
        let pending = self.pending.ok_or(RankAdaptationError::MissingSelection)?;
        if pending.selection_id != selection_id {
            return Err(RankAdaptationError::SelectionMismatch {
                expected: pending.selection_id,
                actual: selection_id,
            });
        }
        Ok(pending)
    }

    fn validate_pending_state(&self) -> Result<(), RankAdaptationError> {
        let bandit_pending = self
            .bandit
            .selection_pending()
            .map_err(|field| RankAdaptationError::InvalidState { field })?;
        if bandit_pending != self.pending.is_some() {
            return Err(RankAdaptationError::InvalidState {
                field: "session.pending_selection",
            });
        }
        Ok(())
    }
}

fn validate_scripts(scripts: &[String]) -> Result<(), RankAdaptationError> {
    if scripts.is_empty() {
        return Err(RankAdaptationError::NoCandidates);
    }
    if scripts.len() > RANK_ADAPTATION_MAX_CANDIDATES {
        return Err(RankAdaptationError::TooManyCandidates {
            maximum: RANK_ADAPTATION_MAX_CANDIDATES,
            actual: scripts.len(),
        });
    }
    let mut total = 0usize;
    for (index, script) in scripts.iter().enumerate() {
        if script.trim().is_empty() {
            return Err(RankAdaptationError::EmptyScript { index });
        }
        if script.len() > RANK_ADAPTATION_MAX_SCRIPT_BYTES {
            return Err(RankAdaptationError::ScriptTooLarge {
                index,
                maximum: RANK_ADAPTATION_MAX_SCRIPT_BYTES,
                actual: script.len(),
            });
        }
        total = total.checked_add(script.len()).ok_or(
            RankAdaptationError::TotalScriptBytesExceeded {
                maximum: RANK_ADAPTATION_MAX_TOTAL_SCRIPT_BYTES,
                actual: usize::MAX,
            },
        )?;
    }
    if total > RANK_ADAPTATION_MAX_TOTAL_SCRIPT_BYTES {
        return Err(RankAdaptationError::TotalScriptBytesExceeded {
            maximum: RANK_ADAPTATION_MAX_TOTAL_SCRIPT_BYTES,
            actual: total,
        });
    }
    Ok(())
}

fn validate_elapsed(elapsed_ms: f64) -> Result<(), RankAdaptationError> {
    if !elapsed_ms.is_finite() {
        return Err(RankAdaptationError::NonFiniteElapsed { value: elapsed_ms });
    }
    if elapsed_ms < 0.0 {
        return Err(RankAdaptationError::NegativeElapsed { value: elapsed_ms });
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::BackendKind;
    use crate::backend::execution_plan::{AcceleratorFallback, ExecutionConfig};
    use crate::backend::unison::RankKind;
    use crate::ops::rank_entry::try_plan_rank_with_config;

    fn base_plan() -> RankPlan {
        try_plan_rank_with_config(
            RankKind::TopK,
            2,
            256,
            8,
            BackendKind::Wgpu.default_caps(),
            ExecutionConfig::new(AcceleratorFallback::Forbid, 1024),
        )
        .expect("valid base plan")
    }

    fn scripts() -> Vec<String> {
        vec!["u2: false;".to_owned(), "u2: true;".to_owned()]
    }

    #[test]
    fn selection_and_correct_observation_share_one_rust_contract() {
        let mut session = RankAdaptationSession::try_from_spiralk(
            &base_plan(),
            &scripts(),
            SoftBanditMode::UCB,
            17,
        )
        .expect("session");
        let selection = session.try_choose().expect("selection");
        assert_eq!(selection.receipt().selection_id, 1);
        assert_eq!(session.pending_selection_id(), Some(1));
        assert!(matches!(
            session.try_choose(),
            Err(RankAdaptationError::PendingSelection { selection_id: 1 })
        ));

        let receipt = session
            .try_observe(1, 4.0, true)
            .expect("correct observation");
        assert!(receipt.credited);
        assert_eq!(receipt.reward, Some(0.2));
        assert_eq!(session.pending_selection_id(), None);
        assert_eq!(
            session
                .snapshot()
                .observation_counts
                .values()
                .flat_map(|counts| counts.values())
                .sum::<u64>(),
            1
        );
    }

    #[test]
    fn failed_correctness_and_abandonment_never_credit_a_candidate() {
        let mut session = RankAdaptationSession::try_from_spiralk(
            &base_plan(),
            &scripts(),
            SoftBanditMode::UCB,
            29,
        )
        .expect("session");
        let first = session.try_choose().expect("first selection");
        let rejected = session
            .try_observe(first.receipt().selection_id, 2.0, false)
            .expect("rejected correctness");
        assert!(!rejected.credited);
        assert_eq!(rejected.reward, None);

        let second = session.try_choose().expect("second selection");
        let abandoned = session
            .try_abandon(second.receipt().selection_id)
            .expect("abandonment");
        assert!(!abandoned.credited);
        assert_eq!(
            session
                .snapshot()
                .observation_counts
                .values()
                .flat_map(|counts| counts.values())
                .sum::<u64>(),
            0
        );
    }

    #[test]
    fn invalid_measurement_and_stale_id_preserve_the_pending_slot() {
        let mut session = RankAdaptationSession::try_from_spiralk(
            &base_plan(),
            &scripts(),
            SoftBanditMode::UCB,
            43,
        )
        .expect("session");
        let selection = session.try_choose().expect("selection");
        assert!(matches!(
            session.try_observe(selection.receipt().selection_id + 1, 1.0, true),
            Err(RankAdaptationError::SelectionMismatch { .. })
        ));
        assert!(matches!(
            session.try_observe(selection.receipt().selection_id, f64::NAN, true),
            Err(RankAdaptationError::NonFiniteElapsed { .. })
        ));
        assert_eq!(session.pending_selection_id(), Some(1));
        session.try_abandon(1).expect("pending slot survives");
    }

    #[test]
    fn duplicate_sources_and_equivalent_plans_fail_before_selection() {
        let base = base_plan();
        assert!(matches!(
            RankAdaptationSession::try_from_spiralk(
                &base,
                &["u2: false;".to_owned(), " u2: false; ".to_owned()],
                SoftBanditMode::UCB,
                1,
            ),
            Err(RankAdaptationError::DuplicateScript { .. })
        ));
        assert!(matches!(
            RankAdaptationSession::try_from_spiralk(
                &base,
                &[
                    format!("u2: {};", base.choice.use_2ce),
                    format!("wg: {};", base.choice.wg),
                ],
                SoftBanditMode::UCB,
                1,
            ),
            Err(RankAdaptationError::DuplicatePlan { .. })
        ));
    }
}
