// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical negative-token plans for repetition-aware causal-LM training.
//!
//! Rust owns token-history matching, candidate ordering, bounds, and plan
//! identity. Model clients retain the differentiable tensor operation described
//! by [`ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE`].

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use thiserror::Error;

pub const ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_repetition_unlikelihood.v2";
pub const ZSPACE_REPETITION_UNLIKELIHOOD_KIND: &str =
    "spiraltorch.zspace_repetition_unlikelihood_plan";
pub const ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER: &str =
    "st-core::runtime::zspace_repetition_unlikelihood";
pub const ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER: &str = "model-client-autograd";
pub const ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_OWNER: &str = "model-client-no-grad";
pub const ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE: &str =
    "for model_topk_history, the model client submits torch.topk token IDs from detached float logits in returned rank order; Rust commits the received proposal stream but does not claim to reproduce model logits or backend tie behavior";
pub const ZSPACE_REPETITION_UNLIKELIHOOD_CANDIDATE_RULE: &str =
    "prior_continuation matches the preceding n-1 token prefix and ranks prior alternative next tokens by occurrence count descending, most-recent distance ascending, then token ID ascending; model_topk_history preserves received proposal rank and retains non-target proposals occurring in bounded valid-token history, then caps the canonical candidate list";
pub const ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE: &str =
    "position_loss=mean(-log(1-p(candidate))) over Rust-planned candidates; auxiliary_loss=mean(position_loss) over active positions; training_loss=causal_lm_loss+strength*auxiliary_loss; evaluation_loss remains causal_lm_loss";
pub const ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON: f64 = 1.0e-6;
pub const ZSPACE_REPETITION_UNLIKELIHOOD_MAX_STRENGTH: f64 = 10.0;
pub const ZSPACE_REPETITION_UNLIKELIHOOD_MIN_NGRAM_ORDER: usize = 2;
pub const ZSPACE_REPETITION_UNLIKELIHOOD_MAX_NGRAM_ORDER: usize = 8;
pub const ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CONTEXT_WINDOW: usize = 16_384;
pub const ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CANDIDATES_PER_POSITION: usize = 64;
pub const ZSPACE_REPETITION_UNLIKELIHOOD_MAX_PROPOSAL_TOP_K: usize = 64;
pub const ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES: usize = 4_096;
pub const ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOKENS_PER_SEQUENCE: usize = 16_384;
pub const ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS: usize = 1_000_000;
pub const ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceRepetitionUnlikelihoodConfig {
    pub strength: f64,
    pub candidate_source: ZSpaceRepetitionUnlikelihoodCandidateSource,
    pub context_window: usize,
    pub max_candidates_per_position: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ZSpaceRepetitionUnlikelihoodCandidateSource {
    PriorContinuation { ngram_order: usize },
    ModelTopkHistory { proposal_top_k: usize },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceRepetitionUnlikelihoodSequence {
    pub token_ids: Vec<u64>,
    pub token_mask: Vec<bool>,
    pub label_mask: Vec<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_token_ids: Option<Vec<Vec<u64>>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceRepetitionUnlikelihoodRequest {
    pub config: ZSpaceRepetitionUnlikelihoodConfig,
    pub sequences: Vec<ZSpaceRepetitionUnlikelihoodSequence>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceRepetitionUnlikelihoodCandidate {
    pub token_id: u64,
    pub occurrence_count: usize,
    pub most_recent_distance: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal_rank: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceRepetitionUnlikelihoodPosition {
    pub sequence_index: usize,
    pub prediction_index: usize,
    pub target_index: usize,
    pub target_token_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_prefix_token_ids: Option<Vec<u64>>,
    pub candidates: Vec<ZSpaceRepetitionUnlikelihoodCandidate>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceRepetitionUnlikelihoodAggregate {
    pub sequence_count: usize,
    pub total_token_count: usize,
    pub eligible_target_count: usize,
    pub active_position_count: usize,
    pub candidate_count: usize,
    pub excluded_target_match_count: usize,
    pub proposal_count: usize,
    pub excluded_target_proposal_count: usize,
    pub excluded_out_of_history_proposal_count: usize,
    pub truncated_candidate_count: usize,
    pub maximum_candidates_per_active_position: usize,
    pub active_position_ratio: Option<f64>,
    pub mean_candidates_per_active_position: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceRepetitionUnlikelihoodPlan {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub differentiation_owner: &'static str,
    pub proposal_owner: &'static str,
    pub plan_validated: bool,
    pub plan_id: String,
    pub status: &'static str,
    pub request: ZSpaceRepetitionUnlikelihoodRequest,
    pub proposal_rule: &'static str,
    pub candidate_rule: &'static str,
    pub objective_rule: &'static str,
    pub probability_epsilon: f64,
    pub positions: Vec<ZSpaceRepetitionUnlikelihoodPosition>,
    pub aggregate: ZSpaceRepetitionUnlikelihoodAggregate,
    pub efficacy_claim_ready: bool,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Error, PartialEq)]
pub enum ZSpaceRepetitionUnlikelihoodError {
    #[error("repetition-unlikelihood sequences must not be empty")]
    EmptySequences,
    #[error("sequence count {count} exceeds maximum {maximum}")]
    SequenceLimit { count: usize, maximum: usize },
    #[error("sequence {index} must contain at least one token")]
    EmptySequence { index: usize },
    #[error("sequence {index} token count {count} exceeds maximum {maximum}")]
    SequenceTokenLimit {
        index: usize,
        count: usize,
        maximum: usize,
    },
    #[error("total token count {count} exceeds maximum {maximum}")]
    TotalTokenLimit { count: usize, maximum: usize },
    #[error("sequence {index} {mask} length {count} does not match token count {tokens}")]
    MaskLength {
        index: usize,
        mask: &'static str,
        count: usize,
        tokens: usize,
    },
    #[error("sequence {sequence_index} label {token_index} is supervised but masked as invalid")]
    SupervisedInvalidToken {
        sequence_index: usize,
        token_index: usize,
    },
    #[error(
        "sequence {sequence_index} token {token_index} ID {token_id} exceeds maximum {maximum}"
    )]
    TokenIdLimit {
        sequence_index: usize,
        token_index: usize,
        token_id: u64,
        maximum: u64,
    },
    #[error("strength must be finite and in [0, {maximum}], got {value}")]
    InvalidStrength { value: f64, maximum: f64 },
    #[error("ngram_order must be in [{minimum}, {maximum}], got {value}")]
    InvalidNgramOrder {
        value: usize,
        minimum: usize,
        maximum: usize,
    },
    #[error("proposal_top_k must be in [1, {maximum}], got {value}")]
    InvalidProposalTopK { value: usize, maximum: usize },
    #[error("context_window must be in [1, {maximum}], got {value}")]
    InvalidContextWindow { value: usize, maximum: usize },
    #[error("max_candidates_per_position must be in [1, {maximum}], got {value}")]
    InvalidCandidateLimit { value: usize, maximum: usize },
    #[error(
        "sequence {sequence_index} must not provide proposal_token_ids for prior_continuation"
    )]
    UnexpectedProposalRows { sequence_index: usize },
    #[error("sequence {sequence_index} must provide proposal_token_ids for model_topk_history")]
    MissingProposalRows { sequence_index: usize },
    #[error(
        "sequence {sequence_index} proposal row count {count} does not match token count {tokens}"
    )]
    ProposalRowCount {
        sequence_index: usize,
        count: usize,
        tokens: usize,
    },
    #[error(
        "sequence {sequence_index} target {token_index} proposal count {count} does not match required {required}"
    )]
    ProposalCount {
        sequence_index: usize,
        token_index: usize,
        count: usize,
        required: usize,
    },
    #[error(
        "sequence {sequence_index} target {token_index} has proposals outside an eligible supervised position"
    )]
    UnexpectedPositionProposals {
        sequence_index: usize,
        token_index: usize,
    },
    #[error("sequence {sequence_index} target {token_index} repeats proposal token ID {token_id}")]
    DuplicateProposalToken {
        sequence_index: usize,
        token_index: usize,
        token_id: u64,
    },
    #[error(
        "sequence {sequence_index} target {token_index} proposal token ID {token_id} exceeds maximum {maximum}"
    )]
    ProposalTokenIdLimit {
        sequence_index: usize,
        token_index: usize,
        token_id: u64,
        maximum: u64,
    },
    #[error("repetition-unlikelihood token count overflow")]
    TokenCountOverflow,
    #[error("malformed repetition-unlikelihood plan: {message}")]
    MalformedPlan { message: String },
}

pub fn plan_zspace_repetition_unlikelihood(
    request: ZSpaceRepetitionUnlikelihoodRequest,
) -> Result<ZSpaceRepetitionUnlikelihoodPlan, ZSpaceRepetitionUnlikelihoodError> {
    validate_config(&request.config)?;
    validate_sequences(&request.sequences, &request.config.candidate_source)?;

    let mut positions = Vec::new();
    let mut total_token_count = 0usize;
    let mut eligible_target_count = 0usize;
    let mut excluded_target_match_count = 0usize;
    let mut proposal_count = 0usize;
    let mut excluded_target_proposal_count = 0usize;
    let mut excluded_out_of_history_proposal_count = 0usize;
    let mut truncated_candidate_count = 0usize;

    for (sequence_index, sequence) in request.sequences.iter().enumerate() {
        total_token_count += sequence.token_ids.len();
        match &request.config.candidate_source {
            ZSpaceRepetitionUnlikelihoodCandidateSource::PriorContinuation { ngram_order } => {
                let prefix_len = ngram_order - 1;
                for target_index in prefix_len..sequence.token_ids.len() {
                    if !sequence.label_mask[target_index]
                        || !sequence.token_mask[target_index]
                        || !sequence.token_mask[target_index - prefix_len..=target_index]
                            .iter()
                            .all(|valid| *valid)
                    {
                        continue;
                    }
                    eligible_target_count += 1;
                    let prefix_start = target_index - prefix_len;
                    let history_start = target_index.saturating_sub(request.config.context_window);
                    let prefix = &sequence.token_ids[prefix_start..target_index];
                    let target_token_id = sequence.token_ids[target_index];
                    let mut matches: BTreeMap<u64, (usize, usize)> = BTreeMap::new();

                    for candidate_start in history_start..prefix_start {
                        let candidate_target_index = candidate_start + prefix_len;
                        if !sequence.token_mask[candidate_start..=candidate_target_index]
                            .iter()
                            .all(|valid| *valid)
                            || &sequence.token_ids[candidate_start..candidate_target_index]
                                != prefix
                        {
                            continue;
                        }
                        let candidate_token_id = sequence.token_ids[candidate_target_index];
                        if candidate_token_id == target_token_id {
                            excluded_target_match_count += 1;
                            continue;
                        }
                        let entry = matches
                            .entry(candidate_token_id)
                            .or_insert((0, candidate_target_index));
                        entry.0 += 1;
                        entry.1 = entry.1.max(candidate_target_index);
                    }

                    let mut candidates = matches
                        .into_iter()
                        .map(|(token_id, (occurrence_count, most_recent_index))| {
                            ZSpaceRepetitionUnlikelihoodCandidate {
                                token_id,
                                occurrence_count,
                                most_recent_distance: target_index - most_recent_index,
                                proposal_rank: None,
                            }
                        })
                        .collect::<Vec<_>>();
                    candidates.sort_by(|left, right| {
                        right
                            .occurrence_count
                            .cmp(&left.occurrence_count)
                            .then_with(|| {
                                left.most_recent_distance.cmp(&right.most_recent_distance)
                            })
                            .then_with(|| left.token_id.cmp(&right.token_id))
                    });
                    truncate_candidates(
                        &mut candidates,
                        request.config.max_candidates_per_position,
                        &mut truncated_candidate_count,
                    );
                    if candidates.is_empty() {
                        continue;
                    }
                    positions.push(ZSpaceRepetitionUnlikelihoodPosition {
                        sequence_index,
                        prediction_index: target_index - 1,
                        target_index,
                        target_token_id,
                        matched_prefix_token_ids: Some(prefix.to_vec()),
                        candidates,
                    });
                }
            }
            ZSpaceRepetitionUnlikelihoodCandidateSource::ModelTopkHistory { .. } => {
                let proposal_rows = sequence
                    .proposal_token_ids
                    .as_ref()
                    .expect("model proposal rows were validated");
                for (target_index, proposals) in proposal_rows.iter().enumerate().skip(1) {
                    if !eligible_model_proposal_target(sequence, target_index) {
                        continue;
                    }
                    eligible_target_count += 1;
                    let target_token_id = sequence.token_ids[target_index];
                    let history_start = target_index.saturating_sub(request.config.context_window);
                    let mut history: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
                    for history_index in history_start..target_index {
                        if !sequence.token_mask[history_index] {
                            continue;
                        }
                        let token_id = sequence.token_ids[history_index];
                        let entry = history.entry(token_id).or_insert((0, history_index));
                        entry.0 += 1;
                        entry.1 = history_index;
                    }

                    proposal_count += proposals.len();
                    let mut candidates = Vec::new();
                    for (proposal_rank, &token_id) in proposals.iter().enumerate() {
                        if token_id == target_token_id {
                            excluded_target_proposal_count += 1;
                            continue;
                        }
                        let Some(&(occurrence_count, most_recent_index)) = history.get(&token_id)
                        else {
                            excluded_out_of_history_proposal_count += 1;
                            continue;
                        };
                        candidates.push(ZSpaceRepetitionUnlikelihoodCandidate {
                            token_id,
                            occurrence_count,
                            most_recent_distance: target_index - most_recent_index,
                            proposal_rank: Some(proposal_rank),
                        });
                    }
                    truncate_candidates(
                        &mut candidates,
                        request.config.max_candidates_per_position,
                        &mut truncated_candidate_count,
                    );
                    if candidates.is_empty() {
                        continue;
                    }
                    positions.push(ZSpaceRepetitionUnlikelihoodPosition {
                        sequence_index,
                        prediction_index: target_index - 1,
                        target_index,
                        target_token_id,
                        matched_prefix_token_ids: None,
                        candidates,
                    });
                }
            }
        }
    }

    let candidate_count = positions
        .iter()
        .map(|position| position.candidates.len())
        .sum();
    let active_position_count = positions.len();
    let aggregate = ZSpaceRepetitionUnlikelihoodAggregate {
        sequence_count: request.sequences.len(),
        total_token_count,
        eligible_target_count,
        active_position_count,
        candidate_count,
        excluded_target_match_count,
        proposal_count,
        excluded_target_proposal_count,
        excluded_out_of_history_proposal_count,
        truncated_candidate_count,
        maximum_candidates_per_active_position: positions
            .iter()
            .map(|position| position.candidates.len())
            .max()
            .unwrap_or(0),
        active_position_ratio: ratio(active_position_count, eligible_target_count),
        mean_candidates_per_active_position: ratio(candidate_count, active_position_count),
    };
    let plan_id = repetition_unlikelihood_plan_id(&request)?;

    Ok(ZSpaceRepetitionUnlikelihoodPlan {
        contract_version: ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION,
        kind: ZSPACE_REPETITION_UNLIKELIHOOD_KIND,
        semantic_owner: ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND,
        differentiation_owner: ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER,
        proposal_owner: ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_OWNER,
        plan_validated: true,
        plan_id,
        status: "ready",
        request,
        proposal_rule: ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE,
        candidate_rule: ZSPACE_REPETITION_UNLIKELIHOOD_CANDIDATE_RULE,
        objective_rule: ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE,
        probability_epsilon: ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON,
        positions,
        aggregate,
        efficacy_claim_ready: false,
        evidence_boundary: "the plan proves deterministic negative-candidate selection and objective wiring, not reduced generation loops or improved language quality",
    })
}

fn truncate_candidates(
    candidates: &mut Vec<ZSpaceRepetitionUnlikelihoodCandidate>,
    maximum: usize,
    truncated_candidate_count: &mut usize,
) {
    if candidates.len() > maximum {
        *truncated_candidate_count += candidates.len() - maximum;
        candidates.truncate(maximum);
    }
}

fn eligible_model_proposal_target(
    sequence: &ZSpaceRepetitionUnlikelihoodSequence,
    target_index: usize,
) -> bool {
    target_index > 0
        && sequence.label_mask[target_index]
        && sequence.token_mask[target_index]
        && sequence.token_mask[target_index - 1]
}

pub fn validate_zspace_repetition_unlikelihood_value(
    report: serde_json::Value,
) -> Result<ZSpaceRepetitionUnlikelihoodPlan, ZSpaceRepetitionUnlikelihoodError> {
    let request_value = report.get("request").cloned().ok_or_else(|| {
        ZSpaceRepetitionUnlikelihoodError::MalformedPlan {
            message: "missing request".to_owned(),
        }
    })?;
    let request = serde_json::from_value(request_value).map_err(|error| {
        ZSpaceRepetitionUnlikelihoodError::MalformedPlan {
            message: error.to_string(),
        }
    })?;
    let canonical = plan_zspace_repetition_unlikelihood(request)?;
    let canonical_value = serde_json::to_value(&canonical).map_err(|error| {
        ZSpaceRepetitionUnlikelihoodError::MalformedPlan {
            message: error.to_string(),
        }
    })?;
    if report != canonical_value {
        return Err(ZSpaceRepetitionUnlikelihoodError::MalformedPlan {
            message: "report does not match the canonical Rust plan".to_owned(),
        });
    }
    Ok(canonical)
}

fn validate_config(
    config: &ZSpaceRepetitionUnlikelihoodConfig,
) -> Result<(), ZSpaceRepetitionUnlikelihoodError> {
    if !config.strength.is_finite()
        || !(0.0..=ZSPACE_REPETITION_UNLIKELIHOOD_MAX_STRENGTH).contains(&config.strength)
    {
        return Err(ZSpaceRepetitionUnlikelihoodError::InvalidStrength {
            value: config.strength,
            maximum: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_STRENGTH,
        });
    }
    if config.context_window == 0
        || config.context_window > ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CONTEXT_WINDOW
    {
        return Err(ZSpaceRepetitionUnlikelihoodError::InvalidContextWindow {
            value: config.context_window,
            maximum: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CONTEXT_WINDOW,
        });
    }
    match config.candidate_source {
        ZSpaceRepetitionUnlikelihoodCandidateSource::PriorContinuation { ngram_order } => {
            if !(ZSPACE_REPETITION_UNLIKELIHOOD_MIN_NGRAM_ORDER
                ..=ZSPACE_REPETITION_UNLIKELIHOOD_MAX_NGRAM_ORDER)
                .contains(&ngram_order)
            {
                return Err(ZSpaceRepetitionUnlikelihoodError::InvalidNgramOrder {
                    value: ngram_order,
                    minimum: ZSPACE_REPETITION_UNLIKELIHOOD_MIN_NGRAM_ORDER,
                    maximum: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_NGRAM_ORDER,
                });
            }
            if config.context_window < ngram_order {
                return Err(ZSpaceRepetitionUnlikelihoodError::InvalidContextWindow {
                    value: config.context_window,
                    maximum: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CONTEXT_WINDOW,
                });
            }
        }
        ZSpaceRepetitionUnlikelihoodCandidateSource::ModelTopkHistory { proposal_top_k } => {
            if !(1..=ZSPACE_REPETITION_UNLIKELIHOOD_MAX_PROPOSAL_TOP_K).contains(&proposal_top_k) {
                return Err(ZSpaceRepetitionUnlikelihoodError::InvalidProposalTopK {
                    value: proposal_top_k,
                    maximum: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_PROPOSAL_TOP_K,
                });
            }
        }
    }
    if !(1..=ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CANDIDATES_PER_POSITION)
        .contains(&config.max_candidates_per_position)
    {
        return Err(ZSpaceRepetitionUnlikelihoodError::InvalidCandidateLimit {
            value: config.max_candidates_per_position,
            maximum: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CANDIDATES_PER_POSITION,
        });
    }
    Ok(())
}

fn validate_sequences(
    sequences: &[ZSpaceRepetitionUnlikelihoodSequence],
    candidate_source: &ZSpaceRepetitionUnlikelihoodCandidateSource,
) -> Result<(), ZSpaceRepetitionUnlikelihoodError> {
    if sequences.is_empty() {
        return Err(ZSpaceRepetitionUnlikelihoodError::EmptySequences);
    }
    if sequences.len() > ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES {
        return Err(ZSpaceRepetitionUnlikelihoodError::SequenceLimit {
            count: sequences.len(),
            maximum: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES,
        });
    }
    let mut total_token_count = 0usize;
    for (sequence_index, sequence) in sequences.iter().enumerate() {
        let token_count = sequence.token_ids.len();
        if token_count == 0 {
            return Err(ZSpaceRepetitionUnlikelihoodError::EmptySequence {
                index: sequence_index,
            });
        }
        if token_count > ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOKENS_PER_SEQUENCE {
            return Err(ZSpaceRepetitionUnlikelihoodError::SequenceTokenLimit {
                index: sequence_index,
                count: token_count,
                maximum: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOKENS_PER_SEQUENCE,
            });
        }
        for (mask, count) in [
            ("token_mask", sequence.token_mask.len()),
            ("label_mask", sequence.label_mask.len()),
        ] {
            if count != token_count {
                return Err(ZSpaceRepetitionUnlikelihoodError::MaskLength {
                    index: sequence_index,
                    mask,
                    count,
                    tokens: token_count,
                });
            }
        }
        total_token_count = total_token_count
            .checked_add(token_count)
            .ok_or(ZSpaceRepetitionUnlikelihoodError::TokenCountOverflow)?;
        if total_token_count > ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS {
            return Err(ZSpaceRepetitionUnlikelihoodError::TotalTokenLimit {
                count: total_token_count,
                maximum: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS,
            });
        }
        for (token_index, &token_id) in sequence.token_ids.iter().enumerate() {
            if token_id > ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER {
                return Err(ZSpaceRepetitionUnlikelihoodError::TokenIdLimit {
                    sequence_index,
                    token_index,
                    token_id,
                    maximum: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER,
                });
            }
            if sequence.label_mask[token_index] && !sequence.token_mask[token_index] {
                return Err(ZSpaceRepetitionUnlikelihoodError::SupervisedInvalidToken {
                    sequence_index,
                    token_index,
                });
            }
        }
        match candidate_source {
            ZSpaceRepetitionUnlikelihoodCandidateSource::PriorContinuation { .. } => {
                if sequence.proposal_token_ids.is_some() {
                    return Err(ZSpaceRepetitionUnlikelihoodError::UnexpectedProposalRows {
                        sequence_index,
                    });
                }
            }
            ZSpaceRepetitionUnlikelihoodCandidateSource::ModelTopkHistory { proposal_top_k } => {
                validate_model_proposals(sequence_index, sequence, *proposal_top_k)?;
            }
        }
    }
    Ok(())
}

fn validate_model_proposals(
    sequence_index: usize,
    sequence: &ZSpaceRepetitionUnlikelihoodSequence,
    proposal_top_k: usize,
) -> Result<(), ZSpaceRepetitionUnlikelihoodError> {
    let proposal_rows = sequence
        .proposal_token_ids
        .as_ref()
        .ok_or(ZSpaceRepetitionUnlikelihoodError::MissingProposalRows { sequence_index })?;
    if proposal_rows.len() != sequence.token_ids.len() {
        return Err(ZSpaceRepetitionUnlikelihoodError::ProposalRowCount {
            sequence_index,
            count: proposal_rows.len(),
            tokens: sequence.token_ids.len(),
        });
    }
    for (token_index, proposals) in proposal_rows.iter().enumerate() {
        if !eligible_model_proposal_target(sequence, token_index) {
            if !proposals.is_empty() {
                return Err(
                    ZSpaceRepetitionUnlikelihoodError::UnexpectedPositionProposals {
                        sequence_index,
                        token_index,
                    },
                );
            }
            continue;
        }
        if proposals.len() != proposal_top_k {
            return Err(ZSpaceRepetitionUnlikelihoodError::ProposalCount {
                sequence_index,
                token_index,
                count: proposals.len(),
                required: proposal_top_k,
            });
        }
        let mut seen = BTreeSet::new();
        for &token_id in proposals {
            if token_id > ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER {
                return Err(ZSpaceRepetitionUnlikelihoodError::ProposalTokenIdLimit {
                    sequence_index,
                    token_index,
                    token_id,
                    maximum: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER,
                });
            }
            if !seen.insert(token_id) {
                return Err(ZSpaceRepetitionUnlikelihoodError::DuplicateProposalToken {
                    sequence_index,
                    token_index,
                    token_id,
                });
            }
        }
    }
    Ok(())
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn repetition_unlikelihood_plan_id(
    request: &ZSpaceRepetitionUnlikelihoodRequest,
) -> Result<String, ZSpaceRepetitionUnlikelihoodError> {
    let encoded = serde_json::to_vec(&(ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION, request))
        .map_err(|error| ZSpaceRepetitionUnlikelihoodError::MalformedPlan {
        message: error.to_string(),
    })?;
    let digest = Sha256::digest(encoded);
    let mut output = String::with_capacity(71);
    output.push_str("sha256:");
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn prior_config() -> ZSpaceRepetitionUnlikelihoodConfig {
        ZSpaceRepetitionUnlikelihoodConfig {
            strength: 0.1,
            candidate_source: ZSpaceRepetitionUnlikelihoodCandidateSource::PriorContinuation {
                ngram_order: 3,
            },
            context_window: 16,
            max_candidates_per_position: 8,
        }
    }

    fn model_config(proposal_top_k: usize) -> ZSpaceRepetitionUnlikelihoodConfig {
        ZSpaceRepetitionUnlikelihoodConfig {
            strength: 0.1,
            candidate_source: ZSpaceRepetitionUnlikelihoodCandidateSource::ModelTopkHistory {
                proposal_top_k,
            },
            context_window: 16,
            max_candidates_per_position: 8,
        }
    }

    fn sequence(token_ids: &[u64]) -> ZSpaceRepetitionUnlikelihoodSequence {
        ZSpaceRepetitionUnlikelihoodSequence {
            token_ids: token_ids.to_vec(),
            token_mask: vec![true; token_ids.len()],
            label_mask: vec![true; token_ids.len()],
            proposal_token_ids: None,
        }
    }

    fn prior_request(token_ids: &[u64]) -> ZSpaceRepetitionUnlikelihoodRequest {
        ZSpaceRepetitionUnlikelihoodRequest {
            config: prior_config(),
            sequences: vec![sequence(token_ids)],
        }
    }

    fn model_request(
        token_ids: &[u64],
        target_index: usize,
        proposals: &[u64],
    ) -> ZSpaceRepetitionUnlikelihoodRequest {
        let mut value = sequence(token_ids);
        value.label_mask.fill(false);
        value.label_mask[target_index] = true;
        let mut proposal_rows = vec![Vec::new(); token_ids.len()];
        proposal_rows[target_index] = proposals.to_vec();
        value.proposal_token_ids = Some(proposal_rows);
        ZSpaceRepetitionUnlikelihoodRequest {
            config: model_config(proposals.len()),
            sequences: vec![value],
        }
    }

    #[test]
    fn plans_prior_alternative_continuations() {
        let plan =
            plan_zspace_repetition_unlikelihood(prior_request(&[1, 2, 3, 1, 2, 4])).expect("plan");

        assert_eq!(plan.status, "ready");
        assert!(plan.plan_validated);
        assert_eq!(plan.positions.len(), 1);
        let position = &plan.positions[0];
        assert_eq!(position.sequence_index, 0);
        assert_eq!(position.prediction_index, 4);
        assert_eq!(position.target_index, 5);
        assert_eq!(position.target_token_id, 4);
        assert_eq!(position.matched_prefix_token_ids, Some(vec![1, 2]));
        assert_eq!(position.candidates.len(), 1);
        assert_eq!(position.candidates[0].token_id, 3);
        assert_eq!(position.candidates[0].occurrence_count, 1);
        assert_eq!(position.candidates[0].most_recent_distance, 3);
        assert_eq!(position.candidates[0].proposal_rank, None);
        assert_eq!(plan.aggregate.active_position_count, 1);
        assert_eq!(plan.aggregate.candidate_count, 1);
        assert_eq!(plan.aggregate.excluded_target_match_count, 0);
        assert_eq!(plan.aggregate.proposal_count, 0);
        assert!(!plan.efficacy_claim_ready);
    }

    #[test]
    fn excludes_the_supervised_target_from_negative_candidates() {
        let plan =
            plan_zspace_repetition_unlikelihood(prior_request(&[1, 2, 3, 1, 2, 3])).expect("plan");

        assert!(plan.positions.is_empty());
        assert_eq!(plan.aggregate.excluded_target_match_count, 1);
        assert_eq!(plan.aggregate.active_position_count, 0);
        assert_eq!(plan.aggregate.mean_candidates_per_active_position, None);
    }

    #[test]
    fn ranks_candidates_by_frequency_then_recency_and_truncates() {
        let mut value = prior_request(&[1, 2, 7, 1, 2, 8, 1, 2, 7, 1, 2, 9]);
        value.config.max_candidates_per_position = 1;
        let plan = plan_zspace_repetition_unlikelihood(value).expect("plan");
        let position = plan.positions.last().expect("active position");

        assert_eq!(position.target_index, 11);
        assert_eq!(position.candidates.len(), 1);
        assert_eq!(position.candidates[0].token_id, 7);
        assert_eq!(position.candidates[0].occurrence_count, 2);
        assert_eq!(plan.aggregate.truncated_candidate_count, 1);
    }

    #[test]
    fn model_topk_history_retains_ranked_non_target_history_tokens() {
        let plan =
            plan_zspace_repetition_unlikelihood(model_request(&[7, 2, 3, 2, 4], 4, &[2, 3, 4, 9]))
                .expect("plan");

        assert_eq!(plan.positions.len(), 1);
        let position = &plan.positions[0];
        assert_eq!(position.prediction_index, 3);
        assert_eq!(position.target_token_id, 4);
        assert_eq!(position.matched_prefix_token_ids, None);
        assert_eq!(position.candidates.len(), 2);
        assert_eq!(position.candidates[0].token_id, 2);
        assert_eq!(position.candidates[0].occurrence_count, 2);
        assert_eq!(position.candidates[0].most_recent_distance, 1);
        assert_eq!(position.candidates[0].proposal_rank, Some(0));
        assert_eq!(position.candidates[1].token_id, 3);
        assert_eq!(position.candidates[1].proposal_rank, Some(1));
        assert_eq!(plan.aggregate.proposal_count, 4);
        assert_eq!(plan.aggregate.excluded_target_proposal_count, 1);
        assert_eq!(plan.aggregate.excluded_out_of_history_proposal_count, 1);
        assert_eq!(plan.aggregate.active_position_ratio, Some(1.0));
    }

    #[test]
    fn model_topk_history_respects_the_bounded_history() {
        let mut value = model_request(&[7, 2, 3, 4], 3, &[2, 3]);
        value.config.context_window = 1;
        let plan = plan_zspace_repetition_unlikelihood(value).expect("plan");

        assert_eq!(plan.positions.len(), 1);
        assert_eq!(plan.positions[0].candidates.len(), 1);
        assert_eq!(plan.positions[0].candidates[0].token_id, 3);
        assert_eq!(plan.aggregate.excluded_out_of_history_proposal_count, 1);
    }

    #[test]
    fn respects_token_and_label_masks() {
        let mut masked = sequence(&[0, 1, 2, 3, 1, 2, 4]);
        masked.token_mask[0] = false;
        masked.label_mask[0] = false;
        masked.label_mask[6] = false;
        let plan = plan_zspace_repetition_unlikelihood(ZSpaceRepetitionUnlikelihoodRequest {
            config: prior_config(),
            sequences: vec![masked],
        })
        .expect("plan");

        assert!(plan.positions.is_empty());
    }

    #[test]
    fn rejects_supervision_on_an_invalid_token() {
        let mut masked = sequence(&[1, 2, 3]);
        masked.token_mask[1] = false;
        assert_eq!(
            plan_zspace_repetition_unlikelihood(ZSpaceRepetitionUnlikelihoodRequest {
                config: prior_config(),
                sequences: vec![masked],
            })
            .expect_err("invalid supervision"),
            ZSpaceRepetitionUnlikelihoodError::SupervisedInvalidToken {
                sequence_index: 0,
                token_index: 1,
            }
        );
    }

    #[test]
    fn validation_recomputes_and_rejects_tampering() {
        let plan =
            plan_zspace_repetition_unlikelihood(prior_request(&[1, 2, 3, 1, 2, 4])).expect("plan");
        let value = serde_json::to_value(&plan).expect("value");
        assert_eq!(
            validate_zspace_repetition_unlikelihood_value(value.clone()).expect("valid"),
            plan
        );

        let mut tampered = value;
        tampered["aggregate"]["candidate_count"] = json!(99);
        assert!(matches!(
            validate_zspace_repetition_unlikelihood_value(tampered),
            Err(ZSpaceRepetitionUnlikelihoodError::MalformedPlan { .. })
        ));
    }

    #[test]
    fn rejects_invalid_configuration() {
        let mut value = prior_request(&[1, 2, 3]);
        value.config.strength = f64::NAN;
        assert!(matches!(
            plan_zspace_repetition_unlikelihood(value),
            Err(ZSpaceRepetitionUnlikelihoodError::InvalidStrength { .. })
        ));

        let mut value = prior_request(&[1, 2, 3]);
        value.config.candidate_source =
            ZSpaceRepetitionUnlikelihoodCandidateSource::PriorContinuation { ngram_order: 1 };
        assert!(matches!(
            plan_zspace_repetition_unlikelihood(value),
            Err(ZSpaceRepetitionUnlikelihoodError::InvalidNgramOrder { .. })
        ));

        let mut value = model_request(&[1, 2], 1, &[1]);
        value.config.candidate_source =
            ZSpaceRepetitionUnlikelihoodCandidateSource::ModelTopkHistory { proposal_top_k: 0 };
        assert!(matches!(
            plan_zspace_repetition_unlikelihood(value),
            Err(ZSpaceRepetitionUnlikelihoodError::InvalidProposalTopK { .. })
        ));
    }

    #[test]
    fn model_topk_history_rejects_missing_and_duplicate_proposals() {
        let mut missing = sequence(&[1, 2]);
        missing.label_mask = vec![false, true];
        assert_eq!(
            plan_zspace_repetition_unlikelihood(ZSpaceRepetitionUnlikelihoodRequest {
                config: model_config(1),
                sequences: vec![missing],
            })
            .expect_err("missing proposals"),
            ZSpaceRepetitionUnlikelihoodError::MissingProposalRows { sequence_index: 0 }
        );

        let mut duplicate = model_request(&[1, 2, 3], 2, &[1, 1]);
        duplicate.config = model_config(2);
        assert_eq!(
            plan_zspace_repetition_unlikelihood(duplicate).expect_err("duplicate proposal"),
            ZSpaceRepetitionUnlikelihoodError::DuplicateProposalToken {
                sequence_index: 0,
                token_index: 2,
                token_id: 1,
            }
        );
    }
}
