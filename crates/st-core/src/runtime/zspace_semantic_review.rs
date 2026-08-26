// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical contracts for blinded semantic review of held-out generations.
//!
//! Clients own review presentation and durable draft storage. This module owns
//! packet commitments, score coverage, complete-response receipts, unblinding,
//! and aggregate semantics so Python, Rust, and browser clients cannot drift.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use thiserror::Error;

pub const ZSPACE_SEMANTIC_REVIEW_PACKET_SCHEMA: &str =
    "spiraltorch.hf_blinded_semantic_review_packet.v1";
pub const ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA: &str = "spiraltorch.hf_blinded_semantic_review_map.v1";
pub const ZSPACE_SEMANTIC_REVIEW_DRAFT_SCHEMA: &str =
    "spiraltorch.hf_blinded_semantic_review_draft.v1";
pub const ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_semantic_review_packet.v1";
pub const ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_semantic_review_draft.v1";
pub const ZSPACE_SEMANTIC_REVIEW_RESPONSE_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_semantic_review_response.v1";
pub const ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION: &str =
    "spiraltorch.zspace_semantic_review_map_commitment.v1";
pub const ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_semantic_review_unblind.v1";
pub const ZSPACE_SEMANTIC_REVIEW_PACKET_KIND: &str = "spiraltorch.zspace_semantic_review_packet";
pub const ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND: &str = "spiraltorch.zspace_semantic_review_draft";
pub const ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND: &str = "spiraltorch.zspace_semantic_review_unblind";
pub const ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER: &str = "st-core::runtime::zspace_semantic_review";
pub const ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_SEMANTIC_REVIEW_PACKET_STATUS: &str = "ready_for_blinded_review";
pub const ZSPACE_SEMANTIC_REVIEW_MAP_STATUS: &str = "sealed_pending_review";
pub const ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM: u8 = 1;
pub const ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM: u8 = 5;
pub const ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS: usize = 10_000;
pub const ZSPACE_SEMANTIC_REVIEW_MAX_PROMPT_BYTES: usize = 16_384;
pub const ZSPACE_SEMANTIC_REVIEW_MAX_CONTINUATION_BYTES: usize = 65_536;
pub const ZSPACE_SEMANTIC_REVIEW_MAX_INSTRUCTIONS_BYTES: usize = 16_384;
pub const ZSPACE_SEMANTIC_REVIEW_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
pub const ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS: [&str; 3] = ["A", "B", "C"];
pub const ZSPACE_SEMANTIC_REVIEW_SCORE_DIMENSIONS: [&str; 4] = [
    "fluency",
    "prompt_relevance",
    "local_coherence",
    "non_repetition",
];
pub const ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES: [&str; 4] = ["A", "B", "C", "tie"];
pub const ZSPACE_SEMANTIC_REVIEW_PACKET_ID_RULE: &str =
    "sha256 of UTF-8 canonical JSON for the packet without packet_id; object keys sorted lexicographically, arrays preserved, no insignificant whitespace";
pub const ZSPACE_SEMANTIC_REVIEW_MAP_ID_RULE: &str =
    "sha256 of UTF-8 canonical JSON for [map commitment version, map entries sorted by group_id]; object keys sorted lexicographically, no insignificant whitespace";
pub const ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY: &str =
    "the contract verifies packet and pre-review map-content commitments, structural blinding inputs, score bounds, coverage, and deterministic aggregation; it cannot prove that a reviewer remained blind or establish statistical or model superiority";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSemanticReviewCandidate {
    pub candidate_label: String,
    pub continuation: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSemanticReviewGroup {
    pub group_id: String,
    pub prompt: String,
    pub candidates: Vec<ZSpaceSemanticReviewCandidate>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSemanticReviewPacket {
    pub schema: String,
    pub status: String,
    pub protocol_id: String,
    pub prompt_set_id: String,
    pub blinding_key_sha256: String,
    pub blinding_map_id: String,
    pub group_count: usize,
    pub candidate_count: usize,
    pub instructions: String,
    pub rubric: BTreeMap<String, String>,
    pub groups: Vec<ZSpaceSemanticReviewGroup>,
    pub packet_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSemanticReviewCandidateScore {
    pub candidate_label: String,
    pub fluency: u8,
    pub prompt_relevance: u8,
    pub local_coherence: u8,
    pub non_repetition: u8,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSemanticReviewGroupResponse {
    pub group_id: String,
    pub scores: Vec<ZSpaceSemanticReviewCandidateScore>,
    pub preference: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSemanticReviewDraft {
    pub schema: String,
    pub packet_id: String,
    pub reviewer_id: String,
    pub review_session_id: String,
    pub responses: Vec<ZSpaceSemanticReviewGroupResponse>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSemanticReviewMapEntry {
    pub group_id: String,
    pub seed: u64,
    pub prompt_id: String,
    pub candidate_to_arm: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSemanticReviewMap {
    pub schema: String,
    pub status: String,
    pub protocol_id: String,
    pub packet_id: String,
    pub blinding_key_sha256: String,
    pub entries: Vec<ZSpaceSemanticReviewMapEntry>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceSemanticReviewPacketReceipt {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub packet_validated: bool,
    pub status: &'static str,
    pub packet_id: String,
    pub protocol_id: String,
    pub prompt_set_id: String,
    pub blinding_key_sha256: String,
    pub blinding_map_id: String,
    pub group_count: usize,
    pub candidate_count: usize,
    pub candidate_labels: [&'static str; 3],
    pub score_dimensions: [&'static str; 4],
    pub score_minimum: u8,
    pub score_maximum: u8,
    pub preference_values: [&'static str; 4],
    pub packet_id_rule: &'static str,
    pub blinding_map_id_rule: &'static str,
    pub human_review_complete: bool,
    pub unblind_ready: bool,
    pub efficacy_claim_ready: bool,
    pub evidence_boundary: &'static str,
    pub packet: ZSpaceSemanticReviewPacket,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSemanticReviewDraftRequest {
    pub packet: ZSpaceSemanticReviewPacket,
    pub draft: ZSpaceSemanticReviewDraft,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceSemanticReviewDraftReceipt {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub draft_validated: bool,
    pub status: &'static str,
    pub draft_id: String,
    pub response_id: Option<String>,
    pub packet_id: String,
    pub reviewer_id: String,
    pub review_session_id: String,
    pub group_count: usize,
    pub completed_group_count: usize,
    pub remaining_group_count: usize,
    pub scored_candidate_count: usize,
    pub completion_ratio: f64,
    pub missing_group_ids: Vec<String>,
    pub human_review_complete: bool,
    pub unblind_ready: bool,
    pub efficacy_claim_ready: bool,
    pub evidence_boundary: &'static str,
    pub request: ZSpaceSemanticReviewDraftRequest,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceSemanticReviewMeanScores {
    pub fluency: f64,
    pub prompt_relevance: f64,
    pub local_coherence: f64,
    pub non_repetition: f64,
    pub overall: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceSemanticReviewArmAggregate {
    pub arm: String,
    pub candidate_count: usize,
    pub mean_scores: ZSpaceSemanticReviewMeanScores,
    pub preference_win_count: usize,
    pub preference_share_of_groups: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceSemanticReviewSeedAggregate {
    pub seed: u64,
    pub group_count: usize,
    pub tie_preference_count: usize,
    pub arms: Vec<ZSpaceSemanticReviewArmAggregate>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSemanticReviewUnblindRequest {
    pub packet: ZSpaceSemanticReviewPacket,
    pub draft: ZSpaceSemanticReviewDraft,
    pub blinding_map: ZSpaceSemanticReviewMap,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceSemanticReviewUnblindReport {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub unblind_validated: bool,
    pub status: &'static str,
    pub unblind_id: String,
    pub packet_id: String,
    pub response_id: String,
    pub blinding_map_id: String,
    pub protocol_id: String,
    pub reviewer_id: String,
    pub review_session_id: String,
    pub reviewed_group_count: usize,
    pub unblinded_candidate_count: usize,
    pub arm_count: usize,
    pub tie_preference_count: usize,
    pub arms: Vec<ZSpaceSemanticReviewArmAggregate>,
    pub seeds: Vec<ZSpaceSemanticReviewSeedAggregate>,
    pub human_review_complete: bool,
    pub structural_blinding_inputs_validated: bool,
    pub reviewer_blinding_behavior_verified: bool,
    pub efficacy_claim_ready: bool,
    pub evidence_boundary: &'static str,
    pub request: ZSpaceSemanticReviewUnblindRequest,
}

#[derive(Debug, Error, PartialEq)]
pub enum ZSpaceSemanticReviewError {
    #[error("{field} must equal {expected}")]
    ContractValue { field: String, expected: String },
    #[error("{field} must be a lowercase sha256 identity")]
    InvalidIdentity { field: String },
    #[error("{field} must be a lowercase 64-character sha256 digest")]
    InvalidDigest { field: String },
    #[error("semantic review packet must contain at least one group")]
    EmptyPacket,
    #[error("semantic review group count {count} exceeds maximum {maximum}")]
    GroupLimit { count: usize, maximum: usize },
    #[error("{field} count {declared} does not match actual count {actual}")]
    CountMismatch {
        field: String,
        declared: usize,
        actual: usize,
    },
    #[error("{field} must not be empty")]
    EmptyText { field: String },
    #[error("{field} byte count {count} exceeds maximum {maximum}")]
    TextLimit {
        field: String,
        count: usize,
        maximum: usize,
    },
    #[error("duplicate semantic review group {group_id}")]
    DuplicateGroup { group_id: String },
    #[error("group {group_id} must contain exactly candidate labels A, B, and C")]
    CandidateLabels { group_id: String },
    #[error("semantic review packet_id does not match canonical packet content")]
    PacketIdentityMismatch,
    #[error("draft packet_id does not match the reviewed packet")]
    DraftPacketMismatch,
    #[error("draft contains unknown group {group_id}")]
    UnknownGroup { group_id: String },
    #[error("draft contains duplicate response for group {group_id}")]
    DuplicateResponse { group_id: String },
    #[error("group {group_id} candidate {candidate_label} score {field}={score} is outside {minimum}..={maximum}")]
    ScoreRange {
        group_id: String,
        candidate_label: String,
        field: String,
        score: u8,
        minimum: u8,
        maximum: u8,
    },
    #[error("group {group_id} preference must be A, B, C, or tie")]
    Preference { group_id: String },
    #[error("semantic review draft is incomplete; {remaining} groups remain")]
    IncompleteDraft { remaining: usize },
    #[error("blinding map {field} does not match the packet")]
    MapPacketMismatch { field: String },
    #[error("blinding map content does not match the packet's pre-review commitment")]
    MapIdentityMismatch,
    #[error("blinding map contains duplicate entry for group {group_id}")]
    DuplicateMapEntry { group_id: String },
    #[error("blinding map is missing group {group_id}")]
    MissingMapEntry { group_id: String },
    #[error("blinding map contains unknown group {group_id}")]
    UnknownMapGroup { group_id: String },
    #[error("blinding map entry {index} seed {seed} exceeds cross-client maximum {maximum}")]
    SeedLimit {
        index: usize,
        seed: u64,
        maximum: u64,
    },
    #[error("blinding map entry for group {group_id} must map A, B, and C bijectively")]
    MapCandidateLabels { group_id: String },
    #[error("blinding map entry for group {group_id} has invalid arm name {arm}")]
    InvalidArm { group_id: String, arm: String },
    #[error("blinding map arm set changes at group {group_id}")]
    InconsistentArmSet { group_id: String },
    #[error("duplicate blinding map sample for prompt {prompt_id} seed {seed}")]
    DuplicateMapSample { prompt_id: String, seed: u64 },
    #[error("malformed semantic review artifact: {message}")]
    MalformedArtifact { message: String },
}

#[derive(Clone, Debug, Default)]
struct ScoreAccumulator {
    candidate_count: usize,
    fluency: u64,
    prompt_relevance: u64,
    local_coherence: u64,
    non_repetition: u64,
    preference_win_count: usize,
}

pub fn validate_zspace_semantic_review_packet(
    packet: ZSpaceSemanticReviewPacket,
) -> Result<ZSpaceSemanticReviewPacketReceipt, ZSpaceSemanticReviewError> {
    validate_packet(&packet)?;
    Ok(packet_receipt(packet))
}

pub fn validate_zspace_semantic_review_packet_receipt_value(
    report: serde_json::Value,
) -> Result<ZSpaceSemanticReviewPacketReceipt, ZSpaceSemanticReviewError> {
    let packet = report
        .get("packet")
        .cloned()
        .ok_or_else(|| malformed("missing packet"))?;
    let packet = serde_json::from_value(packet).map_err(|error| malformed(error.to_string()))?;
    let canonical = validate_zspace_semantic_review_packet(packet)?;
    require_canonical_report(report, &canonical, "packet receipt")?;
    Ok(canonical)
}

pub fn summarize_zspace_semantic_review_draft(
    packet: ZSpaceSemanticReviewPacket,
    draft: ZSpaceSemanticReviewDraft,
) -> Result<ZSpaceSemanticReviewDraftReceipt, ZSpaceSemanticReviewError> {
    validate_packet(&packet)?;
    let (draft, missing_group_ids) = normalize_and_validate_draft(&packet, draft)?;
    let group_count = packet.groups.len();
    let completed_group_count = draft.responses.len();
    let remaining_group_count = missing_group_ids.len();
    let complete = remaining_group_count == 0;
    let draft_id = semantic_identity(ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION, &draft)?;
    let response_id = complete
        .then(|| semantic_identity(ZSPACE_SEMANTIC_REVIEW_RESPONSE_CONTRACT_VERSION, &draft))
        .transpose()?;
    let packet_id = packet.packet_id.clone();
    let reviewer_id = draft.reviewer_id.clone();
    let review_session_id = draft.review_session_id.clone();
    Ok(ZSpaceSemanticReviewDraftReceipt {
        contract_version: ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION,
        kind: ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND,
        semantic_owner: ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND,
        draft_validated: true,
        status: if complete {
            "ready_for_unblind"
        } else {
            "in_progress"
        },
        draft_id,
        response_id,
        packet_id,
        reviewer_id,
        review_session_id,
        group_count,
        completed_group_count,
        remaining_group_count,
        scored_candidate_count: completed_group_count
            * ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS.len(),
        completion_ratio: completed_group_count as f64 / group_count as f64,
        missing_group_ids,
        human_review_complete: complete,
        unblind_ready: complete,
        efficacy_claim_ready: false,
        evidence_boundary: ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY,
        request: ZSpaceSemanticReviewDraftRequest { packet, draft },
    })
}

pub fn validate_zspace_semantic_review_draft_receipt_value(
    report: serde_json::Value,
) -> Result<ZSpaceSemanticReviewDraftReceipt, ZSpaceSemanticReviewError> {
    let request = report
        .get("request")
        .cloned()
        .ok_or_else(|| malformed("missing request"))?;
    let request: ZSpaceSemanticReviewDraftRequest =
        serde_json::from_value(request).map_err(|error| malformed(error.to_string()))?;
    let canonical = summarize_zspace_semantic_review_draft(request.packet, request.draft)?;
    require_canonical_report(report, &canonical, "draft receipt")?;
    Ok(canonical)
}

pub fn unblind_zspace_semantic_review(
    packet: ZSpaceSemanticReviewPacket,
    draft: ZSpaceSemanticReviewDraft,
    blinding_map: ZSpaceSemanticReviewMap,
) -> Result<ZSpaceSemanticReviewUnblindReport, ZSpaceSemanticReviewError> {
    let draft_receipt = summarize_zspace_semantic_review_draft(packet, draft)?;
    if !draft_receipt.human_review_complete {
        return Err(ZSpaceSemanticReviewError::IncompleteDraft {
            remaining: draft_receipt.remaining_group_count,
        });
    }
    let response_id = draft_receipt
        .response_id
        .clone()
        .expect("complete drafts always carry a response identity");
    let ZSpaceSemanticReviewDraftRequest { packet, draft } = draft_receipt.request;
    let blinding_map = normalize_and_validate_map(&packet, blinding_map)?;
    let blinding_map_id = packet.blinding_map_id.clone();
    let entry_by_group = blinding_map
        .entries
        .iter()
        .map(|entry| (entry.group_id.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    let mut overall = BTreeMap::<String, ScoreAccumulator>::new();
    let mut by_seed = BTreeMap::<u64, BTreeMap<String, ScoreAccumulator>>::new();
    let mut groups_by_seed = BTreeMap::<u64, usize>::new();
    let mut ties_by_seed = BTreeMap::<u64, usize>::new();
    let mut tie_preference_count = 0usize;

    for response in &draft.responses {
        let entry = entry_by_group
            .get(response.group_id.as_str())
            .expect("validated map covers every draft group");
        *groups_by_seed.entry(entry.seed).or_default() += 1;
        for score in &response.scores {
            let arm = entry
                .candidate_to_arm
                .get(&score.candidate_label)
                .expect("validated map covers every candidate label");
            add_score(overall.entry(arm.clone()).or_default(), score);
            add_score(
                by_seed
                    .entry(entry.seed)
                    .or_default()
                    .entry(arm.clone())
                    .or_default(),
                score,
            );
        }
        if response.preference == "tie" {
            tie_preference_count += 1;
            *ties_by_seed.entry(entry.seed).or_default() += 1;
        } else {
            let arm = entry
                .candidate_to_arm
                .get(&response.preference)
                .expect("validated preference labels are covered by the map");
            overall.entry(arm.clone()).or_default().preference_win_count += 1;
            by_seed
                .entry(entry.seed)
                .or_default()
                .entry(arm.clone())
                .or_default()
                .preference_win_count += 1;
        }
    }

    let group_count = packet.groups.len();
    let arms = aggregates(overall, group_count);
    let seeds = by_seed
        .into_iter()
        .map(|(seed, values)| {
            let seed_group_count = groups_by_seed[&seed];
            ZSpaceSemanticReviewSeedAggregate {
                seed,
                group_count: seed_group_count,
                tie_preference_count: ties_by_seed.get(&seed).copied().unwrap_or(0),
                arms: aggregates(values, seed_group_count),
            }
        })
        .collect::<Vec<_>>();
    let unblind_id = semantic_identity(
        ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION,
        &(
            packet.packet_id.as_str(),
            response_id.as_str(),
            blinding_map_id.as_str(),
        ),
    )?;
    let protocol_id = packet.protocol_id.clone();
    let packet_id = packet.packet_id.clone();
    let reviewer_id = draft.reviewer_id.clone();
    let review_session_id = draft.review_session_id.clone();
    Ok(ZSpaceSemanticReviewUnblindReport {
        contract_version: ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION,
        kind: ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND,
        semantic_owner: ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND,
        unblind_validated: true,
        status: "unblinded",
        unblind_id,
        packet_id,
        response_id,
        blinding_map_id,
        protocol_id,
        reviewer_id,
        review_session_id,
        reviewed_group_count: group_count,
        unblinded_candidate_count: group_count * ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS.len(),
        arm_count: arms.len(),
        tie_preference_count,
        arms,
        seeds,
        human_review_complete: true,
        structural_blinding_inputs_validated: true,
        reviewer_blinding_behavior_verified: false,
        efficacy_claim_ready: false,
        evidence_boundary: ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY,
        request: ZSpaceSemanticReviewUnblindRequest {
            packet,
            draft,
            blinding_map,
        },
    })
}

pub fn validate_zspace_semantic_review_unblind_value(
    report: serde_json::Value,
) -> Result<ZSpaceSemanticReviewUnblindReport, ZSpaceSemanticReviewError> {
    let request = report
        .get("request")
        .cloned()
        .ok_or_else(|| malformed("missing request"))?;
    let request: ZSpaceSemanticReviewUnblindRequest =
        serde_json::from_value(request).map_err(|error| malformed(error.to_string()))?;
    let canonical =
        unblind_zspace_semantic_review(request.packet, request.draft, request.blinding_map)?;
    require_canonical_report(report, &canonical, "unblind report")?;
    Ok(canonical)
}

fn packet_receipt(packet: ZSpaceSemanticReviewPacket) -> ZSpaceSemanticReviewPacketReceipt {
    ZSpaceSemanticReviewPacketReceipt {
        contract_version: ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION,
        kind: ZSPACE_SEMANTIC_REVIEW_PACKET_KIND,
        semantic_owner: ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND,
        packet_validated: true,
        status: "ready",
        packet_id: packet.packet_id.clone(),
        protocol_id: packet.protocol_id.clone(),
        prompt_set_id: packet.prompt_set_id.clone(),
        blinding_key_sha256: packet.blinding_key_sha256.clone(),
        blinding_map_id: packet.blinding_map_id.clone(),
        group_count: packet.groups.len(),
        candidate_count: packet.candidate_count,
        candidate_labels: ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS,
        score_dimensions: ZSPACE_SEMANTIC_REVIEW_SCORE_DIMENSIONS,
        score_minimum: ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM,
        score_maximum: ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM,
        preference_values: ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES,
        packet_id_rule: ZSPACE_SEMANTIC_REVIEW_PACKET_ID_RULE,
        blinding_map_id_rule: ZSPACE_SEMANTIC_REVIEW_MAP_ID_RULE,
        human_review_complete: false,
        unblind_ready: false,
        efficacy_claim_ready: false,
        evidence_boundary: ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY,
        packet,
    }
}

fn validate_packet(packet: &ZSpaceSemanticReviewPacket) -> Result<(), ZSpaceSemanticReviewError> {
    require_value(
        "schema",
        &packet.schema,
        ZSPACE_SEMANTIC_REVIEW_PACKET_SCHEMA,
    )?;
    require_value(
        "status",
        &packet.status,
        ZSPACE_SEMANTIC_REVIEW_PACKET_STATUS,
    )?;
    require_sha256_id("protocol_id", &packet.protocol_id)?;
    require_sha256_id("prompt_set_id", &packet.prompt_set_id)?;
    require_digest("blinding_key_sha256", &packet.blinding_key_sha256)?;
    require_sha256_id("blinding_map_id", &packet.blinding_map_id)?;
    require_sha256_id("packet_id", &packet.packet_id)?;
    if packet.groups.is_empty() {
        return Err(ZSpaceSemanticReviewError::EmptyPacket);
    }
    if packet.groups.len() > ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS {
        return Err(ZSpaceSemanticReviewError::GroupLimit {
            count: packet.groups.len(),
            maximum: ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS,
        });
    }
    if packet.group_count != packet.groups.len() {
        return Err(ZSpaceSemanticReviewError::CountMismatch {
            field: "group_count".to_owned(),
            declared: packet.group_count,
            actual: packet.groups.len(),
        });
    }
    let actual_candidate_count = packet
        .groups
        .iter()
        .map(|group| group.candidates.len())
        .sum::<usize>();
    if packet.candidate_count != actual_candidate_count {
        return Err(ZSpaceSemanticReviewError::CountMismatch {
            field: "candidate_count".to_owned(),
            declared: packet.candidate_count,
            actual: actual_candidate_count,
        });
    }
    require_text(
        "instructions",
        &packet.instructions,
        ZSPACE_SEMANTIC_REVIEW_MAX_INSTRUCTIONS_BYTES,
    )?;
    validate_rubric(&packet.rubric)?;
    let mut seen = BTreeSet::new();
    for (index, group) in packet.groups.iter().enumerate() {
        require_sha256_id(&format!("groups[{index}].group_id"), &group.group_id)?;
        if !seen.insert(group.group_id.clone()) {
            return Err(ZSpaceSemanticReviewError::DuplicateGroup {
                group_id: group.group_id.clone(),
            });
        }
        require_text(
            &format!("groups[{index}].prompt"),
            &group.prompt,
            ZSPACE_SEMANTIC_REVIEW_MAX_PROMPT_BYTES,
        )?;
        require_candidate_labels(
            &group.group_id,
            group
                .candidates
                .iter()
                .map(|candidate| candidate.candidate_label.as_str()),
        )?;
        for (candidate_index, candidate) in group.candidates.iter().enumerate() {
            require_text(
                &format!("groups[{index}].candidates[{candidate_index}].continuation"),
                &candidate.continuation,
                ZSPACE_SEMANTIC_REVIEW_MAX_CONTINUATION_BYTES,
            )?;
        }
    }
    if packet.candidate_count != packet.groups.len() * ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS.len()
    {
        return Err(ZSpaceSemanticReviewError::CountMismatch {
            field: "candidate_count".to_owned(),
            declared: packet.candidate_count,
            actual: packet.groups.len() * ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS.len(),
        });
    }
    if packet_identity(packet)? != packet.packet_id {
        return Err(ZSpaceSemanticReviewError::PacketIdentityMismatch);
    }
    Ok(())
}

fn normalize_and_validate_draft(
    packet: &ZSpaceSemanticReviewPacket,
    mut draft: ZSpaceSemanticReviewDraft,
) -> Result<(ZSpaceSemanticReviewDraft, Vec<String>), ZSpaceSemanticReviewError> {
    require_value(
        "draft.schema",
        &draft.schema,
        ZSPACE_SEMANTIC_REVIEW_DRAFT_SCHEMA,
    )?;
    if draft.packet_id != packet.packet_id {
        return Err(ZSpaceSemanticReviewError::DraftPacketMismatch);
    }
    require_sha256_id("draft.reviewer_id", &draft.reviewer_id)?;
    require_sha256_id("draft.review_session_id", &draft.review_session_id)?;
    let positions = packet
        .groups
        .iter()
        .enumerate()
        .map(|(index, group)| (group.group_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    for response in &mut draft.responses {
        if !positions.contains_key(response.group_id.as_str()) {
            return Err(ZSpaceSemanticReviewError::UnknownGroup {
                group_id: response.group_id.clone(),
            });
        }
        if !seen.insert(response.group_id.clone()) {
            return Err(ZSpaceSemanticReviewError::DuplicateResponse {
                group_id: response.group_id.clone(),
            });
        }
        require_candidate_labels(
            &response.group_id,
            response
                .scores
                .iter()
                .map(|score| score.candidate_label.as_str()),
        )?;
        response
            .scores
            .sort_by(|left, right| left.candidate_label.cmp(&right.candidate_label));
        for score in &response.scores {
            for (field, value) in [
                ("fluency", score.fluency),
                ("prompt_relevance", score.prompt_relevance),
                ("local_coherence", score.local_coherence),
                ("non_repetition", score.non_repetition),
            ] {
                if !(ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM..=ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM)
                    .contains(&value)
                {
                    return Err(ZSpaceSemanticReviewError::ScoreRange {
                        group_id: response.group_id.clone(),
                        candidate_label: score.candidate_label.clone(),
                        field: field.to_owned(),
                        score: value,
                        minimum: ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM,
                        maximum: ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM,
                    });
                }
            }
        }
        if !ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES.contains(&response.preference.as_str()) {
            return Err(ZSpaceSemanticReviewError::Preference {
                group_id: response.group_id.clone(),
            });
        }
    }
    draft.responses.sort_by_key(|response| {
        positions
            .get(response.group_id.as_str())
            .copied()
            .expect("draft groups were validated")
    });
    let missing = packet
        .groups
        .iter()
        .filter(|group| !seen.contains(&group.group_id))
        .map(|group| group.group_id.clone())
        .collect();
    Ok((draft, missing))
}

pub fn zspace_semantic_review_map_id(
    entries: Vec<ZSpaceSemanticReviewMapEntry>,
) -> Result<String, ZSpaceSemanticReviewError> {
    let entries = normalize_and_validate_map_entries(entries)?;
    semantic_identity(ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION, &entries)
}

fn normalize_and_validate_map_entries(
    mut entries: Vec<ZSpaceSemanticReviewMapEntry>,
) -> Result<Vec<ZSpaceSemanticReviewMapEntry>, ZSpaceSemanticReviewError> {
    let mut seen_groups = BTreeSet::new();
    let mut seen_samples = BTreeSet::new();
    let mut expected_arms = None::<BTreeSet<String>>;
    for (index, entry) in entries.iter().enumerate() {
        require_sha256_id(
            &format!("blinding_map.entries[{index}].group_id"),
            &entry.group_id,
        )?;
        if !seen_groups.insert(entry.group_id.clone()) {
            return Err(ZSpaceSemanticReviewError::DuplicateMapEntry {
                group_id: entry.group_id.clone(),
            });
        }
        require_sha256_id(
            &format!("blinding_map.entries[{index}].prompt_id"),
            &entry.prompt_id,
        )?;
        if entry.seed > ZSPACE_SEMANTIC_REVIEW_MAX_SAFE_INTEGER {
            return Err(ZSpaceSemanticReviewError::SeedLimit {
                index,
                seed: entry.seed,
                maximum: ZSPACE_SEMANTIC_REVIEW_MAX_SAFE_INTEGER,
            });
        }
        if !seen_samples.insert((entry.prompt_id.clone(), entry.seed)) {
            return Err(ZSpaceSemanticReviewError::DuplicateMapSample {
                prompt_id: entry.prompt_id.clone(),
                seed: entry.seed,
            });
        }
        require_candidate_labels(
            &entry.group_id,
            entry.candidate_to_arm.keys().map(String::as_str),
        )
        .map_err(|_| ZSpaceSemanticReviewError::MapCandidateLabels {
            group_id: entry.group_id.clone(),
        })?;
        let arms = entry
            .candidate_to_arm
            .values()
            .cloned()
            .collect::<BTreeSet<_>>();
        if arms.len() != ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS.len() {
            return Err(ZSpaceSemanticReviewError::MapCandidateLabels {
                group_id: entry.group_id.clone(),
            });
        }
        for arm in &arms {
            if !valid_arm_name(arm) {
                return Err(ZSpaceSemanticReviewError::InvalidArm {
                    group_id: entry.group_id.clone(),
                    arm: arm.clone(),
                });
            }
        }
        if let Some(expected) = &expected_arms {
            if expected != &arms {
                return Err(ZSpaceSemanticReviewError::InconsistentArmSet {
                    group_id: entry.group_id.clone(),
                });
            }
        } else {
            expected_arms = Some(arms);
        }
    }
    entries.sort_by(|left, right| left.group_id.cmp(&right.group_id));
    Ok(entries)
}

fn normalize_and_validate_map(
    packet: &ZSpaceSemanticReviewPacket,
    mut blinding_map: ZSpaceSemanticReviewMap,
) -> Result<ZSpaceSemanticReviewMap, ZSpaceSemanticReviewError> {
    require_value(
        "blinding_map.schema",
        &blinding_map.schema,
        ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA,
    )?;
    require_value(
        "blinding_map.status",
        &blinding_map.status,
        ZSPACE_SEMANTIC_REVIEW_MAP_STATUS,
    )?;
    for (field, matches) in [
        (
            "protocol_id",
            blinding_map.protocol_id == packet.protocol_id,
        ),
        ("packet_id", blinding_map.packet_id == packet.packet_id),
        (
            "blinding_key_sha256",
            blinding_map.blinding_key_sha256 == packet.blinding_key_sha256,
        ),
    ] {
        if !matches {
            return Err(ZSpaceSemanticReviewError::MapPacketMismatch {
                field: field.to_owned(),
            });
        }
    }
    if blinding_map.entries.len() != packet.groups.len() {
        return Err(ZSpaceSemanticReviewError::CountMismatch {
            field: "blinding_map.entries".to_owned(),
            declared: blinding_map.entries.len(),
            actual: packet.groups.len(),
        });
    }
    let positions = packet
        .groups
        .iter()
        .enumerate()
        .map(|(index, group)| (group.group_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut entries = normalize_and_validate_map_entries(blinding_map.entries)?;
    let seen_groups = entries
        .iter()
        .map(|entry| entry.group_id.as_str())
        .collect::<BTreeSet<_>>();
    for entry in &entries {
        if !positions.contains_key(entry.group_id.as_str()) {
            return Err(ZSpaceSemanticReviewError::UnknownMapGroup {
                group_id: entry.group_id.clone(),
            });
        }
    }
    for group in &packet.groups {
        if !seen_groups.contains(group.group_id.as_str()) {
            return Err(ZSpaceSemanticReviewError::MissingMapEntry {
                group_id: group.group_id.clone(),
            });
        }
    }
    let observed_map_id =
        semantic_identity(ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION, &entries)?;
    if observed_map_id != packet.blinding_map_id {
        return Err(ZSpaceSemanticReviewError::MapIdentityMismatch);
    }
    entries.sort_by_key(|entry| {
        positions
            .get(entry.group_id.as_str())
            .copied()
            .expect("map groups were validated")
    });
    blinding_map.entries = entries;
    Ok(blinding_map)
}

fn validate_rubric(rubric: &BTreeMap<String, String>) -> Result<(), ZSpaceSemanticReviewError> {
    let expected = BTreeMap::from([
        ("fluency".to_owned(), "integer 1 through 5".to_owned()),
        (
            "local_coherence".to_owned(),
            "integer 1 through 5".to_owned(),
        ),
        (
            "non_repetition".to_owned(),
            "integer 1 through 5".to_owned(),
        ),
        ("preference".to_owned(), "A, B, C, or tie".to_owned()),
        (
            "prompt_relevance".to_owned(),
            "integer 1 through 5".to_owned(),
        ),
    ]);
    if rubric == &expected {
        Ok(())
    } else {
        Err(ZSpaceSemanticReviewError::ContractValue {
            field: "rubric".to_owned(),
            expected: "the v1 fluency, prompt_relevance, local_coherence, non_repetition, and preference rubric".to_owned(),
        })
    }
}

fn require_candidate_labels<'a>(
    group_id: &str,
    labels: impl Iterator<Item = &'a str>,
) -> Result<(), ZSpaceSemanticReviewError> {
    let labels = labels.collect::<Vec<_>>();
    let label_count = labels.len();
    let labels = labels.into_iter().collect::<BTreeSet<_>>();
    let expected = ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS
        .into_iter()
        .collect::<BTreeSet<_>>();
    if labels == expected && label_count == ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS.len() {
        Ok(())
    } else {
        Err(ZSpaceSemanticReviewError::CandidateLabels {
            group_id: group_id.to_owned(),
        })
    }
}

fn add_score(accumulator: &mut ScoreAccumulator, score: &ZSpaceSemanticReviewCandidateScore) {
    accumulator.candidate_count += 1;
    accumulator.fluency += u64::from(score.fluency);
    accumulator.prompt_relevance += u64::from(score.prompt_relevance);
    accumulator.local_coherence += u64::from(score.local_coherence);
    accumulator.non_repetition += u64::from(score.non_repetition);
}

fn aggregates(
    values: BTreeMap<String, ScoreAccumulator>,
    group_count: usize,
) -> Vec<ZSpaceSemanticReviewArmAggregate> {
    values
        .into_iter()
        .map(|(arm, value)| {
            let denominator = value.candidate_count as f64;
            let fluency = value.fluency as f64 / denominator;
            let prompt_relevance = value.prompt_relevance as f64 / denominator;
            let local_coherence = value.local_coherence as f64 / denominator;
            let non_repetition = value.non_repetition as f64 / denominator;
            ZSpaceSemanticReviewArmAggregate {
                arm,
                candidate_count: value.candidate_count,
                mean_scores: ZSpaceSemanticReviewMeanScores {
                    fluency,
                    prompt_relevance,
                    local_coherence,
                    non_repetition,
                    overall: (fluency + prompt_relevance + local_coherence + non_repetition) / 4.0,
                },
                preference_win_count: value.preference_win_count,
                preference_share_of_groups: value.preference_win_count as f64 / group_count as f64,
            }
        })
        .collect()
}

fn packet_identity(
    packet: &ZSpaceSemanticReviewPacket,
) -> Result<String, ZSpaceSemanticReviewError> {
    let mut value = serde_json::to_value(packet).map_err(|error| malformed(error.to_string()))?;
    value
        .as_object_mut()
        .expect("serialized packet is an object")
        .remove("packet_id");
    canonical_value_identity(&value)
}

fn semantic_identity(
    contract_version: &str,
    payload: &impl Serialize,
) -> Result<String, ZSpaceSemanticReviewError> {
    let value = serde_json::to_value((contract_version, payload))
        .map_err(|error| malformed(error.to_string()))?;
    canonical_value_identity(&value)
}

fn canonical_value_identity(
    value: &serde_json::Value,
) -> Result<String, ZSpaceSemanticReviewError> {
    let mut encoded = Vec::new();
    write_canonical_json(value, &mut encoded)?;
    let digest = Sha256::digest(encoded);
    let mut output = String::with_capacity(71);
    output.push_str("sha256:");
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

fn write_canonical_json(
    value: &serde_json::Value,
    output: &mut Vec<u8>,
) -> Result<(), ZSpaceSemanticReviewError> {
    match value {
        serde_json::Value::Null => output.extend_from_slice(b"null"),
        serde_json::Value::Bool(value) => {
            output.extend_from_slice(if *value { b"true" } else { b"false" })
        }
        serde_json::Value::Number(value) => output.extend_from_slice(value.to_string().as_bytes()),
        serde_json::Value::String(value) => {
            serde_json::to_writer(output, value).map_err(|error| malformed(error.to_string()))?
        }
        serde_json::Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(b']');
        }
        serde_json::Value::Object(values) => {
            output.push(b'{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                serde_json::to_writer(&mut *output, key)
                    .map_err(|error| malformed(error.to_string()))?;
                output.push(b':');
                write_canonical_json(&values[key], output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn require_value(
    field: &str,
    actual: &str,
    expected: &str,
) -> Result<(), ZSpaceSemanticReviewError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ZSpaceSemanticReviewError::ContractValue {
            field: field.to_owned(),
            expected: expected.to_owned(),
        })
    }
}

fn require_sha256_id(field: &str, value: &str) -> Result<(), ZSpaceSemanticReviewError> {
    if value.strip_prefix("sha256:").is_some_and(valid_digest) {
        Ok(())
    } else {
        Err(ZSpaceSemanticReviewError::InvalidIdentity {
            field: field.to_owned(),
        })
    }
}

fn require_digest(field: &str, value: &str) -> Result<(), ZSpaceSemanticReviewError> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(ZSpaceSemanticReviewError::InvalidDigest {
            field: field.to_owned(),
        })
    }
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(field: &str, value: &str, maximum: usize) -> Result<(), ZSpaceSemanticReviewError> {
    if value.trim().is_empty() {
        return Err(ZSpaceSemanticReviewError::EmptyText {
            field: field.to_owned(),
        });
    }
    if value.len() > maximum {
        return Err(ZSpaceSemanticReviewError::TextLimit {
            field: field.to_owned(),
            count: value.len(),
            maximum,
        });
    }
    Ok(())
}

fn valid_arm_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-' | b'.')
        })
}

fn malformed(message: impl Into<String>) -> ZSpaceSemanticReviewError {
    ZSpaceSemanticReviewError::MalformedArtifact {
        message: message.into(),
    }
}

fn require_canonical_report(
    actual: serde_json::Value,
    canonical: &impl Serialize,
    label: &str,
) -> Result<(), ZSpaceSemanticReviewError> {
    let canonical =
        serde_json::to_value(canonical).map_err(|error| malformed(error.to_string()))?;
    if actual == canonical {
        Ok(())
    } else {
        Err(malformed(format!(
            "{label} does not match the canonical Rust artifact"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn identity(character: char) -> String {
        format!("sha256:{}", character.to_string().repeat(64))
    }

    fn packet() -> ZSpaceSemanticReviewPacket {
        let mut packet = ZSpaceSemanticReviewPacket {
            schema: ZSPACE_SEMANTIC_REVIEW_PACKET_SCHEMA.to_owned(),
            status: ZSPACE_SEMANTIC_REVIEW_PACKET_STATUS.to_owned(),
            protocol_id: identity('a'),
            prompt_set_id: identity('b'),
            blinding_key_sha256: "c".repeat(64),
            blinding_map_id: identity('9'),
            group_count: 2,
            candidate_count: 6,
            instructions: "Score while blind.".to_owned(),
            rubric: BTreeMap::from([
                ("fluency".to_owned(), "integer 1 through 5".to_owned()),
                (
                    "prompt_relevance".to_owned(),
                    "integer 1 through 5".to_owned(),
                ),
                (
                    "local_coherence".to_owned(),
                    "integer 1 through 5".to_owned(),
                ),
                (
                    "non_repetition".to_owned(),
                    "integer 1 through 5".to_owned(),
                ),
                ("preference".to_owned(), "A, B, C, or tie".to_owned()),
            ]),
            groups: vec![group('1', "A prompt"), group('2', "Another prompt")],
            packet_id: identity('0'),
        };
        packet.blinding_map_id = zspace_semantic_review_map_id(blinding_map(&packet).entries)
            .expect("blinding map identity");
        packet.packet_id = packet_identity(&packet).expect("packet identity");
        packet
    }

    fn group(character: char, prompt: &str) -> ZSpaceSemanticReviewGroup {
        ZSpaceSemanticReviewGroup {
            group_id: identity(character),
            prompt: prompt.to_owned(),
            candidates: ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS
                .into_iter()
                .map(|label| ZSpaceSemanticReviewCandidate {
                    candidate_label: label.to_owned(),
                    continuation: format!(" continuation {label}"),
                })
                .collect(),
        }
    }

    fn score(label: &str, value: u8) -> ZSpaceSemanticReviewCandidateScore {
        ZSpaceSemanticReviewCandidateScore {
            candidate_label: label.to_owned(),
            fluency: value,
            prompt_relevance: value,
            local_coherence: value,
            non_repetition: value,
        }
    }

    fn response(
        group_id: String,
        preference: &str,
        values: [u8; 3],
    ) -> ZSpaceSemanticReviewGroupResponse {
        ZSpaceSemanticReviewGroupResponse {
            group_id,
            scores: vec![
                score("C", values[2]),
                score("A", values[0]),
                score("B", values[1]),
            ],
            preference: preference.to_owned(),
        }
    }

    fn draft(packet: &ZSpaceSemanticReviewPacket, complete: bool) -> ZSpaceSemanticReviewDraft {
        let mut responses = vec![response(packet.groups[0].group_id.clone(), "A", [5, 3, 1])];
        if complete {
            responses.push(response(
                packet.groups[1].group_id.clone(),
                "tie",
                [4, 2, 3],
            ));
        }
        ZSpaceSemanticReviewDraft {
            schema: ZSPACE_SEMANTIC_REVIEW_DRAFT_SCHEMA.to_owned(),
            packet_id: packet.packet_id.clone(),
            reviewer_id: identity('d'),
            review_session_id: identity('e'),
            responses,
        }
    }

    fn blinding_map(packet: &ZSpaceSemanticReviewPacket) -> ZSpaceSemanticReviewMap {
        ZSpaceSemanticReviewMap {
            schema: ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA.to_owned(),
            status: ZSPACE_SEMANTIC_REVIEW_MAP_STATUS.to_owned(),
            protocol_id: packet.protocol_id.clone(),
            packet_id: packet.packet_id.clone(),
            blinding_key_sha256: packet.blinding_key_sha256.clone(),
            entries: vec![
                ZSpaceSemanticReviewMapEntry {
                    group_id: packet.groups[1].group_id.clone(),
                    seed: 17,
                    prompt_id: identity('4'),
                    candidate_to_arm: BTreeMap::from([
                        ("A".to_owned(), "periodic".to_owned()),
                        ("B".to_owned(), "baseline".to_owned()),
                        ("C".to_owned(), "history".to_owned()),
                    ]),
                },
                ZSpaceSemanticReviewMapEntry {
                    group_id: packet.groups[0].group_id.clone(),
                    seed: 13,
                    prompt_id: identity('3'),
                    candidate_to_arm: BTreeMap::from([
                        ("A".to_owned(), "baseline".to_owned()),
                        ("B".to_owned(), "history".to_owned()),
                        ("C".to_owned(), "periodic".to_owned()),
                    ]),
                },
            ],
        }
    }

    #[test]
    fn validates_packet_commitment_and_rejects_tampering() {
        let packet = packet();
        let receipt = validate_zspace_semantic_review_packet(packet.clone()).expect("packet");
        assert_eq!(receipt.packet_id, packet.packet_id);
        assert_eq!(receipt.group_count, 2);
        assert!(!receipt.human_review_complete);

        let mut tampered = packet;
        tampered.groups[0].prompt.push('!');
        assert_eq!(
            validate_zspace_semantic_review_packet(tampered),
            Err(ZSpaceSemanticReviewError::PacketIdentityMismatch)
        );
    }

    #[test]
    fn canonical_json_matches_python_packet_rule() {
        let value = json!({"z": "灯", "a": [2, 1]});
        assert_eq!(
            canonical_value_identity(&value).expect("identity"),
            "sha256:122d459644d023581d43e964672b330b068a6115a91f0db2754a713df4bfcabd"
        );
    }

    #[test]
    fn draft_reports_progress_and_only_seals_complete_responses() {
        let packet = packet();
        let partial = summarize_zspace_semantic_review_draft(packet.clone(), draft(&packet, false))
            .expect("partial draft");
        assert_eq!(partial.status, "in_progress");
        assert_eq!(partial.completed_group_count, 1);
        assert_eq!(partial.remaining_group_count, 1);
        assert_eq!(partial.response_id, None);
        assert_eq!(
            partial.missing_group_ids,
            vec![packet.groups[1].group_id.clone()]
        );

        let complete = summarize_zspace_semantic_review_draft(packet.clone(), draft(&packet, true))
            .expect("complete draft");
        assert_eq!(complete.status, "ready_for_unblind");
        assert!(complete.response_id.is_some());
        assert!(complete.human_review_complete);
        assert_eq!(
            complete.request.draft.responses[0].scores[0].candidate_label,
            "A"
        );

        let mut invalid = draft(&packet, false);
        invalid.responses[0].scores[0].fluency = 0;
        assert!(matches!(
            summarize_zspace_semantic_review_draft(packet, invalid),
            Err(ZSpaceSemanticReviewError::ScoreRange { .. })
        ));
    }

    #[test]
    fn unblind_requires_completion_and_aggregates_arms_and_seeds() {
        let packet = packet();
        assert!(matches!(
            unblind_zspace_semantic_review(
                packet.clone(),
                draft(&packet, false),
                blinding_map(&packet),
            ),
            Err(ZSpaceSemanticReviewError::IncompleteDraft { remaining: 1 })
        ));

        let report = unblind_zspace_semantic_review(
            packet.clone(),
            draft(&packet, true),
            blinding_map(&packet),
        )
        .expect("unblind");
        assert_eq!(report.reviewed_group_count, 2);
        assert_eq!(report.unblinded_candidate_count, 6);
        assert_eq!(report.tie_preference_count, 1);
        assert_eq!(report.arms.len(), 3);
        assert_eq!(report.seeds.len(), 2);
        let baseline = report
            .arms
            .iter()
            .find(|row| row.arm == "baseline")
            .expect("baseline");
        assert_eq!(baseline.candidate_count, 2);
        assert_eq!(baseline.mean_scores.fluency, 3.5);
        assert_eq!(baseline.preference_win_count, 1);
        assert_eq!(baseline.preference_share_of_groups, 0.5);
        assert!(!report.reviewer_blinding_behavior_verified);
        assert!(!report.efficacy_claim_ready);

        let value = serde_json::to_value(&report).expect("report value");
        assert_eq!(
            validate_zspace_semantic_review_unblind_value(value).expect("validate"),
            report
        );
    }

    #[test]
    fn rejects_map_arm_drift_and_report_tampering() {
        let packet = packet();
        let mut map = blinding_map(&packet);
        map.entries[1]
            .candidate_to_arm
            .insert("C".to_owned(), "other".to_owned());
        assert!(matches!(
            unblind_zspace_semantic_review(packet.clone(), draft(&packet, true), map),
            Err(ZSpaceSemanticReviewError::InconsistentArmSet { .. })
        ));

        let receipt = summarize_zspace_semantic_review_draft(packet.clone(), draft(&packet, true))
            .expect("draft receipt");
        let mut value = serde_json::to_value(receipt).expect("receipt value");
        value["completed_group_count"] = json!(1);
        assert!(matches!(
            validate_zspace_semantic_review_draft_receipt_value(value),
            Err(ZSpaceSemanticReviewError::MalformedArtifact { .. })
        ));
    }

    #[test]
    fn rejects_a_structurally_valid_post_review_assignment_swap() {
        let packet = packet();
        let mut map = blinding_map(&packet);
        let entry = &mut map.entries[0].candidate_to_arm;
        let arm_a = entry["A"].clone();
        let arm_b = entry["B"].clone();
        entry.insert("A".to_owned(), arm_b);
        entry.insert("B".to_owned(), arm_a);

        assert_eq!(
            unblind_zspace_semantic_review(packet.clone(), draft(&packet, true), map),
            Err(ZSpaceSemanticReviewError::MapIdentityMismatch)
        );
    }

    #[test]
    fn rejects_duplicate_candidate_score_rows() {
        let packet = packet();
        let mut invalid = draft(&packet, false);
        invalid.responses[0].scores.push(score("A", 3));

        assert!(matches!(
            summarize_zspace_semantic_review_draft(packet, invalid),
            Err(ZSpaceSemanticReviewError::CandidateLabels { .. })
        ));
    }
}
