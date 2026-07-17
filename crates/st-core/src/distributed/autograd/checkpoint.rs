// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Portable, integrity-bound state and replay evidence for [`AmebaAutograd`].

use std::collections::VecDeque;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    AgentConfig, AmebaAutograd, AutogradError, CommittedRound, ExecutionOutcome, GradientMessage,
};
use crate::causal::NodeId;

pub const AMEBA_AUTOGRAD_CHECKPOINT_KIND: &str = "spiraltorch.ameba_autograd_checkpoint";
pub const AMEBA_AUTOGRAD_CHECKPOINT_CONTRACT_VERSION: &str =
    "spiraltorch.ameba_autograd_checkpoint.v1";
pub const AMEBA_AUTOGRAD_CHECKPOINT_SEMANTIC_OWNER: &str =
    "st-core::distributed::autograd::checkpoint";
pub const AMEBA_AUTOGRAD_CHECKPOINT_SEMANTIC_BACKEND: &str = "rust";
pub const AMEBA_AUTOGRAD_CHECKPOINT_RESUME_SCOPE: &str =
    "agent_topology_weights_queue_order_max_hops_and_tolerance";

pub const AMEBA_AUTOGRAD_REPLAY_KIND: &str = "spiraltorch.ameba_autograd_replay_receipt";
pub const AMEBA_AUTOGRAD_REPLAY_CONTRACT_VERSION: &str =
    "spiraltorch.ameba_autograd_replay_receipt.v1";
pub const AMEBA_AUTOGRAD_REPLAY_SEMANTIC_OWNER: &str = "st-core::distributed::autograd::checkpoint";
pub const AMEBA_AUTOGRAD_REPLAY_SEMANTIC_BACKEND: &str = "rust";

const STATE_DIGEST_DOMAIN: &[u8] = b"spiraltorch.ameba_autograd.state.v1\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"spiraltorch.ameba_autograd.receipt.v1\0";

mod canonical_u64 {
    use serde::{de::Error as _, Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        super::parse_canonical_u64(&encoded).map_err(D::Error::custom)
    }
}

/// Versioned, integrity-bound snapshot of the complete Ameba runtime state.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AmebaAutogradCheckpoint {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub resume_scope: String,
    pub state_sha256: String,
    pub state: AmebaAutogradCheckpointState,
}

/// Canonical state payload. Agents are sorted numerically; queue order is semantic.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AmebaAutogradCheckpointState {
    #[serde(with = "canonical_u64")]
    pub max_hops: u64,
    pub tolerance: f32,
    pub agents: Vec<AmebaAutogradAgentCheckpoint>,
    pub pending: Vec<AmebaAutogradMessageCheckpoint>,
}

/// One agent in a portable checkpoint.
///
/// Node IDs use canonical decimal strings so every `u64` remains exact through
/// JSON, Python, JavaScript, and WASM transports.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AmebaAutogradAgentCheckpoint {
    pub id: String,
    pub neighbors: Vec<String>,
    pub learning_rate: f32,
    pub damping: f32,
    pub weights: Vec<f32>,
}

/// One queued gradient message in exact FIFO order.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AmebaAutogradMessageCheckpoint {
    pub from: String,
    pub to: String,
    #[serde(with = "canonical_u64")]
    pub hops: u64,
    pub payload: Vec<f32>,
}

/// Validation receipt returned by checkpoint preflight and native restore.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AmebaAutogradCheckpointValidation {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub resume_scope: &'static str,
    pub state_sha256: String,
    pub agent_count: usize,
    pub edge_count: usize,
    pub pending_messages: usize,
    pub pending_values: usize,
    #[serde(serialize_with = "canonical_u64::serialize")]
    pub max_hops: u64,
    pub tolerance: f32,
    pub canonical_agent_order: bool,
    pub queue_order_preserved: bool,
    pub payload_complete: bool,
    pub deterministic_resume_ready: bool,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AmebaAutogradCheckpointError {
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error("Ameba checkpoint {field} must be {expected}, got {actual}")]
    InvalidMetadata {
        field: &'static str,
        expected: &'static str,
        actual: String,
    },
    #[error("Ameba checkpoint {field} must be a 64-digit lowercase SHA-256")]
    InvalidDigest { field: &'static str },
    #[error("Ameba checkpoint state integrity mismatch: expected {expected}, got {actual}")]
    IntegrityMismatch { expected: String, actual: String },
    #[error("invalid Ameba checkpoint state at {field}: {message}")]
    InvalidState { field: String, message: String },
}

/// Transaction whose exact state transition is bound by a replay receipt.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AmebaAutogradReplayOperation {
    PropagateRound,
    Drain,
}

/// Deterministic summary of one committed message-passing round.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AmebaAutogradRoundReceipt {
    #[serde(with = "canonical_u64")]
    pub round_index: u64,
    #[serde(with = "canonical_u64")]
    pub pending_before: u64,
    #[serde(with = "canonical_u64")]
    pub pending_after: u64,
    #[serde(with = "canonical_u64")]
    pub pending_values_before: u64,
    #[serde(with = "canonical_u64")]
    pub pending_values_after: u64,
    #[serde(with = "canonical_u64")]
    pub processed: u64,
    #[serde(with = "canonical_u64")]
    pub forwarded: u64,
    #[serde(with = "canonical_u64")]
    pub absorbed_tolerance: u64,
    #[serde(with = "canonical_u64")]
    pub stopped_max_hops: u64,
    #[serde(with = "canonical_u64")]
    pub terminal_no_neighbor: u64,
    #[serde(with = "canonical_u64")]
    pub unknown_targets: u64,
    pub signal_sum: f64,
    pub forwarded_signal_sum: f64,
    #[serde(with = "canonical_u64")]
    pub forwarded_values: u64,
    pub weight_update_requested_backend: String,
    pub weight_update_selected_backend: String,
    #[serde(with = "canonical_u64")]
    pub weight_update_operations: u64,
    #[serde(with = "canonical_u64")]
    pub weight_update_values: u64,
    pub forwarding_requested_backend: String,
    pub forwarding_selected_backend: String,
    #[serde(with = "canonical_u64")]
    pub forwarding_operations: u64,
    #[serde(with = "canonical_u64")]
    pub forwarding_values: u64,
}

/// Integrity-bound evidence for one successful round or full drain transaction.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AmebaAutogradReplayReceipt {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub operation: AmebaAutogradReplayOperation,
    pub before_state_sha256: String,
    pub after_state_sha256: String,
    #[serde(with = "canonical_u64")]
    pub total_processed: u64,
    pub rounds: Vec<AmebaAutogradRoundReceipt>,
    pub receipt_sha256: String,
}

/// Result of receipt preflight or a successful native replay.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AmebaAutogradReplayValidation {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub operation: AmebaAutogradReplayOperation,
    pub before_state_sha256: String,
    pub after_state_sha256: String,
    pub receipt_sha256: String,
    #[serde(serialize_with = "canonical_u64::serialize")]
    pub total_processed: u64,
    pub round_count: usize,
    pub receipt_valid: bool,
    pub replay_matched: bool,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AmebaAutogradReplayError {
    #[error(transparent)]
    Checkpoint(#[from] AmebaAutogradCheckpointError),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error("Ameba replay receipt {field} must be {expected}, got {actual}")]
    InvalidMetadata {
        field: &'static str,
        expected: &'static str,
        actual: String,
    },
    #[error("Ameba replay receipt {field} must be a 64-digit lowercase SHA-256")]
    InvalidDigest { field: &'static str },
    #[error("Ameba replay receipt integrity mismatch: expected {expected}, got {actual}")]
    IntegrityMismatch { expected: String, actual: String },
    #[error("invalid Ameba replay receipt at {field}: {message}")]
    InvalidReceipt { field: String, message: String },
    #[error("replay starts from {actual}, but receipt requires {expected}")]
    BeforeStateMismatch { expected: String, actual: String },
    #[error("replay receipt mismatch: expected {expected}, got {actual}")]
    ReplayMismatch { expected: String, actual: String },
}

impl AmebaAutograd {
    /// Captures a canonical checkpoint only after the complete state validates.
    pub fn checkpoint(&self) -> Result<AmebaAutogradCheckpoint, AmebaAutogradCheckpointError> {
        self.validate_topology()?;
        validate_internal_queue(self)?;
        let state = checkpoint_state(self);
        let checkpoint = AmebaAutogradCheckpoint {
            kind: AMEBA_AUTOGRAD_CHECKPOINT_KIND.to_owned(),
            contract_version: AMEBA_AUTOGRAD_CHECKPOINT_CONTRACT_VERSION.to_owned(),
            semantic_owner: AMEBA_AUTOGRAD_CHECKPOINT_SEMANTIC_OWNER.to_owned(),
            semantic_backend: AMEBA_AUTOGRAD_CHECKPOINT_SEMANTIC_BACKEND.to_owned(),
            resume_scope: AMEBA_AUTOGRAD_CHECKPOINT_RESUME_SCOPE.to_owned(),
            state_sha256: state_sha256(self),
            state,
        };
        prepare_checkpoint(&checkpoint)?;
        Ok(checkpoint)
    }

    /// Constructs a new runtime from a fully validated checkpoint.
    pub fn from_checkpoint(
        checkpoint: &AmebaAutogradCheckpoint,
    ) -> Result<Self, AmebaAutogradCheckpointError> {
        prepare_checkpoint(checkpoint).map(|(mesh, _)| mesh)
    }

    /// Restores atomically; failed validation leaves the current runtime untouched.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &AmebaAutogradCheckpoint,
    ) -> Result<AmebaAutogradCheckpointValidation, AmebaAutogradCheckpointError> {
        let (restored, validation) = prepare_checkpoint(checkpoint)?;
        *self = restored;
        Ok(validation)
    }

    /// Commits one round and returns integrity-bound transition evidence.
    pub fn propagate_round_with_receipt(
        &mut self,
    ) -> Result<AmebaAutogradReplayReceipt, AmebaAutogradReplayError> {
        let before = self.checkpoint()?;
        let outcome = self.transact_round()?;
        let after = self.checkpoint()?;
        Ok(build_replay_receipt(
            AmebaAutogradReplayOperation::PropagateRound,
            before.state_sha256,
            after.state_sha256,
            &outcome,
        ))
    }

    /// Commits a complete drain and returns integrity-bound transition evidence.
    pub fn drain_with_receipt(
        &mut self,
    ) -> Result<AmebaAutogradReplayReceipt, AmebaAutogradReplayError> {
        let before = self.checkpoint()?;
        let outcome = self.transact_drain()?;
        let after = self.checkpoint()?;
        Ok(build_replay_receipt(
            AmebaAutogradReplayOperation::Drain,
            before.state_sha256,
            after.state_sha256,
            &outcome,
        ))
    }

    /// Replays a checkpoint under the active Rust execution policy and requires
    /// exact state, round, and selected-route agreement with the receipt.
    pub fn replay_checkpoint(
        checkpoint: &AmebaAutogradCheckpoint,
        receipt: &AmebaAutogradReplayReceipt,
    ) -> Result<(Self, AmebaAutogradReplayValidation), AmebaAutogradReplayError> {
        let (mut replay, _) = prepare_checkpoint(checkpoint)?;
        let mut validation = evaluate_ameba_autograd_replay_receipt(receipt)?;
        if checkpoint.state_sha256 != receipt.before_state_sha256 {
            return Err(AmebaAutogradReplayError::BeforeStateMismatch {
                expected: receipt.before_state_sha256.clone(),
                actual: checkpoint.state_sha256.clone(),
            });
        }

        let observed_before = replay.checkpoint()?;
        let outcome = match receipt.operation {
            AmebaAutogradReplayOperation::PropagateRound => {
                replay.transact_round_with_meta(false)?
            }
            AmebaAutogradReplayOperation::Drain => replay.transact_drain_with_meta(false)?,
        };
        let observed_after = replay.checkpoint()?;
        let observed = build_replay_receipt(
            receipt.operation,
            observed_before.state_sha256,
            observed_after.state_sha256,
            &outcome,
        );
        if observed != *receipt {
            return Err(AmebaAutogradReplayError::ReplayMismatch {
                expected: receipt.receipt_sha256.clone(),
                actual: observed.receipt_sha256,
            });
        }
        validation.replay_matched = true;
        Ok((replay, validation))
    }

    /// Replays into scratch state and commits only after exact receipt agreement.
    pub fn restore_replay_checkpoint(
        &mut self,
        checkpoint: &AmebaAutogradCheckpoint,
        receipt: &AmebaAutogradReplayReceipt,
    ) -> Result<AmebaAutogradReplayValidation, AmebaAutogradReplayError> {
        let (replayed, validation) = Self::replay_checkpoint(checkpoint, receipt)?;
        *self = replayed;
        emit_replay_commit_meta(self, receipt);
        Ok(validation)
    }
}

/// Validates a portable checkpoint without mutating a runtime.
pub fn evaluate_ameba_autograd_checkpoint(
    checkpoint: &AmebaAutogradCheckpoint,
) -> Result<AmebaAutogradCheckpointValidation, AmebaAutogradCheckpointError> {
    prepare_checkpoint(checkpoint).map(|(_, validation)| validation)
}

/// Validates receipt metadata, structure, and integrity without executing it.
pub fn evaluate_ameba_autograd_replay_receipt(
    receipt: &AmebaAutogradReplayReceipt,
) -> Result<AmebaAutogradReplayValidation, AmebaAutogradReplayError> {
    validate_replay_metadata(receipt)?;
    for (field, digest) in [
        ("before_state_sha256", receipt.before_state_sha256.as_str()),
        ("after_state_sha256", receipt.after_state_sha256.as_str()),
        ("receipt_sha256", receipt.receipt_sha256.as_str()),
    ] {
        validate_replay_digest(field, digest)?;
    }
    validate_receipt_structure(receipt)?;
    let actual = receipt_sha256(receipt);
    if receipt.receipt_sha256 != actual {
        return Err(AmebaAutogradReplayError::IntegrityMismatch {
            expected: receipt.receipt_sha256.clone(),
            actual,
        });
    }
    Ok(AmebaAutogradReplayValidation {
        kind: AMEBA_AUTOGRAD_REPLAY_KIND,
        contract_version: AMEBA_AUTOGRAD_REPLAY_CONTRACT_VERSION,
        semantic_owner: AMEBA_AUTOGRAD_REPLAY_SEMANTIC_OWNER,
        semantic_backend: AMEBA_AUTOGRAD_REPLAY_SEMANTIC_BACKEND,
        operation: receipt.operation,
        before_state_sha256: receipt.before_state_sha256.clone(),
        after_state_sha256: receipt.after_state_sha256.clone(),
        receipt_sha256: receipt.receipt_sha256.clone(),
        total_processed: receipt.total_processed,
        round_count: receipt.rounds.len(),
        receipt_valid: true,
        replay_matched: false,
    })
}

fn prepare_checkpoint(
    checkpoint: &AmebaAutogradCheckpoint,
) -> Result<(AmebaAutograd, AmebaAutogradCheckpointValidation), AmebaAutogradCheckpointError> {
    validate_checkpoint_metadata(checkpoint)?;
    validate_checkpoint_digest("state_sha256", &checkpoint.state_sha256)?;

    let max_hops = usize::try_from(checkpoint.state.max_hops).map_err(|_| {
        checkpoint_state_error(
            "state.max_hops",
            "does not fit this target's usize execution width",
        )
    })?;
    let mut mesh = AmebaAutograd::new(max_hops, checkpoint.state.tolerance)?;
    let mut previous_id = None;
    for (index, agent) in checkpoint.state.agents.iter().enumerate() {
        let id = parse_node_id(&format!("state.agents[{index}].id"), &agent.id)?;
        if previous_id.is_some_and(|previous| id <= previous) {
            return Err(checkpoint_state_error(
                &format!("state.agents[{index}].id"),
                "agents must be strictly ordered by numeric NodeId",
            ));
        }
        previous_id = Some(id);
        let neighbors = agent
            .neighbors
            .iter()
            .enumerate()
            .map(|(neighbor_index, neighbor)| {
                parse_node_id(
                    &format!("state.agents[{index}].neighbors[{neighbor_index}]"),
                    neighbor,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        mesh.register_agent(
            AgentConfig::new(id, agent.weights.len())
                .with_neighbors(neighbors)
                .with_learning_rate(agent.learning_rate)
                .with_damping(agent.damping)
                .with_weights(agent.weights.clone()),
        )?;
    }
    mesh.validate_topology()?;

    let mut pending = VecDeque::with_capacity(checkpoint.state.pending.len());
    for (index, message) in checkpoint.state.pending.iter().enumerate() {
        let from = parse_node_id(&format!("state.pending[{index}].from"), &message.from)?;
        let to = parse_node_id(&format!("state.pending[{index}].to"), &message.to)?;
        let hops = usize::try_from(message.hops).map_err(|_| {
            checkpoint_state_error(
                &format!("state.pending[{index}].hops"),
                "does not fit this target's usize execution width",
            )
        })?;
        validate_message(&mesh, index, from, to, hops, &message.payload)?;
        pending.push_back(GradientMessage {
            from,
            to,
            hops,
            payload: message.payload.clone(),
        });
    }
    mesh.pending = pending;

    let actual = state_sha256(&mesh);
    if checkpoint.state_sha256 != actual {
        return Err(AmebaAutogradCheckpointError::IntegrityMismatch {
            expected: checkpoint.state_sha256.clone(),
            actual,
        });
    }

    let edge_count = mesh.agents.values().fold(0usize, |total, agent| {
        total.saturating_add(agent.neighbors.len())
    });
    let pending_values = mesh.pending.iter().fold(0usize, |total, message| {
        total.saturating_add(message.payload.len())
    });
    let validation = AmebaAutogradCheckpointValidation {
        kind: AMEBA_AUTOGRAD_CHECKPOINT_KIND,
        contract_version: AMEBA_AUTOGRAD_CHECKPOINT_CONTRACT_VERSION,
        semantic_owner: AMEBA_AUTOGRAD_CHECKPOINT_SEMANTIC_OWNER,
        semantic_backend: AMEBA_AUTOGRAD_CHECKPOINT_SEMANTIC_BACKEND,
        resume_scope: AMEBA_AUTOGRAD_CHECKPOINT_RESUME_SCOPE,
        state_sha256: checkpoint.state_sha256.clone(),
        agent_count: mesh.agents.len(),
        edge_count,
        pending_messages: mesh.pending.len(),
        pending_values,
        max_hops: checkpoint.state.max_hops,
        tolerance: mesh.tolerance,
        canonical_agent_order: true,
        queue_order_preserved: true,
        payload_complete: true,
        deterministic_resume_ready: true,
    };
    Ok((mesh, validation))
}

fn checkpoint_state(mesh: &AmebaAutograd) -> AmebaAutogradCheckpointState {
    let mut ids = mesh.agents.keys().copied().collect::<Vec<_>>();
    ids.sort_unstable();
    let agents = ids
        .into_iter()
        .map(|id| {
            let agent = &mesh.agents[&id];
            AmebaAutogradAgentCheckpoint {
                id: id.to_string(),
                neighbors: agent.neighbors.iter().map(ToString::to_string).collect(),
                learning_rate: agent.learning_rate,
                damping: agent.damping,
                weights: agent.weights.clone(),
            }
        })
        .collect();
    let pending = mesh
        .pending
        .iter()
        .map(|message| AmebaAutogradMessageCheckpoint {
            from: message.from.to_string(),
            to: message.to.to_string(),
            hops: usize_to_u64(message.hops),
            payload: message.payload.clone(),
        })
        .collect();
    AmebaAutogradCheckpointState {
        max_hops: usize_to_u64(mesh.max_hops),
        tolerance: mesh.tolerance,
        agents,
        pending,
    }
}

fn validate_internal_queue(mesh: &AmebaAutograd) -> Result<(), AmebaAutogradCheckpointError> {
    for (index, message) in mesh.pending.iter().enumerate() {
        validate_message(
            mesh,
            index,
            message.from,
            message.to,
            message.hops,
            &message.payload,
        )?;
    }
    Ok(())
}

fn validate_message(
    mesh: &AmebaAutograd,
    index: usize,
    from: NodeId,
    to: NodeId,
    hops: usize,
    payload: &[f32],
) -> Result<(), AmebaAutogradCheckpointError> {
    let from_agent = mesh.agents.get(&from).ok_or_else(|| {
        checkpoint_state_error(
            &format!("state.pending[{index}].from"),
            "must name a registered agent",
        )
    })?;
    let to_agent = mesh.agents.get(&to).ok_or_else(|| {
        checkpoint_state_error(
            &format!("state.pending[{index}].to"),
            "must name a registered agent",
        )
    })?;
    if hops > mesh.max_hops {
        return Err(checkpoint_state_error(
            &format!("state.pending[{index}].hops"),
            "must not exceed max_hops",
        ));
    }
    if hops == 0 && from != to {
        return Err(checkpoint_state_error(
            &format!("state.pending[{index}]"),
            "a seed message at hop zero must originate at its target",
        ));
    }
    if hops > 0 && !from_agent.neighbors.contains(&to) {
        return Err(checkpoint_state_error(
            &format!("state.pending[{index}]"),
            "a forwarded message must follow an edge from source to target",
        ));
    }
    if payload.len() != to_agent.weights.len() {
        return Err(checkpoint_state_error(
            &format!("state.pending[{index}].payload"),
            "gradient dimension must match the target agent",
        ));
    }
    AmebaAutograd::validate_finite_slice("checkpoint_message_gradient", payload)?;
    Ok(())
}

fn build_replay_receipt(
    operation: AmebaAutogradReplayOperation,
    before_state_sha256: String,
    after_state_sha256: String,
    outcome: &ExecutionOutcome,
) -> AmebaAutogradReplayReceipt {
    let rounds = outcome
        .rounds
        .iter()
        .enumerate()
        .map(|(index, round)| round_receipt(index, round))
        .collect();
    let mut receipt = AmebaAutogradReplayReceipt {
        kind: AMEBA_AUTOGRAD_REPLAY_KIND.to_owned(),
        contract_version: AMEBA_AUTOGRAD_REPLAY_CONTRACT_VERSION.to_owned(),
        semantic_owner: AMEBA_AUTOGRAD_REPLAY_SEMANTIC_OWNER.to_owned(),
        semantic_backend: AMEBA_AUTOGRAD_REPLAY_SEMANTIC_BACKEND.to_owned(),
        operation,
        before_state_sha256,
        after_state_sha256,
        total_processed: usize_to_u64(outcome.total_processed),
        rounds,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_sha256(&receipt);
    receipt
}

fn emit_replay_commit_meta(mesh: &AmebaAutograd, receipt: &AmebaAutogradReplayReceipt) {
    st_tensor::emit_tensor_op(
        "ameba_autograd_replay_commit",
        &[receipt.rounds.len().max(1), mesh.agents.len().max(1)],
        &[mesh.pending.len().max(1), mesh.agents.len().max(1)],
    );
    st_tensor::emit_tensor_op_meta("ameba_autograd_replay_commit", || {
        serde_json::json!({
            "backend": "cpu",
            "requested_backend": "receipt_bound",
            "kind": "st_core_ameba_autograd_replay_commit",
            "semantic_owner": AMEBA_AUTOGRAD_REPLAY_SEMANTIC_OWNER,
            "semantic_backend": AMEBA_AUTOGRAD_REPLAY_SEMANTIC_BACKEND,
            "operation": match receipt.operation {
                AmebaAutogradReplayOperation::PropagateRound => "propagate_round",
                AmebaAutogradReplayOperation::Drain => "drain",
            },
            "commit_mode": "verified_scratch_replace",
            "control_backend": "cpu_hashmap_vecdeque",
            "numeric_backend_semantics": "selected_routes_bound_in_receipt",
            "actual_numeric_backend_source": "add_scaled_and_scale_tensor_op_meta",
            "receipt_sha256": receipt.receipt_sha256,
            "before_state_sha256": receipt.before_state_sha256,
            "after_state_sha256": receipt.after_state_sha256,
            "total_processed": receipt.total_processed,
            "rounds": receipt.rounds.len(),
            "agents": mesh.agents.len(),
            "pending_after": mesh.pending.len(),
            "replay_verified": true,
        })
    });
}

fn round_receipt(index: usize, round: &CommittedRound) -> AmebaAutogradRoundReceipt {
    let stats = round.stats;
    AmebaAutogradRoundReceipt {
        round_index: usize_to_u64(index),
        pending_before: usize_to_u64(stats.pending_before),
        pending_after: usize_to_u64(round.pending_after),
        pending_values_before: usize_to_u64(stats.pending_values_before),
        pending_values_after: usize_to_u64(stats.pending_values_after),
        processed: usize_to_u64(stats.processed),
        forwarded: usize_to_u64(stats.forwarded),
        absorbed_tolerance: usize_to_u64(stats.absorbed_tolerance),
        stopped_max_hops: usize_to_u64(stats.stopped_max_hops),
        terminal_no_neighbor: usize_to_u64(stats.terminal_no_neighbor),
        unknown_targets: usize_to_u64(stats.unknown_targets),
        signal_sum: stats.signal_sum,
        forwarded_signal_sum: stats.forwarded_signal_sum,
        forwarded_values: usize_to_u64(stats.forwarded_values),
        weight_update_requested_backend: stats.weight_update_routes.requested_label().to_owned(),
        weight_update_selected_backend: stats.weight_update_routes.selected_label().to_owned(),
        weight_update_operations: usize_to_u64(stats.weight_update_routes.operations),
        weight_update_values: usize_to_u64(stats.weight_update_routes.values),
        forwarding_requested_backend: stats.forwarding_routes.requested_label().to_owned(),
        forwarding_selected_backend: stats.forwarding_routes.selected_label().to_owned(),
        forwarding_operations: usize_to_u64(stats.forwarding_routes.operations),
        forwarding_values: usize_to_u64(stats.forwarding_routes.values),
    }
}

fn validate_receipt_structure(
    receipt: &AmebaAutogradReplayReceipt,
) -> Result<(), AmebaAutogradReplayError> {
    match receipt.operation {
        AmebaAutogradReplayOperation::PropagateRound if receipt.rounds.len() != 1 => {
            return Err(replay_state_error(
                "rounds",
                "propagate_round receipts must contain exactly one round",
            ));
        }
        AmebaAutogradReplayOperation::Drain
            if receipt
                .rounds
                .last()
                .is_some_and(|round| round.pending_after != 0) =>
        {
            return Err(replay_state_error(
                "rounds",
                "drain receipts must end with an empty queue",
            ));
        }
        _ => {}
    }

    let mut processed = 0u64;
    let mut previous_pending_after = None;
    for (index, round) in receipt.rounds.iter().enumerate() {
        if round.round_index != usize_to_u64(index) {
            return Err(replay_state_error(
                &format!("rounds[{index}].round_index"),
                "round indices must be contiguous from zero",
            ));
        }
        if previous_pending_after.is_some_and(|previous| previous != round.pending_before) {
            return Err(replay_state_error(
                &format!("rounds[{index}].pending_before"),
                "must equal the previous round's pending_after",
            ));
        }
        if !round.signal_sum.is_finite()
            || round.signal_sum < 0.0
            || !round.forwarded_signal_sum.is_finite()
            || round.forwarded_signal_sum < 0.0
        {
            return Err(replay_state_error(
                &format!("rounds[{index}].signal_sum"),
                "signal summaries must be finite and non-negative",
            ));
        }
        validate_route_summary(
            index,
            "weight_update",
            &round.weight_update_requested_backend,
            &round.weight_update_selected_backend,
            round.weight_update_operations,
        )?;
        validate_route_summary(
            index,
            "forwarding",
            &round.forwarding_requested_backend,
            &round.forwarding_selected_backend,
            round.forwarding_operations,
        )?;
        if round.weight_update_operations != round.processed {
            return Err(replay_state_error(
                &format!("rounds[{index}].weight_update_operations"),
                "must equal processed messages",
            ));
        }
        if round.forwarding_operations > round.processed {
            return Err(replay_state_error(
                &format!("rounds[{index}].forwarding_operations"),
                "must not exceed processed messages",
            ));
        }
        processed = processed.checked_add(round.processed).ok_or_else(|| {
            replay_state_error("total_processed", "round message count overflowed u64")
        })?;
        previous_pending_after = Some(round.pending_after);
    }
    if processed != receipt.total_processed {
        return Err(replay_state_error(
            "total_processed",
            "must equal the sum of all round message counts",
        ));
    }
    Ok(())
}

fn validate_route_summary(
    round_index: usize,
    field: &str,
    requested: &str,
    selected: &str,
    operations: u64,
) -> Result<(), AmebaAutogradReplayError> {
    let valid = |value: &str| matches!(value, "none" | "auto" | "cpu" | "wgpu" | "mixed");
    if !valid(requested) || !valid(selected) {
        return Err(replay_state_error(
            &format!("rounds[{round_index}].{field}_backend"),
            "route labels must use the Rust-owned tensor utility vocabulary",
        ));
    }
    if operations == 0 && (requested != "none" || selected != "none") {
        return Err(replay_state_error(
            &format!("rounds[{round_index}].{field}_operations"),
            "a zero-operation route summary must use none/none",
        ));
    }
    if operations > 0 && (requested == "none" || selected == "none") {
        return Err(replay_state_error(
            &format!("rounds[{round_index}].{field}_operations"),
            "a non-empty route summary must name requested and selected routes",
        ));
    }
    Ok(())
}

fn state_sha256(mesh: &AmebaAutograd) -> String {
    let mut hasher = Sha256::new();
    hasher.update(STATE_DIGEST_DOMAIN);
    hash_u64(&mut hasher, usize_to_u64(mesh.max_hops));
    hash_u32(&mut hasher, mesh.tolerance.to_bits());
    let mut ids = mesh.agents.keys().copied().collect::<Vec<_>>();
    ids.sort_unstable();
    hash_u64(&mut hasher, usize_to_u64(ids.len()));
    for id in ids {
        let agent = &mesh.agents[&id];
        hash_u64(&mut hasher, id);
        hash_u64(&mut hasher, usize_to_u64(agent.neighbors.len()));
        for neighbor in &agent.neighbors {
            hash_u64(&mut hasher, *neighbor);
        }
        hash_u32(&mut hasher, agent.learning_rate.to_bits());
        hash_u32(&mut hasher, agent.damping.to_bits());
        hash_f32_slice(&mut hasher, &agent.weights);
    }
    hash_u64(&mut hasher, usize_to_u64(mesh.pending.len()));
    for message in &mesh.pending {
        hash_u64(&mut hasher, message.from);
        hash_u64(&mut hasher, message.to);
        hash_u64(&mut hasher, usize_to_u64(message.hops));
        hash_f32_slice(&mut hasher, &message.payload);
    }
    finish_sha256(hasher)
}

fn receipt_sha256(receipt: &AmebaAutogradReplayReceipt) -> String {
    let mut hasher = Sha256::new();
    hasher.update(RECEIPT_DIGEST_DOMAIN);
    hash_string(&mut hasher, &receipt.kind);
    hash_string(&mut hasher, &receipt.contract_version);
    hash_string(&mut hasher, &receipt.semantic_owner);
    hash_string(&mut hasher, &receipt.semantic_backend);
    hash_u8(
        &mut hasher,
        match receipt.operation {
            AmebaAutogradReplayOperation::PropagateRound => 0,
            AmebaAutogradReplayOperation::Drain => 1,
        },
    );
    hash_string(&mut hasher, &receipt.before_state_sha256);
    hash_string(&mut hasher, &receipt.after_state_sha256);
    hash_u64(&mut hasher, receipt.total_processed);
    hash_u64(&mut hasher, usize_to_u64(receipt.rounds.len()));
    for round in &receipt.rounds {
        for value in [
            round.round_index,
            round.pending_before,
            round.pending_after,
            round.pending_values_before,
            round.pending_values_after,
            round.processed,
            round.forwarded,
            round.absorbed_tolerance,
            round.stopped_max_hops,
            round.terminal_no_neighbor,
            round.unknown_targets,
        ] {
            hash_u64(&mut hasher, value);
        }
        hash_u64(&mut hasher, round.signal_sum.to_bits());
        hash_u64(&mut hasher, round.forwarded_signal_sum.to_bits());
        hash_u64(&mut hasher, round.forwarded_values);
        hash_string(&mut hasher, &round.weight_update_requested_backend);
        hash_string(&mut hasher, &round.weight_update_selected_backend);
        hash_u64(&mut hasher, round.weight_update_operations);
        hash_u64(&mut hasher, round.weight_update_values);
        hash_string(&mut hasher, &round.forwarding_requested_backend);
        hash_string(&mut hasher, &round.forwarding_selected_backend);
        hash_u64(&mut hasher, round.forwarding_operations);
        hash_u64(&mut hasher, round.forwarding_values);
    }
    finish_sha256(hasher)
}

fn hash_f32_slice(hasher: &mut Sha256, values: &[f32]) {
    hash_u64(hasher, usize_to_u64(values.len()));
    for value in values {
        hash_u32(hasher, value.to_bits());
    }
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hash_u64(hasher, usize_to_u64(value.len()));
    hasher.update(value.as_bytes());
}

fn hash_u8(hasher: &mut Sha256, value: u8) {
    hasher.update([value]);
}

fn hash_u32(hasher: &mut Sha256, value: u32) {
    hasher.update(value.to_be_bytes());
}

fn hash_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_be_bytes());
}

fn finish_sha256(hasher: Sha256) -> String {
    let digest = hasher.finalize();
    let mut hexadecimal = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hexadecimal, "{byte:02x}").expect("writing to a String cannot fail");
    }
    hexadecimal
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).expect("supported Rust targets have at most 64-bit usize")
}

fn parse_canonical_u64(value: &str) -> Result<u64, &'static str> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err("must be a canonical unsigned decimal u64");
    }
    value.parse().map_err(|_| "must fit the complete u64 range")
}

fn parse_node_id(field: &str, value: &str) -> Result<NodeId, AmebaAutogradCheckpointError> {
    parse_canonical_u64(value).map_err(|message| checkpoint_state_error(field, message))
}

fn validate_checkpoint_metadata(
    checkpoint: &AmebaAutogradCheckpoint,
) -> Result<(), AmebaAutogradCheckpointError> {
    for (field, actual, expected) in [
        (
            "kind",
            checkpoint.kind.as_str(),
            AMEBA_AUTOGRAD_CHECKPOINT_KIND,
        ),
        (
            "contract_version",
            checkpoint.contract_version.as_str(),
            AMEBA_AUTOGRAD_CHECKPOINT_CONTRACT_VERSION,
        ),
        (
            "semantic_owner",
            checkpoint.semantic_owner.as_str(),
            AMEBA_AUTOGRAD_CHECKPOINT_SEMANTIC_OWNER,
        ),
        (
            "semantic_backend",
            checkpoint.semantic_backend.as_str(),
            AMEBA_AUTOGRAD_CHECKPOINT_SEMANTIC_BACKEND,
        ),
        (
            "resume_scope",
            checkpoint.resume_scope.as_str(),
            AMEBA_AUTOGRAD_CHECKPOINT_RESUME_SCOPE,
        ),
    ] {
        if actual != expected {
            return Err(AmebaAutogradCheckpointError::InvalidMetadata {
                field,
                expected,
                actual: actual.to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_replay_metadata(
    receipt: &AmebaAutogradReplayReceipt,
) -> Result<(), AmebaAutogradReplayError> {
    for (field, actual, expected) in [
        ("kind", receipt.kind.as_str(), AMEBA_AUTOGRAD_REPLAY_KIND),
        (
            "contract_version",
            receipt.contract_version.as_str(),
            AMEBA_AUTOGRAD_REPLAY_CONTRACT_VERSION,
        ),
        (
            "semantic_owner",
            receipt.semantic_owner.as_str(),
            AMEBA_AUTOGRAD_REPLAY_SEMANTIC_OWNER,
        ),
        (
            "semantic_backend",
            receipt.semantic_backend.as_str(),
            AMEBA_AUTOGRAD_REPLAY_SEMANTIC_BACKEND,
        ),
    ] {
        if actual != expected {
            return Err(AmebaAutogradReplayError::InvalidMetadata {
                field,
                expected,
                actual: actual.to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_checkpoint_digest(
    field: &'static str,
    digest: &str,
) -> Result<(), AmebaAutogradCheckpointError> {
    if valid_sha256(digest) {
        Ok(())
    } else {
        Err(AmebaAutogradCheckpointError::InvalidDigest { field })
    }
}

fn validate_replay_digest(
    field: &'static str,
    digest: &str,
) -> Result<(), AmebaAutogradReplayError> {
    if valid_sha256(digest) {
        Ok(())
    } else {
        Err(AmebaAutogradReplayError::InvalidDigest { field })
    }
}

fn valid_sha256(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn checkpoint_state_error(field: &str, message: &str) -> AmebaAutogradCheckpointError {
    AmebaAutogradCheckpointError::InvalidState {
        field: field.to_owned(),
        message: message.to_owned(),
    }
}

fn replay_state_error(field: &str, message: &str) -> AmebaAutogradReplayError {
    AmebaAutogradReplayError::InvalidReceipt {
        field: field.to_owned(),
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::DeviceCaps;
    use crate::backend::execution::{push_backend_policy, BackendPolicy, ExecutionConfig};

    fn mesh_with_pending_cascade() -> AmebaAutograd {
        let mut mesh = AmebaAutograd::new(3, 1e-6).unwrap();
        mesh.register_agent(
            AgentConfig::new(3, 2)
                .with_neighbors(vec![2])
                .with_learning_rate(0.025)
                .with_weights(vec![0.5, -0.5]),
        )
        .unwrap();
        mesh.register_agent(
            AgentConfig::new(1, 2)
                .with_neighbors(vec![2])
                .with_learning_rate(0.1)
                .with_weights(vec![1.0, -1.0]),
        )
        .unwrap();
        mesh.register_agent(
            AgentConfig::new(2, 2)
                .with_neighbors(vec![1, 3])
                .with_learning_rate(0.05)
                .with_damping(0.75),
        )
        .unwrap();
        mesh.seed_gradient(2, vec![1.0, -0.5]).unwrap();
        mesh
    }

    #[test]
    fn checkpoint_is_canonical_portable_and_restores_fifo_state() {
        let mesh = mesh_with_pending_cascade();
        let checkpoint = mesh.checkpoint().unwrap();
        let encoded = serde_json::to_string(&checkpoint).unwrap();
        let wire: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        let decoded: AmebaAutogradCheckpoint = serde_json::from_str(&encoded).unwrap();
        let validation = evaluate_ameba_autograd_checkpoint(&decoded).unwrap();
        let restored = AmebaAutograd::from_checkpoint(&decoded).unwrap();

        assert_eq!(checkpoint, decoded);
        assert_eq!(wire["state"]["max_hops"], serde_json::json!("3"));
        assert_eq!(wire["state"]["pending"][0]["hops"], serde_json::json!("0"));
        assert_eq!(checkpoint.state.agents[0].id, "1");
        assert_eq!(checkpoint.state.agents[1].id, "2");
        assert_eq!(checkpoint.state.agents[2].id, "3");
        assert_eq!(checkpoint.state.pending[0].from, "2");
        assert_eq!(checkpoint.state.pending[0].to, "2");
        assert_eq!(checkpoint.state_sha256.len(), 64);
        assert_eq!(validation.agent_count, 3);
        assert_eq!(validation.edge_count, 4);
        assert!(validation.deterministic_resume_ready);
        assert_eq!(
            serde_json::to_value(&validation).unwrap()["max_hops"],
            serde_json::json!("3")
        );
        assert_eq!(restored.checkpoint().unwrap(), checkpoint);
    }

    #[test]
    fn checkpoint_preserves_the_complete_u64_node_id_range() {
        let mut mesh = AmebaAutograd::new(0, 0.0).unwrap();
        mesh.register_agent(AgentConfig::new(u64::MAX, 1)).unwrap();
        let checkpoint = mesh.checkpoint().unwrap();

        assert_eq!(checkpoint.state.agents[0].id, u64::MAX.to_string());
        assert_eq!(
            AmebaAutograd::from_checkpoint(&checkpoint)
                .unwrap()
                .checkpoint()
                .unwrap(),
            checkpoint
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn checkpoint_wire_hop_fields_survive_the_javascript_integer_boundary() {
        const BEYOND_JS_SAFE_INTEGER: u64 = 9_007_199_254_740_993;

        let mut mesh = mesh_with_pending_cascade();
        mesh.max_hops = usize::try_from(BEYOND_JS_SAFE_INTEGER).unwrap();
        let message = mesh.pending.front_mut().unwrap();
        message.from = 1;
        message.hops = usize::try_from(BEYOND_JS_SAFE_INTEGER).unwrap();
        let checkpoint = mesh.checkpoint().unwrap();
        let wire = serde_json::to_value(&checkpoint).unwrap();

        assert_eq!(
            wire["state"]["max_hops"],
            serde_json::json!(BEYOND_JS_SAFE_INTEGER.to_string())
        );
        assert_eq!(
            wire["state"]["pending"][0]["hops"],
            serde_json::json!(BEYOND_JS_SAFE_INTEGER.to_string())
        );
        let decoded: AmebaAutogradCheckpoint = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(decoded, checkpoint);
        evaluate_ameba_autograd_checkpoint(&decoded).unwrap();

        let mut numeric = wire.clone();
        numeric["state"]["max_hops"] = serde_json::json!(BEYOND_JS_SAFE_INTEGER);
        assert!(serde_json::from_value::<AmebaAutogradCheckpoint>(numeric).is_err());

        let mut noncanonical = wire;
        noncanonical["state"]["pending"][0]["hops"] = serde_json::json!("09007199254740993");
        assert!(serde_json::from_value::<AmebaAutogradCheckpoint>(noncanonical).is_err());
    }

    #[test]
    fn queue_and_neighbor_order_are_bound_by_the_state_digest() {
        let mut first = mesh_with_pending_cascade();
        first.seed_gradient(1, vec![0.25, 0.5]).unwrap();
        let mut second = mesh_with_pending_cascade();
        let original = second.pending.pop_front().unwrap();
        second.seed_gradient(1, vec![0.25, 0.5]).unwrap();
        second.pending.push_back(original);

        assert_ne!(
            first.checkpoint().unwrap().state_sha256,
            second.checkpoint().unwrap().state_sha256
        );

        let original_neighbors = mesh_with_pending_cascade();
        let mut reordered_neighbors = original_neighbors.clone();
        reordered_neighbors
            .agents
            .get_mut(&2)
            .unwrap()
            .neighbors
            .reverse();
        assert_ne!(
            original_neighbors.checkpoint().unwrap().state_sha256,
            reordered_neighbors.checkpoint().unwrap().state_sha256
        );
    }

    #[test]
    fn checkpoint_tampering_is_rejected_without_mutating_the_target() {
        let source = mesh_with_pending_cascade();
        let mut tampered = source.checkpoint().unwrap();
        tampered.state.agents[0].weights[0] += 0.25;

        let mut target = AmebaAutograd::new(0, 0.0).unwrap();
        target.register_agent(AgentConfig::new(99, 1)).unwrap();
        let before = target.checkpoint().unwrap();
        assert!(matches!(
            target.restore_checkpoint(&tampered),
            Err(AmebaAutogradCheckpointError::IntegrityMismatch { .. })
        ));
        assert_eq!(target.checkpoint().unwrap(), before);
    }

    #[test]
    fn checkpoint_rejects_noncanonical_ids_and_malformed_queue_provenance() {
        let source = mesh_with_pending_cascade();
        let mut id = source.checkpoint().unwrap();
        id.state.agents[0].id = "01".to_owned();
        assert!(matches!(
            evaluate_ameba_autograd_checkpoint(&id),
            Err(AmebaAutogradCheckpointError::InvalidState { .. })
        ));

        let mut order = source.checkpoint().unwrap();
        order.state.agents.swap(0, 1);
        assert!(matches!(
            evaluate_ameba_autograd_checkpoint(&order),
            Err(AmebaAutogradCheckpointError::InvalidState { .. })
        ));

        let mut provenance = source.checkpoint().unwrap();
        provenance.state.pending[0].from = "1".to_owned();
        assert!(matches!(
            evaluate_ameba_autograd_checkpoint(&provenance),
            Err(AmebaAutogradCheckpointError::InvalidState { .. })
        ));
    }

    #[test]
    fn drain_receipt_replays_to_the_exact_state_and_route_summary() {
        let mut source = mesh_with_pending_cascade();
        let before = source.checkpoint().unwrap();
        let receipt = source.drain_with_receipt().unwrap();
        let expected = source.checkpoint().unwrap();
        let wire = serde_json::to_value(&receipt).unwrap();
        let decoded: AmebaAutogradReplayReceipt = serde_json::from_value(wire.clone()).unwrap();
        let preflight = evaluate_ameba_autograd_replay_receipt(&receipt).unwrap();
        let (replayed, validation) = AmebaAutograd::replay_checkpoint(&before, &receipt).unwrap();

        assert_eq!(
            before.state_sha256,
            "6d8a33466342a823a634bf21db415084e85003acece518e033fdfa48f99ede27"
        );
        assert_eq!(
            receipt.receipt_sha256,
            "9e095940680f58a66d9781710df87b82f8b03eeffd0c81bc8574e9e94aa5d034"
        );
        assert!(preflight.receipt_valid);
        assert!(!preflight.replay_matched);
        assert_eq!(
            serde_json::to_value(&preflight).unwrap()["total_processed"],
            serde_json::json!("3")
        );
        assert!(validation.replay_matched);
        assert_eq!(receipt.before_state_sha256, before.state_sha256);
        assert_eq!(receipt.after_state_sha256, expected.state_sha256);
        assert_eq!(receipt.total_processed, 3);
        assert_eq!(receipt.rounds.len(), 2);
        assert_eq!(wire["total_processed"], serde_json::json!("3"));
        assert_eq!(wire["rounds"][0]["round_index"], serde_json::json!("0"));
        assert_eq!(wire["rounds"][0]["processed"], serde_json::json!("1"));
        assert_eq!(decoded, receipt);
        assert_eq!(replayed.checkpoint().unwrap(), expected);
    }

    #[test]
    fn restore_replay_emits_only_the_verified_commit_composite() {
        use std::sync::{Arc, Mutex};

        let mut source = mesh_with_pending_cascade();
        let checkpoint = source.checkpoint().unwrap();
        let receipt = source.drain_with_receipt().unwrap();

        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));
        let mut target = AmebaAutograd::new(0, 0.0).unwrap();
        target.register_agent(AgentConfig::new(99, 1)).unwrap();
        let validation = target
            .restore_replay_checkpoint(&checkpoint, &receipt)
            .unwrap();
        st_tensor::set_thread_meta_observer(previous);

        assert!(validation.replay_matched);
        let events = events.lock().unwrap();
        assert!(!events
            .iter()
            .any(|(op_name, _)| *op_name == "ameba_autograd_round"));
        let commit = events
            .iter()
            .find(|(op_name, _)| *op_name == "ameba_autograd_replay_commit")
            .expect("verified replay commit event");
        assert_eq!(commit.1["receipt_sha256"], receipt.receipt_sha256);
        assert_eq!(commit.1["commit_mode"], "verified_scratch_replace");
        assert_eq!(commit.1["replay_verified"], true);
    }

    #[test]
    fn replay_receipts_are_observer_independent() {
        use std::sync::Arc;

        let mut unobserved = mesh_with_pending_cascade();
        let unobserved_receipt = unobserved.drain_with_receipt().unwrap();

        let _lock = crate::telemetry::tensor_observer_lock();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(|_| {})));
        let mut observed = mesh_with_pending_cascade();
        let observed_receipt = observed.drain_with_receipt().unwrap();
        st_tensor::set_thread_meta_observer(previous);

        assert_eq!(observed_receipt, unobserved_receipt);
        assert_eq!(
            observed.checkpoint().unwrap(),
            unobserved.checkpoint().unwrap()
        );
    }

    #[test]
    fn empty_queue_receipts_preserve_explicit_round_and_drain_semantics() {
        let mut round = AmebaAutograd::new(0, 0.0).unwrap();
        round.register_agent(AgentConfig::new(1, 1)).unwrap();
        let round_receipt = round.propagate_round_with_receipt().unwrap();
        assert_eq!(round_receipt.total_processed, 0);
        assert_eq!(round_receipt.rounds.len(), 1);
        assert_eq!(
            round_receipt.rounds[0].weight_update_selected_backend,
            "none"
        );
        evaluate_ameba_autograd_replay_receipt(&round_receipt).unwrap();

        let mut drain = AmebaAutograd::new(0, 0.0).unwrap();
        drain.register_agent(AgentConfig::new(1, 1)).unwrap();
        let drain_receipt = drain.drain_with_receipt().unwrap();
        assert_eq!(drain_receipt.total_processed, 0);
        assert!(drain_receipt.rounds.is_empty());
        assert_eq!(
            drain_receipt.before_state_sha256,
            drain_receipt.after_state_sha256
        );
        evaluate_ameba_autograd_replay_receipt(&drain_receipt).unwrap();
    }

    #[test]
    fn replay_receipt_binds_the_active_tensor_route() {
        use std::sync::{Arc, Mutex};

        let mut source = mesh_with_pending_cascade();
        let before = source.checkpoint().unwrap();
        let receipt = source.propagate_round_with_receipt().unwrap();
        assert_eq!(receipt.rounds[0].weight_update_requested_backend, "auto");

        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured.lock().unwrap().push(event.op_name);
        })));
        let policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::cpu(),
            ExecutionConfig::default(),
        );
        let result = {
            let _guard = push_backend_policy(policy);
            AmebaAutograd::replay_checkpoint(&before, &receipt)
        };
        st_tensor::set_thread_meta_observer(previous);

        assert!(matches!(
            result,
            Err(AmebaAutogradReplayError::ReplayMismatch { .. })
        ));
        let events = events.lock().unwrap();
        assert!(events.contains(&"add_scaled"));
        assert!(!events.contains(&"ameba_autograd_round"));
    }

    #[test]
    fn replay_receipt_cannot_be_applied_to_a_different_checkpoint() {
        let mut source = mesh_with_pending_cascade();
        let checkpoint = source.checkpoint().unwrap();
        let receipt = source.propagate_round_with_receipt().unwrap();

        let mut other = AmebaAutograd::from_checkpoint(&checkpoint).unwrap();
        other.seed_gradient(1, vec![0.25, 0.5]).unwrap();
        let other_checkpoint = other.checkpoint().unwrap();
        assert!(matches!(
            AmebaAutograd::replay_checkpoint(&other_checkpoint, &receipt),
            Err(AmebaAutogradReplayError::BeforeStateMismatch { .. })
        ));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_selected_receipt_replays_under_the_same_policy() {
        use crate::backend::execution::AcceleratorFallback;

        let policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::wgpu(32, true, 256),
            ExecutionConfig::new(AcceleratorFallback::Allow, 1),
        );
        let _guard = push_backend_policy(policy);
        let mut source = mesh_with_pending_cascade();
        let checkpoint = source.checkpoint().unwrap();
        let receipt = source.drain_with_receipt().unwrap();
        let expected = source.checkpoint().unwrap();
        let (replayed, validation) =
            AmebaAutograd::replay_checkpoint(&checkpoint, &receipt).unwrap();

        assert!(receipt.rounds.iter().all(|round| {
            round.weight_update_requested_backend == "wgpu"
                && round.weight_update_selected_backend == "wgpu"
        }));
        assert!(validation.replay_matched);
        assert_eq!(replayed.checkpoint().unwrap(), expected);
    }

    #[test]
    fn receipt_tampering_and_failed_replay_never_mutate_the_target() {
        let mut source = mesh_with_pending_cascade();
        let checkpoint = source.checkpoint().unwrap();
        let mut receipt = source.drain_with_receipt().unwrap();
        receipt.total_processed += 1;

        assert!(matches!(
            evaluate_ameba_autograd_replay_receipt(&receipt),
            Err(AmebaAutogradReplayError::InvalidReceipt { .. })
                | Err(AmebaAutogradReplayError::IntegrityMismatch { .. })
        ));

        let mut target = AmebaAutograd::new(0, 0.0).unwrap();
        target.register_agent(AgentConfig::new(99, 1)).unwrap();
        let before = target.checkpoint().unwrap();
        assert!(target
            .restore_replay_checkpoint(&checkpoint, &receipt)
            .is_err());
        assert_eq!(target.checkpoint().unwrap(), before);
    }

    #[test]
    fn failed_receipted_drain_preserves_the_complete_source_state() {
        let mut mesh = AmebaAutograd::new(2, 1e-6).unwrap();
        mesh.register_agent(
            AgentConfig::new(1, 1)
                .with_neighbors(vec![2])
                .with_learning_rate(0.1)
                .with_damping(1.0)
                .with_weights(vec![1.0]),
        )
        .unwrap();
        mesh.register_agent(
            AgentConfig::new(2, 1)
                .with_neighbors(vec![1])
                .with_learning_rate(f32::MAX)
                .with_weights(vec![1.0]),
        )
        .unwrap();
        mesh.seed_gradient(1, vec![2.0]).unwrap();
        let before = mesh.checkpoint().unwrap();

        assert!(matches!(
            mesh.drain_with_receipt(),
            Err(AmebaAutogradReplayError::Autograd(
                AutogradError::NonFiniteValue {
                    label: "weight_delta",
                    ..
                }
            ))
        ));
        assert_eq!(mesh.checkpoint().unwrap(), before);
    }

    #[test]
    fn checkpoint_json_rejects_unknown_fields() {
        let checkpoint = mesh_with_pending_cascade().checkpoint().unwrap();
        let mut payload = serde_json::to_value(checkpoint).unwrap();
        payload["state"]["surprise"] = serde_json::json!(true);

        assert!(serde_json::from_value::<AmebaAutogradCheckpoint>(payload).is_err());
    }
}
