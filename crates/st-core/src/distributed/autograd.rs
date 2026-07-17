// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::backend::execution::{current_tensor_util_route, TensorUtilRoute};
use crate::causal::NodeId;
use st_tensor::{
    emit_tensor_op, emit_tensor_op_meta, tensor_op_meta_observer_installed, Tensor, TensorError,
};

mod checkpoint;

pub use checkpoint::{
    evaluate_ameba_autograd_checkpoint, evaluate_ameba_autograd_replay_receipt,
    AmebaAutogradAgentCheckpoint, AmebaAutogradCheckpoint, AmebaAutogradCheckpointError,
    AmebaAutogradCheckpointState, AmebaAutogradCheckpointValidation,
    AmebaAutogradMessageCheckpoint, AmebaAutogradReplayError, AmebaAutogradReplayOperation,
    AmebaAutogradReplayReceipt, AmebaAutogradReplayValidation, AmebaAutogradRoundReceipt,
    AMEBA_AUTOGRAD_CHECKPOINT_CONTRACT_VERSION, AMEBA_AUTOGRAD_CHECKPOINT_KIND,
    AMEBA_AUTOGRAD_CHECKPOINT_RESUME_SCOPE, AMEBA_AUTOGRAD_CHECKPOINT_SEMANTIC_BACKEND,
    AMEBA_AUTOGRAD_CHECKPOINT_SEMANTIC_OWNER, AMEBA_AUTOGRAD_REPLAY_CONTRACT_VERSION,
    AMEBA_AUTOGRAD_REPLAY_KIND, AMEBA_AUTOGRAD_REPLAY_SEMANTIC_BACKEND,
    AMEBA_AUTOGRAD_REPLAY_SEMANTIC_OWNER,
};

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AutogradError {
    #[error("agent {0} already registered")]
    DuplicateAgent(NodeId),
    #[error("unknown agent {0}")]
    UnknownAgent(NodeId),
    #[error("gradient dimensionality mismatch: expected {expected}, got {got}")]
    GradientDimension { expected: usize, got: usize },
    #[error("tolerance must be non-negative, got {0}")]
    NegativeTolerance(f32),
    #[error("damping must be non-negative, got {0}")]
    NegativeDamping(f32),
    #[error("damping must not exceed one, got {0}")]
    DampingAboveOne(f32),
    #[error("learning rate must be positive and finite, got {0}")]
    NonPositiveLearningRate(f32),
    #[error("agent {0} must have at least one weight")]
    EmptyAgentDimension(NodeId),
    #[error("agent {agent} cannot list itself as a neighbor")]
    SelfNeighbor { agent: NodeId },
    #[error("agent {agent} lists neighbor {neighbor} more than once")]
    DuplicateNeighbor { agent: NodeId, neighbor: NodeId },
    #[error("agent {agent} references unknown neighbor {neighbor}")]
    UnknownNeighbor { agent: NodeId, neighbor: NodeId },
    #[error(
        "agent {agent} dimension {agent_dimension} does not match neighbor {neighbor} dimension {neighbor_dimension}"
    )]
    NeighborDimension {
        agent: NodeId,
        neighbor: NodeId,
        agent_dimension: usize,
        neighbor_dimension: usize,
    },
    #[error("{label} contained non-finite value at index {index}: {value}")]
    NonFiniteValue {
        label: &'static str,
        index: usize,
        value: f32,
    },
    #[error("tensor operation {operation} failed: {source}")]
    TensorOperation {
        operation: &'static str,
        #[source]
        source: TensorError,
    },
}

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub id: NodeId,
    pub neighbors: Vec<NodeId>,
    pub learning_rate: f32,
    pub damping: f32,
    pub weights: Vec<f32>,
}

impl AgentConfig {
    pub fn new(id: NodeId, dimension: usize) -> Self {
        Self {
            id,
            neighbors: Vec::new(),
            learning_rate: 0.05,
            damping: 0.6,
            weights: vec![0.0; dimension],
        }
    }

    pub fn with_neighbors(mut self, neighbors: Vec<NodeId>) -> Self {
        self.neighbors = neighbors;
        self
    }

    pub fn with_learning_rate(mut self, rate: f32) -> Self {
        self.learning_rate = rate;
        self
    }

    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        self.weights = weights;
        self
    }
}

#[derive(Clone, Debug)]
struct AgentState {
    neighbors: Vec<NodeId>,
    learning_rate: f32,
    damping: f32,
    weights: Vec<f32>,
}

#[derive(Clone, Debug)]
struct GradientMessage {
    from: NodeId,
    to: NodeId,
    hops: usize,
    payload: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct NumericRouteSummary {
    requested_backend: Option<&'static str>,
    selected_backend: Option<&'static str>,
    requested_mixed: bool,
    selected_mixed: bool,
    operations: usize,
    values: usize,
}

impl NumericRouteSummary {
    fn record(&mut self, route: TensorUtilRoute) {
        let requested = route.requested_backend_label();
        let selected = route.selected_backend_label();
        if self
            .requested_backend
            .is_some_and(|current| current != requested)
        {
            self.requested_mixed = true;
        }
        if self
            .selected_backend
            .is_some_and(|current| current != selected)
        {
            self.selected_mixed = true;
        }
        self.requested_backend.get_or_insert(requested);
        self.selected_backend.get_or_insert(selected);
        self.operations += 1;
        self.values = self.values.saturating_add(route.values);
    }

    fn requested_label(self) -> &'static str {
        if self.requested_mixed {
            "mixed"
        } else {
            self.requested_backend.unwrap_or("none")
        }
    }

    fn selected_label(self) -> &'static str {
        if self.selected_mixed {
            "mixed"
        } else {
            self.selected_backend.unwrap_or("none")
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct MessageOutcome {
    forwarded: usize,
    absorbed_tolerance: usize,
    stopped_max_hops: usize,
    terminal_no_neighbor: usize,
    unknown_targets: usize,
    signal_sum: f64,
    forwarded_signal_sum: f64,
    update_l2_sum: f64,
    forwarded_values: usize,
    weight_update_route: Option<TensorUtilRoute>,
    forwarding_route: Option<TensorUtilRoute>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RoundStats {
    pending_before: usize,
    pending_values_before: usize,
    pending_values_after: usize,
    processed: usize,
    forwarded: usize,
    absorbed_tolerance: usize,
    stopped_max_hops: usize,
    terminal_no_neighbor: usize,
    unknown_targets: usize,
    signal_sum: f64,
    forwarded_signal_sum: f64,
    update_l2_sum: f64,
    forwarded_values: usize,
    weight_update_routes: NumericRouteSummary,
    forwarding_routes: NumericRouteSummary,
}

#[derive(Clone, Copy, Debug)]
struct CommittedRound {
    stats: RoundStats,
    pending_after: usize,
}

#[derive(Clone, Debug, Default)]
struct ExecutionOutcome {
    total_processed: usize,
    rounds: Vec<CommittedRound>,
}

impl RoundStats {
    fn add(&mut self, outcome: MessageOutcome) {
        self.forwarded += outcome.forwarded;
        self.absorbed_tolerance += outcome.absorbed_tolerance;
        self.stopped_max_hops += outcome.stopped_max_hops;
        self.terminal_no_neighbor += outcome.terminal_no_neighbor;
        self.unknown_targets += outcome.unknown_targets;
        self.signal_sum += outcome.signal_sum;
        self.forwarded_signal_sum += outcome.forwarded_signal_sum;
        self.update_l2_sum += outcome.update_l2_sum;
        self.forwarded_values = self
            .forwarded_values
            .saturating_add(outcome.forwarded_values);
        if let Some(route) = outcome.weight_update_route {
            self.weight_update_routes.record(route);
        }
        if let Some(route) = outcome.forwarding_route {
            self.forwarding_routes.record(route);
        }
    }
}

fn finite_meta_f32(value: f32) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value as f64 } else { 0.0 })
}

fn finite_meta_f64(value: f64) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value } else { 0.0 })
}

fn l1_signal(values: &[f32]) -> f64 {
    values.iter().map(|value| value.abs() as f64).sum()
}

fn emit_autograd_round_meta(
    mesh: &AmebaAutograd,
    stats: RoundStats,
    pending_after: usize,
    capture_meta: bool,
) {
    emit_tensor_op(
        "ameba_autograd_round",
        &[stats.pending_before.max(1), mesh.agents.len().max(1)],
        &[pending_after.max(1), mesh.agents.len().max(1)],
    );
    if !capture_meta {
        return;
    }
    let max_gradient_dim = mesh
        .agents
        .values()
        .map(|agent| agent.weights.len())
        .max()
        .unwrap_or(0);
    emit_tensor_op_meta("ameba_autograd_round", || {
        serde_json::json!({
            "backend": "cpu",
            "requested_backend": stats.weight_update_routes.requested_label(),
            "kind": "st_core_ameba_autograd_round",
            "control_backend": "cpu_hashmap_vecdeque",
            "message_queue_backend": "cpu_vecdeque",
            "agent_state_backend": "cpu_hashmap",
            "weight_update_backend": format!(
                "tensor_util_{}",
                stats.weight_update_routes.selected_label()
            ),
            "forwarding_backend": format!(
                "tensor_util_{}",
                stats.forwarding_routes.selected_label()
            ),
            "weight_update_requested_backend": stats.weight_update_routes.requested_label(),
            "forwarding_requested_backend": stats.forwarding_routes.requested_label(),
            "weight_update_operations": stats.weight_update_routes.operations,
            "forwarding_operations": stats.forwarding_routes.operations,
            "forwarding_transport_backend": "cpu_vec_clone",
            "propagation_mode": "stateful_agent_message_passing",
            "round_commit_mode": "transactional_scratch",
            "route_blocker": "stateful_agent_graph_and_message_queue",
            "numeric_execution_owner": "st-core::backend::execution",
            "numeric_backend_semantics": "selected_route",
            "actual_numeric_backend_source": "add_scaled_and_scale_tensor_op_meta",
            "agents": mesh.agents.len(),
            "max_gradient_dim": max_gradient_dim,
            "pending_before": stats.pending_before,
            "pending_after": pending_after,
            "pending_values_before": stats.pending_values_before,
            "pending_values_after": stats.pending_values_after,
            "processed": stats.processed,
            "forwarded": stats.forwarded,
            "absorbed_tolerance": stats.absorbed_tolerance,
            "stopped_max_hops": stats.stopped_max_hops,
            "terminal_no_neighbor": stats.terminal_no_neighbor,
            "unknown_targets": stats.unknown_targets,
            "max_hops": mesh.max_hops,
            "tolerance": finite_meta_f32(mesh.tolerance),
            "signal_sum": finite_meta_f64(stats.signal_sum),
            "forwarded_signal_sum": finite_meta_f64(stats.forwarded_signal_sum),
            "update_l2_sum": finite_meta_f64(stats.update_l2_sum),
            "estimated_weight_update_values": stats.weight_update_routes.values,
            "estimated_forwarding_compute_values": stats.forwarding_routes.values,
            "estimated_forwarded_gradient_values": stats.forwarded_values,
            "estimated_queue_values_after": stats.pending_values_after,
        })
    });
}

#[derive(Clone, Debug)]
pub struct AmebaAutograd {
    agents: HashMap<NodeId, AgentState>,
    pending: VecDeque<GradientMessage>,
    max_hops: usize,
    tolerance: f32,
}

impl AmebaAutograd {
    pub fn new(max_hops: usize, tolerance: f32) -> Result<Self, AutogradError> {
        if !tolerance.is_finite() {
            return Err(AutogradError::NonFiniteValue {
                label: "tolerance",
                index: 0,
                value: tolerance,
            });
        }
        if tolerance < 0.0 {
            return Err(AutogradError::NegativeTolerance(tolerance));
        }
        Ok(Self {
            agents: HashMap::new(),
            pending: VecDeque::new(),
            max_hops,
            tolerance,
        })
    }

    pub fn register_agent(&mut self, config: AgentConfig) -> Result<(), AutogradError> {
        if self.agents.contains_key(&config.id) {
            return Err(AutogradError::DuplicateAgent(config.id));
        }
        if config.weights.is_empty() {
            return Err(AutogradError::EmptyAgentDimension(config.id));
        }
        if config.learning_rate <= 0.0 || !config.learning_rate.is_finite() {
            return Err(AutogradError::NonPositiveLearningRate(config.learning_rate));
        }
        Self::validate_finite_slice("damping", &[config.damping])?;
        if config.damping < 0.0 {
            return Err(AutogradError::NegativeDamping(config.damping));
        }
        if config.damping > 1.0 {
            return Err(AutogradError::DampingAboveOne(config.damping));
        }
        Self::validate_finite_slice("weights", &config.weights)?;
        let mut neighbors = HashSet::with_capacity(config.neighbors.len());
        for &neighbor in &config.neighbors {
            if neighbor == config.id {
                return Err(AutogradError::SelfNeighbor { agent: config.id });
            }
            if !neighbors.insert(neighbor) {
                return Err(AutogradError::DuplicateNeighbor {
                    agent: config.id,
                    neighbor,
                });
            }
        }
        self.agents.insert(
            config.id,
            AgentState {
                neighbors: config.neighbors,
                learning_rate: config.learning_rate,
                damping: config.damping,
                weights: config.weights,
            },
        );
        Ok(())
    }

    /// Verifies that every routing edge exists and preserves gradient dimension.
    pub fn validate_topology(&self) -> Result<(), AutogradError> {
        let mut agent_ids = self.agents.keys().copied().collect::<Vec<_>>();
        agent_ids.sort_unstable();
        for agent_id in agent_ids {
            let agent = &self.agents[&agent_id];
            for &neighbor_id in &agent.neighbors {
                let neighbor =
                    self.agents
                        .get(&neighbor_id)
                        .ok_or(AutogradError::UnknownNeighbor {
                            agent: agent_id,
                            neighbor: neighbor_id,
                        })?;
                if agent.weights.len() != neighbor.weights.len() {
                    return Err(AutogradError::NeighborDimension {
                        agent: agent_id,
                        neighbor: neighbor_id,
                        agent_dimension: agent.weights.len(),
                        neighbor_dimension: neighbor.weights.len(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn seed_gradient(&mut self, id: NodeId, gradient: Vec<f32>) -> Result<(), AutogradError> {
        let expected = self
            .agents
            .get(&id)
            .map(|agent| agent.weights.len())
            .ok_or(AutogradError::UnknownAgent(id))?;
        if expected != gradient.len() {
            return Err(AutogradError::GradientDimension {
                expected,
                got: gradient.len(),
            });
        }
        Self::validate_finite_slice("gradient", &gradient)?;
        self.pending.push_back(GradientMessage {
            from: id,
            to: id,
            hops: 0,
            payload: gradient,
        });
        Ok(())
    }

    pub fn propagate_round(&mut self) -> Result<usize, AutogradError> {
        Ok(self.transact_round()?.total_processed)
    }

    fn transact_round(&mut self) -> Result<ExecutionOutcome, AutogradError> {
        self.transact_round_with_meta(true)
    }

    fn transact_round_with_meta(
        &mut self,
        emit_round_meta: bool,
    ) -> Result<ExecutionOutcome, AutogradError> {
        self.validate_topology()?;
        let capture_meta = emit_round_meta && tensor_op_meta_observer_installed();
        let mut scratch = self.clone();
        let (processed, stats) = scratch.process_round_in_place(capture_meta)?;
        let pending_after = scratch.pending.len();
        *self = scratch;
        if emit_round_meta {
            emit_autograd_round_meta(self, stats, pending_after, capture_meta);
        }
        Ok(ExecutionOutcome {
            total_processed: processed,
            rounds: vec![CommittedRound {
                stats,
                pending_after,
            }],
        })
    }

    fn process_round_in_place(
        &mut self,
        capture_meta: bool,
    ) -> Result<(usize, RoundStats), AutogradError> {
        let mut processed = 0usize;
        let mut per_round = self.pending.len();
        let mut stats = RoundStats {
            pending_before: per_round,
            pending_values_before: self.pending.iter().fold(0usize, |total, message| {
                total.saturating_add(message.payload.len())
            }),
            ..RoundStats::default()
        };
        while per_round > 0 {
            if let Some(message) = self.pending.pop_front() {
                processed += 1;
                per_round -= 1;
                let outcome = self.handle_message(message, capture_meta)?;
                stats.add(outcome);
            }
        }
        stats.processed = processed;
        stats.pending_values_after = self.pending.iter().fold(0usize, |total, message| {
            total.saturating_add(message.payload.len())
        });
        Ok((processed, stats))
    }

    pub fn drain(&mut self) -> Result<usize, AutogradError> {
        Ok(self.transact_drain()?.total_processed)
    }

    fn transact_drain(&mut self) -> Result<ExecutionOutcome, AutogradError> {
        self.transact_drain_with_meta(true)
    }

    fn transact_drain_with_meta(
        &mut self,
        emit_round_meta: bool,
    ) -> Result<ExecutionOutcome, AutogradError> {
        self.validate_topology()?;
        let capture_meta = emit_round_meta && tensor_op_meta_observer_installed();
        let mut scratch = self.clone();
        let mut total = 0usize;
        let mut completed_rounds = Vec::new();
        while !scratch.pending.is_empty() {
            let (processed, stats) = scratch.process_round_in_place(capture_meta)?;
            total = total.saturating_add(processed);
            completed_rounds.push(CommittedRound {
                stats,
                pending_after: scratch.pending.len(),
            });
        }
        *self = scratch;
        if emit_round_meta {
            for round in &completed_rounds {
                emit_autograd_round_meta(self, round.stats, round.pending_after, capture_meta);
            }
        }
        Ok(ExecutionOutcome {
            total_processed: total,
            rounds: completed_rounds,
        })
    }

    pub fn weights(&self, id: NodeId) -> Result<&[f32], AutogradError> {
        self.agents
            .get(&id)
            .map(|agent| agent.weights.as_slice())
            .ok_or(AutogradError::UnknownAgent(id))
    }

    fn handle_message(
        &mut self,
        message: GradientMessage,
        capture_meta: bool,
    ) -> Result<MessageOutcome, AutogradError> {
        let mut outcome = MessageOutcome::default();
        let agent = self
            .agents
            .get(&message.to)
            .cloned()
            .ok_or(AutogradError::UnknownAgent(message.to))?;
        if agent.weights.len() != message.payload.len() {
            return Err(AutogradError::GradientDimension {
                expected: agent.weights.len(),
                got: message.payload.len(),
            });
        }
        Self::validate_finite_slice("message_gradient", &message.payload)?;

        let gradient = Tensor::from_vec(1, message.payload.len(), message.payload.clone())
            .map_err(|source| Self::tensor_error("gradient_tensor", source))?;
        let mut next_weights = Tensor::from_vec(1, agent.weights.len(), agent.weights.clone())
            .map_err(|source| Self::tensor_error("weight_tensor", source))?;
        let update_route = current_tensor_util_route(agent.weights.len());
        next_weights
            .add_scaled_with_backend(
                &gradient,
                -agent.learning_rate,
                update_route.selected_backend,
            )
            .map_err(|source| {
                Self::map_update_error(
                    &agent.weights,
                    &message.payload,
                    agent.learning_rate,
                    source,
                )
            })?;
        let next_weights = next_weights.data().to_vec();
        outcome.weight_update_route = Some(update_route);
        if capture_meta {
            outcome.update_l2_sum = message
                .payload
                .iter()
                .map(|gradient| {
                    let delta = *gradient as f64 * agent.learning_rate as f64;
                    delta * delta
                })
                .sum::<f64>()
                .sqrt();
        }

        let signal = l1_signal(&message.payload);
        outcome.signal_sum = signal;
        if signal < self.tolerance as f64 && message.hops > 0 {
            self.commit_agent_weights(message.to, next_weights)?;
            outcome.absorbed_tolerance = 1;
            return Ok(outcome);
        }

        if message.hops >= self.max_hops {
            self.commit_agent_weights(message.to, next_weights)?;
            outcome.stopped_max_hops = 1;
            return Ok(outcome);
        }

        let neighbors = agent
            .neighbors
            .iter()
            .copied()
            .filter(|neighbor| *neighbor != message.from)
            .collect::<Vec<_>>();
        if neighbors.is_empty() {
            self.commit_agent_weights(message.to, next_weights)?;
            outcome.terminal_no_neighbor = 1;
            return Ok(outcome);
        }

        let forwarding_route = current_tensor_util_route(message.payload.len());
        let forwarded = gradient
            .scale_with_backend(agent.damping, forwarding_route.selected_backend)
            .map_err(|source| Self::map_forward_error(&message.payload, agent.damping, source))?;
        let forwarded = forwarded.data().to_vec();
        outcome.forwarding_route = Some(forwarding_route);
        let forwarded_signal = l1_signal(&forwarded);
        outcome.forwarded_signal_sum = forwarded_signal;
        if forwarded_signal < self.tolerance as f64 {
            self.commit_agent_weights(message.to, next_weights)?;
            outcome.absorbed_tolerance = 1;
            return Ok(outcome);
        }

        self.commit_agent_weights(message.to, next_weights)?;
        outcome.forwarded_values = neighbors.len().saturating_mul(forwarded.len());
        for neighbor in neighbors {
            self.pending.push_back(GradientMessage {
                from: message.to,
                to: neighbor,
                hops: message.hops + 1,
                payload: forwarded.clone(),
            });
            outcome.forwarded += 1;
        }
        Ok(outcome)
    }

    fn commit_agent_weights(&mut self, id: NodeId, weights: Vec<f32>) -> Result<(), AutogradError> {
        let agent = self
            .agents
            .get_mut(&id)
            .ok_or(AutogradError::UnknownAgent(id))?;
        agent.weights = weights;
        Ok(())
    }

    fn tensor_error(operation: &'static str, source: TensorError) -> AutogradError {
        AutogradError::TensorOperation { operation, source }
    }

    fn map_update_error(
        weights: &[f32],
        gradients: &[f32],
        learning_rate: f32,
        source: TensorError,
    ) -> AutogradError {
        if matches!(source, TensorError::NonFiniteValue { .. }) {
            for (index, (&weight, &gradient)) in weights.iter().zip(gradients).enumerate() {
                let delta = learning_rate * gradient;
                if !delta.is_finite() {
                    return AutogradError::NonFiniteValue {
                        label: "weight_delta",
                        index,
                        value: delta,
                    };
                }
                let next = weight - delta;
                if !next.is_finite() {
                    return AutogradError::NonFiniteValue {
                        label: "weight_update",
                        index,
                        value: next,
                    };
                }
            }
        }
        Self::tensor_error("weight_update", source)
    }

    fn map_forward_error(gradients: &[f32], damping: f32, source: TensorError) -> AutogradError {
        if matches!(source, TensorError::NonFiniteValue { .. }) {
            for (index, gradient) in gradients.iter().copied().enumerate() {
                let forwarded = gradient * damping;
                if !forwarded.is_finite() {
                    return AutogradError::NonFiniteValue {
                        label: "forwarded_gradient",
                        index,
                        value: forwarded,
                    };
                }
            }
        }
        Self::tensor_error("gradient_forwarding", source)
    }

    fn validate_finite_slice(label: &'static str, values: &[f32]) -> Result<(), AutogradError> {
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(AutogradError::NonFiniteValue {
                    label,
                    index,
                    value,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn gradient_cascades_without_central_server() {
        let mut mesh = AmebaAutograd::new(4, 1e-4).unwrap();
        mesh.register_agent(
            AgentConfig::new(1, 2)
                .with_neighbors(vec![2])
                .with_learning_rate(0.1),
        )
        .unwrap();
        mesh.register_agent(
            AgentConfig::new(2, 2)
                .with_neighbors(vec![1, 3])
                .with_learning_rate(0.1),
        )
        .unwrap();
        mesh.register_agent(
            AgentConfig::new(3, 2)
                .with_neighbors(vec![2])
                .with_learning_rate(0.1),
        )
        .unwrap();

        mesh.seed_gradient(2, vec![1.0, -1.0]).unwrap();
        mesh.drain().unwrap();

        let w1 = mesh.weights(1).unwrap();
        let w2 = mesh.weights(2).unwrap();
        let w3 = mesh.weights(3).unwrap();

        assert_ne!(w1[0], 0.0);
        assert_ne!(w2[0], 0.0);
        assert_ne!(w3[0], 0.0);
        assert!(w1[0] < 0.0 && w3[0] < 0.0);
    }

    #[test]
    fn propagate_round_emits_backend_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut mesh = AmebaAutograd::new(4, 1e-4).unwrap();
        mesh.register_agent(
            AgentConfig::new(1, 2)
                .with_neighbors(vec![2])
                .with_learning_rate(0.1),
        )
        .unwrap();
        mesh.register_agent(
            AgentConfig::new(2, 2)
                .with_neighbors(vec![1, 3])
                .with_learning_rate(0.1),
        )
        .unwrap();
        mesh.register_agent(
            AgentConfig::new(3, 2)
                .with_neighbors(vec![2])
                .with_learning_rate(0.1),
        )
        .unwrap();
        mesh.seed_gradient(2, vec![1.0, -1.0]).unwrap();
        assert_eq!(mesh.propagate_round().unwrap(), 1);
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "ameba_autograd_round"
                    && data["kind"] == "st_core_ameba_autograd_round"
                    && data["processed"] == 1
                    && data["agents"] == 3
                    && data["pending_after"] == 2
                    && data["forwarded"] == 2
            })
            .expect("ameba autograd round metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["requested_backend"], "auto");
        assert_eq!(meta.1["agents"], 3);
        assert_eq!(meta.1["control_backend"], "cpu_hashmap_vecdeque");
        assert_eq!(meta.1["message_queue_backend"], "cpu_vecdeque");
        assert_eq!(meta.1["agent_state_backend"], "cpu_hashmap");
        assert_eq!(meta.1["weight_update_backend"], "tensor_util_auto");
        assert_eq!(meta.1["forwarding_backend"], "tensor_util_auto");
        assert_eq!(meta.1["weight_update_requested_backend"], "auto");
        assert_eq!(meta.1["forwarding_requested_backend"], "auto");
        assert_eq!(meta.1["weight_update_operations"], 1);
        assert_eq!(meta.1["forwarding_operations"], 1);
        assert_eq!(meta.1["forwarding_transport_backend"], "cpu_vec_clone");
        assert_eq!(meta.1["propagation_mode"], "stateful_agent_message_passing");
        assert_eq!(meta.1["round_commit_mode"], "transactional_scratch");
        assert_eq!(
            meta.1["route_blocker"],
            "stateful_agent_graph_and_message_queue"
        );
        assert_eq!(
            meta.1["numeric_execution_owner"],
            "st-core::backend::execution"
        );
        assert_eq!(meta.1["numeric_backend_semantics"], "selected_route");
        assert_eq!(
            meta.1["actual_numeric_backend_source"],
            "add_scaled_and_scale_tensor_op_meta"
        );
        assert_eq!(meta.1["max_gradient_dim"], 2);
        assert_eq!(meta.1["pending_before"], 1);
        assert_eq!(meta.1["pending_after"], 2);
        assert_eq!(meta.1["pending_values_before"], 2);
        assert_eq!(meta.1["pending_values_after"], 4);
        assert_eq!(meta.1["forwarded"], 2);
        assert_eq!(meta.1["estimated_weight_update_values"], 2);
        assert_eq!(meta.1["estimated_forwarding_compute_values"], 2);
        assert_eq!(meta.1["estimated_forwarded_gradient_values"], 4);
        assert_eq!(meta.1["estimated_queue_values_after"], 4);
        assert!(meta.1["signal_sum"].as_f64().unwrap() > 0.0);
        assert!(meta.1["update_l2_sum"].as_f64().unwrap() > 0.0);

        let add_scaled = events
            .iter()
            .find(|(op_name, _)| *op_name == "add_scaled")
            .expect("weight update tensor metadata event");
        assert_eq!(add_scaled.1["requested_backend"], "auto");
        let scale = events
            .iter()
            .find(|(op_name, _)| *op_name == "scale")
            .expect("forwarding tensor metadata event");
        assert_eq!(scale.1["requested_backend"], "auto");
    }

    #[test]
    fn tolerance_drops_small_residuals() {
        let mut mesh = AmebaAutograd::new(3, 0.5).unwrap();
        mesh.register_agent(AgentConfig::new(1, 1).with_neighbors(vec![2]))
            .unwrap();
        mesh.register_agent(AgentConfig::new(2, 1).with_neighbors(vec![1]))
            .unwrap();
        mesh.seed_gradient(1, vec![0.1]).unwrap();
        mesh.drain().unwrap();
        let w1 = mesh.weights(1).unwrap();
        let w2 = mesh.weights(2).unwrap();
        assert_eq!(w2[0], 0.0);
        assert!(w1[0] < 0.0);
    }

    #[test]
    fn rejects_non_finite_tolerance() {
        let err = AmebaAutograd::new(3, f32::NAN).unwrap_err();
        assert!(matches!(
            err,
            AutogradError::NonFiniteValue {
                label: "tolerance",
                index: 0,
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn register_rejects_invalid_agent_scalars_and_weights() {
        let mut mesh = AmebaAutograd::new(3, 1e-4).unwrap();
        assert_eq!(
            mesh.register_agent(AgentConfig::new(1, 0)).unwrap_err(),
            AutogradError::EmptyAgentDimension(1)
        );
        assert_eq!(
            mesh.register_agent(AgentConfig::new(1, 1).with_learning_rate(0.0))
                .unwrap_err(),
            AutogradError::NonPositiveLearningRate(0.0)
        );
        assert!(matches!(
            mesh.register_agent(AgentConfig::new(1, 1).with_damping(f32::NAN))
                .unwrap_err(),
            AutogradError::NonFiniteValue {
                label: "damping",
                index: 0,
                value,
            } if value.is_nan()
        ));
        assert_eq!(
            mesh.register_agent(AgentConfig::new(1, 1).with_damping(-0.1))
                .unwrap_err(),
            AutogradError::NegativeDamping(-0.1)
        );
        assert_eq!(
            mesh.register_agent(AgentConfig::new(1, 1).with_damping(1.01))
                .unwrap_err(),
            AutogradError::DampingAboveOne(1.01)
        );
        assert_eq!(
            mesh.register_agent(AgentConfig::new(1, 1).with_weights(vec![f32::INFINITY]))
                .unwrap_err(),
            AutogradError::NonFiniteValue {
                label: "weights",
                index: 0,
                value: f32::INFINITY,
            }
        );
        assert_eq!(
            mesh.register_agent(AgentConfig::new(1, 1).with_neighbors(vec![1]))
                .unwrap_err(),
            AutogradError::SelfNeighbor { agent: 1 }
        );
        assert_eq!(
            mesh.register_agent(AgentConfig::new(1, 1).with_neighbors(vec![2, 2]))
                .unwrap_err(),
            AutogradError::DuplicateNeighbor {
                agent: 1,
                neighbor: 2,
            }
        );
        assert_eq!(mesh.weights(1).unwrap_err(), AutogradError::UnknownAgent(1));
    }

    #[test]
    fn topology_preflight_rejects_unknown_and_dimension_mismatched_edges() {
        let mut unknown = AmebaAutograd::new(2, 1e-6).unwrap();
        unknown
            .register_agent(AgentConfig::new(1, 2).with_neighbors(vec![2]))
            .unwrap();
        assert_eq!(
            unknown.validate_topology().unwrap_err(),
            AutogradError::UnknownNeighbor {
                agent: 1,
                neighbor: 2,
            }
        );
        unknown.seed_gradient(1, vec![1.0, -1.0]).unwrap();
        assert_eq!(
            unknown.propagate_round().unwrap_err(),
            AutogradError::UnknownNeighbor {
                agent: 1,
                neighbor: 2,
            }
        );
        assert_eq!(unknown.pending.len(), 1);
        assert_eq!(unknown.weights(1).unwrap(), &[0.0, 0.0]);

        let mut mismatch = AmebaAutograd::new(2, 1e-6).unwrap();
        mismatch
            .register_agent(AgentConfig::new(1, 2).with_neighbors(vec![2]))
            .unwrap();
        mismatch.register_agent(AgentConfig::new(2, 3)).unwrap();
        assert_eq!(
            mismatch.validate_topology().unwrap_err(),
            AutogradError::NeighborDimension {
                agent: 1,
                neighbor: 2,
                agent_dimension: 2,
                neighbor_dimension: 3,
            }
        );
    }

    #[test]
    fn zero_max_hops_is_an_explicit_local_only_update() {
        let mut mesh = AmebaAutograd::new(0, 0.0).unwrap();
        mesh.register_agent(
            AgentConfig::new(1, 1)
                .with_neighbors(vec![2])
                .with_learning_rate(0.1),
        )
        .unwrap();
        mesh.register_agent(AgentConfig::new(2, 1).with_neighbors(vec![1]))
            .unwrap();
        mesh.seed_gradient(1, vec![2.0]).unwrap();

        assert_eq!(mesh.drain().unwrap(), 1);

        assert!((mesh.weights(1).unwrap()[0] + 0.2).abs() < 1e-6);
        assert_eq!(mesh.weights(2).unwrap(), &[0.0]);
    }

    #[test]
    fn seed_gradient_rejects_non_finite_without_queueing() {
        let mut mesh = AmebaAutograd::new(2, 1e-6).unwrap();
        mesh.register_agent(AgentConfig::new(1, 1).with_weights(vec![1.0]))
            .unwrap();

        let err = mesh.seed_gradient(1, vec![f32::NEG_INFINITY]).unwrap_err();
        assert_eq!(
            err,
            AutogradError::NonFiniteValue {
                label: "gradient",
                index: 0,
                value: f32::NEG_INFINITY,
            }
        );
        assert_eq!(mesh.drain().unwrap(), 0);
        assert_eq!(mesh.weights(1).unwrap(), &[1.0]);
    }

    #[test]
    fn local_update_overflow_does_not_mutate_weights() {
        let mut mesh = AmebaAutograd::new(2, 1e-6).unwrap();
        mesh.register_agent(
            AgentConfig::new(1, 1)
                .with_learning_rate(f32::MAX)
                .with_weights(vec![1.0]),
        )
        .unwrap();
        mesh.seed_gradient(1, vec![2.0]).unwrap();

        let err = mesh.drain().unwrap_err();
        assert_eq!(
            err,
            AutogradError::NonFiniteValue {
                label: "weight_delta",
                index: 0,
                value: f32::INFINITY,
            }
        );
        assert_eq!(mesh.weights(1).unwrap(), &[1.0]);
        assert_eq!(mesh.pending.len(), 1);
    }

    #[test]
    fn round_failure_rolls_back_every_message_and_preserves_the_queue() {
        let mut mesh = AmebaAutograd::new(2, 1e-6).unwrap();
        mesh.register_agent(
            AgentConfig::new(1, 1)
                .with_learning_rate(0.1)
                .with_weights(vec![1.0]),
        )
        .unwrap();
        mesh.register_agent(
            AgentConfig::new(2, 1)
                .with_learning_rate(f32::MAX)
                .with_weights(vec![1.0]),
        )
        .unwrap();
        mesh.seed_gradient(1, vec![1.0]).unwrap();
        mesh.seed_gradient(2, vec![2.0]).unwrap();

        let err = mesh.propagate_round().unwrap_err();
        assert_eq!(
            err,
            AutogradError::NonFiniteValue {
                label: "weight_delta",
                index: 0,
                value: f32::INFINITY,
            }
        );
        assert_eq!(mesh.weights(1).unwrap(), &[1.0]);
        assert_eq!(mesh.weights(2).unwrap(), &[1.0]);
        assert_eq!(mesh.pending.len(), 2);
    }

    #[test]
    fn drain_failure_rolls_back_every_completed_round() {
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

        let err = mesh.drain().unwrap_err();

        assert_eq!(
            err,
            AutogradError::NonFiniteValue {
                label: "weight_delta",
                index: 0,
                value: f32::INFINITY,
            }
        );
        assert_eq!(mesh.weights(1).unwrap(), &[1.0]);
        assert_eq!(mesh.weights(2).unwrap(), &[1.0]);
        assert_eq!(mesh.pending.len(), 1);
    }

    #[test]
    fn telemetry_observer_does_not_change_near_tolerance_routing() {
        let dimension = 10_000;
        let mut base = AmebaAutograd::new(2, 1.000_02).unwrap();
        base.register_agent(
            AgentConfig::new(1, dimension)
                .with_neighbors(vec![2])
                .with_damping(1.0),
        )
        .unwrap();
        base.register_agent(AgentConfig::new(2, dimension).with_neighbors(vec![1]))
            .unwrap();

        let gradient = vec![1.0e-4; dimension];
        let mut unobserved = base.clone();
        unobserved.seed_gradient(1, gradient.clone()).unwrap();
        unobserved.drain().unwrap();

        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));
        let mut observed = base;
        observed.seed_gradient(1, gradient).unwrap();
        observed.drain().unwrap();
        st_tensor::set_thread_meta_observer(previous);

        for id in [1, 2] {
            assert_eq!(
                unobserved.weights(id).unwrap(),
                observed.weights(id).unwrap()
            );
        }
        assert!(observed
            .weights(2)
            .unwrap()
            .iter()
            .all(|weight| *weight == 0.0));
        assert!(events
            .lock()
            .unwrap()
            .iter()
            .any(|(op_name, _)| *op_name == "ameba_autograd_round"));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn cpu_and_wgpu_routes_preserve_transactional_autograd_results() {
        use crate::backend::device_caps::DeviceCaps;
        use crate::backend::execution::{
            push_backend_policy, AcceleratorFallback, BackendPolicy, ExecutionConfig,
        };

        fn mesh() -> AmebaAutograd {
            let mut mesh = AmebaAutograd::new(3, 1e-6).unwrap();
            mesh.register_agent(
                AgentConfig::new(1, 4)
                    .with_neighbors(vec![2])
                    .with_learning_rate(0.1)
                    .with_damping(0.5)
                    .with_weights(vec![0.5, -0.25, 1.0, -1.5]),
            )
            .unwrap();
            mesh.register_agent(
                AgentConfig::new(2, 4)
                    .with_neighbors(vec![1, 3])
                    .with_learning_rate(0.05)
                    .with_damping(0.75)
                    .with_weights(vec![0.0, 0.5, -0.5, 1.0]),
            )
            .unwrap();
            mesh.register_agent(
                AgentConfig::new(3, 4)
                    .with_neighbors(vec![2])
                    .with_learning_rate(0.025)
                    .with_weights(vec![-0.5, 0.25, 0.75, -1.0]),
            )
            .unwrap();
            mesh
        }

        let config = ExecutionConfig::new(AcceleratorFallback::Allow, 1);
        let mut cpu = mesh();
        {
            let policy = BackendPolicy::from_device_caps_with_config(DeviceCaps::cpu(), config);
            let _guard = push_backend_policy(policy);
            cpu.seed_gradient(1, vec![1.0, -2.0, 0.5, 0.25]).unwrap();
            cpu.drain().unwrap();
        }

        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));
        let mut wgpu = mesh();
        {
            let policy = BackendPolicy::from_device_caps_with_config(
                DeviceCaps::wgpu(32, true, 256),
                config,
            );
            let _guard = push_backend_policy(policy);
            wgpu.seed_gradient(1, vec![1.0, -2.0, 0.5, 0.25]).unwrap();
            wgpu.drain().unwrap();
        }
        st_tensor::set_thread_meta_observer(previous);

        for id in [1, 2, 3] {
            for (cpu_weight, wgpu_weight) in cpu
                .weights(id)
                .unwrap()
                .iter()
                .zip(wgpu.weights(id).unwrap())
            {
                assert!(
                    (cpu_weight - wgpu_weight).abs() <= 3.0e-4,
                    "agent={id} cpu={cpu_weight} wgpu={wgpu_weight}"
                );
            }
        }

        let events = events.lock().unwrap();
        for op_name in ["add_scaled", "scale"] {
            let routed = events
                .iter()
                .filter(|(name, _)| *name == op_name)
                .collect::<Vec<_>>();
            assert!(!routed.is_empty(), "missing {op_name} tensor events");
            assert!(routed
                .iter()
                .all(|(_, data)| data["requested_backend"] == "wgpu"));
            assert!(routed
                .iter()
                .all(|(_, data)| matches!(data["backend"].as_str(), Some("cpu" | "wgpu_dense"))));
        }
        let forwarding_round = events
            .iter()
            .find(|(op_name, data)| *op_name == "ameba_autograd_round" && data["forwarded"] == 1)
            .expect("policy-routed forwarding round");
        assert_eq!(forwarding_round.1["requested_backend"], "wgpu");
        assert_eq!(
            forwarding_round.1["weight_update_backend"],
            "tensor_util_wgpu"
        );
        assert_eq!(forwarding_round.1["forwarding_backend"], "tensor_util_wgpu");
        assert_eq!(
            forwarding_round.1["control_backend"],
            "cpu_hashmap_vecdeque"
        );
    }

    #[test]
    fn weights_update_once_per_message() {
        let mut mesh = AmebaAutograd::new(2, 1e-6).unwrap();
        mesh.register_agent(
            AgentConfig::new(1, 1)
                .with_neighbors(Vec::new())
                .with_learning_rate(0.1)
                .with_weights(vec![1.0]),
        )
        .unwrap();

        mesh.seed_gradient(1, vec![2.0]).unwrap();
        mesh.drain().unwrap();

        let w = mesh.weights(1).unwrap();
        assert!((w[0] - 0.8).abs() < 1e-6, "weight updated more than once");
    }
}
