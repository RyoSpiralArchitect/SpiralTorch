// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::{HashMap, VecDeque};

use crate::causal::NodeId;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, tensor_op_meta_observer_installed};

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
    #[error("learning rate must be positive and finite, got {0}")]
    NonPositiveLearningRate(f32),
    #[error("{label} contained non-finite value at index {index}: {value}")]
    NonFiniteValue {
        label: &'static str,
        index: usize,
        value: f32,
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
struct MessageOutcome {
    forwarded: usize,
    absorbed_tolerance: usize,
    stopped_max_hops: usize,
    terminal_no_neighbor: usize,
    unknown_targets: usize,
    signal_sum: f32,
    forwarded_signal_sum: f32,
    update_l2_sum: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct RoundStats {
    pending_before: usize,
    processed: usize,
    forwarded: usize,
    absorbed_tolerance: usize,
    stopped_max_hops: usize,
    terminal_no_neighbor: usize,
    unknown_targets: usize,
    signal_sum: f32,
    forwarded_signal_sum: f32,
    update_l2_sum: f32,
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
    }
}

fn finite_meta_f32(value: f32) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value as f64 } else { 0.0 })
}

fn l1_signal(values: &[f32]) -> f32 {
    values
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| value.abs())
        .sum()
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
            "requested_backend": "auto",
            "kind": "st_core_ameba_autograd_round",
            "message_queue_backend": "cpu_vecdeque",
            "agent_state_backend": "cpu_hashmap",
            "weight_update_backend": "cpu_loop",
            "forwarding_backend": "cpu_vec_clone",
            "propagation_mode": "stateful_agent_message_passing",
            "route_blocker": "stateful_agent_graph_and_message_queue",
            "agents": mesh.agents.len(),
            "max_gradient_dim": max_gradient_dim,
            "pending_before": stats.pending_before,
            "pending_after": pending_after,
            "processed": stats.processed,
            "forwarded": stats.forwarded,
            "absorbed_tolerance": stats.absorbed_tolerance,
            "stopped_max_hops": stats.stopped_max_hops,
            "terminal_no_neighbor": stats.terminal_no_neighbor,
            "unknown_targets": stats.unknown_targets,
            "max_hops": mesh.max_hops,
            "tolerance": finite_meta_f32(mesh.tolerance),
            "signal_sum": finite_meta_f32(stats.signal_sum),
            "forwarded_signal_sum": finite_meta_f32(stats.forwarded_signal_sum),
            "update_l2_sum": finite_meta_f32(stats.update_l2_sum),
            "estimated_weight_update_values": stats.processed.saturating_mul(max_gradient_dim),
            "estimated_forwarded_gradient_values": stats.forwarded.saturating_mul(max_gradient_dim),
            "estimated_queue_values_after": pending_after.saturating_mul(max_gradient_dim),
        })
    });
}

#[derive(Debug)]
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
            max_hops: max_hops.max(1),
            tolerance,
        })
    }

    pub fn register_agent(&mut self, config: AgentConfig) -> Result<(), AutogradError> {
        if self.agents.contains_key(&config.id) {
            return Err(AutogradError::DuplicateAgent(config.id));
        }
        if config.learning_rate <= 0.0 || !config.learning_rate.is_finite() {
            return Err(AutogradError::NonPositiveLearningRate(config.learning_rate));
        }
        Self::validate_finite_slice("damping", &[config.damping])?;
        if config.damping < 0.0 {
            return Err(AutogradError::NegativeDamping(config.damping));
        }
        Self::validate_finite_slice("weights", &config.weights)?;
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
        let mut processed = 0usize;
        let mut per_round = self.pending.len();
        let capture_meta = tensor_op_meta_observer_installed();
        let mut stats = RoundStats {
            pending_before: per_round,
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
        emit_autograd_round_meta(self, stats, self.pending.len(), capture_meta);
        Ok(processed)
    }

    pub fn drain(&mut self) -> Result<usize, AutogradError> {
        let mut total = 0usize;
        while !self.pending.is_empty() {
            total += self.propagate_round()?;
        }
        Ok(total)
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
        if let Some(agent) = self.agents.get_mut(&message.to) {
            if agent.weights.len() != message.payload.len() {
                return Err(AutogradError::GradientDimension {
                    expected: agent.weights.len(),
                    got: message.payload.len(),
                });
            }
            Self::validate_finite_slice("message_gradient", &message.payload)?;

            // Apply the local weight update once before deciding whether to
            // forward the gradient. The previous implementation updated the
            // weights twice which effectively doubled the learning rate.
            let mut next_weights = Vec::with_capacity(agent.weights.len());
            let mut update_l2 = 0.0f32;
            for (idx, (w, g)) in agent.weights.iter().zip(message.payload.iter()).enumerate() {
                let delta = agent.learning_rate * g;
                let next = *w - delta;
                if !delta.is_finite() {
                    return Err(AutogradError::NonFiniteValue {
                        label: "weight_delta",
                        index: idx,
                        value: delta,
                    });
                }
                if !next.is_finite() {
                    return Err(AutogradError::NonFiniteValue {
                        label: "weight_update",
                        index: idx,
                        value: next,
                    });
                }
                if capture_meta {
                    update_l2 += delta * delta;
                }
                next_weights.push(next);
            }
            if capture_meta {
                outcome.update_l2_sum = update_l2.sqrt();
            }

            let signal = if capture_meta {
                l1_signal(&message.payload)
            } else {
                message.payload.iter().map(|v| v.abs() as f64).sum::<f64>() as f32
            };
            outcome.signal_sum = signal;
            if signal < self.tolerance && message.hops > 0 {
                agent.weights = next_weights;
                outcome.absorbed_tolerance = 1;
                return Ok(outcome);
            }

            if message.hops >= self.max_hops {
                agent.weights = next_weights;
                outcome.stopped_max_hops = 1;
                return Ok(outcome);
            }

            let forwarded: Vec<f32> = message.payload.iter().map(|g| g * agent.damping).collect();
            Self::validate_finite_slice("forwarded_gradient", &forwarded)?;
            let forwarded_signal = if capture_meta {
                l1_signal(&forwarded)
            } else {
                forwarded.iter().map(|v| v.abs() as f64).sum::<f64>() as f32
            };
            outcome.forwarded_signal_sum = forwarded_signal;
            if forwarded_signal < self.tolerance {
                agent.weights = next_weights;
                outcome.absorbed_tolerance = 1;
                return Ok(outcome);
            }

            agent.weights = next_weights;
            for &neighbor in &agent.neighbors {
                if neighbor == message.from {
                    continue;
                }
                self.pending.push_back(GradientMessage {
                    from: message.to,
                    to: neighbor,
                    hops: message.hops + 1,
                    payload: forwarded.clone(),
                });
                outcome.forwarded += 1;
            }
            if outcome.forwarded == 0 {
                outcome.terminal_no_neighbor = 1;
            }
        } else {
            outcome.unknown_targets = 1;
        }
        Ok(outcome)
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
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
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
        st_tensor::set_tensor_op_meta_observer(previous);

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
        assert_eq!(meta.1["agents"], 3);
        assert_eq!(meta.1["message_queue_backend"], "cpu_vecdeque");
        assert_eq!(meta.1["agent_state_backend"], "cpu_hashmap");
        assert_eq!(meta.1["weight_update_backend"], "cpu_loop");
        assert_eq!(meta.1["forwarding_backend"], "cpu_vec_clone");
        assert_eq!(meta.1["propagation_mode"], "stateful_agent_message_passing");
        assert_eq!(
            meta.1["route_blocker"],
            "stateful_agent_graph_and_message_queue"
        );
        assert_eq!(meta.1["max_gradient_dim"], 2);
        assert_eq!(meta.1["pending_before"], 1);
        assert_eq!(meta.1["pending_after"], 2);
        assert_eq!(meta.1["forwarded"], 2);
        assert_eq!(meta.1["estimated_weight_update_values"], 2);
        assert_eq!(meta.1["estimated_forwarded_gradient_values"], 4);
        assert_eq!(meta.1["estimated_queue_values_after"], 4);
        assert!(meta.1["signal_sum"].as_f64().unwrap() > 0.0);
        assert!(meta.1["update_l2_sum"].as_f64().unwrap() > 0.0);
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
            mesh.register_agent(AgentConfig::new(1, 1).with_weights(vec![f32::INFINITY]))
                .unwrap_err(),
            AutogradError::NonFiniteValue {
                label: "weights",
                index: 0,
                value: f32::INFINITY,
            }
        );
        assert_eq!(mesh.weights(1).unwrap_err(), AutogradError::UnknownAgent(1));
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
    }

    #[test]
    fn forward_overflow_does_not_mutate_weights() {
        let mut mesh = AmebaAutograd::new(2, 1e-6).unwrap();
        mesh.register_agent(
            AgentConfig::new(1, 1)
                .with_neighbors(vec![2])
                .with_learning_rate(0.1)
                .with_damping(f32::MAX)
                .with_weights(vec![1.0]),
        )
        .unwrap();
        mesh.register_agent(AgentConfig::new(2, 1)).unwrap();
        mesh.seed_gradient(1, vec![2.0]).unwrap();

        let err = mesh.drain().unwrap_err();
        assert_eq!(
            err,
            AutogradError::NonFiniteValue {
                label: "forwarded_gradient",
                index: 0,
                value: f32::INFINITY,
            }
        );
        assert_eq!(mesh.weights(1).unwrap(), &[1.0]);
        assert_eq!(mesh.weights(2).unwrap(), &[0.0]);
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
