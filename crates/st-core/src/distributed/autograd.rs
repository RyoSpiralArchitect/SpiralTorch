use std::collections::{HashMap, VecDeque};

use crate::causal::NodeId;

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

pub struct AmebaAutograd {
    agents: HashMap<NodeId, AgentState>,
    pending: VecDeque<GradientMessage>,
    max_hops: usize,
    tolerance: f32,
}

impl AmebaAutograd {
    pub fn new(max_hops: usize, tolerance: f32) -> Result<Self, AutogradError> {
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
        while per_round > 0 {
            if let Some(message) = self.pending.pop_front() {
                processed += 1;
                per_round -= 1;
                self.handle_message(message)?;
            }
        }
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

    fn handle_message(&mut self, message: GradientMessage) -> Result<(), AutogradError> {
        if let Some(agent) = self.agents.get_mut(&message.to) {
            if agent.weights.len() != message.payload.len() {
                return Err(AutogradError::GradientDimension {
                    expected: agent.weights.len(),
                    got: message.payload.len(),
                });
            }

            let signal = message.payload.iter().map(|v| v.abs()).sum::<f32>();
            if signal < self.tolerance {
                return Ok(());
            }

            for (w, g) in agent.weights.iter_mut().zip(message.payload.iter()) {
                *w -= agent.learning_rate * g;
            }

            if message.hops >= self.max_hops {
                return Ok(());
            }

            let forwarded: Vec<f32> = message.payload.iter().map(|g| g * agent.damping).collect();
            if forwarded.iter().map(|v| v.abs()).sum::<f32>() < self.tolerance {
                return Ok(());
            }

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
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
