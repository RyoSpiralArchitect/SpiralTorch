// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use rand::{thread_rng, Rng};

use crate::{RlResult, SpiralRlError};
use st_tensor::TensorError;

/// Discrete-action Deep Q Network agent implemented with a lightweight Q-table.
#[derive(Clone, Debug)]
pub struct DqnAgent {
    state_dim: usize,
    action_dim: usize,
    discount: f32,
    learning_rate: f32,
    epsilon: f32,
    table: Vec<f32>,
}

impl DqnAgent {
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        discount: f32,
        learning_rate: f32,
    ) -> RlResult<Self> {
        if state_dim == 0 || action_dim == 0 {
            return Err(SpiralRlError::InvalidStateShape {
                expected: state_dim,
                rows: state_dim,
                cols: action_dim,
            });
        }
        if !(0.0..=1.0).contains(&discount) {
            return Err(SpiralRlError::InvalidDiscount { discount });
        }
        if learning_rate <= 0.0 {
            return Err(SpiralRlError::Tensor(
                TensorError::NonPositiveLearningRate {
                    rate: learning_rate,
                },
            ));
        }
        Ok(Self {
            state_dim,
            action_dim,
            discount,
            learning_rate,
            epsilon: 0.1,
            table: vec![0.0; state_dim * action_dim],
        })
    }

    fn q(&self, state: usize, action: usize) -> f32 {
        self.table[state * self.action_dim + action]
    }

    fn q_mut(&mut self, state: usize, action: usize) -> &mut f32 {
        &mut self.table[state * self.action_dim + action]
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon.clamp(0.0, 1.0);
    }

    pub fn select_action(&self, state: usize) -> usize {
        if thread_rng().gen::<f32>() < self.epsilon {
            thread_rng().gen_range(0..self.action_dim)
        } else {
            (0..self.action_dim)
                .max_by(|&lhs, &rhs| self.q(state, lhs).total_cmp(&self.q(state, rhs)))
                .unwrap_or(0)
        }
    }

    pub fn update(&mut self, state: usize, action: usize, reward: f32, next_state: usize) {
        let best_next = (0..self.action_dim)
            .map(|a| self.q(next_state, a))
            .fold(f32::NEG_INFINITY, f32::max);
        let target = reward + self.discount * best_next;
        let current = self.q(state, action);
        *self.q_mut(state, action) = current + self.learning_rate * (target - current);
    }
}

/// Lightweight PPO agent that keeps vector parameters for the policy and value heads.
#[derive(Clone, Debug)]
pub struct PpoAgent {
    policy_weights: Vec<f32>,
    value_weights: Vec<f32>,
    state_dim: usize,
    action_dim: usize,
    learning_rate: f32,
    clip_range: f32,
}

impl PpoAgent {
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        learning_rate: f32,
        clip_range: f32,
    ) -> RlResult<Self> {
        if state_dim == 0 || action_dim == 0 {
            return Err(SpiralRlError::InvalidStateShape {
                expected: state_dim,
                rows: state_dim,
                cols: action_dim,
            });
        }
        let mut rng = thread_rng();
        let mut policy_weights = vec![0.0f32; state_dim * action_dim];
        for weight in &mut policy_weights {
            *weight = rng.gen_range(-0.05..=0.05);
        }
        let value_weights = vec![0.0f32; state_dim];
        Ok(Self {
            policy_weights,
            value_weights,
            state_dim,
            action_dim,
            learning_rate,
            clip_range: clip_range.clamp(0.0, 1.0),
        })
    }

    fn policy_row(&self, action: usize) -> &[f32] {
        let offset = action * self.state_dim;
        &self.policy_weights[offset..offset + self.state_dim]
    }

    fn policy_row_mut(&mut self, action: usize) -> &mut [f32] {
        let offset = action * self.state_dim;
        &mut self.policy_weights[offset..offset + self.state_dim]
    }

    pub fn score_actions(&self, state: &[f32]) -> Vec<f32> {
        let mut logits = Vec::with_capacity(self.action_dim);
        for action in 0..self.action_dim {
            let mut dot = 0.0f32;
            for idx in 0..self.state_dim {
                dot += self.policy_row(action)[idx] * state[idx];
            }
            logits.push(dot);
        }
        logits
    }

    pub fn value(&self, state: &[f32]) -> f32 {
        self.value_weights
            .iter()
            .zip(state.iter())
            .map(|(w, s)| w * s)
            .sum()
    }

    pub fn update(&mut self, state: &[f32], action: usize, advantage: f32, old_log_prob: f32) {
        let logits = self.score_actions(state);
        let log_sum = logits.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let normaliser: f32 = logits
            .iter()
            .map(|logit| (logit - log_sum).exp())
            .sum::<f32>()
            .ln()
            + log_sum;
        let new_log_prob = logits[action] - normaliser;
        let ratio = (new_log_prob - old_log_prob).exp();
        let clipped = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range);
        let policy_grad = clipped * advantage;
        for idx in 0..self.state_dim {
            self.policy_row_mut(action)[idx] += self.learning_rate * policy_grad * state[idx];
        }
        for idx in 0..self.state_dim {
            self.value_weights[idx] += self.learning_rate * advantage * state[idx];
        }
    }
}

/// Simplified SAC agent that jitters a stochastic policy to track an entropy target.
#[derive(Clone, Debug)]
pub struct SacAgent {
    state_dim: usize,
    action_dim: usize,
    temperature: f32,
    policy_weights: Vec<f32>,
}

impl SacAgent {
    pub fn new(state_dim: usize, action_dim: usize, temperature: f32) -> RlResult<Self> {
        if temperature <= 0.0 {
            return Err(SpiralRlError::InvalidDiscount {
                discount: temperature,
            });
        }
        let mut rng = thread_rng();
        let mut policy_weights = vec![0.0f32; state_dim * action_dim];
        for weight in &mut policy_weights {
            *weight = rng.gen_range(-temperature..=temperature);
        }
        Ok(Self {
            state_dim,
            action_dim,
            temperature,
            policy_weights,
        })
    }

    fn row(&self, action: usize) -> &[f32] {
        let offset = action * self.state_dim;
        &self.policy_weights[offset..offset + self.state_dim]
    }

    pub fn sample_action(&self, state: &[f32]) -> usize {
        let mut scores = Vec::with_capacity(self.action_dim);
        for action in 0..self.action_dim {
            let mut dot = 0.0f32;
            for idx in 0..self.state_dim {
                dot += self.row(action)[idx] * state[idx];
            }
            scores.push((dot / self.temperature).exp());
        }
        let sum: f32 = scores.iter().sum();
        if sum <= f32::EPSILON {
            return 0;
        }
        let mut rng = thread_rng();
        let mut cumulative = 0.0f32;
        let draw: f32 = rng.gen::<f32>() * sum;
        for (idx, score) in scores.iter().enumerate() {
            cumulative += *score;
            if draw <= cumulative {
                return idx;
            }
        }
        self.action_dim - 1
    }

    pub fn jitter(&mut self, entropy_target: f32) {
        let mut rng = thread_rng();
        for weight in &mut self.policy_weights {
            let noise: f32 = rng.gen_range(-self.temperature..=self.temperature);
            *weight = (*weight + noise * entropy_target).tanh();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dqn_updates_state_action_value() {
        let mut agent = DqnAgent::new(2, 3, 0.9, 0.1).unwrap();
        let before = agent.q(0, 1);
        agent.update(0, 1, 1.0, 1);
        assert!(agent.q(0, 1) > before);
    }

    #[test]
    fn ppo_tracks_advantages() {
        let mut agent = PpoAgent::new(3, 2, 0.05, 0.2).unwrap();
        let state = vec![1.0, 0.0, -1.0];
        agent.update(&state, 0, 0.5, -0.1);
        assert!(agent.score_actions(&state)[0] != 0.0);
    }

    #[test]
    fn sac_returns_valid_action() {
        let agent = SacAgent::new(2, 3, 0.5).unwrap();
        let state = vec![0.2, -0.7];
        let action = agent.sample_action(&state);
        assert!(action < 3);
    }
}
