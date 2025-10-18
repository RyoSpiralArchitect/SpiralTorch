// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::geometry::{SemanticBridge, SparseKernel};
use crate::PureResult;
use st_core::coop::ai::{CoopAgent, CoopProposal};
use st_tensor::TensorError;
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct DistanceMatrix {
    size: usize,
    data: Vec<f32>,
}

impl DistanceMatrix {
    pub fn new(size: usize, data: Vec<f32>) -> PureResult<Self> {
        if size == 0 {
            return Err(TensorError::InvalidValue {
                label: "distance matrix must be non-empty",
            });
        }
        if data.len() != size * size {
            return Err(TensorError::InvalidValue {
                label: "distance matrix must be square",
            });
        }
        for value in &data {
            if !value.is_finite() || *value < 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "distance entries must be finite and non-negative",
                });
            }
        }
        Ok(Self { size, data })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.size + j]
    }
}

#[derive(Clone, Debug)]
pub struct EntropicGwSolver {
    pub epsilon: f32,
    pub anchor_strength: f32,
    pub max_iter: usize,
    pub sinkhorn_iter: usize,
}

impl Default for EntropicGwSolver {
    fn default() -> Self {
        Self {
            epsilon: 0.5,
            anchor_strength: 3.0,
            max_iter: 25,
            sinkhorn_iter: 40,
        }
    }
}

impl EntropicGwSolver {
    pub fn solve(
        &self,
        symbol_distances: &DistanceMatrix,
        concept_distances: &DistanceMatrix,
        symbol_marginal: &[f32],
        concept_marginal: &[f32],
        anchors: &[(usize, usize)],
        concept_kernel: SparseKernel,
    ) -> PureResult<SemanticBridge> {
        let n = symbol_distances.size();
        let m = concept_distances.size();
        if symbol_marginal.len() != n {
            return Err(TensorError::InvalidValue {
                label: "symbol marginal must match symbol distance matrix",
            });
        }
        if concept_marginal.len() != m {
            return Err(TensorError::InvalidValue {
                label: "concept marginal must match concept distance matrix",
            });
        }
        let a_target = normalise(symbol_marginal)?;
        let b_target = normalise(concept_marginal)?;
        let mut pi = vec![0.0f32; n * m];
        for i in 0..n {
            for j in 0..m {
                pi[i * m + j] = a_target[i] * b_target[j];
            }
        }
        let anchors_set: HashSet<(usize, usize)> = anchors.iter().copied().collect();
        let mut kernel = vec![0.0f32; n * m];
        let mut cost_buffer = vec![0.0f32; n * m];
        let anchor_boost = (self.anchor_strength / self.epsilon).exp();

        for _ in 0..self.max_iter {
            compute_gw_cost(symbol_distances, concept_distances, &pi, &mut cost_buffer);
            for i in 0..n {
                for j in 0..m {
                    let idx = i * m + j;
                    let mut value = (-cost_buffer[idx] / self.epsilon).exp();
                    if anchors_set.contains(&(i, j)) {
                        value *= anchor_boost;
                    }
                    kernel[idx] = value.max(1e-12);
                }
            }

            sinkhorn(&mut pi, &kernel, &a_target, &b_target, self.sinkhorn_iter);
        }

        let mut log_pi_rows = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::new();
            for j in 0..m {
                let prob = pi[i * m + j].max(1e-9);
                row.push((j, prob.ln()));
            }
            log_pi_rows.push(row);
        }

        let mut row_sums = vec![0.0f32; n];
        let mut col_sums = vec![0.0f32; m];
        for i in 0..n {
            for j in 0..m {
                let value = pi[i * m + j];
                row_sums[i] += value;
                col_sums[j] += value;
            }
        }
        normalise_in_place(&mut row_sums);
        normalise_in_place(&mut col_sums);

        let bridge = SemanticBridge::new(
            log_pi_rows,
            row_sums,
            col_sums,
            anchors_set,
            1e-6,
            concept_kernel,
        )?;
        Ok(bridge)
    }
}

impl CoopAgent for EntropicGwSolver {
    fn propose(&mut self) -> CoopProposal {
        let z_bias = (self.anchor_strength - self.epsilon).tanh();
        let weight = (self.anchor_strength + self.epsilon).max(1e-3);
        CoopProposal::new(z_bias, weight)
    }

    fn observe(&mut self, team_reward: f32, credit: f32) {
        let credit_push = credit.tanh();
        let reward_push = team_reward.tanh();

        if credit_push >= 0.0 {
            self.anchor_strength =
                (self.anchor_strength * (1.0 + 0.12 * credit_push)).clamp(0.5, 24.0);
            self.epsilon = (self.epsilon * (1.0 - 0.04 * credit_push)).clamp(1e-3, 8.0);
        } else {
            let neg = -credit_push;
            self.anchor_strength = (self.anchor_strength * (1.0 - 0.05 * neg)).clamp(0.5, 24.0);
            self.epsilon = (self.epsilon * (1.0 + 0.1 * neg)).clamp(1e-3, 8.0);
        }

        self.epsilon = (self.epsilon * (1.0 + (-0.03 * reward_push))).clamp(1e-3, 8.0);
    }
}

#[cfg(test)]
mod coop_tests {
    use super::*;

    #[test]
    fn gw_solver_coop_agent_balances_parameters() {
        let mut solver = EntropicGwSolver::default();
        let proposal = CoopAgent::propose(&mut solver);
        assert!(proposal.weight > 0.0);

        let before_anchor = solver.anchor_strength;
        let before_eps = solver.epsilon;
        CoopAgent::observe(&mut solver, -0.4, 0.6);
        assert!(solver.anchor_strength >= before_anchor);
        let mid_eps = solver.epsilon;
        assert!(mid_eps <= before_eps);

        CoopAgent::observe(&mut solver, 0.2, -0.7);
        assert!(solver.epsilon >= mid_eps);
    }
}

fn compute_gw_cost(symbol: &DistanceMatrix, concept: &DistanceMatrix, pi: &[f32], out: &mut [f32]) {
    let n = symbol.size();
    let m = concept.size();
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0f32;
            for ip in 0..n {
                let d_s = symbol.at(i, ip);
                for jp in 0..m {
                    let d_z = concept.at(j, jp);
                    let diff = d_s - d_z;
                    sum += diff * diff * pi[ip * m + jp];
                }
            }
            out[i * m + j] = sum;
        }
    }
}

fn sinkhorn(pi: &mut [f32], kernel: &[f32], a_target: &[f32], b_target: &[f32], iterations: usize) {
    let n = a_target.len();
    let m = b_target.len();
    let mut u = vec![1.0f32; n];
    let mut v = vec![1.0f32; m];
    for _ in 0..iterations {
        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..m {
                sum += kernel[i * m + j] * v[j];
            }
            if sum > 0.0 {
                u[i] = a_target[i] / sum;
            }
        }
        for j in 0..m {
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += kernel[i * m + j] * u[i];
            }
            if sum > 0.0 {
                v[j] = b_target[j] / sum;
            }
        }
    }

    for i in 0..n {
        for j in 0..m {
            pi[i * m + j] = kernel[i * m + j] * u[i] * v[j];
        }
    }
}

fn normalise(weights: &[f32]) -> PureResult<Vec<f32>> {
    if weights.is_empty() {
        return Err(TensorError::InvalidValue {
            label: "marginals cannot be empty",
        });
    }
    let mut total = 0.0f32;
    for &w in weights {
        if w < 0.0 || !w.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "marginals must be finite and non-negative",
            });
        }
        total += w;
    }
    if total <= f32::EPSILON {
        return Err(TensorError::InvalidValue {
            label: "marginal sum must be positive",
        });
    }
    Ok(weights.iter().map(|w| w / total).collect())
}

fn normalise_in_place(weights: &mut [f32]) {
    let mut total = 0.0f32;
    for &w in weights.iter() {
        total += w;
    }
    if total <= f32::EPSILON {
        return;
    }
    for value in weights.iter_mut() {
        *value /= total;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gw_solver_respects_anchors() {
        let symbol_dist = DistanceMatrix::new(2, vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let concept_dist = DistanceMatrix::new(2, vec![0.0, 0.5, 0.5, 0.0]).unwrap();
        let symbol_marg = vec![0.6, 0.4];
        let concept_marg = vec![0.5, 0.5];
        let anchors = vec![(0, 0)];
        let kernel = SparseKernel::from_rows(vec![vec![(0, 1.0)], vec![(1, 1.0)]], 1e-6).unwrap();
        let solver = EntropicGwSolver {
            epsilon: 0.3,
            anchor_strength: 5.0,
            max_iter: 5,
            sinkhorn_iter: 10,
        };
        let bridge = solver
            .solve(
                &symbol_dist,
                &concept_dist,
                &symbol_marg,
                &concept_marg,
                &anchors,
                kernel,
            )
            .unwrap();
        let row = &bridge.log_rows()[0];
        let mut anchor = 0.0f32;
        let mut other = 0.0f32;
        for &(concept, logp) in row {
            if concept == 0 {
                anchor = logp.exp();
            } else {
                other = logp.exp();
            }
        }
        assert!(anchor > other);
    }
}
