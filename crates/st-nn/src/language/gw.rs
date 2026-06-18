// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::geometry::{SemanticBridge, SparseKernel};
use crate::execution::current_tensor_util_backend_for_values;
use crate::PureResult;
use st_core::coop::ai::{CoopAgent, CoopProposal};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, Tensor, TensorError, TensorUtilBackend};
use std::collections::HashSet;

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

fn scale_probability_values(values: Vec<f32>, scale: f32) -> (Vec<f32>, &'static str) {
    let backend = current_tensor_util_backend_for_values(values.len());
    let backend_label = tensor_util_backend_label(backend);
    if values.is_empty() {
        return (values, backend_label);
    }

    let fallback_values = values.clone();
    match Tensor::from_vec(1, values.len(), values)
        .and_then(|tensor| tensor.scale_with_backend(scale, backend))
    {
        Ok(tensor) => (tensor.data().to_vec(), backend_label),
        Err(_) => {
            let mut values = fallback_values;
            for value in &mut values {
                *value *= scale;
            }
            (values, "cpu")
        }
    }
}

fn sum_probability_values(values: &[f32]) -> (f32, &'static str) {
    let backend = current_tensor_util_backend_for_values(values.len());
    let backend_label = tensor_util_backend_label(backend);
    if values.is_empty() {
        return (0.0, backend_label);
    }

    match Tensor::from_vec(1, values.len(), values.to_vec())
        .and_then(|tensor| tensor.sum_abs_with_backend(backend))
    {
        Ok(sum) => (sum, backend_label),
        Err(_) => (values.iter().copied().sum(), "cpu"),
    }
}

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
        let a_target = normalise("gw_symbol_marginal", symbol_marginal)?;
        let b_target = normalise("gw_concept_marginal", concept_marginal)?;
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
        normalise_in_place("gw_row_sums", &mut row_sums);
        normalise_in_place("gw_col_sums", &mut col_sums);

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

fn normalise(label: &'static str, weights: &[f32]) -> PureResult<Vec<f32>> {
    if weights.is_empty() {
        return Err(TensorError::InvalidValue {
            label: "marginals cannot be empty",
        });
    }
    for &w in weights {
        if w < 0.0 || !w.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "marginals must be finite and non-negative",
            });
        }
    }
    let (total, marginal_sum_backend) = sum_probability_values(weights);
    if total <= f32::EPSILON {
        return Err(TensorError::InvalidValue {
            label: "marginal sum must be positive",
        });
    }
    let (normalised, distribution_scale_backend) =
        scale_probability_values(weights.to_vec(), 1.0 / total);
    let distribution_sum = normalised.iter().copied().sum::<f32>();
    let dominant_probability = normalised
        .iter()
        .copied()
        .fold(0.0f32, |best, value| best.max(value));
    emit_tensor_op(
        "gw_marginal_normalise",
        &[weights.len()],
        &[normalised.len()],
    );
    emit_tensor_op_meta("gw_marginal_normalise", || {
        serde_json::json!({
            "backend": "hybrid",
            "requested_backend": distribution_scale_backend,
            "kind": "gromov_wasserstein_marginal_normalise",
            "marginal_sum_backend": marginal_sum_backend,
            "distribution_scale_backend": distribution_scale_backend,
            "label": label,
            "values": weights.len(),
            "raw_sum": total,
            "distribution_sum": distribution_sum,
            "dominant_probability": dominant_probability,
            "zero_fallback": false,
        })
    });
    Ok(normalised)
}

fn normalise_in_place(label: &'static str, weights: &mut [f32]) {
    let (total, marginal_sum_backend) = sum_probability_values(weights);
    if total <= f32::EPSILON {
        emit_tensor_op(
            "gw_marginal_normalise_in_place",
            &[weights.len()],
            &[weights.len()],
        );
        emit_tensor_op_meta("gw_marginal_normalise_in_place", || {
            serde_json::json!({
                "backend": "probability_cpu",
                "requested_backend": "auto",
                "kind": "gromov_wasserstein_marginal_normalise_in_place",
                "label": label,
                "values": weights.len(),
                "raw_sum": total,
                "distribution_sum": weights.iter().copied().sum::<f32>(),
                "dominant_probability": 0.0f32,
                "zero_fallback": true,
            })
        });
        return;
    }
    let (scaled, distribution_scale_backend) =
        scale_probability_values(weights.to_vec(), 1.0 / total);
    for (value, scaled) in weights.iter_mut().zip(scaled.into_iter()) {
        *value = scaled;
    }
    let distribution_sum = weights.iter().copied().sum::<f32>();
    let dominant_probability = weights
        .iter()
        .copied()
        .fold(0.0f32, |best, value| best.max(value));
    emit_tensor_op(
        "gw_marginal_normalise_in_place",
        &[weights.len()],
        &[weights.len()],
    );
    emit_tensor_op_meta("gw_marginal_normalise_in_place", || {
        serde_json::json!({
            "backend": "hybrid",
            "requested_backend": distribution_scale_backend,
            "kind": "gromov_wasserstein_marginal_normalise_in_place",
            "marginal_sum_backend": marginal_sum_backend,
            "distribution_scale_backend": distribution_scale_backend,
            "label": label,
            "values": weights.len(),
            "raw_sum": total,
            "distribution_sum": distribution_sum,
            "dominant_probability": dominant_probability,
            "zero_fallback": false,
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::DeviceCaps;
    #[cfg(feature = "wgpu")]
    use st_tensor::backend::wgpu_dense;
    use std::sync::{Arc, Mutex, OnceLock};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        static OBSERVER_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        OBSERVER_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("observer lock available")
    }

    #[cfg(feature = "wgpu")]
    fn restore_tensor_util_wgpu_min_values(previous: Option<String>) {
        const KEY: &str = "SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES";
        if let Some(value) = previous {
            std::env::set_var(KEY, value);
        } else {
            std::env::remove_var(KEY);
        }
    }

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

    #[test]
    fn gw_marginal_normalise_paths_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let normalised = normalise("gw_test_marginal", &[0.25, 0.75]).unwrap();
        let mut in_place = vec![2.0, 1.0, 1.0];
        normalise_in_place("gw_test_in_place", &mut in_place);
        st_tensor::set_tensor_op_meta_observer(previous);

        assert!((normalised.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((in_place.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let events = events.lock().unwrap();
        let marginal = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "gw_marginal_normalise" && data["label"] == "gw_test_marginal"
            })
            .expect("gw_marginal_normalise metadata event");
        assert_eq!(marginal.1["backend"], "hybrid");
        assert_eq!(marginal.1["requested_backend"], "auto");
        assert_eq!(marginal.1["kind"], "gromov_wasserstein_marginal_normalise");
        assert_eq!(marginal.1["marginal_sum_backend"], "auto");
        assert_eq!(marginal.1["distribution_scale_backend"], "auto");
        assert_eq!(marginal.1["values"], 2);
        assert_eq!(marginal.1["zero_fallback"], false);
        assert!((marginal.1["distribution_sum"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-6);

        let row = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "gw_marginal_normalise_in_place" && data["label"] == "gw_test_in_place"
            })
            .expect("gw_marginal_normalise_in_place metadata event");
        assert_eq!(row.1["backend"], "hybrid");
        assert_eq!(row.1["requested_backend"], "auto");
        assert_eq!(
            row.1["kind"],
            "gromov_wasserstein_marginal_normalise_in_place"
        );
        assert_eq!(row.1["marginal_sum_backend"], "auto");
        assert_eq!(row.1["distribution_scale_backend"], "auto");
        assert_eq!(row.1["values"], 3);
        assert_eq!(row.1["zero_fallback"], false);
        assert!((row.1["distribution_sum"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-6);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn gw_marginal_normalise_forced_wgpu_routes_sum() {
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        const KEY: &str = "SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES";
        let previous_min = std::env::var(KEY).ok();
        std::env::set_var(KEY, "0");
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let (normalised, in_place) = {
            let _guard = push_backend_policy(policy);
            let normalised = normalise("gw_test_marginal_wgpu", &[0.25, 0.75]).unwrap();
            let mut in_place = vec![2.0, 1.0, 1.0];
            normalise_in_place("gw_test_in_place_wgpu", &mut in_place);
            (normalised, in_place)
        };
        st_tensor::set_tensor_op_meta_observer(previous);
        restore_tensor_util_wgpu_min_values(previous_min);

        assert!((normalised.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((in_place.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let events = events.lock().unwrap();
        let marginal = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "gw_marginal_normalise" && data["label"] == "gw_test_marginal_wgpu"
            })
            .expect("gw_marginal_normalise metadata event");
        assert_eq!(marginal.1["backend"], "hybrid");
        assert_eq!(marginal.1["requested_backend"], "wgpu");
        assert_eq!(marginal.1["marginal_sum_backend"], "wgpu");
        assert_eq!(marginal.1["distribution_scale_backend"], "wgpu");
        let row = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "gw_marginal_normalise_in_place"
                    && data["label"] == "gw_test_in_place_wgpu"
            })
            .expect("gw_marginal_normalise_in_place metadata event");
        assert_eq!(row.1["backend"], "hybrid");
        assert_eq!(row.1["requested_backend"], "wgpu");
        assert_eq!(row.1["marginal_sum_backend"], "wgpu");
        assert_eq!(row.1["distribution_scale_backend"], "wgpu");
        let sum_abs = events
            .iter()
            .find(|(op_name, data)| *op_name == "sum_abs" && data["backend"] == "wgpu_dense")
            .expect("sum_abs WGPU marginal sum metadata event");
        assert_eq!(sum_abs.1["backend"], "wgpu_dense");
    }
}
