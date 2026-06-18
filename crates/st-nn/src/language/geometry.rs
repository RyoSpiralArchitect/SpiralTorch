// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::PureResult;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, Tensor, TensorError, TensorUtilBackend};
use std::collections::HashSet;

const DEFAULT_EPSILON: f32 = 1e-6;

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

fn scale_semantic_distribution(values: Vec<f32>, scale: f32) -> (Vec<f32>, &'static str) {
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

fn sum_semantic_distribution(values: &[f32]) -> (f32, &'static str) {
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

struct SemanticWindowAccumulation {
    accum: Vec<f32>,
    total: f32,
    skipped_tokens: usize,
    skipped_non_positive_weights: usize,
    contributions: usize,
    backend: &'static str,
}

#[derive(Clone, Debug)]
pub(crate) struct SemanticWindowDistributionReport {
    pub(crate) distribution: Vec<f32>,
    pub(crate) backend: &'static str,
    pub(crate) requested_backend: &'static str,
    pub(crate) semantic_sparse_scan_backend: &'static str,
    pub(crate) semantic_accumulation_backend: &'static str,
    pub(crate) distribution_scale_backend: &'static str,
    pub(crate) inference_mode: &'static str,
    pub(crate) route_blocker: &'static str,
    pub(crate) raw_total: f32,
    pub(crate) contributions: usize,
    pub(crate) skipped_tokens: usize,
    pub(crate) skipped_non_positive_weights: usize,
    pub(crate) uniform_fallback: bool,
    pub(crate) empty_window: bool,
}

fn accumulate_semantic_window(
    log_pi: &[Vec<(usize, f32)>],
    concepts: usize,
    window: &[(usize, f32)],
    smoothing: f32,
) -> SemanticWindowAccumulation {
    let mut skipped_tokens = 0usize;
    let mut skipped_non_positive_weights = 0usize;
    let mut contributions = 0usize;
    let mut cpu_accum = vec![smoothing; concepts];
    let mut dense_rows = cpu_accum.clone();
    let mut row_count = 1usize;

    for &(token, weight) in window {
        let Some(row) = log_pi.get(token) else {
            skipped_tokens = skipped_tokens.saturating_add(1);
            continue;
        };
        if weight <= 0.0 {
            skipped_non_positive_weights = skipped_non_positive_weights.saturating_add(1);
            continue;
        }
        let mut dense_row = vec![0.0f32; concepts];
        for &(concept, logp) in row {
            let value = weight * logp.exp();
            dense_row[concept] += value;
            cpu_accum[concept] += value;
            contributions = contributions.saturating_add(1);
        }
        dense_rows.extend_from_slice(&dense_row);
        row_count = row_count.saturating_add(1);
    }

    let backend = current_tensor_util_backend_for_values(dense_rows.len());
    let backend_label = tensor_util_backend_label(backend);
    let (accum, backend_label) = match Tensor::from_vec(row_count, concepts, dense_rows)
        .and_then(|tensor| tensor.try_sum_axis0_with_backend(backend))
    {
        Ok(accum) => (accum, backend_label),
        Err(_) => (cpu_accum, "cpu"),
    };
    let total = accum.iter().copied().sum::<f32>();
    SemanticWindowAccumulation {
        accum,
        total,
        skipped_tokens,
        skipped_non_positive_weights,
        contributions,
        backend: backend_label,
    }
}

struct DistributionSanitise {
    values: Vec<f32>,
    clipped_negative: usize,
    non_finite: usize,
    backend: &'static str,
}

fn sanitise_distribution_hint_values(dist: &[f32], concepts: usize) -> DistributionSanitise {
    let mut buffer = vec![DEFAULT_EPSILON; concepts];
    let mut clipped_negative = 0usize;
    let mut non_finite = 0usize;
    for (idx, value) in dist.iter().copied().enumerate().take(concepts) {
        if !value.is_finite() {
            non_finite = non_finite.saturating_add(1);
        } else if value < 0.0 {
            clipped_negative = clipped_negative.saturating_add(1);
        }
        buffer[idx] = value;
    }

    if non_finite > 0 {
        for value in &mut buffer {
            *value = value.max(0.0);
        }
        return DistributionSanitise {
            values: buffer,
            clipped_negative,
            non_finite,
            backend: "semantic_cpu",
        };
    }

    let backend = current_tensor_util_backend_for_values(buffer.len());
    let backend_label = tensor_util_backend_label(backend);
    let fallback = buffer
        .iter()
        .copied()
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    let (values, backend_label) = match Tensor::from_vec(1, concepts, buffer)
        .and_then(|tensor| tensor.relu_with_backend(backend))
    {
        Ok(tensor) => (tensor.data().to_vec(), backend_label),
        Err(_) => (fallback, "cpu"),
    };
    DistributionSanitise {
        values,
        clipped_negative,
        non_finite,
        backend: backend_label,
    }
}

fn ensure_probabilities(
    row_index: usize,
    weights: &[(usize, f32)],
    epsilon: f32,
) -> PureResult<Vec<(usize, f32)>> {
    if weights.is_empty() {
        emit_tensor_op("sparse_kernel_probability_row", &[0], &[0]);
        emit_tensor_op_meta("sparse_kernel_probability_row", || {
            serde_json::json!({
                "backend": "probability_cpu",
                "requested_backend": "auto",
                "kind": "language_sparse_kernel_probability_row",
                "row_index": row_index,
                "entries": 0,
                "epsilon": epsilon,
                "raw_sum": 0.0f32,
                "smoothing": 0.0f32,
                "denominator": epsilon,
                "distribution_sum": 0.0f32,
                "dominant_probability": 0.0f32,
                "empty": true,
            })
        });
        return Ok(Vec::new());
    }
    let mut raw_values = Vec::with_capacity(weights.len());
    for &(_, w) in weights {
        if w.is_nan() || w.is_infinite() {
            return Err(TensorError::InvalidValue {
                label: "kernel weight must be finite",
            });
        }
        if w < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "kernel weight must be non-negative",
            });
        }
        raw_values.push(w);
    }
    let (total, row_sum_backend) = sum_semantic_distribution(&raw_values);
    let smoothing = epsilon * weights.len() as f32;
    let denom = (total + smoothing).max(epsilon);
    let smoothed = weights
        .iter()
        .map(|(_, weight)| *weight + epsilon)
        .collect::<Vec<_>>();
    let (scaled, distribution_scale_backend) = scale_semantic_distribution(smoothed, 1.0 / denom);
    let normalised = weights
        .iter()
        .map(|(j, _)| *j)
        .zip(scaled.into_iter().map(|value| value.max(epsilon)))
        .collect::<Vec<_>>();
    let distribution_sum = normalised.iter().map(|(_, prob)| *prob).sum::<f32>();
    let dominant_probability = normalised
        .iter()
        .map(|(_, prob)| *prob)
        .fold(0.0f32, |best, value| best.max(value));
    emit_tensor_op(
        "sparse_kernel_probability_row",
        &[weights.len()],
        &[normalised.len()],
    );
    emit_tensor_op_meta("sparse_kernel_probability_row", || {
        serde_json::json!({
            "backend": "hybrid",
            "requested_backend": distribution_scale_backend,
            "kind": "language_sparse_kernel_probability_row",
            "row_sum_backend": row_sum_backend,
            "distribution_scale_backend": distribution_scale_backend,
            "row_index": row_index,
            "entries": weights.len(),
            "epsilon": epsilon,
            "raw_sum": total,
            "smoothing": smoothing,
            "denominator": denom,
            "distribution_sum": distribution_sum,
            "dominant_probability": dominant_probability,
            "empty": false,
        })
    });
    Ok(normalised)
}

#[derive(Clone, Debug)]
pub struct SparseKernel {
    rows: Vec<Vec<(usize, f32)>>,
    log_epsilon: f32,
}

impl SparseKernel {
    pub fn from_rows(rows: Vec<Vec<(usize, f32)>>, epsilon: f32) -> PureResult<Self> {
        let mut norm_rows = Vec::with_capacity(rows.len());
        for (row_index, row) in rows.into_iter().enumerate() {
            let normalised = ensure_probabilities(row_index, &row, epsilon)?;
            let mut log_row = Vec::with_capacity(normalised.len());
            for (j, prob) in normalised {
                log_row.push((j, prob.ln()));
            }
            norm_rows.push(log_row);
        }
        Ok(Self {
            rows: norm_rows,
            log_epsilon: epsilon.ln(),
        })
    }

    pub fn from_dense(dense: Vec<Vec<f32>>, epsilon: f32) -> PureResult<Self> {
        let mut rows = Vec::with_capacity(dense.len());
        for row in dense {
            let mut entries = Vec::new();
            for (idx, weight) in row.into_iter().enumerate() {
                if weight.is_nan() || weight.is_infinite() {
                    return Err(TensorError::InvalidValue {
                        label: "kernel weight must be finite",
                    });
                }
                if weight <= 0.0 {
                    continue;
                }
                entries.push((idx, weight));
            }
            rows.push(entries);
        }
        Self::from_rows(rows, epsilon)
    }

    pub fn vocab_size(&self) -> usize {
        self.rows.len()
    }

    pub fn row(&self, idx: usize) -> &[(usize, f32)] {
        self.rows.get(idx).map(|r| r.as_slice()).unwrap_or(&[])
    }

    pub fn log_value(&self, i: usize, j: usize) -> f32 {
        for &(col, logv) in self.row(i) {
            if col == j {
                return logv;
            }
        }
        self.log_epsilon
    }

    pub fn to_dense(&self) -> Vec<Vec<f32>> {
        let size = self.vocab_size();
        let mut dense = vec![vec![self.log_epsilon.exp(); size]; size];
        for (i, row) in self.rows.iter().enumerate() {
            for &(j, logv) in row {
                dense[i][j] = logv.exp();
            }
        }
        dense
    }
}

#[derive(Clone, Debug)]
pub struct SymbolGeometry {
    syn: SparseKernel,
    par: SparseKernel,
}

impl SymbolGeometry {
    pub fn new(syn: SparseKernel, par: SparseKernel) -> PureResult<Self> {
        if syn.vocab_size() != par.vocab_size() {
            return Err(TensorError::InvalidValue {
                label: "syntactic and paradigm kernels must share vocabulary",
            });
        }
        Ok(Self { syn, par })
    }

    pub fn vocab_size(&self) -> usize {
        self.syn.vocab_size()
    }

    pub fn log_syn(&self, i: usize, j: usize) -> f32 {
        self.syn.log_value(i, j)
    }

    pub fn log_par(&self, i: usize, j: usize) -> f32 {
        self.par.log_value(i, j)
    }

    pub fn syn_row(&self, idx: usize) -> &[(usize, f32)] {
        self.syn.row(idx)
    }

    pub fn par_row(&self, idx: usize) -> &[(usize, f32)] {
        self.par.row(idx)
    }
}

#[derive(Clone, Debug)]
pub struct RepressionField {
    values: Vec<f32>,
}

impl RepressionField {
    pub fn new(values: Vec<f32>) -> PureResult<Self> {
        for &v in &values {
            if v.is_nan() || v.is_infinite() {
                return Err(TensorError::InvalidValue {
                    label: "repression must be finite",
                });
            }
            if v < 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "repression values must be non-negative",
                });
            }
        }
        Ok(Self { values })
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn value(&self, idx: usize) -> f32 {
        self.values.get(idx).copied().unwrap_or(0.0)
    }
}

#[derive(Clone, Debug)]
pub struct SemanticBridge {
    log_pi: Vec<Vec<(usize, f32)>>,
    pub(crate) row_sums: Vec<f32>,
    pub(crate) col_sums: Vec<f32>,
    anchors: HashSet<(usize, usize)>,
    log_epsilon: f32,
    concept_kernel: SparseKernel,
}

impl SemanticBridge {
    pub fn new(
        log_pi: Vec<Vec<(usize, f32)>>,
        row_sums: Vec<f32>,
        col_sums: Vec<f32>,
        anchors: HashSet<(usize, usize)>,
        epsilon: f32,
        concept_kernel: SparseKernel,
    ) -> PureResult<Self> {
        let vocab = log_pi.len();
        if vocab == 0 {
            return Err(TensorError::InvalidValue {
                label: "semantic bridge requires at least one symbol",
            });
        }
        if row_sums.len() != vocab {
            return Err(TensorError::InvalidValue {
                label: "row marginals must match symbol vocabulary",
            });
        }
        if col_sums.is_empty() {
            return Err(TensorError::InvalidValue {
                label: "concept marginal cannot be empty",
            });
        }
        if concept_kernel.vocab_size() != col_sums.len() {
            return Err(TensorError::InvalidValue {
                label: "concept kernel must match semantic nodes",
            });
        }
        for row in &log_pi {
            for &(k, logv) in row {
                if k >= col_sums.len() {
                    return Err(TensorError::InvalidValue {
                        label: "concept index out of range",
                    });
                }
                if !logv.is_finite() {
                    return Err(TensorError::InvalidValue {
                        label: "log coupling must be finite",
                    });
                }
            }
        }
        Ok(Self {
            log_pi,
            row_sums,
            col_sums,
            anchors,
            log_epsilon: epsilon.ln(),
            concept_kernel,
        })
    }

    pub fn from_dense(
        couplings: Vec<Vec<f32>>,
        anchors: impl IntoIterator<Item = (usize, usize)>,
        epsilon: f32,
        concept_kernel: SparseKernel,
    ) -> PureResult<Self> {
        let tokens = couplings.len();
        if tokens == 0 {
            return Err(TensorError::InvalidValue {
                label: "semantic bridge requires at least one symbol",
            });
        }
        let concepts = concept_kernel.vocab_size();
        if concepts == 0 {
            return Err(TensorError::InvalidValue {
                label: "semantic bridge requires at least one concept",
            });
        }
        let mut log_rows = Vec::with_capacity(tokens);
        let mut row_sums = Vec::with_capacity(tokens);
        let mut col_sums = vec![0.0f32; concepts];
        for row in couplings.into_iter() {
            if row.len() != concepts {
                return Err(TensorError::InvalidValue {
                    label: "semantic bridge row must match concept kernel",
                });
            }
            let mut entries = Vec::new();
            let mut sum = 0.0f32;
            for (concept_idx, weight) in row.into_iter().enumerate() {
                if weight.is_nan() || weight.is_infinite() {
                    return Err(TensorError::InvalidValue {
                        label: "semantic bridge weight must be finite",
                    });
                }
                if weight <= 0.0 {
                    continue;
                }
                entries.push((concept_idx, weight.ln()));
                sum += weight;
                col_sums[concept_idx] += weight;
            }
            row_sums.push(if sum <= 0.0 {
                epsilon.max(f32::EPSILON)
            } else {
                sum
            });
            log_rows.push(entries);
        }
        for value in &mut col_sums {
            if *value <= 0.0 {
                *value = epsilon.max(f32::EPSILON);
            }
        }
        let anchors: HashSet<(usize, usize)> = anchors.into_iter().collect();
        for &(token, concept) in &anchors {
            if token >= tokens {
                return Err(TensorError::InvalidValue {
                    label: "semantic bridge anchor token out of range",
                });
            }
            if concept >= concepts {
                return Err(TensorError::InvalidValue {
                    label: "semantic bridge anchor concept out of range",
                });
            }
        }
        Self::new(
            log_rows,
            row_sums,
            col_sums,
            anchors,
            epsilon,
            concept_kernel,
        )
    }

    pub fn vocab_size(&self) -> usize {
        self.log_pi.len()
    }

    pub fn concept_count(&self) -> usize {
        self.col_sums.len()
    }

    pub fn log_pi(&self, token: usize, concept: usize) -> f32 {
        for &(idx, logp) in self.log_pi.get(token).map(|r| r.as_slice()).unwrap_or(&[]) {
            if idx == concept {
                return logp;
            }
        }
        self.log_epsilon
    }

    pub fn expectation(&self, token: usize, concept_dist: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        let mut total = 0.0f32;
        for (k, weight) in concept_dist.iter().enumerate() {
            let clamped = weight.max(0.0);
            if clamped <= f32::EPSILON {
                continue;
            }
            let logp = self.log_pi(token, k);
            acc += clamped * logp;
            total += clamped;
        }
        if total <= f32::EPSILON {
            self.log_epsilon
        } else {
            (acc / total).max(self.log_epsilon)
        }
    }

    pub fn anchors(&self) -> &HashSet<(usize, usize)> {
        &self.anchors
    }

    pub(crate) fn infer_from_window_report(
        &self,
        window: &[(usize, f32)],
        smoothing: f32,
    ) -> SemanticWindowDistributionReport {
        let concepts = self.concept_count();
        let accumulation = accumulate_semantic_window(&self.log_pi, concepts, window, smoothing);
        let accum = accumulation.accum;
        let total = accumulation.total;
        let skipped_tokens = accumulation.skipped_tokens;
        let skipped_non_positive_weights = accumulation.skipped_non_positive_weights;
        let contributions = accumulation.contributions;
        let semantic_accumulation_backend = accumulation.backend;
        if total <= f32::EPSILON {
            let distribution = vec![1.0 / concepts as f32; concepts];
            let distribution_sum = distribution.iter().copied().sum::<f32>();
            let dominant_probability = distribution
                .iter()
                .copied()
                .fold(0.0f32, |best, value| best.max(value));
            emit_tensor_op(
                "semantic_bridge_window_distribution",
                &[window.len(), self.vocab_size(), concepts],
                &[1, distribution.len()],
            );
            emit_tensor_op_meta("semantic_bridge_window_distribution", || {
                serde_json::json!({
                    "backend": "semantic_cpu",
                    "requested_backend": "auto",
                    "kind": "language_semantic_bridge_window_distribution",
                    "semantic_sparse_scan_backend": "semantic_cpu",
                    "semantic_accumulation_backend": semantic_accumulation_backend,
                    "distribution_scale_backend": "none",
                    "inference_mode": "uniform_fallback",
                    "route_blocker": "sparse_semantic_window_scan",
                    "window_len": window.len(),
                    "vocab_size": self.vocab_size(),
                    "concepts": concepts,
                    "smoothing": smoothing,
                    "raw_total": total,
                    "distribution_sum": distribution_sum,
                    "dominant_probability": dominant_probability,
                    "contributions": contributions,
                    "skipped_tokens": skipped_tokens,
                    "skipped_non_positive_weights": skipped_non_positive_weights,
                    "uniform_fallback": true,
                    "empty_window": window.is_empty(),
                    "estimated_sparse_scan_values": window.len(),
                    "estimated_accumulation_values": concepts.saturating_mul(window.len().saturating_add(1)),
                    "estimated_distribution_scale_values": 0usize,
                })
            });
            return SemanticWindowDistributionReport {
                distribution,
                backend: "semantic_cpu",
                requested_backend: "auto",
                semantic_sparse_scan_backend: "semantic_cpu",
                semantic_accumulation_backend,
                distribution_scale_backend: "none",
                inference_mode: "uniform_fallback",
                route_blocker: "sparse_semantic_window_scan",
                raw_total: total,
                contributions,
                skipped_tokens,
                skipped_non_positive_weights,
                uniform_fallback: true,
                empty_window: window.is_empty(),
            };
        }
        let (scaled, distribution_scale_backend) = scale_semantic_distribution(accum, 1.0 / total);
        let distribution = scaled
            .into_iter()
            .map(|v| v.max(f32::EPSILON))
            .collect::<Vec<_>>();
        let distribution_sum = distribution.iter().copied().sum::<f32>();
        let dominant_probability = distribution
            .iter()
            .copied()
            .fold(0.0f32, |best, value| best.max(value));
        emit_tensor_op(
            "semantic_bridge_window_distribution",
            &[window.len(), self.vocab_size(), concepts],
            &[1, distribution.len()],
        );
        emit_tensor_op_meta("semantic_bridge_window_distribution", || {
            serde_json::json!({
                "backend": "hybrid",
                "requested_backend": distribution_scale_backend,
                "kind": "language_semantic_bridge_window_distribution",
                "semantic_sparse_scan_backend": "semantic_cpu",
                "semantic_accumulation_backend": semantic_accumulation_backend,
                "distribution_scale_backend": distribution_scale_backend,
                "inference_mode": "sparse_window_accumulate_then_scale",
                "route_blocker": "sparse_semantic_window_scan",
                "window_len": window.len(),
                "vocab_size": self.vocab_size(),
                "concepts": concepts,
                "smoothing": smoothing,
                "raw_total": total,
                "distribution_sum": distribution_sum,
                "dominant_probability": dominant_probability,
                "contributions": contributions,
                "skipped_tokens": skipped_tokens,
                "skipped_non_positive_weights": skipped_non_positive_weights,
                "uniform_fallback": false,
                "empty_window": window.is_empty(),
                "estimated_sparse_scan_values": window.len(),
                "estimated_accumulation_values": concepts.saturating_mul(window.len().saturating_add(1)),
                "estimated_distribution_scale_values": distribution.len(),
            })
        });
        SemanticWindowDistributionReport {
            distribution,
            backend: "hybrid",
            requested_backend: distribution_scale_backend,
            semantic_sparse_scan_backend: "semantic_cpu",
            semantic_accumulation_backend,
            distribution_scale_backend,
            inference_mode: "sparse_window_accumulate_then_scale",
            route_blocker: "sparse_semantic_window_scan",
            raw_total: total,
            contributions,
            skipped_tokens,
            skipped_non_positive_weights,
            uniform_fallback: false,
            empty_window: window.is_empty(),
        }
    }

    pub fn infer_from_window(&self, window: &[(usize, f32)], smoothing: f32) -> Vec<f32> {
        self.infer_from_window_report(window, smoothing)
            .distribution
    }

    pub fn concept_kernel(&self) -> &SparseKernel {
        &self.concept_kernel
    }
}

impl SemanticBridge {
    pub fn log_rows(&self) -> &[Vec<(usize, f32)>] {
        &self.log_pi
    }

    pub fn row_marginal(&self) -> &[f32] {
        &self.row_sums
    }

    pub fn col_marginal(&self) -> &[f32] {
        &self.col_sums
    }
}

#[derive(Clone, Debug)]
pub enum ConceptHint {
    Distribution(Vec<f32>),
    Window(Vec<(usize, f32)>),
}

impl ConceptHint {
    pub fn as_distribution(&self, bridge: &SemanticBridge) -> Vec<f32> {
        match self {
            ConceptHint::Distribution(dist) => {
                let concepts = bridge.concept_count();
                let distribution = if dist.is_empty() {
                    vec![1.0 / concepts as f32; concepts]
                } else {
                    let sanitised = sanitise_distribution_hint_values(dist, concepts);
                    let buffer = sanitised.values;
                    let clipped_negative = sanitised.clipped_negative;
                    let non_finite = sanitised.non_finite;
                    let semantic_sanitize_backend = sanitised.backend;
                    let sum: f32 = buffer.iter().sum();
                    let (distribution, backend, requested_backend, distribution_scale_backend) =
                        if sum <= f32::EPSILON {
                            (
                                vec![1.0 / concepts as f32; concepts],
                                "semantic_cpu",
                                "auto",
                                "host",
                            )
                        } else {
                            let (scaled, scale_backend) =
                                scale_semantic_distribution(buffer, 1.0 / sum);
                            (
                                scaled.into_iter().map(|v| v.max(f32::EPSILON)).collect(),
                                "hybrid",
                                scale_backend,
                                scale_backend,
                            )
                        };
                    emit_tensor_op(
                        "concept_hint_distribution",
                        &[dist.len(), concepts],
                        &[1, distribution.len()],
                    );
                    emit_tensor_op_meta("concept_hint_distribution", || {
                        serde_json::json!({
                            "backend": backend,
                            "requested_backend": requested_backend,
                            "kind": "language_concept_hint_distribution",
                            "source": "distribution",
                            "semantic_sanitize_backend": semantic_sanitize_backend,
                            "distribution_scale_backend": distribution_scale_backend,
                            "input_len": dist.len(),
                            "concepts": concepts,
                            "raw_sum": sum,
                            "distribution_sum": distribution.iter().copied().sum::<f32>(),
                            "dominant_probability": distribution
                                .iter()
                                .copied()
                                .fold(0.0f32, |best, value| best.max(value)),
                            "truncated_values": dist.len().saturating_sub(concepts),
                            "clipped_negative": clipped_negative,
                            "non_finite_values": non_finite,
                            "uniform_fallback": sum <= f32::EPSILON,
                            "empty_hint": false,
                        })
                    });
                    return distribution;
                };
                emit_tensor_op(
                    "concept_hint_distribution",
                    &[dist.len(), concepts],
                    &[1, distribution.len()],
                );
                emit_tensor_op_meta("concept_hint_distribution", || {
                    serde_json::json!({
                        "backend": "semantic_cpu",
                        "requested_backend": "auto",
                        "kind": "language_concept_hint_distribution",
                        "source": "distribution",
                        "input_len": dist.len(),
                        "concepts": concepts,
                        "raw_sum": 0.0f32,
                        "distribution_sum": distribution.iter().copied().sum::<f32>(),
                        "dominant_probability": distribution
                            .iter()
                            .copied()
                            .fold(0.0f32, |best, value| best.max(value)),
                        "truncated_values": 0,
                        "clipped_negative": 0,
                        "non_finite_values": 0,
                        "uniform_fallback": true,
                        "empty_hint": true,
                    })
                });
                distribution
            }
            ConceptHint::Window(window) => {
                let report = bridge.infer_from_window_report(window, DEFAULT_EPSILON);
                let distribution = report.distribution.clone();
                emit_tensor_op(
                    "concept_hint_distribution",
                    &[window.len(), bridge.concept_count()],
                    &[1, distribution.len()],
                );
                emit_tensor_op_meta("concept_hint_distribution", || {
                    serde_json::json!({
                        "backend": report.backend,
                        "requested_backend": report.requested_backend,
                        "kind": "language_concept_hint_distribution",
                        "source": "window",
                        "semantic_inference_backend": "semantic_bridge_window_distribution",
                        "semantic_sparse_scan_backend": report.semantic_sparse_scan_backend,
                        "semantic_accumulation_backend": report.semantic_accumulation_backend,
                        "distribution_scale_backend": report.distribution_scale_backend,
                        "inference_mode": report.inference_mode,
                        "route_blocker": report.route_blocker,
                        "input_len": window.len(),
                        "concepts": bridge.concept_count(),
                        "raw_sum": report.raw_total,
                        "distribution_sum": distribution.iter().copied().sum::<f32>(),
                        "dominant_probability": distribution
                            .iter()
                            .copied()
                            .fold(0.0f32, |best, value| best.max(value)),
                        "truncated_values": 0,
                        "clipped_negative": report.skipped_non_positive_weights,
                        "non_finite_values": window.iter().filter(|(_, weight)| !weight.is_finite()).count(),
                        "contributions": report.contributions,
                        "skipped_tokens": report.skipped_tokens,
                        "uniform_fallback": report.uniform_fallback,
                        "empty_hint": report.empty_window,
                        "estimated_inference_values": bridge.concept_count().saturating_mul(window.len().saturating_add(1)),
                    })
                });
                distribution
            }
        }
    }
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

    fn sample_kernel() -> SparseKernel {
        SparseKernel::from_rows(
            vec![vec![(0, 0.7), (1, 0.3)], vec![(0, 0.2), (1, 0.8)]],
            1e-6,
        )
        .unwrap()
    }

    #[test]
    fn dense_kernel_matches_rows() {
        let dense = vec![vec![0.7, 0.3], vec![0.2, 0.8]];
        let dense_kernel = SparseKernel::from_dense(dense, 1e-6).unwrap();
        assert_eq!(dense_kernel.vocab_size(), 2);
        assert!((dense_kernel.log_value(0, 0).exp() - 0.7).abs() < 1e-6);
        assert!((dense_kernel.log_value(1, 1).exp() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn sparse_kernel_probability_rows_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let kernel =
            SparseKernel::from_rows(vec![vec![(0, 2.0), (1, 1.0)], vec![(1, 3.0)]], 1e-6).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(kernel.vocab_size(), 2);
        let events = events.lock().unwrap();
        let row = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "sparse_kernel_probability_row" && data["row_index"] == 0
            })
            .expect("sparse_kernel_probability_row metadata event");
        assert_eq!(row.1["backend"], "hybrid");
        assert_eq!(row.1["requested_backend"], "auto");
        assert_eq!(row.1["kind"], "language_sparse_kernel_probability_row");
        assert_eq!(row.1["row_sum_backend"], "auto");
        assert_eq!(row.1["distribution_scale_backend"], "auto");
        assert_eq!(row.1["entries"], 2);
        assert_eq!(row.1["empty"], false);
        assert!(row.1["raw_sum"].as_f64().unwrap_or(0.0) > 0.0);
        assert!((row.1["distribution_sum"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-5);
        assert!(row.1["dominant_probability"].as_f64().unwrap_or(0.0) > 0.0);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn sparse_kernel_probability_rows_forced_wgpu_route_sum() {
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
        let kernel = {
            let _guard = push_backend_policy(policy);
            SparseKernel::from_rows(vec![vec![(0, 2.0), (1, 1.0)], vec![(1, 1.0)]], 1e-6).unwrap()
        };
        st_tensor::set_tensor_op_meta_observer(previous);
        restore_tensor_util_wgpu_min_values(previous_min);

        assert_eq!(kernel.vocab_size(), 2);
        let events = events.lock().unwrap();
        let row = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "sparse_kernel_probability_row" && data["row_index"] == 0
            })
            .expect("sparse_kernel_probability_row metadata event");
        assert_eq!(row.1["backend"], "hybrid");
        assert_eq!(row.1["requested_backend"], "wgpu");
        assert_eq!(row.1["row_sum_backend"], "wgpu");
        assert_eq!(row.1["distribution_scale_backend"], "wgpu");
        let sum_abs = events
            .iter()
            .find(|(op_name, data)| *op_name == "sum_abs" && data["backend"] == "wgpu_dense")
            .expect("sum_abs WGPU sparse row metadata event");
        assert_eq!(sum_abs.1["backend"], "wgpu_dense");
    }

    #[test]
    fn concept_hint_and_window_distribution_emit_backend_meta() {
        let concept_kernel = sample_kernel();
        let bridge = SemanticBridge::from_dense(
            vec![vec![0.8, 0.2], vec![0.3, 0.7]],
            Vec::new(),
            1e-6,
            concept_kernel,
        )
        .unwrap();

        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let hinted = ConceptHint::Distribution(vec![0.6, -0.1, 0.4]).as_distribution(&bridge);
        let inferred =
            ConceptHint::Window(vec![(0, 0.8), (99, 0.4), (1, -0.2)]).as_distribution(&bridge);
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(hinted.len(), bridge.concept_count());
        assert_eq!(inferred.len(), bridge.concept_count());
        let events = events.lock().unwrap();
        let distribution_hint = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "concept_hint_distribution" && data["source"] == "distribution"
            })
            .expect("concept_hint_distribution metadata event");
        assert_eq!(distribution_hint.1["backend"], "hybrid");
        assert_eq!(distribution_hint.1["requested_backend"], "auto");
        assert_eq!(
            distribution_hint.1["kind"],
            "language_concept_hint_distribution"
        );
        assert_eq!(distribution_hint.1["semantic_sanitize_backend"], "auto");
        assert_eq!(distribution_hint.1["distribution_scale_backend"], "auto");
        assert_eq!(distribution_hint.1["input_len"], 3);
        assert_eq!(distribution_hint.1["concepts"], 2);
        assert_eq!(distribution_hint.1["truncated_values"], 1);
        assert_eq!(distribution_hint.1["clipped_negative"], 1);
        assert!(
            distribution_hint.1["distribution_sum"]
                .as_f64()
                .unwrap_or(0.0)
                > 0.0
        );

        let window_distribution = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "semantic_bridge_window_distribution" && data["window_len"] == 3
            })
            .expect("semantic_bridge_window_distribution metadata event");
        assert_eq!(window_distribution.1["backend"], "hybrid");
        assert_eq!(window_distribution.1["requested_backend"], "auto");
        assert_eq!(
            window_distribution.1["kind"],
            "language_semantic_bridge_window_distribution"
        );
        assert_eq!(
            window_distribution.1["semantic_accumulation_backend"],
            "auto"
        );
        assert_eq!(
            window_distribution.1["semantic_sparse_scan_backend"],
            "semantic_cpu"
        );
        assert_eq!(window_distribution.1["distribution_scale_backend"], "auto");
        assert_eq!(window_distribution.1["skipped_tokens"], 1);
        assert_eq!(window_distribution.1["skipped_non_positive_weights"], 1);
        assert_eq!(window_distribution.1["uniform_fallback"], false);
        assert!(
            window_distribution.1["distribution_sum"]
                .as_f64()
                .unwrap_or(0.0)
                > 0.0
        );

        let window_hint = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "concept_hint_distribution" && data["source"] == "window"
            })
            .expect("concept_hint_distribution window metadata event");
        assert_eq!(window_hint.1["backend"], "hybrid");
        assert_eq!(window_hint.1["requested_backend"], "auto");
        assert_eq!(
            window_hint.1["semantic_inference_backend"],
            "semantic_bridge_window_distribution"
        );
        assert_eq!(
            window_hint.1["semantic_sparse_scan_backend"],
            "semantic_cpu"
        );
        assert_eq!(window_hint.1["semantic_accumulation_backend"], "auto");
        assert_eq!(window_hint.1["distribution_scale_backend"], "auto");
        assert_eq!(
            window_hint.1["inference_mode"],
            "sparse_window_accumulate_then_scale"
        );
        assert_eq!(
            window_hint.1["route_blocker"],
            "sparse_semantic_window_scan"
        );
        assert_eq!(window_hint.1["input_len"], 3);
        assert_eq!(window_hint.1["contributions"], 2);
        assert_eq!(window_hint.1["skipped_tokens"], 1);
        assert!(
            window_hint.1["dominant_probability"]
                .as_f64()
                .unwrap_or(0.0)
                > 0.0
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn concept_hint_distribution_forced_wgpu_routes_sanitise() {
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

        let bridge = SemanticBridge::from_dense(
            vec![vec![0.8, 0.2], vec![0.3, 0.7]],
            Vec::new(),
            1e-6,
            sample_kernel(),
        )
        .unwrap();
        let policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let distribution = {
            let _guard = push_backend_policy(policy);
            ConceptHint::Distribution(vec![0.6, -0.1, 0.4]).as_distribution(&bridge)
        };
        st_tensor::set_tensor_op_meta_observer(previous);
        restore_tensor_util_wgpu_min_values(previous_min);

        assert_eq!(distribution.len(), bridge.concept_count());
        let events = events.lock().unwrap();
        let hint = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "concept_hint_distribution" && data["source"] == "distribution"
            })
            .expect("concept_hint_distribution metadata event");
        assert_eq!(hint.1["backend"], "hybrid");
        assert_eq!(hint.1["requested_backend"], "wgpu");
        assert_eq!(hint.1["semantic_sanitize_backend"], "wgpu");
        assert_eq!(hint.1["distribution_scale_backend"], "wgpu");
        let relu = events
            .iter()
            .find(|(op_name, data)| *op_name == "relu" && data["requested_backend"] == "wgpu")
            .expect("relu WGPU sanitise metadata event");
        assert_eq!(relu.1["backend"], "wgpu_dense");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn semantic_bridge_window_distribution_forced_wgpu_routes_accumulation() {
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

        let concept_kernel = sample_kernel();
        let bridge = SemanticBridge::from_dense(
            vec![vec![0.8, 0.2], vec![0.3, 0.7]],
            Vec::new(),
            1e-6,
            concept_kernel,
        )
        .unwrap();
        let policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let distribution = {
            let _guard = push_backend_policy(policy);
            bridge.infer_from_window(&[(0, 0.8), (1, 0.4)], 1e-6)
        };
        st_tensor::set_tensor_op_meta_observer(previous);
        restore_tensor_util_wgpu_min_values(previous_min);

        assert_eq!(distribution.len(), bridge.concept_count());
        let events = events.lock().unwrap();
        let window_distribution = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "semantic_bridge_window_distribution" && data["window_len"] == 2
            })
            .expect("semantic_bridge_window_distribution metadata event");
        assert_eq!(window_distribution.1["backend"], "hybrid");
        assert_eq!(window_distribution.1["requested_backend"], "wgpu");
        assert_eq!(
            window_distribution.1["semantic_sparse_scan_backend"],
            "semantic_cpu"
        );
        assert_eq!(
            window_distribution.1["semantic_accumulation_backend"],
            "wgpu"
        );
        assert_eq!(window_distribution.1["distribution_scale_backend"], "wgpu");
        let reduction = events
            .iter()
            .find(|(op_name, data)| *op_name == "sum_axis0" && data["requested_backend"] == "wgpu")
            .expect("sum_axis0 WGPU accumulation metadata event");
        assert_eq!(reduction.1["backend"], "wgpu_dense");
    }

    #[test]
    fn semantic_bridge_from_dense_matches_manual() {
        let concept_kernel = sample_kernel();
        let dense = vec![vec![0.6, 0.4], vec![0.3, 0.7]];
        let bridge =
            SemanticBridge::from_dense(dense.clone(), Vec::new(), 1e-6, concept_kernel.clone())
                .unwrap();
        assert_eq!(bridge.vocab_size(), 2);
        assert_eq!(bridge.concept_count(), 2);
        let manual = SemanticBridge::new(
            dense
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .enumerate()
                        .filter(|(_, value)| *value > 0.0)
                        .map(|(idx, value)| (idx, value.ln()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
            vec![1.0, 1.0],
            vec![0.9, 1.1],
            HashSet::new(),
            1e-6,
            concept_kernel,
        )
        .unwrap();
        assert!((bridge.log_pi(0, 0) - manual.log_pi(0, 0)).abs() < 1e-6);
        assert!((bridge.row_marginal()[0] - 1.0).abs() < 1e-6);
        assert!(bridge.col_marginal()[0] > 0.0);
    }
}
