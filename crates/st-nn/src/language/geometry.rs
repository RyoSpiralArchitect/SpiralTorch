// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::PureResult;
use st_tensor::pure::TensorError;
use std::collections::HashSet;

const DEFAULT_EPSILON: f32 = 1e-6;

fn ensure_probabilities(weights: &[(usize, f32)], epsilon: f32) -> PureResult<Vec<(usize, f32)>> {
    if weights.is_empty() {
        return Ok(Vec::new());
    }
    let mut total = 0.0f32;
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
        total += w;
    }
    let smoothing = epsilon * weights.len() as f32;
    let denom = (total + smoothing).max(epsilon);
    let mut normalised = Vec::with_capacity(weights.len());
    for &(j, w) in weights {
        let value = (w + epsilon) / denom;
        normalised.push((j, value.max(epsilon)));
    }
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
        for row in rows {
            let normalised = ensure_probabilities(&row, epsilon)?;
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

    pub fn infer_from_window(&self, window: &[(usize, f32)], smoothing: f32) -> Vec<f32> {
        let concepts = self.concept_count();
        let mut accum = vec![smoothing; concepts];
        let mut total = smoothing * concepts as f32;
        for &(token, weight) in window {
            if token >= self.vocab_size() {
                continue;
            }
            if weight <= 0.0 {
                continue;
            }
            let row = &self.log_pi[token];
            for &(concept, logp) in row {
                let prob = logp.exp();
                accum[concept] += weight * prob;
                total += weight * prob;
            }
        }
        if total <= f32::EPSILON {
            return vec![1.0 / concepts as f32; concepts];
        }
        accum
            .into_iter()
            .map(|v| (v / total).max(f32::EPSILON))
            .collect()
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
                if dist.is_empty() {
                    vec![1.0 / concepts as f32; concepts]
                } else {
                    let mut buffer = vec![DEFAULT_EPSILON; concepts];
                    for (idx, value) in dist.iter().copied().enumerate().take(concepts) {
                        buffer[idx] = value.max(0.0);
                    }
                    let sum: f32 = buffer.iter().sum();
                    if sum <= f32::EPSILON {
                        vec![1.0 / concepts as f32; concepts]
                    } else {
                        buffer
                            .into_iter()
                            .map(|v| (v / sum).max(f32::EPSILON))
                            .collect()
                    }
                }
            }
            ConceptHint::Window(window) => bridge.infer_from_window(window, DEFAULT_EPSILON),
        }
    }
}
