// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::{PureResult, Tensor, TensorError};
use std::cell::RefCell;

const DIST_FLOOR: f32 = 1.0e-6;
const SCORE_FLOOR: f32 = 1.0e-12;

#[derive(Clone, Debug)]
struct CoherenceScanCache {
    batch: usize,
    steps: usize,
    dim: usize,
    weights: Vec<f32>,
}

/// Temporal coherence scan that aggregates the last `memory` tokens of a
/// flattened sequence into a single context vector.
///
/// Unlike attention (Q·Kᵀ softmax), this layer derives fractional coherence
/// weights by comparing each token embedding to the most recent token under
/// the configured Z-space curvature, then normalises by a simple energy sum.
#[derive(Debug)]
pub struct ZSpaceCoherenceScan {
    dim: usize,
    steps: usize,
    memory: usize,
    curvature: f32,
    temperature: f32,
    cache: RefCell<Option<CoherenceScanCache>>,
}

impl ZSpaceCoherenceScan {
    pub fn new(
        dim: usize,
        steps: usize,
        memory: usize,
        curvature: f32,
        temperature: f32,
    ) -> PureResult<Self> {
        if dim == 0 || steps == 0 || memory == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: steps.max(1),
                cols: dim.max(1),
            });
        }
        if memory > steps {
            return Err(TensorError::InvalidDimensions {
                rows: memory,
                cols: steps,
            });
        }
        if curvature >= 0.0 || !curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_coherence_scan_temperature",
                value: temperature,
            });
        }

        Ok(Self {
            dim,
            steps,
            memory,
            curvature,
            temperature,
            cache: RefCell::new(None),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn memory(&self) -> usize {
        self.memory
    }

    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    fn coherence_order(&self) -> f32 {
        1.0 + (-self.curvature).sqrt().min(4.0)
    }

    fn score_pair(&self, query: &[f32], value: &[f32]) -> f32 {
        debug_assert_eq!(query.len(), self.dim);
        debug_assert_eq!(value.len(), self.dim);
        let mut mse = 0.0f32;
        for (&q, &v) in query.iter().zip(value.iter()) {
            let diff = q - v;
            mse += diff * diff;
        }
        let denom = (self.dim as f32).max(1.0);
        let dist = ((mse / denom).sqrt() * (-self.curvature).sqrt() / self.temperature)
            .max(DIST_FLOOR);
        let score = 1.0 / (dist.powf(self.coherence_order()) + SCORE_FLOOR);
        if score.is_finite() { score } else { 0.0 }
    }
}

impl Module for ZSpaceCoherenceScan {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        let expected_cols = self.dim * self.steps;
        if cols != expected_cols {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: (batch, expected_cols),
            });
        }

        let data = input.data();
        let mut output = Tensor::zeros(batch, self.dim)?;
        let mut weights = vec![0.0f32; batch * self.steps];
        let start_step = self.steps.saturating_sub(self.memory);

        for b in 0..batch {
            let base = b * cols;
            let query_start = base + (self.steps - 1) * self.dim;
            let query = &data[query_start..query_start + self.dim];

            let mut scores = vec![0.0f32; self.steps];
            let mut total = 0.0f32;
            for step in start_step..self.steps {
                let value_start = base + step * self.dim;
                let value = &data[value_start..value_start + self.dim];
                let score = self.score_pair(query, value);
                scores[step] = score;
                total += score;
            }

            if !total.is_finite() || total <= 0.0 {
                let uniform = 1.0 / (self.memory as f32).max(1.0);
                for step in start_step..self.steps {
                    weights[b * self.steps + step] = uniform;
                }
            } else {
                let inv = 1.0 / total;
                for step in start_step..self.steps {
                    weights[b * self.steps + step] = scores[step] * inv;
                }
            }

            let out_slice = &mut output.data_mut()[b * self.dim..(b + 1) * self.dim];
            for step in start_step..self.steps {
                let w = weights[b * self.steps + step];
                if w == 0.0 {
                    continue;
                }
                let value_start = base + step * self.dim;
                let value = &data[value_start..value_start + self.dim];
                for (dst, &src) in out_slice.iter_mut().zip(value.iter()) {
                    *dst += w * src;
                }
            }
        }

        *self.cache.borrow_mut() = Some(CoherenceScanCache {
            batch,
            steps: self.steps,
            dim: self.dim,
            weights,
        });
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let cache = self
            .cache
            .borrow_mut()
            .take()
            .ok_or(TensorError::EmptyInput("zspace_coherence_scan_cache"))?;

        let (batch, cols) = input.shape();
        let expected_cols = cache.steps * cache.dim;
        if (batch, cols) != (cache.batch, expected_cols) {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: (cache.batch, expected_cols),
            });
        }
        if grad_output.shape() != (cache.batch, cache.dim) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (cache.batch, cache.dim),
            });
        }

        let mut grad_input = Tensor::zeros(cache.batch, expected_cols)?;
        let go = grad_output.data();
        let weights = &cache.weights;
        {
            let gi = grad_input.data_mut();
            for b in 0..cache.batch {
                let go_offset = b * cache.dim;
                for step in 0..cache.steps {
                    let w = weights[b * cache.steps + step];
                    if w == 0.0 {
                        continue;
                    }
                    let gi_offset = b * expected_cols + step * cache.dim;
                    for d in 0..cache.dim {
                        gi[gi_offset + d] += w * go[go_offset + d];
                    }
                }
            }
        }
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherence_scan_shapes_match() {
        let mut scan = ZSpaceCoherenceScan::new(4, 3, 2, -1.0, 1.0).unwrap();
        let input = Tensor::from_vec(
            2,
            12,
            vec![
                // batch 0, step 0..2
                0.0, 0.0, 0.0, 0.0, //
                1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, //
                // batch 1, step 0..2
                0.5, 0.25, -0.5, -0.25, //
                0.0, 0.0, 0.0, 0.0, //
                0.5, 0.25, -0.5, -0.25, //
            ],
        )
        .unwrap();
        let out = scan.forward(&input).unwrap();
        assert_eq!(out.shape(), (2, 4));

        let grad_out = Tensor::from_vec(2, 4, vec![0.1; 8]).unwrap();
        let grad_in = scan.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), (2, 12));
    }

    #[test]
    fn coherence_scan_prefers_matching_recent_tokens() {
        let scan = ZSpaceCoherenceScan::new(2, 4, 3, -1.0, 1.0).unwrap();
        let input = Tensor::from_vec(
            1,
            8,
            vec![
                0.0, 0.0, // step0 (ignored by memory=3)
                1.0, 1.0, // step1
                1.0, 1.0, // step2
                1.0, 1.0, // step3 query
            ],
        )
        .unwrap();
        let out = scan.forward(&input).unwrap();
        let row = out.data();
        assert!((row[0] - 1.0).abs() < 1e-4);
        assert!((row[1] - 1.0).abs() < 1e-4);
    }
}

