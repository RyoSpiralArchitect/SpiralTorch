// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::Module;
use crate::{PureResult, Tensor, TensorError};
#[cfg(feature = "wgpu")]
use st_tensor::wgpu_dense;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorUtilBackend};
use std::cell::RefCell;

const DIST_FLOOR: f32 = 1.0e-6;
const SCORE_FLOOR: f32 = 1.0e-12;

fn coherence_weight_stats(weights: &[f32]) -> (usize, f32, f32) {
    let mut nonzero = 0usize;
    let mut max_weight = 0.0f32;
    let mut min_nonzero = f32::MAX;
    for &weight in weights {
        if weight.is_finite() && weight > 0.0 {
            nonzero += 1;
            max_weight = max_weight.max(weight);
            min_nonzero = min_nonzero.min(weight);
        }
    }
    if nonzero == 0 {
        min_nonzero = 0.0;
    }
    (nonzero, max_weight, min_nonzero)
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

#[cfg(feature = "wgpu")]
fn validate_finite_slice(label: &'static str, values: &[f32]) -> PureResult<()> {
    for &value in values {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue { label, value });
        }
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
fn strict_gpu_path() -> bool {
    std::env::var("SPIRALTORCH_STRICT_GPU")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

#[cfg(feature = "wgpu")]
fn coherence_scan_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_coherence_scan_meta(
    op_name: &'static str,
    kind: &'static str,
    batch: usize,
    steps: usize,
    dim: usize,
    memory: usize,
    curvature: f32,
    temperature: f32,
    self_score_scale: f32,
    query_residual_scale: f32,
    weights: &[f32],
    backward: bool,
    backend: &'static str,
    requested_backend: &'static str,
    kernel: &'static str,
    fallback: Option<&str>,
) {
    let active_window = memory.min(steps);
    let input_values = batch.saturating_mul(steps).saturating_mul(dim);
    let output_values = batch.saturating_mul(dim);
    let score_pairs = batch.saturating_mul(active_window);
    let weighted_values = score_pairs.saturating_mul(dim);
    let (nonzero_weights, max_weight, min_nonzero_weight) = coherence_weight_stats(weights);
    let residual_values = if query_residual_scale > 0.0 {
        output_values
    } else {
        0
    };
    emit_tensor_op(op_name, &[batch, steps, dim], &[batch, dim]);
    emit_tensor_op_meta(op_name, || {
        let mut data = serde_json::json!({
            "backend": backend,
            "requested_backend": requested_backend,
            "kernel": kernel,
            "kind": kind,
            "batch": batch,
            "steps": steps,
            "dim": dim,
            "memory": memory,
            "active_window": active_window,
            "input_values": input_values,
            "output_values": output_values,
            "score_pairs": score_pairs,
            "pairwise_values": weighted_values,
            "weighted_values": weighted_values,
            "residual_values": residual_values,
            "nonzero_weights": nonzero_weights,
            "max_weight": max_weight,
            "min_nonzero_weight": min_nonzero_weight,
            "curvature": curvature,
            "temperature": temperature,
            "self_score_scale": self_score_scale,
            "query_residual_scale": query_residual_scale,
            "estimated_backward_adds": if backward {
                nonzero_weights.saturating_mul(dim).saturating_add(residual_values)
            } else {
                0
            },
            "score_gradient": backward,
            "normalization_gradient": backward,
            "estimated_score_gradient_ops": if backward {
                score_pairs.saturating_mul(dim).saturating_mul(4)
            } else {
                0
            },
            "backward": backward,
            "empty": input_values == 0,
        });
        if let Some(message) = fallback {
            if let Some(map) = data.as_object_mut() {
                map.insert(
                    "fallback".to_string(),
                    serde_json::json!({"from": "wgpu", "message": message}),
                );
            }
        }
        data
    });
}

#[derive(Clone, Debug)]
struct CoherenceScanCache {
    batch: usize,
    steps: usize,
    dim: usize,
    weights: Vec<f32>,
    query_residual_scale: f32,
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
    self_score_scale: f32,
    query_residual_scale: f32,
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
        Self::with_output_scales(dim, steps, memory, curvature, temperature, 1.0, 0.0)
    }

    pub fn with_self_score_scale(
        dim: usize,
        steps: usize,
        memory: usize,
        curvature: f32,
        temperature: f32,
        self_score_scale: f32,
    ) -> PureResult<Self> {
        Self::with_output_scales(
            dim,
            steps,
            memory,
            curvature,
            temperature,
            self_score_scale,
            0.0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_output_scales(
        dim: usize,
        steps: usize,
        memory: usize,
        curvature: f32,
        temperature: f32,
        self_score_scale: f32,
        query_residual_scale: f32,
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
        if self_score_scale < 0.0 || !self_score_scale.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_coherence_scan_self_score_scale",
                value: self_score_scale,
            });
        }
        if query_residual_scale < 0.0 || !query_residual_scale.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_coherence_scan_query_residual_scale",
                value: query_residual_scale,
            });
        }

        Ok(Self {
            dim,
            steps,
            memory,
            curvature,
            temperature,
            self_score_scale,
            query_residual_scale,
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

    pub fn self_score_scale(&self) -> f32 {
        self.self_score_scale
    }

    pub fn query_residual_scale(&self) -> f32 {
        self.query_residual_scale
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
        let dist =
            ((mse / denom).sqrt() * (-self.curvature).sqrt() / self.temperature).max(DIST_FLOOR);
        let score = 1.0 / (dist.powf(self.coherence_order()) + SCORE_FLOOR);
        if score.is_finite() {
            score
        } else {
            0.0
        }
    }

    fn score_pair_gradient_common(
        &self,
        query: &[f32],
        value: &[f32],
        score_scale: f32,
    ) -> Option<f32> {
        if score_scale == 0.0 {
            return None;
        }
        debug_assert_eq!(query.len(), self.dim);
        debug_assert_eq!(value.len(), self.dim);
        let mut mse = 0.0f32;
        for (&q, &v) in query.iter().zip(value.iter()) {
            let diff = q - v;
            mse += diff * diff;
        }
        let dim = (self.dim as f32).max(1.0);
        let mean = mse / dim;
        if !mean.is_finite() || mean <= 0.0 {
            return None;
        }
        let sqrt_mean = mean.sqrt();
        let alpha = (-self.curvature).sqrt() / self.temperature;
        let unclamped_dist = sqrt_mean * alpha;
        if !unclamped_dist.is_finite() || unclamped_dist <= DIST_FLOOR {
            return None;
        }

        let order = self.coherence_order();
        let dist_pow = unclamped_dist.powf(order);
        let denom = dist_pow + SCORE_FLOOR;
        if !denom.is_finite() || denom <= 0.0 {
            return None;
        }
        let dscore_ddist = -order * unclamped_dist.powf(order - 1.0) / (denom * denom);
        let common = score_scale * dscore_ddist * alpha / (dim * sqrt_mean);
        if common.is_finite() {
            Some(common)
        } else {
            None
        }
    }

    fn fallback_steps(&self, start_step: usize) -> Vec<usize> {
        let query_step = self.steps - 1;
        let mut steps = (start_step..self.steps)
            .filter(|&step| step != query_step || self.self_score_scale > 0.0)
            .collect::<Vec<_>>();
        if steps.is_empty() {
            steps.push(query_step);
        }
        steps
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
        let route_backend = current_tensor_util_backend_for_values(
            batch
                .saturating_mul(self.memory.min(self.steps))
                .saturating_mul(self.dim),
        );
        let requested_backend = tensor_util_backend_label(route_backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && batch > 0
                && self.steps > 0
                && self.dim > 0
                && self.memory > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::zspace_coherence_scan_forward(
                    data,
                    batch,
                    self.steps,
                    self.dim,
                    self.memory,
                    self.curvature,
                    self.temperature,
                    self.self_score_scale,
                    self.query_residual_scale,
                ) {
                    Ok((context, weights)) => {
                        validate_finite_slice("zspace_coherence_scan_output", &context)?;
                        validate_finite_slice("zspace_coherence_scan_weights", &weights)?;
                        let output = Tensor::from_vec(batch, self.dim, context)?;
                        emit_coherence_scan_meta(
                            "zspace_coherence_scan_forward",
                            "zspace_coherence_scan_forward",
                            batch,
                            self.steps,
                            self.dim,
                            self.memory,
                            self.curvature,
                            self.temperature,
                            self.self_score_scale,
                            self.query_residual_scale,
                            &weights,
                            false,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.zspace_coherence_scan_forward",
                            None,
                        );
                        *self.cache.borrow_mut() = Some(CoherenceScanCache {
                            batch,
                            steps: self.steps,
                            dim: self.dim,
                            weights,
                            query_residual_scale: self.query_residual_scale,
                        });
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(coherence_scan_wgpu_error(
                            "zspace_coherence_scan_forward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut output = Tensor::zeros(batch, self.dim)?;
        let mut weights = vec![0.0f32; batch * self.steps];
        let start_step = self.steps.saturating_sub(self.memory);

        for b in 0..batch {
            let base = b * cols;
            let query_start = base + (self.steps - 1) * self.dim;
            let query = &data[query_start..query_start + self.dim];

            let mut scores = vec![0.0f32; self.steps];
            let mut total = 0.0f32;
            for (step, slot) in scores
                .iter_mut()
                .enumerate()
                .take(self.steps)
                .skip(start_step)
            {
                let value_start = base + step * self.dim;
                let value = &data[value_start..value_start + self.dim];
                let mut score = self.score_pair(query, value);
                if step == self.steps - 1 {
                    score *= self.self_score_scale;
                }
                *slot = score;
                total += score;
            }

            if !total.is_finite() || total <= 0.0 {
                let fallback_steps = self.fallback_steps(start_step);
                let uniform = 1.0 / (fallback_steps.len() as f32).max(1.0);
                for step in fallback_steps {
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
            if self.query_residual_scale > 0.0 {
                for (dst, &src) in out_slice.iter_mut().zip(query.iter()) {
                    *dst += self.query_residual_scale * src;
                }
            }
        }

        emit_coherence_scan_meta(
            "zspace_coherence_scan_forward",
            "zspace_coherence_scan_forward",
            batch,
            self.steps,
            self.dim,
            self.memory,
            self.curvature,
            self.temperature,
            self.self_score_scale,
            self.query_residual_scale,
            &weights,
            false,
            "cpu",
            requested_backend,
            "coherence_scan.scalar",
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure.as_deref()
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        *self.cache.borrow_mut() = Some(CoherenceScanCache {
            batch,
            steps: self.steps,
            dim: self.dim,
            weights,
            query_residual_scale: self.query_residual_scale,
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

        let go = grad_output.data();
        let data = input.data();
        let weights = &cache.weights;
        let route_backend = current_tensor_util_backend_for_values(
            cache
                .batch
                .saturating_mul(self.memory.min(cache.steps))
                .saturating_mul(cache.dim),
        );
        let requested_backend = tensor_util_backend_label(route_backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && cache.batch > 0
                && cache.steps > 0
                && cache.dim > 0
                && self.memory > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::zspace_coherence_scan_backward(
                    data,
                    go,
                    weights,
                    cache.batch,
                    cache.steps,
                    cache.dim,
                    self.memory,
                    self.curvature,
                    self.temperature,
                    self.self_score_scale,
                    cache.query_residual_scale,
                ) {
                    Ok(gradient) => {
                        validate_finite_slice("zspace_coherence_scan_grad_input", &gradient)?;
                        let grad_input = Tensor::from_vec(cache.batch, expected_cols, gradient)?;
                        emit_coherence_scan_meta(
                            "zspace_coherence_scan_backward",
                            "zspace_coherence_scan_backward_scatter",
                            cache.batch,
                            cache.steps,
                            cache.dim,
                            self.memory,
                            self.curvature,
                            self.temperature,
                            self.self_score_scale,
                            cache.query_residual_scale,
                            weights,
                            true,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.zspace_coherence_scan_backward",
                            None,
                        );
                        return Ok(grad_input);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(coherence_scan_wgpu_error(
                            "zspace_coherence_scan_backward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut grad_input = Tensor::zeros(cache.batch, expected_cols)?;
        let start_step = cache.steps.saturating_sub(self.memory);
        let query_step = cache.steps - 1;
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
            if cache.query_residual_scale > 0.0 {
                for b in 0..cache.batch {
                    let go_offset = b * cache.dim;
                    let gi_offset = b * expected_cols + (cache.steps - 1) * cache.dim;
                    for d in 0..cache.dim {
                        gi[gi_offset + d] += cache.query_residual_scale * go[go_offset + d];
                    }
                }
            }

            for b in 0..cache.batch {
                let base = b * expected_cols;
                let go_offset = b * cache.dim;
                let query_offset = base + query_step * cache.dim;
                let query = &data[query_offset..query_offset + cache.dim];
                let grad_row = &go[go_offset..go_offset + cache.dim];

                let mut scores = vec![0.0f32; cache.steps];
                let mut total = 0.0f32;
                for (step, slot) in scores
                    .iter_mut()
                    .enumerate()
                    .take(cache.steps)
                    .skip(start_step)
                {
                    let value_offset = base + step * cache.dim;
                    let value = &data[value_offset..value_offset + cache.dim];
                    let mut score = self.score_pair(query, value);
                    if step == query_step {
                        score *= self.self_score_scale;
                    }
                    *slot = score;
                    total += score;
                }
                if !total.is_finite() || total <= 0.0 {
                    continue;
                }

                let mut weighted_dot = 0.0f32;
                for step in start_step..cache.steps {
                    let weight = weights[b * cache.steps + step];
                    if weight == 0.0 {
                        continue;
                    }
                    let value_offset = base + step * cache.dim;
                    let value = &data[value_offset..value_offset + cache.dim];
                    let dot = grad_row
                        .iter()
                        .zip(value.iter())
                        .map(|(&grad, &src)| grad * src)
                        .sum::<f32>();
                    weighted_dot += weight * dot;
                }

                for (step, score) in scores.iter().enumerate().take(cache.steps).skip(start_step) {
                    if *score == 0.0 {
                        continue;
                    }
                    let value_offset = base + step * cache.dim;
                    let value = &data[value_offset..value_offset + cache.dim];
                    let dot = grad_row
                        .iter()
                        .zip(value.iter())
                        .map(|(&grad, &src)| grad * src)
                        .sum::<f32>();
                    let dloss_dscore = (dot - weighted_dot) / total;
                    if !dloss_dscore.is_finite() || dloss_dscore == 0.0 {
                        continue;
                    }
                    let score_scale = if step == query_step {
                        self.self_score_scale
                    } else {
                        1.0
                    };
                    let Some(common) = self.score_pair_gradient_common(query, value, score_scale)
                    else {
                        continue;
                    };
                    for d in 0..cache.dim {
                        let delta = query[d] - value[d];
                        let contribution = dloss_dscore * common * delta;
                        if !contribution.is_finite() {
                            continue;
                        }
                        gi[query_offset + d] += contribution;
                        gi[value_offset + d] -= contribution;
                    }
                }
            }
        }
        emit_coherence_scan_meta(
            "zspace_coherence_scan_backward",
            "zspace_coherence_scan_backward_scatter",
            cache.batch,
            cache.steps,
            cache.dim,
            self.memory,
            self.curvature,
            self.temperature,
            self.self_score_scale,
            cache.query_residual_scale,
            weights,
            true,
            "cpu",
            requested_backend,
            "coherence_scan.scalar_backward",
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure.as_deref()
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
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
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::DeviceCaps;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn observer_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[cfg(feature = "wgpu")]
    fn approx_eq(lhs: &[f32], rhs: &[f32], tolerance: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (*a - *b).abs() <= tolerance,
                "mismatch at {idx}: left={a}, right={b}, tolerance={tolerance}"
            );
        }
    }

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
    fn coherence_scan_forward_backward_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut scan =
            ZSpaceCoherenceScan::with_output_scales(3, 5, 4, -1.0, 1.0, 0.25, 0.25).unwrap();
        let input = Tensor::from_vec(
            1,
            15,
            vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, //
                1.0, 1.0, 1.0, //
                2.0, 1.0, 0.5, //
            ],
        )
        .unwrap();
        let out = scan.forward(&input).unwrap();
        let grad_out = Tensor::from_vec(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let _ = scan.backward(&input, &grad_out).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(out.shape(), (1, 3));
        let events = events.lock().unwrap();
        for (op_name, kind, backward) in [
            (
                "zspace_coherence_scan_forward",
                "zspace_coherence_scan_forward",
                false,
            ),
            (
                "zspace_coherence_scan_backward",
                "zspace_coherence_scan_backward_scatter",
                true,
            ),
        ] {
            let event = events
                .iter()
                .find(|(name, data)| {
                    *name == op_name
                        && data["kind"] == kind
                        && data["steps"] == 5
                        && data["dim"] == 3
                        && data["query_residual_scale"].as_f64() == Some(0.25)
                })
                .unwrap_or_else(|| panic!("{op_name} metadata event"));
            assert_eq!(event.1["backend"], "cpu");
            assert_eq!(event.1["requested_backend"], "auto");
            assert_eq!(event.1["batch"], 1);
            assert_eq!(event.1["steps"], 5);
            assert_eq!(event.1["dim"], 3);
            assert_eq!(event.1["memory"], 4);
            assert_eq!(event.1["score_pairs"], 4);
            assert_eq!(event.1["weighted_values"], 12);
            assert_eq!(event.1["residual_values"], 3);
            assert_eq!(event.1["score_gradient"], backward);
            assert_eq!(event.1["normalization_gradient"], backward);
            assert_eq!(event.1["backward"], backward);
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn coherence_scan_forced_wgpu_forward_matches_cpu_reference_and_emits_backend() {
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        let previous_threshold = std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES").ok();
        std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "1");

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = Tensor::from_fn(2, 12, |row, col| {
            ((row * 17 + col * 5) % 29) as f32 * 0.031 - 0.33
        })
        .unwrap();
        let grad_out = Tensor::from_fn(2, 3, |row, col| {
            ((row * 11 + col * 7) % 19) as f32 * 0.023 - 0.14
        })
        .unwrap();

        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_scan =
            ZSpaceCoherenceScan::with_output_scales(3, 4, 3, -1.0, 1.0, 0.25, 0.5).unwrap();
        let cpu_forward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_scan.forward(&input).unwrap()
        };
        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_scan.backward(&input, &grad_out).unwrap()
        };

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_scan =
            ZSpaceCoherenceScan::with_output_scales(3, 4, 3, -1.0, 1.0, 0.25, 0.5).unwrap();
        let (wgpu_forward, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_scan.forward(&input).unwrap(),
                wgpu_scan.backward(&input, &grad_out).unwrap(),
            )
        };

        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        approx_eq(cpu_forward.data(), wgpu_forward.data(), 1.0e-3);
        approx_eq(cpu_grad_input.data(), wgpu_grad_input.data(), 2.0e-3);

        let events = events.lock().unwrap();
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "zspace_coherence_scan_forward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
                && data["kernel"] == "tensor_util.zspace_coherence_scan_forward"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "zspace_coherence_scan_backward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
                && data["kernel"] == "tensor_util.zspace_coherence_scan_backward"
                && data["score_gradient"] == true
                && data["normalization_gradient"] == true
        }));
    }

    #[test]
    fn coherence_scan_backward_matches_score_finite_difference() {
        let mut scan = ZSpaceCoherenceScan::with_self_score_scale(2, 3, 3, -1.0, 1.0, 0.0).unwrap();
        let input_values = vec![
            0.0, 0.0, //
            1.0, 0.25, //
            1.5, -0.5, //
        ];
        let input = Tensor::from_vec(1, 6, input_values.clone()).unwrap();
        let grad_out = Tensor::from_vec(1, 2, vec![0.7, -0.2]).unwrap();

        let _ = scan.forward(&input).unwrap();
        let direct_only = {
            let cache = scan.cache.borrow();
            let weights = &cache.as_ref().unwrap().weights;
            weights[1] * grad_out.data()[0]
        };
        let grad_input = scan.backward(&input, &grad_out).unwrap();
        let analytic = grad_input.data()[2];

        let epsilon = 1.0e-3f32;
        let loss_at = |values: Vec<f32>| {
            let tensor = Tensor::from_vec(1, 6, values).unwrap();
            let out = scan.forward(&tensor).unwrap();
            out.data()
                .iter()
                .zip(grad_out.data().iter())
                .map(|(&value, &grad)| value * grad)
                .sum::<f32>()
        };
        let mut plus = input_values.clone();
        plus[2] += epsilon;
        let mut minus = input_values;
        minus[2] -= epsilon;
        let finite_difference = (loss_at(plus) - loss_at(minus)) / (2.0 * epsilon);

        assert!(
            (analytic - finite_difference).abs() < 2.0e-2,
            "analytic={analytic} finite_difference={finite_difference}"
        );
        assert!(
            (analytic - direct_only).abs() > 1.0e-4,
            "score/normalization gradient should move beyond direct weighted aggregation"
        );
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

    #[test]
    fn coherence_scan_can_skip_query_self_match() {
        let scan = ZSpaceCoherenceScan::with_self_score_scale(2, 3, 2, -1.0, 1.0, 0.0).unwrap();
        let input = Tensor::from_vec(
            1,
            6,
            vec![
                0.0, 0.0, // step0 ignored by memory=2
                2.0, 0.0, // step1 becomes the only non-self memory item
                10.0, 0.0, // step2 query would dominate without self-score suppression
            ],
        )
        .unwrap();
        let out = scan.forward(&input).unwrap();
        let row = out.data();
        assert!((row[0] - 2.0).abs() < 1e-4);
        assert!(row[1].abs() < 1e-4);
    }

    #[test]
    fn coherence_scan_can_blend_query_residual() {
        let mut scan =
            ZSpaceCoherenceScan::with_output_scales(2, 3, 2, -1.0, 1.0, 0.0, 0.5).unwrap();
        let input = Tensor::from_vec(
            1,
            6,
            vec![
                0.0, 0.0, //
                2.0, 4.0, //
                10.0, 20.0, //
            ],
        )
        .unwrap();
        let out = scan.forward(&input).unwrap();
        assert_eq!(out.data(), &[7.0, 14.0]);

        let grad_out = Tensor::from_vec(1, 2, vec![1.0, 2.0]).unwrap();
        let grad_in = scan.backward(&input, &grad_out).unwrap();
        assert_eq!(
            grad_in.data(),
            &[
                0.0, 0.0, //
                1.0, 2.0, //
                0.5, 1.0, //
            ]
        );
    }
}
