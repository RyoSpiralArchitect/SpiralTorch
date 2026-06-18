// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Z-RBF attention specialised for Z-space indices.

use crate::execution::{current_attention_backend, current_matmul_backend};
use crate::z_rba::ZTensor;
use crate::{PureResult, Tensor, TensorError};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

fn validate_finite_value(label: &'static str, value: f32) -> PureResult<()> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(())
}

fn checked_value(label: &'static str, value: f32) -> PureResult<f32> {
    validate_finite_value(label, value)?;
    Ok(value)
}

fn validate_finite_slice(label: &'static str, values: &[f32]) -> PureResult<()> {
    for &value in values {
        validate_finite_value(label, value)?;
    }
    Ok(())
}

fn validate_finite_tensor(label: &'static str, tensor: &Tensor) -> PureResult<()> {
    validate_finite_slice(label, tensor.data())
}

fn relabel_non_finite<T>(result: PureResult<T>, label: &'static str) -> PureResult<T> {
    match result {
        Err(TensorError::NonFiniteValue { value, .. }) => {
            Err(TensorError::NonFiniteValue { label, value })
        }
        other => other,
    }
}

fn finite_meta_f32(value: f32) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value as f64 } else { 0.0 })
}

/// Identifies a token's position inside the active Z-frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ZIndex {
    pub band: usize,
    pub sheet: usize,
    pub echo: usize,
}

/// Distances sourced from a Z-frame or compatible descriptor.
pub trait ZFrameGeometry {
    fn band_distance(&self, a: usize, b: usize) -> f32;
    fn sheet_distance(&self, a: usize, b: usize) -> f32;
    fn echo_circular_distance(&self, a: usize, b: usize) -> f32;
}

/// Lightweight geometry helper for tests and offline experiments.
#[derive(Clone, Debug)]
pub struct SimpleZFrame {
    band_count: usize,
    sheet_count: usize,
    echo_period: usize,
}

impl SimpleZFrame {
    pub fn new(band_count: usize, sheet_count: usize, echo_period: usize) -> Self {
        Self {
            band_count: band_count.max(1),
            sheet_count: sheet_count.max(1),
            echo_period: echo_period.max(1),
        }
    }
}

impl ZFrameGeometry for SimpleZFrame {
    fn band_distance(&self, a: usize, b: usize) -> f32 {
        let a = (a % self.band_count) as isize;
        let b = (b % self.band_count) as isize;
        (a - b).abs() as f32
    }

    fn sheet_distance(&self, a: usize, b: usize) -> f32 {
        let a = (a % self.sheet_count) as isize;
        let b = (b % self.sheet_count) as isize;
        (a - b).abs() as f32
    }

    fn echo_circular_distance(&self, a: usize, b: usize) -> f32 {
        let period = self.echo_period as isize;
        let a = (a % self.echo_period) as isize;
        let b = (b % self.echo_period) as isize;
        let diff = (a - b).abs();
        let wrapped = period - diff;
        diff.min(wrapped).max(0) as f32
    }
}

/// Weights applied to each Z-distance component.
#[derive(Clone, Debug)]
pub struct ZMetricWeights {
    pub w_band: f32,
    pub w_sheet: f32,
    pub w_echo: f32,
}

impl Default for ZMetricWeights {
    fn default() -> Self {
        Self {
            w_band: 1.0,
            w_sheet: 0.7,
            w_echo: 0.5,
        }
    }
}

impl ZMetricWeights {
    pub fn normalised(&self) -> Self {
        let sum = self.w_band + self.w_sheet + self.w_echo;
        if !sum.is_finite() || sum <= f32::EPSILON {
            return Self::default();
        }
        Self {
            w_band: self.w_band / sum,
            w_sheet: self.w_sheet / sum,
            w_echo: self.w_echo / sum,
        }
    }

    fn validate(&self) -> PureResult<()> {
        validate_finite_value("zrba_metric_weight", self.w_band)?;
        validate_finite_value("zrba_metric_weight", self.w_sheet)?;
        validate_finite_value("zrba_metric_weight", self.w_echo)
    }
}

fn emit_metric_weights_normalise(weights: &ZMetricWeights) {
    let sum = weights.w_band + weights.w_sheet + weights.w_echo;
    let normalised = weights.normalised();
    let distribution_sum = normalised.w_band + normalised.w_sheet + normalised.w_echo;
    let dominant_weight = normalised
        .w_band
        .max(normalised.w_sheet)
        .max(normalised.w_echo);
    emit_tensor_op("zrba_metric_weights_normalise", &[3], &[3]);
    emit_tensor_op_meta("zrba_metric_weights_normalise", || {
        serde_json::json!({
            "backend": "control_cpu",
            "requested_backend": "host",
            "kind": "zrba_metric_weights_normalise",
            "values": 3,
            "raw_sum": finite_meta_f32(sum),
            "distribution_sum": finite_meta_f32(distribution_sum),
            "dominant_weight": finite_meta_f32(dominant_weight),
            "w_band": finite_meta_f32(weights.w_band),
            "w_sheet": finite_meta_f32(weights.w_sheet),
            "w_echo": finite_meta_f32(weights.w_echo),
            "normalised_w_band": finite_meta_f32(normalised.w_band),
            "normalised_w_sheet": finite_meta_f32(normalised.w_sheet),
            "normalised_w_echo": finite_meta_f32(normalised.w_echo),
            "default_fallback": !sum.is_finite() || sum <= f32::EPSILON,
        })
    });
}

fn kernel_component(distance: f32, ell: f32) -> PureResult<f32> {
    validate_finite_value("zrba_kernel_distance", distance)?;
    validate_finite_value("zrba_kernel_length_scale", ell)?;
    let ell = ell.max(1e-3);
    let ratio = checked_value("zrba_kernel_ratio", distance / ell)?;
    let exponent = checked_value("zrba_kernel_exponent", -0.5 * ratio.powi(2))?;
    checked_value("zrba_kernel_component", exponent.exp())
}

/// Computes a product RBF kernel using band, sheet, and echo distances.
pub fn product_kernel<G: ZFrameGeometry>(
    frame: &G,
    weights: &ZMetricWeights,
    indices_a: &ZIndex,
    indices_b: &ZIndex,
    ard: &ArdParameters,
) -> f32 {
    product_kernel_checked(frame, weights, indices_a, indices_b, ard).unwrap_or(0.0)
}

fn product_kernel_checked<G: ZFrameGeometry>(
    frame: &G,
    weights: &ZMetricWeights,
    indices_a: &ZIndex,
    indices_b: &ZIndex,
    ard: &ArdParameters,
) -> PureResult<f32> {
    weights.validate()?;
    ard.validate()?;
    let w = weights.normalised();
    let band = checked_value(
        "zrba_kernel_distance",
        frame.band_distance(indices_a.band, indices_b.band) * w.w_band.max(1e-6),
    )?;
    let sheet = checked_value(
        "zrba_kernel_distance",
        frame.sheet_distance(indices_a.sheet, indices_b.sheet) * w.w_sheet.max(1e-6),
    )?;
    let echo = checked_value(
        "zrba_kernel_distance",
        frame.echo_circular_distance(indices_a.echo, indices_b.echo) * w.w_echo.max(1e-6),
    )?;
    let k_band = kernel_component(band, ard.ell_band)?;
    let k_sheet = kernel_component(sheet, ard.ell_sheet)?;
    let k_echo = kernel_component(echo, ard.ell_echo)?;
    let kernel = checked_value(
        "zrba_product_kernel",
        ard.sigma2 * k_band * k_sheet * k_echo,
    )?;
    Ok(kernel)
}

/// Automatic relevance determination parameters per attention head.
#[derive(Clone, Debug)]
pub struct ArdParameters {
    pub ell_band: f32,
    pub ell_sheet: f32,
    pub ell_echo: f32,
    pub sigma2: f32,
}

impl ArdParameters {
    pub fn new(ell_band: f32, ell_sheet: f32, ell_echo: f32, sigma2: f32) -> Self {
        Self {
            ell_band: ell_band.max(1e-2),
            ell_sheet: ell_sheet.max(1e-2),
            ell_echo: ell_echo.max(1e-2),
            sigma2: sigma2.max(1e-4),
        }
    }

    fn validate(&self) -> PureResult<()> {
        validate_finite_value("zrba_ard_ell_band", self.ell_band)?;
        validate_finite_value("zrba_ard_ell_sheet", self.ell_sheet)?;
        validate_finite_value("zrba_ard_ell_echo", self.ell_echo)?;
        validate_finite_value("zrba_ard_sigma2", self.sigma2)
    }
}

impl Default for ArdParameters {
    fn default() -> Self {
        Self {
            ell_band: 1.0,
            ell_sheet: 1.0,
            ell_echo: 1.0,
            sigma2: 1.0,
        }
    }
}

/// Telemetry produced by the attention stack for each head.
#[derive(Clone, Debug)]
pub struct AttentionTelemetry {
    pub kernel_mean: Vec<f32>,
    pub kernel_min: Vec<f32>,
    pub kernel_max: Vec<f32>,
    pub head_entropy: Vec<f32>,
    pub length_scales: Vec<ArdParameters>,
}

impl AttentionTelemetry {
    pub fn new(heads: usize) -> Self {
        Self {
            kernel_mean: vec![0.0; heads],
            kernel_min: vec![0.0; heads],
            kernel_max: vec![0.0; heads],
            head_entropy: vec![0.0; heads],
            length_scales: vec![ArdParameters::default(); heads],
        }
    }
}

/// Output bundle from the attention module.
#[derive(Clone, Debug)]
pub struct ZRBFAttentionOutput {
    pub mean: Tensor,
    pub variance: Tensor,
    pub telemetry: AttentionTelemetry,
}

/// Multi-head scaled dot attention augmented with Z-RBF kernels.
#[derive(Debug)]
pub struct ZRBFAttention {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    metric: ZMetricWeights,
    ard: bool,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    head_params: Vec<ArdParameters>,
}

impl ZRBFAttention {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        metric: ZMetricWeights,
        ard: bool,
    ) -> PureResult<Self> {
        if d_model == 0 || n_heads == 0 || !d_model.is_multiple_of(n_heads) {
            return Err(TensorError::InvalidDimensions {
                rows: d_model,
                cols: n_heads,
            });
        }
        metric.validate()?;
        let head_dim = d_model / n_heads;
        let initializer = |rows: usize, cols: usize| {
            Tensor::from_fn(rows, cols, |r, c| {
                let seed = (r * cols + c) as f32;
                (seed.sin() * 0.02 + 0.02 * seed.cos()).tanh()
            })
        };
        let query = initializer(d_model, d_model)?;
        let key = initializer(d_model, d_model)?;
        let value = initializer(d_model, d_model)?;
        let output = initializer(d_model, d_model)?;
        let head_params = (0..n_heads)
            .map(|h| {
                if ard {
                    let scale = 1.0 + 0.1 * h as f32;
                    ArdParameters::new(1.0 * scale, 0.8 * scale, 0.6 * scale, 1.0)
                } else {
                    ArdParameters::default()
                }
            })
            .collect();
        Ok(Self {
            d_model,
            n_heads,
            head_dim,
            metric,
            ard,
            query,
            key,
            value,
            output,
            head_params,
        })
    }

    fn apply_linear(
        &self,
        tensor: &Tensor,
        weight: &Tensor,
        label: &'static str,
    ) -> PureResult<Tensor> {
        let output = relabel_non_finite(
            tensor.matmul_with_backend(weight, current_matmul_backend()),
            label,
        )?;
        validate_finite_tensor(label, &output)?;
        Ok(output)
    }

    pub fn forward<G: ZFrameGeometry>(
        &self,
        input: &ZTensor,
        frame: &G,
    ) -> PureResult<ZRBFAttentionOutput> {
        let (rows, cols) = input.mu.shape();
        if cols != self.d_model {
            return Err(TensorError::ShapeMismatch {
                left: input.mu.shape(),
                right: (rows, self.d_model),
            });
        }
        validate_finite_tensor("zrba_attention_mu", &input.mu)?;
        validate_finite_tensor("zrba_attention_sigma", &input.sigma)?;
        self.metric.validate()?;
        for head_params in &self.head_params {
            head_params.validate()?;
        }
        emit_metric_weights_normalise(&self.metric);
        if rows == 0 {
            return Ok(ZRBFAttentionOutput {
                mean: Tensor::zeros(rows, self.d_model)?,
                variance: Tensor::zeros(rows, self.d_model)?,
                telemetry: AttentionTelemetry::new(self.n_heads),
            });
        }
        let q = self.apply_linear(&input.mu, &self.query, "zrba_attention_query")?;
        let k = self.apply_linear(&input.mu, &self.key, "zrba_attention_key")?;
        let v = self.apply_linear(&input.mu, &self.value, "zrba_attention_value")?;
        let mut head_outputs = vec![0.0f32; rows * self.d_model];
        let mut head_variances = vec![0.0f32; rows * self.d_model];
        let mut telemetry = AttentionTelemetry::new(self.n_heads);

        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();
        let sigma_data = input.sigma.data();
        let mut workspace = vec![0.0f32; rows * rows];
        let mut kernel_matrix = vec![0.0f32; rows * rows];

        for head in 0..self.n_heads {
            let head_params = if self.ard {
                self.head_params[head].clone()
            } else {
                ArdParameters::default()
            };
            telemetry.length_scales[head] = head_params.clone();
            let offset = head * self.head_dim;
            let mut q_head = Vec::with_capacity(rows * self.head_dim);
            let mut k_head = Vec::with_capacity(rows * self.head_dim);
            let mut v_head = Vec::with_capacity(rows * self.head_dim);
            for row in 0..rows {
                let start = row * self.d_model + offset;
                let end = start + self.head_dim;
                q_head.extend_from_slice(&q_data[start..end]);
                k_head.extend_from_slice(&k_data[start..end]);
                v_head.extend_from_slice(&v_data[start..end]);
            }
            workspace.fill(0.0);
            kernel_matrix.fill(0.0);
            for i in 0..rows {
                for j in 0..rows {
                    let mut dot = 0.0f32;
                    let q_start = i * self.d_model + offset;
                    let k_start = j * self.d_model + offset;
                    for d in 0..self.head_dim {
                        let product = checked_value(
                            "zrba_attention_score_product",
                            q_data[q_start + d] * k_data[k_start + d],
                        )?;
                        dot = checked_value("zrba_attention_score_sum", dot + product)?;
                    }
                    let scaled = checked_value(
                        "zrba_attention_scaled_score",
                        dot / (self.head_dim as f32).sqrt(),
                    )?;
                    let kernel = product_kernel_checked(
                        frame,
                        &self.metric,
                        &input.indices[i],
                        &input.indices[j],
                        &head_params,
                    )?;
                    workspace[i * rows + j] =
                        checked_value("zrba_attention_workspace_logit", scaled + kernel)?;
                    kernel_matrix[i * rows + j] = kernel;
                }
            }
            // Row-wise softmax.
            let mut normalised_rows = 0usize;
            let mut zero_sum_rows = 0usize;
            let mut exp_sum_total = 0.0f32;
            let mut normalised_sum_total = 0.0f32;
            let mut entropy_total = 0.0f32;
            let mut entropy_max = 0.0f32;
            let mut dominant_probability = 0.0f32;
            for i in 0..rows {
                let row = &mut workspace[i * rows..(i + 1) * rows];
                let max = row
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, |acc, v| acc.max(v));
                validate_finite_value("zrba_attention_workspace_row_max", max)?;
                let mut sum = 0.0f32;
                for value in row.iter_mut() {
                    let shifted = checked_value("zrba_attention_workspace_shifted", *value - max)?;
                    *value = checked_value("zrba_attention_workspace_exp", shifted.exp())?;
                    sum = checked_value("zrba_attention_workspace_exp_sum", sum + *value)?;
                }
                if sum <= f32::EPSILON {
                    zero_sum_rows = zero_sum_rows.saturating_add(1);
                    continue;
                }
                exp_sum_total = checked_value(
                    "zrba_attention_workspace_exp_sum_total",
                    exp_sum_total + sum,
                )?;
                for value in row.iter_mut() {
                    *value = checked_value("zrba_attention_workspace_probability", *value / sum)?;
                }
                let mut normalised_sum = 0.0f32;
                for &value in row.iter() {
                    normalised_sum = checked_value(
                        "zrba_attention_workspace_probability_sum",
                        normalised_sum + value,
                    )?;
                }
                let row_dominant = row
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .fold(0.0f32, |best, value| best.max(value));
                let entropy = row
                    .iter()
                    .filter(|v| **v > 0.0)
                    .map(|v| -v * (v.max(1e-9)).ln())
                    .sum::<f32>();
                validate_finite_value("zrba_attention_workspace_entropy", entropy)?;
                normalised_rows = normalised_rows.saturating_add(1);
                normalised_sum_total = checked_value(
                    "zrba_attention_workspace_probability_sum_total",
                    normalised_sum_total + normalised_sum,
                )?;
                entropy_total = checked_value(
                    "zrba_attention_workspace_entropy_total",
                    entropy_total + entropy,
                )?;
                entropy_max = entropy_max.max(entropy);
                dominant_probability = dominant_probability.max(row_dominant);
                telemetry.head_entropy[head] = checked_value(
                    "zrba_attention_head_entropy",
                    telemetry.head_entropy[head] + entropy / rows as f32,
                )?;
            }
            emit_tensor_op(
                "zrba_workspace_softmax_summary",
                &[rows, rows, self.head_dim],
                &[rows, rows],
            );
            emit_tensor_op_meta("zrba_workspace_softmax_summary", || {
                serde_json::json!({
                    "backend": "summary_cpu",
                    "requested_backend": "host",
                    "kind": "zrba_workspace_softmax_summary",
                    "head": head,
                    "rows": rows,
                    "cols": rows,
                    "head_dim": self.head_dim,
                    "normalised_rows": normalised_rows,
                    "zero_sum_rows": zero_sum_rows,
                    "exp_sum_total": exp_sum_total,
                    "normalised_sum_total": normalised_sum_total,
                    "mean_row_sum": if normalised_rows == 0 {
                        0.0
                    } else {
                        normalised_sum_total / normalised_rows as f32
                    },
                    "entropy_total": entropy_total,
                    "mean_entropy": if rows == 0 {
                        0.0
                    } else {
                        entropy_total / rows as f32
                    },
                    "max_entropy": entropy_max,
                    "dominant_probability": dominant_probability,
                })
            });

            let mut kernel_acc = 0.0f32;
            let mut kernel_min = f32::INFINITY;
            let mut kernel_max = f32::NEG_INFINITY;

            for kernel in kernel_matrix.iter() {
                validate_finite_value("zrba_attention_kernel", *kernel)?;
                kernel_acc = checked_value("zrba_attention_kernel_sum", kernel_acc + *kernel)?;
                kernel_min = kernel_min.min(*kernel);
                kernel_max = kernel_max.max(*kernel);
            }
            let normaliser = (rows * rows) as f32;
            telemetry.kernel_mean[head] =
                checked_value("zrba_attention_kernel_mean", kernel_acc / normaliser)?;
            telemetry.kernel_min[head] = if kernel_min.is_finite() {
                kernel_min
            } else {
                0.0
            };
            telemetry.kernel_max[head] = if kernel_max.is_finite() {
                kernel_max
            } else {
                0.0
            };

            let q_head = Tensor::from_vec(rows, self.head_dim, q_head)?;
            let k_head = Tensor::from_vec(rows, self.head_dim, k_head)?;
            let v_head = Tensor::from_vec(rows, self.head_dim, v_head)?;
            let kernel_bias = Tensor::from_vec(rows, rows, kernel_matrix.clone())?;
            let head_mean = q_head.scaled_dot_attention_with_backend(
                &k_head,
                &v_head,
                1,
                rows,
                1.0 / (self.head_dim as f32).sqrt(),
                None,
                Some(&kernel_bias),
                current_attention_backend(),
            )?;
            validate_finite_tensor("zrba_attention_head_mean", &head_mean)?;
            let head_mean_data = head_mean.data();

            for i in 0..rows {
                let output_start = i * self.d_model + offset;
                let mean_start = i * self.head_dim;
                head_outputs[output_start..output_start + self.head_dim]
                    .copy_from_slice(&head_mean_data[mean_start..mean_start + self.head_dim]);
                for dim in 0..self.head_dim {
                    let mut variance = head_params.sigma2;
                    for j in 0..rows {
                        let weight = workspace[i * rows + j];
                        let sigma_index = j * self.d_model + offset + dim;
                        let weight_sq =
                            checked_value("zrba_attention_variance_weight", weight * weight)?;
                        let contribution = checked_value(
                            "zrba_attention_variance_contribution",
                            weight_sq * sigma_data[sigma_index].abs(),
                        )?;
                        variance =
                            checked_value("zrba_attention_variance", variance + contribution)?;
                    }
                    head_variances[i * self.d_model + offset + dim] =
                        checked_value("zrba_attention_variance", variance.max(1e-6))?;
                }
            }
        }

        validate_finite_slice("zrba_attention_head_outputs", &head_outputs)?;
        let head_outputs = relabel_non_finite(
            Tensor::from_vec(rows, self.d_model, head_outputs)?
                .matmul_with_backend(&self.output, current_matmul_backend()),
            "zrba_attention_output",
        )?;
        validate_finite_tensor("zrba_attention_output", &head_outputs)?;
        let head_variances = Tensor::from_vec(rows, self.d_model, head_variances)?;
        validate_finite_tensor("zrba_attention_variance", &head_variances)?;
        Ok(ZRBFAttentionOutput {
            mean: head_outputs,
            variance: head_variances,
            telemetry,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, OnceLock};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn reference_attention_mean<G: ZFrameGeometry>(
        attention: &ZRBFAttention,
        input: &ZTensor,
        frame: &G,
    ) -> PureResult<Tensor> {
        let (rows, _) = input.mu.shape();
        let q = attention.apply_linear(&input.mu, &attention.query, "zrba_attention_query")?;
        let k = attention.apply_linear(&input.mu, &attention.key, "zrba_attention_key")?;
        let v = attention.apply_linear(&input.mu, &attention.value, "zrba_attention_value")?;
        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();
        let mut head_outputs = vec![0.0f32; rows * attention.d_model];

        for head in 0..attention.n_heads {
            let head_params = if attention.ard {
                attention.head_params[head].clone()
            } else {
                ArdParameters::default()
            };
            let offset = head * attention.head_dim;
            let mut logits = vec![0.0f32; rows * rows];
            for i in 0..rows {
                for j in 0..rows {
                    let mut dot = 0.0f32;
                    let q_start = i * attention.d_model + offset;
                    let k_start = j * attention.d_model + offset;
                    for d in 0..attention.head_dim {
                        dot += q_data[q_start + d] * k_data[k_start + d];
                    }
                    let kernel = product_kernel(
                        frame,
                        &attention.metric,
                        &input.indices[i],
                        &input.indices[j],
                        &head_params,
                    );
                    logits[i * rows + j] = dot / (attention.head_dim as f32).sqrt() + kernel;
                }
            }

            for i in 0..rows {
                let row = &mut logits[i * rows..(i + 1) * rows];
                let max = row
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
                let mut sum = 0.0f32;
                for value in row.iter_mut() {
                    *value = (*value - max).exp();
                    sum += *value;
                }
                if sum <= f32::EPSILON {
                    continue;
                }
                for value in row.iter_mut() {
                    *value /= sum;
                }

                for dim in 0..attention.head_dim {
                    let mut value = 0.0f32;
                    for j in 0..rows {
                        let weight = row[j];
                        let v_index = j * attention.d_model + offset + dim;
                        value += weight * v_data[v_index];
                    }
                    head_outputs[i * attention.d_model + offset + dim] = value;
                }
            }
        }

        Tensor::from_vec(rows, attention.d_model, head_outputs)?.matmul(&attention.output)
    }

    #[test]
    fn product_kernel_respects_weights() {
        let frame = SimpleZFrame::new(4, 4, 8);
        let weights = ZMetricWeights {
            w_band: 1.0,
            w_sheet: 0.5,
            w_echo: 0.25,
        };
        let ard = ArdParameters::new(1.0, 1.0, 1.0, 1.0);
        let a = ZIndex {
            band: 0,
            sheet: 0,
            echo: 0,
        };
        let b = ZIndex {
            band: 1,
            sheet: 2,
            echo: 3,
        };
        let kernel = product_kernel(&frame, &weights, &a, &b, &ard);
        let norm = weights.normalised();
        let band = frame.band_distance(a.band, b.band) * norm.w_band.max(1e-6);
        let sheet = frame.sheet_distance(a.sheet, b.sheet) * norm.w_sheet.max(1e-6);
        let echo = frame.echo_circular_distance(a.echo, b.echo) * norm.w_echo.max(1e-6);
        let expected = ard.sigma2
            * (-0.5 * (band / ard.ell_band).powi(2)).exp()
            * (-0.5 * (sheet / ard.ell_sheet).powi(2)).exp()
            * (-0.5 * (echo / ard.ell_echo).powi(2)).exp();
        assert!((kernel - expected).abs() < 1e-6);
    }

    #[test]
    fn attention_shapes_match() {
        let frame = SimpleZFrame::new(3, 3, 4);
        let indices = vec![
            ZIndex {
                band: 0,
                sheet: 0,
                echo: 0,
            },
            ZIndex {
                band: 1,
                sheet: 1,
                echo: 2,
            },
            ZIndex {
                band: 2,
                sheet: 2,
                echo: 3,
            },
        ];
        let mu = Tensor::from_vec(
            3,
            6,
            vec![
                0.1, 0.0, 0.2, -0.1, 0.05, 0.3, 0.2, 0.1, -0.2, 0.0, 0.1, -0.1, 0.05, -0.05, 0.2,
                0.3, -0.2, 0.1,
            ],
        )
        .unwrap();
        let sigma = Tensor::from_vec(
            3,
            6,
            vec![
                0.05, 0.04, 0.03, 0.02, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02,
                0.02, 0.02, 0.02, 0.02,
            ],
        )
        .unwrap();
        let tensor = ZTensor::new(mu, sigma, indices).unwrap();
        let attention = ZRBFAttention::new(6, 2, ZMetricWeights::default(), true).unwrap();
        let output = attention.forward(&tensor, &frame).unwrap();
        let reference = reference_attention_mean(&attention, &tensor, &frame).unwrap();
        assert_eq!(output.mean.shape(), (3, 6));
        assert_eq!(output.variance.shape(), (3, 6));
        assert_eq!(output.telemetry.kernel_mean.len(), 2);
        for (actual, expected) in output.mean.data().iter().zip(reference.data().iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn attention_empty_sequence_preserves_shape_and_zero_telemetry() {
        let frame = SimpleZFrame::new(3, 3, 4);
        let mu = Tensor::from_vec(0, 6, Vec::new()).unwrap();
        let sigma = Tensor::from_vec(0, 6, Vec::new()).unwrap();
        let tensor = ZTensor::new(mu, sigma, Vec::new()).unwrap();
        let attention = ZRBFAttention::new(6, 2, ZMetricWeights::default(), true).unwrap();

        let output = attention.forward(&tensor, &frame).unwrap();

        assert_eq!(output.mean.shape(), (0, 6));
        assert_eq!(output.variance.shape(), (0, 6));
        assert!(output.mean.data().is_empty());
        assert!(output.variance.data().is_empty());
        assert_eq!(output.telemetry.kernel_mean, vec![0.0, 0.0]);
        assert_eq!(output.telemetry.kernel_min, vec![0.0, 0.0]);
        assert_eq!(output.telemetry.kernel_max, vec![0.0, 0.0]);
        assert_eq!(output.telemetry.head_entropy, vec![0.0, 0.0]);
        assert_eq!(output.telemetry.length_scales.len(), 2);
    }

    #[test]
    fn attention_rejects_non_finite_metric_weight() {
        let err = ZRBFAttention::new(
            6,
            2,
            ZMetricWeights {
                w_band: f32::NAN,
                w_sheet: 0.7,
                w_echo: 0.5,
            },
            true,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zrba_metric_weight",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn attention_rejects_non_finite_mu_before_projection() {
        let frame = SimpleZFrame::new(3, 3, 4);
        let indices = vec![ZIndex {
            band: 0,
            sheet: 0,
            echo: 0,
        }];
        let mu = Tensor::from_vec(1, 2, vec![0.1, f32::INFINITY]).unwrap();
        let sigma = Tensor::from_vec(1, 2, vec![0.05, 0.04]).unwrap();
        let tensor = ZTensor::new(mu, sigma, indices).unwrap();
        let attention = ZRBFAttention::new(2, 1, ZMetricWeights::default(), true).unwrap();

        let err = attention.forward(&tensor, &frame).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zrba_attention_mu",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn attention_rejects_overflowing_workspace_score_product() {
        let frame = SimpleZFrame::new(3, 3, 4);
        let indices = vec![
            ZIndex {
                band: 0,
                sheet: 0,
                echo: 0,
            },
            ZIndex {
                band: 1,
                sheet: 1,
                echo: 1,
            },
        ];
        let mu = Tensor::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let sigma = Tensor::from_vec(2, 2, vec![0.05, 0.04, 0.03, 0.02]).unwrap();
        let tensor = ZTensor::new(mu, sigma, indices).unwrap();
        let mut attention = ZRBFAttention::new(2, 1, ZMetricWeights::default(), true).unwrap();
        attention.query = Tensor::from_vec(2, 2, vec![1.0e20, 0.0, 0.0, 1.0e20]).unwrap();
        attention.key = Tensor::from_vec(2, 2, vec![1.0e20, 0.0, 0.0, 1.0e20]).unwrap();

        let err = attention.forward(&tensor, &frame).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zrba_attention_score_product",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn attention_forward_emits_scaled_dot_attention_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let frame = SimpleZFrame::new(3, 3, 4);
        let indices = vec![
            ZIndex {
                band: 0,
                sheet: 0,
                echo: 0,
            },
            ZIndex {
                band: 1,
                sheet: 1,
                echo: 2,
            },
            ZIndex {
                band: 2,
                sheet: 2,
                echo: 3,
            },
        ];
        let mu = Tensor::from_vec(
            3,
            6,
            vec![
                0.1, 0.0, 0.2, -0.1, 0.05, 0.3, 0.2, 0.1, -0.2, 0.0, 0.1, -0.1, 0.05, -0.05, 0.2,
                0.3, -0.2, 0.1,
            ],
        )
        .unwrap();
        let sigma = Tensor::from_vec(
            3,
            6,
            vec![
                0.05, 0.04, 0.03, 0.02, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02,
                0.02, 0.02, 0.02, 0.02,
            ],
        )
        .unwrap();
        let tensor = ZTensor::new(mu, sigma, indices).unwrap();
        let attention = ZRBFAttention::new(6, 2, ZMetricWeights::default(), true).unwrap();
        let _ = attention.forward(&tensor, &frame).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let metric_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zrba_metric_weights_normalise" && data["values"] == 3
            })
            .expect("zrba_metric_weights_normalise metadata event");
        assert_eq!(metric_meta.1["backend"], "control_cpu");
        assert_eq!(metric_meta.1["requested_backend"], "host");
        assert_eq!(metric_meta.1["kind"], "zrba_metric_weights_normalise");
        assert_eq!(metric_meta.1["default_fallback"], false);
        assert!((metric_meta.1["distribution_sum"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-6);

        let workspace_softmax = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zrba_workspace_softmax_summary"
                    && data["head"] == 0
                    && data["rows"] == 3
                    && data["head_dim"] == 3
            })
            .expect("zrba_workspace_softmax_summary metadata event");
        assert_eq!(workspace_softmax.1["backend"], "summary_cpu");
        assert_eq!(workspace_softmax.1["requested_backend"], "host");
        assert_eq!(
            workspace_softmax.1["kind"],
            "zrba_workspace_softmax_summary"
        );
        assert_eq!(workspace_softmax.1["normalised_rows"], 3);
        assert_eq!(workspace_softmax.1["zero_sum_rows"], 0);
        assert!((workspace_softmax.1["mean_row_sum"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-6);
        assert!(workspace_softmax.1["mean_entropy"].as_f64().unwrap_or(0.0) > 0.0);
        assert!(
            workspace_softmax.1["dominant_probability"]
                .as_f64()
                .unwrap_or(0.0)
                > 0.0
        );

        let attention_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "scaled_dot_attention"
                    && data["contexts"] == 1
                    && data["sequence"] == 3
                    && data["head_dim"] == 3
            })
            .expect("scaled dot attention metadata event");
        assert!(attention_meta.1["backend"].as_str().is_some());
    }
}
