// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Z-RBF attention specialised for Z-space indices.

use crate::z_rba::ZTensor;
use crate::{PureResult, Tensor, TensorError};

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
        if sum <= f32::EPSILON {
            return Self::default();
        }
        Self {
            w_band: self.w_band / sum,
            w_sheet: self.w_sheet / sum,
            w_echo: self.w_echo / sum,
        }
    }
}

fn kernel_component(distance: f32, ell: f32) -> f32 {
    let ell = ell.max(1e-3);
    (-0.5 * (distance / ell).powi(2)).exp()
}

/// Computes a product RBF kernel using band, sheet, and echo distances.
pub fn product_kernel<G: ZFrameGeometry>(
    frame: &G,
    weights: &ZMetricWeights,
    indices_a: &ZIndex,
    indices_b: &ZIndex,
    ard: &ArdParameters,
) -> f32 {
    let w = weights.normalised();
    let band = frame.band_distance(indices_a.band, indices_b.band) * w.w_band.max(1e-6);
    let sheet = frame.sheet_distance(indices_a.sheet, indices_b.sheet) * w.w_sheet.max(1e-6);
    let echo = frame.echo_circular_distance(indices_a.echo, indices_b.echo) * w.w_echo.max(1e-6);
    let k_band = kernel_component(band, ard.ell_band);
    let k_sheet = kernel_component(sheet, ard.ell_sheet);
    let k_echo = kernel_component(echo, ard.ell_echo);
    ard.sigma2 * k_band * k_sheet * k_echo
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
        if d_model == 0 || n_heads == 0 || d_model % n_heads != 0 {
            return Err(TensorError::InvalidDimensions {
                rows: d_model,
                cols: n_heads,
            });
        }
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

    fn apply_linear(&self, tensor: &Tensor, weight: &Tensor) -> PureResult<Tensor> {
        tensor.matmul(weight)
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
        let q = self.apply_linear(&input.mu, &self.query)?;
        let k = self.apply_linear(&input.mu, &self.key)?;
        let v = self.apply_linear(&input.mu, &self.value)?;
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
            workspace.fill(0.0);
            kernel_matrix.fill(0.0);
            for i in 0..rows {
                for j in 0..rows {
                    let mut dot = 0.0f32;
                    let q_start = i * self.d_model + offset;
                    let k_start = j * self.d_model + offset;
                    for d in 0..self.head_dim {
                        dot += q_data[q_start + d] * k_data[k_start + d];
                    }
                    let scaled = dot / (self.head_dim as f32).sqrt();
                    let kernel = product_kernel(
                        frame,
                        &self.metric,
                        &input.indices[i],
                        &input.indices[j],
                        &head_params,
                    );
                    workspace[i * rows + j] = scaled + kernel;
                    kernel_matrix[i * rows + j] = kernel;
                }
            }
            // Row-wise softmax.
            for i in 0..rows {
                let row = &mut workspace[i * rows..(i + 1) * rows];
                let max = row
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, |acc, v| acc.max(v));
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
                let entropy = row
                    .iter()
                    .filter(|v| **v > 0.0)
                    .map(|v| -v * (v.max(1e-9)).ln())
                    .sum::<f32>();
                telemetry.head_entropy[head] += entropy / rows as f32;
            }

            let mut kernel_acc = 0.0f32;
            let mut kernel_min = f32::INFINITY;
            let mut kernel_max = f32::NEG_INFINITY;

            for kernel in kernel_matrix.iter() {
                kernel_acc += *kernel;
                kernel_min = kernel_min.min(*kernel);
                kernel_max = kernel_max.max(*kernel);
            }
            let normaliser = (rows * rows) as f32;
            telemetry.kernel_mean[head] = kernel_acc / normaliser;
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

            for i in 0..rows {
                for dim in 0..self.head_dim {
                    let mut value = 0.0f32;
                    let mut variance = head_params.sigma2;
                    for j in 0..rows {
                        let weight = workspace[i * rows + j];
                        let v_index = j * self.d_model + offset + dim;
                        value += weight * v_data[v_index];
                        let sigma_index = j * self.d_model + offset + dim;
                        variance += weight * weight * sigma_data[sigma_index].abs();
                    }
                    head_outputs[i * self.d_model + offset + dim] = value;
                    head_variances[i * self.d_model + offset + dim] = variance.max(1e-6);
                }
            }
        }

        let head_outputs =
            Tensor::from_vec(rows, self.d_model, head_outputs)?.matmul(&self.output)?;
        let head_variances = Tensor::from_vec(rows, self.d_model, head_variances)?;
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
        let band = frame.band_distance(a.band, b.band) as f32 * norm.w_band.max(1e-6);
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
        assert_eq!(output.mean.shape(), (3, 6));
        assert_eq!(output.variance.shape(), (3, 6));
        assert_eq!(output.telemetry.kernel_mean.len(), 2);
    }
}
