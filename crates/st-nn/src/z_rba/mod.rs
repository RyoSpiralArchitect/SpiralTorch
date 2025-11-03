// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Z-RBA: Residual Bayesian Attention modules specialised for Z-space geometry.
//!
//! This module glues together the geometric Z-frame distance metrics, residual
//! Beta gating, and covariance telemetry described in the Z-RBA blueprint.  The
//! implementation focuses on offering a pragmatic Rust surface that can be wired
//! into existing `st-nn` pipelines while keeping all intermediate uncertainty
//! statistics inspectable.

pub mod attention;
pub mod beta_residual;
pub mod cov_head;
pub mod telemetry;

use crate::{PureResult, Tensor, TensorError};
use st_core::ops::zspace_round::SpectralFeatureSample;

/// Mean + diagonal variance tensor tracked alongside Z-space indices.
#[derive(Clone, Debug)]
pub struct ZTensor {
    pub mu: Tensor,
    pub sigma: Tensor,
    pub indices: Vec<attention::ZIndex>,
}

impl ZTensor {
    /// Validates tensor dimensions and wraps them into a `ZTensor` bundle.
    pub fn new(mu: Tensor, sigma: Tensor, indices: Vec<ZIndex>) -> PureResult<Self> {
        let (rows, cols) = mu.shape();
        let (s_rows, s_cols) = sigma.shape();
        if rows != s_rows || cols != s_cols {
            return Err(TensorError::ShapeMismatch {
                left: mu.shape(),
                right: sigma.shape(),
            });
        }
        if indices.len() != rows {
            return Err(TensorError::DataLength {
                expected: rows,
                got: indices.len(),
            });
        }
        Ok(Self { mu, sigma, indices })
    }

    /// Returns the number of token rows tracked by this bundle.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Covariance head result paired with the gate and attention telemetry.
#[derive(Clone, Debug)]
pub struct ZCov {
    pub covariance: Tensor,
    pub gate: beta_residual::BetaGateSample,
    pub attention: attention::AttentionTelemetry,
    pub covariance_stats: cov_head::CovHeadTelemetry,
}

/// Configuration for the Z-RBA stack.
#[derive(Clone, Debug)]
pub struct ZRBAConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub metric: ZMetricWeights,
    pub ard: bool,
    pub cov_rank: usize,
    /// Momentum for the residual Beta gate's running statistics.
    pub gate_momentum: f32,
    /// Seed controlling the stochastic Beta samples.
    pub gate_seed: u64,
    /// When true the residual path uses the Beta expectation instead of samples.
    pub gate_use_expected: bool,
}

impl Default for ZRBAConfig {
    fn default() -> Self {
        Self {
            d_model: 128,
            n_heads: 4,
            metric: ZMetricWeights::default(),
            ard: true,
            cov_rank: 8,
            gate_momentum: 0.05,
            gate_seed: 42,
            gate_use_expected: true,
        }
    }
}

/// High-level module combining Z-RBF attention, residual Beta gating, and
/// covariance reconstruction.
#[derive(Debug)]
pub struct ZRBA {
    attn: attention::ZRBFAttention,
    gate: beta_residual::BetaGate,
    cov: cov_head::CovHead,
    use_expected_gate: bool,
}

impl ZRBA {
    /// Builds a new Z-RBA block using the provided configuration.
    pub fn new(config: ZRBAConfig) -> PureResult<Self> {
        let attn = attention::ZRBFAttention::new(
            config.d_model,
            config.n_heads,
            config.metric.clone(),
            config.ard,
        )?;
        let gate = beta_residual::BetaGate::new(beta_residual::BetaGateConfig {
            momentum: config.gate_momentum,
            seed: config.gate_seed,
        });
        let cov = cov_head::CovHead::new(config.cov_rank);
        Ok(Self {
            attn,
            gate,
            cov,
            use_expected_gate: config.gate_use_expected,
        })
    }

    /// Exposes the underlying attention module for fine-grained control.
    pub fn attention(&self) -> &attention::ZRBFAttention {
        &self.attn
    }

    /// Forward pass computing the residual attention update, Beta gate, and
    /// covariance telemetry.
    pub fn forward<G: ZFrameGeometry>(
        &self,
        input: &ZTensor,
        frame: &G,
        stats: &SpectralFeatureSample,
    ) -> PureResult<(ZTensor, ZCov, ZRBATelemetry)> {
        if input.is_empty() {
            return Err(TensorError::EmptyInput("zrba::forward"));
        }
        let attn_out: attention::ZRBFAttentionOutput = self.attn.forward(input, frame)?;
        let mut gate = self.gate.forward(stats, &input.indices);
        let gate_value = if self.use_expected_gate {
            gate.expected
        } else {
            gate.sample
        };
        gate.applied = gate_value;

        let gated_mu = input.mu.scale(gate_value)?;
        let mu = gated_mu.add(&attn_out.mean)?;

        let gated_sigma = input.sigma.scale(gate_value * gate_value)?;
        let sigma = gated_sigma.add(&attn_out.variance)?;

        let cov_out: cov_head::CovHeadOutput = self.cov.forward(&mu, &sigma)?;
        let output = ZTensor::new(mu.clone(), sigma.clone(), input.indices.clone())?;
        let cov_bundle = ZCov {
            covariance: cov_out.covariance.clone(),
            gate: gate.clone(),
            attention: attn_out.telemetry.clone(),
            covariance_stats: cov_out.telemetry.clone(),
        };
        let telemetry =
            telemetry::ZRBATelemetry::new(attn_out.telemetry, gate.clone(), cov_out.telemetry);
        Ok((output, cov_bundle, telemetry))
    }
}

pub use attention::{AttentionTelemetry, SimpleZFrame, ZFrameGeometry, ZIndex, ZMetricWeights};
pub use beta_residual::{BetaGate, BetaGateConfig, BetaGateSample};
pub use cov_head::{CovHead, CovHeadTelemetry};
pub use telemetry::{ReliabilityBin, ZRBAMetrics, ZRBATelemetry, ZTelemetryBundle};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn zrba_forward_produces_outputs() {
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
        let stats = SpectralFeatureSample {
            sheet_index: 1,
            sheet_confidence: 0.7,
            curvature: 0.2,
            spin: 0.1,
            energy: 0.5,
        };

        let config = ZRBAConfig {
            d_model: 6,
            n_heads: 2,
            metric: ZMetricWeights::default(),
            ard: true,
            cov_rank: 3,
            gate_momentum: 0.05,
            gate_seed: 7,
            gate_use_expected: true,
        };
        let zrba = ZRBA::new(config).unwrap();
        let (out, cov, telemetry) = zrba.forward(&tensor, &frame, &stats).unwrap();
        assert_eq!(out.mu.shape(), (3, 6));
        assert_eq!(out.sigma.shape(), (3, 6));
        assert_eq!(cov.covariance.shape(), (6, 6));
        assert_eq!(cov.attention.kernel_mean.len(), 2);
        assert_eq!(
            telemetry.gate.expected_value(),
            telemetry.gate.expected_value()
        );
        assert_eq!(telemetry.gate.applied, telemetry.gate.expected);
    }
}
