// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Residual Beta gate that carries spectral uncertainty between layers.

use super::attention::ZIndex;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Beta, Distribution};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::{PureResult, TensorError};
use st_core::ops::zspace_round::SpectralFeatureSample;

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

/// Snapshot of the sampled gate and its parameters.
#[derive(Clone, Debug)]
pub struct BetaGateSample {
    pub sample: f32,
    pub alpha: f32,
    pub beta: f32,
    pub expected: f32,
    pub variance: f32,
    pub features: [f32; 7],
    pub applied: f32,
}

impl BetaGateSample {
    pub fn expected_value(&self) -> f32 {
        self.expected
    }

    fn validate(&self) -> PureResult<()> {
        validate_finite_value("beta_gate_sample", self.sample)?;
        validate_finite_value("beta_gate_alpha", self.alpha)?;
        validate_finite_value("beta_gate_beta", self.beta)?;
        validate_finite_value("beta_gate_expected", self.expected)?;
        validate_finite_value("beta_gate_variance", self.variance)?;
        validate_finite_slice("beta_gate_feature", &self.features)?;
        validate_finite_value("beta_gate_applied", self.applied)
    }
}

/// Configuration surface for the residual Beta gate.
#[derive(Clone, Debug)]
pub struct BetaGateConfig {
    pub momentum: f32,
    pub seed: u64,
}

impl Default for BetaGateConfig {
    fn default() -> Self {
        Self {
            momentum: 0.05,
            seed: 42,
        }
    }
}

/// Residual gate parameterised by a two-output linear map.
#[derive(Debug)]
pub struct BetaGate {
    weights: [[f32; 7]; 2],
    bias: [f32; 2],
    rng: RefCell<ChaCha20Rng>,
    ema_mean: Cell<f32>,
    ema_var: Cell<f32>,
    momentum: f32,
}

impl Default for BetaGate {
    fn default() -> Self {
        Self::new(BetaGateConfig::default())
    }
}

impl BetaGate {
    pub fn new(config: BetaGateConfig) -> Self {
        let mut weights = [[0.0f32; 7]; 2];
        for (row, row_weights) in weights.iter_mut().enumerate() {
            for (col, weight) in row_weights.iter_mut().enumerate() {
                let seed = (row * 7 + col) as f32;
                *weight = (seed.cos() * 0.05 + seed.sin() * 0.02).tanh();
            }
        }
        let bias = [0.1, 0.2];
        let momentum = if config.momentum.is_finite() {
            config.momentum.clamp(1e-4, 1.0)
        } else {
            BetaGateConfig::default().momentum
        };
        Self {
            weights,
            bias,
            rng: RefCell::new(ChaCha20Rng::seed_from_u64(config.seed)),
            ema_mean: Cell::new(0.5),
            ema_var: Cell::new(0.25),
            momentum,
        }
    }

    pub fn forward(&self, stats: &SpectralFeatureSample, indices: &[ZIndex]) -> BetaGateSample {
        self.try_forward(stats, indices)
            .unwrap_or_else(|_| self.neutral_sample(stats, indices))
    }

    pub fn try_forward(
        &self,
        stats: &SpectralFeatureSample,
        indices: &[ZIndex],
    ) -> PureResult<BetaGateSample> {
        let features = self.build_features(stats, indices);
        validate_finite_slice("beta_gate_feature", &features)?;
        let [alpha_raw, beta_raw] = self.linear_checked(&features)?;
        let alpha = checked_value("beta_gate_alpha", self.softplus_checked(alpha_raw)? + 1e-3)?;
        let beta = checked_value("beta_gate_beta", self.softplus_checked(beta_raw)? + 1e-3)?;
        let concentration = checked_value("beta_gate_concentration", alpha + beta)?;
        let expected = checked_value("beta_gate_expected", alpha / concentration)?;
        let variance_denominator = checked_value(
            "beta_gate_variance",
            concentration.powi(2) * (concentration + 1.0),
        )?;
        let variance = checked_value("beta_gate_variance", (alpha * beta) / variance_denominator)?;
        let sample = self.sample_beta_checked(alpha, beta)?;
        let result = BetaGateSample {
            sample,
            alpha,
            beta,
            expected,
            variance,
            features,
            applied: sample,
        };
        result.validate()?;
        self.update_ema_checked(expected, variance)?;
        Ok(result)
    }

    pub fn ema(&self) -> (f32, f32) {
        (self.ema_mean.get(), self.ema_var.get())
    }

    fn build_features(&self, stats: &SpectralFeatureSample, indices: &[ZIndex]) -> [f32; 7] {
        let mut features = [0.0f32; 7];
        features[0] = stats.sheet_confidence;
        features[1] = stats.curvature;
        features[2] = stats.spin;
        features[3] = stats.energy;
        let (band, echo) = dominant_band_and_echo(indices);
        features[4] = band;
        features[5] = stats.sheet_index as f32;
        features[6] = echo;
        features
    }

    fn linear_checked(&self, features: &[f32; 7]) -> PureResult<[f32; 2]> {
        let mut outputs = [0.0f32; 2];
        for (row, weights) in self.weights.iter().enumerate() {
            let mut acc = checked_value("beta_gate_bias", self.bias[row])?;
            for (feature, weight) in features.iter().zip(weights.iter()) {
                let product = checked_value("beta_gate_linear_product", feature * weight)?;
                acc = checked_value("beta_gate_linear_output", acc + product)?;
            }
            outputs[row] = acc;
        }
        Ok(outputs)
    }

    fn softplus_checked(&self, x: f32) -> PureResult<f32> {
        validate_finite_value("beta_gate_softplus_input", x)?;
        let output = if x > 0.0 {
            x + (-x).exp().ln_1p()
        } else {
            x.exp().ln_1p()
        };
        checked_value("beta_gate_softplus", output)
    }

    fn sample_beta_checked(&self, alpha: f32, beta: f32) -> PureResult<f32> {
        validate_finite_value("beta_gate_alpha", alpha)?;
        validate_finite_value("beta_gate_beta", beta)?;
        let mut rng = self.rng.borrow_mut();
        let distribution =
            Beta::new(alpha as f64, beta as f64).unwrap_or_else(|_| Beta::new(1.0, 1.0).unwrap());
        checked_value("beta_gate_sample", distribution.sample(&mut *rng) as f32)
    }

    fn update_ema_checked(&self, mean: f32, variance: f32) -> PureResult<()> {
        validate_finite_value("beta_gate_expected", mean)?;
        validate_finite_value("beta_gate_variance", variance)?;
        validate_finite_value("beta_gate_momentum", self.momentum)?;
        let ema_mean = self.ema_mean.get();
        let ema_var = self.ema_var.get();
        validate_finite_value("beta_gate_ema_mean", ema_mean)?;
        validate_finite_value("beta_gate_ema_var", ema_var)?;
        let new_mean = checked_value(
            "beta_gate_ema_mean",
            (1.0 - self.momentum) * ema_mean + self.momentum * mean,
        )?;
        let new_var = checked_value(
            "beta_gate_ema_var",
            (1.0 - self.momentum) * ema_var + self.momentum * variance,
        )?;
        self.ema_mean.set(new_mean);
        self.ema_var.set(new_var.max(1e-6));
        Ok(())
    }

    fn neutral_sample(&self, stats: &SpectralFeatureSample, indices: &[ZIndex]) -> BetaGateSample {
        let mut features = self.build_features(stats, indices);
        for value in &mut features {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        BetaGateSample {
            sample: 0.5,
            alpha: 1.0,
            beta: 1.0,
            expected: 0.5,
            variance: 1.0 / 12.0,
            features,
            applied: 0.5,
        }
    }
}

fn dominant_band_and_echo(indices: &[ZIndex]) -> (f32, f32) {
    if indices.is_empty() {
        return (0.0, 0.0);
    }
    let mut band_counts: HashMap<usize, usize> = HashMap::new();
    let mut echo_counts: HashMap<usize, usize> = HashMap::new();
    for index in indices {
        *band_counts.entry(index.band).or_insert(0) += 1;
        *echo_counts.entry(index.echo).or_insert(0) += 1;
    }
    let mut dominant_band = 0usize;
    let mut band_count = 0usize;
    for (band, count) in band_counts.into_iter() {
        if count > band_count || (count == band_count && band < dominant_band) {
            dominant_band = band;
            band_count = count;
        }
    }
    let mut dominant_echo = 0usize;
    let mut echo_count = 0usize;
    for (echo, count) in echo_counts.into_iter() {
        if count > echo_count || (count == echo_count && echo < dominant_echo) {
            dominant_echo = echo;
            echo_count = count;
        }
    }
    let dominant_band = dominant_band as f32;
    let dominant_echo = dominant_echo as f32;
    (dominant_band, dominant_echo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_gate_emits_sample() {
        let gate = BetaGate::new(BetaGateConfig::default());
        let sample = gate.forward(
            &SpectralFeatureSample {
                sheet_index: 1,
                sheet_confidence: 0.8,
                curvature: 0.1,
                spin: 0.05,
                energy: 0.2,
            },
            &[ZIndex {
                band: 1,
                sheet: 2,
                echo: 3,
            }],
        );
        assert!(sample.sample >= 0.0 && sample.sample <= 1.0);
        assert!(sample.alpha > 0.0 && sample.beta > 0.0);
        assert!((sample.applied - sample.sample).abs() <= f32::EPSILON);
        let (ema_mean, ema_var) = gate.ema();
        assert!(ema_mean > 0.0 && ema_var > 0.0);
    }

    #[test]
    fn beta_gate_try_forward_rejects_non_finite_features_without_updating_ema() {
        let gate = BetaGate::new(BetaGateConfig::default());
        let before = gate.ema();

        let err = gate
            .try_forward(
                &SpectralFeatureSample {
                    sheet_index: 1,
                    sheet_confidence: 0.8,
                    curvature: f32::INFINITY,
                    spin: 0.05,
                    energy: 0.2,
                },
                &[ZIndex {
                    band: 1,
                    sheet: 2,
                    echo: 3,
                }],
            )
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "beta_gate_feature",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(gate.ema(), before);
    }

    #[test]
    fn beta_gate_sanitizes_non_finite_momentum() {
        let gate = BetaGate::new(BetaGateConfig {
            momentum: f32::NAN,
            seed: 7,
        });

        let _ = gate
            .try_forward(
                &SpectralFeatureSample {
                    sheet_index: 1,
                    sheet_confidence: 0.8,
                    curvature: 0.1,
                    spin: 0.05,
                    energy: 0.2,
                },
                &[ZIndex {
                    band: 1,
                    sheet: 2,
                    echo: 3,
                }],
            )
            .unwrap();

        let (ema_mean, ema_var) = gate.ema();
        assert!(ema_mean.is_finite());
        assert!(ema_var.is_finite());
    }
}
