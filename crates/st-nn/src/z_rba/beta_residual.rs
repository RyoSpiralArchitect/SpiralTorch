// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Residual Beta gate that carries spectral uncertainty between layers.

use super::attention::ZIndex;
use crate::layers::linear::Linear;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Beta, Distribution};
use std::cell::{Cell, RefCell};

use st_core::ops::zspace_round::SpectralFeatureSample;

/// Snapshot of the sampled gate and its parameters.
#[derive(Clone, Debug)]
pub struct BetaGateSample {
    pub sample: f32,
    pub alpha: f32,
    pub beta: f32,
    pub expected: f32,
    pub variance: f32,
    pub features: [f32; 7],
}

impl BetaGateSample {
    pub fn expected_value(&self) -> f32 {
        self.expected
    }
}

/// Residual gate parameterised by a two-output linear map.
#[derive(Debug)]
pub struct BetaGate {
    phi: Linear,
    rng: RefCell<ChaCha20Rng>,
    ema_mean: Cell<f32>,
    ema_var: Cell<f32>,
    momentum: f32,
}

impl BetaGate {
    pub fn new() -> PureResult<Self> {
        let phi = Linear::new("zrba.beta_gate", 7, 2)?;
        Ok(Self {
            phi,
            rng: RefCell::new(ChaCha20Rng::seed_from_u64(42)),
            ema_mean: Cell::new(0.5),
            ema_var: Cell::new(0.25),
            momentum: 0.05,
        })
    }

    pub fn forward(
        &self,
        stats: &SpectralFeatureSample,
        indices: &[ZIndex],
    ) -> PureResult<BetaGateSample> {
        let features = self.build_features(stats, indices);
        let features_tensor = Tensor::from_vec(1, features.len(), features.to_vec())?;
        let logits = self.phi.forward(&features_tensor)?;
        let data = logits.data();
        let alpha_raw = data[0];
        let beta_raw = data[1];
        let alpha = self.softplus(alpha_raw) + 1e-3;
        let beta = self.softplus(beta_raw) + 1e-3;
        let expected = alpha / (alpha + beta);
        let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
        let sample = self.sample_beta(alpha, beta);
        self.update_ema(expected, variance);
        Ok(BetaGateSample {
            sample,
            alpha,
            beta,
            expected,
            variance,
            features,
        })
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
        if indices.is_empty() {
            features[4] = 0.0;
            features[5] = stats.sheet_index as f32;
            features[6] = 0.0;
            return features;
        }
        let mut mean_band = 0.0f32;
        let mut mean_sheet = 0.0f32;
        let mut mean_echo = 0.0f32;
        for index in indices {
            mean_band += index.band as f32;
            mean_sheet += index.sheet as f32;
            mean_echo += index.echo as f32;
        }
        let denom = indices.len() as f32;
        let avg_band = mean_band / denom;
        let avg_sheet = mean_sheet / denom;
        let avg_echo = mean_echo / denom;
        features[4] = avg_band;
        features[5] = 0.5 * (stats.sheet_index as f32 + avg_sheet);
        features[6] = avg_echo;
        features
    }

    fn softplus(&self, x: f32) -> f32 {
        (1.0 + x.exp()).ln()
    }

    fn sample_beta(&self, alpha: f32, beta: f32) -> f32 {
        let mut rng = self.rng.borrow_mut();
        let distribution =
            Beta::new(alpha as f64, beta as f64).unwrap_or_else(|_| Beta::new(1.0, 1.0).unwrap());
        distribution.sample(&mut *rng) as f32
    }

    fn update_ema(&self, mean: f32, variance: f32) {
        let ema_mean = self.ema_mean.get();
        let ema_var = self.ema_var.get();
        let new_mean = (1.0 - self.momentum) * ema_mean + self.momentum * mean;
        let new_var = (1.0 - self.momentum) * ema_var + self.momentum * variance;
        self.ema_mean.set(new_mean);
        self.ema_var.set(new_var.max(1e-6));
    }

    pub fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.phi.visit_parameters(visitor)
    }

    pub fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.phi.visit_parameters_mut(visitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_gate_emits_sample() {
        let gate = BetaGate::new().unwrap();
        let sample = gate
            .forward(
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
        assert!(sample.sample >= 0.0 && sample.sample <= 1.0);
        assert!(sample.alpha > 0.0 && sample.beta > 0.0);
        let (ema_mean, ema_var) = gate.ema();
        assert!(ema_mean > 0.0 && ema_var > 0.0);
    }
}
