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
    pub applied: f32,
}

impl BetaGateSample {
    pub fn expected_value(&self) -> f32 {
        self.expected
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
        Self {
            weights,
            bias,
            rng: RefCell::new(ChaCha20Rng::seed_from_u64(config.seed)),
            ema_mean: Cell::new(0.5),
            ema_var: Cell::new(0.25),
            momentum: config.momentum.clamp(1e-4, 1.0),
        }
    }

    pub fn forward(&self, stats: &SpectralFeatureSample, indices: &[ZIndex]) -> BetaGateSample {
        let features = self.build_features(stats, indices);
        let [alpha_raw, beta_raw] = self.linear(&features);
        let alpha = self.softplus(alpha_raw) + 1e-3;
        let beta = self.softplus(beta_raw) + 1e-3;
        let expected = alpha / (alpha + beta);
        let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
        let sample = self.sample_beta(alpha, beta);
        self.update_ema(expected, variance);
        BetaGateSample {
            sample,
            alpha,
            beta,
            expected,
            variance,
            features,
            applied: sample,
        }
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

    fn linear(&self, features: &[f32; 7]) -> [f32; 2] {
        let mut outputs = [0.0f32; 2];
        for (row, weights) in self.weights.iter().enumerate() {
            let mut acc = self.bias[row];
            for (feature, weight) in features.iter().zip(weights.iter()) {
                acc += feature * weight;
            }
            outputs[row] = acc;
        }
        outputs
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
}
