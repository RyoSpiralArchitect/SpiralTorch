// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// Mellin basis capturing log-scale harmonics of Z-space coordinates.
#[derive(Clone, Debug)]
pub struct MellinBasis {
    exponents: DVector<f64>,
}

impl MellinBasis {
    /// Builds a basis from Mellin exponents. Each exponent warps the
    /// corresponding coordinate prior to encoding in the VAE.
    pub fn new(exponents: Vec<f64>) -> Self {
        Self {
            exponents: DVector::from_vec(exponents),
        }
    }

    pub fn dimension(&self) -> usize {
        self.exponents.len()
    }

    /// Projects an input vector using the Mellin transform idea: x^s / ||x^s||.
    pub fn project(&self, input: &DVector<f64>) -> DVector<f64> {
        let mut projected = DVector::zeros(input.len());
        for (i, value) in input.iter().enumerate() {
            let exponent = self.exponents.get(i).cloned().unwrap_or(1.0);
            let warped = value.abs().max(1e-9).powf(exponent);
            projected[i] = if value.is_sign_negative() {
                -warped
            } else {
                warped
            };
        }
        let norm = projected.norm();
        if norm > 0.0 {
            projected / norm
        } else {
            projected
        }
    }
}

/// Minimal variational autoencoder tailored for Z-space telemetry.
#[derive(Clone, Debug)]
pub struct ZSpaceVae {
    latent_dim: usize,
    input_dim: usize,
    encoder_mu: DMatrix<f64>,
    encoder_logvar: DMatrix<f64>,
    decoder: DMatrix<f64>,
    bias_mu: DVector<f64>,
    bias_logvar: DVector<f64>,
    bias_decoder: DVector<f64>,
    rng: StdRng,
}

impl ZSpaceVae {
    /// Creates a VAE with orthogonal initial weights.
    pub fn new(input_dim: usize, latent_dim: usize, seed: u64) -> Self {
        let encoder_mu = DMatrix::identity(latent_dim, input_dim);
        let encoder_logvar = DMatrix::zeros(latent_dim, input_dim);
        let decoder = DMatrix::identity(input_dim, latent_dim);
        let bias_mu = DVector::zeros(latent_dim);
        let bias_logvar = DVector::from_element(latent_dim, -3.0);
        let bias_decoder = DVector::zeros(input_dim);
        Self {
            latent_dim,
            input_dim,
            encoder_mu,
            encoder_logvar,
            decoder,
            bias_mu,
            bias_logvar,
            bias_decoder,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Encodes an input vector into mean and log-variance.
    pub fn encode(&self, input: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        let mu = &self.encoder_mu * input + &self.bias_mu;
        let logvar = &self.encoder_logvar * input + &self.bias_logvar;
        (mu, logvar)
    }

    /// Mellin-projected encode step, used for Z-space harmonic compression.
    pub fn encode_with_mellin(
        &self,
        input: &DVector<f64>,
        basis: &MellinBasis,
    ) -> (DVector<f64>, DVector<f64>) {
        let projected = basis.project(input);
        self.encode(&projected)
    }

    /// Reparameterises the latent vector.
    pub fn sample_latent(&mut self, mu: &DVector<f64>, logvar: &DVector<f64>) -> DVector<f64> {
        let mut sample = DVector::zeros(self.latent_dim);
        for i in 0..self.latent_dim {
            let eps: f64 = self.rng.sample(StandardNormal);
            let std = (0.5 * logvar[i]).exp();
            sample[i] = mu[i] + std * eps;
        }
        sample
    }

    /// Deterministic latent sample useful for evaluation.
    pub fn mean_latent(&self, mu: &DVector<f64>) -> DVector<f64> {
        mu.clone()
    }

    /// Decodes a latent vector back into Z-space coordinates.
    pub fn decode(&self, latent: &DVector<f64>) -> DVector<f64> {
        &self.decoder * latent + &self.bias_decoder
    }

    /// Decodes a latent vector using a Mellin basis. Useful for generating
    /// narrative perturbations anchored in the harmonic space.
    pub fn decode_with_mellin(&self, latent: &DVector<f64>, basis: &MellinBasis) -> DVector<f64> {
        let decoded = self.decode(latent);
        basis.project(&decoded)
    }

    /// Full forward pass returning latent, reconstruction, and losses.
    pub fn forward(&mut self, input: &DVector<f64>) -> ZSpaceVaeState {
        let (mu, logvar) = self.encode(input);
        let latent = self.sample_latent(&mu, &logvar);
        let reconstruction = self.decode(&latent);
        let stats = ZSpaceVaeStats::from_forward(input, &reconstruction, &mu, &logvar);
        ZSpaceVaeState {
            latent,
            reconstruction,
            mu,
            logvar,
            stats,
        }
    }

    /// Lightweight parameter update that blends the decoder toward the provided
    /// reconstruction target using statistics from the forward pass.
    pub fn refine_decoder(&mut self, state: &ZSpaceVaeState, learning_rate: f64) {
        let lr = learning_rate.clamp(1e-6, 1e-1);
        let error = &state.reconstruction - &state.stats.target;
        let gradient = &error * state.latent.transpose();
        self.decoder = (&self.decoder - gradient * lr).clone_owned();
        self.bias_decoder = (&self.bias_decoder - error * lr).clone_owned();
    }
}

/// Book-keeping for a single VAE pass.
#[derive(Clone, Debug)]
pub struct ZSpaceVaeState {
    pub latent: DVector<f64>,
    pub reconstruction: DVector<f64>,
    pub mu: DVector<f64>,
    pub logvar: DVector<f64>,
    pub stats: ZSpaceVaeStats,
}

/// Loss bundle describing KL and reconstruction energy.
#[derive(Clone, Debug)]
pub struct ZSpaceVaeStats {
    pub recon_loss: f64,
    pub kl_loss: f64,
    pub evidence_lower_bound: f64,
    pub target: DVector<f64>,
}

impl ZSpaceVaeStats {
    fn from_forward(
        input: &DVector<f64>,
        reconstruction: &DVector<f64>,
        mu: &DVector<f64>,
        logvar: &DVector<f64>,
    ) -> Self {
        let diff = input - reconstruction;
        let recon_loss = diff.dot(&diff) / input.len() as f64;
        let mut kl = 0.0;
        for i in 0..mu.len() {
            kl += -0.5 * (1.0 + logvar[i] - mu[i].powi(2) - logvar[i].exp());
        }
        let elbo = -(recon_loss + kl);
        Self {
            recon_loss,
            kl_loss: kl,
            evidence_lower_bound: elbo,
            target: input.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mellin_projection_normalises() {
        let basis = MellinBasis::new(vec![1.0, 2.0, 0.5]);
        let input = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let projected = basis.project(&input);
        assert!((projected.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vae_forward_produces_losses() {
        let mut vae = ZSpaceVae::new(3, 2, 42);
        let input = DVector::from_vec(vec![0.5, -0.25, 0.75]);
        let state = vae.forward(&input);
        assert_eq!(state.latent.len(), 2);
        assert_eq!(state.reconstruction.len(), 3);
        assert!(state.stats.recon_loss >= 0.0);
        assert!(state.stats.kl_loss >= 0.0);
        assert!(state.stats.evidence_lower_bound <= 0.0);
    }

    #[test]
    fn refine_decoder_adjusts_bias() {
        let mut vae = ZSpaceVae::new(2, 2, 7);
        let input = DVector::from_vec(vec![0.2, -0.1]);
        let state = vae.forward(&input);
        let before = vae.bias_decoder.clone();
        vae.refine_decoder(&state, 1e-2);
        assert!((&before - &vae.bias_decoder).norm() > 0.0);
    }
}
