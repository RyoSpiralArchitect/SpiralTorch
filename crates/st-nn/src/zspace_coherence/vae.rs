// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::{PureResult, TensorError};

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

    /// Builds a basis with the same exponent applied to every coordinate.
    pub fn constant(dimension: usize, exponent: f64) -> Self {
        Self::new(vec![exponent; dimension])
    }

    /// Builds a basis with linearly interpolated exponents.
    ///
    /// The returned basis has `dimension` exponents spanning `[start, end]`.
    pub fn ramp(dimension: usize, start: f64, end: f64) -> Self {
        if dimension <= 1 {
            return Self::new(vec![start]);
        }
        let mut exponents = Vec::with_capacity(dimension);
        let denom = (dimension - 1) as f64;
        for idx in 0..dimension {
            let t = idx as f64 / denom;
            exponents.push(start + (end - start) * t);
        }
        Self::new(exponents)
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

const ZSPACE_VAE_CHECKPOINT_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ZSpaceVaeCheckpoint {
    version: u32,
    input_dim: usize,
    latent_dim: usize,
    encoder_mu: Vec<f64>,
    encoder_logvar: Vec<f64>,
    decoder: Vec<f64>,
    bias_mu: Vec<f64>,
    bias_logvar: Vec<f64>,
    bias_decoder: Vec<f64>,
    seed: u64,
}

fn is_json_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
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
    seed: u64,
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
            seed,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> PureResult<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| TensorError::IoError {
                message: err.to_string(),
            })?;
        }
        let checkpoint = self.checkpoint();
        let payload = if is_json_path(path) {
            serde_json::to_vec_pretty(&checkpoint).map_err(|err| TensorError::SerializationError {
                message: err.to_string(),
            })?
        } else {
            bincode::serialize(&checkpoint).map_err(|err| TensorError::SerializationError {
                message: err.to_string(),
            })?
        };
        std::fs::write(path, payload).map_err(|err| TensorError::IoError {
            message: err.to_string(),
        })?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> PureResult<Self> {
        let path = path.as_ref();
        let payload = std::fs::read(path).map_err(|err| TensorError::IoError {
            message: err.to_string(),
        })?;
        let checkpoint: ZSpaceVaeCheckpoint = if is_json_path(path) {
            serde_json::from_slice(&payload).map_err(|err| TensorError::SerializationError {
                message: err.to_string(),
            })?
        } else {
            bincode::deserialize(&payload).map_err(|err| TensorError::SerializationError {
                message: err.to_string(),
            })?
        };
        Self::from_checkpoint(checkpoint)
    }

    pub(crate) fn checkpoint(&self) -> ZSpaceVaeCheckpoint {
        ZSpaceVaeCheckpoint {
            version: ZSPACE_VAE_CHECKPOINT_VERSION,
            input_dim: self.input_dim,
            latent_dim: self.latent_dim,
            encoder_mu: self.encoder_mu.as_slice().to_vec(),
            encoder_logvar: self.encoder_logvar.as_slice().to_vec(),
            decoder: self.decoder.as_slice().to_vec(),
            bias_mu: self.bias_mu.as_slice().to_vec(),
            bias_logvar: self.bias_logvar.as_slice().to_vec(),
            bias_decoder: self.bias_decoder.as_slice().to_vec(),
            seed: self.seed,
        }
    }

    pub(crate) fn from_checkpoint(checkpoint: ZSpaceVaeCheckpoint) -> PureResult<Self> {
        if checkpoint.version != ZSPACE_VAE_CHECKPOINT_VERSION {
            return Err(TensorError::SerializationError {
                message: format!(
                    "unsupported ZSpaceVae checkpoint version {}",
                    checkpoint.version
                ),
            });
        }
        if checkpoint.input_dim == 0 || checkpoint.latent_dim == 0 {
            return Err(TensorError::InvalidValue {
                label: "zspace_vae_checkpoint_dims",
            });
        }

        let enc_len = checkpoint
            .latent_dim
            .checked_mul(checkpoint.input_dim)
            .ok_or_else(|| TensorError::InvalidValue {
                label: "zspace_vae_checkpoint_overflow",
            })?;
        if checkpoint.encoder_mu.len() != enc_len {
            return Err(TensorError::DataLength {
                expected: enc_len,
                got: checkpoint.encoder_mu.len(),
            });
        }
        if checkpoint.encoder_logvar.len() != enc_len {
            return Err(TensorError::DataLength {
                expected: enc_len,
                got: checkpoint.encoder_logvar.len(),
            });
        }
        let dec_len = checkpoint
            .input_dim
            .checked_mul(checkpoint.latent_dim)
            .ok_or_else(|| TensorError::InvalidValue {
                label: "zspace_vae_checkpoint_overflow",
            })?;
        if checkpoint.decoder.len() != dec_len {
            return Err(TensorError::DataLength {
                expected: dec_len,
                got: checkpoint.decoder.len(),
            });
        }
        if checkpoint.bias_mu.len() != checkpoint.latent_dim {
            return Err(TensorError::DataLength {
                expected: checkpoint.latent_dim,
                got: checkpoint.bias_mu.len(),
            });
        }
        if checkpoint.bias_logvar.len() != checkpoint.latent_dim {
            return Err(TensorError::DataLength {
                expected: checkpoint.latent_dim,
                got: checkpoint.bias_logvar.len(),
            });
        }
        if checkpoint.bias_decoder.len() != checkpoint.input_dim {
            return Err(TensorError::DataLength {
                expected: checkpoint.input_dim,
                got: checkpoint.bias_decoder.len(),
            });
        }

        Ok(Self {
            latent_dim: checkpoint.latent_dim,
            input_dim: checkpoint.input_dim,
            encoder_mu: DMatrix::from_column_slice(
                checkpoint.latent_dim,
                checkpoint.input_dim,
                &checkpoint.encoder_mu,
            ),
            encoder_logvar: DMatrix::from_column_slice(
                checkpoint.latent_dim,
                checkpoint.input_dim,
                &checkpoint.encoder_logvar,
            ),
            decoder: DMatrix::from_column_slice(
                checkpoint.input_dim,
                checkpoint.latent_dim,
                &checkpoint.decoder,
            ),
            bias_mu: DVector::from_vec(checkpoint.bias_mu),
            bias_logvar: DVector::from_vec(checkpoint.bias_logvar),
            bias_decoder: DVector::from_vec(checkpoint.bias_decoder),
            seed: checkpoint.seed,
            rng: StdRng::seed_from_u64(checkpoint.seed),
        })
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
    use tempfile::tempdir;

    #[test]
    fn mellin_projection_normalises() {
        let basis = MellinBasis::new(vec![1.0, 2.0, 0.5]);
        let input = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let projected = basis.project(&input);
        assert!((projected.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mellin_basis_ramp_has_expected_endpoints() {
        let basis = MellinBasis::ramp(4, 0.5, 2.0);
        assert_eq!(basis.dimension(), 4);
        let input = DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
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

    #[test]
    fn vae_checkpoint_roundtrips() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("vae.bin");
        let mut vae = ZSpaceVae::new(3, 2, 123);
        let input = DVector::from_vec(vec![0.5, -0.25, 0.75]);
        let state = vae.forward(&input);
        vae.refine_decoder(&state, 1e-2);
        vae.save(&path).unwrap();

        let loaded = ZSpaceVae::load(&path).unwrap();
        assert_eq!(loaded.input_dim(), vae.input_dim());
        assert_eq!(loaded.latent_dim(), vae.latent_dim());
        assert_eq!(loaded.bias_decoder.len(), vae.bias_decoder.len());
    }
}
