// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::vae::{
    MellinBasis, ZSpaceVae, ZSpaceVaeBatchStats, ZSpaceVaeCheckpoint, ZSpaceVaeOptimizerKind,
    ZSpaceVaeState,
};
use crate::{PureResult, TensorError};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use st_tensor::LanguageWaveEncoder;
use std::path::Path;

/// Minimal text -> Z-space -> VAE pipeline (no tokenizer).
///
/// This is intended as the entry point for "Z-Space native" NLP experiments:
/// raw text is lifted into a hyperbolic chart via [`LanguageWaveEncoder`], then
/// compressed by [`ZSpaceVae`].
#[derive(Clone, Debug)]
pub struct ZSpaceTextVae {
    window_chars: usize,
    input_dim: usize,
    encoder: LanguageWaveEncoder,
    vae: ZSpaceVae,
}

const ZSPACE_TEXT_VAE_CHECKPOINT_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ZSpaceTextVaeCheckpoint {
    version: u32,
    window_chars: usize,
    curvature: f32,
    temperature: f32,
    vae: ZSpaceVaeCheckpoint,
}

fn is_json_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
}

impl ZSpaceTextVae {
    pub fn new(
        window_chars: usize,
        latent_dim: usize,
        curvature: f32,
        temperature: f32,
        seed: u64,
    ) -> PureResult<Self> {
        if window_chars == 0 {
            return Err(TensorError::InvalidValue {
                label: "zspace_text_vae_window_chars",
            });
        }
        if latent_dim == 0 {
            return Err(TensorError::InvalidValue {
                label: "zspace_text_vae_latent_dim",
            });
        }

        let encoder = LanguageWaveEncoder::new(curvature, temperature)?;
        let input_dim = window_chars
            .checked_mul(2)
            .ok_or(TensorError::InvalidValue {
                label: "zspace_text_vae_input_dim_overflow",
            })?;
        let vae = ZSpaceVae::new(input_dim, latent_dim, seed);
        Ok(Self {
            window_chars,
            input_dim,
            encoder,
            vae,
        })
    }

    pub fn window_chars(&self) -> usize {
        self.window_chars
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn latent_dim(&self) -> usize {
        self.vae.latent_dim()
    }

    pub fn curvature(&self) -> f32 {
        self.encoder.curvature()
    }

    pub fn temperature(&self) -> f32 {
        self.encoder.temperature()
    }

    pub fn optimizer_kind(&self) -> ZSpaceVaeOptimizerKind {
        self.vae.optimizer_kind()
    }

    pub fn configure_optimizer(
        &mut self,
        kind: ZSpaceVaeOptimizerKind,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        rms_decay: f64,
        grad_clip: Option<f64>,
    ) -> PureResult<()> {
        self.vae
            .configure_optimizer(kind, beta1, beta2, epsilon, rms_decay, grad_clip)
    }

    pub fn configure_optimizer_by_name(
        &mut self,
        optimizer: &str,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        rms_decay: f64,
        grad_clip: Option<f64>,
    ) -> PureResult<()> {
        self.vae
            .configure_optimizer_by_name(optimizer, beta1, beta2, epsilon, rms_decay, grad_clip)
    }

    pub fn reset_optimizer(&mut self) {
        self.vae.reset_optimizer();
    }

    pub fn encode_text(&self, text: &str) -> PureResult<DVector<f64>> {
        let window = fit_text_window(text, self.window_chars);
        let tensor = self.encoder.encode_z_space(&window)?;
        let (rows, cols) = tensor.shape();
        if rows != 1 || cols != self.input_dim {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        Ok(DVector::from_iterator(
            cols,
            tensor.data().iter().map(|value| f64::from(*value)),
        ))
    }

    pub fn encode_text_with_mellin(
        &self,
        text: &str,
        basis: &MellinBasis,
    ) -> PureResult<DVector<f64>> {
        let encoded = self.encode_text(text)?;
        if basis.dimension() != encoded.len() {
            return Err(TensorError::InvalidDimensions {
                rows: basis.dimension(),
                cols: encoded.len(),
            });
        }
        Ok(basis.project(&encoded))
    }

    pub fn forward_encoded(&mut self, encoded: &DVector<f64>) -> PureResult<ZSpaceVaeState> {
        if encoded.len() != self.input_dim {
            return Err(TensorError::InvalidDimensions {
                rows: encoded.len(),
                cols: self.input_dim,
            });
        }
        Ok(self.vae.forward(encoded))
    }

    pub fn forward_mean_encoded(&self, encoded: &DVector<f64>) -> PureResult<ZSpaceVaeState> {
        if encoded.len() != self.input_dim {
            return Err(TensorError::InvalidDimensions {
                rows: encoded.len(),
                cols: self.input_dim,
            });
        }
        self.vae.forward_mean(encoded)
    }

    pub fn forward_text(&mut self, text: &str) -> PureResult<ZSpaceVaeState> {
        let encoded = self.encode_text(text)?;
        self.forward_encoded(&encoded)
    }

    pub fn forward_mean_text(&self, text: &str) -> PureResult<ZSpaceVaeState> {
        let encoded = self.encode_text(text)?;
        self.forward_mean_encoded(&encoded)
    }

    pub fn forward_text_with_mellin(
        &mut self,
        text: &str,
        basis: &MellinBasis,
    ) -> PureResult<ZSpaceVaeState> {
        let projected = self.encode_text_with_mellin(text, basis)?;
        self.forward_encoded(&projected)
    }

    pub fn forward_mean_text_with_mellin(
        &self,
        text: &str,
        basis: &MellinBasis,
    ) -> PureResult<ZSpaceVaeState> {
        let projected = self.encode_text_with_mellin(text, basis)?;
        self.forward_mean_encoded(&projected)
    }

    pub fn train_encoded(
        &mut self,
        encoded: &DVector<f64>,
        learning_rate: f64,
        kl_weight: f64,
    ) -> PureResult<ZSpaceVaeState> {
        if encoded.len() != self.input_dim {
            return Err(TensorError::InvalidDimensions {
                rows: encoded.len(),
                cols: self.input_dim,
            });
        }
        self.vae.train_step(encoded, learning_rate, kl_weight)
    }

    pub fn train_encoded_batch(
        &mut self,
        encoded: &[DVector<f64>],
        learning_rate: f64,
        kl_weight: f64,
    ) -> PureResult<ZSpaceVaeBatchStats> {
        self.vae.train_batch(encoded, learning_rate, kl_weight)
    }

    pub fn evaluate_encoded_batch(
        &self,
        encoded: &[DVector<f64>],
        kl_weight: f64,
    ) -> PureResult<ZSpaceVaeBatchStats> {
        self.vae.evaluate_batch(encoded, kl_weight)
    }

    pub fn train_text(
        &mut self,
        text: &str,
        learning_rate: f64,
        kl_weight: f64,
    ) -> PureResult<ZSpaceVaeState> {
        let encoded = self.encode_text(text)?;
        self.train_encoded(&encoded, learning_rate, kl_weight)
    }

    pub fn train_text_with_mellin(
        &mut self,
        text: &str,
        basis: &MellinBasis,
        learning_rate: f64,
        kl_weight: f64,
    ) -> PureResult<ZSpaceVaeState> {
        let projected = self.encode_text_with_mellin(text, basis)?;
        self.train_encoded(&projected, learning_rate, kl_weight)
    }

    pub fn train_text_batch(
        &mut self,
        texts: &[String],
        learning_rate: f64,
        kl_weight: f64,
    ) -> PureResult<ZSpaceVaeBatchStats> {
        let mut encoded = Vec::with_capacity(texts.len());
        for text in texts {
            encoded.push(self.encode_text(text)?);
        }
        self.train_encoded_batch(&encoded, learning_rate, kl_weight)
    }

    pub fn train_text_batch_with_mellin(
        &mut self,
        texts: &[String],
        basis: &MellinBasis,
        learning_rate: f64,
        kl_weight: f64,
    ) -> PureResult<ZSpaceVaeBatchStats> {
        let mut encoded = Vec::with_capacity(texts.len());
        for text in texts {
            encoded.push(self.encode_text_with_mellin(text, basis)?);
        }
        self.train_encoded_batch(&encoded, learning_rate, kl_weight)
    }

    pub fn evaluate_text_batch(
        &self,
        texts: &[String],
        kl_weight: f64,
    ) -> PureResult<ZSpaceVaeBatchStats> {
        let mut encoded = Vec::with_capacity(texts.len());
        for text in texts {
            encoded.push(self.encode_text(text)?);
        }
        self.evaluate_encoded_batch(&encoded, kl_weight)
    }

    pub fn evaluate_text_batch_with_mellin(
        &self,
        texts: &[String],
        basis: &MellinBasis,
        kl_weight: f64,
    ) -> PureResult<ZSpaceVaeBatchStats> {
        let mut encoded = Vec::with_capacity(texts.len());
        for text in texts {
            encoded.push(self.encode_text_with_mellin(text, basis)?);
        }
        self.evaluate_encoded_batch(&encoded, kl_weight)
    }

    pub fn refine_decoder(&mut self, state: &ZSpaceVaeState, learning_rate: f64) {
        self.vae.refine_decoder(state, learning_rate);
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
            serde_json::to_vec_pretty(&checkpoint).map_err(|err| {
                TensorError::SerializationError {
                    message: err.to_string(),
                }
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
        let checkpoint: ZSpaceTextVaeCheckpoint = if is_json_path(path) {
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

    fn checkpoint(&self) -> ZSpaceTextVaeCheckpoint {
        ZSpaceTextVaeCheckpoint {
            version: ZSPACE_TEXT_VAE_CHECKPOINT_VERSION,
            window_chars: self.window_chars,
            curvature: self.curvature(),
            temperature: self.temperature(),
            vae: self.vae.checkpoint(),
        }
    }

    fn from_checkpoint(checkpoint: ZSpaceTextVaeCheckpoint) -> PureResult<Self> {
        if checkpoint.version != ZSPACE_TEXT_VAE_CHECKPOINT_VERSION {
            return Err(TensorError::SerializationError {
                message: format!(
                    "unsupported ZSpaceTextVae checkpoint version {}",
                    checkpoint.version
                ),
            });
        }
        if checkpoint.window_chars == 0 {
            return Err(TensorError::InvalidValue {
                label: "zspace_text_vae_checkpoint_window_chars",
            });
        }
        let input_dim =
            checkpoint
                .window_chars
                .checked_mul(2)
                .ok_or(TensorError::InvalidValue {
                    label: "zspace_text_vae_checkpoint_input_dim_overflow",
                })?;

        let encoder = LanguageWaveEncoder::new(checkpoint.curvature, checkpoint.temperature)?;
        let vae = ZSpaceVae::from_checkpoint(checkpoint.vae)?;
        if vae.input_dim() != input_dim {
            return Err(TensorError::InvalidDimensions {
                rows: vae.input_dim(),
                cols: input_dim,
            });
        }
        Ok(Self {
            window_chars: checkpoint.window_chars,
            input_dim,
            encoder,
            vae,
        })
    }
}

fn fit_text_window(text: &str, window_chars: usize) -> String {
    let mut out = String::with_capacity(window_chars);
    let mut count = 0usize;
    for ch in text.chars() {
        if count >= window_chars {
            break;
        }
        out.push(ch);
        count += 1;
    }
    if count < window_chars {
        out.extend(std::iter::repeat_n(' ', window_chars - count));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn text_vae_encodes_fixed_window() {
        let vae = ZSpaceTextVae::new(32, 8, -1.0, 1.0, 42).unwrap();
        let encoded = vae.encode_text("hello").unwrap();
        assert_eq!(encoded.len(), 64);
    }

    #[test]
    fn text_vae_forward_matches_dims() {
        let mut vae = ZSpaceTextVae::new(16, 4, -1.0, 1.0, 7).unwrap();
        let state = vae.forward_text("SpiralTorch").unwrap();
        assert_eq!(state.reconstruction.len(), vae.input_dim());
        assert_eq!(state.latent.len(), vae.latent_dim());
    }

    #[test]
    fn text_vae_train_text_updates_reconstruction_path() {
        let mut vae = ZSpaceTextVae::new(16, 4, -1.0, 1.0, 7).unwrap();
        let before = vae
            .forward_mean_text("SpiralTorch")
            .unwrap()
            .stats
            .recon_loss;

        for _ in 0..16 {
            vae.train_text("SpiralTorch", 1e-2, 1e-3).unwrap();
        }

        let after = vae
            .forward_mean_text("SpiralTorch")
            .unwrap()
            .stats
            .recon_loss;
        assert!(
            after < before,
            "expected text VAE training to reduce recon loss: before={before}, after={after}"
        );
    }

    #[test]
    fn text_vae_train_batch_with_mellin_reports_validation_stats() {
        let mut vae = ZSpaceTextVae::new(16, 4, -1.0, 1.0, 7).unwrap();
        vae.configure_optimizer_by_name("adam", 0.9, 0.999, 1e-8, 0.99, Some(5.0))
            .unwrap();
        let basis = MellinBasis::ramp(vae.input_dim(), 0.8, 1.2);
        let batch = vec![
            "SpiralTorch learns".to_string(),
            "Z space breathes".to_string(),
            "VAE batches align".to_string(),
        ];
        let before = vae
            .evaluate_text_batch_with_mellin(&batch, &basis, 1e-4)
            .unwrap()
            .weighted_loss;

        let mut stats = None;
        for _ in 0..24 {
            stats = Some(
                vae.train_text_batch_with_mellin(&batch, &basis, 1e-2, 1e-4)
                    .unwrap(),
            );
        }

        let stats = stats.unwrap();
        let after = vae
            .evaluate_text_batch_with_mellin(&batch, &basis, 1e-4)
            .unwrap()
            .weighted_loss;
        assert_eq!(stats.batch_size, 3);
        assert_eq!(stats.optimizer, ZSpaceVaeOptimizerKind::Adam);
        assert!(stats.gradient_l2 > 0.0);
        assert!(
            after < before,
            "expected text VAE batch training to reduce weighted loss: before={before}, after={after}"
        );
    }

    #[test]
    fn text_vae_checkpoint_roundtrips() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("text_vae.json");
        let mut vae = ZSpaceTextVae::new(16, 4, -1.0, 1.0, 7).unwrap();
        let state = vae.forward_text("SpiralTorch").unwrap();
        vae.refine_decoder(&state, 1e-2);
        vae.save(&path).unwrap();

        let loaded = ZSpaceTextVae::load(&path).unwrap();
        assert_eq!(loaded.window_chars(), vae.window_chars());
        assert_eq!(loaded.input_dim(), vae.input_dim());
        assert_eq!(loaded.latent_dim(), vae.latent_dim());
        assert!((loaded.curvature() - vae.curvature()).abs() < 1e-6);
        assert!((loaded.temperature() - vae.temperature()).abs() < 1e-6);
    }
}
