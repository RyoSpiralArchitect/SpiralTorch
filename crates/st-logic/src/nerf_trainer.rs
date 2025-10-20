// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "nerf")]

use rand::{rngs::StdRng, Rng, SeedableRng};
use st_nn::module::Module;
use st_tensor::{PureResult, Tensor, TensorError};
use st_vision::datasets::{MultiViewDatasetAdapter, RayBatch};
use st_vision::nerf::NerfField;

/// Configuration for the [`NerfTrainer`].
#[derive(Clone, Debug)]
pub struct NerfTrainingConfig {
    pub samples_per_ray: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub steps_per_epoch: usize,
    pub stratified: bool,
    pub seed: u64,
}

impl Default for NerfTrainingConfig {
    fn default() -> Self {
        Self {
            samples_per_ray: 32,
            batch_size: 1024,
            learning_rate: 5e-4,
            steps_per_epoch: 1,
            stratified: true,
            seed: 13,
        }
    }
}

impl NerfTrainingConfig {
    fn validate(&self) -> PureResult<()> {
        if self.samples_per_ray == 0 || self.batch_size == 0 || self.steps_per_epoch == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: self
                    .samples_per_ray
                    .max(self.batch_size)
                    .max(self.steps_per_epoch),
                cols: 0,
            });
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "learning_rate",
                value: self.learning_rate,
            });
        }
        Ok(())
    }
}

/// Telemetry gathered over a training run.
#[derive(Clone, Copy, Debug, Default)]
pub struct NerfTrainingStats {
    pub loss: f32,
    pub avg_transmittance: f32,
}

/// Minimal NeRF trainer that reuses the SpiralTorch module abstraction.
#[derive(Debug)]
pub struct NerfTrainer {
    field: NerfField,
    config: NerfTrainingConfig,
    rng: StdRng,
}

impl NerfTrainer {
    /// Builds a trainer for the provided field and configuration.
    pub fn new(field: NerfField, config: NerfTrainingConfig) -> PureResult<Self> {
        config.validate()?;
        Ok(Self {
            field,
            rng: StdRng::seed_from_u64(config.seed),
            config,
        })
    }

    /// Returns a shared reference to the underlying field.
    pub fn field(&self) -> &NerfField {
        &self.field
    }

    /// Returns a mutable reference to the underlying field.
    pub fn field_mut(&mut self) -> &mut NerfField {
        &mut self.field
    }

    /// Runs a single optimisation step over a random ray batch.
    pub fn train_step(
        &mut self,
        dataset: &MultiViewDatasetAdapter,
    ) -> PureResult<NerfTrainingStats> {
        let batch = dataset.sample_batch(&mut self.rng, self.config.batch_size)?;
        let (positions, maybe_directions, deltas) = self.sample_points(&batch)?;
        let input = if let Some(directions) = maybe_directions.as_ref() {
            self.field.assemble_input(&positions, Some(directions))?
        } else {
            self.field.assemble_input(&positions, None)?
        };
        let outputs = self.field.forward(&input)?;
        let stats = self.backpropagate(&batch, &outputs, &deltas, &input)?;
        self.field.apply_step(self.config.learning_rate)?;
        Ok(stats)
    }

    /// Trains for a full epoch accumulating the average loss/opacity.
    pub fn train_epoch(
        &mut self,
        dataset: &MultiViewDatasetAdapter,
    ) -> PureResult<NerfTrainingStats> {
        let mut total_loss = 0.0f32;
        let mut total_trans = 0.0f32;
        for _ in 0..self.config.steps_per_epoch {
            let stats = self.train_step(dataset)?;
            total_loss += stats.loss;
            total_trans += stats.avg_transmittance;
        }
        let steps = self.config.steps_per_epoch as f32;
        Ok(NerfTrainingStats {
            loss: total_loss / steps,
            avg_transmittance: total_trans / steps,
        })
    }

    /// Renders a batch of rays without updating the field parameters.
    pub fn render_batch(&mut self, batch: &RayBatch) -> PureResult<Tensor> {
        let (positions, maybe_directions, deltas) = self.sample_points(batch)?;
        let input = if let Some(directions) = maybe_directions.as_ref() {
            self.field.assemble_input(&positions, Some(directions))?
        } else {
            self.field.assemble_input(&positions, None)?
        };
        let outputs = self.field.forward(&input)?;
        let colors = self.composite_colors(batch, &outputs, &deltas);
        Tensor::from_vec(batch.len(), 3, colors)
    }

    fn sample_points(
        &mut self,
        batch: &RayBatch,
    ) -> PureResult<(Tensor, Option<Tensor>, Vec<f32>)> {
        let layout = self.field.sample_layout();
        let origin_shape = batch.origins.shape();
        if origin_shape.1 != layout.position_dims {
            return Err(TensorError::ShapeMismatch {
                left: (origin_shape.0, layout.position_dims),
                right: origin_shape,
            });
        }
        let direction_shape = batch.directions.shape();
        if layout.direction_dims != 0 && direction_shape.1 != layout.direction_dims {
            return Err(TensorError::ShapeMismatch {
                left: (direction_shape.0, layout.direction_dims),
                right: direction_shape,
            });
        }
        let batch_size = batch.len();
        let total_samples = batch_size * self.config.samples_per_ray;
        let mut positions = Vec::with_capacity(total_samples * layout.position_dims);
        let mut directions = if layout.direction_dims > 0 {
            Vec::with_capacity(total_samples * layout.direction_dims)
        } else {
            Vec::new()
        };
        let mut deltas = Vec::with_capacity(total_samples);
        let origins = batch.origins.data();
        let dirs = batch.directions.data();
        let bounds = batch.bounds.data();
        let stratified = self.config.stratified;
        for ray in 0..batch_size {
            let origin_offset = ray * layout.position_dims;
            let dir_offset = ray * layout.direction_dims;
            let near = bounds[ray * 2];
            let far = bounds[ray * 2 + 1];
            let span = (far - near).max(1e-3);
            let base_interval = span / self.config.samples_per_ray as f32;
            for sample in 0..self.config.samples_per_ray {
                let jitter = if stratified {
                    self.rng.gen::<f32>()
                } else {
                    0.5
                };
                let t = near + (sample as f32 + jitter.clamp(0.0, 0.999)) * base_interval;
                let next_t = if sample + 1 == self.config.samples_per_ray {
                    far
                } else {
                    near + (sample as f32 + 1.0) * base_interval
                };
                let delta = (next_t - t).max(1e-3);
                deltas.push(delta);
                for dim in 0..layout.position_dims {
                    let origin = origins[origin_offset + dim];
                    let direction = if layout.direction_dims > 0 {
                        dirs[dir_offset + dim]
                    } else {
                        0.0
                    };
                    positions.push(origin + direction * t);
                }
                for dim in 0..layout.direction_dims {
                    directions.push(dirs[dir_offset + dim]);
                }
            }
        }
        let position_tensor = Tensor::from_vec(total_samples, layout.position_dims, positions)?;
        let direction_tensor = if layout.direction_dims > 0 {
            Some(Tensor::from_vec(
                total_samples,
                layout.direction_dims,
                directions,
            )?)
        } else {
            None
        };
        Ok((position_tensor, direction_tensor, deltas))
    }

    fn composite_colors(&self, batch: &RayBatch, outputs: &Tensor, deltas: &[f32]) -> Vec<f32> {
        let batch_size = batch.len();
        let samples_per_ray = self.config.samples_per_ray;
        let mut colors = vec![0.0f32; batch_size * 3];
        let mut offset = 0usize;
        for ray in 0..batch_size {
            let mut trans = 1.0f32;
            let mut accum = [0.0f32; 3];
            for sample in 0..samples_per_ray {
                let idx = ray * samples_per_ray + sample;
                let sample_offset = idx * 4;
                let sigma = outputs.data()[sample_offset].max(0.0);
                let rgb = &outputs.data()[sample_offset + 1..sample_offset + 4];
                let delta = deltas[idx];
                let alpha = 1.0 - (-sigma * delta).exp();
                let weight = trans * alpha;
                accum[0] += weight * rgb[0];
                accum[1] += weight * rgb[1];
                accum[2] += weight * rgb[2];
                trans *= 1.0 - alpha;
            }
            colors[offset] = accum[0];
            colors[offset + 1] = accum[1];
            colors[offset + 2] = accum[2];
            offset += 3;
        }
        colors
    }

    fn backpropagate(
        &mut self,
        batch: &RayBatch,
        outputs: &Tensor,
        deltas: &[f32],
        input: &Tensor,
    ) -> PureResult<NerfTrainingStats> {
        let batch_size = batch.len();
        let samples_per_ray = self.config.samples_per_ray;
        let total_samples = batch_size * samples_per_ray;
        let mut trans_before = vec![0.0f32; total_samples];
        let mut weights = vec![0.0f32; total_samples];
        let mut alphas = vec![0.0f32; total_samples];
        let mut sigma_grad_scale = vec![0.0f32; total_samples];
        let outputs_data = outputs.data();
        let colors_target = batch.colors.data();
        let mut predicted = vec![0.0f32; batch_size * 3];
        let mut final_transmittance = vec![0.0f32; batch_size];
        for ray in 0..batch_size {
            let mut trans = 1.0f32;
            let mut accum = [0.0f32; 3];
            for sample in 0..samples_per_ray {
                let idx = ray * samples_per_ray + sample;
                let off = idx * 4;
                let raw_sigma = outputs_data[off];
                let (sigma, grad_scale) = if raw_sigma > 0.0 {
                    (raw_sigma, 1.0)
                } else {
                    (0.0, 0.0)
                };
                sigma_grad_scale[idx] = grad_scale;
                trans_before[idx] = trans;
                let delta = deltas[idx];
                let alpha = 1.0 - (-sigma * delta).exp();
                alphas[idx] = alpha;
                let weight = trans * alpha;
                weights[idx] = weight;
                accum[0] += weight * outputs_data[off + 1];
                accum[1] += weight * outputs_data[off + 2];
                accum[2] += weight * outputs_data[off + 3];
                trans *= 1.0 - alpha;
            }
            final_transmittance[ray] = trans;
            let base = ray * 3;
            predicted[base] = accum[0];
            predicted[base + 1] = accum[1];
            predicted[base + 2] = accum[2];
        }

        let mut grad = vec![0.0f32; total_samples * 4];
        let mut total_loss = 0.0f32;
        let grad_scale = 2.0 / (batch_size as f32 * 3.0);
        for ray in 0..batch_size {
            let base = ray * 3;
            let mut diff = [0.0f32; 3];
            let mut grad_color = [0.0f32; 3];
            for channel in 0..3 {
                diff[channel] = predicted[base + channel] - colors_target[base + channel];
                total_loss += diff[channel] * diff[channel];
                grad_color[channel] = diff[channel] * grad_scale;
            }
            let mut future_sum = 0.0f32;
            for sample in (0..samples_per_ray).rev() {
                let idx = ray * samples_per_ray + sample;
                let off = idx * 4;
                let rgb = &outputs_data[off + 1..off + 4];
                let dot = grad_color[0] * rgb[0] + grad_color[1] * rgb[1] + grad_color[2] * rgb[2];
                grad[off + 1] = grad_color[0] * weights[idx];
                grad[off + 2] = grad_color[1] * weights[idx];
                grad[off + 3] = grad_color[2] * weights[idx];
                let delta = deltas[idx];
                let exp_term = 1.0 - alphas[idx];
                let self_term = trans_before[idx] * delta * exp_term * dot;
                let grad_sigma = (self_term - delta * future_sum) * sigma_grad_scale[idx];
                grad[off] = grad_sigma;
                future_sum += weights[idx] * dot;
            }
        }

        self.field.zero_accumulators()?;
        let grad_tensor = Tensor::from_vec(total_samples, 4, grad)?;
        let _ = self.field.backward(input, &grad_tensor)?;
        let avg_trans = final_transmittance.iter().sum::<f32>() / batch_size as f32;
        Ok(NerfTrainingStats {
            loss: total_loss / (batch_size as f32 * 3.0),
            avg_transmittance: avg_trans,
        })
    }
}
