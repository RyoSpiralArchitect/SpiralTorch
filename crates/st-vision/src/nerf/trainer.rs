// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use rand::{rngs::StdRng, Rng, SeedableRng};
use st_nn::module::Module;
use st_tensor::{PureResult, Tensor, TensorError};

use super::integral::integrate_rays;
use super::{checked_elements, validate_finite};
use crate::datasets::{MultiViewDatasetAdapter, RayBatch};
use crate::nerf::NerfField;
use crate::tensor_contract::checked_allocation;

/// Configuration for the [`NerfTrainer`].
#[derive(Clone, Debug)]
pub struct NerfTrainingConfig {
    pub samples_per_ray: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub steps_per_epoch: usize,
    /// Draw a point uniformly inside each bin instead of its midpoint.
    /// Jitter changes the evaluation point, never the bin's integration width.
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

/// NeRF trainer with black-background, piecewise-constant ray quadrature.
///
/// Bounds describe the parameter `t` in `origin + direction * t`; density is
/// measured per unit of `t`. Directions are not implicitly normalized. Use unit
/// directions if bounds/density are expressed in physical distance units.
/// A field may omit direction conditioning, but rays still require physical
/// directions with the same dimension as their origins.
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
        validate_sample_layout(&field, &config)?;
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

    /// Runs one optimisation step. Sampling state advances only on success.
    /// Gradient/parameter mutation follows [`Module`] semantics; this is not a
    /// transactional rollback of all parameters if a later module update fails.
    pub fn train_step(
        &mut self,
        dataset: &MultiViewDatasetAdapter,
    ) -> PureResult<NerfTrainingStats> {
        let mut rng = self.rng.clone();
        let batch = dataset.sample_batch(&mut rng, self.config.batch_size)?;
        let (input, widths) = self.sample_input(&batch, &mut rng)?;
        let outputs = self.field.forward(&input)?;
        let integral = integrate_rays(
            &outputs,
            &widths,
            self.config.samples_per_ray,
            Some(&batch.colors),
        )?;
        let gradient = integral.gradient.ok_or(TensorError::InvalidValue {
            label: "nerf_missing_output_gradient",
        })?;
        self.field.zero_accumulators()?;
        let _ = self.field.backward(&input, &gradient)?;
        self.field.apply_step(self.config.learning_rate)?;
        self.rng = rng;
        Ok(NerfTrainingStats {
            loss: integral.loss,
            avg_transmittance: integral.avg_transmittance,
        })
    }

    /// Trains for a full epoch accumulating the average loss/transmittance.
    pub fn train_epoch(
        &mut self,
        dataset: &MultiViewDatasetAdapter,
    ) -> PureResult<NerfTrainingStats> {
        let mut total_loss = 0.0f64;
        let mut total_trans = 0.0f64;
        for _ in 0..self.config.steps_per_epoch {
            let stats = self.train_step(dataset)?;
            total_loss += f64::from(stats.loss);
            total_trans += f64::from(stats.avg_transmittance);
        }
        let steps = self.config.steps_per_epoch as f64;
        Ok(NerfTrainingStats {
            loss: (total_loss / steps) as f32,
            avg_transmittance: (total_trans / steps) as f32,
        })
    }

    /// Renders validated rays. Zero-width intervals are transparent; negative
    /// or nonfinite intervals are errors. Failed calls do not consume jitter.
    pub fn render_batch(&mut self, batch: &RayBatch) -> PureResult<Tensor> {
        let mut rng = self.rng.clone();
        let (input, widths) = self.sample_input(batch, &mut rng)?;
        let outputs = self.field.forward(&input)?;
        let integral = integrate_rays(&outputs, &widths, self.config.samples_per_ray, None)?;
        self.rng = rng;
        Ok(integral.colors)
    }

    fn sample_input(&self, batch: &RayBatch, rng: &mut StdRng) -> PureResult<(Tensor, Vec<f64>)> {
        let layout = self.field.sample_layout();
        // field_mut() permits replacing the entire field after construction.
        let total_samples = validate_sample_layout(&self.field, &self.config)?;
        if batch.origins.shape() != (self.config.batch_size, layout.position_dims) {
            return Err(TensorError::ShapeMismatch {
                left: (self.config.batch_size, layout.position_dims),
                right: batch.origins.shape(),
            });
        }
        let rays = batch.validated()?;
        // Write the assembled field input directly, avoiding intermediate
        // position/direction tensors and their subsequent concatenation copy.
        let mut input = Tensor::zeros(total_samples, layout.try_total_dims()?)?;
        let mut widths = Vec::with_capacity(total_samples);
        let dims = input.shape().1;
        for ray in 0..batch.len() {
            let origin_offset = ray * layout.position_dims;
            let near = f64::from(rays.bounds.data()[ray * 2]);
            let far = f64::from(rays.bounds.data()[ray * 2 + 1]);
            let width = (far - near) / self.config.samples_per_ray as f64;
            for sample in 0..self.config.samples_per_ray {
                let jitter = if self.config.stratified {
                    f64::from(rng.gen::<f32>())
                } else {
                    0.5
                };
                let t = near + (sample as f64 + jitter) * width;
                let offset = (ray * self.config.samples_per_ray + sample) * dims;
                let row = &mut input.data_mut()[offset..offset + dims];
                for (dim, value) in row[..layout.position_dims].iter_mut().enumerate() {
                    let origin = f64::from(rays.origins.data()[origin_offset + dim]);
                    let direction = f64::from(rays.directions.data()[origin_offset + dim]);
                    *value = (origin + direction * t) as f32;
                }
                if layout.direction_dims > 0 {
                    row[layout.position_dims..].copy_from_slice(
                        &rays.directions.data()
                            [origin_offset..origin_offset + layout.position_dims],
                    );
                }
                widths.push(width);
            }
        }
        validate_finite(input.data(), "nerf_sample_coordinates")?;
        Ok((input, widths))
    }
}

fn validate_sample_layout(field: &NerfField, config: &NerfTrainingConfig) -> PureResult<usize> {
    let layout = field.sample_layout();
    if layout.direction_dims != 0 && layout.direction_dims != layout.position_dims {
        return Err(TensorError::InvalidDimensions {
            rows: layout.position_dims,
            cols: layout.direction_dims,
        });
    }
    let samples = checked_allocation::<f64>(config.batch_size, config.samples_per_ray)?;
    checked_elements(samples, layout.try_total_dims()?)?;
    checked_elements(samples, 4)?;
    checked_elements(config.batch_size, 3)?;
    Ok(samples)
}
