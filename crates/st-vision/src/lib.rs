// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
//
// =============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// =============================================================================

//! Z-space native vision utilities for SpiralTorch.
//!
//! The [`ZSpaceVolume`] acts as a volumetric canvas capable of storing
//! resonant feature slices along the Z axis. Coupled with a
//! [`VisionProjector`], the volume can be collapsed back into 2D feature maps
//! while respecting Z-space curvature, resonance energy and the live telemetry
//! streamed through [`AtlasFrame`] snapshots.

use st_core::telemetry::atlas::AtlasFrame;
use st_core::telemetry::chrono::ChronoSummary;
use st_tensor::{DifferentialResonance, PureResult, Tensor, TensorError};

/// Volumetric container that holds planar tensors along the Z axis.
#[derive(Clone, Debug, PartialEq)]
pub struct ZSpaceVolume {
    depth: usize,
    height: usize,
    width: usize,
    voxels: Vec<f32>,
}

impl ZSpaceVolume {
    /// Creates a Z-space volume filled with zeros.
    pub fn zeros(depth: usize, height: usize, width: usize) -> PureResult<Self> {
        if depth == 0 || height == 0 || width == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: depth,
                cols: height.saturating_mul(width),
            });
        }
        Ok(Self {
            depth,
            height,
            width,
            voxels: vec![0.0; depth * height * width],
        })
    }

    /// Builds a Z-space volume from planar tensor slices.
    pub fn from_slices(slices: &[Tensor]) -> PureResult<Self> {
        if slices.is_empty() {
            return Err(TensorError::EmptyInput("z_space_volume_slices"));
        }
        let (height, width) = slices[0].shape();
        let mut voxels = Vec::with_capacity(slices.len() * height * width);
        for slice in slices {
            let (rows, cols) = slice.shape();
            if rows != height || cols != width {
                return Err(TensorError::ShapeMismatch {
                    left: (height, width),
                    right: (rows, cols),
                });
            }
            voxels.extend_from_slice(slice.data());
        }
        Ok(Self {
            depth: slices.len(),
            height,
            width,
            voxels,
        })
    }

    /// Returns the depth (number of Z slices).
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the height of each slice.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the width of each slice.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Immutable access to the raw voxel buffer.
    pub fn voxels(&self) -> &[f32] {
        &self.voxels
    }

    /// Mutable access to the raw voxel buffer.
    pub fn voxels_mut(&mut self) -> &mut [f32] {
        &mut self.voxels
    }

    /// Extracts a slice at the requested depth index.
    pub fn slice(&self, index: usize) -> PureResult<Tensor> {
        if index >= self.depth {
            return Err(TensorError::InvalidValue {
                label: "z_slice_index",
            });
        }
        let slice_len = self.height * self.width;
        let start = index * slice_len;
        let end = start + slice_len;
        Tensor::from_vec(self.height, self.width, self.voxels[start..end].to_vec())
    }

    /// Collapses the volume into a 2D tensor using the provided depth weights.
    pub fn collapse_with_weights(&self, weights: &[f32]) -> PureResult<Tensor> {
        if weights.len() != self.depth {
            return Err(TensorError::DataLength {
                expected: self.depth,
                got: weights.len(),
            });
        }
        let slice_len = self.height * self.width;
        let mut canvas = vec![0.0; slice_len];
        for (z, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "z_space_weight",
                    value: weight,
                });
            }
            let start = z * slice_len;
            let end = start + slice_len;
            let slice = &self.voxels[start..end];
            for (idx, voxel) in slice.iter().enumerate() {
                canvas[idx] += voxel * weight;
            }
        }
        Tensor::from_vec(self.height, self.width, canvas)
    }

    /// Derives per-depth weights from a differential resonance.
    pub fn resonance_weights(&self, resonance: &DifferentialResonance) -> PureResult<Vec<f32>> {
        let energy = resonance.infinity_energy.data();
        let objective = resonance.recursive_objective.data();
        if energy.is_empty() || objective.is_empty() {
            return Err(TensorError::EmptyInput("differential_resonance"));
        }
        let homotopy = resonance.homotopy_flow.data();
        let mut weights = Vec::with_capacity(self.depth);
        let energy_len = energy.len();
        let objective_len = objective.len();
        let homotopy_len = homotopy.len().max(1);
        for idx in 0..self.depth {
            let e = energy[idx % energy_len].abs() + 1e-6;
            let o = 1.0 + objective[idx % objective_len].tanh();
            let h = if homotopy.is_empty() {
                1.0
            } else {
                1.0 + homotopy[idx % homotopy_len].tanh()
            };
            let value = (e * o * h).max(0.0);
            weights.push(value);
        }
        Self::normalise_weights(&mut weights);
        Ok(weights)
    }

    fn normalise_weights(weights: &mut [f32]) {
        let mut total = 0.0f32;
        for weight in weights.iter() {
            if weight.is_finite() {
                total += *weight;
            }
        }
        if !total.is_finite() || total <= f32::EPSILON {
            let uniform = 1.0 / weights.len().max(1) as f32;
            for weight in weights.iter_mut() {
                *weight = uniform;
            }
        } else {
            for weight in weights.iter_mut() {
                *weight /= total;
            }
        }
    }

    /// Collapses the volume according to resonance-derived weights.
    pub fn project_resonance(&self, resonance: &DifferentialResonance) -> PureResult<Tensor> {
        let weights = self.resonance_weights(resonance)?;
        self.collapse_with_weights(&weights)
    }
}

/// Adaptive projector that fuses resonance telemetry with chrono summaries.
#[derive(Clone, Debug)]
pub struct VisionProjector {
    focus: f32,
    spread: f32,
    energy_bias: f32,
}

impl VisionProjector {
    /// Creates a new projector with explicit Z focus, spread, and energy bias.
    pub fn new(focus: f32, spread: f32, energy_bias: f32) -> Self {
        let focus = focus.clamp(0.0, 1.0);
        let spread = if spread.is_finite() && spread > 1e-3 {
            spread
        } else {
            0.35
        };
        let energy_bias = if energy_bias.is_finite() {
            energy_bias
        } else {
            0.0
        };
        Self {
            focus,
            spread,
            energy_bias,
        }
    }

    /// Builds a projector from a chrono summary.
    pub fn from_summary(summary: &ChronoSummary) -> Self {
        let span = (summary.max_energy - summary.min_energy).abs().max(1e-3);
        let focus = ((summary.mean_energy - summary.min_energy) / span).clamp(0.0, 1.0);
        let mut spread = (summary.energy_std / span).clamp(0.05, 1.0);
        if summary.mean_abs_drift.is_finite() {
            spread /= 1.0 + summary.mean_abs_drift.abs();
        }
        let energy_bias = summary.mean_decay;
        Self::new(focus, spread, energy_bias)
    }

    /// Updates the projector using live atlas telemetry.
    pub fn calibrate_from_atlas(&mut self, frame: &AtlasFrame) {
        if let Some(signal) = frame.z_signal {
            if signal.is_finite() {
                self.focus = signal.clamp(0.0, 1.0);
            }
        }
        if let Some(pressure) = frame.suggested_pressure {
            if pressure.is_finite() {
                let candidate = (pressure / 32.0).abs().clamp(0.05, 2.0);
                self.spread = candidate.max(0.05);
            }
        }
        if let Some(total) = frame.collapse_total {
            if total.is_finite() {
                self.energy_bias = total.tanh();
            }
        }
    }

    /// Returns the current focus along the Z axis.
    pub fn focus(&self) -> f32 {
        self.focus
    }

    /// Returns the current spread of the focus window.
    pub fn spread(&self) -> f32 {
        self.spread
    }

    /// Returns the current energy bias modifier.
    pub fn energy_bias(&self) -> f32 {
        self.energy_bias
    }

    fn compute_depth_weights(
        &self,
        volume: &ZSpaceVolume,
        resonance: &DifferentialResonance,
    ) -> PureResult<Vec<f32>> {
        let mut weights = volume.resonance_weights(resonance)?;
        if weights.is_empty() {
            return Ok(weights);
        }
        let spread = self.spread.max(1e-3);
        let energy_mod = 1.0 + self.energy_bias.tanh();
        if volume.depth() > 1 {
            let denom = (volume.depth() - 1) as f32;
            for (idx, weight) in weights.iter_mut().enumerate() {
                let position = if denom > 0.0 { idx as f32 / denom } else { 0.0 };
                let delta = position - self.focus;
                let gaussian = (-0.5 * (delta / spread).powi(2)).exp();
                *weight *= gaussian.max(1e-6) * energy_mod.max(0.1);
            }
        } else {
            for weight in weights.iter_mut() {
                *weight *= energy_mod.max(0.1);
            }
        }
        ZSpaceVolume::normalise_weights(&mut weights);
        Ok(weights)
    }

    /// Produces the final depth weights as a tensor.
    pub fn depth_weights(
        &self,
        volume: &ZSpaceVolume,
        resonance: &DifferentialResonance,
    ) -> PureResult<Tensor> {
        let weights = self.compute_depth_weights(volume, resonance)?;
        Tensor::from_vec(1, weights.len(), weights)
    }

    /// Projects the volume into a single 2D tensor using calibrated weights.
    pub fn project(
        &self,
        volume: &ZSpaceVolume,
        resonance: &DifferentialResonance,
    ) -> PureResult<Tensor> {
        let weights = self.compute_depth_weights(volume, resonance)?;
        volume.collapse_with_weights(&weights)
    }
}

impl Default for VisionProjector {
    fn default() -> Self {
        Self::new(0.5, 0.35, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_from(values: &[f32], rows: usize, cols: usize) -> Tensor {
        Tensor::from_vec(rows, cols, values.to_vec()).unwrap()
    }

    fn toy_resonance(depth: usize) -> DifferentialResonance {
        let mut energy = Vec::with_capacity(depth);
        let mut objective = Vec::with_capacity(depth);
        let mut homotopy = Vec::with_capacity(depth);
        for idx in 0..depth {
            energy.push(1.0 + idx as f32);
            objective.push((idx as f32 * 0.1) - 0.2);
            homotopy.push((idx as f32 * 0.05).sin());
        }
        DifferentialResonance {
            homotopy_flow: tensor_from(&homotopy, 1, depth),
            functor_linearisation: tensor_from(&objective, 1, depth),
            recursive_objective: tensor_from(&objective, 1, depth),
            infinity_projection: tensor_from(&objective, 1, depth),
            infinity_energy: tensor_from(&energy, 1, depth),
        }
    }

    #[test]
    fn volume_from_slices_respects_shapes() {
        let slice_a = tensor_from(&[0.0, 1.0, 2.0, 3.0], 2, 2);
        let slice_b = tensor_from(&[4.0, 5.0, 6.0, 7.0], 2, 2);
        let volume = ZSpaceVolume::from_slices(&[slice_a.clone(), slice_b.clone()]).unwrap();
        assert_eq!(volume.depth(), 2);
        assert_eq!(volume.height(), 2);
        assert_eq!(volume.width(), 2);
        let slice = volume.slice(1).unwrap();
        assert_eq!(slice.data(), slice_b.data());
    }

    #[test]
    fn resonance_projection_matches_manual_weighting() {
        let slices = vec![
            tensor_from(&[1.0, 2.0, 3.0, 4.0], 2, 2),
            tensor_from(&[2.0, 3.0, 4.0, 5.0], 2, 2),
            tensor_from(&[3.0, 4.0, 5.0, 6.0], 2, 2),
        ];
        let volume = ZSpaceVolume::from_slices(&slices).unwrap();
        let resonance = toy_resonance(3);
        let weights = volume.resonance_weights(&resonance).unwrap();
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        let projected = volume.project_resonance(&resonance).unwrap();
        let mut expected = vec![0.0f32; 4];
        for (z, weight) in weights.iter().enumerate() {
            for (idx, value) in slices[z].data().iter().enumerate() {
                expected[idx] += value * weight;
            }
        }
        assert_eq!(projected.shape(), (2, 2));
        for (a, b) in projected.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn projector_respects_atlas_calibration() {
        let slices = vec![
            tensor_from(&[1.0, 0.0, 0.0, 1.0], 2, 2),
            tensor_from(&[0.0, 1.0, 1.0, 0.0], 2, 2),
        ];
        let volume = ZSpaceVolume::from_slices(&slices).unwrap();
        let resonance = toy_resonance(2);
        let summary = ChronoSummary {
            frames: 4,
            duration: 1.0,
            latest_timestamp: 1.0,
            mean_drift: 0.1,
            mean_abs_drift: 0.2,
            drift_std: 0.05,
            mean_energy: 2.5,
            energy_std: 0.5,
            mean_decay: -0.3,
            min_energy: 1.0,
            max_energy: 4.0,
        };
        let mut projector = VisionProjector::from_summary(&summary);
        let weights_before = projector.depth_weights(&volume, &resonance).unwrap();
        let mut atlas = AtlasFrame::new(0.5);
        atlas.z_signal = Some(1.0);
        atlas.collapse_total = Some(2.0);
        atlas.suggested_pressure = Some(64.0);
        projector.calibrate_from_atlas(&atlas);
        let weights_after = projector.depth_weights(&volume, &resonance).unwrap();
        assert!((weights_after.data()[0] - weights_before.data()[0]).abs() > 1e-3);
        let projection = projector.project(&volume, &resonance).unwrap();
        assert_eq!(projection.shape(), (2, 2));
    }
}
