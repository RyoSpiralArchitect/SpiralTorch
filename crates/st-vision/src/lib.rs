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
//! In addition to the SpiralTorch-specific data structures, this crate keeps a
//! concise index of **TorchVision**'s representative functionality so that
//! downstream consumers can quickly align their expectations when porting
//! workflows.
//!
//! ### TorchVision standard capabilities
//! - **Datasets** &mdash; canonical image classification, detection, segmentation,
//!   optical flow, stereo matching, paired-image, captioning, and video
//!   collections are exposed through ready-to-use `torchvision.datasets`
//!   classes.
//! - **Models** &mdash; reference architectures for classification (from AlexNet to
//!   modern ViT variants), quantized classifiers, semantic segmentation,
//!   detection/instance segmentation/keypoint estimation, video classification,
//!   and optical flow provide drop-in baselines.
//! - **Transforms v2** &mdash; unified augmentation pipelines (`torchvision.transforms.v2`)
//!   handle images, videos, bounding boxes, masks, and keypoints while
//!   remaining compatible with the v1 API.
//! - **TVTensors** &mdash; tensor subclasses (Image, Video, BoundingBoxes, etc.)
//!   enable automatic dispatch and metadata propagation inside the v2 pipeline.
//! - **Utilities** &mdash; rendering helpers such as `draw_bounding_boxes`,
//!   `draw_segmentation_masks`, `make_grid`, and `save_image` simplify rapid
//!   inspection.
//! - **Custom ops** &mdash; `torchvision.ops` implements NMS, RoI operators, box
//!   algebra, detection-friendly losses, and common vision blocks that remain
//!   TorchScript compatible.
//! - **IO** &mdash; accelerated codecs (JPEG/PNG/WEBP/GIF/AVIF/HEIC) with tensor
//!   interop plus video read/write utilities support efficient pipelines.
//! - **Feature extraction** &mdash; helpers like `create_feature_extractor` expose
//!   intermediate activations for transfer learning, visualization, and FPN
//!   style integration.
//!
//! The [`ZSpaceVolume`] acts as a volumetric canvas capable of storing
//! resonant feature slices along the Z axis. Coupled with a
//! [`VisionProjector`], the volume can be collapsed back into 2D feature maps
//! while respecting Z-space curvature, resonance energy and the live telemetry
//! streamed through [`AtlasFrame`] snapshots.

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::min;
use std::f32::consts::PI;
use std::sync::Arc;

use st_core::telemetry::atlas::AtlasFrame;
use st_core::telemetry::chrono::ChronoSummary;
use st_nn::layers::spiral_rnn::SpiralRnn;
use st_nn::module::Module;
use st_tensor::{DifferentialResonance, PureResult, Tensor, TensorError};

const RESONANCE_FEATURES_PER_SLICE: usize = 10;

/// Volumetric container that holds planar tensors along the Z axis.
#[derive(Clone, Debug, PartialEq)]
pub struct ZSpaceVolume {
    depth: usize,
    height: usize,
    width: usize,
    voxels: Vec<f32>,
}

/// Statistical summary describing each slice inside a [`ZSpaceVolume`].
#[derive(Clone, Debug, PartialEq)]
pub struct ZSliceProfile {
    means: Vec<f32>,
    stds: Vec<f32>,
    energies: Vec<f32>,
}

impl ZSliceProfile {
    /// Creates a new profile from explicit per-slice statistics.
    pub fn new(means: Vec<f32>, stds: Vec<f32>, energies: Vec<f32>) -> PureResult<Self> {
        if means.is_empty() {
            return Err(TensorError::EmptyInput("z_slice_profile"));
        }
        let depth = means.len();
        if stds.len() != depth {
            return Err(TensorError::DataLength {
                expected: depth,
                got: stds.len(),
            });
        }
        if energies.len() != depth {
            return Err(TensorError::DataLength {
                expected: depth,
                got: energies.len(),
            });
        }
        Ok(Self {
            means,
            stds,
            energies,
        })
    }

    /// Number of slices contained in the profile.
    pub fn depth(&self) -> usize {
        self.means.len()
    }

    /// Returns an immutable view of the slice means.
    pub fn means(&self) -> &[f32] {
        &self.means
    }

    /// Returns an immutable view of the slice standard deviations.
    pub fn stds(&self) -> &[f32] {
        &self.stds
    }

    /// Returns an immutable view of the slice energies.
    pub fn energies(&self) -> &[f32] {
        &self.energies
    }

    /// Fetches the mean intensity for the given slice index.
    pub fn mean(&self, index: usize) -> f32 {
        self.means[index]
    }

    /// Fetches the standard deviation for the given slice index.
    pub fn std(&self, index: usize) -> f32 {
        self.stds[index]
    }

    /// Fetches the mean squared energy for the given slice index.
    pub fn energy(&self, index: usize) -> f32 {
        self.energies[index]
    }
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

    /// Builds a Z-space volume directly from a raw voxel buffer.
    pub fn from_voxels(
        depth: usize,
        height: usize,
        width: usize,
        voxels: Vec<f32>,
    ) -> PureResult<Self> {
        if depth == 0 || height == 0 || width == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: depth.max(1),
                cols: height.saturating_mul(width).max(1),
            });
        }
        let expected = depth
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .ok_or(TensorError::InvalidDimensions {
                rows: depth,
                cols: height.saturating_mul(width),
            })?;
        if voxels.len() != expected {
            return Err(TensorError::DataLength {
                expected,
                got: voxels.len(),
            });
        }
        Ok(Self {
            depth,
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

    /// Interpolates between existing Z slices, returning a densified volume.
    pub fn interpolate(&self, method: InterpolationMethod) -> PureResult<Self> {
        if self.depth <= 1 || self.height == 0 || self.width == 0 {
            return Ok(self.clone());
        }
        let slice_len = self.height * self.width;
        let mut voxels = Vec::with_capacity(slice_len * (self.depth * 2 - 1));
        for z in 0..self.depth - 1 {
            let start = z * slice_len;
            let current = &self.voxels[start..start + slice_len];
            let next_start = (z + 1) * slice_len;
            let next = &self.voxels[next_start..next_start + slice_len];
            voxels.extend_from_slice(current);
            let interpolated = match method {
                InterpolationMethod::Nearest => current.to_vec(),
                InterpolationMethod::Linear => current
                    .iter()
                    .zip(next.iter())
                    .map(|(a, b)| 0.5 * (a + b))
                    .collect(),
                InterpolationMethod::Cubic => {
                    let prev = if z == 0 {
                        current
                    } else {
                        let prev_start = (z - 1) * slice_len;
                        &self.voxels[prev_start..prev_start + slice_len]
                    };
                    let ahead = if z + 2 < self.depth {
                        let ahead_start = (z + 2) * slice_len;
                        &self.voxels[ahead_start..ahead_start + slice_len]
                    } else {
                        next
                    };
                    let mut buffer = Vec::with_capacity(slice_len);
                    for idx in 0..slice_len {
                        let value =
                            Self::catmull_rom(prev[idx], current[idx], next[idx], ahead[idx], 0.5);
                        buffer.push(value);
                    }
                    buffer
                }
            };
            voxels.extend_from_slice(&interpolated);
        }
        let last_start = (self.depth - 1) * slice_len;
        voxels.extend_from_slice(&self.voxels[last_start..last_start + slice_len]);
        Self::from_voxels(self.depth * 2 - 1, self.height, self.width, voxels)
    }

    /// Upscales each slice using bilinear interpolation and returns a refined volume.
    pub fn upscale(&self, factor: usize) -> PureResult<Self> {
        if factor == 0 {
            return Err(TensorError::InvalidValue {
                label: "z_upscale_factor",
            });
        }
        if factor == 1 || self.height == 0 || self.width == 0 {
            return Ok(self.clone());
        }
        let new_height = self
            .height
            .checked_mul(factor)
            .ok_or(TensorError::InvalidDimensions {
                rows: self.height,
                cols: factor,
            })?;
        let new_width = self
            .width
            .checked_mul(factor)
            .ok_or(TensorError::InvalidDimensions {
                rows: self.width,
                cols: factor,
            })?;
        let slice_len = self.height * self.width;
        let mut voxels = Vec::with_capacity(self.depth * new_height * new_width);
        for z in 0..self.depth {
            let start = z * slice_len;
            let slice = &self.voxels[start..start + slice_len];
            let upscaled =
                Self::bilinear_resample(slice, self.height, self.width, new_height, new_width);
            voxels.extend_from_slice(&upscaled);
        }
        Self::from_voxels(self.depth, new_height, new_width, voxels)
    }

    fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
        let a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
        let a1 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
        let a2 = -0.5 * p0 + 0.5 * p2;
        let a3 = p1;
        ((a0 * t + a1) * t + a2) * t + a3
    }

    fn bilinear_resample(
        slice: &[f32],
        height: usize,
        width: usize,
        new_height: usize,
        new_width: usize,
    ) -> Vec<f32> {
        if height == 0 || width == 0 || new_height == 0 || new_width == 0 {
            return Vec::new();
        }
        let mut output = vec![0.0; new_height * new_width];
        let h_scale = if new_height > 1 {
            (height.saturating_sub(1)) as f32 / (new_height.saturating_sub(1)) as f32
        } else {
            0.0
        };
        let w_scale = if new_width > 1 {
            (width.saturating_sub(1)) as f32 / (new_width.saturating_sub(1)) as f32
        } else {
            0.0
        };
        for y in 0..new_height {
            let src_y = h_scale * y as f32;
            let y0 = src_y.floor() as usize;
            let y1 = min(y0 + 1, height - 1);
            let ty = src_y - y0 as f32;
            for x in 0..new_width {
                let src_x = w_scale * x as f32;
                let x0 = src_x.floor() as usize;
                let x1 = min(x0 + 1, width - 1);
                let tx = src_x - x0 as f32;
                let top_left = slice[y0 * width + x0];
                let top_right = slice[y0 * width + x1];
                let bottom_left = slice[y1 * width + x0];
                let bottom_right = slice[y1 * width + x1];
                let top = top_left + (top_right - top_left) * tx;
                let bottom = bottom_left + (bottom_right - bottom_left) * tx;
                output[y * new_width + x] = top + (bottom - top) * ty;
            }
        }
        output
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

    /// Computes a spectral energy response for each depth slice using the provided window.
    pub fn spectral_response(&self, window: &SpectralWindow) -> Vec<f32> {
        let slice_len = self.height.saturating_mul(self.width);
        if slice_len == 0 || self.depth == 0 {
            return Vec::new();
        }
        let mut response = Vec::with_capacity(self.depth);
        let window_weights = window.weights(self.depth);
        for (z, coeff) in window_weights.iter().enumerate() {
            let start = z * slice_len;
            let end = start + slice_len;
            let slice = &self.voxels[start..end];
            let energy = if slice_len > 0 {
                slice.iter().map(|v| v.abs()).sum::<f32>() / slice_len as f32
            } else {
                0.0
            };
            response.push(energy * coeff);
        }
        response
    }

    /// Performs an exponential moving average with another volume in-place.
    pub fn accumulate(&mut self, next: &ZSpaceVolume, alpha: f32) -> PureResult<()> {
        if self.depth != next.depth || self.height != next.height || self.width != next.width {
            return Err(TensorError::ShapeMismatch {
                left: (self.depth, self.height * self.width),
                right: (next.depth, next.height * next.width),
            });
        }
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(TensorError::InvalidValue {
                label: "temporal_alpha",
            });
        }
        let retain = 1.0 - alpha;
        for (current, incoming) in self.voxels.iter_mut().zip(next.voxels.iter()) {
            *current = (*current * retain) + (incoming * alpha);
        }
        Ok(())
    }

    /// Returns a blended copy that incorporates the next volume using EMA weighting.
    pub fn accumulated(&self, next: &ZSpaceVolume, alpha: f32) -> PureResult<Self> {
        let mut blended = Self {
            depth: self.depth,
            height: self.height,
            width: self.width,
            voxels: self.voxels.clone(),
        };
        blended.accumulate(next, alpha)?;
        Ok(blended)
    }
}

/// Interpolation methods available for Z-space resampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Keeps neighbouring voxels unchanged when interpolating.
    Nearest,
    /// Uses linear interpolation between neighbouring voxels.
    Linear,
    /// Applies a Catmull-Rom cubic interpolation across neighbours.
    Cubic,
}

/// Spectral window functions used to modulate depth resonance weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralWindow {
    kind: SpectralWindowKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SpectralWindowKind {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    Gaussian { sigma: f32 },
}

impl SpectralWindow {
    /// Creates a rectangular window (no modulation).
    pub fn rectangular() -> Self {
        Self {
            kind: SpectralWindowKind::Rectangular,
        }
    }

    /// Creates a Hann window.
    pub fn hann() -> Self {
        Self {
            kind: SpectralWindowKind::Hann,
        }
    }

    /// Creates a Hamming window.
    pub fn hamming() -> Self {
        Self {
            kind: SpectralWindowKind::Hamming,
        }
    }

    /// Creates a Blackman window.
    pub fn blackman() -> Self {
        Self {
            kind: SpectralWindowKind::Blackman,
        }
    }

    /// Creates a Gaussian window with the provided sigma parameter.
    pub fn gaussian(sigma: f32) -> Self {
        let sigma = if sigma.is_finite() && sigma > 1e-3 {
            sigma
        } else {
            0.4
        };
        Self {
            kind: SpectralWindowKind::Gaussian { sigma },
        }
    }

    /// Generates normalised weights for the configured window.
    pub fn weights(&self, depth: usize) -> Vec<f32> {
        if depth == 0 {
            return Vec::new();
        }
        if depth == 1 {
            return vec![1.0];
        }
        let mut weights = Vec::with_capacity(depth);
        let n_minus_1 = (depth - 1) as f32;
        match self.kind {
            SpectralWindowKind::Rectangular => {
                weights.resize(depth, 1.0);
            }
            SpectralWindowKind::Hann => {
                for n in 0..depth {
                    let coeff = 0.5 * (1.0 - (2.0 * PI * n as f32 / n_minus_1).cos());
                    weights.push(coeff.max(0.0));
                }
            }
            SpectralWindowKind::Hamming => {
                for n in 0..depth {
                    let coeff = 0.54 - 0.46 * (2.0 * PI * n as f32 / n_minus_1).cos();
                    weights.push(coeff.max(0.0));
                }
            }
            SpectralWindowKind::Blackman => {
                for n in 0..depth {
                    let ratio = 2.0 * PI * n as f32 / n_minus_1;
                    let coeff = 0.42 - 0.5 * ratio.cos() + 0.08 * (2.0 * ratio).cos();
                    weights.push(coeff.max(0.0));
                }
            }
            SpectralWindowKind::Gaussian { sigma } => {
                let centre = n_minus_1 / 2.0;
                let denom = 2.0 * sigma.powi(2) * (centre + 1.0).powi(2);
                for n in 0..depth {
                    let delta = n as f32 - centre;
                    let coeff = (-delta.powi(2) / denom.max(1e-6)).exp();
                    weights.push(coeff.max(0.0));
                }
            }
        }
        ZSpaceVolume::normalise_weights(&mut weights);
        weights
    }
}

/// Maintains a temporal exponential moving average of depth attention weights.
#[derive(Clone, Debug)]
pub struct TemporalResonanceBuffer {
    decay: f32,
    history: Option<Vec<f32>>,
    frames: usize,
}

impl TemporalResonanceBuffer {
    /// Creates a new temporal buffer using the provided decay coefficient.
    pub fn new(decay: f32) -> Self {
        let decay = if decay.is_finite() {
            decay.clamp(0.0, 1.0)
        } else {
            0.5
        };
        Self {
            decay,
            history: None,
            frames: 0,
        }
    }

    /// Returns the exponential decay factor applied to new weights.
    pub fn decay(&self) -> f32 {
        self.decay
    }

    /// Returns how many frames have been fused into the buffer.
    pub fn frames_accumulated(&self) -> usize {
        self.frames
    }

    /// Returns the current temporal history if it exists.
    pub fn history(&self) -> Option<&[f32]> {
        self.history.as_deref()
    }

    /// Clears the stored history and resets the buffer.
    pub fn clear(&mut self) {
        self.history = None;
        self.frames = 0;
    }

    /// Applies the temporal smoothing to a new set of weights and returns the fused profile.
    pub fn apply(&mut self, weights: &[f32]) -> PureResult<Vec<f32>> {
        if let Some(value) = weights.iter().find(|value| !value.is_finite()) {
            return Err(TensorError::NonFiniteValue {
                label: "temporal_resonance_weight",
                value: *value,
            });
        }
        if weights.is_empty() {
            self.clear();
            return Ok(Vec::new());
        }
        match self.history {
            Some(ref mut history) if history.len() == weights.len() => {
                let alpha = self.decay;
                let retain = 1.0 - alpha;
                for (stored, &incoming) in history.iter_mut().zip(weights.iter()) {
                    *stored = (*stored * retain) + (incoming * alpha);
                }
                ZSpaceVolume::normalise_weights(history);
                self.frames = self.frames.saturating_add(1);
                Ok(history.clone())
            }
            _ => {
                let mut history = weights.to_vec();
                ZSpaceVolume::normalise_weights(&mut history);
                self.history = Some(history.clone());
                self.frames = 1;
                Ok(history)
            }
        }
    }
}

/// Diffuses voxels along the spatial and depth axes to fill sparse slices.
#[derive(Clone, Debug)]
pub struct ZDiffuser {
    iterations: usize,
    rate: f32,
}

impl ZDiffuser {
    /// Creates a new diffuser with the desired iteration count and diffusion rate.
    pub fn new(iterations: usize, rate: f32) -> Self {
        let rate = if rate.is_finite() {
            rate.clamp(0.0, 1.0)
        } else {
            0.25
        };
        Self { iterations, rate }
    }

    /// Number of diffusion passes that will be applied.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Blend rate used when combining neighbours.
    pub fn rate(&self) -> f32 {
        self.rate
    }

    /// Applies diffusion and returns a smoothed Z-space volume.
    pub fn diffuse(&self, volume: &ZSpaceVolume) -> PureResult<ZSpaceVolume> {
        if volume.depth() == 0 || volume.height() == 0 || volume.width() == 0 {
            return Ok(volume.clone());
        }
        let mut current = volume.clone();
        for _ in 0..self.iterations {
            current = self.diffuse_once(&current)?;
        }
        Ok(current)
    }

    fn diffuse_once(&self, volume: &ZSpaceVolume) -> PureResult<ZSpaceVolume> {
        let depth = volume.depth();
        let height = volume.height();
        let width = volume.width();
        let slice_len = height * width;
        let voxels = volume.voxels().to_vec();
        let mut next = voxels.clone();
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let idx = z * slice_len + y * width + x;
                    let centre = voxels[idx];
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    if x > 0 {
                        sum += voxels[idx - 1];
                        count += 1.0;
                    }
                    if x + 1 < width {
                        sum += voxels[idx + 1];
                        count += 1.0;
                    }
                    if y > 0 {
                        sum += voxels[idx - width];
                        count += 1.0;
                    }
                    if y + 1 < height {
                        sum += voxels[idx + width];
                        count += 1.0;
                    }
                    if z > 0 {
                        sum += voxels[idx - slice_len];
                        count += 1.0;
                    }
                    if z + 1 < depth {
                        sum += voxels[idx + slice_len];
                        count += 1.0;
                    }
                    if count == 0.0 {
                        continue;
                    }
                    let average = sum / count;
                    next[idx] = centre * (1.0 - self.rate) + average * self.rate;
                }
            }
        }
        ZSpaceVolume::from_voxels(depth, height, width, next)
    }
}

/// Synthesises differential resonances using a [`SpiralRnn`] conditioned on Z-space telemetry.
#[derive(Debug)]
pub struct ResonanceGenerator {
    rnn: SpiralRnn,
    features_per_slice: usize,
    steps: usize,
    hidden_dim: usize,
}

impl ResonanceGenerator {
    /// Creates a generator that uses the default feature set per slice.
    pub fn new(name: impl Into<String>, hidden_dim: usize, steps: usize) -> PureResult<Self> {
        Self::with_features(name, RESONANCE_FEATURES_PER_SLICE, hidden_dim, steps)
    }

    /// Creates a generator with an explicit feature dimensionality per slice.
    pub fn with_features(
        name: impl Into<String>,
        features_per_slice: usize,
        hidden_dim: usize,
        steps: usize,
    ) -> PureResult<Self> {
        if features_per_slice == 0 || hidden_dim == 0 || steps == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: features_per_slice.max(1),
                cols: hidden_dim.max(1),
            });
        }
        let rnn = SpiralRnn::new(name, features_per_slice, hidden_dim, steps)?;
        Ok(Self {
            rnn,
            features_per_slice,
            steps,
            hidden_dim,
        })
    }

    /// Number of conditioning features encoded for each Z slice.
    pub fn features_per_slice(&self) -> usize {
        self.features_per_slice
    }

    /// Number of temporal steps expected by the underlying [`SpiralRnn`].
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Hidden dimensionality of the [`SpiralRnn`].
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Immutable access to the internal [`SpiralRnn`].
    pub fn rnn(&self) -> &SpiralRnn {
        &self.rnn
    }

    /// Mutable access to the internal [`SpiralRnn`] for fine-tuning.
    pub fn rnn_mut(&mut self) -> &mut SpiralRnn {
        &mut self.rnn
    }

    /// Generates a [`DifferentialResonance`] conditioned on the provided telemetry.
    pub fn generate(
        &mut self,
        volume: &ZSpaceVolume,
        projector: &VisionProjector,
        chrono: Option<&ChronoSummary>,
        atlas: Option<&AtlasFrame>,
        previous: Option<&DifferentialResonance>,
    ) -> PureResult<DifferentialResonance> {
        if volume.depth() > self.steps {
            return Err(TensorError::InvalidDimensions {
                rows: volume.depth(),
                cols: self.steps,
            });
        }
        let profile = volume.slice_profile()?;
        let encoded = self.encode(volume, projector, chrono, atlas, previous, &profile)?;
        let latent = self.rnn.forward(&encoded)?;
        self.decode(
            volume, projector, chrono, atlas, previous, &profile, &latent,
        )
    }

    fn encode(
        &self,
        volume: &ZSpaceVolume,
        projector: &VisionProjector,
        chrono: Option<&ChronoSummary>,
        atlas: Option<&AtlasFrame>,
        previous: Option<&DifferentialResonance>,
        profile: &ZSliceProfile,
    ) -> PureResult<Tensor> {
        let depth = profile.depth();
        let mut buffer = vec![0.0f32; self.steps * self.features_per_slice];
        let atlas_signal = atlas
            .and_then(|frame| frame.z_signal)
            .unwrap_or_else(|| projector.focus());
        let atlas_pressure = atlas
            .and_then(|frame| frame.suggested_pressure)
            .unwrap_or_else(|| projector.spread());
        let atlas_total = atlas
            .and_then(|frame| frame.collapse_total)
            .unwrap_or_else(|| projector.energy_bias());
        let chrono_mean = chrono
            .map(|summary| summary.mean_energy)
            .unwrap_or(atlas_total);
        let chrono_std = chrono.map(|summary| summary.energy_std).unwrap_or(0.0);
        let global_energy = volume.total_energy();
        let aspect = if volume.width() > 0 {
            volume.height() as f32 / volume.width() as f32
        } else {
            1.0
        };
        let prev_energy = previous.map(|res| res.infinity_energy.data().to_vec());
        let prev_objective = previous.map(|res| res.recursive_objective.data().to_vec());
        for idx in 0..depth {
            let offset = idx * self.features_per_slice;
            if self.features_per_slice > 0 {
                buffer[offset] = profile.mean(idx);
            }
            if self.features_per_slice > 1 {
                buffer[offset + 1] = profile.std(idx);
            }
            if self.features_per_slice > 2 {
                buffer[offset + 2] = profile.energy(idx);
            }
            if self.features_per_slice > 3 {
                buffer[offset + 3] = projector.focus();
            }
            if self.features_per_slice > 4 {
                buffer[offset + 4] = projector.spread();
            }
            if self.features_per_slice > 5 {
                buffer[offset + 5] = projector.energy_bias();
            }
            if self.features_per_slice > 6 {
                buffer[offset + 6] = atlas_signal;
            }
            if self.features_per_slice > 7 {
                buffer[offset + 7] = atlas_pressure;
            }
            if self.features_per_slice > 8 {
                buffer[offset + 8] = prev_energy
                    .as_ref()
                    .map(|values| values[idx % values.len()])
                    .unwrap_or(chrono_mean);
            }
            if self.features_per_slice > 9 {
                buffer[offset + 9] = prev_objective
                    .as_ref()
                    .map(|values| values[idx % values.len()])
                    .unwrap_or(chrono_std);
            }
            if self.features_per_slice > 10 {
                buffer[offset + 10] = global_energy;
            }
            if self.features_per_slice > 11 {
                buffer[offset + 11] = aspect;
            }
        }
        Tensor::from_vec(1, buffer.len(), buffer)
    }

    fn decode(
        &self,
        volume: &ZSpaceVolume,
        projector: &VisionProjector,
        chrono: Option<&ChronoSummary>,
        atlas: Option<&AtlasFrame>,
        previous: Option<&DifferentialResonance>,
        profile: &ZSliceProfile,
        latent: &Tensor,
    ) -> PureResult<DifferentialResonance> {
        let depth = volume.depth();
        let hidden = latent.data();
        if hidden.is_empty() {
            return Err(TensorError::EmptyInput("spiral_resonance_latent"));
        }
        let chrono_drift = chrono.map(|summary| summary.mean_drift).unwrap_or(0.0);
        let chrono_energy_std = chrono.map(|summary| summary.energy_std).unwrap_or(0.0);
        let chrono_drift_std = chrono.map(|summary| summary.drift_std).unwrap_or(0.0);
        let atlas_feedback = atlas
            .and_then(|frame| frame.collapse_total)
            .unwrap_or_else(|| projector.energy_bias());
        let prev_energy = previous.map(|res| res.infinity_energy.data().to_vec());
        let prev_objective = previous.map(|res| res.recursive_objective.data().to_vec());
        let prev_homotopy = previous.map(|res| res.homotopy_flow.data().to_vec());
        let prev_projection = previous.map(|res| res.infinity_projection.data().to_vec());

        let mut energies = Vec::with_capacity(depth);
        let mut objectives = Vec::with_capacity(depth);
        let mut homotopies = Vec::with_capacity(depth);
        let mut projections = Vec::with_capacity(depth);
        let mut functors = Vec::with_capacity(depth);

        for idx in 0..depth {
            let base = hidden[idx % hidden.len()];
            let mean = profile.mean(idx);
            let std = profile.std(idx);
            let slice_energy = profile.energy(idx);
            let prev_e = prev_energy
                .as_ref()
                .map(|values| values[idx % values.len()])
                .unwrap_or(slice_energy);
            let prev_o = prev_objective
                .as_ref()
                .map(|values| values[idx % values.len()])
                .unwrap_or(mean);
            let prev_h = prev_homotopy
                .as_ref()
                .map(|values| values[idx % values.len()])
                .unwrap_or(0.0);
            let prev_p = prev_projection
                .as_ref()
                .map(|values| values[idx % values.len()])
                .unwrap_or(slice_energy.tanh());

            let intensity = mean.abs() + std + slice_energy;
            let energy_value = (base + intensity + prev_e + atlas_feedback).abs() + 1e-3;
            energies.push(energy_value);

            let objective_value = (base + prev_o + projector.energy_bias() + chrono_drift).tanh();
            objectives.push(objective_value);

            let homotopy_value = (base + prev_h + projector.focus() - 0.5).tanh();
            homotopies.push(homotopy_value);

            let projection_value = (base + prev_p + slice_energy - chrono_energy_std).tanh();
            projections.push(projection_value);

            let functor_value =
                objective_value * 0.5 + slice_energy * 0.1 + chrono_drift_std * 0.05;
            functors.push(functor_value);
        }

        let homotopy_tensor = Tensor::from_vec(1, depth, homotopies)?;
        let functor_tensor = Tensor::from_vec(1, depth, functors)?;
        let objective_tensor = Tensor::from_vec(1, depth, objectives)?;
        let projection_tensor = Tensor::from_vec(1, depth, projections)?;
        let energy_tensor = Tensor::from_vec(1, depth, energies)?;

        Ok(DifferentialResonance {
            homotopy_flow: homotopy_tensor,
            functor_linearisation: functor_tensor,
            recursive_objective: objective_tensor,
            infinity_projection: projection_tensor,
            infinity_energy: energy_tensor,
        })
    }
}

/// Decodes latent feature vectors into volumetric Z-space tensors.
#[derive(Debug)]
pub struct ZDecoder {
    depth: usize,
    height: usize,
    width: usize,
    rng: StdRng,
}

impl ZDecoder {
    /// Creates a decoder with the expected target dimensions and RNG seed.
    pub fn new(depth: usize, height: usize, width: usize, seed: u64) -> PureResult<Self> {
        if depth == 0 || height == 0 || width == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: depth.max(1),
                cols: height.saturating_mul(width).max(1),
            });
        }
        Ok(Self {
            depth,
            height,
            width,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Decodes the latent tensor into a Z-space volume using stochastic projection.
    pub fn decode(&mut self, latent: &Tensor) -> PureResult<ZSpaceVolume> {
        let features = latent.data();
        if features.is_empty() {
            return Err(TensorError::EmptyInput("z_decoder_latent"));
        }
        let voxel_count = self
            .depth
            .checked_mul(self.height)
            .and_then(|value| value.checked_mul(self.width))
            .ok_or(TensorError::InvalidDimensions {
                rows: self.depth,
                cols: self.height.saturating_mul(self.width),
            })?;
        let mut voxels = Vec::with_capacity(voxel_count);
        for _ in 0..voxel_count {
            let mut value = 0.0f32;
            for &feature in features.iter() {
                let weight: f32 = self.rng.gen_range(-1.0..1.0);
                value += feature * weight;
            }
            let normalised = value / features.len() as f32;
            voxels.push(normalised.tanh());
        }
        ZSpaceVolume::from_voxels(self.depth, self.height, self.width, voxels)
    }

    /// Decodes the latent tensor and applies optional refinement stages.
    pub fn decode_with_refinement(
        &mut self,
        latent: &Tensor,
        diffuser: Option<&ZDiffuser>,
        interpolation: Option<InterpolationMethod>,
        upscale_factor: Option<usize>,
    ) -> PureResult<ZSpaceVolume> {
        let mut volume = self.decode(latent)?;
        if let Some(diffuser) = diffuser {
            volume = diffuser.diffuse(&volume)?;
        }
        if let Some(method) = interpolation {
            volume = volume.interpolate(method)?;
        }
        if let Some(factor) = upscale_factor {
            if factor > 1 {
                volume = volume.upscale(factor)?;
            }
        }
        Ok(volume)
    }

    /// Returns the configured depth.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the configured height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the configured width.
    pub fn width(&self) -> usize {
        self.width
    }
}

/// Streams Z-space volumes while generating resonances and temporal projections.
pub struct VideoStreamProjector {
    projector: VisionProjector,
    generator: ResonanceGenerator,
    smoothing: TemporalResonanceBuffer,
    previous_resonance: Option<Arc<DifferentialResonance>>,
    diffuser: Option<ZDiffuser>,
    super_resolution: Option<(InterpolationMethod, usize)>,
}

impl VideoStreamProjector {
    /// Creates a new video stream projector with the desired smoothing decay.
    pub fn new(projector: VisionProjector, generator: ResonanceGenerator, decay: f32) -> Self {
        Self {
            projector,
            generator,
            smoothing: TemporalResonanceBuffer::new(decay),
            previous_resonance: None,
            diffuser: None,
            super_resolution: None,
        }
    }

    /// Configures a diffuser that is applied before projection.
    pub fn with_diffuser(mut self, diffuser: ZDiffuser) -> Self {
        self.diffuser = Some(diffuser);
        self
    }

    /// Enables Z-space super resolution using interpolation and spatial upscaling.
    pub fn with_super_resolution(mut self, method: InterpolationMethod, factor: usize) -> Self {
        self.super_resolution = Some((method, factor));
        self
    }

    /// Resets accumulated temporal state.
    pub fn reset(&mut self) {
        self.smoothing.clear();
        self.previous_resonance = None;
    }

    /// Accessor for the underlying projector.
    pub fn projector(&self) -> &VisionProjector {
        &self.projector
    }

    /// Mutable accessor for the underlying projector.
    pub fn projector_mut(&mut self) -> &mut VisionProjector {
        &mut self.projector
    }

    /// Returns the previously generated resonance when available.
    pub fn last_resonance(&self) -> Option<&Arc<DifferentialResonance>> {
        self.previous_resonance.as_ref()
    }

    /// Processes a single frame and returns the projection together with the resonance.
    pub fn step(
        &mut self,
        volume: &ZSpaceVolume,
        chrono: Option<&ChronoSummary>,
        atlas: Option<&AtlasFrame>,
    ) -> PureResult<(Tensor, Arc<DifferentialResonance>)> {
        if let Some(frame) = atlas {
            self.projector.calibrate_from_atlas(frame);
        }
        let mut working = volume.clone();
        if let Some(diffuser) = &self.diffuser {
            working = diffuser.diffuse(&working)?;
        }
        if let Some((method, factor)) = self.super_resolution {
            working = working.interpolate(method)?;
            if factor > 1 {
                working = working.upscale(factor)?;
            }
        }
        let resonance = self.generator.generate(
            &working,
            &self.projector,
            chrono,
            atlas,
            self.previous_resonance.as_deref(),
        )?;
        let resonance = Arc::new(resonance);
        let projection = self.projector.project_with_temporal(
            &working,
            resonance.as_ref(),
            &mut self.smoothing,
        )?;
        self.previous_resonance = Some(resonance.clone());
        Ok((projection, resonance))
    }

    /// Projects a full sequence of volumes using optional telemetry and atlas data.
    pub fn project_sequence(
        &mut self,
        volumes: &[ZSpaceVolume],
        chrono: &[Option<ChronoSummary>],
        atlas: &[Option<AtlasFrame>],
    ) -> PureResult<Vec<Tensor>> {
        if !chrono.is_empty() && chrono.len() != volumes.len() {
            return Err(TensorError::DataLength {
                expected: volumes.len(),
                got: chrono.len(),
            });
        }
        if !atlas.is_empty() && atlas.len() != volumes.len() {
            return Err(TensorError::DataLength {
                expected: volumes.len(),
                got: atlas.len(),
            });
        }
        let mut projections = Vec::with_capacity(volumes.len());
        for (idx, volume) in volumes.iter().enumerate() {
            let chrono_ref = chrono.get(idx).and_then(|summary| summary.as_ref());
            let atlas_ref = atlas.get(idx).and_then(|frame| frame.as_ref());
            let (projection, _) = self.step(volume, chrono_ref, atlas_ref)?;
            projections.push(projection);
        }
        Ok(projections)
    }
}

/// Metadata describing a registered camera/view that contributes to a Z-space volume.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewDescriptor {
    id: Arc<str>,
    origin: [f32; 3],
    forward: [f32; 3],
    baseline_weight: f32,
}

impl ViewDescriptor {
    /// Creates a new view descriptor with the provided identifier, origin, and forward vector.
    pub fn new(id: impl Into<String>, origin: [f32; 3], forward: [f32; 3]) -> Self {
        let id: Arc<str> = Arc::from(id.into());
        let mut descriptor = Self {
            id,
            origin,
            forward: normalise_direction(forward),
            baseline_weight: 1.0,
        };
        if !descriptor.forward.iter().any(|value| value.abs() > 0.0) {
            descriptor.forward = [0.0, 0.0, 1.0];
        }
        descriptor
    }

    /// Sets a baseline importance weight for the view during fusion.
    pub fn with_baseline_weight(mut self, weight: f32) -> Self {
        self.baseline_weight = weight.max(0.0);
        self
    }

    /// Returns the identifier of the view.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the origin of the camera in world coordinates.
    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }

    /// Updates the origin of the camera in world coordinates.
    pub fn set_origin(&mut self, origin: [f32; 3]) {
        self.origin = origin;
    }

    /// Returns the forward direction of the camera.
    pub fn forward(&self) -> [f32; 3] {
        self.forward
    }

    /// Updates the forward direction (normalised) of the camera.
    pub fn set_forward(&mut self, forward: [f32; 3]) {
        self.forward = normalise_direction(forward);
    }

    /// Returns the baseline fusion weight associated with this view.
    pub fn baseline_weight(&self) -> f32 {
        self.baseline_weight.max(0.0)
    }

    fn alignment(&self, focus: [f32; 3]) -> f32 {
        let focus = normalise_direction(focus);
        let dot =
            self.forward[0] * focus[0] + self.forward[1] * focus[1] + self.forward[2] * focus[2];
        dot.max(0.0)
    }
}

/// Helper that fuses multi-view registrations into Z-space attention profiles.
#[derive(Clone, Debug)]
pub struct MultiViewFusion {
    views: Vec<ViewDescriptor>,
    focus_direction: [f32; 3],
    alignment_gamma: f32,
}

impl MultiViewFusion {
    /// Builds a new fusion helper from the provided view descriptors.
    pub fn new(views: Vec<ViewDescriptor>) -> PureResult<Self> {
        if views.is_empty() {
            return Err(TensorError::EmptyInput("multi_view_fusion"));
        }
        Ok(Self {
            views,
            focus_direction: [0.0, 0.0, 1.0],
            alignment_gamma: 1.0,
        })
    }

    /// Returns the registered views.
    pub fn views(&self) -> &[ViewDescriptor] {
        &self.views
    }

    /// Returns the number of registered views.
    pub fn view_count(&self) -> usize {
        self.views.len()
    }

    /// Updates the focus direction used when modulating view weights.
    pub fn with_focus_direction(mut self, focus_direction: [f32; 3]) -> Self {
        self.focus_direction = normalise_direction(focus_direction);
        self
    }

    /// Updates the alignment gamma used to sharpen or soften orientation bias.
    pub fn with_alignment_gamma(mut self, gamma: f32) -> Self {
        self.alignment_gamma = if gamma.is_finite() {
            gamma.clamp(0.25, 8.0)
        } else {
            1.0
        };
        self
    }

    /// Returns the current focus direction.
    pub fn focus_direction(&self) -> [f32; 3] {
        self.focus_direction
    }

    /// Returns the alignment gamma used for orientation bias.
    pub fn alignment_gamma(&self) -> f32 {
        self.alignment_gamma
    }

    /// Returns the current normalised bias profile applied during fusion.
    pub fn view_bias_profile(&self) -> Vec<f32> {
        self.normalised_biases()
    }

    fn raw_biases(&self) -> Vec<f32> {
        let focus = self.focus_direction;
        let gamma = self.alignment_gamma.max(1e-3);
        self.views
            .iter()
            .map(|view| {
                let alignment = view.alignment(focus).max(1e-3).powf(gamma);
                alignment * view.baseline_weight().max(1e-3)
            })
            .collect()
    }

    fn normalised_biases(&self) -> Vec<f32> {
        let mut biases = self.raw_biases();
        if biases.is_empty() {
            return biases;
        }
        ZSpaceVolume::normalise_weights(&mut biases);
        biases
    }
}

fn normalise_direction(direction: [f32; 3]) -> [f32; 3] {
    let norm =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    if norm <= 1e-6 || !norm.is_finite() {
        [0.0, 0.0, 1.0]
    } else {
        [
            direction[0] / norm,
            direction[1] / norm,
            direction[2] / norm,
        ]
    }
}

/// Adaptive projector that fuses resonance telemetry with chrono summaries.
#[derive(Clone, Debug)]
pub struct VisionProjector {
    focus: f32,
    spread: f32,
    energy_bias: f32,
    window: SpectralWindow,
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
            window: SpectralWindow::hann(),
        }
    }

    /// Sets a custom spectral window that modulates the depth attention profile.
    pub fn with_spectral_window(mut self, window: SpectralWindow) -> Self {
        self.window = window;
        self
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

    /// Returns the configured spectral window.
    pub fn spectral_window(&self) -> SpectralWindow {
        self.window
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
        let window = self.window.weights(volume.depth());
        let spread = self.spread.max(1e-3);
        let energy_mod = 1.0 + self.energy_bias.tanh();
        if volume.depth() > 1 {
            let denom = (volume.depth() - 1) as f32;
            for (idx, weight) in weights.iter_mut().enumerate() {
                let position = if denom > 0.0 { idx as f32 / denom } else { 0.0 };
                let delta = position - self.focus;
                let gaussian = (-0.5 * (delta / spread).powi(2)).exp();
                let window_coeff = window.get(idx).copied().unwrap_or(1.0);
                *weight *= gaussian.max(1e-6) * energy_mod.max(0.1) * window_coeff.max(1e-6);
            }
        } else {
            let coeff = window.first().copied().unwrap_or(1.0);
            for weight in weights.iter_mut() {
                *weight *= energy_mod.max(0.1) * coeff.max(1e-6);
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

    /// Produces temporally smoothed depth weights using a [`TemporalResonanceBuffer`].
    pub fn depth_weights_with_temporal(
        &self,
        volume: &ZSpaceVolume,
        resonance: &DifferentialResonance,
        buffer: &mut TemporalResonanceBuffer,
    ) -> PureResult<Tensor> {
        let weights = self.compute_depth_weights(volume, resonance)?;
        let smoothed = buffer.apply(&weights)?;
        if !smoothed.is_empty() && smoothed.len() != weights.len() {
            return Err(TensorError::ShapeMismatch {
                left: (weights.len(), 1),
                right: (smoothed.len(), 1),
            });
        }
        Tensor::from_vec(1, smoothed.len(), smoothed)
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

    /// Projects the volume while folding in temporally smoothed resonance weights.
    pub fn project_with_temporal(
        &self,
        volume: &ZSpaceVolume,
        resonance: &DifferentialResonance,
        buffer: &mut TemporalResonanceBuffer,
    ) -> PureResult<Tensor> {
        let weights = self.compute_depth_weights(volume, resonance)?;
        let smoothed = buffer.apply(&weights)?;
        if !smoothed.is_empty() && smoothed.len() != weights.len() {
            return Err(TensorError::ShapeMismatch {
                left: (weights.len(), 1),
                right: (smoothed.len(), 1),
            });
        }
        volume.collapse_with_weights(&smoothed)
    }

    fn apply_multi_view_biases(
        &self,
        weights: &mut Vec<f32>,
        fusion: &MultiViewFusion,
    ) -> PureResult<()> {
        if weights.len() != fusion.view_count() {
            return Err(TensorError::ShapeMismatch {
                left: (weights.len(), 1),
                right: (fusion.view_count(), 1),
            });
        }
        let biases = fusion.normalised_biases();
        if biases.len() != weights.len() {
            return Err(TensorError::ShapeMismatch {
                left: (weights.len(), 1),
                right: (biases.len(), 1),
            });
        }
        for (weight, bias) in weights.iter_mut().zip(biases.iter()) {
            *weight *= bias.max(1e-6);
        }
        ZSpaceVolume::normalise_weights(weights);
        Ok(())
    }

    /// Produces per-view weights for a multi-camera fusion configuration.
    pub fn depth_weights_multi_view(
        &self,
        fusion: &MultiViewFusion,
        volume: &ZSpaceVolume,
        resonance: &DifferentialResonance,
    ) -> PureResult<Tensor> {
        if fusion.view_count() != volume.depth() {
            return Err(TensorError::ShapeMismatch {
                left: (fusion.view_count(), volume.height() * volume.width()),
                right: (volume.depth(), volume.height() * volume.width()),
            });
        }
        let mut weights = self.compute_depth_weights(volume, resonance)?;
        self.apply_multi_view_biases(&mut weights, fusion)?;
        Tensor::from_vec(1, weights.len(), weights)
    }

    /// Produces temporally smoothed per-view weights for a multi-camera fusion configuration.
    pub fn depth_weights_multi_view_with_temporal(
        &self,
        fusion: &MultiViewFusion,
        volume: &ZSpaceVolume,
        resonance: &DifferentialResonance,
        buffer: &mut TemporalResonanceBuffer,
    ) -> PureResult<Tensor> {
        if fusion.view_count() != volume.depth() {
            return Err(TensorError::ShapeMismatch {
                left: (fusion.view_count(), volume.height() * volume.width()),
                right: (volume.depth(), volume.height() * volume.width()),
            });
        }
        let mut weights = self.compute_depth_weights(volume, resonance)?;
        self.apply_multi_view_biases(&mut weights, fusion)?;
        let smoothed = buffer.apply(&weights)?;
        if !smoothed.is_empty() && smoothed.len() != weights.len() {
            return Err(TensorError::ShapeMismatch {
                left: (weights.len(), 1),
                right: (smoothed.len(), 1),
            });
        }
        Tensor::from_vec(1, smoothed.len(), smoothed)
    }

    /// Projects a multi-view Z-space volume into a fused 2D representation using calibrated weights.
    pub fn project_multi_view(
        &self,
        fusion: &MultiViewFusion,
        volume: &ZSpaceVolume,
        resonance: &DifferentialResonance,
    ) -> PureResult<Tensor> {
        if fusion.view_count() != volume.depth() {
            return Err(TensorError::ShapeMismatch {
                left: (fusion.view_count(), volume.height() * volume.width()),
                right: (volume.depth(), volume.height() * volume.width()),
            });
        }
        let mut weights = self.compute_depth_weights(volume, resonance)?;
        self.apply_multi_view_biases(&mut weights, fusion)?;
        volume.collapse_with_weights(&weights)
    }

    /// Projects a multi-view Z-space volume while applying temporal smoothing of the view weights.
    pub fn project_multi_view_with_temporal(
        &self,
        fusion: &MultiViewFusion,
        volume: &ZSpaceVolume,
        resonance: &DifferentialResonance,
        buffer: &mut TemporalResonanceBuffer,
    ) -> PureResult<Tensor> {
        if fusion.view_count() != volume.depth() {
            return Err(TensorError::ShapeMismatch {
                left: (fusion.view_count(), volume.height() * volume.width()),
                right: (volume.depth(), volume.height() * volume.width()),
            });
        }
        let mut weights = self.compute_depth_weights(volume, resonance)?;
        self.apply_multi_view_biases(&mut weights, fusion)?;
        let smoothed = buffer.apply(&weights)?;
        if !smoothed.is_empty() && smoothed.len() != weights.len() {
            return Err(TensorError::ShapeMismatch {
                left: (weights.len(), 1),
                right: (smoothed.len(), 1),
            });
        }
        volume.collapse_with_weights(&smoothed)
    }
}

impl Default for VisionProjector {
    fn default() -> Self {
        Self::new(0.5, 0.35, 0.0)
    }
}

/// Lightweight 3D tensor storing an image or feature map in `CHW` layout.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageTensor {
    channels: usize,
    height: usize,
    width: usize,
    data: Vec<f32>,
}

impl ImageTensor {
    /// Creates a new image tensor from raw data in `CHW` order.
    pub fn new(channels: usize, height: usize, width: usize, data: Vec<f32>) -> PureResult<Self> {
        if channels == 0 || height == 0 || width == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: channels.max(1),
                cols: height.saturating_mul(width),
            });
        }
        let expected = channels
            .checked_mul(height)
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| TensorError::InvalidDimensions {
                rows: channels,
                cols: height.saturating_mul(width),
            })?;
        if expected != data.len() {
            return Err(TensorError::DataLength {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            channels,
            height,
            width,
            data,
        })
    }

    /// Creates an image tensor filled with zeros.
    pub fn zeros(channels: usize, height: usize, width: usize) -> PureResult<Self> {
        let volume = channels
            .checked_mul(height)
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| TensorError::InvalidDimensions {
                rows: channels,
                cols: height.saturating_mul(width),
            })?;
        Ok(Self {
            channels,
            height,
            width,
            data: vec![0.0; volume],
        })
    }

    /// Builds an image tensor from a matrix-shaped tensor interpreted as `C x (H * W)`.
    pub fn from_tensor(
        tensor: &Tensor,
        channels: usize,
        height: usize,
        width: usize,
    ) -> PureResult<Self> {
        let (rows, cols) = tensor.shape();
        if rows != channels || cols != height.saturating_mul(width) {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (channels, height.saturating_mul(width)),
            });
        }
        Self::new(channels, height, width, tensor.data().to_vec())
    }

    /// Converts the image tensor back into a `Tensor` with shape `C x (H * W)`.
    pub fn into_tensor(&self) -> PureResult<Tensor> {
        Tensor::from_vec(self.channels, self.height * self.width, self.data.clone())
    }

    /// Returns the number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the image height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the image width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns `(channels, height, width)` describing the tensor.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.channels, self.height, self.width)
    }

    /// Immutable view of the underlying data buffer in `CHW` order.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable view of the underlying data buffer in `CHW` order.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    fn offset(&self, channel: usize, y: usize, x: usize) -> PureResult<usize> {
        if channel >= self.channels || y >= self.height || x >= self.width {
            return Err(TensorError::InvalidValue {
                label: "image_tensor_coordinate",
            });
        }
        Ok(((channel * self.height) + y) * self.width + x)
    }

    /// Reads a pixel at the specified coordinate.
    pub fn pixel(&self, channel: usize, y: usize, x: usize) -> PureResult<f32> {
        let idx = self.offset(channel, y, x)?;
        Ok(self.data[idx])
    }

    /// Updates a pixel at the specified coordinate.
    pub fn set_pixel(&mut self, channel: usize, y: usize, x: usize, value: f32) -> PureResult<()> {
        let idx = self.offset(channel, y, x)?;
        self.data[idx] = value;
        Ok(())
    }

    /// Applies an element-wise transform to the image data in-place.
    pub fn map_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(f32) -> f32,
    {
        for value in self.data.iter_mut() {
            *value = f(*value);
        }
    }

    /// Applies a ReLU non-linearity in-place.
    pub fn relu_inplace(&mut self) {
        for value in self.data.iter_mut() {
            if *value < 0.0 {
                *value = 0.0;
            }
        }
    }

    /// Returns a flattened copy of the image data suitable for dense layers.
    pub fn flatten(&self) -> Vec<f32> {
        self.data.clone()
    }

    /// Builds an image tensor from a [`ZSpaceVolume`], treating depth slices as channels.
    pub fn from_volume(volume: &ZSpaceVolume) -> PureResult<Self> {
        Self::new(
            volume.depth(),
            volume.height(),
            volume.width(),
            volume.voxels().to_vec(),
        )
    }
}

impl ZSpaceVolume {
    /// Creates a volume by interpreting each channel of the provided image as a Z slice.
    pub fn from_image_tensor(image: &ImageTensor) -> PureResult<Self> {
        Ok(Self {
            depth: image.channels(),
            height: image.height(),
            width: image.width(),
            voxels: image.as_slice().to_vec(),
        })
    }

    /// Converts the volume back into an [`ImageTensor`].
    pub fn to_image_tensor(&self) -> PureResult<ImageTensor> {
        ImageTensor::new(self.depth, self.height, self.width, self.voxels.clone())
    }

    /// Computes per-slice summary statistics for resonance synthesis.
    pub fn slice_profile(&self) -> PureResult<ZSliceProfile> {
        let slice_len = self.height * self.width;
        if slice_len == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: self.height,
                cols: self.width,
            });
        }
        let mut means = Vec::with_capacity(self.depth);
        let mut stds = Vec::with_capacity(self.depth);
        let mut energies = Vec::with_capacity(self.depth);
        let normaliser = slice_len as f32;
        for idx in 0..self.depth {
            let start = idx * slice_len;
            let end = start + slice_len;
            let slice = &self.voxels[start..end];
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for &value in slice.iter() {
                sum += value;
                sum_sq += value * value;
            }
            let mean = sum / normaliser;
            let variance = (sum_sq / normaliser) - mean.powi(2);
            means.push(mean);
            stds.push(variance.max(0.0).sqrt());
            energies.push(sum_sq / normaliser);
        }
        ZSliceProfile::new(means, stds, energies)
    }

    /// Computes the mean squared energy across the entire volume.
    pub fn total_energy(&self) -> f32 {
        if self.voxels.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = self.voxels.iter().map(|value| value * value).sum();
        sum_sq / (self.voxels.len() as f32)
    }
}

/// Canonical computer-vision task categories inspired by TorchVision's taxonomy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionTask {
    Classification,
    Segmentation,
    Detection,
    Keypoint,
    Video,
    OpticalFlow,
    Depth,
    MultiLabel,
}

/// Metadata describing a well-known dataset within the TorchVision ecosystem.
#[derive(Clone, Debug)]
pub struct DatasetDescriptor {
    pub name: &'static str,
    pub task: VisionTask,
    pub description: &'static str,
    pub supports_download: bool,
    pub homepage: &'static str,
    pub paper_url: Option<&'static str>,
}

const DATASET_CATALOG: &[DatasetDescriptor] = &[
    DatasetDescriptor {
        name: "MNIST",
        task: VisionTask::Classification,
        description: "手書き数字 28x28 の画像分類タスク",
        supports_download: true,
        homepage: "http://yann.lecun.com/exdb/mnist/",
        paper_url: Some("http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf"),
    },
    DatasetDescriptor {
        name: "FashionMNIST",
        task: VisionTask::Classification,
        description: "Zalando ファッション画像の 10 クラス分類",
        supports_download: true,
        homepage: "https://github.com/zalandoresearch/fashion-mnist",
        paper_url: None,
    },
    DatasetDescriptor {
        name: "CIFAR10",
        task: VisionTask::Classification,
        description: "32x32 カラー画像の 10 クラス分類",
        supports_download: true,
        homepage: "https://www.cs.toronto.edu/~kriz/cifar.html",
        paper_url: None,
    },
    DatasetDescriptor {
        name: "CIFAR100",
        task: VisionTask::Classification,
        description: "CIFAR10 と同条件で 100 クラス分類",
        supports_download: true,
        homepage: "https://www.cs.toronto.edu/~kriz/cifar.html",
        paper_url: None,
    },
    DatasetDescriptor {
        name: "ImageNet-1K",
        task: VisionTask::Classification,
        description: "ILSVRC 1K クラス大規模分類ベンチマーク",
        supports_download: false,
        homepage: "https://image-net.org/",
        paper_url: Some("https://arxiv.org/abs/1409.0575"),
    },
    DatasetDescriptor {
        name: "COCO-2017",
        task: VisionTask::Detection,
        description: "Common Objects in Context の検出/セグメンテーション",
        supports_download: true,
        homepage: "https://cocodataset.org/",
        paper_url: Some("https://arxiv.org/abs/1405.0312"),
    },
    DatasetDescriptor {
        name: "Cityscapes",
        task: VisionTask::Segmentation,
        description: "自動運転向け都市シーンの高解像度セマンティックセグメンテーション",
        supports_download: false,
        homepage: "https://www.cityscapes-dataset.com/",
        paper_url: Some("https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.pdf"),
    },
    DatasetDescriptor {
        name: "VOC-2012",
        task: VisionTask::Detection,
        description: "PASCAL VOC 検出/セグメンテーション競技用データセット",
        supports_download: true,
        homepage: "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/",
        paper_url: None,
    },
    DatasetDescriptor {
        name: "Kinetics-400",
        task: VisionTask::Video,
        description: "400 クラスの人間行動動画分類",
        supports_download: false,
        homepage: "https://deepmind.com/research/open-source/kinetics",
        paper_url: Some("https://arxiv.org/abs/1705.06950"),
    },
    DatasetDescriptor {
        name: "FlyingChairs",
        task: VisionTask::OpticalFlow,
        description: "光学フロー合成データセット",
        supports_download: true,
        homepage: "https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html",
        paper_url: Some("https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/"),
    },
];

/// Returns the static catalog of datasets aligned with TorchVision.
pub fn dataset_catalog() -> &'static [DatasetDescriptor] {
    DATASET_CATALOG
}

/// Finds a dataset descriptor by name (case-insensitive).
pub fn find_dataset_descriptor(name: &str) -> Option<&'static DatasetDescriptor> {
    DATASET_CATALOG
        .iter()
        .find(|descriptor| descriptor.name.eq_ignore_ascii_case(name))
}

/// Sample item returned by a [`VisionDataset`].
#[derive(Clone, Debug)]
pub struct DatasetSample {
    pub image: ImageTensor,
    pub target: Option<Tensor>,
    pub label: Option<String>,
    pub boxes: Option<Vec<[f32; 4]>>,
    pub masks: Option<Vec<ImageTensor>>,
}

impl DatasetSample {
    pub fn new(image: ImageTensor) -> Self {
        Self {
            image,
            target: None,
            label: None,
            boxes: None,
            masks: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_target(mut self, target: Tensor) -> Self {
        self.target = Some(target);
        self
    }

    pub fn with_boxes(mut self, boxes: Vec<[f32; 4]>) -> Self {
        self.boxes = Some(boxes);
        self
    }

    pub fn with_masks(mut self, masks: Vec<ImageTensor>) -> Self {
        self.masks = Some(masks);
        self
    }
}

/// Unified dataset trait mirroring TorchVision's pythonic interfaces.
pub trait VisionDataset: Send + Sync {
    fn descriptor(&self) -> &DatasetDescriptor;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> PureResult<DatasetSample>;
}

/// Simple in-memory dataset backed by vectors of samples.
#[derive(Clone, Debug)]
pub struct TensorVisionDataset {
    descriptor: DatasetDescriptor,
    expected_shape: Option<(usize, usize, usize)>,
    samples: Vec<DatasetSample>,
}

impl TensorVisionDataset {
    pub fn new(descriptor: DatasetDescriptor) -> Self {
        Self {
            descriptor,
            expected_shape: None,
            samples: Vec::new(),
        }
    }

    pub fn from_samples(
        descriptor: DatasetDescriptor,
        samples: Vec<DatasetSample>,
    ) -> PureResult<Self> {
        let mut dataset = Self::new(descriptor);
        for sample in samples {
            dataset.push_sample(sample)?;
        }
        Ok(dataset)
    }

    pub fn push_sample(&mut self, sample: DatasetSample) -> PureResult<()> {
        if let Some(shape) = self.expected_shape {
            if sample.image.shape() != shape {
                let left = sample.image.shape();
                let right = shape;
                return Err(TensorError::ShapeMismatch {
                    left: (left.0, left.1 * left.2),
                    right: (right.0, right.1 * right.2),
                });
            }
        } else {
            self.expected_shape = Some(sample.image.shape());
        }
        self.samples.push(sample);
        Ok(())
    }
}

impl VisionDataset for TensorVisionDataset {
    fn descriptor(&self) -> &DatasetDescriptor {
        &self.descriptor
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> PureResult<DatasetSample> {
        self.samples
            .get(index)
            .cloned()
            .ok_or(TensorError::InvalidValue {
                label: "tensor_dataset_index",
            })
    }
}

/// A batch of dataset samples produced by a [`DataLoader`].
#[derive(Clone, Debug)]
pub struct VisionBatch {
    pub images: Vec<ImageTensor>,
    pub targets: Vec<Option<Tensor>>,
    pub labels: Vec<Option<String>>,
    pub boxes: Vec<Option<Vec<[f32; 4]>>>,
    pub masks: Vec<Option<Vec<ImageTensor>>>,
}

impl VisionBatch {
    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Stacks the batch into a `(N, C*H*W)` matrix.
    pub fn stack(&self) -> PureResult<Tensor> {
        if self.images.is_empty() {
            return Err(TensorError::EmptyInput("vision_batch"));
        }
        let (channels, height, width) = self.images[0].shape();
        let mut data = Vec::with_capacity(self.images.len() * channels * height * width);
        for image in &self.images {
            if image.shape() != (channels, height, width) {
                let left = image.shape();
                return Err(TensorError::ShapeMismatch {
                    left: (left.0, left.1 * left.2),
                    right: (channels, height * width),
                });
            }
            data.extend_from_slice(image.as_slice());
        }
        Tensor::from_vec(self.images.len(), channels * height * width, data)
    }
}

/// Iterator over dataset samples with optional transforms and shuffling.
pub struct DataLoader<D: VisionDataset> {
    dataset: Arc<D>,
    batch_size: usize,
    order: Vec<usize>,
    position: usize,
    shuffle: bool,
    shuffle_rng: StdRng,
    pipeline: Option<TransformPipeline>,
}

impl<D: VisionDataset> DataLoader<D> {
    pub fn new(dataset: Arc<D>, batch_size: usize, seed: Option<u64>) -> PureResult<Self> {
        if batch_size == 0 {
            return Err(TensorError::InvalidValue {
                label: "dataloader_batch_size",
            });
        }
        let len = dataset.len();
        let order: Vec<usize> = (0..len).collect();
        let shuffle_rng = match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => StdRng::from_entropy(),
        };
        Ok(Self {
            dataset,
            batch_size,
            order,
            position: 0,
            shuffle: false,
            shuffle_rng,
            pipeline: None,
        })
    }

    pub fn with_pipeline(mut self, pipeline: TransformPipeline) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    pub fn enable_shuffle(&mut self, shuffle: bool) {
        self.shuffle = shuffle;
        if self.shuffle {
            self.shuffle_order();
        } else {
            self.order = (0..self.dataset.len()).collect();
            self.position = 0;
        }
    }

    fn shuffle_order(&mut self) {
        self.order = (0..self.dataset.len()).collect();
        // Fisher-Yates shuffle
        for i in (1..self.order.len()).rev() {
            let j = self.shuffle_rng.gen_range(0..=i);
            self.order.swap(i, j);
        }
        self.position = 0;
    }

    pub fn reset(&mut self) {
        self.position = 0;
        if self.shuffle {
            self.shuffle_order();
        }
    }

    pub fn next_batch(&mut self) -> PureResult<Option<VisionBatch>> {
        if self.position >= self.order.len() {
            return Ok(None);
        }
        let end = min(self.position + self.batch_size, self.order.len());
        let mut images = Vec::with_capacity(end - self.position);
        let mut targets = Vec::with_capacity(end - self.position);
        let mut labels = Vec::with_capacity(end - self.position);
        let mut boxes = Vec::with_capacity(end - self.position);
        let mut masks = Vec::with_capacity(end - self.position);
        for &idx in &self.order[self.position..end] {
            let mut sample = self.dataset.get(idx)?;
            if let Some(pipeline) = self.pipeline.as_mut() {
                pipeline.apply(&mut sample.image)?;
            }
            targets.push(sample.target.clone());
            labels.push(sample.label.clone());
            boxes.push(sample.boxes.clone());
            masks.push(sample.masks.clone());
            images.push(sample.image);
        }
        self.position = end;
        Ok(Some(VisionBatch {
            images,
            targets,
            labels,
            boxes,
            masks,
        }))
    }
}

/// Transform operations that mimic `torchvision.transforms` behaviour.
#[derive(Clone, Debug)]
pub enum TransformOperation {
    Normalize(Normalize),
    Resize(Resize),
    CenterCrop(CenterCrop),
    RandomHorizontalFlip(RandomHorizontalFlip),
}

impl TransformOperation {
    pub fn name(&self) -> &'static str {
        match self {
            TransformOperation::Normalize(_) => "Normalize",
            TransformOperation::Resize(_) => "Resize",
            TransformOperation::CenterCrop(_) => "CenterCrop",
            TransformOperation::RandomHorizontalFlip(_) => "RandomHorizontalFlip",
        }
    }

    fn apply(&self, image: &mut ImageTensor, rng: &mut StdRng) -> PureResult<()> {
        match self {
            TransformOperation::Normalize(op) => op.apply(image),
            TransformOperation::Resize(op) => op.apply(image),
            TransformOperation::CenterCrop(op) => op.apply(image),
            TransformOperation::RandomHorizontalFlip(op) => op.apply(image, rng),
        }
    }
}

/// Sequential container for image transforms.
#[derive(Clone, Debug)]
pub struct TransformPipeline {
    ops: Vec<TransformOperation>,
    rng: StdRng,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            rng: StdRng::from_entropy(),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            ops: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn add(&mut self, op: TransformOperation) -> &mut Self {
        self.ops.push(op);
        self
    }

    pub fn apply(&mut self, image: &mut ImageTensor) -> PureResult<()> {
        for op in &self.ops {
            op.apply(image, &mut self.rng)?;
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Per-channel normalization.
#[derive(Clone, Debug)]
pub struct Normalize {
    means: Vec<f32>,
    stds: Vec<f32>,
}

impl Normalize {
    pub fn new(means: Vec<f32>, stds: Vec<f32>) -> PureResult<Self> {
        if means.is_empty() || stds.is_empty() {
            return Err(TensorError::EmptyInput("normalize_stats"));
        }
        if means.len() != stds.len() {
            return Err(TensorError::DataLength {
                expected: means.len(),
                got: stds.len(),
            });
        }
        for &std in &stds {
            if std <= 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "normalize_std",
                });
            }
        }
        Ok(Self { means, stds })
    }

    fn channel_value(values: &[f32], channel: usize) -> f32 {
        if values.len() == 1 {
            values[0]
        } else {
            values[channel.min(values.len() - 1)]
        }
    }

    pub fn apply(&self, image: &mut ImageTensor) -> PureResult<()> {
        let channels = image.channels();
        let spatial = image.height() * image.width();
        for c in 0..channels {
            let mean = Self::channel_value(&self.means, c);
            let std = Self::channel_value(&self.stds, c);
            for idx in 0..spatial {
                let offset = c * spatial + idx;
                image.as_mut_slice()[offset] = (image.as_slice()[offset] - mean) / std;
            }
        }
        Ok(())
    }
}

/// Bilinear resize matching TorchVision's default interpolation.
#[derive(Clone, Debug)]
pub struct Resize {
    height: usize,
    width: usize,
}

impl Resize {
    pub fn new(height: usize, width: usize) -> PureResult<Self> {
        if height == 0 || width == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: height,
                cols: width,
            });
        }
        Ok(Self { height, width })
    }

    pub fn apply(&self, image: &mut ImageTensor) -> PureResult<()> {
        let channels = image.channels();
        let in_h = image.height();
        let in_w = image.width();
        if in_h == self.height && in_w == self.width {
            return Ok(());
        }
        let mut output = vec![0.0f32; channels * self.height * self.width];
        let scale_y = in_h as f32 / self.height as f32;
        let scale_x = in_w as f32 / self.width as f32;
        for c in 0..channels {
            for y in 0..self.height {
                let src_y = (y as f32 + 0.5) * scale_y - 0.5;
                let y0 = src_y.floor().clamp(0.0, (in_h - 1) as f32) as usize;
                let y1 = min(y0 + 1, in_h - 1);
                let ly = src_y - y0 as f32;
                for x in 0..self.width {
                    let src_x = (x as f32 + 0.5) * scale_x - 0.5;
                    let x0 = src_x.floor().clamp(0.0, (in_w - 1) as f32) as usize;
                    let x1 = min(x0 + 1, in_w - 1);
                    let lx = src_x - x0 as f32;
                    let top_left = image.pixel(c, y0, x0)?;
                    let top_right = image.pixel(c, y0, x1)?;
                    let bottom_left = image.pixel(c, y1, x0)?;
                    let bottom_right = image.pixel(c, y1, x1)?;
                    let top = top_left * (1.0 - lx) + top_right * lx;
                    let bottom = bottom_left * (1.0 - lx) + bottom_right * lx;
                    let value = top * (1.0 - ly) + bottom * ly;
                    let offset = ((c * self.height) + y) * self.width + x;
                    output[offset] = value;
                }
            }
        }
        *image = ImageTensor::new(channels, self.height, self.width, output)?;
        Ok(())
    }
}

/// Crops the spatial center of the image.
#[derive(Clone, Debug)]
pub struct CenterCrop {
    height: usize,
    width: usize,
}

impl CenterCrop {
    pub fn new(height: usize, width: usize) -> PureResult<Self> {
        if height == 0 || width == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: height,
                cols: width,
            });
        }
        Ok(Self { height, width })
    }

    pub fn apply(&self, image: &mut ImageTensor) -> PureResult<()> {
        if self.height > image.height() || self.width > image.width() {
            return Err(TensorError::InvalidValue {
                label: "center_crop_size",
            });
        }
        let top = (image.height() - self.height) / 2;
        let left = (image.width() - self.width) / 2;
        let channels = image.channels();
        let mut output = vec![0.0f32; channels * self.height * self.width];
        for c in 0..channels {
            for y in 0..self.height {
                for x in 0..self.width {
                    let value = image.pixel(c, top + y, left + x)?;
                    let offset = ((c * self.height) + y) * self.width + x;
                    output[offset] = value;
                }
            }
        }
        *image = ImageTensor::new(channels, self.height, self.width, output)?;
        Ok(())
    }
}

/// Randomly flips the image horizontally.
#[derive(Clone, Debug)]
pub struct RandomHorizontalFlip {
    probability: f32,
}

impl RandomHorizontalFlip {
    pub fn new(probability: f32) -> PureResult<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(TensorError::InvalidValue {
                label: "horizontal_flip_probability",
            });
        }
        Ok(Self { probability })
    }

    pub fn apply(&self, image: &mut ImageTensor, rng: &mut StdRng) -> PureResult<()> {
        if rng.gen::<f32>() >= self.probability {
            return Ok(());
        }
        let channels = image.channels();
        let height = image.height();
        let width = image.width();
        let data = image.as_mut_slice();
        let stride_c = height * width;
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width / 2 {
                    let left = c * stride_c + y * width + x;
                    let right = c * stride_c + y * width + (width - 1 - x);
                    data.swap(left, right);
                }
            }
        }
        Ok(())
    }
}

/// Helper that reproduces TorchVision の ImageNet 分類向け前処理パイプライン。
pub fn standard_classification_pipeline(
    image_size: usize,
    seed: Option<u64>,
) -> PureResult<TransformPipeline> {
    let mut pipeline = match seed {
        Some(value) => TransformPipeline::with_seed(value),
        None => TransformPipeline::new(),
    };
    pipeline
        .add(TransformOperation::Resize(Resize::new(
            image_size + 32,
            image_size + 32,
        )?))
        .add(TransformOperation::CenterCrop(CenterCrop::new(
            image_size, image_size,
        )?))
        .add(TransformOperation::RandomHorizontalFlip(
            RandomHorizontalFlip::new(0.5)?,
        ))
        .add(TransformOperation::Normalize(Normalize::new(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        )?));
    Ok(pipeline)
}

/// TorchVision モデルの主要カテゴリ。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelKind {
    ResNet18,
    ResNet50,
    MobileNetV3Small,
    MobileNetV3Large,
    EfficientNetB0,
    ConvNeXtTiny,
}

/// 静的なモデル記述。
#[derive(Clone, Debug)]
pub struct ModelDescriptor {
    pub kind: ModelKind,
    pub name: &'static str,
    pub task: VisionTask,
    pub default_input_channels: usize,
    pub default_image_size: (usize, usize),
    pub has_pretrained: bool,
}

const MODEL_CATALOG: &[ModelDescriptor] = &[
    ModelDescriptor {
        kind: ModelKind::ResNet18,
        name: "resnet18",
        task: VisionTask::Classification,
        default_input_channels: 3,
        default_image_size: (224, 224),
        has_pretrained: true,
    },
    ModelDescriptor {
        kind: ModelKind::ResNet50,
        name: "resnet50",
        task: VisionTask::Classification,
        default_input_channels: 3,
        default_image_size: (224, 224),
        has_pretrained: true,
    },
    ModelDescriptor {
        kind: ModelKind::MobileNetV3Small,
        name: "mobilenet_v3_small",
        task: VisionTask::Classification,
        default_input_channels: 3,
        default_image_size: (224, 224),
        has_pretrained: true,
    },
    ModelDescriptor {
        kind: ModelKind::MobileNetV3Large,
        name: "mobilenet_v3_large",
        task: VisionTask::Classification,
        default_input_channels: 3,
        default_image_size: (224, 224),
        has_pretrained: true,
    },
    ModelDescriptor {
        kind: ModelKind::EfficientNetB0,
        name: "efficientnet_b0",
        task: VisionTask::Classification,
        default_input_channels: 3,
        default_image_size: (224, 224),
        has_pretrained: true,
    },
    ModelDescriptor {
        kind: ModelKind::ConvNeXtTiny,
        name: "convnext_tiny",
        task: VisionTask::Classification,
        default_input_channels: 3,
        default_image_size: (224, 224),
        has_pretrained: true,
    },
];

pub fn model_catalog() -> &'static [ModelDescriptor] {
    MODEL_CATALOG
}

pub fn find_model_descriptor(name: &str) -> Option<&'static ModelDescriptor> {
    MODEL_CATALOG
        .iter()
        .find(|descriptor| descriptor.name.eq_ignore_ascii_case(name))
}

/// ランタイムのモデルメタデータ。
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub kind: ModelKind,
    pub name: &'static str,
    pub task: VisionTask,
    pub input_channels: usize,
    pub image_size: (usize, usize),
    pub num_classes: usize,
    pub has_pretrained: bool,
}

impl ModelMetadata {
    fn from_descriptor(descriptor: &ModelDescriptor, num_classes: usize) -> Self {
        Self {
            kind: descriptor.kind,
            name: descriptor.name,
            task: descriptor.task,
            input_channels: descriptor.default_input_channels,
            image_size: descriptor.default_image_size,
            num_classes,
            has_pretrained: descriptor.has_pretrained,
        }
    }
}

/// 特徴抽出段を制御するステージ。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureStage {
    Stem,
    Head,
    Logits,
}

/// 共通インターフェース (TorchVision の `nn.Module`)。
pub trait VisionModel: Send + Sync {
    fn metadata(&self) -> &ModelMetadata;
    fn forward(&self, batch: &[ImageTensor]) -> PureResult<Tensor>;
    fn extract_features(&self, stage: FeatureStage, image: &ImageTensor) -> PureResult<Tensor>;
}

fn random_weight(rng: &mut StdRng, scale: f32) -> f32 {
    rng.gen_range(-scale..scale)
}

struct Conv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv2d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        rng: &mut StdRng,
    ) -> PureResult<Self> {
        if kernel_size == 0 {
            return Err(TensorError::InvalidValue {
                label: "conv_kernel_size",
            });
        }
        let weight_count = out_channels
            .checked_mul(in_channels)
            .and_then(|v| v.checked_mul(kernel_size * kernel_size))
            .ok_or(TensorError::InvalidDimensions {
                rows: out_channels,
                cols: in_channels * kernel_size * kernel_size,
            })?;
        let mut weights = Vec::with_capacity(weight_count);
        for _ in 0..weight_count {
            weights.push(random_weight(rng, 0.05));
        }
        let mut bias = Vec::with_capacity(out_channels);
        for _ in 0..out_channels {
            bias.push(random_weight(rng, 0.01));
        }
        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
        })
    }

    fn output_dims(&self, input: &ImageTensor) -> PureResult<(usize, usize)> {
        let padded_h = input.height() + 2 * self.padding;
        let padded_w = input.width() + 2 * self.padding;
        if padded_h < self.kernel_size || padded_w < self.kernel_size {
            return Err(TensorError::InvalidValue {
                label: "conv_output_dims",
            });
        }
        let out_h = (padded_h - self.kernel_size) / self.stride + 1;
        let out_w = (padded_w - self.kernel_size) / self.stride + 1;
        Ok((out_h, out_w))
    }

    fn apply(&self, input: &ImageTensor) -> PureResult<ImageTensor> {
        if input.channels() != self.in_channels {
            return Err(TensorError::ShapeMismatch {
                left: (input.channels(), input.height() * input.width()),
                right: (self.in_channels, input.height() * input.width()),
            });
        }
        let (out_h, out_w) = self.output_dims(input)?;
        let mut output = vec![0.0f32; self.out_channels * out_h * out_w];
        let kernel = self.kernel_size;
        for oc in 0..self.out_channels {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = self.bias[oc];
                    for ic in 0..self.in_channels {
                        for ky in 0..kernel {
                            for kx in 0..kernel {
                                let iy = oy * self.stride + ky;
                                let ix = ox * self.stride + kx;
                                let in_y = iy as isize - self.padding as isize;
                                let in_x = ix as isize - self.padding as isize;
                                if in_y >= 0
                                    && in_y < input.height() as isize
                                    && in_x >= 0
                                    && in_x < input.width() as isize
                                {
                                    let input_value =
                                        input.pixel(ic, in_y as usize, in_x as usize)?;
                                    let w_index = (((oc * self.in_channels + ic) * kernel + ky)
                                        * kernel)
                                        + kx;
                                    acc += input_value * self.weights[w_index];
                                }
                            }
                        }
                    }
                    let offset = ((oc * out_h) + oy) * out_w + ox;
                    output[offset] = acc;
                }
            }
        }
        ImageTensor::new(self.out_channels, out_h, out_w, output)
    }
}

struct MaxPool2d {
    kernel_size: usize,
    stride: usize,
}

impl MaxPool2d {
    fn new(kernel_size: usize, stride: usize) -> PureResult<Self> {
        if kernel_size == 0 || stride == 0 {
            return Err(TensorError::InvalidValue {
                label: "maxpool_params",
            });
        }
        Ok(Self {
            kernel_size,
            stride,
        })
    }

    fn apply(&self, input: &ImageTensor) -> PureResult<ImageTensor> {
        let channels = input.channels();
        let out_h = (input.height() - self.kernel_size) / self.stride + 1;
        let out_w = (input.width() - self.kernel_size) / self.stride + 1;
        if out_h == 0 || out_w == 0 {
            return Err(TensorError::InvalidValue {
                label: "maxpool_output",
            });
        }
        let mut output = vec![0.0f32; channels * out_h * out_w];
        for c in 0..channels {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut max_value = f32::MIN;
                    for ky in 0..self.kernel_size {
                        for kx in 0..self.kernel_size {
                            let value =
                                input.pixel(c, oy * self.stride + ky, ox * self.stride + kx)?;
                            if value > max_value {
                                max_value = value;
                            }
                        }
                    }
                    let offset = ((c * out_h) + oy) * out_w + ox;
                    output[offset] = max_value;
                }
            }
        }
        ImageTensor::new(channels, out_h, out_w, output)
    }
}

struct LinearLayer {
    in_features: usize,
    out_features: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearLayer {
    fn new(in_features: usize, out_features: usize, rng: &mut StdRng) -> PureResult<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(TensorError::InvalidValue {
                label: "linear_features",
            });
        }
        let mut weights = Vec::with_capacity(out_features * in_features);
        for _ in 0..out_features * in_features {
            weights.push(random_weight(rng, 0.05));
        }
        let mut bias = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            bias.push(random_weight(rng, 0.01));
        }
        Ok(Self {
            in_features,
            out_features,
            weights,
            bias,
        })
    }

    fn forward(&self, input: &[f32]) -> PureResult<Vec<f32>> {
        if input.len() != self.in_features {
            return Err(TensorError::DataLength {
                expected: self.in_features,
                got: input.len(),
            });
        }
        let mut output = vec![0.0f32; self.out_features];
        for o in 0..self.out_features {
            let mut acc = self.bias[o];
            let row_offset = o * self.in_features;
            for i in 0..self.in_features {
                acc += self.weights[row_offset + i] * input[i];
            }
            output[o] = acc;
        }
        Ok(output)
    }
}

/// シンプルな CNN を用いた TorchVision 互換モデルのラッパー。
pub struct SimpleCnn {
    metadata: ModelMetadata,
    conv1: Conv2d,
    conv2: Conv2d,
    pool1: MaxPool2d,
    pool2: MaxPool2d,
    fc1: LinearLayer,
    fc2: LinearLayer,
}

impl SimpleCnn {
    pub fn with_seed(kind: ModelKind, num_classes: usize, seed: Option<u64>) -> PureResult<Self> {
        let descriptor = MODEL_CATALOG.iter().find(|item| item.kind == kind).ok_or(
            TensorError::InvalidValue {
                label: "model_kind",
            },
        )?;
        let mut rng = match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => StdRng::from_entropy(),
        };
        let metadata = ModelMetadata::from_descriptor(descriptor, num_classes);
        let (conv1_out, conv2_out, hidden) = match kind {
            ModelKind::ResNet18 | ModelKind::ResNet50 => (32, 64, 128),
            ModelKind::MobileNetV3Small => (16, 32, 64),
            ModelKind::MobileNetV3Large => (24, 48, 96),
            ModelKind::EfficientNetB0 => (24, 56, 112),
            ModelKind::ConvNeXtTiny => (32, 64, 128),
        };
        let conv1 = Conv2d::new(metadata.input_channels, conv1_out, 3, 2, 1, &mut rng)?;
        let conv2 = Conv2d::new(conv1_out, conv2_out, 3, 2, 1, &mut rng)?;
        let pool1 = MaxPool2d::new(2, 2)?;
        let pool2 = MaxPool2d::new(2, 2)?;

        // compute spatial dims after conv/pool pipeline
        let dummy = ImageTensor::zeros(
            metadata.input_channels,
            metadata.image_size.0,
            metadata.image_size.1,
        )?;
        let stem = conv1.apply(&dummy)?;
        let stem_pooled = pool1.apply(&stem)?;
        let head = conv2.apply(&stem_pooled)?;
        let pooled = pool2.apply(&head)?;
        let flattened_len = pooled.flatten().len();

        let fc1 = LinearLayer::new(flattened_len, hidden, &mut rng)?;
        let fc2 = LinearLayer::new(hidden, num_classes, &mut rng)?;

        Ok(Self {
            metadata,
            conv1,
            conv2,
            pool1,
            pool2,
            fc1,
            fc2,
        })
    }

    fn validate_input(&self, image: &ImageTensor) -> PureResult<()> {
        if image.channels() != self.metadata.input_channels
            || image.height() != self.metadata.image_size.0
            || image.width() != self.metadata.image_size.1
        {
            return Err(TensorError::ShapeMismatch {
                left: (image.channels(), image.height() * image.width()),
                right: (
                    self.metadata.input_channels,
                    self.metadata.image_size.0 * self.metadata.image_size.1,
                ),
            });
        }
        Ok(())
    }

    fn forward_logits(&self, image: &ImageTensor) -> PureResult<Vec<f32>> {
        self.validate_input(image)?;
        let mut stem = self.conv1.apply(image)?;
        stem.relu_inplace();
        let stem_pooled = self.pool1.apply(&stem)?;
        let mut head = self.conv2.apply(&stem_pooled)?;
        head.relu_inplace();
        let head_pooled = self.pool2.apply(&head)?;
        let mut hidden = self.fc1.forward(&head_pooled.flatten())?;
        for value in hidden.iter_mut() {
            if *value < 0.0 {
                *value = 0.0;
            }
        }
        self.fc2.forward(&hidden)
    }
}

impl VisionModel for SimpleCnn {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn forward(&self, batch: &[ImageTensor]) -> PureResult<Tensor> {
        if batch.is_empty() {
            return Err(TensorError::EmptyInput("vision_batch_forward"));
        }
        let mut logits = Vec::with_capacity(batch.len() * self.metadata.num_classes);
        for image in batch {
            let mut scores = self.forward_logits(image)?;
            logits.append(&mut scores);
        }
        Tensor::from_vec(batch.len(), self.metadata.num_classes, logits)
    }

    fn extract_features(&self, stage: FeatureStage, image: &ImageTensor) -> PureResult<Tensor> {
        self.validate_input(image)?;
        match stage {
            FeatureStage::Stem => {
                let mut stem = self.conv1.apply(image)?;
                stem.relu_inplace();
                stem.into_tensor()
            }
            FeatureStage::Head => {
                let mut stem = self.conv1.apply(image)?;
                stem.relu_inplace();
                let stem_pooled = self.pool1.apply(&stem)?;
                let mut head = self.conv2.apply(&stem_pooled)?;
                head.relu_inplace();
                let pooled = self.pool2.apply(&head)?;
                let flat = pooled.flatten();
                Tensor::from_vec(1, flat.len(), flat)
            }
            FeatureStage::Logits => {
                let logits = self.forward_logits(image)?;
                Tensor::from_vec(1, logits.len(), logits)
            }
        }
    }
}

/// Feature extractor mirroring `torchvision.models.feature_extraction`.
pub struct FeatureExtractor {
    model: Arc<dyn VisionModel>,
    stage: FeatureStage,
}

impl FeatureExtractor {
    pub fn new(model: Arc<dyn VisionModel>, stage: FeatureStage) -> Self {
        Self { model, stage }
    }

    pub fn extract(&self, image: &ImageTensor) -> PureResult<Tensor> {
        self.model.extract_features(self.stage, image)
    }
}

/// Instantiates a simplified TorchVision モデル。
pub fn create_classification_model(
    kind: ModelKind,
    num_classes: usize,
    seed: Option<u64>,
) -> PureResult<Arc<dyn VisionModel>> {
    let model = SimpleCnn::with_seed(kind, num_classes, seed)?;
    Ok(Arc::new(model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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

    fn toy_summary() -> ChronoSummary {
        ChronoSummary {
            frames: 4,
            duration: 0.25,
            latest_timestamp: 1.5,
            mean_drift: 0.05,
            mean_abs_drift: 0.06,
            drift_std: 0.02,
            mean_energy: 1.4,
            energy_std: 0.3,
            mean_decay: -0.1,
            min_energy: 1.1,
            max_energy: 1.8,
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
        let resonance = DifferentialResonance {
            homotopy_flow: tensor_from(&[0.0, 0.0, 0.0], 1, 3),
            functor_linearisation: tensor_from(&[0.0, 0.0, 0.0], 1, 3),
            recursive_objective: tensor_from(&[0.0, 0.0, 0.0], 1, 3),
            infinity_projection: tensor_from(&[0.0, 0.0, 0.0], 1, 3),
            infinity_energy: tensor_from(&[1.0, 1.0, 1.0], 1, 3),
        };
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
    fn volume_accumulate_performs_temporal_ema() {
        let front_a = [0.0, 1.0, 2.0, 3.0];
        let back_a = [4.0, 5.0, 6.0, 7.0];
        let front_b = [8.0, 6.0, 4.0, 2.0];
        let back_b = [1.0, 3.0, 5.0, 7.0];
        let base_slices = vec![tensor_from(&front_a, 2, 2), tensor_from(&back_a, 2, 2)];
        let next_slices = vec![tensor_from(&front_b, 2, 2), tensor_from(&back_b, 2, 2)];
        let mut ema = ZSpaceVolume::from_slices(&base_slices).unwrap();
        let next = ZSpaceVolume::from_slices(&next_slices).unwrap();
        let alpha = 0.25;
        ema.accumulate(&next, alpha).unwrap();
        let retain = 1.0 - alpha;
        let mut expected = Vec::new();
        for (current, incoming) in front_a
            .iter()
            .chain(back_a.iter())
            .zip(front_b.iter().chain(back_b.iter()))
        {
            expected.push((current * retain) + (incoming * alpha));
        }
        for (observed, anticipated) in ema.voxels().iter().zip(expected.iter()) {
            assert!((observed - anticipated).abs() < 1e-6);
        }
        let blended = ZSpaceVolume::from_slices(&base_slices)
            .unwrap()
            .accumulated(&next, alpha)
            .unwrap();
        assert_eq!(blended.voxels(), ema.voxels());
        let mismatched = ZSpaceVolume::from_slices(&[tensor_from(&front_a, 2, 2)]).unwrap();
        assert!(ema.accumulate(&mismatched, alpha).is_err());
        assert!(ema.accumulated(&next, 1.5).is_err());
    }

    #[test]
    fn temporal_buffer_applies_decay_and_handles_resets() {
        let mut buffer = TemporalResonanceBuffer::new(0.5);
        let first = vec![0.2, 0.8];
        let fused_first = buffer.apply(&first).unwrap();
        for (a, b) in fused_first.iter().zip(first.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(buffer.frames_accumulated(), 1);

        let second = vec![0.6, 0.4];
        let fused_second = buffer.apply(&second).unwrap();
        assert!((fused_second[0] - 0.4).abs() < 1e-6);
        assert!((fused_second[1] - 0.6).abs() < 1e-6);
        assert_eq!(buffer.frames_accumulated(), 2);

        let third = vec![0.1, 0.2, 0.7];
        let fused_third = buffer.apply(&third).unwrap();
        assert_eq!(fused_third.len(), 3);
        assert_eq!(buffer.frames_accumulated(), 1);
        assert!(fused_third
            .iter()
            .zip(third.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6));

        buffer.clear();
        assert!(buffer.history().is_none());
        let invalid = vec![f32::NAN, 0.5];
        assert!(buffer.apply(&invalid).is_err());
    }

    #[test]
    fn projector_temporal_projection_matches_manual_smoothing() {
        let slices = vec![
            tensor_from(&[1.0, 0.0, 0.0, 1.0], 2, 2),
            tensor_from(&[0.5, 1.0, 1.5, 2.0], 2, 2),
        ];
        let volume = ZSpaceVolume::from_slices(&slices).unwrap();
        let resonance_a = toy_resonance(2);
        let mut resonance_b = toy_resonance(2);
        resonance_b.infinity_energy = tensor_from(&[6.0, 0.2], 1, 2);
        resonance_b.recursive_objective = tensor_from(&[0.5, -0.8], 1, 2);
        let projector =
            VisionProjector::new(0.9, 0.6, 0.0).with_spectral_window(SpectralWindow::rectangular());

        let mut buffer_weights = TemporalResonanceBuffer::new(0.25);
        let first_temporal = projector
            .depth_weights_with_temporal(&volume, &resonance_a, &mut buffer_weights)
            .unwrap();
        let second_direct = projector.depth_weights(&volume, &resonance_b).unwrap();
        let expected: Vec<f32> = first_temporal
            .data()
            .iter()
            .zip(second_direct.data().iter())
            .map(|(prev, next)| prev * 0.75 + next * 0.25)
            .collect();
        let second_temporal = projector
            .depth_weights_with_temporal(&volume, &resonance_b, &mut buffer_weights)
            .unwrap();
        for (observed, anticipated) in second_temporal.data().iter().zip(expected.iter()) {
            assert!((observed - anticipated).abs() < 1e-6);
        }

        let mut buffer_projection = TemporalResonanceBuffer::new(0.25);
        projector
            .project_with_temporal(&volume, &resonance_a, &mut buffer_projection)
            .unwrap();
        let second_projection = projector
            .project_with_temporal(&volume, &resonance_b, &mut buffer_projection)
            .unwrap();
        let manual = volume.collapse_with_weights(&expected).unwrap();
        assert_eq!(manual.shape(), second_projection.shape());
        for (lhs, rhs) in second_projection.data().iter().zip(manual.data().iter()) {
            assert!((lhs - rhs).abs() < 1e-6);
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

    #[test]
    fn slice_profile_matches_manual_stats() {
        let slices = vec![
            tensor_from(&[1.0, 3.0], 1, 2),
            tensor_from(&[2.0, 4.0], 1, 2),
        ];
        let volume = ZSpaceVolume::from_slices(&slices).unwrap();
        let profile = volume.slice_profile().unwrap();
        assert_eq!(profile.depth(), 2);
        let expected_mean0 = (1.0 + 3.0) / 2.0;
        assert!((profile.mean(0) - expected_mean0).abs() < 1e-6);
        let variance0 =
            (((1.0 - expected_mean0).powi(2) + (3.0 - expected_mean0).powi(2)) / 2.0).max(0.0);
        assert!((profile.std(0) - variance0.sqrt()).abs() < 1e-6);
        let energy1 = (2.0f32.powi(2) + 4.0f32.powi(2)) / 2.0;
        assert!((profile.energy(1) - energy1).abs() < 1e-6);
    }

    #[test]
    fn resonance_generator_produces_projectable_resonance() {
        let slices = vec![
            tensor_from(&[0.1, 0.2, 0.3, 0.4], 2, 2),
            tensor_from(&[0.5, 0.4, 0.3, 0.2], 2, 2),
            tensor_from(&[0.9, 0.8, 0.7, 0.6], 2, 2),
        ];
        let volume = ZSpaceVolume::from_slices(&slices).unwrap();
        let mut generator = ResonanceGenerator::new("loop", 8, 3).unwrap();
        let projector =
            VisionProjector::new(0.6, 0.4, 0.2).with_spectral_window(SpectralWindow::hann());
        let chrono = toy_summary();
        let mut atlas = AtlasFrame::new(0.2);
        atlas.z_signal = Some(0.75);
        atlas.suggested_pressure = Some(48.0);
        atlas.collapse_total = Some(1.1);

        let resonance = generator
            .generate(&volume, &projector, Some(&chrono), Some(&atlas), None)
            .unwrap();
        assert_eq!(resonance.infinity_energy.shape(), (1, volume.depth()));
        let weights = projector.depth_weights(&volume, &resonance).unwrap();
        assert_eq!(weights.data().len(), volume.depth());
        assert!((weights.data().iter().sum::<f32>() - 1.0).abs() < 1e-3);
        let projection = projector.project(&volume, &resonance).unwrap();
        assert_eq!(projection.shape(), (volume.height(), volume.width()));
    }

    #[test]
    fn resonance_generator_respects_feedback_history() {
        let slices = vec![
            tensor_from(&[0.2, 0.1, 0.0, 0.3], 2, 2),
            tensor_from(&[0.4, 0.6, 0.8, 1.0], 2, 2),
            tensor_from(&[0.9, 0.7, 0.5, 0.3], 2, 2),
        ];
        let volume = ZSpaceVolume::from_slices(&slices).unwrap();
        let mut generator = ResonanceGenerator::new("loop-feedback", 10, 3).unwrap();
        let projector = VisionProjector::new(0.45, 0.5, -0.2);
        let chrono = toy_summary();
        let atlas = AtlasFrame::new(0.4);

        let first = generator
            .generate(&volume, &projector, Some(&chrono), Some(&atlas), None)
            .unwrap();
        let second = generator
            .generate(
                &volume,
                &projector,
                Some(&chrono),
                Some(&atlas),
                Some(&first),
            )
            .unwrap();
        let first_energy = first.infinity_energy.data()[0];
        let second_energy = second.infinity_energy.data()[0];
        assert!((second_energy - first_energy).abs() > 1e-6);
    }

    #[test]
    fn volume_interpolation_and_upscale_refine_geometry() {
        let slices = vec![
            tensor_from(&[0.0, 1.0, 2.0, 3.0], 2, 2),
            tensor_from(&[4.0, 5.0, 6.0, 7.0], 2, 2),
        ];
        let volume = ZSpaceVolume::from_slices(&slices).unwrap();
        let interpolated = volume.interpolate(InterpolationMethod::Linear).unwrap();
        assert_eq!(interpolated.depth(), volume.depth() * 2 - 1);
        let mid_slice = interpolated.slice(1).unwrap();
        let expected: Vec<f32> = slices[0]
            .data()
            .iter()
            .zip(slices[1].data().iter())
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        assert_eq!(mid_slice.data(), expected.as_slice());

        let upscaled = interpolated.upscale(2).unwrap();
        assert_eq!(upscaled.height(), interpolated.height() * 2);
        assert_eq!(upscaled.width(), interpolated.width() * 2);
        let top_left = upscaled.slice(0).unwrap().data()[0];
        assert!((top_left - slices[0].data()[0]).abs() < 1e-6);
    }

    #[test]
    fn diffuser_softens_sharp_transitions() {
        let slices = vec![
            tensor_from(&[10.0, 0.0, 0.0, 0.0], 2, 2),
            tensor_from(&[0.0, 0.0, 0.0, 0.0], 2, 2),
        ];
        let volume = ZSpaceVolume::from_slices(&slices).unwrap();
        let diffuser = ZDiffuser::new(1, 0.5);
        let diffused = diffuser.diffuse(&volume).unwrap();
        assert!(diffused.voxels()[0] < 10.0);
        assert!(diffused.voxels()[1] > 0.0);
    }

    #[test]
    fn decoder_is_deterministic_for_seed() {
        let latent = tensor_from(&[0.3, -0.1, 0.5], 1, 3);
        let mut decoder_a = ZDecoder::new(2, 2, 2, 7).unwrap();
        let mut decoder_b = ZDecoder::new(2, 2, 2, 7).unwrap();
        let volume_a = decoder_a.decode(&latent).unwrap();
        let volume_b = decoder_b.decode(&latent).unwrap();
        assert_eq!(volume_a.voxels(), volume_b.voxels());
    }

    #[test]
    fn video_stream_projector_handles_sequences() {
        let frame_a = vec![
            tensor_from(&[0.2, 0.4, 0.6, 0.8], 2, 2),
            tensor_from(&[0.1, 0.3, 0.5, 0.7], 2, 2),
            tensor_from(&[0.0, 0.2, 0.4, 0.6], 2, 2),
        ];
        let frame_b = vec![
            tensor_from(&[0.5, 0.3, 0.1, -0.1], 2, 2),
            tensor_from(&[0.6, 0.4, 0.2, 0.0], 2, 2),
            tensor_from(&[0.7, 0.5, 0.3, 0.1], 2, 2),
        ];
        let volume_a = ZSpaceVolume::from_slices(&frame_a).unwrap();
        let volume_b = ZSpaceVolume::from_slices(&frame_b).unwrap();
        let projector = VisionProjector::new(0.5, 0.4, 0.1);
        let generator = ResonanceGenerator::new("video", 12, 5).unwrap();
        let mut stream = VideoStreamProjector::new(projector, generator, 0.3)
            .with_diffuser(ZDiffuser::new(1, 0.25))
            .with_super_resolution(InterpolationMethod::Linear, 2);
        let chrono_frames = vec![Some(toy_summary()), Some(toy_summary())];
        let mut atlas_first = AtlasFrame::new(0.1);
        atlas_first.z_signal = Some(0.6);
        let mut atlas_second = AtlasFrame::new(0.2);
        atlas_second.z_signal = Some(0.8);
        atlas_second.suggested_pressure = Some(40.0);
        let atlas_frames = vec![Some(atlas_first), Some(atlas_second)];
        let projections = stream
            .project_sequence(
                &[volume_a.clone(), volume_b.clone()],
                &chrono_frames,
                &atlas_frames,
            )
            .unwrap();
        assert_eq!(projections.len(), 2);
        assert_eq!(
            projections[0].shape(),
            (volume_a.height() * 2, volume_a.width() * 2)
        );
        assert!(stream.last_resonance().is_some());
    }

    #[test]
    fn multi_view_biases_respect_orientation_and_baseline() {
        let descriptors = vec![
            ViewDescriptor::new("front", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ViewDescriptor::new("right", [0.0, 0.0, 0.0], [1.0, 0.0, 0.1])
                .with_baseline_weight(0.5),
            ViewDescriptor::new("sky", [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]).with_baseline_weight(2.0),
        ];
        let fusion = MultiViewFusion::new(descriptors)
            .unwrap()
            .with_focus_direction([0.1, 0.2, 1.0])
            .with_alignment_gamma(2.0);
        let slices = vec![
            tensor_from(&[1.0, 1.0, 1.0, 1.0], 2, 2),
            tensor_from(&[0.5, 0.5, 0.5, 0.5], 2, 2),
            tensor_from(&[0.25, 0.25, 0.25, 0.25], 2, 2),
        ];
        let volume = ZSpaceVolume::from_slices(&slices).unwrap();
        let resonance = toy_resonance(3);
        let projector = VisionProjector::default();
        let baseline = projector.depth_weights(&volume, &resonance).unwrap();
        let weights = projector
            .depth_weights_multi_view(&fusion, &volume, &resonance)
            .unwrap();
        let base = baseline.data();
        let data = weights.data();
        assert_eq!(data.len(), 3);
        assert!((data.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(data[2] > base[2]);
        assert!(data[1] < base[1]);
    }

    #[test]
    fn projector_multi_view_temporal_projection_matches_manual_smoothing() {
        let descriptors = vec![
            ViewDescriptor::new("front", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ViewDescriptor::new("right", [0.0, 0.0, 0.0], [1.0, 0.0, 0.5])
                .with_baseline_weight(1.5),
        ];
        let fusion = MultiViewFusion::new(descriptors)
            .unwrap()
            .with_focus_direction([0.4, 0.0, 1.0])
            .with_alignment_gamma(1.5);
        let slices = vec![
            tensor_from(&[1.0, 0.0, 0.0, 1.0], 2, 2),
            tensor_from(&[0.0, 1.0, 1.0, 0.0], 2, 2),
        ];
        let volume = ZSpaceVolume::from_slices(&slices).unwrap();
        let resonance_a = toy_resonance(2);
        let mut resonance_b = toy_resonance(2);
        resonance_b.infinity_energy = tensor_from(&[3.0, 6.0], 1, 2);
        resonance_b.recursive_objective = tensor_from(&[0.1, 0.9], 1, 2);
        let projector = VisionProjector::default();

        let mut buffer_weights = TemporalResonanceBuffer::new(0.3);
        let first_temporal = projector
            .depth_weights_multi_view_with_temporal(
                &fusion,
                &volume,
                &resonance_a,
                &mut buffer_weights,
            )
            .unwrap();
        let second_direct = projector
            .depth_weights_multi_view(&fusion, &volume, &resonance_b)
            .unwrap();
        let expected: Vec<f32> = first_temporal
            .data()
            .iter()
            .zip(second_direct.data().iter())
            .map(|(prev, next)| prev * 0.7 + next * 0.3)
            .collect();
        let second_temporal = projector
            .depth_weights_multi_view_with_temporal(
                &fusion,
                &volume,
                &resonance_b,
                &mut buffer_weights,
            )
            .unwrap();
        for (observed, anticipated) in second_temporal.data().iter().zip(expected.iter()) {
            assert!((observed - anticipated).abs() < 1e-6);
        }

        let mut buffer_projection = TemporalResonanceBuffer::new(0.3);
        projector
            .project_multi_view_with_temporal(
                &fusion,
                &volume,
                &resonance_a,
                &mut buffer_projection,
            )
            .unwrap();
        let second_projection = projector
            .project_multi_view_with_temporal(
                &fusion,
                &volume,
                &resonance_b,
                &mut buffer_projection,
            )
            .unwrap();
        let manual = volume.collapse_with_weights(&expected).unwrap();
        assert_eq!(manual.shape(), second_projection.shape());
        for (lhs, rhs) in second_projection.data().iter().zip(manual.data().iter()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn image_tensor_volume_roundtrip() {
        let image = ImageTensor::new(
            3,
            2,
            2,
            vec![
                0.0, 1.0, 2.0, 3.0, // channel 0
                4.0, 5.0, 6.0, 7.0, // channel 1
                8.0, 9.0, 10.0, 11.0, // channel 2
            ],
        )
        .unwrap();
        let volume = ZSpaceVolume::from_image_tensor(&image).unwrap();
        let restored = volume.to_image_tensor().unwrap();
        assert_eq!(image, restored);
        let tensor = image.into_tensor().unwrap();
        let rebuilt = ImageTensor::from_tensor(&tensor, 3, 2, 2).unwrap();
        assert_eq!(rebuilt.shape(), (3, 2, 2));
    }

    #[test]
    fn transform_pipeline_matches_torchvision_basics() {
        let mut pipeline = standard_classification_pipeline(64, Some(7)).unwrap();
        let mut image = ImageTensor::new(3, 96, 96, vec![0.5; 3 * 96 * 96]).unwrap();
        pipeline.apply(&mut image).unwrap();
        assert_eq!(image.shape(), (3, 64, 64));
        let data = image.as_slice();
        let area = 64 * 64;
        let expected = [
            (0.5 - 0.485) / 0.229,
            (0.5 - 0.456) / 0.224,
            (0.5 - 0.406) / 0.225,
        ];
        for (channel, &value) in expected.iter().enumerate() {
            let idx = channel * area;
            assert!((data[idx] - value).abs() < 1e-3);
        }
    }

    #[test]
    fn dataloader_produces_batches() {
        let descriptor = dataset_catalog()[0].clone();
        let mut dataset = TensorVisionDataset::new(descriptor);
        for idx in 0..8 {
            let image = ImageTensor::new(3, 64, 64, vec![idx as f32; 3 * 64 * 64]).unwrap();
            dataset
                .push_sample(DatasetSample::new(image).with_label(format!("label-{idx}")))
                .unwrap();
        }
        let dataset = Arc::new(dataset);
        let mut loader = DataLoader::new(dataset, 3, Some(42)).unwrap();
        loader.enable_shuffle(true);
        let mut seen = 0;
        while let Some(batch) = loader.next_batch().unwrap() {
            assert!(!batch.is_empty());
            let tensor = batch.stack().unwrap();
            assert_eq!(tensor.shape().0, batch.len());
            seen += batch.len();
        }
        assert_eq!(seen, 8);
    }

    #[test]
    fn simple_cnn_forward_produces_logits() {
        let model = create_classification_model(ModelKind::ResNet18, 10, Some(99)).unwrap();
        let metadata = model.metadata().clone();
        let image = ImageTensor::new(
            metadata.input_channels,
            metadata.image_size.0,
            metadata.image_size.1,
            vec![0.25; metadata.input_channels * metadata.image_size.0 * metadata.image_size.1],
        )
        .unwrap();
        let logits = model.forward(&[image.clone()]).unwrap();
        assert_eq!(logits.shape(), (1, metadata.num_classes));
        let extractor = FeatureExtractor::new(model.clone(), FeatureStage::Stem);
        let stem = extractor.extract(&image).unwrap();
        assert_eq!(
            stem.shape().1,
            (metadata.image_size.0 / 2) * (metadata.image_size.1 / 2)
        );
    }
}
