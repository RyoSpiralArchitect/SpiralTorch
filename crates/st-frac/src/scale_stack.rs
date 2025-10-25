// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Scale-linked interface persistence for microlocal ⇄ macrolocal bridging.
//!
//! This module implements a *scale stack* that tracks how often interfaces are
//! observed at increasing probe radii.  It turns the conceptual recipe from the
//! microlocal continuum notes into executable Rust code:
//!
//! 1. For each scale `s`, probe every lattice point with a metric ball of radius
//!    `s` and flag the location if the ball cuts through an interface.
//! 2. Average the flags over the lattice to obtain the monotone gate curve
//!    `\bar{R}_s`.
//! 3. Differentiate the gate curve to obtain a discrete persistence measure that
//!    describes how much interface density is contributed by each scale band.
//!
//! The resulting `ScaleStack` exposes both micro-scale summaries (interface
//! density, fractal dimension estimator) and macro-scale aggregates (moments of
//! the persistence measure corresponding to Hadwiger-style invariants).

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use thiserror::Error;

/// Errors produced while constructing a [`ScaleStack`].
#[derive(Debug, Error)]
pub enum ScaleStackError {
    /// The list of scales was empty.
    #[error("scale stack requires at least one scale")]
    EmptyScales,
    /// A scale radius was non-positive or not strictly increasing.
    #[error("scales must be strictly increasing and positive")]
    InvalidScales,
    /// The interface detection tolerance must be non-negative.
    #[error("threshold must be non-negative")]
    InvalidThreshold,
}

/// Aggregate statistics at a single probing scale.
#[derive(Clone, Debug)]
pub struct ScaleSample {
    /// Probe radius (in lattice units).
    pub scale: f64,
    /// Average of the local interface gate over the lattice.
    pub gate_mean: f64,
}

/// A discrete approximation to the persistent interface measure `\mu(ds)`.
#[derive(Clone, Debug)]
pub struct PersistenceBin {
    /// Lower bound of the contributing scale band.
    pub scale_low: f64,
    /// Upper bound of the contributing scale band.
    pub scale_high: f64,
    /// Integrated mass contributed by this band.
    pub mass: f64,
}

/// Microlocal ⇄ macrolocal bridge for binary interface fields.
#[derive(Clone, Debug)]
pub struct ScaleStack {
    threshold: f32,
    samples: Vec<ScaleSample>,
}

impl ScaleStack {
    /// Construct a [`ScaleStack`] from a scalar field and a list of probe radii.
    ///
    /// * `field` – scalar observable sampled on a regular lattice (any
    ///   dimensionality supported by [`ndarray`]).
    /// * `scales` – strictly increasing probe radii (in lattice units).
    /// * `threshold` – minimal contrast required to treat two points as being on
    ///   different sides of an interface.
    pub fn from_scalar_field(
        field: ArrayViewD<'_, f32>,
        scales: &[f64],
        threshold: f32,
    ) -> Result<Self, ScaleStackError> {
        if scales.is_empty() {
            return Err(ScaleStackError::EmptyScales);
        }
        if threshold < 0.0 {
            return Err(ScaleStackError::InvalidThreshold);
        }
        if !is_strictly_increasing_positive(scales) {
            return Err(ScaleStackError::InvalidScales);
        }

        let mut samples = Vec::with_capacity(scales.len());
        let mut prev_gate = 0.0f64;
        for &scale in scales {
            let radius = (scale.ceil() as usize).max(1);
            let gate = detect_interfaces(&field, radius, threshold);
            let active = gate.iter().filter(|&&flag| flag).count() as f64;
            let total = gate.len() as f64;
            let mean = if total.abs() < f64::EPSILON {
                0.0
            } else {
                active / total
            };
            let gate_mean = mean.max(prev_gate);
            samples.push(ScaleSample { scale, gate_mean });
            prev_gate = gate_mean;
        }

        Ok(Self { threshold, samples })
    }

    /// Access the raw scale samples (monotone gate curve).
    pub fn samples(&self) -> &[ScaleSample] {
        &self.samples
    }

    /// Compute the discrete persistence measure `\mu(ds)` via finite
    /// differencing of the gate curve.
    pub fn persistence_measure(&self) -> Vec<PersistenceBin> {
        let mut bins = Vec::new();
        let mut prev_scale = 0.0f64;
        let mut prev_gate = 0.0f64;
        for sample in &self.samples {
            let delta = (sample.gate_mean - prev_gate).max(0.0);
            if delta > 0.0 {
                bins.push(PersistenceBin {
                    scale_low: prev_scale,
                    scale_high: sample.scale,
                    mass: delta,
                });
            }
            prev_scale = sample.scale;
            prev_gate = sample.gate_mean;
        }
        bins
    }

    /// Estimate the interface density (Hadwiger first coefficient) from the
    /// smallest scale band.
    pub fn interface_density(&self) -> Option<f64> {
        let first = self.persistence_measure().into_iter().next()?;
        let span = (first.scale_high - first.scale_low).max(f64::EPSILON);
        Some(first.mass / span)
    }

    /// Compute the `k`-th raw moment of the persistence measure.
    pub fn moment(&self, order: u32) -> f64 {
        let exponent = order as i32;
        self.persistence_measure()
            .into_iter()
            .map(|bin| {
                let centroid = if bin.scale_high > bin.scale_low {
                    0.5 * (bin.scale_high + bin.scale_low)
                } else {
                    bin.scale_high
                };
                bin.mass * centroid.powi(exponent)
            })
            .sum()
    }

    /// Estimate the boundary (Hausdorff) dimension by fitting the small-scale
    /// slope of the gate curve in log-log space.
    ///
    /// `ambient_dim` is the embedding dimension `d`.  The estimator uses the
    /// first `window` samples (minimum of two) to form a least-squares fit.
    pub fn estimate_boundary_dimension(&self, ambient_dim: f64, window: usize) -> Option<f64> {
        if self.samples.len() < 2 || window < 2 {
            return None;
        }
        let end = window.min(self.samples.len());
        let mut xs = Vec::with_capacity(end);
        let mut ys = Vec::with_capacity(end);
        for sample in &self.samples[..end] {
            if sample.gate_mean <= 0.0 {
                continue;
            }
            if sample.gate_mean >= 1.0 {
                if xs.len() >= 2 {
                    break;
                }
                let clipped = (1.0 - f64::EPSILON).max(f64::EPSILON);
                xs.push((sample.scale.max(f64::EPSILON)).ln());
                ys.push(clipped.ln());
                continue;
            }
            xs.push((sample.scale.max(f64::EPSILON)).ln());
            ys.push((sample.gate_mean.max(f64::EPSILON)).ln());
        }
        if xs.len() < 2 {
            return None;
        }
        let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
        let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (&x, &y) in xs.iter().zip(&ys) {
            let dx = x - mean_x;
            let dy = y - mean_y;
            num += dx * dy;
            den += dx * dx;
        }
        if den <= f64::EPSILON {
            return None;
        }
        let slope = num / den;
        Some(ambient_dim - slope)
    }

    /// Interface detection threshold used while constructing the stack.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

fn is_strictly_increasing_positive(scales: &[f64]) -> bool {
    if scales.is_empty() {
        return false;
    }
    let mut prev = 0.0f64;
    for &s in scales {
        if s <= 0.0 || s <= prev {
            return false;
        }
        prev = s;
    }
    true
}

fn detect_interfaces(field: &ArrayViewD<'_, f32>, radius: usize, threshold: f32) -> ArrayD<bool> {
    let ndim = field.ndim();
    if ndim == 0 {
        return ArrayD::<bool>::from_elem(IxDyn(&[1]), false);
    }
    let offsets = ball_offsets(ndim, radius);
    let mut out = ArrayD::<bool>::from_elem(field.raw_dim(), false);
    for (idx, flag) in out.indexed_iter_mut() {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for offset in &offsets {
            let mut neigh = Vec::with_capacity(ndim);
            let mut valid = true;
            for (axis, (&delta, &dim_len)) in offset.iter().zip(field.shape().iter()).enumerate() {
                let coord = idx[axis] as isize;
                let p = coord + delta;
                if p < 0 || p >= dim_len as isize {
                    valid = false;
                    break;
                }
                neigh.push(p as usize);
            }
            if !valid {
                continue;
            }
            let value = field[IxDyn(&neigh)];
            min_val = min_val.min(value);
            max_val = max_val.max(value);
            if max_val - min_val > threshold {
                *flag = true;
                break;
            }
        }
    }
    out
}

fn ball_offsets(ndim: usize, radius: usize) -> Vec<Vec<isize>> {
    let mut offsets = Vec::new();
    let mut current = vec![0isize; ndim];
    enumerate_offsets_recursive(0, radius as isize, &mut current, &mut offsets);
    offsets
}

fn enumerate_offsets_recursive(
    dim: usize,
    radius: isize,
    current: &mut [isize],
    out: &mut Vec<Vec<isize>>,
) {
    if dim == current.len() {
        let norm_sq: isize = current.iter().map(|x| x * x).sum();
        if norm_sq <= radius * radius {
            out.push(current.to_vec());
        }
        return;
    }
    for delta in -radius..=radius {
        current[dim] = delta;
        enumerate_offsets_recursive(dim + 1, radius, current, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn persistence_mass_matches_gate_limit() {
        let field = array![
            [0.0f32, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ]
        .into_dyn();
        let stack = ScaleStack::from_scalar_field(field.view(), &[1.0, 2.0, 3.0], 0.01).unwrap();
        let last_gate = stack.samples().last().unwrap().gate_mean;
        let total_mass: f64 = stack
            .persistence_measure()
            .into_iter()
            .map(|bin| bin.mass)
            .sum();
        assert_relative_eq!(last_gate, total_mass, epsilon = 1e-6);
    }

    #[test]
    fn boundary_dimension_recovers_planar_interface() {
        let field = array![
            [0.0f32, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ]
        .into_dyn();
        let stack = ScaleStack::from_scalar_field(field.view(), &[1.0, 2.0, 3.0], 0.01).unwrap();
        let dim = stack.estimate_boundary_dimension(2.0, 3).unwrap();
        assert_relative_eq!(dim, 1.0, epsilon = 0.25);
    }

    #[test]
    fn invalid_scales_rejected() {
        let field = ArrayD::<f32>::zeros(IxDyn(&[2, 2]));
        let err = ScaleStack::from_scalar_field(field.view(), &[1.0, 0.5], 0.01).unwrap_err();
        assert!(matches!(err, ScaleStackError::InvalidScales));
    }
}
