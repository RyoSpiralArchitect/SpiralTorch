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

/// Distance metrics used for semantic interface detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticMetric {
    /// Euclidean distance in feature space.
    Euclidean,
    /// Cosine distance (1 - cosine similarity) in feature space.
    Cosine,
}

impl SemanticMetric {
    fn distance(self, a: &[f32], b: &[f32]) -> f64 {
        match self {
            SemanticMetric::Euclidean => {
                let mut sum = 0.0f64;
                for (&lhs, &rhs) in a.iter().zip(b.iter()) {
                    let diff = (lhs - rhs) as f64;
                    sum += diff * diff;
                }
                sum.sqrt()
            }
            SemanticMetric::Cosine => {
                let mut dot = 0.0f64;
                let mut norm_a = 0.0f64;
                let mut norm_b = 0.0f64;
                for (&lhs, &rhs) in a.iter().zip(b.iter()) {
                    let lhs = lhs as f64;
                    let rhs = rhs as f64;
                    dot += lhs * rhs;
                    norm_a += lhs * lhs;
                    norm_b += rhs * rhs;
                }
                if norm_a <= f64::EPSILON || norm_b <= f64::EPSILON {
                    return 0.0;
                }
                let denom = norm_a.sqrt() * norm_b.sqrt();
                let cos = (dot / denom).clamp(-1.0, 1.0);
                1.0 - cos
            }
        }
    }
}

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
    /// The feature axis used for semantic fields was out of bounds.
    #[error("feature axis {axis} out of bounds for field with {field_ndim} axes")]
    InvalidFeatureAxis { axis: usize, field_ndim: usize },
    /// Semantic interface detection requires at least one feature component.
    #[error("semantic fields require a non-empty feature axis")]
    EmptyFeatureAxis,
    /// Provided tensor could not be reshaped into the requested layout.
    #[error("field shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
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

/// How the interface detector interpreted the source field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterfaceMode {
    /// Scalar thresholding on a single-valued field.
    Scalar,
    /// Semantic thresholding over vector-valued embeddings.
    Semantic {
        /// Axis containing the embedding channels.
        feature_axis: usize,
        /// Metric used to compare feature vectors.
        metric: SemanticMetric,
    },
}

/// Microlocal ⇄ macrolocal bridge for binary interface fields.
#[derive(Clone, Debug)]
pub struct ScaleStack {
    threshold: f32,
    samples: Vec<ScaleSample>,
    mode: InterfaceMode,
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

        Ok(Self {
            threshold,
            samples,
            mode: InterfaceMode::Scalar,
        })
    }

    /// Construct a [`ScaleStack`] from a semantic embedding field.
    ///
    /// The provided `field` is interpreted as a tensor where one axis stores
    /// feature channels (selected by `feature_axis`) and the remaining axes
    /// describe the spatial neighbourhood that is probed across the supplied
    /// `scales`.
    pub fn from_semantic_field(
        field: ArrayViewD<'_, f32>,
        scales: &[f64],
        threshold: f32,
        feature_axis: usize,
        metric: SemanticMetric,
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
        let ndim = field.ndim();
        if feature_axis >= ndim {
            return Err(ScaleStackError::InvalidFeatureAxis {
                axis: feature_axis,
                field_ndim: ndim,
            });
        }
        let feature_dim = field.shape()[feature_axis];
        if feature_dim == 0 {
            return Err(ScaleStackError::EmptyFeatureAxis);
        }

        let mut samples = Vec::with_capacity(scales.len());
        let mut prev_gate = 0.0f64;
        for &scale in scales {
            let radius = (scale.ceil() as usize).max(1);
            let gate = detect_semantic_interfaces(&field, radius, threshold, feature_axis, metric);
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

        Ok(Self {
            threshold,
            samples,
            mode: InterfaceMode::Semantic {
                feature_axis,
                metric,
            },
        })
    }

    /// Access the raw scale samples (monotone gate curve).
    pub fn samples(&self) -> &[ScaleSample] {
        &self.samples
    }

    /// Returns how the source field was interpreted by the detector.
    pub fn mode(&self) -> &InterfaceMode {
        &self.mode
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

    /// Returns the first scale where the gate curve crosses `level`.
    pub fn coherence_break_scale(&self, level: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&level) {
            return None;
        }
        self.samples
            .iter()
            .find(|sample| sample.gate_mean >= level)
            .map(|sample| sample.scale)
    }

    /// Returns the breakpoints for each requested coherence level.
    pub fn coherence_profile(&self, levels: &[f64]) -> Vec<Option<f64>> {
        levels
            .iter()
            .map(|&level| self.coherence_break_scale(level))
            .collect()
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

fn detect_semantic_interfaces(
    field: &ArrayViewD<'_, f32>,
    radius: usize,
    threshold: f32,
    feature_axis: usize,
    metric: SemanticMetric,
) -> ArrayD<bool> {
    let ndim = field.ndim();
    let feature_dim = field.shape()[feature_axis];
    let spatial_axes: Vec<usize> = (0..ndim).filter(|&axis| axis != feature_axis).collect();
    let spatial_shape: Vec<usize> = spatial_axes
        .iter()
        .map(|&axis| field.shape()[axis])
        .collect();
    let mut out = ArrayD::<bool>::from_elem(IxDyn(&spatial_shape), false);
    if feature_dim == 0 {
        return out;
    }
    if spatial_shape.contains(&0) {
        return out;
    }
    let offsets = ball_offsets(spatial_axes.len(), radius);
    let mut index = vec![0usize; spatial_axes.len()];
    let mut center_coords = vec![0usize; ndim];
    let mut neighbor_coords = vec![0usize; ndim];
    let mut center_vec = vec![0f32; feature_dim];
    let mut neighbor_vec = vec![0f32; feature_dim];

    loop {
        for (slot, &axis) in spatial_axes.iter().enumerate() {
            center_coords[axis] = index.get(slot).copied().unwrap_or(0);
        }

        for (feature_index, slot) in center_vec.iter_mut().enumerate() {
            center_coords[feature_axis] = feature_index;
            *slot = field[IxDyn(&center_coords)];
        }

        let mut triggered = false;
        for offset in &offsets {
            if offset.iter().all(|&delta| delta == 0) {
                continue;
            }
            neighbor_coords.copy_from_slice(&center_coords);
            let mut valid = true;
            for (slot, &axis) in spatial_axes.iter().enumerate() {
                let delta = offset.get(slot).copied().unwrap_or(0);
                let coord = center_coords[axis] as isize + delta;
                if coord < 0 || coord >= field.shape()[axis] as isize {
                    valid = false;
                    break;
                }
                neighbor_coords[axis] = coord as usize;
            }
            if !valid {
                continue;
            }
            for (feature_index, slot) in neighbor_vec.iter_mut().enumerate() {
                neighbor_coords[feature_axis] = feature_index;
                *slot = field[IxDyn(&neighbor_coords)];
            }
            if metric.distance(&center_vec, &neighbor_vec) > threshold as f64 {
                triggered = true;
                break;
            }
        }

        if let Some(flag) = out.get_mut(IxDyn(&index)) {
            *flag = triggered;
        }

        if spatial_axes.is_empty() {
            break;
        }

        let mut axis = spatial_axes.len();
        loop {
            if axis == 0 {
                return out;
            }
            axis -= 1;
            index[axis] += 1;
            if index[axis] < spatial_shape[axis] {
                break;
            }
            index[axis] = 0;
            if axis == 0 {
                return out;
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

    #[test]
    fn semantic_stack_marks_interface_transition() {
        let field = array![[0.0f32, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]].into_dyn();
        let stack = ScaleStack::from_semantic_field(
            field.view(),
            &[1.0, 2.0],
            0.25,
            1,
            SemanticMetric::Euclidean,
        )
        .unwrap();
        assert_eq!(
            stack.mode(),
            &InterfaceMode::Semantic {
                feature_axis: 1,
                metric: SemanticMetric::Euclidean
            }
        );
        assert!(stack.samples()[0].gate_mean > 0.0);
        assert!(stack.samples()[1].gate_mean >= stack.samples()[0].gate_mean);
    }

    #[test]
    fn semantic_stack_supports_cosine_metric() {
        let field = array![[1.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]].into_dyn();
        let stack = ScaleStack::from_semantic_field(
            field.view(),
            &[1.0, 2.0, 3.0],
            0.1,
            1,
            SemanticMetric::Cosine,
        )
        .unwrap();
        assert!(stack.samples().iter().any(|sample| sample.gate_mean > 0.0));
    }

    #[test]
    fn coherence_break_scale_reports_threshold() {
        let field = array![[0.0f32, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]].into_dyn();
        let stack = ScaleStack::from_semantic_field(
            field.view(),
            &[1.0, 2.0, 3.0],
            0.25,
            1,
            SemanticMetric::Euclidean,
        )
        .unwrap();
        let breakpoint = stack.coherence_break_scale(0.5);
        assert!(breakpoint.is_some());
        let profile = stack.coherence_profile(&[0.25, 0.5, 0.75]);
        assert_eq!(profile.len(), 3);
    }

    #[test]
    fn semantic_field_with_invalid_axis_rejected() {
        let field = ArrayD::<f32>::zeros(IxDyn(&[2, 2]));
        let err = ScaleStack::from_semantic_field(
            field.view(),
            &[1.0, 2.0],
            0.1,
            2,
            SemanticMetric::Euclidean,
        )
        .unwrap_err();
        assert!(matches!(err, ScaleStackError::InvalidFeatureAxis { .. }));
    }
}
