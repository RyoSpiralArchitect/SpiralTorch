// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Utilities for warping temporal axes so Z-space telemetry can be replayed under
//! different subjective clocks.

use core::fmt;

/// Applies a [`TemporalWarp`] to a collection of timestamps in-place.
///
/// # Errors
///
/// Returns [`TemporalWarpError::Empty`] when the axis has no samples,
/// [`TemporalWarpError::NonFinite`] when either the warp parameters or input
/// values are not finite, [`TemporalWarpError::NonMonotonic`] when timestamps
/// are not in ascending order, and [`TemporalWarpError::Degenerate`] when a
/// non-identity dilation is requested for a zero-span axis.
pub fn warp_axis_in_place(axis: &mut [f32], warp: TemporalWarp) -> Result<(), TemporalWarpError> {
    if axis.is_empty() {
        return Err(TemporalWarpError::Empty);
    }

    warp.validate()?;

    let mut iter = axis.iter();
    let first = *iter.next().unwrap();
    if !first.is_finite() {
        return Err(TemporalWarpError::NonFinite);
    }
    if !warp.apply(first).is_finite() {
        return Err(TemporalWarpError::NonFinite);
    }

    let mut min = first;
    let mut max = first;
    let mut previous = first;
    for &value in iter {
        if !value.is_finite() {
            return Err(TemporalWarpError::NonFinite);
        }
        if value < previous {
            return Err(TemporalWarpError::NonMonotonic);
        }
        min = min.min(value);
        max = max.max(value);
        let warped = warp.apply(value);
        if !warped.is_finite() {
            return Err(TemporalWarpError::NonFinite);
        }
        previous = value;
    }

    if (max - min).abs() <= f32::EPSILON && warp.scale != 1.0 {
        return Err(TemporalWarpError::Degenerate);
    }

    for value in axis.iter_mut() {
        *value = warp.apply(*value);
    }

    Ok(())
}

/// Returns a warped copy of the provided axis, leaving the input untouched.
///
/// This is a convenience wrapper around [`warp_axis_in_place`] that first
/// clones the provided slice into an owned [`Vec`].
pub fn warped_axis(axis: &[f32], warp: TemporalWarp) -> Result<Vec<f32>, TemporalWarpError> {
    let mut warped = axis.to_vec();
    warp_axis_in_place(&mut warped, warp)?;
    Ok(warped)
}

/// Affine transform applied to a time axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalWarp {
    /// Multiplicative dilation applied around the pivot.
    pub scale: f32,
    /// Additive shift applied after the dilation.
    pub offset: f32,
    /// Pivot used when dilating the axis.
    pub pivot: f32,
}

impl TemporalWarp {
    /// Attempts to build a warp with the provided components, validating the result.
    pub fn try_new(scale: f32, offset: f32, pivot: f32) -> Result<Self, TemporalWarpError> {
        let warp = Self {
            scale,
            offset,
            pivot,
        };
        warp.validate()?;
        Ok(warp)
    }

    /// Returns the identity warp that leaves timestamps untouched.
    pub const fn identity() -> Self {
        Self {
            scale: 1.0,
            offset: 0.0,
            pivot: 0.0,
        }
    }

    /// Creates a warp that only scales timestamps around the origin.
    pub const fn dilation(scale: f32) -> Self {
        Self {
            scale,
            offset: 0.0,
            pivot: 0.0,
        }
    }

    /// Creates a warp that only shifts timestamps by the provided offset.
    pub const fn translation(offset: f32) -> Self {
        Self {
            scale: 1.0,
            offset,
            pivot: 0.0,
        }
    }

    /// Creates a warp that dilates around an arbitrary pivot without translation.
    pub const fn about(pivot: f32, scale: f32) -> Self {
        Self {
            scale,
            offset: 0.0,
            pivot,
        }
    }

    /// Applies the warp to a timestamp.
    #[inline]
    pub fn apply(&self, t: f32) -> f32 {
        ((t - self.pivot) * self.scale) + self.pivot + self.offset
    }

    /// Ensures the warp parameters are finite and well defined.
    pub fn validate(&self) -> Result<(), TemporalWarpError> {
        if !self.scale.is_finite() {
            return Err(TemporalWarpError::NonFinite);
        }
        if self.scale <= 0.0 {
            return Err(TemporalWarpError::InvalidScale(self.scale));
        }
        if !self.offset.is_finite() || !self.pivot.is_finite() {
            return Err(TemporalWarpError::NonFinite);
        }
        Ok(())
    }

    /// Returns the affine bias component equivalent to this warp.
    #[inline]
    fn bias(&self) -> f32 {
        self.offset + self.pivot * (1.0 - self.scale)
    }

    /// Composes this warp with another, applying `other` first and then `self`.
    pub fn compose(self, other: TemporalWarp) -> Self {
        let scale = self.scale * other.scale;
        let bias = self.scale * other.bias() + self.bias();
        let pivot = self.pivot;
        let offset = bias - pivot * (1.0 - scale);
        Self {
            scale,
            offset,
            pivot,
        }
    }

    /// Returns the inverse warp which reverts this transform.
    pub fn inverse(&self) -> Result<Self, TemporalWarpError> {
        self.validate()?;

        let inv_scale = 1.0 / self.scale;
        if !inv_scale.is_finite() {
            return Err(TemporalWarpError::InvalidScale(inv_scale));
        }
        let inv_offset = -self.offset * inv_scale;
        Self::try_new(inv_scale, inv_offset, self.pivot)
    }

    /// Expresses the same transform around a different pivot.
    pub fn with_pivot(&self, pivot: f32) -> Result<Self, TemporalWarpError> {
        if !pivot.is_finite() {
            return Err(TemporalWarpError::NonFinite);
        }
        let bias = self.bias();
        let offset = bias - pivot * (1.0 - self.scale);
        Self::try_new(self.scale, offset, pivot)
    }

    /// Builds a warp that maps one span of time onto another.
    ///
    /// The resulting warp sends `source_start` to `target_start` and
    /// `source_end` to `target_end`. Both spans must be finite and the source
    /// span must have positive extent unless both spans collapse to a single
    /// instant, in which case a pure translation is returned.
    pub fn fit_span(
        source_start: f32,
        source_end: f32,
        target_start: f32,
        target_end: f32,
    ) -> Result<Self, TemporalWarpError> {
        if !source_start.is_finite()
            || !source_end.is_finite()
            || !target_start.is_finite()
            || !target_end.is_finite()
        {
            return Err(TemporalWarpError::NonFinite);
        }

        let source_span = source_end - source_start;
        let target_span = target_end - target_start;

        if source_span.abs() <= f32::EPSILON {
            if target_span.abs() <= f32::EPSILON {
                return Self::try_new(1.0, target_start - source_start, 0.0);
            }
            return Err(TemporalWarpError::Degenerate);
        }

        let scale = target_span / source_span;
        if scale <= 0.0 {
            return Err(TemporalWarpError::InvalidScale(scale));
        }
        let offset = target_start - source_start * scale;
        Self::try_new(scale, offset, 0.0)
    }
}

impl Default for TemporalWarp {
    fn default() -> Self {
        Self::identity()
    }
}

/// Error emitted when a temporal warp cannot be applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TemporalWarpError {
    /// Axis cannot be warped because it is empty.
    Empty,
    /// Encountered a non-finite parameter while attempting to warp.
    NonFinite,
    /// Requested dilation is invalid (non-positive).
    InvalidScale(f32),
    /// Axis span is zero so dilation would collapse timestamps.
    Degenerate,
    /// Axis samples are not monotonically increasing.
    NonMonotonic,
}

impl fmt::Display for TemporalWarpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "temporal axis is empty"),
            Self::NonFinite => write!(f, "temporal warp parameters must be finite"),
            Self::InvalidScale(scale) => {
                write!(f, "temporal warp scale must be > 0, got {scale}")
            }
            Self::Degenerate => write!(f, "temporal axis has zero span"),
            Self::NonMonotonic => write!(f, "temporal axis must be monotonically increasing"),
        }
    }
}

impl std::error::Error for TemporalWarpError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_axis_in_place_applies_transform() {
        let mut axis = [0.0, 1.0, 2.0, 3.0];
        let warp = TemporalWarp {
            scale: 2.0,
            offset: 1.0,
            pivot: 1.0,
        };
        warp_axis_in_place(&mut axis, warp).unwrap();
        assert_eq!(axis, [0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn warp_axis_rejects_non_finite_values() {
        let mut axis = [0.0, f32::NAN];
        let warp = TemporalWarp::identity();
        let err = warp_axis_in_place(&mut axis, warp).unwrap_err();
        assert_eq!(err, TemporalWarpError::NonFinite);
    }

    #[test]
    fn warp_axis_rejects_empty_axis() {
        let mut axis: [f32; 0] = [];
        let err = warp_axis_in_place(&mut axis, TemporalWarp::identity()).unwrap_err();
        assert_eq!(err, TemporalWarpError::Empty);
    }

    #[test]
    fn warp_axis_detects_degenerate_dilation() {
        let mut axis = [1.0, 1.0];
        let warp = TemporalWarp::dilation(2.0);
        let err = warp_axis_in_place(&mut axis, warp).unwrap_err();
        assert_eq!(err, TemporalWarpError::Degenerate);
    }

    #[test]
    fn warp_axis_rejects_non_finite_outputs() {
        let original = [f32::MAX, 0.0];
        let mut axis = original;
        let warp = TemporalWarp::translation(f32::MAX);
        let err = warp_axis_in_place(&mut axis, warp).unwrap_err();
        assert_eq!(err, TemporalWarpError::NonFinite);
        assert_eq!(axis, original);
    }

    #[test]
    fn warp_axis_rejects_non_monotonic_axis() {
        let mut axis = [0.0, 2.0, 1.0];
        let warp = TemporalWarp::identity();
        let err = warp_axis_in_place(&mut axis, warp).unwrap_err();
        assert_eq!(err, TemporalWarpError::NonMonotonic);
    }

    #[test]
    fn warped_axis_returns_new_buffer() {
        let axis = [0.0, 1.0];
        let warp = TemporalWarp::translation(2.0);
        let warped = warped_axis(&axis, warp).unwrap();
        assert_eq!(warped, vec![2.0, 3.0]);
        assert_eq!(axis, [0.0, 1.0]);
    }

    #[test]
    fn try_new_validates_inputs() {
        let warp = TemporalWarp::try_new(1.5, 2.0, -3.0).expect("warp");
        assert!(
            (warp.apply(4.0)
                - TemporalWarp {
                    scale: 1.5,
                    offset: 2.0,
                    pivot: -3.0
                }
                .apply(4.0))
            .abs()
                < 1e-6
        );

        let err = TemporalWarp::try_new(-0.25, 0.0, 0.0).unwrap_err();
        assert_eq!(err, TemporalWarpError::InvalidScale(-0.25));
    }

    #[test]
    fn compose_applies_other_then_self() {
        let dilation = TemporalWarp::dilation(2.0);
        let translation = TemporalWarp::translation(-1.0);
        let composed = translation.compose(dilation);
        composed.validate().expect("composed warp");

        for sample in [-3.0, 0.5, 7.25] {
            let sequential = translation.apply(dilation.apply(sample));
            let fused = composed.apply(sample);
            assert!((sequential - fused).abs() < 1e-6);
        }
    }

    #[test]
    fn inverse_roundtrips_samples() {
        let warp = TemporalWarp::try_new(2.5, -3.0, 1.5).expect("warp");
        let inverse = warp.inverse().expect("inverse");
        for sample in [-10.0, -0.25, 0.0, 2.0, 17.5] {
            let warped = warp.apply(sample);
            let restored = inverse.apply(warped);
            assert!((restored - sample).abs() < 1e-5);
        }
    }

    #[test]
    fn with_pivot_preserves_transform() {
        let warp = TemporalWarp::try_new(1.25, -0.75, -3.0).expect("warp");
        let rebased = warp.with_pivot(5.5).expect("rebase");
        for sample in [-6.0, -1.0, 0.0, 2.0, 4.0] {
            let original = warp.apply(sample);
            let rebased_value = rebased.apply(sample);
            assert!((original - rebased_value).abs() < 1e-5);
        }
    }

    #[test]
    fn fit_span_maps_endpoints() {
        let warp = TemporalWarp::fit_span(2.0, 6.0, 0.0, 20.0).expect("fit");
        assert!((warp.apply(2.0) - 0.0).abs() < 1e-6);
        assert!((warp.apply(6.0) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn fit_span_handles_collapsed_spans() {
        let warp = TemporalWarp::fit_span(5.0, 5.0, 3.0, 3.0).expect("fit");
        assert!((warp.apply(5.0) - 3.0).abs() < 1e-6);
        assert_eq!(warp.scale, 1.0);
    }

    #[test]
    fn fit_span_rejects_negative_scale() {
        let err = TemporalWarp::fit_span(0.0, 10.0, 5.0, -5.0).unwrap_err();
        assert_eq!(err, TemporalWarpError::InvalidScale(-1.0));
    }
}
