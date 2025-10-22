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
/// values are not finite, and [`TemporalWarpError::Degenerate`] when a
/// non-identity dilation is requested for a zero-span axis.
pub fn warp_axis_in_place(axis: &mut [f32], warp: TemporalWarp) -> Result<(), TemporalWarpError> {
    if axis.is_empty() {
        return Err(TemporalWarpError::Empty);
    }

    warp.validate()?;

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in axis.iter().copied() {
        if !value.is_finite() {
            return Err(TemporalWarpError::NonFinite);
        }
        min = min.min(value);
        max = max.max(value);
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
    fn warp_axis_detects_degenerate_dilation() {
        let mut axis = [1.0, 1.0];
        let warp = TemporalWarp::dilation(2.0);
        let err = warp_axis_in_place(&mut axis, warp).unwrap_err();
        assert_eq!(err, TemporalWarpError::Degenerate);
    }

    #[test]
    fn warped_axis_returns_new_buffer() {
        let axis = [0.0, 1.0];
        let warp = TemporalWarp::translation(2.0);
        let warped = warped_axis(&axis, warp).unwrap();
        assert_eq!(warped, vec![2.0, 3.0]);
        assert_eq!(axis, [0.0, 1.0]);
    }
}
