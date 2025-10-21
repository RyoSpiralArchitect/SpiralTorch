// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Utilities for warping temporal axes so Z-space telemetry can be replayed under
//! different subjective clocks.

use core::fmt;

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
