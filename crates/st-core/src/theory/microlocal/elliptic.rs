// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::f32::consts::{PI, TAU};

use crate::telemetry::hub::SoftlogicEllipticSample;

/// Positive-curvature warp that remaps microlocal orientations onto an elliptic
/// Z-frame.
#[derive(Clone, Debug)]
pub struct EllipticWarp {
    curvature_radius: f32,
    sheet_count: usize,
    spin_harmonics: usize,
}

impl EllipticWarp {
    /// Creates a warp anchored to the provided curvature radius.
    pub fn new(curvature_radius: f32) -> Self {
        let radius = curvature_radius.max(1e-6);
        Self {
            curvature_radius: radius,
            sheet_count: 2,
            spin_harmonics: 1,
        }
    }

    /// Configures the number of discrete sheets representing the χ axis.
    pub fn with_sheet_count(mut self, sheet_count: usize) -> Self {
        self.sheet_count = sheet_count.max(1);
        self
    }

    /// Configures the number of spin harmonics applied while computing ν.
    pub fn with_spin_harmonics(mut self, harmonics: usize) -> Self {
        self.spin_harmonics = harmonics.max(1);
        self
    }

    /// Returns the curvature radius associated with the warp.
    pub fn curvature_radius(&self) -> f32 {
        self.curvature_radius
    }

    /// Returns the number of χ sheets encoded by the warp.
    pub fn sheet_count(&self) -> usize {
        self.sheet_count
    }

    /// Returns the number of spin harmonics applied to the ν axis.
    pub fn spin_harmonics(&self) -> usize {
        self.spin_harmonics
    }

    /// Maximum geodesic radius reachable on the warp.
    pub fn max_geodesic(&self) -> f32 {
        self.curvature_radius * PI
    }

    /// Maps an orientation vector to elliptic telemetry describing the warped
    /// coordinates. Returns `None` when the orientation is degenerate.
    pub fn map_orientation(&self, orientation: &[f32]) -> Option<EllipticTelemetry> {
        if orientation.is_empty() {
            return None;
        }

        let mut dir = [0.0f32; 3];
        for (dst, &value) in dir.iter_mut().zip(orientation.iter()).take(3) {
            *dst = value;
        }
        let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if !norm.is_finite() || norm <= 1e-6 {
            return None;
        }
        for component in dir.iter_mut() {
            *component /= norm;
        }

        let polar = dir[2].clamp(-1.0, 1.0).acos();
        let geodesic_radius = polar * self.curvature_radius;
        let azimuth = dir[1].atan2(dir[0]);
        let mut spin_alignment = (azimuth / PI).clamp(-1.0, 1.0);
        if self.spin_harmonics > 1 {
            spin_alignment = (spin_alignment * self.spin_harmonics as f32).sin();
        }

        let normalized = ((azimuth + PI) / TAU).rem_euclid(1.0);
        let sheet_f = normalized * self.sheet_count as f32;
        let mut sheet_index = sheet_f.floor() as usize;
        if sheet_index >= self.sheet_count {
            sheet_index = self.sheet_count - 1;
        }
        let sheet_position = if self.sheet_count <= 1 {
            0.0
        } else {
            (sheet_f / self.sheet_count as f32).clamp(0.0, 1.0)
        };

        Some(EllipticTelemetry {
            curvature_radius: self.curvature_radius,
            geodesic_radius,
            spin_alignment,
            sheet_index,
            sheet_position,
            normal_bias: dir[2].clamp(-1.0, 1.0),
            sheet_count: self.sheet_count,
        })
    }
}

/// Telemetry describing an elliptic Z-frame projection.
#[derive(Clone, Debug, Default)]
pub struct EllipticTelemetry {
    pub curvature_radius: f32,
    pub geodesic_radius: f32,
    pub spin_alignment: f32,
    pub sheet_index: usize,
    pub sheet_position: f32,
    pub normal_bias: f32,
    pub sheet_count: usize,
}

impl EllipticTelemetry {
    /// Normalised geodesic radius within \([0, 1]\).
    pub fn normalized_radius(&self) -> f32 {
        if self.curvature_radius <= 0.0 {
            0.0
        } else {
            (self.geodesic_radius / (self.curvature_radius * PI)).clamp(0.0, 1.0)
        }
    }

    /// Interpolates two telemetry samples.
    pub fn lerp(&self, other: &EllipticTelemetry, t: f32) -> EllipticTelemetry {
        let clamped = t.clamp(0.0, 1.0);
        let sheet_count = self.sheet_count.max(other.sheet_count).max(1);
        let sheet_position =
            (super::lerp(self.sheet_position, other.sheet_position, clamped)).clamp(0.0, 1.0);
        let sheet_index = ((sheet_position * sheet_count as f32).round() as usize)
            .min(sheet_count.saturating_sub(1));
        EllipticTelemetry {
            curvature_radius: super::lerp(self.curvature_radius, other.curvature_radius, clamped)
                .max(1e-6),
            geodesic_radius: super::lerp(self.geodesic_radius, other.geodesic_radius, clamped)
                .max(0.0),
            spin_alignment: super::lerp(self.spin_alignment, other.spin_alignment, clamped)
                .clamp(-1.0, 1.0),
            sheet_index,
            sheet_position,
            normal_bias: super::lerp(self.normal_bias, other.normal_bias, clamped).clamp(-1.0, 1.0),
            sheet_count,
        }
    }

    /// Returns event tags that summarise the elliptic telemetry.
    pub fn event_tags(&self) -> [String; 3] {
        [
            format!("elliptic.sheet:{:02}", self.sheet_index),
            format!("elliptic.radius:{:.4}", self.normalized_radius()),
            format!("elliptic.spin:{:.3}", self.spin_alignment),
        ]
    }
}

impl From<&EllipticTelemetry> for SoftlogicEllipticSample {
    fn from(telemetry: &EllipticTelemetry) -> Self {
        SoftlogicEllipticSample {
            curvature_radius: telemetry.curvature_radius,
            geodesic_radius: telemetry.geodesic_radius,
            normalized_radius: telemetry.normalized_radius(),
            spin_alignment: telemetry.spin_alignment,
            sheet_index: telemetry.sheet_index as u32,
            sheet_position: telemetry.sheet_position,
            normal_bias: telemetry.normal_bias,
            sheet_count: telemetry.sheet_count as u32,
        }
    }
}

#[derive(Default)]
pub(crate) struct EllipticAccumulator {
    curvature_sum: f32,
    radius_sum: f32,
    bias_sum: f32,
    spin_sum: f32,
    sheet_sum: f32,
    weight: f32,
    sheet_count: usize,
}

impl EllipticAccumulator {
    pub(crate) fn accumulate(&mut self, telemetry: &EllipticTelemetry, weight: f32) {
        if !weight.is_finite() || weight <= 0.0 {
            return;
        }
        self.curvature_sum += telemetry.curvature_radius * weight;
        self.radius_sum += telemetry.geodesic_radius * weight;
        self.bias_sum += telemetry.normal_bias * weight;
        self.spin_sum += telemetry.spin_alignment * weight;
        self.sheet_sum += telemetry.sheet_position * weight;
        self.weight += weight;
        if telemetry.sheet_count > self.sheet_count {
            self.sheet_count = telemetry.sheet_count;
        }
    }

    pub(crate) fn finish(self) -> Option<EllipticTelemetry> {
        if self.weight <= 0.0 {
            return None;
        }
        let sheet_count = self.sheet_count.max(1);
        let sheet_position = (self.sheet_sum / self.weight).clamp(0.0, 1.0);
        let sheet_index = ((sheet_position * sheet_count as f32).round() as usize)
            .min(sheet_count.saturating_sub(1));
        Some(EllipticTelemetry {
            curvature_radius: (self.curvature_sum / self.weight).max(1e-6),
            geodesic_radius: (self.radius_sum / self.weight).max(0.0),
            spin_alignment: (self.spin_sum / self.weight).clamp(-1.0, 1.0),
            sheet_index,
            sheet_position,
            normal_bias: (self.bias_sum / self.weight).clamp(-1.0, 1.0),
            sheet_count,
        })
    }
}
