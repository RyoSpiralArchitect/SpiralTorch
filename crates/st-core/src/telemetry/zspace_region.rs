// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Helpers to classify Softlogic Z-space feedback into coarse regions so
//! consumers can react to spin/radius buckets without duplicating parsing
//! logic.

use crate::telemetry::hub::SoftlogicEllipticSample;

/// Coarse spin buckets extracted from [`SoftlogicEllipticSample`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZSpaceSpinBand {
    /// Strongly aligned with the leading sheet (positive spin).
    Leading,
    /// Near-neutral alignment.
    Neutral,
    /// Strongly aligned with the trailing sheet (negative spin).
    Trailing,
}

impl ZSpaceSpinBand {
    /// Ordered enumeration of all supported spin bands.
    pub const fn values() -> [Self; 3] {
        [Self::Leading, Self::Neutral, Self::Trailing]
    }

    /// Classifies a raw spin alignment into a discrete band.
    pub fn classify(spin_alignment: f32) -> Self {
        if spin_alignment > 0.33 {
            ZSpaceSpinBand::Leading
        } else if spin_alignment < -0.33 {
            ZSpaceSpinBand::Trailing
        } else {
            ZSpaceSpinBand::Neutral
        }
    }

    /// Returns a short diagnostic label for the band.
    pub fn label(&self) -> &'static str {
        match self {
            ZSpaceSpinBand::Leading => "spin.leading",
            ZSpaceSpinBand::Neutral => "spin.neutral",
            ZSpaceSpinBand::Trailing => "spin.trailing",
        }
    }
}

/// Coarse radius buckets derived from the normalised geodesic radius.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZSpaceRadiusBand {
    /// Interior region near the pulse core.
    Core,
    /// Intermediate band covering the mantle.
    Mantle,
    /// Outer edge of the pulse.
    Edge,
}

impl ZSpaceRadiusBand {
    /// Ordered enumeration of all supported radius bands.
    pub const fn values() -> [Self; 3] {
        [Self::Core, Self::Mantle, Self::Edge]
    }

    /// Classifies the normalised radius into a band.
    pub fn classify(normalized_radius: f32) -> Self {
        if normalized_radius < 0.33 {
            ZSpaceRadiusBand::Core
        } else if normalized_radius < 0.66 {
            ZSpaceRadiusBand::Mantle
        } else {
            ZSpaceRadiusBand::Edge
        }
    }

    /// Returns a short diagnostic label for the band.
    pub fn label(&self) -> &'static str {
        match self {
            ZSpaceRadiusBand::Core => "radius.core",
            ZSpaceRadiusBand::Mantle => "radius.mantle",
            ZSpaceRadiusBand::Edge => "radius.edge",
        }
    }
}

/// Discrete key describing a Z-space region bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ZSpaceRegionKey {
    /// Spin band component of the key.
    pub spin: ZSpaceSpinBand,
    /// Radius band component of the key.
    pub radius: ZSpaceRadiusBand,
}

impl ZSpaceRegionKey {
    /// Builds a key directly from the supplied bands.
    pub const fn new(spin: ZSpaceSpinBand, radius: ZSpaceRadiusBand) -> Self {
        Self { spin, radius }
    }

    /// Returns a composite label combining both bands.
    pub fn label(&self) -> String {
        format!("{}.{}", self.spin.label(), self.radius.label())
    }
}

/// Parsed region descriptor surfaced from a Softlogic elliptic sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZSpaceRegionDescriptor {
    /// Raw spin alignment in \([-1, 1]\).
    pub spin_alignment: f32,
    /// Normalised geodesic radius in \([0, 1]\).
    pub normalized_radius: f32,
    /// Curvature radius associated with the pulse.
    pub curvature_radius: f32,
    /// Geodesic radius prior to normalisation.
    pub geodesic_radius: f32,
    /// Sheet index reported by the telemetry sample.
    pub sheet_index: u32,
    /// Total number of sheets for the telemetry sample.
    pub sheet_count: u32,
    /// Topological sector identifier.
    pub topological_sector: u32,
}

impl ZSpaceRegionDescriptor {
    /// Builds a descriptor from the provided elliptic sample.
    pub fn from_elliptic(sample: &SoftlogicEllipticSample) -> Self {
        Self {
            spin_alignment: sample.spin_alignment.clamp(-1.0, 1.0),
            normalized_radius: sample.normalized_radius.clamp(0.0, 1.0),
            curvature_radius: sample.curvature_radius.max(0.0),
            geodesic_radius: sample.geodesic_radius.max(0.0),
            sheet_index: sample.sheet_index,
            sheet_count: sample.sheet_count,
            topological_sector: sample.topological_sector,
        }
    }

    /// Returns the region key derived from the descriptor.
    pub fn key(&self) -> ZSpaceRegionKey {
        ZSpaceRegionKey::new(
            ZSpaceSpinBand::classify(self.spin_alignment),
            ZSpaceRadiusBand::classify(self.normalized_radius),
        )
    }
}

impl From<&SoftlogicEllipticSample> for ZSpaceRegionDescriptor {
    fn from(sample: &SoftlogicEllipticSample) -> Self {
        Self::from_elliptic(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(spin: f32, radius: f32) -> SoftlogicEllipticSample {
        SoftlogicEllipticSample {
            curvature_radius: 1.0,
            geodesic_radius: radius,
            normalized_radius: radius,
            spin_alignment: spin,
            sheet_index: 2,
            sheet_position: 0.0,
            normal_bias: 0.0,
            sheet_count: 4,
            topological_sector: 3,
            homology_index: 0,
            rotor_field: [0.0; 3],
            flow_vector: [0.0; 3],
            curvature_tensor: [[0.0; 3]; 3],
            resonance_heat: 0.0,
            noise_density: 0.0,
            quaternion: [0.0; 4],
            rotation: [0.0; 9],
        }
    }

    #[test]
    fn spin_band_classification_tracks_alignment() {
        assert_eq!(ZSpaceSpinBand::classify(0.8), ZSpaceSpinBand::Leading);
        assert_eq!(ZSpaceSpinBand::classify(-0.9), ZSpaceSpinBand::Trailing);
        assert_eq!(ZSpaceSpinBand::classify(0.1), ZSpaceSpinBand::Neutral);
        assert_eq!(
            ZSpaceSpinBand::values(),
            [
                ZSpaceSpinBand::Leading,
                ZSpaceSpinBand::Neutral,
                ZSpaceSpinBand::Trailing,
            ]
        );
    }

    #[test]
    fn radius_band_classification_tracks_radius() {
        assert_eq!(ZSpaceRadiusBand::classify(0.1), ZSpaceRadiusBand::Core);
        assert_eq!(ZSpaceRadiusBand::classify(0.5), ZSpaceRadiusBand::Mantle);
        assert_eq!(ZSpaceRadiusBand::classify(0.9), ZSpaceRadiusBand::Edge);
        assert_eq!(
            ZSpaceRadiusBand::values(),
            [
                ZSpaceRadiusBand::Core,
                ZSpaceRadiusBand::Mantle,
                ZSpaceRadiusBand::Edge,
            ]
        );
    }

    #[test]
    fn descriptor_produces_region_key() {
        let descriptor = ZSpaceRegionDescriptor::from_elliptic(&sample(0.7, 0.2));
        let key = descriptor.key();
        assert_eq!(key.spin, ZSpaceSpinBand::Leading);
        assert_eq!(key.radius, ZSpaceRadiusBand::Core);
    }
}
