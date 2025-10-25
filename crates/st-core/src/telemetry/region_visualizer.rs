// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use serde_json::{json, Value};
use std::collections::HashMap;

use crate::telemetry::xai_report::{AttributionMetadata, AttributionReport};
use crate::telemetry::zspace_region::{
    ZSpaceRadiusBand, ZSpaceRegionDescriptor, ZSpaceRegionKey, ZSpaceSpinBand,
};

const SPIN_ORDER: [ZSpaceSpinBand; 3] = ZSpaceSpinBand::values();
const RADIUS_ORDER: [ZSpaceRadiusBand; 3] = ZSpaceRadiusBand::values();

fn spin_index(band: ZSpaceSpinBand) -> usize {
    match band {
        ZSpaceSpinBand::Leading => 0,
        ZSpaceSpinBand::Neutral => 1,
        ZSpaceSpinBand::Trailing => 2,
    }
}

fn radius_index(band: ZSpaceRadiusBand) -> usize {
    match band {
        ZSpaceRadiusBand::Core => 0,
        ZSpaceRadiusBand::Mantle => 1,
        ZSpaceRadiusBand::Edge => 2,
    }
}

/// Cell describing the composed weight for a single Z-space region bucket.
#[derive(Debug, Clone)]
pub struct RegionHeatmapCell {
    /// Region key identifying the spin/radius bucket.
    pub key: ZSpaceRegionKey,
    /// Static multiplier configured for the region.
    pub base_weight: f32,
    /// Dynamic multiplier inferred by the adaptive policy.
    pub adaptive_multiplier: f32,
    /// Combined multiplier that is currently applied.
    pub combined_weight: f32,
    /// Number of observations used to derive the adaptive multiplier.
    pub samples: u32,
    /// Smoothed loss estimate tracked for the region.
    pub ema_loss: Option<f32>,
}

impl RegionHeatmapCell {
    /// Builds a new cell, clamping all multipliers to non-negative values.
    pub fn new(
        key: ZSpaceRegionKey,
        base_weight: f32,
        adaptive_multiplier: f32,
        samples: u32,
        ema_loss: Option<f32>,
    ) -> Self {
        let base = base_weight.max(0.0);
        let adaptive = adaptive_multiplier.max(0.0);
        Self {
            key,
            base_weight: base,
            adaptive_multiplier: adaptive,
            combined_weight: base * adaptive,
            samples,
            ema_loss,
        }
    }
}

/// Snapshot describing all region multipliers with optional highlight metadata.
#[derive(Debug, Clone)]
pub struct RegionHeatmapSnapshot {
    default_weight: f32,
    highlight: Option<ZSpaceRegionDescriptor>,
    global_loss_ema: Option<f32>,
    condition_min_spin: Option<f32>,
    condition_min_radius: Option<f32>,
    cells: HashMap<ZSpaceRegionKey, RegionHeatmapCell>,
}

impl RegionHeatmapSnapshot {
    /// Creates an empty snapshot using the supplied default weight as baseline.
    pub fn new(default_weight: f32) -> Self {
        Self {
            default_weight: default_weight.max(0.0),
            highlight: None,
            global_loss_ema: None,
            condition_min_spin: None,
            condition_min_radius: None,
            cells: HashMap::new(),
        }
    }

    /// Annotates the snapshot with the most recent region descriptor.
    pub fn with_highlight(mut self, highlight: Option<ZSpaceRegionDescriptor>) -> Self {
        self.highlight = highlight;
        self
    }

    /// Records the global loss EMA tracked by the adaptive policy.
    pub fn with_global_loss(mut self, global: Option<f32>) -> Self {
        self.global_loss_ema = global;
        self
    }

    /// Records the region gating thresholds enforced by the trainer.
    pub fn with_condition(mut self, min_spin: Option<f32>, min_radius: Option<f32>) -> Self {
        self.condition_min_spin = min_spin;
        self.condition_min_radius = min_radius;
        self
    }

    /// Inserts a new cell into the snapshot, overwriting any previous entry.
    pub fn insert_cell(&mut self, cell: RegionHeatmapCell) {
        self.cells.insert(cell.key, cell);
    }

    fn cell(&self, key: ZSpaceRegionKey) -> &RegionHeatmapCell {
        self.cells
            .get(&key)
            .expect("all region keys must be present in the snapshot")
    }

    /// Converts the snapshot into an attribution report consumable by frontends.
    pub fn into_report(self) -> AttributionReport {
        let mut metadata = AttributionMetadata::for_algorithm("zspace-region-weights");
        metadata.insert_extra_text(
            "spin_axis",
            SPIN_ORDER
                .iter()
                .map(|band| band.label())
                .collect::<Vec<_>>()
                .join(","),
        );
        metadata.insert_extra_text(
            "radius_axis",
            RADIUS_ORDER
                .iter()
                .map(|band| band.label())
                .collect::<Vec<_>>()
                .join(","),
        );
        metadata.insert_extra_number("default_weight", self.default_weight as f64);
        if let Some(global) = self.global_loss_ema {
            metadata.insert_extra_number("global_loss_ema", global as f64);
        }
        if let Some(threshold) = self.condition_min_spin {
            metadata.insert_extra_number("condition_min_spin", threshold as f64);
        }
        if let Some(threshold) = self.condition_min_radius {
            metadata.insert_extra_number("condition_min_radius", threshold as f64);
        }

        let mut cells_meta: Vec<Value> = Vec::with_capacity(SPIN_ORDER.len() * RADIUS_ORDER.len());
        let mut values: Vec<f32> = Vec::with_capacity(SPIN_ORDER.len() * RADIUS_ORDER.len());
        for spin in SPIN_ORDER.iter().copied() {
            for radius in RADIUS_ORDER.iter().copied() {
                let key = ZSpaceRegionKey::new(spin, radius);
                let cell = self.cell(key);
                values.push(cell.combined_weight);
                cells_meta.push(json!({
                    "key": cell.key.label(),
                    "spin": cell.key.spin.label(),
                    "radius": cell.key.radius.label(),
                    "base_weight": cell.base_weight,
                    "adaptive_multiplier": cell.adaptive_multiplier,
                    "combined_weight": cell.combined_weight,
                    "samples": cell.samples,
                    "ema_loss": cell.ema_loss,
                }));
            }
        }
        metadata.insert_extra("cells", Value::Array(cells_meta));

        if let Some(descriptor) = self.highlight {
            let key = descriptor.key();
            metadata.insert_extra(
                "highlight",
                json!({
                    "key": key.label(),
                    "spin": key.spin.label(),
                    "radius": key.radius.label(),
                    "spin_alignment": descriptor.spin_alignment,
                    "normalized_radius": descriptor.normalized_radius,
                    "curvature_radius": descriptor.curvature_radius,
                    "geodesic_radius": descriptor.geodesic_radius,
                    "sheet_index": descriptor.sheet_index,
                    "sheet_count": descriptor.sheet_count,
                    "topological_sector": descriptor.topological_sector,
                }),
            );
            metadata.insert_extra_number("highlight_row", spin_index(key.spin) as f64);
            metadata.insert_extra_number("highlight_col", radius_index(key.radius) as f64);
        }

        AttributionReport::new(metadata, SPIN_ORDER.len(), RADIUS_ORDER.len(), values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_produces_report_with_expected_shape() {
        let descriptor = ZSpaceRegionDescriptor {
            spin_alignment: 0.9,
            normalized_radius: 0.75,
            curvature_radius: 1.5,
            geodesic_radius: 0.5,
            sheet_index: 2,
            sheet_count: 4,
            topological_sector: 1,
        };

        let mut snapshot = RegionHeatmapSnapshot::new(1.0)
            .with_highlight(Some(descriptor))
            .with_global_loss(Some(0.8))
            .with_condition(Some(0.2), Some(0.3));
        for spin in SPIN_ORDER.iter().copied() {
            for radius in RADIUS_ORDER.iter().copied() {
                let key = ZSpaceRegionKey::new(spin, radius);
                snapshot.insert_cell(RegionHeatmapCell::new(key, 1.0, 1.0, 3, Some(0.5)));
            }
        }

        let report = snapshot.into_report();
        assert_eq!(report.shape(), (3, 3));
        assert_eq!(report.values.len(), 9);
        assert_eq!(report.metadata.algorithm, "zspace-region-weights");
        assert!(report.metadata.extras.contains_key("cells"));
        assert!(report.metadata.extras.contains_key("highlight"));
        let spin_threshold = report
            .metadata
            .extras
            .get("condition_min_spin")
            .and_then(|value| value.as_f64())
            .unwrap();
        assert!((spin_threshold - 0.2).abs() < 1e-6);
    }
}
