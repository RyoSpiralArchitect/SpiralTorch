// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};

use crate::telemetry::xai_report::{AttributionMetadata, AttributionReport};
use crate::telemetry::zspace_region::{
    ZSpaceRadiusBand, ZSpaceRegionDescriptor, ZSpaceRegionKey, ZSpaceSpinBand,
};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

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

    /// Returns the baseline multiplier applied when no overrides are configured.
    pub fn default_weight(&self) -> f32 {
        self.default_weight
    }

    /// Returns the latest global loss EMA recorded by the adaptive policy.
    pub fn global_loss(&self) -> Option<f32> {
        self.global_loss_ema
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
        let mut total_samples = 0u64;
        let mut total_combined_weight = 0.0f32;
        let mut max_combined_weight = 0.0f32;
        let has_highlight = self.highlight.is_some();
        let highlight_key = self
            .highlight
            .map(|descriptor| descriptor.key().label())
            .unwrap_or_default();
        for spin in SPIN_ORDER.iter().copied() {
            for radius in RADIUS_ORDER.iter().copied() {
                let key = ZSpaceRegionKey::new(spin, radius);
                let cell = self.cell(key);
                values.push(cell.combined_weight);
                total_samples = total_samples.saturating_add(cell.samples as u64);
                total_combined_weight += cell.combined_weight;
                max_combined_weight = max_combined_weight.max(cell.combined_weight);
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

        emit_tensor_op(
            "zspace_region_heatmap_report",
            &[values.len()],
            &[SPIN_ORDER.len(), RADIUS_ORDER.len()],
        );
        emit_tensor_op_meta("zspace_region_heatmap_report", || {
            serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "kind": "st_core_zspace_region_heatmap_report",
                "cells": values.len(),
                "rows": SPIN_ORDER.len(),
                "cols": RADIUS_ORDER.len(),
                "default_weight": self.default_weight,
                "global_loss_ema": self.global_loss_ema.unwrap_or(0.0),
                "has_global_loss_ema": self.global_loss_ema.is_some(),
                "has_highlight": has_highlight,
                "highlight_key": highlight_key,
                "has_condition_min_spin": self.condition_min_spin.is_some(),
                "condition_min_spin": self.condition_min_spin.unwrap_or(0.0),
                "has_condition_min_radius": self.condition_min_radius.is_some(),
                "condition_min_radius": self.condition_min_radius.unwrap_or(0.0),
                "total_samples": total_samples,
                "mean_combined_weight": if values.is_empty() {
                    0.0
                } else {
                    total_combined_weight / values.len() as f32
                },
                "max_combined_weight": max_combined_weight,
            })
        });
        AttributionReport::new(metadata, SPIN_ORDER.len(), RADIUS_ORDER.len(), values)
    }
}

#[derive(Clone, Debug)]
struct RegionSnapshotEntry {
    step: u64,
    snapshot: RegionHeatmapSnapshot,
}

/// Rolling window of region heatmap snapshots that can surface temporal trends.
#[derive(Clone, Debug)]
pub struct RegionHeatmapHistory {
    capacity: usize,
    timeline: VecDeque<RegionSnapshotEntry>,
}

impl RegionHeatmapHistory {
    /// Builds a new history buffer with the provided capacity.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(2);
        Self {
            capacity,
            timeline: VecDeque::with_capacity(capacity),
        }
    }

    /// Returns the maximum number of snapshots retained in the history buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of snapshots currently stored.
    pub fn len(&self) -> usize {
        self.timeline.len()
    }

    /// Returns true if the history buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.timeline.is_empty()
    }

    /// Pushes a new snapshot into the buffer, evicting the oldest entry if required.
    pub fn push(&mut self, step: u64, snapshot: RegionHeatmapSnapshot) {
        if self.timeline.len() == self.capacity {
            self.timeline.pop_front();
        }
        self.timeline
            .push_back(RegionSnapshotEntry { step, snapshot });
    }

    /// Returns the most recent step recorded in the history.
    pub fn last_step(&self) -> Option<u64> {
        self.timeline.back().map(|entry| entry.step)
    }

    /// Produces a heatmap describing the delta between the newest and oldest snapshot.
    pub fn delta_report(&self) -> Option<AttributionReport> {
        let first = self.timeline.front()?;
        let last = self.timeline.back()?;
        if core::ptr::eq(first, last) {
            return None;
        }
        let mut metadata = AttributionMetadata::for_algorithm("zspace-region-delta");
        metadata.insert_extra_number("history_window", self.timeline.len() as f64);
        metadata.insert_extra_number("history_capacity", self.capacity as f64);
        metadata.insert_extra_number(
            "delta_step_span",
            last.step.saturating_sub(first.step) as f64,
        );
        metadata.insert_extra_number("delta_step_start", first.step as f64);
        metadata.insert_extra_number("delta_step_end", last.step as f64);
        if let Some(global) = first.snapshot.global_loss() {
            metadata.insert_extra_number("delta_global_start", global as f64);
        }
        if let Some(global) = last.snapshot.global_loss() {
            metadata.insert_extra_number("delta_global_end", global as f64);
        }
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
        metadata.insert_extra_number("default_weight", last.snapshot.default_weight() as f64);

        let mut cells_meta: Vec<Value> = Vec::with_capacity(SPIN_ORDER.len() * RADIUS_ORDER.len());
        let mut values: Vec<f32> = Vec::with_capacity(SPIN_ORDER.len() * RADIUS_ORDER.len());
        let mut max_delta_abs = 0.0f32;
        let mut max_relative_delta_abs = 0.0f32;
        let mut delta_total = 0.0f32;
        let mut positive_cells = 0usize;
        let mut negative_cells = 0usize;
        for spin in SPIN_ORDER.iter().copied() {
            for radius in RADIUS_ORDER.iter().copied() {
                let key = ZSpaceRegionKey::new(spin, radius);
                let start_cell = first.snapshot.cell(key);
                let end_cell = last.snapshot.cell(key);
                let delta = end_cell.combined_weight - start_cell.combined_weight;
                let pct = if start_cell.combined_weight.abs() > f32::EPSILON {
                    delta / start_cell.combined_weight
                } else {
                    0.0
                };
                values.push(delta);
                max_delta_abs = max_delta_abs.max(delta.abs());
                max_relative_delta_abs = max_relative_delta_abs.max(pct.abs());
                delta_total += delta;
                if delta > 0.0 {
                    positive_cells += 1;
                } else if delta < 0.0 {
                    negative_cells += 1;
                }
                cells_meta.push(json!({
                    "key": key.label(),
                    "spin": key.spin.label(),
                    "radius": key.radius.label(),
                    "start_weight": start_cell.combined_weight,
                    "end_weight": end_cell.combined_weight,
                    "delta": delta,
                    "relative_delta": pct,
                }));
            }
        }
        metadata.insert_extra("cells", Value::Array(cells_meta));
        let values_len = values.len();
        emit_tensor_op(
            "zspace_region_delta_report",
            &[self.timeline.len(), values_len],
            &[SPIN_ORDER.len(), RADIUS_ORDER.len()],
        );
        emit_tensor_op_meta("zspace_region_delta_report", || {
            serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "kind": "st_core_zspace_region_delta_report",
                "history_window": self.timeline.len(),
                "history_capacity": self.capacity,
                "step_span": last.step.saturating_sub(first.step),
                "cells": values_len,
                "max_delta_abs": max_delta_abs,
                "max_relative_delta_abs": max_relative_delta_abs,
                "mean_delta": if values_len == 0 {
                    0.0
                } else {
                    delta_total / values_len as f32
                },
                "positive_cells": positive_cells,
                "negative_cells": negative_cells,
                "has_global_start": first.snapshot.global_loss().is_some(),
                "global_start": first.snapshot.global_loss().unwrap_or(0.0),
                "has_global_end": last.snapshot.global_loss().is_some(),
                "global_end": last.snapshot.global_loss().unwrap_or(0.0),
            })
        });
        Some(AttributionReport::new(
            metadata,
            SPIN_ORDER.len(),
            RADIUS_ORDER.len(),
            values,
        ))
    }

    /// Produces a heatmap capturing the standard deviation of weights across the window.
    pub fn volatility_report(&self) -> Option<AttributionReport> {
        if self.timeline.len() < 2 {
            return None;
        }
        let first = self.timeline.front().expect("checked length");
        let last = self.timeline.back().expect("checked length");
        let mut metadata = AttributionMetadata::for_algorithm("zspace-region-volatility");
        metadata.insert_extra_number("history_window", self.timeline.len() as f64);
        metadata.insert_extra_number("history_capacity", self.capacity as f64);
        metadata.insert_extra_number(
            "volatility_step_span",
            last.step.saturating_sub(first.step) as f64,
        );
        metadata.insert_extra_number("volatility_step_start", first.step as f64);
        metadata.insert_extra_number("volatility_step_end", last.step as f64);
        metadata.insert_extra_number("default_weight", last.snapshot.default_weight() as f64);
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

        let mut cells_meta: Vec<Value> = Vec::with_capacity(SPIN_ORDER.len() * RADIUS_ORDER.len());
        let mut values: Vec<f32> = Vec::with_capacity(SPIN_ORDER.len() * RADIUS_ORDER.len());
        let mut std_total = 0.0f32;
        let mut max_std_dev = 0.0f32;
        let mut max_range = 0.0f32;
        let mut active_cells = 0usize;
        for spin in SPIN_ORDER.iter().copied() {
            for radius in RADIUS_ORDER.iter().copied() {
                let key = ZSpaceRegionKey::new(spin, radius);
                let mut mean = 0.0f32;
                let mut m2 = 0.0f32;
                let mut count = 0u32;
                let mut min = f32::MAX;
                let mut max = f32::MIN;
                for entry in self.timeline.iter() {
                    let value = entry.snapshot.cell(key).combined_weight;
                    count = count.saturating_add(1);
                    let delta = value - mean;
                    mean += delta / count as f32;
                    let delta2 = value - mean;
                    m2 += delta * delta2;
                    min = min.min(value);
                    max = max.max(value);
                }
                let variance = if count > 1 {
                    m2 / (count as f32 - 1.0)
                } else {
                    0.0
                };
                let variance = variance.max(0.0);
                let std_dev = variance.sqrt();
                values.push(std_dev);
                std_total += std_dev;
                max_std_dev = max_std_dev.max(std_dev);
                if std_dev > 0.0 {
                    active_cells += 1;
                }
                if min.is_finite() && max.is_finite() {
                    max_range = max_range.max((max - min).abs());
                }
                cells_meta.push(json!({
                    "key": key.label(),
                    "spin": key.spin.label(),
                    "radius": key.radius.label(),
                    "std_dev": std_dev,
                    "mean": mean,
                    "min": if min.is_finite() { min } else { 0.0 },
                    "max": if max.is_finite() { max } else { 0.0 },
                    "samples": count,
                }));
            }
        }
        metadata.insert_extra("cells", Value::Array(cells_meta));
        let values_len = values.len();
        emit_tensor_op(
            "zspace_region_volatility_report",
            &[self.timeline.len(), values_len],
            &[SPIN_ORDER.len(), RADIUS_ORDER.len()],
        );
        emit_tensor_op_meta("zspace_region_volatility_report", || {
            serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "kind": "st_core_zspace_region_volatility_report",
                "history_window": self.timeline.len(),
                "history_capacity": self.capacity,
                "step_span": last.step.saturating_sub(first.step),
                "cells": values_len,
                "active_cells": active_cells,
                "mean_std_dev": if values_len == 0 {
                    0.0
                } else {
                    std_total / values_len as f32
                },
                "max_std_dev": max_std_dev,
                "max_weight_range": max_range,
            })
        });
        Some(AttributionReport::new(
            metadata,
            SPIN_ORDER.len(),
            RADIUS_ORDER.len(),
            values,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::telemetry::tensor_observer_lock()
    }

    fn full_snapshot(default_weight: f32, leading_edge_multiplier: f32) -> RegionHeatmapSnapshot {
        let mut snapshot = RegionHeatmapSnapshot::new(default_weight)
            .with_global_loss(Some(0.8))
            .with_condition(Some(0.2), Some(0.3));
        for spin in SPIN_ORDER.iter().copied() {
            for radius in RADIUS_ORDER.iter().copied() {
                let key = ZSpaceRegionKey::new(spin, radius);
                let multiplier =
                    if spin == ZSpaceSpinBand::Leading && radius == ZSpaceRadiusBand::Edge {
                        leading_edge_multiplier
                    } else {
                        1.0
                    };
                snapshot.insert_cell(RegionHeatmapCell::new(key, 1.0, multiplier, 3, Some(0.5)));
            }
        }
        snapshot
    }

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

    #[test]
    fn history_reports_delta_and_volatility() {
        let mut history = RegionHeatmapHistory::new(3);
        let mut snapshot = RegionHeatmapSnapshot::new(1.0);
        for spin in SPIN_ORDER.iter().copied() {
            for radius in RADIUS_ORDER.iter().copied() {
                let key = ZSpaceRegionKey::new(spin, radius);
                snapshot.insert_cell(RegionHeatmapCell::new(key, 1.0, 1.0, 1, Some(0.5)));
            }
        }
        history.push(0, snapshot.clone());

        let mut snapshot_b = snapshot.clone();
        snapshot_b.insert_cell(RegionHeatmapCell::new(
            ZSpaceRegionKey::new(ZSpaceSpinBand::Leading, ZSpaceRadiusBand::Edge),
            1.0,
            2.0,
            4,
            Some(1.5),
        ));
        history.push(1, snapshot_b.clone());

        let mut snapshot_c = snapshot_b.clone();
        snapshot_c.insert_cell(RegionHeatmapCell::new(
            ZSpaceRegionKey::new(ZSpaceSpinBand::Trailing, ZSpaceRadiusBand::Core),
            1.0,
            0.5,
            6,
            Some(0.25),
        ));
        history.push(2, snapshot_c);

        let delta = history
            .delta_report()
            .expect("delta heatmap available once two frames exist");
        assert_eq!(delta.shape(), (3, 3));
        assert!(delta
            .metadata
            .extras
            .get("cells")
            .expect("cell metadata present")
            .is_array());

        let volatility = history
            .volatility_report()
            .expect("volatility heatmap requires >=2 frames");
        assert_eq!(volatility.shape(), (3, 3));
        assert!(volatility
            .metadata
            .extras
            .get("cells")
            .expect("cell metadata present")
            .is_array());
    }

    #[test]
    fn region_reports_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let descriptor = ZSpaceRegionDescriptor {
            spin_alignment: 0.9,
            normalized_radius: 0.75,
            curvature_radius: 1.5,
            geodesic_radius: 0.5,
            sheet_index: 2,
            sheet_count: 4,
            topological_sector: 1,
        };
        let snapshot = full_snapshot(1.0, 1.5).with_highlight(Some(descriptor));
        let report = snapshot.clone().into_report();

        let mut history = RegionHeatmapHistory::new(3);
        history.push(0, full_snapshot(1.0, 1.0));
        history.push(2, snapshot);
        let delta = history.delta_report().expect("delta report");
        let volatility = history.volatility_report().expect("volatility report");
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(report.shape(), (3, 3));
        assert_eq!(delta.shape(), (3, 3));
        assert_eq!(volatility.shape(), (3, 3));

        let events = events.lock().unwrap();
        let heatmap = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_region_heatmap_report"
                    && data["cells"] == 9
                    && data["has_highlight"] == true
                    && data["max_combined_weight"].as_f64().unwrap_or(0.0) > 1.0
            })
            .expect("zspace_region_heatmap_report metadata event");
        assert_eq!(heatmap.1["backend"], "cpu");
        assert_eq!(heatmap.1["kind"], "st_core_zspace_region_heatmap_report");
        assert!(heatmap.1["max_combined_weight"].as_f64().unwrap_or(0.0) > 1.0);

        let delta_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_region_delta_report"
                    && data["history_window"] == 2
                    && data["cells"] == 9
            })
            .expect("zspace_region_delta_report metadata event");
        assert_eq!(delta_meta.1["kind"], "st_core_zspace_region_delta_report");
        assert!(delta_meta.1["max_delta_abs"].as_f64().unwrap_or(0.0) > 0.0);

        let volatility_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_region_volatility_report"
                    && data["history_window"] == 2
                    && data["cells"] == 9
            })
            .expect("zspace_region_volatility_report metadata event");
        assert_eq!(
            volatility_meta.1["kind"],
            "st_core_zspace_region_volatility_report"
        );
        assert!(volatility_meta.1["max_std_dev"].as_f64().unwrap_or(0.0) > 0.0);
    }
}
