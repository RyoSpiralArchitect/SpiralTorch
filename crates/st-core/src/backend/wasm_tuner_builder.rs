// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::wasm_tuner::{WasmTunerRecord, WasmTunerTable};
use crate::backend::unison_heuristics::RankKind;
use crate::ecosystem::{
    EcosystemReport, HeuristicDecision, HeuristicSource, RankPlanSummary, RoundtableSummary,
};
use std::collections::{BTreeMap, HashMap};

/// Adaptive builder that aggregates heuristic telemetry and produces
/// [`WasmTunerRecord`] buckets that mirror the workloads seen in practice.
///
/// The builder consumes [`HeuristicDecision`]s, [`RoundtableSummary`] records,
/// and full [`EcosystemReport`] snapshots.  Each observation contributes votes
/// towards the parameters stored in the resulting WASM tuner table so that the
/// generated overrides reflect the dominant runtime configuration.
#[derive(Default, Debug)]
pub struct WasmTunerAutoBuilder {
    buckets: HashMap<BucketKey, BucketStats>,
}

impl WasmTunerAutoBuilder {
    /// Creates a new, empty builder.
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    /// Ingests a full ecosystem report, folding roundtable plans and
    /// heuristic decisions into the adaptive buckets.
    pub fn ingest_report(&mut self, report: &EcosystemReport) {
        for summary in report.roundtables() {
            self.ingest_roundtable(summary);
        }
        for decision in report.heuristics() {
            self.ingest_heuristic(decision);
        }
    }

    /// Records a single roundtable summary.  Every contained
    /// [`RankPlanSummary`] contributes votes towards the buckets matching its
    /// dimensionality.
    pub fn ingest_roundtable(&mut self, summary: &RoundtableSummary) {
        for plan in &summary.plans {
            let key = BucketKey::from_dims(plan.rows, plan.cols, plan.k);
            let stats = self.buckets.entry(key).or_insert_with(BucketStats::new);
            stats.observe_plan(plan);
        }
    }

    /// Records a single heuristic decision emitted by the WGPU planner.
    pub fn ingest_heuristic(&mut self, decision: &HeuristicDecision) {
        let key = BucketKey::from_dims(decision.rows, decision.cols, decision.k);
        let stats = self.buckets.entry(key).or_insert_with(BucketStats::new);
        stats.observe_heuristic(decision);
    }

    /// Returns the aggregated tuner records without consuming the builder.
    pub fn records(&self) -> Vec<WasmTunerRecord> {
        let mut records: Vec<WasmTunerRecord> = self
            .buckets
            .values()
            .filter_map(BucketStats::to_record)
            .collect();
        records.sort_by(|a, b| {
            let rows_a = a.rows_min.unwrap_or(0);
            let rows_b = b.rows_min.unwrap_or(0);
            rows_a
                .cmp(&rows_b)
                .then_with(|| a.cols_min.cmp(&b.cols_min))
                .then_with(|| a.k_min.cmp(&b.k_min))
        });
        records
    }

    /// Builds a [`WasmTunerTable`] containing the aggregated overrides.
    pub fn build_table(&self) -> WasmTunerTable {
        WasmTunerTable::from_records(self.records())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BucketKey {
    rows_bucket: u32,
    cols_bucket: u32,
    k_bucket: u32,
}

impl BucketKey {
    fn from_dims(rows: u32, cols: u32, k: u32) -> Self {
        Self {
            rows_bucket: bucket_index(rows),
            cols_bucket: bucket_index(cols),
            k_bucket: bucket_index(k.max(1)),
        }
    }
}

#[derive(Debug)]
struct BucketStats {
    rows_min: usize,
    rows_max: usize,
    cols_min: usize,
    cols_max: usize,
    k_min: usize,
    k_max: usize,
    subgroup: Vote<bool>,
    use_two_stage: Vote<bool>,
    workgroup: Vote<u32>,
    lanes: Vote<u32>,
    channel_stride: Vote<u32>,
    compaction_tile: Vote<u32>,
    algo_topk: Vote<u8>,
    mode_midk: Vote<u8>,
    mode_bottomk: Vote<u8>,
    fft_tile_cols: Vote<u32>,
    fft_radix: Vote<u32>,
    fft_segments: Vote<u32>,
    observations: usize,
}

impl BucketStats {
    fn new() -> Self {
        Self {
            rows_min: usize::MAX,
            rows_max: 0,
            cols_min: usize::MAX,
            cols_max: 0,
            k_min: usize::MAX,
            k_max: 0,
            subgroup: Vote::default(),
            use_two_stage: Vote::default(),
            workgroup: Vote::default(),
            lanes: Vote::default(),
            channel_stride: Vote::default(),
            compaction_tile: Vote::default(),
            algo_topk: Vote::default(),
            mode_midk: Vote::default(),
            mode_bottomk: Vote::default(),
            fft_tile_cols: Vote::default(),
            fft_radix: Vote::default(),
            fft_segments: Vote::default(),
            observations: 0,
        }
    }

    fn observe_plan(&mut self, plan: &RankPlanSummary) {
        self.observe_dims(plan.rows, plan.cols, plan.k);
        self.observations += 1;
        self.subgroup.push(plan.subgroup);
        if plan.workgroup > 0 {
            self.workgroup.push(plan.workgroup);
        }
        if plan.lanes > 0 {
            self.lanes.push(plan.lanes);
        }
        if plan.channel_stride > 0 {
            self.channel_stride.push(plan.channel_stride);
        }
        if plan.compaction_tile > 0 {
            self.compaction_tile.push(plan.compaction_tile);
        }
        if plan.fft_tile > 0 {
            self.fft_tile_cols.push(plan.fft_tile);
        }
        if plan.fft_radix > 0 {
            self.fft_radix.push(plan.fft_radix);
        }
        if plan.fft_segments > 0 {
            self.fft_segments.push(plan.fft_segments);
        }
    }

    fn observe_heuristic(&mut self, decision: &HeuristicDecision) {
        self.observe_dims(decision.rows, decision.cols, decision.k);
        self.observations += 1;
        let choice = &decision.choice;
        self.use_two_stage.push(choice.use_two_stage);
        if choice.workgroup > 0 {
            self.workgroup.push(choice.workgroup);
        }
        if choice.lanes > 0 {
            self.lanes.push(choice.lanes);
        }
        if choice.channel_stride > 0 {
            self.channel_stride.push(choice.channel_stride);
        }
        if choice.compaction_tile > 0 {
            self.compaction_tile.push(choice.compaction_tile);
        }
        if choice.fft_tile_cols > 0 {
            self.fft_tile_cols.push(choice.fft_tile_cols);
        }
        if choice.fft_radix > 0 {
            self.fft_radix.push(choice.fft_radix);
        }
        if choice.fft_segments > 0 {
            self.fft_segments.push(choice.fft_segments);
        }

        match decision.kind.as_str() {
            "topk" => {
                if let Some(value) = decode_topk_algo(choice.algo_hint.as_deref()) {
                    self.algo_topk.push(value);
                }
            }
            "midk" => {
                if let Some(value) = decode_midbottom_mode(choice.algo_hint.as_deref()) {
                    self.mode_midk.push(value);
                }
            }
            "bottomk" => {
                if let Some(value) = decode_midbottom_mode(choice.algo_hint.as_deref()) {
                    self.mode_bottomk.push(value);
                }
            }
            _ => {}
        }
    }

    fn observe_dims(&mut self, rows: u32, cols: u32, k: u32) {
        update_bounds(&mut self.rows_min, &mut self.rows_max, rows as usize);
        update_bounds(&mut self.cols_min, &mut self.cols_max, cols as usize);
        update_bounds(&mut self.k_min, &mut self.k_max, k as usize);
    }

    fn to_record(&self) -> Option<WasmTunerRecord> {
        if self.observations == 0 || self.cols_min == usize::MAX {
            return None;
        }

        let rows_min = if self.rows_min == usize::MAX {
            None
        } else {
            Some(self.rows_min)
        };
        let rows_max = if self.rows_max > self.rows_min {
            Some(self.rows_max)
        } else {
            None
        };
        let cols_min = if self.cols_min == usize::MAX {
            0
        } else {
            self.cols_min
        };
        let cols_max = if self.cols_max >= cols_min {
            self.cols_max
        } else {
            cols_min
        };
        let k_min = if self.k_min == usize::MAX {
            0
        } else {
            self.k_min
        };
        let k_max = if self.k_max >= k_min {
            self.k_max
        } else {
            k_min
        };

        let mut record = WasmTunerRecord {
            rows_min,
            rows_max,
            cols_min,
            cols_max,
            k_min,
            k_max,
            subgroup: self.subgroup.dominant(),
            algo_topk: self.algo_topk.most_frequent().filter(|value| *value != 0),
            ctile: self
                .compaction_tile
                .most_frequent()
                .filter(|value| *value != 0),
            wg: self.workgroup.most_frequent().filter(|value| *value != 0),
            kl: self.lanes.most_frequent().filter(|value| *value != 0),
            ch: self
                .channel_stride
                .most_frequent()
                .filter(|value| *value != 0),
            mode_midk: self.mode_midk.most_frequent().filter(|value| *value != 0),
            mode_bottomk: self
                .mode_bottomk
                .most_frequent()
                .filter(|value| *value != 0),
            tile_cols: self
                .fft_tile_cols
                .most_frequent()
                .filter(|value| *value != 0),
            radix: self.fft_radix.most_frequent().filter(|value| *value != 0),
            segments: self
                .fft_segments
                .most_frequent()
                .filter(|value| *value != 0),
            use_2ce: self.use_two_stage.dominant(),
        };

        // Normalise ranges so we do not emit degenerate spans.
        if record.cols_max < record.cols_min {
            record.cols_max = record.cols_min;
        }
        if record.k_max < record.k_min {
            record.k_max = record.k_min;
        }

        Some(record)
    }
}

#[derive(Default, Debug, Clone)]
struct Vote<T>
where
    T: Copy + Ord,
{
    counts: BTreeMap<T, usize>,
}

impl<T> Vote<T>
where
    T: Copy + Ord,
{
    fn push(&mut self, value: T) {
        *self.counts.entry(value).or_insert(0) += 1;
    }

    fn total(&self) -> usize {
        self.counts.values().copied().sum()
    }

    fn most_frequent(&self) -> Option<T> {
        self.counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(value, _)| *value)
    }

    fn dominant(&self) -> Option<T> {
        let total = self.total();
        let (value, count) = self.counts.iter().max_by_key(|(_, count)| *count)?;
        if count * 2 > total {
            Some(*value)
        } else {
            None
        }
    }
}

fn bucket_index(value: u32) -> u32 {
    if value <= 1 {
        1
    } else {
        value.next_power_of_two()
    }
}

fn update_bounds(min: &mut usize, max: &mut usize, value: usize) {
    if value == 0 {
        return;
    }
    if *min == usize::MAX || value < *min {
        *min = value;
    }
    if value > *max {
        *max = value;
    }
}

fn decode_topk_algo(hint: Option<&str>) -> Option<u8> {
    let hint = hint?;
    let value = hint.strip_prefix("algo=")?;
    match value {
        "heap" => Some(1),
        "bitonic" => Some(2),
        "kway" => Some(3),
        _ => None,
    }
}

fn decode_midbottom_mode(hint: Option<&str>) -> Option<u8> {
    let hint = hint?;
    let value = hint.strip_prefix("mode=")?;
    match value {
        "1ce" => Some(1),
        "2ce" => Some(2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecosystem::{HeuristicChoiceSummary, RoundtableConfigSummary};
    use std::time::SystemTime;

    fn plan_summary(kind: RankKind, rows: u32, cols: u32, k: u32) -> RankPlanSummary {
        let mut plan = RankPlanSummary::new(kind, rows, cols, k);
        plan.workgroup = 128;
        plan.lanes = 32;
        plan.channel_stride = 4;
        plan.compaction_tile = 1024;
        plan.subgroup = true;
        plan.fft_tile = 2048;
        plan.fft_radix = 4;
        plan.fft_segments = 2;
        plan
    }

    fn heuristic_decision(kind: &str, rows: u32, cols: u32, k: u32) -> HeuristicDecision {
        HeuristicDecision {
            subsystem: "wgpu".to_string(),
            kind: kind.to_string(),
            rows,
            cols,
            k,
            choice: HeuristicChoiceSummary::new(
                true,
                128,
                32,
                4,
                Some(match kind {
                    "topk" => "algo=heap".to_string(),
                    "midk" => "mode=2ce".to_string(),
                    "bottomk" => "mode=1ce".to_string(),
                    _ => "".to_string(),
                }),
                1024,
                2048,
                4,
                2,
            ),
            score_hint: Some(0.9),
            source: HeuristicSource::Generated,
            issued_at: SystemTime::now(),
        }
    }

    #[test]
    fn builder_merges_roundtables_and_heuristics() {
        let mut builder = WasmTunerAutoBuilder::new();
        let roundtable = RoundtableSummary {
            rows: 320,
            cols: 8192,
            config: RoundtableConfigSummary::new(64, 32, 16, 0.1),
            plans: vec![plan_summary(RankKind::TopK, 320, 8192, 64)],
            autopilot_enabled: false,
            distribution: None,
            issued_at: SystemTime::now(),
        };
        builder.ingest_roundtable(&roundtable);

        builder.ingest_heuristic(&heuristic_decision("topk", 320, 8192, 64));

        let records = builder.records();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.rows_min, Some(320));
        assert_eq!(record.rows_max, None);
        assert_eq!(record.cols_min, 8192);
        assert_eq!(record.cols_max, 8192);
        assert_eq!(record.k_min, 64);
        assert_eq!(record.k_max, 64);
        assert_eq!(record.subgroup, Some(true));
        assert_eq!(record.wg, Some(128));
        assert_eq!(record.kl, Some(32));
        assert_eq!(record.ch, Some(4));
        assert_eq!(record.ctile, Some(1024));
        assert_eq!(record.tile_cols, Some(2048));
        assert_eq!(record.radix, Some(4));
        assert_eq!(record.segments, Some(2));
        assert_eq!(record.use_2ce, Some(true));
        assert_eq!(record.algo_topk, Some(1));
    }

    #[test]
    fn builder_tracks_kind_specific_modes() {
        let mut builder = WasmTunerAutoBuilder::new();
        builder.ingest_heuristic(&heuristic_decision("midk", 512, 16384, 128));
        builder.ingest_heuristic(&heuristic_decision("bottomk", 512, 16384, 128));
        let records = builder.records();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.mode_midk, Some(2));
        assert_eq!(record.mode_bottomk, Some(1));
    }

    #[test]
    fn ambiguous_votes_are_cleared() {
        let mut builder = WasmTunerAutoBuilder::new();
        let decision_true = HeuristicDecision {
            choice: HeuristicChoiceSummary::new(true, 64, 16, 4, None, 0, 0, 0, 0),
            ..heuristic_decision("topk", 128, 2048, 32)
        };
        let decision_false = HeuristicDecision {
            choice: HeuristicChoiceSummary::new(false, 64, 16, 4, None, 0, 0, 0, 0),
            ..heuristic_decision("topk", 128, 2048, 32)
        };
        builder.ingest_heuristic(&decision_true);
        builder.ingest_heuristic(&decision_false);
        let records = builder.records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].use_2ce, None);
    }
}
