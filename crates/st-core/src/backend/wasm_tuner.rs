// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::backend::wgpu_heuristics::Choice;
use serde::{Deserialize, Serialize};
use std::cmp::{max, min, Ordering};
use std::fmt;

/// Error emitted when a WASM tuner table cannot be parsed.
#[derive(Debug, thiserror::Error)]
#[error("failed to parse WASM tuner table: {0}")]
pub struct WasmTunerError(String);

impl From<serde_json::Error> for WasmTunerError {
    fn from(err: serde_json::Error) -> Self {
        Self(err.to_string())
    }
}

/// Row in the WASM tuner table.  Each entry represents a bucket of workloads and
/// optional overrides that will be applied to the base heuristic `Choice`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WasmTunerRecord {
    /// Minimum number of rows that the entry applies to.
    #[serde(default, alias = "rows")]
    pub rows_min: Option<usize>,
    /// Maximum number of rows that the entry applies to.
    #[serde(default)]
    pub rows_max: Option<usize>,
    /// Minimum number of columns.
    #[serde(default)]
    pub cols_min: usize,
    /// Maximum number of columns.
    #[serde(default = "default_cols_max")]
    pub cols_max: usize,
    /// Minimum `k` value supported by the entry.
    #[serde(default)]
    pub k_min: usize,
    /// Maximum `k` value supported by the entry.
    #[serde(default = "default_k_max", alias = "k_max")]
    pub k_max: usize,
    /// Whether the rule applies to subgroup execution.
    #[serde(default, alias = "sg")]
    pub subgroup: Option<bool>,
    /// Override for the Top-K algorithm (`algo_topk`).
    #[serde(default, alias = "mk")]
    pub algo_topk: Option<u8>,
    /// Override for the compaction tile size (`ctile`).
    #[serde(default, alias = "tile")]
    pub ctile: Option<u32>,
    /// Override for the work-group size.
    #[serde(default)]
    pub wg: Option<u32>,
    /// Override for the K loop tiling (`kl`).
    #[serde(default)]
    pub kl: Option<u32>,
    /// Override for the channel stride (`ch`).
    #[serde(default)]
    pub ch: Option<u32>,
    /// Override for the mid-K mode.
    #[serde(default)]
    pub mode_midk: Option<u8>,
    /// Override for the bottom-K mode.
    #[serde(default)]
    pub mode_bottomk: Option<u8>,
    /// Override for FFT tile columns.
    #[serde(default)]
    pub tile_cols: Option<u32>,
    /// Override for the FFT radix.
    #[serde(default)]
    pub radix: Option<u32>,
    /// Override for the ND segment count.
    #[serde(default)]
    pub segments: Option<u32>,
    /// Override for enabling two-stage compaction.
    #[serde(default)]
    pub use_2ce: Option<bool>,
}

fn default_cols_max() -> usize {
    usize::MAX
}

fn default_k_max() -> usize {
    usize::MAX
}

impl WasmTunerRecord {
    fn matches(&self, rows: usize, cols: usize, k: usize, subgroup: bool) -> bool {
        if let Some(rows_min) = self.rows_min {
            if rows < rows_min {
                return false;
            }
        }
        if let Some(rows_max) = self.rows_max {
            if rows > rows_max {
                return false;
            }
        }
        if cols < self.cols_min || cols > self.cols_max {
            return false;
        }
        if k < self.k_min || k > self.k_max {
            return false;
        }
        if let Some(expected) = self.subgroup {
            if expected != subgroup {
                return false;
            }
        }
        true
    }

    fn apply(&self, choice: &mut Choice) {
        if let Some(value) = self.use_2ce {
            choice.use_2ce = value;
        }
        if let Some(value) = self.wg {
            choice.wg = value;
        }
        if let Some(value) = self.kl {
            choice.kl = value;
        }
        if let Some(value) = self.ch {
            choice.ch = value;
        }
        if let Some(value) = self.algo_topk {
            choice.algo_topk = value;
        }
        if let Some(value) = self.ctile {
            choice.ctile = value;
        }
        if let Some(value) = self.mode_midk {
            choice.mode_midk = value;
        }
        if let Some(value) = self.mode_bottomk {
            choice.mode_bottomk = value;
        }
        if let Some(value) = self.tile_cols {
            choice.tile_cols = value;
        }
        if let Some(value) = self.radix {
            choice.radix = max(2, value);
        }
        if let Some(value) = self.segments {
            choice.segments = min(8, value);
        }
    }
}

/// WASM tuner table that can be shipped to the browser and queried on the
/// client.  The table keeps the dataset in its original JSON form so it can be
/// serialised back without losing information.
#[derive(Clone)]
pub struct WasmTunerTable {
    records: Vec<WasmTunerRecord>,
}

impl fmt::Debug for WasmTunerTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WasmTunerTable")
            .field("records", &self.records.len())
            .finish()
    }
}

impl WasmTunerTable {
    /// Create an empty table.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Construct a table from a collection of records.
    pub fn from_records(records: Vec<WasmTunerRecord>) -> Self {
        let mut table = Self { records };
        table.sort_internal();
        table
    }

    /// Create a table from JSON text.
    pub fn from_json_str(json: &str) -> Result<Self, WasmTunerError> {
        let records: Vec<WasmTunerRecord> = serde_json::from_str(json)?;
        Ok(Self::from_records(records))
    }

    /// Serialise the table back to JSON.
    pub fn to_json(&self) -> Result<String, WasmTunerError> {
        Ok(serde_json::to_string_pretty(&self.records)?)
    }

    /// Number of records stored in the table.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` when the table contains no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Push a new record.
    pub fn push(&mut self, record: WasmTunerRecord) {
        self.records.push(record);
    }

    /// Push a record while keeping the ordering stable.
    pub fn push_sorted(&mut self, record: WasmTunerRecord) {
        self.records.push(record);
        self.sort_internal();
    }

    /// Extend the table with new records and keep them ordered.
    pub fn extend_sorted<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = WasmTunerRecord>,
    {
        self.records.extend(iter);
        self.sort_internal();
    }

    /// Remove every record stored in the table.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Re-sort the internal dataset.
    pub fn sort(&mut self) {
        self.sort_internal();
    }

    fn sort_internal(&mut self) {
        self.records.sort_by(Self::compare_records);
    }

    fn compare_records(a: &WasmTunerRecord, b: &WasmTunerRecord) -> Ordering {
        match (a.subgroup, b.subgroup) {
            (Some(true), Some(false)) => Ordering::Less,
            (Some(false), Some(true)) => Ordering::Greater,
            _ => a
                .rows_min
                .unwrap_or(0)
                .cmp(&b.rows_min.unwrap_or(0))
                .then(a.cols_min.cmp(&b.cols_min))
                .then(b.cols_max.cmp(&a.cols_max)),
        }
    }

    /// Iterate over the internal records.
    pub fn iter(&self) -> impl Iterator<Item = &WasmTunerRecord> {
        self.records.iter()
    }

    /// Pick an entry for the provided workload parameters.  The caller should
    /// start from a fallback `Choice` and then apply the override from the
    /// matching record.
    pub fn choose(
        &self,
        base: Choice,
        rows: usize,
        cols: usize,
        k: usize,
        subgroup: bool,
    ) -> Option<Choice> {
        for record in &self.records {
            if record.matches(rows, cols, k, subgroup) {
                let mut out = base;
                record.apply(&mut out);
                return Some(out);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_choice() -> Choice {
        Choice {
            use_2ce: false,
            wg: 128,
            kl: 8,
            ch: 0,
            algo_topk: 0,
            ctile: 256,
            mode_midk: 0,
            mode_bottomk: 0,
            tile_cols: 512,
            radix: 2,
            segments: 1,
        }
    }

    #[test]
    fn parses_and_matches_records() {
        let json = r#"[
            {"rows": 512, "cols_min": 0, "cols_max": 8192, "k_max": 128, "sg": true,
             "wg": 256, "tile_cols": 1024, "radix": 4},
            {"rows": 0, "cols_min": 8193, "cols_max": 262143, "k_max": 2048, "sg": false,
             "use_2ce": true, "segments": 4}
        ]"#;
        let table = WasmTunerTable::from_json_str(json).unwrap();
        assert_eq!(table.len(), 2);

        let choice = table
            .choose(base_choice(), 512, 4096, 64, true)
            .expect("should match first record");
        assert_eq!(choice.wg, 256);
        assert_eq!(choice.tile_cols, 1024);
        assert_eq!(choice.radix, 4);

        let choice = table
            .choose(base_choice(), 1024, 9000, 256, false)
            .expect("should match second record");
        assert!(choice.use_2ce);
        assert_eq!(choice.segments, 4);
    }

    #[test]
    fn serialises_back_to_json() {
        let json = r#"[{"cols_min":0,"cols_max":1,"k_max":1} ]"#;
        let table = WasmTunerTable::from_json_str(json).unwrap();
        let out = table.to_json().unwrap();
        assert!(out.contains("\"cols_min\": 0"));
    }

    fn record(
        rows_min: Option<usize>,
        rows_max: Option<usize>,
        cols_min: usize,
        cols_max: usize,
        subgroup: Option<bool>,
    ) -> WasmTunerRecord {
        WasmTunerRecord {
            rows_min,
            rows_max,
            cols_min,
            cols_max,
            k_min: 0,
            k_max: usize::MAX,
            subgroup,
            algo_topk: None,
            ctile: None,
            wg: None,
            kl: None,
            ch: None,
            mode_midk: None,
            mode_bottomk: None,
            tile_cols: None,
            radix: None,
            segments: None,
            use_2ce: None,
        }
    }

    #[test]
    fn new_table_is_empty() {
        let table = WasmTunerTable::new();
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn push_sorted_keeps_ordering() {
        let mut table = WasmTunerTable::new();
        table.push_sorted(record(Some(256), None, 4096, 8192, Some(false)));
        table.push_sorted(record(Some(128), None, 0, 4095, Some(true)));
        table.push_sorted(record(Some(512), None, 8193, 16_384, None));
        let mut iter = table.iter();
        assert_eq!(iter.next().unwrap().subgroup, Some(true));
        assert_eq!(iter.next().unwrap().rows_min, Some(256));
        assert!(iter.next().unwrap().rows_min.unwrap() >= 512);
    }

    #[test]
    fn extend_sorted_merges_records() {
        let mut table =
            WasmTunerTable::from_records(vec![record(Some(256), None, 0, 1024, Some(true))]);
        let extra = vec![
            record(Some(256), None, 1025, 2048, Some(true)),
            record(Some(512), None, 0, 1024, Some(false)),
        ];
        table.extend_sorted(extra);
        assert_eq!(table.len(), 3);
        let mut iter = table.iter();
        let first = iter.next().unwrap();
        assert_eq!(first.cols_min, 0);
        assert_eq!(first.subgroup, Some(true));
        let second = iter.next().unwrap();
        assert_eq!(second.cols_min, 1025);
        let third = iter.next().unwrap();
        assert_eq!(third.subgroup, Some(false));
    }
}
