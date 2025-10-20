// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Machine-readable representation of the backend feature matrix documented in
//! `docs/backend_matrix.md`.
//!
//! The matrix is exposed as structured data so higher level tooling (benches,
//! telemetry diffing, CLI status printers) can remain in sync with the
//! documentation. Each capability row mirrors the entries in the Markdown table
//! and stores the qualitative readiness state alongside the note rendered in the
//! document.

use serde::Serialize;
use std::fmt;

/// Backends tracked in the feature matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Cpu,
    Wgpu,
    Mps,
    Cuda,
    Hip,
}

impl Backend {
    /// Number of backends currently tracked in the matrix.
    pub const COUNT: usize = 5;

    /// Ordered slice containing every backend.
    pub const ALL: [Backend; Backend::COUNT] = [
        Backend::Cpu,
        Backend::Wgpu,
        Backend::Mps,
        Backend::Cuda,
        Backend::Hip,
    ];

    /// Returns the canonical name used in human facing documentation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Wgpu => "WGPU",
            Backend::Mps => "MPS",
            Backend::Cuda => "CUDA",
            Backend::Hip => "HIP / ROCm",
        }
    }

    /// Parses the canonical backend label back into a [`Backend`].
    pub fn from_str(name: &str) -> Option<Self> {
        Backend::ALL
            .iter()
            .copied()
            .find(|backend| backend.as_str().eq_ignore_ascii_case(name.trim()))
    }

    pub(crate) const fn index(self) -> usize {
        match self {
            Backend::Cpu => 0,
            Backend::Wgpu => 1,
            Backend::Mps => 2,
            Backend::Cuda => 3,
            Backend::Hip => 4,
        }
    }
}

impl Serialize for Backend {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// Qualitative readiness level used in the documentation matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CapabilityState {
    Ready,
    Watchlist,
    Blocked,
}

impl CapabilityState {
    /// Returns the emoji marker used in the Markdown table.
    pub const fn marker(self) -> &'static str {
        match self {
            CapabilityState::Ready => "✅",
            CapabilityState::Watchlist => "⚠️",
            CapabilityState::Blocked => "❌",
        }
    }
}

impl fmt::Display for CapabilityState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.marker())
    }
}

/// Entry describing how a backend fulfils a given capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CapabilityEntry {
    /// Optional readiness classification. Rows that convey free-form notes
    /// (e.g. build flags) omit a state marker.
    pub state: Option<CapabilityState>,
    /// Additional note matching the Markdown table text.
    pub note: &'static str,
}

impl CapabilityEntry {
    const fn note_only(note: &'static str) -> Self {
        Self { state: None, note }
    }

    const fn with_state(state: CapabilityState, note: &'static str) -> Self {
        Self {
            state: Some(state),
            note,
        }
    }
}

/// Capability row in the backend matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CapabilityRow {
    /// Capability name as rendered in the first column of the table.
    pub capability: &'static str,
    /// Matrix entries ordered using [`Backend::ALL`].
    pub entries: [CapabilityEntry; Backend::COUNT],
}

/// Wrapper providing utility accessors around the static capability matrix.
#[derive(Debug, Clone, Copy)]
pub struct CapabilityMatrix {
    rows: &'static [CapabilityRow],
}

impl CapabilityMatrix {
    /// Construct a view over the provided capability rows.
    pub const fn new(rows: &'static [CapabilityRow]) -> Self {
        Self { rows }
    }

    /// Returns the raw capability rows in definition order.
    pub const fn rows(&self) -> &'static [CapabilityRow] {
        self.rows
    }

    /// Finds a capability row by name (case-insensitive).
    pub fn capability(&self, name: &str) -> Option<&'static CapabilityRow> {
        capability_by_name(name)
    }

    /// Summarises how `backend` fares across every capability.
    pub fn backend_summary(&self, backend: Backend) -> BackendSummary {
        summarize_backend(backend)
    }

    /// Returns the matrix-wide readiness counters.
    pub fn summary(&self) -> MatrixSummary {
        matrix_summary()
    }

    /// Lists every capability where `backend` still requires follow-up work.
    pub fn pending_for_backend(&self, backend: Backend) -> Vec<&'static CapabilityRow> {
        pending_capabilities_for_backend(backend)
    }

    /// Returns capability rows whose notes mention `query` (case-insensitive).
    pub fn capabilities_with_note(&self, query: &str) -> Vec<&'static CapabilityRow> {
        capabilities_with_note_containing(query)
    }
}

impl CapabilityRow {
    /// Retrieve the entry for `backend`.
    pub const fn entry(&self, backend: Backend) -> &CapabilityEntry {
        &self.entries[backend.index()]
    }

    /// Computes a [`CapabilitySummary`] describing how every backend fares for this row.
    pub const fn summarize(&self) -> CapabilitySummary {
        let mut ready = 0;
        let mut watchlist = 0;
        let mut blocked = 0;
        let mut idx = 0;
        while idx < Backend::COUNT {
            let state = self.entries[idx].state;
            match state {
                Some(CapabilityState::Ready) => ready += 1,
                Some(CapabilityState::Watchlist) => watchlist += 1,
                Some(CapabilityState::Blocked) => blocked += 1,
                None => {}
            }
            idx += 1;
        }

        CapabilitySummary {
            capability: self.capability,
            ready,
            watchlist,
            blocked,
        }
    }
}

/// Aggregated readiness information for a backend across the entire matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackendNote {
    /// Capability row the note originates from.
    pub capability: &'static str,
    /// Additional context mirrored from the documentation table.
    pub note: &'static str,
}

/// Aggregated readiness information for a backend across the entire matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackendSummary {
    /// Backend identifier.
    pub backend: Backend,
    /// Number of ready capabilities.
    pub ready: usize,
    /// Number of capabilities tracked on the watchlist.
    pub watchlist: usize,
    /// Number of blocked capabilities.
    pub blocked: usize,
    /// Collected notes for non-ready capabilities and informational rows.
    pub notes: Vec<BackendNote>,
}

impl BackendSummary {
    /// Returns `true` if the backend has no watchlist or blocked capabilities.
    pub fn is_fully_ready(&self) -> bool {
        self.watchlist == 0 && self.blocked == 0
    }

    /// Number of capabilities that carry a readiness marker for this backend.
    pub fn tracked_capabilities(&self) -> usize {
        self.ready + self.watchlist + self.blocked
    }

    /// Fraction of tracked capabilities that are marked as ready.
    pub fn readiness_ratio(&self) -> f32 {
        let total = self.tracked_capabilities();
        if total == 0 {
            0.0
        } else {
            self.ready as f32 / total as f32
        }
    }

    /// Number of capabilities that are not yet ready.
    pub fn pending(&self) -> usize {
        self.watchlist + self.blocked
    }
}

/// Aggregate counters describing the entire backend matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MatrixSummary {
    /// Number of entries marked ready across the matrix.
    pub ready: usize,
    /// Number of entries tracked on the watchlist.
    pub watchlist: usize,
    /// Number of entries marked as blocked.
    pub blocked: usize,
    /// Number of informational entries without readiness markers.
    pub informational: usize,
}

impl MatrixSummary {
    /// Number of entries that have an explicit readiness marker.
    pub const fn tracked_entries(&self) -> usize {
        self.ready + self.watchlist + self.blocked
    }

    /// Total number of entries including informational notes.
    pub const fn total_entries(&self) -> usize {
        self.tracked_entries() + self.informational
    }

    /// Fraction of tracked entries that are marked ready.
    pub fn readiness_ratio(&self) -> f32 {
        let tracked = self.tracked_entries();
        if tracked == 0 {
            0.0
        } else {
            self.ready as f32 / tracked as f32
        }
    }
}

/// Aggregated readiness information across all backends for a single capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CapabilitySummary {
    /// Capability name as rendered in the documentation matrix.
    pub capability: &'static str,
    /// Number of backends marked as ready.
    pub ready: usize,
    /// Number of backends on the watchlist.
    pub watchlist: usize,
    /// Number of backends marked as blocked.
    pub blocked: usize,
}

impl CapabilitySummary {
    /// Total number of backends that have an explicit readiness marker for the capability.
    pub const fn tracked_backends(&self) -> usize {
        self.ready + self.watchlist + self.blocked
    }

    /// Dominant state if exactly one readiness tier has the highest count.
    pub const fn dominant_state(&self) -> Option<CapabilityState> {
        let mut max = 0usize;
        let mut ties = 0usize;
        let mut candidate: Option<CapabilityState> = None;

        let ready = self.ready;
        if ready > max {
            max = ready;
            ties = 1;
            candidate = Some(CapabilityState::Ready);
        } else if ready != 0 && ready == max {
            ties += 1;
        }

        let watchlist = self.watchlist;
        if watchlist > max {
            max = watchlist;
            ties = 1;
            candidate = Some(CapabilityState::Watchlist);
        } else if watchlist != 0 && watchlist == max {
            ties += 1;
        }

        let blocked = self.blocked;
        if blocked > max {
            max = blocked;
            ties = 1;
            candidate = Some(CapabilityState::Blocked);
        } else if blocked != 0 && blocked == max {
            ties += 1;
        }

        if max == 0 || ties != 1 {
            None
        } else {
            candidate
        }
    }
}

static CAPABILITY_ROWS: &[CapabilityRow] = &[
    CapabilityRow {
        capability: "Build flag",
        entries: [
            CapabilityEntry::note_only("_none_"),
            CapabilityEntry::note_only("`--features wgpu`"),
            CapabilityEntry::note_only("`--features mps`"),
            CapabilityEntry::note_only("`--features cuda`"),
            CapabilityEntry::note_only("`--features \"hip,st-backend-hip/hip-real\"`"),
        ],
    },
    CapabilityRow {
        capability: "Min toolchain",
        entries: [
            CapabilityEntry::note_only("Stable Rust"),
            CapabilityEntry::note_only("Stable Rust + system WebGPU drivers"),
            CapabilityEntry::note_only("Stable Rust + macOS 14 SDK"),
            CapabilityEntry::note_only("Stable Rust + CUDA 12 Toolkit & NVRTC"),
            CapabilityEntry::note_only("Stable Rust + ROCm 6 toolchain"),
        ],
    },
    CapabilityRow {
        capability: "Tensor ops",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Full"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Full (verify image/texture paths)",
            ),
            CapabilityEntry::with_state(CapabilityState::Ready, "Full"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Full"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Full"),
        ],
    },
    CapabilityRow {
        capability: "Autodiff / hypergrad",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Validated"),
        ],
    },
    CapabilityRow {
        capability: "Planner & scheduler",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Async queues tuned"),
        ],
    },
    CapabilityRow {
        capability: "Telemetry",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Structured logging"),
            CapabilityEntry::with_state(CapabilityState::Ready, "GPU timelines"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Instruments via macOS unified logging",
            ),
            CapabilityEntry::with_state(CapabilityState::Ready, "CUPTI hooks planned"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Counter wiring in place"),
        ],
    },
    CapabilityRow {
        capability: "Python wheel support",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready (default build)"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Wheel audit complete"),
        ],
    },
    CapabilityRow {
        capability: "Kernel autotuning",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Parameter sweeps nightly"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Shader cache heuristics stabilized",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Convolution coverage complete",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Heuristic tuner with offline database",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Wavefront search stabilized",
            ),
        ],
    },
    CapabilityRow {
        capability: "Sparse tensor ops",
        entries: [
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "CSR kernels merged",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Subgroup atomics coverage complete",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Metal sparse pipeline primitives integrated",
            ),
            CapabilityEntry::with_state(CapabilityState::Ready, "CUSPARSE integration validated"),
            CapabilityEntry::with_state(CapabilityState::Ready, "ROCm sparse kernels merged"),
        ],
    },
    CapabilityRow {
        capability: "Quantized inference",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "INT8/BF16 calibrations stable"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Shader range calibrated"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Shader range calibration automated",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Metal Performance Shaders INT8 path enabled",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Tensor cores validated for INT8/BF16",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "rocWMMA quantized path upstreamed",
            ),
        ],
    },
    CapabilityRow {
        capability: "Mixed precision training",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "AMP via BF16 accumulation"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "FP16 gradient scaling tuned",
            ),
            CapabilityEntry::with_state(CapabilityState::Ready, "Metal AMP validated on A17"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Apex parity across optimizers"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Wavefront loss scaling optimized",
            ),
        ],
    },
    CapabilityRow {
        capability: "Dynamic shape compilation",
        entries: [
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Shape polymorphic kernels validated",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Runtime shape lowering stabilized",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Metal dynamic pipeline caching optimized",
            ),
            CapabilityEntry::with_state(CapabilityState::Ready, "NVRTC specialization stable"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "rocDynamic shape specialization merged",
            ),
        ],
    },
    CapabilityRow {
        capability: "Graph fusion pipeline",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Stable scheduler passes"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Texture graph fusion benchmarked",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Tile buffer heuristics tuned",
            ),
            CapabilityEntry::with_state(CapabilityState::Ready, "NVRTC fusion coverage nightly"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "ROC graph capture instrumentation complete",
            ),
        ],
    },
    CapabilityRow {
        capability: "ONNX export parity",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Parity score ≥ 0.9"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Dynamic shape operators covered",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Gradient suite expanded",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Validated nightly against reference ops",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Upstream complex kernel coverage achieved",
            ),
        ],
    },
    CapabilityRow {
        capability: "CI coverage",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Nightly smoke + perf matrix"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Weekly adapter matrix job green",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Weekly adapter matrix job green",
            ),
            CapabilityEntry::with_state(CapabilityState::Ready, "Nightly + gated release pipeline"),
            CapabilityEntry::with_state(
                CapabilityState::Ready,
                "Hardware allocation secured",
            ),
        ],
    },
];

/// Canonical capability matrix view mirroring the documentation table.
pub const CAPABILITY_MATRIX: CapabilityMatrix = CapabilityMatrix::new(CAPABILITY_ROWS);

/// Returns the backend feature matrix mirrored from `docs/backend_matrix.md`.
pub const fn capability_matrix_view() -> &'static CapabilityMatrix {
    &CAPABILITY_MATRIX
}

/// Returns the backend feature matrix mirrored from `docs/backend_matrix.md`.
pub fn capability_matrix() -> &'static [CapabilityRow] {
    CAPABILITY_MATRIX.rows()
}

/// Locates a capability row by name (case-insensitive).
pub fn capability_by_name(name: &str) -> Option<&'static CapabilityRow> {
    let query = name.trim();
    if query.is_empty() {
        return None;
    }

    CAPABILITY_ROWS
        .iter()
        .find(|row| row.capability.eq_ignore_ascii_case(query))
}

/// Produces a summary describing how `backend` fares across all capabilities.
pub fn summarize_backend(backend: Backend) -> BackendSummary {
    let mut summary = BackendSummary {
        backend,
        ready: 0,
        watchlist: 0,
        blocked: 0,
        notes: Vec::new(),
    };

    for row in CAPABILITY_ROWS {
        let entry = row.entry(backend);
        match entry.state {
            Some(CapabilityState::Ready) => summary.ready += 1,
            Some(CapabilityState::Watchlist) => {
                summary.watchlist += 1;
                summary.notes.push(BackendNote {
                    capability: row.capability,
                    note: entry.note,
                });
            }
            Some(CapabilityState::Blocked) => {
                summary.blocked += 1;
                summary.notes.push(BackendNote {
                    capability: row.capability,
                    note: entry.note,
                });
            }
            None => {
                if !entry.note.is_empty() {
                    summary.notes.push(BackendNote {
                        capability: row.capability,
                        note: entry.note,
                    });
                }
            }
        }
    }

    summary
}

/// Convenience helper returning summaries for every backend in the matrix.
pub fn backend_summaries() -> Vec<BackendSummary> {
    Backend::ALL
        .iter()
        .copied()
        .map(summarize_backend)
        .collect()
}

/// Convenience helper returning capability summaries for every row in the matrix.
pub fn capability_summaries() -> Vec<CapabilitySummary> {
    CAPABILITY_ROWS
        .iter()
        .map(CapabilityRow::summarize)
        .collect()
}

/// Returns capability rows that contain at least one backend marked with the requested state.
pub fn capabilities_with_state(state: CapabilityState) -> Vec<&'static CapabilityRow> {
    CAPABILITY_ROWS
        .iter()
        .filter(|row| row.entries.iter().any(|entry| entry.state == Some(state)))
        .collect()
}

/// Returns capability rows where `backend` is marked with the requested state.
pub fn capabilities_for_backend_with_state(
    backend: Backend,
    state: CapabilityState,
) -> Vec<&'static CapabilityRow> {
    CAPABILITY_ROWS
        .iter()
        .filter(|row| row.entry(backend).state == Some(state))
        .collect()
}

/// Returns capability rows whose notes mention `query` (case-insensitive).
pub fn capabilities_with_note_containing(query: &str) -> Vec<&'static CapabilityRow> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let needle = trimmed.to_ascii_lowercase();
    CAPABILITY_ROWS
        .iter()
        .filter(|row| {
            row.entries
                .iter()
                .any(|entry| entry.note.to_ascii_lowercase().contains(needle.as_str()))
        })
        .collect()
}

/// Returns capability rows where `backend` is not yet marked ready.
pub fn pending_capabilities_for_backend(backend: Backend) -> Vec<&'static CapabilityRow> {
    CAPABILITY_ROWS
        .iter()
        .filter(|row| match row.entry(backend).state {
            Some(CapabilityState::Ready) => false,
            Some(CapabilityState::Watchlist) | Some(CapabilityState::Blocked) => true,
            None => false,
        })
        .collect()
}

/// Backend summaries ordered by readiness ratio (descending) and ready count.
pub fn readiness_leaderboard() -> Vec<BackendSummary> {
    let mut summaries = backend_summaries();
    summaries.sort_by(|a, b| {
        b.readiness_ratio()
            .total_cmp(&a.readiness_ratio())
            .then(b.ready.cmp(&a.ready))
            .then_with(|| a.backend.as_str().cmp(b.backend.as_str()))
    });
    summaries
}

/// Serialises the matrix into a JSON value for downstream tooling.
pub fn capability_matrix_json() -> serde_json::Value {
    serde_json::to_value(CAPABILITY_ROWS).expect("matrix is serialisable")
}

/// Computes aggregate readiness counts across the entire matrix.
pub fn matrix_summary() -> MatrixSummary {
    let mut summary = MatrixSummary {
        ready: 0,
        watchlist: 0,
        blocked: 0,
        informational: 0,
    };

    for row in CAPABILITY_ROWS {
        for entry in &row.entries {
            match entry.state {
                Some(CapabilityState::Ready) => summary.ready += 1,
                Some(CapabilityState::Watchlist) => summary.watchlist += 1,
                Some(CapabilityState::Blocked) => summary.blocked += 1,
                None => summary.informational += 1,
            }
        }
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_contains_expected_rows() {
        let matrix = capability_matrix();
        assert!(matrix.iter().any(|row| row.capability == "Tensor ops"));
        let telemetry = matrix
            .iter()
            .find(|row| row.capability == "Telemetry")
            .expect("telemetry row present");
        let hip_entry = telemetry.entry(Backend::Hip);
        assert_eq!(hip_entry.state, Some(CapabilityState::Ready));
        assert!(hip_entry.note.contains("counter"));
    }

    #[test]
    fn backend_names_match_document() {
        let names: Vec<&str> = Backend::ALL
            .iter()
            .map(|backend| backend.as_str())
            .collect();
        assert_eq!(names, vec!["CPU", "WGPU", "MPS", "CUDA", "HIP / ROCm"]);
    }

    #[test]
    fn backend_parsing_is_case_insensitive() {
        assert_eq!(Backend::from_str("cuda"), Some(Backend::Cuda));
        assert_eq!(Backend::from_str(" HIP / ROCM"), Some(Backend::Hip));
        assert_eq!(Backend::from_str("unknown"), None);
    }

    #[test]
    fn summarizes_backend_reports_full_readiness() {
        let hip = summarize_backend(Backend::Hip);
        assert_eq!(hip.backend, Backend::Hip);
        assert_eq!(hip.watchlist, 0);
        assert_eq!(hip.blocked, 0);
        assert!(hip.is_fully_ready());
        assert_eq!(hip.tracked_capabilities(), 13);
        assert_eq!(hip.ready, 13);
        assert_eq!(hip.readiness_ratio(), 1.0);
        assert_eq!(hip.pending(), 0);
        assert!(hip.notes.is_empty());
    }

    #[test]
    fn matrix_serialises_to_json() {
        let value = capability_matrix_json();
        let rows = value.as_array().expect("array of rows");
        let first = rows
            .iter()
            .find(|row| row["capability"].as_str() == Some("Telemetry"))
            .expect("telemetry row serialized");
        let hip_note = first["entries"][Backend::Hip.index()]["note"].as_str();
        assert!(hip_note.expect("hip note present").contains("counter"));
    }

    #[test]
    fn capability_summaries_reflect_full_readiness() {
        let summaries = capability_summaries();
        let onnx = summaries
            .iter()
            .find(|summary| summary.capability == "ONNX export parity")
            .expect("onnx row present");
        assert_eq!(onnx.ready, 5);
        assert_eq!(onnx.watchlist, 0);
        assert_eq!(onnx.blocked, 0);
        assert_eq!(onnx.tracked_backends(), 5);
        assert_eq!(onnx.dominant_state(), Some(CapabilityState::Ready));

        let ci = summaries
            .iter()
            .find(|summary| summary.capability == "CI coverage")
            .expect("ci row present");
        assert_eq!(ci.ready, 5);
        assert_eq!(ci.watchlist, 0);
        assert_eq!(ci.blocked, 0);
        assert_eq!(ci.dominant_state(), Some(CapabilityState::Ready));
    }

    #[test]
    fn capabilities_with_state_filters_rows() {
        let blocked_rows = capabilities_with_state(CapabilityState::Blocked);
        assert!(blocked_rows.is_empty());

        let watchlist_rows = capabilities_with_state(CapabilityState::Watchlist);
        assert!(watchlist_rows.is_empty());

        let ready_rows = capabilities_with_state(CapabilityState::Ready);
        assert!(ready_rows.iter().any(|row| row.capability == "Tensor ops"));
    }

    #[test]
    fn capabilities_for_backend_with_state_lists_pending_items() {
        let hip_watchlist =
            capabilities_for_backend_with_state(Backend::Hip, CapabilityState::Watchlist);
        assert!(hip_watchlist.is_empty());

        let cuda_blocked =
            capabilities_for_backend_with_state(Backend::Cuda, CapabilityState::Blocked);
        assert!(cuda_blocked.is_empty());

        let hip_ready = capabilities_for_backend_with_state(Backend::Hip, CapabilityState::Ready);
        assert_eq!(
            hip_ready.len(),
            hip_ready
                .iter()
                .filter(|row| row.entry(Backend::Hip).state == Some(CapabilityState::Ready))
                .count()
        );
    }

    #[test]
    fn capabilities_with_note_containing_is_case_insensitive() {
        let wavefront = capabilities_with_note_containing("wavefront");
        assert!(wavefront
            .iter()
            .any(|row| row.capability == "Kernel autotuning"));
        assert!(wavefront
            .iter()
            .any(|row| row.capability == "Mixed precision training"));

        assert!(capabilities_with_note_containing(" ").is_empty());
    }

    #[test]
    fn matrix_summary_matches_manual_counts() {
        let summary = matrix_summary();
        let mut ready = 0usize;
        let mut watchlist = 0usize;
        let mut blocked = 0usize;
        let mut informational = 0usize;

        for row in capability_matrix() {
            for entry in &row.entries {
                match entry.state {
                    Some(CapabilityState::Ready) => ready += 1,
                    Some(CapabilityState::Watchlist) => watchlist += 1,
                    Some(CapabilityState::Blocked) => blocked += 1,
                    None => informational += 1,
                }
            }
        }

        assert_eq!(summary.ready, ready);
        assert_eq!(summary.watchlist, watchlist);
        assert_eq!(summary.blocked, blocked);
        assert_eq!(summary.informational, informational);
        assert_eq!(
            summary.total_entries(),
            ready + watchlist + blocked + informational
        );
        if summary.tracked_entries() > 0 {
            let expected_ratio = ready as f32 / summary.tracked_entries() as f32;
            assert!((summary.readiness_ratio() - expected_ratio).abs() <= f32::EPSILON);
        } else {
            assert_eq!(summary.readiness_ratio(), 0.0);
        }
    }

    #[test]
    fn capability_matrix_view_matches_slice() {
        let view = capability_matrix_view();
        assert_eq!(view.rows().len(), capability_matrix().len());
        assert!(view
            .capability("Telemetry")
            .is_some_and(|row| row.capability == "Telemetry"));
    }

    #[test]
    fn capability_matrix_view_provides_note_search() {
        let view = capability_matrix_view();
        let matches = view.capabilities_with_note("dynamic shape");
        assert!(matches
            .iter()
            .any(|row| row.capability == "Dynamic shape compilation"));
    }

    #[test]
    fn pending_capabilities_collects_watchlist_and_blocked() {
        let hip_pending = pending_capabilities_for_backend(Backend::Hip);
        assert!(hip_pending.is_empty());
    }

    #[test]
    fn readiness_leaderboard_prefers_higher_ratios() {
        let leaderboard = readiness_leaderboard();
        assert_eq!(leaderboard.len(), Backend::COUNT);
        assert_eq!(leaderboard.first().expect("non-empty").backend, Backend::Cpu);
        assert!(leaderboard
            .windows(2)
            .all(|pair| pair[0].readiness_ratio() >= pair[1].readiness_ratio()));
    }
}
