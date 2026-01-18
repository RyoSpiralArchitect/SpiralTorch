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
    Planned,
    Stub,
}

impl CapabilityState {
    /// Returns the emoji marker used in the Markdown table.
    pub const fn marker(self) -> &'static str {
        match self {
            CapabilityState::Ready => "✅",
            CapabilityState::Planned => "⚠️",
            CapabilityState::Stub => "❌",
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
        let mut planned = 0;
        let mut stub = 0;
        let mut idx = 0;
        while idx < Backend::COUNT {
            let state = self.entries[idx].state;
            match state {
                Some(CapabilityState::Ready) => ready += 1,
                Some(CapabilityState::Planned) => planned += 1,
                Some(CapabilityState::Stub) => stub += 1,
                None => {}
            }
            idx += 1;
        }

        CapabilitySummary {
            capability: self.capability,
            ready,
            planned,
            stub,
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
    /// Number of capabilities that are planned but not yet complete.
    pub planned: usize,
    /// Number of capabilities that are currently stubbed/unwired.
    pub stub: usize,
    /// Collected notes for non-ready capabilities and informational rows.
    pub notes: Vec<BackendNote>,
}

impl BackendSummary {
    /// Returns `true` if the backend has no planned or stubbed capabilities.
    pub fn is_fully_ready(&self) -> bool {
        self.planned == 0 && self.stub == 0
    }

    /// Number of capabilities that carry a readiness marker for this backend.
    pub fn tracked_capabilities(&self) -> usize {
        self.ready + self.planned + self.stub
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
        self.planned + self.stub
    }
}

/// Aggregate counters describing the entire backend matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MatrixSummary {
    /// Number of entries marked ready across the matrix.
    pub ready: usize,
    /// Number of entries tracked as planned.
    pub planned: usize,
    /// Number of entries tracked as stubbed.
    pub stub: usize,
    /// Number of informational entries without readiness markers.
    pub informational: usize,
}

impl MatrixSummary {
    /// Number of entries that have an explicit readiness marker.
    pub const fn tracked_entries(&self) -> usize {
        self.ready + self.planned + self.stub
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
    /// Number of backends tracked as planned.
    pub planned: usize,
    /// Number of backends tracked as stubbed.
    pub stub: usize,
}

impl CapabilitySummary {
    /// Total number of backends that have an explicit readiness marker for the capability.
    pub const fn tracked_backends(&self) -> usize {
        self.ready + self.planned + self.stub
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

        let planned = self.planned;
        if planned > max {
            max = planned;
            ties = 1;
            candidate = Some(CapabilityState::Planned);
        } else if planned != 0 && planned == max {
            ties += 1;
        }

        let stub = self.stub;
        if stub > max {
            max = stub;
            ties = 1;
            candidate = Some(CapabilityState::Stub);
        } else if stub != 0 && stub == max {
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
            CapabilityEntry::with_state(CapabilityState::Ready, "Full (cpu/faer)"),
            CapabilityEntry::with_state(CapabilityState::Ready, "WGPU dense + frac kernels"),
            CapabilityEntry::with_state(
                CapabilityState::Stub,
                "Feature placeholder (no kernels wired)",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Stub,
                "Feature placeholder (no kernels wired)",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Planned,
                "hip GEMM (matmul); extend op coverage",
            ),
        ],
    },
    CapabilityRow {
        capability: "Autodiff / hypergrad",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(
                CapabilityState::Planned,
                "Validate tapes with WGPU execution",
            ),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(
                CapabilityState::Planned,
                "Validate tapes with HIP execution",
            ),
        ],
    },
    CapabilityRow {
        capability: "Planner & scheduler",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready (backend-agnostic)"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready (backend-agnostic)"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready (backend-agnostic)"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready (backend-agnostic)"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready (backend-agnostic)"),
        ],
    },
    CapabilityRow {
        capability: "Telemetry",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Tracing + structured logging"),
            CapabilityEntry::with_state(CapabilityState::Planned, "GPU timing hooks planned"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Planned, "CUPTI hooks not wired"),
            CapabilityEntry::with_state(CapabilityState::Planned, "ROCm counters pending"),
        ],
    },
    CapabilityRow {
        capability: "Python wheel support",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready (default build)"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Feature placeholder"),
            CapabilityEntry::with_state(
                CapabilityState::Planned,
                "Requires CUDA toolchain build",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Planned,
                "Requires ROCm toolchain build",
            ),
        ],
    },
    CapabilityRow {
        capability: "Kernel autotuning",
        entries: [
            CapabilityEntry::with_state(
                CapabilityState::Planned,
                "CPU tiling heuristics (faer + autotune)",
            ),
            CapabilityEntry::with_state(CapabilityState::Planned, "Shader cache heuristics"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
        ],
    },
    CapabilityRow {
        capability: "Sparse tensor ops",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Stub, "Not implemented"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Not implemented"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Not implemented"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Not implemented"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Not implemented"),
        ],
    },
    CapabilityRow {
        capability: "Quantized inference",
        entries: [
            CapabilityEntry::with_state(
                CapabilityState::Planned,
                "i8 matmul path present; validate end-to-end",
            ),
            CapabilityEntry::with_state(
                CapabilityState::Planned,
                "int8 kernels present; validate end-to-end",
            ),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
        ],
    },
    CapabilityRow {
        capability: "Mixed precision training",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Planned, "BF16/FP16 roadmap"),
            CapabilityEntry::with_state(CapabilityState::Planned, "wgpu_f16 feature (validate)"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
        ],
    },
    CapabilityRow {
        capability: "Dynamic shape compilation",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Planned, "Planned"),
            CapabilityEntry::with_state(CapabilityState::Planned, "Planned"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
        ],
    },
    CapabilityRow {
        capability: "Graph fusion pipeline",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Planned, "Planned"),
            CapabilityEntry::with_state(CapabilityState::Planned, "Planned"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Backend placeholder"),
        ],
    },
    CapabilityRow {
        capability: "ONNX export parity",
        entries: [
            CapabilityEntry::with_state(
                CapabilityState::Planned,
                "Export scaffolding (JSON artefacts); ONNX pending",
            ),
            CapabilityEntry::with_state(CapabilityState::Stub, "Not implemented"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Not implemented"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Not implemented"),
            CapabilityEntry::with_state(CapabilityState::Stub, "Not implemented"),
        ],
    },
    CapabilityRow {
        capability: "CI coverage",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Planned, "Unit tests + docs checks"),
            CapabilityEntry::with_state(CapabilityState::Planned, "GPU CI planned"),
            CapabilityEntry::with_state(CapabilityState::Stub, "No CI"),
            CapabilityEntry::with_state(CapabilityState::Stub, "No CI"),
            CapabilityEntry::with_state(CapabilityState::Stub, "No CI"),
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
        planned: 0,
        stub: 0,
        notes: Vec::new(),
    };

    for row in CAPABILITY_ROWS {
        let entry = row.entry(backend);
        match entry.state {
            Some(CapabilityState::Ready) => summary.ready += 1,
            Some(CapabilityState::Planned) => {
                summary.planned += 1;
                summary.notes.push(BackendNote {
                    capability: row.capability,
                    note: entry.note,
                });
            }
            Some(CapabilityState::Stub) => {
                summary.stub += 1;
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
            Some(CapabilityState::Planned) | Some(CapabilityState::Stub) => true,
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

const BACKEND_MATRIX_AUTOGEN_BEGIN: &str = "<!-- AUTOGEN:BEGIN backend-matrix -->";
const BACKEND_MATRIX_AUTOGEN_END: &str = "<!-- AUTOGEN:END backend-matrix -->";

fn backend_doc_heading(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "CPU (default)",
        Backend::Wgpu => "WGPU",
        Backend::Mps => "MPS",
        Backend::Cuda => "CUDA",
        Backend::Hip => "HIP / ROCm",
    }
}

/// Render the capability matrix as a Markdown table.
pub fn backend_matrix_markdown_table() -> String {
    let headers = [
        "Capability",
        backend_doc_heading(Backend::Cpu),
        backend_doc_heading(Backend::Wgpu),
        backend_doc_heading(Backend::Mps),
        backend_doc_heading(Backend::Cuda),
        backend_doc_heading(Backend::Hip),
    ];

    let mut out = String::new();
    out.push_str("| ");
    out.push_str(&headers.join(" | "));
    out.push_str(" |\n");
    out.push_str("| ");
    out.push_str(&headers.iter().map(|_| "---").collect::<Vec<_>>().join(" | "));
    out.push_str(" |\n");

    for row in CAPABILITY_ROWS {
        out.push_str("| ");
        out.push_str(row.capability);
        for backend in Backend::ALL {
            let entry = row.entry(backend);
            out.push_str(" | ");
            match entry.state {
                Some(state) => {
                    out.push_str(state.marker());
                    if !entry.note.is_empty() {
                        out.push(' ');
                        out.push_str(entry.note);
                    }
                }
                None => out.push_str(entry.note),
            }
        }
        out.push_str(" |\n");
    }

    out
}

/// Render the autogen block inserted into `docs/backend_matrix.md`.
pub fn backend_matrix_autogen_block() -> String {
    let mut out = String::new();
    out.push_str(BACKEND_MATRIX_AUTOGEN_BEGIN);
    out.push('\n');
    out.push_str(&backend_matrix_markdown_table());
    out.push_str(BACKEND_MATRIX_AUTOGEN_END);
    out.push('\n');
    out
}

/// Replace the autogen block inside `docs/backend_matrix.md`.
pub fn sync_backend_matrix_markdown(doc: &str) -> Result<String, String> {
    let begin = doc
        .find(BACKEND_MATRIX_AUTOGEN_BEGIN)
        .ok_or_else(|| format!("missing {BACKEND_MATRIX_AUTOGEN_BEGIN}"))?;
    let end = doc
        .find(BACKEND_MATRIX_AUTOGEN_END)
        .ok_or_else(|| format!("missing {BACKEND_MATRIX_AUTOGEN_END}"))?;
    if end < begin {
        return Err("backend matrix autogen markers are out of order".to_string());
    }

    let mut end_after = end + BACKEND_MATRIX_AUTOGEN_END.len();
    if doc[end_after..].starts_with("\r\n") {
        end_after += 2;
    } else if doc[end_after..].starts_with('\n') {
        end_after += 1;
    }
    let mut out = String::new();
    out.push_str(&doc[..begin]);
    out.push_str(&backend_matrix_autogen_block());
    out.push_str(&doc[end_after..]);
    Ok(out)
}

/// Computes aggregate readiness counts across the entire matrix.
pub fn matrix_summary() -> MatrixSummary {
    let mut summary = MatrixSummary {
        ready: 0,
        planned: 0,
        stub: 0,
        informational: 0,
    };

    for row in CAPABILITY_ROWS {
        for entry in &row.entries {
            match entry.state {
                Some(CapabilityState::Ready) => summary.ready += 1,
                Some(CapabilityState::Planned) => summary.planned += 1,
                Some(CapabilityState::Stub) => summary.stub += 1,
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
        assert_eq!(hip_entry.state, Some(CapabilityState::Planned));
        assert_eq!(hip_entry.note, "ROCm counters pending");
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
    fn summarizes_backend_reports_counts_and_notes() {
        let cpu = summarize_backend(Backend::Cpu);
        assert_eq!(cpu.backend, Backend::Cpu);
        assert_eq!(cpu.tracked_capabilities(), 13);
        assert_eq!(cpu.ready + cpu.planned + cpu.stub, cpu.tracked_capabilities());
        assert!(cpu.ready > 0);
        assert!(!cpu.is_fully_ready());
        assert!(cpu.notes.iter().any(|note| {
            note.capability == "Build flag" && note.note == "_none_"
        }));
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
        assert_eq!(hip_note.expect("hip note present"), "ROCm counters pending");
    }

    #[test]
    fn capability_summaries_reflect_state_counts() {
        let summaries = capability_summaries();
        let onnx = summaries
            .iter()
            .find(|summary| summary.capability == "ONNX export parity")
            .expect("onnx row present");
        assert_eq!(onnx.ready, 0);
        assert_eq!(onnx.planned, 1);
        assert_eq!(onnx.stub, 4);
        assert_eq!(onnx.tracked_backends(), 5);
        assert_eq!(onnx.dominant_state(), Some(CapabilityState::Stub));

        let ci = summaries
            .iter()
            .find(|summary| summary.capability == "CI coverage")
            .expect("ci row present");
        assert_eq!(ci.ready, 0);
        assert_eq!(ci.planned, 2);
        assert_eq!(ci.stub, 3);
        assert_eq!(ci.dominant_state(), Some(CapabilityState::Stub));
    }

    #[test]
    fn capabilities_with_state_filters_rows() {
        let stub_rows = capabilities_with_state(CapabilityState::Stub);
        assert!(stub_rows.iter().any(|row| row.capability == "Sparse tensor ops"));

        let planned_rows = capabilities_with_state(CapabilityState::Planned);
        assert!(planned_rows
            .iter()
            .any(|row| row.capability == "Kernel autotuning"));

        let ready_rows = capabilities_with_state(CapabilityState::Ready);
        assert!(ready_rows.iter().any(|row| row.capability == "Tensor ops"));
    }

    #[test]
    fn capabilities_for_backend_with_state_lists_pending_items() {
        let hip_planned =
            capabilities_for_backend_with_state(Backend::Hip, CapabilityState::Planned);
        assert!(hip_planned.iter().any(|row| row.capability == "Telemetry"));

        let cuda_stub = capabilities_for_backend_with_state(Backend::Cuda, CapabilityState::Stub);
        assert!(cuda_stub.iter().any(|row| row.capability == "Tensor ops"));

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
        let autotune = capabilities_with_note_containing("AUTOTUNE");
        assert!(autotune
            .iter()
            .any(|row| row.capability == "Kernel autotuning"));

        let placeholders = capabilities_with_note_containing("backend placeholder");
        assert!(!placeholders.is_empty());

        assert!(capabilities_with_note_containing(" ").is_empty());
    }

    #[test]
    fn matrix_summary_matches_manual_counts() {
        let summary = matrix_summary();
        let mut ready = 0usize;
        let mut planned = 0usize;
        let mut stub = 0usize;
        let mut informational = 0usize;

        for row in capability_matrix() {
            for entry in &row.entries {
                match entry.state {
                    Some(CapabilityState::Ready) => ready += 1,
                    Some(CapabilityState::Planned) => planned += 1,
                    Some(CapabilityState::Stub) => stub += 1,
                    None => informational += 1,
                }
            }
        }

        assert_eq!(summary.ready, ready);
        assert_eq!(summary.planned, planned);
        assert_eq!(summary.stub, stub);
        assert_eq!(summary.informational, informational);
        assert_eq!(
            summary.total_entries(),
            ready + planned + stub + informational
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
        let matches = view.capabilities_with_note("json artefacts");
        assert!(matches.iter().any(|row| row.capability == "ONNX export parity"));
    }

    #[test]
    fn pending_capabilities_collects_planned_and_stub() {
        let hip_pending = pending_capabilities_for_backend(Backend::Hip);
        assert!(hip_pending.iter().any(|row| row.capability == "Telemetry"));
    }

    #[test]
    fn readiness_leaderboard_prefers_higher_ratios() {
        let leaderboard = readiness_leaderboard();
        assert_eq!(leaderboard.len(), Backend::COUNT);
        assert_eq!(
            leaderboard.first().expect("non-empty").backend,
            Backend::Cpu
        );
        assert!(leaderboard
            .windows(2)
            .all(|pair| pair[0].readiness_ratio() >= pair[1].readiness_ratio()));
    }
}
