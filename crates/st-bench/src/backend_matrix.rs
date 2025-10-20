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

/// Qualitative readiness level used in the documentation matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityRow {
    /// Capability name as rendered in the first column of the table.
    pub capability: &'static str,
    /// Matrix entries ordered using [`Backend::ALL`].
    pub entries: [CapabilityEntry; Backend::COUNT],
}

impl CapabilityRow {
    /// Retrieve the entry for `backend`.
    pub const fn entry(&self, backend: Backend) -> &CapabilityEntry {
        &self.entries[backend.index()]
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
            CapabilityEntry::with_state(CapabilityState::Watchlist, "Incomplete complex kernels"),
        ],
    },
    CapabilityRow {
        capability: "Autodiff / hypergrad",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Watchlist, "Requires additional testing"),
        ],
    },
    CapabilityRow {
        capability: "Planner & scheduler",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Watchlist, "Needs async queue profiling"),
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
            CapabilityEntry::with_state(CapabilityState::Watchlist, "Pending counter wiring"),
        ],
    },
    CapabilityRow {
        capability: "Python wheel support",
        entries: [
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready (default build)"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Ready, "Ready"),
            CapabilityEntry::with_state(CapabilityState::Watchlist, "Needs wheel audit"),
        ],
    },
];

/// Returns the backend feature matrix mirrored from `docs/backend_matrix.md`.
pub fn capability_matrix() -> &'static [CapabilityRow] {
    CAPABILITY_ROWS
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
        assert_eq!(hip_entry.state, Some(CapabilityState::Watchlist));
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
}
