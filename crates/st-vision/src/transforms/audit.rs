// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Metadata describing the current CPU transform surface.

/// High-level transform groupings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransformCategory {
    /// Brightness/contrast/colour space modifications.
    Photometric,
    /// Spatial resampling or cropping operations.
    Geometric,
    /// Normalisation or statistics driven transforms.
    Normalisation,
}

/// Summary of a CPU transform implementation and whether a GPU path exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransformAuditEntry {
    pub name: &'static str,
    pub category: TransformCategory,
    pub gpu_candidate: bool,
    pub gpu_available: bool,
    pub notes: &'static str,
}

/// Return the current set of CPU transforms along with GPU coverage metadata.
pub fn audit_cpu_transforms() -> Vec<TransformAuditEntry> {
    vec![
        TransformAuditEntry {
            name: "Normalize",
            category: TransformCategory::Normalisation,
            gpu_candidate: false,
            gpu_available: false,
            notes: "Stateless per-channel statistics are cheap on CPU; GPU path optional.",
        },
        TransformAuditEntry {
            name: "Resize",
            category: TransformCategory::Geometric,
            gpu_candidate: true,
            gpu_available: true,
            notes: "Bilinear sampler benefits from parallel threads for large batches.",
        },
        TransformAuditEntry {
            name: "CenterCrop",
            category: TransformCategory::Geometric,
            gpu_candidate: true,
            gpu_available: true,
            notes: "Simple index mapping suited for compute shader dispatch.",
        },
        TransformAuditEntry {
            name: "RandomHorizontalFlip",
            category: TransformCategory::Geometric,
            gpu_candidate: true,
            gpu_available: true,
            notes: "Branch-free swizzle is a natural GPU candidate when rng masks are precomputed.",
        },
        TransformAuditEntry {
            name: "ColorJitter",
            category: TransformCategory::Photometric,
            gpu_candidate: true,
            gpu_available: true,
            notes: "Per-pixel colour transforms map directly to compute pipelines.",
        },
    ]
}
