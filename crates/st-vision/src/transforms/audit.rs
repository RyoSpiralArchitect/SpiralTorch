// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Metadata describing the current CPU transform surface.

use std::collections::HashMap;

#[cfg(test)]
use crate::TransformOperation;
use crate::TransformPipeline;

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

/// Report describing a particular pipeline instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineAuditReport {
    pub stages: Vec<TransformAuditEntry>,
    pub unknown_operations: Vec<String>,
    pub has_gpu_dispatcher: bool,
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

/// Audits a [`TransformPipeline`] returning metadata for each stage.
pub fn audit_pipeline(pipeline: &TransformPipeline) -> PipelineAuditReport {
    let registry: HashMap<&'static str, TransformAuditEntry> = audit_cpu_transforms()
        .into_iter()
        .map(|entry| (entry.name, entry))
        .collect();
    let mut stages = Vec::new();
    let mut unknown_operations = Vec::new();
    for op in pipeline.operations() {
        let name = op.name();
        match registry.get(name) {
            Some(entry) => stages.push(entry.clone()),
            None => unknown_operations.push(name.to_string()),
        }
    }
    PipelineAuditReport {
        stages,
        unknown_operations,
        has_gpu_dispatcher: pipeline.has_gpu_dispatcher(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Normalize, Resize};

    #[test]
    fn pipeline_audit_matches_registered_ops() {
        let mut pipeline = TransformPipeline::with_seed(7);
        pipeline
            .add(TransformOperation::Normalize(
                Normalize::new(vec![0.5], vec![0.5]).unwrap(),
            ))
            .add(TransformOperation::Resize(Resize::new(2, 2).unwrap()));
        let report = audit_pipeline(&pipeline);
        assert_eq!(report.stages.len(), 2);
        assert!(report.unknown_operations.is_empty());
        assert!(!report.has_gpu_dispatcher);
        assert!(report.stages.iter().any(|entry| entry.name == "Normalize"));
        assert!(report.stages.iter().any(|entry| entry.name == "Resize"));
    }
}
