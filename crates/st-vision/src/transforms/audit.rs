// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Metadata describing the current CPU transform surface.

use std::collections::{BTreeMap, HashMap};

#[cfg(test)]
use crate::TransformOperation;
use crate::TransformPipeline;

/// High-level transform groupings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

/// Aggregate summary of transform coverage across pipelines.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TransformCoverageSummary {
    pub by_category: BTreeMap<TransformCategory, usize>,
    pub gpu_supported: usize,
    pub gpu_candidates_missing: usize,
    pub total_stages: usize,
    pub unknown_operations: usize,
}

impl TransformCoverageSummary {
    fn record_stage(&mut self, entry: &TransformAuditEntry) {
        *self.by_category.entry(entry.category).or_insert(0) += 1;
        self.total_stages += 1;
        if entry.gpu_available {
            self.gpu_supported += 1;
        } else if entry.gpu_candidate {
            self.gpu_candidates_missing += 1;
        }
    }

    fn record_unknown(&mut self, count: usize) {
        self.unknown_operations += count;
    }

    /// Builds a coverage summary from a single pipeline report.
    pub fn from_report(report: &PipelineAuditReport) -> Self {
        let mut summary = Self::default();
        for stage in &report.stages {
            summary.record_stage(stage);
        }
        summary.record_unknown(report.unknown_operations.len());
        summary
    }

    /// Merges another summary into the current aggregate.
    pub fn merge(&mut self, other: &Self) {
        for (category, count) in &other.by_category {
            *self.by_category.entry(*category).or_insert(0) += count;
        }
        self.gpu_supported += other.gpu_supported;
        self.gpu_candidates_missing += other.gpu_candidates_missing;
        self.total_stages += other.total_stages;
        self.unknown_operations += other.unknown_operations;
    }
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

/// Summarises a single pipeline audit into coverage statistics.
pub fn summarize_pipeline(report: &PipelineAuditReport) -> TransformCoverageSummary {
    TransformCoverageSummary::from_report(report)
}

/// Aggregates multiple pipeline reports into a combined summary.
pub fn summarize_pipelines(reports: &[PipelineAuditReport]) -> TransformCoverageSummary {
    let mut summary = TransformCoverageSummary::default();
    for report in reports {
        let local = TransformCoverageSummary::from_report(report);
        summary.merge(&local);
    }
    summary
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

    #[test]
    fn summarize_pipeline_counts_gpu_support() {
        let mut pipeline = TransformPipeline::with_seed(11);
        pipeline
            .add(TransformOperation::Normalize(
                Normalize::new(vec![0.0], vec![1.0]).unwrap(),
            ))
            .add(TransformOperation::Resize(Resize::new(4, 4).unwrap()));
        let report = audit_pipeline(&pipeline);
        let summary = summarize_pipeline(&report);
        assert_eq!(summary.total_stages, 2);
        assert_eq!(
            summary
                .by_category
                .get(&TransformCategory::Normalisation)
                .copied(),
            Some(1)
        );
        assert_eq!(
            summary
                .by_category
                .get(&TransformCategory::Geometric)
                .copied(),
            Some(1)
        );
        assert_eq!(summary.gpu_supported, 1);
        assert_eq!(summary.gpu_candidates_missing, 0);
        assert_eq!(summary.unknown_operations, 0);
    }

    #[test]
    fn summarize_pipelines_accumulates_unknowns() {
        let mut pipeline = TransformPipeline::with_seed(3);
        pipeline.add(TransformOperation::Normalize(
            Normalize::new(vec![0.25], vec![0.25]).unwrap(),
        ));
        let known_report = audit_pipeline(&pipeline);
        let mut unknown_report = known_report.clone();
        unknown_report.stages.clear();
        unknown_report.unknown_operations = vec!["Custom".into(), "Other".into()];
        let summary = summarize_pipelines(&[known_report.clone(), unknown_report.clone()]);
        assert_eq!(summary.total_stages, known_report.stages.len());
        assert_eq!(
            summary.unknown_operations,
            unknown_report.unknown_operations.len()
        );
        assert_eq!(
            summary
                .by_category
                .get(&TransformCategory::Normalisation)
                .copied(),
            Some(1)
        );
        assert_eq!(summary.gpu_supported, 0);
        assert_eq!(summary.gpu_candidates_missing, 0);
    }
}
