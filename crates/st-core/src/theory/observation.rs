// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Microlocal → macrolocal observation bridge combining coalgebra diagnostics
//! with macro template feedback.
//!
//! The bridge inspects [`InterfaceZReport`] records emitted by the microlocal
//! conductor, counts how many unique gauges and scale slices participate in the
//! report, and pushes those counts through the [`ObservationalCoalgebra`].  The
//! resulting [`ObservabilityAssessment`] exposes the efficiency `η` of each
//! depth relative to the theoretical Pólya upper bound while the matched macro
//! templates contribute their [`MicrolocalFeedback`] payloads.  Downstream
//! controllers can therefore reason about structural observability and
//! macroscopic feedback using a single snapshot.

use std::collections::HashSet;

use crate::telemetry::hub::SoftlogicZFeedback;
use crate::theory::macro_model::{MacroDrive, MacroTemplateBank};
use crate::theory::microlocal::{InterfaceZReport, MicrolocalFeedback};
use crate::theory::microlocal_bank::GaugeBank;
use crate::theory::observability::{
    ObservabilityAssessment, ObservabilityConfig, ObservationalCoalgebra,
};

/// Snapshot returned after fusing microlocal telemetry with observation
/// diagnostics.
#[derive(Clone, Debug)]
pub struct ObservationBridgeSnapshot {
    /// Observed counts per depth (root, gauge ids, scale slices, macro drives).
    pub observed_counts: Vec<u128>,
    /// Comparison between the observed counts and the theoretical upper bound.
    pub assessment: ObservabilityAssessment,
    /// Aggregated microlocal feedback emitted by the matched macro templates.
    pub microlocal_feedback: Option<MicrolocalFeedback>,
    /// Macro drives keyed by the template identifier.
    pub drives: GaugeBank<MacroDrive>,
    /// Softlogic feedback emitted by the microlocal conductor.
    pub softlogic_feedback: SoftlogicZFeedback,
}

impl ObservationBridgeSnapshot {
    /// Returns `true` when at least one interface was detected at the microlocal level.
    pub fn has_interface(&self) -> bool {
        self.observed_counts.first().copied().unwrap_or(0) > 0
    }
}

/// Couples microlocal interface reports with observation coalgebra analytics
/// and macro template feedback.
#[derive(Clone, Debug)]
pub struct ObservationBridge {
    coalgebra: ObservationalCoalgebra,
    templates: MacroTemplateBank,
}

impl ObservationBridge {
    /// Builds a bridge from an [`ObservabilityConfig`] and an optional template bank.
    pub fn with_templates(config: ObservabilityConfig, templates: MacroTemplateBank) -> Self {
        Self {
            coalgebra: ObservationalCoalgebra::new(config),
            templates,
        }
    }

    /// Builds a bridge with an empty [`MacroTemplateBank`].
    pub fn new(config: ObservabilityConfig) -> Self {
        Self::with_templates(config, MacroTemplateBank::new())
    }

    /// Provides immutable access to the registered macro templates.
    pub fn templates(&self) -> &MacroTemplateBank {
        &self.templates
    }

    /// Provides mutable access to the registered macro templates.
    pub fn templates_mut(&mut self) -> &mut MacroTemplateBank {
        &mut self.templates
    }

    /// Ingests a microlocal [`InterfaceZReport`] and produces a fused snapshot.
    pub fn ingest(&mut self, report: &InterfaceZReport) -> ObservationBridgeSnapshot {
        let drives = self.templates.drive_matched(report);
        let microlocal_feedback = merge_feedback(&drives);
        let counts = observed_counts(report, &drives);
        let assessment = self.coalgebra.assess(&counts);

        ObservationBridgeSnapshot {
            observed_counts: counts,
            assessment,
            microlocal_feedback,
            drives,
            softlogic_feedback: report.feedback.clone(),
        }
    }
}

fn merge_feedback(drives: &GaugeBank<MacroDrive>) -> Option<MicrolocalFeedback> {
    let mut combined: Option<MicrolocalFeedback> = None;
    for (_, drive) in drives.iter() {
        let feedback = drive.microlocal_feedback().clone();
        combined = Some(match combined {
            Some(existing) => existing.merge(&feedback),
            None => feedback,
        });
    }
    combined
}

fn observed_counts(report: &InterfaceZReport, drives: &GaugeBank<MacroDrive>) -> Vec<u128> {
    let root = if report.has_interface() { 1 } else { 0 };
    let mut gauge_ids: HashSet<String> = HashSet::new();
    let mut scale_slots: HashSet<(String, isize, u32)> = HashSet::new();

    for (index, signature) in report.signatures.iter().enumerate() {
        if !signature.has_interface() {
            continue;
        }
        let id = report
            .gauge_id(index)
            .map(|value| value.to_string())
            .unwrap_or_else(|| format!("gauge#{index}"));
        gauge_ids.insert(id.clone());
        scale_slots.insert((id, signature.radius, signature.physical_radius.to_bits()));
    }

    vec![
        root,
        gauge_ids.len() as u128,
        scale_slots.len() as u128,
        drives.len() as u128,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::hub::SoftlogicZFeedback;
    use crate::theory::macro_model::{MacroCard, MacroTemplateBank, MinimalCardConfig, PhasePair};
    use crate::theory::microlocal::{
        InterfaceSignature, InterfaceZFused, InterfaceZLift, InterfaceZPulse,
    };
    use crate::theory::observability::{ObservabilityConfig, SlotSymmetry};
    use crate::theory::zpulse::{ZPulse, ZScale, ZSource, ZSupport};
    use crate::util::math::LeechProjector;
    use ndarray::{ArrayD, IxDyn};

    fn sample_signature(active: bool) -> InterfaceSignature {
        let shape = IxDyn(&[1]);
        let r_value = if active { 1.0 } else { 0.0 };
        InterfaceSignature {
            r_machine: ArrayD::from_elem(shape.clone(), r_value),
            raw_density: ArrayD::from_elem(shape.clone(), 0.5),
            perimeter_density: ArrayD::from_elem(shape.clone(), 1.0),
            mean_curvature: ArrayD::from_elem(shape.clone(), 0.25),
            signed_mean_curvature: None,
            orientation: None,
            kappa_d: 1.0,
            radius: 1,
            physical_radius: 1.0,
        }
    }

    fn sample_pulse() -> InterfaceZPulse {
        let mut pulse = InterfaceZPulse::default();
        pulse.support = 1.0;
        pulse.interface_cells = 1.0;
        pulse.band_energy = (0.2, 0.6, 0.2);
        pulse.scale = ZScale::new(1.0);
        pulse.drift = 0.1;
        pulse.z_bias = 0.05;
        pulse.quality_hint = Some(0.8);
        pulse
    }

    fn sample_feedback() -> SoftlogicZFeedback {
        SoftlogicZFeedback {
            psi_total: 0.0,
            weighted_loss: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_signal: 0.0,
            scale: None,
            events: Vec::new(),
            attributions: Vec::new(),
        }
    }

    fn report_with_signature(active: bool) -> InterfaceZReport {
        let signature = sample_signature(active);
        let lift = InterfaceZLift::new(&[1.0], LeechProjector::new(1, 0.0));
        let pulse = if active {
            sample_pulse()
        } else {
            InterfaceZPulse::default()
        };
        let fused = InterfaceZFused {
            pulse: ZPulse {
                source: ZSource::Microlocal,
                ts: 0,
                tempo: 1.0,
                band_energy: pulse.band_energy,
                drift: pulse.drift,
                z_bias: pulse.z_bias,
                support: ZSupport::from_band_energy(pulse.band_energy),
                scale: pulse.scale,
                quality: 0.8,
                stderr: 0.0,
                latency_ms: 0.0,
            },
            z: pulse.z_bias,
            support: pulse.support,
            attributions: vec![(ZSource::Microlocal, 1.0)],
            events: Vec::new(),
        };
        InterfaceZReport {
            gauge_ids: vec![Some("macro".into())],
            signatures: vec![signature],
            lift,
            pulses: vec![pulse.clone()],
            qualities: vec![0.9],
            fused_pulse: pulse,
            fused_z: fused,
            feedback: sample_feedback(),
            budget_scale: 1.0,
        }
    }

    #[test]
    fn bridge_links_microlocal_and_macro_feedback() {
        let config = ObservabilityConfig::new(1, 3, SlotSymmetry::Symmetric);
        let mut templates = MacroTemplateBank::new();
        templates.register_card(
            "macro",
            MacroCard::Minimal(MinimalCardConfig {
                phase_pair: PhasePair::new("A", "B"),
                sigma: 1.0,
                mobility: 1.0,
                volume: None,
                physical_scales: None,
            }),
        );

        let mut bridge = ObservationBridge::with_templates(config, templates);
        let report = report_with_signature(true);
        let snapshot = bridge.ingest(&report);

        assert_eq!(snapshot.observed_counts, vec![1, 1, 1, 1]);
        assert!(snapshot.assessment.expected.len() >= snapshot.observed_counts.len());
        assert!(snapshot.microlocal_feedback.is_some());
        assert_eq!(snapshot.drives.len(), 1);
        assert!(snapshot.has_interface());
    }

    #[test]
    fn bridge_handles_empty_reports() {
        let config = ObservabilityConfig::new(1, 2, SlotSymmetry::Symmetric);
        let templates = MacroTemplateBank::new();
        let mut bridge = ObservationBridge::with_templates(config, templates);
        let report = report_with_signature(false);
        let snapshot = bridge.ingest(&report);

        assert_eq!(snapshot.observed_counts, vec![0, 0, 0, 0]);
        assert!(snapshot.microlocal_feedback.is_none());
        assert_eq!(snapshot.drives.len(), 0);
        assert!(!snapshot.has_interface());
    }
}
