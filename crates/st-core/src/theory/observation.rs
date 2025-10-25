// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Microlocal → macrolocal observation bridge combining coalgebra diagnostics
//! with macro template feedback.
//!
//! The bridge inspects [`InterfaceZReport`] records emitted by the microlocal
//! conductor, counts how many unique gauges and scale slices participate in the
//! report, measures how many macroscopic drives fired, and pushes those counts
//! through the [`ObservationalCoalgebra`].  The resulting
//! [`ObservabilityAssessment`] exposes the efficiency `η` of each depth relative
//! to the theoretical Pólya upper bound while the matched macro templates
//! contribute their [`MicrolocalFeedback`] payloads.  Downstream controllers can
//! therefore reason about structural observability, gauge coverage, and
//! macroscopic feedback using a single snapshot.

use std::collections::HashSet;

use crate::telemetry::hub::SoftlogicZFeedback;
use crate::theory::macro_model::{MacroDrive, MacroTemplateBank};
use crate::theory::microlocal::{InterfaceZReport, MicrolocalFeedback};
use crate::theory::microlocal_bank::GaugeBank;
use crate::theory::observability::{
    ObservabilityAssessment, ObservabilityConfig, ObservationalCoalgebra,
};

/// Named observation counts grouped by coalgebra depth.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObservationCounts {
    /// Total number of gauges reported by the microlocal conductor.
    pub total_gauges: u128,
    /// Root occupancy flag (`1` when any interface was detected overall).
    pub root_interfaces: u128,
    /// Number of gauges that reported an active interface.
    pub active_gauges: u128,
    /// Number of distinct `(gauge id, radius, physical radius)` tuples with
    /// active interfaces.
    pub active_scale_slots: u128,
    /// Number of macro drives triggered by the matched templates.
    pub matched_macro_drives: u128,
    /// Number of microlocal pulses that carried non-zero support.
    pub active_pulses: u128,
}

impl ObservationCounts {
    /// Returns the depth counts as a vector suitable for the observation coalgebra.
    pub fn depths(&self) -> Vec<u128> {
        vec![
            self.root_interfaces,
            self.active_gauges,
            self.active_scale_slots,
            self.matched_macro_drives,
            self.active_pulses,
        ]
    }

    /// Ratio of active gauges over the total gauges reported by the conductor.
    pub fn coverage(&self) -> f32 {
        if self.total_gauges == 0 {
            0.0
        } else {
            (self.active_gauges as f32) / (self.total_gauges as f32)
        }
    }

    /// Number of gauges that remained inactive during the report.
    pub fn inactive_gauges(&self) -> u128 {
        self.total_gauges.saturating_sub(self.active_gauges)
    }
}

/// Summary describing the activity detected for a single gauge.
#[derive(Clone, Debug, PartialEq)]
pub struct GaugeSummary {
    /// Identifier reported by the microlocal conductor (or a synthetic fallback).
    pub id: String,
    /// Whether the gauge detected any interface cells.
    pub active: bool,
    /// Radius (in grid steps) used to analyse the gauge neighbourhood.
    pub radius: isize,
    /// Physical radius attached to the gauge neighbourhood.
    pub physical_radius: f32,
    /// Whether a macro template generated a drive for this gauge.
    pub matched_template: bool,
    /// Support carried by the microlocal pulse, if available.
    pub pulse_support: Option<f32>,
    /// Drift-corrected Z bias associated with the pulse.
    pub z_bias: Option<f32>,
    /// Quality hint emitted by the microlocal conductor.
    pub quality: Option<f32>,
}

impl GaugeSummary {
    /// Convenience helper exposing whether the gauge registered an interface.
    pub fn is_active(&self) -> bool {
        self.active
    }
}

/// Snapshot returned after fusing microlocal telemetry with observation
/// diagnostics.
#[derive(Clone, Debug)]
pub struct ObservationBridgeSnapshot {
    /// Observation counts grouped by coalgebra depth.
    pub counts: ObservationCounts,
    /// Observed counts per depth (root, gauge ids, scale slices, macro drives,
    /// microlocal pulses).
    pub depth_counts: Vec<u128>,
    /// Comparison between the observed counts and the theoretical upper bound.
    pub assessment: ObservabilityAssessment,
    /// Aggregated microlocal feedback emitted by the matched macro templates.
    pub microlocal_feedback: Option<MicrolocalFeedback>,
    /// Macro drives keyed by the template identifier.
    pub drives: GaugeBank<MacroDrive>,
    /// Softlogic feedback emitted by the microlocal conductor.
    pub softlogic_feedback: SoftlogicZFeedback,
    /// Activity summary for every gauge reported by the microlocal conductor.
    pub gauges: Vec<GaugeSummary>,
}

impl ObservationBridgeSnapshot {
    /// Returns `true` when at least one interface was detected at the microlocal level.
    pub fn has_interface(&self) -> bool {
        self.counts.root_interfaces > 0
    }

    /// Returns the depth counts slice used for the coalgebra assessment.
    pub fn observed_counts(&self) -> &[u128] {
        &self.depth_counts
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
        let gauges = gauge_summaries(report, &drives);
        let counts = observed_counts(report, &gauges, &drives);
        let depth_counts = counts.depths();
        let assessment = self.coalgebra.assess(&depth_counts);
        let microlocal_feedback = merge_feedback(&drives);

        ObservationBridgeSnapshot {
            counts,
            depth_counts,
            assessment,
            microlocal_feedback,
            drives,
            softlogic_feedback: report.feedback.clone(),
            gauges,
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

fn observed_counts(
    report: &InterfaceZReport,
    gauges: &[GaugeSummary],
    drives: &GaugeBank<MacroDrive>,
) -> ObservationCounts {
    let mut scale_slots: HashSet<(String, isize, u32)> = HashSet::new();

    for summary in gauges.iter().filter(|summary| summary.active) {
        scale_slots.insert((
            summary.id.clone(),
            summary.radius,
            summary.physical_radius.to_bits(),
        ));
    }

    let active_pulses = report
        .pulses
        .iter()
        .filter(|pulse| !pulse.is_empty())
        .count() as u128;

    ObservationCounts {
        total_gauges: gauges.len() as u128,
        root_interfaces: if report.has_interface() { 1 } else { 0 },
        active_gauges: gauges.iter().filter(|summary| summary.active).count() as u128,
        active_scale_slots: scale_slots.len() as u128,
        matched_macro_drives: drives.len() as u128,
        active_pulses,
    }
}

fn gauge_summaries(report: &InterfaceZReport, drives: &GaugeBank<MacroDrive>) -> Vec<GaugeSummary> {
    report
        .signatures
        .iter()
        .enumerate()
        .map(|(index, signature)| {
            let id = report
                .gauge_id(index)
                .map(|value| value.to_string())
                .unwrap_or_else(|| format!("gauge#{index}"));
            let matched_template = drives.get(id.as_str()).is_some();
            let pulse_support = report.pulses.get(index).map(|pulse| pulse.support);
            let z_bias = report.pulses.get(index).map(|pulse| pulse.z_bias);
            let quality = report.qualities.get(index).copied();

            GaugeSummary {
                id,
                active: signature.has_interface(),
                radius: signature.radius,
                physical_radius: signature.physical_radius,
                matched_template,
                pulse_support,
                z_bias,
                quality,
            }
        })
        .collect()
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
            elliptic: None,
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

        assert_eq!(snapshot.depth_counts, vec![1, 1, 1, 1, 1]);
        assert_eq!(snapshot.counts.total_gauges, 1);
        assert_eq!(snapshot.counts.active_gauges, 1);
        assert_eq!(snapshot.counts.active_pulses, 1);
        assert!(snapshot.assessment.expected.len() >= snapshot.depth_counts.len());
        assert!(snapshot.microlocal_feedback.is_some());
        assert_eq!(snapshot.drives.len(), 1);
        assert!(snapshot.has_interface());
        assert!((snapshot.counts.coverage() - 1.0).abs() < f32::EPSILON);
        assert_eq!(snapshot.gauges.len(), 1);
        let gauge = &snapshot.gauges[0];
        assert!(gauge.matched_template);
        assert!(gauge.is_active());
        assert_eq!(gauge.pulse_support, Some(1.0));
        assert_eq!(gauge.quality, Some(0.9));
    }

    #[test]
    fn bridge_handles_empty_reports() {
        let config = ObservabilityConfig::new(1, 2, SlotSymmetry::Symmetric);
        let templates = MacroTemplateBank::new();
        let mut bridge = ObservationBridge::with_templates(config, templates);
        let report = report_with_signature(false);
        let snapshot = bridge.ingest(&report);

        assert_eq!(snapshot.depth_counts, vec![0, 0, 0, 0, 0]);
        assert_eq!(snapshot.counts.total_gauges, 1);
        assert_eq!(snapshot.counts.active_gauges, 0);
        assert!(snapshot.microlocal_feedback.is_none());
        assert_eq!(snapshot.drives.len(), 0);
        assert!(!snapshot.has_interface());
        assert_eq!(snapshot.counts.inactive_gauges(), 1);
        assert_eq!(snapshot.gauges.len(), 1);
        assert!(!snapshot.gauges[0].is_active());
    }
}
