use std::collections::BTreeMap;

use crate::desire::EnergyReport;
use crate::sensors::FusedFrame;
use crate::telemetry::TelemetryReport;
use spiral_safety::drift_response::{
    self, AnalysisOptions, DrlMetrics, FrameState, FrameThreshold, WordState,
};

/// Safety summary emitted by a [`SafetyPlugin`].
#[derive(Debug, Clone)]
pub struct SafetyReview {
    pub metrics: DrlMetrics,
    pub hazard_total: f32,
    pub refused: bool,
    pub flagged_frames: Vec<String>,
}

/// Hook that allows robotics runtimes to participate in the spiral-safety governance loop.
pub trait SafetyPlugin: Send {
    fn review(
        &mut self,
        frame: &FusedFrame,
        energy: &EnergyReport,
        telemetry: &TelemetryReport,
    ) -> SafetyReview;
}

/// Adapter that maps ψ telemetry and desire energy into drift-response metrics.
#[derive(Debug, Clone)]
pub struct DriftSafetyPlugin {
    word_name: String,
    thresholds: BTreeMap<String, FrameThreshold>,
    options: AnalysisOptions,
    hazard_cut: f32,
}

impl DriftSafetyPlugin {
    pub fn new(word_name: impl Into<String>) -> Self {
        Self {
            word_name: word_name.into(),
            thresholds: BTreeMap::new(),
            options: AnalysisOptions::default(),
            hazard_cut: 0.8,
        }
    }

    pub fn with_thresholds(mut self, thresholds: BTreeMap<String, FrameThreshold>) -> Self {
        self.thresholds = thresholds;
        self
    }

    pub fn with_analysis_options(mut self, options: AnalysisOptions) -> Self {
        self.options = options;
        self
    }

    pub fn set_hazard_cut(&mut self, hazard_cut: f32) {
        self.hazard_cut = hazard_cut.max(0.0);
    }

    pub fn hazard_cut(&self) -> f32 {
        self.hazard_cut
    }

    pub fn set_threshold(&mut self, channel: impl Into<String>, hazard: f32) {
        let channel = channel.into();
        let threshold = FrameThreshold::new(0.05, 0.1, hazard);
        self.thresholds.insert(channel, threshold);
    }

    pub fn thresholds(&self) -> &BTreeMap<String, FrameThreshold> {
        &self.thresholds
    }

    pub fn thresholds_mut(&mut self) -> &mut BTreeMap<String, FrameThreshold> {
        &mut self.thresholds
    }

    pub fn options(&self) -> &AnalysisOptions {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut AnalysisOptions {
        &mut self.options
    }

    fn ensure_threshold(&mut self, channel: &str) {
        self.thresholds
            .entry(channel.to_string())
            .or_insert_with(|| FrameThreshold::new(0.05, 0.1, self.hazard_cut));
    }

    fn build_frame_state(
        &self,
        channel: &str,
        frame: &FusedFrame,
        energy: &EnergyReport,
        telemetry: &TelemetryReport,
    ) -> FrameState {
        let channel_energy = energy
            .per_channel
            .get(channel)
            .copied()
            .unwrap_or(0.0)
            .abs();
        let gravitational = energy
            .gravitational_per_channel
            .get(channel)
            .copied()
            .unwrap_or(0.0)
            .abs();
        let radius = frame.norm(channel).unwrap_or_else(|| {
            frame
                .coordinates
                .get(channel)
                .map(|values| values.iter().map(|value| value * value).sum::<f32>().sqrt())
                .unwrap_or(0.0)
        });
        let anomaly_prefix = format!("stale:{channel}");
        let anomaly_hits = telemetry
            .anomalies
            .iter()
            .filter(|tag| {
                tag.as_str() == "instability"
                    || tag.as_str() == "energy_overflow"
                    || tag.as_str() == "gravity_overflow"
                    || tag.as_str() == "norm_overflow"
                    || tag.starts_with(&anomaly_prefix)
            })
            .count() as f32;
        let phi = (1.0 - telemetry.stability).clamp(0.0, 1.0);
        let failsafe_penalty = if telemetry.failsafe { 1.0 } else { 0.0 };

        let mut state = FrameState::default();
        state.phi = phi;
        state.c = 1.0 + channel_energy + gravitational + anomaly_hits + failsafe_penalty;
        state.s = 1.0 + radius + anomaly_hits;
        state.a_den = -(channel_energy + failsafe_penalty);
        state.a_con = -(channel_energy + gravitational + anomaly_hits + failsafe_penalty);
        let base_b = 1.0 + channel_energy + anomaly_hits + failsafe_penalty;
        state.b_den = base_b;
        state.b_con = base_b * (1.0 + phi);
        state.kappa = 1.0 + radius + anomaly_hits + failsafe_penalty;
        state.kappa_slope = anomaly_hits + failsafe_penalty;
        state.timing_scale = 1.0 + anomaly_hits;
        state
    }
}

impl Default for DriftSafetyPlugin {
    fn default() -> Self {
        Self::new("Robotics")
    }
}

impl SafetyPlugin for DriftSafetyPlugin {
    fn review(
        &mut self,
        frame: &FusedFrame,
        energy: &EnergyReport,
        telemetry: &TelemetryReport,
    ) -> SafetyReview {
        let entropy = 1.0 + telemetry.anomalies.len() as f32;
        let mut word = WordState::new(self.word_name.clone(), entropy);
        word.base_lambda = 1.0;
        word.beta = 1.0 + energy.total.abs().min(2.0);
        word.timing_signal = 1.0 + telemetry.anomalies.len() as f32;

        for channel in frame.coordinates.keys() {
            self.ensure_threshold(channel);
            let state = self.build_frame_state(channel, frame, energy, telemetry);
            word.frames.insert(channel.clone(), state);
        }

        let metrics =
            drift_response::analyse_word_with_options(&word, &self.thresholds, &self.options);
        let hazard_total = metrics.frame_hazards.values().copied().sum();
        let mut flagged_frames = Vec::new();
        for (name, hazard) in &metrics.frame_hazards {
            if let Some(threshold) = self.thresholds.get(name) {
                let cut = self.options.hazard_cut.unwrap_or(threshold.hazard);
                if *hazard >= cut {
                    flagged_frames.push(name.clone());
                }
            }
        }
        let refused = telemetry.failsafe || metrics.strict_mode || !flagged_frames.is_empty();

        SafetyReview {
            metrics,
            hazard_total,
            refused,
            flagged_frames,
        }
    }
}
