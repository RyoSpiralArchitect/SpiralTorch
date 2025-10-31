use std::collections::VecDeque;

use crate::desire::EnergyReport;
use crate::sensors::FusedFrame;

/// Snapshot of runtime vitals surfaced to higher-level controllers.
#[derive(Debug, Clone)]
pub struct TelemetryReport {
    pub energy: f32,
    pub stability: f32,
    pub failsafe: bool,
    pub anomalies: Vec<String>,
}

/// Sliding-window telemetry tracking energetic stability and physical bounds.
#[derive(Debug, Clone)]
pub struct PsiTelemetry {
    window: usize,
    stability_threshold: f32,
    failure_energy: f32,
    norm_limit: f32,
    history: VecDeque<f32>,
}

impl PsiTelemetry {
    pub fn new(
        window: usize,
        stability_threshold: f32,
        failure_energy: f32,
        norm_limit: f32,
    ) -> Self {
        let capacity = window.max(1);
        Self {
            window: capacity,
            stability_threshold,
            failure_energy,
            norm_limit,
            history: VecDeque::with_capacity(capacity),
        }
    }

    pub fn observe(&mut self, frame: &FusedFrame, energy: &EnergyReport) -> TelemetryReport {
        if self.history.len() == self.window {
            self.history.pop_front();
        }
        self.history.push_back(energy.total);

        let stability = if self.history.len() < 2 {
            1.0
        } else {
            let mean = self.history.iter().copied().sum::<f32>() / self.history.len() as f32;
            let variance = self
                .history
                .iter()
                .map(|value| {
                    let diff = value - mean;
                    diff * diff
                })
                .sum::<f32>()
                / self.history.len() as f32;
            let deviation = variance.sqrt();
            1.0 / (1.0 + deviation)
        };

        let mut anomalies = Vec::new();
        if stability < self.stability_threshold {
            anomalies.push("instability".to_string());
        }
        if energy.total > self.failure_energy {
            anomalies.push("energy_overflow".to_string());
        }
        for values in frame.coordinates.values() {
            let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
            if norm > self.norm_limit {
                anomalies.push("norm_overflow".to_string());
            }
        }
        anomalies.sort();
        anomalies.dedup();

        let failsafe = anomalies.iter().any(|tag| tag.starts_with("norm_overflow"))
            || energy.total > self.failure_energy;

        TelemetryReport {
            energy: energy.total,
            stability,
            failsafe,
            anomalies,
        }
    }
}

impl Default for PsiTelemetry {
    fn default() -> Self {
        Self::new(8, 0.5, 5.0, 10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desire::{Desire, DesireLagrangianField};
    use crate::sensors::SensorFusionHub;

    #[test]
    fn telemetry_detects_overflow() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("pose", 2).unwrap();
        let mut desires = std::collections::HashMap::new();
        desires.insert(
            "pose".to_string(),
            Desire {
                target_norm: 0.0,
                tolerance: 0.0,
                weight: 1.0,
            },
        );
        let field = DesireLagrangianField::new(desires);
        let mut telemetry = PsiTelemetry::new(4, 0.8, 10.0, 1.0);
        let frame = hub
            .fuse(&std::collections::HashMap::from([(
                "pose".to_string(),
                vec![2.0, 0.0],
            )]))
            .unwrap();
        let energy = field.energy(&frame);
        let report = telemetry.observe(&frame, &energy);
        assert!(report.failsafe);
        assert!(report
            .anomalies
            .iter()
            .any(|tag| tag.contains("norm_overflow")));
    }
}
