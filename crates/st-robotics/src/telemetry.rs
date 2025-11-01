use std::collections::VecDeque;

use crate::desire::EnergyReport;
use crate::geometry::ZSpaceGeometry;
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
    geometry: ZSpaceGeometry,
}

impl PsiTelemetry {
    pub fn new(
        window: usize,
        stability_threshold: f32,
        failure_energy: f32,
        norm_limit: f32,
    ) -> Self {
        Self::with_geometry(
            window,
            stability_threshold,
            failure_energy,
            norm_limit,
            ZSpaceGeometry::default(),
        )
    }

    pub fn with_geometry(
        window: usize,
        stability_threshold: f32,
        failure_energy: f32,
        norm_limit: f32,
        geometry: ZSpaceGeometry,
    ) -> Self {
        let capacity = window.max(1);
        Self {
            window: capacity,
            stability_threshold,
            failure_energy,
            norm_limit,
            history: VecDeque::with_capacity(capacity),
            geometry,
        }
    }

    pub fn observe(&mut self, frame: &FusedFrame, energy: &EnergyReport) -> TelemetryReport {
        if self.history.len() == self.window {
            self.history.pop_front();
        }
        self.history.push_back(energy.total);

        let mut stability = if self.history.len() < 2 {
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
        if energy.gravitational.abs() > self.failure_energy {
            anomalies.push("gravity_overflow".to_string());
        }
        for values in frame.coordinates.values() {
            let norm = self.geometry.metric_norm(values);
            if norm > self.norm_limit {
                anomalies.push("norm_overflow".to_string());
            }
        }
        for (name, health) in &frame.health {
            if health.stale {
                anomalies.push(format!("stale:{name}"));
            }
            if health.stale {
                stability *= 0.5;
            }
        }
        anomalies.sort();
        anomalies.dedup();

        let failsafe = anomalies.iter().any(|tag| tag.starts_with("norm_overflow"))
            || energy.total > self.failure_energy
            || energy.gravitational.abs() > self.failure_energy;

        TelemetryReport {
            energy: energy.total,
            stability,
            failsafe,
            anomalies,
        }
    }

    pub fn set_geometry(&mut self, geometry: ZSpaceGeometry) {
        self.geometry = geometry;
    }

    pub fn geometry(&self) -> &ZSpaceGeometry {
        &self.geometry
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
    use crate::geometry::{ZSpaceDynamics, ZSpaceGeometry};
    use crate::sensors::SensorFusionHub;
    use crate::{GravityField, GravityRegime, GravityWell};
    use std::collections::HashMap;

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

    #[test]
    fn telemetry_marks_stale_channels() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel_with_options("imu", 3, None, true, Some(0.001))
            .unwrap();
        let frame = hub.fuse(&std::collections::HashMap::new()).unwrap();
        let mut desires = std::collections::HashMap::new();
        desires.insert(
            "imu".to_string(),
            Desire {
                target_norm: 0.0,
                tolerance: 0.0,
                weight: 1.0,
            },
        );
        let field = DesireLagrangianField::new(desires);
        let mut telemetry = PsiTelemetry::default();
        let energy = field.energy(&frame);
        let report = telemetry.observe(&frame, &energy);
        assert!(report.anomalies.iter().any(|tag| tag.starts_with("stale:")));
        assert!(report.stability < 1.0);
    }

    #[test]
    fn telemetry_detects_gravity_overflow() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("pose", 3).unwrap();
        let frame = hub
            .fuse(&std::collections::HashMap::from([(
                "pose".to_string(),
                vec![0.001, 0.0, 0.0],
            )]))
            .unwrap();
        let mut gravity = GravityField::default();
        gravity.add_well(
            "pose",
            GravityWell::new(
                5.0e10,
                GravityRegime::Relativistic {
                    speed_of_light: 1.0,
                },
            ),
        );
        let dynamics = ZSpaceDynamics::new(ZSpaceGeometry::euclidean(), Some(gravity));
        let field = DesireLagrangianField::with_dynamics(HashMap::new(), dynamics.clone());
        let mut telemetry =
            PsiTelemetry::with_geometry(4, 0.5, 1.0, 1.0, dynamics.geometry().clone());
        let energy = field.energy(&frame);
        let report = telemetry.observe(&frame, &energy);
        assert!(report
            .anomalies
            .iter()
            .any(|tag| tag.contains("gravity_overflow")));
        assert!(report.failsafe);
    }
}
