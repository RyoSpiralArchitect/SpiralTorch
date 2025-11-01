use std::collections::HashMap;

use crate::desire::EnergyReport;
use crate::telemetry::TelemetryReport;

/// Lightweight policy-gradient helper that adapts a scalar learning rate.
#[derive(Debug, Clone)]
pub struct PolicyGradientController {
    base_learning_rate: f32,
    smoothing: f32,
    gauge: f32,
}

impl PolicyGradientController {
    pub fn new(base_learning_rate: f32, smoothing: f32) -> Self {
        Self {
            base_learning_rate,
            smoothing,
            gauge: 0.0,
        }
    }

    pub fn update(
        &mut self,
        energy: &EnergyReport,
        telemetry: &TelemetryReport,
    ) -> HashMap<String, f32> {
        let effective = self.base_learning_rate / (1.0 + energy.total.max(0.0));
        self.gauge = self.gauge * self.smoothing + telemetry.stability * (1.0 - self.smoothing);
        let mut commands = HashMap::new();
        commands.insert("learning_rate".to_string(), effective);
        commands.insert("gauge".to_string(), self.gauge);
        commands
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desire::EnergyReport;
    use crate::telemetry::TelemetryReport;

    #[test]
    fn controller_updates_learning_rate() {
        let mut controller = PolicyGradientController::new(0.05, 0.5);
        let mut report = EnergyReport::zero();
        report.total = 2.0;
        let telemetry = TelemetryReport {
            energy: 2.0,
            stability: 0.8,
            failsafe: false,
            anomalies: vec![],
        };
        let commands = controller.update(&report, &telemetry);
        assert!(commands.get("learning_rate").unwrap() < &0.05);
        assert!(commands.get("gauge").is_some());
    }
}
