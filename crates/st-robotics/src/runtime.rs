use std::collections::HashMap;

use crate::desire::{DesireLagrangianField, EnergyReport};
use crate::error::RoboticsError;
use crate::policy::PolicyGradientController;
use crate::sensors::{FusedFrame, SensorFusionHub};
use crate::telemetry::{PsiTelemetry, TelemetryReport};

/// Outputs of a single robotics runtime tick.
#[derive(Debug, Clone)]
pub struct RuntimeStep {
    pub frame: FusedFrame,
    pub energy: EnergyReport,
    pub telemetry: TelemetryReport,
    pub commands: HashMap<String, f32>,
    pub halted: bool,
}

/// Coordinates sensing, instinct evaluation, and optional control policies.
pub struct RoboticsRuntime {
    sensors: SensorFusionHub,
    desires: DesireLagrangianField,
    telemetry: PsiTelemetry,
    policy: Option<PolicyGradientController>,
}

impl RoboticsRuntime {
    pub fn new(
        sensors: SensorFusionHub,
        desires: DesireLagrangianField,
        telemetry: PsiTelemetry,
    ) -> Self {
        Self {
            sensors,
            desires,
            telemetry,
            policy: None,
        }
    }

    pub fn attach_policy_gradient(&mut self, controller: PolicyGradientController) {
        self.policy = Some(controller);
    }

    pub fn step(
        &mut self,
        payloads: HashMap<String, Vec<f32>>,
    ) -> Result<RuntimeStep, RoboticsError> {
        let frame = self.sensors.fuse(&payloads)?;
        let energy = self.desires.energy(&frame);
        let report = self.telemetry.observe(&frame, &energy);
        let mut commands = HashMap::new();
        if let Some(policy) = &mut self.policy {
            commands.extend(policy.update(&energy, &report));
        }
        let halted = report.failsafe;
        if halted {
            commands.insert("halt".to_string(), 1.0);
        }
        Ok(RuntimeStep {
            frame,
            energy,
            telemetry: report,
            commands,
            halted,
        })
    }

    pub fn sensors(&self) -> &SensorFusionHub {
        &self.sensors
    }

    pub fn sensors_mut(&mut self) -> &mut SensorFusionHub {
        &mut self.sensors
    }

    pub fn desires(&self) -> &DesireLagrangianField {
        &self.desires
    }

    pub fn telemetry(&self) -> &PsiTelemetry {
        &self.telemetry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desire::Desire;

    #[test]
    fn runtime_emits_policy_commands() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("imu", 3).unwrap();
        let desires = DesireLagrangianField::new(HashMap::from([(
            "imu".to_string(),
            Desire {
                target_norm: 0.0,
                tolerance: 0.0,
                weight: 1.0,
            },
        )]));
        let telemetry = PsiTelemetry::default();
        let mut runtime = RoboticsRuntime::new(hub, desires, telemetry);
        runtime.attach_policy_gradient(PolicyGradientController::new(0.05, 0.5));
        let step = runtime
            .step(HashMap::from([("imu".to_string(), vec![0.1, -0.2, 0.05])]))
            .unwrap();
        assert!(!step.halted);
        assert!(step.commands.contains_key("learning_rate"));
        assert!(step.commands.contains_key("gauge"));
    }
}
