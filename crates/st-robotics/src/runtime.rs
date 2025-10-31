use std::collections::{HashMap, VecDeque};

use crate::desire::{DesireLagrangianField, EnergyReport};
use crate::error::RoboticsError;
use crate::geometry::ZSpaceDynamics;
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
    recorder: Option<TrajectoryRecorder>,
}

impl RoboticsRuntime {
    pub fn new(
        sensors: SensorFusionHub,
        desires: DesireLagrangianField,
        telemetry: PsiTelemetry,
    ) -> Self {
        let mut runtime = Self {
            sensors,
            desires,
            telemetry,
            policy: None,
            recorder: None,
        };
        let geometry = runtime.desires.dynamics().geometry().clone();
        runtime.telemetry.set_geometry(geometry);
        runtime
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
        let step = RuntimeStep {
            frame,
            energy,
            telemetry: report,
            commands,
            halted,
        };
        if let Some(recorder) = &mut self.recorder {
            recorder.push(&step);
        }
        Ok(step)
    }

    pub fn configure_dynamics(&mut self, dynamics: ZSpaceDynamics) {
        self.desires.set_dynamics(dynamics.clone());
        self.telemetry.set_geometry(dynamics.geometry().clone());
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

    pub fn enable_recording(&mut self, capacity: usize) {
        self.recorder = Some(TrajectoryRecorder::new(capacity.max(1)));
    }

    pub fn recording_len(&self) -> usize {
        self.recorder.as_ref().map(|rec| rec.len()).unwrap_or(0)
    }

    pub fn drain_trajectory(&mut self) -> Vec<RuntimeStep> {
        if let Some(recorder) = &mut self.recorder {
            recorder.drain()
        } else {
            Vec::new()
        }
    }
}

/// Circular buffer retaining the most recent runtime steps for training datasets.
#[derive(Debug, Clone)]
pub struct TrajectoryRecorder {
    capacity: usize,
    buffer: VecDeque<RuntimeStep>,
}

impl TrajectoryRecorder {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
        }
    }

    fn push(&mut self, step: &RuntimeStep) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(step.clone());
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn drain(&mut self) -> Vec<RuntimeStep> {
        self.buffer.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desire::Desire;
    use crate::geometry::{
        GravityField, GravityRegime, GravityWell, ZSpaceDynamics, ZSpaceGeometry,
    };

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

    #[test]
    fn runtime_records_steps_for_training() {
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
        runtime.enable_recording(3);
        for value in [0.1, 0.2, 0.3, 0.4] {
            runtime
                .step(HashMap::from([("imu".to_string(), vec![value, 0.0, 0.0])]))
                .unwrap();
        }
        assert_eq!(runtime.recording_len(), 3);
        let trajectory = runtime.drain_trajectory();
        assert_eq!(trajectory.len(), 3);
        assert!(trajectory
            .iter()
            .all(|step| !step.commands.contains_key("halt")));
    }

    #[test]
    fn runtime_applies_custom_dynamics() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("pose", 3).unwrap();
        let desires = DesireLagrangianField::new(HashMap::new());
        let telemetry = PsiTelemetry::default();
        let mut runtime = RoboticsRuntime::new(hub, desires, telemetry);
        let mut gravity = GravityField::default();
        gravity.add_well("pose", GravityWell::new(5.0, GravityRegime::Newtonian));
        let dynamics = ZSpaceDynamics::new(ZSpaceGeometry::euclidean(), Some(gravity));
        runtime.configure_dynamics(dynamics);
        let result = runtime
            .step(HashMap::from([("pose".to_string(), vec![1.0, 0.0, 0.0])]))
            .unwrap();
        assert!(result.energy.gravitational_per_channel.contains_key("pose"));
    }
}
