//! Robotics utilities for SpiralTorch.

mod desire;
mod error;
mod policy;
mod runtime;
mod sensors;
mod telemetry;

pub use desire::{Desire, DesireLagrangianField, EnergyReport};
pub use error::RoboticsError;
pub use policy::PolicyGradientController;
pub use runtime::{RoboticsRuntime, RuntimeStep};
pub use sensors::{FusedFrame, SensorFusionHub};
pub use telemetry::{PsiTelemetry, TelemetryReport};
