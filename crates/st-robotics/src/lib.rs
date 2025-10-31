//! Robotics utilities for SpiralTorch.

mod desire;
mod error;
mod geometry;
mod policy;
mod runtime;
mod sensors;
mod telemetry;

pub use desire::{Desire, DesireLagrangianField, EnergyReport};
pub use error::RoboticsError;
pub use geometry::{
    GeometryKind, GravityField, GravityRegime, GravityWell, ZSpaceDynamics, ZSpaceGeometry,
};
pub use policy::PolicyGradientController;
pub use runtime::{RoboticsRuntime, RuntimeStep, TrajectoryRecorder};
pub use sensors::{ChannelHealth, FusedFrame, SensorFusionHub};
pub use telemetry::{PsiTelemetry, TelemetryReport};
