//! Robotics utilities for SpiralTorch.

mod desire;
mod error;
mod geometry;
mod gravity;
mod policy;
mod relativity;
mod runtime;
mod sensors;
mod telemetry;

pub use desire::{Desire, DesireLagrangianField, EnergyReport};
pub use error::RoboticsError;
pub use geometry::{GeometryKind, ZSpaceDynamics, ZSpaceGeometry};
pub use gravity::{GravityField, GravityRegime, GravityWell};
pub use policy::PolicyGradientController;
pub use relativity::{RelativityBridge, SymmetryAnsatz};
pub use runtime::{RoboticsRuntime, RuntimeStep, TrajectoryRecorder};
pub use sensors::{ChannelHealth, FusedFrame, SensorFusionHub};
pub use telemetry::{PsiTelemetry, TelemetryReport};
