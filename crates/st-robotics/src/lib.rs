//! Robotics utilities for SpiralTorch.

mod desire;
mod error;
mod geometry;
mod gravity;
mod policy;
mod relativity;
mod runtime;
mod safety;
mod sensors;
mod telemetry;
mod training;
mod vision;

pub use desire::{Desire, DesireLagrangianField, EnergyReport};
pub use error::RoboticsError;
pub use geometry::{GeometryKind, ZSpaceDynamics, ZSpaceGeometry};
pub use gravity::{GravityField, GravityRegime, GravityWell};
pub use policy::PolicyGradientController;
pub use relativity::{RelativityBridge, SymmetryAnsatz};
pub use runtime::{RoboticsRuntime, RuntimeStep, TrajectoryRecorder};
pub use safety::{DriftSafetyPlugin, SafetyPlugin, SafetyReview};
pub use sensors::{ChannelHealth, FusedFrame, SensorFusionHub};
pub use telemetry::{PsiTelemetry, TelemetryReport};
pub use training::{
    TemporalFeedbackLearner, TemporalFeedbackSample, TemporalFeedbackSummary, TrainerEpisode,
    TrainerMetrics, ZSpacePartialObservation, ZSpaceTrainerBridge, ZSpaceTrainerEpisodeBuilder,
    ZSpaceTrainerSample,
};
pub use vision::{VisionFeedbackSnapshot, VisionFeedbackSynchronizer};
