#![allow(dead_code)]

//! Placeholder implementations for the proprietary NeRF trainer interfaces.
//! The full trainer is not available in this repository snapshot, but the
//! stub types allow downstream crates to compile when the `nerf` feature is
//! toggled on.

/// Stub trainer type retained for API compatibility.
#[derive(Debug, Default, Clone, Copy)]
pub struct NerfTrainer;

/// Configuration placeholder for the stub trainer.
#[derive(Debug, Default, Clone)]
pub struct NerfTrainingConfig;

/// Telemetry placeholder emitted by the stub trainer.
#[derive(Debug, Default, Clone)]
pub struct NerfTrainingStats;
