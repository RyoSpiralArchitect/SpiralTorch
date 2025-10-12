//! HIP backend (ROCm). Default: stubs. Enable `hip-real` for real path.
use serde::{Serialize, Deserialize};
use thiserror::Error;
#[derive(Debug, Error)] pub enum HipErr{ #[error("HIP not enabled (build with feature 'hip-real')")] NotEnabled, #[error("Other: {0}")] Other(String) }
#[derive(Debug, Clone, Copy, Serialize, Deserialize)] pub struct DeviceInfo{ pub id:u32, pub name:&'static str, pub multi_node: bool }

#[cfg(feature="hip-real")] pub mod real;
#[cfg(feature="hip-real")] pub mod rccl_comm;
#[cfg(not(feature="hip-real"))] pub mod stub { use super::*; pub fn hip_available()->bool{false} pub fn device_info()->Vec<DeviceInfo>{vec![]} }
