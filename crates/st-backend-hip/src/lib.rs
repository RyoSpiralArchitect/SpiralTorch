//! HIP backend skeleton (feature-gated).
//! By default, provides **stubs** that compile everywhere.
//! Enable `hip-real` feature and link against HIP/RCCL to activate real paths.

use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HipErr{
    #[error("HIP not enabled (build with feature 'hip-real')")]
    NotEnabled,
    #[error("Other: {0}")]
    Other(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DeviceInfo{
    pub id: u32,
    pub name: &'static str,
    pub multi_node: bool,
}

#[cfg(feature="hip-real")]
mod real {
    use super::*;
    // Placeholders: here we'd bind to HIP APIs (hipMalloc/hipMemcpyAsync) and RCCL allreduce.
    pub fn hip_available()->bool { true }
    pub fn device_info()->Vec<DeviceInfo> {
        vec![DeviceInfo{ id:0, name:"HIP-Device-0", multi_node:true }]
    }
    pub fn hip_allreduce_i32(buf:&mut [i32]) -> Result<(),HipErr> {
        // call into RCCL; placeholder
        Ok(())
    }
    pub fn hip_memcpy_async()->Result<(),HipErr>{ Ok(()) }
}

#[cfg(not(feature="hip-real"))]
mod real {
    use super::*;
    pub fn hip_available()->bool { false }
    pub fn device_info()->Vec<DeviceInfo> { vec![] }
    pub fn hip_allreduce_i32(_buf:&mut [i32]) -> Result<(),HipErr> { Err(HipErr::NotEnabled) }
    pub fn hip_memcpy_async()->Result<(),HipErr>{ Err(HipErr::NotEnabled) }
}

pub use real::*;
