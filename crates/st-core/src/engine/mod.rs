//! Minimal engine skeleton (CPU fallback).
//! Backends should extend this module; ops route through st_core::ops for now.

use crate::{device::Device, error::Result};

#[derive(Clone, Copy)]
pub struct Engine { pub device: Device }
impl Engine {
    pub fn new(device: Device) -> Self { Self { device } }
}
