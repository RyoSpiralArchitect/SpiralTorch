// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::ffi::c_void;
use std::sync::Arc;

/// Minimal subset of the DLPack data type codes required for CPU `f32` tensors.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDataTypeCode {
    /// Signed integer types.
    Int = 0,
    /// Unsigned integer types.
    UInt = 1,
    /// IEEE floating point types.
    Float = 2,
    /// Complex numbers backed by interleaved floating point lanes.
    Complex = 5,
}

/// Representation of a DLPack data type.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

/// Enumeration of device kinds supported by DLPack.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDeviceType {
    Cpu = 1,
    Cuda = 2,
    CudaHost = 3,
    Opencl = 4,
    Vulkan = 7,
    Metal = 8,
    Vpi = 9,
    Rocm = 10,
    RocmHost = 11,
    OneApi = 17,
}

/// Device descriptor for a DLPack tensor.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

/// Raw tensor view exported through DLPack.
#[repr(C)]
#[derive(Debug)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: usize,
}

/// Externally managed tensor with a custom deleter.
#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// Internal state retained while exporting a tensor to a DLPack capsule.
#[derive(Debug)]
pub struct ManagedTensorState {
    pub data: Arc<Vec<f32>>,
    pub shape: Box<[i64]>,
    pub strides: Box<[i64]>,
}

impl ManagedTensorState {
    pub fn new(data: Arc<Vec<f32>>, shape: Box<[i64]>, strides: Box<[i64]>) -> Self {
        Self {
            data,
            shape,
            strides,
        }
    }
}

/// Calls the deleter associated with a managed tensor, if one exists.
pub unsafe fn call_managed_deleter(ptr: *mut DLManagedTensor) {
    if ptr.is_null() {
        return;
    }
    if let Some(deleter) = (*ptr).deleter {
        deleter(ptr);
    }
}

/// Restores the managed tensor state allocated during export.
pub unsafe fn drop_exported_state(ptr: *mut DLManagedTensor) {
    if ptr.is_null() {
        return;
    }
    let boxed = Box::from_raw(ptr);
    if !boxed.manager_ctx.is_null() {
        let state = Box::from_raw(boxed.manager_ctx as *mut ManagedTensorState);
        drop(state);
    }
    // `boxed` drops here, releasing the DLTensor wrapper itself.
}

/// Capsule name required by the DLPack specification for live tensors.
pub static DLPACK_CAPSULE_NAME: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"dltensor\0") };

/// Capsule name used once the tensor has been consumed by a downstream framework.
pub static USED_DLPACK_CAPSULE_NAME: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"used_dltensor\0") };
