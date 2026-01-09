// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::ffi::c_void;
use std::mem;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;

use crate::memory::AlignedVec;

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
struct ForeignTensorInner {
    managed: NonNull<DLManagedTensor>,
    data: NonNull<f32>,
    len: usize,
}

// SAFETY: `ForeignTensorInner` only stores raw pointers to externally managed data and never
// provides mutable access to the underlying buffer. The DLPack deleter is assumed to be safe to
// call from any thread because `Tensor` can be shared across rayon workers.
unsafe impl Send for ForeignTensorInner {}
unsafe impl Sync for ForeignTensorInner {}

impl Drop for ForeignTensorInner {
    fn drop(&mut self) {
        unsafe {
            call_managed_deleter(self.managed.as_ptr());
        }
    }
}

#[derive(Clone, Debug)]
pub struct ForeignTensor {
    inner: Arc<ForeignTensorInner>,
}

impl ForeignTensor {
    /// # Safety
    /// `managed` must reference a valid `DLManagedTensor` whose lifetime is
    /// owned by the caller. `data` must point to the first element of the
    /// tensor and remain valid until the deleter runs.
    pub unsafe fn new(managed: NonNull<DLManagedTensor>, data: NonNull<f32>, len: usize) -> Self {
        Self {
            inner: Arc::new(ForeignTensorInner { managed, data, len }),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.inner.data.as_ptr(), self.inner.len) }
    }

    pub fn len(&self) -> usize {
        self.inner.len
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.inner.data.as_ptr()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }
}

#[derive(Clone, Debug)]
pub enum ExportData {
    Owned(Arc<AlignedVec>),
    Foreign(ForeignTensor),
}

impl ExportData {
    pub fn as_ptr(&self) -> *const f32 {
        match self {
            ExportData::Owned(data) => data.as_ptr(),
            ExportData::Foreign(buffer) => buffer.as_ptr(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ExportData::Owned(data) => data.len(),
            ExportData::Foreign(buffer) => buffer.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug)]
pub struct ManagedTensorState {
    pub data: ExportData,
    pub shape: Box<[i64]>,
    pub strides: Box<[i64]>,
}

impl ManagedTensorState {
    pub fn new(data: ExportData, shape: Box<[i64]>, strides: Box<[i64]>) -> Self {
        Self {
            data,
            shape,
            strides,
        }
    }
}

/// Calls the deleter associated with a managed tensor, if one exists.
///
/// # Safety
/// `ptr` must either be null or point to a valid `DLManagedTensor` allocated by
/// a DLPack producer. The pointed-to tensor must remain valid for the duration
/// of this call.
pub unsafe fn call_managed_deleter(ptr: *mut DLManagedTensor) {
    if ptr.is_null() {
        return;
    }
    if let Some(deleter) = (*ptr).deleter {
        deleter(ptr);
    }
}

/// Restores the managed tensor state allocated during export.
///
/// # Safety
/// `ptr` must either be null or point to a `DLManagedTensor` that was created
/// by [`Tensor::to_dlpack`](crate::pure::Tensor::to_dlpack) / this module. The
/// function takes ownership of the pointed-to allocation and must be called at
/// most once per managed tensor.
pub unsafe extern "C" fn drop_exported_state(ptr: *mut DLManagedTensor) {
    if ptr.is_null() {
        return;
    }
    let mut boxed = Box::from_raw(ptr);
    if !boxed.manager_ctx.is_null() {
        let state = Box::from_raw(boxed.manager_ctx as *mut ManagedTensorState);
        drop(state);
    }
    // Prevent the original deleter from running since we've taken
    // responsibility for dropping the wrapper here.
    boxed.deleter = None;
    mem::drop(boxed);
}

/// Capsule name required by the DLPack specification for live tensors.
pub static DLPACK_CAPSULE_NAME: &std::ffi::CStr = c"dltensor";

/// Capsule name used once the tensor has been consumed by a downstream framework.
pub static USED_DLPACK_CAPSULE_NAME: &std::ffi::CStr = c"used_dltensor";

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TestCtx {
        drops: Arc<AtomicUsize>,
        data: Box<[f32]>,
    }

    unsafe extern "C" fn test_deleter(ptr: *mut DLManagedTensor) {
        if ptr.is_null() {
            return;
        }
        let mut boxed = Box::from_raw(ptr);
        if !boxed.manager_ctx.is_null() {
            let ctx = Box::from_raw(boxed.manager_ctx as *mut TestCtx);
            ctx.drops.fetch_add(1, Ordering::SeqCst);
            drop(ctx);
            boxed.manager_ctx = ptr::null_mut();
        }
        boxed.deleter = None;
        drop(boxed);
    }

    #[test]
    fn foreign_tensor_invokes_deleter_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let ctx = Box::new(TestCtx {
            drops: Arc::clone(&drops),
            data: vec![1.0, 2.0, 3.0].into_boxed_slice(),
        });
        let len = ctx.data.len();
        assert!(len > 0);
        let data_ptr = ctx.data.as_ptr() as *mut f32;
        let managed = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr as *mut c_void,
                device: DLDevice {
                    device_type: DLDeviceType::Cpu as i32,
                    device_id: 0,
                },
                ndim: 1,
                dtype: DLDataType {
                    code: DLDataTypeCode::Float as u8,
                    bits: 32,
                    lanes: 1,
                },
                shape: ptr::null_mut(),
                strides: ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: Box::into_raw(ctx) as *mut c_void,
            deleter: Some(test_deleter),
        });
        let managed_ptr = match NonNull::new(Box::into_raw(managed)) {
            Some(ptr) => ptr,
            None => panic!("managed tensor pointer is null"),
        };
        let data_ptr = match NonNull::new(data_ptr) {
            Some(ptr) => ptr,
            None => panic!("data pointer is null"),
        };

        let foreign = unsafe { ForeignTensor::new(managed_ptr, data_ptr, len) };
        assert_eq!(foreign.as_slice(), &[1.0, 2.0, 3.0]);

        let clone = foreign.clone();
        drop(foreign);
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        drop(clone);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn call_managed_deleter_ignores_null() {
        unsafe { call_managed_deleter(ptr::null_mut()) };
    }

    #[test]
    fn drop_exported_state_ignores_null() {
        unsafe { drop_exported_state(ptr::null_mut()) };
    }
}
