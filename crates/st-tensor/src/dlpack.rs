// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::ffi::c_void;
use std::mem;
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::Arc;

use crate::memory::{aligned_from_slice, AlignedVec};
use crate::{PureResult, TensorError};

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
    OneApi = 14,
    WebGpu = 15,
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
    pub byte_offset: u64,
}

/// Externally managed tensor with a custom deleter.
#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// ABI version carried by a versioned managed tensor.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLPackVersion {
    pub major: u32,
    pub minor: u32,
}

/// Version emitted by SpiralTorch. Later minor versions can be imported when
/// their dtype, device, flags, and layout are understood.
pub const DLPACK_VERSION: DLPackVersion = DLPackVersion { major: 1, minor: 0 };
pub const DLPACK_FLAG_READ_ONLY: u64 = 1;
pub const DLPACK_FLAG_IS_COPIED: u64 = 2;

/// DLPack 1.x managed tensor. Field order is part of the C ABI.
#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DlpackProtocol {
    Legacy,
    #[default]
    Versioned,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DlpackCopyPolicy {
    /// Fail rather than materialize a new buffer.
    Never,
    /// Copy only when the legacy protocol cannot preserve read-only storage.
    #[default]
    IfNeeded,
    /// Export an independent writable buffer, even if sharing is possible.
    Always,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DlpackExportOptions {
    pub protocol: DlpackProtocol,
    pub copy: DlpackCopyPolicy,
}

#[derive(Clone, Copy, Debug)]
enum ManagedPointer {
    Legacy(NonNull<DLManagedTensor>),
    Versioned(NonNull<DLManagedTensorVersioned>),
}

/// Owns one DLPack release obligation. Dropping an unconsumed handle calls the
/// producer's deleter; moving it into a Tensor transfers that obligation.
#[derive(Debug)]
pub struct ManagedTensor {
    pointer: ManagedPointer,
}

impl ManagedTensor {
    /// Take ownership of a legacy producer handle, including on import errors.
    ///
    /// # Safety
    /// The pointer must be a live, aligned DLPack handle, consumed exactly once.
    /// Its descriptor and buffer must remain valid until release, without
    /// concurrent mutation during Rust reads. A deleter must be callable on any
    /// thread (acquiring the GIL itself if needed). With no deleter, the producer
    /// must independently keep the descriptor and storage alive until release.
    pub unsafe fn from_legacy(pointer: *mut DLManagedTensor) -> PureResult<Self> {
        let pointer = NonNull::new(pointer).ok_or(TensorError::EmptyInput("dlpack tensor"))?;
        Ok(Self {
            pointer: ManagedPointer::Legacy(pointer),
        })
    }

    /// Take ownership of a versioned producer handle, including on import errors.
    ///
    /// # Safety
    /// The ownership, lifetime, and synchronization requirements of
    /// [`Self::from_legacy`] apply. For an unknown major version, only the stable
    /// version/context/deleter prefix must be readable; no later fields are read.
    pub unsafe fn from_versioned(pointer: *mut DLManagedTensorVersioned) -> PureResult<Self> {
        let pointer = NonNull::new(pointer).ok_or(TensorError::EmptyInput("dlpack tensor"))?;
        Ok(Self {
            pointer: ManagedPointer::Versioned(pointer),
        })
    }

    pub fn protocol(&self) -> DlpackProtocol {
        match self.pointer {
            ManagedPointer::Legacy(_) => DlpackProtocol::Legacy,
            ManagedPointer::Versioned(_) => DlpackProtocol::Versioned,
        }
    }

    /// Borrow the raw handle without transferring ownership. Its type is given
    /// by [`Self::protocol`]; callers must not modify or release it through this pointer.
    pub fn as_ptr(&self) -> *mut c_void {
        match self.pointer {
            ManagedPointer::Legacy(pointer) => pointer.as_ptr().cast(),
            ManagedPointer::Versioned(pointer) => pointer.as_ptr().cast(),
        }
    }

    /// Transfer the handle to an FFI consumer. That consumer must release it
    /// exactly once using the deleter for [`Self::protocol`].
    pub fn into_raw(self) -> *mut c_void {
        let pointer = self.as_ptr();
        mem::forget(self);
        pointer
    }

    fn descriptor(&self) -> PureResult<(&DLTensor, u64)> {
        unsafe {
            match self.pointer {
                ManagedPointer::Legacy(pointer) => Ok((&(*pointer.as_ptr()).dl_tensor, 0)),
                ManagedPointer::Versioned(pointer) => {
                    let pointer = pointer.as_ptr();
                    let version = ptr::addr_of!((*pointer).version).read();
                    if version.major != DLPACK_VERSION.major {
                        return Err(dlpack_error(format!(
                            "unsupported DLPack major version {}",
                            version.major
                        )));
                    }
                    let flags = ptr::addr_of!((*pointer).flags).read();
                    if flags & !(DLPACK_FLAG_READ_ONLY | DLPACK_FLAG_IS_COPIED) != 0 {
                        return Err(dlpack_error(format!("unsupported DLPack flags {flags:#x}")));
                    }
                    let tensor = &(*pointer).dl_tensor;
                    if version.minor >= 2 && tensor.ndim != 0 && tensor.strides.is_null() {
                        return Err(dlpack_error("DLPack 1.2+ requires explicit strides"));
                    }
                    Ok((tensor, flags))
                }
            }
        }
    }
}

impl Drop for ManagedTensor {
    fn drop(&mut self) {
        unsafe {
            match self.pointer {
                ManagedPointer::Legacy(pointer) => call_managed_deleter(pointer.as_ptr()),
                ManagedPointer::Versioned(pointer) => call_versioned_deleter(pointer.as_ptr()),
            }
        }
    }
}

/// Internal state retained while exporting a tensor to a DLPack capsule.
#[derive(Debug)]
struct ForeignTensorInner {
    _managed: ManagedTensor,
    data: NonNull<f32>,
    len: usize,
    read_only: bool,
}

// SAFETY: `ForeignTensorInner` only stores raw pointers to externally managed data and never
// provides mutable access to the underlying buffer. The DLPack deleter is assumed to be safe to
// call from any thread because `Tensor` can be shared across rayon workers.
unsafe impl Send for ForeignTensorInner {}
unsafe impl Sync for ForeignTensorInner {}

#[derive(Clone, Debug)]
pub struct ForeignTensor {
    inner: Arc<ForeignTensorInner>,
}

impl ForeignTensor {
    /// # Safety
    /// `managed` must reference a valid `DLManagedTensor` whose lifetime is
    /// owned by the caller. `data` must point to the first element of the
    /// tensor and remain valid for `len` initialized elements until release.
    /// The byte span must not exceed `isize::MAX`; the ownership and
    /// synchronization requirements of [`ManagedTensor::from_legacy`] apply.
    pub unsafe fn new(managed: NonNull<DLManagedTensor>, data: NonNull<f32>, len: usize) -> Self {
        Self {
            inner: Arc::new(ForeignTensorInner {
                _managed: ManagedTensor {
                    pointer: ManagedPointer::Legacy(managed),
                },
                data,
                len,
                read_only: false,
            }),
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

    pub fn is_read_only(&self) -> bool {
        self.inner.read_only
    }
}

#[derive(Clone, Debug)]
pub enum ExportData {
    Owned(Arc<AlignedVec>),
    Foreign(ForeignTensor),
}

impl ExportData {
    pub fn is_read_only(&self) -> bool {
        matches!(self, Self::Foreign(buffer) if buffer.is_read_only())
    }

    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Owned(data) => data.as_slice(),
            Self::Foreign(buffer) => buffer.as_slice(),
        }
    }

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

/// Release a versioned producer handle using only the stable ABI prefix.
///
/// # Safety
/// `ptr` must be null or a live, aligned versioned handle whose release
/// obligation is owned by the caller. It must not be released again.
pub unsafe fn call_versioned_deleter(ptr: *mut DLManagedTensorVersioned) {
    if !ptr.is_null() {
        if let Some(deleter) = ptr::addr_of!((*ptr).deleter).read() {
            deleter(ptr);
        }
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

/// Release a versioned handle created by SpiralTorch.
///
/// # Safety
/// `ptr` must be null or an unconsumed handle returned by a SpiralTorch
/// versioned export. Ownership is consumed by this call.
pub unsafe extern "C" fn drop_exported_versioned_state(ptr: *mut DLManagedTensorVersioned) {
    if !ptr.is_null() {
        let boxed = Box::from_raw(ptr);
        if !boxed.manager_ctx.is_null() {
            drop(Box::from_raw(boxed.manager_ctx as *mut ManagedTensorState));
        }
    }
}

/// Capsule name required by the DLPack specification for live tensors.
pub static DLPACK_CAPSULE_NAME: &std::ffi::CStr = c"dltensor";

/// Capsule name used once the tensor has been consumed by a downstream framework.
pub static USED_DLPACK_CAPSULE_NAME: &std::ffi::CStr = c"used_dltensor";

pub static DLPACK_VERSIONED_CAPSULE_NAME: &std::ffi::CStr = c"dltensor_versioned";
pub static USED_DLPACK_VERSIONED_CAPSULE_NAME: &std::ffi::CStr = c"used_dltensor_versioned";

fn dlpack_error(message: impl Into<String>) -> TensorError {
    TensorError::DlpackError {
        message: message.into(),
    }
}

pub(crate) fn export_tensor(
    mut data: ExportData,
    rows: usize,
    cols: usize,
    options: DlpackExportOptions,
) -> PureResult<ManagedTensor> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| dlpack_error("tensor volume overflow"))?;
    if expected != data.len() {
        return Err(TensorError::DataLength {
            expected,
            got: data.len(),
        });
    }
    let rows = i64::try_from(rows).map_err(|_| dlpack_error("tensor rows exceed i64 range"))?;
    let cols = i64::try_from(cols).map_err(|_| dlpack_error("tensor cols exceed i64 range"))?;
    let needs_copy = options.protocol == DlpackProtocol::Legacy && data.is_read_only();
    let copied = match options.copy {
        DlpackCopyPolicy::Never if needs_copy => return Err(TensorError::DlpackCopyRequired),
        DlpackCopyPolicy::Always => true,
        DlpackCopyPolicy::IfNeeded => needs_copy,
        DlpackCopyPolicy::Never => false,
    };
    if copied {
        data = ExportData::Owned(Arc::new(aligned_from_slice(data.as_slice())));
    }
    let flags = if copied {
        DLPACK_FLAG_IS_COPIED
    } else if data.is_read_only() {
        DLPACK_FLAG_READ_ONLY
    } else {
        0
    };
    let mut state = Box::new(ManagedTensorState::new(
        data,
        vec![rows, cols].into_boxed_slice(),
        vec![cols, 1].into_boxed_slice(),
    ));
    let dl_tensor = DLTensor {
        data: if state.data.is_empty() {
            ptr::null_mut()
        } else {
            state.data.as_ptr() as *mut c_void
        },
        device: DLDevice {
            device_type: DLDeviceType::Cpu as i32,
            device_id: 0,
        },
        ndim: 2,
        dtype: DLDataType {
            code: DLDataTypeCode::Float as u8,
            bits: 32,
            lanes: 1,
        },
        shape: state.shape.as_mut_ptr(),
        strides: state.strides.as_mut_ptr(),
        byte_offset: 0,
    };
    let manager_ctx = Box::into_raw(state).cast();
    let pointer = match options.protocol {
        DlpackProtocol::Legacy => {
            ManagedPointer::Legacy(NonNull::from(Box::leak(Box::new(DLManagedTensor {
                dl_tensor,
                manager_ctx,
                deleter: Some(drop_exported_state),
            }))))
        }
        DlpackProtocol::Versioned => ManagedPointer::Versioned(NonNull::from(Box::leak(Box::new(
            DLManagedTensorVersioned {
                version: DLPACK_VERSION,
                manager_ctx,
                deleter: Some(drop_exported_versioned_state),
                flags,
                dl_tensor,
            },
        )))),
    };
    Ok(ManagedTensor { pointer })
}

pub(crate) fn import_tensor(managed: ManagedTensor) -> PureResult<(ForeignTensor, usize, usize)> {
    let (tensor, flags) = managed.descriptor()?;
    if tensor.ndim != 2 {
        return Err(dlpack_error(format!(
            "expected 2 dimensions, got {}",
            tensor.ndim
        )));
    }
    if tensor.device.device_type != DLDeviceType::Cpu as i32 || tensor.device.device_id != 0 {
        return Err(dlpack_error("only CPU device 0 tensors are accepted"));
    }
    if tensor.dtype
        != (DLDataType {
            code: DLDataTypeCode::Float as u8,
            bits: 32,
            lanes: 1,
        })
    {
        return Err(dlpack_error("only f32 tensors are supported"));
    }
    if tensor.shape.is_null() || !(tensor.shape as usize).is_multiple_of(mem::align_of::<i64>()) {
        return Err(dlpack_error(
            "dlpack tensor shape pointer is null or misaligned",
        ));
    }
    // The producer's safety contract supplies two readable entries; alignment
    // checks do not prove the allocation's extent.
    let shape = unsafe { slice::from_raw_parts(tensor.shape, 2) };
    let rows = usize::try_from(shape[0]).map_err(|_| dlpack_error("invalid row count"))?;
    let cols = usize::try_from(shape[1]).map_err(|_| dlpack_error("invalid column count"))?;
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| dlpack_error("tensor volume overflow"))?;
    let bytes = len
        .checked_mul(mem::size_of::<f32>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| dlpack_error("tensor byte span exceeds Rust slice limits"))?;
    if !tensor.strides.is_null() {
        if !(tensor.strides as usize).is_multiple_of(mem::align_of::<i64>()) {
            return Err(dlpack_error("dlpack tensor strides pointer is misaligned"));
        }
        let strides = unsafe { slice::from_raw_parts(tensor.strides, 2) };
        if len != 0 && ((rows > 1 && strides[0] != shape[1]) || (cols > 1 && strides[1] != 1)) {
            return Err(dlpack_error(format!(
                "only contiguous row-major tensors are supported; received strides {strides:?}"
            )));
        }
    }
    let offset = usize::try_from(tensor.byte_offset)
        .map_err(|_| dlpack_error("byte offset exceeds address range"))?;
    if !offset.is_multiple_of(mem::size_of::<f32>()) {
        return Err(dlpack_error("byte offset is not aligned to f32 elements"));
    }
    let data = if len == 0 {
        NonNull::dangling()
    } else {
        let address = (tensor.data as usize)
            .checked_add(offset)
            .filter(|address| address.checked_add(bytes).is_some())
            .ok_or_else(|| dlpack_error("tensor byte span causes pointer overflow"))?;
        if tensor.data.is_null() || !address.is_multiple_of(mem::align_of::<f32>()) {
            return Err(dlpack_error(
                "dlpack tensor data pointer is null or misaligned",
            ));
        }
        NonNull::new(tensor.data.cast::<u8>().wrapping_add(offset).cast::<f32>())
            .ok_or_else(|| dlpack_error("dlpack tensor data pointer is null"))?
    };
    let foreign = ForeignTensor {
        inner: Arc::new(ForeignTensorInner {
            _managed: managed,
            data,
            len,
            read_only: flags & DLPACK_FLAG_READ_ONLY != 0,
        }),
    };
    Ok((foreign, rows, cols))
}

// These assertions are also checked in cross-compiles, without executing tests.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(mem::size_of::<DLTensor>() == 48);
    assert!(mem::size_of::<DLManagedTensorVersioned>() == 80);
    assert!(mem::offset_of!(DLTensor, byte_offset) == 40);
};

#[cfg(target_arch = "wasm32")]
const _: () = {
    assert!(mem::size_of::<DLTensor>() == 40);
    assert!(mem::size_of::<DLManagedTensorVersioned>() == 64);
    assert!(mem::offset_of!(DLTensor, byte_offset) == 32);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
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

    struct TrackedExport {
        _owner: ManagedTensor,
        drops: Arc<AtomicUsize>,
    }

    unsafe extern "C" fn drop_tracked_export(pointer: *mut DLManagedTensorVersioned) {
        let header = Box::from_raw(pointer);
        let ctx = Box::from_raw(header.manager_ctx.cast::<TrackedExport>());
        ctx.drops.fetch_add(1, Ordering::SeqCst);
    }

    fn tracked_export(flags: u64) -> (*mut DLManagedTensorVersioned, Arc<AtomicUsize>) {
        let tensor = Tensor::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let owner = tensor
            .export_dlpack(DlpackExportOptions::default())
            .unwrap();
        // Copy only the descriptor. The original owner retains all allocations.
        let dl_tensor =
            unsafe { ptr::read(&(*owner.as_ptr().cast::<DLManagedTensorVersioned>()).dl_tensor) };
        let drops = Arc::new(AtomicUsize::new(0));
        let ctx = Box::new(TrackedExport {
            _owner: owner,
            drops: drops.clone(),
        });
        let header = Box::new(DLManagedTensorVersioned {
            version: DLPACK_VERSION,
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(drop_tracked_export),
            flags,
            dl_tensor,
        });
        (Box::into_raw(header), drops)
    }

    #[test]
    fn managed_roundtrip_preserves_storage_for_both_protocols() {
        for protocol in [DlpackProtocol::Legacy, DlpackProtocol::Versioned] {
            let tensor = Tensor::from_vec(2, 2, vec![1., 2., 3., 4.]).unwrap();
            let owner = tensor
                .export_dlpack(DlpackExportOptions {
                    protocol,
                    copy: DlpackCopyPolicy::Never,
                })
                .unwrap();
            assert_eq!(owner.protocol(), protocol);
            let restored = Tensor::from_managed_dlpack(owner).unwrap();
            assert_eq!(tensor.data().as_ptr(), restored.data().as_ptr());
            drop(tensor);
            assert_eq!(restored.data(), &[1., 2., 3., 4.]);
        }
    }

    #[test]
    fn explicit_copy_is_independent_and_flagged() {
        let tensor = Tensor::from_vec(1, 2, vec![3., 5.]).unwrap();
        let owner = tensor
            .export_dlpack(DlpackExportOptions {
                copy: DlpackCopyPolicy::Always,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(owner.descriptor().unwrap().1, DLPACK_FLAG_IS_COPIED);
        let restored = Tensor::from_managed_dlpack(owner).unwrap();
        assert_ne!(tensor.data().as_ptr(), restored.data().as_ptr());
        assert_eq!(tensor.data(), restored.data());
        let shared = restored
            .export_dlpack(DlpackExportOptions::default())
            .unwrap();
        assert_eq!(
            shared.descriptor().unwrap().1,
            0,
            "an earlier producer copy is not our copy"
        );
    }

    #[test]
    fn read_only_storage_is_shared_then_detached_on_mutation() {
        let (pointer, drops) = tracked_export(DLPACK_FLAG_READ_ONLY);
        let source_ptr = unsafe { (*pointer).dl_tensor.data.cast::<f32>() };
        let mut tensor = unsafe { Tensor::from_dlpack_versioned(pointer).unwrap() };
        assert_eq!(tensor.data().as_ptr(), source_ptr);
        let shared = tensor
            .export_dlpack(DlpackExportOptions::default())
            .unwrap();
        assert_eq!(shared.descriptor().unwrap().1, DLPACK_FLAG_READ_ONLY);
        let legacy_options = DlpackExportOptions {
            protocol: DlpackProtocol::Legacy,
            copy: DlpackCopyPolicy::Never,
        };
        assert!(matches!(
            tensor.export_dlpack(legacy_options),
            Err(TensorError::DlpackCopyRequired)
        ));
        for protocol in [DlpackProtocol::Legacy, DlpackProtocol::Versioned] {
            let copy = if protocol == DlpackProtocol::Legacy {
                DlpackCopyPolicy::IfNeeded
            } else {
                DlpackCopyPolicy::Always
            };
            let exported = tensor
                .export_dlpack(DlpackExportOptions { protocol, copy })
                .unwrap();
            assert_eq!(
                exported.descriptor().unwrap().1,
                if protocol == DlpackProtocol::Versioned {
                    DLPACK_FLAG_IS_COPIED
                } else {
                    0
                }
            );
            let detached = Tensor::from_managed_dlpack(exported).unwrap();
            assert_ne!(detached.data().as_ptr(), source_ptr);
            assert_eq!(detached.data(), tensor.data());
        }
        tensor.data_mut()[0] = 42.;
        assert_ne!(tensor.data().as_ptr(), source_ptr);
        let original = Tensor::from_managed_dlpack(shared).unwrap();
        assert_eq!(original.data()[0], 1.);
        drop(tensor);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(original);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unconsumed_versioned_handle_releases_once() {
        let (pointer, drops) = tracked_export(0);
        let owner = unsafe { ManagedTensor::from_versioned(pointer).unwrap() };
        drop(owner);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        unsafe {
            call_versioned_deleter(ptr::null_mut());
            drop_exported_versioned_state(ptr::null_mut());
        }
    }

    #[test]
    fn malformed_versioned_headers_release_once() {
        let mutations: &[fn(&mut DLManagedTensorVersioned)] = &[
            |h| h.flags = 1 << 63,
            |h| h.dl_tensor.device.device_type = DLDeviceType::WebGpu as i32,
            |h| h.dl_tensor.device.device_id = 1,
            |h| h.dl_tensor.ndim = 3,
            |h| h.dl_tensor.dtype.bits = 64,
            |h| h.dl_tensor.dtype.lanes = 2,
            |h| h.dl_tensor.shape = ptr::null_mut(),
            |h| h.dl_tensor.shape = h.dl_tensor.shape.cast::<u8>().wrapping_add(1).cast(),
            |h| h.dl_tensor.strides = h.dl_tensor.strides.cast::<u8>().wrapping_add(1).cast(),
            |h| h.dl_tensor.data = ptr::null_mut(),
            |h| h.dl_tensor.data = h.dl_tensor.data.cast::<u8>().wrapping_add(1).cast(),
            |h| h.dl_tensor.byte_offset = 1,
            |h| h.dl_tensor.byte_offset = u64::MAX - 3,
            |h| unsafe { *h.dl_tensor.shape = -1 },
            |h| unsafe { *h.dl_tensor.shape = i64::MAX },
            |h| unsafe { *h.dl_tensor.shape = isize::MAX as i64 / 4 },
            |h| unsafe { *h.dl_tensor.strides = 4 },
            |h| {
                h.version.minor = 2;
                h.dl_tensor.strides = ptr::null_mut();
            },
        ];
        for (index, mutate) in mutations.iter().enumerate() {
            let (pointer, drops) = tracked_export(0);
            unsafe { mutate(&mut *pointer) };
            let result = unsafe { Tensor::from_dlpack_versioned(pointer) };
            assert!(
                matches!(result, Err(TensorError::DlpackError { .. })),
                "case {index}: {result:?}"
            );
            assert_eq!(drops.load(Ordering::SeqCst), 1, "case {index}");
        }
    }

    #[test]
    fn unknown_major_version_reads_only_stable_prefix() {
        #[repr(C)]
        struct Prefix {
            version: DLPackVersion,
            manager_ctx: *mut c_void,
            deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
        }
        unsafe extern "C" fn release_prefix(pointer: *mut DLManagedTensorVersioned) {
            let prefix = Box::from_raw(pointer.cast::<Prefix>());
            let drops = Box::from_raw(prefix.manager_ctx.cast::<Arc<AtomicUsize>>());
            drops.fetch_add(1, Ordering::SeqCst);
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let prefix = Box::new(Prefix {
            version: DLPackVersion { major: 2, minor: 0 },
            manager_ctx: Box::into_raw(Box::new(drops.clone())).cast(),
            deleter: Some(release_prefix),
        });
        let result = unsafe { Tensor::from_dlpack_versioned(Box::into_raw(prefix).cast()) };
        assert!(matches!(result, Err(TensorError::DlpackError { .. })));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn missing_deleter_preserves_externally_owned_handle() {
        let (pointer, drops) = tracked_export(0);
        unsafe { (*pointer).deleter = None };
        let tensor = unsafe { Tensor::from_dlpack_versioned(pointer).unwrap() };
        assert_eq!(tensor.data(), &[1., 2., 3., 4., 5., 6.]);
        drop(tensor);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        // This producer owns the handle independently when no deleter is supplied.
        unsafe { drop_tracked_export(pointer) };
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn empty_tensors_export_null_data_and_roundtrip() {
        for (rows, cols) in [(0, 0), (0, 3), (2, 0)] {
            for protocol in [DlpackProtocol::Legacy, DlpackProtocol::Versioned] {
                let tensor = Tensor::from_vec(rows, cols, vec![]).unwrap();
                let owner = tensor
                    .export_dlpack(DlpackExportOptions {
                        protocol,
                        ..Default::default()
                    })
                    .unwrap();
                assert!(owner.descriptor().unwrap().0.data.is_null());
                let restored = Tensor::from_managed_dlpack(owner).unwrap();
                assert_eq!(restored.shape(), (rows, cols));
                assert!(restored.data().is_empty());
            }
        }
    }

    #[test]
    fn singleton_strides_and_newer_minor_versions_are_supported() {
        for (rows, cols, strides) in [(1, 6, [-7, 1]), (6, 1, [1, -7])] {
            let (pointer, drops) = tracked_export(0);
            unsafe {
                (*pointer).version.minor = 3;
                ptr::copy_nonoverlapping([rows, cols].as_ptr(), (*pointer).dl_tensor.shape, 2);
                ptr::copy_nonoverlapping(strides.as_ptr(), (*pointer).dl_tensor.strides, 2);
            }
            let tensor = unsafe { Tensor::from_dlpack_versioned(pointer).unwrap() };
            assert_eq!(tensor.shape(), (rows as usize, cols as usize));
            assert_eq!(tensor.data(), &[1., 2., 3., 4., 5., 6.]);
            drop(tensor);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn overflowed_views_cannot_create_exportable_shapes() {
        let tensor = Tensor::from_vec(0, 0, vec![]).unwrap();
        assert!(matches!(
            tensor.view(usize::MAX / 2 + 1, 2),
            Err(TensorError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn abi_layout_and_device_codes_match_dlpack() {
        assert_eq!(mem::size_of::<DLPackVersion>(), 8);
        assert_eq!(mem::size_of::<DLDevice>(), 8);
        assert_eq!(mem::size_of::<DLDataType>(), 4);
        assert_eq!(mem::offset_of!(DLManagedTensorVersioned, manager_ctx), 8);
        assert_eq!(
            mem::offset_of!(DLManagedTensorVersioned, deleter),
            8 + mem::size_of::<usize>()
        );
        assert_eq!(DLDeviceType::OneApi as i32, 14);
        assert_eq!(DLDeviceType::WebGpu as i32, 15);
    }
}
