// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::{rccl_allgather_layout, DeviceInfo, GemmActivation, HipErr};
use libloading::Library;
use std::convert::TryFrom;
use std::ffi::{c_char, c_void, CStr};
use std::sync::{Mutex, OnceLock};

pub type HipPtr = *mut c_void;
#[allow(non_camel_case_types)]
type hipError_t = i32;
#[allow(non_camel_case_types)]
pub(crate) type hipStream_t = *mut c_void;

const HIP_SUCCESS: hipError_t = 0;
const RCCL_SUCCESS: i32 = 0;
const RCCL_UINT64: i32 = 5;
const HIP_DEVICE_NAME_MAX: usize = 256;

const HIP_HOST_MALLOC_DEFAULT: u32 = 0;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct RcclComm {
    pub internal: *mut c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct RcclUniqueId {
    pub internal: [u8; 128],
}

#[repr(i32)]
#[derive(Copy, Clone)]
enum HipMemcpyKind {
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

extern "C" {
    fn hipMalloc(ptr: *mut HipPtr, size: usize) -> hipError_t;
    fn hipFree(ptr: HipPtr) -> hipError_t;
    fn hipMemcpyAsync(
        dst: HipPtr,
        src: *const c_void,
        size_bytes: usize,
        kind: HipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;
    fn hipMemsetAsync(
        dst: HipPtr,
        value: i32,
        size_bytes: usize,
        stream: hipStream_t,
    ) -> hipError_t;
    fn hipGetDevice(device: *mut i32) -> hipError_t;
    fn hipSetDevice(device: i32) -> hipError_t;
    fn hipGetDeviceCount(count: *mut i32) -> hipError_t;
    fn hipDeviceGetName(name: *mut c_char, len: i32, device: i32) -> hipError_t;
    fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    fn hipDeviceSynchronize() -> hipError_t;
    fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    fn hipHostMalloc(ptr: *mut HipPtr, size: usize, flags: u32) -> hipError_t;
    fn hipHostFree(ptr: HipPtr) -> hipError_t;
    fn hipGetErrorName(error: hipError_t) -> *const c_char;
    fn hipGetErrorString(error: hipError_t) -> *const c_char;

    fn rcclAllGather(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: i32,
        comm: RcclComm,
        stream: hipStream_t,
    ) -> i32;
    fn rcclGetErrorName(result: i32) -> *const c_char;
    fn rcclGetErrorString(result: i32) -> *const c_char;

    fn st_pack_vals_idx_u64(
        vals: *const f32,
        idx: *const i32,
        out: *mut u64,
        total: i32,
        stream: hipStream_t,
    ) -> hipError_t;
    fn st_kway_merge_shared_heap_real_keepk_u64(
        cand_packed: *const u64,
        rows: i32,
        total: i32,
        k_final: i32,
        out_vals: *mut f32,
        out_idx: *mut i32,
        stream: hipStream_t,
    ) -> hipError_t;
    fn st_kway_merge_warp_coop_keepk_u64(
        cand_packed: *const u64,
        rows: i32,
        total: i32,
        k_final: i32,
        out_vals: *mut f32,
        out_idx: *mut i32,
        stream: hipStream_t,
    ) -> hipError_t;
    fn st_kway_merge_bitonic_f32(
        cand_vals: *const f32,
        cand_idx: *const i32,
        rows: i32,
        total: i32,
        k_final: i32,
        out_vals: *mut f32,
        out_idx: *mut i32,
        stream: hipStream_t,
    ) -> hipError_t;
    fn st_kway_merge_bitonic_u64(
        cand_packed: *const u64,
        rows: i32,
        total: i32,
        k_final: i32,
        out_vals: *mut f32,
        out_idx: *mut i32,
        stream: hipStream_t,
    ) -> hipError_t;
    fn st_topk_tile_bitonic_u64(
        cand_packed: *const u64,
        rows: i32,
        total: i32,
        k_final: i32,
        out: *mut u64,
        stream: hipStream_t,
    ) -> hipError_t;
    fn st_topk_pass1_f32(
        input: *const f32,
        rows: i32,
        cols: i32,
        stride: i32,
        k: i32,
        out_vals: *mut f32,
        out_idx: *mut i32,
        stream: hipStream_t,
    ) -> hipError_t;
    fn st_gemm_epilogue_f32(
        output: *mut f32,
        bias: *const f32,
        residual: *const f32,
        total: i32,
        cols: i32,
        activation: i32,
        has_residual: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

fn read_cstring(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return "<null>".into();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

pub(crate) fn hip_result(rc: hipError_t, ctx: &str) -> Result<(), HipErr> {
    if rc == HIP_SUCCESS {
        return Ok(());
    }
    let name = read_cstring(unsafe { hipGetErrorName(rc) });
    let desc = read_cstring(unsafe { hipGetErrorString(rc) });
    Err(HipErr::Other(format!("{ctx}: {name} ({rc}) - {desc}")))
}

fn rccl_result(rc: i32, ctx: &str) -> Result<(), HipErr> {
    if rc == RCCL_SUCCESS {
        return Ok(());
    }
    let name = read_cstring(unsafe { rcclGetErrorName(rc) });
    let desc = read_cstring(unsafe { rcclGetErrorString(rc) });
    Err(HipErr::Other(format!("{ctx}: {name} ({rc}) - {desc}")))
}

/// Owning HIP stream handle.
///
/// Dropping the stream performs a best-effort synchronization before releasing
/// the native handle. Safe high-level operations additionally install an
/// explicit completion guard so their device allocations outlive all queued
/// work even when an intermediate launch fails.
pub struct HipStream(hipStream_t);

impl HipStream {
    pub fn create() -> Result<Self, HipErr> {
        let mut raw: hipStream_t = std::ptr::null_mut();
        hip_result(unsafe { hipStreamCreate(&mut raw) }, "hipStreamCreate")?;
        Ok(Self(raw))
    }

    #[inline]
    pub(crate) fn raw(&self) -> hipStream_t {
        self.0
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = hipStreamSynchronize(self.0);
                let _ = hipStreamDestroy(self.0);
            }
        }
    }
}

// HIP runtime entry points are thread-safe, and stream operations remain
// serialized by the native stream. Ownership still prevents destruction while
// a borrowed stream is in use.
unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

pub fn malloc(size: usize) -> Result<HipPtr, HipErr> {
    let mut ptr: HipPtr = std::ptr::null_mut();
    hip_result(unsafe { hipMalloc(&mut ptr, size) }, "hipMalloc")?;
    Ok(ptr)
}

/// Releases a device allocation returned by [`malloc`].
///
/// # Safety
///
/// `ptr` must be null or an allocation returned by `malloc` that has not
/// already been released. No queued operation may access it after this call.
pub unsafe fn free(ptr: HipPtr) -> Result<(), HipErr> {
    if ptr.is_null() {
        return Ok(());
    }
    hip_result(hipFree(ptr), "hipFree")
}

pub fn host_malloc(size: usize) -> Result<HipPtr, HipErr> {
    let mut ptr: HipPtr = std::ptr::null_mut();
    hip_result(
        unsafe { hipHostMalloc(&mut ptr, size, HIP_HOST_MALLOC_DEFAULT) },
        "hipHostMalloc",
    )?;
    Ok(ptr)
}

/// Releases a pinned host allocation returned by [`host_malloc`].
///
/// # Safety
///
/// `ptr` must be null or an allocation returned by `host_malloc` that has not
/// already been released. No queued transfer may access it after this call.
pub unsafe fn host_free(ptr: HipPtr) -> Result<(), HipErr> {
    if ptr.is_null() {
        return Ok(());
    }
    hip_result(hipHostFree(ptr), "hipHostFree")
}

/// Enqueues a host-to-device byte copy.
///
/// # Safety
///
/// `dst` and `src` must be valid for `size` bytes and remain valid until
/// `stream` has completed the copy. `dst` must reference device memory and
/// `src` host memory.
pub unsafe fn memcpy_h2d_async(
    dst: HipPtr,
    src: *const u8,
    size: usize,
    stream: &HipStream,
) -> Result<(), HipErr> {
    hip_result(
        hipMemcpyAsync(
            dst,
            src as *const c_void,
            size,
            HipMemcpyKind::HostToDevice,
            stream.raw(),
        ),
        "hipMemcpyAsync(H2D)",
    )
}

/// Enqueues a byte-wise device memset.
///
/// # Safety
///
/// `dst` must reference device memory valid for `size` bytes and remain valid
/// until `stream` has completed the operation.
pub unsafe fn memset_async(
    dst: HipPtr,
    value: u8,
    size: usize,
    stream: &HipStream,
) -> Result<(), HipErr> {
    hip_result(
        hipMemsetAsync(dst, value as i32, size, stream.raw()),
        "hipMemsetAsync",
    )
}

/// Enqueues a device-to-host byte copy.
///
/// # Safety
///
/// `dst` and `src` must be valid for `size` bytes and remain valid until
/// `stream` has completed the copy. `dst` must reference writable host memory
/// and `src` device memory.
pub unsafe fn memcpy_d2h_async(
    dst: *mut u8,
    src: HipPtr,
    size: usize,
    stream: &HipStream,
) -> Result<(), HipErr> {
    hip_result(
        hipMemcpyAsync(
            dst as HipPtr,
            src as *const c_void,
            size,
            HipMemcpyKind::DeviceToHost,
            stream.raw(),
        ),
        "hipMemcpyAsync(D2H)",
    )
}

/// Enqueues a device-to-device byte copy.
///
/// # Safety
///
/// `dst` and `src` must reference device memory valid for `size` bytes and
/// remain valid until `stream` has completed the copy. The ranges must satisfy
/// HIP's overlap requirements.
pub unsafe fn memcpy_d2d_async(
    dst: HipPtr,
    src: HipPtr,
    size: usize,
    stream: &HipStream,
) -> Result<(), HipErr> {
    hip_result(
        hipMemcpyAsync(
            dst,
            src as *const c_void,
            size,
            HipMemcpyKind::DeviceToDevice,
            stream.raw(),
        ),
        "hipMemcpyAsync(D2D)",
    )
}

pub fn device_synchronize() -> Result<(), HipErr> {
    hip_result(unsafe { hipDeviceSynchronize() }, "hipDeviceSynchronize")
}

pub fn stream_synchronize(stream: &HipStream) -> Result<(), HipErr> {
    hip_result(
        unsafe { hipStreamSynchronize(stream.raw()) },
        "hipStreamSynchronize",
    )
}

pub fn get_device() -> Result<i32, HipErr> {
    let mut device = 0i32;
    hip_result(unsafe { hipGetDevice(&mut device) }, "hipGetDevice")?;
    Ok(device)
}

pub fn set_device(device: i32) -> Result<(), HipErr> {
    hip_result(unsafe { hipSetDevice(device) }, "hipSetDevice")
}

pub fn device_count() -> Result<i32, HipErr> {
    let mut count = 0i32;
    hip_result(
        unsafe { hipGetDeviceCount(&mut count) },
        "hipGetDeviceCount",
    )?;
    Ok(count)
}

fn device_name(device: i32) -> Result<String, HipErr> {
    let mut buf = [0i8; HIP_DEVICE_NAME_MAX];
    hip_result(
        unsafe { hipDeviceGetName(buf.as_mut_ptr(), HIP_DEVICE_NAME_MAX as i32, device) },
        "hipDeviceGetName",
    )?;
    Ok(read_cstring(buf.as_ptr()))
}

pub fn enumerate_devices() -> Result<Vec<DeviceInfo>, HipErr> {
    let total = device_count()?;
    let mut devices = Vec::new();
    for device in 0..total {
        let name = device_name(device)?;
        devices.push(DeviceInfo::new(device as u32, name, total > 1));
    }
    Ok(devices)
}

#[derive(Copy, Clone)]
enum GemmMode {
    Standard,
    LhsTranspose,
}

#[derive(Copy, Clone)]
struct GemmSpec {
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    mode: GemmMode,
}

mod rocblas {
    use super::{
        hipStream_t, read_cstring, GemmMode, GemmSpec, HipErr, HipPtr, HipStream, Library, Mutex,
        OnceLock, TryFrom,
    };
    use std::ffi::c_char;
    use std::ptr;

    type RocblasHandle = *mut std::ffi::c_void;
    type RocblasStatus = i32;

    const ROCBLAS_STATUS_SUCCESS: RocblasStatus = 0;

    #[repr(i32)]
    #[derive(Copy, Clone)]
    enum Operation {
        None = 111,
        Transpose = 112,
    }

    struct Symbols {
        create_handle: unsafe extern "C" fn(*mut RocblasHandle) -> RocblasStatus,
        set_stream: unsafe extern "C" fn(RocblasHandle, hipStream_t) -> RocblasStatus,
        sgemm: unsafe extern "C" fn(
            RocblasHandle,
            Operation,
            Operation,
            i32,
            i32,
            i32,
            *const f32,
            *const f32,
            i32,
            *const f32,
            i32,
            *const f32,
            *mut f32,
            i32,
        ) -> RocblasStatus,
        status_to_string: Option<unsafe extern "C" fn(RocblasStatus) -> *const c_char>,
    }

    struct HandleState {
        handle: RocblasHandle,
        current_stream: hipStream_t,
    }
    unsafe impl Send for HandleState {}

    fn library() -> Result<&'static Library, HipErr> {
        static LIB: OnceLock<Result<&'static Library, String>> = OnceLock::new();
        LIB.get_or_init(|| unsafe {
            Library::new("librocblas.so")
                .or_else(|_| Library::new("librocblas.so.0"))
                .map(|lib| {
                    let leaked: &'static mut Library = Box::leak(Box::new(lib));
                    leaked as &'static Library
                })
                .map_err(|err| err.to_string())
        })
        .as_ref()
        .map(|lib| *lib)
        .map_err(|err| HipErr::Other(format!("failed to load rocBLAS: {err}")))
    }

    unsafe fn load_symbols() -> Result<Symbols, HipErr> {
        let lib = library()?;
        let create_handle = *lib
            .get::<unsafe extern "C" fn(*mut RocblasHandle) -> RocblasStatus>(
                b"rocblas_create_handle\0",
            )
            .map_err(|err| HipErr::Other(format!("failed to load rocblas_create_handle: {err}")))?;
        let set_stream = *lib
            .get::<unsafe extern "C" fn(RocblasHandle, hipStream_t) -> RocblasStatus>(
                b"rocblas_set_stream\0",
            )
            .map_err(|err| HipErr::Other(format!("failed to load rocblas_set_stream: {err}")))?;
        let sgemm = *lib
            .get::<unsafe extern "C" fn(
                RocblasHandle,
                Operation,
                Operation,
                i32,
                i32,
                i32,
                *const f32,
                *const f32,
                i32,
                *const f32,
                i32,
                *const f32,
                *mut f32,
                i32,
            ) -> RocblasStatus>(b"rocblas_sgemm\0")
            .map_err(|err| HipErr::Other(format!("failed to load rocblas_sgemm: {err}")))?;
        let status_to_string = lib
            .get::<unsafe extern "C" fn(RocblasStatus) -> *const c_char>(
                b"rocblas_status_to_string\0",
            )
            .map(|sym| *sym)
            .ok();

        Ok(Symbols {
            create_handle,
            set_stream,
            sgemm,
            status_to_string,
        })
    }

    fn symbols() -> Result<&'static Symbols, HipErr> {
        static SYMBOLS: OnceLock<Result<Symbols, String>> = OnceLock::new();
        SYMBOLS
            .get_or_init(|| unsafe { load_symbols().map_err(|err| err.to_string()) })
            .as_ref()
            .map_err(|err| HipErr::Other(err.clone()))
    }

    fn handle_slot() -> &'static Mutex<Option<HandleState>> {
        static HANDLE: OnceLock<Mutex<Option<HandleState>>> = OnceLock::new();
        HANDLE.get_or_init(|| Mutex::new(None))
    }

    fn create_handle(symbols: &Symbols) -> Result<RocblasHandle, HipErr> {
        let mut handle: RocblasHandle = ptr::null_mut();
        rocblas_result(
            unsafe { (symbols.create_handle)(&mut handle) },
            "rocblas_create_handle",
            symbols,
        )?;
        if handle.is_null() {
            return Err(HipErr::Other("rocBLAS returned a null handle".into()));
        }
        Ok(handle)
    }

    fn rocblas_result(status: RocblasStatus, ctx: &str, symbols: &Symbols) -> Result<(), HipErr> {
        if status == ROCBLAS_STATUS_SUCCESS {
            return Ok(());
        }
        let description = symbols
            .status_to_string
            .map(|func| unsafe { func(status) })
            .filter(|ptr| !ptr.is_null())
            .map(read_cstring)
            .unwrap_or_else(|| format!("rocBLAS status {status}"));
        Err(HipErr::Other(format!("{ctx}: {description}")))
    }

    fn with_handle<F, R>(stream: &HipStream, mut f: F) -> Result<R, HipErr>
    where
        F: FnMut(RocblasHandle, &Symbols) -> Result<R, HipErr>,
    {
        let symbols = symbols()?;
        let slot = handle_slot();
        let mut guard = slot
            .lock()
            .map_err(|_| HipErr::Other("failed to lock rocBLAS handle slot".into()))?;
        if guard.is_none() {
            let handle = create_handle(symbols)?;
            *guard = Some(HandleState {
                handle,
                current_stream: ptr::null_mut(),
            });
        }
        let state = guard.as_mut().ok_or_else(|| {
            HipErr::Other("rocBLAS handle slot remained empty after creation".into())
        })?;
        if state.current_stream != stream.raw() {
            rocblas_result(
                unsafe { (symbols.set_stream)(state.handle, stream.raw()) },
                "rocblas_set_stream",
                symbols,
            )?;
            state.current_stream = stream.raw();
        }
        f(state.handle, symbols)
    }

    pub(super) fn sgemm(
        stream: &HipStream,
        spec: GemmSpec,
        lhs: HipPtr,
        rhs: HipPtr,
        out: HipPtr,
    ) -> Result<(), HipErr> {
        let m_i32 = i32::try_from(spec.n)
            .map_err(|_| HipErr::Other("rocBLAS: n dimension overflow".into()))?;
        let n_i32 = i32::try_from(spec.m)
            .map_err(|_| HipErr::Other("rocBLAS: m dimension overflow".into()))?;
        let k_i32 = i32::try_from(spec.k)
            .map_err(|_| HipErr::Other("rocBLAS: k dimension overflow".into()))?;
        // Row-major C is column-major C.T. Swapping operands gives the standard
        // path; lhs.T additionally transposes lhs's column-major view in place.
        let lda = m_i32;
        let (lhs_operation, ldb) = if matches!(spec.mode, GemmMode::LhsTranspose) {
            (Operation::Transpose, n_i32)
        } else {
            (Operation::None, k_i32)
        };
        let ldc = m_i32;

        with_handle(stream, |handle, symbols| {
            let beta = 0.0f32;
            rocblas_result(
                unsafe {
                    (symbols.sgemm)(
                        handle,
                        Operation::None,
                        lhs_operation,
                        m_i32,
                        n_i32,
                        k_i32,
                        &spec.scale,
                        rhs as *const f32,
                        lda,
                        lhs as *const f32,
                        ldb,
                        &beta,
                        out as *mut f32,
                        ldc,
                    )
                },
                "rocblas_sgemm",
                symbols,
            )
        })
    }
}

struct DeviceBuffer(HipPtr);

impl DeviceBuffer {
    fn new(size: usize) -> Result<Self, HipErr> {
        Ok(Self(malloc(size)?))
    }

    fn as_ptr(&self) -> HipPtr {
        self.0
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = free(self.0);
        }
    }
}

struct StreamCompletionGuard<'a> {
    stream: &'a HipStream,
    armed: bool,
}

impl<'a> StreamCompletionGuard<'a> {
    fn new(stream: &'a HipStream) -> Self {
        Self {
            stream,
            armed: true,
        }
    }

    fn finish(mut self) -> Result<(), HipErr> {
        let result = stream_synchronize(self.stream);
        if result.is_ok() {
            self.armed = false;
        }
        result
    }
}

impl Drop for StreamCompletionGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = stream_synchronize(self.stream);
        }
    }
}

fn gemm_scaled_impl(
    spec: GemmSpec,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    let lhs_bytes = lhs
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HipErr::Other("lhs buffer byte length overflow".into()))?;
    let rhs_bytes = rhs
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HipErr::Other("rhs buffer byte length overflow".into()))?;
    let out_bytes = out
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HipErr::Other("output buffer byte length overflow".into()))?;

    let lhs_dev = DeviceBuffer::new(lhs_bytes)?;
    let rhs_dev = DeviceBuffer::new(rhs_bytes)?;
    let out_dev = DeviceBuffer::new(out_bytes)?;
    let stream = HipStream::create()?;
    let completion = StreamCompletionGuard::new(&stream);

    unsafe {
        memcpy_h2d_async(
            lhs_dev.as_ptr(),
            lhs.as_ptr() as *const u8,
            lhs_bytes,
            &stream,
        )?;
        memcpy_h2d_async(
            rhs_dev.as_ptr(),
            rhs.as_ptr() as *const u8,
            rhs_bytes,
            &stream,
        )?;
    }

    rocblas::sgemm(
        &stream,
        spec,
        lhs_dev.as_ptr(),
        rhs_dev.as_ptr(),
        out_dev.as_ptr(),
    )?;

    unsafe {
        memcpy_d2h_async(
            out.as_mut_ptr() as *mut u8,
            out_dev.as_ptr(),
            out_bytes,
            &stream,
        )?;
    }
    completion.finish()?;
    Ok(())
}

pub fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    gemm_scaled_f32(m, n, k, 1.0, lhs, rhs, out)
}

pub fn gemm_scaled_f32(
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    gemm_scaled_impl(
        GemmSpec {
            m,
            n,
            k,
            scale,
            mode: GemmMode::Standard,
        },
        lhs,
        rhs,
        out,
    )
}

pub fn gemm_lhs_transpose_scaled_f32(
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    gemm_scaled_impl(
        GemmSpec {
            m,
            n,
            k,
            scale,
            mode: GemmMode::LhsTranspose,
        },
        lhs,
        rhs,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_bias_activation_f32(
    m: usize,
    n: usize,
    k: usize,
    activation: GemmActivation,
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    residual: Option<&[f32]>,
    out: &mut [f32],
) -> Result<(), HipErr> {
    if out.is_empty() {
        return Ok(());
    }

    let lhs_bytes = lhs
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HipErr::Other("lhs buffer byte length overflow".into()))?;
    let rhs_bytes = rhs
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HipErr::Other("rhs buffer byte length overflow".into()))?;
    let bias_bytes = bias
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HipErr::Other("bias buffer byte length overflow".into()))?;
    let out_bytes = out
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HipErr::Other("output buffer byte length overflow".into()))?;
    let total = i32::try_from(out.len())
        .map_err(|_| HipErr::Other("GEMM epilogue element count exceeds i32".into()))?;
    let cols = i32::try_from(n)
        .map_err(|_| HipErr::Other("GEMM epilogue column count exceeds i32".into()))?;

    let out_dev = DeviceBuffer::new(out_bytes)?;
    let lhs_dev = (k != 0).then(|| DeviceBuffer::new(lhs_bytes)).transpose()?;
    let rhs_dev = (k != 0).then(|| DeviceBuffer::new(rhs_bytes)).transpose()?;
    let bias_dev = DeviceBuffer::new(bias_bytes)?;
    let residual_dev = if let Some(residual) = residual {
        let bytes = residual
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| HipErr::Other("residual buffer byte length overflow".into()))?;
        Some((DeviceBuffer::new(bytes)?, residual, bytes))
    } else {
        None
    };
    let stream = HipStream::create()?;
    let completion = StreamCompletionGuard::new(&stream);

    if k == 0 {
        unsafe { memset_async(out_dev.as_ptr(), 0, out_bytes, &stream)? };
    } else {
        let lhs_dev = lhs_dev
            .as_ref()
            .ok_or_else(|| HipErr::Other("missing GEMM lhs device buffer".into()))?;
        let rhs_dev = rhs_dev
            .as_ref()
            .ok_or_else(|| HipErr::Other("missing GEMM rhs device buffer".into()))?;
        unsafe {
            memcpy_h2d_async(
                lhs_dev.as_ptr(),
                lhs.as_ptr() as *const u8,
                lhs_bytes,
                &stream,
            )?;
            memcpy_h2d_async(
                rhs_dev.as_ptr(),
                rhs.as_ptr() as *const u8,
                rhs_bytes,
                &stream,
            )?;
        }
        rocblas::sgemm(
            &stream,
            GemmSpec {
                m,
                n,
                k,
                scale: 1.0,
                mode: GemmMode::Standard,
            },
            lhs_dev.as_ptr(),
            rhs_dev.as_ptr(),
            out_dev.as_ptr(),
        )?;
    }

    unsafe {
        memcpy_h2d_async(
            bias_dev.as_ptr(),
            bias.as_ptr() as *const u8,
            bias_bytes,
            &stream,
        )?;
    }

    if let Some((device, residual, bytes)) = residual_dev.as_ref() {
        unsafe {
            memcpy_h2d_async(
                device.as_ptr(),
                residual.as_ptr() as *const u8,
                *bytes,
                &stream,
            )?;
        }
    }

    let residual_ptr = residual_dev
        .as_ref()
        .map(|(device, _, _)| device.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    hip_result(
        unsafe {
            st_gemm_epilogue_f32(
                out_dev.as_ptr() as *mut f32,
                bias_dev.as_ptr() as *const f32,
                residual_ptr as *const f32,
                total,
                cols,
                activation.hip_code(),
                i32::from(residual_dev.is_some()),
                stream.raw(),
            )
        },
        "st_gemm_epilogue_f32",
    )?;

    unsafe {
        memcpy_d2h_async(
            out.as_mut_ptr() as *mut u8,
            out_dev.as_ptr(),
            out_bytes,
            &stream,
        )?;
    }
    completion.finish()?;
    Ok(())
}

/// Enqueues packing `(value, index)` pairs into `u64` words.
///
/// # Safety
///
/// For positive `total`, `vals`, `idx`, and `out` must be valid device
/// allocations for at least `total` elements of their respective types and
/// remain alive until `stream` completes.
pub unsafe fn pack_vals_idx_u64(
    vals: *const f32,
    idx: *const i32,
    out: *mut u64,
    total: i32,
    stream: &HipStream,
) -> Result<(), HipErr> {
    hip_result(
        unsafe { st_pack_vals_idx_u64(vals, idx, out, total, stream.raw()) },
        "st_pack_vals_idx_u64",
    )
}

/// Enqueues the shared-heap packed k-way merge kernel.
///
/// # Safety
///
/// All pointers must refer to device allocations satisfying the native
/// `(rows, total, k_final)` layout and remain alive until `stream` completes.
/// Output ranges must be writable and must not overlap the input.
pub unsafe fn kway_merge_shared_heap_real_keepk_u64(
    cand_packed: *const u64,
    rows: i32,
    total: i32,
    k_final: i32,
    out_vals: *mut f32,
    out_idx: *mut i32,
    stream: &HipStream,
) -> Result<(), HipErr> {
    hip_result(
        unsafe {
            st_kway_merge_shared_heap_real_keepk_u64(
                cand_packed,
                rows,
                total,
                k_final,
                out_vals,
                out_idx,
                stream.raw(),
            )
        },
        "st_kway_merge_shared_heap_real_keepk_u64",
    )
}

/// Alias for [`kway_merge_shared_heap_real_keepk_u64`].
///
/// # Safety
///
/// The same pointer validity, layout, lifetime, and non-overlap requirements as
/// [`kway_merge_shared_heap_real_keepk_u64`] apply.
pub unsafe fn kway_merge_shared_heap_keepk_u64(
    cand_packed: *const u64,
    rows: i32,
    total: i32,
    k_final: i32,
    out_vals: *mut f32,
    out_idx: *mut i32,
    stream: &HipStream,
) -> Result<(), HipErr> {
    unsafe {
        kway_merge_shared_heap_real_keepk_u64(
            cand_packed,
            rows,
            total,
            k_final,
            out_vals,
            out_idx,
            stream,
        )
    }
}

/// Enqueues the cooperative-warp packed k-way merge kernel.
///
/// # Safety
///
/// All pointers must refer to device allocations satisfying the native
/// `(rows, total, k_final)` layout and remain alive until `stream` completes.
/// Output ranges must be writable and must not overlap the input.
pub unsafe fn kway_merge_warp_coop_keepk_u64(
    cand_packed: *const u64,
    rows: i32,
    total: i32,
    k_final: i32,
    out_vals: *mut f32,
    out_idx: *mut i32,
    stream: &HipStream,
) -> Result<(), HipErr> {
    hip_result(
        unsafe {
            st_kway_merge_warp_coop_keepk_u64(
                cand_packed,
                rows,
                total,
                k_final,
                out_vals,
                out_idx,
                stream.raw(),
            )
        },
        "st_kway_merge_warp_coop_keepk_u64",
    )
}

/// Alias for [`kway_merge_warp_coop_keepk_u64`].
///
/// # Safety
///
/// The same pointer validity, layout, lifetime, and non-overlap requirements as
/// [`kway_merge_warp_coop_keepk_u64`] apply.
pub unsafe fn kway_merge_warp_heap_keepk_u64(
    cand_packed: *const u64,
    rows: i32,
    total: i32,
    k_final: i32,
    out_vals: *mut f32,
    out_idx: *mut i32,
    stream: &HipStream,
) -> Result<(), HipErr> {
    unsafe {
        kway_merge_warp_coop_keepk_u64(cand_packed, rows, total, k_final, out_vals, out_idx, stream)
    }
}

/// Enqueues the packed bitonic k-way merge kernel.
///
/// # Safety
///
/// All pointers must refer to device allocations satisfying the native
/// `(rows, total, k_final)` layout and remain alive until `stream` completes.
/// Output ranges must be writable and must not overlap the input.
pub unsafe fn kway_merge_bitonic_u64(
    cand_packed: *const u64,
    rows: i32,
    total: i32,
    k_final: i32,
    out_vals: *mut f32,
    out_idx: *mut i32,
    stream: &HipStream,
) -> Result<(), HipErr> {
    hip_result(
        unsafe {
            st_kway_merge_bitonic_u64(
                cand_packed,
                rows,
                total,
                k_final,
                out_vals,
                out_idx,
                stream.raw(),
            )
        },
        "st_kway_merge_bitonic_u64",
    )
}

/// Raw arguments for the f32 bitonic k-way merge kernel.
pub struct KwayMergeBitonicF32Args<'a> {
    pub cand_vals: *const f32,
    pub cand_idx: *const i32,
    pub rows: i32,
    pub total: i32,
    pub k_final: i32,
    pub out_vals: *mut f32,
    pub out_idx: *mut i32,
    pub stream: &'a HipStream,
}

/// Enqueues the f32 bitonic k-way merge kernel.
///
/// # Safety
///
/// Pointer fields must refer to device allocations satisfying the native
/// `(rows, total, k_final)` layout and remain alive until `stream` completes.
/// Output ranges must be writable and must not overlap either input.
pub unsafe fn kway_merge_bitonic_f32(args: KwayMergeBitonicF32Args<'_>) -> Result<(), HipErr> {
    if args.rows <= 0 || args.total <= 0 || args.k_final <= 0 {
        return Ok(());
    }
    hip_result(
        st_kway_merge_bitonic_f32(
            args.cand_vals,
            args.cand_idx,
            args.rows,
            args.total,
            args.k_final,
            args.out_vals,
            args.out_idx,
            args.stream.raw(),
        ),
        "st_kway_merge_bitonic_f32",
    )
}

/// Enqueues the packed bitonic tile top-k kernel.
///
/// # Safety
///
/// Pointer arguments must refer to device allocations satisfying the native
/// `(rows, total, k_final)` layout and remain alive until `stream` completes.
/// `out` must be writable and must not overlap `cand_packed`.
pub unsafe fn topk_tile_bitonic_u64(
    cand_packed: *const u64,
    rows: i32,
    total: i32,
    k_final: i32,
    out: *mut u64,
    stream: &HipStream,
) -> Result<(), HipErr> {
    hip_result(
        unsafe { st_topk_tile_bitonic_u64(cand_packed, rows, total, k_final, out, stream.raw()) },
        "st_topk_tile_bitonic_u64",
    )
}

/// Raw arguments for the first-pass f32 top-k kernel.
pub struct TopkPass1F32Args<'a> {
    pub input: *const f32,
    pub rows: i32,
    pub cols: i32,
    pub stride: i32,
    pub k: i32,
    pub out_vals: *mut f32,
    pub out_idx: *mut i32,
    pub stream: &'a HipStream,
}

/// Enqueues the first-pass f32 top-k kernel.
///
/// # Safety
///
/// Pointer fields must refer to correctly sized device allocations for the
/// declared row/column/stride/top-k layout and remain alive until `stream`
/// completes. Output ranges must be writable and non-overlapping.
pub unsafe fn topk_pass1_f32(args: TopkPass1F32Args<'_>) -> Result<(), HipErr> {
    if args.rows <= 0 || args.cols <= 0 || args.k <= 0 {
        return Ok(());
    }
    let stride = if args.stride > 0 {
        args.stride
    } else {
        args.cols
    };
    hip_result(
        st_topk_pass1_f32(
            args.input,
            args.rows,
            args.cols,
            stride,
            args.k,
            args.out_vals,
            args.out_idx,
            args.stream.raw(),
        ),
        "st_topk_pass1_f32",
    )
}

/// Enqueues an RCCL all-gather between device buffers.
///
/// # Safety
///
/// `send` must contain at least `count` `u64` values. `recv` must contain
/// enough writable space for `count * world_size` values as configured by
/// `comm`. Both buffers and the communicator must remain valid until `stream`
/// completes, and the buffers must not overlap.
pub(crate) unsafe fn allgather_u64_dev(
    comm: RcclComm,
    stream: &HipStream,
    send: HipPtr,
    recv: HipPtr,
    count: usize,
) -> Result<(), HipErr> {
    rccl_result(
        unsafe {
            rcclAllGather(
                send as *const c_void,
                recv,
                count,
                RCCL_UINT64,
                comm,
                stream.raw(),
            )
        },
        "rcclAllGather",
    )
}

pub(crate) fn allgather_u64_host(
    comm: RcclComm,
    send: &[u64],
    world_size: usize,
) -> Result<Vec<u64>, HipErr> {
    let layout = rccl_allgather_layout(send.len(), world_size)?;
    if send.is_empty() {
        return Ok(Vec::new());
    }

    let send_dev = DeviceBuffer::new(layout.send_bytes)?;
    let receive_dev = DeviceBuffer::new(layout.receive_bytes)?;
    let stream = HipStream::create()?;
    let mut receive = vec![0u64; layout.receive_count];
    let completion = StreamCompletionGuard::new(&stream);

    unsafe {
        memcpy_h2d_async(
            send_dev.as_ptr(),
            send.as_ptr().cast::<u8>(),
            layout.send_bytes,
            &stream,
        )?;
        allgather_u64_dev(
            comm,
            &stream,
            send_dev.as_ptr(),
            receive_dev.as_ptr(),
            send.len(),
        )?;
        memcpy_d2h_async(
            receive.as_mut_ptr().cast::<u8>(),
            receive_dev.as_ptr(),
            layout.receive_bytes,
            &stream,
        )?;
    }
    completion.finish()?;
    Ok(receive)
}
