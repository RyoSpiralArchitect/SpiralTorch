// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::{DeviceInfo, HipErr};
use std::ffi::{c_char, c_void, CStr};

pub type HipPtr = *mut c_void;
type hipError_t = i32;
pub type hipStream_t = *mut c_void;

const HIP_SUCCESS: hipError_t = 0;
const RCCL_SUCCESS: i32 = 0;
const RCCL_UINT64: i32 = 5;
const HIP_DEVICE_NAME_MAX: usize = 256;

const HIP_HOST_MALLOC_DEFAULT: u32 = 0;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RcclComm {
    pub internal: *mut c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RcclUniqueId {
    pub internal: [u8; 128],
}

#[repr(i32)]
#[derive(Copy, Clone)]
enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
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

pub struct HipStream(pub hipStream_t);

impl HipStream {
    pub fn create() -> Result<Self, HipErr> {
        let mut raw: hipStream_t = std::ptr::null_mut();
        hip_result(unsafe { hipStreamCreate(&mut raw) }, "hipStreamCreate")?;
        Ok(Self(raw))
    }

    #[inline]
    pub fn raw(&self) -> hipStream_t {
        self.0
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = hipStreamDestroy(self.0);
            }
        }
    }
}

unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

pub fn malloc(size: usize) -> Result<HipPtr, HipErr> {
    let mut ptr: HipPtr = std::ptr::null_mut();
    hip_result(unsafe { hipMalloc(&mut ptr, size) }, "hipMalloc")?;
    Ok(ptr)
}

pub fn free(ptr: HipPtr) -> Result<(), HipErr> {
    if ptr.is_null() {
        return Ok(());
    }
    hip_result(unsafe { hipFree(ptr) }, "hipFree")
}

pub fn host_malloc(size: usize) -> Result<HipPtr, HipErr> {
    let mut ptr: HipPtr = std::ptr::null_mut();
    hip_result(
        unsafe { hipHostMalloc(&mut ptr, size, HIP_HOST_MALLOC_DEFAULT) },
        "hipHostMalloc",
    )?;
    Ok(ptr)
}

pub fn host_free(ptr: HipPtr) -> Result<(), HipErr> {
    if ptr.is_null() {
        return Ok(());
    }
    hip_result(unsafe { hipHostFree(ptr) }, "hipHostFree")
}

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

pub fn pack_vals_idx_u64(
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

pub fn kway_merge_shared_heap_real_keepk_u64(
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

pub fn kway_merge_shared_heap_keepk_u64(
    cand_packed: *const u64,
    rows: i32,
    total: i32,
    k_final: i32,
    out_vals: *mut f32,
    out_idx: *mut i32,
    stream: &HipStream,
) -> Result<(), HipErr> {
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

pub fn kway_merge_warp_coop_keepk_u64(
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

pub fn kway_merge_warp_heap_keepk_u64(
    cand_packed: *const u64,
    rows: i32,
    total: i32,
    k_final: i32,
    out_vals: *mut f32,
    out_idx: *mut i32,
    stream: &HipStream,
) -> Result<(), HipErr> {
    kway_merge_warp_coop_keepk_u64(cand_packed, rows, total, k_final, out_vals, out_idx, stream)
}

pub fn kway_merge_bitonic_u64(
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

pub fn kway_merge_bitonic_f32(
    cand_vals: *const f32,
    cand_idx: *const i32,
    rows: i32,
    total: i32,
    k_final: i32,
    out_vals: *mut f32,
    out_idx: *mut i32,
    stream: &HipStream,
) -> Result<(), HipErr> {
    if rows <= 0 || total <= 0 || k_final <= 0 {
        return Ok(());
    }
    hip_result(
        unsafe {
            st_kway_merge_bitonic_f32(
                cand_vals,
                cand_idx,
                rows,
                total,
                k_final,
                out_vals,
                out_idx,
                stream.raw(),
            )
        },
        "st_kway_merge_bitonic_f32",
    )
}

pub fn topk_tile_bitonic_u64(
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

pub fn topk_pass1_f32(
    input: *const f32,
    rows: i32,
    cols: i32,
    stride: i32,
    k: i32,
    out_vals: *mut f32,
    out_idx: *mut i32,
    stream: &HipStream,
) -> Result<(), HipErr> {
    if rows <= 0 || cols <= 0 || k <= 0 {
        return Ok(());
    }
    let stride = if stride > 0 { stride } else { cols };
    hip_result(
        unsafe {
            st_topk_pass1_f32(
                input,
                rows,
                cols,
                stride,
                k,
                out_vals,
                out_idx,
                stream.raw(),
            )
        },
        "st_topk_pass1_f32",
    )
}

pub fn allgather_u64_dev(
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
                recv as *mut c_void,
                count,
                RCCL_UINT64,
                comm,
                stream.raw(),
            )
        },
        "rcclAllGather",
    )
}
