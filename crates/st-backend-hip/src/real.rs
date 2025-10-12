use crate::{HipErr, DeviceInfo};
use std::ffi::c_void;

pub type HipPtr = *mut c_void;
type hipError_t = i32;
pub type hipStream_t = *mut c_void;

const HIP_SUCCESS: hipError_t = 0;
const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;

extern "C" {
    fn hipGetDeviceCount(count: *mut i32) -> hipError_t;
    fn hipDeviceSynchronize() -> hipError_t;
    fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    fn hipMalloc(ptr: *mut HipPtr, size: usize) -> hipError_t;
    fn hipFree(ptr: HipPtr) -> hipError_t;
    fn hipMemcpyAsync(dst: HipPtr, src: HipPtr, size: usize, kind: i32, stream: hipStream_t) -> hipError_t;

    fn st_topk_pass1_f32(dX:*const f32, rows:i32, cols:i32, stride:i32, k:i32, dVals:*mut f32, dIdx:*mut i32, stream: hipStream_t) -> i32;
    fn st_kway_merge_bitonic_f32(cand_vals:*const f32, cand_idx:*const i32, rows:i32, total:i32, k_final:i32, out_vals:*mut f32, out_idx:*mut i32, stream: hipStream_t) -> i32;
    fn st_pack_vals_idx_u64(vals:*const f32, idx:*const i32, out:*mut u64, total:i32, stream: hipStream_t) -> i32;

    // RCCL
    pub fn rcclAllReduce(send:*const c_void, recv:*mut c_void, count: usize, dtype:i32, op:i32, comm: super::rccl_comm::RcclComm, stream: hipStream_t) -> i32;
    pub fn rcclAllGather(send:*const c_void, recv:*mut c_void, count: usize, dtype:i32, comm: super::rccl_comm::RcclComm, stream: hipStream_t) -> i32;
}

pub fn hip_available()->bool { let mut c=0i32; unsafe{ hipGetDeviceCount(&mut c as *mut i32); } c>0 }
pub fn device_info()->Vec<DeviceInfo> { if !hip_available(){ return vec![]; } vec![DeviceInfo{ id:0, name:"HIP-Device-0", multi_node:true }] }

pub struct HipStream(pub hipStream_t);
impl HipStream {
    pub fn create()->Result<Self,HipErr>{ unsafe{ let mut s: hipStream_t = std::ptr::null_mut(); let e = hipStreamCreate(&mut s as *mut _);
        if e==HIP_SUCCESS { Ok(Self(s)) } else { Err(HipErr::Other(format!("hipStreamCreate {}", e))) } } }
    pub fn destroy(self){ unsafe{ let _ = hipStreamDestroy(self.0); } }
}
pub fn malloc(size:usize)->Result<HipPtr,HipErr>{ unsafe{ let mut p: HipPtr = std::ptr::null_mut(); let e = hipMalloc(&mut p as *mut _, size);
    if e==HIP_SUCCESS { Ok(p) } else { Err(HipErr::Other(format!("hipMalloc {}", e))) } } }
pub fn free(p:HipPtr){ unsafe{ let _ = hipFree(p); } }

pub fn memcpy_h2d_async(dst:HipPtr, src:*const u8, size:usize, stream:&HipStream)->Result<(),HipErr>{ unsafe {
    let e = hipMemcpyAsync(dst, src as HipPtr, size, HIP_MEMCPY_HOST_TO_DEVICE, stream.0);
    if e==HIP_SUCCESS { Ok(()) } else { Err(HipErr::Other(format!("hipMemcpyAsync H2D {}", e))) }
}}
pub fn memcpy_d2h_async(dst:*mut u8, src:HipPtr, size:usize, stream:&HipStream)->Result<(),HipErr>{ unsafe {
    let e = hipMemcpyAsync(dst as HipPtr, src, size, HIP_MEMCPY_DEVICE_TO_HOST, stream.0);
    if e==HIP_SUCCESS { Ok(()) } else { Err(HipErr::Other(format!("hipMemcpyAsync D2H {}", e))) }
}}
pub fn device_synchronize()->Result<(),HipErr>{ unsafe{ let e = hipDeviceSynchronize(); if e==HIP_SUCCESS { Ok(()) } else { Err(HipErr::Other(format!("hipDeviceSynchronize {}", e))) } } }

// RCCL helpers
const DTYPE_I32: i32 = 3; // rcclInt32
const DTYPE_F32: i32 = 7; // rcclFloat32
const DTYPE_U64: i32 = 8; // rcclUint64 (assumed value; may differ per header)

const OP_SUM: i32 = 0;    // rcclSum

pub fn allgather_f32_dev(comm: super::rccl_comm::RcclComm, stream:&HipStream, send_dev: HipPtr, recv_dev: HipPtr, count_per_rank: usize) -> Result<(),HipErr> {
    let rc = unsafe{ rcclAllGather(send_dev as *const _, recv_dev, count_per_rank, DTYPE_F32, comm, stream.0) };
    if rc==0 { Ok(()) } else { Err(HipErr::Other(format!("rcclAllGather f32 {}", rc))) }
}
pub fn allgather_i32_dev(comm: super::rccl_comm::RcclComm, stream:&HipStream, send_dev: HipPtr, recv_dev: HipPtr, count_per_rank: usize) -> Result<(),HipErr> {
    let rc = unsafe{ rcclAllGather(send_dev as *const _, recv_dev, count_per_rank, DTYPE_I32, comm, stream.0) };
    if rc==0 { Ok(()) } else { Err(HipErr::Other(format!("rcclAllGather i32 {}", rc))) }
}
pub fn allgather_u64_dev(comm: super::rccl_comm::RcclComm, stream:&HipStream, send_dev: HipPtr, recv_dev: HipPtr, count_per_rank: usize) -> Result<(),HipErr> {
    let rc = unsafe{ rcclAllGather(send_dev as *const _, recv_dev, count_per_rank, DTYPE_U64, comm, stream.0) };
    if rc==0 { Ok(()) } else { Err(HipErr::Other(format!("rcclAllGather u64 {}", rc))) }
}

// Kernels
pub fn topk_pass1_f32(dX:*const f32, rows:i32, cols:i32, stride:i32, k:i32, dVals:*mut f32, dIdx:*mut i32, stream:&HipStream)->Result<(),HipErr>{
    let rc = unsafe{ st_topk_pass1_f32(dX, rows, cols, stride, k, dVals, dIdx, stream.0) };
    if rc==0 { Ok(()) } else { Err(HipErr::Other(format!("st_topk_pass1_f32 {}", rc))) }
}
pub fn kway_merge_bitonic_f32(cand_vals:*const f32, cand_idx:*const i32, rows:i32, total:i32, k_final:i32, out_vals:*mut f32, out_idx:*mut i32, stream:&HipStream)->Result<(),HipErr>{
    let rc = unsafe{ st_kway_merge_bitonic_f32(cand_vals, cand_idx, rows, total, k_final, out_vals, out_idx, stream.0) };
    if rc==0 { Ok(()) } else { Err(HipErr::Other(format!("st_kway_merge_bitonic_f32 {}", rc))) }
}
pub fn pack_vals_idx_u64(vals:*const f32, idx:*const i32, out:*mut u64, total:i32, stream:&HipStream)->Result<(),HipErr>{
    let rc = unsafe{ st_pack_vals_idx_u64(vals, idx, out, total, stream.0) };
    if rc==0 { Ok(()) } else { Err(HipErr::Other(format!("st_pack_vals_idx_u64 {}", rc))) }
}
