use crate::{HipErr, DeviceInfo};
use std::ffi::c_void;

type hipError_t = i32;
type hipStream_t = *mut c_void;
type hipDeviceptr_t = *mut c_void;
const HIP_SUCCESS: hipError_t = 0;
const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RcclUniqueId { pub internal: [u8; 128] }
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RcclComm { pub internal: *mut c_void }

extern "C" {
    fn hipGetDeviceCount(count:*mut i32) -> hipError_t;
    fn hipDeviceSynchronize() -> hipError_t;
    fn hipStreamCreate(stream:*mut hipStream_t) -> hipError_t;
    fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    fn hipMalloc(ptr:*mut hipDeviceptr_t, size: usize) -> hipError_t;
    fn hipFree(ptr: hipDeviceptr_t) -> hipError_t;
    fn hipMemcpyAsync(dst: hipDeviceptr_t, src: hipDeviceptr_t, size: usize, kind: i32, stream: hipStream_t) -> hipError_t;

    fn st_topk_pass1_f32(dX:*const f32, rows:i32, cols:i32, stride:i32, k:i32, dVals:*mut f32, dIdx:*mut i32, stream: hipStream_t) -> i32;

    pub fn rcclGetUniqueId(id:*mut RcclUniqueId) -> i32;
    pub fn rcclCommInitRank(comm:*mut RcclComm, nranks:i32, id:RcclUniqueId, rank:i32) -> i32;
    pub fn rcclAllReduce(send:*const c_void, recv:*mut c_void, count:usize, dtype:i32, op:i32, comm:RcclComm, stream:hipStream_t) -> i32;
    pub fn rcclAllGather(send:*const c_void, recv:*mut c_void, count:usize, dtype:i32, comm:RcclComm, stream:hipStream_t) -> i32;
    pub fn rcclCommDestroy(comm: RcclComm) -> i32;
}

pub fn hip_available()->bool { unsafe { let mut c=0; hipGetDeviceCount(&mut c); c>0 } }
pub fn device_info()->Vec<DeviceInfo> {
    if hip_available() { vec![DeviceInfo{ id:0, name:"HIP-Device-0", multi_node:true }] } else { vec![] }
}

pub struct HipStream(pub hipStream_t);
impl HipStream {
    pub fn create()->Result<Self,HipErr>{ unsafe{ let mut s=std::ptr::null_mut(); let e=hipStreamCreate(&mut s); if e==HIP_SUCCESS {Ok(Self(s))} else {Err(HipErr::Other(format!("hipStreamCreate {}",e)))}}}
    pub fn destroy(self){ unsafe{ let _=hipStreamDestroy(self.0); } }
}

pub fn malloc(size:usize)->Result<hipDeviceptr_t,HipErr>{ unsafe{ let mut p=std::ptr::null_mut(); let e=hipMalloc(&mut p,size); if e==HIP_SUCCESS {Ok(p)} else {Err(HipErr::Other(format!("hipMalloc {}",e)))}}}
pub fn free(p:hipDeviceptr_t){ unsafe{ let _=hipFree(p); } }

pub fn memcpy_h2d_async(dst:hipDeviceptr_t, src:*const u8, size:usize, stream:&HipStream)->Result<(),HipErr>{ unsafe{ let e=hipMemcpyAsync(dst, src as hipDeviceptr_t, size, HIP_MEMCPY_HOST_TO_DEVICE, stream.0); if e==HIP_SUCCESS {Ok(())} else {Err(HipErr::Other(format!("hipMemcpyAsync H2D {}",e)))}}}
pub fn memcpy_d2h_async(dst:*mut u8, src:hipDeviceptr_t, size:usize, stream:&HipStream)->Result<(),HipErr>{ unsafe{ let e=hipMemcpyAsync(dst as hipDeviceptr_t, src, size, HIP_MEMCPY_DEVICE_TO_HOST, stream.0); if e==HIP_SUCCESS {Ok(())} else {Err(HipErr::Other(format!("hipMemcpyAsync D2H {}",e)))}}}

pub fn device_synchronize()->Result<(),HipErr>{ unsafe{ let e=hipDeviceSynchronize(); if e==HIP_SUCCESS {Ok(())} else {Err(HipErr::Other(format!("hipDeviceSynchronize {}",e)))}}}

// HIP kernel (Pass1)
pub fn topk_pass1_f32(dX:*const f32, rows:i32, cols:i32, stride:i32, k:i32, dVals:*mut f32, dIdx:*mut i32, stream:&HipStream)->Result<(),HipErr>{
    let rc = unsafe{ st_topk_pass1_f32(dX, rows, cols, stride, k, dVals, dIdx, stream.0) };
    if rc==0 { Ok(()) } else { Err(HipErr::Other(format!("st_topk_pass1_f32 {}",rc))) }
}

// RCCL helpers
pub fn allreduce_i32_sum(comm: RcclComm, stream:&HipStream, buf:&mut [i32]) -> Result<(),HipErr>{
    let rc = unsafe{ rcclAllReduce(buf.as_ptr() as _, buf.as_mut_ptr() as _, buf.len(), 3 /*int32*/, 0/*sum*/, comm, stream.0) };
    if rc==0 { Ok(()) } else { Err(HipErr::Other(format!("rcclAllReduce {}",rc))) }
}
pub fn allgather_f32(comm: RcclComm, stream:&HipStream, send:&[f32], recv:&mut [f32]) -> Result<(),HipErr>{
    let rc = unsafe{ rcclAllGather(send.as_ptr() as _, recv.as_mut_ptr() as _, send.len(), 7 /*float*/, comm, stream.0) };
    if rc==0 { Ok(()) } else { Err(HipErr::Other(format!("rcclAllGather {}",rc))) }
}
