use crate::real::{HipStream};
use crate::HipErr;
use std::ffi::c_void;

extern "C" {
    fn hip_compaction_1ce_kernel(vin:*const f32, iin:*const i32, rows:i32, cols:i32, low:f32, high:f32,
                                 vout:*mut f32, iout:*mut i32) -> i32;
}

pub fn compaction_1ce(_stream:&HipStream,
                      _vin:*const f32, _iin:*const i32, _rows:i32, _cols:i32, _low:f32, _high:f32,
                      _vout:*mut f32, _iout:*mut i32) -> Result<(), HipErr> {
    // NOTE: wire the launch via hipLaunchKernelGGL in build.rs/externs similarly to other kernels.
    Ok(())
}
