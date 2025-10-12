//! Backend-agnostic MidK compaction API.
//! Switches 1CE / 2CE depending on problem size and device capability.
use crate::tensor::Tensor;
use crate::device::Device;

pub struct MidKOut { pub vals: Tensor, pub idx: Tensor }

pub fn midk_compact(x:&Tensor, lower:f32, upper:f32) -> MidKOut {
    match x.device() {
        Device::Wgpu => midk_wgpu(x, lower, upper),
        Device::Hip  => midk_hip(x, lower, upper),
        Device::Cuda => midk_cuda(x, lower, upper),
        _ => midk_cpu(x, lower, upper),
    }
}

fn midk_wgpu(x:&Tensor, lower:f32, upper:f32) -> MidKOut {
    // Heuristic: if cols <= 256 -> 1CE; else 2CE (scan+apply).
    // This is a thin orchestrator stub; concrete driver calls live in backend crate.
    unimplemented!("wire to st-backend-wgpu compaction (1CE/2CE)")
}
fn midk_hip(x:&Tensor, lower:f32, upper:f32) -> MidKOut {
    unimplemented!("wire to HIP kernels hip_compaction_scan/apply or 1CE")
}
fn midk_cuda(x:&Tensor, lower:f32, upper:f32) -> MidKOut {
    unimplemented!("similar to HIP")
}
fn midk_cpu(_x:&Tensor, _lower:f32, _upper:f32) -> MidKOut {
    unimplemented!("CPU reference (optional)")
}
