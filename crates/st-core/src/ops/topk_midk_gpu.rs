use ndarray::{Array2, ArrayView2};
use super::topk_family::{topk2d_cpu, bottomk2d_cpu, midk2d_cpu};

#[derive(Clone, Copy)]
pub enum Device { Auto, Cpu, Wgpu, Cuda, Mps }

pub fn topk2d(x: ArrayView2<'_, f32>, k: usize, device: Device) -> (Array2<f32>, Array2<i32>) {
    match device {
        Device::Cpu | Device::Auto => topk2d_cpu(x, k),
        _ => topk2d_cpu(x, k),
    }
}
pub fn bottomk2d(x: ArrayView2<'_, f32>, k: usize, device: Device) -> (Array2<f32>, Array2<i32>) {
    match device {
        Device::Cpu | Device::Auto => bottomk2d_cpu(x, k),
        _ => bottomk2d_cpu(x, k),
    }
}
pub fn midk2d(x: ArrayView2<'_, f32>, k: usize, device: Device) -> (Array2<f32>, Array2<i32>) {
    match device {
        Device::Cpu | Device::Auto => midk2d_cpu(x, k),
        _ => midk2d_cpu(x, k),
    }
}
