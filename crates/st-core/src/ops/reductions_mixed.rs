use crate::{Tensor, device::Device, error::Result};

pub fn sum_axes_mixed(x: &Tensor, axes: &[usize], keepdim: bool, _accum_fp32: bool) -> Result<Tensor> {
    if matches!(x.device(), Device::Wgpu) {
        // For now, same as sum_axes. The policy hook remains for future f16/bf16.
        return super::reductions::sum_axes(x, axes, keepdim);
    }
    super::reductions::sum_axes(x, axes, keepdim)
}
