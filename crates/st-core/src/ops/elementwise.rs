use crate::{Tensor, device::Device, error::Result};

pub fn add(a:&Tensor, b:&Tensor) -> Result<Tensor> {
    match a.device() {
        Device::Wgpu => {
            #[cfg(feature="wgpu")]
            {
                let be = crate::backend::WgpuBackend::new();
                if a.device_array().is_none() { a.ensure_device().ok(); }
                if b.device_array().is_none() { b.ensure_device().ok(); }
                let out = be.add(&a.device_array().unwrap(), &b.device_array().unwrap())?;
                return Ok(Tensor::from_device_array(out, a.shape(), Device::Wgpu, a.0.borrow().requires_grad || b.0.borrow().requires_grad));
            }
        }
        Device::Cuda => { /* TODO: device add via PTX (available) */ }
        _ => {}
    }
    // CPU fallback
    let c = a.data() + &b.data();
    Ok(Tensor::from_array(c))
}
