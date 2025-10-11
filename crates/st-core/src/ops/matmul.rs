use ndarray::Array2;
use crate::{Tensor, device::Device, error::Result};

pub fn matmul2d(a:&Tensor, b:&Tensor) -> Result<Tensor> {
    let (m,k) = { let s=a.shape(); (s[s.len()-2], s[s.len()-1]) };
    let (k2,n) = { let s=b.shape(); (s[s.len()-2], s[s.len()-1]) };
    if k!=k2 { return Err(crate::error::msg("matmul2d: inner dim mismatch")); }

    match a.device() {
        Device::Wgpu => {
            #[cfg(feature="wgpu")]
            {
                let be = crate::backend::WgpuBackend::new();
                if a.device_array().is_none() { a.ensure_device().ok(); }
                if b.device_array().is_none() { b.ensure_device().ok(); }
                let c = be.matmul2d_tiled(&a.device_array().unwrap(), &b.device_array().unwrap(), m, k, n)?;
                return Ok(Tensor::from_device_array(c, vec![m,n], Device::Wgpu, a.0.borrow().requires_grad||b.0.borrow().requires_grad));
            }
        }
        Device::Cuda => {
            #[cfg(feature="cuda")]
            {
                let be = crate::backend::CudaBackend::new();
                if a.device_array().is_none() { a.ensure_device().ok(); }
                if b.device_array().is_none() { b.ensure_device().ok(); }
                let c = be.gemm2d_tiled(&a.device_array().unwrap(), &b.device_array().unwrap(), m, k, n)?;
                return Ok(Tensor::from_device_array(c, vec![m,n], Device::Cuda, a.0.borrow().requires_grad||b.0.borrow().requires_grad));
            }
        }
        _ => {}
    }

    // CPU fallback
    let a2 = a.data().into_dimensionality::<ndarray::Ix2>().unwrap();
    let b2 = b.data().into_dimensionality::<ndarray::Ix2>().unwrap();
    let mut c = Array2::<f32>::zeros((m,n));
    ndarray::linalg::general_mat_mul(1.0, &a2, &b2, 0.0, &mut c);
    Ok(Tensor::from_array(c.into_dyn()))
}
