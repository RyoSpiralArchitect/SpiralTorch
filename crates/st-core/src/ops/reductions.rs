
use ndarray::{ArrayD, Axis, IxDyn};
use crate::{Tensor, error::Result};
use crate::device::Device;

pub fn sum_all(x: &Tensor) -> Result<Tensor> {
    let axes: Vec<usize> = (0..x.shape().len()).collect();
    sum_axes(x, &axes, false)
}

pub fn sum_axes(x: &Tensor, axes: &[usize], keepdim: bool) -> Result<Tensor> {
    if axes.is_empty() { return Ok(x.clone()); }
    let nd = x.shape().len();
    let mut red: Vec<usize> = axes.iter().copied().collect();
    red.sort_unstable();
    red.dedup();
    for &ax in &red { assert!(ax < nd, "sum_axes: axis out of range"); }

    #[cfg(not(all(feature="mps", target_os="macos")))]
    {
        let mut a = x.data();
        for &ax in red.iter().rev() { a = a.sum_axis(Axis(ax)); }
        let out = if keepdim {
            let mut shape = x.shape();
            for &ax in &red { shape[ax] = 1; }
            a.into_shape(IxDyn(&shape)).unwrap()
        } else { a.into_dyn() };
        return Ok(Tensor::from_array(out));
    }

    #[cfg(all(feature="mps", target_os="macos"))]
    {
        use crate::backend::{Backend, MpsBackend};
        use crate::backend::mps_impl::NdWGInfo;
        if !matches!(x.device(), Device::Mps) {
            let mut a = x.data();
            for &ax in red.iter().rev() { a = a.sum_axis(Axis(ax)); }
            let out = if keepdim {
                let mut shape = x.shape();
                for &ax in &red { shape[ax] = 1; }
                a.into_shape(IxDyn(&shape)).unwrap()
            } else { a.into_dyn() };
            return Ok(Tensor::from_array(out));
        }
        let be = MpsBackend::new();
        if x.device_array().is_none() { x.ensure_device().ok(); }
        let dx = x.device_array().expect("device buffer");

        let shape = x.shape();
        let mut strides = vec![1i32; nd];
        for i in (0..nd-1).rev() { strides[i] = strides[i+1] * shape[i+1] as i32; }
        let kept: Vec<usize> = (0..nd).filter(|i| !red.contains(i)).collect();

        let n_rows: usize = kept.iter().map(|&i| shape[i]).product();
        let n_cols: usize = red.iter().map(|&i| shape[i]).product();

        let mut info = NdWGInfo {
            n_rows: n_rows as u32, n_cols: n_cols as u32,
            kdims: kept.len() as u32, rdims: red.len() as u32,
            kshape: [1;6], rshape: [1;6], kstride: [0;6], rstride: [0;6],
        };
        for (j,&i) in kept.iter().rev().take(6).enumerate() {
            let idx = 5 - j; info.kshape[idx] = shape[i] as u32; info.kstride[idx] = strides[i];
        }
        for (j,&i) in red.iter().rev().take(6).enumerate() {
            let idx = 5 - j; info.rshape[idx] = shape[i] as u32; info.rstride[idx] = strides[i];
        }

        let out_dev = be.reduce_nd_sum_auto(&dx, &info).expect("reduce auto");
        let mut out_shape = if kept.is_empty() { vec![] } else { kept.iter().map(|&i| shape[i]).collect::<Vec<_>>() };
        if keepdim {
            let mut full = vec![0usize; nd];
            for (pos,&i) in kept.iter().enumerate() { full[i] = out_shape[pos]; }
            for &i in &red { full[i] = 1; }
            out_shape = full;
        }
        let out = Tensor::from_device_array(out_dev, out_shape, Device::Mps, x.0.borrow().requires_grad);
        return Ok(out);
    }
}
