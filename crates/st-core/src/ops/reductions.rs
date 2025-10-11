use ndarray::{ArrayD, IxDyn, Axis};
use crate::{Tensor, device::Device, error::Result};

pub fn sum_axes(x: &Tensor, axes: &[usize], keepdim: bool) -> Result<Tensor> {
    let shp = x.shape();
    let ndim = shp.len();
    let mut reduce = vec![false; ndim];
    for &ax in axes { reduce[ax] = true; }
    let mut rows = 1usize; let mut cols = 1usize;
    for (i, &d) in shp.iter().enumerate() {
        if reduce[i] { cols *= d; } else { rows *= d; }
    }

    match x.device() {
        Device::Wgpu => {
            #[cfg(feature="wgpu")]
            {
                use crate::backend::WgpuBackend;
                if x.device_array().is_none() { x.ensure_device().ok(); }
                let be = WgpuBackend::new();
                let d = x.device_array().unwrap();
                let out = be.reduce_rows_sum(&d, rows, cols)?;
                let t = Tensor::from_device_array(out, vec![rows], Device::Wgpu, x.0.borrow().requires_grad);
                return Ok(reshape_back(&t, &shp, &reduce, keepdim));
            }
        }
        _ => {}
    }

    // CPU fallback
    let mut a = x.data();
    // simple path: move reduced axes to the end (approximate)
    let mut order: Vec<usize> = (0..ndim).collect();
    order.sort_by_key(|&i| if reduce[i] { 1 } else { 0 });
    if order != (0..ndim).collect::<Vec<_>>() {
        a = a.permuted_axes(order);
    }
    let mut y = a.into_shape(IxDyn(&[rows, cols])).unwrap().sum_axis(Axis(1));
    if keepdim {
        let mut newshape: Vec<usize> = shp.iter().enumerate().filter(|(i,_)| !reduce[*i]).map(|(_, &d)| d).collect();
        for (i, r) in reduce.iter().enumerate() { if *r { newshape.insert(i, 1); } }
        y = y.into_shape(IxDyn(&newshape)).unwrap();
    } else {
        let newshape: Vec<usize> = shp.iter().enumerate().filter(|(i,_)| !reduce[*i]).map(|(_, &d)| d).collect();
        y = y.into_shape(IxDyn(&newshape)).unwrap();
    }
    Ok(Tensor::from_array(y))
}

pub fn reshape_back(t: &Tensor, orig: &[usize], reduce: &[bool], keepdim: bool) -> Tensor {
    let mut kept: Vec<usize> = orig.iter().enumerate().filter(|(i,_)| !reduce[*i]).map(|(_, &d)| d).collect();
    let mut newshape = Vec::new(); let mut ki=0;
    for &r in reduce {
        if r { if keepdim { newshape.push(1); } } else { newshape.push(kept[ki]); ki+=1; }
    }
    let mut out = t.clone();
    out.0.borrow_mut().shape = if newshape.is_empty() { vec![] } else { newshape.clone() };
    out
}
