
use ndarray::{Array2, ArrayD, Ix2};
use crate::{Tensor, error::Result, autograd::{GradFn, BackwardNode, GradBuf}};
use crate::device::Device;

pub fn softmax2d(x: &Tensor) -> Result<Tensor> {
    let x2 = x.data().clone().into_dimensionality::<Ix2>().expect("softmax2d: expects 2D");
    let (rows, cols) = (x2.shape()[0], x2.shape()[1]);
    let mut out = Array2::<f32>::zeros((rows, cols));
    for i in 0..rows {
        let row = x2.row(i);
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0f32;
        for j in 0..cols { s += (row[j] - m).exp(); }
        for j in 0..cols { out[[i,j]] = (row[j]-m).exp() / s; }
    }
    let out_t = Tensor::from_array(out.clone().into_dyn());
    if x.0.borrow().requires_grad {
        struct Node { y: Tensor }
        impl BackwardNode for Node {
            fn name(&self) -> &'static str { "softmax2d" }
            fn parents(&self) -> Vec<Tensor> { vec![self.y.clone()] }
            fn num_outputs(&self) -> usize { 1 }
            fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
                let go = grads_out[0].as_ref().unwrap().clone().into_dimensionality::<Ix2>().unwrap();
                let y  = self.y.data().clone().into_dimensionality::<Ix2>().unwrap();
                let (rows, cols) = (y.shape()[0], y.shape()[1]);
                let mut gx = ndarray::Array2::<f32>::zeros((rows, cols));
                for i in 0..rows {
                    let yi = y.row(i); let gi = go.row(i);
                    let dot = (yi.clone() * gi.clone()).sum();
                    for j in 0..cols { gx[[i,j]] = (gi[j] - dot) * yi[j]; }
                }
                vec![Some(gx.into_dyn())]
            }
            fn supports_device(&self) -> bool { matches!(self.y.device(), Device::Mps) }
            fn backward_multi_dev(&self, grads_out: &[Option<GradBuf>]) -> Option<Vec<Option<GradBuf>>> {
                #[cfg(all(feature="mps", target_os="macos"))] {
                    use crate::backend::{Backend, MpsBackend};
                    let be = MpsBackend::new();
                    if self.y.device_array().is_none() { self.y.ensure_device().ok(); }
                    let y_dev = self.y.device_array()?;
                    let go_dev = match &grads_out[0] {
                        Some(GradBuf::Device{arr, ..}) => arr.clone(),
                        Some(GradBuf::Host(h)) => be.from_host_f32(h).ok()?,
                        None => return Some(vec![None]),
                    };
                    let gx = be.softmax_backward2d(&y_dev, &go_dev).ok()?;
                    return Some(vec![Some(GradBuf::Device{ arr: gx, shape: self.y.shape(), device: Device::Mps })]);
                }
                #[allow(unreachable_code)]
                None
            }
        }
        out_t.attach_grad_fn(GradFn::new(Node{ y: out_t.clone() }), 0, 1, true);
    }
    Ok(out_t)
}
