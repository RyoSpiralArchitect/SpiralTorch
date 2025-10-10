
use ndarray::{Array2, ArrayD, Ix2};
use crate::{Tensor, error::Result, autograd::GradFn};

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
        impl crate::autograd::BackwardNode for Node {
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
        }
        out_t.attach_grad_fn(GradFn::new(Node{ y: out_t.clone() }), 0, 1, true);
    }
    Ok(out_t)
}
