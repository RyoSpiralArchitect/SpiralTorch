
use crate::{Tensor};
use crate::autograd::{BackwardNode, GradFn};
use crate::error::Result;
use ndarray::ArrayD;

pub struct SoftmaxF16Node { pub x: Tensor }
impl BackwardNode for SoftmaxF16Node {
    fn name(&self)-> &'static str { "SoftmaxF16Node" }
    fn parents(&self)->Vec<Tensor>{ vec![self.x.clone()] }
    fn backward_multi(&self, grads_out:&[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
        let go = grads_out[0].as_ref().expect("grad required");
        let x = self.x.data();
        let m = x.map_axis(ndarray::Axis(x.ndim()-1), |row| row.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        let e = x - &m.insert_axis(ndarray::Axis(m.ndim()));
        let eexp = e.map(|v| v.exp());
        let s = eexp.sum_axis(ndarray::Axis(eexp.ndim()-1)).insert_axis(ndarray::Axis( eexp.ndim()-1));
        let y = eexp / &s;
        let go2 = go.clone().into_shape(y.raw_dim()).unwrap();
        let dot = (&y * &go2).sum_axis(ndarray::Axis(y.ndim()-1)).insert_axis(ndarray::Axis(y.ndim()-1));
        let gx = &y * (&go2 - &dot);
        vec![Some(gx.into_dyn())]
    }
}
pub fn softmax_lastdim_with_grad(x:&Tensor)->Result<Tensor> {
    let y = x.clone(); Ok(y)
}
