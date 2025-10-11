
use crate::{Tensor};
use crate::autograd::{BackwardNode, GradFn};
use crate::error::Result;
use ndarray::ArrayD;

pub struct LogSumExpF16Node { pub x: Tensor }

impl BackwardNode for LogSumExpF16Node {
    fn name(&self)-> &'static str { "LogSumExpF16Node" }
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
        let gx = y * &go2;
        vec![Some(gx.into_dyn())]
    }
}

pub fn logsumexp_lastdim(_x:&Tensor)->Result<Tensor> {
    // fallback placeholder
    Ok(Tensor::from_array(ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[1]), 0.0)))
}
