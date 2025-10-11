use crate::{Tensor, error::Result};

pub fn softmax_lastdim(x:&Tensor) -> Result<Tensor> {
    let s = x.shape();
    let rows: usize = s[..s.len()-1].iter().product();
    let cols: usize = s[s.len()-1];
    let a = x.data().into_shape((rows, cols)).unwrap();
    let max = a.map_axis(ndarray::Axis(1), |r| r.fold(f32::NEG_INFINITY, |m,&v| m.max(v)));
    let a_shift = a - &max.insert_axis(ndarray::Axis(1));
    let e = a_shift.mapv(|v| v.exp());
    let sum = e.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
    let y = e / &sum;
    Ok(Tensor::from_array(y.into_dyn().into_shape(s).unwrap()))
}
