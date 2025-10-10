
use ndarray::ArrayD;
use crate::{Tensor, error::Result};

pub fn sum_all(x: &Tensor) -> Result<Tensor> {
    Ok(Tensor::from_array(ArrayD::from_elem([], x.data().sum())))
}
