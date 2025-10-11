use ndarray::{ArrayD, IxDyn};
use std::collections::{HashMap, HashSet};
use crate::{tensor::Tensor, dtype::DType};

/// Global backward engine (thin wrapper; Tensor::backward uses similar logic).
pub fn backward(t: &Tensor, grad: Option<ArrayD<f32>>) -> crate::error::Result<()> {
    let seed = match grad {
        Some(g) => g,
        None => if t.ndim() == 0 {
            ArrayD::<f32>::from_elem(IxDyn(&[]), 1.0)
        } else {
            ArrayD::<f32>::from_elem(IxDyn(&t.shape()), 1.0)
        },
    };
    t.backward_with_grad(&seed)
}
