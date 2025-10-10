
use ndarray::{ArrayD, Ix2};
use crate::{Tensor, error::Result, autograd::GradFn};

pub fn matmul2d(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a2 = a.data().clone().into_dimensionality::<Ix2>().expect("matmul2d: A 2D");
    let b2 = b.data().clone().into_dimensionality::<Ix2>().expect("matmul2d: B 2D");
    let (m, k) = (a2.shape()[0], a2.shape()[1]);
    let (kb, n) = (b2.shape()[0], b2.shape()[1]);
    assert_eq!(k, kb, "matmul2d: inner mismatch");
    let y = a2.dot(&b2);
    let out = Tensor::from_array(y.into_dyn());
    if a.0.borrow().requires_grad || b.0.borrow().requires_grad {
        struct Node { a: Tensor, b: Tensor }
        impl crate::autograd::BackwardNode for Node {
            fn name(&self) -> &'static str { "matmul2d" }
            fn parents(&self) -> Vec<Tensor> { vec![self.a.clone(), self.b.clone()] }
            fn num_outputs(&self) -> usize { 1 }
            fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
                let go = grads_out[0].as_ref().unwrap().clone().into_dimensionality::<Ix2>().unwrap();
                let a2 = self.a.data().clone().into_dimensionality::<Ix2>().unwrap();
                let b2 = self.b.data().clone().into_dimensionality::<Ix2>().unwrap();
                let ga = go.dot(&b2.t());
                let gb = a2.t().dot(&go);
                vec![Some(ga.into_dyn()), Some(gb.into_dyn())]
            }
        }
        out.attach_grad_fn(GradFn::new(Node{ a: a.clone(), b: b.clone() }), 0, 1, true);
    }
    Ok(out)
}
