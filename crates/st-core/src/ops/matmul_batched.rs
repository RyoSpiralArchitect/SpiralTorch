
use ndarray::{Array3, ArrayD, Ix3, Axis};
use crate::{Tensor, error::Result, autograd::GradFn};

pub fn matmul_batched3d(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a3 = a.data().clone().into_dimensionality::<Ix3>().expect("matmul_batched3d: A 3D");
    let b3 = b.data().clone().into_dimensionality::<Ix3>().expect("matmul_batched3d: B 3D");
    let (ba, m, k) = (a3.shape()[0], a3.shape()[1], a3.shape()[2]);
    let (bb, kb, n) = (b3.shape()[0], b3.shape()[1], b3.shape()[2]);
    assert_eq!(k, kb, "matmul_batched3d: inner mismatch");
    let bout = ba.max(bb);
    let mut out = ndarray::Array3::<f32>::zeros((bout, m, n));
    for i in 0..bout {
        let ai = a3.index_axis(Axis(0), if ba==1 { 0 } else { i });
        let bi = b3.index_axis(Axis(0), if bb==1 { 0 } else { i });
        let yi = ai.dot(&bi);
        out.index_axis_mut(Axis(0), i).assign(&yi);
    }
    let out_t = Tensor::from_array(out.into_dyn());
    if a.0.borrow().requires_grad || b.0.borrow().requires_grad {
        struct Node { a: Tensor, b: Tensor }
        impl crate::autograd::BackwardNode for Node {
            fn name(&self) -> &'static str { "matmul_batched3d" }
            fn parents(&self) -> Vec<Tensor> { vec![self.a.clone(), self.b.clone()] }
            fn num_outputs(&self) -> usize { 1 }
            fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
                let go = grads_out[0].as_ref().unwrap().clone().into_dimensionality::<Ix3>().unwrap();
                let a3 = self.a.data().clone().into_dimensionality::<Ix3>().unwrap();
                let b3 = self.b.data().clone().into_dimensionality::<Ix3>().unwrap();
                let (ba, m, k) = (a3.shape()[0], a3.shape()[1], a3.shape()[2]);
                let (bb, kb, n) = (b3.shape()[0], b3.shape()[1], b3.shape()[2]);
                let bout = ba.max(bb);
                let mut ga = ndarray::Array3::<f32>::zeros((ba, m, k));
                let mut gb = ndarray::Array3::<f32>::zeros((bb, kb, n));
                for i in 0..bout {
                    let ai = a3.index_axis(Axis(0), if ba==1 { 0 } else { i });
                    let bi = b3.index_axis(Axis(0), if bb==1 { 0 } else { i });
                    let goi = go.index_axis(Axis(0), i);
                    let gai = goi.dot(&bi.t());
                    let gbi = ai.t().dot(&goi);
                    if ba==1 {
                        ga.index_axis_mut(Axis(0), 0).assign(&(ga.index_axis(Axis(0), 0).to_owned() + &gai));
                    } else {
                        ga.index_axis_mut(Axis(0), i).assign(&gai);
                    }
                    if bb==1 {
                        gb.index_axis_mut(Axis(0), 0).assign(&(gb.index_axis(Axis(0), 0).to_owned() + &gbi));
                    } else {
                        gb.index_axis_mut(Axis(0), i).assign(&gbi);
                    }
                }
                vec![Some(ga.into_dyn()), Some(gb.into_dyn())]
            }
        }
        out_t.attach_grad_fn(GradFn::new(Node{ a: a.clone(), b: b.clone() }), 0, 1, true);
    }
    Ok(out_t)
}
