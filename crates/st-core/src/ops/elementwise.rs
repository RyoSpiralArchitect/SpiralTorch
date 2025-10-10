
use ndarray::{ArrayD, IxDyn};
use crate::{Tensor, error::Result, autograd::{GradFn, BackwardNode, GradBuf}};
use crate::device::Device;

fn broadcast_shape(sa: &[usize], sb: &[usize]) -> Vec<usize> {
    let nd = sa.len().max(sb.len());
    let mut out = vec![1usize; nd];
    for i in 0..nd {
        let a = *sa.get(sa.len().wrapping_sub(1).wrapping_sub(i)).unwrap_or(&1);
        let b = *sb.get(sb.len().wrapping_sub(1).wrapping_sub(i)).unwrap_or(&1);
        if a == b || a == 1 || b == 1 { out[nd-1-i] = a.max(b); } else { panic!("broadcast: incompatible shapes"); }
    }
    out
}
fn unbroadcast(mut g: ArrayD<f32>, target: &[usize]) -> ArrayD<f32> {
    let gshape = g.shape().to_vec();
    let mut t = target.to_vec();
    if gshape.len() > t.len() { let mut pad = vec![1usize; gshape.len() - t.len()]; pad.extend_from_slice(&t); t = pad; }
    for ax in (0..gshape.len()).rev() {
        if t[ax] == 1 && gshape[ax] != 1 { g = g.sum_axis(ndarray::Axis(ax)); }
    }
    g.into_dimensionality::<IxDyn>().unwrap()
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let sa = a.shape(); let sb = b.shape();
    let out_shape = broadcast_shape(&sa, &sb);
    let aa = a.data().broadcast(IxDyn(&out_shape)).expect("broadcast a").to_owned();
    let bb = b.data().broadcast(IxDyn(&out_shape)).expect("broadcast b").to_owned();
    let y = &aa + &bb;
    let out = Tensor::from_array(y.into_dyn());
    if a.0.borrow().requires_grad || b.0.borrow().requires_grad {
        struct Node { a: Tensor, b: Tensor }
        impl BackwardNode for Node {
            fn name(&self) -> &'static str { "add" }
            fn parents(&self) -> Vec<Tensor> { vec![self.a.clone(), self.b.clone()] }
            fn num_outputs(&self) -> usize { 1 }
            fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
                let go = grads_out[0].as_ref().unwrap().clone();
                let ga = unbroadcast(go.clone(), &self.a.shape());
                let gb = unbroadcast(go, &self.b.shape());
                vec![Some(ga), Some(gb)]
            }
        }
        out.attach_grad_fn(GradFn::new(Node{ a: a.clone(), b: b.clone() }), 0, 1, true);
    }
    Ok(out)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let sa = a.shape(); let sb = b.shape();
    let out_shape = broadcast_shape(&sa, &sb);
    let aa = a.data().broadcast(IxDyn(&out_shape)).expect("broadcast a").to_owned();
    let bb = b.data().broadcast(IxDyn(&out_shape)).expect("broadcast b").to_owned();
    let y = &aa * &bb;
    let out = Tensor::from_array(y.into_dyn());
    if a.0.borrow().requires_grad || b.0.borrow().requires_grad {
        struct Node { a: Tensor, b: Tensor }
        impl BackwardNode for Node {
            fn name(&self) -> &'static str { "mul" }
            fn parents(&self) -> Vec<Tensor> { vec![self.a.clone(), self.b.clone()] }
            fn num_outputs(&self) -> usize { 1 }
            fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
                let go = grads_out[0].as_ref().unwrap().clone();
                let a = self.a.data(); let b = self.b.data();
                let ga = unbroadcast(&go * &b.broadcast(a.raw_dim()).unwrap().to_owned(), &self.a.shape());
                let gb = unbroadcast(&go * &a.broadcast(b.raw_dim()).unwrap().to_owned(), &self.b.shape());
                vec![Some(ga), Some(gb)]
            }
        }
        out.attach_grad_fn(GradFn::new(Node{ a: a.clone(), b: b.clone() }), 0, 1, true);
    }
    Ok(out)
}

pub fn relu(x: &Tensor) -> Result<Tensor> {
    let y = x.data().mapv(|v| if v>0.0 { v } else { 0.0 });
    let out = Tensor::from_array(y.into_dyn());
    if x.0.borrow().requires_grad {
        struct Node { x: Tensor }
        impl BackwardNode for Node {
            fn name(&self) -> &'static str { "relu" }
            fn parents(&self) -> Vec<Tensor> { vec![self.x.clone()] }
            fn num_outputs(&self) -> usize { 1 }
            fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
                let go = grads_out[0].as_ref().unwrap();
                let x = self.x.data();
                let mask = x.mapv(|v| if v>0.0 { 1.0 } else { 0.0 });
                let gx = go * &mask;
                vec![Some(gx)]
            }
            fn supports_device(&self) -> bool { matches!(self.x.device(), Device::Mps) }
            fn backward_multi_dev(&self, grads_out: &[Option<GradBuf>]) -> Option<Vec<Option<GradBuf>>> {
                #[cfg(all(feature="mps", target_os="macos"))] {
                    use crate::backend::{Backend, MpsBackend};
                    use crate::backend::mps_impl::NdInfoA;
                    let be = MpsBackend::new();
                    if self.x.device_array().is_none() { self.x.ensure_device().ok(); }
                    let dx = self.x.device_array()?;
                    let go_dev = match &grads_out[0] {
                        Some(GradBuf::Device{arr, ..}) => arr.clone(),
                        Some(GradBuf::Host(h)) => be.from_host_f32(h).ok()?,
                        None => return Some(vec![None]),
                    };
                    let nd = self.x.shape().len().min(6);
                    let mut info = NdInfoA { ndim: nd as u32, n: self.x.shape().iter().product::<usize>() as u32,
                                              shape:[1;6], stride_x:[0;6], stride_go:[0;6] };
                    let shp = self.x.shape();
                    for i in 0..nd { info.shape[6-nd+i] = shp[i] as u32; }
                    let mut s = 1i32;
                    for i in (0..nd).rev() { info.stride_x[6-nd+i]=s; info.stride_go[6-nd+i]=s; s *= shp[i] as i32; }
                    let dy = be.relu_backward_strided(&dx, &go_dev, &info).ok()?;
                    return Some(vec![Some(GradBuf::Device{ arr: dy, shape: self.x.shape(), device: Device::Mps })]);
                }
                #[allow(unreachable_code)]
                None
            }
        }
        out.attach_grad_fn(GradFn::new(Node{ x: x.clone() }), 0, 1, true);
    }
    Ok(out)
}
