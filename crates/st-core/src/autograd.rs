use std::{cell::RefCell, rc::Rc};
use ndarray::ArrayD;
use crate::{device::Device};

#[derive(Clone)]
pub struct GradFn(pub Rc<RefCell<dyn BackwardNode>>>);

impl GradFn {
    pub fn new<N: BackwardNode + 'static>(node: N) -> Self { GradFn(Rc::new(RefCell::new(node))) }
    pub fn key(&self) -> usize { &*self.0 as *const _ as usize }
    pub fn parents(&self) -> Vec<crate::tensor::Tensor> { self.0.borrow().parents() }
    pub fn num_outputs(&self) -> usize { self.0.borrow().num_outputs() }
    pub fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
        self.0.borrow().backward_multi(grads_out)
    }
    pub fn supports_device(&self) -> bool { self.0.borrow().supports_device() }
}

pub enum GradBuf {
    Host(ArrayD<f32>),
    Device { arr: crate::backend::BackendArrayF32, shape: Vec<usize>, device: Device },
}

pub trait BackwardNode {
    fn name(&self) -> &'static str;
    fn parents(&self) -> Vec<crate::tensor::Tensor>;
    fn num_outputs(&self) -> usize;
    fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>>;

    fn supports_device(&self) -> bool { false }
    fn backward_multi_dev(&self, _grads_out: &[Option<GradBuf>]) -> Option<Vec<Option<GradBuf>>> { None }
}
