use std::{cell::RefCell, rc::Rc};
use ndarray::ArrayD;
use crate::Tensor;

pub trait BackwardNode {
    fn name(&self) -> &'static str;
    fn parents(&self) -> Vec<Tensor>;
    fn num_outputs(&self) -> usize { 1 }
    fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>>;
}

#[derive(Clone)]
pub struct GradFn(pub Rc<RefCell<dyn BackwardNode>>);

impl GradFn {
    pub fn new<N: BackwardNode + 'static>(node: N) -> Self { GradFn(Rc::new(RefCell::new(node))) }
    pub fn key(&self) -> usize { self.0.as_ptr() as *const () as usize }
}
