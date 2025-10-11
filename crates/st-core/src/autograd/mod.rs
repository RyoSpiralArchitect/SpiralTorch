
use ndarray::ArrayD;
use crate::Tensor;

pub trait BackwardNode: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn parents(&self) -> Vec<Tensor>;
    fn num_outputs(&self) -> usize { 1 }
    fn backward_multi(&self, _grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> { vec![None] }
}

pub struct GradFn(pub std::rc::Rc<std::cell::RefCell<GradFnInner>>);
pub struct GradFnInner {
    node: Box<dyn BackwardNode>,
}
impl GradFn {
    pub fn new<N: BackwardNode>(node: N) -> Self {
        GradFn(std::rc::Rc::new(std::cell::RefCell::new(GradFnInner{ node: Box::new(node) })))
    }
    pub fn parents(&self)->Vec<Tensor>{ self.0.borrow().node.parents() }
    pub fn num_outputs(&self)->usize{ self.0.borrow().node.num_outputs() }
    pub fn backward_multi(&self, grads_out:&[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
        self.0.borrow().node.backward_multi(grads_out)
    }
    pub fn key(&self)->usize { (&*self.0.as_ptr()) as *const GradFnInner as usize }
}
