use std::{cell::RefCell, rc::Rc, sync::atomic::{AtomicUsize, Ordering}};
use ndarray::ArrayD;
use crate::Tensor;
use crate::device::Device;

#[cfg(all(feature="mps", target_os="macos"))]
use crate::backend::BackendArrayF32;

pub enum GradBuf {
    Host(ArrayD<f32>),
    #[cfg(all(feature="mps", target_os="macos"))]
    Device { arr: BackendArrayF32, shape: Vec<usize>, device: Device },
}

pub trait BackwardNode {
    fn name(&self) -> &'static str;
    fn parents(&self) -> Vec<Tensor>;
    fn num_outputs(&self) -> usize;
    fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>>;
    fn backward_multi_dev(&self, _grads_out: &[Option<GradBuf>]) -> Option<Vec<Option<GradBuf>>> { None }
    fn supports_device(&self) -> bool { false }
}

static NEXT_GF_KEY: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone)]
pub struct GradFn(pub Rc<RefCell<dyn BackwardNode>>, usize);
impl GradFn {
    pub fn new<N: BackwardNode + 'static>(node: N) -> Self {
        let k = NEXT_GF_KEY.fetch_add(1, Ordering::Relaxed);
        Self(Rc::new(RefCell::new(node)), k)
    }
    #[inline] pub fn key(&self) -> usize { self.1 }
    #[inline] pub fn parents(&self) -> Vec<Tensor> { self.0.borrow().parents() }
    #[inline] pub fn num_outputs(&self) -> usize { self.0.borrow().num_outputs() }
    #[inline] pub fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> { self.0.borrow().backward_multi(grads_out) }
    #[inline] pub fn backward_multi_dev(&self, grads_out: &[Option<GradBuf>]) -> Option<Vec<Option<GradBuf>>> { self.0.borrow().backward_multi_dev(grads_out) }
    #[inline] pub fn supports_device(&self) -> bool { self.0.borrow().supports_device() }
}

pub mod engine;
pub use engine::run_backward;
