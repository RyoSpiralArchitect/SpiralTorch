
use std::{cell::RefCell, rc::Rc, sync::atomic::{AtomicUsize, Ordering}};
use ndarray::{ArrayD, IxDyn, Axis};
use crate::{device::Device, dtype::DType, error::Result, autograd::GradFn, autograd::run_backward};

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<Inner>>);

pub enum Storage {
    F32(ArrayD<f32>),
    I32(ArrayD<i32>),
    Bool(ArrayD<bool>),
}

#[cfg(all(feature="mps", target_os="macos"))]
use crate::backend::{Backend, BackendArrayF32, MpsBackend};

#[derive(Default)]
struct DevState {
    #[cfg(all(feature="mps", target_os="macos"))]
    arr: Option<BackendArrayF32>,
    host_dirty: bool,
    dev_dirty: bool,
}

pub struct Inner {
    id: usize,
    pub storage: Storage,
    pub dtype: DType,
    pub device: Device,
    pub requires_grad: bool,
    pub grad: Option<ArrayD<f32>>,
    #[cfg(all(feature="mps", target_os="macos"))]
    pub grad_dev: Option<BackendArrayF32>,

    pub grad_fn: Option<GradFn>,
    pub out_index: usize,
    pub num_outputs: usize,
    dev: DevState,
}

impl Tensor {
    pub fn from_array(data: ArrayD<f32>) -> Self {
        let inner = Inner {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            storage: Storage::F32(data),
            dtype: DType::F32,
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
            #[cfg(all(feature="mps", target_os="macos"))]
            grad_dev: None,
            grad_fn: None,
            out_index: 0,
            num_outputs: 1,
            dev: DevState::default(),
        };
        Tensor(Rc::new(RefCell::new(inner)))
    }
    pub fn from_i32(data: ArrayD<i32>) -> Self {
        let inner = Inner {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            storage: Storage::I32(data),
            dtype: DType::I32,
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
            #[cfg(all(feature="mps", target_os="macos"))]
            grad_dev: None,
            grad_fn: None,
            out_index: 0,
            num_outputs: 1,
            dev: DevState::default(),
        };
        Tensor(Rc::new(RefCell::new(inner)))
    }
    pub fn from_bool(data: ArrayD<bool>) -> Self {
        let inner = Inner {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            storage: Storage::Bool(data),
            dtype: DType::Bool,
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
            #[cfg(all(feature="mps", target_os="macos"))]
            grad_dev: None,
            grad_fn: None,
            out_index: 0,
            num_outputs: 1,
            dev: DevState::default(),
        };
        Tensor(Rc::new(RefCell::new(inner)))
    }
    pub fn zeros(shape: &[usize]) -> Self { Self::from_array(ArrayD::<f32>::zeros(IxDyn(shape))) }
    pub fn ones(shape: &[usize]) -> Self { Self::from_array(ArrayD::<f32>::from_elem(IxDyn(shape), 1.0)) }

    pub fn id(&self) -> usize { self.0.borrow().id }
    pub fn dtype(&self) -> DType { self.0.borrow().dtype }
    pub fn device(&self) -> Device { self.0.borrow().device }
    pub fn shape(&self) -> Vec<usize> {
        match &self.0.borrow().storage {
            Storage::F32(a) => a.shape().to_vec(),
            Storage::I32(a) => a.shape().to_vec(),
            Storage::Bool(a) => a.shape().to_vec(),
        }
    }
    pub fn ndim(&self) -> usize { self.shape().len() }
    pub fn requires_grad(mut self, flag: bool) -> Self { if matches!(self.0.borrow().dtype, DType::F32) { self.0.borrow_mut().requires_grad = flag; } self }

    // ==== Phase-2 host/device sync ====
    pub fn ensure_host(&self) -> Result<()> {
        #[cfg(all(feature="mps", target_os="macos"))] {
            let mut inner = self.0.borrow_mut();
            if inner.dev.host_dirty {
                if let Some(ref arr) = inner.dev.arr {
                    if let Device::Mps = inner.device {
                        let host = MpsBackend::new().to_host_f32(arr)?;
                        inner.storage = Storage::F32(host);
                        inner.dev.host_dirty = false;
                        inner.dev.dev_dirty = false;
                    }
                }
            }
        }
        Ok(())
    }
    pub fn ensure_device(&self) -> Result<()> {
        #[cfg(all(feature="mps", target_os="macos"))] {
            let mut inner = self.0.borrow_mut();
            if let Device::Mps = inner.device {
                if inner.dev.arr.is_none() || inner.dev.dev_dirty {
                    let be = MpsBackend::new();
                    let host = match &inner.storage { Storage::F32(a) => a.clone(), _ => panic!("upload requires f32") };
                    let d = be.from_host_f32(&host)?;
                    inner.dev.arr = Some(d);
                    inner.dev.host_dirty = false;
                    inner.dev.dev_dirty = false;
                }
            }
        }
        Ok(())
    }
    #[cfg(all(feature="mps", target_os="macos"))]
    pub fn device_array(&self) -> Option<BackendArrayF32> { self.0.borrow().dev.arr.clone() }
    #[cfg(all(feature="mps", target_os="macos"))]
    pub fn from_device_array(arr: BackendArrayF32, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor {
        let host = ArrayD::<f32>::zeros(IxDyn(&shape));
        let inner = Inner {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            storage: Storage::F32(host),
            dtype: DType::F32,
            device,
            requires_grad,
            grad: None,
            grad_dev: None,
            grad_fn: None,
            out_index: 0,
            num_outputs: 1,
            dev: DevState { arr: Some(arr), host_dirty: true, dev_dirty: false },
        };
        Tensor(Rc::new(RefCell::new(inner)))
    }
    #[inline] pub fn mark_device_fresh(&self) { self.0.borrow_mut().dev.host_dirty = true; }
    #[inline] pub fn mark_host_fresh(&self) { self.0.borrow_mut().dev.dev_dirty = true; }

    pub fn to_device(mut self, device: Device) -> Self {
        let mut inner = self.0.borrow_mut();
        if inner.device != device {
            inner.device = device;
            #[cfg(all(feature="mps", target_os="macos"))]
            match device {
                Device::Cpu => { inner.dev.arr = None; inner.dev.host_dirty = false; inner.dev.dev_dirty = false; }
                Device::Mps => { inner.dev.dev_dirty = true; }
                _ => {}
            }
        }
        drop(inner);
        self
    }

    // ==== Accessors ====
    pub fn data(&self) -> ArrayD<f32> { self.ensure_host().ok(); match &self.0.borrow().storage { Storage::F32(a) => a.clone(), _ => panic!("dtype != f32") } }
    pub fn data_i32(&self) -> ArrayD<i32> { match &self.0.borrow().storage { Storage::I32(a) => a.clone(), _ => panic!("dtype != i32") } }
    pub fn data_bool(&self) -> ArrayD<bool> { match &self.0.borrow().storage { Storage::Bool(a) => a.clone(), _ => panic!("dtype != bool") } }
    pub fn grad(&self) -> Option<ArrayD<f32>> { self.0.borrow().grad.clone() }
    pub fn set_grad(&self, g: Option<ArrayD<f32>>) { self.0.borrow_mut().grad = g; }
    #[cfg(all(feature="mps", target_os="macos"))]
    pub fn grad_device(&self) -> Option<BackendArrayF32> { self.0.borrow().grad_dev.clone() }

    pub fn attach_grad_fn(&self, gf: GradFn, out_index: usize, num_outputs: usize, requires_grad: bool) {
        self.0.borrow_mut().grad_fn = Some(gf);
        self.0.borrow_mut().out_index = out_index;
        self.0.borrow_mut().num_outputs = num_outputs;
        if requires_grad && matches!(self.0.borrow().dtype, DType::F32) { self.0.borrow_mut().requires_grad = true; }
    }
    #[inline] pub fn grad_fn(&self) -> Option<GradFn> { self.0.borrow().grad_fn.clone() }
    #[inline] pub fn out_index(&self) -> usize { self.0.borrow().out_index }
    #[inline] pub fn num_outputs(&self) -> usize { self.0.borrow().num_outputs }

    // ==== Backward ====
    pub fn backward(&self) -> Result<()> {
        let seed = if self.ndim() == 0 { ArrayD::<f32>::from_elem(IxDyn(&[]), 1.0) } else { ArrayD::<f32>::from_elem(IxDyn(&self.shape()), 1.0) };
        run_backward(self, seed)
    }
    pub fn backward_with_grad(&self, grad: &ArrayD<f32>) -> Result<()> { run_backward(self, grad.clone()) }

    // ==== Engine helpers ====
    pub(crate) fn accumulate_host_grad(&self, g: &ArrayD<f32>) -> Result<()> {
        if matches!(self.0.borrow().dtype, DType::F32) {
            let mut inner = self.0.borrow_mut();
            inner.grad = Some(if let Some(old) = &inner.grad { old.clone() + g } else { g.clone() });
        }
        Ok(())
    }
    #[cfg(all(feature="mps", target_os="macos"))]
    pub(crate) fn accumulate_device_grad(&self, arr: &BackendArrayF32) -> Result<()> {
        if !matches!(self.0.borrow().dtype, DType::F32) { return Ok(()); }
        let be = MpsBackend::new();
        let mut inner = self.0.borrow_mut();
        match (&mut inner.grad_dev) {
            Some(old) => { inner.grad_dev = Some(be.add(old, arr)?); inner.grad = None; },
            None => { inner.grad_dev = Some(arr.clone()); inner.grad = None; }
        }
        Ok(())
    }
}
pub fn unbroadcast(mut g: ArrayD<f32>, target: &[usize]) -> ArrayD<f32> {
    let gshape = g.shape().to_vec();
    let mut t = target.to_vec();
    if gshape.len() > t.len() { let mut pad = vec![1usize; gshape.len() - t.len()]; pad.extend_from_slice(&t); t = pad; }
    for ax in (0..gshape.len()).rev() { if t[ax] == 1 && gshape[ax] != 1 { g = g.sum_axis(Axis(ax)); } }
    g.into_dimensionality::<IxDyn>().unwrap()
}
