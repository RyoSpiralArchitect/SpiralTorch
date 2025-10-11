
use std::{cell::RefCell, rc::Rc, sync::atomic::{AtomicUsize, Ordering}};
use ndarray::{ArrayD, IxDyn, Axis};
use crate::{device::Device, dtype::DType, error::Result, autograd::GradFn};
use crate::backend::BackendArrayF32;

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<Inner>>);

pub enum Storage {
    F32(ArrayD<f32>),
    I32(ArrayD<i32>),
    Bool(ArrayD<bool>),
}

pub struct Inner {
    id: usize,
    pub storage: Storage,
    pub dtype: DType,
    pub device: Device,
    pub requires_grad: bool,
    pub grad: Option<ArrayD<f32>>,
    pub grad_fn: Option<GradFn>,
    pub out_index: usize,
    pub num_outputs: usize,
    pub dev: Option<BackendArrayF32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn from_array(data: ArrayD<f32>) -> Self {
        let shape = data.shape().to_vec();
        let inner = Inner {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            storage: Storage::F32(data),
            dtype: DType::F32,
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            out_index: 0,
            num_outputs: 1,
            dev: None,
            shape,
        };
        Tensor(Rc::new(RefCell::new(inner)))
    }
    pub fn from_i32(data: ArrayD<i32>) -> Self {
        let shape = data.shape().to_vec();
        let inner = Inner {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            storage: Storage::I32(data),
            dtype: DType::I32,
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            out_index: 0,
            num_outputs: 1,
            dev: None,
            shape,
        };
        Tensor(Rc::new(RefCell::new(inner)))
    }
    pub fn from_bool(data: ArrayD<bool>) -> Self {
        let shape = data.shape().to_vec();
        let inner = Inner {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            storage: Storage::Bool(data),
            dtype: DType::Bool,
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            out_index: 0,
            num_outputs: 1,
            dev: None,
            shape,
        };
        Tensor(Rc::new(RefCell::new(inner)))
    }
    pub fn from_device_array(arr: BackendArrayF32, shape: Vec<usize>, device: Device, requires_grad: bool) -> Self {
        let inner = Inner {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            storage: Storage::F32(ArrayD::<f32>::zeros(IxDyn(&shape))), // lazy host fetch
            dtype: DType::F32, device, requires_grad, grad: None, grad_fn: None,
            out_index: 0, num_outputs: 1, dev: Some(arr), shape,
        };
        Tensor(Rc::new(RefCell::new(inner)))
    }
    pub fn zeros(shape: &[usize]) -> Self { Self::from_array(ArrayD::<f32>::zeros(IxDyn(shape))) }
    pub fn ones(shape: &[usize]) -> Self { Self::from_array(ArrayD::<f32>::from_elem(IxDyn(shape), 1.0)) }

    pub fn id(&self) -> usize { self.0.borrow().id }
    pub fn dtype(&self) -> DType { self.0.borrow().dtype }
    pub fn shape(&self) -> Vec<usize> { self.0.borrow().shape.clone() }
    pub fn ndim(&self) -> usize { self.0.borrow().shape.len() }
    pub fn device(&self) -> Device { self.0.borrow().device }
    pub fn requires_grad(mut self, flag: bool) -> Self {
        if matches!(self.0.borrow().dtype, DType::F32) { self.0.borrow_mut().requires_grad = flag; } self
    }
    pub fn data(&self) -> ArrayD<f32> { match &self.0.borrow().storage { Storage::F32(a) => a.clone(), _ => panic!("dtype != f32") } }
    pub fn data_i32(&self) -> ArrayD<i32> { match &self.0.borrow().storage { Storage::I32(a) => a.clone(), _ => panic!("dtype != i32") } }
    pub fn data_bool(&self) -> ArrayD<bool> { match &self.0.borrow().storage { Storage::Bool(a) => a.clone(), _ => panic!("dtype != bool") } }
    pub fn grad(&self) -> Option<ArrayD<f32>> { self.0.borrow().grad.clone() }
    pub fn device_array(&self) -> Option<BackendArrayF32> { self.0.borrow().dev.clone() }

    pub fn ensure_device(&self) -> Result<()> {
        if self.0.borrow().device == Device::Cpu { return Ok(()); }
        if self.0.borrow().dev.is_some() { return Ok(()); }
        let host = self.data();
        let arr = match self.0.borrow().device {
            Device::Wgpu => { let be = crate::backend::WgpuBackend::new(); be.from_host_f32(&host)? }
            Device::Mps  => { let be = crate::backend::MpsBackend::new();  be.from_host_f32(&host)? }
            Device::Cuda => { let be = crate::backend::CudaBackend::new(); be.from_host_f32(&host)? }
            Device::Cpu  => unreachable!(),
        };
        self.0.borrow_mut().dev = Some(arr);
        Ok(())
    }
    pub fn to_device(mut self, device: Device) -> Self {
        self.0.borrow_mut().device = device;
        if device != Device::Cpu { let _ = self.ensure_device(); } else { self.0.borrow_mut().dev = None; }
        self
    }
    pub fn attach_grad_fn(&self, gf: GradFn, out_index: usize, num_outputs: usize, requires_grad: bool) {
        self.0.borrow_mut().grad_fn = Some(gf); self.0.borrow_mut().out_index = out_index; self.0.borrow_mut().num_outputs = num_outputs;
        if requires_grad && matches!(self.0.borrow().dtype, crate::dtype::DType::F32) { self.0.borrow_mut().requires_grad = true; }
    }
    pub fn backward(&self) -> Result<()> {
        let seed = if self.ndim() == 0 { ArrayD::<f32>::from_elem(IxDyn(&[]), 1.0) } else { ArrayD::<f32>::from_elem(IxDyn(&self.shape()), 1.0) };
        self.backward_with_grad(&seed)
    }
    pub fn backward_with_grad(&self, grad: &ArrayD<f32>) -> Result<()> {
        if self.0.borrow().grad_fn.is_none() {
            if matches!(self.0.borrow().dtype, DType::F32) {
                let g = if let Some(old) = &self.0.borrow().grad { old.clone() + grad } else { grad.clone() };
                self.0.borrow_mut().grad = Some(g);
            }
            return Ok(())
        }
        let mut topo: Vec<GradFn> = Vec::new();
        let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
        fn collect_from_tensor(t: &Tensor, topo: &mut Vec<GradFn>, visited: &mut std::collections::HashSet<usize>) {
            if let Some(gf) = t.0.borrow().grad_fn.clone() {
                let key = gf.key();
                if !visited.contains(&key) {
                    visited.insert(key);
                    for p in gf.parents() { collect_from_tensor(&p, topo, visited); }
                    topo.push(gf);
                }
            }
        }
        collect_from_tensor(self, &mut topo, &mut visited);
        let mut out_grads: std::collections::HashMap<usize, Vec<Option<ArrayD<f32>>>> = std::collections::HashMap::new();
        fn accum_out_grad(t: &Tensor, g: ArrayD<f32>, out_grads: &mut std::collections::HashMap<usize, Vec<Option<ArrayD<f32>>>>) {
            if matches!(t.0.borrow().dtype, crate::dtype::DType::F32) {
                let updated = if let Some(old) = &t.0.borrow().grad { old.clone() + &g } else { g.clone() };
                t.0.borrow_mut().grad = Some(updated);
            }
            if let Some(gf) = t.0.borrow().grad_fn.clone() {
                let key = gf.key(); let num = t.0.borrow().num_outputs; let idx = t.0.borrow().out_index;
                let entry = out_grads.entry(key).or_insert_with(|| vec![None; num]);
                if let Some(prev) = &entry[idx] { entry[idx] = Some(prev.clone() + &g); } else { entry[idx] = Some(g); }
            }
        }
        accum_out_grad(self, grad.clone(), &mut out_grads);
        for gf in topo.into_iter().rev() {
            let key = gf.key();
            let num = gf.num_outputs();
            let grads_vec = out_grads.remove(&key).unwrap_or_else(|| vec![None; num]);
            let gin = gf.backward_multi(&grads_vec);
            let parents = gf.parents();
            for (p, maybe_g) in parents.into_iter().zip(gin.into_iter()) {
                if let Some(gp) = maybe_g {
                    if matches!(p.0.borrow().dtype, DType::F32) {
                        let updated = if let Some(old) = &p.0.borrow().grad { old.clone() + &gp } else { gp.clone() };
                        p.0.borrow_mut().grad = Some(updated.clone());
                    }
                    if p.0.borrow().grad_fn.is_some() { accum_out_grad(&p, gp, &mut out_grads); }
                }
            }
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
