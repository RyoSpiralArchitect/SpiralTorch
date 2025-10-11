
use ndarray::ArrayD;
use once_cell::sync::OnceCell;
use cust::prelude::*;
use crate::{device::Device, error::{Result, device as dev_err}};

pub struct CudaBackend;
impl CudaBackend { pub fn new() -> Self { CudaBackend } }

pub enum BackendArrayF32 {
    Cuda { rows: usize, cols: usize, ptr: std::sync::Arc<cust::memory::DeviceBuffer<f32>> },
    #[allow(dead_code)] HostStub,
}

pub trait Backend {
    fn name(&self)->&'static str;
    fn device(&self)->Device;
    fn from_host_f32(&self, host:&ArrayD<f32>) -> Result<BackendArrayF32>;
    fn to_host_f32(&self, arr:&BackendArrayF32) -> Result<ArrayD<f32>>;
}

static PTX: &str = include_str!("cuda_kernels.ptx");

struct Ctx {
    _ctx: Context,
    stream: Stream,
    module: Module,
    add_fn: Function<'static>,
    t2d_fn: Function<'static>,
}
static CTX: OnceCell<Ctx> = OnceCell::new();
fn ctx()->&'static Ctx {
    CTX.get_or_init(|| {
        let ctx = cust::quick_init().expect("cuda");
        let stream = Stream::new(StreamFlags::DEFAULT, None).expect("stream");
        let module = Module::from_ptx(PTX, &[]).expect("ptx");
        let module_static: &'static Module = Box::leak(Box::new(module));
        let add_fn = module_static.get_function("add_vec").expect("add_vec");
        let t2d_fn = module_static.get_function("transpose_2d").expect("transpose_2d");
        Ctx{ _ctx: ctx, stream, module: (*module_static).clone(), add_fn, t2d_fn }
    })
}

impl super::Backend for CudaBackend {
    fn name(&self)->&'static str { "cuda" }
    fn device(&self)->Device { Device::Cuda }
    fn from_host_f32(&self, host:&ArrayD<f32>) -> Result<BackendArrayF32> {
        let slice = host.as_slice().ok_or_else(|| dev_err("host not contiguous"))?;
        let mut buf = DeviceBuffer::<f32>::uninitialized(slice.len()).map_err(|e| dev_err(&format!("alloc: {e}")))?;
        buf.copy_from(slice).map_err(|e| dev_err(&format!("H2D: {e}")))?;
        Ok(BackendArrayF32::Cuda{ rows: slice.len(), cols: 1, ptr: std::sync::Arc::new(buf) })
    }
    fn to_host_f32(&self, arr:&BackendArrayF32) -> Result<ArrayD<f32>> {
        match arr {
            BackendArrayF32::Cuda{ rows, cols, ptr } => {
                let len = rows*cols; let mut v=vec![0f32; len];
                unsafe { ptr.copy_to(v.as_mut_slice()).map_err(|e| dev_err(&format!("D2H: {e}")))?; }
                Ok(ndarray::Array1::from_vec(v).into_dyn())
            }
            _ => Err(dev_err("to_host: non-cuda"))
        }
    }
}

impl CudaBackend {
    pub fn add(&self, a:&BackendArrayF32, b:&BackendArrayF32) -> Result<BackendArrayF32> {
        let (n, ap, bp) = match (a,b) {
            (BackendArrayF32::Cuda{ rows:ra, cols:ca, ptr:ap }, BackendArrayF32::Cuda{ rows:rb, cols:cb, ptr:bp }) => {
                assert!(ra*ca==rb*cb); (ra*ca, ap, bp)
            }, _=> return Err(dev_err("add: non-cuda"))
        };
        let mut out = DeviceBuffer::<f32>::uninitialized(*n).map_err(|e| dev_err(&format!("alloc: {e}")))?;
        let grid = (((*n as u32)+255)/256, 1, 1);
        unsafe {
            launch!(ctx().add_fn<<<grid.0, 256, 0, ctx().stream>>>(
                ap.as_device_ptr(), bp.as_device_ptr(), out.as_device_ptr(), *n as u32
            )).map_err(|e| dev_err(&format!("launch add: {e}")))?;
        }
        ctx().stream.synchronize().ok();
        Ok(BackendArrayF32::Cuda{ rows:*n, cols:1, ptr: std::sync::Arc::new(out) })
    }
    pub fn transpose2d(&self, x:&BackendArrayF32, rows:usize, cols:usize) -> Result<BackendArrayF32> {
        let xp = match x { BackendArrayF32::Cuda{ ptr, .. } => ptr, _=> return Err(dev_err("non-cuda")) };
        let mut out = DeviceBuffer::<f32>::uninitialized(rows*cols).map_err(|e| dev_err(&format!("alloc: {e}")))?;
        let total = (rows*cols) as u32; let grid = ((total+255)/256, 1, 1);
        unsafe {
            launch!(ctx().t2d_fn<<<grid.0, 256, 0, ctx().stream>>>(
                xp.as_device_ptr(), out.as_device_ptr(), rows as u32, cols as u32
            )).map_err(|e| dev_err(&format!("launch t2d: {e}")))?;
        }
        ctx().stream.synchronize().ok();
        Ok(BackendArrayF32::Cuda{ rows: cols, cols: rows, ptr: std::sync::Arc::new(out) })
    }
}
