use once_cell::sync::OnceCell;
use cust::prelude::*;
use crate::error::{Result, device as dev_err};
use super::BackendArrayF32;

pub struct CudaBackend;
impl CudaBackend { pub fn new() -> Self { CudaBackend } }

static PTX: &str = include_str!("cuda_kernels.ptx");

struct Ctx {
    _ctx: Context, stream: Stream, module: Module,
    fn_add: Function<'static>, fn_t2d: Function<'static>, fn_tiled: Function<'static>,
    fn_wmma: Option<Function<'static>>,
}
static CTX: OnceCell<Ctx> = OnceCell::new();
fn ctx()->&'static Ctx {
    CTX.get_or_init(|| {
        let ctx = cust::quick_init().expect("cuda");
        let stream = Stream::new(StreamFlags::DEFAULT, None).expect("stream");
        let module = Module::from_ptx(PTX, &[]).expect("ptx");
        let m: &'static Module = Box::leak(Box::new(module));
        let fn_add = m.get_function("add_vec").expect("add");
        let fn_t2d = m.get_function("transpose_2d").expect("t2d");
        let fn_tiled = m.get_function("gemm_tiled_16").expect("gemm_tiled_16");
        let fn_wmma = m.get_function("gemm_wmma_16x16x16_stub").ok();
        Ctx{ _ctx: ctx, stream, module: (*m).clone(), fn_add, fn_t2d, fn_tiled, fn_wmma }
    })
}

impl CudaBackend {
    pub fn from_host_f32(&self, host:&ndarray::ArrayD<f32>) -> Result<BackendArrayF32> {
        let slice = host.as_slice().ok_or_else(|| dev_err("host not contiguous"))?;
        let mut buf = DeviceBuffer::<f32>::uninitialized(slice.len()).map_err(|e| dev_err(&format!("alloc: {e}")))?;
        buf.copy_from(slice).map_err(|e| dev_err(&format!("H2D: {e}")))?;
        Ok(BackendArrayF32::Cuda { rows: slice.len(), cols: 1, ptr: std::sync::Arc::new(buf) })
    }
    pub fn to_host_f32(&self, arr:&BackendArrayF32) -> Result<ndarray::ArrayD<f32>> {
        match arr {
            BackendArrayF32::Cuda{ rows, cols, ptr } => {
                let len = rows*cols; let mut v=vec![0f32; len];
                unsafe { ptr.copy_to(v.as_mut_slice()).map_err(|e| dev_err(&format!("D2H: {e}")))?; }
                Ok(ndarray::Array1::from_vec(v).into_dyn())
            }
            _ => Err(dev_err("to_host: non-cuda"))
        }
    }
    pub fn transpose2d(&self, x:&BackendArrayF32, rows:usize, cols:usize) -> Result<BackendArrayF32> {
        let xp = match x { BackendArrayF32::Cuda{ ptr, .. } => ptr, _=> return Err(dev_err("non-cuda")) };
        let mut out = DeviceBuffer::<f32>::uninitialized(rows*cols).map_err(|e| dev_err(&format!("alloc: {e}")))?;
        let total = (rows*cols) as u32; let grid = ((total+255)/256, 1, 1);
        unsafe {
            launch!(ctx().fn_t2d<<<grid.0, 256, 0, ctx().stream>>>(
                xp.as_device_ptr(), out.as_device_ptr(), rows as u32, cols as u32
            ))?;
        }
        ctx().stream.synchronize().ok();
        Ok(BackendArrayF32::Cuda{ rows: cols, cols: rows, ptr: std::sync::Arc::new(out) })
    }
    pub fn gemm2d_tiled(&self, a:&BackendArrayF32, b:&BackendArrayF32, m:usize, k:usize, n:usize) -> Result<BackendArrayF32> {
        let (ap,bp) = match (a,b) { (BackendArrayF32::Cuda{ptr:ap,..}, BackendArrayF32::Cuda{ptr:bp,..}) => (ap,bp), _=> return Err(dev_err("gemm2d: non-cuda")) };
        let mut c = DeviceBuffer::<f32>::uninitialized(m*n).map_err(|e| dev_err(&format!("alloc: {e}")))?;
        let grid = (((n as u32)+15)//16, ((m as u32)+15)//16, 1u32);
        let block = (16u32,16u32,1u32);
        unsafe {
            launch!(ctx().fn_tiled<<<grid, block, 0, ctx().stream>>>(
                ap.as_device_ptr(), bp.as_device_ptr(), c.as_device_ptr(), m as u32, k as u32, n as u32
            ))?;
        }
        ctx().stream.synchronize().ok();
        Ok(BackendArrayF32::Cuda{ rows:m, cols:n, ptr: std::sync::Arc::new(c) })
    }
}
