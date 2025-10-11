use once_cell::sync::OnceCell;
use crate::error::{Result, device as dev_err};
use super::BackendArrayF32;
mod pool { pub use crate::backend::mps_pool::*; }

pub struct MpsBackend;
impl MpsBackend { pub fn new() -> Self { MpsBackend } }

pub struct Ctx { pub device: metal::Device, pub queue: metal::CommandQueue }
static CTX: OnceCell<Ctx> = OnceCell::new();
pub(crate) fn ctx()->&'static Ctx{
    CTX.get_or_init(|| {
        let device = metal::Device::system_default().expect("No MTLDevice");
        let queue = device.new_command_queue();
        Ctx{ device, queue }
    })
}

impl MpsBackend {
    pub fn from_host_f32(&self, host:&ndarray::ArrayD<f32>) -> Result<BackendArrayF32> {
        let len = host.len(); let bytes = (len*4) as u64;
        let buf = ctx().device.new_buffer(bytes, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        unsafe { std::ptr::copy_nonoverlapping(host.as_ptr(), buf.contents() as *mut f32, len); }
        Ok(BackendArrayF32::Mps{ rows: len, cols:1, buffer: buf })
    }
    pub fn to_host_f32(&self, arr:&BackendArrayF32) -> Result<ndarray::ArrayD<f32>> {
        match arr {
            BackendArrayF32::Mps{ rows, cols, buffer } => {
                let len = rows*cols; let mut v=vec![0f32; len];
                unsafe { std::ptr::copy_nonoverlapping(buffer.contents() as *const f32, v.as_mut_ptr(), len); }
                Ok(ndarray::Array1::from_vec(v).into_dyn())
            }
            _ => Err(dev_err("to_host: non-mps"))
        }
    }
    // (省略) matmul2d_batch_bwd の β-accumulate 実装は v1.3.6 参照: 転置は pool::temp を用いて一時を確保→pool::recycle
}
