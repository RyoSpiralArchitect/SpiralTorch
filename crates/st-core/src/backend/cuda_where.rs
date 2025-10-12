
#[cfg(feature="cuda")]
use cust::{prelude::*, memory::DeviceBuffer};
#[cfg(feature="cuda")]
use crate::error::{Result, device as dev_err};
#[cfg(feature="cuda")]
static PTX_WHERE_V4: &str = include_str!("cuda_where_nd_strided_u8_v4.ptx");
#[cfg(feature="cuda")]
pub struct CudaWhereND { module: Module, f_v4: cust::function::Function<'static> }
#[cfg(feature="cuda")]
impl CudaWhereND {
    pub fn new() -> Result<Self> {
        let _ = cust::quick_init().map_err(|e| dev_err(&format!("cuda init: {e}")))?;
        let module = Module::from_ptx(PTX_WHERE_V4, &[]).map_err(|e| dev_err(&format!("ptx load: {e}")))?;
        let f_v4 = unsafe { std::mem::transmute::<_, cust::function::Function<'static>>(module.get_function("where_nd_strided_u8_v4").unwrap()) };
        Ok(Self { module, f_v4 })
    }
}
