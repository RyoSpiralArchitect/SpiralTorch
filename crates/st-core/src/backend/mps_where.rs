
use crate::error::{Result, device as dev_err};
pub struct MpsWhereND;
impl MpsWhereND {
    pub fn new() -> Result<Self> { Ok(Self) }
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_base(&self,
        _b_cond:&metal::Buffer, _b_x:&metal::Buffer, _b_y:&metal::Buffer, _out:&metal::Buffer,
        _b_out_shape:&metal::Buffer, _b_out_strides:&metal::Buffer,
        _b_c_shape:&metal::Buffer, _b_c_strides:&metal::Buffer, _c_base:u32,
        _b_x_shape:&metal::Buffer, _b_x_strides:&metal::Buffer, _x_base:u32,
        _b_y_shape:&metal::Buffer, _b_y_strides:&metal::Buffer, _y_base:u32,
        _nd:u32, _n:u32) -> Result<()> {
        Err(dev_err("MPS where_nd stub in this artifact"))
    }
}
