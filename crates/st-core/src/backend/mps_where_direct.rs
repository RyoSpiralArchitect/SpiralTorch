use crate::error::{Result, device as dev_err};
#[repr(C)] #[derive(Clone, Copy)] struct RC { nd: u32, n: u32 }
#[repr(C)] #[derive(Clone, Copy)] struct RB { c_base: u32, x_base: u32, y_base: u32 }
pub struct MpsWhereDirect { dev: metal::Device, q: metal::CommandQueue, p: metal::ComputePipelineState }
impl MpsWhereDirect {
    pub fn new() -> Result<Self> {
        let dev = metal::Device::system_default().ok_or_else(|| dev_err("MPS device not found"))?;
        let q = dev.new_command_queue();
        let lib = dev.new_library_with_source(crate::backend::MSL_WHERE, &metal::CompileOptions::new()).map_err(|e| dev_err(&format!("MSL: {:?}", e)))?;
        let f = lib.get_function("where_nd_strided_u8", None).map_err(|e| dev_err(&format!("fn: {:?}", e)))?;
        let p = dev.new_compute_pipeline_state_with_function(&f).map_err(|e| dev_err(&format!("pipeline: {:?}", e)))?;
        Ok(Self{ dev, q, p })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn run_direct(&self,
        c_blob:&[u8], x_blob:&[u8], y_blob:&[u8],
        b_out_shape:&metal::Buffer, b_out_strides:&metal::Buffer,
        b_c_shape:&metal::Buffer, b_c_strides:&metal::Buffer, c_base:u32,
        b_x_shape:&metal::Buffer, b_x_strides:&metal::Buffer, x_base:u32,
        b_y_shape:&metal::Buffer, b_y_strides:&metal::Buffer, y_base:u32,
        nd:u32, n:u32
    ) -> Result<Vec<f32>> {
        let pad = (c_blob.len() + 3) & !3; let mut c_pad = vec![0u8; pad];
        c_pad[..c_blob.len()].copy_from_slice(c_blob);
        let b_cond = self.dev.new_buffer_with_data(&c_pad, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let b_x    = self.dev.new_buffer_with_data(x_blob, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let b_y    = self.dev.new_buffer_with_data(y_blob, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let out = self.dev.new_buffer((n as u64)*4, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let rc = RC{ nd, n }; let rb = RB{ c_base, x_base, y_base };
        let b_rc = self.dev.new_buffer_with_data(bytemuck::bytes_of(&rc), metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let b_rb = self.dev.new_buffer_with_data(bytemuck::bytes_of(&rb), metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let cb = self.q.new_command_buffer(); let ce = cb.new_compute_command_encoder();
        ce.set_compute_pipeline_state(&self.p);
        ce.set_buffer(0, Some(&b_cond), 0); ce.set_buffer(1, Some(&b_x), 0); ce.set_buffer(2, Some(&b_y), 0);
        ce.set_buffer(3, Some(&out), 0);
        ce.set_buffer(4, Some(b_out_shape), 0); ce.set_buffer(5, Some(b_out_strides), 0);
        ce.set_buffer(6, Some(b_c_shape), 0); ce.set_buffer(7, Some(b_c_strides), 0);
        ce.set_buffer(8, Some(b_x_shape), 0); ce.set_buffer(9, Some(b_x_strides), 0);
        ce.set_buffer(10, Some(b_y_shape), 0); ce.set_buffer(11, Some(b_y_strides), 0);
        ce.set_buffer(12, Some(&b_rc), 0); ce.set_buffer(13, Some(&b_rb), 0);
        let w = self.p.thread_execution_width();
        let tg = metal::MTLSize { width: w, height: 1, depth: 1 };
        let grid = metal::MTLSize { width: n as u64, height: 1, depth: 1 };
        ce.dispatch_threads(grid, tg); ce.end_encoding(); cb.commit(); cb.wait_until_completed();
        let outv = unsafe{ std::slice::from_raw_parts(out.contents() as *const f32, n as usize) }.to_vec();
        Ok(outv)
    }
}
