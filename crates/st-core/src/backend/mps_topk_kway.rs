use crate::error::{Result, device as dev_err};

#[repr(C)] #[derive(Clone, Copy)]
struct Meta { rows:u32, cols:u32, k:u32, k_lane:u32, chunk_cols:u32, cand_cols:u32 }

pub struct MpsTopk { dev: metal::Device, q: metal::CommandQueue, p1: metal::ComputePipelineState }
impl MpsTopk {
    pub fn new() -> Result<Self> {
        let dev = metal::Device::system_default().ok_or_else(|| dev_err("MPS device not found"))?;
        let q = dev.new_command_queue();
        let lib = dev.new_library_with_source(crate::backend::MSL_TOPK, &metal::CompileOptions::new()).map_err(|e| dev_err(&format!("MSL: {:?}", e)))?;
        let f1 = lib.get_function("topk_kway_1ce", None).map_err(|e| dev_err(&format!("fn: {:?}", e)))?;
        let p1 = dev.new_compute_pipeline_state_with_function(&f1).map_err(|e| dev_err(&format!("pipeline: {:?}", e)))?;
        Ok(Self{ dev, q, p1 })
    }
    pub fn run_1ce(&self, x:&[f32], rows:u32, cols:u32, k:u32, k_lane:u32, chunk_cols:u32) -> Result<(Vec<f32>, Vec<i32>)> {
        let outv_len = (rows*k) as usize; let outi_len = outv_len;
        let b_x = self.dev.new_buffer_with_data(bytemuck::cast_slice(x), metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let outv = self.dev.new_buffer((outv_len*4) as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let outi = self.dev.new_buffer((outi_len*4) as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let meta = Meta{ rows, cols, k, k_lane, chunk_cols, cand_cols: 256*k_lane };
        let b_meta = self.dev.new_buffer_with_data(bytemuck::bytes_of(&meta), metal::MTLResourceOptions::CPUCacheModeDefaultCache);

        let cb = self.q.new_command_buffer(); let ce = cb.new_compute_command_encoder();
        ce.set_compute_pipeline_state(&self.p1);
        ce.set_buffer(0, Some(&b_x), 0);
        ce.set_buffer(1, Some(&outv), 0);
        ce.set_buffer(2, Some(&outi), 0);
        ce.set_buffer(3, Some(&b_meta), 0);
        let tg = metal::MTLSize{ width: 256, height: 1, depth: 1 };
        let grid = metal::MTLSize{ width: rows as u64, height: 1, depth: 1 };
        ce.dispatch_threadgroups(grid, tg);
        ce.end_encoding(); cb.commit(); cb.wait_until_completed();
        let vals = unsafe { std::slice::from_raw_parts(outv.contents() as *const f32, outv_len) }.to_vec();
        let idxs = unsafe { std::slice::from_raw_parts(outi.contents() as *const i32, outi_len) }.to_vec();
        Ok((vals, idxs))
    }
}

pub fn topk_kway_2d_autotuned(x:&[f32], rows:u32, cols:u32, k:u32) -> Result<(Vec<f32>, Vec<i32>)> {
    let ch = crate::backend::wgpu_heuristics::choose(rows as usize, cols as usize, k as usize, false);
    let (k_lane, chunk_cols) = if let Some(ch) = ch { (ch.kl, ch.ch) } else { (if k>=32 {32} else if k>=16 {16} else {8}, if cols>16384 {8192} else {0}) };
    let m = MpsTopk::new()?;
    m.run_1ce(x, rows, cols, k, k_lane, chunk_cols)
}
