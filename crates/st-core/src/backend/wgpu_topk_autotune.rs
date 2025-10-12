
use wgpu::{Instance, RequestAdapterOptions};
pub struct WgpuAutotune;
impl WgpuAutotune {
    pub fn pick_k_lane(k: usize, cols: usize) -> usize {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default()));
        if adapter.is_none() { return if k<=32 {4} else {8}; }
        let adapter = adapter.unwrap();
        let limits = adapter.limits();
        let info = adapter.get_info();
        let wg_ok = limits.max_compute_workgroup_size_x >= 256;
        let prefer8 = cols.saturating_div(k.max(1)) >= 32 || wg_ok;
        match info.vendor {
            0x1002 | 0x106b => 8,
            0x10de | 0x8086 => if prefer8 { 8 } else { 4 },
            _ => if prefer8 { 8 } else { 4 },
        }
    }
}
