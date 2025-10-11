
use crate::error::{Result, device as dev_err};
use super::BackendArrayF32;
use super::wgpu_topk_blockmerge_kway::WgpuTopKBlockMergeKWay;
use super::wgpu_topk_bigk::WgpuTopKBigK;
use super::wgpu_topk_passk::{WgpuTopKPassK, adapter_info};
use super::wgpu_mask_indices::WgpuMaskIndices;

fn autotune_k_lane(k: usize, cols: usize) -> usize {
    let info = adapter_info();
    let vendor = info.vendor;
    // simple mapping: assume NVIDIA (0x10de) subgroup 32 -> prefer 4; AMD (0x1002) subgroup 64 -> prefer 8; Apple (0x106b) -> 8; Intel (0x8086) -> 4
    if k <= 32 {
        return 4;
    }
    match vendor {
        0x1002 | 0x106b => 8, // AMD/Apple
        0x10de | 0x8086 => if (cols / k) >= 32 { 8 } else { 4 }, // NVIDIA/Intel
        _ => 8,
    }
}

pub struct WgpuTopKUnified;
impl WgpuTopKUnified {
    pub fn new()->Self{ WgpuTopKUnified }

    pub fn topk_lastdim(&self, x:&BackendArrayF32, rows:usize, cols:usize, k:usize) -> Result<(ndarray::ArrayD<f32>, ndarray::ArrayD<i32>)> {
        match x { BackendArrayF32::Wgpu{..} => {}, _ => return Err(dev_err("WGPU TopK Unified: input is not WGPU backend")) }
        if k==0 { return Err(dev_err("TopK: k must be > 0")); }
        if k <= 128 {
            let k_lane = autotune_k_lane(k, cols);
            return WgpuTopKBlockMergeKWay::new().topk_lastdim(x, rows, cols, k, k_lane);
        }
        if k <= 256 { return WgpuTopKBigK::new().topk_lastdim(x, rows, cols, k); }

        // k > 256 multi-pass with pass_k(rem)
        let pass = WgpuTopKPassK::new();
        let mask = WgpuMaskIndices::new();
        let (dev, queue) = WgpuTopKPassK::device();

        // working copy (mask in place)
        let xb = match x { BackendArrayF32::Wgpu{ buffer, .. } => buffer, _=> unreachable!() };
        let bytes = (rows*cols*4) as u64;
        let x_work = dev.create_buffer(&wgpu::BufferDescriptor{
            label: Some("x_work"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });
        { let mut e = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("copy-x") });
          e.copy_buffer_to_buffer(&xb, 0, &x_work, 0, bytes); queue.submit(std::iter::once(e.finish())); }

        let acc_v = dev.create_buffer(&wgpu::BufferDescriptor{
            label: Some("acc_v"), size: (rows*k*4) as u64,
            usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false
        });
        let acc_i = dev.create_buffer(&wgpu::BufferDescriptor{
            label: Some("acc_i"), size: (rows*k*4) as u64,
            usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false
        });

        let full = k / 256;
        let rem  = k % 256;
        let row_stride = (k*4) as u64;
        for p in 0..full {
            let (pv, pi) = pass.pass_k(&x, rows, cols, 256)?;
            let off = (p*256*4) as u64;
            let mut e = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("acc-copy") });
            for r in 0..rows {
                let src_off = (r*256*4) as u64;
                let dst_off = (r as u64)*row_stride + off;
                e.copy_buffer_to_buffer(&pv, src_off, &acc_v, dst_off, 256*4);
                e.copy_buffer_to_buffer(&pi, src_off, &acc_i, dst_off, 256*4);
            }
            queue.submit(std::iter::once(e.finish()));
            mask.mask(&x_work, &pi, rows, cols, 256)?;
        }
        if rem > 0 {
            let (pv, pi) = pass.pass_k(&x, rows, cols, rem)?;
            let off = (full*256*4) as u64;
            let bytes_rem = (rem*4) as u64;
            let mut e = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("acc-copy-rem") });
            for r in 0..rows {
                let src_off = (r*rem*4) as u64;
                let dst_off = (r as u64)*row_stride + off;
                e.copy_buffer_to_buffer(&pv, src_off, &acc_v, dst_off, bytes_rem);
                e.copy_buffer_to_buffer(&pi, src_off, &acc_i, dst_off, bytes_rem);
            }
            queue.submit(std::iter::once(e.finish()));
        }

        // map once
        let slice_v = acc_v.slice(..);
        let slice_i = acc_i.slice(..);
        let _ = slice_v.map_async(wgpu::MapMode::Read);
        let _ = slice_i.map_async(wgpu::MapMode::Read);
        dev.poll(wgpu::Maintain::Wait);
        let dv = slice_v.get_mapped_range().to_vec();
        let di = slice_i.get_mapped_range().to_vec();
        drop(slice_v); drop(slice_i);
        acc_v.unmap(); acc_i.unmap();
        let vals: Vec<f32> = bytemuck::cast_slice::<u8,f32>(&dv).to_vec();
        let idxu: Vec<u32> = bytemuck::cast_slice::<u8,u32>(&di).to_vec();
        let idxs: Vec<i32> = idxu.into_iter().map(|u| u as i32).collect();
        let v = ndarray::Array2::from_shape_vec((rows, k), vals).unwrap().into_dyn();
        let i = ndarray::Array2::from_shape_vec((rows, k), idxs).unwrap().into_dyn();
        Ok((v,i))
    }
}
