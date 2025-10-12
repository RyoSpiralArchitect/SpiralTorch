
use crate::error::Result;
use ndarray::ArrayD;

#[derive(Clone)]
pub enum BackendArrayF32 {
    Cpu{ data: ArrayD<f32> },
    #[cfg(feature="cuda")]
    Cuda{ rows: usize, cols: usize, ptr: std::sync::Arc<cust::memory::DeviceBuffer<f32>> },
    #[cfg(feature="wgpu")]
    Wgpu{ rows: usize, cols: usize, buffer: wgpu::Buffer },
    #[cfg(feature="mps")]
    Mps{ rows: usize, cols: usize, buffer: metal::Buffer },
}

#[cfg(feature="wgpu")]
pub mod wgpu_where_nd;
#[cfg(feature="wgpu")]
pub mod wgpu_topk_passk;
#[cfg(feature="wgpu")]
pub mod wgpu_mask_indices;
#[cfg(feature="wgpu")]
pub mod wgpu_topk_unified;

// Placeholders for compatibility (implemented via pass_k internally)
#[cfg(feature="wgpu")]
pub mod wgpu_topk_blockmerge_kway { use crate::error::Result; use super::{BackendArrayF32, wgpu_topk_passk::WgpuTopKPassK}; pub struct WgpuTopKBlockMergeKWay; impl WgpuTopKBlockMergeKWay{ pub fn new()->Self{Self} pub fn topk_lastdim(&self, x:&BackendArrayF32, rows:usize, cols:usize, k:usize, _k_lane:usize)->Result<(ndarray::ArrayD<f32>, ndarray::ArrayD<i32>)>{ let (dev, _q)=super::wgpu_topk_passk::WgpuTopKPassK::device(); let (pv, pi)=WgpuTopKPassK::new().pass_k(x, rows, cols, k.min(256))?; let slice_v=pv.slice(..); let slice_i=pi.slice(..); let _=slice_v.map_async(wgpu::MapMode::Read); let _=slice_i.map_async(wgpu::MapMode::Read); dev.poll(wgpu::Maintain::Wait); let dv=slice_v.get_mapped_range().to_vec(); let di=slice_i.get_mapped_range().to_vec(); drop(slice_v); drop(slice_i); pv.unmap(); pi.unmap(); let vals:Vec<f32>=bytemuck::cast_slice::<u8,f32>(&dv).to_vec(); let idxu:Vec<u32>=bytemuck::cast_slice::<u8,u32>(&di).to_vec(); let idxs:Vec<i32>=idxu.into_iter().map(|u|u as i32).collect(); Ok((ndarray::Array2::from_shape_vec((rows, k.min(256)), vals).unwrap().into_dyn(), ndarray::Array2::from_shape_vec((rows, k.min(256)), idxs).unwrap().into_dyn())) } } }
#[cfg(feature="wgpu")]
pub mod wgpu_topk_bigk { use crate::error::Result; use super::{BackendArrayF32, wgpu_topk_passk::WgpuTopKPassK}; pub struct WgpuTopKBigK; impl WgpuTopKBigK{ pub fn new()->Self{Self} pub fn topk_lastdim(&self, x:&BackendArrayF32, rows:usize, cols:usize, k:usize)->Result<(ndarray::ArrayD<f32>, ndarray::ArrayD<i32>)>{ let (dev, _q)=super::wgpu_topk_passk::WgpuTopKPassK::device(); let (pv, pi)=WgpuTopKPassK::new().pass_k(x, rows, cols, k.min(256))?; let slice_v=pv.slice(..); let slice_i=pi.slice(..); let _=slice_v.map_async(wgpu::MapMode::Read); let _=slice_i.map_async(wgpu::MapMode::Read); dev.poll(wgpu::Maintain::Wait); let dv=slice_v.get_mapped_range().to_vec(); let di=slice_i.get_mapped_range().to_vec(); drop(slice_v); drop(slice_i); pv.unmap(); pi.unmap(); let vals:Vec<f32>=bytemuck::cast_slice::<u8,f32>(&dv).to_vec(); let idxu:Vec<u32>=bytemuck::cast_slice::<u8,u32>(&di).to_vec(); let idxs:Vec<i32>=idxu.into_iter().map(|u|u as i32).collect(); Ok((ndarray::Array2::from_shape_vec((rows, k.min(256)), vals).unwrap().into_dyn(), ndarray::Array2::from_shape_vec((rows, k.min(256)), idxs).unwrap().into_dyn())) } } }

#[cfg(feature="wgpu")]
pub const WGPU_KERNELS_ALL: &str = include_str!("wgpu_kernels_all.wgsl");
#[cfg(feature="wgpu")]
pub mod wgpu_where_segments;
