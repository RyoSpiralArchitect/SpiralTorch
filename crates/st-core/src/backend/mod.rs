#[cfg(feature="wgpu")]
pub const WGSL_BASE: &str = include_str!("wgpu_kernels_all.wgsl");
#[cfg(feature="wgpu")]
pub const WGSL_WHERE_APPEND: &str = include_str!("wgpu_kernels_where_nd_strided_u8.append.wgsl");
#[cfg(feature="wgpu")]
pub mod wgpu_where_segments;
#[cfg(feature="wgpu")]
pub mod wgpu_topk_kway;

#[cfg(feature="mps")]
pub const MSL_WHERE: &str = include_str!("mps_where_nd_strided_u8.metal");
#[cfg(feature="mps")]
pub mod mps_where;
#[cfg(feature="mps")]
pub mod mps_where_segments;

#[cfg(feature="cuda")]
pub mod cuda_topk_kway;
