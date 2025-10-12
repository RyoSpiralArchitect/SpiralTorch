
#[cfg(feature="cuda")] pub mod cuda_topk_passk;
#[cfg(feature="cuda")] pub mod cuda_where;

#[cfg(feature="wgpu")] pub mod wgpu_where_nd;
#[cfg(feature="wgpu")] pub mod wgpu_topk_autotune;

#[cfg(feature="mps")]  pub mod mps_where;

#[cfg(feature="cuda")] pub const PTX_TOPK: &str = include_str!("cuda_topk_all.ptx");
#[cfg(feature="cuda")] pub const PTX_WHERE: &str = include_str!("cuda_where_nd_strided_u8_v4.ptx");
#[cfg(feature="wgpu")] pub const WGSL_BASE: &str = include_str!("wgpu_kernels_all.wgsl");
#[cfg(feature="wgpu")] pub const WGSL_WHERE_APPEND: &str = include_str!("wgpu_kernels_where_nd_strided_u8.append.wgsl");
#[cfg(feature="mps")]  pub const MSL_WHERE: &str = include_str!("mps_where_nd_strided_u8.metal");
