//! WGPU runtime & binder for Rank‑K kernels (feature `wgpu-rt`)
//! v1.7.8: add single-CE TopK (subgroups) and Mid/Bottom compaction (1CE/2CE)
#![allow(unused)]

use once_cell::sync::OnceCell;
use std::sync::Arc;

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub struct WgpuCtx {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    // lazy pipelines
    topk_1ce_pl: OnceCell<wgpu::ComputePipeline>,
    compact1_pl: OnceCell<wgpu::ComputePipeline>,
    scan_pl:     OnceCell<wgpu::ComputePipeline>,
    apply_pl:    OnceCell<wgpu::ComputePipeline>,
    layout_topk: OnceCell<wgpu::BindGroupLayout>,
    layout_cmp:  OnceCell<wgpu::BindGroupLayout>,
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
impl WgpuCtx {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self{
            device, queue,
            topk_1ce_pl: OnceCell::new(),
            compact1_pl: OnceCell::new(),
            scan_pl:     OnceCell::new(),
            apply_pl:    OnceCell::new(),
            layout_topk: OnceCell::new(),
            layout_cmp:  OnceCell::new(),
        }
    }
}

pub struct KernelIO {
    pub rows: u32, pub cols: u32, pub k: u32,
    pub row_stride: u32, pub k_lane: u32, pub tile_cols: u32,
    pub x: wgpu::Buffer, pub out_vals: wgpu::Buffer, pub out_idx: wgpu::Buffer,
}
pub struct CompactionIO {
    pub rows: u32, pub cols: u32, pub row_stride: u32,
    pub x: wgpu::Buffer, pub mask: wgpu::Buffer,
    pub out_pos: wgpu::Buffer, pub out_val: wgpu::Buffer,
    pub prefix: Option<wgpu::Buffer>, // for 2CE if needed
}

pub trait WgpuBinder: Send + Sync + 'static {
    fn bind_topk(&self, plan:&crate::ops::rank_entry::RankPlan) -> Result<KernelIO, String>;
    fn bind_compaction(&self, plan:&crate::ops::rank_entry::RankPlan) -> Result<CompactionIO, String>;
}

// Global ctx/binder
static CTX: OnceCell<Arc<WgpuCtx>> = OnceCell::new();
static BINDER: OnceCell<Arc<dyn WgpuBinder>> = OnceCell::new();
pub fn install_ctx(ctx: Arc<WgpuCtx>) { let _ = CTX.set(ctx); }
pub fn install_binder(b: Arc<dyn WgpuBinder>) { let _ = BINDER.set(b); }

fn ctx() -> Result<Arc<WgpuCtx>, String> { CTX.get().cloned().ok_or("WGPU ctx not installed".into()) }
fn binder() -> Result<Arc<dyn WgpuBinder>, String> { BINDER.get().cloned().ok_or("WGPU binder not installed".into()) }

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params { rows:u32, cols:u32, k:u32, row_stride:u32, k_lane:u32, tile_cols:u32, _pad:u32, _pad2:u32 }
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CParams { rows:u32, cols:u32, row_stride:u32, kind:u32 }

const WGSL_SRC: &str = include_str!("wgpu_kernels_rankk.wgsl");

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_layout_topk(ctx:&WgpuCtx) -> &wgpu::BindGroupLayout {
    ctx.layout_topk.get_or_init(|| {
        ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("st.rankk.layout.topk"),
            entries: &[
                // X
                wgpu::BindGroupLayoutEntry{ binding:0, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset:false, min_binding_size: None }, count: None },
                // OUT_VALS
                wgpu::BindGroupLayoutEntry{ binding:1, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset:false, min_binding_size: None }, count: None },
                // OUT_IDX
                wgpu::BindGroupLayoutEntry{ binding:2, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset:false, min_binding_size: None }, count: None },
                // Params
                wgpu::BindGroupLayoutEntry{ binding:3, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset:false, min_binding_size: None }, count: None },
            ],
        })
    })
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_layout_cmp(ctx:&WgpuCtx) -> &wgpu::BindGroupLayout {
    ctx.layout_cmp.get_or_init(|| {
        ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("st.rankk.layout.compact"),
            entries: &[
                // X
                wgpu::BindGroupLayoutEntry{ binding:0, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset:false, min_binding_size: None }, count: None },
                // MASK
                wgpu::BindGroupLayoutEntry{ binding:1, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset:false, min_binding_size: None }, count: None },
                // OUTPOS
                wgpu::BindGroupLayoutEntry{ binding:2, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset:false, min_binding_size: None }, count: None },
                // OUTVAL
                wgpu::BindGroupLayoutEntry{ binding:3, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset:false, min_binding_size: None }, count: None },
                // Params
                wgpu::BindGroupLayoutEntry{ binding:4, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset:false, min_binding_size: None }, count: None },
            ],
        })
    })
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_pl(ctx:&WgpuCtx, entry:&'static str, layout:&wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor{
        label: Some("st.wgsl.rankk"), source: wgpu::ShaderSource::Wgsl(WGSL_SRC.into())
    });
    let pl_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
        label: Some("st.rankk.pl"), bind_group_layouts: &[layout], push_constant_ranges: &[]
    });
    ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
        label: Some(entry), layout: Some(&pl_layout), module: &module, entry_point: entry
    })
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_topk_1ce_pl(ctx:&WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.topk_1ce_pl.get_or_init(|| ensure_pl(ctx, "topk_subgroups_1ce", ensure_layout_topk(ctx)))
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_compact1_pl(ctx:&WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.compact1_pl.get_or_init(|| ensure_pl(ctx, "midk_compact_1ce", ensure_layout_cmp(ctx)))
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_scan_pl(ctx:&WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.scan_pl.get_or_init(|| ensure_pl(ctx, "midk_compact_scan_pass", ensure_layout_cmp(ctx)))
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_apply_pl(ctx:&WgpuCtx) -> &wgpu::ComputePipeline {
    ctx.apply_pl.get_or_init(|| ensure_pl(ctx, "midk_compact_apply_pass", ensure_layout_cmp(ctx)))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_topk_1ce(plan:&crate::ops::rank_entry::RankPlan) -> Result<(), String> {
    let ctx = ctx()?; let b = binder()?;
    let io = b.bind_topk(plan)?;
    // Force 1CE assumption: one WG per row in X dimension
    let params = Params{ rows: io.rows, cols: io.cols, k: io.k,
        row_stride: io.row_stride, k_lane: io.k_lane, tile_cols: io.cols, _pad:0, _pad2:0 };
    let ubuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("st.rankk.params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let layout = ensure_layout_topk(&ctx);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("st.rankk.bg.topk"), layout,
        entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: io.x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: io.out_vals.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: io.out_idx.as_entire_binding()  },
            wgpu::BindGroupEntry{ binding:3, resource: ubuf.as_entire_binding() },
        ]
    });
    let pl = ensure_topk_1ce_pl(&ctx);
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("st.rankk.enc") });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("st.rankk.pass.topk") });
        cpass.set_pipeline(pl); cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(1, io.rows.max(1), 1);
    }
    ctx.queue.submit(Some(enc.finish())); Ok(())
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_compaction_1ce(plan:&crate::ops::rank_entry::RankPlan, kind:u32) -> Result<(), String> {
    let ctx = ctx()?; let b = binder()?;
    let io = b.bind_compaction(plan)?;
    let params = CParams{ rows: io.rows, cols: io.cols, row_stride: io.row_stride, kind };
    let ubuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("st.rankk.cparams"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let layout = ensure_layout_cmp(&ctx);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("st.rankk.bg.compact"), layout,
        entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: io.x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: io.mask.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: io.out_pos.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:3, resource: io.out_val.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:4, resource: ubuf.as_entire_binding()  },
        ]
    });
    let pl = ensure_compact1_pl(&ctx);
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("st.rankk.enc2") });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("st.rankk.pass.compact") });
        cpass.set_pipeline(pl); cpass.set_bind_group(0, &bg, &[]);
        let gx = (io.cols + 255) / 256;
        cpass.dispatch_workgroups(gx.max(1), io.rows.max(1), 1);
    }
    ctx.queue.submit(Some(enc.finish())); Ok(())
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_compaction_2ce(plan:&crate::ops::rank_entry::RankPlan, kind:u32) -> Result<(), String> {
    let ctx = ctx()?; let b = binder()?;
    let io = b.bind_compaction(plan)?;
    let params = CParams{ rows: io.rows, cols: io.cols, row_stride: io.row_stride, kind };
    let ubuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("st.rankk.cparams"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let layout = ensure_layout_cmp(&ctx);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("st.rankk.bg.compact"), layout,
        entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: io.x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: io.mask.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: io.out_pos.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:3, resource: io.out_val.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:4, resource: ubuf.as_entire_binding()  },
        ]
    });
    let scan_pl  = ensure_scan_pl(&ctx);
    let apply_pl = ensure_apply_pl(&ctx);
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("st.rankk.enc3") });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("st.rankk.pass.scan") });
        p.set_pipeline(scan_pl);  p.set_bind_group(0, &bg, &[]);
        let gx = (io.cols + 255) / 256; p.dispatch_workgroups(gx.max(1), io.rows.max(1), 1);
    }
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("st.rankk.pass.apply") });
        p.set_pipeline(apply_pl); p.set_bind_group(0, &bg, &[]);
        let gx = (io.cols + 255) / 256; p.dispatch_workgroups(gx.max(1), io.rows.max(1), 1);
    }
    ctx.queue.submit(Some(enc.finish())); Ok(())
}

// --- helper to re-export DeviceExt ---
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
mod util_reexp { pub use wgpu::util::DeviceExt; }
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
use util_reexp::*;
