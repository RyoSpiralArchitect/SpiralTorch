// crates/st-core/src/backend/wgpu_rt.rs  (v1.8.7) — excerpted TopK dispatch
#![allow(unused)]
use once_cell::sync::OnceCell;
use std::sync::Arc;

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub struct WgpuCtx {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    topk_heap_pl:   OnceCell<wgpu::ComputePipeline>,
    topk_heap_sgintrin_pl: OnceCell<wgpu::ComputePipeline>,
    topk_bit_pl:    OnceCell<wgpu::ComputePipeline>,
    topk_wg_pl:     OnceCell<wgpu::ComputePipeline>,
    layout_topk: OnceCell<wgpu::BindGroupLayout>,
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
impl WgpuCtx {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self{
            device, queue,
            topk_heap_pl: OnceCell::new(),
            topk_heap_sgintrin_pl: OnceCell::new(),
            topk_bit_pl:  OnceCell::new(),
            topk_wg_pl:   OnceCell::new(),
            layout_topk: OnceCell::new(),
        }
    }
}

#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params { rows:u32, cols:u32, k:u32, row_stride:u32, k_lane:u32, tile_cols:u32, _pad:u32, _pad2:u32 }

static CTX: OnceCell<Arc<WgpuCtx>> = OnceCell::new();
pub fn install_ctx(ctx: Arc<WgpuCtx>) { let _ = CTX.set(ctx); }
fn ctx() -> Result<Arc<WgpuCtx>, String> { CTX.get().cloned().ok_or("WGPU ctx not installed".into()) }
const WGSL_RANKK: &str = include_str!("wgpu_kernels_rankk.wgsl");

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn module(ctx:&WgpuCtx, src:&'static str) -> wgpu::ShaderModule {
    ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor{ label: Some("st.wgsl"), source: wgpu::ShaderSource::Wgsl(src.into()) })
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_layout_topk(ctx:&WgpuCtx) -> &wgpu::BindGroupLayout {
    ctx.layout_topk.get_or_init(|| {
        ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("st.rankk.layout.topk"),
            entries: &[ bge_storage(0, true), bge_storage(1, false), bge_storage(2, false), bge_uniform(3) ],
        })
    })
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn bge_storage(binding:u32, read_only:bool)->wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry{
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Storage{ read_only }, has_dynamic_offset:false, min_binding_size: None },
        count: None
    }
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn bge_uniform(binding:u32)->wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry{
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset:false, min_binding_size: None },
        count: None
    }
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn pl(ctx:&WgpuCtx, entry:&'static str, layout:&wgpu::BindGroupLayout)->wgpu::ComputePipeline {
    let module = module(ctx, WGSL_RANKK);
    let pl_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{ label: Some("st.pl"), bind_group_layouts: &[layout], push_constant_ranges: &[] });
    ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{ label: Some(entry), layout: Some(&pl_layout), module: &module, entry_point: entry })
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_topk_heap_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline { ctx.topk_heap_pl.get_or_init(|| pl(ctx, "topk_subgroups_heap_1ce", ensure_layout_topk(ctx))) }
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_topk_heap_sgintrin_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline { ctx.topk_heap_sgintrin_pl.get_or_init(|| pl(ctx, "topk_subgroups_heap_sgintrin_1ce", ensure_layout_topk(ctx))) }
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_topk_bit_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline { ctx.topk_bit_pl.get_or_init(|| pl(ctx, "topk_subgroups_bitonic_1ce", ensure_layout_topk(ctx))) }
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_topk_wg_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline { ctx.topk_wg_pl.get_or_init(|| pl(ctx, "topk_workgroup_1ce", ensure_layout_topk(ctx))) }

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_topk_1ce(rows:u32, cols:u32, k:u32, row_stride:u32, k_lane:u32, x:&wgpu::Buffer, out_vals:&wgpu::Buffer, out_idx:&wgpu::Buffer)->Result<(),String>{
    use wgpu::util::DeviceExt; use wgpu::Features;
    let ctx = ctx()?;
    let mut tile_cols = cols;
    // SpiralK/Tuner choice for TopK
    let has_sub = ctx.device.features().contains(Features::SUBGROUPS);
    let mut algo_hint: u8 = 0;
    let mut ctile_hint: u32 = 0;
    if let Some(ch) = crate::backend::wgpu_heuristics::choose_topk(rows, cols, k, has_sub) {
        algo_hint = ch.algo_topk; ctile_hint = ch.ctile;
    }
    if ctile_hint != 0 { tile_cols = ctile_hint; }

    let params = Params{ rows, cols, k, row_stride, k_lane, tile_cols, _pad:0, _pad2:0 };
    let ub = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label: Some("st.rankk.params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });
    let layout = ensure_layout_topk(&ctx);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{ label: Some("st.rankk.bg.topk"), layout, entries: &[
        wgpu::BindGroupEntry{ binding:0, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry{ binding:1, resource: out_vals.as_entire_binding() },
        wgpu::BindGroupEntry{ binding:2, resource: out_idx.as_entire_binding()  },
        wgpu::BindGroupEntry{ binding:3, resource: ub.as_entire_binding() },
    ]});
    let sgc_hint = if has_sub { 8 } else { 1 };
    let prefer_intrin = std::env::var("ST_USE_SG_INTRIN").ok().as_deref()==Some("1");
    let prefer_heap_default = (k <= 32 && cols >= 2048) || (k <= 16 && sgc_hint >= 8);
    let prefer_heap = match algo_hint {
        1 => true, 2 => false, _ => prefer_heap_default
    };
    let pl =
        if has_sub {
            if prefer_heap {
                if prefer_intrin { ensure_topk_heap_sgintrin_pl(&ctx) } else { ensure_topk_heap_pl(&ctx) }
            } else { ensure_topk_bit_pl(&ctx) }
        } else { ensure_topk_wg_pl(&ctx) };

    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("st.rankk.enc.topk") });
    { let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("st.rankk.pass.topk") });
      c.set_pipeline(pl); c.set_bind_group(0, &bg, &[]);
      c.dispatch_workgroups(1, rows.max(1), 1);
    }
    ctx.queue.submit(Some(enc.finish())); Ok(())
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn bge_storage(binding:u32, read_only:bool)->wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry{
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Storage{ read_only }, has_dynamic_offset:false, min_binding_size: None },
        count: None
    }
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn bge_uniform(binding:u32)->wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry{
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset:false, min_binding_size: None },
        count: None
    }
}
