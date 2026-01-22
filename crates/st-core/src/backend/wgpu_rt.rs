// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-core/src/backend/wgpu_rt.rs  (v1.9.0) — TopK + linear primitives
#![allow(unused)]
use once_cell::sync::OnceCell;
use std::{
    any::Any,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

// ===== SpiralTorch: WGPU runtime hardening additions (non-breaking) =====
#[derive(Debug, thiserror::Error)]
pub enum WgpuRtError {
    #[error("queue submit timed out after {0:?}")]
    SubmitTimeout(std::time::Duration),
    #[error("buffer map failed: {0}")]
    MapFailed(String),
}

/// Submit command buffers and actively poll until completion or timeout.
/// 呼び出し側で既存の `queue.submit(cmd_bufs)` をこれで置換可能。
#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
pub fn st_submit_with_timeout(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cmd_bufs: &[wgpu::CommandBuffer],
    timeout: std::time::Duration,
) -> Result<(), WgpuRtError> {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    queue.submit(cmd_bufs.iter().cloned());
    queue.on_submitted_work_done(move || {
        let _ = tx.send(());
    });

    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        device.poll(wgpu::Maintain::Poll);
        if rx.try_recv().is_ok() {
            return Ok(());
        }
        std::thread::yield_now();
    }
    Err(WgpuRtError::SubmitTimeout(timeout))
}

/// MAP_READ バッファを安全に読み出す（タイムアウト付き）
#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
pub fn st_map_read_with_timeout(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    range: std::ops::Range<u64>,
    timeout: std::time::Duration,
) -> Result<Vec<u8>, WgpuRtError> {
    use std::sync::mpsc::channel;
    let (tx, rx) = channel();
    buffer
        .slice(range.clone())
        .map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
    let start = std::time::Instant::now();
    loop {
        device.poll(wgpu::Maintain::Poll);
        if let Ok(res) = rx.try_recv() {
            res.map_err(|e| WgpuRtError::MapFailed(format!("{e:?}")))?;
            let view = buffer.slice(range.clone()).get_mapped_range();
            let data = view.to_vec();
            drop(view);
            buffer.unmap();
            return Ok(data);
        }
        if start.elapsed() >= timeout {
            return Err(WgpuRtError::SubmitTimeout(timeout));
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
// ===== end additions =====

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub struct WgpuCtx {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    // RankK/TopK pipelines
    topk_heap_pl:   OnceCell<wgpu::ComputePipeline>,
    topk_heap_sgintrin_pl: OnceCell<wgpu::ComputePipeline>,
    topk_bit_pl:    OnceCell<wgpu::ComputePipeline>,
    topk_wg_pl:     OnceCell<wgpu::ComputePipeline>,
    layout_topk:    OnceCell<wgpu::BindGroupLayout>,
    // Linear primitives
    lin_axpy_pl:        OnceCell<wgpu::ComputePipeline>,
    lin_scale_pl:       OnceCell<wgpu::ComputePipeline>,
    lin_copy_pl:        OnceCell<wgpu::ComputePipeline>,
    lin_fill_pl:        OnceCell<wgpu::ComputePipeline>,
    lin_dot_partials_pl:OnceCell<wgpu::ComputePipeline>,
    lin_dot_finalize_pl:OnceCell<wgpu::ComputePipeline>,
    lin_dot_partials_subgroup_pl:OnceCell<wgpu::ComputePipeline>,
    lin_dot_finalize_subgroup_pl:OnceCell<wgpu::ComputePipeline>,
    layout_lin:         OnceCell<wgpu::BindGroupLayout>,
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
            layout_topk:  OnceCell::new(),
            lin_axpy_pl:        OnceCell::new(),
            lin_scale_pl:       OnceCell::new(),
            lin_copy_pl:        OnceCell::new(),
            lin_fill_pl:        OnceCell::new(),
            lin_dot_partials_pl:OnceCell::new(),
            lin_dot_finalize_pl:OnceCell::new(),
            lin_dot_partials_subgroup_pl:OnceCell::new(),
            lin_dot_finalize_subgroup_pl:OnceCell::new(),
            layout_lin:         OnceCell::new(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    rows:u32, cols:u32, k:u32, row_stride:u32,
    k_lane:u32, tile_cols:u32, radix:u32, segments:u32
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LinParams {
    dims: [u32;4],
    scalars: [f32;4],
}

static CTX: OnceCell<Arc<WgpuCtx>> = OnceCell::new();
pub fn install_ctx(ctx: Arc<WgpuCtx>) { let _ = CTX.set(ctx); }
fn ctx() -> Result<Arc<WgpuCtx>, String> { CTX.get().cloned().ok_or("WGPU ctx not installed".into()) }

const WGSL_RANKK: &str = include_str!("wgpu_kernels_rankk.wgsl");
const WGSL_LINOPS: &str = include_str!("wgpu_kernels_linops.wgsl");
const WGSL_LINOPS_SUBGROUP: &str = include_str!("wgpu_kernels_linops_subgroup.wgsl");

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn module(ctx: &WgpuCtx, label: &str, src: &'static str) -> Result<wgpu::ShaderModule, String> {
    catch_unwind(AssertUnwindSafe(|| {
        ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        })
    }))
    .map_err(|payload| panic_payload_to_string(payload))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_layout_topk(ctx:&WgpuCtx) -> &wgpu::BindGroupLayout {
    ctx.layout_topk.get_or_init(|| {
        ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("st.rankk.layout.topk"),
            entries: &[
                bge_storage(0, true),
                bge_storage(1, false),
                bge_storage(2, false),
                bge_uniform(3),
            ],
        })
    })
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn ensure_layout_lin(ctx:&WgpuCtx) -> &wgpu::BindGroupLayout {
    ctx.layout_lin.get_or_init(|| {
        ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("st.lin.layout"),
            entries: &[
                bge_storage(0, true),
                bge_storage(1, true),
                bge_storage_rw(2),
                bge_uniform(3),
            ],
        })
    })
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn bge_storage(binding:u32, read_only:bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry{
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Storage{ read_only }, has_dynamic_offset:false, min_binding_size: None },
        count: None,
    }
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn bge_storage_rw(binding:u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry{
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Storage{ read_only:false }, has_dynamic_offset:false, min_binding_size: None },
        count: None,
    }
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn bge_uniform(binding:u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry{
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer{ ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset:false, min_binding_size: None },
        count: None,
    }
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn pl(
    ctx: &WgpuCtx,
    entry: &'static str,
    layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::ComputePipeline, String> {
    let module = module(ctx, "st.rankk", WGSL_RANKK)?;
    let pl_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
        label: Some("st.rankk.pl"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    catch_unwind(AssertUnwindSafe(|| {
        ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some(entry),
            layout: Some(&pl_layout),
            module: &module,
            entry_point: entry,
            compilation_options: Default::default(),
        })
    }))
    .map_err(|payload| panic_payload_to_string(payload))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn lin_pl(
    ctx: &WgpuCtx,
    entry: &'static str,
    layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::ComputePipeline, String> {
    let module = module(ctx, "st.lin", WGSL_LINOPS)?;
    let pl_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
        label: Some("st.lin.pl"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    catch_unwind(AssertUnwindSafe(|| {
        ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some(entry),
            layout: Some(&pl_layout),
            module: &module,
            entry_point: entry,
            compilation_options: Default::default(),
        })
    }))
    .map_err(|payload| panic_payload_to_string(payload))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn lin_subgroup_pl(
    ctx: &WgpuCtx,
    entry: &'static str,
    layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::ComputePipeline, String> {
    let module = module(ctx, "st.lin.subgroup", WGSL_LINOPS_SUBGROUP)?;
    let pl_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
        label: Some("st.lin.pl.subgroup"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    catch_unwind(AssertUnwindSafe(|| {
        ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some(entry),
            layout: Some(&pl_layout),
            module: &module,
            entry_point: entry,
            compilation_options: Default::default(),
        })
    }))
    .map_err(|payload| panic_payload_to_string(payload))
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_topk_heap_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.topk_heap_pl
        .get_or_try_init(|| pl(ctx, "topk_subgroups_heap_1ce", ensure_layout_topk(ctx)))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_topk_heap_sgintrin_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.topk_heap_sgintrin_pl.get_or_try_init(|| {
        pl(ctx, "topk_subgroups_heap_sgintrin_1ce", ensure_layout_topk(ctx))
    })
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_topk_bit_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.topk_bit_pl
        .get_or_try_init(|| pl(ctx, "topk_subgroups_bitonic_1ce", ensure_layout_topk(ctx)))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_topk_wg_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.topk_wg_pl
        .get_or_try_init(|| pl(ctx, "topk_workgroup_1ce", ensure_layout_topk(ctx)))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_lin_axpy_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.lin_axpy_pl
        .get_or_try_init(|| lin_pl(ctx, "axpy_inplace", ensure_layout_lin(ctx)))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_lin_scale_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.lin_scale_pl
        .get_or_try_init(|| lin_pl(ctx, "scale_inplace", ensure_layout_lin(ctx)))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_lin_copy_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.lin_copy_pl
        .get_or_try_init(|| lin_pl(ctx, "copy_vec", ensure_layout_lin(ctx)))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_lin_fill_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.lin_fill_pl
        .get_or_try_init(|| lin_pl(ctx, "fill_vec", ensure_layout_lin(ctx)))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_lin_dot_partials_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.lin_dot_partials_pl
        .get_or_try_init(|| lin_pl(ctx, "dot_partials", ensure_layout_lin(ctx)))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_lin_dot_finalize_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.lin_dot_finalize_pl
        .get_or_try_init(|| lin_pl(ctx, "dot_finalize", ensure_layout_lin(ctx)))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_lin_dot_partials_subgroup_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.lin_dot_partials_subgroup_pl.get_or_try_init(|| {
        lin_subgroup_pl(ctx, "dot_partials_subgroup", ensure_layout_lin(ctx))
    })
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_lin_dot_finalize_subgroup_pl(ctx: &WgpuCtx) -> Result<&wgpu::ComputePipeline, String> {
    ctx.lin_dot_finalize_subgroup_pl.get_or_try_init(|| {
        lin_subgroup_pl(ctx, "dot_finalize_subgroup", ensure_layout_lin(ctx))
    })
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_topk_1ce(
    rows:u32,
    cols:u32,
    k:u32,
    row_stride:u32,
    k_lane:u32,
    x:&wgpu::Buffer,
    out_vals:&wgpu::Buffer,
    out_idx:&wgpu::Buffer
) -> Result<(),String> {
    use wgpu::util::DeviceExt;
    use wgpu::Features;

    let ctx = ctx()?;
    let mut tile_cols = cols;
    let mut radix_hint: u32 = if k.is_power_of_two() { 4 } else { 2 };
    let mut segments_hint: u32 = if cols > 131_072 { 4 } else if cols > 32_768 { 2 } else { 1 };

    let has_sub = ctx.device.features().contains(Features::SUBGROUPS);
    let mut algo_hint: u8 = 0;
    let mut ctile_hint: u32 = 0;
    if let Some(ch) = crate::backend::wgpu_heuristics::choose_topk(rows, cols, k, has_sub) {
        algo_hint = ch.algo_topk;
        ctile_hint = ch.ctile;
        if ch.tile_cols != 0 { tile_cols = ch.tile_cols; }
        radix_hint = ch.radix.max(1);
        segments_hint = ch.segments.max(1);
    }
    if ctile_hint != 0 { tile_cols = ctile_hint; }

    let params = Params{ rows, cols, k, row_stride, k_lane, tile_cols, radix: radix_hint, segments: segments_hint };
    let ub = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("st.rankk.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = ensure_layout_topk(&ctx);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("st.rankk.bg.topk"),
        layout,
        entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: out_vals.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: out_idx.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:3, resource: ub.as_entire_binding() },
        ],
    });

    let sgc_hint = if has_sub { 8 } else { 1 };
    let prefer_intrin = std::env::var("ST_USE_SG_INTRIN").ok().as_deref() == Some("1");
    let prefer_heap_default = (k <= 32 && cols >= 2048) || (k <= 16 && sgc_hint >= 8);
    let prefer_heap = match algo_hint { 1 => true, 2 => false, _ => prefer_heap_default };
    let pipeline = if has_sub {
        if prefer_heap {
            if prefer_intrin {
                ensure_topk_heap_sgintrin_pl(&ctx)?
            } else {
                ensure_topk_heap_pl(&ctx)?
            }
        } else {
            ensure_topk_bit_pl(&ctx)?
        }
    } else {
        ensure_topk_wg_pl(&ctx)?
    };

    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("st.rankk.enc.topk") });
    {
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("st.rankk.pass.topk") });
        c.set_pipeline(pipeline);
        c.set_bind_group(0, &bg, &[]);
        c.dispatch_workgroups(1, rows.max(1), 1);
    }
    let cmd = enc.finish();
    let cmd_bufs = [cmd];
    st_submit_with_timeout(
        &ctx.device,
        &ctx.queue,
        &cmd_bufs,
        std::time::Duration::from_secs(30),
    )
    .map_err(|e| e.to_string())
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn div_ceil_u32(x:u32, d:u32) -> u32 {
    if x == 0 { 0 } else { (x + d - 1) / d }
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn create_lin_params_buffer(ctx:&WgpuCtx, params:LinParams, label:&str) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some(label),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn create_lin_bind_group(
    ctx:&WgpuCtx,
    layout:&wgpu::BindGroupLayout,
    x:&wgpu::Buffer,
    y:&wgpu::Buffer,
    z:&wgpu::Buffer,
    params:&wgpu::Buffer
) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("st.lin.bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: y.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: z.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:3, resource: params.as_entire_binding() },
        ],
    })
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn dispatch_lin_kernel(
    ctx:&WgpuCtx,
    pipeline:&wgpu::ComputePipeline,
    bind_group:&wgpu::BindGroup,
    workgroups:u32,
    label:&str
) -> Result<(), String> {
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some(label) });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("st.lin.pass") });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        if workgroups > 0 {
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
    let cmd = enc.finish();
    let cmd_bufs = [cmd];
    st_submit_with_timeout(
        &ctx.device,
        &ctx.queue,
        &cmd_bufs,
        std::time::Duration::from_secs(30),
    )
    .map_err(|e| e.to_string())
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_lin_axpy(n:u32, alpha:f32, x:&wgpu::Buffer, y:&wgpu::Buffer, out:&wgpu::Buffer) -> Result<(), String> {
    let ctx = ctx()?;
    let params = LinParams{ dims:[n, 0, 0, 0], scalars:[alpha, 0.0, 0.0, 0.0] };
    let ub = create_lin_params_buffer(&ctx, params, "st.lin.params.axpy");
    let layout = ensure_layout_lin(&ctx);
    let bg = create_lin_bind_group(&ctx, layout, x, y, out, &ub);
    let wg = div_ceil_u32(n, 256);
    let pipeline = ensure_lin_axpy_pl(&ctx)?;
    dispatch_lin_kernel(&ctx, pipeline, &bg, wg, "st.lin.axpy")
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_lin_scale(n:u32, scale:f32, y:&wgpu::Buffer, out:&wgpu::Buffer) -> Result<(), String> {
    let ctx = ctx()?;
    let params = LinParams{ dims:[n, 0, 0, 0], scalars:[scale, 0.0, 0.0, 0.0] };
    let ub = create_lin_params_buffer(&ctx, params, "st.lin.params.scale");
    let layout = ensure_layout_lin(&ctx);
    let bg = create_lin_bind_group(&ctx, layout, out, y, out, &ub);
    let wg = div_ceil_u32(n, 256);
    let pipeline = ensure_lin_scale_pl(&ctx)?;
    dispatch_lin_kernel(&ctx, pipeline, &bg, wg, "st.lin.scale")
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_lin_copy(n:u32, src:&wgpu::Buffer, dst:&wgpu::Buffer) -> Result<(), String> {
    let ctx = ctx()?;
    let params = LinParams{ dims:[n, 0, 0, 0], scalars:[0.0, 0.0, 0.0, 0.0] };
    let ub = create_lin_params_buffer(&ctx, params, "st.lin.params.copy");
    let layout = ensure_layout_lin(&ctx);
    let bg = create_lin_bind_group(&ctx, layout, src, dst, dst, &ub);
    let wg = div_ceil_u32(n, 256);
    let pipeline = ensure_lin_copy_pl(&ctx)?;
    dispatch_lin_kernel(&ctx, pipeline, &bg, wg, "st.lin.copy")
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_lin_fill(n:u32, value:f32, dst:&wgpu::Buffer) -> Result<(), String> {
    let ctx = ctx()?;
    let params = LinParams{ dims:[n, 0, 0, 0], scalars:[0.0, value, 0.0, 0.0] };
    let ub = create_lin_params_buffer(&ctx, params, "st.lin.params.fill");
    let layout = ensure_layout_lin(&ctx);
    let bg = create_lin_bind_group(&ctx, layout, dst, dst, dst, &ub);
    let wg = div_ceil_u32(n, 256);
    let pipeline = ensure_lin_fill_pl(&ctx)?;
    dispatch_lin_kernel(&ctx, pipeline, &bg, wg, "st.lin.fill")
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_lin_dot(n:u32, x:&wgpu::Buffer, y:&wgpu::Buffer, scratch:&wgpu::Buffer) -> Result<(), String> {
    let ctx = ctx()?;
    let partials = div_ceil_u32(n, 256);
    let has_subgroups = {
        use wgpu::Features;
        ctx.device.features().contains(Features::SUBGROUPS)
    };

    let params_partials = LinParams{ dims:[n, 0, 0, 0], scalars:[0.0, 0.0, 0.0, 0.0] };
    let ub_partials = create_lin_params_buffer(&ctx, params_partials, "st.lin.params.dot.partials");
    let layout = ensure_layout_lin(&ctx);
    let bg_partials = create_lin_bind_group(&ctx, layout, x, y, scratch, &ub_partials);
    let pipeline_partials = if has_subgroups {
        ensure_lin_dot_partials_subgroup_pl(&ctx)?
    } else {
        ensure_lin_dot_partials_pl(&ctx)?
    };
    let label_partials = if has_subgroups {
        "st.lin.dot.partials.sg"
    } else {
        "st.lin.dot.partials"
    };
    dispatch_lin_kernel(&ctx, pipeline_partials, &bg_partials, partials, label_partials)?;

    let params_finalize = LinParams{ dims:[n, partials, 0, 0], scalars:[0.0, 0.0, 0.0, 0.0] };
    let ub_finalize = create_lin_params_buffer(&ctx, params_finalize, "st.lin.params.dot.finalize");
    let bg_finalize = create_lin_bind_group(&ctx, layout, scratch, scratch, scratch, &ub_finalize);
    let pipeline_finalize = if has_subgroups {
        ensure_lin_dot_finalize_subgroup_pl(&ctx)?
    } else {
        ensure_lin_dot_finalize_pl(&ctx)?
    };
    let wg_finalize = div_ceil_u32(partials, 256).max(1);
    let label_finalize = if has_subgroups {
        "st.lin.dot.finalize.sg"
    } else {
        "st.lin.dot.finalize"
    };
    dispatch_lin_kernel(&ctx, pipeline_finalize, &bg_finalize, wg_finalize, label_finalize)
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub mod util_reexp {
    pub use wgpu::{BindGroup, BindGroupLayout, Buffer, CommandEncoder, ComputePipeline, Device, Queue};
    pub use wgpu::util::DeviceExt;
}

#[cfg(not(all(feature="wgpu", feature="wgpu-rt")))]
pub mod util_reexp {}
