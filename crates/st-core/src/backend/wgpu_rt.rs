//! WGPU runtime (v1.8.0): TopK fast + Mid/Bottom improved, LinOps
#![allow(unused)]
use once_cell::sync::OnceCell;
use std::sync::Arc;

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub struct WgpuCtx {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    // rank-k
    topk_sub_pl: OnceCell<wgpu::ComputePipeline>,
    topk_wg_pl:  OnceCell<wgpu::ComputePipeline>,
    scan_tiles_pl:  OnceCell<wgpu::ComputePipeline>,
    row_prefix_pl:  OnceCell<wgpu::ComputePipeline>,
    apply_pl:       OnceCell<wgpu::ComputePipeline>,
    apply_sg_pl:    OnceCell<wgpu::ComputePipeline>,
    layout_topk: OnceCell<wgpu::BindGroupLayout>,
    layout_cmp:  OnceCell<wgpu::BindGroupLayout>,
    // linops
    layout_lin: OnceCell<wgpu::BindGroupLayout>,
    axpy_pl:    OnceCell<wgpu::ComputePipeline>,
    dotp_pl:    OnceCell<wgpu::ComputePipeline>,
    dotf_pl:    OnceCell<wgpu::ComputePipeline>,
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
impl WgpuCtx {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self{
            device, queue,
            topk_sub_pl: OnceCell::new(),
            topk_wg_pl:  OnceCell::new(),
            scan_tiles_pl: OnceCell::new(),
            row_prefix_pl: OnceCell::new(),
            apply_pl: OnceCell::new(),
            apply_sg_pl: OnceCell::new(),
            layout_topk: OnceCell::new(),
            layout_cmp:  OnceCell::new(),
            layout_lin:  OnceCell::new(),
            axpy_pl:     OnceCell::new(),
            dotp_pl:     OnceCell::new(),
            dotf_pl:     OnceCell::new(),
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
    pub prefix: Option<wgpu::Buffer>,
}

pub trait WgpuBinder: Send + Sync + 'static {
    fn bind_topk(&self, plan:&crate::ops::rank_entry::RankPlan) -> Result<KernelIO, String>;
    fn bind_compaction(&self, plan:&crate::ops::rank_entry::RankPlan) -> Result<CompactionIO, String>;
}

static CTX: OnceCell<Arc<WgpuCtx>> = OnceCell::new();
static BINDER: OnceCell<Arc<dyn WgpuBinder>> = OnceCell::new();
pub fn install_ctx(ctx: Arc<WgpuCtx>) { let _ = CTX.set(ctx); }
pub fn install_binder(b: Arc<dyn WgpuBinder>) { let _ = BINDER.set(b); }
fn ctx() -> Result<Arc<WgpuCtx>, String> { CTX.get().cloned().ok_or("WGPU ctx not installed".into()) }
fn binder() -> Result<Arc<dyn WgpuBinder>, String> { BINDER.get().cloned().ok_or("WGPU binder not installed".into()) }

#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params { rows:u32, cols:u32, k:u32, row_stride:u32, k_lane:u32, tile_cols:u32, _pad:u32, _pad2:u32 }
#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CParams { rows:u32, cols:u32, row_stride:u32, kind:u32, tiles_x:u32, _pad:u32, _pad2:u32, _pad3:u32 }
#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LParams { n:u32, _pad:u32, _p2:u32, _p3:u32 }

const WGSL_RANKK: &str = include_str!("wgpu_kernels_rankk.wgsl");
const WGSL_LIN:   &str = include_str!("wgpu_kernels_linops.wgsl");

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn module(ctx:&WgpuCtx, src:&'static str) -> wgpu::ShaderModule {
    ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor{
        label: Some("st.wgsl"), source: wgpu::ShaderSource::Wgsl(src.into())
    })
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
fn ensure_layout_cmp(ctx:&WgpuCtx) -> &wgpu::BindGroupLayout {
    ctx.layout_cmp.get_or_init(|| {
        ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("st.rankk.layout.compact"),
            entries: &[ bge_storage(0,true), bge_storage(1,true), bge_storage(2,false),
                        bge_storage(3,false), bge_uniform(4), bge_storage(5,false) ],
        })
    })
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_layout_lin(ctx:&WgpuCtx) -> &wgpu::BindGroupLayout {
    ctx.layout_lin.get_or_init(|| {
        ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("st.lin.layout"),
            entries: &[ bge_storage(0,true), bge_storage(1,true), bge_storage(2,false), bge_uniform(3) ],
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
fn pl(ctx:&WgpuCtx, entry:&'static str, layout:&wgpu::BindGroupLayout, src:&'static str)->wgpu::ComputePipeline {
    let module = module(ctx, src);
    let pl_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
        label: Some("st.pl"), bind_group_layouts: &[layout], push_constant_ranges: &[]
    });
    ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
        label: Some(entry), layout: Some(&pl_layout), module: &module, entry_point: entry
    })
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_topk_sub_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline {
    ctx.topk_sub_pl.get_or_init(|| pl(ctx, "topk_subgroups_1ce", ensure_layout_topk(ctx), WGSL_RANKK))
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_topk_wg_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline {
    ctx.topk_wg_pl.get_or_init(|| pl(ctx, "topk_workgroup_1ce", ensure_layout_topk(ctx), WGSL_RANKK))
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_scan_tiles_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline {
    ctx.scan_tiles_pl.get_or_init(|| pl(ctx, "midk_compact_scan_tiles", ensure_layout_cmp(ctx), WGSL_RANKK))
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_row_prefix_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline {
    ctx.row_prefix_pl.get_or_init(|| pl(ctx, "midk_compact_row_prefix", ensure_layout_cmp(ctx), WGSL_RANKK))
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_apply_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline {
    ctx.apply_pl.get_or_init(|| pl(ctx, "midk_compact_apply", ensure_layout_cmp(ctx), WGSL_RANKK))
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_apply_sg_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline {
    ctx.apply_sg_pl.get_or_init(|| pl(ctx, "midk_compact_apply_sg", ensure_layout_cmp(ctx), WGSL_RANKK))
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_axpy_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline {
    ctx.axpy_pl.get_or_init(|| pl(ctx, "axpy_inplace", ensure_layout_lin(ctx), WGSL_LIN))
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_dotp_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline {
    ctx.dotp_pl.get_or_init(|| pl(ctx, "dot_partials", ensure_layout_lin(ctx), WGSL_LIN))
}
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn ensure_dotf_pl(ctx:&WgpuCtx)->&wgpu::ComputePipeline {
    ctx.dotf_pl.get_or_init(|| pl(ctx, "dot_finalize", ensure_layout_lin(ctx), WGSL_LIN))
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_topk_1ce(plan:&crate::ops::rank_entry::RankPlan)->Result<(),String>{
    use wgpu::util::DeviceExt; use wgpu::Features;
    let ctx = ctx()?; let b = binder()?; let io = b.bind_topk(plan)?;
    let params = Params{ rows:io.rows, cols:io.cols, k:io.k, row_stride:io.row_stride, k_lane:io.k_lane, tile_cols:io.cols, _pad:0, _pad2:0 };
    let ub = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("st.rankk.params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM
    });
    let layout = ensure_layout_topk(&ctx);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("st.rankk.bg.topk"), layout, entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: io.x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: io.out_vals.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: io.out_idx.as_entire_binding()  },
            wgpu::BindGroupEntry{ binding:3, resource: ub.as_entire_binding() },
        ]
    });
    let use_sub = ctx.device.features().contains(Features::SUBGROUPS);
    let pl = if use_sub { ensure_topk_sub_pl(&ctx) } else { ensure_topk_wg_pl(&ctx) };
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("st.rankk.enc.topk") });
    { let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("st.rankk.pass.topk") });
      c.set_pipeline(pl); c.set_bind_group(0, &bg, &[]);
      c.dispatch_workgroups(1, io.rows.max(1), 1);
    }
    ctx.queue.submit(Some(enc.finish())); Ok(())
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_compaction_2ce(plan:&crate::ops::rank_entry::RankPlan, kind:u32)->Result<(),String>{
    use wgpu::util::DeviceExt; use wgpu::Features;
    let ctx = ctx()?; let b = binder()?; let io = b.bind_compaction(plan)?;
    let tiles_x = (io.cols + 255) / 256;
    let params = CParams{ rows:io.rows, cols:io.cols, row_stride:io.row_stride, kind, tiles_x, _pad:0, _pad2:0, _pad3:0 };
    let ub = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("st.rankk.cparams"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM
    });
    let prefix = match io.prefix {
        Some(p) => p,
        None => ctx.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("st.rankk.prefix.tmp"), size: (io.rows as u64 * tiles_x as u64 * 4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
    };
    let layout = ensure_layout_cmp(&ctx);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("st.rankk.bg.compact"), layout, entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: io.x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: io.mask.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: io.out_pos.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:3, resource: io.out_val.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:4, resource: ub.as_entire_binding()  },
            wgpu::BindGroupEntry{ binding:5, resource: prefix.as_entire_binding()  },
        ]
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("st.rankk.enc.compact2") });
    // scan tiles
    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("scan_tiles") });
      p.set_pipeline(ensure_scan_tiles_pl(&ctx)); p.set_bind_group(0, &bg, &[]);
      p.dispatch_workgroups(tiles_x.max(1), io.rows.max(1), 1);
    }
    // row prefix (parallel chunks)
    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("row_prefix") });
      p.set_pipeline(ensure_row_prefix_pl(&ctx)); p.set_bind_group(0, &bg, &[]);
      p.dispatch_workgroups(1, io.rows.max(1), 1);
    }
    // apply
    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("apply") });
      let use_sub = ctx.device.features().contains(Features::SUBGROUPS);
      let pl = if use_sub { ensure_apply_sg_pl(&ctx) } else { ensure_apply_pl(&ctx) };
      p.set_pipeline(pl); p.set_bind_group(0, &bg, &[]);
      p.dispatch_workgroups(tiles_x.max(1), io.rows.max(1), 1);
    }
    ctx.queue.submit(Some(enc.finish())); Ok(())
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_compaction_1ce(plan:&crate::ops::rank_entry::RankPlan, kind:u32)->Result<(),String>{
    // reuse Blelloch apply as 1CE fallback (unstable if OUTPOS is used as atomic counter elsewhere)
    use wgpu::util::DeviceExt;
    let ctx = ctx()?; let b = binder()?; let io = b.bind_compaction(plan)?;
    let tiles_x = (io.cols + 255) / 256;
    let params = CParams{ rows:io.rows, cols:io.cols, row_stride:io.row_stride, kind, tiles_x, _pad:0, _pad2:0, _pad3:0 };
    let ub = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("st.rankk.cparams"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM
    });
    let prefix = ctx.device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("st.rankk.prefix.dummy"), size: 4, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false
    });
    let layout = ensure_layout_cmp(&ctx);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("st.rankk.bg.compact1"), layout, entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: io.x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: io.mask.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: io.out_pos.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:3, resource: io.out_val.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:4, resource: ub.as_entire_binding()  },
            wgpu::BindGroupEntry{ binding:5, resource: prefix.as_entire_binding()  },
        ]
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("st.rankk.enc.compact1") });
    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("compact1") });
      p.set_pipeline(ensure_apply_pl(&ctx)); p.set_bind_group(0, &bg, &[]);
      p.dispatch_workgroups(tiles_x.max(1), io.rows.max(1), 1);
    }
    ctx.queue.submit(Some(enc.finish())); Ok(())
}

// --------------- LinOps (dot/axpy) ---------------
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub struct LinOpsIO { pub n:u32, pub x:wgpu::Buffer, pub y:wgpu::Buffer, pub z:wgpu::Buffer }

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_axpy(n:u32, alpha:f32, x:&wgpu::Buffer, y:&wgpu::Buffer, z:&wgpu::Buffer)->Result<(),String>{
    use wgpu::util::DeviceExt;
    let ctx = ctx()?;
    // write alpha to z[0] temporarily
    let tmp = vec![alpha];
    ctx.queue.write_buffer(z, 0, bytemuck::cast_slice(&tmp));
    let lp = LParams{ n, _pad:0, _p2:0, _p3:0 };
    let ub = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("st.lin.params"), contents: bytemuck::bytes_of(&lp), usage: wgpu::BufferUsages::UNIFORM
    });
    let layout = ensure_layout_lin(&ctx);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("st.lin.bg.axpy"), layout, entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: y.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: z.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:3, resource: ub.as_entire_binding() },
        ]
    });
    let pl = ensure_axpy_pl(&ctx);
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("st.lin.enc.axpy") });
    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("st.lin.pass.axpy") });
      p.set_pipeline(pl); p.set_bind_group(0, &bg, &[]);
      let gx = (n + 255)/256; p.dispatch_workgroups(gx.max(1), 1, 1);
    }
    ctx.queue.submit(Some(enc.finish())); Ok(())
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
pub fn dispatch_dot(n:u32, x:&wgpu::Buffer, y:&wgpu::Buffer)->Result<(wgpu::Buffer, f32),String>{
    use wgpu::util::DeviceExt;
    let ctx = ctx()?;
    let lp = LParams{ n, _pad:0, _p2:0, _p3:0 };
    let ub = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("st.lin.params"), contents: bytemuck::bytes_of(&lp), usage: wgpu::BufferUsages::UNIFORM
    });
    let layout = ensure_layout_lin(&ctx);
    let gx = (n + 255)/256;
    let partials = ctx.device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("st.lin.partials"), size: gx as u64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation:false
    });
    let bgp = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("st.lin.bg.dotp"), layout, entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: y.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: partials.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:3, resource: ub.as_entire_binding() },
        ]
    });
    // finalize uses the same BG but number of WGs = ceil(gx/256)
    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("st.lin.dot.out"), size: 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation:false
    });
    // We'll reuse `partials` as input and output; copy final sum into `out` with a trivial pass or map from partials[0]
    let plp = ensure_dotp_pl(&ctx);
    let plf = ensure_dotf_pl(&ctx);

    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("st.lin.enc.dot") });
    // pass1
    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("dot.partials") });
      p.set_pipeline(plp); p.set_bind_group(0, &bgp, &[]);
      p.dispatch_workgroups(gx.max(1), 1, 1);
    }
    // pass2 finalize; we reuse same bindgroup; compute dims chosen to cover partials buffer
    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("dot.finalize") });
      p.set_pipeline(plf); p.set_bind_group(0, &bgp, &[]);
      let gx2 = ((gx + 255)/256).max(1);
      p.dispatch_workgroups(gx2, 1, 1);
    }
    // copy result partials[0] -> out[0]
    enc.copy_buffer_to_buffer(&partials, 0, &out, 0, 4);
    ctx.queue.submit(Some(enc.finish()));

    // Map `out` to host to get scalar (control loop uses it)
    // (In a real async runtime we would use callbacks/futures; here we block via map + pollster)
    let buf_slice = out.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buf_slice.map_async(wgpu::MapMode::Read, move |v| { tx.send(v).ok(); });
    pollster::block_on(async { rx.receive().await; });
    let data = buf_slice.get_mapped_range();
    let val = bytemuck::cast_slice::<u8,f32>(&data)[0];
    drop(data); out.unmap();
    Ok((out, val))
}

// reexport DeviceExt
#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
mod util_reexp { pub use wgpu::util::DeviceExt; }
