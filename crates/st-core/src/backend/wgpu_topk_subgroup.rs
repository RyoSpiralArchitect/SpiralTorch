use once_cell::sync::OnceCell;
use wgpu::util::DeviceExt;
use crate::error::{Result, device as dev_err};

#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Meta { rows:u32, cols:u32, k:u32, k_lane:u32, chunk_cols:u32, cand_cols:u32 }

pub struct SubgroupCtx {
    device: wgpu::Device, queue: wgpu::Queue, adapter: wgpu::Adapter,
    p_1ce: Option<wgpu::ComputePipeline>,
}
static CTX: OnceCell<SubgroupCtx> = OnceCell::new();

fn ctx()->&'static SubgroupCtx {
    CTX.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("wgpu adapter");
        let features = adapter.features();
        let use_subgroup = features.contains(wgpu::Features::SUBGROUPS);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{
            label: Some("st-wgpu-topk-subgroup"),
            features: if use_subgroup { wgpu::Features::SUBGROUPS } else { wgpu::Features::empty() },
            limits: wgpu::Limits::downlevel_defaults()
        }, None)).expect("wgpu device");
        let mut p_1ce = None;
        if use_subgroup {
            let code = [
                "enable subgroups;",
                include_str!("wgpu_topk_kway.wgsl")
            ].join("\n");
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
                label: Some("st-topk-subgroup"), source: wgpu::ShaderSource::Wgsl(code.into())
            });
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
                label: Some("topk-subgroup-1ce"), layout: None, module: &shader, entry_point: "topk_kway_1ce_256"
            });
            p_1ce = Some(p);
        }
        SubgroupCtx{ device, queue, adapter, p_1ce }
    })
}

pub fn available() -> bool { ctx().p_1ce.is_some() }

pub fn topk_kway_2d_subgroup(x:&[f32], rows:u32, cols:u32, k:u32) -> Result<(Vec<f32>, Vec<i32>)> {
    if !available() { return Err(dev_err("subgroup not available")); }
    let (use_2ce, wg, k_lane, chunk_cols) = {
        let sg = true;
        if let Some(ch) = crate::backend::wgpu_heuristics::choose(rows as usize, cols as usize, k as usize, sg) {
            (ch.use_2ce, ch.wg, ch.kl, ch.ch)
        } else { (false, 256, if k>=32 {32} else if k>=16 {16} else {8}, 0) }
    };
    let outv_len = (rows*k) as usize; let outi_len = outv_len;
    let b_x = ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("x"), contents: bytemuck::cast_slice(x), usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST
    });
    let b_outv = ctx().device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("outv"), size: (outv_len*4) as u64, usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC, mapped_at_creation:false
    });
    let b_outi = ctx().device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("outi"), size: (outi_len*4) as u64, usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC, mapped_at_creation:false
    });
    let meta = Meta{ rows, cols, k, k_lane, chunk_cols, cand_cols: 256*k_lane };
    let b_meta = ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("meta"), contents: bytemuck::bytes_of(&meta), usage: wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST
    });
    let p = ctx().p_1ce.as_ref().unwrap();
    let bind = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("bind-sg"), layout: &p.get_bind_group_layout(0), entries:&[
            wgpu::BindGroupEntry{ binding:0, resource: b_x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: b_outv.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: b_outi.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:3, resource: b_meta.as_entire_binding() },
        ]
    });
    let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("enc") });
    { let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("pass") });
      pass.set_pipeline(p); pass.set_bind_group(0, &bind, &[]); pass.dispatch_workgroups(rows,1,1); }
    ctx().queue.submit(std::iter::once(e.finish()));
    // RB
    let rbv = ctx().device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("rbv"), size: (outv_len*4) as u64, usage: wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST, mapped_at_creation:false
    });
    let rbi = ctx().device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("rbi"), size: (outi_len*4) as u64, usage: wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST, mapped_at_creation:false
    });
    let mut e2 = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("rb") });
    e2.copy_buffer_to_buffer(&b_outv, 0, &rbv, 0, (outv_len*4) as u64);
    e2.copy_buffer_to_buffer(&b_outi, 0, &rbi, 0, (outi_len*4) as u64);
    ctx().queue.submit(std::iter::once(e2.finish()));
    let s = rbv.slice(..); let _=s.map_async(wgpu::MapMode::Read); ctx().device.poll(wgpu::Maintain::Wait);
    let d = s.get_mapped_range(); let vals = bytemuck::cast_slice::<u8,f32>(&d).to_vec(); drop(d); rbv.unmap();
    let s2 = rbi.slice(..); let _=s2.map_async(wgpu::MapMode::Read); ctx().device.poll(wgpu::Maintain::Wait);
    let d2 = s2.get_mapped_range(); let idxs = bytemuck::cast_slice::<u8,i32>(&d2).to_vec(); drop(d2); rbi.unmap();
    Ok((vals, idxs))
}
