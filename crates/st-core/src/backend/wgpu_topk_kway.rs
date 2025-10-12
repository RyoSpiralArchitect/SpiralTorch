use once_cell::sync::OnceCell;
use wgpu::util::DeviceExt;
use crate::error::{Result, device as dev_err};

#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Meta { rows:u32, cols:u32, k:u32, k_lane:u32, chunk_cols:u32, cand_cols:u32 }

struct Ctx {
    device: wgpu::Device, queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    p1c128: wgpu::ComputePipeline, p1c256: wgpu::ComputePipeline,
    pp1_128: wgpu::ComputePipeline, pp1_256: wgpu::ComputePipeline,
    pp2_128: wgpu::ComputePipeline, pp2_256: wgpu::ComputePipeline,
}
static CTX: OnceCell<Ctx> = OnceCell::new();

fn ctx()->&'static Ctx {
    CTX.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("wgpu adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{
            label: Some("st-wgpu-topk-kway"),
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::downlevel_defaults()
        }, None)).expect("wgpu device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("st-topk-kway"),
            source: wgpu::ShaderSource::Wgsl([
                crate::backend::WGSL_BASE,
                include_str!("wgpu_topk_kway.wgsl")
            ].join("\n").into())
        });
        let p1c128 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("topk-kway-1ce-128"), layout: None, module: &shader, entry_point: "topk_kway_1ce_128"
        });
        let p1c256 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("topk-kway-1ce-256"), layout: None, module: &shader, entry_point: "topk_kway_1ce_256"
        });
        let pp1_128 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("topk-kway-pass1-128"), layout: None, module: &shader, entry_point: "topk_kway_pass1_128"
        });
        let pp1_256 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("topk-kway-pass1-256"), layout: None, module: &shader, entry_point: "topk_kway_pass1_256"
        });
        let pp2_128 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("topk-kway-pass2-128"), layout: None, module: &shader, entry_point: "topk_kway_pass2_128"
        });
        let pp2_256 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("topk-kway-pass2-256"), layout: None, module: &shader, entry_point: "topk_kway_pass2_256"
        });
        Ctx{ device, queue, adapter, p1c128, p1c256, pp1_128, pp1_256, pp2_128, pp2_256 }
    })
}

fn sbuf_f32(data:&[f32], label:&str)->wgpu::Buffer {
    ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some(label), contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST
    })
}
fn sbuf_i32(len:usize, label:&str)->wgpu::Buffer {
    ctx().device.create_buffer(&wgpu::BufferDescriptor{
        label: Some(label), size: (len*4) as u64, usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false
    })
}
fn ubuf<T: bytemuck::Pod>(v:&T, label:&str)->wgpu::Buffer {
    ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some(label), contents: bytemuck::bytes_of(v),
        usage: wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST
    })
}
fn readback_f32(buf:&wgpu::Buffer, len:usize)->Vec<f32>{
    let rb = ctx().device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("rbf32"), size: (len*4) as u64, usage: wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST, mapped_at_creation: false
    });
    let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("rb-enc") });
    e.copy_buffer_to_buffer(buf, 0, &rb, 0, (len*4) as u64);
    ctx().queue.submit(std::iter::once(e.finish()));
    let s = rb.slice(..); let _ = s.map_async(wgpu::MapMode::Read); ctx().device.poll(wgpu::Maintain::Wait);
    let d = s.get_mapped_range(); let v = bytemuck::cast_slice::<u8,f32>(&d).to_vec(); drop(d); rb.unmap(); v
}
fn readback_i32(buf:&wgpu::Buffer, len:usize)->Vec<i32>{
    let rb = ctx().device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("rbi32"), size: (len*4) as u64, usage: wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST, mapped_at_creation: false
    });
    let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("rb-enc") });
    e.copy_buffer_to_buffer(buf, 0, &rb, 0, (len*4) as u64);
    ctx().queue.submit(std::iter::once(e.finish()));
    let s = rb.slice(..); let _ = s.map_async(wgpu::MapMode::Read); ctx().device.poll(wgpu::Maintain::Wait);
    let d = s.get_mapped_range(); let v = bytemuck::cast_slice::<u8,i32>(&d).to_vec(); drop(d); rb.unmap(); v
}

fn heuristic_choose(rows:u32, cols:u32, k:u32) -> (bool, u32, u32, u32) {
    // subgroup detection left as bool hook for tuner or kdsl
    let subgroup = false;
    if let Some((u2, wg, kl, ch)) = crate::backend::wgpu_heuristics::choose(rows, cols, k, subgroup) {
        return (u2, wg, kl, ch);
    }
    // fallback heuristic
    let use_2ce = cols > 32768 || k > 128;
    let wg = if cols < 4096 { 128 } else { 256 };
    let k_lane = if k >= 32 { 32 } else if k >= 16 { 16 } else { 8 };
    let chunk_cols = if cols > 16384 { 8192 } else { 0 };
    (use_2ce, wg, k_lane, chunk_cols)
}

pub fn topk_kway_2d_autotuned(x:&[f32], rows:u32, cols:u32, k:u32) -> Result<(Vec<f32>, Vec<i32>)> {
    if k==0 || k>cols { return Err(dev_err("topk: invalid k")); }
    let (use_2ce, wg, k_lane, chunk_cols) = heuristic_choose(rows, cols, k);
    let outv_len = (rows*k) as usize; let outi_len = outv_len;
    let b_x = sbuf_f32(x, "x");
    let b_outv = sbuf_i32(outv_len, "outv");
    let b_outi = sbuf_i32(outi_len, "outi");
    let b_meta = ubuf(&Meta{ rows, cols, k, k_lane, chunk_cols, cand_cols: (wg*k_lane) }, "meta");

    let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("topk-enc") });
    if !use_2ce {
        let p = if wg==128 { &ctx().p1c128 } else { &ctx().p1c256 };
        let bind = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("bind-1ce"), layout: &p.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry{ binding:0, resource: b_x.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:1, resource: b_outv.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:2, resource: b_outi.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:3, resource: b_meta.as_entire_binding() },
            ]
        });
        { let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("topk-1ce") });
          pass.set_pipeline(p); pass.set_bind_group(0, &bind, &[]);
          pass.dispatch_workgroups(rows, 1, 1);
        }
    } else {
        let cand_len = (rows * wg * k_lane) as usize;
        let b_candv = ctx().device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("candv"), size: (cand_len*4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false
        });
        let b_candi = ctx().device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("candi"), size: (cand_len*4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false
        });
        let p1 = if wg==128 { &ctx().pp1_128 } else { &ctx().pp1_256 };
        let p2 = if wg==128 { &ctx().pp2_128 } else { &ctx().pp2_256 };
        let bind_p1 = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("bind-p1"), layout: &p1.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry{ binding:0, resource: b_x.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:1, resource: b_outv.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:2, resource: b_outi.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:3, resource: b_meta.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:4, resource: b_candv.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:5, resource: b_candi.as_entire_binding() },
            ]
        });
        { let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("topk-2ce-pass1") });
          pass.set_pipeline(p1); pass.set_bind_group(0, &bind_p1, &[]);
          pass.dispatch_workgroups(rows, 1, 1);
        }
        let bind_p2 = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("bind-p2"), layout: &p2.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry{ binding:0, resource: b_x.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:1, resource: b_outv.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:2, resource: b_outi.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:3, resource: b_meta.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:4, resource: b_candv.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:5, resource: b_candi.as_entire_binding() },
            ]
        });
        { let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("topk-2ce-pass2") });
          pass.set_pipeline(p2); pass.set_bind_group(0, &bind_p2, &[]);
          pass.dispatch_workgroups(rows, 1, 1);
        }
    }
    ctx().queue.submit(std::iter::once(e.finish()));
    let vals = readback_f32(&b_outv, outv_len);
    let idxs = readback_i32(&b_outi, outi_len);
    Ok((vals, idxs))
}
