
use once_cell::sync::OnceCell;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use crate::error::Result;

#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct RC { rows:u32, cols:u32, k:u32, neg_inf:f32 }

pub struct WgpuMaskIndices;
impl WgpuMaskIndices { pub fn new()->Self{ WgpuMaskIndices } }

struct Ctx { device: wgpu::Device, queue: wgpu::Queue, p: wgpu::ComputePipeline }
static CTX: OnceCell<Ctx> = OnceCell::new();
fn ctx()->&'static Ctx {
    CTX.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("wgpu");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{
            label: Some("st-wgpu-mask-indices"), features: wgpu::Features::empty(), limits: wgpu::Limits::downlevel_defaults()
        }, None)).expect("device");
        let base = include_str!("wgpu_kernels_all.wgsl");
        let app  = include_str!("wgpu_kernels_mask_indices.append.wgsl");
        let shader_src = [base, app].join("\n");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("st-mask-indices"), source: wgpu::ShaderSource::Wgsl(shader_src.into())
        });
        let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("mask_indices"), layout: None, module: &shader, entry_point: "mask_indices"
        });
        Ctx{ device, queue, p }
    })
}
fn ub<T: Pod>(v:&T, label:&str)->wgpu::Buffer{
    ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label: Some(label), contents: bytemuck::bytes_of(v), usage: wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST })
}

impl WgpuMaskIndices {
    pub fn mask(&self, x:&wgpu::Buffer, idx:&wgpu::Buffer, rows:usize, cols:usize, k:usize) -> Result<()> {
        let rc = RC{ rows: rows as u32, cols: cols as u32, k: k as u32, neg_inf: -3.40282347e+38 };
        let b_rc = ub(&rc, "rc");
        let bind = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("mask-bind"), layout: &ctx().p.get_bind_group_layout(111),
            entries: &[
                wgpu::BindGroupEntry{ binding:0, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:1, resource: idx.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:2, resource: b_rc.as_entire_binding() },
            ]
        });
        let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("mask-enc") });
        { let mut p = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("mask-pass") });
          p.set_pipeline(&ctx().p); p.set_bind_group(111, &bind, &[]); p.dispatch_workgroups(rows as u32, 1, 1); }
        ctx().queue.submit(std::iter::once(e.finish()));
        Ok(())
    }
    pub fn device()->(&'static wgpu::Device, &'static wgpu::Queue) { (&ctx().device, &ctx().queue) }
}
