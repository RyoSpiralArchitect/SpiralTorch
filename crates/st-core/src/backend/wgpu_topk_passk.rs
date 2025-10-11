
use once_cell::sync::OnceCell;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use crate::error::Result;
use super::BackendArrayF32;

#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct RC { rows:u32, cols:u32, k:u32 }

pub struct WgpuTopKPassK;
impl WgpuTopKPassK { pub fn new()->Self{ WgpuTopKPassK } }

struct Ctx { device: wgpu::Device, queue: wgpu::Queue, p: wgpu::ComputePipeline, adapter_info: wgpu::AdapterInfo }
static CTX: OnceCell<Ctx> = OnceCell::new();
fn ctx()->&'static Ctx {
    CTX.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("wgpu");
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{
            label: Some("st-wgpu-topk-passk"), features: wgpu::Features::empty(), limits: wgpu::Limits::downlevel_defaults()
        }, None)).expect("device");
        let base = include_str!("wgpu_kernels_all.wgsl");
        let app  = include_str!("wgpu_kernels_topk_pass256.append.wgsl");
        let shader_src = [base, app].join("\\n");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("st-topk-passk"), source: wgpu::ShaderSource::Wgsl(shader_src.into())
        });
        let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{ label: Some("topk_passk"), layout: None, module: &shader, entry_point: "topk_pass256" });
        Ctx{ device, queue, p, adapter_info: info }
    })
}
pub fn adapter_info()->&'static wgpu::AdapterInfo { &ctx().adapter_info }
fn buf(size:u64, usage: wgpu::BufferUsages, label:&str)->wgpu::Buffer{
    ctx().device.create_buffer(&wgpu::BufferDescriptor{ label: Some(label), size, usage, mapped_at_creation:false })
}
fn ub<T: Pod>(v:&T, label:&str)->wgpu::Buffer{
    ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label: Some(label), contents: bytemuck::bytes_of(v), usage: wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST })
}

impl WgpuTopKPassK {
    pub fn pass_k(&self, x:&BackendArrayF32, rows:usize, cols:usize, k:usize) -> Result<(wgpu::Buffer, wgpu::Buffer)> {
        assert!(k<=256);
        let xb = match x { BackendArrayF32::Wgpu{ buffer, .. } => buffer, _=> return Err(crate::error::device("topk_passk: non-wgpu")) };
        let k32 = k as u32;
        let outv = buf((rows as u64)*k32 as u64*4, wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ, "passk-v");
        let outi = buf((rows as u64)*k32 as u64*4, wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ, "passk-i");
        let rc = RC{ rows: rows as u32, cols: cols as u32, k: k32 };
        let b_rc = ub(&rc, "rc");
        let bind = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("passk-bind"),
            layout: &ctx().p.get_bind_group_layout(110),
            entries: &[
                wgpu::BindGroupEntry{ binding:0, resource: xb.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:1, resource: outv.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:2, resource: outi.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:3, resource: b_rc.as_entire_binding() },
            ]
        });
        let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("passk-enc") });
        { let mut p = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("passk") });
          p.set_pipeline(&ctx().p); p.set_bind_group(110, &bind, &[]); p.dispatch_workgroups(rows as u32, 1, 1); }
        ctx().queue.submit(std::iter::once(e.finish()));
        Ok((outv, outi))
    }
    pub fn device()->(&'static wgpu::Device, &'static wgpu::Queue) { (&ctx().device, &ctx().queue) }
}
