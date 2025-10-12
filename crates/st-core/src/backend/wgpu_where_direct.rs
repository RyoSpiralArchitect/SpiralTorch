use once_cell::sync::OnceCell;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use crate::error::{Result, device as dev_err};

#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
pub struct RC { pub nd:u32, pub n:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
pub struct RB { pub c_base:u32, pub x_base:u32, pub y_base:u32 }

pub struct Ctx { pub device: wgpu::Device, pub queue: wgpu::Queue, pub p_u8: wgpu::ComputePipeline }
static CTX: OnceCell<Ctx> = OnceCell::new();
fn ctx()->&'static Ctx {
    CTX.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("wgpu adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{
            label: Some("st-wgpu-where-direct"),
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::downlevel_defaults()
        }, None)).expect("wgpu device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("st-where-u8-direct"),
            source: wgpu::ShaderSource::Wgsl([
                crate::backend::WGSL_BASE,
                crate::backend::WGSL_WHERE_APPEND
            ].join("\n").into())
        });
        let p_u8 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("where-u8"), layout: None, module: &shader, entry_point: "where_nd_strided_u8"
        });
        Ctx{ device, queue, p_u8 }
    })
}
fn sbuf_u32(data: &[u32], label:&str)->wgpu::Buffer{
    ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some(label), contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST
    })
}
fn ub<T: Pod>(v:&T, label:&str)->wgpu::Buffer{
    ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some(label), contents: bytemuck::bytes_of(v),
        usage: wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST
    })
}
fn rowmajor_strides(shape:&[u32])->Vec<u32>{
    let nd=shape.len(); let mut st=vec![0u32;nd]; let mut acc=1u32;
    for d in (0..nd).rev(){ st[d]=acc; acc=acc.saturating_mul(shape[d].max(1)); }
    st
}
#[allow(clippy::too_many_arguments)]
pub fn where_nd_strided_bytes_direct(
    cond_blob: &[u8], c_base:u32, c_shape:&[u32], c_strides:&[u32],
    x_blob: &[u8], x_base:u32, x_shape:&[u32], x_strides:&[u32],
    y_blob: &[u8], y_base:u32, y_shape:&[u32], y_strides:&[u32],
    out_shape:&[u32]
) -> Result<Vec<f32>> {
    let n: u32 = out_shape.iter().copied().fold(1u32, |a,b| a.saturating_mul(b));
    let out_strides = rowmajor_strides(out_shape);
    let c_pad = (cond_blob.len() + 3) & !3; let mut c_padded = vec![0u8; c_pad];
    c_padded[0..cond_blob.len()].copy_from_slice(cond_blob);
    let b_cond = ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("cond-blob"), contents: &c_padded, usage: wgpu::BufferUsages::STORAGE
    });
    let b_x = ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("x-blob"), contents: x_blob, usage: wgpu::BufferUsages::STORAGE
    });
    let b_y = ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("y-blob"), contents: y_blob, usage: wgpu::BufferUsages::STORAGE
    });
    let out = ctx().device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("where-out"), size: (n as u64)*4, usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false
    });
    let b_out_shape   = sbuf_u32(out_shape, "out_shape");
    let b_out_strides = sbuf_u32(&out_strides, "out_strides");
    let b_c_shape     = sbuf_u32(c_shape, "c_shape");
    let b_c_strides   = sbuf_u32(c_strides, "c_strides");
    let b_x_shape     = sbuf_u32(x_shape, "x_shape");
    let b_x_strides   = sbuf_u32(x_strides, "x_strides");
    let b_y_shape     = sbuf_u32(y_shape, "y_shape");
    let b_y_strides   = sbuf_u32(y_strides, "y_strides");
    let rc = RC{ nd: out_shape.len() as u32, n };
    let rb = RB{ c_base, x_base, y_base };
    let b_rc = ub(&rc, "rc"); let b_rb = ub(&rb, "rb");
    let bind = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("where-direct-bind"),
        layout: &ctx().p_u8.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: b_cond.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: b_x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: b_y.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:3, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:4, resource: b_out_shape.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:5, resource: b_out_strides.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:6, resource: b_c_shape.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:7, resource: b_c_strides.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:8, resource: b_x_shape.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:9, resource: b_x_strides.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:10, resource: b_y_shape.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:11, resource: b_y_strides.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:12, resource: b_rc.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:13, resource: b_rb.as_entire_binding() },
        ]
    });
    let groups = ((n + 255) / 256) as u32;
    let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("where-enc") });
    { let mut p = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("where-pass") });
      p.set_pipeline(&ctx().p_u8); p.set_bind_group(0, &bind, &[]); p.dispatch_workgroups(groups, 1, 1); }
    ctx().queue.submit(std::iter::once(e.finish()));
    // readback
    let staging = ctx().device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("stage"), size: (n as u64)*4, usage: wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST, mapped_at_creation: false
    });
    let mut e2 = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("rb-enc") });
    e2.copy_buffer_to_buffer(&out, 0, &staging, 0, (n as u64)*4);
    ctx().queue.submit(std::iter::once(e2.finish()));
    let slice = staging.slice(..); let _ = slice.map_async(wgpu::MapMode::Read); ctx().device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range(); let v = bytemuck::cast_slice::<u8,f32>(&data).to_vec(); drop(data); staging.unmap();
    Ok(v)
}
