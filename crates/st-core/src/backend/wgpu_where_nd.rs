
use once_cell::sync::OnceCell;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use crate::error::{Result, device as dev_err};

#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct RC { nd:u32, n:u32 }

pub struct WgpuWhereND;
impl WgpuWhereND { pub fn new()->Self{ WgpuWhereND } }

struct Ctx { device: wgpu::Device, queue: wgpu::Queue, p_nd: wgpu::ComputePipeline, p_str: wgpu::ComputePipeline, p_u8: wgpu::ComputePipeline }
static CTX: OnceCell<Ctx> = OnceCell::new();
fn ctx()->&'static Ctx {
    CTX.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("wgpu");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{
            label: Some("st-wgpu-where-nd"), features: wgpu::Features::empty(), limits: wgpu::Limits::downlevel_defaults()
        }, None)).expect("device");
        let base = include_str!("wgpu_kernels_all.wgsl");
        let app0 = include_str!("wgpu_kernels_where_nd.append.wgsl");
        let app1 = include_str!("wgpu_kernels_where_nd_strided.append.wgsl");
        let app2 = include_str!("wgpu_kernels_where_nd_strided_u8.append.wgsl");
        let shader_nd  = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("st-where-nd"), source: wgpu::ShaderSource::Wgsl([base, app0].join("\n").into())
        });
        let shader_str = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("st-where-nd-strided"), source: wgpu::ShaderSource::Wgsl([base, app1].join("\n").into())
        });
        let shader_u8 = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("st-where-nd-strided-u8"), source: wgpu::ShaderSource::Wgsl([base, app2].join("\n").into())
        });
        let p_nd = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{ label: Some("where_nd"), layout: None, module: &shader_nd, entry_point: "where_nd" });
        let p_str= device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{ label: Some("where_nd_strided"), layout: None, module: &shader_str, entry_point: "where_nd_strided" });
        let p_u8 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{ label: Some("where_nd_strided_u8"), layout: None, module: &shader_u8, entry_point: "where_nd_strided_u8" });
        Ctx{ device, queue, p_nd, p_str, p_u8 }
    })
}

fn buf(size:u64, usage: wgpu::BufferUsages, label:&str)->wgpu::Buffer{
    ctx().device.create_buffer(&wgpu::BufferDescriptor{ label: Some(label), size, usage, mapped_at_creation: false })
}
fn sbuf_u32(data: &[u32], label:&str)->wgpu::Buffer{
    ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label: Some(label), contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST })
}
fn ub<T: Pod>(v:&T, label:&str)->wgpu::Buffer{
    ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label: Some(label), contents: bytemuck::bytes_of(v), usage: wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST })
}

fn strides_of(shape: &[u32]) -> Vec<u32> { let nd=shape.len(); let mut st=vec![0u32; nd]; let mut acc=1u32; for d in (0..nd).rev(){ st[d]=acc; acc = acc.saturating_mul(shape[d]); } st }

pub fn where_nd(&self,
    cond:&wgpu::Buffer, cond_shape:&[u32],
    x:&wgpu::Buffer, x_shape:&[u32],
    y:&wgpu::Buffer, y_shape:&[u32],
    out_shape:&[u32]) -> Result<wgpu::Buffer>
{
    // use strided path with contiguous strides
    let st_out = strides_of(out_shape);
    self.where_nd_strided(cond, cond_shape, &strides_of(cond_shape), x, x_shape, &strides_of(x_shape), y, y_shape, &strides_of(y_shape), out_shape, &st_out)
}

pub fn where_nd_strided(&self,
    cond:&wgpu::Buffer, cond_shape:&[u32], cond_strides:&[u32],
    x:&wgpu::Buffer, x_shape:&[u32], x_strides:&[u32],
    y:&wgpu::Buffer, y_shape:&[u32], y_strides:&[u32],
    out_shape:&[u32], out_strides:&[u32]) -> Result<wgpu::Buffer>
{
    let n: u32 = out_shape.iter().cloned().fold(1u32, |a,b| a.saturating_mul(b));
    let out = buf((n as u64)*4, wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC, "where-nd-out");
    let rc = RC{ nd: out_shape.len() as u32, n };
    let b_rc = ub(&rc, "rc");
    let b_out_shape = sbuf_u32(out_shape, "out_shape");
    let b_out_strides= sbuf_u32(out_strides, "out_strides");
    let b_c_shape = sbuf_u32(cond_shape, "c_shape");
    let b_c_strides = sbuf_u32(cond_strides, "c_strides");
    let b_x_shape = sbuf_u32(x_shape, "x_shape");
    let b_x_strides = sbuf_u32(x_strides, "x_strides");
    let b_y_shape = sbuf_u32(y_shape, "y_shape");
    let b_y_strides = sbuf_u32(y_strides, "y_strides");

    let bind = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("where-nd-bind-strided"),
        layout: &ctx().p_str.get_bind_group_layout(93),
        entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: cond.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: y.as_entire_binding() },
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
        ]
    });
    let groups = ((n + 255) / 256) as u32;
    let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("where-nd-enc-str") });
    { let mut p = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("where-nd-pass-str") });
      p.set_pipeline(&ctx().p_str); p.set_bind_group(93, &bind, &[]); p.dispatch_workgroups(groups, 1, 1); }
    ctx().queue.submit(std::iter::once(e.finish()));
    Ok(out)
}

/// u8-cond optimized path: cond bytes are packed (4 per u32 word). length ceil(n/4).
pub fn where_nd_strided_u8(&self,
    cond_bytes:&wgpu::Buffer,
    x:&wgpu::Buffer, x_shape:&[u32], x_strides:&[u32],
    y:&wgpu::Buffer, y_shape:&[u32], y_strides:&[u32],
    out_shape:&[u32], out_strides:&[u32],
    c_shape:&[u32], c_strides:&[u32]) -> Result<wgpu::Buffer>
{
    let n: u32 = out_shape.iter().cloned().fold(1u32, |a,b| a.saturating_mul(b));
    let out = buf((n as u64)*4, wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC, "where-nd-out");
    let rc = RC{ nd: out_shape.len() as u32, n };
    let b_rc = ub(&rc, "rc");
    let b_out_shape = sbuf_u32(out_shape, "out_shape");
    let b_out_strides= sbuf_u32(out_strides, "out_strides");
    let b_c_shape = sbuf_u32(c_shape, "c_shape");
    let b_c_strides = sbuf_u32(c_strides, "c_strides");
    let b_x_shape = sbuf_u32(x_shape, "x_shape");
    let b_x_strides = sbuf_u32(x_strides, "x_strides");
    let b_y_shape = sbuf_u32(y_shape, "y_shape");
    let b_y_strides = sbuf_u32(y_strides, "y_strides");

    let bind = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: Some("where-nd-bind-u8"),
        layout: &ctx().p_u8.get_bind_group_layout(94),
        entries: &[
            wgpu::BindGroupEntry{ binding:0, resource: cond_bytes.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry{ binding:2, resource: y.as_entire_binding() },
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
        ]
    });
    let groups = ((n + 255) / 256) as u32;
    let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("where-nd-enc-u8") });
    { let mut p = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("where-nd-pass-u8") });
      p.set_pipeline(&ctx().p_u8); p.set_bind_group(94, &bind, &[]); p.dispatch_workgroups(groups, 1, 1); }
    ctx().queue.submit(std::iter::once(e.finish()));
    Ok(out)
}
