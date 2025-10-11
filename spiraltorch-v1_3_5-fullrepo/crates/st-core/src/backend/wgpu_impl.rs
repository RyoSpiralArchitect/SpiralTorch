
use ndarray::ArrayD;
use once_cell::sync::OnceCell;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{device::Device, error::{Result, device as dev_err}};

pub struct WgpuBackend;
impl WgpuBackend { pub fn new() -> Self { WgpuBackend } }

const WGSL_SRC: &str = include_str!("wgpu_kernels_all.wgsl");

#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)] pub struct U32One { pub n:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)] pub struct T2 { pub rows:u32, pub cols:u32, pub stride_x:u32, pub stride_y:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)] pub struct TB { pub rows:u32, pub cols:u32, pub batches:u32, pub _p:u32, pub stride_x:u32, pub stride_y:u32, pub _p1:u32, pub _p2:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)] pub struct MM { pub m:u32, pub n:u32, pub k:u32, pub tile:u32, pub lda:u32, pub ldb:u32, pub ldc:u32, pub batch:u32, pub stride_a:u32, pub stride_b:u32, pub stride_c:u32, pub _p:u32 }

pub enum BackendArrayF32 {
    Wgpu { rows: usize, cols: usize, buffer: wgpu::Buffer },
    #[allow(dead_code)] HostStub,
}

pub trait Backend {
    fn name(&self) -> &'static str;
    fn device(&self) -> Device;
    fn from_host_f32(&self, host: &ArrayD<f32>) -> Result<BackendArrayF32>;
    fn to_host_f32(&self, arr: &BackendArrayF32) -> Result<ArrayD<f32>>;
}

struct Ctx {
    device: wgpu::Device,
    queue:  wgpu::Queue,
    p_add: wgpu::ComputePipeline,
    p_t2d: wgpu::ComputePipeline,
    p_t2db: wgpu::ComputePipeline,
    p_mm2d: wgpu::ComputePipeline,
    p_mm2db: wgpu::ComputePipeline,
    p_dot: wgpu::ComputePipeline,
    p_smax: wgpu::ComputePipeline,
}
static CTX: OnceCell<Ctx> = OnceCell::new();
fn ctx() -> &'static Ctx {
    CTX.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("No adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{
            label: Some("spiraltorch-wgpu"),
            features: wgpu::Features::empty(), limits: wgpu::Limits::downlevel_defaults(),
        }, None)).expect("device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("st-wgsl"), source: wgpu::ShaderSource::Wgsl(WGSL_SRC.into()),
        });
        let pipe = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some(entry), layout: None, module: &shader, entry_point: entry
        });
        Ctx {
            device, queue,
            p_add: pipe("add_vec"),
            p_t2d: pipe("transpose_2d"),
            p_t2db: pipe("transpose_2d_batched"),
            p_mm2d: pipe("matmul_tiled"),
            p_mm2db: pipe("matmul_tiled_batched"),
            p_dot: pipe("rowwise_dot_gy_wg"),
            p_smax: pipe("softmax_bw_from_dot"),
        }
    })
}

fn buf(size:u64, usage: wgpu::BufferUsages, label:&str)->wgpu::Buffer{
    ctx().device.create_buffer(&wgpu::BufferDescriptor{ label:Some(label), size, usage, mapped_at_creation:false })
}

impl Backend for WgpuBackend {
    fn name(&self)->&'static str{ "wgpu" }
    fn device(&self)->Device{ Device::Wgpu }
    fn from_host_f32(&self, host:&ArrayD<f32>) -> Result<BackendArrayF32> {
        let bytes = bytemuck::cast_slice(host.as_slice().ok_or_else(|| dev_err("host not contiguous"))?);
        let b = ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("upload"), contents: bytes, usage: wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST
        });
        Ok(BackendArrayF32::Wgpu{ rows: host.len(), cols:1, buffer: b })
    }
    fn to_host_f32(&self, arr:&BackendArrayF32) -> Result<ArrayD<f32>> {
        let (rows, cols, src) = match arr { BackendArrayF32::Wgpu{rows, cols, buffer} => (*rows,*cols,buffer), _=> return Err(dev_err("non-wgpu")) };
        let size = (rows*cols*4) as u64;
        let read = buf(size, wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST, "read");
        let mut enc = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("rb-enc") });
        enc.copy_buffer_to_buffer(src, 0, &read, 0, size);
        ctx().queue.submit(std::iter::once(enc.finish()));
        let slice = read.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _=tx.send(r); });
        pollster::block_on(rx.receive());
        let data = slice.get_mapped_range().to_vec();
        read.unmap();
        let v:Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        Ok(ndarray::Array1::from_vec(v).into_dyn())
    }
}

impl WgpuBackend {
    pub fn add(&self, a:&BackendArrayF32, b:&BackendArrayF32) -> Result<BackendArrayF32> {
        let (n, ab, bb) = match (a,b) {
            (BackendArrayF32::Wgpu{rows:ra, cols:ca, buffer:ab}, BackendArrayF32::Wgpu{rows:rb, cols:cb, buffer:bb}) => {
                assert!(ra*ca==rb*cb); (ra*ca, ab, bb)
            }, _ => return Err(dev_err("wgpu add: buffers"))};
        let out = buf((n*4) as u64, wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST, "add-out");
        let u = U32One{ n: n as u32 };
        let ub = ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label:Some("uN"), contents: bytemuck::bytes_of(&u), usage: wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST });
        let bind = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("add-bind"),
            layout: &ctx().p_add.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry{ binding:0, resource: ab.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:1, resource: bb.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:2, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:3, resource: ub.as_entire_binding() },
            ]
        });
        let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("add-enc") });
        { let mut p = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("add-pass") });
          p.set_pipeline(&ctx().p_add); p.set_bind_group(0, &bind, &[]); p.dispatch_workgroups(((n as u32)+255)/256, 1, 1); }
        ctx().queue.submit(std::iter::once(e.finish()));
        Ok(BackendArrayF32::Wgpu{ rows:n, cols:1, buffer: out })
    }

    pub fn transpose2d(&self, x:&BackendArrayF32, rows:usize, cols:usize) -> Result<BackendArrayF32> {
        let xb = match x { BackendArrayF32::Wgpu{buffer, ..} => buffer, _=>return Err(dev_err("non-wgpu")) };
        let out = buf((rows*cols*4) as u64, wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST, "t2d-out");
        let info = T2{ rows: rows as u32, cols: cols as u32, stride_x: cols as u32, stride_y: rows as u32 };
        let ub = ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label:Some("T2"), contents: bytemuck::bytes_of(&info), usage: wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST });
        let bind = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("t2d-bind"),
            layout: &ctx().p_t2d.get_bind_group_layout(6),
            entries: &[
                wgpu::BindGroupEntry{ binding:0, resource: xb.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:1, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:2, resource: ub.as_entire_binding() },
            ]
        });
        let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("t2d-enc") });
        { let mut p = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("t2d-pass") });
          p.set_pipeline(&ctx().p_t2d); p.set_bind_group(6, &bind, &[]); p.dispatch_workgroups((((rows*cols) as u32)+255)/256, 1, 1); }
        ctx().queue.submit(std::iter::once(e.finish()));
        Ok(BackendArrayF32::Wgpu{ rows: cols, cols: rows, buffer: out })
    }

    pub fn transpose2d_batched(&self, x:&BackendArrayF32, rows:usize, cols:usize, bsz:usize) -> Result<BackendArrayF32> {
        let xb = match x { BackendArrayF32::Wgpu{buffer, ..} => buffer, _=>return Err(dev_err("non-wgpu")) };
        let per = rows*cols;
        let out = buf(((per*bsz)*4) as u64, wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST, "t2db-out");
        let info = TB{ rows: rows as u32, cols: cols as u32, batches: bsz as u32, _p:0, stride_x: per as u32, stride_y: per as u32, _p1:0, _p2:0 };
        let ub = ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label:Some("TB"), contents: bytemuck::bytes_of(&info), usage: wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST });
        let bind = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("t2db-bind"),
            layout: &ctx().p_t2db.get_bind_group_layout(7),
            entries: &[
                wgpu::BindGroupEntry{ binding:0, resource: xb.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:1, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:2, resource: ub.as_entire_binding() },
            ]
        });
        let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("t2db-enc") });
        { let mut p = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("t2db-pass") });
          p.set_pipeline(&ctx().p_t2db); p.set_bind_group(7, &bind, &[]);
          let total = (per*bsz) as u32; p.dispatch_workgroups((total+255)/256, 1, 1); }
        ctx().queue.submit(std::iter::once(e.finish()));
        Ok(BackendArrayF32::Wgpu{ rows: bsz*cols, cols: rows, buffer: out })
    }

    pub fn matmul2d_tiled(&self, a:&BackendArrayF32, b:&BackendArrayF32, m:usize, k:usize, n:usize) -> Result<BackendArrayF32> {
        let (ab, bb) = match (a,b) { (BackendArrayF32::Wgpu{buffer:ab,..}, BackendArrayF32::Wgpu{buffer:bb,..}) => (ab,bb), _=>return Err(dev_err("non-wgpu")) };
        let c = buf((m*n*4) as u64, wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST, "mm2d-out");
        let info = MM{ m:m as u32,n:n as u32,k:k as u32,tile:16,lda:k as u32,ldb:n as u32,ldc:n as u32,batch:1,stride_a:0,stride_b:0,stride_c:0,_p:0 };
        let ub = ctx().device.create_buffer_init(&wgpu::util::BufferInitDescriptor{ label:Some("MM"), contents: bytemuck::bytes_of(&info), usage: wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST });
        let bind = ctx().device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("mm2d-bind"), layout: &ctx().p_mm2d.get_bind_group_layout(5),
            entries: &[
                wgpu::BindGroupEntry{ binding:0, resource: ab.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:1, resource: bb.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:2, resource: c.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:3, resource: ub.as_entire_binding() },
            ]
        });
        let mut e = ctx().device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("mm2d-enc") });
        { let mut p = e.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("mm2d-pass") });
          p.set_pipeline(&ctx().p_mm2d); p.set_bind_group(5, &bind, &[]);
          p.dispatch_workgroups(((n as u32)+15)/16, ((m as u32)+15)/16, 1); }
        ctx().queue.submit(std::iter::once(e.finish()));
        Ok(BackendArrayF32::Wgpu{ rows:m, cols:n, buffer: c })
    }

    pub fn matmul2d_bwd(&self, go:&BackendArrayF32, a:&BackendArrayF32, b:&BackendArrayF32, m:usize, k:usize, n:usize) -> Result<(BackendArrayF32, BackendArrayF32)> {
        let bt = self.transpose2d(b, k, n)?;
        let da = self.matmul2d_tiled(go, &bt, m, n, k)?;
        let at = self.transpose2d(a, m, k)?;
        let db = self.matmul2d_tiled(&at, go, k, m, n)?;
        Ok((da, db))
    }
}
