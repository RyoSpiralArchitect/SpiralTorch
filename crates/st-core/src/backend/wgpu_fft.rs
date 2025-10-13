//! WGPU row-wise FFT (complex float2 interleaved). Minimal skeleton.
#![cfg(feature = "wgpu")]
use wgpu::*;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    rows: u32,
    cols: u32,
    inverse: u32,
    stages: u32,
    radix: u32, // 2 or 4
    _pad: u32,
}

#[derive(Clone, Copy)]
pub enum FftNorm { None, Backward, Ortho }
#[derive(Clone, Copy)]
pub enum FftRadix { Auto, Radix2, Radix4 }

fn next_pow2(x: usize) -> usize { x.next_power_of_two() }

pub fn fft_1d_rows_wgpu(
    device: &Device,
    queue: &Queue,
    x_reim: &[f32], // len = rows*cols*2
    rows: usize,
    cols: usize,
    inverse: bool,
    _norm: FftNorm,
    radix: Option<FftRadix>,
) -> anyhow::Result<Vec<f32>> {
    assert_eq!(x_reim.len(), rows*cols*2);
    let cols_pow2 = next_pow2(cols);
    if cols_pow2 != cols { anyhow::bail!("cols must be power-of-two for this kernel"); }

    let chosen_radix = match radix.unwrap_or(FftRadix::Auto) {
        FftRadix::Auto => if cols % 4 == 0 { 4 } else { 2 },
        FftRadix::Radix2 => 2,
        FftRadix::Radix4 => 4,
    } as u32;

    // Load shader
    let shader_src = if chosen_radix == 4 { include_str!("shaders/wgpu_fft_radix4.wgsl") } else { include_str!("shaders/wgpu_fft_radix2.wgsl") };
    let module = device.create_shader_module(ShaderModuleDescriptor{
        label: Some("fft-wgsl"),
        source: ShaderSource::Wgsl(shader_src.into()),
    });

    // Buffers
    let in_size = (x_reim.len()*4) as u64;
    let out_size = in_size;
    let buf_in = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("fft-in"),
        contents: bytemuck::cast_slice(x_reim),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let buf_out = device.create_buffer(&BufferDescriptor{
        label: Some("fft-out"),
        size: out_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = Params {
        rows: rows as u32,
        cols: cols as u32,
        inverse: if inverse {1} else {0},
        stages: (cols as f32).log2().round() as u32,
        radix: chosen_radix,
        _pad: 0,
    };
    let buf_params = device.create_buffer_init(&util::BufferInitDescriptor{
        label: Some("fft-params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    // Layouts
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor{
        label: Some("fft-bgl"),
        entries: &[
            BindGroupLayoutEntry{ binding:0, visibility:ShaderStages::COMPUTE, ty:BindingType::Buffer{ty:BufferBindingType::Storage{ read_only:true }, has_dynamic_offset:false, min_binding_size:None}, count:None },
            BindGroupLayoutEntry{ binding:1, visibility:ShaderStages::COMPUTE, ty:BindingType::Buffer{ty:BufferBindingType::Storage{ read_only:false }, has_dynamic_offset:false, min_binding_size:None}, count:None },
            BindGroupLayoutEntry{ binding:2, visibility:ShaderStages::COMPUTE, ty:BindingType::Buffer{ty:BufferBindingType::Uniform, has_dynamic_offset:false, min_binding_size:None}, count:None },
        ]
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor{
        label: Some("fft-pl"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[]
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor{
        label: Some("fft-pipe"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: "main",
    });
    let bind = device.create_bind_group(&BindGroupDescriptor{
        label: Some("fft-bg"),
        layout: &layout,
        entries: &[
            BindGroupEntry{ binding:0, resource: buf_in.as_entire_binding() },
            BindGroupEntry{ binding:1, resource: buf_out.as_entire_binding() },
            BindGroupEntry{ binding:2, resource: buf_params.as_entire_binding() },
        ]
    });

    // Dispatch
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor{ label: Some("fft-enc") });
    {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor{ label: Some("fft-cp") });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind, &[]);
        // 1 thread per row (or more). Here: wg_size=64, grid = ceil(rows/64)
        let wg = 64u32;
        let nx = ((rows as u32) + wg - 1)/wg;
        cpass.dispatch_workgroups(nx, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    // Readback
    let buf_read = device.create_buffer(&BufferDescriptor{
        label: Some("fft-read"),
        size: out_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc2 = device.create_command_encoder(&CommandEncoderDescriptor{ label: Some("fft-enc2") });
    enc2.copy_buffer_to_buffer(&buf_out, 0, &buf_read, 0, out_size);
    queue.submit(Some(enc2.finish()));

    // Map
    let slice = buf_read.slice(..);
    let (sender, recv) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(MapMode::Read, move |res| { sender.send(res).ok(); });
    futures_lite::future::block_on(recv.receive());
    let data = slice.get_mapped_range();
    let mut out = vec![0f32; x_reim.len()];
    out.copy_from_slice(bytemuck::cast_slice(&data));
    drop(data);
    buf_read.unmap();
    Ok(out)
}
