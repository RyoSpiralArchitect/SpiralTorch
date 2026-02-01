// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "wgpu")]

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};

use bytemuck::{cast_slice, Pod, Zeroable};
use pollster::block_on;
use thiserror::Error;

use crate::mellin_types::{ComplexScalar, Scalar};

#[derive(Clone, Debug, Error)]
pub enum MellinGpuError {
    #[error("weighted series must not be empty")]
    EmptySeries,
    #[error("evaluation batch must not be empty")]
    EmptyBatch,
    #[error("unsupported scalar width for GPU evaluation")]
    UnsupportedScalar,
    #[error("no compatible WGPU adapter was found")]
    NoAdapter,
    #[error("failed to acquire WGPU device: {0}")]
    RequestDevice(String),
    #[error("failed to compile Mellin WGSL shader: {0}")]
    Shader(String),
    #[error("failed to map GPU buffer for readback")]
    Map,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ComplexPod {
    re: f32,
    im: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsPod {
    len: u32,
    count: u32,
    _pad: [u32; 2],
}

struct MellinGpuExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    buffers: Mutex<MellinGpuBuffers>,
}

static EXECUTOR: OnceLock<Result<MellinGpuExecutor, MellinGpuError>> = OnceLock::new();

pub fn evaluate_weighted_series_many_gpu(
    weighted: &[ComplexScalar],
    z_values: &[ComplexScalar],
) -> Result<Vec<ComplexScalar>, MellinGpuError> {
    if weighted.is_empty() {
        return Err(MellinGpuError::EmptySeries);
    }
    if z_values.is_empty() {
        return Err(MellinGpuError::EmptyBatch);
    }
    ensure_scalar_supported()?;

    let executor = match EXECUTOR.get_or_init(MellinGpuExecutor::new) {
        Ok(exec) => exec,
        Err(err) => return Err(err.clone()),
    };
    executor.evaluate(weighted, z_values)
}

impl MellinGpuExecutor {
    fn new() -> Result<Self, MellinGpuError> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok_or(MellinGpuError::NoAdapter)?;

        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                .map_err(|err| MellinGpuError::RequestDevice(err.to_string()))?;

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("st.mellin.gpu.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("st.mellin.gpu.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let shader = catch_unwind(AssertUnwindSafe(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("st.mellin.gpu.shader"),
                source: wgpu::ShaderSource::Wgsl(MELLIN_WGSL.into()),
            })
        }))
        .map_err(|payload| MellinGpuError::Shader(panic_payload_to_string(payload)))?;

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("st.mellin.gpu.pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            buffers: Mutex::new(MellinGpuBuffers::default()),
        })
    }

    fn evaluate(
        &self,
        weighted: &[ComplexScalar],
        z_values: &[ComplexScalar],
    ) -> Result<Vec<ComplexScalar>, MellinGpuError> {
        let coeffs = to_pod_vec(weighted)?;
        let zs = to_pod_vec(z_values)?;
        let coeff_bytes = (std::mem::size_of::<ComplexPod>() * weighted.len()) as u64;
        let z_bytes = (std::mem::size_of::<ComplexPod>() * z_values.len()) as u64;
        let output_size = z_bytes;

        let (coeff_buffer, z_buffer, output_buffer, staging_buffer, params_buffer) = {
            let mut buffers = self.buffers.lock().expect("mellin gpu buffers poisoned");

            let coeff = buffers.ensure(
                &self.device,
                "st.mellin.gpu.coeffs",
                coeff_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let z = buffers.ensure(
                &self.device,
                "st.mellin.gpu.zs",
                z_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let output = buffers.ensure(
                &self.device,
                "st.mellin.gpu.output",
                output_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let staging = buffers.ensure(
                &self.device,
                "st.mellin.gpu.staging",
                output_size,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            );
            let params = buffers.ensure_params(&self.device);
            (coeff, z, output, staging, params)
        };

        self.queue.write_buffer(
            coeff_buffer.as_ref(),
            0,
            cast_slice(&coeffs),
        );
        self.queue.write_buffer(
            z_buffer.as_ref(),
            0,
            cast_slice(&zs),
        );

        let params = ParamsPod {
            len: weighted.len() as u32,
            count: z_values.len() as u32,
            _pad: [0, 0],
        };
        self.queue
            .write_buffer(params_buffer.as_ref(), 0, cast_slice(&[params]));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.mellin.gpu.bind_group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: coeff_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("st.mellin.gpu.encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("st.mellin.gpu.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (z_values.len() as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            output_buffer.as_ref(),
            0,
            staging_buffer.as_ref(),
            0,
            output_size,
        );
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let slice = staging_buffer.slice(0..output_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match receiver.recv().map_err(|_| MellinGpuError::Map)? {
            Ok(()) => {}
            Err(_) => return Err(MellinGpuError::Map),
        }
        let data = slice.get_mapped_range();
        let pods: &[ComplexPod] = cast_slice(&data);
        let mut result = Vec::with_capacity(z_values.len());
        for pod in pods.iter() {
            result.push(ComplexScalar::new(pod.re as Scalar, pod.im as Scalar));
        }
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }
}

#[derive(Default)]
struct MellinGpuBuffers {
    coeffs: Option<CachedBuffer>,
    zs: Option<CachedBuffer>,
    output: Option<CachedBuffer>,
    staging: Option<CachedBuffer>,
    params: Option<Arc<wgpu::Buffer>>,
}

#[derive(Clone)]
struct CachedBuffer {
    buffer: Arc<wgpu::Buffer>,
    bytes: u64,
}

impl MellinGpuBuffers {
    fn ensure(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        bytes: u64,
        usage: wgpu::BufferUsages,
    ) -> Arc<wgpu::Buffer> {
        let slot = match label {
            "st.mellin.gpu.coeffs" => &mut self.coeffs,
            "st.mellin.gpu.zs" => &mut self.zs,
            "st.mellin.gpu.output" => &mut self.output,
            "st.mellin.gpu.staging" => &mut self.staging,
            _ => {
                // Fallback: allocate a fresh buffer (label is only used internally).
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes.max(4),
                    usage,
                    mapped_at_creation: false,
                });
                return Arc::new(buffer);
            }
        };

        let needs_alloc = slot.as_ref().map(|buf| buf.bytes < bytes).unwrap_or(true);
        if needs_alloc {
            let padded = bytes.max(4);
            let padded = padded.saturating_mul(11) / 10;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: padded,
                usage,
                mapped_at_creation: false,
            });
            *slot = Some(CachedBuffer {
                buffer: Arc::new(buffer),
                bytes: padded,
            });
        }

        slot.as_ref().expect("buffer must be allocated").buffer.clone()
    }

    fn ensure_params(&mut self, device: &wgpu::Device) -> Arc<wgpu::Buffer> {
        if let Some(buf) = &self.params {
            return buf.clone();
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("st.mellin.gpu.params"),
            size: std::mem::size_of::<ParamsPod>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf = Arc::new(buffer);
        self.params = Some(buf.clone());
        buf
    }
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

fn ensure_scalar_supported() -> Result<(), MellinGpuError> {
    if std::mem::size_of::<Scalar>() != std::mem::size_of::<f32>() {
        return Err(MellinGpuError::UnsupportedScalar);
    }
    Ok(())
}

fn to_pod_vec(values: &[ComplexScalar]) -> Result<Vec<ComplexPod>, MellinGpuError> {
    ensure_scalar_supported()?;
    Ok(values
        .iter()
        .map(|value| ComplexPod {
            re: value.re as f32,
            im: value.im as f32,
        })
        .collect())
}

const MELLIN_WGSL: &str = r#"
struct Complex { re: f32, im: f32; };

fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re,
    );
}

struct Params {
    len: u32,
    count: u32,
};

@group(0) @binding(0) var<storage, read> coeffs: array<Complex>;
@group(0) @binding(1) var<storage, read> zs: array<Complex>;
@group(0) @binding(2) var<storage, read_write> out: array<Complex>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count {
        return;
    }
    var acc = Complex(0.0, 0.0);
    var power = Complex(1.0, 0.0);
    let z = zs[idx];
    for (var k: u32 = 0u; k < params.len; k = k + 1u) {
        let coeff = coeffs[k];
        acc = Complex(
            acc.re + coeff.re * power.re - coeff.im * power.im,
            acc.im + coeff.re * power.im + coeff.im * power.re,
        );
        power = complex_mul(power, z);
    }
    out[idx] = acc;
}
"#;
