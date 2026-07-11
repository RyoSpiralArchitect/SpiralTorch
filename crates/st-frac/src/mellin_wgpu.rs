// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "wgpu")]
#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use bytemuck::{cast_slice, try_cast_slice, Pod, Zeroable};
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
    #[error("timed out waiting for GPU buffer readback")]
    MapTimeout,
    #[error("{kind}[{index}] is not finite: re={re}, im={im}")]
    NonFiniteInput {
        kind: &'static str,
        index: usize,
        re: Scalar,
        im: Scalar,
    },
    #[error("GPU output[{index}] is not finite: re={re}, im={im}")]
    NonFiniteOutput {
        index: usize,
        re: Scalar,
        im: Scalar,
    },
    #[error("{kind} length exceeds the GPU u32 limit: {len}")]
    LengthOverflow { kind: &'static str, len: usize },
    #[error("{kind} byte size overflow for {len} elements")]
    BufferSizeOverflow { kind: &'static str, len: usize },
    #[error("{kind} requires {bytes} bytes but the device limit is {limit}")]
    BufferLimit {
        kind: &'static str,
        bytes: u64,
        limit: u64,
    },
    #[error("dispatch requires {workgroups} workgroups but the device limit is {limit}")]
    DispatchLimit { workgroups: u32, limit: u32 },
    #[error(
        "workgroup size {required} exceeds device limits (size_x={size_x}, invocations={invocations})"
    )]
    WorkgroupLimit {
        required: u32,
        size_x: u32,
        invocations: u32,
    },
    #[error("GPU readback length mismatch (expected={expected}, got={got})")]
    ReadbackLength { expected: usize, got: usize },
    #[error("GPU readback buffer has an invalid complex layout")]
    ReadbackLayout,
    #[error("GPU buffer cache invariant failed for {kind}")]
    BufferInvariant { kind: &'static str },
    #[error("failed to reserve {len} elements for {kind}")]
    Allocation { kind: &'static str, len: usize },
    #[error("WGPU {stage} panicked: {message}")]
    RuntimePanic {
        stage: &'static str,
        message: String,
    },
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

const WORKGROUP_SIZE: u32 = 64;
const MAP_TIMEOUT: Duration = Duration::from_secs(30);
const MAP_POLL_INTERVAL: Duration = Duration::from_millis(2);

#[derive(Clone, Copy, Debug)]
struct GpuDispatch {
    len: u32,
    count: u32,
    coeff_bytes: u64,
    z_bytes: u64,
    workgroups: u32,
}

impl GpuDispatch {
    fn new(weighted_len: usize, z_len: usize) -> Result<Self, MellinGpuError> {
        let len = u32::try_from(weighted_len).map_err(|_| MellinGpuError::LengthOverflow {
            kind: "weighted series",
            len: weighted_len,
        })?;
        let count = u32::try_from(z_len).map_err(|_| MellinGpuError::LengthOverflow {
            kind: "evaluation batch",
            len: z_len,
        })?;
        Ok(Self {
            len,
            count,
            coeff_bytes: checked_byte_size("weighted series", weighted_len)?,
            z_bytes: checked_byte_size("evaluation batch", z_len)?,
            workgroups: count.div_ceil(WORKGROUP_SIZE),
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum BufferKind {
    Coefficients,
    ZValues,
    Output,
}

impl BufferKind {
    fn name(self) -> &'static str {
        match self {
            Self::Coefficients => "coefficient buffer",
            Self::ZValues => "z-value buffer",
            Self::Output => "output buffer",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Coefficients => "st.mellin.gpu.coeffs",
            Self::ZValues => "st.mellin.gpu.zs",
            Self::Output => "st.mellin.gpu.output",
        }
    }
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
    let dispatch = GpuDispatch::new(weighted.len(), z_values.len())?;
    validate_finite_input(weighted, "weighted coefficient")?;
    validate_finite_input(z_values, "z value")?;

    let executor =
        match EXECUTOR.get_or_init(|| guard_wgpu_panic("initialization", MellinGpuExecutor::new)) {
            Ok(exec) => exec,
            Err(err) => return Err(err.clone()),
        };
    guard_wgpu_panic("evaluation", || {
        executor.evaluate(weighted, z_values, dispatch)
    })
}

fn guard_wgpu_panic<T, F>(stage: &'static str, operation: F) -> Result<T, MellinGpuError>
where
    F: FnOnce() -> Result<T, MellinGpuError>,
{
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(payload) => Err(MellinGpuError::RuntimePanic {
            stage,
            message: panic_payload_to_string(payload),
        }),
    }
}

fn validate_finite_input(
    values: &[ComplexScalar],
    kind: &'static str,
) -> Result<(), MellinGpuError> {
    for (index, value) in values.iter().enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            return Err(MellinGpuError::NonFiniteInput {
                kind,
                index,
                re: value.re,
                im: value.im,
            });
        }
    }
    Ok(())
}

fn checked_byte_size(kind: &'static str, len: usize) -> Result<u64, MellinGpuError> {
    let bytes = std::mem::size_of::<ComplexPod>()
        .checked_mul(len)
        .ok_or(MellinGpuError::BufferSizeOverflow { kind, len })?;
    u64::try_from(bytes).map_err(|_| MellinGpuError::BufferSizeOverflow { kind, len })
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
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
        dispatch: GpuDispatch,
    ) -> Result<Vec<ComplexScalar>, MellinGpuError> {
        let storage_limit = self.validate_limits(dispatch)?;
        let coeffs = to_pod_vec(weighted, "coefficient upload")?;
        let zs = to_pod_vec(z_values, "z-value upload")?;

        let mut buffers = lock_recover(&self.buffers);
        let coeff_buffer = buffers.ensure(
            &self.device,
            BufferKind::Coefficients,
            dispatch.coeff_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            storage_limit,
        )?;
        let z_buffer = buffers.ensure(
            &self.device,
            BufferKind::ZValues,
            dispatch.z_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            storage_limit,
        )?;
        let output_buffer = buffers.ensure(
            &self.device,
            BufferKind::Output,
            dispatch.z_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            storage_limit,
        )?;
        let params_buffer = buffers.ensure_params(&self.device)?;
        let staging_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("st.mellin.gpu.staging"),
            size: dispatch.z_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        self.queue
            .write_buffer(coeff_buffer.as_ref(), 0, cast_slice(&coeffs));
        self.queue
            .write_buffer(z_buffer.as_ref(), 0, cast_slice(&zs));

        let params = ParamsPod {
            len: dispatch.len,
            count: dispatch.count,
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
            pass.dispatch_workgroups(dispatch.workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            output_buffer.as_ref(),
            0,
            staging_buffer.as_ref(),
            0,
            dispatch.z_bytes,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(0..dispatch.z_bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        let started = Instant::now();
        let mapped = loop {
            self.device.poll(wgpu::Maintain::Poll);
            let remaining = MAP_TIMEOUT.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(MellinGpuError::MapTimeout);
            }
            match receiver.recv_timeout(remaining.min(MAP_POLL_INTERVAL)) {
                Ok(mapped) => break mapped,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return Err(MellinGpuError::Map),
            }
        };
        match mapped {
            Ok(()) => {}
            Err(_) => return Err(MellinGpuError::Map),
        }
        let result = {
            let data = slice.get_mapped_range();
            readback_complex(&data, z_values.len())
        };
        staging_buffer.unmap();
        drop(buffers);
        result
    }

    fn validate_limits(&self, dispatch: GpuDispatch) -> Result<u64, MellinGpuError> {
        let limits = self.device.limits();
        validate_dispatch_limits(dispatch, &limits)
    }
}

fn validate_dispatch_limits(
    dispatch: GpuDispatch,
    limits: &wgpu::Limits,
) -> Result<u64, MellinGpuError> {
    if limits.max_compute_workgroup_size_x < WORKGROUP_SIZE
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
    {
        return Err(MellinGpuError::WorkgroupLimit {
            required: WORKGROUP_SIZE,
            size_x: limits.max_compute_workgroup_size_x,
            invocations: limits.max_compute_invocations_per_workgroup,
        });
    }
    if dispatch.workgroups > limits.max_compute_workgroups_per_dimension {
        return Err(MellinGpuError::DispatchLimit {
            workgroups: dispatch.workgroups,
            limit: limits.max_compute_workgroups_per_dimension,
        });
    }

    let storage_limit =
        u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
    validate_buffer_limit("coefficient buffer", dispatch.coeff_bytes, storage_limit)?;
    validate_buffer_limit("z-value buffer", dispatch.z_bytes, storage_limit)?;
    validate_buffer_limit("output buffer", dispatch.z_bytes, storage_limit)?;
    let uniform_limit =
        u64::from(limits.max_uniform_buffer_binding_size).min(limits.max_buffer_size);
    validate_buffer_limit(
        "parameter buffer",
        std::mem::size_of::<ParamsPod>() as u64,
        uniform_limit,
    )?;
    Ok(storage_limit)
}

fn validate_buffer_limit(kind: &'static str, bytes: u64, limit: u64) -> Result<(), MellinGpuError> {
    if bytes > limit {
        Err(MellinGpuError::BufferLimit { kind, bytes, limit })
    } else {
        Ok(())
    }
}

fn padded_buffer_size(bytes: u64, limit: u64) -> u64 {
    let requested = bytes.max(4);
    requested.saturating_add(requested / 10).min(limit)
}

fn readback_complex(data: &[u8], expected: usize) -> Result<Vec<ComplexScalar>, MellinGpuError> {
    let pods: &[ComplexPod] = try_cast_slice(data).map_err(|_| MellinGpuError::ReadbackLayout)?;
    if pods.len() != expected {
        return Err(MellinGpuError::ReadbackLength {
            expected,
            got: pods.len(),
        });
    }

    let mut result = Vec::new();
    result
        .try_reserve_exact(expected)
        .map_err(|_| MellinGpuError::Allocation {
            kind: "GPU readback",
            len: expected,
        })?;
    for (index, pod) in pods.iter().enumerate() {
        if !(pod.re.is_finite() && pod.im.is_finite()) {
            return Err(MellinGpuError::NonFiniteOutput {
                index,
                re: pod.re,
                im: pod.im,
            });
        }
        result.push(ComplexScalar::new(pod.re, pod.im));
    }
    Ok(result)
}

#[derive(Default)]
struct MellinGpuBuffers {
    coeffs: Option<CachedBuffer>,
    zs: Option<CachedBuffer>,
    output: Option<CachedBuffer>,
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
        kind: BufferKind,
        bytes: u64,
        usage: wgpu::BufferUsages,
        limit: u64,
    ) -> Result<Arc<wgpu::Buffer>, MellinGpuError> {
        validate_buffer_limit(kind.name(), bytes.max(4), limit)?;
        let slot = match kind {
            BufferKind::Coefficients => &mut self.coeffs,
            BufferKind::ZValues => &mut self.zs,
            BufferKind::Output => &mut self.output,
        };

        let needs_alloc = slot.as_ref().map(|buf| buf.bytes < bytes).unwrap_or(true);
        if needs_alloc {
            let padded = padded_buffer_size(bytes, limit);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(kind.label()),
                size: padded,
                usage,
                mapped_at_creation: false,
            });
            *slot = Some(CachedBuffer {
                buffer: Arc::new(buffer),
                bytes: padded,
            });
        }

        match slot.as_ref() {
            Some(cached) => Ok(cached.buffer.clone()),
            None => Err(MellinGpuError::BufferInvariant { kind: kind.name() }),
        }
    }

    fn ensure_params(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<Arc<wgpu::Buffer>, MellinGpuError> {
        if let Some(buf) = &self.params {
            return Ok(buf.clone());
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("st.mellin.gpu.params"),
            size: std::mem::size_of::<ParamsPod>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf = Arc::new(buffer);
        self.params = Some(buf.clone());
        Ok(buf)
    }
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    let payload = match payload.downcast::<String>() {
        Ok(message) => return *message,
        Err(payload) => payload,
    };
    let payload = match payload.downcast::<&'static str>() {
        Ok(message) => return (*message).to_string(),
        Err(payload) => payload,
    };

    if let Err(secondary_payload) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
        std::mem::forget(secondary_payload);
    }
    "non-string panic payload".to_string()
}

fn ensure_scalar_supported() -> Result<(), MellinGpuError> {
    if std::mem::size_of::<Scalar>() != std::mem::size_of::<f32>() {
        return Err(MellinGpuError::UnsupportedScalar);
    }
    Ok(())
}

fn to_pod_vec(
    values: &[ComplexScalar],
    kind: &'static str,
) -> Result<Vec<ComplexPod>, MellinGpuError> {
    ensure_scalar_supported()?;
    let mut pods = Vec::new();
    pods.try_reserve_exact(values.len())
        .map_err(|_| MellinGpuError::Allocation {
            kind,
            len: values.len(),
        })?;
    pods.extend(values.iter().map(|value| ComplexPod {
        re: value.re,
        im: value.im,
    }));
    Ok(pods)
}

const MELLIN_WGSL: &str = r#"
struct Complex { re: f32, im: f32, };

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
    if params.len == 0u {
        out[idx] = Complex(0.0, 0.0);
        return;
    }
    let z = zs[idx];
    var acc = coeffs[params.len - 1u];
    for (var k: u32 = params.len - 1u; k > 0u; k = k - 1u) {
        let coeff = coeffs[k - 1u];
        let prod = complex_mul(z, acc);
        acc = Complex(coeff.re + prod.re, coeff.im + prod.im);
    }
    out[idx] = acc;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_is_unavailable(error: &MellinGpuError) -> bool {
        matches!(
            error,
            MellinGpuError::NoAdapter
                | MellinGpuError::RequestDevice(_)
                | MellinGpuError::Shader(_)
                | MellinGpuError::RuntimePanic {
                    stage: "initialization",
                    ..
                }
        )
    }

    #[test]
    fn invalid_inputs_are_rejected_before_gpu_execution() {
        assert!(matches!(
            evaluate_weighted_series_many_gpu(&[], &[ComplexScalar::new(1.0, 0.0)]),
            Err(MellinGpuError::EmptySeries)
        ));
        assert!(matches!(
            evaluate_weighted_series_many_gpu(&[ComplexScalar::new(1.0, 0.0)], &[]),
            Err(MellinGpuError::EmptyBatch)
        ));
        assert!(matches!(
            evaluate_weighted_series_many_gpu(
                &[ComplexScalar::new(Scalar::NAN, 0.0)],
                &[ComplexScalar::new(1.0, 0.0)]
            ),
            Err(MellinGpuError::NonFiniteInput {
                kind: "weighted coefficient",
                index: 0,
                ..
            })
        ));
        assert!(matches!(
            evaluate_weighted_series_many_gpu(
                &[ComplexScalar::new(1.0, 0.0)],
                &[ComplexScalar::new(Scalar::INFINITY, 0.0)]
            ),
            Err(MellinGpuError::NonFiniteInput {
                kind: "z value",
                index: 0,
                ..
            })
        ));
    }

    #[test]
    fn dispatch_shape_uses_checked_lengths_and_bytes() {
        let dispatch = GpuDispatch::new(3, 65).unwrap();
        assert_eq!(dispatch.len, 3);
        assert_eq!(dispatch.count, 65);
        assert_eq!(dispatch.coeff_bytes, 24);
        assert_eq!(dispatch.z_bytes, 520);
        assert_eq!(dispatch.workgroups, 2);

        assert!(matches!(
            checked_byte_size("test", usize::MAX),
            Err(MellinGpuError::BufferSizeOverflow { .. })
        ));
        assert!(matches!(
            GpuDispatch::new(usize::MAX, 1),
            Err(MellinGpuError::LengthOverflow { .. })
                | Err(MellinGpuError::BufferSizeOverflow { .. })
        ));
    }

    #[test]
    fn dispatch_limits_are_checked_before_allocation() {
        let dispatch = GpuDispatch::new(1, 65).unwrap();

        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..Default::default()
        };
        assert!(matches!(
            validate_dispatch_limits(dispatch, &limits),
            Err(MellinGpuError::DispatchLimit {
                workgroups: 2,
                limit: 1
            })
        ));

        let limits = wgpu::Limits {
            max_compute_workgroup_size_x: WORKGROUP_SIZE - 1,
            ..Default::default()
        };
        assert!(matches!(
            validate_dispatch_limits(dispatch, &limits),
            Err(MellinGpuError::WorkgroupLimit { .. })
        ));

        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 4,
            max_buffer_size: 4,
            ..Default::default()
        };
        assert!(matches!(
            validate_dispatch_limits(dispatch, &limits),
            Err(MellinGpuError::BufferLimit {
                kind: "coefficient buffer",
                bytes: 8,
                limit: 4
            })
        ));
    }

    #[test]
    fn buffer_growth_never_exceeds_device_limit() {
        assert_eq!(padded_buffer_size(100, 200), 110);
        assert_eq!(padded_buffer_size(100, 105), 105);
        assert_eq!(padded_buffer_size(u64::MAX, u64::MAX), u64::MAX);
    }

    #[test]
    fn readback_validates_layout_length_and_finiteness() {
        let pods = [ComplexPod { re: 1.0, im: -2.0 }];
        let values = readback_complex(cast_slice(&pods), 1).unwrap();
        assert_eq!(values, vec![ComplexScalar::new(1.0, -2.0)]);

        assert!(matches!(
            readback_complex(&[0], 1),
            Err(MellinGpuError::ReadbackLayout)
        ));
        assert!(matches!(
            readback_complex(cast_slice(&pods), 2),
            Err(MellinGpuError::ReadbackLength {
                expected: 2,
                got: 1
            })
        ));

        let non_finite = [ComplexPod {
            re: Scalar::INFINITY,
            im: 0.0,
        }];
        assert!(matches!(
            readback_complex(cast_slice(&non_finite), 1),
            Err(MellinGpuError::NonFiniteOutput { index: 0, .. })
        ));
    }

    #[test]
    fn panic_boundary_and_poison_recovery_do_not_escape() {
        let error = guard_wgpu_panic("test", || -> Result<(), MellinGpuError> {
            panic!("gpu test panic")
        })
        .unwrap_err();
        assert!(matches!(
            error,
            MellinGpuError::RuntimePanic {
                stage: "test",
                ref message
            } if message == "gpu test panic"
        ));

        let value = Mutex::new(0_u8);
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let _guard = value.lock().unwrap();
            panic!("poison test mutex");
        }));
        assert!(value.is_poisoned());
        *lock_recover(&value) = 7;
        assert_eq!(*lock_recover(&value), 7);
    }

    #[test]
    fn panic_payload_with_panicking_destructor_is_contained() {
        struct PanicOnDrop;
        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                panic!("payload drop panic");
            }
        }

        let payload = catch_unwind(|| std::panic::panic_any(PanicOnDrop)).unwrap_err();
        assert_eq!(panic_payload_to_string(payload), "non-string panic payload");
    }

    #[test]
    fn raw_gpu_matches_cpu_when_an_adapter_is_available() {
        let weighted = [
            ComplexScalar::new(1.0, 0.0),
            ComplexScalar::new(2.0, -0.5),
            ComplexScalar::new(-1.0, 0.25),
        ];
        let z_values = [ComplexScalar::new(0.5, 0.0), ComplexScalar::new(0.2, -0.4)];

        match evaluate_weighted_series_many_gpu(&weighted, &z_values) {
            Ok(gpu) => {
                let cpu =
                    crate::zspace::evaluate_weighted_series_many(&weighted, &z_values).unwrap();
                assert_eq!(gpu.len(), cpu.len());
                for (gpu, cpu) in gpu.iter().zip(cpu.iter()) {
                    assert!((*gpu - *cpu).norm() < 1.0e-5);
                }
            }
            Err(error) if gpu_is_unavailable(&error) => {}
            Err(error) => panic!("unexpected GPU error: {error}"),
        }
    }

    #[test]
    fn raw_gpu_rejects_non_finite_readback_when_available() {
        let weighted = [
            ComplexScalar::new(Scalar::MAX, 0.0),
            ComplexScalar::new(Scalar::MAX, 0.0),
        ];
        let z_values = [ComplexScalar::new(2.0, 0.0)];

        match evaluate_weighted_series_many_gpu(&weighted, &z_values) {
            Err(MellinGpuError::NonFiniteOutput { index: 0, .. }) => {}
            Err(error) if gpu_is_unavailable(&error) => {}
            Err(error) => panic!("unexpected GPU error: {error}"),
            Ok(values) => panic!("expected non-finite output error, got {values:?}"),
        }
    }
}
