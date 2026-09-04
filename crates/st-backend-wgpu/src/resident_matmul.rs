//! Explicit float32 matmul with persistent GPU storage, shared by native and WASM.
//!
//! Dispatch only enqueues work. A readback snapshots output at request time; it
//! owns its staging buffer and remains valid if inputs or outputs change later.
//! This is an execution workspace, not an autograd tape or a backend selector.

use crate::runtime::{self, WgpuContext, WgpuRuntime, WgpuRuntimeError};
pub use crate::shader_sources::{MatmulAccumulation, MatmulKernel};
use bytemuck::{Pod, Zeroable};
use thiserror::Error;

const MAX_TILE_AXIS: u32 = 64;
pub const MAX_REPETITIONS: u32 = 1024;

#[derive(Debug, Error)]
pub enum MatmulError {
    #[error("resident matmul requires nonzero, u32-addressable dimensions and buffers")]
    InvalidShape,
    #[error("tile axes must be in 1..=64 with at most 256 output lanes")]
    InvalidTile,
    #[error("register_2x2 requires even output tile dimensions")]
    UnsupportedKernelTile,
    #[error("resident matmul exceeds device limit: {0}")]
    DeviceLimit(&'static str),
    #[error("{operand} requires {expected} elements, received {actual}")]
    InputLength {
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("upload both operands before dispatch")]
    MissingInputs,
    #[error("dispatch current inputs before reading or copying the output")]
    StaleOutput,
    #[error("dispatch repetitions must be between 1 and {MAX_REPETITIONS}")]
    InvalidRepetitions,
    #[error("GPU copy requires matching matrix dimensions and the same device/queue")]
    IncompatibleCopy,
    #[error("input generation counter exhausted")]
    GenerationOverflow,
    #[error(transparent)]
    Runtime(#[from] WgpuRuntimeError),
}

/// Checked workgroup geometry, ordered M/N/K (unlike matrix shape M/K/N).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulTile([u32; 3]);

impl Default for MatmulTile {
    fn default() -> Self {
        Self([8, 8, 16])
    }
}

impl MatmulTile {
    pub fn new(rows: u32, cols: u32, inner: u32) -> Result<Self, MatmulError> {
        if [rows, cols, inner]
            .iter()
            .any(|&v| v == 0 || v > MAX_TILE_AXIS)
            || u64::from(rows) * u64::from(cols) > 256
        {
            return Err(MatmulError::InvalidTile);
        }
        Ok(Self([rows, cols, inner]))
    }

    pub fn dimensions(self) -> [u32; 3] {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulShape {
    rows: u32,
    inner: u32,
    cols: u32,
}

impl MatmulShape {
    pub fn new(rows: usize, inner: usize, cols: usize) -> Result<Self, MatmulError> {
        let dim = |x| {
            u32::try_from(x)
                .ok()
                .filter(|&v| v > 0 && v <= u32::MAX - MAX_TILE_AXIS)
        };
        let shape = Self {
            rows: dim(rows).ok_or(MatmulError::InvalidShape)?,
            inner: dim(inner).ok_or(MatmulError::InvalidShape)?,
            cols: dim(cols).ok_or(MatmulError::InvalidShape)?,
        };
        for (a, b) in [
            (shape.rows, shape.inner),
            (shape.inner, shape.cols),
            (shape.rows, shape.cols),
        ] {
            let elements = a.checked_mul(b).ok_or(MatmulError::InvalidShape)?;
            runtime::checked_byte_len::<f32>("resident matmul", elements as usize)?;
        }
        Ok(shape)
    }

    pub fn dimensions(self) -> (usize, usize, usize) {
        (self.rows as usize, self.inner as usize, self.cols as usize)
    }

    fn lengths(self) -> [usize; 3] {
        [
            (self.rows * self.inner) as usize,
            (self.inner * self.cols) as usize,
            (self.rows * self.cols) as usize,
        ]
    }

    fn validate(
        self,
        limits: &wgpu::Limits,
        tile: MatmulTile,
        kernel: MatmulKernel,
    ) -> Result<(), MatmulError> {
        let [tile_m, tile_n, tile_k] = tile.dimensions();
        let [wg_x, wg_y, _] = kernel
            .workgroup_size(tile.dimensions())
            .ok_or(MatmulError::UnsupportedKernelTile)?;
        for len in self.lengths() {
            let bytes = runtime::checked_byte_len::<f32>("resident matmul", len)?;
            if bytes > limits.max_buffer_size
                || bytes > u64::from(limits.max_storage_buffer_binding_size)
            {
                return Err(MatmulError::DeviceLimit("storage buffer bytes"));
            }
        }
        if self.rows.div_ceil(tile_m) > limits.max_compute_workgroups_per_dimension
            || self.cols.div_ceil(tile_n) > limits.max_compute_workgroups_per_dimension
        {
            return Err(MatmulError::DeviceLimit("dispatch workgroups"));
        }
        if limits.max_compute_workgroup_size_x < wg_x
            || limits.max_compute_workgroup_size_y < wg_y
            || limits.max_compute_invocations_per_workgroup < wg_x * wg_y
            || limits.max_compute_workgroup_storage_size < (tile_m + tile_n) * tile_k * 4
            || limits.max_storage_buffers_per_shader_stage < 6
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_bind_groups < 1
            || limits.max_bindings_per_bind_group < 7
            || limits.max_uniform_buffer_binding_size < 32
        {
            return Err(MatmulError::DeviceLimit("workgroup or binding limits"));
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    rows: u32,
    cols: u32,
    inner: u32,
    flags: u32,
    output_scale: f32,
    padding: [f32; 3],
}

pub struct ResidentMatmul {
    runtime: WgpuRuntime,
    shape: MatmulShape,
    tile: MatmulTile,
    kernel: MatmulKernel,
    accumulation: MatmulAccumulation,
    lhs: wgpu::Buffer,
    rhs: wgpu::Buffer,
    output: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    inputs_ready: [bool; 2],
    generation: u64,
    output_generation: Option<u64>,
}

impl ResidentMatmul {
    pub fn new(runtime: WgpuRuntime, shape: MatmulShape) -> Result<Self, MatmulError> {
        Self::with_tile(runtime, shape, MatmulTile::default())
    }

    pub fn with_tile(
        runtime: WgpuRuntime,
        shape: MatmulShape,
        tile: MatmulTile,
    ) -> Result<Self, MatmulError> {
        Self::with_kernel(runtime, shape, tile, MatmulKernel::Scalar)
    }

    pub fn with_kernel(
        runtime: WgpuRuntime,
        shape: MatmulShape,
        tile: MatmulTile,
        kernel: MatmulKernel,
    ) -> Result<Self, MatmulError> {
        Self::with_options(runtime, shape, tile, kernel, MatmulAccumulation::Sequential)
    }

    pub fn with_options(
        runtime: WgpuRuntime,
        shape: MatmulShape,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, MatmulError> {
        let context = runtime.context();
        let device = context.device();
        shape.validate(&device.limits(), tile, kernel)?;
        let [lhs_len, rhs_len, out_len] = shape.lengths();
        let storage = wgpu::BufferUsages::STORAGE;
        let lhs = runtime::empty_buffer::<f32>(
            device,
            "resident.lhs",
            lhs_len,
            storage | wgpu::BufferUsages::COPY_DST,
        )?;
        let rhs = runtime::empty_buffer::<f32>(
            device,
            "resident.rhs",
            rhs_len,
            storage | wgpu::BufferUsages::COPY_DST,
        )?;
        let output = runtime::empty_buffer::<f32>(
            device,
            "resident.out",
            out_len,
            storage | wgpu::BufferUsages::COPY_SRC,
        )?;
        let dummy = runtime::upload_slice(device, "resident.unused", &[0.0f32], storage)?;
        let params = Uniforms {
            rows: shape.rows,
            cols: shape.cols,
            inner: shape.inner,
            flags: 0,
            output_scale: 1.0,
            padding: [0.0; 3],
        };
        let uniform = runtime::upload_slice(
            device,
            "resident.params",
            &[params],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let entries: Vec<_> = (0..7)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 6 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 2,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident.matmul.bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resident.matmul.layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident.matmul.shader"),
            source: wgpu::ShaderSource::Wgsl(
                crate::shader_sources::dense_matmul_source_with_options(
                    tile.dimensions(),
                    false,
                    kernel,
                    accumulation,
                )
                .map_err(|_| MatmulError::UnsupportedKernelTile)?
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resident.matmul"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let buffers = [&lhs, &rhs, &output, &dummy, &dummy, &dummy, &uniform];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident.matmul.bound"),
            layout: &layout,
            entries: &entries,
        });
        Ok(Self {
            runtime,
            shape,
            tile,
            kernel,
            accumulation,
            lhs,
            rhs,
            output,
            pipeline,
            bind_group,
            inputs_ready: [false; 2],
            generation: 0,
            output_generation: None,
        })
    }

    pub fn shape(&self) -> MatmulShape {
        self.shape
    }
    pub fn tile(&self) -> MatmulTile {
        self.tile
    }
    pub fn kernel(&self) -> MatmulKernel {
        self.kernel
    }
    pub fn accumulation(&self) -> MatmulAccumulation {
        self.accumulation
    }
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.kernel
            .workgroup_size(self.tile.dimensions())
            .expect("validated workspace geometry")
    }
    pub fn outputs_per_thread(&self) -> [u32; 2] {
        self.kernel.outputs_per_thread()
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.runtime.adapter_info()
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    /// Logical freshness of the queued output, not a GPU completion receipt.
    pub fn output_is_current(&self) -> bool {
        self.output_generation == Some(self.generation)
    }

    fn check_length(
        &self,
        operand: &'static str,
        actual: usize,
        index: usize,
    ) -> Result<(), MatmulError> {
        let expected = self.shape.lengths()[index];
        if actual != expected {
            return Err(MatmulError::InputLength {
                operand,
                expected,
                actual,
            });
        }
        Ok(())
    }

    fn invalidate(&mut self) -> Result<(), MatmulError> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(MatmulError::GenerationOverflow)?;
        self.output_generation = None;
        Ok(())
    }

    /// Validates both inputs before changing either buffer or generation.
    pub fn upload(&mut self, lhs: &[f32], rhs: &[f32]) -> Result<(), MatmulError> {
        self.check_length("lhs", lhs.len(), 0)?;
        self.check_length("rhs", rhs.len(), 1)?;
        self.invalidate()?;
        let queue = self.runtime.context().queue();
        queue.write_buffer(&self.lhs, 0, bytemuck::cast_slice(lhs));
        queue.write_buffer(&self.rhs, 0, bytemuck::cast_slice(rhs));
        self.inputs_ready = [true; 2];
        Ok(())
    }

    pub fn upload_rhs(&mut self, rhs: &[f32]) -> Result<(), MatmulError> {
        self.check_length("rhs", rhs.len(), 1)?;
        self.invalidate()?;
        self.runtime
            .context()
            .queue()
            .write_buffer(&self.rhs, 0, bytemuck::cast_slice(rhs));
        self.inputs_ready[1] = true;
        Ok(())
    }

    /// Connect workspaces without a CPU round trip; the source must be current.
    pub fn set_lhs_from(&mut self, source: &Self) -> Result<(), MatmulError> {
        if (self.shape.rows, self.shape.inner) != (source.shape.rows, source.shape.cols)
            || !self
                .runtime
                .context()
                .shares_handles_with(source.runtime.context())
        {
            return Err(MatmulError::IncompatibleCopy);
        }
        source.require_current()?;
        self.invalidate()?;
        let context = self.runtime.context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&source.output, 0, &self.lhs, 0, self.lhs.size());
        context.queue().submit(Some(encoder.finish()));
        self.inputs_ready[0] = true;
        Ok(())
    }

    pub fn dispatch(&mut self, repetitions: u32) -> Result<u64, MatmulError> {
        if repetitions == 0 || repetitions > MAX_REPETITIONS {
            return Err(MatmulError::InvalidRepetitions);
        }
        if self.inputs_ready != [true; 2] {
            return Err(MatmulError::MissingInputs);
        }
        let context = self.runtime.context();
        let [tile_m, tile_n, _] = self.tile.dimensions();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resident.matmul.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            for _ in 0..repetitions {
                pass.dispatch_workgroups(
                    self.shape.cols.div_ceil(tile_n),
                    self.shape.rows.div_ceil(tile_m),
                    1,
                );
            }
        }
        context.queue().submit(Some(encoder.finish()));
        self.output_generation = Some(self.generation);
        Ok(self.generation)
    }

    fn require_current(&self) -> Result<(), MatmulError> {
        if !self.output_is_current() {
            return Err(MatmulError::StaleOutput);
        }
        Ok(())
    }

    /// Enqueues the copy now, not when the returned readback is later awaited.
    pub fn snapshot(&self) -> Result<MatmulReadback, MatmulError> {
        self.require_current()?;
        let context = self.runtime.context();
        let staging = runtime::empty_buffer::<f32>(
            context.device(),
            "resident.snapshot",
            self.shape.lengths()[2],
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.output, 0, &staging, 0, staging.size());
        context.queue().submit(Some(encoder.finish()));
        Ok(MatmulReadback {
            context: context.clone(),
            staging,
            shape: self.shape,
            generation: self.generation,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn synchronize(&self) -> Result<(), MatmulError> {
        let context = self.runtime.context();
        runtime::submit_with_timeout(
            context.device(),
            context.queue(),
            [],
            std::time::Duration::from_secs(30),
            "resident.matmul",
        )?;
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    pub fn synchronize_async(
        &self,
    ) -> Result<impl std::future::Future<Output = Result<(), MatmulError>> + 'static, MatmulError>
    {
        let context = self.runtime.context().clone();
        // wgpu 0.20's browser on_submitted_work_done is unimplemented. Mapping
        // a four-byte copy queued after the work gives an actual completion fence.
        let staging = runtime::empty_buffer::<f32>(
            context.device(),
            "resident.fence",
            1,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.output, 0, &staging, 0, 4);
        context.queue().submit(Some(encoder.finish()));
        Ok(async move { map_staging(context, staging).await.map(|_| ()) })
    }
}

pub struct MatmulReadback {
    context: WgpuContext,
    staging: wgpu::Buffer,
    shape: MatmulShape,
    generation: u64,
}

impl MatmulReadback {
    pub fn shape(&self) -> MatmulShape {
        self.shape
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<Vec<f32>, MatmulError> {
        let bytes = runtime::map_read_bytes_with_timeout(
            self.context.device(),
            &self.staging,
            0..self.staging.size(),
            std::time::Duration::from_secs(30),
            "resident.snapshot",
        )?;
        Ok(decode_f32(&bytes))
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<Vec<f32>, MatmulError> {
        map_staging(self.context, self.staging).await
    }
}

#[cfg(target_arch = "wasm32")]
async fn map_staging(context: WgpuContext, staging: wgpu::Buffer) -> Result<Vec<f32>, MatmulError> {
    let slice = staging.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    receiver
        .await
        .map_err(|_| WgpuRuntimeError::MapCallbackDisconnected {
            resource: "resident.snapshot".into(),
        })?
        .map_err(|error| WgpuRuntimeError::Map {
            resource: "resident.snapshot".into(),
            message: error.to_string(),
        })?;
    let mapped = slice.get_mapped_range();
    let output = decode_f32(&mapped);
    drop(mapped);
    staging.unmap();
    // Keep the device/queue alive throughout the asynchronous readback.
    drop(context);
    Ok(output)
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    #[allow(clippy::chunks_exact_to_as_chunks)]
    bytes
        .chunks_exact(4)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_checks_shader_addressing_and_device_limits() {
        for shape in [(0, 2, 3), (1, usize::MAX, 1), (65536, 65536, 1)] {
            assert!(MatmulShape::new(shape.0, shape.1, shape.2).is_err());
        }
        let shape = MatmulShape::new(7, 63, 65).unwrap();
        assert_eq!(shape.lengths(), [441, 4095, 455]);
        let mut limits = wgpu::Limits::default();
        shape
            .validate(&limits, MatmulTile::default(), MatmulKernel::Scalar)
            .unwrap();
        limits.max_storage_buffer_binding_size = 1024;
        assert!(shape
            .validate(&limits, MatmulTile::default(), MatmulKernel::Scalar)
            .is_err());
        limits = wgpu::Limits::default();
        limits.max_compute_workgroups_per_dimension = 1;
        assert!(shape
            .validate(&limits, MatmulTile::default(), MatmulKernel::Scalar)
            .is_err());
    }

    #[test]
    fn shared_shader_variants_validate() {
        for tile in [
            [8, 8, 16],
            [16, 16, 64],
            [64, 4, 16],
            [4, 64, 16],
            [3, 5, 7],
        ] {
            for int8 in [false, true] {
                for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
                    for accumulation in [
                        MatmulAccumulation::Sequential,
                        MatmulAccumulation::Tiled,
                        MatmulAccumulation::Compensated,
                    ] {
                        let Ok(source) = crate::shader_sources::dense_matmul_source_with_options(
                            tile,
                            int8,
                            kernel,
                            accumulation,
                        ) else {
                            assert!(kernel.workgroup_size(tile).is_none());
                            continue;
                        };
                        let module = naga::front::wgsl::parse_str(&source).unwrap();
                        naga::valid::Validator::new(
                            naga::valid::ValidationFlags::all(),
                            naga::valid::Capabilities::all(),
                        )
                        .validate(&module)
                        .unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn output_tile_and_invocation_geometry_agree() {
        assert_eq!(
            MatmulAccumulation::default(),
            MatmulAccumulation::Sequential
        );
        assert_eq!(
            "compensated"
                .parse::<MatmulAccumulation>()
                .unwrap()
                .as_str(),
            "compensated"
        );
        assert!("auto".parse::<MatmulAccumulation>().is_err());
        for m in 1..=64 {
            for n in 1..=64 {
                let Ok(tile) = MatmulTile::new(m, n, 16) else {
                    continue;
                };
                for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
                    let Some([x, y, z]) = kernel.workgroup_size(tile.dimensions()) else {
                        assert!(!m.is_multiple_of(2) || !n.is_multiple_of(2));
                        continue;
                    };
                    let [thread_m, thread_n] = kernel.outputs_per_thread();
                    assert_eq!([y * thread_m, x * thread_n, z], [m, n, 1]);
                }
            }
        }
        let blocked = MatmulTile::new(16, 16, 16).unwrap();
        assert_eq!(
            MatmulKernel::Register2x2.workgroup_size(blocked.dimensions()),
            Some([8, 8, 1])
        );
        assert_eq!(MatmulKernel::Register2x2.outputs_per_thread(), [2, 2]);
        assert!("auto".parse::<MatmulKernel>().is_err());
        let shape = MatmulShape::new(7, 63, 65).unwrap();
        let mut limits = wgpu::Limits {
            max_compute_invocations_per_workgroup: 64,
            ..wgpu::Limits::default()
        };
        shape
            .validate(&limits, blocked, MatmulKernel::Register2x2)
            .unwrap();
        assert!(shape
            .validate(&limits, blocked, MatmulKernel::Scalar)
            .is_err());
        limits.max_compute_invocations_per_workgroup = 63;
        assert!(shape
            .validate(&limits, blocked, MatmulKernel::Register2x2)
            .is_err());
    }

    #[test]
    fn tile_checks_are_structural_and_device_specific() {
        for [m, n, k] in [
            [0, 8, 16],
            [8, 0, 16],
            [8, 8, 0],
            [65, 1, 1],
            [8, 8, 65],
            [32, 32, 16],
            [u32::MAX, 1, 1],
        ] {
            assert!(MatmulTile::new(m, n, k).is_err());
        }
        let tile = MatmulTile::new(16, 16, 64).unwrap();
        let shape = MatmulShape::new(7, 63, 65).unwrap();
        let mut limits = wgpu::Limits::default();
        shape.validate(&limits, tile, MatmulKernel::Scalar).unwrap();
        limits.max_compute_workgroup_storage_size = 4096;
        assert!(shape.validate(&limits, tile, MatmulKernel::Scalar).is_err());
        shape
            .validate(&limits, MatmulTile::default(), MatmulKernel::Scalar)
            .unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn resident_browser_long_k_frozen_cells_match_reference_when_enabled() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        fn next_value(state: &mut u32) -> f32 {
            *state ^= *state << 13;
            *state ^= *state >> 17;
            *state ^= *state << 5;
            (f64::from(*state) / 4_294_967_296.0 - 0.5) as f32
        }
        let (runtime, _) =
            runtime::ensure_default_runtime_blocking("resident.long-k.test").unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        // Browser xorshift inputs, not the Python benchmark's random.Random inputs.
        // These sequential failures were captured at 128x3072x768 on Apple WebGPU.
        for (seed, row, col, frozen_reference) in [
            (29, 124, 630, 0.008646938055694997),
            (43, 26, 183, 0.01428703216586541),
        ] {
            let mut state = seed;
            let lhs: Vec<_> = (0..(row + 1) * 3072)
                .map(|_| next_value(&mut state))
                .skip(row * 3072)
                .collect();
            state = seed + 1;
            let rhs: Vec<_> = (0..3072 * 768)
                .map(|_| next_value(&mut state))
                .enumerate()
                .filter_map(|(i, value)| (i % 768 == col).then_some(value))
                .collect();
            let expected: f64 = lhs
                .iter()
                .zip(&rhs)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            assert!((expected - frozen_reference).abs() < 1e-12);
            for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
                for accumulation in [MatmulAccumulation::Tiled, MatmulAccumulation::Compensated] {
                    let mut workspace = ResidentMatmul::with_options(
                        runtime.clone(),
                        MatmulShape::new(1, 3072, 1).unwrap(),
                        MatmulTile::new(16, 16, 16).unwrap(),
                        kernel,
                        accumulation,
                    )
                    .unwrap();
                    workspace.upload(&lhs, &rhs).unwrap();
                    workspace.dispatch(1).unwrap();
                    let actual = f64::from(workspace.snapshot().unwrap().read().unwrap()[0]);
                    assert!(
                        (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                        "seed={seed} {kernel:?} {accumulation:?}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn rounded_add_bits_matches_f32_when_enabled() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let (runtime, _) = runtime::ensure_default_runtime_blocking("rounded-add.test").unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let boundary = [
            0, 1, 0x7fffff, 0x800000, 0x3f800000, 0x33800000, 0x3f800001, 0x7f7fffff, 0x7f800000,
            0x7fc00000,
        ];
        let mut pairs = Vec::new();
        for a in boundary {
            for b in boundary {
                for sign_a in [0, 0x80000000] {
                    for sign_b in [0, 0x80000000] {
                        pairs.push([a | sign_a, b | sign_b]);
                    }
                }
            }
        }
        let mut state = 17u32;
        for _ in 0..16_384 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let a = state;
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            pairs.push([a, state]);
        }
        let source = format!(
            "{}\n{}",
            crate::shader_sources::ROUNDED_ADD_WGSL,
            r#"
@group(0) @binding(0) var<storage, read> pairs: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&pairs)) {
        output[id.x] = rounded_add_bits(pairs[id.x].x, pairs[id.x].y);
    }
}"#
        );
        let ctx = runtime.context();
        let device = ctx.device();
        let input = runtime::upload_slice(
            device,
            "rounded-add.inputs",
            &pairs,
            wgpu::BufferUsages::STORAGE,
        )
        .unwrap();
        let output = runtime::empty_buffer::<u32>(
            device,
            "rounded-add.output",
            pairs.len(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
        .unwrap();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rounded-add.shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rounded-add.pipeline"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rounded-add.group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((pairs.len() as u32).div_ceil(64), 1, 1);
        }
        ctx.queue().submit(Some(encoder.finish()));
        let actual = runtime::read_buffer::<u32>(
            device,
            ctx.queue(),
            &output,
            pairs.len(),
            "rounded-add.read",
        )
        .unwrap();
        for ([a, b], actual) in pairs.into_iter().zip(actual) {
            let expected = f32::from_bits(a) + f32::from_bits(b);
            if expected.is_nan() {
                assert!(f32::from_bits(actual).is_nan());
            } else {
                assert_eq!(actual, expected.to_bits(), "a={a:08x} b={b:08x}");
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn resident_compensation_recovers_small_terms_when_enabled() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let (runtime, _) =
            runtime::ensure_default_runtime_blocking("resident.compensation.test").unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let shape = MatmulShape::new(3, 3075, 5).unwrap();
        let lhs: Vec<_> = (0..3 * 3075)
            .map(|index| match index % 3 {
                0 => 100_000_000.0,
                1 => 1.0,
                _ => -100_000_000.0,
            })
            .collect();
        let rhs = vec![1.0; 3075 * 5];
        for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
            for accumulation in [
                MatmulAccumulation::Sequential,
                MatmulAccumulation::Compensated,
            ] {
                let mut workspace = ResidentMatmul::with_options(
                    runtime.clone(),
                    shape,
                    MatmulTile::new(16, 16, 16).unwrap(),
                    kernel,
                    accumulation,
                )
                .unwrap();
                workspace.upload(&lhs, &rhs).unwrap();
                workspace.dispatch(2).unwrap();
                let values = workspace.snapshot().unwrap().read().unwrap();
                assert_eq!(workspace.accumulation(), accumulation);
                if accumulation == MatmulAccumulation::Compensated {
                    assert_eq!(values, vec![1025.0; 15], "{kernel:?}");
                } else {
                    assert!(values.iter().all(|v| v.is_finite()));
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn resident_tiles_match_f64_reference_when_enabled() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let (runtime, _) = runtime::ensure_default_runtime_blocking("resident-tiles").unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let shape = MatmulShape::new(7, 63, 65).unwrap();
        let lhs: Vec<_> = (0..7 * 63).map(|i| (i % 13) as f32 / 13.0 - 0.5).collect();
        let rhs: Vec<_> = (0..63 * 65).map(|i| (i % 17) as f32 / 17.0 - 0.5).collect();
        for [m, n, k] in [
            [8, 16, 16],
            [16, 8, 16],
            [16, 16, 16],
            [16, 16, 32],
            [16, 16, 64],
            [3, 5, 7],
        ] {
            let tile = MatmulTile::new(m, n, k).unwrap();
            for kernel in [MatmulKernel::Scalar, MatmulKernel::Register2x2] {
                for accumulation in [
                    MatmulAccumulation::Sequential,
                    MatmulAccumulation::Tiled,
                    MatmulAccumulation::Compensated,
                ] {
                    let candidate = ResidentMatmul::with_options(
                        runtime.clone(),
                        shape,
                        tile,
                        kernel,
                        accumulation,
                    );
                    if kernel.workgroup_size(tile.dimensions()).is_none() {
                        assert!(matches!(candidate, Err(MatmulError::UnsupportedKernelTile)));
                        continue;
                    }
                    let mut workspace = candidate.unwrap();
                    assert_eq!(workspace.tile(), tile);
                    assert_eq!(workspace.kernel(), kernel);
                    assert_eq!(workspace.accumulation(), accumulation);
                    workspace.upload(&lhs, &rhs).unwrap();
                    workspace.dispatch(2).unwrap();
                    let actual = workspace.snapshot().unwrap().read().unwrap();
                    for row in 0..7 {
                        for col in 0..65 {
                            let expected: f64 = (0..63)
                                .map(|i| {
                                    f64::from(lhs[row * 63 + i]) * f64::from(rhs[i * 65 + col])
                                })
                                .sum();
                            assert!(
                                (f64::from(actual[row * 65 + col]) - expected).abs() < 1e-5,
                                "tile={tile:?} accumulation={accumulation:?} row={row} col={col}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn resident_device_reuse_snapshots_and_chaining() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let (runtime, _) = runtime::ensure_default_runtime_blocking("resident-test").unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let mut workspace =
            ResidentMatmul::new(runtime.clone(), MatmulShape::new(7, 63, 65).unwrap()).unwrap();
        assert!(matches!(
            workspace.dispatch(1),
            Err(MatmulError::MissingInputs)
        ));
        assert!(matches!(
            workspace.snapshot(),
            Err(MatmulError::StaleOutput)
        ));
        let lhs: Vec<_> = (0..7 * 63).map(|i| (i % 13) as f32 / 13.0 - 0.5).collect();
        let rhs: Vec<_> = (0..63 * 65).map(|i| (i % 17) as f32 / 17.0 - 0.5).collect();
        workspace.upload(&lhs, &rhs).unwrap();
        workspace.dispatch(3).unwrap();
        let first = workspace.snapshot().unwrap();
        let generation = workspace.generation();
        assert!(workspace.upload(&lhs, &rhs[..3]).is_err());
        assert_eq!(generation, workspace.generation());
        assert!(workspace.output_is_current());
        workspace.upload(&vec![0.0; lhs.len()], &rhs).unwrap();
        assert!(matches!(
            workspace.snapshot(),
            Err(MatmulError::StaleOutput)
        ));
        workspace.dispatch(1).unwrap();
        let second = workspace.snapshot().unwrap();
        // Read in reverse order after new work: the first snapshot must not change.
        assert!(second.read().unwrap().iter().all(|&x| x == 0.0));
        let actual = first.read().unwrap();
        for row in 0..7 {
            for col in 0..65 {
                let expected: f64 = (0..63)
                    .map(|k| f64::from(lhs[row * 63 + k]) * f64::from(rhs[k * 65 + col]))
                    .sum();
                assert!((f64::from(actual[row * 65 + col]) - expected).abs() < 1e-5);
            }
        }
        workspace.upload(&lhs, &rhs).unwrap();
        workspace.dispatch(1).unwrap();
        let mut next = ResidentMatmul::new(runtime, MatmulShape::new(7, 65, 1).unwrap()).unwrap();
        next.upload_rhs(&[1.0; 65]).unwrap();
        next.set_lhs_from(&workspace).unwrap();
        next.dispatch(1).unwrap();
        let snapshot = next.snapshot().unwrap();
        drop(next);
        drop(workspace);
        for (row, value) in snapshot.read().unwrap().into_iter().enumerate() {
            let expected: f32 = actual[row * 65..(row + 1) * 65].iter().sum();
            assert!((value - expected).abs() < 1e-4);
        }
    }
}
