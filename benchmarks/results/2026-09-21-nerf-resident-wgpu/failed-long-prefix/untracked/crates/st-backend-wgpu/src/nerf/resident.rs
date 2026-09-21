//! Immutable ray sampling and compositing on an existing tensor device.
use crate::{
    resident_tensor::{storage_limit, ResidentTensor, TensorDevice, TensorError},
    runtime::{self, Shared, WgpuRuntimeError},
};
use st_kernel_contracts::layout::NdLayout;
use thiserror::Error;

const SAMPLER: &str = include_str!("../shaders/nerf_volume_utils.wgsl");
const COMPOSITOR: &str = include_str!("../shaders/nerf_raymarch.wgsl");

#[derive(Debug, Error)]
pub enum NerfError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Runtime(#[from] WgpuRuntimeError),
    #[error("NeRF ray {index} has non-finite coordinates, unordered bounds or an unrepresentable f32 span")]
    InvalidRay { index: usize },
    #[error("NeRF requires nonempty rays and a positive sample count")]
    Empty,
    #[error("NeRF field must have shape [rays, samples_per_ray, 4] (sigma, r, g, b)")]
    FieldShape,
}

/// A physical ray, parameterized as origin + direction * t. Directions need not
/// be normalized. Integration uses t widths, not direction-vector lengths.
#[derive(Clone, Copy, Debug)]
pub struct NerfRay {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
    pub near: f32,
    pub far: f32,
}

/// GPU-local deterministic jitter is counter-based, not the CPU trainer's RNG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaySampling {
    Midpoint,
    Stratified { seed: u32 },
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRay {
    origin: [f32; 3],
    padding: f32,
    direction: [f32; 3],
    near: f32,
    far: f32,
    tail: [f32; 3],
}

impl NerfRay {
    fn checked(self, index: usize) -> Result<GpuRay, NerfError> {
        if !self
            .origin
            .iter()
            .chain(&self.direction)
            .all(|v| v.is_finite())
            || !self.near.is_finite()
            || !self.far.is_finite()
            || self.far < self.near
            || !(self.far - self.near).is_finite()
        {
            return Err(NerfError::InvalidRay { index });
        }
        Ok(GpuRay {
            origin: self.origin,
            padding: 0.0,
            direction: self.direction,
            near: self.near,
            far: self.far,
            tail: [0.0; 3],
        })
    }
}

#[derive(Debug)]
struct Pipelines {
    sampling: wgpu::ComputePipeline,
    compositing: wgpu::ComputePipeline,
    device: TensorDevice,
}

/// Reusable forward-only NeRF stages. No discovery, fallback, filesystem reads
/// or host observation occur in sample/composite. All intermediate tensors use
/// the existing ResidentTensor ownership, view and finite-value contracts.
#[derive(Clone, Debug)]
pub struct ResidentNerf(Shared<Pipelines>);

/// Immutable, validated rays uploaded once and reusable for many sample draws.
#[derive(Clone, Debug)]
pub struct ResidentRays {
    buffer: Shared<wgpu::Buffer>,
    count: usize,
    device: TensorDevice,
}

impl ResidentRays {
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// A frozen sampling result. Later draws cannot alter these tensors or guards.
#[derive(Clone, Debug)]
pub struct NerfSamples {
    points: ResidentTensor,
    widths: ResidentTensor,
    shape: [usize; 2],
}

impl NerfSamples {
    /// Logical shape [ray_count, samples_per_ray].
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    /// Shape [rays, samples, 4], channels [x, y, z, t].
    pub fn points(&self) -> &ResidentTensor {
        &self.points
    }
    /// A zero-copy [rays, samples, 3] position view, suitable as an NN graph input.
    pub fn positions(&self) -> Result<ResidentTensor, TensorError> {
        self.points.narrow(2, 0, 3)
    }
    /// Full bin widths, shape [rays, samples], independent of jitter.
    pub fn widths(&self) -> &ResidentTensor {
        &self.widths
    }
}

fn dimensions(rows: usize, samples: usize, limits: &wgpu::Limits) -> Result<usize, NerfError> {
    if rows == 0 || samples == 0 {
        return Err(NerfError::Empty);
    }
    let count = rows
        .checked_mul(samples)
        .ok_or(TensorError::Limit("NeRF sample count"))?;
    let elements = count
        .checked_mul(4)
        .ok_or(TensorError::Limit("NeRF sample storage"))?;
    storage_limit(elements, limits)?;
    storage_limit(
        rows.checked_mul(12)
            .ok_or(TensorError::Limit("NeRF ray storage"))?,
        limits,
    )?;
    if rows.div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize {
        return Err(TensorError::Limit("NeRF dispatch grid").into());
    }
    Ok(count)
}

fn bind(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nerf.bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &buffers
            .iter()
            .map(|&(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    })
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    binding: &wgpu::BindGroup,
    rows: usize,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("nerf.dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, binding, &[]);
    pass.dispatch_workgroups((rows as u32).div_ceil(64), 1, 1);
}

impl ResidentNerf {
    /// Use the same TensorDevice as a resident NN graph to avoid device copies.
    /// Embedded WGSL is identical on native and wasm32 targets.
    pub fn new(device: TensorDevice) -> Result<Self, NerfError> {
        let gpu = device.runtime().context().device();
        let limits = gpu.limits();
        if limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_uniform_buffer_binding_size < 16
        {
            return Err(TensorError::Limit("NeRF pipeline").into());
        }
        let pipeline = |label, source: &str, entry_point| {
            let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point,
                compilation_options: Default::default(),
            })
        };
        Ok(Self(Shared::new(Pipelines {
            sampling: pipeline("nerf.sampling", SAMPLER, "checked_main"),
            compositing: pipeline("nerf.compositing", COMPOSITOR, "resident_main"),
            device,
        })))
    }

    pub fn tensor_device(&self) -> &TensorDevice {
        &self.0.device
    }

    pub fn upload_rays(&self, rays: &[NerfRay]) -> Result<ResidentRays, NerfError> {
        let gpu = self.0.device.runtime().context().device();
        dimensions(rays.len(), 1, &gpu.limits())?;
        let packed = rays
            .iter()
            .enumerate()
            .map(|(index, &ray)| ray.checked(index))
            .collect::<Result<Vec<_>, _>>()?;
        let buffer = runtime::upload_slice(gpu, "nerf.rays", &packed, wgpu::BufferUsages::STORAGE)?;
        Ok(ResidentRays {
            buffer: Shared::new(buffer),
            count: rays.len(),
            device: self.0.device.clone(),
        })
    }

    /// Submit one draw and its guard captures without any readback. Arithmetic
    /// is f32: unrepresentable spans are rejected on upload; non-finite sample
    /// positions or underflowed positive widths reject the eventual snapshot.
    pub fn sample(
        &self,
        rays: &ResidentRays,
        samples_per_ray: usize,
        mode: RaySampling,
    ) -> Result<NerfSamples, NerfError> {
        let context = self.0.device.runtime().context();
        if !context.shares_handles_with(rays.device.runtime().context()) {
            return Err(TensorError::DeviceMismatch.into());
        }
        let gpu = context.device();
        let rows = rays.count;
        let count = dimensions(rows, samples_per_ray, &gpu.limits())?;
        let usage = wgpu::BufferUsages::STORAGE;
        let points = runtime::empty_buffer::<[f32; 4]>(gpu, "nerf.points", count, usage)?;
        let widths = runtime::empty_buffer::<f32>(gpu, "nerf.widths", count, usage)?;
        let flags = runtime::upload_slice(gpu, "nerf.sample_flags", &[0u32], usage)?;
        let (stratified, seed) = match mode {
            RaySampling::Midpoint => (0, 0),
            RaySampling::Stratified { seed } => (1, seed),
        };
        let params = runtime::upload_slice(
            gpu,
            "nerf.sample_params",
            &[rows as u32, samples_per_ray as u32, stratified, seed],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let binding = bind(
            gpu,
            &self.0.sampling,
            &[
                (0, &rays.buffer),
                (1, &points),
                (2, &widths),
                (3, &params),
                (4, &flags),
            ],
        );
        let mut encoder = gpu.create_command_encoder(&Default::default());
        dispatch(&mut encoder, &self.0.sampling, &binding, rows);
        let points = self.0.device.capture_into(
            &mut encoder,
            &NdLayout::contiguous(&[rows, samples_per_ray, 4]).map_err(TensorError::from)?,
            &points,
            &flags,
        )?;
        let widths = self.0.device.capture_into(
            &mut encoder,
            &NdLayout::contiguous(&[rows, samples_per_ray]).map_err(TensorError::from)?,
            &widths,
            &flags,
        )?;
        context.queue().submit(Some(encoder.finish()));
        Ok(NerfSamples {
            points,
            widths,
            shape: [rows, samples_per_ray],
        })
    }

    /// Composite a resident field [rays, samples, 4] into [rays, 4] RGBA on a
    /// black background. Sigma uses ReLU; radiance is not clamped. Noncontiguous
    /// inputs are packed on GPU. Output ownership and upstream guards survive
    /// later graph dispatches, draws and drops. This does not provide a VJP.
    ///
    /// A submission is not numerical acceptance. Snapshot reading also rejects
    /// non-finite f32 optical depth/accumulation and any inherited tensor error.
    pub fn composite(
        &self,
        samples: &NerfSamples,
        field: &ResidentTensor,
    ) -> Result<ResidentTensor, NerfError> {
        let context = self.0.device.runtime().context();
        samples.points.require_context(context)?;
        field.require_context(context)?;
        let [rows, count] = samples.shape;
        if field.layout().shape() != [rows, count, 4] {
            return Err(NerfError::FieldShape);
        }
        let gpu = context.device();
        dimensions(rows, count, &gpu.limits())?;
        let output =
            runtime::empty_buffer::<[f32; 4]>(gpu, "nerf.rgba", rows, wgpu::BufferUsages::STORAGE)?;
        let flags = runtime::upload_slice(
            gpu,
            "nerf.composite_flags",
            &[0u32; 3],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        let params = runtime::upload_slice(
            gpu,
            "nerf.composite_params",
            &[rows as u32, count as u32, 1f32.to_bits(), 0],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let field = field.contiguous_into(&mut encoder)?;
        encoder.copy_buffer_to_buffer(field.flags(), 0, &flags, 4, 4);
        encoder.copy_buffer_to_buffer(samples.widths.flags(), 0, &flags, 8, 4);
        let binding = bind(
            gpu,
            &self.0.compositing,
            &[
                (2, samples.widths.values()),
                (3, &output),
                (4, &params),
                (5, &flags),
                (6, field.values()),
            ],
        );
        dispatch(&mut encoder, &self.0.compositing, &binding, rows);
        let output = self.0.device.capture_into(
            &mut encoder,
            &NdLayout::contiguous(&[rows, 4]).map_err(TensorError::from)?,
            &output,
            &flags,
        )?;
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
