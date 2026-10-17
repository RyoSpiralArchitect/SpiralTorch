//! Global active-gradient clipping without readback, before optimizer arithmetic.
use super::*;
use st_kernel_contracts::gradient_clip::{GlobalNormClip, MAX_SCALE_FACTORS};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClipParams {
    len: u32,
    groups_x: u32,
    offset: u32,
    partials: u32,
    enabled: u32,
    limit_mantissa: f32,
    limit_exponent: i32,
    padding: u32,
}

pub(super) struct ClippingKernels {
    layout: wgpu::BindGroupLayout,
    partials: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
}

fn source() -> String {
    (st_kernel_contracts::gradient_clip::wgsl_source()
        + include_str!("../shaders/parameter_clip.wgsl"))
    .replace("INVALID_TENSOR_FLAG", &format!("{INVALID_TENSOR_FLAG}u"))
}

impl ClippingKernels {
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..7)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 6 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 2,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("parameters.clip.bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("parameters.clip.pipeline"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("parameters.clip.shader"),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point,
                compilation_options: Default::default(),
            })
        };
        Self {
            partials: pipeline("clip_partials"),
            reduce: pipeline("clip_reduce"),
            apply: pipeline("clip_apply"),
            layout,
        }
    }
}

impl ResidentParameters {
    /// Clip active derivatives globally before plain SGD. Zero-rate tensors do
    /// not contribute to the norm but their guards remain part of acceptance.
    pub fn sgd_with_rates_clipped(
        &mut self,
        gradients: &ResidentParameterGradients,
        rates: &[f32],
        clip: GlobalNormClip,
    ) -> Result<ResidentParameterUpdate, TrainingError> {
        if rates.len() != self.current.values.len() {
            return Err(TrainingError::ParameterLayout);
        }
        if self.adamw.is_some() {
            return Err(TrainingError::OptimizerState(
                "SGD cannot discard AdamW history",
            ));
        }
        let active = rates
            .iter()
            .map(|&r| {
                SgdStep::new(r)
                    .map(|r| r.rate() != 0.0)
                    .map_err(|_| TrainingError::LearningRate)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let clipped = self.clip_gradients(gradients, clip, &active)?;
        self.sgd_with_rates(&clipped, rates)
    }

    /// Clip before moment accumulation, with the same zero-rate freeze policy
    /// and all-or-none weight/moment/clock commit as the unclipped AdamW path.
    pub fn adamw_with_steps_clipped(
        &mut self,
        gradients: &ResidentParameterGradients,
        steps: &[st_kernel_contracts::adamw::AdamWStep],
        clip: GlobalNormClip,
    ) -> Result<ResidentParameterUpdate, TrainingError> {
        if self.adamw.is_none() {
            return Err(TrainingError::OptimizerState("AdamW not enabled"));
        }
        if steps.len() != self.current.values.len() {
            return Err(TrainingError::ParameterLayout);
        }
        let active: Vec<_> = steps.iter().map(|s| s.rate() != 0.0).collect();
        let clipped = self.clip_gradients(gradients, clip, &active)?;
        self.adamw_with_steps(&clipped, steps)
    }

    /// Return version-bound, immutable gradients clipped together in one global
    /// L2 norm. Inactive tensors are excluded from that norm and copied bitwise,
    /// but every tensor and inherited guard still validates. The caller owns the
    /// active policy (normally nonzero update rates); no optimizer is applied.
    /// The first call compiles kernels; subsequent calls reuse them. No GPU map.
    pub fn clip_gradients(
        &mut self,
        gradients: &ResidentParameterGradients,
        clip: GlobalNormClip,
        active: &[bool],
    ) -> Result<ResidentParameterGradients, TrainingError> {
        if !self.current.version.matches(&gradients.version) {
            return Err(TrainingError::ParameterVersion);
        }
        if active.len() != gradients.values.len() {
            return Err(TrainingError::ParameterLayout);
        }
        let context = self.device.runtime().context();
        let gpu = context.device();
        let counts: Vec<_> = gradients
            .values
            .iter()
            .map(|g| g.layout().len().div_ceil(256))
            .collect();
        let total = counts
            .iter()
            .try_fold(0usize, |a, &b| a.checked_add(b))
            .ok_or(TrainingError::Overflow)?;
        let total_u32 = u32::try_from(total).map_err(|_| TrainingError::Overflow)?;
        storage_limit(
            total.checked_mul(2).ok_or(TrainingError::Overflow)?,
            &gpu.limits(),
        )?;
        let kernels = self
            .clipping
            .get_or_insert_with(|| ClippingKernels::new(gpu));
        let partials = runtime::empty_buffer::<[f32; 2]>(
            gpu,
            "parameters.clip.partials",
            total,
            wgpu::BufferUsages::STORAGE,
        )?;
        let factors = runtime::empty_buffer::<f32>(
            gpu,
            "parameters.clip.factors",
            1 + MAX_SCALE_FACTORS,
            wgpu::BufferUsages::STORAGE,
        )?;
        let guard = Shared::new(runtime::empty_buffer::<u32>(
            gpu,
            "parameters.clip.guard",
            1,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?);
        let (limit_mantissa, limit_exponent) = clip.binary_parts();
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let mut passes = Vec::with_capacity(active.len());
        let mut values = Vec::with_capacity(active.len());
        let mut offset = 0;
        for (index, gradient) in gradients.values.iter().enumerate() {
            let gradient = gradient.contiguous_into(&mut encoder)?;
            let output = self
                .device
                .allocate_output_with_guard(gradient.layout(), Some(guard.clone()))?;
            let grid = groups(gradient.layout().len(), &gpu.limits())?;
            let params = ClipParams {
                len: gradient.layout().len() as u32,
                groups_x: grid[0],
                offset,
                partials: total_u32,
                enabled: u32::from(active[index]),
                limit_mantissa,
                limit_exponent,
                padding: 0,
            };
            let uniform = runtime::upload_slice(
                gpu,
                "parameters.clip.config",
                &[params],
                wgpu::BufferUsages::UNIFORM,
            )?;
            let binding = binding(
                gpu,
                &kernels.layout,
                &[
                    gradient.values(),
                    gradient.flags(),
                    &partials,
                    &factors,
                    output.values(),
                    &guard,
                    &uniform,
                ],
            );
            passes.push((binding, grid));
            values.push(output);
            offset += counts[index] as u32;
        }
        for (binding, grid) in &passes {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&kernels.partials);
            pass.set_bind_group(0, binding, &[]);
            pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&kernels.reduce);
            pass.set_bind_group(0, &passes[0].0, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        for (binding, grid) in &passes {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&kernels.apply);
            pass.set_bind_group(0, binding, &[]);
            pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
        }
        context.queue().submit(Some(encoder.finish()));
        Ok(ResidentParameterGradients {
            version: gradients.version.clone(),
            values,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
