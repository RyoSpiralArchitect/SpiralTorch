//! Class-last resident cross entropy; ordinary Loss semantics, no host labels.
use super::*;
use st_kernel_contracts::classification::CrossEntropySpec;

#[derive(Debug)]
pub(super) struct ClassificationKernels {
    layout: wgpu::BindGroupLayout,
    labels: wgpu::ComputePipeline,
    rows: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    guard: guard_capture::GuardCapture,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    rows: u32,
    classes: u32,
    row_groups_x: u32,
    label_groups_x: u32,
    reduction: u32,
    ignore_enabled: u32,
    ignore: f32,
    smoothing: f32,
    uniform_mass: f32,
    target_adjustment: f32,
    nll_m: f32,
    nll_e: i32,
    uniform_m: f32,
    uniform_e: i32,
    pad0: u32,
    pad1: u32,
}

impl ClassificationKernels {
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 7 {
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
            label: Some("loss.ce.bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("loss.ce.layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("loss.ce.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cross_entropy.wgsl").into()),
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
            labels: pipeline("count_labels"),
            rows: pipeline("classify_rows"),
            reduce: pipeline("reduce_loss"),
            guard: guard_capture::GuardCapture::new(device),
            layout,
        }
    }
}

impl ResidentTensor {
    /// Integer labels use the same finite f32/i64 transport as ordinary st-nn.
    /// None returns sample axes plus one and uses the ordinary all-ones loss seed.
    /// Value/gradient share a whole-loss guard; invalid GPU labels reject both.
    pub fn cross_entropy_with_logits(
        &self,
        target: &Self,
        spec: CrossEntropySpec,
    ) -> Result<loss::ResidentLoss, TensorError> {
        let context = self.device.runtime().context();
        target.require_context(context)?;
        let (rows, classes, output_shape) =
            spec.shapes(self.layout.shape(), target.layout.shape())?;
        let gpu = context.device();
        let limits = gpu.limits();
        if limits.max_bindings_per_bind_group < 8
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_uniform_buffer_binding_size < 64
            || limits.max_compute_workgroup_storage_size < 2048
        {
            return Err(TensorError::Limit("cross entropy pipeline"));
        }
        storage_limit(rows, &limits)?;
        let label_grid = grid(rows, &limits)?;
        let row_x = (rows.max(1) as u32).min(limits.max_compute_workgroups_per_dimension);
        let row_y = (rows.max(1) as u32).div_ceil(row_x.max(1));
        if row_x == 0 || row_y > limits.max_compute_workgroups_per_dimension {
            return Err(TensorError::Limit("cross entropy rows"));
        }
        let kernels = self
            .device
            .0
            .classification
            .get_or_init(|| ClassificationKernels::new(gpu));
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let prediction = self.contiguous_into(&mut encoder)?;
        let target = target.contiguous_into(&mut encoder)?;
        let gradient = self.device.allocate_output(&self.layout)?;
        let output_layout = NdLayout::contiguous(&output_shape)?;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let value = ResidentTensor {
            storage: Shared::new(Storage {
                values: runtime::empty_buffer::<f32>(
                    gpu,
                    "loss.ce.value",
                    output_layout.len().max(1),
                    usage,
                )?,
                flags: gradient.storage.flags.clone(),
            }),
            layout: output_layout,
            device: self.device.clone(),
        };
        let row_loss = runtime::empty_buffer::<f32>(
            gpu,
            "loss.ce.rows",
            rows.max(1),
            wgpu::BufferUsages::STORAGE,
        )?;
        let active = runtime::empty_buffer::<u32>(
            gpu,
            "loss.ce.active",
            1,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        let validation = runtime::empty_buffer::<u32>(
            gpu,
            "loss.ce.validation",
            3,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        let ignore = spec.ignore_transport();
        let nll = spec.nll_scale();
        let uniform = spec.uniform_scale(classes)?;
        let params = runtime::upload_slice(
            gpu,
            "loss.ce.params",
            &[Params {
                rows: rows as u32,
                classes: classes as u32,
                row_groups_x: row_x,
                label_groups_x: label_grid[0],
                reduction: spec.reduction() as u32,
                ignore_enabled: u32::from(ignore.is_some()),
                ignore: ignore.unwrap_or(0.),
                smoothing: spec.smoothing() as f32,
                uniform_mass: (spec.smoothing() / classes as f64) as f32,
                target_adjustment: (spec.smoothing() * (1. - 1. / classes as f64)) as f32,
                nll_m: nll.mantissa,
                nll_e: nll.exponent,
                uniform_m: uniform.mantissa,
                uniform_e: uniform.exponent,
                pad0: 0,
                pad1: 0,
            }],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let buffers = [
            prediction.values(),
            target.values(),
            gradient.values(),
            &row_loss,
            value.values(),
            &active,
            &validation,
            &params,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("loss.ce"),
            layout: &kernels.layout,
            entries: &entries,
        });
        let guard = kernels.guard.bind(gpu, &validation, gradient.flags());
        encoder.clear_buffer(&validation, 0, None);
        encoder.clear_buffer(&active, 0, None);
        encoder.copy_buffer_to_buffer(prediction.flags(), 0, &validation, 4, 4);
        encoder.copy_buffer_to_buffer(target.flags(), 0, &validation, 8, 4);
        for (pipeline, x, y) in [
            (&kernels.labels, label_grid[0], label_grid[1]),
            (&kernels.rows, row_x, row_y),
            (&kernels.reduce, 1, 1),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("loss.ce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("loss.ce.guard"),
                timestamp_writes: None,
            });
            kernels.guard.encode_in_pass(&mut pass, &guard);
        }
        context.queue().submit(Some(encoder.finish()));
        Ok(loss::ResidentLoss {
            value,
            prediction_gradient: gradient,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
