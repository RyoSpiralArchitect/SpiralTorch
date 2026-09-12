//! Shared training MSE kernels exposed as an owning loss/seed pair.
use super::*;

/// Mean loss and exact prediction cotangent, with one shared whole-loss guard.
/// Both stay resident and survive reuse/drop. A submission is not acceptance:
/// reading either tensor checks inputs, intermediate squares and the reduction.
#[derive(Clone)]
pub struct ResidentLoss {
    value: ResidentTensor,
    prediction_gradient: ResidentTensor,
}

impl ResidentLoss {
    pub fn value(&self) -> &ResidentTensor {
        &self.value
    }
    pub fn prediction_gradient(&self) -> &ResidentTensor {
        &self.prediction_gradient
    }
}

#[derive(Debug)]
pub(super) struct MseKernels {
    layout: wgpu::BindGroupLayout,
    partials: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    guard: guard_capture::GuardCapture,
}

impl MseKernels {
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding >= 7 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("loss.mse.bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("loss.mse.layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("loss.mse.shared_training_shader"),
            source: wgpu::ShaderSource::Wgsl(
                crate::resident_training::training_scalar_source().into(),
            ),
        });
        let pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point,
                compilation_options: Default::default(),
            })
        };
        Self {
            partials: pipeline("mse_partials"),
            reduce: pipeline("mse_reduce"),
            guard: guard_capture::GuardCapture::new(device),
            layout,
        }
    }

    fn bind(&self, device: &wgpu::Device, buffers: [&wgpu::Buffer; 9]) -> wgpu::BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("loss.mse"),
            layout: &self.layout,
            entries: &entries,
        })
    }
}

impl ResidentTensor {
    /// Exact mean-MSE and 2 * (prediction - target) / numel on the owning device.
    /// Shape equality is required; explicit N-D views are packed on GPU.
    /// Empty inputs follow SpiralTorch's existing MSE contract: zero loss/seed.
    /// No host observation, optimizer, loss scaling or CPU fallback is added.
    pub fn mean_squared_error(&self, target: &Self) -> Result<ResidentLoss, TensorError> {
        let context = self.device.runtime().context();
        target.require_context(context)?;
        if self.layout.shape() != target.layout.shape() {
            return Err(TensorError::LossShape);
        }
        let device = context.device();
        let limits = device.limits();
        if limits.max_bindings_per_bind_group < 9
            || limits.max_uniform_buffers_per_shader_stage < 2
            || limits.max_uniform_buffer_binding_size < 32
            || limits.max_compute_workgroup_storage_size < 1024
        {
            return Err(TensorError::Limit("mean MSE pipeline"));
        }
        let [x, y, count] = grid(self.layout.len(), &limits)?;
        storage_limit(count as usize, &limits)?;
        let kernels = self.device.0.mse.get_or_init(|| MseKernels::new(device));
        let mut encoder = device.create_command_encoder(&Default::default());
        let prediction = self.contiguous_into(&mut encoder)?;
        let target = target.contiguous_into(&mut encoder)?;
        let gradient = self.device.allocate_output(&self.layout)?;
        let value = ResidentTensor {
            storage: Shared::new(Storage {
                values: runtime::empty_buffer::<f32>(
                    device,
                    "loss.mse.value",
                    1,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                )?,
                flags: gradient.storage.flags.clone(),
            }),
            layout: NdLayout::contiguous(&[1, 1])?,
            device: self.device.clone(),
        };
        let partials = runtime::empty_buffer::<f32>(
            device,
            "loss.mse.partials",
            count as usize,
            wgpu::BufferUsages::STORAGE,
        )?;
        let validation = runtime::empty_buffer::<u32>(
            device,
            "loss.mse.validation",
            3,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        // The shared training Params layout; only len/stage/groups/partials
        // participate in these two unchanged shader entry points.
        let params = runtime::upload_slice(
            device,
            "loss.mse.params",
            &[crate::resident_training::Params::standalone_mse(
                self.layout.len() as u32,
                x,
                count,
            )],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let step = runtime::upload_slice(
            device,
            "loss.mse.unused_step",
            &[0f32; 4],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let first = kernels.bind(
            device,
            [
                prediction.values(),
                target.values(),
                prediction.values(),
                prediction.values(),
                gradient.values(),
                &partials,
                &validation,
                &params,
                &step,
            ],
        );
        let second = kernels.bind(
            device,
            [
                &partials,
                prediction.values(),
                prediction.values(),
                prediction.values(),
                gradient.values(),
                value.values(),
                &validation,
                &params,
                &step,
            ],
        );
        let guard = kernels.guard.bind(device, &validation, gradient.flags());
        encoder.clear_buffer(&validation, 0, None);
        encoder.copy_buffer_to_buffer(prediction.flags(), 0, &validation, 4, 4);
        encoder.copy_buffer_to_buffer(target.flags(), 0, &validation, 8, 4);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("loss.mse.partials"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernels.partials);
            pass.set_bind_group(0, &first, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("loss.mse.reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernels.reduce);
            pass.set_bind_group(0, &second, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("loss.mse.guard"),
                timestamp_writes: None,
            });
            kernels.guard.encode_in_pass(&mut pass, &guard);
        }
        context.queue().submit(Some(encoder.finish()));
        Ok(ResidentLoss {
            value,
            prediction_gradient: gradient,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
