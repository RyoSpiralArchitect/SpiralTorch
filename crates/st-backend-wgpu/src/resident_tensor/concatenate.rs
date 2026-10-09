//! Join logical views in one submission, without staging or arithmetic copies.
use super::*;
use st_kernel_contracts::layout::concatenate_shape;

#[derive(Debug)]
pub(super) struct ConcatenateKernels {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

fn shader_source() -> String {
    include_str!("shaders/concatenate.wgsl")
        .replace("INVALID_TENSOR_FLAG", &format!("{INVALID_TENSOR_FLAG}u"))
}

impl ConcatenateKernels {
    fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: ![1, 4].contains(&binding),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tensor.concatenate.layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tensor.concatenate.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor.concatenate.shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tensor.concatenate"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Self { layout, pipeline }
    }
}

impl ResidentTensor {
    /// Concatenate same-device tensors along an existing axis. Other dimensions
    /// must match exactly. Reads strides, offsets and broadcasts directly;
    /// the returned version is canonical and independent of every input.
    /// One dispatch per input (including empty inputs for inherited guards),
    /// one submission, and no host activation readback or implicit autograd.
    pub fn concatenate(inputs: &[&Self], axis: usize) -> Result<Self, TensorError> {
        let shape = concatenate_shape(
            &inputs.iter().map(|t| t.layout.shape()).collect::<Vec<_>>(),
            axis,
        )?;
        let first = inputs[0];
        let device = &first.device;
        let context = device.runtime().context();
        let gpu = context.device();
        let limits = gpu.limits();
        let layout = NdLayout::contiguous(&shape)?;
        validate_view(&layout, layout.len(), &limits)?;
        for input in inputs {
            input.require_context(context)?;
            validate_view(&input.layout, (input.values().size() / 4) as usize, &limits)?;
            storage_limit((input.flags().size() / 4) as usize, &limits)?;
            grid(input.layout.len(), &limits)?;
        }
        let output = device.allocate_output(&layout)?;
        let kernels = device
            .0
            .concatenate
            .get_or_init(|| ConcatenateKernels::new(gpu));
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let mut offset = 0usize;
        for input in inputs {
            let [x, y, groups] = grid(input.layout.len(), &limits)?;
            let mut meta = vec![
                input.layout.len() as u32,
                layout.rank() as u32,
                x,
                groups,
                input.layout.offset() as u32,
                offset as u32,
                (input.flags().size() / 4) as u32,
            ];
            for values in [
                input.layout.shape(),
                input.layout.strides(),
                layout.strides(),
            ] {
                meta.extend(values.iter().map(|&v| v as u32));
            }
            let meta = runtime::upload_slice(
                gpu,
                "tensor.concatenate.meta",
                &meta,
                wgpu::BufferUsages::STORAGE,
            )?;
            let buffers = [
                input.values(),
                output.values(),
                &meta,
                input.flags(),
                output.flags(),
            ];
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(i, buffer)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let binding = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tensor.concatenate.binding"),
                layout: &kernels.layout,
                entries: &entries,
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tensor.concatenate.pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&kernels.pipeline);
                pass.set_bind_group(0, &binding, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
            // No values are addressable when a non-joined dimension is empty.
            if !layout.is_empty() {
                offset += input.layout.shape()[axis] * layout.strides()[axis];
            }
        }
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
