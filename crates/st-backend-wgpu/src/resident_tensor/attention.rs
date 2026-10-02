//! Owning, forward-only attention. Q/K/V, optional score biases, view packing,
//! online softmax, and inherited validity guards stay on the same GPU queue.

use super::*;
pub use st_kernel_contracts::attention::{AttentionMask, AttentionSpec};

const SHADER_SOURCE: &str = include_str!("shaders/attention.wgsl");
const MAX_HEAD_DIM: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    contexts: u32,
    queries: u32,
    keys: u32,
    head_dim: u32,
    scale: f32,
    flags: u32,
    query_offset: u32,
    groups_x: u32,
}

#[derive(Debug)]
pub(crate) struct AttentionKernels {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    guard: guard_capture::GuardCapture,
}

impl AttentionKernels {
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
                            read_only: binding < 5,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attention.bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("attention.layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attention.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("attention.forward"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "forward",
            compilation_options: Default::default(),
        });
        Self {
            layout,
            pipeline,
            guard: guard_capture::GuardCapture::new(device),
        }
    }
}

/// Validate this kernel's capabilities without creating GPU resources.
pub fn validate_limits(spec: AttentionSpec, limits: &wgpu::Limits) -> Result<(), TensorError> {
    preflight(spec, limits).map(|_| ())
}

fn preflight(spec: AttentionSpec, limits: &wgpu::Limits) -> Result<[u32; 2], TensorError> {
    if spec.head_dim() > MAX_HEAD_DIM
        || limits.max_bindings_per_bind_group < 8
        || limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_uniform_buffer_binding_size < 32
        || limits.max_compute_workgroup_storage_size < (256 * 2 + 64 + 10) * 4
    {
        return Err(TensorError::Limit(
            "attention pipeline (head dimension <= 256)",
        ));
    }
    for extent in spec.query_shape().into_iter().chain(spec.key_shape()) {
        u32::try_from(extent).map_err(|_| TensorError::Limit("attention dimension"))?;
    }
    u32::try_from(spec.contexts()).map_err(|_| TensorError::Limit("attention contexts"))?;
    storage_limit(spec.contexts() * spec.queries() * spec.head_dim(), limits)?;
    storage_limit(spec.contexts() * spec.keys() * spec.head_dim(), limits)?;
    let rows = u32::try_from((spec.contexts() * spec.queries()).max(1))
        .map_err(|_| TensorError::Limit("attention rows"))?;
    let x = rows.min(limits.max_compute_workgroups_per_dimension);
    if x == 0
        || rows.div_ceil(x) > limits.max_compute_workgroups_per_dimension
        || x.checked_mul(rows.div_ceil(x)).is_none()
    {
        return Err(TensorError::Limit("attention dispatch grid"));
    }
    Ok([x, rows.div_ceil(x)])
}

impl ResidentTensor {
    /// `softmax(scale * QK^T + z_bias + pair_bias) V`, with an optional
    /// structural causal mask. Biases cannot unmask future keys.
    ///
    /// Q has shape `[B,H,Q,D]`, K/V `[B,H,K,D]`, Z-bias `[B,H,K]`, and
    /// pairwise bias `[B,H,Q,K]`. Broadcast explicitly using views. All values
    /// must be finite; score/weighted-output overflow fails on readback and is
    /// inherited by downstream operations. Arbitrary views are packed on GPU.
    ///
    /// This initial forward kernel supports D <= 256, no dropout or backward
    /// tape. Unsupported shapes/devices return errors, never a CPU fallback.
    pub fn scaled_dot_attention(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: AttentionMask,
        z_bias: Option<&Self>,
        pair_bias: Option<&Self>,
    ) -> Result<Self, TensorError> {
        let spec = AttentionSpec::new(
            self.layout.shape(),
            keys.layout.shape(),
            values.layout.shape(),
            scale,
            mask,
        )?;
        spec.validate_bias_shapes(
            z_bias.map(|bias| bias.layout.shape()),
            pair_bias.map(|bias| bias.layout.shape()),
        )?;
        let context = self.device.runtime().context();
        for tensor in [Some(keys), Some(values), z_bias, pair_bias]
            .into_iter()
            .flatten()
        {
            tensor.require_context(context)?;
            storage_limit(tensor.layout.len(), &context.device().limits())?;
        }
        let gpu = context.device();
        let grid = preflight(spec, &gpu.limits())?;
        let kernels = self
            .device
            .0
            .attention
            .get_or_init(|| AttentionKernels::new(gpu));
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let query = self.contiguous_into(&mut encoder)?;
        let keys = keys.contiguous_into(&mut encoder)?;
        let values = values.contiguous_into(&mut encoder)?;
        let z_bias = z_bias
            .map(|bias| bias.contiguous_into(&mut encoder))
            .transpose()?;
        let pair_bias = pair_bias
            .map(|bias| bias.contiguous_into(&mut encoder))
            .transpose()?;
        let output = self.device.allocate_output(&self.layout)?;
        let inputs: Vec<_> = [
            Some(&query),
            Some(&keys),
            Some(&values),
            z_bias.as_ref(),
            pair_bias.as_ref(),
        ]
        .into_iter()
        .flatten()
        .collect();
        // Concatenate inherited guards, then reduce them into the owned output
        // guard. No host read or mutable shared guard is needed.
        let flag_words = inputs.iter().try_fold(1usize, |total, tensor| {
            let bytes = tensor.flags().size();
            if bytes % 4 != 0 {
                return Err(TensorError::Limit("attention guard alignment"));
            }
            let words = usize::try_from(bytes / 4)
                .map_err(|_| TensorError::Limit("attention guard size"))?;
            total
                .checked_add(words)
                .ok_or(TensorError::Limit("attention guard size"))
        })?;
        storage_limit(flag_words, &gpu.limits())?;
        let flags = runtime::empty_buffer::<u32>(
            gpu,
            "attention.flags",
            flag_words,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        encoder.clear_buffer(&flags, 0, None);
        let mut offset = 4;
        for input in inputs {
            encoder.copy_buffer_to_buffer(input.flags(), 0, &flags, offset, input.flags().size());
            offset += input.flags().size();
        }
        let (causal, query_offset) = match mask {
            AttentionMask::None => (0, 0),
            AttentionMask::Causal { query_offset } => (4, query_offset as u32),
        };
        let params = runtime::upload_slice(
            gpu,
            "attention.params",
            &[Params {
                contexts: spec.contexts() as u32,
                queries: spec.queries() as u32,
                keys: spec.keys() as u32,
                head_dim: spec.head_dim() as u32,
                scale,
                flags: u32::from(z_bias.is_some()) | (u32::from(pair_bias.is_some()) << 1) | causal,
                query_offset,
                groups_x: grid[0],
            }],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let dummy =
            runtime::empty_buffer::<f32>(gpu, "attention.no_bias", 1, wgpu::BufferUsages::STORAGE)?;
        let buffers = [
            query.values(),
            keys.values(),
            values.values(),
            z_bias.as_ref().map_or(&dummy, ResidentTensor::values),
            pair_bias.as_ref().map_or(&dummy, ResidentTensor::values),
            output.values(),
            &flags,
            &params,
        ];
        let entries: Vec<_> = buffers
            .into_iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let binding = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attention.forward"),
            layout: &kernels.layout,
            entries: &entries,
        });
        if spec.contexts() != 0 && spec.queries() != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attention.forward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernels.pipeline);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(grid[0], grid[1], 1);
        }
        let binding = kernels.guard.bind(gpu, &flags, output.flags());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attention.guard"),
                timestamp_writes: None,
            });
            kernels.guard.encode_in_pass(&mut pass, &binding);
        }
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
