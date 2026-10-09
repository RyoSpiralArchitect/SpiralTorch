//! Owning attention. Q/K/V, optional score biases, strided reads,
//! online softmax, and inherited validity guards stay on the same GPU queue.

use super::*;
use crate::runtime::timestamps::PassTimestampCursor;
pub use st_kernel_contracts::attention::{AttentionMask, AttentionSpec};
pub(crate) mod vjp;
pub use vjp::{validate_vjp_limits, ResidentAttentionGradients};

const SHADER_SOURCE: &str = include_str!("shaders/attention.wgsl");
const MAX_HEAD_DIM: usize = 256;

#[derive(Clone, Copy)]
enum OutputOrder {
    HeadMajor,
    MergedHeads,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct View {
    strides: [u32; 4],
    offset: u32,
    padding: [u32; 3],
}

fn view_descriptor(
    layout: &NdLayout,
    storage_len: usize,
    limits: &wgpu::Limits,
) -> Result<View, TensorError> {
    validate_view(layout, storage_len, limits)?;
    let axes = match layout.rank() {
        3 => [Some(0), Some(1), None, Some(2)], // Z bias [B,H,K].
        4 => [Some(0), Some(1), Some(2), Some(3)],
        _ => return Err(TensorError::Limit("attention view rank")),
    };
    Ok(View {
        strides: axes.map(|axis| axis.map_or(0, |i| layout.strides()[i] as u32)),
        offset: layout.offset() as u32,
        padding: [0; 3],
    })
}

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
    heads: u32,
    padding: [u32; 3],
    query: View,
    key: View,
    value: View,
    z_bias: View,
    pair_bias: View,
}

#[derive(Debug)]
pub(crate) struct AttentionKernels {
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    pipelines: [std::sync::OnceLock<wgpu::ComputePipeline>; 4],
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
        Self {
            layout,
            pipeline_layout,
            module,
            pipelines: Default::default(),
            guard: guard_capture::GuardCapture::new(device),
        }
    }

    fn pipeline(
        &self,
        device: &wgpu::Device,
        spec: AttentionSpec,
        order: OutputOrder,
    ) -> &wgpu::ComputePipeline {
        let tile = key_tile(spec);
        let merged = u32::from(matches!(order, OutputOrder::MergedHeads));
        self.pipelines[usize::from(tile == 8) * 2 + merged as usize].get_or_init(|| {
            let constants = std::collections::HashMap::from([
                ("KEY_TILE".to_owned(), f64::from(tile)),
                ("MERGED_HEADS".to_owned(), f64::from(merged)),
            ]);
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("attention.forward"),
                layout: Some(&self.pipeline_layout),
                module: &self.module,
                entry_point: "forward",
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
            })
        })
    }
}

fn key_tile(spec: AttentionSpec) -> u32 {
    // Matched full-chain measurements support this range, not short sequences
    // or wider heads. Keep their established reduction until measured too.
    if spec.keys() >= 128 && spec.head_dim() <= 32 {
        8
    } else {
        1
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
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<Params>() as u32
        || limits.max_compute_workgroup_storage_size < (256 * 2 + 64 + 18) * 4
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
    /// inherited by downstream operations. Strided and broadcast inputs are
    /// read directly from their immutable storage, without intermediate packing.
    /// Returns owned, contiguous head-major output [B,H,Q,D].
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
        self.attention_forward(
            keys,
            values,
            scale,
            mask,
            z_bias,
            pair_bias,
            OutputOrder::HeadMajor,
            &mut PassTimestampCursor::default(),
        )
    }

    /// Same arithmetic, masks, input shapes and guards as
    /// [Self::scaled_dot_attention], but returns owned, contiguous [B,Q,H*D].
    /// Element (b,q,h*D+d) equals the head-major output (b,h,q,d).
    /// The attention kernel writes this order directly: no head-merge dispatch,
    /// intermediate output copy or host readback is needed before a Linear.
    pub fn scaled_dot_attention_merged_heads(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: AttentionMask,
        z_bias: Option<&Self>,
        pair_bias: Option<&Self>,
    ) -> Result<Self, TensorError> {
        self.attention_forward(
            keys,
            values,
            scale,
            mask,
            z_bias,
            pair_bias,
            OutputOrder::MergedHeads,
            &mut PassTimestampCursor::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_forward(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: AttentionMask,
        z_bias: Option<&Self>,
        pair_bias: Option<&Self>,
        order: OutputOrder,
        timestamps: &mut PassTimestampCursor<'_>,
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
        let inputs = [Some(self), Some(keys), Some(values), z_bias, pair_bias];
        let mut views = [View::default(); 5];
        for (tensor, view) in inputs.iter().zip(&mut views) {
            let Some(tensor) = tensor else { continue };
            tensor.require_context(context)?;
            storage_limit(tensor.layout.len(), &context.device().limits())?;
            *view = view_descriptor(
                &tensor.layout,
                (tensor.values().size() / 4) as usize,
                &context.device().limits(),
            )?;
        }
        let gpu = context.device();
        let grid = preflight(spec, &gpu.limits())?;
        let output_layout = match order {
            OutputOrder::HeadMajor => NdLayout::contiguous(&spec.query_shape())?,
            OutputOrder::MergedHeads => NdLayout::contiguous(&spec.merged_output_shape()?)?,
        };
        validate_view(&output_layout, output_layout.len(), &gpu.limits())?;
        let kernels = self
            .device
            .0
            .attention
            .get_or_init(|| AttentionKernels::new(gpu));
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let output = self.device.allocate_output(&output_layout)?;
        // Concatenate inherited guards, then reduce them into the owned output
        // guard. No host read or mutable shared guard is needed.
        let flag_words = inputs.iter().flatten().try_fold(1usize, |total, tensor| {
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
        for input in inputs.into_iter().flatten() {
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
                heads: spec.query_shape()[1] as u32,
                padding: [0; 3],
                query: views[0],
                key: views[1],
                value: views[2],
                z_bias: views[3],
                pair_bias: views[4],
            }],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let dummy =
            runtime::empty_buffer::<f32>(gpu, "attention.no_bias", 1, wgpu::BufferUsages::STORAGE)?;
        let buffers = [
            self.values(),
            keys.values(),
            values.values(),
            z_bias.map_or(&dummy, ResidentTensor::values),
            pair_bias.map_or(&dummy, ResidentTensor::values),
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
                timestamp_writes: timestamps.next(),
            });
            pass.set_pipeline(kernels.pipeline(gpu, spec, order));
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(grid[0], grid[1], 1);
        }
        let binding = kernels.guard.bind(gpu, &flags, output.flags());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attention.guard"),
                timestamp_writes: timestamps.next(),
            });
            kernels.guard.encode_in_pass(&mut pass, &binding);
        }
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod profile_probe;
