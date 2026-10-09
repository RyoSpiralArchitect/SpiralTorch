//! First-order attention pullbacks with immutable, resident gradient ownership.

use super::*;

/// Logical input gradients, without implicit averaging or broadcast reduction.
/// All fields share an immutable allocation and one whole-operation guard.
/// Bias gradients are absent exactly when the corresponding input was absent.
#[derive(Clone)]
pub struct ResidentAttentionGradients {
    pub query: ResidentTensor,
    pub key: ResidentTensor,
    pub value: ResidentTensor,
    pub z_bias: Option<ResidentTensor>,
    pub pair_bias: Option<ResidentTensor>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VjpParams {
    contexts: u32,
    queries: u32,
    keys: u32,
    head_dim: u32,
    scale: f32,
    flags: u32,
    query_offset: u32,
    heads: u32,
    query_groups_x: u32,
    key_groups_x: u32,
    stats_offset: u32,
    query_offset_out: u32,
    key_offset_out: u32,
    value_offset_out: u32,
    z_offset_out: u32,
    pair_offset_out: u32,
    query: View,
    key: View,
    value: View,
    z_bias: View,
    pair_bias: View,
    upstream: View,
}

fn shader_source() -> String {
    include_str!("../shaders/attention_backward.wgsl")
        .replace(
            "ROUNDED_ADD",
            include_str!("../../shaders/rounded_add.wgsl"),
        )
        .replace(
            "WIDE_ARITHMETIC",
            include_str!("../shaders/layer_norm_wide.wgsl"),
        )
}

#[derive(Debug)]
pub(crate) struct AttentionVjpKernels {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 3],
    guard: guard_capture::GuardCapture,
}

impl AttentionVjpKernels {
    fn new(gpu: &wgpu::Device) -> Self {
        let layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attention.vjp.bindings"),
            entries: &(0..9)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 8 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 6,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("attention.vjp.layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attention.vjp.shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source().into()),
        });
        let pipelines = ["statistics", "query_pullback", "key_value_pullback"].map(|entry_point| {
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point,
                compilation_options: Default::default(),
            })
        });
        Self {
            layout,
            pipelines,
            guard: guard_capture::GuardCapture::new(gpu),
        }
    }
}

struct PackedLayout {
    offsets: [usize; 5],
    lengths: [usize; 5],
    total: usize,
    query_grid: [u32; 2],
    key_grid: [u32; 2],
}

impl PackedLayout {
    fn new(
        spec: AttentionSpec,
        z_bias: bool,
        pair_bias: bool,
        limits: &wgpu::Limits,
    ) -> Result<Self, TensorError> {
        let query_grid = preflight(spec, limits)?;
        if limits.max_bindings_per_bind_group < 9
            || limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_uniform_buffer_binding_size < std::mem::size_of::<VjpParams>() as u32
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_storage_size < 10_240
        {
            return Err(TensorError::Limit("attention VJP pipeline"));
        }
        let rows = spec.contexts() * spec.queries();
        let key_rows = spec.contexts() * spec.keys();
        let lengths = [
            rows * spec.head_dim(),
            key_rows * spec.head_dim(),
            key_rows * spec.head_dim(),
            if z_bias { key_rows } else { 0 },
            if pair_bias { rows * spec.keys() } else { 0 },
        ];
        let mut total = rows
            .checked_mul(9)
            .ok_or(TensorError::Limit("attention VJP allocation"))?;
        let mut offsets = [0; 5];
        for (offset, length) in offsets.iter_mut().zip(lengths) {
            *offset = total;
            total = total
                .checked_add(length)
                .ok_or(TensorError::Limit("attention VJP allocation"))?;
        }
        storage_limit(total, limits)?;
        let key_grid = normalization::groups(key_rows, limits)?;
        Ok(Self {
            offsets,
            lengths,
            total,
            query_grid,
            key_grid,
        })
    }
}

/// Validate the complete gradient allocation and dispatch before creating GPU
/// resources. Scratch is nine words per query, not a quadratic probability tape.
pub fn validate_vjp_limits(
    spec: AttentionSpec,
    z_bias: bool,
    pair_bias: bool,
    limits: &wgpu::Limits,
) -> Result<(), TensorError> {
    PackedLayout::new(spec, z_bias, pair_bias, limits).map(|_| ())
}

impl ResidentTensor {
    /// VJP of `scaled_dot_attention`, with upstream shape `[B,H,Q,D]`.
    /// Q/K/V, upstream and optional biases may be strided or broadcast views.
    /// Gradients have the logical input shapes; reducing a broadcast back to its
    /// source remains explicit. Causal future pairs have zero gradient.
    ///
    /// This recomputes normalization in three GPU passes and submits once, with
    /// no host readback, CPU fallback, float atomics or retained probability
    /// matrix. Extended-exponent accumulation avoids avoidable f32 intermediate
    /// overflow; it is not IEEE f64 or a claim of bitwise oracle equivalence.
    /// Returned immutable views survive subsequent forwards/pullbacks. A failed
    /// inherited input or unrepresentable requested gradient invalidates all
    /// outputs together, including empty outputs, on snapshot or consumption.
    #[allow(clippy::too_many_arguments)]
    pub fn scaled_dot_attention_vjp(
        &self,
        keys: &Self,
        values: &Self,
        upstream: &Self,
        scale: f32,
        mask: AttentionMask,
        z_bias: Option<&Self>,
        pair_bias: Option<&Self>,
    ) -> Result<ResidentAttentionGradients, TensorError> {
        let spec = AttentionSpec::new(
            self.layout.shape(),
            keys.layout.shape(),
            values.layout.shape(),
            scale,
            mask,
        )?;
        spec.validate_bias_shapes(
            z_bias.map(|v| v.layout.shape()),
            pair_bias.map(|v| v.layout.shape()),
        )?;
        if upstream.layout.shape() != spec.query_shape() {
            return Err(TensorError::Attention(
                st_kernel_contracts::attention::AttentionError::Shape,
            ));
        }
        let context = self.device.runtime().context();
        let gpu = context.device();
        let limits = gpu.limits();
        let packed = PackedLayout::new(spec, z_bias.is_some(), pair_bias.is_some(), &limits)?;
        let inputs = [
            Some(self),
            Some(keys),
            Some(values),
            z_bias,
            pair_bias,
            Some(upstream),
        ];
        let mut views = [View::default(); 6];
        let mut flag_words = 1usize;
        for (input, view) in inputs.iter().zip(&mut views) {
            let Some(input) = input else { continue };
            input.require_context(context)?;
            *view = view_descriptor(&input.layout, (input.values().size() / 4) as usize, &limits)?;
            if input.flags().size() % 4 != 0 {
                return Err(TensorError::Limit("attention VJP guard alignment"));
            }
            flag_words = flag_words
                .checked_add(
                    usize::try_from(input.flags().size() / 4)
                        .map_err(|_| TensorError::Limit("attention VJP guards"))?,
                )
                .ok_or(TensorError::Limit("attention VJP guards"))?;
        }
        storage_limit(flag_words, &limits)?;
        let output = self
            .device
            .allocate_output(&NdLayout::contiguous(&[packed.total])?)?;
        let gradient = |index, shape: &[usize]| {
            output
                .narrow(0, packed.offsets[index], packed.lengths[index])?
                .reshape(shape)
        };
        let gradients = ResidentAttentionGradients {
            query: gradient(0, &spec.query_shape())?,
            key: gradient(1, &spec.key_shape())?,
            value: gradient(2, &spec.key_shape())?,
            z_bias: z_bias
                .map(|_| gradient(3, &spec.z_bias_shape()))
                .transpose()?,
            pair_bias: pair_bias
                .map(|_| gradient(4, &spec.pair_bias_shape()))
                .transpose()?,
        };
        let flags = runtime::empty_buffer::<u32>(
            gpu,
            "attention.vjp.flags",
            flag_words,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        let (causal, query_offset) = match mask {
            AttentionMask::None => (0, 0),
            AttentionMask::Causal { query_offset } => (4, query_offset as u32),
        };
        let params = runtime::upload_slice(
            gpu,
            "attention.vjp.params",
            &[VjpParams {
                contexts: spec.contexts() as u32,
                queries: spec.queries() as u32,
                keys: spec.keys() as u32,
                head_dim: spec.head_dim() as u32,
                scale,
                flags: u32::from(z_bias.is_some())
                    | (u32::from(pair_bias.is_some()) << 1)
                    | causal
                    | (u32::from(key_tile(spec) == 8) << 3),
                query_offset,
                heads: spec.query_shape()[1] as u32,
                query_groups_x: packed.query_grid[0],
                key_groups_x: packed.key_grid[0],
                stats_offset: 0,
                query_offset_out: packed.offsets[0] as u32,
                key_offset_out: packed.offsets[1] as u32,
                value_offset_out: packed.offsets[2] as u32,
                z_offset_out: packed.offsets[3] as u32,
                pair_offset_out: packed.offsets[4] as u32,
                query: views[0],
                key: views[1],
                value: views[2],
                z_bias: views[3],
                pair_bias: views[4],
                upstream: views[5],
            }],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let kernels = self
            .device
            .0
            .attention_vjp
            .get_or_init(|| AttentionVjpKernels::new(gpu));
        let dummy = runtime::empty_buffer::<f32>(
            gpu,
            "attention.vjp.no_bias",
            1,
            wgpu::BufferUsages::STORAGE,
        )?;
        let buffers = [
            self.values(),
            keys.values(),
            values.values(),
            z_bias.map_or(&dummy, Self::values),
            pair_bias.map_or(&dummy, Self::values),
            upstream.values(),
            output.values(),
            &flags,
            &params,
        ];
        let binding = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attention.vjp"),
            layout: &kernels.layout,
            entries: &buffers
                .into_iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut encoder = gpu.create_command_encoder(&Default::default());
        encoder.clear_buffer(&flags, 0, None);
        let mut offset = 4;
        for input in inputs.into_iter().flatten() {
            encoder.copy_buffer_to_buffer(input.flags(), 0, &flags, offset, input.flags().size());
            offset += input.flags().size();
        }
        for (index, (count, grid)) in [
            (spec.contexts() * spec.queries(), packed.query_grid),
            (spec.contexts() * spec.queries(), packed.query_grid),
            (spec.contexts() * spec.keys(), packed.key_grid),
        ]
        .into_iter()
        .enumerate()
        {
            if count == 0 {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attention.vjp"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernels.pipelines[index]);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(grid[0], grid[1], 1);
        }
        let captured = kernels.guard.bind(gpu, &flags, output.flags());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            kernels.guard.encode_in_pass(&mut pass, &captured);
        }
        context.queue().submit(Some(encoder.finish()));
        Ok(gradients)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
