//! Owning, last-axis affine LayerNorm and requested VJPs on one GPU queue.
//! No statistics, tape, or intermediate gradients are read back to the host.

use super::*;
use st_kernel_contracts::normalization::{
    validate_epsilon, validate_gradient_scale, LayerNormError, LayerNormShape,
};

/// The forward value plus immutable centered values, row statistics and gamma.
/// Cloning, dropping other operations, or running backward again cannot recycle
/// this tape. Reading any result checks its inherited whole-operation guard.
#[derive(Clone)]
pub struct ResidentLayerNorm {
    value: ResidentTensor,
    tape: LayerNormBackwardTape,
}

/// Statistics-only backward tape. Affine forward overflow cannot invalidate
/// finite requested VJPs because no forward affine output is computed.
#[derive(Clone)]
pub struct ResidentLayerNormVjp {
    tape: LayerNormBackwardTape,
}

#[derive(Clone)]
struct LayerNormBackwardTape {
    centered: Shared<wgpu::Buffer>,
    row_stats: Shared<wgpu::Buffer>,
    gamma: ResidentTensor,
    layout: NdLayout,
    flags: Shared<wgpu::Buffer>,
    device: TensorDevice,
    shape: LayerNormShape,
    epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    rows: u32,
    cols: u32,
    groups_x: u32,
    requested: u32,
    epsilon: f32,
    scale: f32,
    beta_offset: u32,
    _pad: u32,
}

fn source(template: &str) -> String {
    template
        .replace("ROUNDED_ADD", include_str!("../shaders/rounded_add.wgsl"))
        .replace(
            "WIDE_ARITHMETIC",
            include_str!("shaders/layer_norm_wide.wgsl"),
        )
}

#[derive(Debug)]
pub(super) struct LayerNormKernels {
    forward_layout: wgpu::BindGroupLayout,
    backward_layout: wgpu::BindGroupLayout,
    forward: wgpu::ComputePipeline,
    statistics: wgpu::ComputePipeline,
    input: wgpu::ComputePipeline,
    affine: wgpu::ComputePipeline,
    guard: guard_capture::GuardCapture,
}

impl LayerNormKernels {
    fn new(device: &wgpu::Device) -> Self {
        let layout = |read_only_until| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("layer_norm.bindings"),
                entries: &(0..8)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if binding == 7 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: binding < read_only_until,
                                }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let forward_layout = layout(3);
        let backward_layout = layout(4);
        let pipeline = |layout, template, entry_point| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("layer_norm.layout"),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry_point),
                source: wgpu::ShaderSource::Wgsl(source(template).into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&layout),
                module: &module,
                entry_point,
                compilation_options: Default::default(),
            })
        };
        Self {
            forward: pipeline(
                &forward_layout,
                include_str!("shaders/layer_norm.wgsl"),
                "forward",
            ),
            statistics: pipeline(
                &forward_layout,
                include_str!("shaders/layer_norm.wgsl"),
                "statistics",
            ),
            input: pipeline(
                &backward_layout,
                include_str!("shaders/layer_norm_backward.wgsl"),
                "backward_input",
            ),
            affine: pipeline(
                &backward_layout,
                include_str!("shaders/layer_norm_backward.wgsl"),
                "backward_affine",
            ),
            forward_layout,
            backward_layout,
            guard: guard_capture::GuardCapture::new(device),
        }
    }
}

fn preflight(shape: LayerNormShape, limits: &wgpu::Limits) -> Result<(), TensorError> {
    if limits.max_bindings_per_bind_group < 8
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_uniform_buffer_binding_size < 32
        // Two 256-element reductions plus four row-constant Wide values.
        || limits.max_compute_workgroup_storage_size < 2 * 256 * 16 + 4 * 16
    {
        return Err(TensorError::Limit("LayerNorm pipeline"));
    }
    storage_limit(
        shape
            .rows
            .checked_mul(shape.cols)
            .and_then(|n| n.checked_mul(4))
            .ok_or(TensorError::Limit("LayerNorm centered tape"))?,
        limits,
    )?;
    storage_limit(
        shape
            .rows
            .checked_mul(8)
            .ok_or(TensorError::Limit("LayerNorm row tape"))?,
        limits,
    )?;
    storage_limit(
        shape
            .cols
            .checked_mul(2)
            .ok_or(TensorError::Limit("LayerNorm affine gradients"))?,
        limits,
    )?;
    groups(shape.rows, limits)?;
    groups(shape.cols, limits)?;
    Ok(())
}

fn groups(count: usize, limits: &wgpu::Limits) -> Result<[u32; 2], TensorError> {
    let count = u32::try_from(count.max(1)).map_err(|_| TensorError::Limit("LayerNorm grid"))?;
    let x = count.min(limits.max_compute_workgroups_per_dimension);
    if x == 0 || count.div_ceil(x) > limits.max_compute_workgroups_per_dimension {
        return Err(TensorError::Limit("LayerNorm grid"));
    }
    Ok([x, count.div_ceil(x)])
}

fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: [&wgpu::Buffer; 8],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("layer_norm"),
        layout,
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    })
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    binding: &wgpu::BindGroup,
    [x, y]: [u32; 2],
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("layer_norm"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, binding, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

fn validation(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    inputs: &[&wgpu::Buffer],
) -> Result<wgpu::Buffer, TensorError> {
    let bytes = inputs.iter().try_fold(4u64, |total, input| {
        if input.size() % 4 != 0 {
            return Err(TensorError::Limit("LayerNorm validation alignment"));
        }
        total
            .checked_add(input.size())
            .ok_or(TensorError::Limit("LayerNorm validation size"))
    })?;
    let words =
        usize::try_from(bytes / 4).map_err(|_| TensorError::Limit("LayerNorm validation size"))?;
    let flags = runtime::empty_buffer::<u32>(
        device,
        "layer_norm.validation",
        words,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    )?;
    encoder.clear_buffer(&flags, 0, None);
    let mut offset = 4;
    for input in inputs {
        // A previous validation may contain several inherited guard words.
        encoder.copy_buffer_to_buffer(input, 0, &flags, offset, input.size());
        offset += input.size();
    }
    Ok(flags)
}

fn capture(
    kernels: &LayerNormKernels,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    flags: &wgpu::Buffer,
    output: &wgpu::Buffer,
) {
    let binding = kernels.guard.bind(device, flags, output);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("layer_norm.guard"),
        timestamp_writes: None,
    });
    kernels.guard.encode_in_pass(&mut pass, &binding);
}

impl ResidentTensor {
    /// Last-axis affine LayerNorm with a reusable first-order backward tape.
    /// Gamma and beta have equal shape, either `[cols]` or `[1, cols]`.
    /// Empty batches are valid; constant rows with zero epsilon fail on read.
    /// Arbitrary N-D views are packed on GPU. There is no CPU fallback.
    pub fn layer_norm_affine(
        &self,
        gamma: &Self,
        beta: &Self,
        epsilon: f32,
    ) -> Result<ResidentLayerNorm, TensorError> {
        validate_epsilon(epsilon)?;
        let shape = LayerNormShape::new(self.layout.shape(), gamma.layout.shape())?;
        if gamma.layout.shape() != beta.layout.shape() {
            return Err(LayerNormError::AffineShape.into());
        }
        let context = self.device.runtime().context();
        gamma.require_context(context)?;
        beta.require_context(context)?;
        let gpu = context.device();
        preflight(shape, &gpu.limits())?;
        let grid = groups(shape.rows, &gpu.limits())?;
        let kernels = self
            .device
            .0
            .normalization
            .get_or_init(|| LayerNormKernels::new(gpu));
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let input = self.contiguous_into(&mut encoder)?;
        let gamma = gamma.contiguous_into(&mut encoder)?;
        let beta = beta.contiguous_into(&mut encoder)?;
        let value = self.device.allocate_output(&self.layout)?;
        let centered = Shared::new(runtime::empty_buffer::<[u32; 4]>(
            gpu,
            "layer_norm.centered",
            self.layout.len().max(1),
            wgpu::BufferUsages::STORAGE,
        )?);
        let row_stats = Shared::new(runtime::empty_buffer::<[u32; 8]>(
            gpu,
            "layer_norm.row_stats",
            shape.rows.max(1),
            wgpu::BufferUsages::STORAGE,
        )?);
        let flags = validation(
            gpu,
            &mut encoder,
            &[input.flags(), gamma.flags(), beta.flags()],
        )?;
        let params = runtime::upload_slice(
            gpu,
            "layer_norm.params",
            &[Params {
                rows: shape.rows as u32,
                cols: shape.cols as u32,
                groups_x: grid[0],
                requested: 0,
                epsilon,
                scale: 1.0,
                beta_offset: 0,
                _pad: 0,
            }],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let binding = bind(
            gpu,
            &kernels.forward_layout,
            [
                input.values(),
                gamma.values(),
                beta.values(),
                &centered,
                &row_stats,
                value.values(),
                &flags,
                &params,
            ],
        );
        if shape.rows != 0 {
            dispatch(&mut encoder, &kernels.forward, &binding, grid);
        }
        capture(kernels, gpu, &mut encoder, &flags, value.flags());
        context.queue().submit(Some(encoder.finish()));
        Ok(ResidentLayerNorm {
            tape: LayerNormBackwardTape {
                centered,
                row_stats,
                gamma,
                layout: value.layout.clone(),
                flags: value.storage.flags.clone(),
                device: self.device.clone(),
                shape,
                epsilon,
            },
            value,
        })
    }

    /// Prepare only the centered row statistics required by the VJP. No
    /// affine output is evaluated or read, and there is no CPU fallback.
    pub fn layer_norm_vjp_tape(
        &self,
        gamma: &Self,
        epsilon: f32,
    ) -> Result<ResidentLayerNormVjp, TensorError> {
        validate_epsilon(epsilon)?;
        let shape = LayerNormShape::new(self.layout.shape(), gamma.layout.shape())?;
        let context = self.device.runtime().context();
        gamma.require_context(context)?;
        let gpu = context.device();
        preflight(shape, &gpu.limits())?;
        let grid = groups(shape.rows, &gpu.limits())?;
        let layout = NdLayout::contiguous(self.layout.shape())?;
        let kernels = self
            .device
            .0
            .normalization
            .get_or_init(|| LayerNormKernels::new(gpu));
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let input = self.contiguous_into(&mut encoder)?;
        let gamma = gamma.contiguous_into(&mut encoder)?;
        let centered = Shared::new(runtime::empty_buffer::<[u32; 4]>(
            gpu,
            "layer_norm.vjp.centered",
            self.layout.len().max(1),
            wgpu::BufferUsages::STORAGE,
        )?);
        let row_stats = Shared::new(runtime::empty_buffer::<[u32; 8]>(
            gpu,
            "layer_norm.vjp.row_stats",
            shape.rows.max(1),
            wgpu::BufferUsages::STORAGE,
        )?);
        let flags = Shared::new(validation(
            gpu,
            &mut encoder,
            &[input.flags(), gamma.flags()],
        )?);
        let unused_output = runtime::empty_buffer::<f32>(
            gpu,
            "layer_norm.vjp.unused_output",
            1,
            wgpu::BufferUsages::STORAGE,
        )?;
        let params = runtime::upload_slice(
            gpu,
            "layer_norm.vjp.params",
            &[Params {
                rows: shape.rows as u32,
                cols: shape.cols as u32,
                groups_x: grid[0],
                requested: 0,
                epsilon,
                scale: 1.0,
                beta_offset: 0,
                _pad: 0,
            }],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let binding = bind(
            gpu,
            &kernels.forward_layout,
            [
                input.values(),
                gamma.values(),
                gamma.values(),
                &centered,
                &row_stats,
                &unused_output,
                &flags,
                &params,
            ],
        );
        if shape.rows != 0 {
            dispatch(&mut encoder, &kernels.statistics, &binding, grid);
        }
        context.queue().submit(Some(encoder.finish()));
        Ok(ResidentLayerNormVjp {
            tape: LayerNormBackwardTape {
                centered,
                row_stats,
                gamma,
                layout,
                flags,
                device: self.device.clone(),
                shape,
                epsilon,
            },
        })
    }
}

impl ResidentLayerNorm {
    pub fn value(&self) -> &ResidentTensor {
        &self.value
    }

    /// Requested `(input, gamma, beta)` VJPs, all immutable and GPU-resident.
    /// The finite affine scale applies after row reduction, not to dx. Every
    /// returned tensor shares the guard for all requested results. Unrequested
    /// affine overflow cannot poison a requested input gradient.
    pub fn backward(
        &self,
        upstream: &ResidentTensor,
        parameter_gradient_scale: f32,
        requested: [bool; 3],
    ) -> Result<[Option<ResidentTensor>; 3], TensorError> {
        self.tape
            .backward(upstream, parameter_gradient_scale, requested)
    }
}

impl ResidentLayerNormVjp {
    /// Return only requested `(input, gamma, beta)` VJPs. The tape has no
    /// forward affine value or beta dependency to poison valid gradients.
    pub fn backward(
        &self,
        upstream: &ResidentTensor,
        parameter_gradient_scale: f32,
        requested: [bool; 3],
    ) -> Result<[Option<ResidentTensor>; 3], TensorError> {
        self.tape
            .backward(upstream, parameter_gradient_scale, requested)
    }
}

impl LayerNormBackwardTape {
    fn backward(
        &self,
        upstream: &ResidentTensor,
        parameter_gradient_scale: f32,
        requested: [bool; 3],
    ) -> Result<[Option<ResidentTensor>; 3], TensorError> {
        validate_gradient_scale(parameter_gradient_scale)?;
        if upstream.layout.shape() != self.layout.shape() {
            return Err(LayerNormError::CotangentShape.into());
        }
        let device = &self.device;
        let context = device.runtime().context();
        upstream.require_context(context)?;
        if !requested.iter().any(|&v| v) {
            return Ok([None, None, None]);
        }
        let gpu = context.device();
        let kernels = device
            .0
            .normalization
            .get()
            .expect("tape created LayerNorm kernels");
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let upstream = upstream.contiguous_into(&mut encoder)?;
        let flags = validation(gpu, &mut encoder, &[&self.flags, upstream.flags()])?;
        let empty = NdLayout::contiguous(&[0])?;
        let mut dx = device.allocate_output(if requested[0] { &self.layout } else { &empty })?;
        let affine_len = self.shape.cols * (usize::from(requested[1]) + usize::from(requested[2]));
        let affine = device.allocate_output(&NdLayout::contiguous(&[affine_len])?)?;
        // Fresh private buffers are not exposed until both kernels and their
        // common guard have been submitted.
        Shared::get_mut(&mut dx.storage)
            .expect("fresh output")
            .flags = affine.storage.flags.clone();
        let mut params = Params {
            rows: self.shape.rows as u32,
            cols: self.shape.cols as u32,
            groups_x: 0,
            requested: u32::from(requested[0])
                | (u32::from(requested[1]) << 1)
                | (u32::from(requested[2]) << 2),
            epsilon: self.epsilon,
            scale: parameter_gradient_scale,
            beta_offset: if requested[1] {
                self.shape.cols as u32
            } else {
                0
            },
            _pad: 0,
        };
        for (enabled, count, pipeline) in [
            (
                requested[0] && self.shape.rows > 0,
                self.shape.rows,
                &kernels.input,
            ),
            (
                requested[1] || requested[2],
                self.shape.cols,
                &kernels.affine,
            ),
        ] {
            if !enabled {
                continue;
            }
            let grid = groups(count, &gpu.limits())?;
            params.groups_x = grid[0];
            let uniform = runtime::upload_slice(
                gpu,
                "layer_norm.backward.params",
                &[params],
                wgpu::BufferUsages::UNIFORM,
            )?;
            let binding = bind(
                gpu,
                &kernels.backward_layout,
                [
                    &self.centered,
                    self.gamma.values(),
                    upstream.values(),
                    &self.row_stats,
                    dx.values(),
                    affine.values(),
                    &flags,
                    &uniform,
                ],
            );
            dispatch(&mut encoder, pipeline, &binding, grid);
        }
        capture(kernels, gpu, &mut encoder, &flags, affine.flags());
        context.queue().submit(Some(encoder.finish()));
        let dg = requested[1]
            .then(|| {
                affine
                    .narrow(0, 0, self.shape.cols)?
                    .reshape(self.gamma.layout.shape())
            })
            .transpose()?;
        let db = requested[2]
            .then(|| {
                affine
                    .narrow(0, params.beta_offset as usize, self.shape.cols)?
                    .reshape(self.gamma.layout.shape())
            })
            .transpose()?;
        Ok([requested[0].then_some(dx), dg, db])
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
