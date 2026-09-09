//! Checked reverse-mode pointwise contributions and deterministic unbroadcast.
//! Forward is recomputed from immutable inputs; no mutable tape escapes.
use super::*;
use st_kernel_contracts::pointwise::BroadcastAdjoint;

#[derive(Debug)]
struct Reduction {
    layout: NdLayout,
    direct: bool,
    source_offset: u64,
    partial_len: usize,
    first: wgpu::Buffer,
    first_grid: [u32; 2],
    second: Option<(wgpu::Buffer, [u32; 2])>,
}

#[derive(Debug)]
pub struct PointwiseVjpPlan {
    forward: PointwisePlan,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    reduction_layout: wgpu::BindGroupLayout,
    reduction_pipeline: wgpu::ComputePipeline,
    reductions: Vec<Reduction>,
    contribution_len: usize,
}

fn groups_grid(groups: usize, limits: &wgpu::Limits) -> Result<[u32; 2], TensorError> {
    let groups = u32::try_from(groups.max(1)).map_err(|_| TensorError::Limit("VJP groups"))?;
    let x = groups.min(limits.max_compute_workgroups_per_dimension);
    if x == 0 || groups.div_ceil(x) > limits.max_compute_workgroups_per_dimension {
        return Err(TensorError::Limit("VJP grid"));
    }
    Ok([x, groups.div_ceil(x)])
}

fn buffer_layout(gpu: &wgpu::Device, count: usize, writable: &[usize]) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = (0..count)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: !writable.contains(&binding),
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pointwise.vjp.layout"),
        entries: &entries,
    })
}

fn pipeline(
    gpu: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: String,
) -> wgpu::ComputePipeline {
    let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pointwise.vjp.shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pointwise.vjp.pipeline_layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pointwise.vjp"),
        layout: Some(&layout),
        module: &module,
        entry_point: "main",
        compilation_options: Default::default(),
    })
}

fn reduction_source() -> String {
    substitute_ops(include_str!("../../shaders/pointwise_unbroadcast.wgsl").to_owned())
}

impl PointwiseVjpPlan {
    /// Preparation is opt-in: existing forward-only plans compile no VJP shader.
    pub fn new(forward: PointwisePlan) -> Result<Self, TensorError> {
        let gpu = forward.device.runtime().context().device();
        let limits = gpu.limits();
        let count = forward.chain.input_count();
        if count + 5 > limits.max_storage_buffers_per_shader_stage as usize
            || count + 5 > limits.max_bindings_per_bind_group as usize
        {
            return Err(TensorError::Limit("VJP input bindings"));
        }
        let n = forward.layouts[0].len();
        let contribution_len = n
            .checked_mul(count)
            .ok_or(TensorError::Limit("VJP contributions"))?;
        storage_limit(contribution_len, &limits)?;
        let mut reductions = Vec::new();
        for (slot, input) in forward.layouts.iter().enumerate() {
            let map = BroadcastAdjoint::new(forward.layouts[0].shape(), input.shape())?;
            let elements = map.input_layout().len();
            let partials = map.reduction_len().div_ceil(256).max(1);
            let partial_len = elements
                .checked_mul(partials)
                .ok_or(TensorError::Limit("VJP partials"))?;
            storage_limit(partial_len, &limits)?;
            let direct = map.reduction_len() == 1;
            let first_grid = groups_grid(if direct { 0 } else { partial_len }, &limits)?;
            let rank = map.output_layout().rank();
            let mut meta = vec![
                first_grid[0],
                partial_len.max(1) as u32,
                0,
                (slot * n) as u32,
                elements as u32,
                map.reduction_len() as u32,
                partials as u32,
                rank as u32,
            ];
            let mut padded = vec![1usize; rank - input.rank()];
            padded.extend_from_slice(input.shape());
            for values in [
                &padded[..],
                map.reduction_shape(),
                map.output_layout().strides(),
            ] {
                meta.extend(values.iter().map(|&n| n as u32));
            }
            storage_limit(meta.len(), &limits)?;
            let first = runtime::upload_slice(
                gpu,
                "vjp.reduction.first",
                &meta,
                wgpu::BufferUsages::STORAGE,
            )?;
            let second = if partials > 1 {
                let grid = groups_grid(elements, &limits)?;
                meta[0] = grid[0];
                meta[1] = elements.max(1) as u32;
                meta[2] = 1;
                meta[3] = 0;
                Some((
                    runtime::upload_slice(
                        gpu,
                        "vjp.reduction.second",
                        &meta,
                        wgpu::BufferUsages::STORAGE,
                    )?,
                    grid,
                ))
            } else {
                None
            };
            reductions.push(Reduction {
                layout: map.input_layout().clone(),
                direct,
                source_offset: (slot * n) as u64 * 4,
                partial_len,
                first,
                first_grid,
                second,
            });
        }
        let layout = buffer_layout(gpu, count + 5, &[count, count + 3]);
        let pipeline = pipeline(gpu, &layout, generated_source(&forward.chain, true));
        let reduction_layout = buffer_layout(gpu, 4, &[1, 3]);
        let reduction_pipeline = self::pipeline(gpu, &reduction_layout, reduction_source());
        Ok(Self {
            forward,
            layout,
            pipeline,
            reduction_layout,
            reduction_pipeline,
            reductions,
            contribution_len,
        })
    }

    pub fn forward(&self) -> &PointwisePlan {
        &self.forward
    }

    pub fn run(
        &self,
        inputs: &[&ResidentTensor],
        cotangent: &ResidentTensor,
    ) -> Result<Vec<ResidentTensor>, TensorError> {
        if inputs.len() != self.forward.layouts.len() {
            return Err(PointwiseError::Operands.into());
        }
        let device = &self.forward.device;
        let context = device.runtime().context();
        for (input, layout) in inputs.iter().zip(&self.forward.layouts) {
            input.require_context(context)?;
            if input.layout() != layout {
                return Err(PointwiseError::LayoutMismatch.into());
            }
        }
        cotangent.require_context(context)?;
        if cotangent.layout().shape() != self.forward.layouts[0].shape() {
            return Err(PointwiseError::LayoutMismatch.into());
        }
        let cotangent = cotangent.contiguous()?;
        let gpu = context.device();
        let storage = wgpu::BufferUsages::STORAGE;
        let copy_storage = storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let flags = runtime::empty_buffer::<u32>(gpu, "vjp.flags", 1, copy_storage)?;
        let inherited = runtime::empty_buffer::<u32>(
            gpu,
            "vjp.inherited",
            inputs.len() + 1,
            storage | wgpu::BufferUsages::COPY_DST,
        )?;
        let mut encoder = gpu.create_command_encoder(&Default::default());
        for (i, input) in inputs.iter().copied().chain([&cotangent]).enumerate() {
            encoder.copy_buffer_to_buffer(input.flags(), 0, &inherited, i as u64 * 4, 4);
        }
        let mut result = Vec::new();
        for reduction in &self.reductions {
            let values = runtime::empty_buffer::<f32>(
                gpu,
                "vjp.gradient",
                reduction.layout.len().max(1),
                copy_storage,
            )?;
            let result_flags =
                runtime::empty_buffer::<u32>(gpu, "vjp.gradient.flags", 1, copy_storage)?;
            result.push(ResidentTensor {
                storage: Shared::new(Storage {
                    values,
                    flags: result_flags,
                }),
                layout: reduction.layout.clone(),
                device: device.clone(),
            });
        }
        self.encode_into(
            &mut encoder,
            &inputs.iter().map(|t| t.values()).collect::<Vec<_>>(),
            cotangent.values(),
            &result.iter().map(|t| t.values()).collect::<Vec<_>>(),
            &inherited,
            &flags,
        )?;
        // Every returned gradient inherits failures from every input and reduction,
        // including a later broadcast sum overflowing after another slot succeeded.
        for gradient in &result {
            encoder.copy_buffer_to_buffer(&flags, 0, gradient.flags(), 0, 4);
        }
        context.queue().submit(Some(encoder.finish()));
        Ok(result)
    }

    /// Same VJP/reductions as run(), recorded into a private graph workspace.
    /// Callers validate device/layout and clear flags before encoding a step.
    pub(crate) fn encode_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &[&wgpu::Buffer],
        cotangent: &wgpu::Buffer,
        gradients: &[&wgpu::Buffer],
        inherited: &wgpu::Buffer,
        flags: &wgpu::Buffer,
    ) -> Result<(), TensorError> {
        assert_eq!(inputs.len(), self.forward.layouts.len());
        assert_eq!(gradients.len(), self.reductions.len());
        let gpu = self.forward.device.runtime().context().device();
        let contributions = runtime::empty_buffer::<f32>(
            gpu,
            "vjp.contributions",
            self.contribution_len.max(1),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let buffers: Vec<_> = inputs
            .iter()
            .copied()
            .chain([
                &contributions,
                &self.forward.metadata,
                inherited,
                flags,
                cotangent,
            ])
            .collect();
        encode(
            gpu,
            encoder,
            &self.layout,
            &self.pipeline,
            &buffers,
            [self.forward.grid[0], self.forward.grid[1]],
        );
        for (reduction, &values) in self.reductions.iter().zip(gradients) {
            if reduction.direct {
                if !reduction.layout.is_empty() {
                    encoder.copy_buffer_to_buffer(
                        &contributions,
                        reduction.source_offset,
                        values,
                        0,
                        reduction.layout.len() as u64 * 4,
                    );
                }
            } else if let Some((metadata, grid)) = &reduction.second {
                let partials = runtime::empty_buffer::<f32>(
                    gpu,
                    "vjp.partials",
                    reduction.partial_len.max(1),
                    wgpu::BufferUsages::STORAGE,
                )?;
                encode(
                    gpu,
                    encoder,
                    &self.reduction_layout,
                    &self.reduction_pipeline,
                    &[&contributions, &partials, &reduction.first, flags],
                    reduction.first_grid,
                );
                encode(
                    gpu,
                    encoder,
                    &self.reduction_layout,
                    &self.reduction_pipeline,
                    &[&partials, values, metadata, flags],
                    *grid,
                );
            } else {
                encode(
                    gpu,
                    encoder,
                    &self.reduction_layout,
                    &self.reduction_pipeline,
                    &[&contributions, values, &reduction.first, flags],
                    reduction.first_grid,
                );
            }
        }
        Ok(())
    }
}

pub(super) fn encode(
    gpu: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    layout: &wgpu::BindGroupLayout,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&wgpu::Buffer],
    grid: [u32; 2],
) {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let binding = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vjp.operands"),
        layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("pointwise.vjp"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &binding, &[]);
    pass.dispatch_workgroups(grid[0], grid[1], 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_kernel_contracts::pointwise::PointwiseStep;
    #[test]
    fn vjp_and_unbroadcast_shaders_validate() {
        let chain = PointwiseChain::new(
            3,
            [
                PointwiseStep {
                    op: ElementwiseOp::Add,
                    rhs: Some(0),
                },
                PointwiseStep {
                    op: ElementwiseOp::Multiply,
                    rhs: Some(1),
                },
                PointwiseStep {
                    op: ElementwiseOp::Multiply,
                    rhs: Some(2),
                },
                PointwiseStep {
                    op: ElementwiseOp::Relu,
                    rhs: None,
                },
                PointwiseStep {
                    op: ElementwiseOp::Gelu,
                    rhs: None,
                },
            ]
            .repeat(20),
        )
        .unwrap();
        for source in [generated_source(&chain, true), reduction_source()] {
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
        assert_eq!(
            groups_grid(
                5,
                &wgpu::Limits {
                    max_compute_workgroups_per_dimension: 3,
                    ..Default::default()
                }
            )
            .unwrap(),
            [3, 2]
        );
    }
}
