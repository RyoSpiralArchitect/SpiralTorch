use super::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ChartParams {
    rows: u32,
    cols: u32,
    input_offset: u32,
    groups_x: u32,
    groups: u32,
    padding: [u32; 3],
}

pub(super) fn source() -> String {
    include_str!("../shaders/fisher_chart.wgsl")
        .replace(
            "ROUNDED_ADD",
            include_str!("../../shaders/rounded_add.wgsl"),
        )
        .replace(
            "WIDE_ARITHMETIC",
            include_str!("../shaders/layer_norm_wide.wgsl"),
        )
        .replace("INVALID_TENSOR_FLAG", &format!("{INVALID_TENSOR_FLAG}u"))
}

#[derive(Debug)]
pub(in crate::resident_tensor) struct FisherChartKernels {
    layout: wgpu::BindGroupLayout,
    forward: wgpu::ComputePipeline,
}

impl FisherChartKernels {
    fn new(gpu: &wgpu::Device) -> Self {
        let layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fisher_chart.layout"),
            entries: &(0..=3)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 3 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding == 0,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fisher_chart.shader"),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let pipeline_layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fisher_chart.pipeline"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = |entry| {
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: entry,
                compilation_options: Default::default(),
            })
        };
        Self {
            forward: pipeline("forward"),
            layout,
        }
    }
}

pub(super) fn roots(input: &ResidentTensor) -> Result<ResidentTensor, TensorError> {
    let [batch, time, cols] =
        <[usize; 3]>::try_from(input.layout.shape()).map_err(|_| FisherRaoError::Shape)?;
    let device = &input.device;
    let context = device.runtime().context();
    let limits = context.device().limits();
    validate_view(&input.layout, (input.values().size() / 4) as usize, &limits)?;
    let [x, y, groups] = grid(batch * time, &limits)?;
    if limits.max_storage_buffers_per_shader_stage < 3
        || limits.max_bindings_per_bind_group < 4
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_uniform_buffer_binding_size < 32
    {
        return Err(TensorError::Limit("Fisher simplex bindings"));
    }
    storage_limit(input.layout.len(), &limits)?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let values = input.contiguous_into(&mut encoder)?;
    let guard = flags(device, input, &mut encoder)?;
    let output = device
        .allocate_output_with_guard(&NdLayout::contiguous(input.layout.shape())?, Some(guard))?;
    let p = ChartParams {
        rows: (batch * time) as u32,
        cols: cols as u32,
        input_offset: values.layout.offset() as u32,
        groups_x: x,
        groups,
        padding: [0; 3],
    };
    let uniform = runtime::upload_slice(
        context.device(),
        "fisher_chart.params",
        &[p],
        wgpu::BufferUsages::UNIFORM,
    )?;
    let kernels = device
        .0
        .fisher_chart
        .get_or_init(|| FisherChartKernels::new(context.device()));
    let buffers = [values.values(), output.values(), output.flags(), &uniform];
    let group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fisher_chart.bind"),
            layout: &kernels.layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&kernels.forward);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
    context.queue().submit(Some(encoder.finish()));
    Ok(device.guard_together(&[&output, &values])?.remove(0))
}
