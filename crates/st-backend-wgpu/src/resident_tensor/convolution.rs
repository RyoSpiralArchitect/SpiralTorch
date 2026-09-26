//! Guarded depthwise convolution over immutable NCHW resident tensors.

use super::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    channels: u32,
    input_h: u32,
    input_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: i32,
    pad_w: i32,
    dilation_h: u32,
    dilation_w: u32,
    output_h: u32,
    output_w: u32,
    span: u32,
    output_len: u32,
    groups_x: u32,
}

#[derive(Debug)]
pub(super) struct DepthwiseKernels {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl DepthwiseKernels {
    fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tensor.depthwise.layout"),
            entries: &(0..9)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 8 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 3 || (4..7).contains(&binding),
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tensor.depthwise.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor.depthwise.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/depthwise_conv2d.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tensor.depthwise"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Self { layout, pipeline }
    }
}

fn output_extent(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize, TensorError> {
    if input == 0 || kernel == 0 || stride == 0 || dilation == 0 {
        return Err(TensorError::ConvolutionShape("zero spatial dimension"));
    }
    let padded = padding
        .checked_mul(2)
        .and_then(|pad| input.checked_add(pad))
        .ok_or(TensorError::ConvolutionShape("padded extent overflow"))?;
    let effective = (kernel - 1)
        .checked_mul(dilation)
        .and_then(|span| span.checked_add(1))
        .ok_or(TensorError::ConvolutionShape("kernel extent overflow"))?;
    if padded < effective || padded > i32::MAX as usize {
        return Err(TensorError::ConvolutionShape("invalid padded extent"));
    }
    Ok((padded - effective) / stride + 1)
}

fn preflight(
    input: &ResidentTensor,
    weights: &ResidentTensor,
    bias: &ResidentTensor,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<([usize; 4], Params, [u32; 2]), TensorError> {
    let context = input.device.runtime().context();
    input.require_context(weights.device.runtime().context())?;
    input.require_context(bias.device.runtime().context())?;
    let [batch, channels, input_h, input_w] = input.layout.shape() else {
        return Err(TensorError::ConvolutionShape("input must be NCHW"));
    };
    let [weight_channels, kernel_h, kernel_w] = weights.layout.shape() else {
        return Err(TensorError::ConvolutionShape("weights must be [C, KH, KW]"));
    };
    if bias.layout.shape() != [*channels] || *channels == 0 || *weight_channels != *channels {
        return Err(TensorError::ConvolutionShape("channel or bias mismatch"));
    }
    let output_h = output_extent(*input_h, *kernel_h, stride.0, padding.0, dilation.0)?;
    let output_w = output_extent(*input_w, *kernel_w, stride.1, padding.1, dilation.1)?;
    let shape = [*batch, *channels, output_h, output_w];
    let output_len = shape
        .iter()
        .try_fold(1usize, |size, &dim| size.checked_mul(dim))
        .ok_or(TensorError::ConvolutionShape("output volume overflow"))?;
    let span = kernel_h
        .checked_mul(*kernel_w)
        .ok_or(TensorError::ConvolutionShape("kernel span overflow"))?;
    let limits = context.device().limits();
    if limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_storage_buffers_per_shader_stage < 8
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_bindings_per_bind_group < 9
        || limits.max_uniform_buffer_binding_size < std::mem::size_of::<Params>() as u32
    {
        return Err(TensorError::Limit("depthwise pipeline"));
    }
    storage_limit(output_len, &limits)?;
    let groups = output_len.div_ceil(64).max(1);
    let groups_x = groups.min(limits.max_compute_workgroups_per_dimension as usize);
    if groups_x == 0
        || groups.div_ceil(groups_x) > limits.max_compute_workgroups_per_dimension as usize
    {
        return Err(TensorError::Limit("depthwise dispatch grid"));
    }
    let to_u32 = |value| u32::try_from(value).map_err(|_| TensorError::Limit("depthwise index"));
    let params = Params {
        channels: to_u32(*channels)?,
        input_h: to_u32(*input_h)?,
        input_w: to_u32(*input_w)?,
        kernel_h: to_u32(*kernel_h)?,
        kernel_w: to_u32(*kernel_w)?,
        stride_h: to_u32(stride.0)?,
        stride_w: to_u32(stride.1)?,
        pad_h: i32::try_from(padding.0).map_err(|_| TensorError::Limit("depthwise padding"))?,
        pad_w: i32::try_from(padding.1).map_err(|_| TensorError::Limit("depthwise padding"))?,
        dilation_h: to_u32(dilation.0)?,
        dilation_w: to_u32(dilation.1)?,
        output_h: to_u32(output_h)?,
        output_w: to_u32(output_w)?,
        span: to_u32(span)?,
        output_len: to_u32(output_len)?,
        groups_x: to_u32(groups_x)?,
    };
    Ok((
        shape,
        params,
        [groups_x as u32, groups.div_ceil(groups_x) as u32],
    ))
}

pub(super) fn forward(
    input: &ResidentTensor,
    weights: &ResidentTensor,
    bias: &ResidentTensor,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<ResidentTensor, TensorError> {
    let (shape, params, grid) = preflight(input, weights, bias, stride, padding, dilation)?;
    let context = input.device.runtime().context();
    let device = context.device();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tensor.depthwise.encoder"),
    });
    let input = input.contiguous_into(&mut encoder)?;
    let weights = weights.contiguous_into(&mut encoder)?;
    let bias = bias.contiguous_into(&mut encoder)?;
    let output = input
        .device
        .allocate_output(&NdLayout::contiguous(&shape)?)?;
    let params = runtime::upload_slice(
        device,
        "tensor.depthwise.params",
        &[params],
        wgpu::BufferUsages::UNIFORM,
    )?;
    let kernels = input
        .device
        .0
        .convolution
        .get_or_init(|| DepthwiseKernels::new(device));
    let buffers = [
        input.values(),
        weights.values(),
        bias.values(),
        output.values(),
        input.flags(),
        weights.flags(),
        bias.flags(),
        output.flags(),
        &params,
    ];
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tensor.depthwise.bind_group"),
        layout: &kernels.layout,
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tensor.depthwise.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernels.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(grid[0], grid[1], 1);
    }
    context.queue().submit(Some(encoder.finish()));
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_is_valid_wgsl() {
        let module =
            naga::front::wgsl::parse_str(include_str!("shaders/depthwise_conv2d.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn output_extent_checks_geometry() {
        assert_eq!(output_extent(5, 2, 2, 1, 2).unwrap(), 3);
        assert!(output_extent(2, 7, 1, 0, 1).is_err());
        assert!(output_extent(4, 3, 0, 1, 1).is_err());
        assert!(output_extent(4, 3, 1, usize::MAX, 1).is_err());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn resident_depthwise_matches_reference_and_preserves_guards_on_real_gpu() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let (runtime, _) =
            runtime::ensure_default_runtime_blocking("tensor.depthwise.test").unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let device = TensorDevice::new(runtime).unwrap();
        let nhwc: Vec<f32> = (0..2 * 5 * 4 * 2)
            .map(|index| (index as f32 - 40.0) / 23.0)
            .collect();
        let input = device
            .upload(&[2, 5, 4, 2], &nhwc)
            .unwrap()
            .permute(&[0, 3, 1, 2])
            .unwrap();
        let raw_weights = [
            0.2, 0.7, -0.3, 0.1, 0.4, -0.5, -0.2, 0.3, 0.8, -0.4, 0.6, 0.2,
        ];
        let weights = device
            .upload(&[2, 3, 2], &raw_weights)
            .unwrap()
            .permute(&[0, 2, 1])
            .unwrap();
        let bias = device
            .upload(&[3], &[99.0, 0.125, -0.25])
            .unwrap()
            .narrow(0, 1, 2)
            .unwrap();
        let output = input
            .depthwise_conv2d(&weights, &bias, (2, 1), (1, 2), (2, 1))
            .unwrap();
        assert_eq!(output.layout().shape(), &[2, 2, 3, 6]);
        let actual = output.snapshot().unwrap().read().unwrap();
        let mut expected = Vec::new();
        for batch in 0..2 {
            for channel in 0..2 {
                for out_y in 0..3 {
                    for out_x in 0..6 {
                        let mut value = [0.125, -0.25][channel];
                        for ky in 0..2 {
                            for kx in 0..3 {
                                let y = out_y * 2 + ky * 2;
                                let x = out_x + kx;
                                if y >= 1 && x >= 2 && y - 1 < 5 && x - 2 < 4 {
                                    let input_index =
                                        ((batch * 5 + (y - 1)) * 4 + (x - 2)) * 2 + channel;
                                    let weight_index = (channel * 3 + kx) * 2 + ky;
                                    value += nhwc[input_index] * raw_weights[weight_index];
                                }
                            }
                        }
                        expected.push(value);
                    }
                }
            }
        }
        for (left, right) in actual.iter().zip(expected) {
            assert!((left - right).abs() <= 1e-6, "{left} != {right}");
        }
        let shared = device.upload(&[1, 1, 1, 1], &[1.]).unwrap();
        let shared_output = shared
            .depthwise_conv2d(
                &shared.reshape(&[1, 1, 1]).unwrap(),
                &shared.reshape(&[1]).unwrap(),
                (1, 1),
                (0, 0),
                (1, 1),
            )
            .unwrap();
        assert_eq!(shared_output.snapshot().unwrap().read().unwrap(), [2.]);
        assert!(matches!(
            input.depthwise_conv2d(&weights, &bias, (0, 1), (0, 0), (1, 1)),
            Err(TensorError::ConvolutionShape(_))
        ));
        assert!(matches!(
            input.depthwise_conv2d(
                &weights,
                &device.upload(&[1], &[0.]).unwrap(),
                (1, 1),
                (0, 0),
                (1, 1)
            ),
            Err(TensorError::ConvolutionShape(_))
        ));

        let invalid = device
            .upload(&[1, 1, 2, 2], &[0., 0., 0., f32::MAX])
            .unwrap()
            .mul(&device.upload(&[], &[2.]).unwrap())
            .unwrap();
        let identity = device.upload(&[1, 1, 1], &[1.]).unwrap();
        let zero_bias = device.upload(&[1], &[0.]).unwrap();
        let finite = device
            .upload(&[1, 1, 2, 2], &[0., 0., 0., f32::MAX])
            .unwrap()
            .depthwise_conv2d(&identity, &zero_bias, (2, 2), (0, 0), (1, 1))
            .unwrap();
        assert_eq!(finite.snapshot().unwrap().read().unwrap(), [0.]);
        let inherited = invalid
            .depthwise_conv2d(&identity, &zero_bias, (2, 2), (0, 0), (1, 1))
            .unwrap();
        assert!(matches!(
            inherited.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
        let empty = invalid
            .broadcast_to(&[0, 1, 2, 2])
            .unwrap()
            .depthwise_conv2d(&identity, &zero_bias, (2, 2), (0, 0), (1, 1))
            .unwrap();
        assert!(matches!(
            empty.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
        let overflow = device
            .upload(&[1, 1, 1, 1], &[f32::MAX])
            .unwrap()
            .depthwise_conv2d(
                &device.upload(&[1, 1, 1], &[2.]).unwrap(),
                &zero_bias,
                (1, 1),
                (0, 0),
                (1, 1),
            )
            .unwrap();
        assert!(matches!(
            overflow.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
}
