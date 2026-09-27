//! Deterministic resident depthwise-convolution VJPs without float atomics.

use super::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VjpParams {
    geometry: Params,
    batch: u32,
    _padding: [u32; 3],
}

#[derive(Debug)]
pub(crate) struct DepthwiseVjpKernels {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 3],
}

impl DepthwiseVjpKernels {
    fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tensor.depthwise_vjp.layout"),
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
            label: Some("tensor.depthwise_vjp.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor.depthwise_vjp.shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/depthwise_conv2d_vjp.wgsl").into(),
            ),
        });
        let pipelines = ["input_vjp", "weight_vjp", "bias_vjp"].map(|entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point,
                compilation_options: Default::default(),
            })
        });
        Self { layout, pipelines }
    }
}

fn grid_for(elements: usize, limits: &wgpu::Limits) -> Result<[u32; 2], TensorError> {
    storage_limit(elements, limits)?;
    let groups = elements.div_ceil(64).max(1);
    let groups_x = groups.min(limits.max_compute_workgroups_per_dimension as usize);
    if groups_x == 0
        || groups.div_ceil(groups_x) > limits.max_compute_workgroups_per_dimension as usize
    {
        return Err(TensorError::Limit("depthwise VJP dispatch grid"));
    }
    Ok([groups_x as u32, groups.div_ceil(groups_x) as u32])
}

pub(crate) fn backward(
    input: &ResidentTensor,
    weights: &ResidentTensor,
    upstream: &ResidentTensor,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<[ResidentTensor; 3], TensorError> {
    let (output_shape, geometry, _) =
        preflight_geometry(input, weights, stride, padding, dilation)?;
    let context = input.device.runtime().context();
    input.require_context(upstream.device.runtime().context())?;
    if upstream.layout.shape() != output_shape {
        return Err(TensorError::ConvolutionShape("cotangent shape mismatch"));
    }
    let gpu = context.device();
    let limits = gpu.limits();
    if limits.max_uniform_buffer_binding_size < std::mem::size_of::<VjpParams>() as u32 {
        return Err(TensorError::Limit("depthwise VJP uniform"));
    }
    let layouts = [
        NdLayout::contiguous(input.layout.shape())?,
        NdLayout::contiguous(weights.layout.shape())?,
        NdLayout::contiguous(&[output_shape[1]])?,
    ];
    let grids = [
        grid_for(layouts[0].len(), &limits)?,
        grid_for(layouts[1].len(), &limits)?,
        grid_for(layouts[2].len(), &limits)?,
    ];
    let guard = Shared::new(runtime::empty_buffer::<u32>(
        gpu,
        "tensor.depthwise_vjp.guard",
        1,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )?);
    let outputs = [
        input
            .device
            .allocate_output_with_guard(&layouts[0], Some(guard.clone()))?,
        input
            .device
            .allocate_output_with_guard(&layouts[1], Some(guard.clone()))?,
        input
            .device
            .allocate_output_with_guard(&layouts[2], Some(guard))?,
    ];
    let mut encoder = gpu.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tensor.depthwise_vjp.encoder"),
    });
    let input = input.contiguous_into(&mut encoder)?;
    let weights = weights.contiguous_into(&mut encoder)?;
    let upstream = upstream.contiguous_into(&mut encoder)?;
    let kernels = input
        .device
        .0
        .depthwise_vjp
        .get_or_init(|| DepthwiseVjpKernels::new(gpu));
    let source_values = [input.values(), weights.values(), upstream.values()];
    let source_flags = [input.flags(), weights.flags(), upstream.flags()];
    for (index, output) in outputs.iter().enumerate() {
        let mut target_geometry = geometry;
        target_geometry.output_len = u32::try_from(layouts[index].len())
            .map_err(|_| TensorError::Limit("depthwise VJP index"))?;
        target_geometry.groups_x = grids[index][0];
        let params = runtime::upload_slice(
            gpu,
            "tensor.depthwise_vjp.params",
            &[VjpParams {
                geometry: target_geometry,
                batch: u32::try_from(output_shape[0])
                    .map_err(|_| TensorError::Limit("depthwise VJP batch"))?,
                _padding: [0; 3],
            }],
            wgpu::BufferUsages::UNIFORM,
        )?;
        let buffers = [
            source_values[0],
            source_values[1],
            source_values[2],
            output.values(),
            source_flags[0],
            source_flags[1],
            source_flags[2],
            output.flags(),
            &params,
        ];
        let binding = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tensor.depthwise_vjp.binding"),
            layout: &kernels.layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tensor.depthwise_vjp.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernels.pipelines[index]);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(grids[index][0], grids[index][1], 1);
        }
    }
    context.queue().submit(Some(encoder.finish()));
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_arch = "wasm32"))]
    use super::*;

    #[test]
    fn shader_is_valid_wgsl() {
        let module =
            naga::front::wgsl::parse_str(include_str!("../shaders/depthwise_conv2d_vjp.wgsl"))
                .unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn depthwise_vjp_matches_host_with_views_and_geometry_on_real_gpu() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let (runtime, _) =
            runtime::ensure_default_runtime_blocking("tensor.depthwise_vjp.test").unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let device = TensorDevice::new(runtime).unwrap();
        let (batch, channels, input_h, input_w) = (2, 2, 4, 5);
        let (kernel_h, kernel_w) = (2, 3);
        let (output_h, output_w) = (2, 7);
        let (stride, padding, dilation) = ((2, 1), (1, 2), (2, 1));
        let input_values: Vec<f32> = (0..batch * channels * input_h * input_w)
            .map(|index| (index as f32 - 34.0) / 29.0)
            .collect();
        let weight_values: Vec<f32> = (0..channels * kernel_h * kernel_w)
            .map(|index| (index as f32 - 5.0) / 17.0)
            .collect();
        let upstream_values: Vec<f32> = (0..batch * channels * output_h * output_w)
            .map(|index| ((index * 7 % 31) as f32 - 15.0) / 19.0)
            .collect();
        let mut transposed_input = vec![0.0; input_values.len()];
        let mut transposed_upstream = vec![0.0; upstream_values.len()];
        for b in 0..batch {
            for c in 0..channels {
                for y in 0..input_h {
                    for x in 0..input_w {
                        transposed_input[((b * channels + c) * input_w + x) * input_h + y] =
                            input_values[((b * channels + c) * input_h + y) * input_w + x];
                    }
                }
                for y in 0..output_h {
                    for x in 0..output_w {
                        transposed_upstream[((b * channels + c) * output_w + x) * output_h + y] =
                            upstream_values[((b * channels + c) * output_h + y) * output_w + x];
                    }
                }
            }
        }
        let input = device
            .upload(&[batch, channels, input_w, input_h], &transposed_input)
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap();
        let upstream = device
            .upload(&[batch, channels, output_w, output_h], &transposed_upstream)
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap();
        let mut transposed_weights = vec![0.0; weight_values.len()];
        for c in 0..channels {
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    transposed_weights[(c * kernel_w + kx) * kernel_h + ky] =
                        weight_values[(c * kernel_h + ky) * kernel_w + kx];
                }
            }
        }
        let weights = device
            .upload(&[channels, kernel_w, kernel_h], &transposed_weights)
            .unwrap()
            .permute(&[0, 2, 1])
            .unwrap();
        let actual = input
            .depthwise_conv2d_vjp(&weights, &upstream, stride, padding, dilation)
            .unwrap();
        assert_eq!(
            actual[0].layout().shape(),
            &[batch, channels, input_h, input_w]
        );
        assert_eq!(actual[1].layout().shape(), &[channels, kernel_h, kernel_w]);
        assert_eq!(actual[2].layout().shape(), &[channels]);

        let mut expected_input = vec![0.0f32; input_values.len()];
        let mut expected_weight = vec![0.0f32; weight_values.len()];
        let mut expected_bias = vec![0.0f32; channels];
        for b in 0..batch {
            for (c, bias_gradient) in expected_bias.iter_mut().enumerate().take(channels) {
                for oy in 0..output_h {
                    for ox in 0..output_w {
                        let output_index = ((b * channels + c) * output_h + oy) * output_w + ox;
                        let seed = upstream_values[output_index];
                        *bias_gradient += seed;
                        for ky in 0..kernel_h {
                            for kx in 0..kernel_w {
                                let iy = oy * stride.0 + ky * dilation.0;
                                let ix = ox * stride.1 + kx * dilation.1;
                                if iy >= padding.0
                                    && iy - padding.0 < input_h
                                    && ix >= padding.1
                                    && ix - padding.1 < input_w
                                {
                                    let input_index =
                                        ((b * channels + c) * input_h + iy - padding.0) * input_w
                                            + ix
                                            - padding.1;
                                    let weight_index = (c * kernel_h + ky) * kernel_w + kx;
                                    expected_input[input_index] +=
                                        seed * weight_values[weight_index];
                                    expected_weight[weight_index] +=
                                        seed * input_values[input_index];
                                }
                            }
                        }
                    }
                }
            }
        }
        for (gradient, expected) in actual.iter().zip([
            expected_input.as_slice(),
            expected_weight.as_slice(),
            expected_bias.as_slice(),
        ]) {
            let values = gradient.snapshot().unwrap().read().unwrap();
            for (index, (&value, &reference)) in values.iter().zip(expected).enumerate() {
                assert!(
                    (value - reference).abs() <= 2e-5 * (1.0 + reference.abs()),
                    "index={index} gpu={value} cpu={reference}"
                );
            }
        }

        assert!(matches!(
            input.depthwise_conv2d_vjp(
                &weights,
                &device.upload(&[1, 1, 1, 1], &[1.0]).unwrap(),
                stride,
                padding,
                dilation
            ),
            Err(TensorError::ConvolutionShape("cotangent shape mismatch"))
        ));
        let empty_input = device.upload(&[0, 1, 2, 2], &[]).unwrap();
        let unit_weight = device.upload(&[1, 1, 1], &[1.0]).unwrap();
        let empty_upstream = device.upload(&[0, 1, 2, 2], &[]).unwrap();
        let empty = empty_input
            .depthwise_conv2d_vjp(&unit_weight, &empty_upstream, (1, 1), (0, 0), (1, 1))
            .unwrap();
        assert!(empty[0].snapshot().unwrap().read().unwrap().is_empty());
        assert_eq!(empty[1].snapshot().unwrap().read().unwrap(), [0.0]);
        assert_eq!(empty[2].snapshot().unwrap().read().unwrap(), [0.0]);

        let invalid = device
            .upload(&[1, 1, 2, 2], &[0.0, 0.0, 0.0, f32::MAX])
            .unwrap()
            .mul(&device.upload(&[], &[2.0]).unwrap())
            .unwrap();
        let finite_upstream = device.upload(&[1, 1, 1, 1], &[1.0]).unwrap();
        let inherited = invalid
            .depthwise_conv2d_vjp(&unit_weight, &finite_upstream, (2, 2), (0, 0), (1, 1))
            .unwrap();
        for gradient in inherited {
            assert!(matches!(
                gradient.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
        }

        let overflowing = device
            .upload(&[1, 1, 1, 1], &[f32::MAX])
            .unwrap()
            .depthwise_conv2d_vjp(
                &unit_weight,
                &device.upload(&[1, 1, 1, 1], &[2.0]).unwrap(),
                (1, 1),
                (0, 0),
                (1, 1),
            )
            .unwrap();
        for gradient in overflowing {
            assert!(matches!(
                gradient.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
        }
    }
}
