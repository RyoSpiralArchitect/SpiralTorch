//! Deterministic dense-convolution VJPs without floating-point atomics.

use super::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VjpParams {
    geometry: Conv2dParams,
    batch: u32,
    _padding: [u32; 3],
}

#[derive(Debug)]
pub(crate) struct Conv2dVjpKernels {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 3],
}

impl Conv2dVjpKernels {
    fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tensor.conv2d_vjp.layout"),
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
            label: Some("tensor.conv2d_vjp.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor.conv2d_vjp.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/conv2d_vjp.wgsl").into()),
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
        return Err(TensorError::Limit("convolution VJP dispatch grid"));
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
        conv2d_preflight_geometry(input, weights, stride, padding, dilation)?;
    let context = input.device.runtime().context();
    input.require_context(upstream.device.runtime().context())?;
    if upstream.layout.shape() != output_shape {
        return Err(TensorError::ConvolutionShape("cotangent shape mismatch"));
    }
    let gpu = context.device();
    let limits = gpu.limits();
    if limits.max_uniform_buffer_binding_size < std::mem::size_of::<VjpParams>() as u32 {
        return Err(TensorError::Limit("convolution VJP uniform"));
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
    let batch =
        u32::try_from(output_shape[0]).map_err(|_| TensorError::Limit("convolution VJP batch"))?;
    let guard = Shared::new(runtime::empty_buffer::<u32>(
        gpu,
        "tensor.conv2d_vjp.guard",
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
        label: Some("tensor.conv2d_vjp.encoder"),
    });
    let input = input.contiguous_into(&mut encoder)?;
    let weights = weights.contiguous_into(&mut encoder)?;
    let upstream = upstream.contiguous_into(&mut encoder)?;
    let kernels = input
        .device
        .0
        .dense_vjp
        .get_or_init(|| Conv2dVjpKernels::new(gpu));
    let source_values = [input.values(), weights.values(), upstream.values()];
    let source_flags = [input.flags(), weights.flags(), upstream.flags()];
    for (index, output) in outputs.iter().enumerate() {
        let mut target_geometry = geometry;
        target_geometry.output_len = u32::try_from(layouts[index].len())
            .map_err(|_| TensorError::Limit("convolution VJP index"))?;
        target_geometry.groups_x = grids[index][0];
        let params = runtime::upload_slice(
            gpu,
            "tensor.conv2d_vjp.params",
            &[VjpParams {
                geometry: target_geometry,
                batch,
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
            label: Some("tensor.conv2d_vjp.binding"),
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
                label: Some("tensor.conv2d_vjp.pass"),
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
            naga::front::wgsl::parse_str(include_str!("../shaders/conv2d_vjp.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn resident_conv2d_vjp_matches_reference_and_propagates_guards() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let (runtime, _) =
            runtime::ensure_default_runtime_blocking("tensor.conv2d_vjp.test").unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let device = TensorDevice::new(runtime).unwrap();
        let input_values: Vec<f32> = (0..80).map(|index| (index as f32 - 35.0) / 41.0).collect();
        let weight_values: Vec<f32> = (0..36).map(|index| (index as f32 - 17.0) / 53.0).collect();
        let upstream_values: Vec<f32> = (0..84).map(|index| (index as f32 - 39.0) / 37.0).collect();
        let transpose_spatial =
            |values: &[f32], batch: usize, channels: usize, h: usize, w: usize| {
                let mut transposed = vec![0.0; values.len()];
                for n in 0..batch {
                    for c in 0..channels {
                        for y in 0..h {
                            for x in 0..w {
                                transposed[((n * channels + c) * w + x) * h + y] =
                                    values[((n * channels + c) * h + y) * w + x];
                            }
                        }
                    }
                }
                transposed
            };
        let input = device
            .upload(&[2, 2, 5, 4], &transpose_spatial(&input_values, 2, 2, 4, 5))
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap();
        let mut transposed_weights = vec![0.0; weight_values.len()];
        for oc in 0..3 {
            for ic in 0..2 {
                for ky in 0..2 {
                    for kx in 0..3 {
                        transposed_weights[((oc * 2 + ic) * 3 + kx) * 2 + ky] =
                            weight_values[((oc * 2 + ic) * 2 + ky) * 3 + kx];
                    }
                }
            }
        }
        let weights = device
            .upload(&[3, 2, 3, 2], &transposed_weights)
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap();
        let upstream = device
            .upload(
                &[2, 3, 7, 2],
                &transpose_spatial(&upstream_values, 2, 3, 2, 7),
            )
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap();
        assert!(!input.layout().is_contiguous());
        assert!(!weights.layout().is_contiguous());
        assert!(!upstream.layout().is_contiguous());
        let actual = input
            .conv2d_vjp(&weights, &upstream, (2, 1), (1, 2), (2, 1))
            .unwrap();
        assert_eq!(actual[0].layout().shape(), &[2, 2, 4, 5]);
        assert_eq!(actual[1].layout().shape(), &[3, 2, 2, 3]);
        assert_eq!(actual[2].layout().shape(), &[3]);
        let mut expected_input = vec![0.0; input_values.len()];
        let mut expected_weight = vec![0.0; weight_values.len()];
        let mut expected_bias = vec![0.0; 3];
        for n in 0..2 {
            for oc in 0..3 {
                for oy in 0..2 {
                    for ox in 0..7 {
                        let seed = upstream_values[((n * 3 + oc) * 2 + oy) * 7 + ox];
                        expected_bias[oc] += seed;
                        for ic in 0..2 {
                            for ky in 0..2 {
                                for kx in 0..3 {
                                    let y = oy * 2 + ky * 2;
                                    let x = ox + kx;
                                    if y >= 1 && x >= 2 && y - 1 < 4 && x - 2 < 5 {
                                        let input_index = ((n * 2 + ic) * 4 + y - 1) * 5 + x - 2;
                                        let weight_index = ((oc * 2 + ic) * 2 + ky) * 3 + kx;
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
            input.conv2d_vjp(
                &weights,
                &device.upload(&[1, 3, 2, 7], &[0.0; 42]).unwrap(),
                (2, 1),
                (1, 2),
                (2, 1)
            ),
            Err(TensorError::ConvolutionShape("cotangent shape mismatch"))
        ));
        let empty = device.upload(&[0, 1, 2, 2], &[]).unwrap();
        let unit_weight = device.upload(&[1, 1, 1, 1], &[1.0]).unwrap();
        let empty_upstream = device.upload(&[0, 1, 2, 2], &[]).unwrap();
        let empty_grads = empty
            .conv2d_vjp(&unit_weight, &empty_upstream, (1, 1), (0, 0), (1, 1))
            .unwrap();
        assert!(empty_grads[0]
            .snapshot()
            .unwrap()
            .read()
            .unwrap()
            .is_empty());
        assert_eq!(empty_grads[1].snapshot().unwrap().read().unwrap(), [0.0]);
        assert_eq!(empty_grads[2].snapshot().unwrap().read().unwrap(), [0.0]);

        let invalid = device
            .upload(&[1, 1, 1, 1], &[f32::MAX])
            .unwrap()
            .mul(&device.upload(&[], &[2.0]).unwrap())
            .unwrap();
        let seed = device.upload(&[1, 1, 1, 1], &[1.0]).unwrap();
        let poisoned = invalid
            .conv2d_vjp(&unit_weight, &seed, (1, 1), (0, 0), (1, 1))
            .unwrap();
        for gradient in poisoned {
            assert!(matches!(
                gradient.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
        }
        let overflowing = device
            .upload(&[1, 1, 1, 1], &[f32::MAX])
            .unwrap()
            .conv2d_vjp(
                &device.upload(&[1, 1, 1, 1], &[2.0]).unwrap(),
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
