use super::*;
use crate::runtime::{self, WgpuRuntime};

#[test]
fn finite_domain_softmax_and_peak_masks_preserve_strides_and_padding() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let runtime = pollster::block_on(WgpuRuntime::request_headless("softmax.domain")).unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let context = runtime.context();
    let gpu = context.device();
    let (file_pipelines, _) = Builder::new(
        gpu,
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders"),
    )
    .supports_subgroup(true)
    .build()
    .unwrap();
    for pipelines in [&file_pipelines, &Pipelines::from_embedded(gpu, true)] {
        let variants = if pipelines.subgroup.is_some() {
            vec![PipelineVariant::Workgroup, PipelineVariant::Subgroup]
        } else {
            vec![PipelineVariant::Workgroup]
        };
        eprintln!(
            "softmax adapter={:?}, variants={variants:?}",
            runtime.adapter_info()
        );
        for cols in [1usize, 3, 31, 256, 257, 1025] {
            for chimera in [false, true] {
                let rows = 4;
                let tile = 16;
                let stripes = cols.div_ceil(tile);
                let stride = if chimera { tile * stripes } else { cols } + 7;
                let index = |row, col| {
                    row * stride
                        + if chimera {
                            col % tile * stripes + col / tile
                        } else {
                            col
                        }
                };
                let mut values = vec![123.0f32; rows * stride];
                let logical: Vec<f32> = (0..rows * cols)
                    .map(|i| match i / cols {
                        0 => -f32::MAX,
                        1 => -2e30,
                        2 => {
                            if i % cols == cols - 1 {
                                -2e30
                            } else {
                                -3e30
                            }
                        }
                        _ => (i % cols % 7) as f32 / 8. - 0.5,
                    })
                    .collect();
                for row in 0..rows {
                    for col in 0..cols {
                        values[index(row, col)] = logical[row * cols + col];
                    }
                }
                let input =
                    runtime::upload_slice(gpu, "softmax.input", &values, BufferUsages::STORAGE)
                        .unwrap();
                for variant in &variants {
                    for mode in [0, 2, 4, 6] {
                        let output = runtime::upload_slice(
                            gpu,
                            "softmax.output",
                            &vec![123.0f32; values.len()],
                            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                        )
                        .unwrap();
                        let mask = runtime::upload_slice(
                            gpu,
                            "softmax.mask",
                            &vec![123.0f32; values.len()],
                            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                        )
                        .unwrap();
                        let params = upload_params(
                            gpu,
                            context.queue(),
                            &Params {
                                rows: rows as u32,
                                cols: cols as u32,
                                in_stride: stride as u32,
                                out_stride: stride as u32,
                                chimera_tile: tile as u32,
                                chimera_stripes: stripes as u32,
                                flags: mode | u32::from(chimera),
                                mask_stride: stride as u32,
                            },
                        );
                        dispatch(
                            gpu,
                            context.queue(),
                            pipelines,
                            &DispatchArgs {
                                values: &input,
                                output: &output,
                                params: &params,
                                mask: Some(&mask),
                            },
                            Dispatch { rows: rows as u32 },
                            *variant,
                        );
                        let actual: Vec<f32> = runtime::read_buffer(
                            gpu,
                            context.queue(),
                            &output,
                            values.len(),
                            "softmax.result",
                        )
                        .unwrap();
                        let actual_mask: Vec<f32> = runtime::read_buffer(
                            gpu,
                            context.queue(),
                            &mask,
                            values.len(),
                            "softmax.peaks",
                        )
                        .unwrap();
                        let mut expected = vec![123.; values.len()];
                        let mut peaks = vec![123.; values.len()];
                        for row in 0..rows {
                            let x = &logical[row * cols..(row + 1) * cols];
                            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f64 = x
                                .iter()
                                .map(|&v| (f64::from(v) - f64::from(max)).exp())
                                .sum();
                            for col in 0..cols {
                                let peak = f32::from(x[col] == max);
                                expected[index(row, col)] = if mode & 2 != 0 {
                                    peak
                                } else {
                                    ((f64::from(x[col]) - f64::from(max)).exp() / sum) as f32
                                };
                                if mode & 4 != 0 {
                                    peaks[index(row, col)] = peak;
                                }
                            }
                        }
                        for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
                            assert!(got.is_finite() && (got-want).abs() <= 2e-7 + 2e-6 * want.abs(),
                                "{variant:?} cols={cols} chimera={chimera} mode={mode} i={i}: {got} != {want}");
                        }
                        assert_eq!(actual_mask, peaks);
                    }
                }
            }
        }
    }
}

#[test]
fn row_subgroup_shader_aliases_reduce_independent_rows() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let runtime = pollster::block_on(WgpuRuntime::request_headless("softmax.row.aliases")).unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let context = runtime.context();
    let gpu = context.device();
    if !device_supports_subgroup(gpu) {
        eprintln!("row subgroup aliases not exercised: device feature unavailable");
        return;
    }
    for source in [
        include_str!("../shaders/row_softmax_subgroup.wgsl"),
        include_str!("../shaders/softmax_row_subgroup.wgsl"),
    ] {
        let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax.alias"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax.alias"),
            layout: None,
            module: &module,
            entry_point: "main_cs",
            compilation_options: Default::default(),
        });
        for (rows, cols) in [(1usize, 1usize), (3, 31), (259, 257)] {
            let stride = cols + 7;
            let mut values = vec![123.0f32; rows * stride];
            let mut expected = values.clone();
            for row in 0..rows {
                let x = &mut values[row * stride..row * stride + cols];
                for (col, v) in x.iter_mut().enumerate() {
                    *v = if row == 0 {
                        -f32::MAX
                    } else {
                        ((col * 7 + row * 17) % 23) as f32 / 8.
                    };
                }
                let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f64 = x
                    .iter()
                    .map(|&v| (f64::from(v) - f64::from(max)).exp())
                    .sum();
                for col in 0..cols {
                    expected[row * stride + col] =
                        ((f64::from(x[col]) - f64::from(max)).exp() / sum) as f32;
                }
            }
            let input =
                runtime::upload_slice(gpu, "alias.input", &values, BufferUsages::STORAGE).unwrap();
            let output = runtime::upload_slice(
                gpu,
                "alias.output",
                &vec![123.0f32; values.len()],
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap();
            let params = runtime::upload_slice(
                gpu,
                "alias.params",
                &[rows as u32, cols as u32, stride as u32, stride as u32],
                BufferUsages::UNIFORM,
            )
            .unwrap();
            let binding = gpu.create_bind_group(&BindGroupDescriptor {
                label: Some("alias.binding"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = gpu.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &binding, &[]);
                pass.dispatch_workgroups(rows as u32, 1, 1);
            }
            context.queue().submit(Some(encoder.finish()));
            let got: Vec<f32> =
                runtime::read_buffer(gpu, context.queue(), &output, values.len(), "alias.result")
                    .unwrap();
            for (i, (&actual, &want)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    actual.is_finite() && (actual - want).abs() <= 2e-7 + 2e-6 * want.abs(),
                    "rows={rows} cols={cols} i={i}: {actual} != {want}"
                );
            }
        }
    }
}
