//! Canonical portable softmax, with a same-source redundant-barrier control.
use serde_json::{json, Value};
use st_backend_wgpu::{
    runtime::{self, Shared, WgpuRuntime},
    shader_sources::SOFTMAX_WORKGROUP_WGSL,
    softmax::{self, Dispatch, DispatchArgs, Params, PipelineVariant, Pipelines},
};
use wgpu::{Buffer, BufferUsages as Usage};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Prepared {
    input: Buffer,
    output: Buffer,
    mask: Buffer,
    uniform: Buffer,
    params: Params,
    len: usize,
}

impl Prepared {
    fn new(runtime: &WgpuRuntime, data: &[f32], params: Params) -> Result<Self> {
        let gpu = runtime.context().device();
        let len = params.rows as usize * params.out_stride as usize;
        if data.len() != params.rows as usize * params.in_stride as usize
            || data.iter().any(|x| !x.is_finite())
            || len == 0
        {
            return Err("invalid softmax fixture".into());
        }
        Ok(Self {
            input: runtime::upload_slice(gpu, "softmax.input", data, Usage::STORAGE)?,
            output: runtime::upload_slice(
                gpu,
                "softmax.output",
                &vec![123.0f32; len],
                Usage::STORAGE | Usage::COPY_SRC,
            )?,
            mask: runtime::upload_slice(
                gpu,
                "softmax.mask",
                &vec![123.0f32; len],
                Usage::STORAGE | Usage::COPY_SRC,
            )?,
            uniform: softmax::upload_params(gpu, runtime.context().queue(), &params),
            params,
            len,
        })
    }

    fn dispatch(&self, runtime: &WgpuRuntime, pipelines: &Pipelines, variant: PipelineVariant) {
        assert!(softmax::dispatch(
            runtime.context().device(),
            runtime.context().queue(),
            pipelines,
            &DispatchArgs {
                values: &self.input,
                output: &self.output,
                params: &self.uniform,
                mask: Some(&self.mask)
            },
            Dispatch {
                rows: self.params.rows
            },
            variant
        ));
    }

    async fn read(&self, runtime: &WgpuRuntime) -> Result<Vec<f32>> {
        let gpu = runtime.context().device();
        let bytes = (self.len * 4) as u64;
        let masked = self.params.flags & 4 != 0;
        let size = bytes * if masked { 2 } else { 1 };
        let staging = gpu.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax.owning_read"),
            size,
            usage: Usage::MAP_READ | Usage::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.output, 0, &staging, 0, bytes);
        if masked {
            encoder.copy_buffer_to_buffer(&self.mask, 0, &staging, bytes, bytes);
        }
        runtime.context().queue().submit(Some(encoder.finish()));
        #[cfg(not(target_arch = "wasm32"))]
        let raw = runtime::map_read_bytes_with_timeout(
            gpu,
            &staging,
            0..size,
            std::time::Duration::from_secs(30),
            "softmax.read",
        )?;
        #[cfg(target_arch = "wasm32")]
        let raw = {
            let (sender, receiver) = futures_channel::oneshot::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
            receiver.await??;
            let bytes = staging.slice(..).get_mapped_range().to_vec();
            staging.unmap();
            bytes
        };
        Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| bytemuck::pod_read_unaligned::<f32>(bytes))
            .collect())
    }
}

fn params(rows: usize, cols: usize, flags: u32, chimera: bool) -> Params {
    let tile = 16;
    let stripes = cols.div_ceil(tile);
    let stride = if chimera { tile * stripes + 7 } else { cols };
    Params {
        rows: rows as u32,
        cols: cols as u32,
        in_stride: stride as u32,
        out_stride: stride as u32,
        chimera_tile: tile as u32,
        chimera_stripes: stripes as u32,
        flags: flags | u32::from(chimera),
        mask_stride: stride as u32,
    }
}

fn index(params: Params, row: usize, col: usize) -> usize {
    row * params.in_stride as usize
        + if params.flags & 1 != 0 {
            col % params.chimera_tile as usize * params.chimera_stripes as usize
                + col / params.chimera_tile as usize
        } else {
            col
        }
}

fn oracle(data: &[f32], params: Params) -> Vec<f32> {
    let len = params.rows as usize * params.out_stride as usize;
    let mut output = vec![123.0f32; len * if params.flags & 4 != 0 { 2 } else { 1 }];
    for row in 0..params.rows as usize {
        let x: Vec<f64> = (0..params.cols as usize)
            .map(|col| f64::from(data[index(params, row, col)]))
            .collect();
        let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = x.iter().map(|v| (v - max).exp()).sum();
        for (col, value) in x.iter().enumerate() {
            let at = index(params, row, col);
            let peak = if *value == max { 1. } else { 0. };
            output[at] = if params.flags & 2 != 0 {
                peak
            } else {
                ((value - max).exp() / sum) as f32
            };
            if params.flags & 4 != 0 {
                output[len + at] = peak;
            }
        }
    }
    output
}

fn close(actual: &[f32], expected: &[f32]) -> Result<f64> {
    if actual.len() != expected.len() {
        return Err("softmax output length".into());
    }
    let mut maximum = 0.0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        let error = (f64::from(a) - f64::from(b)).abs();
        if !a.is_finite() || !b.is_finite() || error > 2e-7 + 2e-6 * f64::from(b).abs() {
            return Err(format!("softmax mismatch {a} versus {b}").into());
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}

fn redundant_control(runtime: &WgpuRuntime, candidate: &Pipelines) -> Result<Pipelines> {
    let mut source = SOFTMAX_WORKGROUP_WGSL.to_owned();
    for step in [
        "reduce_max(tid, WORKGROUP_SIZE);",
        "reduce_sum(tid, WORKGROUP_SIZE);",
    ] {
        if source.matches(step).count() != 1 {
            return Err("control specialization point".into());
        }
        source = source.replace(step, &format!("workgroupBarrier();\n    {step}"));
    }
    let gpu = runtime.context().device();
    let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("softmax.redundant_control"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("softmax.redundant_control"),
        bind_group_layouts: &[&candidate.bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("softmax.redundant_control"),
        layout: Some(&layout),
        module: &module,
        entry_point: "main_cs",
        compilation_options: Default::default(),
    });
    Ok(Pipelines {
        bind_layout: pipeline.get_bind_group_layout(0),
        workgroup: Shared::new(pipeline),
        subgroup: None,
    })
}

async fn domains(runtime: &WgpuRuntime, pipelines: &Pipelines) -> Result<usize> {
    let mut cases = 0;
    for cols in [1, 3, 31, 256, 257, 1025] {
        for chimera in [false, true] {
            for mode in [0, 2, 4, 6] {
                let params = params(4, cols, mode, chimera);
                let mut data = vec![123.0f32; params.rows as usize * params.in_stride as usize];
                for row in 0..4 {
                    for col in 0..cols {
                        data[index(params, row, col)] = match row {
                            0 => -f32::MAX,
                            1 => -2e30,
                            2 => {
                                if col + 1 == cols {
                                    -2e30
                                } else {
                                    -3e30
                                }
                            }
                            _ => (col % 7) as f32 / 8. - 0.5,
                        };
                    }
                }
                let prepared = Prepared::new(runtime, &data, params)?;
                let expected = oracle(&data, params);
                for variant in [PipelineVariant::Workgroup, PipelineVariant::Subgroup] {
                    if variant == PipelineVariant::Subgroup && pipelines.subgroup.is_none() {
                        continue;
                    }
                    prepared.dispatch(runtime, pipelines, variant);
                    close(&prepared.read(runtime).await?, &expected)?;
                }
                cases += 1;
            }
        }
    }
    Ok(cases)
}

pub async fn run(runtime: WgpuRuntime, now: fn() -> f64) -> Result<Value> {
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("real GPU required".into());
    }
    let candidate = Pipelines::from_embedded(runtime.context().device(), true);
    let domain_cases = domains(&runtime, &candidate).await?;
    let subgroup_exercised = candidate.subgroup.is_some();
    let pipelines = [redundant_control(&runtime, &candidate)?, candidate];
    let names = ["redundant", "deduplicated"];
    let mut cases = Vec::new();
    for (rows, cols) in [
        (1, 31),
        (1, 256),
        (17, 257),
        (65, 1025),
        (128, 1025),
        (17, 4096),
    ] {
        for mode in [0, 4] {
            let params = params(rows, cols, mode, false);
            let data: Vec<f32> = (0..rows * cols)
                .map(|i| ((i * 37 + 17) % 1009) as f32 / 128. - 504. / 128.)
                .collect();
            let expected = oracle(&data, params);
            let prepared = [
                Prepared::new(&runtime, &data, params)?,
                Prepared::new(&runtime, &data, params)?,
            ];
            for route in 0..2 {
                prepared[route].dispatch(&runtime, &pipelines[route], PipelineVariant::Workgroup);
                close(&prepared[route].read(&runtime).await?, &expected)?;
            }
            let mut intervals = Vec::new();
            let mut outputs = [Vec::new(), Vec::new()];
            for burst in [1, 4] {
                for block in 0..12 {
                    let order = if (block + rows + cols + mode as usize).is_multiple_of(2) {
                        [0, 1]
                    } else {
                        [1, 0]
                    };
                    for route in order {
                        let start = now();
                        for _ in 0..burst {
                            prepared[route].dispatch(
                                &runtime,
                                &pipelines[route],
                                PipelineVariant::Workgroup,
                            );
                        }
                        let actual = prepared[route].read(&runtime).await?;
                        let elapsed_ms = now() - start;
                        let error = close(&actual, &expected)?;
                        if !elapsed_ms.is_finite() || elapsed_ms <= 0. {
                            return Err("invalid softmax interval".into());
                        }
                        outputs[route] = actual;
                        if block >= 3 {
                            intervals
                                .push(json!({"block":block-3,"burst":burst,"route":names[route],
                                "order":order,"elapsed_ms":elapsed_ms,"max_abs_error":error}));
                        }
                    }
                }
            }
            cases.push(
                json!({"rows":rows,"cols":cols,"mode":mode,"input":data,"reference":expected,
                "last_outputs":outputs,"intervals":intervals}),
            );
        }
    }
    Ok(
        json!({"schema":"spiraltorch.softmax_portable_bench.v1","status":"passed",
        "comparison":"redundant_vs_deduplicated_barriers","finite_domain_fix_in_both":true,
        "adapter":format!("{:?}",runtime.adapter_info()),"domain_cases":domain_cases,"subgroup_exercised":subgroup_exercised,
        "warmup":3,"blocks":9,"bursts":[1,4],"cases":cases,
        "boundary":"Prepared f32 input and output buffers, canonical portable workgroup shader. Both routes use the corrected finite maximum; control restores only two redundant entry barriers. Encoding/binding creation and submission per softmax plus one owning terminal CPU copy/map/completion per 1/4-operation interval included. Compilation, uploads, validation and serialization excluded. Subgroups checked outside timing only when enabled. No isolated GPU timestamps, training, ResidentTensor/autograd migration, or universal speed claim."}),
    )
}
