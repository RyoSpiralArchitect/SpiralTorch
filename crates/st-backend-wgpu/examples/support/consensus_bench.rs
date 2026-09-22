//! Matched 2x2 reduction/readback experiment; pair mode is a shader placebo.
#[path = "readback_cases.rs"]
mod readback_cases;
use serde_json::{json, Value};
use st_backend_wgpu::{
    runtime::{self, ReadbackBatch, Shared, WgpuRuntime},
    shader_sources::SOFTMAX_SPIRAL_CONSENSUS_WGSL,
    softmax::{self, consensus, Dispatch, DispatchArgs, PipelineVariant, Pipelines},
};
use wgpu::{Buffer, BufferUsages as Usage};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[cfg(not(target_arch = "wasm32"))]
fn checkpoint(_stage: &str) {}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(
    inline_js = "export function consensusCheckpoint(s) { globalThis.publishConsensusProgress?.(s); } export async function consensusCase(raw) { return await globalThis.publishConsensusCase(raw); }"
)]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = consensusCheckpoint)]
    fn checkpoint(stage: &str);
    #[wasm_bindgen::prelude::wasm_bindgen(catch, js_name = consensusCase)]
    async fn publish_case(
        raw: String,
    ) -> std::result::Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue>;
}
const NAMES: [&str; 4] = [
    "separate_separate",
    "paired_separate",
    "separate_batch",
    "paired_batch",
];

struct Prepared {
    input: Buffer,
    outputs: Vec<Buffer>,
    uniform: Buffer,
    consensus_uniform: Buffer,
    params: softmax::Params,
    consensus: consensus::Params,
    count: usize,
}

impl Prepared {
    fn new(
        runtime: &WgpuRuntime,
        data: &[f32],
        rows: usize,
        cols: usize,
        count: usize,
        chimera: bool,
    ) -> Result<Self> {
        let gpu = runtime.context().device();
        let tile = 16;
        let stripes = cols.div_ceil(tile);
        let stride = if chimera { tile * stripes + 7 } else { cols };
        let params = softmax::Params {
            rows: rows as u32,
            cols: cols as u32,
            in_stride: stride as u32,
            out_stride: stride as u32,
            mask_stride: stride as u32,
            chimera_tile: tile as u32,
            chimera_stripes: stripes as u32,
            flags: 4 | u32::from(chimera),
        };
        if data.len() != rows * stride
            || data.iter().any(|v| !v.is_finite())
            || ![2, 4].contains(&count)
        {
            return Err("invalid consensus fixture".into());
        }
        let consensus = consensus::Params {
            rows: params.rows,
            cols: params.cols,
            soft_stride: params.out_stride,
            mask_stride: params.out_stride,
            spiral_stride: params.out_stride,
            chimera_tile: params.chimera_tile,
            chimera_stripes: params.chimera_stripes,
            flags: u32::from(chimera),
            phi: 1.618_034,
            phi_conjugate: 0.618_034,
            phi_bias: 0.381_966,
            leech_scale: (0.75 * 0.001_929_574_309_403_922_5 * 24f64.sqrt()) as f32,
            ramanujan_ratio: 1.,
            inv_cols: 1. / cols as f32,
            entropy_epsilon: 1e-7,
            _pad: 0.,
        };
        let outputs = [rows * stride, rows * stride, rows * stride, rows * 4]
            .into_iter()
            .map(|len| {
                runtime::upload_slice(
                    gpu,
                    "consensus.output",
                    &vec![123f32; len],
                    Usage::STORAGE | Usage::COPY_SRC,
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            input: runtime::upload_slice(gpu, "consensus.input", data, Usage::STORAGE)?,
            outputs,
            uniform: softmax::upload_params(gpu, runtime.context().queue(), &params),
            consensus_uniform: runtime::upload_slice(
                gpu,
                "consensus.params",
                &[consensus],
                Usage::UNIFORM,
            )?,
            params,
            consensus,
            count,
        })
    }

    fn dispatch(&self, runtime: &WgpuRuntime, softmax: &Pipelines, pipeline: &consensus::Pipeline) {
        let gpu = runtime.context().device();
        let mut encoder = gpu.create_command_encoder(&Default::default());
        assert!(softmax::encode_into(
            gpu,
            &mut encoder,
            softmax,
            &DispatchArgs {
                values: &self.input,
                output: &self.outputs[0],
                params: &self.uniform,
                mask: Some(&self.outputs[1]),
            },
            Dispatch {
                rows: self.params.rows
            },
            PipelineVariant::Workgroup
        ));
        if self.count == 4 {
            let binding = consensus::bind(
                gpu,
                &pipeline.bind_layout,
                &self.outputs[0],
                &self.outputs[1],
                &self.outputs[2],
                &self.outputs[3],
                &self.consensus_uniform,
            );
            assert!(pipeline.encode_into(&mut encoder, &binding, self.params.rows));
        }
        runtime.context().queue().submit(Some(encoder.finish()));
    }

    async fn read(&self, runtime: &WgpuRuntime, batch: bool) -> Result<Vec<f32>> {
        let len = self.params.rows as usize * self.params.out_stride as usize;
        let sources: Vec<_> = self.outputs[..self.count]
            .iter()
            .enumerate()
            .map(|(i, b)| {
                (
                    b,
                    if i == 3 {
                        self.params.rows as usize * 4
                    } else {
                        len
                    },
                )
            })
            .collect();
        let mut output = Vec::new();
        let group_size = if batch { self.count } else { 1 };
        for group in sources.chunks(group_size) {
            // Both controls use identical snapshot ownership/decoding, differing
            // only in grouping. Native original read_buffer is checked separately.
            let snapshot = ReadbackBatch::<f32>::copy(runtime.context(), group, "consensus.read")?;
            assert_eq!(snapshot.staging_buffer_count(), 1);
            #[cfg(not(target_arch = "wasm32"))]
            let values = snapshot.read()?;
            #[cfg(target_arch = "wasm32")]
            let values = snapshot.read_async().await?;
            output.extend(values.into_iter().flatten());
        }
        Ok(output)
    }

    fn index(&self, row: usize, col: usize) -> usize {
        row * self.params.in_stride as usize
            + if self.params.flags & 1 != 0 {
                col % self.params.chimera_tile as usize * self.params.chimera_stripes as usize
                    + col / self.params.chimera_tile as usize
            } else {
                col
            }
    }

    fn oracle(&self, data: &[f32]) -> Vec<f32> {
        let rows = self.params.rows as usize;
        let cols = self.params.cols as usize;
        let len = rows * self.params.out_stride as usize;
        let mut output = vec![
            123f32;
            if self.count == 2 {
                2 * len
            } else {
                3 * len + rows * 4
            }
        ];
        let p = self.consensus;
        for row in 0..rows {
            let logits: Vec<_> = (0..cols)
                .map(|c| f64::from(data[self.index(row, c)]))
                .collect();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = logits.iter().map(|v| (v - max).exp()).sum();
            let probs: Vec<_> = logits.iter().map(|v| (v - max).exp() / sum).collect();
            let mask: Vec<_> = logits
                .iter()
                .map(|v| if *v == max { 1f64 } else { 0. })
                .collect();
            let entropy: f64 = probs
                .iter()
                .map(|v| -v * v.max(f64::from(p.entropy_epsilon)).ln())
                .sum();
            let mass: f64 = mask.iter().sum();
            let geodesic = entropy * f64::from(p.ramanujan_ratio) + mass * f64::from(p.phi);
            let enrichment = if geodesic.abs() > f64::from(p.entropy_epsilon) {
                f64::from(p.leech_scale) * geodesic
            } else {
                0.
            };
            let coherence = ((entropy / (entropy + 1.)).clamp(0., 1.)
                + (mass * f64::from(p.inv_cols)).clamp(0., 1.)
                + (enrichment / (1. + enrichment.abs())).clamp(0., 1.))
                / 3.;
            for col in 0..cols {
                let at = self.index(row, col);
                output[at] = probs[col] as f32;
                output[len + at] = mask[col] as f32;
                if self.count == 4 {
                    output[2 * len + at] = ((1. + enrichment)
                        * (f64::from(p.phi_conjugate) * probs[col]
                            + f64::from(p.phi_bias) * mask[col]))
                        as f32;
                }
            }
            if self.count == 4 {
                output[3 * len + row * 4..3 * len + row * 4 + 4].copy_from_slice(&[
                    entropy as f32,
                    mass as f32,
                    enrichment as f32,
                    coherence as f32,
                ]);
            }
        }
        output
    }
}

fn close(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
    if actual.len() != expected.len() {
        return Err("consensus output length".into());
    }
    let (mut maximum, mut scaled) = (0f64, 0f64);
    for (&a, &b) in actual.iter().zip(expected) {
        let error = (f64::from(a) - f64::from(b)).abs();
        let ratio = error / (2e-6 + 5e-6 * f64::from(b).abs());
        if !a.is_finite() || !b.is_finite() || ratio > 1. {
            return Err(format!("consensus mismatch {a} vs {b}").into());
        }
        maximum = maximum.max(error);
        scaled = scaled.max(ratio);
    }
    Ok((maximum, scaled))
}

fn separate_control(runtime: &WgpuRuntime) -> Result<(consensus::Pipeline, String)> {
    let source = SOFTMAX_SPIRAL_CONSENSUS_WGSL;
    let start = source
        .find("fn reduce_sum_statistics(")
        .ok_or("missing reduction")?;
    let end = source[start..].find("\n@compute").ok_or("missing entry")? + start;
    let mut functions = String::new();
    for statistic in ["entropy", "hardmass"] {
        functions.push_str(&format!("fn reduce_sum_{statistic}(local_id: u32) {{\n    var stride = WORKGROUP_SIZE / 2u;\n    loop {{\n        if (stride == 0u) {{\n            break;\n        }}\n        workgroupBarrier();\n        if (local_id < stride) {{\n            shared_{statistic}[local_id] =\n                shared_{statistic}[local_id] + shared_{statistic}[local_id + stride];\n        }}\n        stride = stride / 2u;\n    }}\n    workgroupBarrier();\n}}\n\n"));
    }
    let mut restored = source.to_owned();
    restored.replace_range(start..end, functions.trim_end_matches('\n'));
    let call = "// Both statistics retain their addition tree and share each required barrier.\n    reduce_sum_statistics(tid);";
    if restored.matches(call).count() != 1 {
        return Err("consensus control specialization".into());
    }
    restored = restored.replace(
        call,
        "workgroupBarrier();\n\n    reduce_sum_entropy(tid);\n    reduce_sum_hardmass(tid);",
    );
    let gpu = runtime.context().device();
    let binding = consensus::bind_layout(gpu);
    let layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("consensus.separate.control"),
        bind_group_layouts: &[&binding],
        push_constant_ranges: &[],
    });
    let shader = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("consensus.separate.control"),
        source: wgpu::ShaderSource::Wgsl(restored.clone().into()),
    });
    let compute = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("consensus.separate.control"),
        layout: Some(&layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
    });
    Ok((
        consensus::Pipeline {
            bind_layout: binding,
            compute: Shared::new(compute),
        },
        restored,
    ))
}

pub async fn domains(runtime: &WgpuRuntime) -> Result<usize> {
    let softmax = Pipelines::from_embedded(runtime.context().device(), false);
    let pipelines = [
        separate_control(runtime)?.0,
        consensus::Pipeline::from_embedded(runtime.context().device()),
    ];
    let mut cases = 0;
    for cols in [1usize, 3, 31, 256, 257, 1025] {
        for chimera in [false, true] {
            let stride = if chimera {
                cols.div_ceil(16) * 16 + 7
            } else {
                cols
            };
            let mut data = vec![123f32; 4 * stride];
            for row in 0..4 {
                for col in 0..cols {
                    let at = row * stride
                        + if chimera {
                            col % 16 * cols.div_ceil(16) + col / 16
                        } else {
                            col
                        };
                    data[at] = match row {
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
            let prepared = Prepared::new(runtime, &data, 4, cols, 4, chimera)?;
            let expected = prepared.oracle(&data);
            let mut original = None;
            for route in 0..4 {
                prepared.dispatch(runtime, &softmax, &pipelines[route % 2]);
                let output = prepared.read(runtime, route >= 2).await?;
                close(&output, &expected)?;
                let bits: Vec<_> = output.iter().map(|v| v.to_bits()).collect();
                if let Some(old) = &original {
                    assert_eq!(&bits, old);
                } else {
                    original = Some(bits);
                }
            }
            cases += 1;
        }
    }
    Ok(cases)
}

pub async fn run(runtime: WgpuRuntime, now: fn() -> f64) -> Result<Value> {
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("real GPU required".into());
    }
    let ownership = readback_cases::run(&runtime, checkpoint).await?;
    checkpoint("ownership-passed");
    let domain_cases = domains(&runtime).await?;
    checkpoint("domains-passed");
    let softmax = Pipelines::from_embedded(runtime.context().device(), false);
    let (control, control_source) = separate_control(&runtime)?;
    let pipelines = [
        control,
        consensus::Pipeline::from_embedded(runtime.context().device()),
    ];
    let mut cases = Vec::new();
    for (rows, cols) in [
        (1, 31),
        (1, 256),
        (17, 257),
        (65, 1025),
        (128, 1025),
        (17, 4096),
    ] {
        for count in [2, 4] {
            checkpoint(&format!("case-{rows}-{cols}-{count}"));
            let data: Vec<_> = (0..rows * cols)
                .map(|i| ((i * 37 + 17) % 1009) as f32 / 128. - 504. / 128.)
                .collect();
            let prepared = (0..4)
                .map(|_| Prepared::new(&runtime, &data, rows, cols, count, false))
                .collect::<Result<Vec<_>>>()?;
            let reference = prepared[0].oracle(&data);
            let mut intervals = Vec::new();
            let mut outputs = vec![Vec::new(); 4];
            for burst in [1, 4] {
                for block in 0..12 {
                    let mut order = [0, 1, 2, 3];
                    order.rotate_left((block + rows + cols + count) % 4);
                    if (block / 4) % 2 == 1 {
                        order.reverse();
                    }
                    for route in order {
                        let start = now();
                        for _ in 0..burst {
                            prepared[route].dispatch(&runtime, &softmax, &pipelines[route % 2]);
                        }
                        let output = prepared[route].read(&runtime, route >= 2).await?;
                        let elapsed_ms = now() - start;
                        let (max_abs_error, max_scaled_error) = close(&output, &reference)?;
                        if !elapsed_ms.is_finite() || elapsed_ms <= 0. {
                            return Err("invalid consensus interval".into());
                        }
                        outputs[route] = output;
                        if block >= 3 {
                            intervals.push(json!({"block":block-3,"burst":burst,"route":NAMES[route],
                            "order":order,"elapsed_ms":elapsed_ms,"max_abs_error":max_abs_error,"max_scaled_error":max_scaled_error}));
                        }
                    }
                }
            }
            let case = json!({"rows":rows,"cols":cols,"count":count,"order_scheme":"balanced-cycle-v1","input":data,"reference":reference,
                             "last_outputs":outputs,"intervals":intervals});
            #[cfg(not(target_arch = "wasm32"))]
            cases.push(case);
            #[cfg(target_arch = "wasm32")]
            {
                // Persist one complete case outside timing instead of retaining
                // and rendering a >100 MB JSON document in the browser.
                publish_case(case.to_string())
                    .await
                    .map_err(|e| format!("case publication failed: {e:?}"))?;
                cases.push(json!({"rows":rows,"cols":cols,"count":count,"streamed":true,"interval_count":72}));
            }
        }
    }
    let mut report = json!({"schema":"spiraltorch.consensus_readback_bench.v1","status":"passed",
        "adapter":format!("{:?}",runtime.adapter_info()),"ownership":ownership,"domain_cases":domain_cases,
        "domain_bitwise_equal":true,"control_shader":control_source,
        "warmup":3,"blocks":9,"bursts":[1,4],"routes":NAMES,
        "boundary":"Portable workgroup softmax/all-peak mask, plus raw GPU consensus for count=4. Same-source separate/paired reductions crossed with separate/batched owning readback. Count=2 is a reduction placebo and matches the ordinary Tensor pair output boundary, not whole API overhead. Prepared input/output; encoding, binding creation, per-operation submission and terminal owning CPU completion per burst included. Native and browser share Rust snapshot decoding. No Tensor CPU consensus blend, autograd, training or universal performance claim."});
    report["cases"] = Value::Array(cases);
    Ok(report)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    #[test]
    fn consensus_domain_and_batch_ownership_when_enabled() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let runtime = pollster::block_on(WgpuRuntime::request_headless("consensus.test")).unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        pollster::block_on(readback_cases::run(&runtime, checkpoint)).unwrap();
        assert_eq!(pollster::block_on(domains(&runtime)).unwrap(), 12);
    }
}
