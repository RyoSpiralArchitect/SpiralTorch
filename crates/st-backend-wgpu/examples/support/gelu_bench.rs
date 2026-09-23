//! Liveness experiment: unused outputs versus unused computation.
use serde_json::{json, Value};
use st_backend_wgpu::{
    gelu_back::{self, plain, Geometry, Plan},
    runtime::{self, ReadbackBatch, Shared, WgpuRuntime},
};
use wgpu::{Buffer, BufferUsages as Usage};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(
    inline_js = "export async function geluCase(raw) { return await globalThis.publishGeluCase(raw); }"
)]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(catch, js_name = geluCase)]
    async fn publish_case(
        raw: String,
    ) -> std::result::Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue>;
}

pub fn routes(count: usize) -> [&'static str; 3] {
    if count == 1 {
        ["legacy_full", "fused_selected", "plain"]
    } else {
        ["legacy_full", "fused_shared", "fused_batch"]
    }
}

pub fn order(block: usize, rows: usize, cols: usize, count: usize) -> [usize; 3] {
    let start = (block + rows + cols + count) % 3;
    let mut order = [start, (start + 1) % 3, (start + 2) % 3];
    if (block / 3) % 2 == 1 {
        order.reverse();
    }
    order
}

struct Kernels {
    fused: gelu_back::Pipelines,
    plain: plain::Pipeline,
    legacy: Shared<wgpu::ComputePipeline>,
}

fn compile_legacy(
    gpu: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: String,
) -> Shared<wgpu::ComputePipeline> {
    let layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gelu.legacy.layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gelu.legacy.shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    Shared::new(
        gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gelu.legacy.pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        }),
    )
}

impl Kernels {
    fn new(runtime: &WgpuRuntime) -> Result<Self> {
        let gpu = runtime.context().device();
        let fused = gelu_back::Pipelines::from_embedded(gpu, Geometry::default())?;
        let source = include_str!("gelu_legacy_tensor.wgsl")
            .replace("{WG_ROWS}", "16")
            .replace("{WG_COLS}", "16")
            .replace("{WG_TILE}", "256");
        let legacy = compile_legacy(gpu, &fused.fused_bind_layout, source);
        Ok(Self {
            fused,
            plain: plain::Pipeline::from_embedded(gpu)?,
            legacy,
        })
    }
}

struct Prepared {
    z: Buffer,
    grad: Buffer,
    gz: Buffer,
    residual: Buffer,
    residual_seed: Buffer,
    partials: Buffer,
    db: Buffer,
    fused_uniform: Buffer,
    reduce_uniform: Buffer,
    plain_uniform: Buffer,
    plan: Plan,
    plain_plan: plain::Plan,
    rows: usize,
    cols: usize,
    stride: usize,
    count: usize,
    values: Vec<f32>,
    seeds: Vec<f32>,
    residuals: Vec<f32>,
}

impl Prepared {
    fn new(
        runtime: &WgpuRuntime,
        rows: usize,
        cols: usize,
        stride: usize,
        count: usize,
        domain: bool,
    ) -> Result<Self> {
        let gpu = runtime.context().device();
        let plan = Plan::new(
            rows as u32,
            cols as u32,
            stride as u32,
            Geometry::default(),
            count == 3,
            &gpu.limits(),
        )?;
        let plain_plan = plain::Plan::new(rows, cols, &gpu.limits())?;
        let domain_values = [
            -f32::MAX,
            -1e20,
            -10.,
            -9.999,
            -5.,
            -1.,
            -0.,
            0.,
            0.8,
            1.,
            5.,
            9.999,
            10.,
            1e20,
            f32::MAX,
        ];
        let values: Vec<_> = (0..rows * cols)
            .map(|i| {
                if domain {
                    domain_values[i % domain_values.len()]
                } else {
                    ((i * 37 + 17) % 1009) as f32 / 128. - 4.
                }
            })
            .collect();
        let seeds: Vec<_> = (0..rows * cols)
            .map(|i| ((i * 13 + 7) % 127) as f32 / 64. - 1.)
            .collect();
        let residuals: Vec<_> = (0..rows * cols)
            .map(|i| (i % 17) as f32 / 32. - 0.25)
            .collect();
        let padded = |v: &[f32]| {
            let mut data = vec![123f32; plan.storage_len()];
            for row in 0..rows {
                data[row * stride..row * stride + cols]
                    .copy_from_slice(&v[row * cols..(row + 1) * cols]);
            }
            data
        };
        let upload = |label, data: &[f32], usage| runtime::upload_slice(gpu, label, data, usage);
        let z = upload("gelu.z", &padded(&values), Usage::STORAGE)?;
        let grad = upload("gelu.seed", &padded(&seeds), Usage::STORAGE)?;
        let gz = upload(
            "gelu.gz",
            &vec![123.; plan.storage_len()],
            Usage::STORAGE | Usage::COPY_SRC,
        )?;
        let residual = upload(
            "gelu.residual",
            &vec![123.; plan.storage_len()],
            Usage::STORAGE | Usage::COPY_SRC | Usage::COPY_DST,
        )?;
        let residual_seed = upload("gelu.residual.seed", &padded(&residuals), Usage::COPY_SRC)?;
        let partials =
            runtime::empty_buffer::<f32>(gpu, "gelu.partials", plan.partial_len(), Usage::STORAGE)?;
        let db =
            runtime::empty_buffer::<f32>(gpu, "gelu.db", cols, Usage::STORAGE | Usage::COPY_SRC)?;
        let fused_uniform = runtime::upload_slice(
            gpu,
            "gelu.uniform",
            &[plan.fused_uniforms()],
            Usage::UNIFORM,
        )?;
        let reduce_uniform = runtime::upload_slice(
            gpu,
            "gelu.reduce.uniform",
            &[plan.reduce_uniforms()],
            Usage::UNIFORM,
        )?;
        let plain_uniform = runtime::upload_slice(
            gpu,
            "gelu.plain.uniform",
            &plain_plan.uniforms(),
            Usage::UNIFORM,
        )?;
        Ok(Self {
            z,
            grad,
            gz,
            residual,
            residual_seed,
            partials,
            db,
            fused_uniform,
            reduce_uniform,
            plain_uniform,
            plan,
            plain_plan,
            rows,
            cols,
            stride,
            count,
            values,
            seeds,
            residuals,
        })
    }

    fn dispatch(&self, runtime: &WgpuRuntime, kernels: &Kernels, route: usize) -> Result<()> {
        let gpu = runtime.context().device();
        let mut encoder = gpu.create_command_encoder(&Default::default());
        if self.count == 1 && route == 2 {
            assert_eq!(self.stride, self.cols);
            let binding = plain::bind(
                gpu,
                &kernels.plain.bind_layout,
                [&self.z, &self.grad, &self.gz, &self.plain_uniform],
            );
            kernels
                .plain
                .encode_into(&mut encoder, &binding, &self.plain_plan);
        } else {
            if self.count == 3 {
                // Reset each operation, including every burst iteration. Both
                // controls and candidates see the same immutable residual seed.
                encoder.copy_buffer_to_buffer(
                    &self.residual_seed,
                    0,
                    &self.residual,
                    0,
                    (self.plan.storage_len() * 4) as u64,
                );
            }
            let fused = gelu_back::fused_bind(
                gpu,
                &kernels.fused.fused_bind_layout,
                [
                    &self.z,
                    &self.grad,
                    &self.gz,
                    &self.residual,
                    &self.partials,
                    &self.fused_uniform,
                ],
            );
            let reduce = gelu_back::reduce_bind(
                gpu,
                &kernels.fused.reduce_bind_layout,
                [&self.partials, &self.db, &self.reduce_uniform],
            );
            if route == 0 || (self.count == 1 && route == 1) {
                for (pipeline, binding, grid) in [
                    (&kernels.legacy, &fused, self.plan.fused_grid()),
                    (&kernels.fused.reduce, &reduce, self.plan.reduce_grid()),
                ] {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, binding, &[]);
                    pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
                }
            } else {
                kernels
                    .fused
                    .encode_into(&mut encoder, &fused, &reduce, &self.plan)?;
            }
        }
        runtime.context().queue().submit(Some(encoder.finish()));
        Ok(())
    }

    async fn observe(&self, runtime: &WgpuRuntime, route: usize) -> Result<Vec<Vec<f32>>> {
        let full = [
            (&self.gz, self.plan.storage_len()),
            (&self.residual, self.plan.storage_len()),
            (&self.db, self.cols),
        ];
        let mut output = if route == 0 || (self.count == 3 && route == 1) {
            let mut output = Vec::new();
            for source in full {
                output.extend(
                    finish(ReadbackBatch::copy(
                        runtime.context(),
                        &[source],
                        "gelu.separate",
                    )?)
                    .await?,
                );
            }
            output
        } else {
            let sources = if self.count == 1 {
                &full[..1]
            } else {
                &full[..]
            };
            finish(ReadbackBatch::copy(
                runtime.context(),
                sources,
                "gelu.selected",
            )?)
            .await?
        };
        if self.count == 1 {
            output.truncate(1);
        }
        Ok(output)
    }

    fn logical(&self, output: &[Vec<f32>]) -> Vec<f32> {
        let mut values = Vec::new();
        for (i, data) in output.iter().enumerate() {
            if i == 2 {
                values.extend(data);
            } else {
                for row in data.chunks(self.stride) {
                    assert!(
                        row[self.cols..].iter().all(|v| *v == 123.),
                        "padding was overwritten"
                    );
                    values.extend(&row[..self.cols]);
                }
            }
        }
        values
    }

    fn reference(&self) -> Vec<f64> {
        let mut output: Vec<_> = self
            .values
            .iter()
            .zip(&self.seeds)
            .map(|(&x, &g)| derivative(f64::from(x)) * f64::from(g))
            .collect();
        if self.count == 3 {
            let db: Vec<f64> = (0..self.cols)
                .map(|c| (0..self.rows).map(|r| output[r * self.cols + c]).sum())
                .collect();
            let residual: Vec<_> = output
                .iter()
                .zip(&self.residuals)
                .map(|(&g, &r)| g + f64::from(r))
                .collect();
            output.extend(residual);
            output.extend(db);
        }
        output
    }

    fn check(&self, output: &[f32], reference: &[f64]) -> Result<(f64, f64)> {
        if output.len() != reference.len() {
            return Err("output length".into());
        }
        let (mut absolute, mut scaled) = (0f64, 0f64);
        for (i, (&a, &b)) in output.iter().zip(reference).enumerate() {
            let scale = if i >= 2 * self.rows * self.cols {
                self.rows as f64
            } else {
                1.
            };
            let error = (f64::from(a) - b).abs();
            let ratio = error / (2e-6 * scale + 1e-5 * b.abs());
            if !a.is_finite() || !b.is_finite() || ratio > 1. {
                return Err(format!("GELU oracle mismatch {i}: {a} != {b}, scaled={ratio}").into());
            }
            absolute = absolute.max(error);
            scaled = scaled.max(ratio);
        }
        Ok((absolute, scaled))
    }
}

pub fn derivative(x: f64) -> f64 {
    if x.abs() >= 10. {
        return if x > 0. { 1. } else { 0. };
    }
    let c = (2. / std::f64::consts::PI).sqrt();
    let t = (c * (x + 0.044715 * x * x * x)).tanh();
    0.5 * (1. + t) + 0.5 * x * (1. - t * t) * c * (1. + 3. * 0.044715 * x * x)
}

async fn finish(batch: ReadbackBatch<f32>) -> Result<Vec<Vec<f32>>> {
    #[cfg(not(target_arch = "wasm32"))]
    let values = batch.read()?;
    #[cfg(target_arch = "wasm32")]
    let values = batch.read_async().await?;
    Ok(values)
}

#[cfg(not(target_arch = "wasm32"))]
fn now() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
        * 1000.
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(
    inline_js = "export function geluNow() { return performance.now(); }"
)]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = geluNow)]
    fn now() -> f64;
}

pub async fn check(runtime: &WgpuRuntime) -> Result<Value> {
    let kernels = Kernels::new(runtime)?;
    let mut cases = Vec::new();
    for (rows, cols) in [(1, 1), (1, 31), (3, 9), (17, 65), (33, 257), (2, 1025)] {
        for count in [1, 3] {
            for padding in [0, 3] {
                let case = Prepared::new(runtime, rows, cols, cols + padding, count, true)?;
                let reference = case.reference();
                let mut errors = Vec::new();
                for route in 0..3 {
                    if padding != 0 && count == 1 && route == 2 {
                        continue;
                    }
                    case.dispatch(runtime, &kernels, route)?;
                    let raw = case.observe(runtime, route).await?;
                    errors.push(case.check(&case.logical(&raw), &reference)?);
                }
                cases.push(json!({"rows":rows,"cols":cols,"count":count,"padding":padding,"errors":errors}));
            }
        }
    }
    // Isolate the old backend's numerical formula after fixing only its syntax.
    // This is not a claim that its original public loader could execute.
    let source = include_str!("gelu_legacy_backend.wgsl").replace("override ", "const ");
    let old = Kernels {
        legacy: compile_legacy(
            runtime.context().device(),
            &kernels.fused.fused_bind_layout,
            source,
        ),
        ..kernels
    };
    let case = Prepared::new(runtime, 1, 15, 15, 1, true)?;
    case.dispatch(runtime, &old, 0)?;
    let raw = case.observe(runtime, 0).await?;
    let output = case.logical(&raw);
    Ok(json!({"cases":cases,"legacy_backend_syntax_repaired_only":{
        "input":case.values, "output_strings":output.iter().map(|v|v.to_string()).collect::<Vec<_>>(),
        "nonfinite_count":output.iter().filter(|v|!v.is_finite()).count()}}))
}

pub async fn run(runtime: &WgpuRuntime) -> Result<Value> {
    let domains = check(runtime).await?;
    let kernels = Kernels::new(runtime)?;
    let mut cases = Vec::new();
    for (rows, cols) in [
        (1, 31),
        (1, 256),
        (17, 257),
        (65, 1025),
        (128, 1025),
        (17, 4096),
    ] {
        for count in [1, 3] {
            let case = Prepared::new(runtime, rows, cols, cols, count, false)?;
            let reference = case.reference();
            let names = routes(count);
            let mut intervals = Vec::new();
            let mut outputs = vec![Vec::new(); 3];
            for block in 0..12 {
                let order = order(block, rows, cols, count);
                for burst in [1, 4] {
                    for route in order {
                        let start = now();
                        for _ in 0..burst {
                            case.dispatch(runtime, &kernels, route)?;
                        }
                        let raw = case.observe(runtime, route).await?;
                        let elapsed_ms = now() - start;
                        let values = case.logical(&raw);
                        let (absolute, scaled) = case.check(&values, &reference)?;
                        if block >= 3 {
                            if elapsed_ms <= 0. {
                                return Err("clock interval not positive".into());
                            }
                            intervals.push(json!({"block":block-3,"burst":burst,"route":names[route],"order":order,
                                "elapsed_ms":elapsed_ms,"max_abs_error":absolute,"max_scaled_error":scaled}));
                        }
                        outputs[route] = values;
                    }
                }
            }
            let report = json!({"rows":rows,"cols":cols,"count":count,"order_scheme":"balanced-cycle-v1",
                "input":case.values,"seed":case.seeds,"residual":case.residuals,"reference":reference,
                "last_outputs":outputs,"intervals":intervals});
            #[cfg(not(target_arch = "wasm32"))]
            cases.push(report);
            #[cfg(target_arch = "wasm32")]
            {
                publish_case(serde_json::to_string(&report)?)
                    .await
                    .map_err(|e| format!("case export: {e:?}"))?;
                cases.push(json!({"rows":rows,"cols":cols,"count":count,"streamed":true,"interval_count":54}));
            }
        }
    }
    Ok(
        json!({"schema":"spiraltorch.gelu_liveness_bench.v1","status":"passed",
        "adapter":format!("{:?}",runtime.adapter_info()),"domains":domains,
        "legacy_tensor_source":include_str!("gelu_legacy_tensor.wgsl"),
        "legacy_backend_source":include_str!("gelu_legacy_backend.wgsl"),
        "blocks":9,"warmup":3,"bursts":[1,4],"cases":cases}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positions_are_balanced() {
        for count in [1, 3] {
            let mut positions = [[0; 3]; 3];
            for block in 3..12 {
                for (p, r) in order(block, 17, 257, count).into_iter().enumerate() {
                    positions[r][p] += 1;
                }
            }
            assert_eq!(positions, [[3; 3]; 3]);
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gelu_domains_on_real_gpu_when_enabled() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let runtime = pollster::block_on(WgpuRuntime::request_headless("gelu.domains")).unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let result = pollster::block_on(check(&runtime)).unwrap();
        assert_eq!(result["cases"].as_array().unwrap().len(), 24);
    }
}
