//! Existing Module execution versus mixed resident forward, with explicit observation boundaries.
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use serde_json::json;
    use st_backend_wgpu::{
        resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
        runtime,
    };
    use st_core::backend::device_caps::DeviceCaps;
    use st_nn::{
        layers::{Gelu, Relu, Scaler},
        module::Module,
        push_backend_policy,
        resident::InferencePlan,
        AcceleratorFallback, BackendPolicy, ExecutionConfig, Linear, Sequential,
    };
    use st_tensor::{NdLayout, Tensor};
    use std::time::Instant;

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const WARMUP: usize = 3;
    const SAMPLES: usize = 9;
    const BURST: usize = 8;

    fn close(actual: &[f32], expected: &[f32]) -> Result<f32> {
        if actual.len() != expected.len() {
            return Err("output shape differs".into());
        }
        let mut maximum = 0f32;
        for (&a, &b) in actual.iter().zip(expected) {
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > 2e-5 + 2e-4 * b.abs() {
                return Err(format!("forward mismatch: {a} != {b}").into());
            }
            maximum = maximum.max((a - b).abs());
        }
        Ok(maximum)
    }

    fn run(
        shape: &[usize],
        depth: usize,
        seed: usize,
        runtime: &runtime::WgpuRuntime,
    ) -> Result<serde_json::Value> {
        let width = *shape.last().ok_or("missing feature axis")?;
        let layout = NdLayout::contiguous(shape)?;
        let mut model = Sequential::new();
        for i in 0..depth {
            model.push(Scaler::new(format!("scale_{i}"), width)?);
            model.push(Linear::new(format!("linear_{i}"), width, width)?);
            model.push(Gelu::new());
            model.push(Relu::new());
        }
        let mut index = seed;
        model.visit_parameters_mut(&mut |parameter| {
            let weight = parameter.name().ends_with("::weight");
            let gain = parameter.name().ends_with("::gain");
            for (i, value) in parameter.value_mut().data_mut().iter_mut().enumerate() {
                let jitter = ((index * 17 % 23) as f32 - 11.) / 1024.;
                *value = if gain {
                    1. + jitter
                } else if weight {
                    jitter + if i / width == i % width { 1. } else { 0. }
                } else {
                    jitter
                };
                index += 1;
            }
            Ok(())
        })?;
        let input = Tensor::from_fn(layout.len() / width, width, |r, c| {
            ((r * width + c + seed) % 29) as f32 / 16. - 0.5
        })?;
        let reference = {
            let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
            model.forward(&input)?
        };
        let plan = InferencePlan::from_module(&model, layout)?;
        let compile_start = Instant::now();
        let mut graphs = [
            plan.compile_graph_wgpu(runtime.clone())?,
            plan.compile_graph_wgpu_with_options(
                runtime.clone(),
                MatmulTile::default(),
                MatmulKernel::Register2x2,
                MatmulAccumulation::Compensated,
            )?,
        ];
        let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.;
        let legacy_policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::wgpu(32, false, 256),
            ExecutionConfig::new(AcceleratorFallback::Forbid, 0),
        );
        let mut samples = Vec::new();
        let mut last_outputs = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for block in 0..WARMUP + SAMPLES {
            let order: Vec<_> = (0..5).map(|i| (i + block + seed) % 5).collect();
            for &route in &order {
                let (output, elapsed_ms, forwards) = if route == 0 {
                    let _policy = push_backend_policy(legacy_policy);
                    let start = Instant::now();
                    let output = model.forward(&input)?;
                    let elapsed = start.elapsed().as_secs_f64() * 1000.;
                    (output.data().to_vec(), elapsed, 1)
                } else {
                    let graph = &mut graphs[(route - 1) % 2];
                    let burst = route >= 3;
                    if burst {
                        graph.upload(input.data())?;
                        graph.dispatch()?;
                        graph.snapshot()?.read()?;
                    }
                    let before = graph.submitted_dispatches();
                    let forwards = if burst { BURST } else { 1 };
                    let start = Instant::now();
                    if !burst {
                        graph.upload(input.data())?;
                    }
                    for _ in 0..forwards {
                        graph.dispatch()?;
                    }
                    let output = graph.snapshot()?.read()?;
                    let elapsed = start.elapsed().as_secs_f64() * 1000.;
                    if graph.submitted_dispatches() - before != forwards as u64 {
                        return Err("dispatch count changed".into());
                    }
                    (output, elapsed, forwards)
                };
                let maximum = close(&output, reference.data())?;
                if !elapsed_ms.is_finite() || elapsed_ms <= 0. {
                    return Err("invalid timing".into());
                }
                last_outputs[route] = output;
                if block >= WARMUP {
                    samples.push(json!({"block":block - WARMUP,"route":route,"order":order,
                        "elapsed_ms":elapsed_ms,"forwards":forwards,"max_abs_error":maximum}));
                }
            }
        }
        Ok(
            json!({"shape":shape,"depth":depth,"seed":seed,"input":input.data(),
            "reference":reference.data(),"plan":serde_json::from_str::<serde_json::Value>(&plan.to_json()?)?,
            "source_operations":plan.source_operation_count(),"gpu_stages":plan.stage_count(),
            "compile_two_graphs_ms":compile_ms,"samples":samples,"last_outputs":last_outputs}),
        )
    }

    pub fn main() -> Result<()> {
        let (runtime, _) = runtime::ensure_default_runtime_blocking("nn.graph.forward.bench")?;
        let adapter = runtime.adapter_info();
        if format!("{:?}", adapter.device_type) == "Cpu" {
            return Err("real GPU required".into());
        }
        let mut report = json!({"schema":"spiraltorch.graph_forward_bench.v1","status":"error",
            "client":"native","warmup":WARMUP,"samples_per_route":SAMPLES,"burst":BURST,
            "routes":["legacy_h2h","scalar_h2h","register_h2h","scalar_burst","register_burst"],
            "adapter":{"name":adapter.name,"backend":format!("{:?}",adapter.backend),
                "device_type":format!("{:?}",adapter.device_type)},"cases":[],
            "boundary":"H2H: fixed host input to owning host f32 output. Burst: eight forwards of the same resident input, final owning readback included. Setup/compilation excluded. Controls rotated; numerical checks outside timing. Legacy Module backend policy forbids accelerator fallback; host-backed operations and existing caches retained."});
        let result: Result<()> = (|| {
            for seed in [17, 29, 43] {
                for (shape, depth) in [
                    (vec![2, 3, 7], 2),
                    (vec![2, 8, 64], 8),
                    (vec![4, 8, 128], 16),
                ] {
                    let case = run(&shape, depth, seed, &runtime)?;
                    eprintln!("completed shape={shape:?} depth={depth} seed={seed}");
                    report["cases"].as_array_mut().unwrap().push(case);
                }
            }
            Ok(())
        })();
        match &result {
            Ok(()) => report["status"] = json!("passed"),
            Err(error) => report["error"] = json!(error.to_string()),
        }
        println!("{report}");
        result
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::main()
}
#[cfg(target_arch = "wasm32")]
fn main() {}
