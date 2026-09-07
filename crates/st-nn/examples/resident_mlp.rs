//! Matched existing-Module versus resident inference, including input/output transfers.
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use serde::Deserialize;
    use serde_json::json;
    use st_backend_wgpu::runtime;
    use st_core::backend::device_caps::DeviceCaps;
    use st_nn::{
        layers::Gelu, module::Module, push_backend_policy, resident::InferencePlan,
        AcceleratorFallback, BackendPolicy, ExecutionConfig, Linear, Sequential,
    };
    use st_tensor::{NdLayout, Tensor};
    use std::{
        io::{self, BufRead},
        time::Instant,
    };

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        shape: Vec<usize>,
        depth: usize,
        seed: u32,
    }

    fn close(actual: &[f32], expected: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
        if actual.len() != expected.len() {
            return Err("output length differs".into());
        }
        let mut error = 0f32;
        for (&a, &b) in actual.iter().zip(expected) {
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-5 + 1e-4 * b.abs() {
                return Err(format!("output differs: {a} versus {b}").into());
            }
            error = error.max((a - b).abs());
        }
        Ok(error)
    }

    fn run(
        request: Request,
        runtime: &runtime::WgpuRuntime,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let layout = NdLayout::contiguous(&request.shape)?;
        let width = *request.shape.last().ok_or("input needs a feature axis")?;
        if width == 0
            || width > 256
            || layout.is_empty()
            || layout.len() > 65536
            || request.depth == 0
            || request.depth > 32
            || request.seed == 0
        {
            return Err("fixture exceeds bounded dimensions/depth/seed".into());
        }
        let mut model = Sequential::new();
        for i in 0..request.depth {
            model.push(Linear::new(format!("linear_{i}"), width, width)?);
            if i + 1 < request.depth {
                model.push(Gelu::new());
            }
        }
        let mut state = request.seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state % 65) as f32 - 32.0) / 256.0
        };
        model.visit_parameters_mut(&mut |parameter| {
            for value in parameter.value_mut().data_mut() {
                *value = next();
            }
            Ok(())
        })?;
        let input = Tensor::from_fn(layout.len() / width, width, |_, _| next())?;
        let reference = {
            let _guard = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
            model.forward(&input)?
        };
        let plan = InferencePlan::from_module(&model, layout)?;
        let mut resident = plan.compile_wgpu(runtime.clone())?;
        let legacy_policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::wgpu(32, false, 256),
            ExecutionConfig::new(AcceleratorFallback::Forbid, 0),
        );
        let mut legacy = Vec::new();
        let mut resident_times = Vec::new();
        let mut samples = Vec::new();
        let mut max_error = 0f32;
        for block in 0..14 {
            let order = if (block + request.seed).is_multiple_of(2) {
                [0, 1]
            } else {
                [1, 0]
            };
            let mut pair = [0f64; 2];
            for mode in order {
                if mode == 0 {
                    let _guard = push_backend_policy(legacy_policy);
                    let start = Instant::now();
                    let output = model.forward(&input)?;
                    pair[0] = start.elapsed().as_secs_f64() * 1000.0;
                    max_error = max_error.max(close(output.data(), reference.data())?);
                } else {
                    let start = Instant::now();
                    resident.upload(input.data())?;
                    resident.dispatch()?;
                    let output = resident.snapshot()?.read()?;
                    pair[1] = start.elapsed().as_secs_f64() * 1000.0;
                    max_error = max_error.max(close(&output, reference.data())?);
                }
            }
            if block >= 2 {
                legacy.push(pair[0]);
                resident_times.push(pair[1]);
                samples.push(json!({"order":order, "legacy_ms":pair[0], "resident_ms":pair[1]}));
            }
        }
        let parameters: Vec<_> = plan.parameter_snapshots().enumerate().map(|(i, (weight, bias))|
            json!({"weight":weight.data(), "bias":bias.data(), "gelu":i + 1 < request.depth})).collect();
        Ok(json!({
            "status":"passed", "shape":request.shape, "depth":request.depth, "seed":request.seed,
            "source_operations":plan.source_operation_count(), "gpu_stages":plan.stage_count(),
            "adapter":{"name":runtime.adapter_info().name, "backend":format!("{:?}",runtime.adapter_info().backend),
                "device_type":format!("{:?}",runtime.adapter_info().device_type)},
            "input":input.data(), "parameters":parameters, "reference":reference.data(),
            "max_abs_error":max_error, "samples":samples,
            "legacy_mean_ms":legacy.iter().sum::<f64>() / legacy.len() as f64,
            "resident_mean_ms":resident_times.iter().sum::<f64>() / resident_times.len() as f64,
            "boundary":"host input to owned host output; fixed model parameters, existing Module prepack/cache behavior retained; Module WGPU matmul + CPU GELU versus checked resident chain with device-persistent weights/bias; setup excluded; 2 warmups and 12 paired retained samples; checks outside timing"
        }))
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args == ["--build-info"] {
            println!(
                "{}",
                json!({"schema":"spiraltorch.native_build_identity.v1",
                "build_fingerprint":st_core::build_fingerprint(),
                "manifest":serde_json::from_str::<serde_json::Value>(st_core::build_manifest_json())?})
            );
            return Ok(());
        }
        if !args.is_empty() {
            return Err("usage: resident_mlp [--build-info]".into());
        }
        let (runtime, _) = runtime::ensure_default_runtime_blocking("nn.resident.mlp")?;
        if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
            return Err("resident diagnostic requires a GPU".into());
        }
        for line in io::stdin().lock().lines() {
            let result = run(serde_json::from_str(&line?)?, &runtime)?;
            println!("{}", result);
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::main()
}
#[cfg(target_arch = "wasm32")]
fn main() {}
