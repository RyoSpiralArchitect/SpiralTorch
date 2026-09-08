//! Source-identified native worker for the N-D resident elementwise benchmark.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use serde::Deserialize;
    use serde_json::json;
    use st_tensor::{NdTensor, WgpuTensorDevice};
    use std::{
        io::{self, BufRead, Write},
        time::Instant,
    };
    #[derive(Clone, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Config {
        shape: [usize; 3],
        seed: usize,
        iterations: usize,
    }
    #[derive(Deserialize)]
    #[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
    enum Request {
        Init { config: Config },
        Sample { capture: bool },
    }
    let identity = json!({"schema":"spiraltorch.native_build_identity.v1",
        "build_fingerprint":st_core::build_fingerprint(),
        "manifest":serde_json::from_str::<serde_json::Value>(st_core::build_manifest_json())?});
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--build-info"] {
        println!("{identity}");
        return Ok(());
    }
    if !args.is_empty() {
        return Err("usage: resident_nd_bench [--build-info]".into());
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("tensor.nd.bench")?;
    if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
        return Err("GPU required".into());
    }
    let device = WgpuTensorDevice::new(runtime)?;
    let mut prepared = None;
    for line in io::stdin().lock().lines() {
        let response = match serde_json::from_str::<Request>(&line?)? {
            Request::Init { config } => {
                let len = config
                    .shape
                    .iter()
                    .try_fold(1usize, |a, &b| a.checked_mul(b))
                    .ok_or("shape overflow")?;
                if config.shape[0] == 0
                    || config.shape[1] < 2
                    || config.shape[2] == 0
                    || len > 1_048_576
                    || config.iterations == 0
                    || config.iterations > 100
                    || config.seed > 1_000_000
                {
                    return Err("benchmark recipe out of bounds".into());
                }
                let input: Vec<_> = (0..len)
                    .map(|i| ((i * 13 + config.seed) % 61) as f32 / 64. - 0.46875)
                    .collect();
                let bias: Vec<_> = (0..config.shape[2])
                    .map(|i| (i % 5) as f32 / 32. - 0.0625)
                    .collect();
                let root = NdTensor::from_vec(&config.shape, input.clone())?.to_wgpu(&device)?;
                let b = NdTensor::from_vec(&[config.shape[2]], bias.clone())?.to_wgpu(&device)?;
                let gain = NdTensor::from_vec(&[], vec![0.75])?.to_wgpu(&device)?;
                // Finish uploads and shader warmup before the first interval.
                root.read_values()?;
                let out = json!({"schema":"spiraltorch.nd_bench.fixture.v1","shape":config.shape,"seed":config.seed,"iterations":config.iterations,
                    "input":input,"bias":bias,"gain":0.75,"adapter":{
                        "name":device.runtime().adapter_info().name,
                        "device_type":format!("{:?}",device.runtime().adapter_info().device_type),
                        "backend":format!("{:?}",device.runtime().adapter_info().backend)},"identity":identity});
                prepared = Some((config, root, b, gain));
                out
            }
            Request::Sample { capture } => {
                let (config, root, b, gain) = prepared.as_ref().ok_or("init first")?;
                let start = Instant::now();
                let mut out = root
                    .permute(&[1, 0, 2])?
                    .narrow(0, 1, config.shape[1] - 1)?;
                for _ in 0..config.iterations {
                    out = out.add(b)?.mul(gain)?.gelu()?;
                }
                let values = out.read_values()?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                json!({"elapsed_ms":elapsed_ms,"shape":out.shape(),"finite_checked":true,
                    "values":if capture {Some(values)} else {None}})
            }
        };
        println!("{response}");
        io::stdout().flush()?;
    }
    Ok(())
}
#[cfg(target_arch = "wasm32")]
fn main() {}
