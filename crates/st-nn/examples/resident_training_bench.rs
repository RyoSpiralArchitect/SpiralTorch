//! Persistent native benchmark worker, controlled by one request/response at a time.
#[cfg(not(target_arch = "wasm32"))]
#[path = "support/resident_training_bench.rs"]
mod fixture;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> fixture::Result<()> {
    use serde::Deserialize;
    use serde_json::json;
    use std::{
        io::{self, BufRead, Write},
        sync::OnceLock,
        time::Instant,
    };
    static CLOCK: OnceLock<Instant> = OnceLock::new();
    fn now() -> f64 {
        CLOCK.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.
    }
    #[derive(Deserialize)]
    #[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
    enum Request {
        Init {
            config: fixture::Config,
        },
        Sample {
            cadence: fixture::Cadence,
            capture: bool,
        },
        Profile {
            policy: String,
        },
        Learn {
            cadence: fixture::Cadence,
            capture: bool,
        },
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
        return Err("usage: resident_training_bench [--build-info]".into());
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("nn.training.bench")?;
    if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
        return Err("GPU required".into());
    }
    let mut benchmark = None;
    for line in io::stdin().lock().lines() {
        let result = match serde_json::from_str::<Request>(&line?)? {
            Request::Init { config } => {
                let value = fixture::Benchmark::new(runtime.clone(), config)?;
                let output = value.fixture()?;
                benchmark = Some(value);
                output
            }
            Request::Sample { cadence, capture } => futures::executor::block_on(
                benchmark
                    .as_ref()
                    .ok_or("initialize a graph first")?
                    .sample(cadence, capture, now),
            )?,
            Request::Profile { policy } => futures::executor::block_on(
                benchmark
                    .as_ref()
                    .ok_or("initialize a graph first")?
                    .profile(policy.parse()?),
            )?,
            Request::Learn { cadence, capture } => futures::executor::block_on(
                benchmark
                    .as_ref()
                    .ok_or("initialize a graph first")?
                    .learn(cadence, capture, now),
            )?,
        };
        println!("{result}");
        io::stdout().flush()?;
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
