//! Existing Sequential -> resident forward/backward/SGD, with independent CPU checks.
#[path = "support/resident_training.rs"]
#[cfg(not(target_arch = "wasm32"))]
mod fixture;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> fixture::Result<()> {
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("nn.training.fixture")?;
    let report = futures::executor::block_on(fixture::run(runtime))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
