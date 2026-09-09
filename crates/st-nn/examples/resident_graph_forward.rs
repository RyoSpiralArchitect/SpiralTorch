#[cfg(not(target_arch = "wasm32"))]
#[path = "support/resident_graph_forward.rs"]
mod fixture;
#[cfg(not(target_arch = "wasm32"))]
fn main() -> fixture::Result<()> {
    if std::env::args().len() != 1 {
        return Err("usage: resident_graph_forward".into());
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("graph.forward.fixture")?;
    let report = futures::executor::block_on(fixture::run(runtime))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
#[cfg(target_arch = "wasm32")]
fn main() {}
