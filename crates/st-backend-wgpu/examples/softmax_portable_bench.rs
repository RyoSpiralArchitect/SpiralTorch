#[cfg(not(target_arch = "wasm32"))]
#[path = "support/softmax_bench.rs"]
mod fixture;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> fixture::Result<()> {
    fn now() -> f64 {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1000.
    }
    let runtime = pollster::block_on(st_backend_wgpu::runtime::WgpuRuntime::request_headless(
        "softmax.native",
    ))?;
    println!("{}", pollster::block_on(fixture::run(runtime, now))?);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
