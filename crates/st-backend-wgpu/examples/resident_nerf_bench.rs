#[cfg(not(target_arch = "wasm32"))]
#[path = "support/nerf_bench.rs"]
mod fixture;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> fixture::Result<()> {
    let mut args = std::env::args().skip(1);
    let comparison = match (args.next().as_deref(), args.next()) {
        (None, None) => fixture::Comparison::StagedDirect,
        (Some("--compare-submissions"), None) => fixture::Comparison::Submissions,
        (Some("--compare-input-layouts"), None) => fixture::Comparison::InputRows,
        _ => {
            return Err(
                "usage: resident_nerf_bench [--compare-submissions|--compare-input-layouts]".into(),
            )
        }
    };
    fn now() -> f64 {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1000.
    }
    let runtime = pollster::block_on(st_backend_wgpu::runtime::WgpuRuntime::request_headless(
        "nerf.bench.native",
    ))?;
    println!(
        "{}",
        pollster::block_on(fixture::run(runtime, now, comparison))?
    );
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
