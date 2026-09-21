#[cfg(not(target_arch = "wasm32"))]
#[path = "support/nerf.rs"]
mod fixture;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> fixture::Result<()> {
    let runtime = pollster::block_on(st_backend_wgpu::runtime::WgpuRuntime::request_headless(
        "nerf.fixture.native",
    ))?;
    println!("{}", pollster::block_on(fixture::run(runtime))?);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
