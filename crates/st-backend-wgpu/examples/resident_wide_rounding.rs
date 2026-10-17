#[path = "../tests/support/wide_rounding.rs"]
mod support;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (runtime, _) = st_backend_wgpu::runtime::ensure_default_runtime_blocking("wide.rounding")?;
    println!("{}", pollster::block_on(support::run(runtime))?);
    Ok(())
}
