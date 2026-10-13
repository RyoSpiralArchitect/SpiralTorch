#[path = "support/querywise_mixture_checks.rs"]
mod checks;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected fixture path")?;
    let input = std::fs::read_to_string(path)?;
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("querywise.native")?;
    println!("{}", pollster::block_on(checks::run(runtime, &input))?);
    Ok(())
}
