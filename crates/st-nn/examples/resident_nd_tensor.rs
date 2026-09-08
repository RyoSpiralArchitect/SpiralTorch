#[path = "support/resident_nd_tensor.rs"]
mod fixture;
fn main() -> fixture::Result<()> {
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("tensor.nd.native")?;
    println!("{}", futures::executor::block_on(fixture::run(runtime))?);
    Ok(())
}
