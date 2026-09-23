#[path = "support/gelu_bench.rs"]
mod bench;
fn main() -> bench::Result<()> {
    let runtime = pollster::block_on(st_backend_wgpu::runtime::WgpuRuntime::request_headless(
        "gelu.bench",
    ))?;
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("CPU adapter rejected".into());
    }
    let report = pollster::block_on(bench::run(&runtime))?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
