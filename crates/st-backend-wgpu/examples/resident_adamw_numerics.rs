#[path = "support/resident_adamw_numerics.rs"]
mod checks;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use st_backend_wgpu::{resident_tensor::TensorDevice, runtime};
    let (runtime, _) = runtime::ensure_default_runtime_blocking("adamw.numerics")?;
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("numerics study requires an actual GPU".into());
    }
    let mut report = pollster::block_on(checks::run(&TensorDevice::new(runtime.clone())?))?;
    report["adapter"] = serde_json::json!(format!("{:?}", runtime.adapter_info()));
    report["backend"] = serde_json::json!(format!("{:?}", runtime.adapter_info().backend));
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report["status"] != "passed" {
        return Err("same-gradient numerical gate failed; all observations retained".into());
    }
    Ok(())
}
