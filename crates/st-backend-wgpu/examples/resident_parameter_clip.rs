#[path = "support/resident_parameter_clip.rs"]
mod checks;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("parameter_clip.native")?.0;
    let info = runtime.adapter_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err("real GPU required".into());
    }
    let mut report = pollster::block_on(checks::run(
        &st_backend_wgpu::resident_tensor::TensorDevice::new(runtime.clone())?,
    ))?;
    report["adapter"] = serde_json::json!(format!("{info:?}"));
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_parameter_clip() -> Result<String, wasm_bindgen::JsValue> {
    async fn run() -> Result<String, Box<dyn std::error::Error>> {
        let runtime =
            st_backend_wgpu::runtime::WgpuRuntime::request_headless("parameter_clip.browser")
                .await?;
        let mut report = checks::run(&st_backend_wgpu::resident_tensor::TensorDevice::new(
            runtime.clone(),
        )?)
        .await?;
        report["backend"] = serde_json::json!(format!("{:?}", runtime.adapter_info().backend));
        report["physical_adapter_verified"] = serde_json::json!(false);
        Ok(serde_json::to_string(&report)?)
    }
    run()
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(target_arch = "wasm32")]
fn main() {}
