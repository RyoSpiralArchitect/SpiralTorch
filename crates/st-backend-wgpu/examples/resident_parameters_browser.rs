#[cfg(target_arch = "wasm32")]
#[path = "support/resident_parameters_checks.rs"]
mod checks;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_parameter_checks() -> Result<String, wasm_bindgen::JsValue> {
    use st_backend_wgpu::{resident_tensor::TensorDevice, runtime::WgpuRuntime};
    async fn run() -> Result<String, Box<dyn std::error::Error>> {
        let runtime = WgpuRuntime::request_headless("parameters.browser").await?;
        let device = TensorDevice::new(runtime.clone())?;
        let mut report = checks::run(&device).await?;
        report["adapter"] = serde_json::Value::String(format!("{:?}", runtime.adapter_info()));
        Ok(serde_json::to_string(&report)?)
    }
    run()
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
