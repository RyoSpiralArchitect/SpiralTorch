#[cfg(target_arch = "wasm32")]
#[path = "support/resident_parameters_checks.rs"]
mod checks;

#[cfg(target_arch = "wasm32")]
#[path = "support/resident_adamw_checks.rs"]
mod adamw_checks;

#[cfg(target_arch = "wasm32")]
#[path = "support/resident_adamw_numerics.rs"]
mod adamw_numerics;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_adamw_numerics() -> Result<String, wasm_bindgen::JsValue> {
    use st_backend_wgpu::{resident_tensor::TensorDevice, runtime::WgpuRuntime};
    async fn run() -> Result<String, Box<dyn std::error::Error>> {
        let runtime = WgpuRuntime::request_headless("adamw.numerics.browser").await?;
        let mut report = adamw_numerics::run(&TensorDevice::new(runtime.clone())?).await?;
        report["adapter"] = serde_json::json!(format!("{:?}", runtime.adapter_info()));
        report["backend"] = serde_json::json!(format!("{:?}", runtime.adapter_info().backend));
        // WGPU 0.20 exposes no selected browser-adapter hardware identity here.
        report["hardware_execution"] = serde_json::json!("unverified");
        Ok(serde_json::to_string(&report)?)
    }
    run()
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_adamw_checks() -> Result<String, wasm_bindgen::JsValue> {
    use st_backend_wgpu::{resident_tensor::TensorDevice, runtime::WgpuRuntime};
    async fn run() -> Result<String, Box<dyn std::error::Error>> {
        let runtime = WgpuRuntime::request_headless("adamw.browser").await?;
        let device = TensorDevice::new(runtime.clone())?;
        let mut report = adamw_checks::run(&device).await?;
        report["adapter"] = serde_json::Value::String(format!("{:?}", runtime.adapter_info()));
        report["backend"] =
            serde_json::Value::String(format!("{:?}", runtime.adapter_info().backend));
        Ok(serde_json::to_string(&report)?)
    }
    run()
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

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
