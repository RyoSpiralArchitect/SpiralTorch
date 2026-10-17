#[cfg(target_arch = "wasm32")]
#[path = "../tests/support/wide_rounding.rs"]
mod support;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_wide_rounding_checks() -> Result<String, wasm_bindgen::JsValue> {
    use st_backend_wgpu::runtime::WgpuRuntime;
    let runtime = WgpuRuntime::request_headless("wide.rounding.browser")
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))?;
    support::run(runtime)
        .await
        .map(|report| report.to_string())
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
