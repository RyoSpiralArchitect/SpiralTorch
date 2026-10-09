#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[path = "support/byte_decoder.rs"]
mod support;

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_resident_byte_decoder_checks() -> Result<String, wasm_bindgen::JsValue> {
    use st_backend_wgpu::runtime::WgpuRuntime;
    use wasm_bindgen::JsValue;
    let runtime = WgpuRuntime::request_headless("byte_decoder.browser")
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let report = support::run(runtime)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(report.to_string())
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_resident_byte_parameter_rates() -> Result<String, wasm_bindgen::JsValue> {
    use st_backend_wgpu::runtime::WgpuRuntime;
    use wasm_bindgen::JsValue;
    let runtime = WgpuRuntime::request_headless("byte_decoder.parameter_rates.browser")
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let report = support::run_parameter_rates(runtime)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(report.to_string())
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_resident_byte_flat_metric(
    fixture_json: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    use st_backend_wgpu::runtime::WgpuRuntime;
    use wasm_bindgen::JsValue;
    let runtime = WgpuRuntime::request_headless("byte_decoder.flat_metric.browser")
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let report = support::run_flat_metric(runtime, fixture_json)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(report.to_string())
}
