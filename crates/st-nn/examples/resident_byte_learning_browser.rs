#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[path = "support/byte_learning.rs"]
mod learning;

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_resident_byte_learning(input: &str) -> Result<String, wasm_bindgen::JsValue> {
    use st_backend_wgpu::runtime::WgpuRuntime;
    use wasm_bindgen::JsValue;
    learning::validate_request(input.as_bytes()).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let runtime = WgpuRuntime::request_headless("byte.corpus.browser")
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    learning::run(runtime, input.as_bytes())
        .await
        .map(|v| v.to_string())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
