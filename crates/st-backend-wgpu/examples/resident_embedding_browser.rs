#[cfg(target_arch = "wasm32")]
#[path = "../tests/support/embedding.rs"]
mod support;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_resident_embedding_checks() -> Result<String, wasm_bindgen::JsValue> {
    use st_backend_wgpu::runtime::WgpuRuntime;
    use wasm_bindgen::JsValue;
    let runtime = WgpuRuntime::request_headless("embedding.browser")
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let report = support::run(runtime)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(report.to_string())
}
