#[cfg(target_arch = "wasm32")]
#[path = "../tests/support/causal_wave.rs"]
mod support;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_causal_zspace_wave_checks() -> Result<String, wasm_bindgen::JsValue> {
    use st_backend_wgpu::runtime::WgpuRuntime;
    use wasm_bindgen::JsValue;
    let runtime = WgpuRuntime::request_headless("causal_wave.browser")
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let report = support::run(runtime)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(report.to_string())
}
