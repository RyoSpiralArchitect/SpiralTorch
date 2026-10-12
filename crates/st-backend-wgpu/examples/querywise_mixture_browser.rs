#[cfg(target_arch = "wasm32")]
#[path = "support/querywise_mixture_checks.rs"]
mod checks;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_querywise_mixture(input: &str) -> Result<String, wasm_bindgen::JsValue> {
    async fn run(input: &str) -> Result<String, Box<dyn std::error::Error>> {
        let runtime =
            st_backend_wgpu::runtime::WgpuRuntime::request_headless("querywise.browser").await?;
        Ok(serde_json::to_string(&checks::run(runtime, input).await?)?)
    }
    run(input)
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}
