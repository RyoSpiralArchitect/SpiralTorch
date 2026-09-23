#[cfg(target_arch = "wasm32")]
#[path = "support/gelu_bench.rs"]
mod bench;
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_gelu_backward_bench() -> Result<String, wasm_bindgen::JsValue> {
    let result: bench::Result<String> = async {
        let runtime =
            st_backend_wgpu::runtime::WgpuRuntime::request_headless("gelu.browser").await?;
        Ok(serde_json::to_string(&bench::run(&runtime).await?)?)
    }
    .await;
    result.map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}
