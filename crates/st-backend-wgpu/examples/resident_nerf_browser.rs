#[cfg(target_arch = "wasm32")]
#[path = "support/nerf.rs"]
mod fixture;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_resident_nerf_fixture() -> Result<String, wasm_bindgen::JsValue> {
    async fn run() -> fixture::Result<String> {
        let runtime =
            st_backend_wgpu::runtime::WgpuRuntime::request_headless("nerf.fixture.browser").await?;
        Ok(fixture::run(runtime).await?.to_string())
    }
    run()
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
