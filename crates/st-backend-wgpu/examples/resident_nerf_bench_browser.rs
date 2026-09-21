#[cfg(target_arch = "wasm32")]
#[path = "support/nerf_bench.rs"]
mod fixture;

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::fixture;
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen(inline_js = "export function nerfBenchNow() { return performance.now(); }")]
    extern "C" {
        #[wasm_bindgen(js_name=nerfBenchNow)]
        fn now() -> f64;
    }
    async fn run(comparison: fixture::Comparison) -> fixture::Result<String> {
        let runtime =
            st_backend_wgpu::runtime::WgpuRuntime::request_headless("nerf.bench.browser").await?;
        Ok(fixture::run(runtime, now, comparison).await?.to_string())
    }
    #[wasm_bindgen]
    pub async fn run_resident_nerf_bench(compare_submissions: bool) -> Result<String, JsValue> {
        run(if compare_submissions {
            fixture::Comparison::Submissions
        } else {
            fixture::Comparison::StagedDirect
        })
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub async fn run_resident_nerf_row_input_bench() -> Result<String, JsValue> {
        run(fixture::Comparison::InputRows)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
