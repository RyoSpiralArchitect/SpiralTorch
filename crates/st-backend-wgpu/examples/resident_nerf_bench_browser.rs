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
    #[wasm_bindgen]
    pub async fn run_resident_nerf_bench(compare_submissions: bool) -> Result<String, JsValue> {
        async fn run(compare_submissions: bool) -> fixture::Result<String> {
            let runtime =
                st_backend_wgpu::runtime::WgpuRuntime::request_headless("nerf.bench.browser")
                    .await?;
            Ok(fixture::run(runtime, now, compare_submissions)
                .await?
                .to_string())
        }
        run(compare_submissions)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
