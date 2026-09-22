#[cfg(target_arch = "wasm32")]
#[path = "support/consensus_bench.rs"]
mod fixture;

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::fixture;
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen(inline_js = "export function consensusNow() { return performance.now(); }")]
    extern "C" {
        #[wasm_bindgen(js_name=consensusNow)]
        fn now() -> f64;
    }
    #[wasm_bindgen]
    pub async fn run_consensus_readback_bench() -> Result<String, JsValue> {
        async fn run() -> fixture::Result<String> {
            let runtime =
                st_backend_wgpu::runtime::WgpuRuntime::request_headless("consensus.browser")
                    .await?;
            Ok(fixture::run(runtime, now).await?.to_string())
        }
        run().await.map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
