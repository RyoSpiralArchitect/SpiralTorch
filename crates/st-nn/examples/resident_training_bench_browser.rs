//! Browser timing of the same Rust workload; no JavaScript optimizer semantics.
#[cfg(target_arch = "wasm32")]
#[path = "support/resident_training_bench.rs"]
mod fixture;

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::fixture;
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen(inline_js = "export function trainingBenchNow() { return performance.now(); }")]
    extern "C" {
        #[wasm_bindgen(js_name=trainingBenchNow)]
        fn now() -> f64;
    }

    #[wasm_bindgen]
    pub struct TrainingBenchmark {
        inner: fixture::Benchmark,
    }
    #[wasm_bindgen]
    impl TrainingBenchmark {
        pub async fn create(config: String) -> Result<TrainingBenchmark, JsValue> {
            async fn build(config: String) -> fixture::Result<fixture::Benchmark> {
                let runtime = st_backend_wgpu::runtime::WgpuRuntime::request_headless(
                    "nn.training.bench.browser",
                )
                .await?;
                fixture::Benchmark::new(runtime, serde_json::from_str(&config)?)
            }
            build(config)
                .await
                .map(|inner| Self { inner })
                .map_err(|e| JsValue::from_str(&e.to_string()))
        }
        pub fn fixture(&self) -> Result<String, JsValue> {
            self.inner
                .fixture()
                .map(|v| v.to_string())
                .map_err(|e| JsValue::from_str(&e.to_string()))
        }
        pub async fn profile(&self, policy: String) -> Result<String, JsValue> {
            let policy = policy.parse().map_err(JsValue::from_str)?;
            self.inner
                .profile(policy)
                .await
                .map(|v| v.to_string())
                .map_err(|e| JsValue::from_str(&e.to_string()))
        }
        pub async fn sample(&self, cadence: String, capture: bool) -> Result<String, JsValue> {
            let cadence =
                serde_json::from_str(&cadence).map_err(|e| JsValue::from_str(&e.to_string()))?;
            self.inner
                .sample(cadence, capture, now)
                .await
                .map(|v| v.to_string())
                .map_err(|e| JsValue::from_str(&e.to_string()))
        }
        pub async fn learn(&self, cadence: String, capture: bool) -> Result<String, JsValue> {
            let cadence =
                serde_json::from_str(&cadence).map_err(|e| JsValue::from_str(&e.to_string()))?;
            self.inner
                .learn(cadence, capture, now)
                .await
                .map(|v| v.to_string())
                .map_err(|e| JsValue::from_str(&e.to_string()))
        }
        #[wasm_bindgen(js_name = learnHostProfile)]
        pub async fn learn_host_profile(
            &self,
            cadence: String,
            capture: bool,
        ) -> Result<String, JsValue> {
            let cadence =
                serde_json::from_str(&cadence).map_err(|e| JsValue::from_str(&e.to_string()))?;
            self.inner
                .learn_host_profile(cadence, capture, now)
                .await
                .map(|v| v.to_string())
                .map_err(|e| JsValue::from_str(&e.to_string()))
        }
    }
}
