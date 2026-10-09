#[cfg(target_arch = "wasm32")]
#[path = "support/residual_attention.rs"]
mod support;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_residual_attention_checks() -> Result<String, wasm_bindgen::JsValue> {
    let report = support::run()
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    serde_json::to_string(&report).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}
