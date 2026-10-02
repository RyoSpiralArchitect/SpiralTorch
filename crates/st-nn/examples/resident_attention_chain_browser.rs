#[cfg(target_arch = "wasm32")]
#[path = "support/attention_chain.rs"]
mod support;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_attention_chain_checks() -> Result<String, wasm_bindgen::JsValue> {
    let result = support::run()
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    serde_json::to_string(&result).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}
