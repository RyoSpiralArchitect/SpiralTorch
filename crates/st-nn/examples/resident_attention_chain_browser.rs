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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_attention_chain_fixture(
    fixture_json: String,
) -> Result<String, wasm_bindgen::JsValue> {
    let fixture = serde_json::from_str(&fixture_json)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let result = support::run_fixture(fixture)
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    serde_json::to_string(&result).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_attention_chain_projection(
    projection: String,
    fixture_json: Option<String>,
) -> Result<String, wasm_bindgen::JsValue> {
    let fixture = serde_json::from_str(
        fixture_json
            .as_deref()
            .unwrap_or(include_str!("../tests/fixtures/attention_chain_torch.json")),
    )
    .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let result = support::run_fixture_with_projection(fixture, &projection)
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    serde_json::to_string(&result).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}
