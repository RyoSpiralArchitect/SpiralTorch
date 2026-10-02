#[cfg(target_arch = "wasm32")]
#[path = "support/convolution_profile_checks.rs"]
mod checks;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_convolution_profile_checks() -> Result<String, wasm_bindgen::JsValue> {
    checks::run()
        .await
        .map(|result| result.to_string())
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
