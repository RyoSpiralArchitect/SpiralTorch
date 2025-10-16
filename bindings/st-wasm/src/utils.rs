use js_sys::JSON;
use std::fmt;
use wasm_bindgen::JsValue;

pub(crate) fn js_error(err: impl fmt::Display) -> JsValue {
    JsValue::from_str(&err.to_string())
}

pub(crate) fn js_value_to_string(value: &JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

pub(crate) fn stringify_js_value(value: &JsValue) -> Result<String, JsValue> {
    let json = JSON::stringify(value).map_err(|err| js_error(js_value_to_string(&err)))?;
    json.as_string()
        .ok_or_else(|| js_error("expected JSON string"))
}

pub(crate) fn json_to_js_value(json: &str) -> Result<JsValue, JsValue> {
    JSON::parse(json).map_err(|err| js_error(js_value_to_string(&err)))
}
