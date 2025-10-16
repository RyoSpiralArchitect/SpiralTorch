use std::fmt;
use wasm_bindgen::JsValue;

pub(crate) fn js_error(err: impl fmt::Display) -> JsValue {
    JsValue::from_str(&err.to_string())
}
