use wasm_bindgen::prelude::*;
#[wasm_bindgen(start)] pub fn main_js(){ console_error_panic_hook::set_once(); }
#[wasm_bindgen] pub fn info() -> String { "SpiralTorch WASM demo: WebGPU TopK/where (skeleton)".to_string() }
