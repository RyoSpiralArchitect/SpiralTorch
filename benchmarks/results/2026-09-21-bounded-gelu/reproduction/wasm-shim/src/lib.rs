//! Measurement-only adapter for the checked Rust host operation, not a public API.
use st_tensor::{Layout, Tensor};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct HostGeluFixture {
    input: Tensor,
}

#[wasm_bindgen]
pub struct HostGeluResult {
    output: Tensor,
}

#[wasm_bindgen]
impl HostGeluFixture {
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize, values: &[f32], layout: u8) -> Result<Self, JsValue> {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(JsValue::from_str("invalid fixture shape"));
        }
        let layout = match layout {
            0 => Layout::RowMajor,
            1 => Layout::ColMajor,
            2 if cols % 3 == 0 => Layout::Chimera {
                stripes: 3,
                tile: (cols / 3).try_into().map_err(|_| JsValue::from_str("tile overflow"))?,
            },
            _ => return Err(JsValue::from_str("invalid fixture layout")),
        };
        let input = Tensor::from_vec(rows, cols, values.to_vec())
            .and_then(|input| input.to_layout(layout))
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
        Ok(Self { input })
    }

    pub fn run(&self) -> Result<HostGeluResult, JsValue> {
        self.input.try_gelu()
            .map(|output| HostGeluResult { output })
            .map_err(|err| JsValue::from_str(&err.to_string()))
    }
}

#[wasm_bindgen]
impl HostGeluResult {
    pub fn values(&self) -> Vec<f32> {
        assert_eq!(self.output.layout(), Layout::RowMajor);
        self.output.data().to_vec()
    }
}
