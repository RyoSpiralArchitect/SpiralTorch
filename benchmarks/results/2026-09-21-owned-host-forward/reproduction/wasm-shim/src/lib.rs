//! Measurement adapter for the real Rust CPU Sequential, not a deployed JS API.
use st_nn::{Gelu, Module, Sequential};
use st_tensor::{Layout, Tensor};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct HostChain {
    input: Tensor,
    model: Sequential,
}

#[wasm_bindgen]
pub struct HostResult {
    tensor: Tensor,
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[wasm_bindgen]
impl HostChain {
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize, values: &[f32], depth: usize, layout: u8, nested: bool) -> Result<Self, JsValue> {
        if rows.checked_mul(cols) != Some(values.len()) || depth > 64 {
            return Err(JsValue::from_str("invalid fixture"));
        }
        let layout = match layout {
            0 => Layout::RowMajor,
            1 => Layout::ColMajor,
            2 if cols % 3 == 0 => Layout::Chimera { stripes: 3, tile: (cols / 3) as u32 },
            _ => return Err(JsValue::from_str("invalid layout")),
        };
        let input = Tensor::from_vec(rows, cols, values.to_vec()).and_then(|x| x.to_layout(layout)).map_err(js_error)?;
        let mut model = Sequential::new();
        for _ in 0..depth {
            if nested {
                let mut child = Sequential::new();
                child.push(Gelu::new());
                model.push(child);
            } else {
                model.push(Gelu::new());
            }
        }
        Ok(Self { input, model })
    }

    pub fn run(&self) -> Result<HostResult, JsValue> {
        self.model.forward(&self.input).map(|tensor| HostResult { tensor }).map_err(js_error)
    }

    pub fn backward(&mut self, seed: &[f32]) -> Result<HostResult, JsValue> {
        if seed.len() != self.input.len() {
            return Err(JsValue::from_str("invalid seed"));
        }
        let (rows, cols) = self.input.shape();
        let seed = Tensor::from_vec(rows, cols, seed.to_vec()).map_err(js_error)?;
        self.model.backward(&self.input, &seed).map(|tensor| HostResult { tensor }).map_err(js_error)
    }

    pub fn input_values(&self) -> Result<Vec<f32>, JsValue> {
        Ok(self.input.to_layout(Layout::RowMajor).map_err(js_error)?.data().to_vec())
    }
}

#[wasm_bindgen]
impl HostResult {
    pub fn values(&self) -> Vec<f32> {
        assert_eq!(self.tensor.layout(), Layout::RowMajor);
        self.tensor.data().to_vec()
    }
}
