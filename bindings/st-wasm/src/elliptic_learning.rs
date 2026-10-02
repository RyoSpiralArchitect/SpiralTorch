#[cfg(target_arch = "wasm32")]
use crate::utils::{js_error, js_u32};
#[cfg(target_arch = "wasm32")]
use js_sys::Number;
#[cfg(target_arch = "wasm32")]
use st_core::theory::microlocal::{EllipticLearningBatch as CoreBatch, EllipticWarp};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct EllipticWarpKernel {
    warp: EllipticWarp,
    max_rows: usize,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct EllipticLearningBatch {
    inner: CoreBatch,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl EllipticWarpKernel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        radius: Number,
        sheets: Number,
        harmonics: Number,
        max_rows: Number,
    ) -> Result<Self, JsValue> {
        let raw: &JsValue = radius.as_ref();
        let radius = raw
            .as_f64()
            .ok_or_else(|| js_error("radius must be a number"))? as f32;
        let sheets = js_u32(sheets.as_ref(), "sheets")? as usize;
        let harmonics = js_u32(harmonics.as_ref(), "harmonics")? as usize;
        let max_rows = js_u32(max_rows.as_ref(), "max_rows")? as usize;
        let warp = EllipticWarp::for_learning(radius, sheets, harmonics).map_err(js_error)?;
        warp.differentiate_batch(&[], max_rows).map_err(js_error)?;
        Ok(Self { warp, max_rows })
    }

    pub fn forward(&self, orientations: &[f32]) -> Result<EllipticLearningBatch, JsValue> {
        self.warp
            .differentiate_batch(orientations, self.max_rows)
            .map(|inner| EllipticLearningBatch { inner })
            .map_err(js_error)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl EllipticLearningBatch {
    #[wasm_bindgen(getter)]
    pub fn features(&self) -> Vec<f32> {
        self.inner.features().to_vec()
    }

    pub fn vjp(&self, upstream: &[f32]) -> Result<Vec<f32>, JsValue> {
        self.inner.vjp(upstream).map_err(js_error)
    }
}
