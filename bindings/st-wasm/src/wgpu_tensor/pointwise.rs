use super::*;
use st_backend_wgpu::resident_tensor::pointwise::{PointwiseInputs, PointwisePlan};
use st_tensor::{PointwiseChain, PointwiseStep};

#[wasm_bindgen(js_name = WgpuPointwiseInputs)]
#[derive(Default)]
pub struct WasmPointwiseInputs {
    inner: PointwiseInputs,
}
#[wasm_bindgen(js_class = WgpuPointwiseInputs)]
impl WasmPointwiseInputs {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }
    pub fn add(&mut self, tensor: &WasmWgpuTensor) -> Result<(), JsValue> {
        self.inner.add(&tensor.inner).map_err(js_error)
    }
    pub fn set(&mut self, slot: Number, tensor: &WasmWgpuTensor) -> Result<(), JsValue> {
        self.inner
            .set(js_u32(slot.as_ref(), "slot")? as usize, &tensor.inner)
            .map_err(js_error)
    }
    /// JSON array of [operation, original-input-slot-or-null] pairs.
    pub fn compile(&self, steps: String) -> Result<WasmPointwisePlan, JsValue> {
        if steps.len() > 65536 {
            return Err(js_error("pointwise recipe exceeds 64 KiB"));
        }
        let steps: Vec<(String, Option<usize>)> = serde_json::from_str(&steps).map_err(js_error)?;
        let steps = steps
            .iter()
            .map(|(name, rhs)| PointwiseStep::named(name, *rhs))
            .collect::<Result<Vec<_>, _>>()
            .map_err(js_error)?;
        let chain = PointwiseChain::new(self.inner.len(), steps).map_err(js_error)?;
        Ok(WasmPointwisePlan {
            inner: self.inner.compile(chain).map_err(js_error)?,
        })
    }
}

#[wasm_bindgen(js_name = WgpuPointwisePlan)]
pub struct WasmPointwisePlan {
    inner: PointwisePlan,
}
#[wasm_bindgen(js_class = WgpuPointwisePlan)]
impl WasmPointwisePlan {
    pub fn run(
        &self,
        inputs: &WasmPointwiseInputs,
        execution: String,
    ) -> Result<WasmWgpuTensor, JsValue> {
        Ok(WasmWgpuTensor {
            inner: inputs
                .inner
                .run(&self.inner, execution.parse().map_err(js_error)?)
                .map_err(js_error)?,
        })
    }
}
