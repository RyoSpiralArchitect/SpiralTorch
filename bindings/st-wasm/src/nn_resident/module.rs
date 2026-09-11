//! A browser-owned real Rust Module, not a separately reconstructed graph.
use super::*;
use crate::wgpu_tensor::{values, WasmWgpuTensor};
use st_nn::{Gelu, Linear, Module, Relu, Scaler, Sequential};
use st_tensor::{NdLayout, Tensor};

#[wasm_bindgen(js_name = ResidentForwardStats)]
pub struct WasmResidentForwardStats {
    #[wasm_bindgen(readonly)]
    pub compilations: u64,
    #[wasm_bindgen(readonly, js_name = cacheHits)]
    pub cache_hits: u64,
    #[wasm_bindgen(readonly, js_name = submittedForwards)]
    pub submitted_forwards: u64,
}

#[wasm_bindgen(js_name = Sequential)]
pub struct WasmSequential {
    pub(super) inner: Sequential,
}

#[wasm_bindgen(js_class = Sequential)]
impl WasmSequential {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Sequential::new(),
        }
    }

    #[wasm_bindgen(js_name = addLinear)]
    pub fn add_linear(
        &mut self,
        name: JsString,
        input_dim: Number,
        output_dim: Number,
    ) -> Result<(), JsValue> {
        let name = name
            .as_string()
            .ok_or_else(|| js_error("name must be a string"))?;
        let input = js_u32(input_dim.as_ref(), "input_dim")? as usize;
        let output = js_u32(output_dim.as_ref(), "output_dim")? as usize;
        self.inner
            .push(Linear::new(name, input, output).map_err(js_error)?);
        Ok(())
    }

    #[wasm_bindgen(js_name = addScaler)]
    pub fn add_scaler(
        &mut self,
        name: JsString,
        #[wasm_bindgen(unchecked_param_type = "Float32Array")] gain: JsValue,
    ) -> Result<(), JsValue> {
        let name = name
            .as_string()
            .ok_or_else(|| js_error("name must be a string"))?;
        let gain = values(gain)?.to_vec();
        let gain = Tensor::from_vec(1, gain.len(), gain).map_err(js_error)?;
        self.inner
            .push(Scaler::from_gain(name, gain).map_err(js_error)?);
        Ok(())
    }

    #[wasm_bindgen(js_name = addGelu)]
    pub fn add_gelu(&mut self) {
        self.inner.push(Gelu::new());
    }

    #[wasm_bindgen(js_name = addRelu)]
    pub fn add_relu(&mut self) {
        self.inner.push(Relu::new());
    }

    /// GPU submission only. The output owns its capture; readback is explicit.
    pub fn forward(&self, input: &WasmWgpuTensor) -> Result<WasmWgpuTensor, JsValue> {
        Ok(WasmWgpuTensor {
            inner: self
                .inner
                .forward_resident(&input.inner)
                .map_err(js_error)?,
        })
    }

    #[wasm_bindgen(js_name = inferencePlan)]
    pub fn inference_plan(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number[]")] shape: Array,
    ) -> Result<WasmInferencePlan, JsValue> {
        let shape = shape
            .iter()
            .map(|v| js_u32(&v, "dimension").map(|v| v as usize))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(WasmInferencePlan {
            inner: InferencePlan::from_module(
                &self.inner,
                NdLayout::contiguous(&shape).map_err(js_error)?,
            )
            .map_err(js_error)?,
        })
    }

    #[wasm_bindgen(js_name = residentCacheInfo)]
    pub fn resident_cache_info(&self) -> WasmResidentForwardStats {
        let s = self.inner.resident_forward_stats().unwrap();
        WasmResidentForwardStats {
            compilations: s.compilations,
            cache_hits: s.cache_hits,
            submitted_forwards: s.submitted_forwards,
        }
    }

    #[wasm_bindgen(js_name = clearResidentCache)]
    pub fn clear_resident_cache(&self) {
        self.inner.clear_resident_forward_cache();
    }
}

impl Default for WasmSequential {
    fn default() -> Self {
        Self::new()
    }
}
