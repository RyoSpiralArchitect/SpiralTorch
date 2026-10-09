//! No browser-side attention semantics: only handle/number conversion.
use super::*;
use st_backend_wgpu::resident_tensor::attention::{AttentionMask, ResidentAttentionGradients};

/// Borrow-safe optional inputs: wasm-bindgen cannot pass Option<&exported type>.
/// Setting a bias clones only its immutable Rust handle, never its GPU storage.
#[wasm_bindgen(js_name = WgpuAttentionBiases)]
#[derive(Default)]
pub struct WasmAttentionBiases {
    z_bias: Option<ResidentTensor>,
    pair_bias: Option<ResidentTensor>,
}

#[wasm_bindgen(js_class = WgpuAttentionBiases)]
impl WasmAttentionBiases {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }
    #[wasm_bindgen(js_name = setZBias)]
    pub fn set_z_bias(&mut self, bias: &WasmWgpuTensor) {
        self.z_bias = Some(bias.inner.clone());
    }
    #[wasm_bindgen(js_name = setPairBias)]
    pub fn set_pair_bias(&mut self, bias: &WasmWgpuTensor) {
        self.pair_bias = Some(bias.inner.clone());
    }
    #[wasm_bindgen(js_name = clearZBias)]
    pub fn clear_z_bias(&mut self) {
        self.z_bias = None;
    }
    #[wasm_bindgen(js_name = clearPairBias)]
    pub fn clear_pair_bias(&mut self) {
        self.pair_bias = None;
    }
}

fn mask(offset: Option<Number>) -> Result<AttentionMask, JsValue> {
    offset
        .map(|v| js_u32(v.as_ref(), "causal_offset").map(|v| v as usize))
        .transpose()
        .map(|value| {
            value.map_or(AttentionMask::None, |query_offset| AttentionMask::Causal {
                query_offset,
            })
        })
}

#[wasm_bindgen(js_class = WgpuTensor)]
impl WasmWgpuTensor {
    /// None is unmasked; causal_offset=0 masks future keys structurally.
    #[wasm_bindgen(js_name = scaledDotAttention)]
    #[allow(clippy::too_many_arguments)]
    pub fn scaled_dot_attention(
        &self,
        keys: &WasmWgpuTensor,
        values: &WasmWgpuTensor,
        scale: f32,
        causal_offset: Option<Number>,
        biases: &WasmAttentionBiases,
    ) -> Result<WasmWgpuTensor, JsValue> {
        Ok(Self {
            inner: self
                .inner
                .scaled_dot_attention(
                    &keys.inner,
                    &values.inner,
                    scale,
                    mask(causal_offset)?,
                    biases.z_bias.as_ref(),
                    biases.pair_bias.as_ref(),
                )
                .map_err(js_error)?,
        })
    }

    /// Upstream and gradients use logical [B,H,Q,D] / input shapes, not means.
    #[wasm_bindgen(js_name = scaledDotAttentionVjp)]
    #[allow(clippy::too_many_arguments)]
    pub fn scaled_dot_attention_vjp(
        &self,
        keys: &WasmWgpuTensor,
        values: &WasmWgpuTensor,
        upstream: &WasmWgpuTensor,
        scale: f32,
        causal_offset: Option<Number>,
        biases: &WasmAttentionBiases,
    ) -> Result<WasmAttentionGradients, JsValue> {
        Ok(WasmAttentionGradients {
            inner: self
                .inner
                .scaled_dot_attention_vjp(
                    &keys.inner,
                    &values.inner,
                    &upstream.inner,
                    scale,
                    mask(causal_offset)?,
                    biases.z_bias.as_ref(),
                    biases.pair_bias.as_ref(),
                )
                .map_err(js_error)?,
        })
    }
}

#[wasm_bindgen(js_name = WgpuAttentionGradients)]
pub struct WasmAttentionGradients {
    inner: ResidentAttentionGradients,
}

#[wasm_bindgen(js_class = WgpuAttentionGradients)]
impl WasmAttentionGradients {
    #[wasm_bindgen(getter)]
    pub fn query(&self) -> WasmWgpuTensor {
        WasmWgpuTensor {
            inner: self.inner.query.clone(),
        }
    }
    #[wasm_bindgen(getter)]
    pub fn key(&self) -> WasmWgpuTensor {
        WasmWgpuTensor {
            inner: self.inner.key.clone(),
        }
    }
    #[wasm_bindgen(getter)]
    pub fn value(&self) -> WasmWgpuTensor {
        WasmWgpuTensor {
            inner: self.inner.value.clone(),
        }
    }
    #[wasm_bindgen(getter, js_name = zBias)]
    pub fn z_bias(&self) -> Option<WasmWgpuTensor> {
        self.inner
            .z_bias
            .clone()
            .map(|inner| WasmWgpuTensor { inner })
    }
    #[wasm_bindgen(getter, js_name = pairBias)]
    pub fn pair_bias(&self) -> Option<WasmWgpuTensor> {
        self.inner
            .pair_bias
            .clone()
            .map(|inner| WasmWgpuTensor { inner })
    }
}
