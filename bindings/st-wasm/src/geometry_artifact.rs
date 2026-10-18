use crate::utils::js_error;
use st_nn::geometry_artifact::GeometryAdapterArtifact as CoreArtifact;
use st_nn::language_windows::CausalDocumentWindows as CoreWindows;
use wasm_bindgen::prelude::*;

/// Shared portable state validation, not a browser HF execution adapter.
#[wasm_bindgen]
pub struct GeometryAdapterArtifact {
    inner: CoreArtifact,
}

/// Token lengths only; document content and tokenization remain client-owned.
#[wasm_bindgen]
pub struct CausalDocumentWindows {
    inner: CoreWindows,
}

#[wasm_bindgen]
impl CausalDocumentWindows {
    #[wasm_bindgen(js_name = createJson)]
    pub fn create_json(payload: &str) -> Result<Self, JsValue> {
        CoreWindows::create_json(payload.as_bytes())
            .map(|inner| Self { inner })
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        self.inner.to_json().map_err(js_error)
    }

    #[wasm_bindgen(getter, js_name = planId)]
    pub fn plan_id(&self) -> Result<String, JsValue> {
        self.inner.plan_id().map_err(js_error)
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.windows().len()
    }

    pub fn window(&self, index: JsValue) -> Result<Vec<u32>, JsValue> {
        let index = index
            .as_f64()
            .ok_or_else(|| js_error("window index must be a number"))?;
        if !index.is_finite() || index.fract() != 0.0 || index < 0.0 || index > u32::MAX as f64 {
            return Err(js_error(
                "window index must be a portable nonnegative integer",
            ));
        }
        self.inner
            .windows()
            .get(index as usize)
            .map(|w| vec![w.document, w.start, w.length])
            .ok_or_else(|| js_error("window index out of range"))
    }
}

#[wasm_bindgen]
impl GeometryAdapterArtifact {
    #[wasm_bindgen(js_name = createJson)]
    pub fn create_json(payload: &str) -> Result<Self, JsValue> {
        CoreArtifact::create_json(payload.as_bytes())
            .map(|inner| Self { inner })
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(
        payload: &str,
        expected_id: &str,
        expected_base_model_identity: &str,
    ) -> Result<Self, JsValue> {
        CoreArtifact::from_json(
            payload.as_bytes(),
            expected_id,
            expected_base_model_identity,
        )
        .map(|inner| Self { inner })
        .map_err(js_error)
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        self.inner.to_json().to_owned()
    }

    #[wasm_bindgen(js_name = stateJson)]
    pub fn state_json(&self) -> Result<String, JsValue> {
        self.inner.state_json().map_err(js_error)
    }

    #[wasm_bindgen(getter, js_name = artifactId)]
    pub fn artifact_id(&self) -> String {
        self.inner.artifact_id().to_owned()
    }

    #[wasm_bindgen(getter, js_name = layoutId)]
    pub fn layout_id(&self) -> Result<String, JsValue> {
        self.inner.layout_id().map_err(js_error)
    }

    #[wasm_bindgen(getter, js_name = baseModelIdentity)]
    pub fn base_model_identity(&self) -> String {
        self.inner.record().base_model_identity.clone()
    }
}
