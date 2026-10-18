use crate::utils::js_error;
use st_nn::geometry_artifact::GeometryAdapterArtifact as CoreArtifact;
use wasm_bindgen::prelude::*;

/// Shared portable state validation, not a browser HF execution adapter.
#[wasm_bindgen]
pub struct GeometryAdapterArtifact {
    inner: CoreArtifact,
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
