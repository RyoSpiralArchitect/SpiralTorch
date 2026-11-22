use super::*;
use st_nn::causal_lm::NativeCausalLmLineTrainingSession as Session;

#[wasm_bindgen]
pub struct NativeCausalLmLineTrainingSession {
    inner: Session,
}

#[wasm_bindgen]
pub struct CausalLmLineCheckpoint {
    payload: String,
    id: String,
}

#[wasm_bindgen]
impl CausalLmLineCheckpoint {
    #[wasm_bindgen(getter)]
    pub fn payload(&self) -> String {
        self.payload.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }
}

#[wasm_bindgen]
impl NativeCausalLmLineTrainingSession {
    #[wasm_bindgen(constructor)]
    pub fn new(
        origin: &NativeCausalLm,
        plan: &JsString,
    ) -> Result<NativeCausalLmLineTrainingSession, JsValue> {
        let bytes = bounded_json_string_from_js(
            plan,
            Session::MAX_PLAN_BYTES as u64,
            "complete-line plan",
        )?;
        Session::new(origin.model.clone(), bytes.as_bytes())
            .map(|inner| Self { inner })
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name=fromJson)]
    pub fn from_json(
        origin: &NativeCausalLm,
        plan: &JsString,
        payload: &JsString,
        expected_id: &JsString,
    ) -> Result<NativeCausalLmLineTrainingSession, JsValue> {
        let id = bounded_json_string_from_js(expected_id, 71, "line checkpoint pin")?;
        let plan = bounded_json_string_from_js(
            plan,
            Session::MAX_PLAN_BYTES as u64,
            "complete-line plan",
        )?;
        let bytes = bounded_json_string_from_js(
            payload,
            Session::MAX_CHECKPOINT_BYTES as u64,
            "line checkpoint",
        )?;
        Session::restore(origin.model.clone(), plan.as_bytes(), bytes.as_bytes(), &id)
            .map(|inner| Self { inner })
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name=modelSnapshot)]
    pub fn model_snapshot(&self) -> NativeCausalLm {
        NativeCausalLm {
            model: self.inner.model().clone(),
        }
    }
    #[wasm_bindgen(getter,js_name=planId)]
    pub fn plan_id(&self) -> String {
        self.inner.plan_id().into()
    }
    #[wasm_bindgen(getter,js_name=completedUpdates)]
    pub fn completed_updates(&self) -> u32 {
        self.inner.completed_updates()
    }
    #[wasm_bindgen(getter,js_name=totalUpdates)]
    pub fn total_updates(&self) -> u32 {
        self.inner.total_updates()
    }
    #[wasm_bindgen(getter)]
    pub fn exhausted(&self) -> bool {
        self.inner.exhausted()
    }

    #[wasm_bindgen(js_name=stepJson)]
    pub fn step_json(&mut self) -> Result<Option<String>, JsValue> {
        self.inner
            .step()
            .map_err(js_error)?
            .map(|r| serde_json::to_string(&r).map_err(js_error))
            .transpose()
    }

    pub fn checkpoint(&self) -> Result<CausalLmLineCheckpoint, JsValue> {
        self.inner
            .checkpoint_with_id()
            .map(|(payload, id)| CausalLmLineCheckpoint { payload, id })
            .map_err(js_error)
    }
}
