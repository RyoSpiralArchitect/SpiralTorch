use super::*;
use st_nn::causal_lm::{
    CausalLmAdaptationMode, CausalLmLineObjective, NativeCausalLmAdaptationGradients,
};

#[wasm_bindgen]
pub struct CausalLmLineGradients {
    gradients: NativeCausalLmAdaptationGradients,
}

#[wasm_bindgen]
impl CausalLmLineGradients {
    #[wasm_bindgen(js_name = reportJson)]
    pub fn report_json(&self) -> Result<String, JsValue> {
        let g = &self.gradients;
        if let Some(r) = g.line_set_report() {
            serde_json::to_string(r)
        } else if let Some(r) = g.line_menu_report() {
            serde_json::to_string(r)
        } else if let Some(r) = g.line_subset_report() {
            serde_json::to_string(r)
        } else {
            return Err(js_error("missing complete-line report"));
        }
        .map_err(js_error)
    }

    #[wasm_bindgen(js_name = trainingWorkJson)]
    pub fn training_work_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.gradients.line_training_work()).map_err(js_error)
    }

    #[wasm_bindgen(js_name = parameterGradientsJson)]
    pub fn parameter_gradients_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self.gradients.parameters()).map_err(js_error)
    }
}

#[wasm_bindgen]
impl NativeCausalLm {
    /// Geometry VJP only: no optimizer or candidate-vocabulary truncation.
    #[wasm_bindgen(js_name = geometryLineGradients)]
    pub fn geometry_line_gradients(
        &self,
        prompt: &Uint32Array,
        content: &Uint32Array,
        lengths: &Uint32Array,
        #[wasm_bindgen(unchecked_param_type = "number")] stop_id: &JsValue,
        objective: &JsString,
        #[wasm_bindgen(unchecked_optional_param_type = "number")] max_tile_values: &JsValue,
    ) -> Result<CausalLmLineGradients, JsValue> {
        self.model
            .validate_adaptation_mode(CausalLmAdaptationMode::Geometry)
            .map_err(js_error)?;
        let stop = js_u32(stop_id, "stop ID")? as usize;
        let cap = tile_limit(max_tile_values)?;
        self.model.line_readout_tile_rows(cap).map_err(js_error)?;
        let spec = bounded_json_string_from_js(objective, 4096, "complete-line objective")?;
        let spec: CausalLmLineObjective = serde_json::from_str(&spec).map_err(js_error)?;
        let n = id_count(prompt, WINDOW_IDS)?;
        let lens = admitted_ids(lengths, id_count(lengths, MAX_LINE_MENU_CANDIDATES)?)?;
        spec.training_shape(&self.model, n, &lens, stop)
            .map_err(js_error)?;
        let lines = packed_lines(content, lens)?;
        let prompt = admitted_ids(prompt, n)?;
        let gradients = spec
            .gradients(&self.model, &prompt, &lines, stop, cap)
            .map_err(js_error)?;
        Ok(CausalLmLineGradients { gradients })
    }

    /// Plain SGD on the owning geometry snapshot, not a finite-decrease guarantee.
    #[wasm_bindgen(js_name = stepLineGeometrySgd)]
    pub fn step_line_geometry_sgd(
        &self,
        gradient: &CausalLmLineGradients,
        #[wasm_bindgen(unchecked_param_type = "number")] rate: &JsValue,
    ) -> Result<NativeCausalLm, JsValue> {
        let rate = rate
            .as_f64()
            .ok_or_else(|| js_error("learning rate must be numeric"))? as f32;
        self.model
            .step_adaptation_sgd(&gradient.gradients, CausalLmAdaptationMode::Geometry, rate)
            .map(|model| NativeCausalLm { model })
            .map_err(js_error)
    }
}
