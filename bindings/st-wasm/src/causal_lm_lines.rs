use super::*;
use st_nn::causal_lm::{
    CausalLmAdaptationMode, CausalLmLineObjective, NativeCausalLmAdaptationGradients,
};

#[wasm_bindgen]
pub struct CausalLmLineGradients {
    gradients: NativeCausalLmAdaptationGradients,
    mode: CausalLmAdaptationMode,
}

#[wasm_bindgen]
pub struct CausalLmLineRayStep {
    model: Decoder,
    report: st_nn::causal_lm::CausalLmAdaptationRayStep,
}

#[wasm_bindgen]
impl CausalLmLineRayStep {
    #[wasm_bindgen(js_name=reportJson)]
    pub fn report_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.report).map_err(js_error)
    }
    #[wasm_bindgen(js_name=intoModel)]
    pub fn into_model(self) -> NativeCausalLm {
        NativeCausalLm { model: self.model }
    }
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
        self.line_vjp(
            prompt,
            content,
            lengths,
            stop_id,
            objective,
            CausalLmAdaptationMode::Geometry,
            max_tile_values,
        )
    }

    #[wasm_bindgen(js_name = lineAdaptationGradients)]
    pub fn line_adaptation_gradients(
        &self,
        prompt: &Uint32Array,
        content: &Uint32Array,
        lengths: &Uint32Array,
        #[wasm_bindgen(unchecked_param_type = "number")] stop_id: &JsValue,
        objective: &JsString,
        mode: &JsString,
        #[wasm_bindgen(unchecked_optional_param_type = "number")] max_tile_values: &JsValue,
    ) -> Result<CausalLmLineGradients, JsValue> {
        let mode = bounded_json_string_from_js(mode, 8, "adaptation mode")?
            .parse()
            .map_err(js_error)?;
        self.line_vjp(
            prompt,
            content,
            lengths,
            stop_id,
            objective,
            mode,
            max_tile_values,
        )
    }

    /// Plain SGD on the owning geometry snapshot, not a finite-decrease guarantee.
    #[wasm_bindgen(js_name = stepLineGeometrySgd)]
    pub fn step_line_geometry_sgd(
        &self,
        gradient: &CausalLmLineGradients,
        #[wasm_bindgen(unchecked_param_type = "number")] rate: &JsValue,
    ) -> Result<NativeCausalLm, JsValue> {
        if gradient.mode != CausalLmAdaptationMode::Geometry {
            return Err(js_error("expected a geometry-only complete-line gradient"));
        }
        self.step_line_adaptation_sgd(gradient, rate)
    }

    #[wasm_bindgen(js_name=stepLineGradientAtDistance)]
    pub fn step_line_gradient_at_distance(
        &self,
        gradient: &CausalLmLineGradients,
        #[wasm_bindgen(unchecked_param_type = "number")] target_l2: &JsValue,
    ) -> Result<CausalLmLineRayStep, JsValue> {
        let target = target_l2
            .as_f64()
            .ok_or_else(|| js_error("parameter distance must be numeric"))?;
        self.model
            .step_adaptation_gradient_at_distance(&gradient.gradients, gradient.mode, target)
            .map(|(model, report)| CausalLmLineRayStep { model, report })
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = stepLineAdaptationSgd)]
    pub fn step_line_adaptation_sgd(
        &self,
        gradient: &CausalLmLineGradients,
        #[wasm_bindgen(unchecked_param_type = "number")] rate: &JsValue,
    ) -> Result<NativeCausalLm, JsValue> {
        let rate = rate
            .as_f64()
            .ok_or_else(|| js_error("learning rate must be numeric"))? as f32;
        self.model
            .step_adaptation_sgd(&gradient.gradients, gradient.mode, rate)
            .map(|model| NativeCausalLm { model })
            .map_err(js_error)
    }
}

impl NativeCausalLm {
    fn line_vjp(
        &self,
        prompt: &Uint32Array,
        content: &Uint32Array,
        lengths: &Uint32Array,
        stop_id: &JsValue,
        objective: &JsString,
        mode: CausalLmAdaptationMode,
        max_tile_values: &JsValue,
    ) -> Result<CausalLmLineGradients, JsValue> {
        self.model
            .validate_adaptation_mode(mode)
            .map_err(js_error)?;
        let stop = js_u32(stop_id, "stop ID")? as usize;
        let cap = tile_limit(max_tile_values)?;
        self.model.line_readout_tile_rows(cap).map_err(js_error)?;
        let spec = bounded_json_string_from_js(
            objective,
            CausalLmLineObjective::MAX_SPEC_BYTES as u64,
            "complete-line objective",
        )?;
        let spec: CausalLmLineObjective = serde_json::from_str(&spec).map_err(js_error)?;
        let n = id_count(prompt, WINDOW_IDS)?;
        let lens = admitted_ids(lengths, id_count(lengths, MAX_LINE_MENU_CANDIDATES)?)?;
        spec.training_shape_for(&self.model, n, &lens, stop, mode)
            .map_err(js_error)?;
        let lines = packed_lines(content, lens)?;
        let prompt = admitted_ids(prompt, n)?;
        let gradients = spec
            .gradients_for(&self.model, &prompt, &lines, stop, cap, mode)
            .map_err(js_error)?;
        Ok(CausalLmLineGradients { gradients, mode })
    }
}
