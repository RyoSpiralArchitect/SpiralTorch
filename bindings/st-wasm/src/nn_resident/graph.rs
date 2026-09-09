//! Browser ownership over the same Rust graph executor as the Python client.
#[cfg(feature = "webgpu")]
use super::training::training_error;
use super::*;
#[cfg(feature = "webgpu")]
use st_backend_wgpu::resident_training::graph as backend;

#[cfg(feature = "webgpu")]
pub(super) fn compile(
    plan: &InferencePlan,
    policy: st_nn::resident::GraphGradientPolicy,
    tile: Option<Array>,
    kernel: Option<JsString>,
    accumulation: Option<JsString>,
) -> Result<Promise, JsValue> {
    let (tile, kernel, accumulation) = gpu_options(tile, kernel, accumulation)?;
    let plan = plan.clone();
    Ok(future_to_promise(async move {
        let runtime = crate::wgpu_resident::ensure_runtime().await?;
        let inner = plan
            .compile_graph_training_wgpu_with_options(runtime, policy, tile, kernel, accumulation)
            .map_err(js_error)?;
        Ok(WasmResidentGraphTraining { inner }.into())
    }))
}

#[wasm_bindgen(js_name = ResidentGraphTraining)]
pub struct WasmResidentGraphTraining {
    #[cfg(feature = "webgpu")]
    inner: backend::ResidentGraphTraining,
}

#[cfg(feature = "webgpu")]
fn shape(layout: &st_tensor::NdLayout) -> Vec<u32> {
    layout.shape().iter().map(|&n| n as u32).collect()
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = ResidentGraphTraining)]
impl WasmResidentGraphTraining {
    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        shape(self.inner.input_layout())
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        shape(self.inner.output_layout())
    }
    #[wasm_bindgen(getter, js_name = stageCount)]
    pub fn stage_count(&self) -> usize {
        self.inner.stage_count()
    }
    #[wasm_bindgen(getter, js_name = parameterCount)]
    pub fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }
    #[wasm_bindgen(getter, js_name = gradientPolicy)]
    pub fn gradient_policy(&self) -> String {
        self.inner.gradient_policy().as_str().into()
    }
    #[wasm_bindgen(getter, js_name = submittedSteps)]
    pub fn submitted_steps(&self) -> u64 {
        self.inner.submitted_steps()
    }
    #[wasm_bindgen(getter, js_name = batchGeneration)]
    pub fn batch_generation(&self) -> u64 {
        self.inner.batch_generation()
    }

    #[wasm_bindgen(js_name = adapterInfo, unchecked_return_type = "{ name: string; backend: string; device_type: string }")]
    pub fn adapter_info(&self) -> Result<JsValue, JsValue> {
        let info = self.inner.adapter_info();
        crate::utils::json_to_js_value(
            &serde_json::json!({
                "name": info.name, "backend": format!("{:?}", info.backend),
                "device_type": format!("{:?}", info.device_type),
            })
            .to_string(),
        )
    }
    #[wasm_bindgen(js_name = uploadBatch)]
    pub fn upload_batch(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "Float32Array")] input: JsValue,
        #[wasm_bindgen(unchecked_param_type = "Float32Array")] target: JsValue,
    ) -> Result<(), JsValue> {
        if !crate::utils::js_is_typed_array(&input, "Float32Array")?
            || !crate::utils::js_is_typed_array(&target, "Float32Array")?
        {
            return Err(js_error("input and target must be Float32Array values"));
        }
        let input: Float32Array = input.unchecked_into();
        let target: Float32Array = target.unchecked_into();
        if input.length() as usize != self.inner.input_layout().len()
            || target.length() as usize != self.inner.output_layout().len()
        {
            return Err(js_error(
                "batch lengths must match the compiled input and output shapes",
            ));
        }
        self.inner
            .upload_batch(&input.to_vec(), &target.to_vec())
            .map_err(training_error)
    }
    /// Enqueued attempt, not proof of acceptance. Read an owning loss/state snapshot.
    pub fn step(&mut self, learning_rate: Number) -> Result<u64, JsValue> {
        let rate = learning_rate
            .as_f64()
            .ok_or_else(|| js_error("learning_rate must be a number"))? as f32;
        self.inner.step(rate).map_err(training_error)
    }
    #[wasm_bindgen(js_name = lossSnapshot)]
    pub fn loss_snapshot(&self) -> Result<WasmTrainingLossSnapshot, JsValue> {
        let inner = self.inner.loss_snapshot().map_err(training_error)?;
        Ok(WasmTrainingLossSnapshot {
            step: inner.submitted_step(),
            generation: inner.batch_generation(),
            inner: Some(inner),
        })
    }
    #[wasm_bindgen(js_name = stateSnapshot)]
    pub fn state_snapshot(&self) -> Result<WasmGraphTrainingSnapshot, JsValue> {
        let inner = self.inner.state_snapshot().map_err(training_error)?;
        Ok(WasmGraphTrainingSnapshot {
            step: inner.submitted_step(),
            generation: inner.batch_generation(),
            input_shape: shape(inner.input_layout()),
            output_shape: shape(inner.output_layout()),
            policy: inner.gradient_policy().as_str(),
            inner: Some(inner),
        })
    }
    #[wasm_bindgen(js_name = parameterSnapshot)]
    pub fn parameter_snapshot(&self) -> Result<WasmGraphTrainingParametersSnapshot, JsValue> {
        Ok(WasmGraphTrainingParametersSnapshot {
            inner: Some(self.inner.parameter_snapshot().map_err(training_error)?),
        })
    }
}

#[wasm_bindgen(js_name = GraphTrainingSnapshot)]
pub struct WasmGraphTrainingSnapshot {
    #[cfg(feature = "webgpu")]
    inner: Option<backend::GraphStateReadback>,
    #[cfg(feature = "webgpu")]
    step: u64,
    #[cfg(feature = "webgpu")]
    generation: u64,
    #[cfg(feature = "webgpu")]
    input_shape: Vec<u32>,
    #[cfg(feature = "webgpu")]
    output_shape: Vec<u32>,
    #[cfg(feature = "webgpu")]
    policy: &'static str,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = GraphTrainingSnapshot)]
impl WasmGraphTrainingSnapshot {
    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        self.input_shape.clone()
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        self.output_shape.clone()
    }
    #[wasm_bindgen(getter, js_name = submittedStep)]
    pub fn submitted_step(&self) -> u64 {
        self.step
    }
    #[wasm_bindgen(getter, js_name = batchGeneration)]
    pub fn batch_generation(&self) -> u64 {
        self.generation
    }
    #[wasm_bindgen(getter, js_name = gradientPolicy)]
    pub fn gradient_policy(&self) -> String {
        self.policy.into()
    }
    #[wasm_bindgen(js_name = readState, unchecked_return_type = "Promise<GraphTrainingState>")]
    pub fn read_state(&mut self) -> Result<Promise, JsValue> {
        // Take before returning: JS may free the handle while the read is pending.
        let inner = self
            .inner
            .take()
            .ok_or_else(|| js_error("snapshot has already been consumed"))?;
        Ok(future_to_promise(async move {
            Ok(WasmGraphTrainingState {
                inner: inner.read_async().await.map_err(training_error)?,
            }
            .into())
        }))
    }
}

#[wasm_bindgen(js_name = GraphTrainingParametersSnapshot)]
pub struct WasmGraphTrainingParametersSnapshot {
    #[cfg(feature = "webgpu")]
    inner: Option<backend::GraphParameterReadback>,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = GraphTrainingParametersSnapshot)]
impl WasmGraphTrainingParametersSnapshot {
    #[wasm_bindgen(js_name = readPlan, unchecked_return_type = "Promise<InferencePlan>")]
    pub fn read_plan(&mut self) -> Result<Promise, JsValue> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| js_error("snapshot has already been consumed"))?;
        Ok(future_to_promise(async move {
            let graph = inner.read_async().await.map_err(training_error)?;
            Ok(WasmInferencePlan {
                inner: InferencePlan::from_graph_definition(graph).map_err(js_error)?,
            }
            .into())
        }))
    }
}

#[wasm_bindgen(js_name = GraphTrainingState)]
pub struct WasmGraphTrainingState {
    #[cfg(feature = "webgpu")]
    inner: backend::GraphState,
}

#[cfg(feature = "webgpu")]
impl WasmGraphTrainingState {
    fn parameter_id(&self, parameter: &Number) -> Result<usize, JsValue> {
        let index = js_u32(parameter.as_ref(), "parameter")? as usize;
        if index >= self.inner.graph.parameters().len() {
            return Err(js_error("parameter out of range"));
        }
        Ok(index)
    }
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = GraphTrainingState)]
impl WasmGraphTrainingState {
    #[wasm_bindgen(getter)]
    pub fn loss(&self) -> f32 {
        self.inner.loss
    }
    #[wasm_bindgen(getter, js_name = submittedStep)]
    pub fn submitted_step(&self) -> u64 {
        self.inner.submitted_step
    }
    #[wasm_bindgen(getter, js_name = batchGeneration)]
    pub fn batch_generation(&self) -> u64 {
        self.inner.batch_generation
    }
    #[wasm_bindgen(getter, js_name = gradientPolicy)]
    pub fn gradient_policy(&self) -> String {
        self.inner.gradient_policy.as_str().into()
    }
    #[wasm_bindgen(getter, js_name = stageCount)]
    pub fn stage_count(&self) -> usize {
        self.inner.graph.stages().len()
    }
    #[wasm_bindgen(getter, js_name = parameterCount)]
    pub fn parameter_count(&self) -> usize {
        self.inner.graph.parameters().len()
    }
    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        shape(self.inner.graph.input_layout())
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        shape(self.inner.graph.output_layout())
    }
    #[wasm_bindgen(js_name = predictionValues)]
    pub fn prediction_values(&self) -> Float32Array {
        Float32Array::from(self.inner.prediction.as_slice())
    }
    #[wasm_bindgen(js_name = inputGradientValues)]
    pub fn input_gradient_values(&self) -> Float32Array {
        Float32Array::from(self.inner.input_gradient.as_slice())
    }
    #[wasm_bindgen(js_name = parameterRole)]
    pub fn parameter_role(&self, parameter: Number) -> Result<String, JsValue> {
        Ok(
            self.inner.graph.parameters()[self.parameter_id(&parameter)?]
                .role
                .as_str()
                .into(),
        )
    }
    #[wasm_bindgen(js_name = parameterShape)]
    pub fn parameter_shape(&self, parameter: Number) -> Result<Vec<u32>, JsValue> {
        Ok(
            self.inner.graph.parameters()[self.parameter_id(&parameter)?]
                .shape
                .iter()
                .map(|&n| n as u32)
                .collect(),
        )
    }
    #[wasm_bindgen(js_name = parameterValues)]
    pub fn parameter_values(&self, parameter: Number) -> Result<Float32Array, JsValue> {
        Ok(Float32Array::from(
            self.inner.graph.parameters()[self.parameter_id(&parameter)?]
                .values
                .as_slice(),
        ))
    }
    #[wasm_bindgen(js_name = parameterGradientValues)]
    pub fn parameter_gradient_values(&self, parameter: Number) -> Result<Float32Array, JsValue> {
        Ok(Float32Array::from(
            self.inner.raw_gradients[self.parameter_id(&parameter)?].as_slice(),
        ))
    }
    #[wasm_bindgen(js_name = effectiveGradientValues)]
    pub fn effective_gradient_values(&self, parameter: Number) -> Result<Float32Array, JsValue> {
        Ok(Float32Array::from(
            self.inner.effective_gradients[self.parameter_id(&parameter)?].as_slice(),
        ))
    }
    /// Weight-only v2 plan; runtime counters, batch and policy are not serialized.
    #[wasm_bindgen(js_name = toPlan)]
    pub fn to_plan(&self) -> Result<WasmInferencePlan, JsValue> {
        Ok(WasmInferencePlan {
            inner: InferencePlan::from_graph_definition(self.inner.graph.clone())
                .map_err(js_error)?,
        })
    }
}
