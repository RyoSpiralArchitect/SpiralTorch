//! Browser ownership and numeric transport over Rust's resident training core.
use super::*;
#[cfg(feature = "webgpu")]
use st_backend_wgpu::resident_training as backend;

#[cfg(feature = "webgpu")]
pub(super) fn compile(
    plan: &InferencePlan,
    tile: Option<Array>,
    kernel: Option<JsString>,
    accumulation: Option<JsString>,
) -> Result<Promise, JsValue> {
    let (tile, kernel, accumulation) = gpu_options(tile, kernel, accumulation)?;
    let plan = plan.clone();
    Ok(future_to_promise(async move {
        let runtime = crate::wgpu_resident::ensure_runtime().await?;
        let inner = plan
            .compile_training_wgpu_with_options(runtime, tile, kernel, accumulation)
            .map_err(js_error)?;
        Ok(WasmResidentTraining { inner, plan }.into())
    }))
}

#[cfg(feature = "webgpu")]
fn training_error(error: backend::TrainingError) -> JsValue {
    if let backend::TrainingError::Rejected { stage, flags } = error {
        let value = js_sys::Error::new(&error.to_string());
        let decorate = || -> Result<JsValue, JsValue> {
            js_sys::Reflect::set(&value, &"stage".into(), &JsValue::from_f64(stage as f64))?;
            js_sys::Reflect::set(
                &value,
                &"flags".into(),
                &JsValue::from_f64(f64::from(flags)),
            )?;
            js_sys::Reflect::set(&value, &"code".into(), &"training_step_rejected".into())?;
            Ok(value.clone().into())
        };
        decorate().unwrap_or_else(|e| e)
    } else {
        js_error(error)
    }
}

#[wasm_bindgen(js_name = ResidentTraining)]
pub struct WasmResidentTraining {
    #[cfg(feature = "webgpu")]
    inner: backend::ResidentDenseTraining,
    #[cfg(feature = "webgpu")]
    plan: InferencePlan,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = ResidentTraining)]
impl WasmResidentTraining {
    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        self.inner
            .input_layout()
            .shape()
            .iter()
            .map(|&n| n as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        self.inner
            .output_layout()
            .shape()
            .iter()
            .map(|&n| n as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = stageCount)]
    pub fn stage_count(&self) -> usize {
        self.inner.stage_count()
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
            &serde_json::json!({"name":info.name,
            "backend":format!("{:?}",info.backend),"device_type":format!("{:?}",info.device_type)})
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

    /// Enqueued attempt, not proof of acceptance. Read the owning snapshot.
    pub fn step(&mut self, learning_rate: Number) -> Result<u64, JsValue> {
        let learning_rate = learning_rate
            .as_f64()
            .ok_or_else(|| js_error("learning_rate must be a number"))?
            as f32;
        self.inner.step(learning_rate).map_err(training_error)
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
    pub fn state_snapshot(&self) -> Result<WasmTrainingSnapshot, JsValue> {
        let inner = self.inner.state_snapshot().map_err(training_error)?;
        Ok(WasmTrainingSnapshot {
            step: inner.submitted_step(),
            generation: inner.batch_generation(),
            input_shape: inner
                .input_layout()
                .shape()
                .iter()
                .map(|&n| n as u32)
                .collect(),
            output_shape: inner
                .output_layout()
                .shape()
                .iter()
                .map(|&n| n as u32)
                .collect(),
            inner: Some(inner),
            plan: self.plan.clone(),
        })
    }

    #[wasm_bindgen(js_name = parameterSnapshot)]
    pub fn parameter_snapshot(&self) -> Result<WasmTrainingParametersSnapshot, JsValue> {
        Ok(WasmTrainingParametersSnapshot {
            inner: Some(self.inner.parameter_snapshot().map_err(training_error)?),
            plan: self.plan.clone(),
        })
    }
}

#[wasm_bindgen(js_name = TrainingLossSnapshot)]
pub struct WasmTrainingLossSnapshot {
    #[cfg(feature = "webgpu")]
    inner: Option<backend::StepReadback>,
    #[cfg(feature = "webgpu")]
    step: u64,
    #[cfg(feature = "webgpu")]
    generation: u64,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = TrainingLossSnapshot)]
impl WasmTrainingLossSnapshot {
    #[wasm_bindgen(getter, js_name = submittedStep)]
    pub fn submitted_step(&self) -> u64 {
        self.step
    }
    #[wasm_bindgen(getter, js_name = batchGeneration)]
    pub fn batch_generation(&self) -> u64 {
        self.generation
    }
    #[wasm_bindgen(unchecked_return_type = "Promise<number>")]
    pub fn read(&mut self) -> Result<Promise, JsValue> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| js_error("snapshot has already been consumed"))?;
        Ok(future_to_promise(async move {
            Ok(JsValue::from_f64(f64::from(
                inner.read_async().await.map_err(training_error)?,
            )))
        }))
    }
}

#[wasm_bindgen(js_name = TrainingSnapshot)]
pub struct WasmTrainingSnapshot {
    #[cfg(feature = "webgpu")]
    inner: Option<backend::TrainingStateReadback>,
    #[cfg(feature = "webgpu")]
    plan: InferencePlan,
    #[cfg(feature = "webgpu")]
    step: u64,
    #[cfg(feature = "webgpu")]
    generation: u64,
    #[cfg(feature = "webgpu")]
    input_shape: Vec<u32>,
    #[cfg(feature = "webgpu")]
    output_shape: Vec<u32>,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = TrainingSnapshot)]
impl WasmTrainingSnapshot {
    #[wasm_bindgen(getter, js_name = submittedStep)]
    pub fn submitted_step(&self) -> u64 {
        self.step
    }
    #[wasm_bindgen(getter, js_name = batchGeneration)]
    pub fn batch_generation(&self) -> u64 {
        self.generation
    }
    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        self.input_shape.clone()
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        self.output_shape.clone()
    }
    #[wasm_bindgen(js_name = readState, unchecked_return_type = "Promise<TrainingState>")]
    pub fn read_state(&mut self) -> Result<Promise, JsValue> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| js_error("snapshot has already been consumed"))?;
        let plan = self.plan.clone();
        Ok(future_to_promise(async move {
            let inner = inner.read_async().await.map_err(training_error)?;
            Ok(WasmTrainingState { inner, plan }.into())
        }))
    }
}

#[wasm_bindgen(js_name = TrainingParametersSnapshot)]
pub struct WasmTrainingParametersSnapshot {
    #[cfg(feature = "webgpu")]
    inner: Option<backend::ParameterReadback>,
    #[cfg(feature = "webgpu")]
    plan: InferencePlan,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = TrainingParametersSnapshot)]
impl WasmTrainingParametersSnapshot {
    #[wasm_bindgen(js_name = readPlan, unchecked_return_type = "Promise<InferencePlan>")]
    pub fn read_plan(&mut self) -> Result<Promise, JsValue> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| js_error("snapshot has already been consumed"))?;
        let plan = self.plan.clone();
        Ok(future_to_promise(async move {
            let layers = inner.read_async().await.map_err(training_error)?;
            let inner = plan.with_dense_parameters(layers).map_err(js_error)?;
            Ok(WasmInferencePlan { inner }.into())
        }))
    }
}

#[wasm_bindgen(js_name = TrainingState)]
pub struct WasmTrainingState {
    #[cfg(feature = "webgpu")]
    inner: backend::TrainingState,
    #[cfg(feature = "webgpu")]
    plan: InferencePlan,
}

#[cfg(feature = "webgpu")]
impl WasmTrainingState {
    fn layer(&self, stage: &Number) -> Result<usize, JsValue> {
        let index = js_u32(stage.as_ref(), "stage")? as usize;
        if index >= self.inner.parameters.len() {
            return Err(js_error("stage out of range"));
        }
        Ok(index)
    }
    fn parameter_values(
        &self,
        stage: &Number,
        gradient: bool,
        bias: bool,
    ) -> Result<Float32Array, JsValue> {
        let index = self.layer(stage)?;
        let values = if gradient {
            let g = &self.inner.parameter_gradients[index];
            if bias {
                &g.bias
            } else {
                &g.weights
            }
        } else {
            let p = &self.inner.parameters[index];
            if bias {
                &p.bias
            } else {
                &p.weights
            }
        };
        Ok(Float32Array::from(values.as_slice()))
    }
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = TrainingState)]
impl WasmTrainingState {
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
    #[wasm_bindgen(getter, js_name = stageCount)]
    pub fn stage_count(&self) -> usize {
        self.inner.parameters.len()
    }
    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        self.inner
            .input_layout
            .shape()
            .iter()
            .map(|&n| n as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        self.inner
            .output_layout
            .shape()
            .iter()
            .map(|&n| n as u32)
            .collect()
    }
    #[wasm_bindgen(js_name = predictionValues)]
    pub fn prediction_values(&self) -> Float32Array {
        Float32Array::from(self.inner.prediction.as_slice())
    }
    #[wasm_bindgen(js_name = inputGradientValues)]
    pub fn input_gradient_values(&self) -> Float32Array {
        Float32Array::from(self.inner.input_gradient.as_slice())
    }
    #[wasm_bindgen(js_name = layerShape)]
    pub fn layer_shape(&self, stage: Number) -> Result<Vec<u32>, JsValue> {
        let p = &self.inner.parameters[self.layer(&stage)?];
        Ok(vec![p.inner as u32, p.cols as u32])
    }
    #[wasm_bindgen(js_name = weightValues)]
    pub fn weight_values(&self, stage: Number) -> Result<Float32Array, JsValue> {
        self.parameter_values(&stage, false, false)
    }
    #[wasm_bindgen(js_name = biasValues)]
    pub fn bias_values(&self, stage: Number) -> Result<Float32Array, JsValue> {
        self.parameter_values(&stage, false, true)
    }
    #[wasm_bindgen(js_name = weightGradientValues)]
    pub fn weight_gradient_values(&self, stage: Number) -> Result<Float32Array, JsValue> {
        self.parameter_values(&stage, true, false)
    }
    #[wasm_bindgen(js_name = biasGradientValues)]
    pub fn bias_gradient_values(&self, stage: Number) -> Result<Float32Array, JsValue> {
        self.parameter_values(&stage, true, true)
    }
    #[wasm_bindgen(js_name = toPlan)]
    pub fn to_plan(&self) -> Result<WasmInferencePlan, JsValue> {
        let inner = self
            .plan
            .with_dense_parameters(self.inner.parameters.clone())
            .map_err(js_error)?;
        Ok(WasmInferencePlan { inner })
    }
}
