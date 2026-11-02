//! Token transport only; model loading, execution, loss and geometry stay in Rust.
use crate::channel_affine::{dimension, reject_bool};
use crate::geometry_artifact::PyGeometryAdapterArtifact;
use crate::geometry_runtime::{adamw_step, PyNativeGeometryAdamW};
use crate::weights::{PyNdTensor, PySafeTensorArchive};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use st_nn::causal_lm::{
    CausalLmLimits, CausalLmObjective, CausalLmReadoutAccumulator, NativeCausalLm,
    NativeCausalLmGradients, NativeCausalLmLoraGradients,
};
use std::collections::BTreeMap;

mod accumulation;
mod context;
mod documents;
mod evaluation;
mod generation;
mod session;
mod session_input;
mod training;
use accumulation::PyNativeCausalLmAccumulator;
use session::PyNativeCausalLmAdaptationGradients;

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

fn shape(value: &Bound<'_, PyAny>) -> PyResult<[usize; 2]> {
    if value.len()? != 2 {
        return Err(error("native causal LM shape must be [batch, sequence]"));
    }
    Ok([
        dimension(&value.get_item(0)?)?,
        dimension(&value.get_item(1)?)?,
    ])
}

fn tokens(value: &Bound<'_, PyAny>, count: usize) -> PyResult<Vec<usize>> {
    if value.len()? != count {
        return Err(error("token ID count differs from shape"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(error)?;
    for i in 0..count {
        result.push(dimension(&value.get_item(i)?)?);
    }
    Ok(result)
}

fn loss_objective(
    value: Option<&Bound<'_, PyAny>>,
    repetition: Option<&Bound<'_, PyAny>>,
) -> PyResult<CausalLmObjective> {
    let label_smoothing = match value {
        Some(value) => {
            reject_bool(value)?;
            value.extract::<f64>()?
        }
        None => 0.,
    };
    let repetition = repetition
        .map(|value| {
            let value = crate::json::py_to_json_bounded(
                value,
                65_536,
                1024,
                8,
                "native repetition objective",
            )?;
            serde_json::from_value(value).map_err(error)
        })
        .transpose()?;
    let objective = CausalLmObjective {
        label_smoothing,
        repetition,
    };
    objective.validate().map_err(error)?;
    Ok(objective)
}

fn loss_labels(value: Option<&Bound<'_, PyAny>>, count: usize) -> PyResult<Option<Vec<i64>>> {
    let Some(value) = value else { return Ok(None) };
    if value.len()? != count {
        return Err(error("label count differs from window shape"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(error)?;
    for i in 0..count {
        let item = value.get_item(i)?;
        reject_bool(&item)?;
        result.push(item.extract::<i64>()?);
    }
    Ok(Some(result))
}

fn limit(value: Option<&Bound<'_, PyAny>>, default: usize) -> PyResult<usize> {
    match value {
        Some(value) => {
            reject_bool(value)?;
            value.extract()
        }
        None => Ok(default),
    }
}

#[pyclass(name = "NativeCausalLm", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLm {
    inner: NativeCausalLm,
}

#[pyclass(name = "NativeCausalLmGradients", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLmGradients {
    inner: NativeCausalLmGradients,
}

#[pymethods]
impl PyNativeCausalLmGradients {
    #[getter]
    fn readout_accumulator(&self) -> &'static str {
        self.inner.readout_accumulator().as_str()
    }
    #[getter]
    fn loss(&self) -> f32 {
        self.inner.loss()
    }
    #[getter]
    fn causal_loss(&self) -> f32 {
        self.inner.causal_loss()
    }
    #[getter]
    fn repetition_loss(&self) -> f32 {
        self.inner.repetition_loss()
    }
    #[getter]
    fn repetition_positions(&self) -> usize {
        self.inner.repetition_positions()
    }
    fn repetition_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.objective().repetition).map_err(error)
    }
    #[getter]
    fn base_model_identity(&self) -> &str {
        self.inner.base_model_identity()
    }
    #[getter]
    fn geometry_artifact_id(&self) -> &str {
        self.inner.geometry_artifact_id()
    }
    #[getter]
    fn window_shape(&self) -> Vec<usize> {
        self.inner.window_shape().to_vec()
    }
    #[getter]
    fn target_tokens(&self) -> usize {
        self.inner.target_tokens()
    }
    #[getter]
    fn label_smoothing(&self) -> f64 {
        self.inner.objective().label_smoothing
    }
    fn parameter_gradients(&self) -> BTreeMap<String, BTreeMap<String, Vec<f32>>> {
        self.inner.parameters().clone()
    }
}

#[pyclass(name = "NativeCausalLmLoraGradients", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLmLoraGradients {
    inner: NativeCausalLmLoraGradients,
}

#[pymethods]
impl PyNativeCausalLmLoraGradients {
    #[getter]
    fn loss(&self) -> f32 {
        self.inner.loss()
    }
    #[getter]
    fn target_tokens(&self) -> usize {
        self.inner.target_tokens()
    }
    #[getter]
    fn window_shape(&self) -> Vec<usize> {
        self.inner.window_shape().to_vec()
    }
    #[getter]
    fn base_model_identity(&self) -> &str {
        self.inner.base_model_identity()
    }
    #[getter]
    fn lora_artifact_id(&self) -> &str {
        self.inner.lora_artifact_id()
    }
    fn parameter_gradients(&self) -> BTreeMap<String, Vec<f32>> {
        BTreeMap::from([
            ("a".into(), self.inner.a().to_vec()),
            ("b".into(), self.inner.b().to_vec()),
        ])
    }
}

#[pymethods]
impl PyNativeCausalLm {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(base, payload, expected_id, *, update, recorded_data_id, post_lora_id, post_geometry_id))]
    fn from_session_checkpoint_at(
        py: Python<'_>,
        base: &Self,
        payload: &Bound<'_, PyAny>,
        expected_id: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        recorded_data_id: &Bound<'_, PyAny>,
        post_lora_id: Option<&Bound<'_, PyAny>>,
        post_geometry_id: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        reject_bool(update)?;
        let update = update.extract::<u64>()?;
        let expected_id = session_input::pin(expected_id)?;
        let data = session_input::pin(recorded_data_id)?;
        let lora = post_lora_id.map(session_input::pin).transpose()?;
        let geometry = post_geometry_id.map(session_input::pin).transpose()?;
        let payload = session_input::payload(payload)?;
        let bytes = payload.as_bytes();
        py.detach(|| {
            NativeCausalLm::from_session_checkpoint_at(
                base.inner.clone(),
                bytes,
                &expected_id,
                update,
                &data,
                lora.as_deref(),
                geometry.as_deref(),
            )
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }
    /// With no mode, expose all attached gradients. Otherwise retain the full
    /// forward function but return only the Rust-selected parameter groups.
    #[pyo3(signature=(windows, shape, *, mode=None))]
    fn adaptation_loss_gradients(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        mode: Option<&str>,
    ) -> PyResult<PyNativeCausalLmAdaptationGradients> {
        let mode = mode.map(session::mode).transpose()?;
        if let Some(mode) = mode {
            self.inner.validate_adaptation_mode(mode).map_err(error)?;
        }
        let shape = self::shape(shape)?;
        let count = self
            .inner
            .adaptation_gradient_input_count(shape)
            .map_err(error)?;
        let windows = tokens(windows, count)?;
        py.detach(|| match mode {
            Some(mode) => self
                .inner
                .adaptation_loss_gradients_for(&windows, shape, mode),
            None => self.inner.adaptation_loss_gradients(&windows, shape),
        })
        .map(|inner| PyNativeCausalLmAdaptationGradients { inner })
        .map_err(error)
    }

    #[pyo3(signature=(gradients, mode, learning_rate))]
    fn step_adaptation_sgd(
        &self,
        py: Python<'_>,
        gradients: &PyNativeCausalLmAdaptationGradients,
        mode: &str,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let mode = session::mode(mode)?;
        reject_bool(learning_rate)?;
        let learning_rate = learning_rate.extract::<f32>()?;
        py.detach(|| {
            self.inner
                .step_adaptation_sgd(&gradients.inner, mode, learning_rate)
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }

    /// Attach one native GPT-2 attention output adapter; no foreign computation.
    fn with_lora(
        &self,
        py: Python<'_>,
        layer: &Bound<'_, PyAny>,
        rank: &Bound<'_, PyAny>,
        alpha: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let layer = dimension(layer)?;
        let rank = dimension(rank)?;
        reject_bool(alpha)?;
        let alpha = alpha.extract::<f32>()?;
        py.detach(|| self.inner.with_lora(layer, rank, alpha))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[getter]
    fn lora_artifact_id(&self) -> Option<&str> {
        self.inner.lora_artifact_id()
    }

    fn lora_to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.lora_to_json()).map_err(error)
    }

    fn with_lora_json(&self, py: Python<'_>, payload: &str) -> PyResult<Self> {
        py.detach(|| self.inner.with_lora_json(payload))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    fn lora_loss_gradients(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
    ) -> PyResult<PyNativeCausalLmLoraGradients> {
        let shape = self::shape(shape)?;
        let count = self.inner.lora_gradient_input_count(shape).map_err(error)?;
        let windows = tokens(windows, count)?;
        py.detach(|| self.inner.lora_loss_gradients(&windows, shape))
            .map(|inner| PyNativeCausalLmLoraGradients { inner })
            .map_err(error)
    }

    fn step_lora_sgd(
        &self,
        py: Python<'_>,
        gradients: &PyNativeCausalLmLoraGradients,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        reject_bool(learning_rate)?;
        let learning_rate = learning_rate.extract::<f32>()?;
        py.detach(|| self.inner.step_lora_sgd(&gradients.inner, learning_rate))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[staticmethod]
    #[pyo3(signature=(variants, token_documents, context_length, *, microbatch_windows=None, labels=None))]
    fn evaluate_variants(
        py: Python<'_>,
        variants: &Bound<'_, PyAny>,
        token_documents: &Bound<'_, PyAny>,
        context_length: &Bound<'_, PyAny>,
        microbatch_windows: Option<&Bound<'_, PyAny>>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<evaluation::PyNativeCausalLmEvaluationComparison> {
        evaluation::compare(
            py,
            variants,
            token_documents,
            context_length,
            microbatch_windows,
            labels,
        )
    }

    #[pyo3(signature=(token_documents, context_length, *, microbatch_windows=None, labels=None))]
    fn evaluate_documents(
        &self,
        py: Python<'_>,
        token_documents: &Bound<'_, PyAny>,
        context_length: &Bound<'_, PyAny>,
        microbatch_windows: Option<&Bound<'_, PyAny>>,
        labels: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<evaluation::PyNativeCausalLmEvaluation> {
        evaluation::evaluate(
            self,
            py,
            token_documents,
            context_length,
            microbatch_windows,
            labels,
        )
    }

    #[getter]
    fn readout_accumulator(&self) -> &'static str {
        self.inner.readout_accumulator().as_str()
    }

    fn with_readout_accumulator(&self, accumulator: &str) -> PyResult<Self> {
        let accumulator = match accumulator {
            "f32" => CausalLmReadoutAccumulator::F32,
            "f64" => CausalLmReadoutAccumulator::F64,
            _ => return Err(error("readout accumulator must be f32 or f64")),
        };
        Ok(Self {
            inner: self.inner.with_readout_accumulator(accumulator),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (base, payload, expected_id, *, update, recorded_data_id, post_geometry_id))]
    fn from_training_checkpoint_at(
        py: Python<'_>,
        base: &Self,
        payload: &str,
        expected_id: &str,
        update: &Bound<'_, PyAny>,
        recorded_data_id: &str,
        post_geometry_id: &str,
    ) -> PyResult<Self> {
        reject_bool(update)?;
        let update = update.extract::<u64>()?;
        py.detach(|| {
            NativeCausalLm::from_training_checkpoint_at(
                base.inner.clone(),
                payload.as_bytes(),
                expected_id,
                update,
                recorded_data_id,
                post_geometry_id,
            )
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }

    #[staticmethod]
    fn from_training_checkpoint(
        py: Python<'_>,
        base: &Self,
        payload: &str,
        expected_id: &str,
    ) -> PyResult<Self> {
        py.detach(|| {
            NativeCausalLm::from_training_checkpoint(
                base.inner.clone(),
                payload.as_bytes(),
                expected_id,
            )
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }

    #[staticmethod]
    #[pyo3(signature = (config_json, weights, *, max_parameter_elements=None, max_activation_elements=None, max_attention_pairs=None, max_logit_elements=None, max_training_checkpoint_elements=None))]
    fn from_gpt2(
        py: Python<'_>,
        config_json: &str,
        weights: &PySafeTensorArchive,
        max_parameter_elements: Option<&Bound<'_, PyAny>>,
        max_activation_elements: Option<&Bound<'_, PyAny>>,
        max_attention_pairs: Option<&Bound<'_, PyAny>>,
        max_logit_elements: Option<&Bound<'_, PyAny>>,
        max_training_checkpoint_elements: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let defaults = CausalLmLimits::default();
        let limits = CausalLmLimits {
            max_parameter_elements: limit(max_parameter_elements, defaults.max_parameter_elements)?,
            max_activation_elements: limit(
                max_activation_elements,
                defaults.max_activation_elements,
            )?,
            max_attention_pairs: limit(max_attention_pairs, defaults.max_attention_pairs)?,
            max_logit_elements: limit(max_logit_elements, defaults.max_logit_elements)?,
            max_training_checkpoint_elements: limit(
                max_training_checkpoint_elements,
                defaults.max_training_checkpoint_elements,
            )?,
        };
        py.detach(|| NativeCausalLm::from_gpt2_json(config_json.as_bytes(), &weights.inner, limits))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[getter]
    fn execution_backend(&self) -> &'static str {
        "rust_f32_cpu_prepacked"
    }

    #[getter]
    fn base_model_identity(&self) -> &str {
        self.inner.base_model_identity()
    }

    #[getter]
    fn geometry_artifact_id(&self) -> Option<&str> {
        self.inner.geometry_artifact_id()
    }

    fn config_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.config()).map_err(error)
    }

    fn geometry_artifact(&self) -> Option<PyGeometryAdapterArtifact> {
        self.inner
            .geometry_artifact()
            .cloned()
            .map(|inner| PyGeometryAdapterArtifact { inner })
    }

    #[pyo3(signature = (*, label_smoothing=None, repetition=None))]
    fn geometry_accumulator(
        &self,
        py: Python<'_>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
        repetition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNativeCausalLmAccumulator> {
        let objective = loss_objective(label_smoothing, repetition)?;
        py.detach(|| self.inner.geometry_accumulator_with_objective(objective))
            .map(|inner| PyNativeCausalLmAccumulator { inner })
            .map_err(error)
    }

    #[pyo3(signature = (windows, shape, *, labels=None, label_smoothing=None, repetition=None))]
    fn geometry_loss_gradients(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
        repetition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNativeCausalLmGradients> {
        let objective = loss_objective(label_smoothing, repetition)?;
        let shape = self::shape(shape)?;
        self.inner
            .validate_objective_shape(shape, objective)
            .map_err(error)?;
        let count = self
            .inner
            .geometry_objective_input_count(shape, labels.is_some())
            .map_err(error)?;
        let windows = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        py.detach(|| {
            self.inner.geometry_loss_gradients_with_objective(
                &windows,
                shape,
                labels.as_deref(),
                objective,
            )
        })
        .map(|inner| PyNativeCausalLmGradients { inner })
        .map_err(error)
    }

    fn step_geometry_sgd(
        &self,
        py: Python<'_>,
        gradients: &PyNativeCausalLmGradients,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        reject_bool(learning_rate)?;
        let learning_rate = learning_rate.extract::<f32>()?;
        py.detach(|| {
            self.inner
                .step_geometry_sgd(&gradients.inner, learning_rate)
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }

    #[pyo3(signature = (gradients, optimizer, learning_rate, *, weight_decay=None))]
    fn step_geometry_adamw(
        &self,
        py: Python<'_>,
        gradients: &PyNativeCausalLmGradients,
        optimizer: &PyNativeGeometryAdamW,
        learning_rate: &Bound<'_, PyAny>,
        weight_decay: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyNativeGeometryAdamW)> {
        let step = adamw_step(learning_rate, weight_decay)?;
        py.detach(|| {
            self.inner
                .step_geometry_adamw(&gradients.inner, &optimizer.inner, step)
        })
        .map(|(inner, optimizer)| (Self { inner }, PyNativeGeometryAdamW { inner: optimizer }))
        .map_err(error)
    }

    fn with_geometry(
        &self,
        py: Python<'_>,
        artifact: &PyGeometryAdapterArtifact,
    ) -> PyResult<Self> {
        py.detach(|| self.inner.with_geometry(artifact.inner.clone()))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[pyo3(signature = (input_ids, shape, *, last_only=false))]
    fn forward(
        &self,
        py: Python<'_>,
        input_ids: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        last_only: bool,
    ) -> PyResult<PyNdTensor> {
        let shape = self::shape(shape)?;
        let count = self.inner.input_count(shape, last_only).map_err(error)?;
        let ids = tokens(input_ids, count)?;
        py.detach(|| self.inner.forward(&ids, shape, last_only))
            .map(|inner| PyNdTensor { inner })
            .map_err(error)
    }

    fn next_token_ids(
        &self,
        py: Python<'_>,
        input_ids: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<usize>> {
        let shape = self::shape(shape)?;
        let count = self.inner.input_count(shape, true).map_err(error)?;
        let ids = tokens(input_ids, count)?;
        py.detach(|| self.inner.next_token_ids(&ids, shape))
            .map_err(error)
    }

    #[pyo3(signature = (windows, shape, *, labels=None, label_smoothing=None, repetition=None))]
    fn next_token_loss(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        labels: Option<&Bound<'_, PyAny>>,
        label_smoothing: Option<&Bound<'_, PyAny>>,
        repetition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f32> {
        let objective = loss_objective(label_smoothing, repetition)?;
        let shape = self::shape(shape)?;
        self.inner
            .validate_objective_shape(shape, objective)
            .map_err(error)?;
        let count = self
            .inner
            .loss_objective_input_count(shape, labels.is_some())
            .map_err(error)?;
        let windows = tokens(windows, count)?;
        let labels = loss_labels(labels, count)?;
        py.detach(|| {
            self.inner
                .next_token_loss_with_objective(&windows, shape, labels.as_deref(), objective)
        })
        .map_err(error)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeCausalLm>()?;
    module.add_class::<PyNativeCausalLmAccumulator>()?;
    module.add_class::<PyNativeCausalLmGradients>()?;
    module.add_class::<PyNativeCausalLmLoraGradients>()?;
    evaluation::register(module)?;
    training::register(module)?;
    session::register(module)?;
    generation::register(module)?;
    context::register(module)?;
    Ok(())
}
