//! Token transport only; model loading, execution, loss and geometry stay in Rust.
use crate::channel_affine::{dimension, reject_bool};
use crate::geometry_artifact::PyGeometryAdapterArtifact;
use crate::geometry_runtime::{adamw_step, PyNativeGeometryAdamW};
use crate::weights::{PyNdTensor, PySafeTensorArchive};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use st_nn::causal_lm::{CausalLmLimits, NativeCausalLm, NativeCausalLmGradients};
use std::collections::BTreeMap;

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
    fn loss(&self) -> f32 {
        self.inner.loss()
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
    fn parameter_gradients(&self) -> BTreeMap<String, BTreeMap<String, Vec<f32>>> {
        self.inner.parameters().clone()
    }
}

#[pymethods]
impl PyNativeCausalLm {
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

    fn geometry_loss_gradients(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
    ) -> PyResult<PyNativeCausalLmGradients> {
        let shape = self::shape(shape)?;
        let count = self
            .inner
            .geometry_gradient_input_count(shape)
            .map_err(error)?;
        let windows = tokens(windows, count)?;
        py.detach(|| self.inner.geometry_loss_gradients(&windows, shape))
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

    fn next_token_loss(
        &self,
        py: Python<'_>,
        windows: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
    ) -> PyResult<f32> {
        let shape = self::shape(shape)?;
        let steps = shape[1]
            .checked_sub(1)
            .filter(|&n| n > 0)
            .ok_or_else(|| error("next-token loss needs at least two IDs per window"))?;
        self.inner
            .input_count([shape[0], steps], false)
            .map_err(error)?;
        let count = shape[0]
            .checked_mul(shape[1])
            .ok_or_else(|| error("window shape overflow"))?;
        let windows = tokens(windows, count)?;
        py.detach(|| self.inner.next_token_loss(&windows, shape))
            .map_err(error)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeCausalLm>()?;
    module.add_class::<PyNativeCausalLmGradients>()?;
    Ok(())
}
