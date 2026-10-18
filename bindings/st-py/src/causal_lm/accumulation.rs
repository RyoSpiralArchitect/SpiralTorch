//! Thin transport for the Rust-owned token-weighted mean and update boundary.
use super::{error, shape, tokens, PyNativeCausalLm, PyNativeCausalLmGradients};
use crate::channel_affine::reject_bool;
use crate::geometry_runtime::{adamw_step, PyNativeGeometryAdamW};
use pyo3::prelude::*;
use st_nn::causal_lm::NativeCausalLmAccumulator;
use std::collections::BTreeMap;

#[pyclass(name = "NativeCausalLmAccumulator", module = "spiraltorch")]
pub struct PyNativeCausalLmAccumulator {
    pub(super) inner: NativeCausalLmAccumulator,
}

#[pymethods]
impl PyNativeCausalLmAccumulator {
    #[getter]
    fn base_model_identity(&self) -> &str {
        self.inner.base_model_identity()
    }
    #[getter]
    fn geometry_artifact_id(&self) -> &str {
        self.inner.geometry_artifact_id()
    }
    #[getter]
    fn target_tokens(&self) -> u64 {
        self.inner.target_tokens()
    }
    #[getter]
    fn microbatches(&self) -> u64 {
        self.inner.microbatches()
    }
    #[getter]
    fn loss(&self) -> Option<f64> {
        self.inner.loss()
    }

    fn add(&mut self, py: Python<'_>, gradients: &PyNativeCausalLmGradients) -> PyResult<()> {
        py.detach(|| self.inner.add(&gradients.inner))
            .map_err(error)
    }

    fn accumulate_loss(
        &mut self,
        py: Python<'_>,
        model: &PyNativeCausalLm,
        windows: &Bound<'_, PyAny>,
        window_shape: &Bound<'_, PyAny>,
    ) -> PyResult<f32> {
        let shape = shape(window_shape)?;
        let count = self.inner.input_count(&model.inner, shape).map_err(error)?;
        let windows = tokens(windows, count)?;
        py.detach(|| self.inner.accumulate_loss(&model.inner, &windows, shape))
            .map_err(error)
    }

    fn parameter_gradients(
        &self,
        py: Python<'_>,
    ) -> PyResult<BTreeMap<String, BTreeMap<String, Vec<f32>>>> {
        py.detach(|| self.inner.parameter_gradients())
            .map_err(error)
    }

    fn step_sgd(
        &self,
        py: Python<'_>,
        model: &PyNativeCausalLm,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<PyNativeCausalLm> {
        reject_bool(learning_rate)?;
        let rate = learning_rate.extract::<f32>()?;
        py.detach(|| self.inner.step_sgd(&model.inner, rate))
            .map(|inner| PyNativeCausalLm { inner })
            .map_err(error)
    }

    #[pyo3(signature = (model, optimizer, learning_rate, *, weight_decay=None))]
    fn step_adamw(
        &self,
        py: Python<'_>,
        model: &PyNativeCausalLm,
        optimizer: &PyNativeGeometryAdamW,
        learning_rate: &Bound<'_, PyAny>,
        weight_decay: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyNativeCausalLm, PyNativeGeometryAdamW)> {
        let step = adamw_step(learning_rate, weight_decay)?;
        py.detach(|| self.inner.step_adamw(&model.inner, &optimizer.inner, step))
            .map(|(model, optimizer)| {
                (
                    PyNativeCausalLm { inner: model },
                    PyNativeGeometryAdamW { inner: optimizer },
                )
            })
            .map_err(error)
    }
}
