//! Thin owning handles for the Rust portable-adapter runtime, without Torch.

use crate::channel_affine::{dimension, reject_bool, sequence};
use crate::f32_buffer::{read_f32, write_f32};
use crate::geometry_artifact::PyGeometryAdapterArtifact;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict};
use st_nn::geometry_runtime::{NativeGeometryBatch, NativeGeometryRuntime};
use std::collections::BTreeMap;

mod optimizer;
pub(crate) use optimizer::{adamw_step, PyNativeGeometryAdamW};

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

fn rate(value: &Bound<'_, PyAny>) -> PyResult<f32> {
    reject_bool(value)?;
    value.extract()
}

#[pyclass(name = "NativeGeometryRuntime", module = "spiraltorch")]
pub struct PyNativeGeometryRuntime {
    inner: NativeGeometryRuntime,
}

#[pyclass(name = "NativeGeometryBatch", module = "spiraltorch", frozen)]
pub struct PyNativeGeometryBatch {
    inner: NativeGeometryBatch,
}

impl PyNativeGeometryRuntime {
    fn shape(&self, path: &str, rows: &Bound<'_, PyAny>) -> PyResult<(usize, usize, usize)> {
        let rows = dimension(rows)?;
        let (features, budget) = self.inner.dimensions(path).map_err(error)?;
        let count = rows
            .checked_mul(features)
            .filter(|&n| n <= budget)
            .ok_or_else(|| error("native geometry shape exceeds value budget"))?;
        Ok((rows, count, budget))
    }

    fn context_shape(&self, path: &str, shape: &Bound<'_, PyAny>) -> PyResult<([usize; 2], usize)> {
        if shape.len()? != 2 {
            return Err(error(
                "sequence shape must be [batch, sequence]; feature width comes from the artifact",
            ));
        }
        let shape = [
            dimension(&shape.get_item(0)?)?,
            dimension(&shape.get_item(1)?)?,
        ];
        let (_, count) = self.inner.sequence_dimensions(path, shape).map_err(error)?;
        Ok((shape, count))
    }
}

#[pymethods]
impl PyNativeGeometryRuntime {
    #[new]
    fn new(artifact: &PyGeometryAdapterArtifact) -> Self {
        Self {
            inner: NativeGeometryRuntime::new(artifact.inner.clone()),
        }
    }

    #[getter]
    fn execution_backend(&self) -> &'static str {
        "rust_f32_cpu"
    }

    fn artifact(&self) -> PyGeometryAdapterArtifact {
        PyGeometryAdapterArtifact {
            inner: self.inner.artifact().clone(),
        }
    }

    fn forward(
        &self,
        py: Python<'_>,
        path: &str,
        input: &Bound<'_, PyAny>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<PyNativeGeometryBatch> {
        let (rows, count, budget) = self.shape(path, rows)?;
        let input = sequence(input, budget, count)?;
        py.detach(|| self.inner.forward(path, &input, rows))
            .map(|inner| PyNativeGeometryBatch { inner })
            .map_err(error)
    }

    fn forward_buffer(
        &self,
        py: Python<'_>,
        path: &str,
        input: &Bound<'_, PyAny>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<PyNativeGeometryBatch> {
        let (rows, count, budget) = self.shape(path, rows)?;
        let input = read_f32(input, budget, Some(count))?;
        py.detach(|| self.inner.forward(path, &input, rows))
            .map(|inner| PyNativeGeometryBatch { inner })
            .map_err(error)
    }

    fn forward_sequence(
        &self,
        py: Python<'_>,
        path: &str,
        input: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
    ) -> PyResult<PyNativeGeometryBatch> {
        let (shape, count) = self.context_shape(path, shape)?;
        let input = sequence(input, count, count)?;
        py.detach(|| self.inner.forward_sequence(path, &input, shape))
            .map(|inner| PyNativeGeometryBatch { inner })
            .map_err(error)
    }

    fn forward_sequence_buffer(
        &self,
        py: Python<'_>,
        path: &str,
        input: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
    ) -> PyResult<PyNativeGeometryBatch> {
        let (shape, count) = self.context_shape(path, shape)?;
        let input = read_f32(input, count, Some(count))?;
        py.detach(|| self.inner.forward_sequence(path, &input, shape))
            .map(|inner| PyNativeGeometryBatch { inner })
            .map_err(error)
    }

    fn step_sgd(
        &mut self,
        py: Python<'_>,
        batch: &PyNativeGeometryBatch,
        upstream: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<String> {
        let count = batch.inner.output().data().len();
        let upstream = sequence(upstream, count, count)?;
        let learning_rate = rate(learning_rate)?;
        py.detach(|| self.inner.step_sgd(&batch.inner, &upstream, learning_rate))
            .map_err(error)?;
        Ok(self.inner.artifact().artifact_id().into())
    }

    #[pyo3(signature = (batch, upstream, optimizer, learning_rate, *, weight_decay=None))]
    fn step_adamw(
        &self,
        py: Python<'_>,
        batch: &PyNativeGeometryBatch,
        upstream: &Bound<'_, PyAny>,
        optimizer: &PyNativeGeometryAdamW,
        learning_rate: &Bound<'_, PyAny>,
        weight_decay: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyNativeGeometryAdamW)> {
        let step = adamw_step(learning_rate, weight_decay)?;
        let count = batch.inner.output().data().len();
        let upstream = sequence(upstream, count, count)?;
        py.detach(|| {
            self.inner
                .step_adamw(&batch.inner, &upstream, &optimizer.inner, step)
        })
        .map(|(inner, optimizer)| (Self { inner }, PyNativeGeometryAdamW { inner: optimizer }))
        .map_err(error)
    }

    #[pyo3(signature = (batch, upstream, optimizer, learning_rate, *, weight_decay=None))]
    fn step_adamw_buffer(
        &self,
        py: Python<'_>,
        batch: &PyNativeGeometryBatch,
        upstream: &Bound<'_, PyAny>,
        optimizer: &PyNativeGeometryAdamW,
        learning_rate: &Bound<'_, PyAny>,
        weight_decay: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyNativeGeometryAdamW)> {
        let step = adamw_step(learning_rate, weight_decay)?;
        let count = batch.inner.output().data().len();
        let upstream = read_f32(upstream, count, Some(count))?;
        py.detach(|| {
            self.inner
                .step_adamw(&batch.inner, &upstream, &optimizer.inner, step)
        })
        .map(|(inner, optimizer)| (Self { inner }, PyNativeGeometryAdamW { inner: optimizer }))
        .map_err(error)
    }

    fn train_mse(
        &mut self,
        py: Python<'_>,
        path: &str,
        input: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
        rows: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<f32> {
        let (rows, count, budget) = self.shape(path, rows)?;
        let input = sequence(input, budget, count)?;
        let target = sequence(target, budget, count)?;
        let learning_rate = rate(learning_rate)?;
        py.detach(|| {
            self.inner
                .train_mse(path, &input, &target, rows, learning_rate)
        })
        .map_err(error)
    }

    fn train_mse_buffer(
        &mut self,
        py: Python<'_>,
        path: &str,
        input: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
        rows: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<f32> {
        let (rows, count, budget) = self.shape(path, rows)?;
        let input = read_f32(input, budget, Some(count))?;
        let target = read_f32(target, budget, Some(count))?;
        let learning_rate = rate(learning_rate)?;
        py.detach(|| {
            self.inner
                .train_mse(path, &input, &target, rows, learning_rate)
        })
        .map_err(error)
    }

    fn train_sequence_mse(
        &mut self,
        py: Python<'_>,
        path: &str,
        input: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<f32> {
        let (shape, count) = self.context_shape(path, shape)?;
        let input = sequence(input, count, count)?;
        let target = sequence(target, count, count)?;
        let learning_rate = rate(learning_rate)?;
        py.detach(|| {
            self.inner
                .train_sequence_mse(path, &input, &target, shape, learning_rate)
        })
        .map_err(error)
    }

    fn train_sequence_mse_buffer(
        &mut self,
        py: Python<'_>,
        path: &str,
        input: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
        shape: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
    ) -> PyResult<f32> {
        let (shape, count) = self.context_shape(path, shape)?;
        let input = read_f32(input, count, Some(count))?;
        let target = read_f32(target, count, Some(count))?;
        let learning_rate = rate(learning_rate)?;
        py.detach(|| {
            self.inner
                .train_sequence_mse(path, &input, &target, shape, learning_rate)
        })
        .map_err(error)
    }
}

#[pymethods]
impl PyNativeGeometryBatch {
    #[getter]
    fn output(&self) -> Vec<f32> {
        self.inner.output().data().to_vec()
    }
    #[getter]
    fn source_artifact_id(&self) -> &str {
        self.inner.source_artifact_id()
    }
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.output().shape()
    }

    #[getter]
    fn sequence_shape(&self) -> Option<(usize, usize, usize)> {
        self.inner
            .sequence_shape()
            .map(|[batch, steps, features]| (batch, steps, features))
    }

    fn output_buffer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyByteArray>> {
        write_f32(py, self.inner.output().data().iter().copied())
    }

    #[allow(clippy::type_complexity)]
    fn vjp(
        &self,
        py: Python<'_>,
        upstream: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<f32>, BTreeMap<String, Vec<f32>>)> {
        let count = self.inner.output().data().len();
        let upstream = sequence(upstream, count, count)?;
        let result = py.detach(|| self.inner.vjp(&upstream)).map_err(error)?;
        Ok((result.input.data().to_vec(), result.parameters))
    }

    fn vjp_buffer<'py>(
        &self,
        py: Python<'py>,
        upstream: &Bound<'_, PyAny>,
    ) -> PyResult<(Bound<'py, PyByteArray>, Bound<'py, PyDict>)> {
        let count = self.inner.output().data().len();
        let upstream = read_f32(upstream, count, Some(count))?;
        let result = py.detach(|| self.inner.vjp(&upstream)).map_err(error)?;
        let parameters = PyDict::new(py);
        for (name, values) in result.parameters {
            parameters.set_item(name, write_f32(py, values.into_iter())?)?;
        }
        Ok((
            write_f32(py, result.input.data().iter().copied())?,
            parameters,
        ))
    }

    fn vjp_parameters(
        &self,
        py: Python<'_>,
        upstream: &Bound<'_, PyAny>,
    ) -> PyResult<BTreeMap<String, Vec<f32>>> {
        let count = self.inner.output().data().len();
        let upstream = sequence(upstream, count, count)?;
        py.detach(|| self.inner.vjp_parameters(&upstream))
            .map_err(error)
    }

    fn vjp_parameters_buffer<'py>(
        &self,
        py: Python<'py>,
        upstream: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let count = self.inner.output().data().len();
        let upstream = read_f32(upstream, count, Some(count))?;
        let result = py
            .detach(|| self.inner.vjp_parameters(&upstream))
            .map_err(error)?;
        let parameters = PyDict::new(py);
        for (name, values) in result {
            parameters.set_item(name, write_f32(py, values.into_iter())?)?;
        }
        Ok(parameters)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeGeometryAdamW>()?;
    module.add_class::<PyNativeGeometryRuntime>()?;
    module.add_class::<PyNativeGeometryBatch>()
}
