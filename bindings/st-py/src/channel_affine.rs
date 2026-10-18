use crate::f32_buffer::{read_f32, write_f32};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray};
use st_nn::{ChannelAffineKernel, ChannelAffineLearningBatch};

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

pub(crate) fn reject_bool(value: &Bound<'_, PyAny>) -> PyResult<()> {
    let boolean_dtype = value
        .getattr("dtype")
        .and_then(|dtype| dtype.str()?.extract::<String>())
        .is_ok_and(|dtype| dtype == "bool" || dtype == "torch.bool");
    if value.is_instance_of::<PyBool>() || boolean_dtype {
        return Err(PyTypeError::new_err(
            "boolean values are not numeric controls",
        ));
    }
    Ok(())
}

pub(crate) fn dimension(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    reject_bool(value)?;
    Ok(value.extract::<u32>()? as usize)
}

pub(crate) fn sequence(
    value: &Bound<'_, PyAny>,
    budget: usize,
    expected: usize,
) -> PyResult<Vec<f32>> {
    let count = value.len()?;
    if count > budget || count != expected {
        return Err(error(
            "float32 sequence exceeds budget or differs from shape",
        ));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(error)?;
    for i in 0..count {
        let item = value.get_item(i)?;
        reject_bool(&item)?;
        result.push(item.extract()?);
    }
    Ok(result)
}

#[pyclass(name = "ChannelAffineKernel", module = "spiraltorch", frozen)]
pub struct PyChannelAffineKernel {
    inner: ChannelAffineKernel,
}

#[pyclass(name = "ChannelAffineLearningBatch", module = "spiraltorch", frozen)]
pub struct PyChannelAffineLearningBatch {
    inner: ChannelAffineLearningBatch,
}

#[pymethods]
impl PyChannelAffineKernel {
    #[new]
    #[pyo3(signature = (*, max_values=None))]
    fn new(max_values: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let budget = max_values.map(dimension).transpose()?.unwrap_or(1_048_576);
        Ok(Self {
            inner: ChannelAffineKernel::new(budget).map_err(error)?,
        })
    }

    #[getter]
    fn max_values(&self) -> usize {
        self.inner.max_values()
    }

    #[getter]
    fn execution_backend(&self) -> &'static str {
        "rust_f32_cpu"
    }

    fn forward(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        gate: &Bound<'_, PyAny>,
        bias: &Bound<'_, PyAny>,
        rows: &Bound<'_, PyAny>,
        features: &Bound<'_, PyAny>,
    ) -> PyResult<PyChannelAffineLearningBatch> {
        let (rows, features) = (dimension(rows)?, dimension(features)?);
        let count = rows
            .checked_mul(features)
            .ok_or_else(|| error("shape overflow"))?;
        let input = sequence(input, self.max_values(), count)?;
        let gate = sequence(gate, self.max_values(), features)?;
        let bias = sequence(bias, self.max_values(), features)?;
        py.detach(|| self.inner.forward(&input, &gate, &bias, rows, features))
            .map(|inner| PyChannelAffineLearningBatch { inner })
            .map_err(error)
    }

    fn forward_buffer(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        gate: &Bound<'_, PyAny>,
        bias: &Bound<'_, PyAny>,
        rows: &Bound<'_, PyAny>,
        features: &Bound<'_, PyAny>,
    ) -> PyResult<PyChannelAffineLearningBatch> {
        let (rows, features) = (dimension(rows)?, dimension(features)?);
        let count = rows
            .checked_mul(features)
            .ok_or_else(|| error("shape overflow"))?;
        let input = read_f32(input, self.max_values(), Some(count))?;
        let gate = read_f32(gate, self.max_values(), Some(features))?;
        let bias = read_f32(bias, self.max_values(), Some(features))?;
        py.detach(|| self.inner.forward(&input, &gate, &bias, rows, features))
            .map(|inner| PyChannelAffineLearningBatch { inner })
            .map_err(error)
    }
}

#[pymethods]
impl PyChannelAffineLearningBatch {
    #[getter]
    fn output(&self) -> Vec<f32> {
        self.inner.output().data().to_vec()
    }

    fn output_buffer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyByteArray>> {
        write_f32(py, self.inner.output().data().iter().copied())
    }

    fn vjp(
        &self,
        py: Python<'_>,
        upstream: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let count = self.inner.output().data().len();
        let upstream = sequence(upstream, count, count)?;
        py.detach(|| self.inner.vjp(&upstream))
            .map(|g| {
                (
                    g.grad_input.data().to_vec(),
                    g.grad_gate.data().to_vec(),
                    g.grad_bias.data().to_vec(),
                )
            })
            .map_err(error)
    }

    #[allow(clippy::type_complexity)]
    fn vjp_buffer<'py>(
        &self,
        py: Python<'py>,
        upstream: &Bound<'_, PyAny>,
    ) -> PyResult<(
        Bound<'py, PyByteArray>,
        Bound<'py, PyByteArray>,
        Bound<'py, PyByteArray>,
    )> {
        let count = self.inner.output().data().len();
        let upstream = read_f32(upstream, count, Some(count))?;
        let g = py.detach(|| self.inner.vjp(&upstream)).map_err(error)?;
        Ok((
            write_f32(py, g.grad_input.data().iter().copied())?,
            write_f32(py, g.grad_gate.data().iter().copied())?,
            write_f32(py, g.grad_bias.data().iter().copied())?,
        ))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyChannelAffineKernel>()?;
    module.add_class::<PyChannelAffineLearningBatch>()
}
