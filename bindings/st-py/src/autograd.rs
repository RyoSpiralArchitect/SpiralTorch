use crate::json::json_to_py;
use crate::tensor::{tensor_err_to_py, PyTensor};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use st_tensor::{
    AutogradBackwardReport, AutogradGraphSummary, AutogradTensor, AUTOGRAD_CONTRACT_VERSION,
    AUTOGRAD_SEMANTIC_OWNER,
};

fn graph_summary_to_py(py: Python<'_>, summary: AutogradGraphSummary) -> PyResult<PyObject> {
    json_to_py(py, &summary.contract_payload())
}

fn backward_report_to_py(py: Python<'_>, report: AutogradBackwardReport) -> PyResult<PyObject> {
    json_to_py(py, &report.contract_payload())
}

/// Thin Python handle over the Rust-owned reverse-mode graph.
#[pyclass(module = "spiraltorch", name = "AutogradTensor")]
#[derive(Clone)]
pub(crate) struct PyAutogradTensor {
    inner: AutogradTensor,
}

impl PyAutogradTensor {
    fn from_inner(inner: AutogradTensor) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAutogradTensor {
    #[new]
    #[pyo3(signature = (value, requires_grad=true))]
    fn new(value: &PyTensor, requires_grad: bool) -> PyResult<Self> {
        AutogradTensor::from_tensor(value.inner.clone(), requires_grad)
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    #[staticmethod]
    fn variable(value: &PyTensor) -> PyResult<Self> {
        AutogradTensor::variable(value.inner.clone())
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    #[staticmethod]
    fn constant(value: &PyTensor) -> PyResult<Self> {
        AutogradTensor::constant(value.inner.clone())
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn id(&self) -> u64 {
        self.inner.id()
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    fn operation_name(&self) -> &'static str {
        self.inner.operation_name()
    }

    fn value(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.value().clone())
    }

    fn grad(&self) -> Option<PyTensor> {
        self.inner.grad().map(PyTensor::from_tensor)
    }

    fn detach(&self) -> PyResult<Self> {
        self.inner
            .detach()
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn zero_grad_graph(&self) {
        self.inner.zero_grad_graph();
    }

    fn add(&self, rhs: &Self) -> PyResult<Self> {
        self.inner
            .add(&rhs.inner)
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn sub(&self, rhs: &Self) -> PyResult<Self> {
        self.inner
            .sub(&rhs.inner)
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn hadamard(&self, rhs: &Self) -> PyResult<Self> {
        self.inner
            .hadamard(&rhs.inner)
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        self.inner
            .matmul(&rhs.inner)
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn scale(&self, factor: f32) -> PyResult<Self> {
        self.inner
            .scale(factor)
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn transpose(&self) -> PyResult<Self> {
        self.inner
            .transpose()
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn sum(&self) -> PyResult<Self> {
        self.inner
            .sum()
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn mean(&self) -> PyResult<Self> {
        self.inner
            .mean()
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn dot(&self, rhs: &Self) -> PyResult<Self> {
        self.inner
            .dot(&rhs.inner)
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn mean_squared_error(&self, target: &Self) -> PyResult<Self> {
        self.inner
            .mean_squared_error(&target.inner)
            .map(Self::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn item(&self) -> PyResult<f32> {
        self.inner.item_f32().map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (seed=None))]
    fn backward(&self, py: Python<'_>, seed: Option<&PyTensor>) -> PyResult<PyObject> {
        let report = match seed {
            Some(seed) => self.inner.backward_with_grad(&seed.inner),
            None => self.inner.backward(),
        }
        .map_err(tensor_err_to_py)?;
        backward_report_to_py(py, report)
    }

    fn graph_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        graph_summary_to_py(py, self.inner.graph_summary())
    }

    fn __add__(&self, rhs: &Self) -> PyResult<Self> {
        self.add(rhs)
    }

    fn __sub__(&self, rhs: &Self) -> PyResult<Self> {
        self.sub(rhs)
    }

    fn __matmul__(&self, rhs: &Self) -> PyResult<Self> {
        self.matmul(rhs)
    }

    fn __repr__(&self) -> String {
        format!(
            "AutogradTensor(id={}, shape={:?}, requires_grad={}, operation='{}')",
            self.inner.id(),
            self.inner.shape(),
            self.inner.requires_grad(),
            self.inner.operation_name()
        )
    }
}

pub(crate) fn register(_py: Python<'_>, module: &Bound<PyModule>) -> PyResult<()> {
    module.add_class::<PyAutogradTensor>()?;
    module.add("AUTOGRAD_CONTRACT_VERSION", AUTOGRAD_CONTRACT_VERSION)?;
    module.add("AUTOGRAD_SEMANTIC_OWNER", AUTOGRAD_SEMANTIC_OWNER)?;
    Ok(())
}
