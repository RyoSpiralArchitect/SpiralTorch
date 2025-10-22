use crate::tensor::PyTensor;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use pyo3::Py;
use pyo3::PyAny;

use st_tensor::{mean_squared_error, LinearModel, Tensor};

#[pyclass(module = "spiraltorch", name = "ModuleTrainer")]
pub(crate) struct PyModuleTrainer {
    model: LinearModel,
    input_dim: usize,
    output_dim: usize,
}

#[pymethods]
impl PyModuleTrainer {
    #[new]
    pub fn new(input_dim: usize, output_dim: usize) -> PyResult<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(PyValueError::new_err(
                "input_dim and output_dim must be positive",
            ));
        }
        let model = LinearModel::new(input_dim, output_dim)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            model,
            input_dim,
            output_dim,
        })
    }

    /// 1 epoch 学習。inputs/targets は [[f32]] (row-major) を想定。
    #[pyo3(signature = (inputs, targets, learning_rate=1e-2, batch_size=32))]
    pub fn train_epoch(
        &mut self,
        inputs: Vec<Vec<f32>>,
        targets: Vec<Vec<f32>>,
        learning_rate: f32,
        batch_size: usize,
    ) -> PyResult<f32> {
        if inputs.len() != targets.len() {
            return Err(PyValueError::new_err("inputs/targets length mismatch"));
        }
        if inputs.is_empty() {
            return Ok(0.0);
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be positive"));
        }
        for row in &inputs {
            if row.len() != self.input_dim {
                return Err(PyValueError::new_err("input row dimension mismatch"));
            }
        }
        for row in &targets {
            if row.len() != self.output_dim {
                return Err(PyValueError::new_err("target row dimension mismatch"));
            }
        }

        let mut avg_loss = 0.0f32;
        let mut seen = 0usize;
        let mut start = 0usize;
        let total = inputs.len();
        while start < total {
            let end = (start + batch_size).min(total);
            let batch_rows = end - start;
            let x: Vec<f32> = inputs[start..end]
                .iter()
                .flat_map(|r| r.iter().copied())
                .collect();
            let y: Vec<f32> = targets[start..end]
                .iter()
                .flat_map(|r| r.iter().copied())
                .collect();
            let x = Tensor::from_vec(batch_rows, self.input_dim, x)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let y = Tensor::from_vec(batch_rows, self.output_dim, y)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let loss = self
                .model
                .train_batch(&x, &y, learning_rate)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            avg_loss += loss * batch_rows as f32;
            seen += batch_rows;
            start = end;
        }
        Ok(avg_loss / seen as f32)
    }

    /// MSE を返す
    pub fn evaluate(&self, inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) -> PyResult<f32> {
        if inputs.len() != targets.len() {
            return Err(PyValueError::new_err("inputs/targets length mismatch"));
        }
        if inputs.is_empty() {
            return Err(PyValueError::new_err("empty inputs"));
        }
        for row in &inputs {
            if row.len() != self.input_dim {
                return Err(PyValueError::new_err("input row dimension mismatch"));
            }
        }
        for row in &targets {
            if row.len() != self.output_dim {
                return Err(PyValueError::new_err("target row dimension mismatch"));
            }
        }
        let rows = inputs.len();
        let x: Vec<f32> = inputs.iter().flat_map(|r| r.iter().copied()).collect();
        let y: Vec<f32> = targets.iter().flat_map(|r| r.iter().copied()).collect();
        let x = Tensor::from_vec(rows, self.input_dim, x)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let y = Tensor::from_vec(rows, self.output_dim, y)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let predictions = self
            .model
            .forward(&x)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        mean_squared_error(&predictions, &y).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Convenience helper to run inference from nested Python lists.
    pub fn predict(&self, inputs: Vec<Vec<f32>>) -> PyResult<PyTensor> {
        let tensor_inputs = lists_to_tensor(&inputs, self.input_dim, "inputs")?;
        let predictions = self
            .model
            .forward(&tensor_inputs)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor::from_tensor(predictions))
    }

    /// Run inference using an existing Tensor without additional allocations.
    pub fn predict_tensor(&self, inputs: &PyTensor) -> PyResult<PyTensor> {
        let predictions = self
            .model
            .forward(&inputs.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor::from_tensor(predictions))
    }

    /// Return a snapshot of the current model weights.
    pub fn weights(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.model.weights().clone()))
    }

    /// Return a copy of the current bias vector.
    pub fn bias(&self) -> Vec<f32> {
        self.model.bias().to_vec()
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

#[pyclass(module = "spiraltorch", name = "LinearModel")]
pub(crate) struct PyLinearModel {
    inner: LinearModel,
}

#[pymethods]
impl PyLinearModel {
    #[new]
    pub fn new(input_dim: usize, output_dim: usize) -> PyResult<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(PyValueError::new_err(
                "input_dim and output_dim must be positive",
            ));
        }
        let inner = LinearModel::new(input_dim, output_dim)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (inputs))]
    pub fn forward(&self, inputs: &Bound<PyAny>) -> PyResult<PyTensor> {
        let py = inputs.py();

        if let Ok(py_tensor) = inputs.extract::<Py<PyTensor>>() {
            let tensor = {
                let borrow = py_tensor.bind(py).borrow();
                self.inner
                    .forward(&borrow.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            };
            return Ok(PyTensor::from_tensor(tensor));
        }

        if let Ok(list_inputs) = inputs.extract::<Vec<Vec<f32>>>() {
            let (input_dim, _) = self.inner.weights().shape();
            let tensor_inputs = lists_to_tensor(&list_inputs, input_dim, "inputs")?;
            let tensor = self
                .inner
                .forward(&tensor_inputs)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(PyTensor::from_tensor(tensor));
        }

        Err(PyTypeError::new_err(
            "inputs must be a Tensor or a sequence of sequences of floats",
        ))
    }

    #[pyo3(signature = (inputs, targets, learning_rate=1e-2))]
    pub fn train_batch(
        &mut self,
        inputs: Vec<Vec<f32>>,
        targets: Vec<Vec<f32>>,
        learning_rate: f32,
    ) -> PyResult<f32> {
        if inputs.len() != targets.len() {
            return Err(PyValueError::new_err("inputs/targets length mismatch"));
        }
        if inputs.is_empty() {
            return Err(PyValueError::new_err("inputs cannot be empty"));
        }
        let (input_dim, output_dim) = self.inner.weights().shape();
        let tensor_inputs = lists_to_tensor(&inputs, input_dim, "inputs")?;
        let tensor_targets = lists_to_tensor(&targets, output_dim, "targets")?;
        self.inner
            .train_batch(&tensor_inputs, &tensor_targets, learning_rate)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (inputs, targets, learning_rate=1e-2))]
    pub fn train_batch_tensor(
        &mut self,
        inputs: &PyTensor,
        targets: &PyTensor,
        learning_rate: f32,
    ) -> PyResult<f32> {
        let (batch_rows, input_dim) = inputs.inner.shape();
        let (target_rows, target_dim) = targets.inner.shape();
        let (expected_input, expected_output) = self.inner.weights().shape();
        if input_dim != expected_input {
            return Err(PyValueError::new_err(format!(
                "input tensor expected {} columns, got {}",
                expected_input, input_dim
            )));
        }
        if target_dim != expected_output {
            return Err(PyValueError::new_err(format!(
                "target tensor expected {} columns, got {}",
                expected_output, target_dim
            )));
        }
        if batch_rows != target_rows {
            return Err(PyValueError::new_err("inputs/targets length mismatch"));
        }
        self.inner
            .train_batch(&inputs.inner, &targets.inner, learning_rate)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn weights(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.weights().clone()))
    }

    pub fn bias(&self) -> Vec<f32> {
        self.inner.bias().to_vec()
    }

    pub fn input_dim(&self) -> usize {
        self.inner.weights().shape().0
    }

    pub fn output_dim(&self) -> usize {
        self.inner.weights().shape().1
    }

    pub fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        let weights = Py::new(py, PyTensor::from_tensor(self.inner.weights().clone()))?;
        dict.set_item("weights", weights)?;
        dict.set_item("bias", self.inner.bias().to_vec())?;
        Ok(dict.into_py(py))
    }
}

#[pyfunction]
#[pyo3(name = "mean_squared_error")]
fn mean_squared_error_py(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f32> {
    mean_squared_error(&predictions.inner, &targets.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_class::<PyLinearModel>()?;
    parent.add_class::<PyModuleTrainer>()?;
    parent.add_function(wrap_pyfunction!(mean_squared_error_py, parent)?)?;
    let _ = py;
    Ok(())
}

fn lists_to_tensor(rows: &[Vec<f32>], expected_cols: usize, label: &str) -> PyResult<Tensor> {
    if rows.is_empty() {
        return Err(PyValueError::new_err(format!("{label} cannot be empty")));
    }
    for (idx, row) in rows.iter().enumerate() {
        if row.len() != expected_cols {
            return Err(PyValueError::new_err(format!(
                "{label} row {idx} expected {expected_cols} values, got {}",
                row.len()
            )));
        }
    }
    let mut flat = Vec::with_capacity(rows.len() * expected_cols);
    for row in rows {
        flat.extend_from_slice(row);
    }
    Tensor::from_vec(rows.len(), expected_cols, flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
