use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Bound;

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
}

pub(crate) fn register(_py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_class::<PyModuleTrainer>()?;
    Ok(())
}
