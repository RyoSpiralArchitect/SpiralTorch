use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

#[cfg(feature = "rl")]
use crate::tensor::tensor_err_to_py;
#[cfg(feature = "rl")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "rl")]
use st_rl::{DqnAgent, PpoAgent, SacAgent, SpiralRlError};

#[cfg(feature = "rl")]
fn rl_err_to_py(err: SpiralRlError) -> PyErr {
    match err {
        SpiralRlError::Tensor(err) => tensor_err_to_py(err),
        SpiralRlError::EmptyEpisode
        | SpiralRlError::InvalidStateShape { .. }
        | SpiralRlError::InvalidAction { .. }
        | SpiralRlError::InvalidDiscount { .. } => PyValueError::new_err(err.to_string()),
    }
}

#[cfg(feature = "rl")]
#[pyclass(module = "spiraltorch.rl")]
pub(crate) struct PyDqnAgent {
    inner: DqnAgent,
}

#[cfg(feature = "rl")]
#[pymethods]
impl PyDqnAgent {
    #[new]
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        discount: f32,
        learning_rate: f32,
    ) -> PyResult<Self> {
        let inner =
            DqnAgent::new(state_dim, action_dim, discount, learning_rate).map_err(rl_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn select_action(&self, state: usize) -> usize {
        self.inner.select_action(state)
    }

    pub fn update(&mut self, state: usize, action: usize, reward: f32, next_state: usize) {
        self.inner.update(state, action, reward, next_state);
    }

    #[getter]
    pub fn epsilon(&self) -> f32 {
        self.inner.epsilon()
    }

    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.inner.set_epsilon(epsilon);
    }
}

#[cfg(feature = "rl")]
#[pyclass(module = "spiraltorch.rl")]
pub(crate) struct PyPpoAgent {
    inner: PpoAgent,
    state_dim: usize,
}

#[cfg(feature = "rl")]
#[pymethods]
impl PyPpoAgent {
    #[new]
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        learning_rate: f32,
        clip_range: f32,
    ) -> PyResult<Self> {
        let inner = PpoAgent::new(state_dim, action_dim, learning_rate, clip_range)
            .map_err(rl_err_to_py)?;
        Ok(Self { inner, state_dim })
    }

    pub fn score_actions(&self, state: Vec<f32>) -> PyResult<Vec<f32>> {
        if state.len() != self.state_dim {
            return Err(PyValueError::new_err(format!(
                "state length {} does not match configured dimension {}",
                state.len(),
                self.state_dim
            )));
        }
        Ok(self.inner.score_actions(&state))
    }

    pub fn value(&self, state: Vec<f32>) -> PyResult<f32> {
        if state.len() != self.state_dim {
            return Err(PyValueError::new_err(format!(
                "state length {} does not match configured dimension {}",
                state.len(),
                self.state_dim
            )));
        }
        Ok(self.inner.value(&state))
    }

    pub fn update(
        &mut self,
        state: Vec<f32>,
        action: usize,
        advantage: f32,
        old_log_prob: f32,
    ) -> PyResult<()> {
        if state.len() != self.state_dim {
            return Err(PyValueError::new_err(format!(
                "state length {} does not match configured dimension {}",
                state.len(),
                self.state_dim
            )));
        }
        self.inner.update(&state, action, advantage, old_log_prob);
        Ok(())
    }
}

#[cfg(feature = "rl")]
#[pyclass(module = "spiraltorch.rl")]
pub(crate) struct PySacAgent {
    inner: SacAgent,
    state_dim: usize,
}

#[cfg(feature = "rl")]
#[pymethods]
impl PySacAgent {
    #[new]
    pub fn new(state_dim: usize, action_dim: usize, temperature: f32) -> PyResult<Self> {
        let inner = SacAgent::new(state_dim, action_dim, temperature).map_err(rl_err_to_py)?;
        Ok(Self { inner, state_dim })
    }

    pub fn sample_action(&self, state: Vec<f32>) -> PyResult<usize> {
        if state.len() != self.state_dim {
            return Err(PyValueError::new_err(format!(
                "state length {} does not match configured dimension {}",
                state.len(),
                self.state_dim
            )));
        }
        Ok(self.inner.sample_action(&state))
    }

    pub fn jitter(&mut self, entropy_target: f32) {
        self.inner.jitter(entropy_target);
    }
}

#[cfg(feature = "rl")]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "rl")?;
    module.add("__doc__", "SpiralTorch reinforcement learning agents")?;
    module.add_class::<PyDqnAgent>()?;
    module.add_class::<PyPpoAgent>()?;
    module.add_class::<PySacAgent>()?;
    module.add("__all__", vec!["DqnAgent", "PpoAgent", "SacAgent"])?;
    parent.add_submodule(&module)?;
    parent.add("DqnAgent", module.getattr("DqnAgent")?)?;
    parent.add("PpoAgent", module.getattr("PpoAgent")?)?;
    parent.add("SacAgent", module.getattr("SacAgent")?)?;
    Ok(())
}

#[cfg(not(feature = "rl"))]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "rl")?;
    module.add("__doc__", "SpiralTorch reinforcement learning agents")?;
    parent.add_submodule(&module)?;
    Ok(())
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    register_impl(py, parent)
}
