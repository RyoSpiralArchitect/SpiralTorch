use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, Bound};

use julia_ffi_poc::{tempo_latency_score as rust_tempo_latency_score, ZTigerOptim};

#[pyclass(module = "spiraltorch.julia", name = "ZTigerOptim")]
#[derive(Clone)]
pub(crate) struct PyZTigerOptim {
    inner: ZTigerOptim,
}

#[pymethods]
impl PyZTigerOptim {
    #[new]
    #[pyo3(signature = (curvature=-1.0))]
    pub fn new(curvature: f64) -> Self {
        Self {
            inner: ZTigerOptim::new(curvature),
        }
    }

    #[getter]
    pub fn curvature(&self) -> f64 {
        self.inner.curvature()
    }

    #[getter]
    pub fn gain(&self) -> f64 {
        self.inner.gain()
    }

    pub fn update(&mut self, lora_pid: f64, resonance: Vec<f64>) -> f64 {
        self.inner.update(lora_pid, &resonance)
    }
}

#[pyfunction]
#[pyo3(signature = (tile, slack))]
fn tempo_latency_score(tile: u32, slack: u32) -> PyResult<f64> {
    rust_tempo_latency_score(tile, slack).map_err(|err| PyValueError::new_err(err.to_string()))
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "julia")?;
    module.add("__doc__", "Julia Z-space bridge helpers")?;
    module.add_class::<PyZTigerOptim>()?;
    module.add_function(wrap_pyfunction!(tempo_latency_score, &module)?)?;
    module.add("__all__", vec!["ZTigerOptim", "tempo_latency_score"])?;

    parent.add_submodule(&module)?;
    let module_obj = module.to_object(py);
    parent.add("julia", module_obj)?;

    let optim = module.getattr("ZTigerOptim")?;
    parent.add("ZTigerOptim", optim.to_object(py))?;
    let tempo = module.getattr("tempo_latency_score")?;
    parent.add("tempo_latency_score", tempo.to_object(py))?;

    Ok(())
}
