//! Preparation is Rust-owned too; Python never refits or reconstructs requests.
use super::*;
use st_nn::resident::{
    ByteCorpusBiasCalibration, ByteCorpusHeadwiseControls, ByteCorpusMixtureControls,
    ByteCorpusPreparedBiasStudy, ByteCorpusQueryMixtureControls,
};

#[derive(Clone)]
#[pyclass(
    name = "ByteCorpusPreparedBiasStudy",
    module = "spiraltorch.nn",
    frozen,
    skip_from_py_object
)]
pub(super) struct PyByteCorpusPreparedBiasStudy {
    inner: Arc<ByteCorpusPreparedBiasStudy>,
}

#[pymethods]
impl PyByteCorpusPreparedBiasStudy {
    #[getter]
    fn request_json(&self) -> &str {
        &self.inner.request_json
    }

    #[getter]
    fn report_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| serde_json::to_string(&self.inner.report).map_err(value_error))
    }

    fn study(&self, py: Python<'_>) -> PyResult<PyByteCorpusStudy> {
        py.detach(|| {
            Ok(PyByteCorpusStudy {
                inner: Arc::new(
                    ByteCorpusStudy::from_json(self.inner.request_json.as_bytes())
                        .map_err(value_error)?,
                ),
            })
        })
    }
}

#[pyclass(name = "ByteCorpusBiasCalibration", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteCorpusBiasCalibration {
    // CPU wheels keep the admitted recipe, but cannot measure a device.
    #[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
    inner: Arc<ByteCorpusBiasCalibration>,
}

#[pymethods]
impl PyByteCorpusBiasCalibration {
    #[staticmethod]
    #[pyo3(signature = (payload, train_batch_indices, *, fisher_rao_control=false))]
    fn from_json(
        py: Python<'_>,
        payload: &Bound<'_, PyAny>,
        train_batch_indices: &Bound<'_, PyAny>,
        fisher_rao_control: bool,
    ) -> PyResult<Self> {
        let input = json_input(payload)?;
        let indices = train_batch_indices
            .try_iter()?
            // Rust admits at most 16 selections; stop after the first excess
            // item so even an unbounded Python iterator reaches that validator.
            .take(17)
            .map(|item| index(&item?))
            .collect::<PyResult<Vec<_>>>()?;
        py.detach(move || {
            let recipe =
                ByteCorpusBiasCalibration::from_json(&input, &indices).map_err(value_error)?;
            Ok(Self {
                inner: Arc::new(if fisher_rao_control {
                    recipe.with_fisher_rao_control().map_err(value_error)?
                } else {
                    recipe
                }),
            })
        })
    }

    /// Fit gain-only controls on selected training windows, never held-out data.
    fn prepare_wgpu(&self, py: Python<'_>) -> PyResult<PyByteCorpusPreparedBiasStudy> {
        #[cfg(feature = "wgpu")]
        {
            on_gpu(py, |runtime| {
                let prepared = pollster::block_on(self.inner.prepare(runtime))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(PyByteCorpusPreparedBiasStudy {
                    inner: Arc::new(prepared),
                })
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = py;
            requires_wgpu()
        }
    }
}

#[pyclass(name = "ByteCorpusHeadwiseControls", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteCorpusHeadwiseControls {
    #[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
    inner: Arc<ByteCorpusHeadwiseControls>,
}

#[pymethods]
impl PyByteCorpusHeadwiseControls {
    #[staticmethod]
    fn from_json(py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        let input = json_input(payload)?;
        py.detach(move || {
            Ok(Self {
                inner: Arc::new(
                    ByteCorpusHeadwiseControls::from_json(&input).map_err(value_error)?,
                ),
            })
        })
    }

    fn prepare_wgpu(&self, py: Python<'_>) -> PyResult<PyByteCorpusPreparedBiasStudy> {
        #[cfg(feature = "wgpu")]
        {
            on_gpu(py, |runtime| {
                let prepared = pollster::block_on(self.inner.prepare(runtime))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(PyByteCorpusPreparedBiasStudy {
                    inner: Arc::new(prepared),
                })
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = py;
            requires_wgpu()
        }
    }
}

#[pyclass(name = "ByteCorpusMixtureControls", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteCorpusMixtureControls {
    #[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
    inner: Arc<ByteCorpusMixtureControls>,
}

#[pymethods]
impl PyByteCorpusMixtureControls {
    #[staticmethod]
    fn from_json(py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        let input = json_input(payload)?;
        py.detach(move || {
            Ok(Self {
                inner: Arc::new(ByteCorpusMixtureControls::from_json(&input).map_err(value_error)?),
            })
        })
    }

    fn prepare_wgpu(&self, py: Python<'_>) -> PyResult<PyByteCorpusPreparedBiasStudy> {
        #[cfg(feature = "wgpu")]
        {
            on_gpu(py, |runtime| {
                let prepared = pollster::block_on(self.inner.prepare(runtime))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(PyByteCorpusPreparedBiasStudy {
                    inner: Arc::new(prepared),
                })
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = py;
            requires_wgpu()
        }
    }
}

#[pyclass(
    name = "ByteCorpusQueryMixtureControls",
    module = "spiraltorch.nn",
    frozen
)]
pub(super) struct PyByteCorpusQueryMixtureControls {
    prepared: PyByteCorpusPreparedBiasStudy,
}

#[pymethods]
impl PyByteCorpusQueryMixtureControls {
    /// Rust performs this structural expansion on CPU without data fitting.
    #[staticmethod]
    fn from_json(py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        let input = json_input(payload)?;
        py.detach(move || {
            let prepared = ByteCorpusQueryMixtureControls::from_json(&input)
                .map_err(value_error)?
                .prepare();
            Ok(Self {
                prepared: PyByteCorpusPreparedBiasStudy {
                    inner: Arc::new(prepared),
                },
            })
        })
    }

    fn prepare(&self) -> PyByteCorpusPreparedBiasStudy {
        self.prepared.clone()
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyByteCorpusPreparedBiasStudy>()?;
    module.add_class::<PyByteCorpusBiasCalibration>()?;
    module.add_class::<PyByteCorpusHeadwiseControls>()?;
    module.add_class::<PyByteCorpusMixtureControls>()?;
    module.add_class::<PyByteCorpusQueryMixtureControls>()
}
