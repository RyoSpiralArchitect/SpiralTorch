//! Input collection only; Rust owns operation names, layout checks and kernels.
use super::*;
#[cfg(feature = "wgpu")]
use st_backend_wgpu::resident_tensor::pointwise::{PointwiseInputs, PointwisePlan};
#[cfg(feature = "wgpu")]
use st_tensor::{PointwiseChain, PointwiseStep};

#[pyclass(name = "WgpuPointwiseInputs", module = "spiraltorch.wgpu")]
struct PyPointwiseInputs {
    #[cfg(feature = "wgpu")]
    inner: PointwiseInputs,
}

#[pymethods]
impl PyPointwiseInputs {
    #[new]
    fn new() -> PyResult<Self> {
        #[cfg(feature = "wgpu")]
        {
            Ok(Self {
                inner: PointwiseInputs::new(),
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "WgpuPointwiseInputs requires the 'wgpu' feature",
            ))
        }
    }
    #[cfg(feature = "wgpu")]
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    #[cfg(feature = "wgpu")]
    fn add(&mut self, tensor: &PyWgpuTensor) -> PyResult<()> {
        self.inner.add(&tensor.inner).map_err(error)
    }
    #[cfg(feature = "wgpu")]
    fn set(&mut self, slot: &Bound<'_, PyAny>, tensor: &PyWgpuTensor) -> PyResult<()> {
        self.inner.set(index(slot)?, &tensor.inner).map_err(error)
    }
    #[cfg(feature = "wgpu")]
    fn compile(&self, py: Python<'_>, steps: &Bound<'_, PyAny>) -> PyResult<PyPointwisePlan> {
        let mut parsed = Vec::new();
        for item in steps.try_iter()? {
            if parsed.len() == 256 {
                return Err(PyValueError::new_err("pointwise step budget exceeded"));
            }
            let (name, rhs): (String, Bound<'_, PyAny>) = item?.extract()?;
            parsed.push(
                PointwiseStep::named(
                    &name,
                    if rhs.is_none() {
                        None
                    } else {
                        Some(index(&rhs)?)
                    },
                )
                .map_err(|e| error(e.into()))?,
            );
        }
        let chain = PointwiseChain::new(self.inner.len(), parsed).map_err(|e| error(e.into()))?;
        Ok(PyPointwisePlan {
            inner: py.detach(|| self.inner.compile(chain)).map_err(error)?,
        })
    }
}

#[pyclass(name = "WgpuPointwisePlan", module = "spiraltorch.wgpu", frozen)]
struct PyPointwisePlan {
    #[cfg(feature = "wgpu")]
    inner: PointwisePlan,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyPointwisePlan {
    #[pyo3(signature = (inputs, execution="fused"))]
    fn run(
        &self,
        py: Python<'_>,
        inputs: &PyPointwiseInputs,
        execution: &str,
    ) -> PyResult<PyWgpuTensor> {
        let execution = execution
            .parse()
            .map_err(|e| error(TensorError::Pointwise(e)))?;
        Ok(PyWgpuTensor {
            inner: py
                .detach(|| inputs.inner.run(&self.inner, execution))
                .map_err(error)?,
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPointwiseInputs>()?;
    module.add_class::<PyPointwisePlan>()
}
