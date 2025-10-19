use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::tensor::PyTensor;

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let compat = PyModule::new_bound(py, "compat")?;
    compat.add(
        "__doc__",
        "Zero-copy bridges into Torch, JAX, and TensorFlow via DLPack.",
    )?;

    torch::register(py, &compat)?;
    jax::register(py, &compat)?;
    tensorflow::register(py, &compat)?;

    parent.add_submodule(&compat)?;
    Ok(())
}

fn import_with_hint<'py>(
    py: Python<'py>,
    module: &str,
    hint: &str,
) -> PyResult<Bound<'py, PyModule>> {
    match PyModule::import_bound(py, module) {
        Ok(m) => Ok(m),
        Err(err) => {
            if err.is_instance_of::<PyImportError>(py) {
                Err(PyImportError::new_err(format!(
                    "failed to import '{module}'; install {hint} or add it to your environment"
                )))
            } else {
                Err(err)
            }
        }
    }
}

mod torch {
    use super::*;
    use pyo3::wrap_pyfunction;

    pub(super) fn register(py: Python<'_>, compat: &Bound<PyModule>) -> PyResult<()> {
        let torch = PyModule::new_bound(py, "torch")?;
        torch.add(
            "__doc__",
            "PyTorch conversion helpers backed by torch.utils.dlpack",
        )?;
        torch.add_function(wrap_pyfunction!(to_torch, &torch)?)?;
        torch.add_function(wrap_pyfunction!(from_torch, &torch)?)?;
        compat.add_submodule(&torch)?;
        Ok(())
    }

    #[pyfunction]
    pub(super) fn to_torch(py: Python<'_>, tensor: &PyTensor) -> PyResult<PyObject> {
        let utils = super::import_with_hint(py, "torch.utils.dlpack", "PyTorch >= 1.10")?;
        let from_dlpack = utils.getattr("from_dlpack")?;
        let capsule = tensor.to_dlpack(py)?;
        let tensor = from_dlpack.call1((capsule,))?;
        Ok(tensor.into_py(py))
    }

    #[pyfunction]
    pub(super) fn from_torch(py: Python<'_>, tensor: &Bound<PyAny>) -> PyResult<PyTensor> {
        let utils = super::import_with_hint(py, "torch.utils.dlpack", "PyTorch >= 1.10")?;
        let to_dlpack = utils.getattr("to_dlpack")?;
        let capsule = to_dlpack.call1((tensor,))?.unbind();
        PyTensor::from_dlpack(py, capsule)
    }
}

mod jax {
    use super::*;
    use pyo3::wrap_pyfunction;

    pub(super) fn register(py: Python<'_>, compat: &Bound<PyModule>) -> PyResult<()> {
        let jax = PyModule::new_bound(py, "jax")?;
        jax.add("__doc__", "JAX conversion helpers backed by jax.dlpack")?;
        jax.add_function(wrap_pyfunction!(to_jax, &jax)?)?;
        jax.add_function(wrap_pyfunction!(from_jax, &jax)?)?;
        compat.add_submodule(&jax)?;
        Ok(())
    }

    #[pyfunction]
    pub(super) fn to_jax(py: Python<'_>, tensor: &PyTensor) -> PyResult<PyObject> {
        let module = super::import_with_hint(py, "jax.dlpack", "JAX >= 0.4")?;
        let from_dlpack = module.getattr("from_dlpack")?;
        let capsule = tensor.to_dlpack(py)?;
        let array = from_dlpack.call1((capsule,))?;
        Ok(array.into_py(py))
    }

    #[pyfunction]
    pub(super) fn from_jax(py: Python<'_>, array: &Bound<PyAny>) -> PyResult<PyTensor> {
        let module = super::import_with_hint(py, "jax.dlpack", "JAX >= 0.4")?;
        let to_dlpack = module.getattr("to_dlpack")?;
        let capsule = to_dlpack.call1((array,))?.unbind();
        PyTensor::from_dlpack(py, capsule)
    }
}

mod tensorflow {
    use super::*;
    use pyo3::wrap_pyfunction;

    pub(super) fn register(py: Python<'_>, compat: &Bound<PyModule>) -> PyResult<()> {
        let tf = PyModule::new_bound(py, "tensorflow")?;
        tf.add(
            "__doc__",
            "TensorFlow conversion helpers backed by tf.experimental.dlpack",
        )?;
        tf.add_function(wrap_pyfunction!(to_tensorflow, &tf)?)?;
        tf.add_function(wrap_pyfunction!(from_tensorflow, &tf)?)?;
        compat.add_submodule(&tf)?;
        Ok(())
    }

    #[pyfunction]
    pub(super) fn to_tensorflow(py: Python<'_>, tensor: &PyTensor) -> PyResult<PyObject> {
        let module =
            super::import_with_hint(py, "tensorflow.experimental.dlpack", "TensorFlow >= 2.12")?;
        let from_dlpack = module.getattr("from_dlpack")?;
        let capsule = tensor.to_dlpack(py)?;
        let tensor = from_dlpack.call1((capsule,))?;
        Ok(tensor.into_py(py))
    }

    #[pyfunction]
    pub(super) fn from_tensorflow(py: Python<'_>, value: &Bound<PyAny>) -> PyResult<PyTensor> {
        let module =
            super::import_with_hint(py, "tensorflow.experimental.dlpack", "TensorFlow >= 2.12")?;
        let to_dlpack = module.getattr("to_dlpack")?;
        let capsule = to_dlpack.call1((value,))?.unbind();
        PyTensor::from_dlpack(py, capsule)
    }
}
