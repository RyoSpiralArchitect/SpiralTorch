use pyo3::exceptions::{PyImportError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;

use crate::tensor::PyTensor;

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let compat = PyModule::new_bound(py, "compat")?;
    compat.add(
        "__doc__",
        "Zero-copy bridges into Torch, JAX, and TensorFlow via DLPack.",
    )?;

    compat.add_function(wrap_pyfunction!(capture, &compat)?)?;
    compat.add_function(wrap_pyfunction!(share, &compat)?)?;

    parent.add_function(wrap_pyfunction!(capture, parent)?)?;
    parent.add_function(wrap_pyfunction!(share, parent)?)?;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    SpiralTorch,
    Torch,
    Jax,
    TensorFlow,
    Dlpack,
}

fn detect_source(value: &Bound<PyAny>) -> Source {
    if value.is_instance_of::<PyTensor>() {
        return Source::SpiralTorch;
    }

    let ty = value.get_type();
    let module = ty
        .getattr("__module__")
        .ok()
        .and_then(|name| name.extract::<String>().ok())
        .unwrap_or_default();

    if module.starts_with("torch") {
        return Source::Torch;
    }

    if module.starts_with("jax") || module.contains("jaxlib") {
        return Source::Jax;
    }

    if module.starts_with("tensorflow") {
        return Source::TensorFlow;
    }

    Source::Dlpack
}

fn capture_impl(py: Python<'_>, value: &Bound<PyAny>) -> PyResult<PyTensor> {
    match detect_source(value) {
        Source::SpiralTorch => {
            let py_tensor: Py<PyTensor> = value.extract()?;
            Ok(py_tensor.bind(py).borrow().clone())
        }
        Source::Torch => torch::from_torch(py, value),
        Source::Jax => jax::from_jax(py, value),
        Source::TensorFlow => tensorflow::from_tensorflow(py, value),
        Source::Dlpack => PyTensor::from_dlpack(py, value.clone().unbind().into_py(py)),
    }
}

fn share_impl(py: Python<'_>, value: &Bound<PyAny>, target: &str) -> PyResult<PyObject> {
    let normalized = target.to_ascii_lowercase();
    let ty = value.get_type();
    let module = ty
        .getattr("__module__")
        .ok()
        .and_then(|name| name.extract::<String>().ok())
        .unwrap_or_default();
    match normalized.as_str() {
        "spiraltorch" | "spiral" | "st" | "tensor" => {
            if value.is_instance_of::<PyTensor>() {
                return Ok(value.clone().unbind().into_py(py));
            }
            let tensor = capture_impl(py, value)?;
            Py::new(py, tensor).map(|py_tensor| py_tensor.into_py(py))
        }
        "torch" | "pytorch" => {
            if module.starts_with("torch") {
                return Ok(value.clone().unbind().into_py(py));
            }
            let tensor = capture_impl(py, value)?;
            torch::to_torch(py, &tensor)
        }
        "jax" => {
            if module.starts_with("jax") || module.contains("jaxlib") {
                return Ok(value.clone().unbind().into_py(py));
            }
            let tensor = capture_impl(py, value)?;
            jax::to_jax(py, &tensor)
        }
        "tensorflow" | "tf" => {
            if module.starts_with("tensorflow") {
                return Ok(value.clone().unbind().into_py(py));
            }
            let tensor = capture_impl(py, value)?;
            tensorflow::to_tensorflow(py, &tensor)
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown target '{target}'; choose 'spiraltorch', 'torch', 'jax', or 'tensorflow'"
        ))),
    }
}

/// Convert any DLPack-capable tensor into a SpiralTorch tensor.
#[pyfunction]
#[pyo3(text_signature = "(value, /)")]
fn capture(py: Python<'_>, value: &Bound<PyAny>) -> PyResult<PyTensor> {
    capture_impl(py, value)
}

/// Share a tensor with another framework without copying the underlying buffer.
#[pyfunction]
#[pyo3(text_signature = "(value, target, /)")]
fn share(py: Python<'_>, value: &Bound<PyAny>, target: &str) -> PyResult<PyObject> {
    share_impl(py, value, target)
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
