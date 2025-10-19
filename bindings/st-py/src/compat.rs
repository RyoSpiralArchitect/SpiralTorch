use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::tensor::PyTensor;

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let compat = PyModule::new_bound(py, "compat")?;
    compat.add(
        "__doc__",
        "Zero-copy bridges into Torch, JAX, TensorFlow, and NumPy via DLPack plus auto-detection helpers.",
    )?;

    torch::register(py, &compat)?;
    jax::register(py, &compat)?;
    tensorflow::register(py, &compat)?;
    numpy::register(py, &compat)?;
    auto::register(py, &compat)?;

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

mod numpy {
    use super::*;
    use pyo3::exceptions::PyTypeError;
    use pyo3::types::PyDict;
    use pyo3::wrap_pyfunction;

    pub(super) fn register(py: Python<'_>, compat: &Bound<PyModule>) -> PyResult<()> {
        let numpy = PyModule::new_bound(py, "numpy")?;
        numpy.add(
            "__doc__",
            "NumPy conversion helpers backed by numpy.from_dlpack and ndarray.__dlpack__",
        )?;
        numpy.add_function(wrap_pyfunction!(to_numpy, &numpy)?)?;
        numpy.add_function(wrap_pyfunction!(from_numpy, &numpy)?)?;
        compat.add_submodule(&numpy)?;
        Ok(())
    }

    #[pyfunction]
    pub(super) fn to_numpy(py: Python<'_>, tensor: &PyTensor) -> PyResult<PyObject> {
        let numpy = super::import_with_hint(py, "numpy", "NumPy >= 1.22")?;
        let from_dlpack = numpy.getattr("from_dlpack")?;
        let capsule = tensor.to_dlpack(py)?;
        let array = from_dlpack.call1((capsule,))?;
        Ok(array.into_py(py))
    }

    #[pyfunction]
    pub(super) fn from_numpy(py: Python<'_>, array: &Bound<PyAny>) -> PyResult<PyTensor> {
        let _numpy = super::import_with_hint(py, "numpy", "NumPy >= 1.22")?;
        let dlpack = array.getattr("__dlpack__").map_err(|_| {
            PyTypeError::new_err(
                "object does not implement __dlpack__; upgrade to NumPy >= 1.22 or pass a dlpack capsule",
            )
        })?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("stream", py.None())?;
        let capsule = dlpack.call((), Some(&kwargs))?.unbind();
        PyTensor::from_dlpack(py, capsule)
    }
}

mod auto {
    use super::*;
    use crate::tensor::{ensure_dlpack_capsule, from_dlpack_capsule};
    use pyo3::exceptions::PyTypeError;
    use pyo3::ffi;
    use pyo3::types::PyType;
    use pyo3::wrap_pyfunction;

    #[derive(Clone, Copy)]
    struct Backend {
        canonical: &'static str,
        import_path: &'static str,
        hint: &'static str,
        display: &'static str,
        aliases: &'static [&'static str],
    }

    const BACKENDS: &[Backend] = &[
        Backend {
            canonical: "torch",
            import_path: "torch.utils.dlpack",
            hint: "PyTorch >= 1.10",
            display: "PyTorch",
            aliases: &["pytorch"],
        },
        Backend {
            canonical: "jax",
            import_path: "jax.dlpack",
            hint: "JAX >= 0.4",
            display: "JAX",
            aliases: &[],
        },
        Backend {
            canonical: "tensorflow",
            import_path: "tensorflow.experimental.dlpack",
            hint: "TensorFlow >= 2.12",
            display: "TensorFlow",
            aliases: &["tf"],
        },
        Backend {
            canonical: "numpy",
            import_path: "numpy",
            hint: "NumPy >= 1.22",
            display: "NumPy",
            aliases: &["np"],
        },
    ];

    pub(super) fn register(py: Python<'_>, compat: &Bound<PyModule>) -> PyResult<()> {
        let auto = PyModule::new_bound(py, "auto")?;
        auto.add(
            "__doc__",
            "Detect dlpack-ready objects, surface friendly hints, and import preferred backends.",
        )?;
        auto.add_function(wrap_pyfunction!(from_any, &auto)?)?;
        auto.add_function(wrap_pyfunction!(describe, &auto)?)?;
        auto.add_function(wrap_pyfunction!(available_backends, &auto)?)?;
        auto.add_function(wrap_pyfunction!(require_backend, &auto)?)?;
        compat.add_submodule(&auto)?;
        Ok(())
    }

    fn backend_by_name(name: &str) -> Option<&'static Backend> {
        let lowered = name.to_ascii_lowercase();
        BACKENDS.iter().find(|backend| {
            backend.canonical == lowered || backend.aliases.iter().any(|a| *a == lowered)
        })
    }

    fn try_import(py: Python<'_>, backend: &Backend) -> bool {
        PyModule::import_bound(py, backend.import_path).is_ok()
    }

    fn describe_type(value: &Bound<PyAny>) -> PyResult<String> {
        let ty: Bound<'_, PyType> = value.get_type();
        let module: Option<String> = ty.getattr("__module__").ok().and_then(|m| m.extract().ok());
        let qualname: Option<String> = ty
            .getattr("__qualname__")
            .ok()
            .and_then(|n| n.extract().ok());
        let fallback = ty
            .name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "unknown".to_string());
        Ok(match (module, qualname) {
            (Some(module), Some(name)) => format!("{module}.{name}"),
            (None, Some(name)) => name,
            _ => fallback,
        })
    }

    fn classify(value: &Bound<PyAny>) -> Option<&'static str> {
        if unsafe { ffi::PyCapsule_CheckExact(value.as_ptr()) == 1 } {
            return Some("dlpack");
        }
        let ty: Bound<'_, PyType> = value.get_type();
        let module: Option<String> = ty.getattr("__module__").ok().and_then(|m| m.extract().ok());
        module.as_ref().and_then(|module| {
            if module.starts_with("torch") {
                Some("torch")
            } else if module.contains("jax") {
                Some("jax")
            } else if module.contains("tensorflow") || module.contains("tf.") {
                Some("tensorflow")
            } else if module.starts_with("numpy") {
                Some("numpy")
            } else {
                None
            }
        })
    }

    fn hint_message(py: Python<'_>) -> String {
        let mut parts = Vec::new();
        for backend in BACKENDS {
            let status = if try_import(py, backend) {
                format!("{} (available)", backend.display)
            } else {
                format!("{} (install {})", backend.display, backend.hint)
            };
            parts.push(status);
        }
        format!(
            "expected a DLPack capsule or object with __dlpack__; backends tried: {}",
            parts.join(", ")
        )
    }

    #[pyfunction]
    pub(super) fn from_any(py: Python<'_>, value: &Bound<PyAny>) -> PyResult<PyTensor> {
        match ensure_dlpack_capsule(py, value) {
            Ok(capsule) => {
                let capsule = capsule.bind(py);
                from_dlpack_capsule(py, &capsule)
            }
            Err(err) => {
                if err.is_instance_of::<PyTypeError>(py) {
                    let descriptor = describe_type(value).unwrap_or_else(|_| "unknown".into());
                    let guess = classify(value)
                        .map(|name| format!(" (detected: {name})"))
                        .unwrap_or_default();
                    let message = format!("{descriptor}{guess}: {}", hint_message(py));
                    Err(PyTypeError::new_err(message))
                } else {
                    Err(err)
                }
            }
        }
    }

    #[pyfunction]
    pub(super) fn describe(_py: Python<'_>, value: &Bound<PyAny>) -> PyResult<String> {
        let descriptor = describe_type(value)?;
        let classification = classify(value).unwrap_or("unknown");
        let dlpack = if unsafe { ffi::PyCapsule_CheckExact(value.as_ptr()) == 1 } {
            "capsule"
        } else if value.hasattr("__dlpack__")? {
            "callable"
        } else {
            "missing"
        };
        Ok(format!(
            "{descriptor} (backend: {classification}, dlpack: {dlpack})"
        ))
    }

    #[pyfunction]
    pub(super) fn available_backends(py: Python<'_>) -> Vec<&'static str> {
        BACKENDS
            .iter()
            .filter(|backend| try_import(py, backend))
            .map(|backend| backend.display)
            .collect()
    }

    #[pyfunction]
    pub(super) fn require_backend(py: Python<'_>, name: &str) -> PyResult<()> {
        let backend = backend_by_name(name).ok_or_else(|| {
            PyTypeError::new_err(format!(
                "unknown backend '{name}', expected one of torch|jax|tensorflow|numpy"
            ))
        })?;
        super::import_with_hint(py, backend.import_path, backend.hint)?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn auto_from_any_respects_dlpack_capsule() {
            Python::with_gil(|py| {
                let tensor = PyTensor::zeros(1, 2).unwrap();
                let capsule = tensor.to_dlpack(py).unwrap();
                let capsule_bound = capsule.bind(py);
                let restored = from_any(py, &capsule_bound).unwrap();
                assert_eq!(
                    tensor.inner().data().as_ptr(),
                    restored.inner().data().as_ptr()
                );
            });
        }

        #[test]
        fn describe_marks_missing_dlpack() {
            Python::with_gil(|py| {
                let none_obj = py.None();
                let none = none_obj.bind(py);
                let desc = describe(py, &none).unwrap();
                assert!(desc.contains("missing"));
            });
        }

        #[test]
        fn from_any_hints_on_missing_dlpack() {
            Python::with_gil(|py| {
                let none_obj = py.None();
                let none = none_obj.bind(py);
                let message = from_any(py, &none).err().unwrap().to_string();
                assert!(message.contains("expected a DLPack capsule"));
                assert!(message.contains("PyTorch"));
                assert!(message.contains("NumPy"));
            });
        }
    }
}
