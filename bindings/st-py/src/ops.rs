use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyList, PyModule};
use pyo3::wrap_pyfunction;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::tensor::{tensor_err_to_py, PyTensor};
use st_core::ops::{global_operator_registry, OperatorBuilder, RegisteredOperator};
use st_tensor::{Tensor, TensorError};

fn collect_tensors(py: Python<'_>, inputs: &Bound<'_, PyAny>) -> PyResult<Vec<Tensor>> {
    if let Ok(handle) = inputs.extract::<Py<PyTensor>>() {
        return Ok(vec![handle.bind(py).borrow().inner.clone()]);
    }

    let iter = PyIterator::from_bound_object(inputs).map_err(|_| {
        PyTypeError::new_err("inputs must be a Tensor or an iterable of Tensor objects")
    })?;
    let mut out = Vec::new();
    for item in iter {
        let item = item?;
        let handle: Py<PyTensor> = item.extract().map_err(|_| {
            PyTypeError::new_err("inputs must contain only Tensor objects")
        })?;
        out.push(handle.bind(py).borrow().inner.clone());
    }
    Ok(out)
}

fn tensors_to_pylist(py: Python<'_>, tensors: &[Tensor]) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for tensor in tensors {
        list.append(Py::new(py, PyTensor::from_tensor(tensor.clone()))?)?;
    }
    Ok(list.into_py(py))
}

fn extract_output_tensors(py: Python<'_>, output: &Bound<'_, PyAny>) -> PyResult<Vec<Tensor>> {
    if let Ok(handle) = output.extract::<Py<PyTensor>>() {
        return Ok(vec![handle.bind(py).borrow().inner.clone()]);
    }

    if let Ok(handles) = output.extract::<Vec<Py<PyTensor>>>() {
        let mut tensors = Vec::with_capacity(handles.len());
        for handle in handles {
            tensors.push(handle.bind(py).borrow().inner.clone());
        }
        return Ok(tensors);
    }

    let iter = PyIterator::from_bound_object(output)
        .map_err(|_| PyTypeError::new_err("operator output must be a Tensor or iterable"))?;
    let mut tensors = Vec::new();
    for item in iter {
        let item = item?;
        let handle: Py<PyTensor> = item.extract().map_err(|_| {
            PyTypeError::new_err("operator output must contain only Tensor objects")
        })?;
        tensors.push(handle.bind(py).borrow().inner.clone());
    }
    if tensors.is_empty() {
        return Err(PyValueError::new_err("operator output must not be empty"));
    }
    Ok(tensors)
}

fn pyerr_to_tensor(err: PyErr, label: &str) -> TensorError {
    TensorError::Generic(format!("{label} callback failed: {err}"))
}

fn call_python_forward(
    callback: &Arc<Mutex<Py<PyAny>>>,
    inputs: &[&Tensor],
    expected_outputs: usize,
) -> Result<Vec<Tensor>, TensorError> {
    Python::with_gil(|py| {
        let mut input_vec = Vec::with_capacity(inputs.len());
        for tensor in inputs {
            input_vec.push((*tensor).clone());
        }
        let py_inputs = tensors_to_pylist(py, &input_vec).map_err(|err| {
            TensorError::Generic(format!("failed to build input list: {err}"))
        })?;
        let callback = callback.lock().map_err(|_| {
            TensorError::Generic("operator callback lock was poisoned".to_string())
        })?;
        let result = callback
            .call1(py, (py_inputs,))
            .map_err(|err| pyerr_to_tensor(err, "operator forward"))?;
        let output = extract_output_tensors(py, &result.bind(py))
            .map_err(|err| pyerr_to_tensor(err, "operator forward"))?;
        if expected_outputs > 0 && output.len() != expected_outputs {
            return Err(TensorError::Generic(format!(
                "operator forward returned {} outputs, expected {}",
                output.len(),
                expected_outputs
            )));
        }
        Ok(output)
    })
}

fn call_python_backward(
    callback: &Arc<Mutex<Py<PyAny>>>,
    inputs: &[&Tensor],
    outputs: &[&Tensor],
    grad_outputs: &[&Tensor],
    expected_grads: usize,
) -> Result<Vec<Tensor>, TensorError> {
    Python::with_gil(|py| {
        let mut input_vec = Vec::with_capacity(inputs.len());
        for tensor in inputs {
            input_vec.push((*tensor).clone());
        }
        let mut output_vec = Vec::with_capacity(outputs.len());
        for tensor in outputs {
            output_vec.push((*tensor).clone());
        }
        let mut grad_vec = Vec::with_capacity(grad_outputs.len());
        for tensor in grad_outputs {
            grad_vec.push((*tensor).clone());
        }
        let py_inputs = tensors_to_pylist(py, &input_vec)
            .map_err(|err| TensorError::Generic(format!("failed to build input list: {err}")))?;
        let py_outputs = tensors_to_pylist(py, &output_vec)
            .map_err(|err| TensorError::Generic(format!("failed to build output list: {err}")))?;
        let py_grads = tensors_to_pylist(py, &grad_vec)
            .map_err(|err| TensorError::Generic(format!("failed to build grad list: {err}")))?;
        let callback = callback.lock().map_err(|_| {
            TensorError::Generic("operator callback lock was poisoned".to_string())
        })?;
        let result = callback
            .call1(py, (py_inputs, py_outputs, py_grads))
            .map_err(|err| pyerr_to_tensor(err, "operator backward"))?;
        let grads = extract_output_tensors(py, &result.bind(py))
            .map_err(|err| pyerr_to_tensor(err, "operator backward"))?;
        if expected_grads > 0 && grads.len() != expected_grads {
            return Err(TensorError::Generic(format!(
                "operator backward returned {} gradients, expected {}",
                grads.len(),
                expected_grads
            )));
        }
        Ok(grads)
    })
}

fn parse_attributes(attrs: Option<&Bound<'_, PyAny>>) -> PyResult<HashMap<String, String>> {
    let mut map = HashMap::new();
    let Some(attrs) = attrs else {
        return Ok(map);
    };
    let dict = attrs.downcast::<PyDict>().map_err(|_| {
        PyTypeError::new_err("attributes must be a dict[str, str]")
    })?;
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        let value: String = value.extract().map_err(|_| {
            PyTypeError::new_err("attributes must map string keys to string values")
        })?;
        map.insert(key, value);
    }
    Ok(map)
}

#[pyfunction]
#[pyo3(signature = (name, num_inputs, num_outputs, forward, *, backward=None, description=None, backends=None, attributes=None, supports_inplace=false, differentiable=None))]
#[allow(clippy::too_many_arguments)]
fn register(
    py: Python<'_>,
    name: &str,
    num_inputs: usize,
    num_outputs: usize,
    forward: PyObject,
    backward: Option<PyObject>,
    description: Option<&str>,
    backends: Option<Vec<String>>,
    attributes: Option<&Bound<'_, PyAny>>,
    supports_inplace: bool,
    differentiable: Option<bool>,
) -> PyResult<()> {
    let forward_any = forward.bind(py);
    if !forward_any.is_callable() {
        return Err(PyTypeError::new_err("forward must be a callable"));
    }
    if let Some(ref backward) = backward {
        if !backward.bind(py).is_callable() {
            return Err(PyTypeError::new_err("backward must be a callable"));
        }
    }
    if differentiable == Some(false) && backward.is_some() {
        return Err(PyValueError::new_err(
            "differentiable=false conflicts with a provided backward callback",
        ));
    }

    let forward_cb = Arc::new(Mutex::new(forward));
    let expected_outputs = num_outputs;
    let forward_fn = Arc::new(move |inputs: &[&Tensor]| {
        call_python_forward(&forward_cb, inputs, expected_outputs)
    });

    let backward_fn = backward.map(|backward| {
        let backward_cb = Arc::new(Mutex::new(backward));
        let expected_grads = num_inputs;
        Arc::new(move |inputs: &[&Tensor], outputs: &[&Tensor], grads: &[&Tensor]| {
            call_python_backward(&backward_cb, inputs, outputs, grads, expected_grads)
        })
            as Arc<dyn Fn(&[&Tensor], &[&Tensor], &[&Tensor]) -> Result<Vec<Tensor>, TensorError> + Send + Sync>
    });

    let mut builder = OperatorBuilder::new(name, num_inputs, num_outputs);
    if let Some(desc) = description {
        builder = builder.with_description(desc);
    }
    let backends = backends.unwrap_or_else(|| vec!["python".to_string()]);
    for backend in backends {
        builder = builder.with_backend(backend);
    }
    if supports_inplace {
        builder = builder.with_inplace(true);
    }
    if let Some(diff) = differentiable {
        builder = builder.with_differentiable(diff);
    }
    let attrs = parse_attributes(attributes)?;
    for (key, value) in attrs {
        builder = builder.with_attribute(key, value);
    }
    builder = builder.with_forward(forward_fn);
    if let Some(backward_fn) = backward_fn {
        builder = builder.with_backward(backward_fn);
    }

    let operator = builder.build().map_err(tensor_err_to_py)?;
    global_operator_registry()
        .register(operator)
        .map_err(tensor_err_to_py)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (name, inputs, *, return_single=false))]
fn execute(py: Python<'_>, name: &str, inputs: &Bound<'_, PyAny>, return_single: bool) -> PyResult<PyObject> {
    let tensors = collect_tensors(py, inputs)?;
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let outputs = global_operator_registry()
        .execute(name, &refs)
        .map_err(tensor_err_to_py)?;
    if return_single && outputs.len() == 1 {
        return Ok(Py::new(py, PyTensor::from_tensor(outputs[0].clone()))?.into_py(py));
    }
    let list = PyList::empty_bound(py);
    for output in outputs {
        list.append(Py::new(py, PyTensor::from_tensor(output))?)?;
    }
    Ok(list.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (name, inputs, outputs, grad_outputs, *, return_single=false))]
fn backward(
    py: Python<'_>,
    name: &str,
    inputs: &Bound<'_, PyAny>,
    outputs: &Bound<'_, PyAny>,
    grad_outputs: &Bound<'_, PyAny>,
    return_single: bool,
) -> PyResult<PyObject> {
    let inputs = collect_tensors(py, inputs)?;
    let outputs = collect_tensors(py, outputs)?;
    let grads = collect_tensors(py, grad_outputs)?;
    let input_refs: Vec<&Tensor> = inputs.iter().collect();
    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    let grad_refs: Vec<&Tensor> = grads.iter().collect();

    let operator = global_operator_registry()
        .get(name)
        .ok_or_else(|| PyValueError::new_err(format!("operator '{name}' not found")))?;
    let grads = operator
        .backward(&input_refs, &output_refs, &grad_refs)
        .map_err(tensor_err_to_py)?;

    if return_single && grads.len() == 1 {
        return Ok(Py::new(py, PyTensor::from_tensor(grads[0].clone()))?.into_py(py));
    }
    let list = PyList::empty_bound(py);
    for grad in grads {
        list.append(Py::new(py, PyTensor::from_tensor(grad))?)?;
    }
    Ok(list.into_py(py))
}

#[pyfunction]
fn list_operators() -> Vec<String> {
    global_operator_registry().list_operators()
}

#[pyfunction]
fn unregister(name: &str) -> bool {
    global_operator_registry().unregister(name)
}

fn metadata_dict(py: Python<'_>, operator: &RegisteredOperator) -> PyResult<PyObject> {
    let meta = operator.metadata();
    let dict = PyDict::new_bound(py);
    dict.set_item("name", meta.signature.name.clone())?;
    dict.set_item("num_inputs", meta.signature.num_inputs)?;
    dict.set_item("num_outputs", meta.signature.num_outputs)?;
    dict.set_item("supports_inplace", meta.signature.supports_inplace)?;
    dict.set_item("differentiable", meta.signature.differentiable)?;
    dict.set_item("description", meta.description.clone())?;
    dict.set_item("backends", meta.backends.clone())?;
    dict.set_item("attributes", meta.attributes.clone())?;
    Ok(dict.into_py(py))
}

#[pyfunction]
fn metadata(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    let operator = global_operator_registry()
        .get(name)
        .ok_or_else(|| PyValueError::new_err(format!("operator '{name}' not found")))?;
    metadata_dict(py, &operator)
}

pub(crate) fn register_module(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "ops")?;
    module.add("__doc__", "Custom operator registry (Python callbacks)")?;
    module.add_function(wrap_pyfunction!(register, &module)?)?;
    module.add_function(wrap_pyfunction!(execute, &module)?)?;
    module.add_function(wrap_pyfunction!(backward, &module)?)?;
    module.add_function(wrap_pyfunction!(list_operators, &module)?)?;
    module.add_function(wrap_pyfunction!(unregister, &module)?)?;
    module.add_function(wrap_pyfunction!(metadata, &module)?)?;
    module.add(
        "__all__",
        vec![
            "register",
            "execute",
            "backward",
            "list_operators",
            "unregister",
            "metadata",
        ],
    )?;
    parent.add_submodule(&module)?;
    parent.add("ops", module.to_object(py))?;
    Ok(())
}
