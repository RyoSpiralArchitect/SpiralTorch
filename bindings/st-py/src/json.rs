use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Number as JsonNumber, Value as JsonValue};

pub fn json_to_py(py: Python<'_>, value: &JsonValue) -> PyResult<PyObject> {
    Ok(match value {
        JsonValue::Null => py.None(),
        JsonValue::Bool(v) => v.into_py(py),
        JsonValue::Number(v) => {
            if let Some(i) = v.as_i64() {
                i.into_py(py)
            } else if let Some(u) = v.as_u64() {
                u.into_py(py)
            } else if let Some(f) = v.as_f64() {
                f.into_py(py)
            } else {
                py.None()
            }
        }
        JsonValue::String(v) => v.into_py(py),
        JsonValue::Array(items) => {
            let list = PyList::empty_bound(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_py(py)
        }
        JsonValue::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (key, item) in map {
                dict.set_item(key, json_to_py(py, item)?)?;
            }
            dict.into_py(py)
        }
    })
}

pub fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    py_to_json_inner(value, 0)
}

fn py_to_json_inner(value: &Bound<'_, PyAny>, depth: usize) -> PyResult<JsonValue> {
    if depth > 64 {
        return Err(PyValueError::new_err(
            "payload is too deeply nested to encode as JSON",
        ));
    }

    if value.is_none() {
        return Ok(JsonValue::Null);
    }

    if let Ok(value) = value.downcast::<PyBool>() {
        return Ok(JsonValue::Bool(value.is_true()));
    }

    if let Ok(value) = value.downcast::<PyInt>() {
        if let Ok(v) = value.extract::<i64>() {
            return Ok(JsonValue::Number(JsonNumber::from(v)));
        }
        if let Ok(v) = value.extract::<u64>() {
            return Ok(JsonValue::Number(JsonNumber::from(v)));
        }
        return Err(PyValueError::new_err(
            "int is out of range for JSON encoding",
        ));
    }

    if let Ok(value) = value.downcast::<PyFloat>() {
        let v = value.extract::<f64>()?;
        if !v.is_finite() {
            return Err(PyValueError::new_err(
                "float payload must be finite for JSON encoding",
            ));
        }
        let Some(number) = JsonNumber::from_f64(v) else {
            return Err(PyValueError::new_err(
                "float payload cannot be represented as JSON number",
            ));
        };
        return Ok(JsonValue::Number(number));
    }

    if let Ok(value) = value.downcast::<PyString>() {
        return Ok(JsonValue::String(value.to_string()));
    }

    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut out = serde_json::Map::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let key: String = key.extract().map_err(|_| {
                PyValueError::new_err("dict keys must be strings for JSON encoding")
            })?;
            out.insert(key, py_to_json_inner(&value, depth + 1)?);
        }
        return Ok(JsonValue::Object(out));
    }

    if let Ok(list) = value.downcast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_json_inner(&item, depth + 1)?);
        }
        return Ok(JsonValue::Array(out));
    }

    if let Ok(tuple) = value.downcast::<PyTuple>() {
        let mut out = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            out.push(py_to_json_inner(&item, depth + 1)?);
        }
        return Ok(JsonValue::Array(out));
    }

    Err(PyValueError::new_err(
        "payload must be JSON-like (None/bool/int/float/str/dict/list/tuple)",
    ))
}
