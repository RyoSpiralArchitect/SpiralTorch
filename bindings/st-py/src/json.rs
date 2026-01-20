use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value as JsonValue;

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

