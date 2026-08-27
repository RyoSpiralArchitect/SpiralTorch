use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyStringMethods, PyTuple,
};
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
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_py(py)
        }
        JsonValue::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, json_to_py(py, item)?)?;
            }
            dict.into_py(py)
        }
    })
}

pub fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    py_to_json_inner(value, 0, 64, None)
}

pub fn py_to_json_bounded(
    value: &Bound<'_, PyAny>,
    maximum_bytes: u64,
    maximum_nodes: u64,
    maximum_depth: usize,
    context: &str,
) -> PyResult<JsonValue> {
    let mut budget = JsonBudget {
        bytes: 0,
        nodes: 0,
        maximum_bytes,
        maximum_nodes,
        context,
    };
    py_to_json_inner(value, 0, maximum_depth, Some(&mut budget))
}

struct JsonBudget<'a> {
    bytes: u64,
    nodes: u64,
    maximum_bytes: u64,
    maximum_nodes: u64,
    context: &'a str,
}

impl JsonBudget<'_> {
    fn charge(&mut self, bytes: u64, nodes: u64) -> PyResult<()> {
        self.bytes = self.bytes.saturating_add(bytes);
        self.nodes = self.nodes.saturating_add(nodes);
        if self.bytes > self.maximum_bytes {
            return Err(PyValueError::new_err(format!(
                "{} exceeds JSON ingress budget of {} bytes",
                self.context, self.maximum_bytes
            )));
        }
        if self.nodes > self.maximum_nodes {
            return Err(PyValueError::new_err(format!(
                "{} exceeds JSON ingress budget of {} nodes",
                self.context, self.maximum_nodes
            )));
        }
        Ok(())
    }

    fn require_child_capacity(&self, count: usize) -> PyResult<()> {
        if (count as u64) > self.maximum_nodes.saturating_sub(self.nodes) {
            Err(PyValueError::new_err(format!(
                "{} exceeds JSON ingress budget of {} nodes",
                self.context, self.maximum_nodes
            )))
        } else {
            Ok(())
        }
    }

    fn require_string_capacity(&self, codepoints: usize) -> PyResult<()> {
        if (codepoints as u64).saturating_mul(4) > self.maximum_bytes {
            Err(PyValueError::new_err(format!(
                "{} exceeds JSON ingress budget of {} bytes",
                self.context, self.maximum_bytes
            )))
        } else {
            Ok(())
        }
    }
}

fn json_string_encoded_bytes(value: &str) -> u64 {
    value
        .chars()
        .map(|character| match character {
            '"' | '\\' | '\u{0008}' | '\t' | '\n' | '\u{000c}' | '\r' => 2,
            '\u{0000}'..='\u{001f}' => 6,
            _ => character.len_utf8() as u64,
        })
        .sum()
}

fn py_to_json_inner(
    value: &Bound<'_, PyAny>,
    depth: usize,
    maximum_depth: usize,
    mut budget: Option<&mut JsonBudget<'_>>,
) -> PyResult<JsonValue> {
    if depth > maximum_depth {
        return Err(PyValueError::new_err(
            "payload is too deeply nested to encode as JSON",
        ));
    }
    if let Some(budget) = budget.as_deref_mut() {
        budget.charge(0, 1)?;
    }

    if value.is_none() {
        if let Some(budget) = budget.as_deref_mut() {
            budget.charge(4, 0)?;
        }
        return Ok(JsonValue::Null);
    }

    if let Ok(value) = value.downcast::<PyBool>() {
        if let Some(budget) = budget.as_deref_mut() {
            budget.charge(5, 0)?;
        }
        return Ok(JsonValue::Bool(value.is_true()));
    }

    if let Ok(value) = value.downcast::<PyInt>() {
        if let Some(budget) = budget.as_deref_mut() {
            budget.charge(24, 0)?;
        }
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
        if let Some(budget) = budget.as_deref_mut() {
            budget.charge(24, 0)?;
        }
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
        if let Some(budget) = budget.as_deref_mut() {
            budget.require_string_capacity(value.len()?)?;
        }
        let value_str = value.to_cow()?;
        if let Some(budget) = budget.as_deref_mut() {
            budget.charge(
                json_string_encoded_bytes(value_str.as_ref()).saturating_add(2),
                0,
            )?;
        }
        return Ok(JsonValue::String(value_str.into_owned()));
    }

    if let Ok(dict) = value.downcast::<PyDict>() {
        if let Some(budget) = budget.as_deref_mut() {
            budget.require_child_capacity(dict.len())?;
            budget.charge((dict.len().saturating_sub(1) as u64).saturating_add(2), 0)?;
        }
        let mut out = serde_json::Map::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let key = key.downcast::<PyString>().map_err(|_| {
                PyValueError::new_err("dict keys must be strings for JSON encoding")
            })?;
            if let Some(budget) = budget.as_deref_mut() {
                budget.require_string_capacity(key.len()?)?;
            }
            let key_str = key.to_cow()?;
            if let Some(budget) = budget.as_deref_mut() {
                budget.charge(
                    json_string_encoded_bytes(key_str.as_ref()).saturating_add(3),
                    0,
                )?;
            }
            out.insert(
                key_str.into_owned(),
                py_to_json_inner(&value, depth + 1, maximum_depth, budget.as_deref_mut())?,
            );
        }
        return Ok(JsonValue::Object(out));
    }

    if let Ok(list) = value.downcast::<PyList>() {
        if let Some(budget) = budget.as_deref_mut() {
            budget.require_child_capacity(list.len())?;
            budget.charge((list.len().saturating_sub(1) as u64).saturating_add(2), 0)?;
        }
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_json_inner(
                &item,
                depth + 1,
                maximum_depth,
                budget.as_deref_mut(),
            )?);
        }
        return Ok(JsonValue::Array(out));
    }

    if let Ok(tuple) = value.downcast::<PyTuple>() {
        if let Some(budget) = budget.as_deref_mut() {
            budget.require_child_capacity(tuple.len())?;
            budget.charge((tuple.len().saturating_sub(1) as u64).saturating_add(2), 0)?;
        }
        let mut out = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            out.push(py_to_json_inner(
                &item,
                depth + 1,
                maximum_depth,
                budget.as_deref_mut(),
            )?);
        }
        return Ok(JsonValue::Array(out));
    }

    Err(PyValueError::new_err(
        "payload must be JSON-like (None/bool/int/float/str/dict/list/tuple)",
    ))
}
