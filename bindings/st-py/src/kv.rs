#[cfg(feature = "kv-redis")]
use crate::json::{json_to_py, py_to_json};
use crate::planner::PyRankPlan;
#[cfg(not(feature = "kv-redis"))]
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use pyo3::{wrap_pyfunction, PyRef};
#[cfg(feature = "kv-redis")]
use serde_json::Value as JsonValue;

const DEFAULT_CHOICE_NAMESPACE: &str = "spiral:heur:v1";

fn condition_label(condition: &str) -> PyResult<&'static str> {
    match condition.to_ascii_lowercase().as_str() {
        "always" | "" => Ok("always"),
        "nx" => Ok("nx"),
        "xx" => Ok("xx"),
        other => Err(PyValueError::new_err(format!(
            "condition must be 'always', 'nx', or 'xx', got '{other}'"
        ))),
    }
}

fn selected_expiry(
    expiry_seconds: Option<u64>,
    expiry_milliseconds: Option<u64>,
    expiry_at_seconds: Option<u64>,
    expiry_at_milliseconds: Option<u64>,
) -> PyResult<Option<(&'static str, &'static str, u64)>> {
    let mut selected = Vec::new();
    if let Some(value) = expiry_seconds {
        selected.push(("seconds", "EX", value));
    }
    if let Some(value) = expiry_milliseconds {
        selected.push(("milliseconds", "PX", value));
    }
    if let Some(value) = expiry_at_seconds {
        selected.push(("at_seconds", "EXAT", value));
    }
    if let Some(value) = expiry_at_milliseconds {
        selected.push(("at_milliseconds", "PXAT", value));
    }
    if selected.len() > 1 {
        return Err(PyValueError::new_err(
            "choose at most one expiry_seconds/expiry_milliseconds/expiry_at_seconds/expiry_at_milliseconds",
        ));
    }
    Ok(selected.into_iter().next())
}

fn validate_json_set_options(
    expiry: Option<(&'static str, &'static str, u64)>,
    keep_ttl: bool,
    persist: bool,
) -> PyResult<()> {
    if keep_ttl && expiry.is_some() {
        return Err(PyValueError::new_err(
            "cannot set both explicit expiry and KEEPTTL",
        ));
    }
    if persist && expiry.is_some() {
        return Err(PyValueError::new_err(
            "cannot use PERSIST alongside an explicit expiry",
        ));
    }
    if persist && keep_ttl {
        return Err(PyValueError::new_err("cannot combine PERSIST with KEEPTTL"));
    }
    Ok(())
}

fn append_option_fragments(
    py: Python<'_>,
    fragments: &Bound<PyList>,
    expiry: Option<(&'static str, &'static str, u64)>,
    keep_ttl: bool,
    persist: bool,
    condition: &str,
) -> PyResult<()> {
    if let Some((_kind, keyword, value)) = expiry {
        fragments.append(keyword)?;
        fragments.append(value)?;
    }
    if persist {
        fragments.append("PERSIST")?;
    }
    if keep_ttl {
        fragments.append("KEEPTTL")?;
    }
    match condition {
        "nx" => fragments.append("NX")?,
        "xx" => fragments.append("XX")?,
        _ => {}
    }
    let _ = py;
    Ok(())
}

fn lg2_bucket(value: u32) -> u32 {
    32 - (value.max(1) - 1).leading_zeros()
}

fn set_choice_field(dict: &Bound<PyDict>, key: &str, value: u64) -> PyResult<()> {
    dict.set_item(key, value)
}

fn merge_kind_name(value: u32) -> &'static str {
    match value {
        1 => "shared",
        2 => "warp",
        _ => "bitonic",
    }
}

fn merge_detail_name(value: u32) -> &'static str {
    match value {
        1 => "heap",
        2 => "kway",
        3 => "bitonic",
        4 => "warp_heap",
        5 => "warp_bitonic",
        _ => "auto",
    }
}

fn two_stage_mode(is_target_kind: bool, use_two_stage: bool) -> u8 {
    if !is_target_kind {
        0
    } else if use_two_stage {
        2
    } else {
        1
    }
}

#[cfg(not(feature = "kv-redis"))]
fn kv_unavailable() -> PyErr {
    PyNotImplementedError::new_err(
        "Redis-backed st-kv helpers require building spiraltorch with the 'kv-redis' feature",
    )
}

#[pyfunction]
fn kv_redis_available() -> bool {
    cfg!(feature = "kv-redis")
}

#[pyfunction]
fn kv_choice_schema_fields() -> Vec<&'static str> {
    vec![
        "use_2ce",
        "wg",
        "kl",
        "ch",
        "algo_topk",
        "ctile",
        "mode_midk",
        "mode_bottomk",
        "tile_cols",
        "radix",
        "segments",
    ]
}

#[pyfunction]
#[pyo3(signature = (rows, cols, k, subgroup=false, namespace=DEFAULT_CHOICE_NAMESPACE))]
fn kv_rank_choice_key(rows: u32, cols: u32, k: u32, subgroup: bool, namespace: &str) -> String {
    let _ = rows;
    format!(
        "{namespace}:sg:{}:c:{}:k:{}",
        if subgroup { 1 } else { 0 },
        lg2_bucket(cols),
        lg2_bucket(k)
    )
}

#[pyfunction]
#[pyo3(signature = (plan, namespace=DEFAULT_CHOICE_NAMESPACE))]
fn kv_choice_key_from_rank_plan(plan: PyRef<'_, PyRankPlan>, namespace: &str) -> String {
    let plan = plan.plan();
    kv_rank_choice_key(
        plan.rows,
        plan.cols,
        plan.k,
        plan.choice.subgroup,
        namespace,
    )
}

#[pyfunction]
fn kv_choice_from_rank_plan(py: Python<'_>, plan: PyRef<'_, PyRankPlan>) -> PyResult<PyObject> {
    let plan = plan.plan();
    let choice = &plan.choice;
    let dict = PyDict::new(py);
    let rank_kind = plan.kind.as_str();
    let algo_topk = if rank_kind == "topk" {
        choice.mkd.min(u8::MAX as u32) as u8
    } else {
        0
    };
    let tile_cols = choice.fft_tile.max(choice.tile).max(1);
    dict.set_item("use_2ce", choice.use_2ce)?;
    set_choice_field(&dict, "wg", choice.wg as u64)?;
    set_choice_field(&dict, "kl", choice.kl as u64)?;
    set_choice_field(&dict, "ch", choice.ch as u64)?;
    set_choice_field(&dict, "algo_topk", algo_topk as u64)?;
    set_choice_field(&dict, "ctile", choice.ctile as u64)?;
    set_choice_field(
        &dict,
        "mode_midk",
        two_stage_mode(rank_kind == "midk", choice.use_2ce) as u64,
    )?;
    set_choice_field(
        &dict,
        "mode_bottomk",
        two_stage_mode(rank_kind == "bottomk", choice.use_2ce) as u64,
    )?;
    set_choice_field(&dict, "tile_cols", tile_cols as u64)?;
    set_choice_field(&dict, "radix", choice.fft_radix.max(1) as u64)?;
    set_choice_field(&dict, "segments", choice.fft_segments.max(1) as u64)?;
    dict.set_item("rank_kind", rank_kind)?;
    dict.set_item("rows", plan.rows)?;
    dict.set_item("cols", plan.cols)?;
    dict.set_item("k", plan.k)?;
    dict.set_item("subgroup", choice.subgroup)?;
    dict.set_item("merge_strategy", merge_kind_name(choice.mk))?;
    dict.set_item("merge_detail", merge_detail_name(choice.mkd))?;
    dict.set_item("merge_kind", choice.mk)?;
    dict.set_item("merge_detail_code", choice.mkd)?;
    dict.set_item("tile", choice.tile)?;
    dict.set_item("fft_tile", choice.fft_tile)?;
    dict.set_item("fft_radix", choice.fft_radix)?;
    dict.set_item("fft_segments", choice.fft_segments)?;
    if let Some(window) = choice.latency_window {
        let latency = PyDict::new(py);
        latency.set_item("target", window.target)?;
        latency.set_item("lower", window.lower)?;
        latency.set_item("upper", window.upper)?;
        latency.set_item("min_lane", window.min_lane)?;
        latency.set_item("max_lane", window.max_lane)?;
        latency.set_item("slack", window.slack)?;
        latency.set_item("stride", window.stride)?;
        dict.set_item("latency_window", latency)?;
    } else {
        dict.set_item("latency_window", py.None())?;
    }
    Ok(dict.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (*, expiry_seconds=None, expiry_milliseconds=None, expiry_at_seconds=None, expiry_at_milliseconds=None, keep_ttl=false, persist=false, condition="always"))]
fn kv_json_set_options(
    py: Python<'_>,
    expiry_seconds: Option<u64>,
    expiry_milliseconds: Option<u64>,
    expiry_at_seconds: Option<u64>,
    expiry_at_milliseconds: Option<u64>,
    keep_ttl: bool,
    persist: bool,
    condition: &str,
) -> PyResult<PyObject> {
    let condition = condition_label(condition)?;
    let expiry = selected_expiry(
        expiry_seconds,
        expiry_milliseconds,
        expiry_at_seconds,
        expiry_at_milliseconds,
    )?;
    validate_json_set_options(expiry, keep_ttl, persist)?;
    let dict = PyDict::new(py);
    if let Some((kind, keyword, value)) = expiry {
        let expiry_dict = PyDict::new(py);
        expiry_dict.set_item("kind", kind)?;
        expiry_dict.set_item("keyword", keyword)?;
        expiry_dict.set_item("value", value)?;
        dict.set_item("expiry", expiry_dict)?;
    } else {
        dict.set_item("expiry", py.None())?;
    }
    dict.set_item("keep_ttl", keep_ttl)?;
    dict.set_item("persist", persist)?;
    dict.set_item("condition", condition)?;
    let fragments = PyList::empty(py);
    append_option_fragments(py, &fragments, expiry, keep_ttl, persist, condition)?;
    dict.set_item("fragments", fragments)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "kv-redis")]
fn kv_err_to_py(err: st_kv::KvErr) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[cfg(feature = "kv-redis")]
fn json_u64(value: &JsonValue, field: &str) -> PyResult<u64> {
    value
        .as_u64()
        .ok_or_else(|| PyValueError::new_err(format!("{field} must be an unsigned integer")))
}

#[cfg(feature = "kv-redis")]
fn parse_redis_json_set_options(
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<st_kv::JsonSetOptions> {
    let Some(options) = options else {
        return Ok(st_kv::JsonSetOptions::new());
    };
    if options.is_none() {
        return Ok(st_kv::JsonSetOptions::new());
    }
    let json = py_to_json(options)?;
    let object = json.as_object().ok_or_else(|| {
        PyValueError::new_err("options must be a mapping returned by kv_json_set_options")
    })?;
    let mut out = st_kv::JsonSetOptions::new();
    if let Some(expiry) = object.get("expiry") {
        if !expiry.is_null() {
            let expiry_obj = expiry.as_object().ok_or_else(|| {
                PyValueError::new_err("options['expiry'] must be null or a mapping")
            })?;
            let kind = expiry_obj
                .get("kind")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| PyValueError::new_err("options['expiry']['kind'] is required"))?;
            let value = expiry_obj
                .get("value")
                .map(|value| json_u64(value, "options['expiry']['value']"))
                .transpose()?
                .ok_or_else(|| PyValueError::new_err("options['expiry']['value'] is required"))?;
            out = match kind {
                "seconds" => out.with_expiry(st_kv::JsonExpiry::seconds(value)),
                "milliseconds" => out.with_expiry(st_kv::JsonExpiry::milliseconds(value)),
                "at_seconds" => out.with_expiry(st_kv::JsonExpiry::at_seconds(value)),
                "at_milliseconds" => out.with_expiry(st_kv::JsonExpiry::at_milliseconds(value)),
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported expiry kind '{other}'"
                    )))
                }
            };
        }
    }
    if object
        .get("keep_ttl")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false)
    {
        out = out.keep_ttl();
    }
    if object
        .get("persist")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false)
    {
        out = out.persist();
    }
    let condition = object
        .get("condition")
        .and_then(JsonValue::as_str)
        .unwrap_or("always");
    out = match condition_label(condition)? {
        "nx" => out.nx(),
        "xx" => out.xx(),
        _ => out,
    };
    Ok(out)
}

#[cfg(feature = "kv-redis")]
fn parse_choice(choice: &Bound<'_, PyAny>) -> PyResult<st_kv::Choice> {
    let json = py_to_json(choice)?;
    serde_json::from_value(json).map_err(|err| {
        PyValueError::new_err(format!("choice must contain st-kv Choice fields: {err}"))
    })
}

#[cfg(feature = "kv-redis")]
#[pyfunction]
#[pyo3(signature = (url, key, value, options=None))]
fn kv_redis_set_json(
    url: &str,
    key: &str,
    value: &Bound<'_, PyAny>,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let value = py_to_json(value)?;
    let options = parse_redis_json_set_options(options)?;
    st_kv::redis_set_json_with_options(url, key, &value, &options).map_err(kv_err_to_py)
}

#[cfg(not(feature = "kv-redis"))]
#[pyfunction]
#[pyo3(signature = (url, key, value, options=None))]
fn kv_redis_set_json(
    url: &str,
    key: &str,
    value: &Bound<'_, PyAny>,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let _ = (url, key, value, options);
    Err(kv_unavailable())
}

#[cfg(feature = "kv-redis")]
#[pyfunction]
fn kv_redis_get_json(py: Python<'_>, url: &str, key: &str) -> PyResult<PyObject> {
    match st_kv::redis_get_json::<JsonValue>(url, key).map_err(kv_err_to_py)? {
        Some(value) => json_to_py(py, &value),
        None => Ok(py.None()),
    }
}

#[cfg(not(feature = "kv-redis"))]
#[pyfunction]
fn kv_redis_get_json(py: Python<'_>, url: &str, key: &str) -> PyResult<PyObject> {
    let _ = (py, url, key);
    Err(kv_unavailable())
}

#[cfg(feature = "kv-redis")]
#[pyfunction]
#[pyo3(signature = (url, key, choice, options=None))]
fn kv_redis_set_choice(
    url: &str,
    key: &str,
    choice: &Bound<'_, PyAny>,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let choice = parse_choice(choice)?;
    let options = parse_redis_json_set_options(options)?;
    st_kv::redis_set_choice_with_options(url, key, &choice, &options).map_err(kv_err_to_py)
}

#[cfg(not(feature = "kv-redis"))]
#[pyfunction]
#[pyo3(signature = (url, key, choice, options=None))]
fn kv_redis_set_choice(
    url: &str,
    key: &str,
    choice: &Bound<'_, PyAny>,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let _ = (url, key, choice, options);
    Err(kv_unavailable())
}

#[cfg(feature = "kv-redis")]
#[pyfunction]
fn kv_redis_get_choice(py: Python<'_>, url: &str, key: &str) -> PyResult<PyObject> {
    match st_kv::redis_get_choice(url, key).map_err(kv_err_to_py)? {
        Some(choice) => {
            let value = serde_json::to_value(choice).map_err(|err| {
                PyValueError::new_err(format!("failed to serialize st-kv Choice: {err}"))
            })?;
            json_to_py(py, &value)
        }
        None => Ok(py.None()),
    }
}

#[cfg(not(feature = "kv-redis"))]
#[pyfunction]
fn kv_redis_get_choice(py: Python<'_>, url: &str, key: &str) -> PyResult<PyObject> {
    let _ = (py, url, key);
    Err(kv_unavailable())
}

#[cfg(feature = "kv-redis")]
#[pyfunction]
#[pyo3(signature = (url, key, choice, max_len=None))]
fn kv_redis_push_choice(
    url: &str,
    key: &str,
    choice: &Bound<'_, PyAny>,
    max_len: Option<usize>,
) -> PyResult<usize> {
    let choice = parse_choice(choice)?;
    st_kv::redis_push_choice(url, key, &choice, max_len).map_err(kv_err_to_py)
}

#[cfg(not(feature = "kv-redis"))]
#[pyfunction]
#[pyo3(signature = (url, key, choice, max_len=None))]
fn kv_redis_push_choice(
    url: &str,
    key: &str,
    choice: &Bound<'_, PyAny>,
    max_len: Option<usize>,
) -> PyResult<usize> {
    let _ = (url, key, choice, max_len);
    Err(kv_unavailable())
}

#[cfg(feature = "kv-redis")]
#[pyfunction]
#[pyo3(signature = (url, key, start=0, stop=-1))]
fn kv_redis_lrange_choice(
    py: Python<'_>,
    url: &str,
    key: &str,
    start: isize,
    stop: isize,
) -> PyResult<PyObject> {
    let choices = st_kv::redis_lrange_choice(url, key, start, stop).map_err(kv_err_to_py)?;
    let value = serde_json::to_value(choices)
        .map_err(|err| PyValueError::new_err(format!("failed to serialize choices: {err}")))?;
    json_to_py(py, &value)
}

#[cfg(not(feature = "kv-redis"))]
#[pyfunction]
#[pyo3(signature = (url, key, start=0, stop=-1))]
fn kv_redis_lrange_choice(
    py: Python<'_>,
    url: &str,
    key: &str,
    start: isize,
    stop: isize,
) -> PyResult<PyObject> {
    let _ = (py, url, key, start, stop);
    Err(kv_unavailable())
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "kv")?;
    module.add(
        "__doc__",
        "Key-value persistence helpers for planner choices and JSON payloads",
    )?;
    module.add_function(wrap_pyfunction!(kv_redis_available, &module)?)?;
    module.add_function(wrap_pyfunction!(kv_choice_schema_fields, &module)?)?;
    module.add_function(wrap_pyfunction!(kv_rank_choice_key, &module)?)?;
    module.add_function(wrap_pyfunction!(kv_choice_key_from_rank_plan, &module)?)?;
    module.add_function(wrap_pyfunction!(kv_choice_from_rank_plan, &module)?)?;
    module.add_function(wrap_pyfunction!(kv_json_set_options, &module)?)?;
    module.add_function(wrap_pyfunction!(kv_redis_set_json, &module)?)?;
    module.add_function(wrap_pyfunction!(kv_redis_get_json, &module)?)?;
    module.add_function(wrap_pyfunction!(kv_redis_set_choice, &module)?)?;
    module.add_function(wrap_pyfunction!(kv_redis_get_choice, &module)?)?;
    module.add_function(wrap_pyfunction!(kv_redis_push_choice, &module)?)?;
    module.add_function(wrap_pyfunction!(kv_redis_lrange_choice, &module)?)?;
    module.add(
        "__all__",
        vec![
            "kv_redis_available",
            "kv_choice_schema_fields",
            "kv_rank_choice_key",
            "kv_choice_key_from_rank_plan",
            "kv_choice_from_rank_plan",
            "kv_json_set_options",
            "kv_redis_set_json",
            "kv_redis_get_json",
            "kv_redis_set_choice",
            "kv_redis_get_choice",
            "kv_redis_push_choice",
            "kv_redis_lrange_choice",
        ],
    )?;
    parent.add_submodule(&module)?;
    parent.add_function(wrap_pyfunction!(kv_redis_available, parent)?)?;
    parent.add_function(wrap_pyfunction!(kv_choice_schema_fields, parent)?)?;
    parent.add_function(wrap_pyfunction!(kv_rank_choice_key, parent)?)?;
    parent.add_function(wrap_pyfunction!(kv_choice_key_from_rank_plan, parent)?)?;
    parent.add_function(wrap_pyfunction!(kv_choice_from_rank_plan, parent)?)?;
    parent.add_function(wrap_pyfunction!(kv_json_set_options, parent)?)?;
    parent.add_function(wrap_pyfunction!(kv_redis_set_json, parent)?)?;
    parent.add_function(wrap_pyfunction!(kv_redis_get_json, parent)?)?;
    parent.add_function(wrap_pyfunction!(kv_redis_set_choice, parent)?)?;
    parent.add_function(wrap_pyfunction!(kv_redis_get_choice, parent)?)?;
    parent.add_function(wrap_pyfunction!(kv_redis_push_choice, parent)?)?;
    parent.add_function(wrap_pyfunction!(kv_redis_lrange_choice, parent)?)?;
    Ok(())
}
