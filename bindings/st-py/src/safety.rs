use crate::json::{json_to_py, py_to_json};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::BTreeMap;

use spiral_safety::{
    aggregate_penalty_with, analyse_word_with_options, default_thresholds, existence_load,
    frame_hazard, frame_summary, safe_radius, trainer_penalty_with, AnalysisOptions,
    DirectionQuery, DrlMetrics, FrameState, FrameThreshold, WordState,
};

fn serde_err_to_py(label: &str, err: serde_json::Error) -> PyErr {
    PyValueError::new_err(format!(
        "{label} must match spiral-safety JSON shape: {err}"
    ))
}

fn parse_json_like<T>(value: &Bound<'_, PyAny>, label: &str) -> PyResult<T>
where
    T: DeserializeOwned,
{
    let json = py_to_json(value)?;
    serde_json::from_value(json).map_err(|err| serde_err_to_py(label, err))
}

fn parse_optional_json_like<T>(value: Option<&Bound<'_, PyAny>>, label: &str) -> PyResult<Option<T>>
where
    T: DeserializeOwned,
{
    match value {
        Some(value) if !value.is_none() => parse_json_like(value, label).map(Some),
        _ => Ok(None),
    }
}

fn serialize_to_py<T>(py: Python<'_>, value: &T) -> PyResult<PyObject>
where
    T: Serialize,
{
    let json = serde_json::to_value(value).map_err(|err| {
        PyValueError::new_err(format!("failed to serialize safety payload: {err}"))
    })?;
    json_to_py(py, &json)
}

fn parse_thresholds(
    thresholds: Option<&Bound<'_, PyAny>>,
) -> PyResult<BTreeMap<String, FrameThreshold>> {
    Ok(parse_optional_json_like(thresholds, "thresholds")?.unwrap_or_else(default_thresholds))
}

fn parse_direction_queries(
    direction_queries: Option<&Bound<'_, PyAny>>,
) -> PyResult<BTreeMap<String, Vec<DirectionQuery>>> {
    Ok(parse_optional_json_like(direction_queries, "direction_queries")?.unwrap_or_default())
}

#[pyfunction]
fn drl_default_thresholds(py: Python<'_>) -> PyResult<PyObject> {
    serialize_to_py(py, &default_thresholds())
}

#[pyfunction]
#[pyo3(signature = (word, thresholds=None, *, hazard_cut=None, min_radius=0.2, direction_queries=None))]
fn drl_analyse_word(
    py: Python<'_>,
    word: &Bound<'_, PyAny>,
    thresholds: Option<&Bound<'_, PyAny>>,
    hazard_cut: Option<f32>,
    min_radius: f32,
    direction_queries: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let word: WordState = parse_json_like(word, "word")?;
    let thresholds = parse_thresholds(thresholds)?;
    let options = AnalysisOptions {
        hazard_cut,
        min_radius,
        direction_queries: parse_direction_queries(direction_queries)?,
    };
    let metrics = analyse_word_with_options(&word, &thresholds, &options);
    serialize_to_py(py, &metrics)
}

#[pyfunction]
#[pyo3(signature = (word, thresholds=None, *, hazard_cut=None, min_radius=0.2, direction_queries=None))]
fn drl_analyze_word(
    py: Python<'_>,
    word: &Bound<'_, PyAny>,
    thresholds: Option<&Bound<'_, PyAny>>,
    hazard_cut: Option<f32>,
    min_radius: f32,
    direction_queries: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    drl_analyse_word(
        py,
        word,
        thresholds,
        hazard_cut,
        min_radius,
        direction_queries,
    )
}

#[pyfunction]
fn drl_existence_load(word: &Bound<'_, PyAny>) -> PyResult<f32> {
    let word: WordState = parse_json_like(word, "word")?;
    Ok(existence_load(&word))
}

#[pyfunction]
#[pyo3(signature = (word, thresholds=None))]
fn drl_safe_radii(
    py: Python<'_>,
    word: &Bound<'_, PyAny>,
    thresholds: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let word: WordState = parse_json_like(word, "word")?;
    let thresholds = parse_thresholds(thresholds)?;
    serialize_to_py(py, &safe_radius(&word, &thresholds))
}

#[pyfunction]
fn drl_frame_hazard(word: &Bound<'_, PyAny>, frame: &Bound<'_, PyAny>) -> PyResult<f32> {
    let word: WordState = parse_json_like(word, "word")?;
    let hazard = match frame.extract::<String>() {
        Ok(frame_name) => {
            let frame = word.frames.get(&frame_name).ok_or_else(|| {
                PyKeyError::new_err(format!("word has no frame named '{frame_name}'"))
            })?;
            frame_hazard(&word, frame)
        }
        Err(_) => {
            let frame: FrameState = parse_json_like(frame, "frame")?;
            frame_hazard(&word, &frame)
        }
    };
    Ok(hazard)
}

#[pyfunction]
#[pyo3(signature = (metrics, min_radius=0.2))]
fn drl_trainer_penalty(metrics: &Bound<'_, PyAny>, min_radius: f32) -> PyResult<f32> {
    let metrics: DrlMetrics = parse_json_like(metrics, "metrics")?;
    Ok(trainer_penalty_with(&metrics, min_radius))
}

#[pyfunction]
#[pyo3(signature = (metrics, min_radius=0.2))]
fn drl_aggregate_penalty(metrics: &Bound<'_, PyAny>, min_radius: f32) -> PyResult<f32> {
    let metrics: Vec<DrlMetrics> = parse_json_like(metrics, "metrics")?;
    Ok(aggregate_penalty_with(metrics.iter(), min_radius))
}

#[pyfunction]
fn drl_frame_summary(py: Python<'_>, metrics: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let metrics: DrlMetrics = parse_json_like(metrics, "metrics")?;
    serialize_to_py(py, &frame_summary(&metrics))
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "safety")?;
    module.add("__doc__", "Safety and drift-response linguistics helpers")?;
    module.add_function(wrap_pyfunction!(drl_default_thresholds, &module)?)?;
    module.add_function(wrap_pyfunction!(drl_analyse_word, &module)?)?;
    module.add_function(wrap_pyfunction!(drl_analyze_word, &module)?)?;
    module.add_function(wrap_pyfunction!(drl_existence_load, &module)?)?;
    module.add_function(wrap_pyfunction!(drl_safe_radii, &module)?)?;
    module.add_function(wrap_pyfunction!(drl_frame_hazard, &module)?)?;
    module.add_function(wrap_pyfunction!(drl_trainer_penalty, &module)?)?;
    module.add_function(wrap_pyfunction!(drl_aggregate_penalty, &module)?)?;
    module.add_function(wrap_pyfunction!(drl_frame_summary, &module)?)?;
    module.add(
        "__all__",
        vec![
            "drl_default_thresholds",
            "drl_analyse_word",
            "drl_analyze_word",
            "drl_existence_load",
            "drl_safe_radii",
            "drl_frame_hazard",
            "drl_trainer_penalty",
            "drl_aggregate_penalty",
            "drl_frame_summary",
        ],
    )?;
    parent.add_submodule(&module)?;
    parent.add_function(wrap_pyfunction!(drl_default_thresholds, parent)?)?;
    parent.add_function(wrap_pyfunction!(drl_analyse_word, parent)?)?;
    parent.add_function(wrap_pyfunction!(drl_analyze_word, parent)?)?;
    parent.add_function(wrap_pyfunction!(drl_existence_load, parent)?)?;
    parent.add_function(wrap_pyfunction!(drl_safe_radii, parent)?)?;
    parent.add_function(wrap_pyfunction!(drl_frame_hazard, parent)?)?;
    parent.add_function(wrap_pyfunction!(drl_trainer_penalty, parent)?)?;
    parent.add_function(wrap_pyfunction!(drl_aggregate_penalty, parent)?)?;
    parent.add_function(wrap_pyfunction!(drl_frame_summary, parent)?)?;
    Ok(())
}
