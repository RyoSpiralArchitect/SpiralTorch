use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use pyo3::IntoPy;
use spiral_selfsup::{contrastive, masked, ObjectiveError};

fn objective_err(err: ObjectiveError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

#[pyfunction]
#[pyo3(signature = (anchors, positives, temperature=0.07, normalize=true))]
fn info_nce(
    py: Python<'_>,
    anchors: Vec<Vec<f32>>,
    positives: Vec<Vec<f32>>,
    temperature: f32,
    normalize: bool,
) -> PyResult<PyObject> {
    let result = contrastive::info_nce_loss(&anchors, &positives, temperature, normalize)
        .map_err(objective_err)?;

    let dict = PyDict::new_bound(py);
    dict.set_item("loss", result.loss)?;
    let mut rows = Vec::with_capacity(result.batch);
    for row in result.logits.chunks(result.batch) {
        rows.push(row.to_vec());
    }
    dict.set_item("logits", rows)?;
    dict.set_item("labels", result.labels)?;
    dict.set_item("batch", result.batch)?;
    dict.set_item("temperature", temperature)?;
    dict.set_item("normalized", normalize)?;
    Ok(dict.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (predictions, targets, mask_indices))]
fn masked_mse(
    py: Python<'_>,
    predictions: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
    mask_indices: Vec<Vec<usize>>,
) -> PyResult<PyObject> {
    let result =
        masked::masked_mse_loss(&predictions, &targets, &mask_indices).map_err(objective_err)?;

    let dict = PyDict::new_bound(py);
    dict.set_item("loss", result.loss)?;
    dict.set_item("total_masked", result.total_masked)?;
    dict.set_item("per_example", result.per_example)?;
    Ok(dict.into_py(py))
}

pub fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(info_nce, m)?)?;
    m.add_function(wrap_pyfunction!(masked_mse, m)?)?;
    m.add(
        "__doc__",
        "Self-supervised objectives exposed from Rust implementations.",
    )?;
    let _ = py;
    Ok(())
}
