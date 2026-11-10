//! Bounded little-endian transport only; the intervention and its pullback are native.
use super::*;
use pyo3::types::{PyDict, PyDictMethods};
use st_nn::causal_lm::{CausalLmProjectionPullback, CausalLmProjectionSpec};

pub(super) fn spec(
    target: &Bound<'_, PyAny>,
    alt: &Bound<'_, PyAny>,
    layer: &Bound<'_, PyAny>,
    site: &str,
    basis: &str,
) -> PyResult<(usize, usize, CausalLmProjectionSpec)> {
    Ok((
        dimension(target)?,
        dimension(alt)?,
        CausalLmProjectionSpec {
            layer: dimension(layer)?,
            site: site.parse().map_err(error)?,
            basis: basis.parse().map_err(error)?,
        },
    ))
}
pub(super) fn decode(bytes: &[u8]) -> PyResult<Vec<f32>> {
    bytes
        .chunks_exact(4)
        .map(|b| {
            let x = f32::from_le_bytes(b.try_into().unwrap());
            if x.is_finite() {
                Ok(x)
            } else {
                Err(error("nonfinite projection delta"))
            }
        })
        .collect()
}
pub(super) fn f64le<'py>(py: Python<'py>, values: &[f64]) -> PyResult<Bound<'py, PyBytes>> {
    PyBytes::new_with(py, 8 * values.len(), |b| {
        for (chunk, x) in b.chunks_exact_mut(8).zip(values) {
            chunk.copy_from_slice(&x.to_le_bytes());
        }
        Ok(())
    })
}
pub(super) fn report(py: Python<'_>, r: CausalLmProjectionPullback) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    out.set_item("base_model_identity", r.base_model_identity)?;
    out.set_item("lora_artifact_id", r.lora_artifact_id)?;
    out.set_item("geometry_artifact_id", r.geometry_artifact_id)?;
    out.set_item("readout_accumulator", r.readout_accumulator.as_str())?;
    out.set_item("layer", r.spec.layer)?;
    out.set_item("site", r.spec.site.as_str())?;
    out.set_item("basis", r.spec.basis.as_str())?;
    out.set_item("width", r.bias.len())?;
    out.set_item(
        "contrast",
        crate::json::json_to_py(py, &serde_json::to_value(r.contrast).map_err(error)?)?,
    )?;
    out.set_item("bias_f64le", f64le(py, &r.bias)?)?;
    out.set_item("weight_f64le", f64le(py, &r.weight)?)?;
    Ok(out.into_any().unbind())
}
