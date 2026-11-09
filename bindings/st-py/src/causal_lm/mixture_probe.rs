use super::*;
use pyo3::types::{PyDict, PyDictMethods};
use st_nn::causal_lm::CausalLmMixtureProbe;

fn f32le<'py>(py: Python<'py>, values: &[f32]) -> PyResult<Bound<'py, PyBytes>> {
    PyBytes::new_with(py, values.len() * 4, |raw| {
        for (out, v) in raw.chunks_exact_mut(4).zip(values) {
            out.copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    })
}

pub(super) fn to_python(py: Python<'_>, probe: CausalLmMixtureProbe) -> PyResult<Py<PyAny>> {
    let report = PyDict::new(py);
    report.set_item("base_model_identity", probe.base_model_identity)?;
    report.set_item("lora_artifact_id", probe.lora_artifact_id)?;
    report.set_item("geometry_artifact_id", probe.geometry_artifact_id)?;
    report.set_item("site", probe.site)?;
    report.set_item("prompt_tokens", probe.prompt_tokens)?;
    report.set_item("target_id", probe.target_id)?;
    report.set_item("rival_id", probe.rival_id)?;
    report.set_item("mix", probe.mix)?;
    report.set_item("forward_readout", "f32")?;
    report.set_item("encoding", "little_endian_f32")?;
    report.set_item("logits", f32le(py, &probe.logits)?)?;
    let objectives = PyDict::new(py);
    for arm in probe.objectives {
        let obj = PyDict::new(py);
        obj.set_item("loss", arm.loss)?;
        obj.set_item("d_logits", f32le(py, &arm.d_logits)?)?;
        let paths = PyDict::new(py);
        for path in arm.paths {
            let route = PyDict::new(py);
            let boundaries = PyDict::new(py);
            for (name, v) in path.boundaries {
                boundaries.set_item(name, f32le(py, &v)?)?;
            }
            let params = PyDict::new(py);
            for (name, v) in path.parameters {
                params.set_item(name, f32le(py, &v)?)?;
            }
            route.set_item("boundaries", boundaries)?;
            route.set_item("parameters", params)?;
            paths.set_item(path.head_backward.as_str(), route)?;
        }
        obj.set_item("paths", paths)?;
        objectives.set_item(arm.objective, obj)?;
    }
    report.set_item("objectives", objectives)?;
    Ok(report.into_any().unbind())
}
