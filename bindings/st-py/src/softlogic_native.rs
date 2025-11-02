use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;
use st_softlogic::spiralk::ir::{
    Backend, Document, FeedbackBlock, Layout, Precision, RefractBlock, RefractOpPolicy, SyncBlock,
    TargetSpec,
};
use st_softlogic::spiralk::parse::parse_spiralk;

pub fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "_softlogic_native")?;
    module.add_function(wrap_pyfunction!(parse_document, &module)?)?;
    module.add("__doc__", "Native SpiralK helpers")?;
    m.add_submodule(&module)?;
    Ok(())
}

#[pyfunction]
fn parse_document(py: Python<'_>, src: &str) -> PyResult<PyObject> {
    let document = parse_spiralk(src).map_err(|err| PyValueError::new_err(err.to_string()))?;
    document_to_py(py, &document)
}

fn document_to_py(py: Python<'_>, doc: &Document) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("refracts", refract_blocks_to_py(py, &doc.refracts)?)?;
    out.set_item("syncs", sync_blocks_to_py(py, &doc.syncs)?)?;
    out.set_item("feedbacks", feedback_blocks_to_py(py, &doc.feedbacks)?)?;
    Ok(out.into())
}

fn refract_blocks_to_py(py: Python<'_>, blocks: &[RefractBlock]) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for block in blocks {
        let item = PyDict::new_bound(py);
        item.set_item("name", &block.name)?;
        item.set_item("target", target_to_py(py, &block.target)?)?;
        match block.precision {
            Some(value) => item.set_item("precision", precision_to_str(value))?,
            None => item.set_item("precision", py.None())?,
        }
        match block.layout {
            Some(value) => item.set_item("layout", layout_to_str(value))?,
            None => item.set_item("layout", py.None())?,
        }
        match block.schedule.as_ref() {
            Some(value) => item.set_item("schedule", value)?,
            None => item.set_item("schedule", py.None())?,
        }
        match block.backend {
            Some(value) => item.set_item("backend", backend_to_str(value))?,
            None => item.set_item("backend", py.None())?,
        }
        item.set_item("policies", policies_to_py(py, &block.policies)?)?;
        list.append(&item)?;
    }
    Ok(list.into())
}

fn policies_to_py(py: Python<'_>, policies: &[RefractOpPolicy]) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for policy in policies {
        let item = PyDict::new_bound(py);
        item.set_item("op", &policy.op)?;
        item.set_item("flags", PyList::new_bound(py, &policy.flags))?;
        list.append(&item)?;
    }
    Ok(list.into())
}

fn sync_blocks_to_py(py: Python<'_>, blocks: &[SyncBlock]) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for block in blocks {
        let item = PyDict::new_bound(py);
        item.set_item("name", &block.name)?;
        item.set_item("pairs", PyList::new_bound(py, &block.pairs))?;
        item.set_item("tolerance", block.tolerance)?;
        list.append(&item)?;
    }
    Ok(list.into())
}

fn feedback_blocks_to_py(py: Python<'_>, blocks: &[FeedbackBlock]) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for block in blocks {
        let item = PyDict::new_bound(py);
        item.set_item("name", &block.name)?;
        item.set_item("export", &block.export_path)?;
        item.set_item("metrics", PyList::new_bound(py, &block.metrics))?;
        list.append(&item)?;
    }
    Ok(list.into())
}

fn target_to_py(py: Python<'_>, target: &TargetSpec) -> PyResult<PyObject> {
    let item = PyDict::new_bound(py);
    match target {
        TargetSpec::Graph(name) => {
            item.set_item("kind", "graph")?;
            item.set_item("name", name)?;
        }
        TargetSpec::Prsn(name) => {
            item.set_item("kind", "prsn")?;
            item.set_item("name", name)?;
        }
    }
    Ok(item.into())
}

fn precision_to_str(value: Precision) -> &'static str {
    match value {
        Precision::Fp32 => "fp32",
        Precision::Fp16 => "fp16",
        Precision::Bf16 => "bf16",
    }
}

fn layout_to_str(value: Layout) -> &'static str {
    match value {
        Layout::NHWC => "nhwc",
        Layout::NCHW => "nchw",
        Layout::Blocked => "blocked",
    }
}

fn backend_to_str(value: Backend) -> &'static str {
    match value {
        Backend::WGPU => "WGPU",
        Backend::MPS => "MPS",
        Backend::CUDA => "CUDA",
        Backend::CPU => "CPU",
    }
}
