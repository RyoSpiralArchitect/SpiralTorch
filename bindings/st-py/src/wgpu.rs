use crate::planner::PyRankPlan;
#[cfg(not(feature = "wgpu"))]
use pyo3::exceptions::PyNotImplementedError;
#[cfg(feature = "wgpu")]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
#[cfg(feature = "wgpu")]
use pyo3::types::{PyDict, PyList};
use pyo3::{wrap_pyfunction, PyRef};

#[cfg(feature = "wgpu")]
fn descriptor_to_dict(
    py: Python<'_>,
    descriptor: &st_backend_wgpu::WgpuKernelDescriptor,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("name", descriptor.name)?;
    dict.set_item("family", descriptor.family)?;
    dict.set_item("operation", descriptor.operation)?;
    dict.set_item("shader", descriptor.shader)?;
    dict.set_item("entry_point", descriptor.entry_point)?;
    dict.set_item("pipeline_label", descriptor.pipeline_label)?;
    dict.set_item("variant", descriptor.variant)?;
    dict.set_item("subgroup", descriptor.subgroup)?;
    dict.set_item("portable", descriptor.portable)?;
    dict.set_item("stages", descriptor.stages.to_vec())?;
    dict.set_item("bindings", descriptor.bindings.to_vec())?;
    dict.set_item("notes", descriptor.notes)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "wgpu")]
fn dispatch_to_dict(
    py: Python<'_>,
    dispatch: st_backend_wgpu::WgpuDispatchGeometry,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item(
        "workgroups",
        (
            dispatch.workgroups.0,
            dispatch.workgroups.1,
            dispatch.workgroups.2,
        ),
    )?;
    dict.set_item("tiles_x", dispatch.tiles_x)?;
    dict.set_item("row_stride", dispatch.row_stride)?;
    dict.set_item("empty", dispatch.empty)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "wgpu")]
fn fft_to_dict(py: Python<'_>, fft: st_backend_wgpu::WgpuFftKernelHints) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("tile_cols", fft.tile_cols)?;
    dict.set_item("radix", fft.radix)?;
    dict.set_item("segments", fft.segments)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "wgpu")]
fn parse_rank_kind(kind: &str) -> PyResult<st_backend_wgpu::WgpuRankKernelKind> {
    match kind.to_ascii_lowercase().as_str() {
        "topk" | "top" => Ok(st_backend_wgpu::WgpuRankKernelKind::TopK),
        "midk" | "middle" | "middlek" => Ok(st_backend_wgpu::WgpuRankKernelKind::MidK),
        "bottomk" | "bottom" => Ok(st_backend_wgpu::WgpuRankKernelKind::BottomK),
        other => Err(PyValueError::new_err(format!(
            "rank kernel kind must be 'topk', 'midk', or 'bottomk', got '{other}'"
        ))),
    }
}

#[cfg(feature = "wgpu")]
fn rank_report_to_dict(
    py: Python<'_>,
    report: st_backend_wgpu::WgpuRankKernelReport,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    let request = PyDict::new(py);
    request.set_item("kind", report.request.kind.as_str())?;
    request.set_item("rows", report.request.rows)?;
    request.set_item("cols", report.request.cols)?;
    request.set_item("k", report.request.k)?;
    request.set_item("subgroup", report.request.subgroup)?;
    request.set_item("use_two_stage", report.request.use_two_stage)?;
    request.set_item("fft_tile", report.request.fft_tile)?;
    request.set_item("fft_radix", report.request.fft_radix)?;
    request.set_item("fft_segments", report.request.fft_segments)?;
    request.set_item("compaction_tile", report.request.compaction_tile)?;
    dict.set_item("request", request)?;
    dict.set_item("primary", descriptor_to_dict(py, report.primary)?)?;
    dict.set_item(
        "fallback",
        match report.fallback {
            Some(descriptor) => descriptor_to_dict(py, descriptor)?,
            None => py.None(),
        },
    )?;
    dict.set_item("dispatch", dispatch_to_dict(py, report.dispatch)?)?;
    dict.set_item("fft", fft_to_dict(py, report.fft)?)?;
    dict.set_item("stages", report.stages.to_vec())?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "wgpu")]
fn softmax_report_to_dict(
    py: Python<'_>,
    report: st_backend_wgpu::WgpuSoftmaxKernelReport,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    let request = PyDict::new(py);
    request.set_item("rows", report.request.rows)?;
    request.set_item("cols", report.request.cols)?;
    request.set_item("subgroup", report.request.subgroup)?;
    request.set_item("hardmax", report.request.hardmax)?;
    request.set_item("mask", report.request.mask)?;
    dict.set_item("request", request)?;
    dict.set_item("primary", descriptor_to_dict(py, report.primary)?)?;
    dict.set_item(
        "fallback",
        match report.fallback {
            Some(descriptor) => descriptor_to_dict(py, descriptor)?,
            None => py.None(),
        },
    )?;
    dict.set_item("dispatch", dispatch_to_dict(py, report.dispatch)?)?;
    dict.set_item("flags", report.flags)?;
    dict.set_item("stages", report.stages.to_vec())?;
    Ok(dict.into_py(py))
}

#[cfg(not(feature = "wgpu"))]
fn wgpu_unavailable() -> PyErr {
    PyNotImplementedError::new_err(
        "WGPU kernel reports require building spiraltorch with the 'wgpu' feature",
    )
}

#[pyfunction]
fn wgpu_kernel_reports_available() -> bool {
    cfg!(feature = "wgpu")
}

#[cfg(feature = "wgpu")]
#[pyfunction]
fn wgpu_kernel_catalog(py: Python<'_>) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for descriptor in st_backend_wgpu::kernel_catalog() {
        list.append(descriptor_to_dict(py, descriptor)?)?;
    }
    Ok(list.into_py(py))
}

#[cfg(not(feature = "wgpu"))]
#[pyfunction]
fn wgpu_kernel_catalog(py: Python<'_>) -> PyResult<PyObject> {
    let _ = py;
    Err(wgpu_unavailable())
}

#[cfg(feature = "wgpu")]
#[pyfunction]
fn wgpu_kernel_descriptor(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    match st_backend_wgpu::kernel_descriptor(name) {
        Some(descriptor) => descriptor_to_dict(py, descriptor),
        None => Ok(py.None()),
    }
}

#[cfg(not(feature = "wgpu"))]
#[pyfunction]
fn wgpu_kernel_descriptor(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    let _ = (py, name);
    Err(wgpu_unavailable())
}

#[cfg(feature = "wgpu")]
#[pyfunction]
#[pyo3(signature = (kind, rows, cols, k, *, subgroup=false, use_two_stage=false, fft_tile=0, fft_radix=0, fft_segments=0, compaction_tile=0))]
fn wgpu_rank_kernel_report(
    py: Python<'_>,
    kind: &str,
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    use_two_stage: bool,
    fft_tile: u32,
    fft_radix: u32,
    fft_segments: u32,
    compaction_tile: u32,
) -> PyResult<PyObject> {
    let request = st_backend_wgpu::WgpuRankKernelRequest {
        kind: parse_rank_kind(kind)?,
        rows,
        cols,
        k,
        subgroup,
        use_two_stage,
        fft_tile,
        fft_radix,
        fft_segments,
        compaction_tile,
    };
    rank_report_to_dict(py, st_backend_wgpu::rank_kernel_report(request))
}

#[cfg(not(feature = "wgpu"))]
#[pyfunction]
#[pyo3(signature = (kind, rows, cols, k, *, subgroup=false, use_two_stage=false, fft_tile=0, fft_radix=0, fft_segments=0, compaction_tile=0))]
fn wgpu_rank_kernel_report(
    py: Python<'_>,
    kind: &str,
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    use_two_stage: bool,
    fft_tile: u32,
    fft_radix: u32,
    fft_segments: u32,
    compaction_tile: u32,
) -> PyResult<PyObject> {
    let _ = (
        py,
        kind,
        rows,
        cols,
        k,
        subgroup,
        use_two_stage,
        fft_tile,
        fft_radix,
        fft_segments,
        compaction_tile,
    );
    Err(wgpu_unavailable())
}

#[cfg(feature = "wgpu")]
#[pyfunction]
fn wgpu_kernel_report_from_rank_plan(
    py: Python<'_>,
    plan: PyRef<'_, PyRankPlan>,
) -> PyResult<PyObject> {
    let plan = plan.plan();
    let request = st_backend_wgpu::WgpuRankKernelRequest {
        kind: parse_rank_kind(plan.kind.as_str())?,
        rows: plan.rows,
        cols: plan.cols,
        k: plan.k,
        subgroup: plan.choice.subgroup,
        use_two_stage: plan.choice.use_2ce,
        fft_tile: plan.choice.fft_tile,
        fft_radix: plan.choice.fft_radix,
        fft_segments: plan.choice.fft_segments,
        compaction_tile: plan.choice.ctile,
    };
    rank_report_to_dict(py, st_backend_wgpu::rank_kernel_report(request))
}

#[cfg(not(feature = "wgpu"))]
#[pyfunction]
fn wgpu_kernel_report_from_rank_plan(
    py: Python<'_>,
    plan: PyRef<'_, PyRankPlan>,
) -> PyResult<PyObject> {
    let _ = (py, plan);
    Err(wgpu_unavailable())
}

#[cfg(feature = "wgpu")]
#[pyfunction]
#[pyo3(signature = (rows, cols, *, subgroup=false, hardmax=false, mask=false))]
fn wgpu_softmax_kernel_report(
    py: Python<'_>,
    rows: u32,
    cols: u32,
    subgroup: bool,
    hardmax: bool,
    mask: bool,
) -> PyResult<PyObject> {
    let request = st_backend_wgpu::WgpuSoftmaxKernelRequest {
        rows,
        cols,
        subgroup,
        hardmax,
        mask,
    };
    softmax_report_to_dict(py, st_backend_wgpu::softmax_kernel_report(request))
}

#[cfg(not(feature = "wgpu"))]
#[pyfunction]
#[pyo3(signature = (rows, cols, *, subgroup=false, hardmax=false, mask=false))]
fn wgpu_softmax_kernel_report(
    py: Python<'_>,
    rows: u32,
    cols: u32,
    subgroup: bool,
    hardmax: bool,
    mask: bool,
) -> PyResult<PyObject> {
    let _ = (py, rows, cols, subgroup, hardmax, mask);
    Err(wgpu_unavailable())
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "wgpu")?;
    module.add("__doc__", "WGPU kernel catalog and selection reports")?;
    module.add_function(wrap_pyfunction!(wgpu_kernel_reports_available, &module)?)?;
    module.add_function(wrap_pyfunction!(wgpu_kernel_catalog, &module)?)?;
    module.add_function(wrap_pyfunction!(wgpu_kernel_descriptor, &module)?)?;
    module.add_function(wrap_pyfunction!(wgpu_rank_kernel_report, &module)?)?;
    module.add_function(wrap_pyfunction!(
        wgpu_kernel_report_from_rank_plan,
        &module
    )?)?;
    module.add_function(wrap_pyfunction!(wgpu_softmax_kernel_report, &module)?)?;
    module.add(
        "__all__",
        vec![
            "wgpu_kernel_reports_available",
            "wgpu_kernel_catalog",
            "wgpu_kernel_descriptor",
            "wgpu_rank_kernel_report",
            "wgpu_kernel_report_from_rank_plan",
            "wgpu_softmax_kernel_report",
        ],
    )?;
    parent.add_submodule(&module)?;
    parent.add_function(wrap_pyfunction!(wgpu_kernel_reports_available, parent)?)?;
    parent.add_function(wrap_pyfunction!(wgpu_kernel_catalog, parent)?)?;
    parent.add_function(wrap_pyfunction!(wgpu_kernel_descriptor, parent)?)?;
    parent.add_function(wrap_pyfunction!(wgpu_rank_kernel_report, parent)?)?;
    parent.add_function(wrap_pyfunction!(wgpu_kernel_report_from_rank_plan, parent)?)?;
    parent.add_function(wrap_pyfunction!(wgpu_softmax_kernel_report, parent)?)?;
    Ok(())
}
