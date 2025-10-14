use pyo3::prelude::*;
use pyo3::types::PyDict;
use st_core::backend::device_caps::DeviceCaps;
use st_core::backend::unison_heuristics::RankKind;
use st_core::ops::rank_entry::plan_rank;

fn parse_kind(kind: &str) -> PyResult<RankKind> {
    match kind.to_ascii_lowercase().as_str() {
        "topk" | "top" => Ok(RankKind::TopK),
        "midk" | "mid" => Ok(RankKind::MidK),
        "bottomk" | "bottom" => Ok(RankKind::BottomK),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported rank kind: {}",
            other
        ))),
    }
}

fn caps_for(device: Option<&str>) -> DeviceCaps {
    match device.map(|d| d.to_ascii_lowercase()) {
        Some(ref name) if name == "cuda" => DeviceCaps::cuda(32, 1024, Some(96 * 1024)),
        Some(ref name) if name == "hip" => DeviceCaps::hip(32, 1024, Some(64 * 1024)),
        Some(ref name) if name == "cpu" => DeviceCaps::cpu(),
        Some(ref name) if name == "mps" => DeviceCaps::wgpu(32, true, 256),
        Some(ref name) if name == "wgpu" => DeviceCaps::wgpu(32, true, 256),
        Some(ref name) if name == "auto" => DeviceCaps::wgpu(32, true, 256),
        Some(ref name) if name == "hip-real" => DeviceCaps::hip(32, 1024, Some(64 * 1024)),
        _ => DeviceCaps::wgpu(32, true, 256),
    }
}

fn choice_dict<'py>(
    py: Python<'py>,
    plan: &st_core::ops::rank_entry::RankPlan,
) -> PyResult<&'py PyDict> {
    let choice = PyDict::new(py);
    choice.set_item("use_2ce", plan.choice.use_2ce)?;
    choice.set_item("workgroup", plan.choice.wg)?;
    choice.set_item("kl", plan.choice.kl)?;
    choice.set_item("channel_stride", plan.choice.ch)?;
    choice.set_item("merge_kind", plan.choice.mk)?;
    choice.set_item("merge_detail", plan.choice.mkd)?;
    choice.set_item("tile", plan.choice.tile)?;
    choice.set_item("compaction_tile", plan.choice.ctile)?;
    Ok(choice)
}

/// Inspect the unified heuristics for the requested rank family.
#[pyfunction]
#[pyo3(signature = (kind, rows, cols, k, device=None))]
fn plan(
    py: Python<'_>,
    kind: &str,
    rows: u32,
    cols: u32,
    k: u32,
    device: Option<&str>,
) -> PyResult<PyObject> {
    let rank_kind = parse_kind(kind)?;
    let caps = caps_for(device);
    let plan = plan_rank(rank_kind, rows, cols, k, caps);

    let out = PyDict::new(py);
    out.set_item("kind", kind.to_ascii_lowercase())?;
    out.set_item("rows", rows)?;
    out.set_item("cols", cols)?;
    out.set_item("k", k)?;
    out.set_item("choice", choice_dict(py, &plan)?)?;
    Ok(out.into())
}

/// Convenience helper for the TopK family.
#[pyfunction]
#[pyo3(signature = (rows, cols, k, device=None))]
fn plan_topk(
    py: Python<'_>,
    rows: u32,
    cols: u32,
    k: u32,
    device: Option<&str>,
) -> PyResult<PyObject> {
    plan(py, "topk", rows, cols, k, device)
}

/// Return a basic capability template for the given device string.
#[pyfunction]
#[pyo3(signature = (device=None))]
fn describe_device(py: Python<'_>, device: Option<&str>) -> PyResult<PyObject> {
    let caps = caps_for(device);
    let out = PyDict::new(py);
    out.set_item("lane_width", caps.lane_width)?;
    out.set_item("max_workgroup", caps.max_workgroup)?;
    out.set_item("subgroup", caps.subgroup)?;
    out.set_item(
        "shared_mem_per_workgroup",
        caps.shared_mem_per_workgroup.map(|v| v as usize),
    )?;
    Ok(out.into())
}

/// SpiralTorch Python module.
#[pymodule]
fn spiraltorch(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(plan_topk, m)?)?;
    m.add_function(wrap_pyfunction!(describe_device, m)?)?;

    m.add("__all__", vec!["plan", "plan_topk", "describe_device"])?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Provide a tiny doc string that highlights the zero-shim approach.
    m.add(
        "__doc__",
        "Rust-first heuristics interface for SpiralTorch.",
    )?;

    Ok(())
}
