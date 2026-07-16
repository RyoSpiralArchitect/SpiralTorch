use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use pyo3::wrap_pyfunction;
use pyo3::Bound;

use st_backend_hip as hip_backend;
#[cfg(feature = "cuda")]
use st_core::backend::cuda_exec::CudaExecutor;
use st_core::backend::device_caps::{
    BackendKind, DeviceCaps, DeviceCapsError, DeviceCapsOverrides,
};
use st_core::backend::execution_plan::{AcceleratorFallback, ExecutionConfig};
#[cfg(feature = "hip")]
use st_core::backend::hip_exec::HipExecutor;
#[cfg(feature = "cuda")]
use st_core::backend::rankk_launch::with_launch_buffers_cuda;
#[cfg(feature = "hip")]
use st_core::backend::rankk_launch::with_launch_buffers_hip;
#[cfg(any(feature = "cuda", feature = "hip"))]
use st_core::backend::rankk_launch::LaunchBuffers;
use st_core::backend::runtime_probe::{
    backend_feature_enabled, backend_placeholder, backend_real_kernels_compiled,
    backend_runtime_ready, backend_runtime_recommendation, backend_runtime_status,
    build_device_report, mps_probe as core_mps_probe, resolve_backend, BackendResolution,
    DeviceReport,
};
use st_core::backend::unison::RankKind;
#[cfg(any(feature = "cuda", feature = "hip"))]
use st_core::ops::rank_entry::execute_rank;

use crate::json::json_to_py;
#[cfg(feature = "kdsl")]
use crate::spiralk::{spiralk_err_to_py, spiralk_out_to_dict, PySpiralKContext};
use st_core::ops::rank_entry::{try_plan_rank, try_plan_rank_with_config, RankPlan};

#[pyclass(module = "spiraltorch", name = "RankPlan")]
pub(crate) struct PyRankPlan {
    inner: RankPlan,
    kind_override: Option<&'static str>,
    requested_backend: Option<&'static str>,
    effective_backend: Option<&'static str>,
}

impl PyRankPlan {
    #[cfg(feature = "nn")]
    pub(crate) fn from_plan(inner: RankPlan) -> Self {
        Self::from_plan_with_metadata(inner, None, None, None)
    }

    pub(crate) fn from_plan_with_metadata(
        inner: RankPlan,
        kind_override: Option<&'static str>,
        requested_backend: Option<&'static str>,
        effective_backend: Option<&'static str>,
    ) -> Self {
        Self {
            inner,
            kind_override,
            requested_backend,
            effective_backend,
        }
    }

    pub(crate) fn plan(&self) -> &RankPlan {
        &self.inner
    }

    fn merge_kind(&self) -> &'static str {
        match self.inner.choice.mk {
            1 => "shared",
            2 => "warp",
            _ => "bitonic",
        }
    }

    fn merge_detail_label(&self) -> &'static str {
        match self.inner.choice.mkd {
            1 => "heap",
            2 => "kway",
            3 => "bitonic",
            4 => "warp_heap",
            5 => "warp_bitonic",
            _ => "auto",
        }
    }
}

#[pymethods]
impl PyRankPlan {
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind_override
            .unwrap_or_else(|| self.inner.kind.as_str())
    }

    #[getter]
    fn rows(&self) -> u32 {
        self.inner.rows
    }

    #[getter]
    fn cols(&self) -> u32 {
        self.inner.cols
    }

    #[getter]
    fn k(&self) -> u32 {
        self.inner.k
    }

    #[getter]
    fn requested_backend(&self) -> Option<&'static str> {
        self.requested_backend
    }

    #[getter]
    fn effective_backend(&self) -> Option<&'static str> {
        self.effective_backend
    }

    #[getter]
    fn accelerator_fallback(&self) -> &'static str {
        self.inner.accelerator_fallback().as_str()
    }

    #[getter]
    fn tensor_util_wgpu_min_values(&self) -> usize {
        self.inner.execution_config.tensor_util_wgpu_min_values
    }

    #[getter]
    fn workgroup(&self) -> u32 {
        self.inner.choice.wg
    }

    #[getter]
    fn lanes(&self) -> u32 {
        self.inner.choice.kl
    }

    #[getter]
    fn channel_stride(&self) -> u32 {
        self.inner.choice.ch
    }

    #[getter]
    fn merge_strategy(&self) -> &'static str {
        self.merge_kind()
    }

    #[getter]
    fn merge_detail(&self) -> &'static str {
        self.merge_detail_label()
    }

    #[getter]
    fn use_two_stage(&self) -> bool {
        self.inner.choice.use_2ce
    }

    #[getter]
    fn subgroup(&self) -> bool {
        self.inner.choice.subgroup
    }

    #[getter]
    fn tile(&self) -> u32 {
        self.inner.choice.tile
    }

    #[getter]
    fn compaction_tile(&self) -> u32 {
        self.inner.choice.ctile
    }

    #[getter]
    fn fft_tile(&self) -> u32 {
        self.inner.choice.fft_tile
    }

    #[getter]
    fn fft_radix(&self) -> u32 {
        self.inner.choice.fft_radix
    }

    #[getter]
    fn fft_segments(&self) -> u32 {
        self.inner.choice.fft_segments
    }

    fn latency_window(&self) -> Option<(u32, u32, u32, u32, u32, u32, u32)> {
        self.inner.choice.latency_window.map(|window| {
            (
                window.target,
                window.lower,
                window.upper,
                window.min_lane,
                window.max_lane,
                window.slack,
                window.stride,
            )
        })
    }

    fn to_unison_script(&self) -> String {
        self.inner.choice.to_unison_script(self.inner.kind)
    }

    fn fft_wgsl(&self) -> PyResult<String> {
        self.inner
            .fft_wgsl()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    fn fft_spiralk_hint(&self) -> PyResult<String> {
        self.inner
            .fft_spiralk_hint()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    fn fft_dispatch_manifest_json(&self) -> PyResult<String> {
        self.inner
            .fft_plan()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
            .dispatch_manifest_json()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// Returns the shared Rust-owned planning contract with client provenance.
    fn contract(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut value = serde_json::to_value(self.inner.snapshot())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        let object = value
            .as_object_mut()
            .expect("rank-plan snapshot serializes as an object");
        object.insert("execution_client".into(), "python".into());
        object.insert(
            "requested_backend".into(),
            self.requested_backend
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
        );
        object.insert(
            "effective_backend".into(),
            self.effective_backend
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
        );
        json_to_py(py, &value)
    }

    #[cfg(feature = "kdsl")]
    fn spiralk_context(&self, py: Python<'_>) -> PyResult<PyObject> {
        let ctx = self
            .plan()
            .spiralk_context()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let wrapper = PySpiralKContext::from_ctx(ctx);
        Ok(Py::new(py, wrapper)?.into_py(py))
    }

    #[cfg(not(feature = "kdsl"))]
    fn spiralk_context(&self, _py: Python<'_>) -> PyResult<PyObject> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "SpiralK support requires enabling the 'kdsl' feature",
        ))
    }

    #[cfg(feature = "kdsl")]
    fn rewrite_with_spiralk(&self, script: &str) -> PyResult<PyRankPlan> {
        let ctx = self
            .plan()
            .spiralk_context()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let out = st_kdsl::eval_program(script, &ctx).map_err(spiralk_err_to_py)?;
        let updated = self
            .inner
            .try_with_spiralk_hard(&out.hard)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(PyRankPlan::from_plan_with_metadata(
            updated,
            self.kind_override,
            self.requested_backend,
            self.effective_backend,
        ))
    }

    #[cfg(feature = "kdsl")]
    #[pyo3(signature = (script, *, max_events=256))]
    fn rewrite_with_spiralk_explain(
        &self,
        py: Python<'_>,
        script: &str,
        max_events: usize,
    ) -> PyResult<(PyRankPlan, PyObject, PyObject)> {
        let ctx = self
            .plan()
            .spiralk_context()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let (out, trace) = st_kdsl::eval_program_with_trace(script, &ctx, max_events)
            .map_err(spiralk_err_to_py)?;
        let updated = self
            .inner
            .try_with_spiralk_hard(&out.hard)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let out_obj = spiralk_out_to_dict(py, &out)?;
        let trace_value = serde_json::to_value(&trace)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        let trace_obj = json_to_py(py, &trace_value)?;
        Ok((
            PyRankPlan::from_plan_with_metadata(
                updated,
                self.kind_override,
                self.requested_backend,
                self.effective_backend,
            ),
            out_obj,
            trace_obj,
        ))
    }

    #[cfg(not(feature = "kdsl"))]
    fn rewrite_with_spiralk(&self, _script: &str) -> PyResult<PyRankPlan> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "SpiralK support requires enabling the 'kdsl' feature",
        ))
    }

    #[cfg(not(feature = "kdsl"))]
    fn rewrite_with_spiralk_explain(
        &self,
        _py: Python<'_>,
        _script: &str,
        _max_events: usize,
    ) -> PyResult<(PyRankPlan, PyObject, PyObject)> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "SpiralK support requires enabling the 'kdsl' feature",
        ))
    }
}

pub(crate) fn parse_backend(name: Option<&str>) -> PyResult<BackendKind> {
    let raw = name.unwrap_or("wgpu");
    if raw.eq_ignore_ascii_case("auto") {
        return Ok(BackendKind::Wgpu);
    }

    raw.parse::<BackendKind>()
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

fn backend_label(kind: BackendKind) -> &'static str {
    kind.as_str()
}

fn caps_to_pydict(py: Python<'_>, caps: DeviceCaps) -> PyResult<Bound<'_, PyDict>> {
    let report = PyDict::new(py);
    report.set_item("backend", backend_label(caps.backend))?;
    report.set_item("subgroup", caps.subgroup)?;
    report.set_item("lane_width", caps.lane_width)?;
    report.set_item("max_workgroup", caps.max_workgroup)?;
    match caps.shared_mem_per_workgroup {
        Some(value) => report.set_item("shared_mem_per_workgroup", value)?,
        None => report.set_item("shared_mem_per_workgroup", py.None())?,
    };
    Ok(report)
}

fn set_backend_runtime_fields(
    out: &Bound<'_, PyDict>,
    prefix: &str,
    backend: BackendKind,
) -> PyResult<()> {
    out.set_item(
        format!("{prefix}_backend_feature_enabled"),
        backend_feature_enabled(backend),
    )?;
    out.set_item(
        format!("{prefix}_backend_real_kernels_compiled"),
        backend_real_kernels_compiled(backend),
    )?;
    out.set_item(
        format!("{prefix}_backend_placeholder"),
        backend_placeholder(backend),
    )?;
    out.set_item(
        format!("{prefix}_backend_runtime_status"),
        backend_runtime_status(backend),
    )?;
    out.set_item(
        format!("{prefix}_backend_runtime_ready"),
        backend_runtime_ready(backend),
    )?;
    out.set_item(
        format!("{prefix}_backend_runtime_recommendation"),
        backend_runtime_recommendation(backend),
    )?;
    Ok(())
}

fn device_report_to_pydict(py: Python<'_>, report: DeviceReport) -> PyResult<Bound<'_, PyDict>> {
    let out = caps_to_pydict(py, report.caps)?;
    out.set_item("backend", backend_label(report.reported_backend))?;
    out.set_item("requested_backend", backend_label(report.reported_backend))?;
    out.set_item(
        "effective_backend",
        backend_label(report.effective_backend()),
    )?;
    set_backend_runtime_fields(&out, "requested", report.reported_backend)?;
    set_backend_runtime_fields(&out, "effective", report.effective_backend())?;
    out.set_item(
        "runtime_status",
        backend_runtime_status(report.effective_backend()),
    )?;
    out.set_item(
        "runtime_ready",
        backend_runtime_ready(report.effective_backend()),
    )?;
    out.set_item(
        "runtime_recommendation",
        backend_runtime_recommendation(report.effective_backend()),
    )?;

    if let Some(mps_probe) = report.mps_probe {
        out.set_item("status", mps_probe.status().as_str())?;
        out.set_item("feature_enabled", mps_probe.feature_enabled)?;
        out.set_item("platform_supported", mps_probe.platform_supported)?;
        out.set_item("host_class", mps_probe.host_class.as_str())?;
        out.set_item("backend_wired", mps_probe.backend_wired)?;
        out.set_item("placeholder", mps_probe.placeholder())?;
        out.set_item("available", mps_probe.available())?;
        out.set_item("initialized", mps_probe.initialized)?;
        out.set_item(
            "planner_surrogate_backend",
            backend_label(mps_probe.planner_surrogate_backend),
        )?;
        out.set_item("planner_route", mps_probe.planner_route())?;
        out.set_item(
            "recommended_backend",
            backend_label(mps_probe.recommended_backend()),
        )?;
        out.set_item("recommendation", mps_probe.recommendation())?;
        out.set_item("error", mps_probe.error())?;
    }

    if let Some(requested) = report.requested_workgroup {
        out.set_item("requested_workgroup", requested)?;
    }
    if let Some(aligned) = report.aligned_workgroup {
        out.set_item("aligned_workgroup", aligned)?;
    }
    if let Some(score) = report.occupancy_score {
        out.set_item("occupancy_score", score)?;
    }
    if let Some(tile) = report.preferred_tile {
        out.set_item("preferred_tile", tile)?;
    }
    if let Some(tile) = report.preferred_compaction_tile {
        out.set_item("preferred_compaction_tile", tile)?;
    }

    Ok(out)
}

fn parse_backend_for_planner(name: Option<&str>) -> PyResult<BackendResolution> {
    let requested_backend = parse_backend(name)?;
    Ok(resolve_backend(requested_backend))
}

fn parse_rank_kind(kind: &str) -> PyResult<RankKind> {
    kind.parse::<RankKind>()
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

fn diagnostic_input(rows: u32, cols: u32) -> Vec<f32> {
    let mut input = Vec::with_capacity((rows as usize).saturating_mul(cols as usize));
    let cols_center = cols as f32 * 0.5;
    for row in 0..rows {
        for col in 0..cols {
            let centered = col as f32 - cols_center;
            input.push(centered + (row as f32 * 0.125));
        }
    }
    input
}

#[allow(unused_mut, unused_variables)]
fn execute_backend_probe(
    backend: BackendKind,
    kind: RankKind,
    rows: u32,
    cols: u32,
    k: u32,
    strict: bool,
) -> Result<(Vec<f32>, Vec<i32>), String> {
    let caps = build_caps(backend, None, None, None, None).map_err(|error| error.to_string())?;
    let mut execution_config = ExecutionConfig::from_env();
    execution_config.accelerator_fallback = if strict {
        AcceleratorFallback::Forbid
    } else {
        AcceleratorFallback::Allow
    };
    let plan = try_plan_rank_with_config(kind, rows, cols, k, caps, execution_config)
        .map_err(|error| error.to_string())?;
    let input = diagnostic_input(rows, cols);
    let mut out_vals = vec![0.0f32; (rows as usize).saturating_mul(k as usize)];
    let mut out_idx = vec![0i32; (rows as usize).saturating_mul(k as usize)];

    let launch: Result<(), String> = match backend {
        BackendKind::Mps => Err("mps GPU probing is not wired yet".to_string()),
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let buffers = LaunchBuffers::new(
                    &input,
                    rows,
                    cols,
                    k,
                    out_vals.as_mut_slice(),
                    out_idx.as_mut_slice(),
                )?;
                with_launch_buffers_cuda(buffers, || execute_rank(&CudaExecutor::default(), &plan))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("cuda feature is not enabled in this build".to_string())
            }
        }
        BackendKind::Hip => {
            #[cfg(feature = "hip")]
            {
                let buffers = LaunchBuffers::new(
                    &input,
                    rows,
                    cols,
                    k,
                    out_vals.as_mut_slice(),
                    out_idx.as_mut_slice(),
                )?;
                with_launch_buffers_hip(buffers, || execute_rank(&HipExecutor::default(), &plan))
            }
            #[cfg(not(feature = "hip"))]
            {
                Err("hip feature is not enabled in this build".to_string())
            }
        }
        _ => Err("gpu probe supports only 'cuda' or 'hip' backends".to_string()),
    };

    launch.map(|_| (out_vals, out_idx))
}

#[pyfunction]
#[pyo3(signature = (kind="bottomk", *, backend="cuda", rows=2, cols=5, k=2))]
fn probe_gpu_path(
    py: Python<'_>,
    kind: &str,
    backend: &str,
    rows: u32,
    cols: u32,
    k: u32,
) -> PyResult<PyObject> {
    if rows == 0 || cols == 0 || k == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rows, cols, and k must be positive",
        ));
    }
    if k > cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "k must not exceed cols",
        ));
    }

    let rank_kind = parse_rank_kind(kind)?;
    let backend_kind = parse_backend(Some(backend))?;
    if !matches!(backend_kind, BackendKind::Cuda | BackendKind::Hip) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "probe_gpu_path supports only 'cuda' or 'hip' backends",
        ));
    }

    let strict_attempt = execute_backend_probe(backend_kind, rank_kind, rows, cols, k, true);
    let non_strict_attempt = execute_backend_probe(backend_kind, rank_kind, rows, cols, k, false);

    let strict_ok = strict_attempt.is_ok();
    let non_strict_ok = non_strict_attempt.is_ok();
    let used_fallback = !strict_ok && non_strict_ok;

    let report = PyDict::new(py);
    report.set_item("backend", backend_label(backend_kind))?;
    report.set_item("kind", kind.to_ascii_lowercase())?;
    report.set_item("rows", rows)?;
    report.set_item("cols", cols)?;
    report.set_item("k", k)?;
    report.set_item("strict_success", strict_ok)?;
    report.set_item("non_strict_success", non_strict_ok)?;
    report.set_item("gpu_path_available", strict_ok)?;
    report.set_item("used_fallback", used_fallback)?;
    report.set_item("hip_real_enabled", cfg!(feature = "hip-real"))?;

    match strict_attempt {
        Ok((vals, idx)) => {
            report.set_item("strict_values", vals)?;
            report.set_item("strict_indices", idx)?;
            report.set_item("strict_error", py.None())?;
        }
        Err(err) => {
            report.set_item("strict_values", py.None())?;
            report.set_item("strict_indices", py.None())?;
            report.set_item("strict_error", err)?;
        }
    }

    match non_strict_attempt {
        Ok((vals, idx)) => {
            report.set_item("non_strict_values", vals)?;
            report.set_item("non_strict_indices", idx)?;
            report.set_item("non_strict_error", py.None())?;
        }
        Err(err) => {
            report.set_item("non_strict_values", py.None())?;
            report.set_item("non_strict_indices", py.None())?;
            report.set_item("non_strict_error", err)?;
        }
    }

    Ok(report.into_py(py))
}

pub(crate) fn build_caps(
    backend: BackendKind,
    lane_width: Option<u32>,
    subgroup: Option<bool>,
    max_workgroup: Option<u32>,
    shared_mem_per_workgroup: Option<u32>,
) -> Result<DeviceCaps, DeviceCapsError> {
    backend
        .default_caps()
        .try_with_overrides(DeviceCapsOverrides {
            lane_width,
            subgroup,
            max_workgroup,
            shared_mem_per_workgroup,
        })
}

#[allow(clippy::too_many_arguments)]
fn plan_impl(
    kind: RankKind,
    rows: u32,
    cols: u32,
    k: u32,
    backend: Option<&str>,
    lane_width: Option<u32>,
    subgroup: Option<bool>,
    max_workgroup: Option<u32>,
    shared_mem_per_workgroup: Option<u32>,
    kind_override: Option<&'static str>,
) -> PyResult<PyRankPlan> {
    let backend_resolution = parse_backend_for_planner(backend)?;
    let backend_kind = backend_resolution.effective_backend;
    let caps = build_caps(
        backend_kind,
        lane_width,
        subgroup,
        max_workgroup,
        shared_mem_per_workgroup,
    )
    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let plan = try_plan_rank(kind, rows, cols, k, caps)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(PyRankPlan::from_plan_with_metadata(
        plan,
        kind_override,
        Some(backend_label(backend_resolution.reported_backend)),
        Some(backend_label(backend_kind)),
    ))
}

#[pyfunction]
#[pyo3(signature = (kind, rows, cols, k, *, backend=None, lane_width=None, subgroup=None, max_workgroup=None, shared_mem_per_workgroup=None))]
#[allow(clippy::too_many_arguments)]
fn plan(
    kind: &str,
    rows: u32,
    cols: u32,
    k: u32,
    backend: Option<&str>,
    lane_width: Option<u32>,
    subgroup: Option<bool>,
    max_workgroup: Option<u32>,
    shared_mem_per_workgroup: Option<u32>,
) -> PyResult<PyRankPlan> {
    let (rank_kind, kind_override) = if kind.eq_ignore_ascii_case("fft") {
        (RankKind::TopK, Some("fft"))
    } else {
        (parse_rank_kind(kind)?, None)
    };
    plan_impl(
        rank_kind,
        rows,
        cols,
        k,
        backend,
        lane_width,
        subgroup,
        max_workgroup,
        shared_mem_per_workgroup,
        kind_override,
    )
}

#[pyfunction]
#[pyo3(signature = (rows, cols, k, *, backend=None, lane_width=None, subgroup=None, max_workgroup=None, shared_mem_per_workgroup=None))]
#[allow(clippy::too_many_arguments)]
fn plan_topk(
    rows: u32,
    cols: u32,
    k: u32,
    backend: Option<&str>,
    lane_width: Option<u32>,
    subgroup: Option<bool>,
    max_workgroup: Option<u32>,
    shared_mem_per_workgroup: Option<u32>,
) -> PyResult<PyRankPlan> {
    plan_impl(
        RankKind::TopK,
        rows,
        cols,
        k,
        backend,
        lane_width,
        subgroup,
        max_workgroup,
        shared_mem_per_workgroup,
        None,
    )
}

#[pyfunction]
#[pyo3(signature = (backend="wgpu", *, lane_width=None, subgroup=None, max_workgroup=None, shared_mem_per_workgroup=None, workgroup=None, cols=None, tile_hint=None, compaction_hint=None))]
#[allow(clippy::too_many_arguments)]
fn describe_device(
    py: Python<'_>,
    backend: &str,
    lane_width: Option<u32>,
    subgroup: Option<bool>,
    max_workgroup: Option<u32>,
    shared_mem_per_workgroup: Option<u32>,
    workgroup: Option<u32>,
    cols: Option<u32>,
    tile_hint: Option<u32>,
    compaction_hint: Option<u32>,
) -> PyResult<PyObject> {
    let backend_resolution = parse_backend_for_planner(Some(backend))?;
    let backend_kind = backend_resolution.effective_backend;
    let caps = build_caps(
        backend_kind,
        lane_width,
        subgroup,
        max_workgroup,
        shared_mem_per_workgroup,
    )
    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let report = build_device_report(
        backend_resolution.reported_backend,
        caps,
        backend_resolution.mps_probe,
        workgroup,
        cols,
        tile_hint,
        compaction_hint,
    );

    Ok(device_report_to_pydict(py, report)?.into_py(py))
}

#[pyfunction]
fn hip_probe(py: Python<'_>) -> PyResult<PyObject> {
    let probe = hip_backend::probe();
    let py_devices = PyList::empty(py);
    for device in probe.devices.iter() {
        let info = PyDict::new(py);
        info.set_item("id", device.id)?;
        info.set_item("name", device.name.to_string())?;
        info.set_item("multi_node", device.multi_node)?;
        py_devices.append(info)?;
    }
    let out = PyDict::new(py);
    out.set_item("available", probe.available)?;
    out.set_item("initialized", probe.initialized)?;
    out.set_item("devices", py_devices)?;
    match &probe.error {
        Some(message) => out.set_item("error", message)?,
        None => out.set_item("error", py.None())?,
    }
    Ok(out.into_py(py))
}

#[pyfunction]
fn mps_probe(py: Python<'_>) -> PyResult<PyObject> {
    let probe = core_mps_probe();

    let out = PyDict::new(py);
    out.set_item("backend", "mps")?;
    out.set_item("status", probe.status().as_str())?;
    out.set_item("feature_enabled", probe.feature_enabled)?;
    out.set_item("platform_supported", probe.platform_supported)?;
    out.set_item("host_class", probe.host_class.as_str())?;
    out.set_item("backend_wired", probe.backend_wired)?;
    out.set_item("placeholder", probe.placeholder())?;
    out.set_item("available", probe.available())?;
    out.set_item("initialized", probe.initialized)?;
    out.set_item("host_os", probe.host_os())?;
    out.set_item("host_arch", probe.host_arch())?;
    out.set_item(
        "planner_surrogate_backend",
        backend_label(probe.planner_surrogate_backend),
    )?;
    out.set_item("planner_route", probe.planner_route())?;
    out.set_item("planner_caps", caps_to_pydict(py, probe.planner_caps)?)?;
    out.set_item(
        "recommended_backend",
        backend_label(probe.recommended_backend()),
    )?;
    out.set_item("recommendation", probe.recommendation())?;
    out.set_item("devices", PyList::empty(py))?;
    out.set_item("error", probe.error())?;
    Ok(out.into_py(py))
}

pub(crate) fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyRankPlan>()?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(plan_topk, m)?)?;
    m.add_function(wrap_pyfunction!(describe_device, m)?)?;
    m.add_function(wrap_pyfunction!(hip_probe, m)?)?;
    m.add_function(wrap_pyfunction!(mps_probe, m)?)?;
    m.add_function(wrap_pyfunction!(probe_gpu_path, m)?)?;
    let _ = py;
    Ok(())
}
