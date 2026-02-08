use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use std::ffi::OsString;
use std::sync::{Mutex, OnceLock};

use st_backend_hip as hip_backend;
#[cfg(feature = "cuda")]
use st_core::backend::cuda_exec::CudaExecutor;
use st_core::backend::device_caps::{BackendKind, DeviceCaps};
#[cfg(feature = "hip")]
use st_core::backend::hip_exec::HipExecutor;
#[cfg(any(feature = "cuda", feature = "hip"))]
use st_core::backend::rankk_launch::LaunchBuffers;
#[cfg(feature = "cuda")]
use st_core::backend::rankk_launch::with_launch_buffers_cuda;
#[cfg(feature = "hip")]
use st_core::backend::rankk_launch::with_launch_buffers_hip;
use st_core::backend::unison_heuristics::RankKind;
#[cfg(any(feature = "cuda", feature = "hip"))]
use st_core::ops::rank_entry::execute_rank;

#[cfg(feature = "kdsl")]
use crate::json::json_to_py;
#[cfg(feature = "kdsl")]
use crate::spiralk::{spiralk_err_to_py, spiralk_out_to_dict, PySpiralKContext};
use st_core::ops::rank_entry::{plan_rank, RankPlan};
#[cfg(feature = "kdsl")]
use st_kdsl::{self, Ctx as SpiralKCtx, Hard as SpiralKHard};

#[pyclass(module = "spiraltorch", name = "RankPlan")]
pub(crate) struct PyRankPlan {
    inner: RankPlan,
    kind_override: Option<&'static str>,
}

impl PyRankPlan {
    pub(crate) fn from_plan(inner: RankPlan) -> Self {
        Self {
            inner,
            kind_override: None,
        }
    }

    pub(crate) fn from_plan_with_override(inner: RankPlan, kind: Option<&'static str>) -> Self {
        Self {
            inner,
            kind_override: kind,
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

#[cfg(feature = "kdsl")]
fn spiralk_ctx_from_plan(plan: &RankPlan) -> SpiralKCtx {
    let subgroup_capacity = if plan.choice.subgroup {
        plan.choice.kl.max(1)
    } else {
        1
    };
    let kernel_capacity = if plan.k <= 1_024 {
        1
    } else if plan.k <= 16_384 {
        2
    } else {
        3
    };
    let tile_cols = if plan.choice.fft_tile != 0 {
        plan.choice.fft_tile
    } else {
        let tiles = plan.cols.max(1).div_ceil(256);
        tiles.max(1) * 256
    };
    let radix = if plan.choice.fft_radix != 0 {
        plan.choice.fft_radix
    } else if plan.k.is_power_of_two() {
        4
    } else {
        2
    };
    let segments = if plan.choice.fft_segments != 0 {
        plan.choice.fft_segments
    } else if plan.cols > 131_072 {
        4
    } else if plan.cols > 32_768 {
        2
    } else {
        1
    };
    SpiralKCtx {
        r: plan.rows,
        c: plan.cols,
        k: plan.k,
        sg: plan.choice.subgroup,
        sgc: subgroup_capacity,
        kc: kernel_capacity,
        tile_cols,
        radix,
        segments,
    }
}

#[cfg(feature = "kdsl")]
fn apply_spiralk_overrides(
    choice: &mut st_core::backend::unison_heuristics::Choice,
    hard: &SpiralKHard,
) {
    if let Some(flag) = hard.use_2ce {
        choice.use_2ce = flag;
    }
    if let Some(value) = hard.wg {
        choice.wg = value.max(1);
    }
    if let Some(value) = hard.kl {
        choice.kl = value.max(1);
    }
    if let Some(value) = hard.ch {
        choice.ch = value;
    }
    if let Some(value) = hard.algo {
        choice.mkd = value as u32;
    }
    if let Some(value) = hard.ctile {
        choice.ctile = value;
    }
    if let Some(value) = hard.tile_cols {
        choice.fft_tile = value.max(1);
    }
    if let Some(value) = hard.radix {
        choice.fft_radix = value.max(1);
    }
    if let Some(value) = hard.segments {
        choice.fft_segments = value.max(1);
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

    fn fft_wgsl(&self) -> String {
        self.inner.fft_wgsl()
    }

    fn fft_spiralk_hint(&self) -> String {
        self.inner.fft_spiralk_hint()
    }

    #[cfg(feature = "kdsl")]
    fn spiralk_context(&self, py: Python<'_>) -> PyResult<PyObject> {
        let ctx = spiralk_ctx_from_plan(self.plan());
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
        let ctx = spiralk_ctx_from_plan(self.plan());
        let out = st_kdsl::eval_program(script, &ctx).map_err(spiralk_err_to_py)?;
        let mut updated = self.inner.clone();
        apply_spiralk_overrides(&mut updated.choice, &out.hard);
        Ok(PyRankPlan::from_plan_with_override(
            updated,
            self.kind_override,
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
        let ctx = spiralk_ctx_from_plan(self.plan());
        let (out, trace) = st_kdsl::eval_program_with_trace(script, &ctx, max_events)
            .map_err(spiralk_err_to_py)?;
        let mut updated = self.inner.clone();
        apply_spiralk_overrides(&mut updated.choice, &out.hard);
        let out_obj = spiralk_out_to_dict(py, &out)?;
        let trace_value = serde_json::to_value(&trace)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        let trace_obj = json_to_py(py, &trace_value)?;
        Ok((
            PyRankPlan::from_plan_with_override(updated, self.kind_override),
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

    match raw.to_ascii_lowercase().as_str() {
        "wgpu" | "webgpu" => Ok(BackendKind::Wgpu),
        "cuda" => Ok(BackendKind::Cuda),
        "hip" | "rocm" => Ok(BackendKind::Hip),
        "cpu" => Ok(BackendKind::Cpu),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown backend '{other}', expected 'wgpu', 'cuda', 'hip', or 'cpu'"
        ))),
    }
}

fn backend_label(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Wgpu => "wgpu",
        BackendKind::Cuda => "cuda",
        BackendKind::Hip => "hip",
        BackendKind::Cpu => "cpu",
    }
}

fn parse_rank_kind(kind: &str) -> PyResult<RankKind> {
    match kind.to_ascii_lowercase().as_str() {
        "topk" | "top_k" => Ok(RankKind::TopK),
        "midk" | "mid_k" => Ok(RankKind::MidK),
        "bottomk" | "bottom_k" => Ok(RankKind::BottomK),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown rank kind '{other}', expected 'topk', 'midk', or 'bottomk'"
        ))),
    }
}

fn strict_env_lock() -> std::sync::MutexGuard<'static, ()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("strict env lock available")
}

struct EnvVarRestore {
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvVarRestore {
    fn capture(key: &'static str) -> Self {
        Self {
            key,
            previous: std::env::var_os(key),
        }
    }
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.as_ref() {
            unsafe { std::env::set_var(self.key, previous) };
        } else {
            unsafe { std::env::remove_var(self.key) };
        }
    }
}

fn with_strict_gpu_env<T>(strict: bool, f: impl FnOnce() -> T) -> T {
    let _guard = strict_env_lock();
    let _restore = EnvVarRestore::capture("SPIRALTORCH_STRICT_GPU");
    if strict {
        unsafe { std::env::set_var("SPIRALTORCH_STRICT_GPU", "1") };
    } else {
        unsafe { std::env::set_var("SPIRALTORCH_STRICT_GPU", "0") };
    }
    f()
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
    let caps = build_caps(backend, None, None, None, None);
    let plan = plan_rank(kind, rows, cols, k, caps);
    let input = diagnostic_input(rows, cols);
    let mut out_vals = vec![0.0f32; (rows as usize).saturating_mul(k as usize)];
    let mut out_idx = vec![0i32; (rows as usize).saturating_mul(k as usize)];

    let launch = with_strict_gpu_env(strict, || -> Result<(), String> {
        match backend {
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
                    with_launch_buffers_cuda(buffers, || {
                        execute_rank(&CudaExecutor::default(), &plan)
                    })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("cuda feature is not enabled in this build".to_string())
                }
            }
            BackendKind::Hip => {
                #[cfg(feature = "hip")]
                {
                    if strict && !cfg!(feature = "hip-real") {
                        return Err(
                            "hip-real feature is required for strict HIP GPU probing".to_string()
                        );
                    }
                    let buffers = LaunchBuffers::new(
                        &input,
                        rows,
                        cols,
                        k,
                        out_vals.as_mut_slice(),
                        out_idx.as_mut_slice(),
                    )?;
                    with_launch_buffers_hip(buffers, || {
                        execute_rank(&HipExecutor::default(), &plan)
                    })
                }
                #[cfg(not(feature = "hip"))]
                {
                    Err("hip feature is not enabled in this build".to_string())
                }
            }
            _ => Err("gpu probe supports only 'cuda' or 'hip' backends".to_string()),
        }
    });

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

    let report = PyDict::new_bound(py);
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

fn apply_overrides(
    mut caps: DeviceCaps,
    lane_width: Option<u32>,
    subgroup: Option<bool>,
    max_workgroup: Option<u32>,
    shared_mem_per_workgroup: Option<u32>,
) -> DeviceCaps {
    if let Some(width) = lane_width {
        caps.lane_width = width.max(1);
    }
    if let Some(flag) = subgroup {
        caps.subgroup = flag;
    }
    if let Some(max_wg) = max_workgroup {
        caps.max_workgroup = max_wg.max(1);
    }
    if let Some(shared) = shared_mem_per_workgroup {
        if shared == 0 {
            caps.shared_mem_per_workgroup = None;
        } else {
            caps.shared_mem_per_workgroup = Some(shared);
        }
    }
    caps
}

pub(crate) fn build_caps(
    backend: BackendKind,
    lane_width: Option<u32>,
    subgroup: Option<bool>,
    max_workgroup: Option<u32>,
    shared_mem_per_workgroup: Option<u32>,
) -> DeviceCaps {
    let base = match backend {
        BackendKind::Wgpu => {
            let lanes = lane_width.unwrap_or(32);
            let subgroup_flag = subgroup.unwrap_or(true);
            let max_wg = max_workgroup.unwrap_or(256);
            DeviceCaps::wgpu(lanes, subgroup_flag, max_wg)
        }
        BackendKind::Cuda => DeviceCaps::cuda(
            lane_width.unwrap_or(32),
            max_workgroup.unwrap_or(1024),
            shared_mem_per_workgroup.or(Some(96 * 1024)),
        ),
        BackendKind::Hip => DeviceCaps::hip(
            lane_width.unwrap_or(32),
            max_workgroup.unwrap_or(1024),
            shared_mem_per_workgroup.or(Some(64 * 1024)),
        ),
        BackendKind::Cpu => DeviceCaps::cpu(),
    };

    apply_overrides(
        base,
        lane_width,
        subgroup,
        max_workgroup,
        shared_mem_per_workgroup,
    )
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
    let backend_kind = parse_backend(backend)?;
    let caps = build_caps(
        backend_kind,
        lane_width,
        subgroup,
        max_workgroup,
        shared_mem_per_workgroup,
    );
    let plan = plan_rank(kind, rows, cols, k, caps);
    if kind_override.is_some() {
        Ok(PyRankPlan::from_plan_with_override(plan, kind_override))
    } else {
        Ok(PyRankPlan::from_plan(plan))
    }
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
    let (rank_kind, kind_override) = match kind.to_ascii_lowercase().as_str() {
        "topk" | "top_k" => (RankKind::TopK, None),
        "midk" | "mid_k" => (RankKind::MidK, None),
        "bottomk" | "bottom_k" => (RankKind::BottomK, None),
        "fft" => (RankKind::TopK, Some("fft")),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown rank kind '{other}', expected 'topk', 'midk', 'bottomk', or 'fft'"
            )))
        }
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
    let backend_kind = parse_backend(Some(backend))?;
    let caps = build_caps(
        backend_kind,
        lane_width,
        subgroup,
        max_workgroup,
        shared_mem_per_workgroup,
    );
    let report = PyDict::new_bound(py);
    report.set_item("backend", backend_label(caps.backend))?;
    report.set_item("subgroup", caps.subgroup)?;
    report.set_item("lane_width", caps.lane_width)?;
    report.set_item("max_workgroup", caps.max_workgroup)?;
    match caps.shared_mem_per_workgroup {
        Some(value) => report.set_item("shared_mem_per_workgroup", value)?,
        None => report.set_item("shared_mem_per_workgroup", py.None())?,
    };

    if let Some(requested) = workgroup {
        report.set_item("requested_workgroup", requested)?;
        report.set_item("aligned_workgroup", caps.align_workgroup(requested))?;
        report.set_item("occupancy_score", caps.occupancy_score(requested))?;
    }

    if let Some(total_cols) = cols {
        let tile = caps.preferred_tile(total_cols, tile_hint.unwrap_or(0));
        let compaction = caps.preferred_compaction_tile(total_cols, compaction_hint.unwrap_or(0));
        report.set_item("preferred_tile", tile)?;
        report.set_item("preferred_compaction_tile", compaction)?;
    }

    Ok(report.into_py(py))
}

#[pyfunction]
fn hip_probe(py: Python<'_>) -> PyResult<PyObject> {
    let probe = hip_backend::probe();
    let py_devices = PyList::empty_bound(py);
    for device in probe.devices.iter() {
        let info = PyDict::new_bound(py);
        info.set_item("id", device.id)?;
        info.set_item("name", device.name.to_string())?;
        info.set_item("multi_node", device.multi_node)?;
        py_devices.append(info)?;
    }
    let out = PyDict::new_bound(py);
    out.set_item("available", probe.available)?;
    out.set_item("initialized", probe.initialized)?;
    out.set_item("devices", py_devices)?;
    match &probe.error {
        Some(message) => out.set_item("error", message)?,
        None => out.set_item("error", py.None())?,
    }
    Ok(out.into_py(py))
}

pub(crate) fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyRankPlan>()?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(plan_topk, m)?)?;
    m.add_function(wrap_pyfunction!(describe_device, m)?)?;
    m.add_function(wrap_pyfunction!(hip_probe, m)?)?;
    m.add_function(wrap_pyfunction!(probe_gpu_path, m)?)?;
    let _ = py;
    Ok(())
}
