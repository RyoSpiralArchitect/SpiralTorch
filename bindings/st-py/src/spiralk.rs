use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyType};
use pyo3::wrap_pyfunction;
use pyo3::Bound;

use crate::planner::PyRankPlan;

use st_core::backend::spiralk_fft::SpiralKFftPlan;
use st_core::theory::maxwell::{MaxwellSpiralKBridge, MaxwellSpiralKHint, MaxwellZPulse};
use st_kdsl::auto::{self, HeuristicHint, WilsonMetrics};
use st_kdsl::{Ctx, Err as KdslErr, Out, SoftRule};

fn kdsl_err_to_py(err: KdslErr) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn canonical_field(field: &str) -> PyResult<&'static str> {
    Ok(match field {
        "use_2ce" | "use_two_stage" => "use_2ce",
        "wg" | "workgroup" => "wg",
        "kl" | "lanes" => "kl",
        "ch" | "channel_stride" => "ch",
        "algo" | "algorithm" => "algo",
        "midk" => "midk",
        "bottomk" => "bottomk",
        "ctile" | "compaction_tile" => "ctile",
        "tile_cols" | "tile" => "tile_cols",
        "radix" => "radix",
        "segments" => "segments",
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown SpiralK field '{other}'"
            )))
        }
    })
}

fn kdsl_out_to_py(py: Python<'_>, out: &Out) -> PyResult<PyObject> {
    fn set_optional<T: ToPyObject>(
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key: &str,
        value: Option<T>,
    ) -> PyResult<()> {
        match value {
            Some(inner) => dict.set_item(key, inner)?,
            None => dict.set_item(key, py.None())?,
        }
        Ok(())
    }

    let hard = PyDict::new_bound(py);
    set_optional(py, &hard, "use_two_stage", out.hard.use_2ce)?;
    set_optional(py, &hard, "workgroup", out.hard.wg)?;
    set_optional(py, &hard, "lanes", out.hard.kl)?;
    set_optional(py, &hard, "channel_stride", out.hard.ch)?;
    set_optional(py, &hard, "algorithm", out.hard.algo)?;
    set_optional(py, &hard, "midk", out.hard.midk)?;
    set_optional(py, &hard, "bottomk", out.hard.bottomk)?;
    set_optional(py, &hard, "compaction_tile", out.hard.ctile)?;
    set_optional(py, &hard, "tile_cols", out.hard.tile_cols)?;
    set_optional(py, &hard, "radix", out.hard.radix)?;
    set_optional(py, &hard, "segments", out.hard.segments)?;

    let soft = PyList::empty_bound(py);
    for rule in &out.soft {
        let entry = PyDict::new_bound(py);
        match rule {
            SoftRule::U2 { val, w } => {
                entry.set_item("field", "use_2ce")?;
                entry.set_item("value", *val)?;
                entry.set_item("weight", *w)?;
            }
            SoftRule::Wg { val, w } => {
                entry.set_item("field", "wg")?;
                entry.set_item("value", *val)?;
                entry.set_item("weight", *w)?;
            }
            SoftRule::Kl { val, w } => {
                entry.set_item("field", "kl")?;
                entry.set_item("value", *val)?;
                entry.set_item("weight", *w)?;
            }
            SoftRule::Ch { val, w } => {
                entry.set_item("field", "ch")?;
                entry.set_item("value", *val)?;
                entry.set_item("weight", *w)?;
            }
            SoftRule::Algo { val, w } => {
                entry.set_item("field", "algo")?;
                entry.set_item("value", *val)?;
                entry.set_item("weight", *w)?;
            }
            SoftRule::Midk { val, w } => {
                entry.set_item("field", "midk")?;
                entry.set_item("value", *val)?;
                entry.set_item("weight", *w)?;
            }
            SoftRule::Bottomk { val, w } => {
                entry.set_item("field", "bottomk")?;
                entry.set_item("value", *val)?;
                entry.set_item("weight", *w)?;
            }
            SoftRule::Ctile { val, w } => {
                entry.set_item("field", "ctile")?;
                entry.set_item("value", *val)?;
                entry.set_item("weight", *w)?;
            }
            SoftRule::TileCols { val, w } => {
                entry.set_item("field", "tile_cols")?;
                entry.set_item("value", *val)?;
                entry.set_item("weight", *w)?;
            }
            SoftRule::Radix { val, w } => {
                entry.set_item("field", "radix")?;
                entry.set_item("value", *val)?;
                entry.set_item("weight", *w)?;
            }
            SoftRule::Segments { val, w } => {
                entry.set_item("field", "segments")?;
                entry.set_item("value", *val)?;
                entry.set_item("weight", *w)?;
            }
        }
        soft.append(entry)?;
    }

    let result = PyDict::new_bound(py);
    result.set_item("hard", hard)?;
    result.set_item("soft", soft)?;
    Ok(result.into())
}

#[pyclass(module = "spiraltorch.spiralk", name = "SpiralKFftPlan")]
pub(crate) struct PySpiralKFftPlan {
    pub(crate) inner: SpiralKFftPlan,
}

impl PySpiralKFftPlan {
    pub(crate) fn from_inner(inner: SpiralKFftPlan) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySpiralKFftPlan {
    #[new]
    #[pyo3(signature = (radix, tile_cols, segments, subgroup))]
    pub fn new(radix: u32, tile_cols: u32, segments: u32, subgroup: bool) -> Self {
        let inner = SpiralKFftPlan {
            radix: radix.clamp(2, 4),
            tile_cols: tile_cols.max(1),
            segments: segments.max(1),
            subgroup,
        };
        Self { inner }
    }

    #[classmethod]
    pub fn from_rank_plan(_cls: &Bound<'_, PyType>, plan: &PyRankPlan) -> PyResult<Self> {
        Ok(Self::from_inner(plan.inner().fft_plan()))
    }

    #[getter]
    pub fn radix(&self) -> u32 {
        self.inner.radix
    }

    #[getter]
    pub fn tile_cols(&self) -> u32 {
        self.inner.tile_cols
    }

    #[getter]
    pub fn segments(&self) -> u32 {
        self.inner.segments
    }

    #[getter]
    pub fn subgroup(&self) -> bool {
        self.inner.subgroup
    }

    pub fn workgroup_size(&self) -> u32 {
        self.inner.workgroup_size()
    }

    pub fn wgsl(&self) -> String {
        self.inner.emit_wgsl()
    }

    pub fn spiralk_hint(&self) -> String {
        self.inner.emit_spiralk_hint()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "SpiralKFftPlan(radix={}, tile_cols={}, segments={}, subgroup={})",
            self.inner.radix, self.inner.tile_cols, self.inner.segments, self.inner.subgroup
        ))
    }
}

#[pyclass(module = "spiraltorch.spiralk", name = "MaxwellSpiralKHint")]
#[derive(Clone)]
pub(crate) struct PyMaxwellSpiralKHint {
    inner: MaxwellSpiralKHint,
}

impl From<MaxwellSpiralKHint> for PyMaxwellSpiralKHint {
    fn from(inner: MaxwellSpiralKHint) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMaxwellSpiralKHint {
    #[getter]
    fn channel(&self) -> &str {
        &self.inner.channel
    }

    #[getter]
    fn blocks(&self) -> u64 {
        self.inner.blocks
    }

    #[getter]
    fn z_score(&self) -> f64 {
        self.inner.z_score
    }

    #[getter]
    fn z_bias(&self) -> f32 {
        self.inner.z_bias
    }

    #[getter]
    fn weight(&self) -> f32 {
        self.inner.weight
    }

    fn script_line(&self) -> String {
        self.inner.script_line()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "MaxwellSpiralKHint(channel='{}', blocks={}, z_score={:.3}, weight={:.3})",
            self.inner.channel, self.inner.blocks, self.inner.z_score, self.inner.weight
        ))
    }
}

#[pyclass(module = "spiraltorch.spiralk", name = "MaxwellSpiralKBridge")]
pub(crate) struct PyMaxwellSpiralKBridge {
    inner: MaxwellSpiralKBridge,
}

#[pymethods]
impl PyMaxwellSpiralKBridge {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: MaxwellSpiralKBridge::new(),
        }
    }

    #[pyo3(signature = (program=None))]
    pub fn set_base_program(&mut self, program: Option<&str>) {
        let inner = std::mem::replace(&mut self.inner, MaxwellSpiralKBridge::new());
        let script = program.unwrap_or("");
        self.inner = inner.with_base_program(script);
    }

    pub fn set_weight_bounds(&mut self, min_weight: f32, max_weight: f32) {
        let inner = std::mem::replace(&mut self.inner, MaxwellSpiralKBridge::new());
        self.inner = inner.with_weight_bounds(min_weight, max_weight);
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.hints().len()
    }

    pub fn push_pulse(
        &mut self,
        channel: &str,
        blocks: u64,
        mean: f64,
        standard_error: f64,
        z_score: f64,
        band_energy: (f32, f32, f32),
        z_bias: f32,
    ) -> PyMaxwellSpiralKHint {
        let pulse = MaxwellZPulse {
            blocks,
            mean,
            standard_error,
            z_score,
            band_energy,
            z_bias,
        };
        self.inner.push_pulse(channel, &pulse).into()
    }

    pub fn hints(&self) -> Vec<PyMaxwellSpiralKHint> {
        self.inner.hints().iter().cloned().map(Into::into).collect()
    }

    pub fn script(&self) -> Option<String> {
        self.inner.script()
    }

    pub fn reset(&mut self) {
        self.inner = MaxwellSpiralKBridge::new();
    }
}

#[pyclass(module = "spiraltorch.spiralk", name = "SpiralKContext")]
#[derive(Clone, Copy)]
pub(crate) struct PySpiralKContext {
    inner: Ctx,
}

#[pymethods]
impl PySpiralKContext {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (rows, cols, k, subgroup, subgroup_capacity, kernel_capacity, tile_cols, radix, segments))]
    pub fn new(
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
        subgroup_capacity: u32,
        kernel_capacity: u32,
        tile_cols: u32,
        radix: u32,
        segments: u32,
    ) -> Self {
        let inner = Ctx {
            r: rows,
            c: cols,
            k,
            sg: subgroup,
            sgc: subgroup_capacity,
            kc: kernel_capacity,
            tile_cols,
            radix,
            segments,
        };
        Self { inner }
    }

    #[getter]
    fn rows(&self) -> u32 {
        self.inner.r
    }

    #[getter]
    fn cols(&self) -> u32 {
        self.inner.c
    }

    #[getter]
    fn k(&self) -> u32 {
        self.inner.k
    }

    #[getter]
    fn subgroup(&self) -> bool {
        self.inner.sg
    }

    #[getter]
    fn subgroup_capacity(&self) -> u32 {
        self.inner.sgc
    }

    #[getter]
    fn kernel_capacity(&self) -> u32 {
        self.inner.kc
    }

    #[getter]
    fn tile_cols(&self) -> u32 {
        self.inner.tile_cols
    }

    #[getter]
    fn radix(&self) -> u32 {
        self.inner.radix
    }

    #[getter]
    fn segments(&self) -> u32 {
        self.inner.segments
    }
}

#[pyclass(module = "spiraltorch.spiralk", name = "SpiralKWilsonMetrics")]
#[derive(Clone, Copy)]
pub(crate) struct PySpiralKWilsonMetrics {
    inner: WilsonMetrics,
}

#[pymethods]
impl PySpiralKWilsonMetrics {
    #[new]
    #[pyo3(signature = (baseline_latency, candidate_latency, wins, trials))]
    pub fn new(baseline_latency: f32, candidate_latency: f32, wins: u32, trials: u32) -> Self {
        let inner = WilsonMetrics {
            baseline_latency,
            candidate_latency,
            wins,
            trials,
        };
        Self { inner }
    }

    #[getter]
    fn baseline_latency(&self) -> f32 {
        self.inner.baseline_latency
    }

    #[getter]
    fn candidate_latency(&self) -> f32 {
        self.inner.candidate_latency
    }

    #[getter]
    fn wins(&self) -> u32 {
        self.inner.wins
    }

    #[getter]
    fn trials(&self) -> u32 {
        self.inner.trials
    }

    fn gain(&self) -> f32 {
        self.inner.gain()
    }
}

#[pyclass(module = "spiraltorch.spiralk", name = "SpiralKHeuristicHint")]
#[derive(Clone)]
pub(crate) struct PySpiralKHeuristicHint {
    inner: HeuristicHint,
}

#[pymethods]
impl PySpiralKHeuristicHint {
    #[new]
    #[pyo3(signature = (field, value_expr, weight, condition_expr))]
    pub fn new(field: &str, value_expr: &str, weight: f32, condition_expr: &str) -> PyResult<Self> {
        let field = canonical_field(field)?;
        let inner = HeuristicHint::new(field, value_expr, weight, condition_expr);
        Ok(Self { inner })
    }

    #[getter]
    fn field(&self) -> &str {
        self.inner.field
    }

    #[getter]
    fn value_expr(&self) -> &str {
        &self.inner.value_expr
    }

    #[getter]
    fn weight_expr(&self) -> &str {
        &self.inner.weight_expr
    }

    #[getter]
    fn condition_expr(&self) -> &str {
        &self.inner.condition_expr
    }
}

#[pyfunction]
pub fn wilson_lower_bound(wins: u32, trials: u32, z: f32) -> f32 {
    auto::wilson_lower_bound(wins, trials, z)
}

#[pyfunction]
#[pyo3(signature = (metrics, min_gain=0.02, min_confidence=0.6))]
pub fn should_rewrite(
    metrics: &PySpiralKWilsonMetrics,
    min_gain: f32,
    min_confidence: f32,
) -> bool {
    auto::should_rewrite(&metrics.inner, min_gain, min_confidence)
}

#[pyfunction]
#[pyo3(signature = (base_src, hints))]
pub fn synthesize_program(
    py: Python<'_>,
    base_src: &str,
    hints: Vec<Py<PySpiralKHeuristicHint>>,
) -> PyResult<String> {
    let hints: Vec<_> = hints
        .into_iter()
        .map(|hint| hint.borrow(py).inner.clone())
        .collect();
    Ok(auto::synthesize_program(base_src, &hints))
}

#[pyfunction]
#[pyo3(signature = (base_src, ctx, metrics, hints, min_gain=0.02, min_confidence=0.6))]
pub fn rewrite_with_wilson(
    py: Python<'_>,
    base_src: &str,
    ctx: &PySpiralKContext,
    metrics: &PySpiralKWilsonMetrics,
    hints: Vec<Py<PySpiralKHeuristicHint>>,
    min_gain: f32,
    min_confidence: f32,
) -> PyResult<(PyObject, String)> {
    let hints_vec: Vec<_> = hints
        .into_iter()
        .map(|hint| hint.borrow(py).inner.clone())
        .collect();
    let (out, script) = auto::rewrite_with_wilson(
        base_src,
        &ctx.inner,
        metrics.inner,
        &hints_vec,
        min_gain,
        min_confidence,
    )
    .map_err(kdsl_err_to_py)?;
    let out_dict = kdsl_out_to_py(py, &out)?;
    Ok((out_dict, script))
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "spiralk")?;
    module.add_class::<PySpiralKFftPlan>()?;
    module.add_class::<PyMaxwellSpiralKBridge>()?;
    module.add_class::<PyMaxwellSpiralKHint>()?;
    module.add_class::<PySpiralKContext>()?;
    module.add_class::<PySpiralKWilsonMetrics>()?;
    module.add_class::<PySpiralKHeuristicHint>()?;
    module.add_function(wrap_pyfunction!(wilson_lower_bound, &module)?)?;
    module.add_function(wrap_pyfunction!(should_rewrite, &module)?)?;
    module.add_function(wrap_pyfunction!(synthesize_program, &module)?)?;
    module.add_function(wrap_pyfunction!(rewrite_with_wilson, &module)?)?;
    module.add(
        "__doc__",
        "SpiralK planners, heuristics, and Maxwell bridges",
    )?;
    module.add(
        "__all__",
        vec![
            "SpiralKFftPlan",
            "MaxwellSpiralKBridge",
            "MaxwellSpiralKHint",
            "SpiralKContext",
            "SpiralKWilsonMetrics",
            "SpiralKHeuristicHint",
            "wilson_lower_bound",
            "should_rewrite",
            "synthesize_program",
            "rewrite_with_wilson",
        ],
    )?;
    parent.add_submodule(&module)?;
    Ok(())
}
