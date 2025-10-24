use pyo3::prelude::*;
use pyo3::types::PyModule;
#[cfg(feature = "kdsl")]
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;
use pyo3::Bound;
#[cfg(feature = "kdsl")]
use pyo3::PyRef;

use crate::planner::PyRankPlan;

use st_core::backend::spiralk_fft::SpiralKFftPlan;

use st_core::theory::maxwell::{
    required_blocks as required_blocks_rs, MaxwellFingerprint, MaxwellSpiralKBridge,
    MaxwellSpiralKHint, MaxwellZProjector, MaxwellZPulse, MeaningGate, SequentialZ,
};
#[cfg(feature = "kdsl")]
use st_kdsl::{
    auto::{self, HeuristicHint, WilsonMetrics},
    Ctx as SpiralKCtx, Err as SpiralKErr, Out as SpiralKOut, SoftRule as SpiralKSoftRule,
};

#[pyclass(module = "spiraltorch.spiralk", name = "FftPlan")]
pub(crate) struct PySpiralKFftPlan {
    pub(crate) inner: SpiralKFftPlan,
}

#[pymethods]
impl PySpiralKFftPlan {
    #[new]
    pub fn new(radix: u32, tile_cols: u32, segments: u32, subgroup: bool) -> Self {
        let plan = SpiralKFftPlan {
            radix,
            tile_cols,
            segments,
            subgroup,
        };
        Self { inner: plan }
    }

    #[staticmethod]
    pub fn from_rank_plan(plan: &PyRankPlan) -> Self {
        let inner = plan.plan().fft_plan();
        Self { inner }
    }

    #[getter]
    fn radix(&self) -> u32 {
        self.inner.radix
    }

    #[getter]
    fn tile_cols(&self) -> u32 {
        self.inner.tile_cols
    }

    #[getter]
    fn segments(&self) -> u32 {
        self.inner.segments
    }

    #[getter]
    fn subgroup(&self) -> bool {
        self.inner.subgroup
    }

    fn workgroup_size(&self) -> u32 {
        self.inner.workgroup_size()
    }

    fn emit_wgsl(&self) -> String {
        self.inner.emit_wgsl()
    }

    fn emit_spiralk_hint(&self) -> String {
        self.inner.emit_spiralk_hint()
    }
}

#[cfg(feature = "kdsl")]
pub(crate) fn spiralk_err_to_py(err: SpiralKErr) -> PyErr {
    use pyo3::exceptions::PyValueError;
    match err {
        SpiralKErr::Parse(pos) => PyValueError::new_err(format!("parse error at pos {pos}")),
        SpiralKErr::Tok => PyValueError::new_err("unexpected token"),
    }
}

#[cfg(feature = "kdsl")]
fn spiralk_out_to_dict(py: Python<'_>, out: &SpiralKOut) -> PyResult<PyObject> {
    let hard = PyDict::new_bound(py);
    if let Some(flag) = out.hard.use_2ce {
        hard.set_item("use_2ce", flag)?;
    }
    if let Some(value) = out.hard.wg {
        hard.set_item("workgroup", value)?;
    }
    if let Some(value) = out.hard.kl {
        hard.set_item("lanes", value)?;
    }
    if let Some(value) = out.hard.ch {
        hard.set_item("channel_stride", value)?;
    }
    if let Some(value) = out.hard.algo {
        hard.set_item("algo", value)?;
    }
    if let Some(value) = out.hard.midk {
        hard.set_item("midk_mode", value)?;
    }
    if let Some(value) = out.hard.bottomk {
        hard.set_item("bottomk_mode", value)?;
    }
    if let Some(value) = out.hard.ctile {
        hard.set_item("compaction_tile", value)?;
    }
    if let Some(value) = out.hard.tile_cols {
        hard.set_item("tile_cols", value)?;
    }
    if let Some(value) = out.hard.radix {
        hard.set_item("radix", value)?;
    }
    if let Some(value) = out.hard.segments {
        hard.set_item("segments", value)?;
    }

    let soft_rules = PyList::empty_bound(py);
    for rule in &out.soft {
        let rule_dict = PyDict::new_bound(py);
        match rule {
            SpiralKSoftRule::U2 { val, w } => {
                rule_dict.set_item("field", "use_2ce")?;
                rule_dict.set_item("value", *val)?;
                rule_dict.set_item("weight", *w)?;
            }
            SpiralKSoftRule::Wg { val, w } => {
                rule_dict.set_item("field", "workgroup")?;
                rule_dict.set_item("value", *val)?;
                rule_dict.set_item("weight", *w)?;
            }
            SpiralKSoftRule::Kl { val, w } => {
                rule_dict.set_item("field", "lanes")?;
                rule_dict.set_item("value", *val)?;
                rule_dict.set_item("weight", *w)?;
            }
            SpiralKSoftRule::Ch { val, w } => {
                rule_dict.set_item("field", "channel_stride")?;
                rule_dict.set_item("value", *val)?;
                rule_dict.set_item("weight", *w)?;
            }
            SpiralKSoftRule::Algo { val, w } => {
                rule_dict.set_item("field", "algo")?;
                rule_dict.set_item("value", *val)?;
                rule_dict.set_item("weight", *w)?;
            }
            SpiralKSoftRule::Midk { val, w } => {
                rule_dict.set_item("field", "midk_mode")?;
                rule_dict.set_item("value", *val)?;
                rule_dict.set_item("weight", *w)?;
            }
            SpiralKSoftRule::Bottomk { val, w } => {
                rule_dict.set_item("field", "bottomk_mode")?;
                rule_dict.set_item("value", *val)?;
                rule_dict.set_item("weight", *w)?;
            }
            SpiralKSoftRule::Ctile { val, w } => {
                rule_dict.set_item("field", "compaction_tile")?;
                rule_dict.set_item("value", *val)?;
                rule_dict.set_item("weight", *w)?;
            }
            SpiralKSoftRule::TileCols { val, w } => {
                rule_dict.set_item("field", "tile_cols")?;
                rule_dict.set_item("value", *val)?;
                rule_dict.set_item("weight", *w)?;
            }
            SpiralKSoftRule::Radix { val, w } => {
                rule_dict.set_item("field", "radix")?;
                rule_dict.set_item("value", *val)?;
                rule_dict.set_item("weight", *w)?;
            }
            SpiralKSoftRule::Segments { val, w } => {
                rule_dict.set_item("field", "segments")?;
                rule_dict.set_item("value", *val)?;
                rule_dict.set_item("weight", *w)?;
            }
        }
        soft_rules.append(rule_dict)?;
    }

    let out_dict = PyDict::new_bound(py);
    out_dict.set_item("hard", hard)?;
    out_dict.set_item("soft", soft_rules)?;
    Ok(out_dict.into_py(py))
}

#[cfg(feature = "kdsl")]
fn normalize_hint_field(field: &str) -> PyResult<&'static str> {
    use pyo3::exceptions::PyValueError;
    match field {
        "use_2ce" | "u2" => Ok("use_2ce"),
        "wg" | "workgroup" => Ok("wg"),
        "kl" | "lanes" => Ok("kl"),
        "ch" | "channel_stride" => Ok("ch"),
        "algo" | "merge" => Ok("algo"),
        "midk" | "midk_mode" => Ok("midk"),
        "bottomk" | "bottomk_mode" => Ok("bottomk"),
        "ctile" | "compaction_tile" => Ok("ctile"),
        "tile_cols" => Ok("tile_cols"),
        "radix" => Ok("radix"),
        "segments" => Ok("segments"),
        other => Err(PyValueError::new_err(format!(
            "unknown SpiralK field '{other}'"
        ))),
    }
}

#[cfg(feature = "kdsl")]
#[pyclass(module = "spiraltorch.spiralk", name = "SpiralKContext")]
pub(crate) struct PySpiralKContext {
    pub(crate) inner: SpiralKCtx,
}

#[cfg(feature = "kdsl")]
impl PySpiralKContext {
    pub(crate) fn from_ctx(ctx: SpiralKCtx) -> Self {
        Self { inner: ctx }
    }
}

#[cfg(feature = "kdsl")]
#[pymethods]
impl PySpiralKContext {
    #[new]
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
        let inner = SpiralKCtx {
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

    fn eval(&self, py: Python<'_>, program: &str) -> PyResult<PyObject> {
        let out = st_kdsl::eval_program(program, &self.inner).map_err(spiralk_err_to_py)?;
        spiralk_out_to_dict(py, &out)
    }
}

#[cfg(feature = "kdsl")]
#[pyclass(module = "spiraltorch.spiralk", name = "SpiralKHeuristicHint")]
pub(crate) struct PySpiralKHeuristicHint {
    inner: HeuristicHint,
}

#[cfg(feature = "kdsl")]
#[pymethods]
impl PySpiralKHeuristicHint {
    #[new]
    pub fn new(
        field: &str,
        value_expr: String,
        weight: f32,
        condition_expr: String,
    ) -> PyResult<Self> {
        let normalized = normalize_hint_field(field)?;
        Ok(Self {
            inner: HeuristicHint::new(normalized, value_expr, weight, condition_expr),
        })
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

#[cfg(feature = "kdsl")]
impl PySpiralKHeuristicHint {
    pub(crate) fn inner(&self) -> &HeuristicHint {
        &self.inner
    }
}

#[cfg(feature = "kdsl")]
#[pyclass(module = "spiraltorch.spiralk", name = "SpiralKWilsonMetrics")]
pub(crate) struct PySpiralKWilsonMetrics {
    inner: WilsonMetrics,
}

#[cfg(feature = "kdsl")]
#[pymethods]
impl PySpiralKWilsonMetrics {
    #[new]
    pub fn new(baseline_latency: f32, candidate_latency: f32, wins: u32, trials: u32) -> Self {
        Self {
            inner: WilsonMetrics {
                baseline_latency,
                candidate_latency,
                wins,
                trials,
            },
        }
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

#[cfg(feature = "kdsl")]
impl PySpiralKWilsonMetrics {
    pub(crate) fn inner(&self) -> WilsonMetrics {
        self.inner
    }
}

#[cfg(feature = "kdsl")]
#[pyfunction]
fn wilson_lower_bound(wins: u32, trials: u32, z: f32) -> f32 {
    auto::wilson_lower_bound(wins, trials, z)
}

#[cfg(feature = "kdsl")]
#[pyfunction]
#[pyo3(signature = (metrics, min_gain=0.02, min_confidence=0.5))]
fn should_rewrite(metrics: &PySpiralKWilsonMetrics, min_gain: f32, min_confidence: f32) -> bool {
    auto::should_rewrite(&metrics.inner(), min_gain, min_confidence)
}

#[cfg(feature = "kdsl")]
#[pyfunction]
fn synthesize_program(
    base_src: &str,
    hints: Vec<PyRef<PySpiralKHeuristicHint>>,
) -> PyResult<String> {
    let collected: Vec<HeuristicHint> =
        hints.into_iter().map(|hint| hint.inner().clone()).collect();
    Ok(auto::synthesize_program(base_src, &collected))
}

#[cfg(feature = "kdsl")]
#[pyfunction]
#[pyo3(signature = (base_src, ctx, metrics, hints, min_gain=0.02, min_confidence=0.5))]
fn rewrite_with_wilson(
    py: Python<'_>,
    base_src: &str,
    ctx: &PySpiralKContext,
    metrics: &PySpiralKWilsonMetrics,
    hints: Vec<PyRef<PySpiralKHeuristicHint>>,
    min_gain: f32,
    min_confidence: f32,
) -> PyResult<(PyObject, String)> {
    let collected: Vec<HeuristicHint> =
        hints.into_iter().map(|hint| hint.inner().clone()).collect();
    let (out, script) = auto::rewrite_with_wilson(
        base_src,
        &ctx.inner,
        metrics.inner(),
        &collected,
        min_gain,
        min_confidence,
    )
    .map_err(spiralk_err_to_py)?;
    let py_out = spiralk_out_to_dict(py, &out)?;
    Ok((py_out, script))
}

#[pyclass(module = "spiraltorch.spiralk", name = "MaxwellHint")]
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
}

#[pyclass(module = "spiraltorch.spiralk", name = "MaxwellBridge")]
pub(crate) struct PyMaxwellSpiralKBridge {
    inner: MaxwellSpiralKBridge,
}

#[pymethods]
impl PyMaxwellSpiralKBridge {
    #[new]
    #[pyo3(signature = (base_program=None, min_weight=0.55, max_weight=0.95))]
    pub fn new(base_program: Option<String>, min_weight: f32, max_weight: f32) -> Self {
        let bridge = MaxwellSpiralKBridge::new().with_weight_bounds(min_weight, max_weight);
        let bridge = if let Some(program) = base_program {
            bridge.with_base_program(program)
        } else {
            bridge
        };
        Self { inner: bridge }
    }

    #[pyo3(signature = (program=None))]
    pub fn set_base_program(&mut self, program: Option<&str>) {
        let mut bridge = self.inner.clone();
        let code = program.unwrap_or_default();
        bridge = bridge.with_base_program(code.to_string());
        self.inner = bridge;
    }

    pub fn set_weight_bounds(&mut self, min_weight: f32, max_weight: f32) {
        self.inner = self
            .inner
            .clone()
            .with_weight_bounds(min_weight, max_weight);
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
        let hint = self.inner.push_pulse(channel, &pulse);
        PyMaxwellSpiralKHint::from(hint)
    }

    pub fn hints(&self) -> Vec<PyMaxwellSpiralKHint> {
        self.inner
            .hints()
            .iter()
            .cloned()
            .map(PyMaxwellSpiralKHint::from)
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn script(&self) -> Option<String> {
        self.inner.script()
    }
}

#[pyfunction]
fn required_blocks(target_z: f64, sigma: f64, kappa: f64, lambda: f64) -> Option<f64> {
    required_blocks_rs(target_z, sigma, kappa, lambda)
}

#[pyclass(module = "spiraltorch.spiralk", name = "MaxwellFingerprint")]
pub(crate) struct PyMaxwellFingerprint {
    inner: MaxwellFingerprint,
}

#[pymethods]
impl PyMaxwellFingerprint {
    #[new]
    pub fn new(
        gamma: f64,
        modulation_depth: f64,
        tissue_response: f64,
        shielding_db: f64,
        transmit_gain: f64,
        polarization_angle: f64,
        distance_m: f64,
    ) -> Self {
        let inner = MaxwellFingerprint::new(
            gamma,
            modulation_depth,
            tissue_response,
            shielding_db,
            transmit_gain,
            polarization_angle,
            distance_m,
        );
        Self { inner }
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma
    }

    #[getter]
    fn modulation_depth(&self) -> f64 {
        self.inner.modulation_depth
    }

    #[getter]
    fn tissue_response(&self) -> f64 {
        self.inner.tissue_response
    }

    #[getter]
    fn shielding_db(&self) -> f64 {
        self.inner.shielding_db
    }

    #[getter]
    fn transmit_gain(&self) -> f64 {
        self.inner.transmit_gain
    }

    #[getter]
    fn polarization_angle(&self) -> f64 {
        self.inner.polarization_angle
    }

    #[getter]
    fn distance_m(&self) -> f64 {
        self.inner.distance_m
    }

    fn shielding_factor(&self) -> f64 {
        self.inner.shielding_factor()
    }

    fn polarization_alignment(&self) -> f64 {
        self.inner.polarization_alignment()
    }

    #[pyo3(name = "lambda_")]
    fn lambda_value(&self) -> f64 {
        self.inner.lambda()
    }

    fn expected_block_mean(&self, kappa: f64) -> f64 {
        self.inner.expected_block_mean(kappa)
    }
}

#[pyclass(module = "spiraltorch.spiralk", name = "MeaningGate")]
pub(crate) struct PyMeaningGate {
    inner: MeaningGate,
}

#[pymethods]
impl PyMeaningGate {
    #[new]
    pub fn new(physical_gain: f64, semantic_gain: f64) -> Self {
        let inner = MeaningGate::new(physical_gain, semantic_gain);
        Self { inner }
    }

    #[getter]
    fn physical_gain(&self) -> f64 {
        self.inner.physical_gain
    }

    #[setter]
    fn set_physical_gain(&mut self, value: f64) {
        self.inner.physical_gain = value;
    }

    #[getter]
    fn semantic_gain(&self) -> f64 {
        self.inner.semantic_gain
    }

    #[setter]
    fn set_semantic_gain(&mut self, value: f64) {
        self.inner.semantic_gain = value.max(0.0);
    }

    fn envelope(&self, rho: f64) -> f64 {
        self.inner.envelope(rho)
    }
}

#[pyclass(module = "spiraltorch.spiralk", name = "SequentialZ")]
#[derive(Clone, Default)]
pub(crate) struct PySequentialZ {
    pub(crate) inner: SequentialZ,
}

#[pymethods]
impl PySequentialZ {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SequentialZ::new(),
        }
    }

    fn push(&mut self, sample: f64) -> Option<f64> {
        self.inner.push(sample)
    }

    fn extend(&mut self, samples: Vec<f64>) -> Option<f64> {
        let mut last = None;
        for sample in samples {
            last = self.inner.push(sample);
        }
        last
    }

    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> Option<f64> {
        self.inner.variance()
    }

    fn standard_error(&self) -> Option<f64> {
        self.inner.standard_error()
    }

    fn z_stat(&self) -> Option<f64> {
        self.inner.z_stat()
    }

    fn reset(&mut self) {
        self.inner = SequentialZ::new();
    }
}

#[pyclass(module = "spiraltorch.spiralk", name = "MaxwellPulse")]
#[derive(Clone)]
pub(crate) struct PyMaxwellPulse {
    inner: MaxwellZPulse,
}

impl From<MaxwellZPulse> for PyMaxwellPulse {
    fn from(inner: MaxwellZPulse) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMaxwellPulse {
    #[new]
    #[pyo3(signature = (blocks, mean, standard_error, z_score, band_energy, z_bias))]
    pub fn new(
        blocks: u64,
        mean: f64,
        standard_error: f64,
        z_score: f64,
        band_energy: (f32, f32, f32),
        z_bias: f32,
    ) -> Self {
        let inner = MaxwellZPulse {
            blocks,
            mean,
            standard_error,
            z_score,
            band_energy,
            z_bias,
        };
        Self { inner }
    }

    #[getter]
    fn blocks(&self) -> u64 {
        self.inner.blocks
    }

    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean
    }

    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.standard_error
    }

    #[getter]
    fn z_score(&self) -> f64 {
        self.inner.z_score
    }

    #[getter]
    fn band_energy(&self) -> (f32, f32, f32) {
        self.inner.band_energy
    }

    #[getter]
    fn z_bias(&self) -> f32 {
        self.inner.z_bias
    }

    fn magnitude(&self) -> f64 {
        self.inner.magnitude()
    }
}

#[pyclass(module = "spiraltorch.spiralk", name = "MaxwellProjector")]
pub(crate) struct PyMaxwellProjector {
    inner: MaxwellZProjector,
    rank: usize,
    weight: f64,
    bias_gain: f32,
    min_blocks: u64,
    min_z: f64,
}

#[pymethods]
impl PyMaxwellProjector {
    #[new]
    #[pyo3(signature = (rank, weight, bias_gain=1.0, min_blocks=2, min_z=0.0))]
    pub fn new(rank: usize, weight: f64, bias_gain: f32, min_blocks: u64, min_z: f64) -> Self {
        let inner = MaxwellZProjector::new(rank, weight)
            .with_bias_gain(bias_gain)
            .with_min_blocks(min_blocks)
            .with_min_z(min_z);
        Self {
            inner,
            rank,
            weight,
            bias_gain,
            min_blocks: min_blocks.max(1),
            min_z: min_z.max(0.0),
        }
    }

    fn project(&self, tracker: &PySequentialZ) -> Option<PyMaxwellPulse> {
        self.inner.project(&tracker.inner).map(PyMaxwellPulse::from)
    }

    fn last_pulse(&self) -> Option<PyMaxwellPulse> {
        self.inner.last_pulse().map(PyMaxwellPulse::from)
    }

    fn bias_gain(&self) -> f32 {
        self.bias_gain
    }

    fn set_bias_gain(&mut self, bias_gain: f32) {
        self.inner = self.inner.clone().with_bias_gain(bias_gain);
        self.bias_gain = bias_gain;
    }

    fn min_blocks(&self) -> u64 {
        self.min_blocks
    }

    fn set_min_blocks(&mut self, min_blocks: u64) {
        let min_blocks = min_blocks.max(1);
        self.inner = self.inner.clone().with_min_blocks(min_blocks);
        self.min_blocks = min_blocks;
    }

    fn min_z(&self) -> f64 {
        self.min_z
    }

    fn set_min_z(&mut self, min_z: f64) {
        let min_z = min_z.max(0.0);
        self.inner = self.inner.clone().with_min_z(min_z);
        self.min_z = min_z;
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn weight(&self) -> f64 {
        self.weight
    }
}

pub(crate) fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "spiralk")?;
    module.add_class::<PySpiralKFftPlan>()?;
    #[cfg(feature = "kdsl")]
    {
        module.add_class::<PySpiralKContext>()?;
        module.add_class::<PySpiralKHeuristicHint>()?;
        module.add_class::<PySpiralKWilsonMetrics>()?;
    }
    module.add_class::<PyMaxwellSpiralKHint>()?;
    module.add_class::<PyMaxwellSpiralKBridge>()?;
    module.add_class::<PyMaxwellFingerprint>()?;
    module.add_class::<PyMeaningGate>()?;
    module.add_class::<PySequentialZ>()?;
    module.add_class::<PyMaxwellPulse>()?;
    module.add_class::<PyMaxwellProjector>()?;
    module.add_function(wrap_pyfunction!(required_blocks, &module)?)?;
    #[cfg(feature = "kdsl")]
    {
        module.add_function(wrap_pyfunction!(wilson_lower_bound, &module)?)?;
        module.add_function(wrap_pyfunction!(should_rewrite, &module)?)?;
        module.add_function(wrap_pyfunction!(synthesize_program, &module)?)?;
        module.add_function(wrap_pyfunction!(rewrite_with_wilson, &module)?)?;
    }
    module.add("__doc__", "SpiralK DSL helpers & Maxwell bridges")?;
    let exports = {
        #[allow(unused_mut)]
        let mut list = vec![
            "FftPlan",
            "MaxwellHint",
            "MaxwellBridge",
            "MaxwellFingerprint",
            "MeaningGate",
            "SequentialZ",
            "MaxwellPulse",
            "MaxwellProjector",
            "required_blocks",
        ];
        #[cfg(feature = "kdsl")]
        {
            list.extend_from_slice(&[
                "SpiralKContext",
                "SpiralKHeuristicHint",
                "SpiralKWilsonMetrics",
                "wilson_lower_bound",
                "should_rewrite",
                "synthesize_program",
                "rewrite_with_wilson",
            ]);
        }
        list
    };
    module.add("__all__", exports)?;
    m.add_submodule(&module)?;
    Ok(())
}
