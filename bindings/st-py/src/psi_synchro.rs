use std::{f64::consts::PI, time::Duration};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use pyo3::{wrap_pyfunction, Bound};

use st_core::telemetry::atlas::{AtlasFragment, ConceptSense};
use st_core::telemetry::chrono::{ChronoHarmonics, ChronoPeak, ChronoSummary};
use st_core::telemetry::maintainer::MaintainerStatus;
#[cfg(feature = "psi")]
use st_core::telemetry::psi::PsiReading;
use st_core::theory::zpulse::{ZPulse, ZScale, ZSource, ZSupport};
#[cfg(feature = "golden")]
use st_nn::golden::{GoldenBlackcatPulse, GoldenCooperativeDirective};
#[cfg(feature = "psi")]
use st_nn::zspace_coherence::psi_synchro::BranchPsiReading;
#[cfg(feature = "golden")]
use st_nn::zspace_coherence::psi_synchro::{heatmaps_to_golden_telemetry, PsiGoldenTelemetry};
use st_nn::zspace_coherence::psi_synchro::{
    run_zspace_learning_pass as run_zspace_learning_rs, ArnoldTongueSummary, BranchAtlasFragment,
    CircleLockMapConfig, HeatmapAnalytics, HeatmapResult, MetaMembConfig, PsiBranchState,
    PsiSynchroConfig, PsiSynchroPulse, PsiSynchroResult, PsiTelemetryConfig,
};

fn vec3_or_default(values: Option<Vec<f64>>, default: [f64; 3], name: &str) -> PyResult<[f64; 3]> {
    match values {
        Some(data) => {
            if data.len() != 3 {
                Err(PyValueError::new_err(format!(
                    "{name} expects exactly 3 values, received {}",
                    data.len()
                )))
            } else {
                Ok([data[0], data[1], data[2]])
            }
        }
        None => Ok(default),
    }
}

fn array_to_tuple(values: [f64; 3]) -> (f64, f64, f64) {
    (values[0], values[1], values[2])
}

fn zsource_to_str(source: ZSource) -> &'static str {
    match source {
        ZSource::Microlocal => "microlocal",
        ZSource::Maxwell => "maxwell",
        ZSource::Graph => "graph",
        ZSource::Desire => "desire",
        ZSource::GW => "gw",
        ZSource::RealGrad => "realgrad",
        ZSource::Other(tag) => tag,
    }
}

fn support_to_tuple(support: ZSupport) -> (f32, f32, f32) {
    (support.leading, support.central, support.trailing)
}

fn scale_to_tuple(scale: Option<ZScale>) -> Option<(f32, f32)> {
    scale.map(|s| (s.physical_radius, s.log_radius))
}

fn chrono_peak_to_py(py: Python<'_>, peak: &ChronoPeak) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("frequency", peak.frequency)?;
    dict.set_item("magnitude", peak.magnitude)?;
    dict.set_item("phase", peak.phase)?;
    Ok(dict.into())
}

fn chrono_summary_to_py(py: Python<'_>, summary: &ChronoSummary) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("frames", summary.frames)?;
    dict.set_item("duration", summary.duration)?;
    dict.set_item("latest_timestamp", summary.latest_timestamp)?;
    dict.set_item("mean_drift", summary.mean_drift)?;
    dict.set_item("mean_abs_drift", summary.mean_abs_drift)?;
    dict.set_item("drift_std", summary.drift_std)?;
    dict.set_item("mean_energy", summary.mean_energy)?;
    dict.set_item("energy_std", summary.energy_std)?;
    dict.set_item("mean_decay", summary.mean_decay)?;
    dict.set_item("min_energy", summary.min_energy)?;
    dict.set_item("max_energy", summary.max_energy)?;
    Ok(dict.into())
}

fn chrono_harmonics_to_py(py: Python<'_>, harmonics: &ChronoHarmonics) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("frames", harmonics.frames)?;
    dict.set_item("duration", harmonics.duration)?;
    dict.set_item("sample_rate", harmonics.sample_rate)?;
    dict.set_item("nyquist", harmonics.nyquist)?;
    dict.set_item("drift_power", harmonics.drift_power.clone())?;
    dict.set_item("energy_power", harmonics.energy_power.clone())?;
    if let Some(peak) = harmonics.dominant_drift.as_ref() {
        dict.set_item("dominant_drift", chrono_peak_to_py(py, peak)?)?;
    } else {
        dict.set_item("dominant_drift", py.None())?;
    }
    if let Some(peak) = harmonics.dominant_energy.as_ref() {
        dict.set_item("dominant_energy", chrono_peak_to_py(py, peak)?)?;
    } else {
        dict.set_item("dominant_energy", py.None())?;
    }
    Ok(dict.into())
}

fn maintainer_status_to_py(status: Option<MaintainerStatus>) -> Option<&'static str> {
    status.map(|value| value.as_str())
}

fn concept_annotation_to_py(
    py: Python<'_>,
    term: &str,
    sense: ConceptSense,
    rationale: Option<&String>,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("term", term)?;
    dict.set_item("sense", sense.label())?;
    dict.set_item("description", sense.description())?;
    if let Some(rationale) = rationale {
        dict.set_item("rationale", rationale.as_str())?;
    } else {
        dict.set_item("rationale", py.None())?;
    }
    Ok(dict.into())
}

fn atlas_fragment_to_py(py: Python<'_>, fragment: AtlasFragment) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("timestamp", fragment.timestamp)?;
    dict.set_item("loop_support", fragment.loop_support)?;
    dict.set_item("collapse_total", fragment.collapse_total)?;
    dict.set_item("z_signal", fragment.z_signal)?;
    dict.set_item("script_hint", fragment.script_hint)?;
    if let Some(summary) = fragment.summary.as_ref() {
        dict.set_item("chrono_summary", chrono_summary_to_py(py, summary)?)?;
    } else {
        dict.set_item("chrono_summary", py.None())?;
    }
    if let Some(harmonics) = fragment.harmonics.as_ref() {
        dict.set_item("chrono_harmonics", chrono_harmonics_to_py(py, harmonics)?)?;
    } else {
        dict.set_item("chrono_harmonics", py.None())?;
    }
    dict.set_item("maintainer_status", maintainer_status_to_py(fragment.maintainer_status))?;
    dict.set_item("maintainer_diagnostic", fragment.maintainer_diagnostic)?;
    dict.set_item("suggested_max_scale", fragment.suggested_max_scale)?;
    dict.set_item("suggested_pressure", fragment.suggested_pressure)?;
    let metrics = PyList::empty_bound(py);
    for metric in fragment.metrics.iter() {
        let metric_dict = PyDict::new_bound(py);
        metric_dict.set_item("name", metric.name.clone())?;
        metric_dict.set_item("value", metric.value)?;
        metric_dict.set_item("district", metric.district.clone())?;
        metrics.append(metric_dict)?;
    }
    dict.set_item("metrics", metrics)?;
    dict.set_item("notes", fragment.notes.clone())?;
    let concepts = PyList::empty_bound(py);
    for concept in fragment.concepts.iter() {
        let annotation = concept_annotation_to_py(
            py,
            &concept.term,
            concept.sense,
            concept.rationale.as_ref(),
        )?;
        concepts.append(annotation)?;
    }
    dict.set_item("concepts", concepts)?;
    Ok(dict.into())
}

#[cfg(feature = "psi")]
fn psi_reading_to_py(py: Python<'_>, reading: PsiReading) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("total", reading.total)?;
    dict.set_item("step", reading.step)?;
    let breakdown = PyDict::new_bound(py);
    for (component, value) in reading.breakdown.iter() {
        breakdown.set_item(component.to_string(), *value)?;
    }
    dict.set_item("breakdown", breakdown)?;
    Ok(dict.into())
}

#[pyclass(module = "spiraltorch.psi", name = "MetaMembConfig")]
#[derive(Clone)]
pub(crate) struct PyMetaMembConfig {
    pub(crate) inner: MetaMembConfig,
}

impl PyMetaMembConfig {
    fn from_config(inner: MetaMembConfig) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMetaMembConfig {
    #[new]
    #[pyo3(signature = (*, delta=None, omega=None, theta=None))]
    pub fn new(
        delta: Option<Vec<f64>>,
        omega: Option<Vec<f64>>,
        theta: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let MetaMembConfig {
            delta: delta_default,
            omega: omega_default,
            theta: theta_default,
        } = MetaMembConfig::default();
        let delta = vec3_or_default(delta, delta_default, "delta")?;
        let omega = vec3_or_default(omega, omega_default, "omega")?;
        let theta = vec3_or_default(theta, theta_default, "theta")?;
        Ok(Self {
            inner: MetaMembConfig {
                delta,
                omega,
                theta,
            },
        })
    }

    #[staticmethod]
    pub fn default() -> Self {
        Self {
            inner: MetaMembConfig::default(),
        }
    }

    #[getter]
    pub fn delta(&self) -> (f64, f64, f64) {
        array_to_tuple(self.inner.delta)
    }

    #[getter]
    pub fn omega(&self) -> (f64, f64, f64) {
        array_to_tuple(self.inner.omega)
    }

    #[getter]
    pub fn theta(&self) -> (f64, f64, f64) {
        array_to_tuple(self.inner.theta)
    }
}

#[pyclass(module = "spiraltorch.psi", name = "CircleLockMapConfig")]
#[derive(Clone)]
pub(crate) struct PyCircleLockMapConfig {
    pub(crate) inner: CircleLockMapConfig,
}

impl PyCircleLockMapConfig {
    fn from_config(inner: CircleLockMapConfig) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCircleLockMapConfig {
    #[new]
    #[pyo3(signature = (*, lam_min=0.0, lam_max=2.0, lam_bins=60, wd_min=0.3, wd_max=1.2, wd_bins=80, burn_in=200, samples=300, qmax=8))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lam_min: f64,
        lam_max: f64,
        lam_bins: usize,
        wd_min: f64,
        wd_max: f64,
        wd_bins: usize,
        burn_in: usize,
        samples: usize,
        qmax: usize,
    ) -> Self {
        Self {
            inner: CircleLockMapConfig {
                lam_min,
                lam_max,
                lam_bins,
                wd_min,
                wd_max,
                wd_bins,
                burn_in,
                samples,
                qmax,
            },
        }
    }

    #[staticmethod]
    pub fn default() -> Self {
        Self {
            inner: CircleLockMapConfig::default(),
        }
    }

    #[getter]
    pub fn lam_min(&self) -> f64 {
        self.inner.lam_min
    }

    #[getter]
    pub fn lam_max(&self) -> f64 {
        self.inner.lam_max
    }

    #[getter]
    pub fn lam_bins(&self) -> usize {
        self.inner.lam_bins
    }

    #[getter]
    pub fn wd_min(&self) -> f64 {
        self.inner.wd_min
    }

    #[getter]
    pub fn wd_max(&self) -> f64 {
        self.inner.wd_max
    }

    #[getter]
    pub fn wd_bins(&self) -> usize {
        self.inner.wd_bins
    }

    #[getter]
    pub fn burn_in(&self) -> usize {
        self.inner.burn_in
    }

    #[getter]
    pub fn samples(&self) -> usize {
        self.inner.samples
    }

    #[getter]
    pub fn qmax(&self) -> usize {
        self.inner.qmax
    }
}

#[pyclass(module = "spiraltorch.psi", name = "PsiTelemetryConfig")]
#[derive(Clone)]
pub(crate) struct PyPsiTelemetryConfig {
    pub(crate) inner: PsiTelemetryConfig,
}

impl PyPsiTelemetryConfig {
    fn from_config(inner: PsiTelemetryConfig) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPsiTelemetryConfig {
    #[cfg(all(feature = "psi", feature = "golden"))]
    #[new]
    #[pyo3(
        signature = (
            *,
            emit_atlas=true,
            atlas_timestamp=None,
            emit_psi=true,
            psi_step_base=0,
            emit_golden=true,
            golden_baseline_interval=30.0,
            golden_baseline_window=32,
        )
    )]
    pub fn new(
        emit_atlas: bool,
        atlas_timestamp: Option<f32>,
        emit_psi: bool,
        psi_step_base: u64,
        emit_golden: bool,
        golden_baseline_interval: f64,
        golden_baseline_window: usize,
    ) -> Self {
        let inner = PsiTelemetryConfig {
            emit_atlas,
            atlas_timestamp,
            emit_psi,
            psi_step_base,
            emit_golden,
            golden_baseline_interval: Duration::from_secs_f64(golden_baseline_interval.max(0.0)),
            golden_baseline_window,
            ..Default::default()
        };
        Self { inner }
    }

    #[cfg(all(feature = "psi", not(feature = "golden")))]
    #[new]
    #[pyo3(signature = (*, emit_atlas=true, atlas_timestamp=None, emit_psi=true, psi_step_base=0))]
    pub fn new_no_golden(
        emit_atlas: bool,
        atlas_timestamp: Option<f32>,
        emit_psi: bool,
        psi_step_base: u64,
    ) -> Self {
        let inner = PsiTelemetryConfig {
            emit_atlas,
            atlas_timestamp,
            emit_psi,
            psi_step_base,
        };
        Self { inner }
    }

    #[cfg(all(not(feature = "psi"), feature = "golden"))]
    #[new]
    #[pyo3(
        signature = (
            *,
            emit_atlas=true,
            atlas_timestamp=None,
            emit_golden=true,
            golden_baseline_interval=30.0,
            golden_baseline_window=32,
        )
    )]
    pub fn new_no_psi(
        emit_atlas: bool,
        atlas_timestamp: Option<f32>,
        emit_golden: bool,
        golden_baseline_interval: f64,
        golden_baseline_window: usize,
    ) -> Self {
        let inner = PsiTelemetryConfig {
            emit_atlas,
            atlas_timestamp,
            emit_golden,
            golden_baseline_interval: Duration::from_secs_f64(golden_baseline_interval.max(0.0)),
            golden_baseline_window,
            ..Default::default()
        };
        Self { inner }
    }

    #[cfg(all(not(feature = "psi"), not(feature = "golden")))]
    #[new]
    #[pyo3(signature = (*, emit_atlas=true, atlas_timestamp=None))]
    pub fn new_minimal(emit_atlas: bool, atlas_timestamp: Option<f32>) -> Self {
        let mut inner = PsiTelemetryConfig::default();
        inner.emit_atlas = emit_atlas;
        inner.atlas_timestamp = atlas_timestamp;
        Self { inner }
    }

    #[getter]
    pub fn emit_atlas(&self) -> bool {
        self.inner.emit_atlas
    }

    #[getter]
    pub fn atlas_timestamp(&self) -> Option<f32> {
        self.inner.atlas_timestamp
    }

    #[cfg(feature = "psi")]
    #[getter]
    pub fn emit_psi(&self) -> bool {
        self.inner.emit_psi
    }

    #[cfg(feature = "psi")]
    #[getter]
    pub fn psi_step_base(&self) -> u64 {
        self.inner.psi_step_base
    }

    #[cfg(feature = "golden")]
    #[getter]
    pub fn emit_golden(&self) -> bool {
        self.inner.emit_golden
    }

    #[cfg(feature = "golden")]
    #[getter]
    pub fn golden_baseline_interval(&self) -> f64 {
        self.inner.golden_baseline_interval.as_secs_f64()
    }

    #[cfg(feature = "golden")]
    #[getter]
    pub fn golden_baseline_window(&self) -> usize {
        self.inner.golden_baseline_window
    }
}

#[pyclass(module = "spiraltorch.psi", name = "PsiSynchroConfig")]
#[derive(Clone)]
pub(crate) struct PyPsiSynchroConfig {
    pub(crate) inner: PsiSynchroConfig,
}

#[pymethods]
impl PyPsiSynchroConfig {
    #[new]
    #[pyo3(signature = (step=0.01, samples=1000, ticker_interval=None, min_ident_points=600, max_ident_points=2400, metamemb=None, circle_map=None, telemetry=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        step: f64,
        samples: usize,
        ticker_interval: Option<f64>,
        min_ident_points: usize,
        max_ident_points: usize,
        metamemb: Option<&Bound<PyMetaMembConfig>>,
        circle_map: Option<&Bound<PyCircleLockMapConfig>>,
        telemetry: Option<&Bound<PyPsiTelemetryConfig>>,
    ) -> PyResult<Self> {
        let mut inner = PsiSynchroConfig {
            step,
            samples,
            ticker_interval: ticker_interval.map(|secs| Duration::from_secs_f64(secs.max(0.0))),
            min_ident_points,
            max_ident_points,
            ..Default::default()
        };
        if let Some(cfg) = metamemb {
            inner.metamemb = cfg.borrow().inner.clone();
        }
        if let Some(cfg) = circle_map {
            inner.circle_map = cfg.borrow().inner.clone();
        }
        if let Some(cfg) = telemetry {
            inner.telemetry = Some(cfg.borrow().inner.clone());
        }
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn default() -> Self {
        Self {
            inner: PsiSynchroConfig::default(),
        }
    }

    #[getter]
    pub fn step(&self) -> f64 {
        self.inner.step
    }

    #[getter]
    pub fn samples(&self) -> usize {
        self.inner.samples
    }

    #[getter]
    pub fn ticker_interval(&self) -> Option<f64> {
        self.inner.ticker_interval.map(|d| d.as_secs_f64())
    }

    #[getter]
    pub fn min_ident_points(&self) -> usize {
        self.inner.min_ident_points
    }

    #[getter]
    pub fn max_ident_points(&self) -> usize {
        self.inner.max_ident_points
    }

    #[getter]
    pub fn metamemb(&self) -> PyMetaMembConfig {
        PyMetaMembConfig::from_config(self.inner.metamemb.clone())
    }

    #[getter]
    pub fn circle_map(&self) -> PyCircleLockMapConfig {
        PyCircleLockMapConfig::from_config(self.inner.circle_map.clone())
    }

    #[getter]
    pub fn telemetry(&self) -> Option<PyPsiTelemetryConfig> {
        self.inner
            .telemetry
            .clone()
            .map(PyPsiTelemetryConfig::from_config)
    }
}

#[pyclass(module = "spiraltorch.psi", name = "PsiBranchState")]
#[derive(Clone)]
pub(crate) struct PyPsiBranchState {
    pub(crate) inner: PsiBranchState,
}

#[pymethods]
impl PyPsiBranchState {
    #[new]
    #[pyo3(signature = (branch_id, *, gamma=1.3, lambda_=1.0, wd=0.7, omega0=0.72, drift_coupled=1.05, phase0=0.0))]
    pub fn new(
        branch_id: String,
        gamma: f64,
        lambda_: f64,
        wd: f64,
        omega0: f64,
        drift_coupled: f64,
        phase0: f64,
    ) -> Self {
        let mut inner = PsiBranchState::new(branch_id);
        inner.gamma = gamma;
        inner.lambda = lambda_;
        inner.wd = wd;
        inner.omega0 = omega0;
        inner.drift_coupled = drift_coupled;
        inner.phase0 = phase0;
        Self { inner }
    }

    #[getter]
    pub fn branch_id(&self) -> &str {
        &self.inner.branch_id
    }

    #[getter]
    pub fn gamma(&self) -> f64 {
        self.inner.gamma
    }

    #[getter]
    pub fn lambda_(&self) -> f64 {
        self.inner.lambda
    }

    #[getter]
    pub fn wd(&self) -> f64 {
        self.inner.wd
    }

    #[getter]
    pub fn omega0(&self) -> f64 {
        self.inner.omega0
    }

    #[getter]
    pub fn drift_coupled(&self) -> f64 {
        self.inner.drift_coupled
    }

    #[getter]
    pub fn phase0(&self) -> f64 {
        self.inner.phase0
    }

    pub fn poincare_period(&self) -> f64 {
        if self.inner.wd.abs() <= f64::EPSILON {
            f64::INFINITY
        } else {
            2.0 * PI / self.inner.wd
        }
    }
}

#[pyclass(module = "spiraltorch.psi", name = "ArnoldTonguePeak")]
#[derive(Clone)]
pub(crate) struct PyArnoldTongue {
    inner: ArnoldTongueSummary,
}

impl PyArnoldTongue {
    fn from_inner(inner: ArnoldTongueSummary) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyArnoldTongue {
    #[getter]
    pub fn ratio_p(&self) -> i64 {
        self.inner.ratio_p
    }

    #[getter]
    pub fn ratio_q(&self) -> i64 {
        self.inner.ratio_q
    }

    #[getter]
    pub fn rotation(&self) -> f64 {
        self.inner.rotation
    }

    #[getter]
    pub fn lam(&self) -> f64 {
        self.inner.lam
    }

    #[getter]
    pub fn wd(&self) -> f64 {
        self.inner.wd
    }

    #[getter]
    pub fn strength(&self) -> f64 {
        self.inner.strength
    }

    #[getter]
    pub fn peak_strength(&self) -> f64 {
        self.inner.peak_strength
    }

    #[getter]
    pub fn error(&self) -> f64 {
        self.inner.error
    }

    #[getter]
    pub fn ratio(&self) -> f64 {
        self.inner.ratio()
    }
}

#[pyclass(module = "spiraltorch.psi", name = "HeatmapAnalytics")]
#[derive(Clone)]
pub(crate) struct PyHeatmapAnalytics {
    inner: HeatmapAnalytics,
}

impl PyHeatmapAnalytics {
    fn from_inner(inner: HeatmapAnalytics) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyHeatmapAnalytics {
    #[getter]
    pub fn total_energy(&self) -> f64 {
        self.inner.total_energy
    }

    #[getter]
    pub fn leading_sum(&self) -> f64 {
        self.inner.leading_sum
    }

    #[getter]
    pub fn central_sum(&self) -> f64 {
        self.inner.central_sum
    }

    #[getter]
    pub fn trailing_sum(&self) -> f64 {
        self.inner.trailing_sum
    }

    #[getter]
    pub fn leading_norm(&self) -> f64 {
        self.inner.leading_norm as f64
    }

    #[getter]
    pub fn central_norm(&self) -> f64 {
        self.inner.central_norm as f64
    }

    #[getter]
    pub fn trailing_norm(&self) -> f64 {
        self.inner.trailing_norm as f64
    }

    #[getter]
    pub fn dominant_lam(&self) -> f64 {
        self.inner.dominant_lam
    }

    #[getter]
    pub fn dominant_wd(&self) -> f64 {
        self.inner.dominant_wd
    }

    #[getter]
    pub fn peak_value(&self) -> f64 {
        self.inner.peak_value
    }

    #[getter]
    pub fn peak_ratio(&self) -> f64 {
        self.inner.peak_ratio as f64
    }

    #[getter]
    pub fn radius(&self) -> f64 {
        self.inner.radius as f64
    }

    #[getter]
    pub fn log_radius(&self) -> f64 {
        self.inner.log_radius as f64
    }

    #[getter]
    pub fn bias(&self) -> f64 {
        self.inner.bias as f64
    }

    #[getter]
    pub fn drift(&self) -> f64 {
        self.inner.drift as f64
    }

    #[getter]
    pub fn quality(&self) -> f64 {
        self.inner.quality as f64
    }

    #[getter]
    pub fn stderr(&self) -> f64 {
        self.inner.stderr as f64
    }

    #[getter]
    pub fn entropy(&self) -> f64 {
        self.inner.entropy as f64
    }

    pub fn band_energy(&self) -> (f64, f64, f64) {
        (
            self.inner.leading_norm as f64,
            self.inner.central_norm as f64,
            self.inner.trailing_norm as f64,
        )
    }
}

#[pyclass(module = "spiraltorch.psi", name = "HeatmapResult")]
#[derive(Clone)]
pub(crate) struct PyHeatmapResult {
    inner: HeatmapResult,
}

impl PyHeatmapResult {
    fn from_inner(inner: HeatmapResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyHeatmapResult {
    #[getter]
    pub fn branch_id(&self) -> &str {
        &self.inner.branch_id
    }

    #[getter]
    pub fn gamma(&self) -> f64 {
        self.inner.gamma
    }

    #[getter]
    pub fn kappa_hat(&self) -> f64 {
        self.inner.kappa_hat
    }

    #[getter]
    pub fn lam_grid(&self) -> Vec<f64> {
        self.inner.lam_grid.clone()
    }

    #[getter]
    pub fn wd_grid(&self) -> Vec<f64> {
        self.inner.wd_grid.clone()
    }

    #[getter]
    pub fn matrix(&self) -> Vec<Vec<f64>> {
        self.inner.matrix.clone()
    }

    #[getter]
    pub fn tongues(&self) -> Vec<PyArnoldTongue> {
        self.inner
            .tongues
            .iter()
            .cloned()
            .map(PyArnoldTongue::from_inner)
            .collect()
    }

    pub fn dominant_tongue(&self) -> Option<PyArnoldTongue> {
        self.inner
            .dominant_tongue()
            .cloned()
            .map(PyArnoldTongue::from_inner)
    }

    pub fn analyse(&self) -> Option<PyHeatmapAnalytics> {
        self.inner.analyse().map(PyHeatmapAnalytics::from_inner)
    }

    #[pyo3(signature = (timestamp=None))]
    pub fn to_atlas_fragment(
        &self,
        py: Python<'_>,
        timestamp: Option<f32>,
    ) -> PyResult<Option<PyObject>> {
        match self.inner.to_atlas_fragment(timestamp) {
            Some(fragment) => atlas_fragment_to_py(py, fragment).map(Some),
            None => Ok(None),
        }
    }

    #[cfg(feature = "psi")]
    pub fn to_psi_reading(&self, py: Python<'_>, step: u64) -> PyResult<Option<PyObject>> {
        match self.inner.to_psi_reading(step) {
            Some(reading) => psi_reading_to_py(py, reading).map(Some),
            None => Ok(None),
        }
    }

    pub fn to_zpulse(&self, ts: u64) -> PyZPulse {
        PyZPulse::from_pulse(self.inner.to_zpulse(ts))
    }
}

#[pyclass(module = "spiraltorch.psi", name = "ZPulseSnapshot")]
#[derive(Clone)]
pub(crate) struct PyZPulse {
    inner: ZPulse,
}

impl PyZPulse {
    pub(crate) fn from_pulse(inner: ZPulse) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyZPulse {
    #[getter]
    pub fn source(&self) -> &'static str {
        zsource_to_str(self.inner.source)
    }

    #[getter]
    pub fn ts(&self) -> u64 {
        self.inner.ts
    }

    #[getter]
    pub fn tempo(&self) -> f32 {
        self.inner.tempo
    }

    #[getter]
    pub fn band_energy(&self) -> (f32, f32, f32) {
        self.inner.band_energy
    }

    #[getter]
    pub fn drift(&self) -> f32 {
        self.inner.drift
    }

    #[getter]
    pub fn z_bias(&self) -> f32 {
        self.inner.z_bias
    }

    #[getter]
    pub fn density_fluctuation(&self) -> f32 {
        self.inner.density_fluctuation
    }

    #[getter]
    pub fn support(&self) -> (f32, f32, f32) {
        support_to_tuple(self.inner.support)
    }

    #[getter]
    pub fn scale(&self) -> Option<(f32, f32)> {
        scale_to_tuple(self.inner.scale)
    }

    #[getter]
    pub fn quality(&self) -> f32 {
        self.inner.quality
    }

    #[getter]
    pub fn stderr(&self) -> f32 {
        self.inner.stderr
    }

    #[getter]
    pub fn latency_ms(&self) -> f32 {
        self.inner.latency_ms
    }
}

#[pyclass(module = "spiraltorch.psi", name = "PsiSynchroPulse")]
#[derive(Clone)]
pub(crate) struct PyPsiSynchroPulse {
    branch_id: String,
    pulse: PyZPulse,
}

impl PyPsiSynchroPulse {
    fn from_pulse(pulse: PsiSynchroPulse) -> Self {
        Self {
            branch_id: pulse.branch_id,
            pulse: PyZPulse::from_pulse(pulse.pulse),
        }
    }
}

#[pymethods]
impl PyPsiSynchroPulse {
    #[getter]
    pub fn branch_id(&self) -> &str {
        &self.branch_id
    }

    #[getter]
    pub fn pulse(&self) -> PyZPulse {
        self.pulse.clone()
    }
}

#[cfg(feature = "golden")]
#[pyclass(module = "spiraltorch.psi", name = "GoldenPulse")]
#[derive(Clone)]
pub(crate) struct PyGoldenPulse {
    inner: GoldenBlackcatPulse,
}

#[cfg(feature = "golden")]
impl PyGoldenPulse {
    fn from_pulse(inner: GoldenBlackcatPulse) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "golden")]
#[pymethods]
impl PyGoldenPulse {
    #[getter]
    pub fn exploration_drive(&self) -> f32 {
        self.inner.exploration_drive
    }

    #[getter]
    pub fn optimization_gain(&self) -> f32 {
        self.inner.optimization_gain
    }

    #[getter]
    pub fn synergy_score(&self) -> f32 {
        self.inner.synergy_score
    }

    #[getter]
    pub fn reinforcement_weight(&self) -> f32 {
        self.inner.reinforcement_weight
    }

    #[getter]
    pub fn mean_support(&self) -> f32 {
        self.inner.mean_support
    }

    #[getter]
    pub fn mean_reward(&self) -> f64 {
        self.inner.mean_reward
    }

    #[getter]
    pub fn mean_psi(&self) -> f32 {
        self.inner.mean_psi
    }

    #[getter]
    pub fn mean_confidence(&self) -> f32 {
        self.inner.mean_confidence
    }

    #[getter]
    pub fn coverage(&self) -> usize {
        self.inner.coverage
    }

    #[getter]
    pub fn heuristics_contributions(&self) -> usize {
        self.inner.heuristics_contributions
    }

    #[getter]
    pub fn append_weight(&self) -> f32 {
        self.inner.append_weight
    }

    #[getter]
    pub fn retract_count(&self) -> usize {
        self.inner.retract_count
    }

    #[getter]
    pub fn annotate_count(&self) -> usize {
        self.inner.annotate_count
    }

    #[getter]
    pub fn dominant_plan(&self) -> Option<String> {
        self.inner.dominant_plan.clone()
    }

    pub fn is_idle(&self) -> bool {
        self.inner.is_idle()
    }
}

#[cfg(feature = "golden")]
#[pyclass(module = "spiraltorch.psi", name = "GoldenDirective")]
#[derive(Clone)]
pub(crate) struct PyGoldenDirective {
    inner: GoldenCooperativeDirective,
}

#[cfg(feature = "golden")]
impl PyGoldenDirective {
    fn from_directive(inner: GoldenCooperativeDirective) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "golden")]
#[pymethods]
impl PyGoldenDirective {
    #[getter]
    pub fn push_interval(&self) -> f64 {
        self.inner.push_interval.as_secs_f64()
    }

    #[getter]
    pub fn summary_window(&self) -> usize {
        self.inner.summary_window
    }

    #[getter]
    pub fn exploration_priority(&self) -> f32 {
        self.inner.exploration_priority
    }

    #[getter]
    pub fn reinforcement_weight(&self) -> f32 {
        self.inner.reinforcement_weight
    }
}

#[cfg(feature = "golden")]
#[pyclass(module = "spiraltorch.psi", name = "GoldenPsiTelemetry")]
#[derive(Clone)]
pub(crate) struct PyGoldenPsiTelemetry {
    inner: PsiGoldenTelemetry,
}

#[cfg(feature = "golden")]
impl PyGoldenPsiTelemetry {
    fn from_inner(inner: PsiGoldenTelemetry) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "golden")]
#[pymethods]
impl PyGoldenPsiTelemetry {
    #[getter]
    pub fn branch_id(&self) -> &str {
        &self.inner.branch_id
    }

    #[getter]
    pub fn pulse(&self) -> PyGoldenPulse {
        PyGoldenPulse::from_pulse(self.inner.pulse.clone())
    }

    #[getter]
    pub fn directive(&self) -> PyGoldenDirective {
        PyGoldenDirective::from_directive(self.inner.directive.clone())
    }
}

#[pyclass(module = "spiraltorch.psi", name = "PsiSynchroResult")]
#[derive(Clone)]
pub(crate) struct PyPsiSynchroResult {
    heatmaps: Vec<HeatmapResult>,
    pulses: Vec<PsiSynchroPulse>,
    atlas: Vec<BranchAtlasFragment>,
    #[cfg(feature = "psi")]
    psi: Vec<BranchPsiReading>,
    #[cfg(feature = "golden")]
    golden: Vec<PsiGoldenTelemetry>,
    #[cfg(feature = "golden")]
    golden_baseline_interval: f64,
    #[cfg(feature = "golden")]
    golden_baseline_window: usize,
}

impl PyPsiSynchroResult {
    fn from_result(result: PsiSynchroResult) -> Self {
        Self {
            heatmaps: result.heatmaps,
            pulses: result.pulses,
            atlas: result.atlas_fragments,
            #[cfg(feature = "psi")]
            psi: result.psi_readings,
            #[cfg(feature = "golden")]
            golden: result.golden_telemetry,
            #[cfg(feature = "golden")]
            golden_baseline_interval: result.golden_baseline_interval.as_secs_f64(),
            #[cfg(feature = "golden")]
            golden_baseline_window: result.golden_baseline_window,
        }
    }
}

#[pymethods]
impl PyPsiSynchroResult {
    #[getter]
    pub fn heatmaps(&self) -> Vec<PyHeatmapResult> {
        self.heatmaps
            .iter()
            .cloned()
            .map(PyHeatmapResult::from_inner)
            .collect()
    }

    #[getter]
    pub fn pulses(&self) -> Vec<PyPsiSynchroPulse> {
        self.pulses
            .iter()
            .cloned()
            .map(PyPsiSynchroPulse::from_pulse)
            .collect()
    }

    pub fn atlas_fragments(&self, py: Python<'_>) -> PyResult<Vec<(String, PyObject)>> {
        self.atlas
            .iter()
            .cloned()
            .map(|entry| {
                let fragment = atlas_fragment_to_py(py, entry.fragment)?;
                Ok((entry.branch_id, fragment))
            })
            .collect()
    }

    #[cfg(feature = "psi")]
    pub fn psi_readings(&self, py: Python<'_>) -> PyResult<Vec<(String, PyObject)>> {
        self.psi
            .iter()
            .cloned()
            .map(|entry| {
                let reading = psi_reading_to_py(py, entry.reading)?;
                Ok((entry.branch_id, reading))
            })
            .collect()
    }

    pub fn by_branch(&self) -> Vec<(String, PyZPulse)> {
        self.pulses
            .iter()
            .cloned()
            .map(|pulse| {
                let PsiSynchroPulse { branch_id, pulse } = pulse;
                (branch_id, PyZPulse::from_pulse(pulse))
            })
            .collect()
    }

    #[cfg(feature = "golden")]
    #[getter]
    pub fn golden(&self) -> Vec<PyGoldenPsiTelemetry> {
        self.golden
            .iter()
            .cloned()
            .map(PyGoldenPsiTelemetry::from_inner)
            .collect()
    }

    #[cfg(feature = "golden")]
    #[getter]
    pub fn golden_baseline_interval(&self) -> f64 {
        self.golden_baseline_interval
    }

    #[cfg(feature = "golden")]
    #[getter]
    pub fn golden_baseline_window(&self) -> usize {
        self.golden_baseline_window
    }

    #[cfg(feature = "golden")]
    #[pyo3(signature = (baseline_interval=30.0, baseline_window=32))]
    pub fn golden_telemetry(
        &self,
        baseline_interval: f64,
        baseline_window: usize,
    ) -> Vec<PyGoldenPsiTelemetry> {
        let interval = Duration::from_secs_f64(baseline_interval.max(0.0));
        if (baseline_interval - self.golden_baseline_interval).abs() <= f64::EPSILON
            && baseline_window == self.golden_baseline_window
        {
            return self
                .golden
                .iter()
                .cloned()
                .map(PyGoldenPsiTelemetry::from_inner)
                .collect();
        }
        heatmaps_to_golden_telemetry(&self.heatmaps, interval, baseline_window)
            .into_iter()
            .map(PyGoldenPsiTelemetry::from_inner)
            .collect()
    }
}

fn parse_branch_states(py: Python<'_>, branches: &Bound<PyAny>) -> PyResult<Vec<PsiBranchState>> {
    let branch_objs: Vec<Py<PyPsiBranchState>> = branches.extract()?;
    branch_objs
        .into_iter()
        .map(|branch| {
            let guard = branch.borrow(py);
            Ok(guard.inner.clone())
        })
        .collect()
}

fn resolve_synchro_config(config: Option<&Bound<PyPsiSynchroConfig>>) -> PsiSynchroConfig {
    config
        .map(|cfg| cfg.borrow().inner.clone())
        .unwrap_or_default()
}

#[pyfunction(name = "run_multibranch_demo")]
#[pyo3(signature = (branches, config=None))]
pub fn run_multibranch_demo_py(
    py: Python<'_>,
    branches: &Bound<PyAny>,
    config: Option<&Bound<PyPsiSynchroConfig>>,
) -> PyResult<Py<PyPsiSynchroResult>> {
    let branch_states = parse_branch_states(py, branches)?;
    let config = resolve_synchro_config(config);
    let result = run_zspace_learning_rs(config, branch_states);
    Py::new(py, PyPsiSynchroResult::from_result(result))
}

#[pyfunction(name = "run_zspace_learning")]
#[pyo3(signature = (branches, config=None))]
pub fn run_zspace_learning_py(
    py: Python<'_>,
    branches: &Bound<PyAny>,
    config: Option<&Bound<PyPsiSynchroConfig>>,
) -> PyResult<Py<PyPsiSynchroResult>> {
    let branch_states = parse_branch_states(py, branches)?;
    let config = resolve_synchro_config(config);
    let result = run_zspace_learning_rs(config, branch_states);
    Py::new(py, PyPsiSynchroResult::from_result(result))
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "psi")?;
    module.add_class::<PyMetaMembConfig>()?;
    module.add_class::<PyCircleLockMapConfig>()?;
    module.add_class::<PyPsiTelemetryConfig>()?;
    module.add_class::<PyPsiSynchroConfig>()?;
    module.add_class::<PyPsiBranchState>()?;
    module.add_class::<PyArnoldTongue>()?;
    module.add_class::<PyHeatmapAnalytics>()?;
    module.add_class::<PyHeatmapResult>()?;
    module.add_class::<PyZPulse>()?;
    module.add_class::<PyPsiSynchroPulse>()?;
    #[cfg(feature = "golden")]
    module.add_class::<PyGoldenPulse>()?;
    #[cfg(feature = "golden")]
    module.add_class::<PyGoldenDirective>()?;
    #[cfg(feature = "golden")]
    module.add_class::<PyGoldenPsiTelemetry>()?;
    module.add_class::<PyPsiSynchroResult>()?;
    module.add_function(wrap_pyfunction!(run_multibranch_demo_py, &module)?)?;
    module.add_function(wrap_pyfunction!(run_zspace_learning_py, &module)?)?;
    #[cfg_attr(not(feature = "golden"), allow(unused_mut))]
    let mut exports = vec![
        "MetaMembConfig",
        "CircleLockMapConfig",
        "PsiTelemetryConfig",
        "PsiSynchroConfig",
        "PsiBranchState",
        "ArnoldTonguePeak",
        "HeatmapAnalytics",
        "HeatmapResult",
        "ZPulseSnapshot",
        "PsiSynchroPulse",
        "PsiSynchroResult",
        "run_multibranch_demo",
        "run_zspace_learning",
    ];
    #[cfg(feature = "golden")]
    {
        exports.push("GoldenPulse");
        exports.push("GoldenDirective");
        exports.push("GoldenPsiTelemetry");
    }
    module.add("__all__", exports)?;
    parent.add_submodule(&module)?;
    Ok(())
}
