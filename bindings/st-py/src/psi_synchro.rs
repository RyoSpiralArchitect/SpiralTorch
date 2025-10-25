use std::{f64::consts::PI, time::Duration};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::{wrap_pyfunction, Bound};

use st_core::theory::zpulse::{ZPulse, ZScale, ZSource, ZSupport};
use st_nn::zspace_coherence::psi_synchro::{
    heatmaps_to_zpulses, run_multibranch_demo as run_multibranch_demo_rs, CircleLockMapConfig,
    HeatmapResult, MetaMembConfig, PsiBranchState, PsiSynchroConfig, PsiSynchroPulse,
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

#[pyclass(module = "spiraltorch.psi", name = "PsiSynchroConfig")]
#[derive(Clone)]
pub(crate) struct PyPsiSynchroConfig {
    pub(crate) inner: PsiSynchroConfig,
}

#[pymethods]
impl PyPsiSynchroConfig {
    #[new]
    #[pyo3(signature = (step=0.01, samples=1000, ticker_interval=None, min_ident_points=600, max_ident_points=2400, metamemb=None, circle_map=None))]
    pub fn new(
        step: f64,
        samples: usize,
        ticker_interval: Option<f64>,
        min_ident_points: usize,
        max_ident_points: usize,
        metamemb: Option<&Bound<PyMetaMembConfig>>,
        circle_map: Option<&Bound<PyCircleLockMapConfig>>,
    ) -> PyResult<Self> {
        let mut inner = PsiSynchroConfig::default();
        inner.step = step;
        inner.samples = samples;
        inner.ticker_interval = ticker_interval.map(|secs| Duration::from_secs_f64(secs.max(0.0)));
        inner.min_ident_points = min_ident_points;
        inner.max_ident_points = max_ident_points;
        if let Some(cfg) = metamemb {
            inner.metamemb = cfg.borrow().inner.clone();
        }
        if let Some(cfg) = circle_map {
            inner.circle_map = cfg.borrow().inner.clone();
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
    fn from_pulse(inner: ZPulse) -> Self {
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

#[pyclass(module = "spiraltorch.psi", name = "PsiSynchroResult")]
#[derive(Clone)]
pub(crate) struct PyPsiSynchroResult {
    heatmaps: Vec<HeatmapResult>,
    pulses: Vec<PsiSynchroPulse>,
}

impl PyPsiSynchroResult {
    fn from_parts(heatmaps: Vec<HeatmapResult>, pulses: Vec<PsiSynchroPulse>) -> Self {
        Self { heatmaps, pulses }
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
}

#[pyfunction(name = "run_multibranch_demo")]
#[pyo3(signature = (branches, config=None))]
pub fn run_multibranch_demo_py(
    py: Python<'_>,
    branches: &Bound<PyAny>,
    config: Option<&Bound<PyPsiSynchroConfig>>,
) -> PyResult<Py<PyPsiSynchroResult>> {
    let branch_objs: Vec<Py<PyPsiBranchState>> = branches.extract()?;
    let branch_states: Vec<PsiBranchState> = branch_objs
        .into_iter()
        .map(|branch| {
            let guard = branch.borrow(py);
            Ok(guard.inner.clone())
        })
        .collect::<PyResult<_>>()?;

    let config = match config {
        Some(cfg) => cfg.borrow().inner.clone(),
        None => PsiSynchroConfig::default(),
    };

    let heatmaps = run_multibranch_demo_rs(config, branch_states);
    let pulses = heatmaps_to_zpulses(&heatmaps);
    Py::new(py, PyPsiSynchroResult::from_parts(heatmaps, pulses))
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "psi")?;
    module.add_class::<PyMetaMembConfig>()?;
    module.add_class::<PyCircleLockMapConfig>()?;
    module.add_class::<PyPsiSynchroConfig>()?;
    module.add_class::<PyPsiBranchState>()?;
    module.add_class::<PyHeatmapResult>()?;
    module.add_class::<PyZPulse>()?;
    module.add_class::<PyPsiSynchroPulse>()?;
    module.add_class::<PyPsiSynchroResult>()?;
    module.add_function(wrap_pyfunction!(run_multibranch_demo_py, &module)?)?;
    module.add(
        "__all__",
        vec![
            "MetaMembConfig",
            "CircleLockMapConfig",
            "PsiSynchroConfig",
            "PsiBranchState",
            "HeatmapResult",
            "ZPulseSnapshot",
            "PsiSynchroPulse",
            "PsiSynchroResult",
            "run_multibranch_demo",
        ],
    )?;
    parent.add_submodule(&module)?;
    Ok(())
}
