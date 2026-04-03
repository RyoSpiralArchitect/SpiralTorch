use num_complex::Complex32 as PyComplex32;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::{wrap_pyfunction, Bound, PyRefMut};

use crate::tensor::{tensor_err_to_py, PyTensor};

use st_core::telemetry::noncollapse::NonCollapseSnapshot as RustNonCollapseSnapshot;
use st_tensor::measure::{
    z_space_barycenter as z_space_barycenter_rs, BarycenterIntermediate, ZSpaceBarycenter,
};
use st_tensor::{
    AmegaHypergrad, AmegaRealgrad, Complex32 as StComplex32, ComplexTensor, DesireGradientControl,
    DesireGradientInterpretation, GradientSummary, HypergradTelemetry, LanguageWaveEncoder,
    OpenCartesianTopos, Tensor, TensorBiome, ZBox, ZBoxSite,
};

fn py_complex_to_st(values: Vec<PyComplex32>) -> Vec<StComplex32> {
    values
        .into_iter()
        .map(|value| StComplex32::new(value.re, value.im))
        .collect()
}

fn st_complex_to_py(values: &[StComplex32]) -> Vec<PyComplex32> {
    values
        .iter()
        .map(|value| PyComplex32::new(value.re, value.im))
        .collect()
}

#[pyclass(module = "spiraltorch", name = "ComplexTensor")]
#[derive(Clone)]
pub(crate) struct PyComplexTensor {
    pub(crate) inner: ComplexTensor,
}

#[pymethods]
impl PyComplexTensor {
    #[new]
    #[pyo3(signature = (rows, cols, data=None))]
    pub fn new(rows: usize, cols: usize, data: Option<Vec<PyComplex32>>) -> PyResult<Self> {
        let inner = match data {
            Some(values) => {
                let converted = py_complex_to_st(values);
                ComplexTensor::from_vec(rows, cols, converted)
            }
            None => ComplexTensor::zeros(rows, cols),
        }
        .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn zeros(rows: usize, cols: usize) -> PyResult<Self> {
        let inner = ComplexTensor::zeros(rows, cols).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn rows(&self) -> usize {
        self.inner.shape().0
    }

    #[getter]
    pub fn cols(&self) -> usize {
        self.inner.shape().1
    }

    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    pub fn to_tensor(&self) -> PyResult<PyTensor> {
        let tensor = self.inner.to_tensor().map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    pub fn data(&self) -> Vec<PyComplex32> {
        st_complex_to_py(self.inner.data())
    }

    pub fn matmul(&self, other: &PyComplexTensor) -> PyResult<Self> {
        let product = self.inner.matmul(&other.inner).map_err(tensor_err_to_py)?;
        Ok(Self { inner: product })
    }
}

#[pyclass(module = "spiraltorch", name = "OpenCartesianTopos")]
#[derive(Clone)]
pub(crate) struct PyOpenCartesianTopos {
    pub(crate) inner: OpenCartesianTopos,
}

impl PyOpenCartesianTopos {
    pub(crate) fn from_topos(inner: OpenCartesianTopos) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpenCartesianTopos {
    #[new]
    pub fn new(
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PyResult<Self> {
        let inner =
            OpenCartesianTopos::new(curvature, tolerance, saturation, max_depth, max_volume)
                .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    pub fn tolerance(&self) -> f32 {
        self.inner.tolerance()
    }

    pub fn saturation(&self) -> f32 {
        self.inner.saturation()
    }

    pub fn porosity(&self) -> f32 {
        self.inner.porosity()
    }

    pub fn max_depth(&self) -> usize {
        self.inner.max_depth()
    }

    pub fn max_volume(&self) -> usize {
        self.inner.max_volume()
    }

    pub fn ensure_loop_free(&self, depth: usize) -> PyResult<()> {
        self.inner.ensure_loop_free(depth).map_err(tensor_err_to_py)
    }

    pub fn saturate(&self, value: f32) -> f32 {
        self.inner.saturate(value)
    }

    pub fn site(&self) -> PyResult<PyZBoxSite> {
        Ok(PyZBoxSite::from_site(self.inner.site().clone()))
    }

    pub fn guard_zbox(&self, zbox: &PyZBox) -> PyResult<()> {
        self.inner.guard_zbox(&zbox.inner).map_err(tensor_err_to_py)
    }

    pub fn guard_cover(&self, cover: Vec<Py<PyZBox>>, py: Python<'_>) -> PyResult<()> {
        let mut boxes = Vec::with_capacity(cover.len());
        for handle in cover {
            boxes.push(handle.borrow(py).inner.clone());
        }
        self.inner.guard_cover(&boxes).map_err(tensor_err_to_py)
    }
}

#[pyclass(module = "spiraltorch", name = "ZBox")]
#[derive(Clone)]
pub(crate) struct PyZBox {
    inner: ZBox,
    centers: Vec<Vec<f32>>,
    radii: Vec<f32>,
}

#[pymethods]
impl PyZBox {
    #[new]
    #[pyo3(signature = (centers, radii, density=1.0))]
    pub fn new(centers: Vec<Vec<f32>>, radii: Vec<f32>, density: f32) -> PyResult<Self> {
        let inner = ZBox::new(centers.clone(), radii.clone(), density).map_err(tensor_err_to_py)?;
        Ok(Self {
            inner,
            centers,
            radii,
        })
    }

    #[getter]
    pub fn centers(&self) -> Vec<Vec<f32>> {
        self.centers.clone()
    }

    #[getter]
    pub fn radii(&self) -> Vec<f32> {
        self.radii.clone()
    }

    pub fn arity(&self) -> usize {
        self.inner.arity()
    }

    pub fn density(&self) -> f32 {
        self.inner.density()
    }

    pub fn factor_dimension(&self, index: usize) -> usize {
        self.inner.factor_dimension(index)
    }

    pub fn hyperbolic_volume(&self, curvature: f32) -> PyResult<f32> {
        self.inner.hyperbolic_volume(curvature).map_err(tensor_err_to_py)
    }

    pub fn probability_mass(&self, curvature: f32) -> PyResult<f32> {
        self.inner.probability_mass(curvature).map_err(tensor_err_to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "ZBox(arity={}, density={:.4})",
            self.inner.arity(),
            self.inner.density()
        )
    }
}

#[pyclass(module = "spiraltorch", name = "ZBoxSite")]
#[derive(Clone)]
pub(crate) struct PyZBoxSite {
    inner: ZBoxSite,
}

impl PyZBoxSite {
    pub(crate) fn from_site(inner: ZBoxSite) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyZBoxSite {
    #[staticmethod]
    pub fn default_for(curvature: f32) -> PyResult<Self> {
        let inner = ZBoxSite::default_for(curvature).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn with_radius_window(&self, min: f32, max: f32) -> PyResult<Self> {
        let inner = self
            .inner
            .clone()
            .with_radius_window(min, max)
            .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    pub fn guard_box(&self, zbox: &PyZBox) -> PyResult<()> {
        self.inner.guard_box(&zbox.inner).map_err(tensor_err_to_py)
    }

    pub fn guard_cover(&self, cover: Vec<Py<PyZBox>>, py: Python<'_>) -> PyResult<()> {
        let mut boxes = Vec::with_capacity(cover.len());
        for handle in cover {
            boxes.push(handle.borrow(py).inner.clone());
        }
        self.inner.guard_cover(&boxes).map_err(tensor_err_to_py)
    }
}

#[pyclass(module = "spiraltorch", name = "LanguageWaveEncoder")]
#[derive(Clone)]
pub(crate) struct PyLanguageWaveEncoder {
    pub(crate) inner: LanguageWaveEncoder,
}

#[pymethods]
impl PyLanguageWaveEncoder {
    #[new]
    pub fn new(curvature: f32, temperature: f32) -> PyResult<Self> {
        let inner = LanguageWaveEncoder::new(curvature, temperature).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    pub fn temperature(&self) -> f32 {
        self.inner.temperature()
    }

    pub fn encode_wave(&self, text: &str) -> PyResult<PyComplexTensor> {
        let wave = self.inner.encode_wave(text).map_err(tensor_err_to_py)?;
        Ok(PyComplexTensor { inner: wave })
    }

    pub fn encode_z_space(&self, text: &str) -> PyResult<PyTensor> {
        let tensor = self.inner.encode_z_space(text).map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }
}

#[pyclass(module = "spiraltorch", name = "GradientSummary")]
#[derive(Clone, Copy)]
pub(crate) struct PyGradientSummary {
    inner: GradientSummary,
}

impl From<GradientSummary> for PyGradientSummary {
    fn from(inner: GradientSummary) -> Self {
        Self { inner }
    }
}

impl PyGradientSummary {
    pub(crate) fn as_inner(&self) -> GradientSummary {
        self.inner
    }
}

#[pymethods]
impl PyGradientSummary {
    #[staticmethod]
    pub fn from_values(values: Vec<f32>) -> Self {
        Self {
            inner: GradientSummary::from_slice(&values),
        }
    }

    pub fn l1(&self) -> f32 {
        self.inner.l1()
    }

    pub fn l2(&self) -> f32 {
        self.inner.l2()
    }

    pub fn linf(&self) -> f32 {
        self.inner.linf()
    }

    pub fn count(&self) -> usize {
        self.inner.count()
    }

    pub fn sum(&self) -> f32 {
        self.inner.sum()
    }

    pub fn mean_abs(&self) -> f32 {
        self.inner.mean_abs()
    }

    pub fn rms(&self) -> f32 {
        self.inner.rms()
    }

    pub fn sum_squares(&self) -> f32 {
        self.inner.sum_squares()
    }

    pub fn sum_cubes(&self) -> f32 {
        self.inner.sum_cubes()
    }

    pub fn sum_quartic(&self) -> f32 {
        self.inner.sum_quartic()
    }

    pub fn mean(&self) -> f32 {
        self.inner.mean()
    }

    pub fn min(&self) -> f32 {
        self.inner.min()
    }

    pub fn max(&self) -> f32 {
        self.inner.max()
    }

    pub fn support_width(&self) -> f32 {
        self.inner.support_width()
    }

    pub fn positive_count(&self) -> usize {
        self.inner.positive_count()
    }

    pub fn negative_count(&self) -> usize {
        self.inner.negative_count()
    }

    pub fn zero_count(&self) -> usize {
        self.inner.zero_count()
    }

    pub fn near_zero_count(&self) -> usize {
        self.inner.near_zero_count()
    }

    pub fn positive_fraction(&self) -> f32 {
        self.inner.positive_fraction()
    }

    pub fn negative_fraction(&self) -> f32 {
        self.inner.negative_fraction()
    }

    pub fn zero_fraction(&self) -> f32 {
        self.inner.zero_fraction()
    }

    pub fn near_zero_fraction(&self) -> f32 {
        self.inner.near_zero_fraction()
    }

    pub fn activation(&self) -> f32 {
        self.inner.activation()
    }

    pub fn sign_lean(&self) -> f32 {
        self.inner.sign_lean()
    }

    pub fn sign_entropy(&self) -> f32 {
        self.inner.sign_entropy()
    }

    pub fn variance(&self) -> f32 {
        self.inner.variance()
    }

    pub fn std(&self) -> f32 {
        self.inner.std()
    }

    pub fn skewness(&self) -> f32 {
        self.inner.skewness()
    }

    pub fn kurtosis(&self) -> f32 {
        self.inner.kurtosis()
    }
}

#[pyclass(module = "spiraltorch", name = "NonCollapseSnapshot")]
#[derive(Clone, Default)]
pub(crate) struct PyNonCollapseSnapshot {
    coherence_entropy: Option<f32>,
    preserved_channels: Option<usize>,
    discarded_channels: Option<usize>,
    z_bias: Option<f32>,
    hypergrad_penalty: Option<f32>,
    phase: Option<String>,
    band_energy: Option<(f32, f32, f32)>,
    dominant_channel: Option<usize>,
    energy_ratio: Option<f32>,
    mean_coherence: Option<f32>,
    hypergrad_l2: Option<f32>,
    hypergrad_linf: Option<f32>,
    hypergrad_non_finite_ratio: Option<f32>,
    pre_discard_preserved_ratio: Option<f32>,
    pre_discard_survivor_energy_ratio: Option<f32>,
}

impl From<RustNonCollapseSnapshot> for PyNonCollapseSnapshot {
    fn from(snapshot: RustNonCollapseSnapshot) -> Self {
        Self {
            coherence_entropy: snapshot.coherence_entropy,
            preserved_channels: snapshot.preserved_channels,
            discarded_channels: snapshot.discarded_channels,
            z_bias: snapshot.z_bias,
            hypergrad_penalty: snapshot.hypergrad_penalty,
            phase: snapshot.phase.map(|phase| phase.as_str().to_string()),
            band_energy: snapshot.band_energy,
            dominant_channel: snapshot.dominant_channel,
            energy_ratio: snapshot.energy_ratio,
            mean_coherence: snapshot.mean_coherence,
            hypergrad_l2: snapshot.hypergrad_l2,
            hypergrad_linf: snapshot.hypergrad_linf,
            hypergrad_non_finite_ratio: snapshot.hypergrad_non_finite_ratio,
            pre_discard_preserved_ratio: snapshot.pre_discard_preserved_ratio,
            pre_discard_survivor_energy_ratio: snapshot.pre_discard_survivor_energy_ratio,
        }
    }
}

impl PyNonCollapseSnapshot {
    pub(crate) fn is_empty_inner(&self) -> bool {
        self.coherence_entropy.is_none()
            && self.preserved_channels.is_none()
            && self.discarded_channels.is_none()
            && self.z_bias.is_none()
            && self.hypergrad_penalty.is_none()
            && self.phase.is_none()
            && self.band_energy.is_none()
            && self.dominant_channel.is_none()
            && self.energy_ratio.is_none()
            && self.mean_coherence.is_none()
            && self.hypergrad_l2.is_none()
            && self.hypergrad_linf.is_none()
            && self.hypergrad_non_finite_ratio.is_none()
            && self.pre_discard_preserved_ratio.is_none()
            && self.pre_discard_survivor_energy_ratio.is_none()
    }

    pub(crate) fn to_pydict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        if let Some(value) = self.coherence_entropy {
            dict.set_item("coherence_entropy", value)?;
        }
        if let Some(value) = self.preserved_channels {
            dict.set_item("preserved_channels", value)?;
        }
        if let Some(value) = self.discarded_channels {
            dict.set_item("discarded_channels", value)?;
        }
        if let Some(value) = self.z_bias {
            dict.set_item("z_bias", value)?;
        }
        if let Some(value) = self.hypergrad_penalty {
            dict.set_item("hypergrad_penalty", value)?;
        }
        if let Some(value) = self.phase.as_deref() {
            dict.set_item("phase", value)?;
        }
        if let Some(value) = self.band_energy {
            dict.set_item("band_energy", value)?;
        }
        if let Some(value) = self.dominant_channel {
            dict.set_item("dominant_channel", value)?;
        }
        if let Some(value) = self.energy_ratio {
            dict.set_item("energy_ratio", value)?;
        }
        if let Some(value) = self.mean_coherence {
            dict.set_item("mean_coherence", value)?;
        }
        if let Some(value) = self.hypergrad_l2 {
            dict.set_item("hypergrad_l2", value)?;
        }
        if let Some(value) = self.hypergrad_linf {
            dict.set_item("hypergrad_linf", value)?;
        }
        if let Some(value) = self.hypergrad_non_finite_ratio {
            dict.set_item("hypergrad_non_finite_ratio", value)?;
        }
        if let Some(value) = self.pre_discard_preserved_ratio {
            dict.set_item("pre_discard_preserved_ratio", value)?;
        }
        if let Some(value) = self.pre_discard_survivor_energy_ratio {
            dict.set_item("pre_discard_survivor_energy_ratio", value)?;
        }
        Ok(dict.into_py(py))
    }
}

#[pymethods]
impl PyNonCollapseSnapshot {
    #[getter]
    pub fn coherence_entropy(&self) -> Option<f32> {
        self.coherence_entropy
    }

    #[getter]
    pub fn preserved_channels(&self) -> Option<usize> {
        self.preserved_channels
    }

    #[getter]
    pub fn discarded_channels(&self) -> Option<usize> {
        self.discarded_channels
    }

    #[getter]
    pub fn z_bias(&self) -> Option<f32> {
        self.z_bias
    }

    #[getter]
    pub fn hypergrad_penalty(&self) -> Option<f32> {
        self.hypergrad_penalty
    }

    #[getter]
    pub fn phase(&self) -> Option<String> {
        self.phase.clone()
    }

    #[getter]
    pub fn band_energy(&self) -> Option<(f32, f32, f32)> {
        self.band_energy
    }

    #[getter]
    pub fn dominant_channel(&self) -> Option<usize> {
        self.dominant_channel
    }

    #[getter]
    pub fn energy_ratio(&self) -> Option<f32> {
        self.energy_ratio
    }

    #[getter]
    pub fn mean_coherence(&self) -> Option<f32> {
        self.mean_coherence
    }

    #[getter]
    pub fn hypergrad_l2(&self) -> Option<f32> {
        self.hypergrad_l2
    }

    #[getter]
    pub fn hypergrad_linf(&self) -> Option<f32> {
        self.hypergrad_linf
    }

    #[getter]
    pub fn hypergrad_non_finite_ratio(&self) -> Option<f32> {
        self.hypergrad_non_finite_ratio
    }

    #[getter]
    pub fn pre_discard_preserved_ratio(&self) -> Option<f32> {
        self.pre_discard_preserved_ratio
    }

    #[getter]
    pub fn pre_discard_survivor_energy_ratio(&self) -> Option<f32> {
        self.pre_discard_survivor_energy_ratio
    }

    pub fn is_empty(&self) -> bool {
        self.is_empty_inner()
    }

    pub fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_pydict(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "NonCollapseSnapshot(empty={}, phase={:?})",
            self.is_empty_inner(),
            self.phase
        )
    }
}

#[pyclass(module = "spiraltorch", name = "HypergradTelemetry")]
#[derive(Clone, Copy)]
pub(crate) struct PyHypergradTelemetry {
    inner: HypergradTelemetry,
}

impl From<HypergradTelemetry> for PyHypergradTelemetry {
    fn from(inner: HypergradTelemetry) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyHypergradTelemetry {
    pub fn summary(&self) -> PyGradientSummary {
        self.inner.summary().into()
    }

    pub fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    pub fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }

    pub fn saturation(&self) -> f32 {
        self.inner.saturation()
    }

    pub fn porosity(&self) -> f32 {
        self.inner.porosity()
    }

    pub fn tolerance(&self) -> f32 {
        self.inner.tolerance()
    }

    pub fn max_depth(&self) -> usize {
        self.inner.max_depth()
    }

    pub fn max_volume(&self) -> usize {
        self.inner.max_volume()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    pub fn volume(&self) -> usize {
        self.inner.volume()
    }

    pub fn finite_count(&self) -> usize {
        self.inner.finite_count()
    }

    pub fn non_finite_count(&self) -> usize {
        self.inner.non_finite_count()
    }

    pub fn non_finite_ratio(&self) -> f32 {
        self.inner.non_finite_ratio()
    }

    pub fn noncollapse_snapshot(&self) -> PyNonCollapseSnapshot {
        RustNonCollapseSnapshot::from(self.inner).into()
    }
}

#[pyclass(
    module = "spiraltorch",
    name = "DesireGradientInterpretation",
    unsendable
)]
#[derive(Clone, Copy)]
pub(crate) struct PyDesireGradientInterpretation {
    inner: DesireGradientInterpretation,
}

impl From<DesireGradientInterpretation> for PyDesireGradientInterpretation {
    fn from(inner: DesireGradientInterpretation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDesireGradientInterpretation {
    pub fn hyper_pressure(&self) -> f32 {
        self.inner.hyper_pressure()
    }

    pub fn real_pressure(&self) -> f32 {
        self.inner.real_pressure()
    }

    pub fn balance(&self) -> f32 {
        self.inner.balance()
    }

    pub fn stability(&self) -> f32 {
        self.inner.stability()
    }

    pub fn saturation(&self) -> f32 {
        self.inner.saturation()
    }

    pub fn hyper_std(&self) -> f32 {
        self.inner.hyper_std()
    }

    pub fn real_std(&self) -> f32 {
        self.inner.real_std()
    }

    pub fn sharpness(&self) -> f32 {
        self.inner.sharpness()
    }

    pub fn penalty_gain(&self) -> f32 {
        self.inner.penalty_gain()
    }

    pub fn activation(&self) -> f32 {
        self.inner.activation()
    }

    pub fn sign_alignment(&self) -> f32 {
        self.inner.sign_alignment()
    }

    pub fn sign_entropy(&self) -> f32 {
        self.inner.sign_entropy()
    }
}

#[pyclass(module = "spiraltorch", name = "DesireGradientControl", unsendable)]
#[derive(Clone, Copy)]
pub(crate) struct PyDesireGradientControl {
    inner: DesireGradientControl,
}

impl From<DesireGradientControl> for PyDesireGradientControl {
    fn from(inner: DesireGradientControl) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDesireGradientControl {
    pub fn penalty_gain(&self) -> f32 {
        self.inner.penalty_gain()
    }

    pub fn bias_mix(&self) -> f32 {
        self.inner.bias_mix()
    }

    pub fn observation_gain(&self) -> f32 {
        self.inner.observation_gain()
    }

    pub fn damping(&self) -> f32 {
        self.inner.damping()
    }

    pub fn hyper_rate_scale(&self) -> f32 {
        self.inner.hyper_rate_scale()
    }

    pub fn real_rate_scale(&self) -> f32 {
        self.inner.real_rate_scale()
    }

    pub fn operator_mix(&self) -> f32 {
        self.inner.operator_mix()
    }

    pub fn operator_gain(&self) -> f32 {
        self.inner.operator_gain()
    }

    pub fn tuning_gain(&self) -> f32 {
        self.inner.tuning_gain()
    }

    pub fn target_entropy(&self) -> f32 {
        self.inner.target_entropy()
    }

    pub fn learning_rate_eta(&self) -> f32 {
        self.inner.learning_rate_eta()
    }

    pub fn learning_rate_min(&self) -> f32 {
        self.inner.learning_rate_min()
    }

    pub fn learning_rate_max(&self) -> f32 {
        self.inner.learning_rate_max()
    }

    pub fn learning_rate_slew(&self) -> f32 {
        self.inner.learning_rate_slew()
    }

    pub fn clip_norm(&self) -> f32 {
        self.inner.clip_norm()
    }

    pub fn clip_floor(&self) -> f32 {
        self.inner.clip_floor()
    }

    pub fn clip_ceiling(&self) -> f32 {
        self.inner.clip_ceiling()
    }

    pub fn clip_ema(&self) -> f32 {
        self.inner.clip_ema()
    }

    pub fn temperature_kappa(&self) -> f32 {
        self.inner.temperature_kappa()
    }

    pub fn temperature_slew(&self) -> f32 {
        self.inner.temperature_slew()
    }

    pub fn quality_gain(&self) -> f32 {
        self.inner.quality_gain()
    }

    pub fn quality_bias(&self) -> f32 {
        self.inner.quality_bias()
    }

    pub fn events(&self) -> Vec<&'static str> {
        self.inner.events().labels()
    }
}

#[pyclass(module = "spiraltorch", name = "Hypergrad", unsendable)]
pub(crate) struct PyHypergrad {
    inner: AmegaHypergrad,
}

#[pymethods]
impl PyHypergrad {
    #[new]
    #[pyo3(signature = (curvature, learning_rate, rows, cols, topos=None))]
    pub fn new(
        curvature: f32,
        learning_rate: f32,
        rows: usize,
        cols: usize,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<Self> {
        let inner = match topos {
            Some(guard) => AmegaHypergrad::with_topos(
                curvature,
                learning_rate,
                rows,
                cols,
                guard.inner.clone(),
            ),
            None => AmegaHypergrad::new(curvature, learning_rate, rows, cols),
        }
        .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    pub fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    pub fn gradient(&self) -> Vec<f32> {
        self.inner.gradient().to_vec()
    }

    pub fn summary(&self) -> PyGradientSummary {
        self.inner.summary().into()
    }

    pub fn finite_count(&self) -> usize {
        self.inner.finite_count()
    }

    pub fn non_finite_count(&self) -> usize {
        self.inner.non_finite_count()
    }

    pub fn non_finite_ratio(&self) -> f32 {
        self.inner.non_finite_ratio()
    }

    pub fn has_non_finite(&self) -> bool {
        self.inner.has_non_finite()
    }

    pub fn telemetry(&self) -> PyHypergradTelemetry {
        self.inner.telemetry().into()
    }

    pub fn noncollapse_snapshot(&self) -> PyNonCollapseSnapshot {
        RustNonCollapseSnapshot::from(&self.inner).into()
    }

    pub fn scale_learning_rate(&mut self, factor: f32) {
        self.inner.scale_learning_rate(factor);
    }

    pub fn scale_gradient(&mut self, factor: f32) {
        self.inner.scale_gradient(factor);
    }

    pub fn rescale_rms(&mut self, target_rms: f32) -> f32 {
        self.inner.rescale_rms(target_rms)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn retune(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        self.inner
            .retune(curvature, learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn accumulate_wave(&mut self, tensor: &PyTensor) -> PyResult<()> {
        self.inner
            .accumulate_wave(&tensor.inner)
            .map_err(tensor_err_to_py)
    }

    pub fn accumulate_complex_wave(&mut self, wave: &PyComplexTensor) -> PyResult<()> {
        self.inner
            .accumulate_complex_wave(&wave.inner)
            .map_err(tensor_err_to_py)
    }

    pub fn absorb_text(&mut self, encoder: &PyLanguageWaveEncoder, text: &str) -> PyResult<()> {
        self.inner
            .absorb_text(&encoder.inner, text)
            .map_err(tensor_err_to_py)
    }

    pub fn accumulate_pair(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<()> {
        self.inner
            .accumulate_pair(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)
    }

    pub fn apply(&mut self, mut weights: PyRefMut<'_, PyTensor>) -> PyResult<()> {
        self.inner
            .apply(&mut weights.inner)
            .map_err(tensor_err_to_py)
    }

    pub fn accumulate_barycenter_path(
        &mut self,
        intermediates: Vec<PyBarycenterIntermediate>,
    ) -> PyResult<()> {
        if intermediates.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "barycenter intermediates cannot be empty",
            ));
        }
        let stages: Vec<BarycenterIntermediate> =
            intermediates.into_iter().map(|stage| stage.inner).collect();
        self.inner
            .accumulate_barycenter_path(&stages)
            .map_err(tensor_err_to_py)
    }

    pub fn topos(&self) -> PyOpenCartesianTopos {
        PyOpenCartesianTopos::from_topos(self.inner.topos().clone())
    }

    #[pyo3(signature = (real, gain=None))]
    pub fn desire_control(
        &self,
        real: &PyGradientSummary,
        gain: Option<f32>,
    ) -> PyDesireGradientControl {
        let factor = gain.unwrap_or(1.0);
        self.inner
            .desire_control_with_gain(real.as_inner(), factor)
            .into()
    }

    pub fn desire_interpretation(
        &self,
        real: &PyGradientSummary,
    ) -> PyDesireGradientInterpretation {
        self.inner.desire_interpretation(real.as_inner()).into()
    }
}

#[pyclass(module = "spiraltorch", name = "Realgrad", unsendable)]
pub(crate) struct PyRealgrad {
    inner: AmegaRealgrad,
}

#[pymethods]
impl PyRealgrad {
    #[new]
    pub fn new(learning_rate: f32, rows: usize, cols: usize) -> PyResult<Self> {
        let inner = AmegaRealgrad::new(learning_rate, rows, cols).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    pub fn gradient(&self) -> Vec<f32> {
        self.inner.gradient().to_vec()
    }

    pub fn summary(&self) -> PyGradientSummary {
        self.inner.summary().into()
    }

    pub fn scale_learning_rate(&mut self, factor: f32) {
        self.inner.scale_learning_rate(factor);
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn accumulate_wave(&mut self, tensor: &PyTensor) -> PyResult<()> {
        self.inner
            .accumulate_wave(&tensor.inner)
            .map_err(tensor_err_to_py)
    }

    pub fn accumulate_complex_wave(&mut self, wave: &PyComplexTensor) -> PyResult<()> {
        self.inner
            .accumulate_complex_wave(&wave.inner)
            .map_err(tensor_err_to_py)
    }

    pub fn absorb_text(&mut self, encoder: &PyLanguageWaveEncoder, text: &str) -> PyResult<()> {
        self.inner
            .absorb_text(&encoder.inner, text)
            .map_err(tensor_err_to_py)
    }

    pub fn accumulate_pair(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<()> {
        self.inner
            .accumulate_pair(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)
    }

    pub fn apply(&mut self, mut weights: PyRefMut<'_, PyTensor>) -> PyResult<()> {
        self.inner
            .apply(&mut weights.inner)
            .map_err(tensor_err_to_py)
    }
}

#[pyclass(module = "spiraltorch", name = "TensorBiome", unsendable)]
pub(crate) struct PyTensorBiome {
    pub(crate) inner: TensorBiome,
}

impl PyTensorBiome {
    pub(crate) fn from_biome(inner: TensorBiome) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "spiraltorch", name = "BarycenterIntermediate", unsendable)]
#[derive(Clone)]
pub(crate) struct PyBarycenterIntermediate {
    inner: BarycenterIntermediate,
}

impl From<BarycenterIntermediate> for PyBarycenterIntermediate {
    fn from(inner: BarycenterIntermediate) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "spiraltorch", name = "ZSpaceBarycenter", unsendable)]
pub(crate) struct PyZSpaceBarycenter {
    inner: ZSpaceBarycenter,
}

impl From<ZSpaceBarycenter> for PyZSpaceBarycenter {
    fn from(inner: ZSpaceBarycenter) -> Self {
        Self { inner }
    }
}

#[pyfunction]
#[pyo3(
    name = "z_space_barycenter",
    signature = (weights, densities, entropy_weight, beta_j, coupling=None)
)]
fn py_z_space_barycenter(
    py: Python<'_>,
    weights: Vec<f32>,
    densities: Vec<Py<PyTensor>>,
    entropy_weight: f32,
    beta_j: f32,
    coupling: Option<Py<PyTensor>>,
) -> PyResult<PyZSpaceBarycenter> {
    let density_clones: Vec<Tensor> = densities
        .into_iter()
        .map(|tensor| tensor.bind(py).borrow().inner.clone())
        .collect();
    let coupling_tensor = coupling.map(|tensor| tensor.bind(py).borrow().inner.clone());
    let barycenter = z_space_barycenter_rs(
        &weights,
        &density_clones,
        entropy_weight,
        beta_j,
        coupling_tensor.as_ref(),
    )
    .map_err(tensor_err_to_py)?;
    Ok(PyZSpaceBarycenter::from(barycenter))
}

#[pymethods]
impl PyTensorBiome {
    #[new]
    pub fn new(topos: &PyOpenCartesianTopos) -> Self {
        Self {
            inner: TensorBiome::new(topos.inner.clone()),
        }
    }

    pub fn topos(&self) -> PyOpenCartesianTopos {
        PyOpenCartesianTopos::from_topos(self.inner.topos().clone())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn total_weight(&self) -> f32 {
        self.inner.total_weight()
    }

    pub fn weights(&self) -> Vec<f32> {
        self.inner.weights().to_vec()
    }

    pub fn absorb(&mut self, tensor: &PyTensor) -> PyResult<()> {
        self.inner
            .absorb("py_tensor_biome_absorb", tensor.inner.clone())
            .map_err(tensor_err_to_py)
    }

    pub fn absorb_weighted(&mut self, tensor: &PyTensor, weight: f32) -> PyResult<()> {
        self.inner
            .absorb_weighted(
                "py_tensor_biome_absorb_weighted",
                tensor.inner.clone(),
                weight,
            )
            .map_err(tensor_err_to_py)
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    pub fn canopy(&self) -> PyResult<PyTensor> {
        let tensor = self.inner.canopy().map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }
}

#[pymethods]
impl PyBarycenterIntermediate {
    #[getter]
    fn interpolation(&self) -> f32 {
        self.inner.interpolation
    }

    #[getter]
    fn density(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.density.clone()))
    }

    #[getter]
    fn kl_energy(&self) -> f32 {
        self.inner.kl_energy
    }

    #[getter]
    fn entropy(&self) -> f32 {
        self.inner.entropy
    }

    #[getter]
    fn objective(&self) -> f32 {
        self.inner.objective
    }
}

#[pymethods]
impl PyZSpaceBarycenter {
    #[getter]
    fn density(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.density.clone()))
    }

    #[getter]
    fn kl_energy(&self) -> f32 {
        self.inner.kl_energy
    }

    #[getter]
    fn entropy(&self) -> f32 {
        self.inner.entropy
    }

    #[getter]
    fn coupling_energy(&self) -> f32 {
        self.inner.coupling_energy
    }

    #[getter]
    fn objective(&self) -> f32 {
        self.inner.objective
    }

    #[getter]
    fn effective_weight(&self) -> f32 {
        self.inner.effective_weight
    }

    fn intermediates(&self) -> Vec<PyBarycenterIntermediate> {
        self.inner
            .intermediates
            .iter()
            .cloned()
            .map(PyBarycenterIntermediate::from)
            .collect()
    }
}

pub(crate) fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyComplexTensor>()?;
    m.add_class::<PyOpenCartesianTopos>()?;
    m.add_class::<PyZBox>()?;
    m.add_class::<PyZBoxSite>()?;
    m.add_class::<PyLanguageWaveEncoder>()?;
    m.add_class::<PyGradientSummary>()?;
    m.add_class::<PyNonCollapseSnapshot>()?;
    m.add_class::<PyHypergradTelemetry>()?;
    m.add_class::<PyDesireGradientInterpretation>()?;
    m.add_class::<PyDesireGradientControl>()?;
    m.add_class::<PyHypergrad>()?;
    m.add_class::<PyRealgrad>()?;
    m.add_class::<PyTensorBiome>()?;
    m.add_class::<PyBarycenterIntermediate>()?;
    m.add_class::<PyZSpaceBarycenter>()?;
    m.add_function(wrap_pyfunction!(py_z_space_barycenter, m)?)?;
    Ok(())
}
