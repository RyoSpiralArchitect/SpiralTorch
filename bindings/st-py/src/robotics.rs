use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::wrap_pyfunction;
use pyo3::PyRef;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::time::UNIX_EPOCH;

use st_robotics::{
    ChannelHealth, Desire, DesireLagrangianField, DriftSafetyPlugin, EnergyReport, FusedFrame,
    GeometryKind, GravityField, GravityRegime, GravityWell, PolicyGradientController, PsiTelemetry,
    RelativityBridge, RoboticsError, RoboticsRuntime, RuntimeStep, SafetyReview, SensorFusionHub,
    SymmetryAnsatz, TelemetryReport, TemporalFeedbackLearner, TemporalFeedbackSummary,
    TrainerEpisode, TrainerMetrics, VisionFeedbackSnapshot, VisionFeedbackSynchronizer,
    ZSpaceDynamics, ZSpaceGeometry, ZSpacePartialObservation, ZSpaceTrainerBridge,
    ZSpaceTrainerEpisodeBuilder, ZSpaceTrainerSample,
};

fn robotics_err_to_py(err: RoboticsError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn extract_vec_f32(value: &Bound<PyAny>) -> PyResult<Vec<f32>> {
    value.extract::<Vec<f32>>()
}

fn dict_to_payloads(dict: &Bound<PyDict>) -> PyResult<HashMap<String, Vec<f32>>> {
    let mut payloads = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let name: String = key.extract()?;
        let vec = extract_vec_f32(&value)?;
        payloads.insert(name, vec);
    }
    Ok(payloads)
}

fn btreemap_to_dict(py: Python<'_>, map: &BTreeMap<String, f32>) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    for (key, value) in map {
        dict.set_item(key, *value)?;
    }
    Ok(dict.into_py(py))
}

fn hashmap_to_dict(py: Python<'_>, map: &HashMap<String, f32>) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    for (key, value) in map {
        dict.set_item(key, *value)?;
    }
    Ok(dict.into_py(py))
}

fn safety_metrics_to_py(py: Python<'_>, review: &SafetyReview) -> PyResult<PyObject> {
    let metrics = &review.metrics;
    let dict = PyDict::new_bound(py);
    dict.set_item("existence_load", metrics.existence_load)?;
    dict.set_item("chi", metrics.chi)?;
    dict.set_item("strict_mode", metrics.strict_mode)?;
    dict.set_item(
        "frame_hazards",
        btreemap_to_dict(py, &metrics.frame_hazards)?,
    )?;
    dict.set_item("safe_radii", btreemap_to_dict(py, &metrics.safe_radii)?)?;
    let word = PyDict::new_bound(py);
    word.set_item("name", &metrics.word.name)?;
    word.set_item("definition_entropy", metrics.word.definition_entropy)?;
    word.set_item("timing_signal", metrics.word.timing_signal)?;
    word.set_item("base_lambda", metrics.word.base_lambda)?;
    word.set_item("beta", metrics.word.beta)?;
    word.set_item("frame_count", metrics.word.frames.len())?;
    dict.set_item("word", word)?;
    dict.set_item("frame_signatures", metrics.frame_signatures.len())?;
    dict.set_item("direction_signatures", metrics.direction_signatures.len())?;
    Ok(dict.into_py(py))
}

fn vectors_from_py(raw: Vec<Vec<f32>>) -> PyResult<Vec<[f32; 4]>> {
    let mut vectors = Vec::with_capacity(raw.len());
    for (idx, vector) in raw.into_iter().enumerate() {
        if vector.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "canvas vector #{idx} must contain four components"
            )));
        }
        vectors.push([vector[0], vector[1], vector[2], vector[3]]);
    }
    Ok(vectors)
}

fn components_from_py(values: Vec<Vec<f64>>) -> PyResult<[[f32; 4]; 4]> {
    if values.len() != 4 {
        return Err(PyValueError::new_err("metric tensor must have four rows"));
    }
    let mut out = [[0.0_f32; 4]; 4];
    for (row_idx, row) in values.into_iter().enumerate() {
        if row.len() != 4 {
            return Err(PyValueError::new_err(
                "metric tensor rows must have four entries",
            ));
        }
        for (col_idx, value) in row.into_iter().enumerate() {
            out[row_idx][col_idx] = value as f32;
        }
    }
    Ok(out)
}

fn parse_ansatz(ansatz: &str) -> SymmetryAnsatz {
    match ansatz.to_ascii_lowercase().as_str() {
        "static_spherical" => SymmetryAnsatz::StaticSpherical,
        "homogeneous_isotropic" => SymmetryAnsatz::HomogeneousIsotropic,
        other => SymmetryAnsatz::Custom(other.to_string()),
    }
}

#[pyfunction]
pub fn relativity_geometry_from_metric(components: Vec<Vec<f64>>) -> PyResult<PyZSpaceGeometry> {
    let matrix = components_from_py(components)?;
    let geometry =
        RelativityBridge::geometry_from_components(matrix).map_err(robotics_err_to_py)?;
    Ok(PyZSpaceGeometry { inner: geometry })
}

#[pyfunction]
#[pyo3(signature = (components, gravity=None))]
pub fn relativity_dynamics_from_metric(
    components: Vec<Vec<f64>>,
    gravity: Option<PyRef<'_, PyGravityField>>,
) -> PyResult<PyZSpaceDynamics> {
    let matrix = components_from_py(components)?;
    let gravity_inner = gravity.map(|value| value.inner.clone());
    let dynamics = RelativityBridge::dynamics_from_components(matrix, gravity_inner)
        .map_err(robotics_err_to_py)?;
    Ok(PyZSpaceDynamics { inner: dynamics })
}

#[pyfunction]
#[pyo3(signature = (ansatz, scale=1.0, gravity=None))]
pub fn relativity_dynamics_from_ansatz(
    ansatz: &str,
    scale: f64,
    gravity: Option<PyRef<'_, PyGravityField>>,
) -> PyResult<PyZSpaceDynamics> {
    let symmetry = parse_ansatz(ansatz);
    let gravity_inner = gravity.map(|value| value.inner.clone());
    let dynamics = RelativityBridge::dynamics_from_ansatz(symmetry, scale, gravity_inner)
        .map_err(robotics_err_to_py)?;
    Ok(PyZSpaceDynamics { inner: dynamics })
}

#[pyclass(module = "spiraltorch.robotics", name = "ZSpaceGeometry")]
#[derive(Clone, Debug)]
pub(crate) struct PyZSpaceGeometry {
    inner: ZSpaceGeometry,
}

#[pymethods]
impl PyZSpaceGeometry {
    #[staticmethod]
    pub fn euclidean() -> Self {
        Self {
            inner: ZSpaceGeometry::euclidean(),
        }
    }

    #[staticmethod]
    pub fn non_euclidean(curvature: f32) -> Self {
        Self {
            inner: ZSpaceGeometry::non_euclidean(curvature),
        }
    }

    #[staticmethod]
    pub fn general_relativity(metric: &Bound<PyAny>, time_dilation: f32) -> PyResult<Self> {
        let matrix = metric.extract::<Vec<Vec<f32>>>()?;
        Ok(Self {
            inner: ZSpaceGeometry::general_relativity(matrix, time_dilation),
        })
    }

    pub fn metric_norm(&self, values: Vec<f32>) -> f32 {
        self.inner.metric_norm(&values)
    }

    #[getter]
    pub fn kind(&self) -> &'static str {
        match self.inner.kind() {
            GeometryKind::Euclidean => "euclidean",
            GeometryKind::NonEuclidean { .. } => "non_euclidean",
            GeometryKind::GeneralRelativity { .. } => "general_relativity",
        }
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "GravityWell")]
#[derive(Clone, Debug)]
pub(crate) struct PyGravityWell {
    inner: GravityWell,
}

#[pymethods]
impl PyGravityWell {
    #[staticmethod]
    pub fn newtonian(mass: f32) -> Self {
        Self {
            inner: GravityWell::new(mass, GravityRegime::Newtonian),
        }
    }

    #[staticmethod]
    pub fn relativistic(mass: f32, speed_of_light: f32) -> Self {
        Self {
            inner: GravityWell::new(mass, GravityRegime::Relativistic { speed_of_light }),
        }
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "GravityField")]
#[derive(Clone, Debug)]
pub(crate) struct PyGravityField {
    inner: GravityField,
}

#[pymethods]
impl PyGravityField {
    #[new]
    #[pyo3(signature = (constant=None))]
    pub fn new(constant: Option<f32>) -> Self {
        let inner = match constant {
            Some(value) => GravityField::new(value),
            None => GravityField::default(),
        };
        Self { inner }
    }

    pub fn add_well(&mut self, channel: &str, well: PyRef<'_, PyGravityWell>) {
        self.inner.add_well(channel.to_string(), well.inner.clone());
    }

    pub fn potential(
        &self,
        channel: &str,
        geometry: PyRef<'_, PyZSpaceGeometry>,
        values: Vec<f32>,
    ) -> Option<f32> {
        let radius = geometry.inner.metric_norm(&values);
        self.inner.potential(channel, radius)
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "ZSpaceDynamics")]
#[derive(Clone, Debug)]
pub(crate) struct PyZSpaceDynamics {
    inner: ZSpaceDynamics,
}

#[pymethods]
impl PyZSpaceDynamics {
    #[new]
    #[pyo3(signature = (geometry=None, gravity=None))]
    pub fn new(
        geometry: Option<PyRef<'_, PyZSpaceGeometry>>,
        gravity: Option<PyRef<'_, PyGravityField>>,
    ) -> Self {
        let geom = geometry
            .map(|value| value.inner.clone())
            .unwrap_or_else(ZSpaceGeometry::default);
        let grav = gravity.map(|field| field.inner.clone());
        Self {
            inner: ZSpaceDynamics::new(geom, grav),
        }
    }

    #[getter]
    pub fn geometry(&self) -> PyZSpaceGeometry {
        PyZSpaceGeometry {
            inner: self.inner.geometry().clone(),
        }
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "SensorFusionHub")]
#[derive(Clone, Debug)]
pub(crate) struct PySensorFusionHub {
    inner: SensorFusionHub,
}

#[pymethods]
impl PySensorFusionHub {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SensorFusionHub::new(),
        }
    }

    #[pyo3(signature = (name, dimension, smoothing=None, optional=false, max_staleness=None))]
    pub fn register_channel(
        &mut self,
        name: &str,
        dimension: usize,
        smoothing: Option<f32>,
        optional: bool,
        max_staleness: Option<f32>,
    ) -> PyResult<()> {
        self.inner
            .register_channel_with_options(name, dimension, smoothing, optional, max_staleness)
            .map_err(robotics_err_to_py)
    }

    #[pyo3(signature = (name, bias=None, scale=None))]
    pub fn calibrate(
        &mut self,
        py: Python<'_>,
        name: &str,
        bias: Option<Py<PyAny>>,
        scale: Option<f32>,
    ) -> PyResult<()> {
        let bias_vec = match bias {
            Some(obj) => Some(obj.bind(py).extract::<Vec<f32>>()?),
            None => None,
        };
        self.inner
            .calibrate(name, bias_vec, scale)
            .map_err(robotics_err_to_py)
    }

    #[pyo3(signature = (name, smoothing))]
    pub fn configure_smoothing(&mut self, name: &str, smoothing: Option<f32>) -> PyResult<()> {
        self.inner
            .configure_smoothing(name, smoothing)
            .map_err(robotics_err_to_py)
    }

    pub fn fuse(&mut self, payloads: &Bound<PyDict>) -> PyResult<PyFusedFrame> {
        let payload_map = dict_to_payloads(payloads)?;
        let frame = self.inner.fuse(&payload_map).map_err(robotics_err_to_py)?;
        Ok(PyFusedFrame { inner: frame })
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "FusedFrame")]
#[derive(Clone, Debug)]
pub(crate) struct PyFusedFrame {
    inner: FusedFrame,
}

#[pymethods]
impl PyFusedFrame {
    #[getter]
    pub fn coordinates(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (key, values) in &self.inner.coordinates {
            dict.set_item(key, values.clone())?;
        }
        Ok(dict.into_py(py))
    }

    pub fn norm(&self, channel: &str) -> Option<f32> {
        self.inner.norm(channel)
    }

    #[getter]
    pub fn timestamp(&self) -> f64 {
        match self.inner.timestamp.duration_since(UNIX_EPOCH) {
            Ok(duration) => duration.as_secs_f64(),
            Err(_) => 0.0,
        }
    }

    #[getter]
    pub fn health(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (key, health) in &self.inner.health {
            let py_health = Py::new(
                py,
                PyChannelHealth {
                    inner: health.clone(),
                },
            )?;
            dict.set_item(key, py_health)?;
        }
        Ok(dict.into_py(py))
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "ChannelHealth")]
#[derive(Clone, Debug)]
pub(crate) struct PyChannelHealth {
    inner: ChannelHealth,
}

#[pymethods]
impl PyChannelHealth {
    #[getter]
    pub fn stale(&self) -> bool {
        self.inner.stale
    }

    #[getter]
    pub fn optional(&self) -> bool {
        self.inner.optional
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "Desire")]
#[derive(Clone, Debug)]
pub(crate) struct PyDesire {
    inner: Desire,
}

#[pymethods]
impl PyDesire {
    #[new]
    #[pyo3(signature = (target_norm, tolerance=0.0, weight=1.0))]
    pub fn new(target_norm: f32, tolerance: f32, weight: f32) -> Self {
        Self {
            inner: Desire {
                target_norm,
                tolerance,
                weight,
            },
        }
    }

    #[getter]
    pub fn target_norm(&self) -> f32 {
        self.inner.target_norm
    }

    #[getter]
    pub fn tolerance(&self) -> f32 {
        self.inner.tolerance
    }

    #[getter]
    pub fn weight(&self) -> f32 {
        self.inner.weight
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "DesireLagrangianField")]
#[derive(Clone, Debug)]
pub(crate) struct PyDesireLagrangianField {
    inner: DesireLagrangianField,
}

#[pymethods]
impl PyDesireLagrangianField {
    #[new]
    #[pyo3(signature = (mapping, dynamics=None))]
    pub fn new(
        mapping: &Bound<PyDict>,
        dynamics: Option<PyRef<'_, PyZSpaceDynamics>>,
    ) -> PyResult<Self> {
        let mut desires = HashMap::with_capacity(mapping.len());
        for (key, value) in mapping.iter() {
            let name: String = key.extract()?;
            let desire = value.extract::<PyRef<PyDesire>>()?;
            desires.insert(name, desire.inner.clone());
        }
        let field = match dynamics {
            Some(spec) => DesireLagrangianField::with_dynamics(desires, spec.inner.clone()),
            None => DesireLagrangianField::new(desires),
        };
        Ok(Self { inner: field })
    }

    pub fn energy(&self, frame: &PyFusedFrame) -> PyEnergyReport {
        PyEnergyReport {
            inner: self.inner.energy(&frame.inner),
        }
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "EnergyReport")]
#[derive(Clone, Debug)]
pub(crate) struct PyEnergyReport {
    inner: EnergyReport,
}

#[pymethods]
impl PyEnergyReport {
    #[getter]
    pub fn total(&self) -> f32 {
        self.inner.total
    }

    #[getter]
    pub fn per_channel(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (key, value) in &self.inner.per_channel {
            dict.set_item(key, *value)?;
        }
        Ok(dict.into_py(py))
    }

    #[getter]
    pub fn gravitational(&self) -> f32 {
        self.inner.gravitational
    }

    #[getter]
    pub fn gravitational_per_channel(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (key, value) in &self.inner.gravitational_per_channel {
            dict.set_item(key, *value)?;
        }
        Ok(dict.into_py(py))
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "PsiTelemetry")]
#[derive(Clone, Debug)]
pub(crate) struct PyPsiTelemetry {
    inner: PsiTelemetry,
}

#[pymethods]
impl PyPsiTelemetry {
    #[new]
    #[pyo3(signature = (window=8, stability_threshold=0.5, failure_energy=5.0, norm_limit=10.0, geometry=None))]
    pub fn new(
        window: usize,
        stability_threshold: f32,
        failure_energy: f32,
        norm_limit: f32,
        geometry: Option<PyRef<'_, PyZSpaceGeometry>>,
    ) -> Self {
        let geom = geometry
            .map(|value| value.inner.clone())
            .unwrap_or_else(ZSpaceGeometry::default);
        Self {
            inner: PsiTelemetry::with_geometry(
                window,
                stability_threshold,
                failure_energy,
                norm_limit,
                geom,
            ),
        }
    }

    pub fn observe(&mut self, frame: &PyFusedFrame, energy: &PyEnergyReport) -> PyTelemetryReport {
        PyTelemetryReport {
            inner: self.inner.observe(&frame.inner, &energy.inner),
        }
    }

    pub fn set_geometry(&mut self, geometry: PyRef<'_, PyZSpaceGeometry>) {
        self.inner.set_geometry(geometry.inner.clone());
    }

    #[getter]
    pub fn window_size(&self) -> usize {
        self.inner.window()
    }

    #[getter]
    pub fn stability_threshold(&self) -> f32 {
        self.inner.stability_threshold()
    }

    #[getter]
    pub fn failure_energy(&self) -> f32 {
        self.inner.failure_energy()
    }

    #[getter]
    pub fn norm_limit(&self) -> f32 {
        self.inner.norm_limit()
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "TelemetryReport")]
#[derive(Clone, Debug)]
pub(crate) struct PyTelemetryReport {
    inner: TelemetryReport,
}

#[pymethods]
impl PyTelemetryReport {
    #[getter]
    pub fn energy(&self) -> f32 {
        self.inner.energy
    }

    #[getter]
    pub fn stability(&self) -> f32 {
        self.inner.stability
    }

    #[getter]
    pub fn failsafe(&self) -> bool {
        self.inner.failsafe
    }

    #[getter]
    pub fn anomalies(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.inner.anomalies.clone().into_py(py))
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "PolicyGradientController")]
#[derive(Clone, Debug)]
pub(crate) struct PyPolicyGradientController {
    inner: PolicyGradientController,
}

#[pymethods]
impl PyPolicyGradientController {
    #[new]
    #[pyo3(signature = (base_learning_rate=0.05, smoothing=0.7))]
    pub fn new(base_learning_rate: f32, smoothing: f32) -> Self {
        Self {
            inner: PolicyGradientController::new(base_learning_rate, smoothing),
        }
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "SafetyReview")]
#[derive(Clone, Debug)]
pub(crate) struct PySafetyReview {
    inner: SafetyReview,
}

#[pymethods]
impl PySafetyReview {
    #[getter]
    pub fn hazard_total(&self) -> f32 {
        self.inner.hazard_total
    }

    #[getter]
    pub fn refused(&self) -> bool {
        self.inner.refused
    }

    #[getter]
    pub fn flagged_frames(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.inner.flagged_frames.clone().into_py(py))
    }

    #[getter]
    pub fn metrics(&self, py: Python<'_>) -> PyResult<PyObject> {
        safety_metrics_to_py(py, &self.inner)
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "DriftSafetyPlugin")]
#[derive(Clone, Debug)]
pub(crate) struct PyDriftSafetyPlugin {
    inner: DriftSafetyPlugin,
}

#[pymethods]
impl PyDriftSafetyPlugin {
    #[new]
    #[pyo3(signature = (word_name="Robotics", hazard_cut=0.8))]
    pub fn new(word_name: &str, hazard_cut: f32) -> Self {
        let mut plugin = DriftSafetyPlugin::new(word_name);
        plugin.set_hazard_cut(hazard_cut);
        Self { inner: plugin }
    }

    pub fn set_threshold(&mut self, channel: &str, hazard: f32) {
        self.inner.set_threshold(channel.to_string(), hazard);
    }

    #[getter]
    pub fn hazard_cut(&self) -> f32 {
        self.inner.hazard_cut()
    }

    #[setter]
    pub fn set_hazard_cut(&mut self, value: f32) {
        self.inner.set_hazard_cut(value);
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "RoboticsRuntime")]
pub(crate) struct PyRoboticsRuntime {
    inner: RoboticsRuntime,
}

impl fmt::Debug for PyRoboticsRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PyRoboticsRuntime").finish()
    }
}

#[pymethods]
impl PyRoboticsRuntime {
    #[new]
    #[pyo3(signature = (sensors, desires, telemetry=None))]
    pub fn new(
        sensors: PyRef<'_, PySensorFusionHub>,
        desires: PyRef<'_, PyDesireLagrangianField>,
        telemetry: Option<PyRef<'_, PyPsiTelemetry>>,
    ) -> Self {
        let telemetry = telemetry
            .map(|wrapper| wrapper.inner.clone())
            .unwrap_or_else(PsiTelemetry::default);
        Self {
            inner: RoboticsRuntime::new(sensors.inner.clone(), desires.inner.clone(), telemetry),
        }
    }

    pub fn attach_policy_gradient(&mut self, controller: PyRef<'_, PyPolicyGradientController>) {
        self.inner.attach_policy_gradient(controller.inner.clone());
    }

    pub fn attach_safety_plugin(&mut self, plugin: PyRef<'_, PyDriftSafetyPlugin>) {
        self.inner.attach_safety_plugin(plugin.inner.clone());
    }

    pub fn clear_safety_plugins(&mut self) {
        self.inner.clear_safety_plugins();
    }

    pub fn step(&mut self, payloads: &Bound<PyDict>) -> PyResult<PyRuntimeStep> {
        let map = dict_to_payloads(payloads)?;
        let step = self.inner.step(map).map_err(robotics_err_to_py)?;
        Ok(PyRuntimeStep { inner: step })
    }

    pub fn configure_dynamics(&mut self, dynamics: PyRef<'_, PyZSpaceDynamics>) {
        self.inner.configure_dynamics(dynamics.inner.clone());
    }

    pub fn enable_recording(&mut self, capacity: usize) {
        self.inner.enable_recording(capacity);
    }

    pub fn recording_len(&self) -> usize {
        self.inner.recording_len()
    }

    pub fn drain_trajectory(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let steps = self.inner.drain_trajectory();
        let list = PyList::empty_bound(py);
        for step in steps {
            let py_step = Py::new(py, PyRuntimeStep { inner: step })?;
            list.append(py_step)?;
        }
        Ok(list.into_py(py))
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "VisionFeedbackSnapshot")]
#[derive(Clone, Debug)]
pub(crate) struct PyVisionFeedbackSnapshot {
    inner: VisionFeedbackSnapshot,
}

#[pymethods]
impl PyVisionFeedbackSnapshot {
    #[getter]
    pub fn channel(&self) -> &str {
        &self.inner.channel
    }

    #[getter]
    pub fn timestamp(&self) -> f64 {
        self.inner
            .timestamp
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs_f64())
            .unwrap_or(0.0)
    }

    #[getter]
    pub fn sensor(&self) -> Vec<f32> {
        self.inner.sensor.clone()
    }

    #[getter]
    pub fn alignment(&self) -> f32 {
        self.inner.alignment
    }

    #[getter]
    pub fn metrics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (key, value) in self.inner.metrics() {
            dict.set_item(key, value)?;
        }
        Ok(dict.into_py(py))
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "VisionFeedbackSynchronizer")]
#[derive(Clone, Debug)]
pub(crate) struct PyVisionFeedbackSynchronizer {
    inner: VisionFeedbackSynchronizer,
}

#[pymethods]
impl PyVisionFeedbackSynchronizer {
    #[new]
    #[pyo3(signature = (channel, *, coherence=1.0, tension=1.0, depth=1))]
    pub fn new(channel: &str, coherence: f32, tension: f32, depth: u32) -> Self {
        Self {
            inner: VisionFeedbackSynchronizer::with_patch(channel, coherence, tension, depth),
        }
    }

    #[getter]
    pub fn channel(&self) -> &str {
        self.inner.channel()
    }

    pub fn set_patch(&mut self, coherence: f32, tension: f32, depth: u32) {
        self.inner.set_patch_parameters(coherence, tension, depth);
    }

    pub fn sync(
        &self,
        step: PyRef<'_, PyRuntimeStep>,
        vectors: Vec<Vec<f32>>,
    ) -> PyResult<PyVisionFeedbackSnapshot> {
        let converted = vectors_from_py(vectors)?;
        let snapshot = self
            .inner
            .sync_with_vectors(&step.inner, &converted)
            .map_err(robotics_err_to_py)?;
        Ok(PyVisionFeedbackSnapshot { inner: snapshot })
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "ZSpacePartialObservation")]
#[derive(Clone, Debug)]
pub(crate) struct PyZSpacePartialObservation {
    inner: ZSpacePartialObservation,
}

#[pymethods]
impl PyZSpacePartialObservation {
    #[getter]
    pub fn metrics(&self, py: Python<'_>) -> PyResult<PyObject> {
        hashmap_to_dict(py, &self.inner.metrics)
    }

    #[getter]
    pub fn commands(&self, py: Python<'_>) -> PyResult<PyObject> {
        hashmap_to_dict(py, &self.inner.commands)
    }

    #[getter]
    pub fn gradient(&self) -> Vec<f32> {
        self.inner.gradient.clone()
    }

    #[getter]
    pub fn weight(&self) -> f32 {
        self.inner.weight
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "TrainerMetrics")]
#[derive(Clone, Debug)]
pub(crate) struct PyTrainerMetrics {
    inner: TrainerMetrics,
}

#[pymethods]
impl PyTrainerMetrics {
    #[getter]
    pub fn speed(&self) -> f32 {
        self.inner.speed
    }

    #[getter]
    pub fn memory(&self) -> f32 {
        self.inner.memory
    }

    #[getter]
    pub fn stability(&self) -> f32 {
        self.inner.stability
    }

    #[getter]
    pub fn gradient(&self) -> Vec<f32> {
        self.inner.gradient.clone()
    }

    #[getter]
    pub fn drs(&self) -> f32 {
        self.inner.drs
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "TemporalFeedbackSummary")]
#[derive(Clone, Debug)]
pub(crate) struct PyTemporalFeedbackSummary {
    inner: TemporalFeedbackSummary,
}

#[pymethods]
impl PyTemporalFeedbackSummary {
    #[getter]
    pub fn discounted_energy(&self) -> f32 {
        self.inner.discounted_energy
    }

    #[getter]
    pub fn discounted_gravity(&self) -> f32 {
        self.inner.discounted_gravity
    }

    #[getter]
    pub fn discounted_stability(&self) -> f32 {
        self.inner.discounted_stability
    }

    #[getter]
    pub fn discounted_alignment(&self) -> Option<f32> {
        self.inner.discounted_alignment
    }

    #[getter]
    pub fn commands(&self, py: Python<'_>) -> PyResult<PyObject> {
        hashmap_to_dict(py, &self.inner.commands)
    }

    #[getter]
    pub fn partial(&self) -> PyZSpacePartialObservation {
        PyZSpacePartialObservation {
            inner: self.inner.partial.clone(),
        }
    }

    #[getter]
    pub fn latest_sensor_norm(&self) -> f32 {
        self.inner.latest_sensor_norm
    }

    #[getter]
    pub fn anomaly_score(&self) -> f32 {
        self.inner.anomaly_score
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "TemporalFeedbackLearner")]
#[derive(Clone, Debug)]
pub(crate) struct PyTemporalFeedbackLearner {
    inner: TemporalFeedbackLearner,
}

#[pymethods]
impl PyTemporalFeedbackLearner {
    #[new]
    #[pyo3(signature = (horizon, *, discount=0.9))]
    pub fn new(horizon: usize, discount: f32) -> PyResult<Self> {
        let inner = TemporalFeedbackLearner::new(horizon, discount).map_err(robotics_err_to_py)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (step, vision=None))]
    pub fn push(
        &mut self,
        step: PyRef<'_, PyRuntimeStep>,
        vision: Option<PyRef<'_, PyVisionFeedbackSnapshot>>,
    ) -> PyResult<PyTemporalFeedbackSummary> {
        let vision_snapshot = vision.map(|value| value.inner.clone());
        let summary = self.inner.push(&step.inner, vision_snapshot.as_ref());
        Ok(PyTemporalFeedbackSummary { inner: summary })
    }

    #[getter]
    pub fn horizon(&self) -> usize {
        self.inner.horizon()
    }

    #[getter]
    pub fn discount(&self) -> f32 {
        self.inner.discount()
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "ZSpaceTrainerSample")]
#[derive(Clone, Debug)]
pub(crate) struct PyZSpaceTrainerSample {
    inner: ZSpaceTrainerSample,
}

#[pymethods]
impl PyZSpaceTrainerSample {
    #[getter]
    pub fn metrics(&self) -> PyTrainerMetrics {
        PyTrainerMetrics {
            inner: self.inner.metrics.clone(),
        }
    }

    #[getter]
    pub fn partial(&self) -> PyZSpacePartialObservation {
        PyZSpacePartialObservation {
            inner: self.inner.partial.clone(),
        }
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "TrainerEpisode")]
#[derive(Clone, Debug)]
pub(crate) struct PyTrainerEpisode {
    inner: TrainerEpisode,
}

#[pymethods]
impl PyTrainerEpisode {
    #[getter]
    pub fn samples(&self) -> Vec<PyZSpaceTrainerSample> {
        self.inner
            .samples
            .iter()
            .cloned()
            .map(|sample| PyZSpaceTrainerSample { inner: sample })
            .collect()
    }

    #[getter]
    pub fn average_memory(&self) -> f32 {
        self.inner.average_memory
    }

    #[getter]
    pub fn average_stability(&self) -> f32 {
        self.inner.average_stability
    }

    #[getter]
    pub fn average_drs(&self) -> f32 {
        self.inner.average_drs
    }

    #[getter]
    pub fn length(&self) -> usize {
        self.inner.length
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "ZSpaceTrainerBridge")]
#[derive(Clone, Debug)]
pub(crate) struct PyZSpaceTrainerBridge {
    inner: ZSpaceTrainerBridge,
}

#[pymethods]
impl PyZSpaceTrainerBridge {
    #[new]
    #[pyo3(signature = (horizon, *, discount=0.9))]
    pub fn new(horizon: usize, discount: f32) -> PyResult<Self> {
        let bridge = ZSpaceTrainerBridge::new(horizon, discount).map_err(robotics_err_to_py)?;
        Ok(Self { inner: bridge })
    }

    #[pyo3(signature = (step, vision=None))]
    pub fn push(
        &mut self,
        step: PyRef<'_, PyRuntimeStep>,
        vision: Option<PyRef<'_, PyVisionFeedbackSnapshot>>,
    ) -> PyResult<PyZSpaceTrainerSample> {
        let vision_snapshot = vision.map(|value| value.inner.clone());
        let sample = self
            .inner
            .push(&step.inner, vision_snapshot.as_ref())
            .map_err(robotics_err_to_py)?;
        Ok(PyZSpaceTrainerSample { inner: sample })
    }

    #[getter]
    pub fn horizon(&self) -> usize {
        self.inner.horizon()
    }

    #[getter]
    pub fn discount(&self) -> f32 {
        self.inner.discount()
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "ZSpaceTrainerEpisodeBuilder")]
#[derive(Clone, Debug)]
pub(crate) struct PyZSpaceTrainerEpisodeBuilder {
    inner: ZSpaceTrainerEpisodeBuilder,
}

#[pymethods]
impl PyZSpaceTrainerEpisodeBuilder {
    #[new]
    #[pyo3(signature = (horizon, *, discount=0.9, capacity=64))]
    pub fn new(horizon: usize, discount: f32, capacity: usize) -> PyResult<Self> {
        let builder = ZSpaceTrainerEpisodeBuilder::new(horizon, discount, capacity)
            .map_err(robotics_err_to_py)?;
        Ok(Self { inner: builder })
    }

    #[pyo3(signature = (step, vision=None, *, end_episode))]
    pub fn push(
        &mut self,
        step: PyRef<'_, PyRuntimeStep>,
        vision: Option<PyRef<'_, PyVisionFeedbackSnapshot>>,
        end_episode: bool,
    ) -> PyResult<Option<PyTrainerEpisode>> {
        let vision_snapshot = vision.map(|value| value.inner.clone());
        let episode = self
            .inner
            .push_step(&step.inner, vision_snapshot.as_ref(), end_episode)
            .map_err(robotics_err_to_py)?;
        Ok(episode.map(|inner| PyTrainerEpisode { inner }))
    }

    pub fn flush(&mut self) -> PyResult<Option<PyTrainerEpisode>> {
        Ok(self.inner.flush().map(|inner| PyTrainerEpisode { inner }))
    }

    #[getter]
    pub fn horizon(&self) -> usize {
        self.inner.horizon()
    }

    #[getter]
    pub fn discount(&self) -> f32 {
        self.inner.discount()
    }

    #[getter]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

#[pyclass(module = "spiraltorch.robotics", name = "RuntimeStep")]
#[derive(Clone, Debug)]
pub(crate) struct PyRuntimeStep {
    inner: RuntimeStep,
}

#[pymethods]
impl PyRuntimeStep {
    #[getter]
    pub fn frame(&self) -> PyFusedFrame {
        PyFusedFrame {
            inner: self.inner.frame.clone(),
        }
    }

    #[getter]
    pub fn energy(&self) -> PyEnergyReport {
        PyEnergyReport {
            inner: self.inner.energy.clone(),
        }
    }

    #[getter]
    pub fn telemetry(&self) -> PyTelemetryReport {
        PyTelemetryReport {
            inner: self.inner.telemetry.clone(),
        }
    }

    #[getter]
    pub fn commands(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (key, value) in &self.inner.commands {
            dict.set_item(key, *value)?;
        }
        Ok(dict.into_py(py))
    }

    #[getter]
    pub fn halted(&self) -> bool {
        self.inner.halted
    }

    #[getter]
    pub fn safety(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty_bound(py);
        for review in &self.inner.safety {
            let review = Py::new(
                py,
                PySafetyReview {
                    inner: review.clone(),
                },
            )?;
            list.append(review)?;
        }
        Ok(list.into_py(py))
    }
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "robotics")?;
    module.add_class::<PySensorFusionHub>()?;
    module.add_class::<PyFusedFrame>()?;
    module.add_class::<PyChannelHealth>()?;
    module.add_class::<PyDesire>()?;
    module.add_class::<PyDesireLagrangianField>()?;
    module.add_class::<PyEnergyReport>()?;
    module.add_class::<PyPsiTelemetry>()?;
    module.add_class::<PyTelemetryReport>()?;
    module.add_class::<PyPolicyGradientController>()?;
    module.add_class::<PyDriftSafetyPlugin>()?;
    module.add_class::<PySafetyReview>()?;
    module.add_class::<PyVisionFeedbackSnapshot>()?;
    module.add_class::<PyVisionFeedbackSynchronizer>()?;
    module.add_class::<PyRoboticsRuntime>()?;
    module.add_class::<PyRuntimeStep>()?;
    module.add_class::<PyZSpacePartialObservation>()?;
    module.add_class::<PyTrainerMetrics>()?;
    module.add_class::<PyTemporalFeedbackSummary>()?;
    module.add_class::<PyTemporalFeedbackLearner>()?;
    module.add_class::<PyZSpaceTrainerSample>()?;
    module.add_class::<PyTrainerEpisode>()?;
    module.add_class::<PyZSpaceTrainerBridge>()?;
    module.add_class::<PyZSpaceTrainerEpisodeBuilder>()?;
    module.add_class::<PyGravityField>()?;
    module.add_class::<PyGravityWell>()?;
    module.add_class::<PyZSpaceDynamics>()?;
    module.add_class::<PyZSpaceGeometry>()?;
    module.add_function(wrap_pyfunction!(relativity_geometry_from_metric, &module)?)?;
    module.add_function(wrap_pyfunction!(relativity_dynamics_from_metric, &module)?)?;
    module.add_function(wrap_pyfunction!(relativity_dynamics_from_ansatz, &module)?)?;
    module.add("__doc__", "Robotics runtime bindings for SpiralTorch")?;
    parent.add_submodule(&module)?;
    Ok(())
}
