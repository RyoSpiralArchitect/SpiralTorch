use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::wrap_pyfunction;
use pyo3::PyRef;
use std::collections::HashMap;
use std::time::UNIX_EPOCH;

use st_robotics::{
    ChannelHealth, Desire, DesireLagrangianField, EnergyReport, FusedFrame, GeometryKind,
    GravityField, GravityRegime, GravityWell, PolicyGradientController, PsiTelemetry,
    RelativityBridge, RoboticsError, RoboticsRuntime, RuntimeStep, SensorFusionHub, SymmetryAnsatz,
    TelemetryReport, ZSpaceDynamics, ZSpaceGeometry,
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
            dict.set_item(
                key,
                PyChannelHealth {
                    inner: health.clone(),
                },
            )?;
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

#[pyclass(module = "spiraltorch.robotics", name = "RoboticsRuntime")]
#[derive(Debug)]
pub(crate) struct PyRoboticsRuntime {
    inner: RoboticsRuntime,
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
            list.append(PyRuntimeStep { inner: step })?;
        }
        Ok(list.into_py(py))
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
    module.add_class::<PyRoboticsRuntime>()?;
    module.add_class::<PyRuntimeStep>()?;
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
