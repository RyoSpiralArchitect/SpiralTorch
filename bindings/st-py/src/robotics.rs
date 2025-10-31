use std::collections::HashMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
use pyo3::Bound;

use st_robotics::{
    Desire, DesireEnergy, DesireLagrangianField, FusedFrame, PsiTelemetry, RoboticsError,
    RoboticsRuntime, SensorFusionHub, TelemetryReport,
};

fn robotics_err_to_py(err: RoboticsError) -> PyErr {
    match err {
        RoboticsError::MissingChannel(_) => PyValueError::new_err(err.to_string()),
        RoboticsError::CalibrationDimension { .. } => PyValueError::new_err(err.to_string()),
        RoboticsError::Policy(message) => PyValueError::new_err(message),
    }
}

fn coerce_payload<'py>(payload: &'py PyAny) -> PyResult<Vec<f64>> {
    if let Ok(value) = payload.extract::<Vec<f64>>() {
        return Ok(value);
    }
    if let Ok(value) = payload.extract::<f64>() {
        return Ok(vec![value]);
    }
    if let Ok(tuple) = payload.extract::<Vec<f32>>() {
        return Ok(tuple.into_iter().map(f64::from).collect());
    }
    Err(PyTypeError::new_err(
        "sensor payload must be a float, list[float], or tuple[float]",
    ))
}

fn coerce_bias<'py>(bias: Option<&'py PyAny>) -> PyResult<Vec<f64>> {
    match bias {
        Some(value) => coerce_payload(value),
        None => Ok(Vec::new()),
    }
}

#[pyclass(name = "FusedFrame", module = "spiraltorch.robotics")]
#[derive(Clone)]
pub(crate) struct PyFusedFrame {
    pub(crate) inner: FusedFrame,
}

#[pymethods]
impl PyFusedFrame {
    #[getter]
    pub fn coordinates(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (name, values) in self.inner.iter() {
            dict.set_item(name, values.to_vec())?;
        }
        Ok(dict.unbind())
    }

    pub fn norm(&self, name: &str) -> Option<f64> {
        self.inner.norm(name)
    }
}

#[pyclass(name = "Desire", module = "spiraltorch.robotics")]
#[derive(Clone)]
pub(crate) struct PyDesire {
    pub(crate) inner: Desire,
}

#[pymethods]
impl PyDesire {
    #[new]
    #[pyo3(signature = (target_norm, tolerance=0.0, weight=1.0))]
    pub fn new(target_norm: f64, tolerance: f64, weight: f64) -> Self {
        Self {
            inner: Desire::new(target_norm, tolerance, weight),
        }
    }

    #[getter]
    pub fn target_norm(&self) -> f64 {
        self.inner.target_norm
    }

    #[getter]
    pub fn tolerance(&self) -> f64 {
        self.inner.tolerance
    }

    #[getter]
    pub fn weight(&self) -> f64 {
        self.inner.weight
    }
}

#[pyclass(name = "DesireEnergy", module = "spiraltorch.robotics")]
#[derive(Clone)]
pub(crate) struct PyDesireEnergy {
    pub(crate) inner: DesireEnergy,
}

#[pymethods]
impl PyDesireEnergy {
    #[getter]
    pub fn total(&self) -> f64 {
        self.inner.total
    }

    #[getter]
    pub fn per_channel(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (name, value) in &self.inner.per_channel {
            dict.set_item(name, *value)?;
        }
        Ok(dict.unbind())
    }
}

#[pyclass(name = "DesireLagrangianField", module = "spiraltorch.robotics")]
#[derive(Clone)]
pub(crate) struct PyDesireField {
    pub(crate) inner: DesireLagrangianField,
}

#[pymethods]
impl PyDesireField {
    #[new]
    pub fn new(mapping: Option<HashMap<String, PyDesire>>) -> Self {
        let desires = mapping
            .unwrap_or_default()
            .into_iter()
            .map(|(name, desire)| (name, desire.inner))
            .collect();
        Self {
            inner: DesireLagrangianField::new(desires),
        }
    }

    pub fn energy(&self, frame: &PyFusedFrame) -> PyDesireEnergy {
        PyDesireEnergy {
            inner: self.inner.energy(&frame.inner),
        }
    }
}

#[pyclass(name = "TelemetryReport", module = "spiraltorch.robotics")]
#[derive(Clone)]
pub(crate) struct PyTelemetryReport {
    pub(crate) inner: TelemetryReport,
}

#[pymethods]
impl PyTelemetryReport {
    #[getter]
    pub fn stability(&self) -> f64 {
        self.inner.stability
    }

    #[getter]
    pub fn energy(&self) -> f64 {
        self.inner.energy
    }

    #[getter]
    pub fn anomalies(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.inner.anomalies.clone().into_py(py))
    }

    #[getter]
    pub fn failsafe(&self) -> bool {
        self.inner.failsafe
    }
}

#[pyclass(name = "PsiTelemetry", module = "spiraltorch.robotics")]
#[derive(Clone)]
pub(crate) struct PyPsiTelemetry {
    pub(crate) inner: PsiTelemetry,
}

#[pymethods]
impl PyPsiTelemetry {
    #[new]
    #[pyo3(signature = (window=8, stability_threshold=0.5, failure_energy=5.0, norm_limit=None))]
    pub fn new(
        window: usize,
        stability_threshold: f64,
        failure_energy: f64,
        norm_limit: Option<f64>,
    ) -> Self {
        let limit = norm_limit.unwrap_or(f64::INFINITY);
        Self {
            inner: PsiTelemetry::new(window, stability_threshold, failure_energy, limit),
        }
    }

    pub fn observe(&mut self, frame: &PyFusedFrame, energy: &PyDesireEnergy) -> PyTelemetryReport {
        PyTelemetryReport {
            inner: self.inner.observe(&frame.inner, &energy.inner),
        }
    }
}

#[pyclass(name = "SensorFusionHub", module = "spiraltorch.robotics")]
#[derive(Clone)]
pub(crate) struct PySensorFusionHub {
    pub(crate) inner: SensorFusionHub,
    extractors: HashMap<String, Py<PyAny>>,
}

#[pymethods]
impl PySensorFusionHub {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SensorFusionHub::new(),
            extractors: HashMap::new(),
        }
    }

    #[pyo3(signature = (name, extractor=None))]
    pub fn register_channel(&mut self, name: &str, extractor: Option<Py<PyAny>>) {
        if let Some(callback) = extractor {
            self.extractors.insert(name.to_string(), callback);
        }
        self.inner.register_channel(name.to_string());
    }

    #[pyo3(signature = (name, bias=None, scale=None))]
    pub fn calibrate(
        &mut self,
        name: &str,
        bias: Option<&PyAny>,
        scale: Option<f64>,
    ) -> PyResult<()> {
        let bias_vec = coerce_bias(bias)?;
        let scale_value = scale.unwrap_or(1.0);
        self.inner
            .calibrate(name, bias_vec, scale_value)
            .map_err(robotics_err_to_py)
    }

    pub fn fuse(&mut self, py: Python<'_>, payloads: &PyAny) -> PyResult<PyFusedFrame> {
        let dict: &PyDict = payloads.downcast()?;
        let mut converted = HashMap::new();
        for (key, value) in dict.iter() {
            let name: String = key.extract()?;
            let data = if let Some(extractor) = self.extractors.get(&name) {
                let result = extractor.as_ref(py).call1((value,))?;
                coerce_payload(result.as_ref(py))?
            } else {
                coerce_payload(value)?
            };
            converted.insert(name, data);
        }
        let frame = self.inner.fuse(&converted).map_err(robotics_err_to_py)?;
        Ok(PyFusedFrame { inner: frame })
    }
}

struct PyPolicyAdapter {
    policy: Py<PyAny>,
}

impl st_robotics::PolicyController for PyPolicyAdapter {
    fn step(&mut self, returns: &[f64]) -> Result<HashMap<String, f64>, RoboticsError> {
        Python::with_gil(|py| {
            let args = (returns.to_vec(),);
            let result = self
                .policy
                .as_ref(py)
                .call_method("step", args, None)
                .map_err(|err| RoboticsError::Policy(err.to_string()))?;
            result
                .extract::<HashMap<String, f64>>()
                .map_err(|err| RoboticsError::Policy(err.to_string()))
        })
    }
}

#[pyclass(name = "RuntimeStep", module = "spiraltorch.robotics")]
pub(crate) struct PyRuntimeStep {
    #[pyo3(get)]
    frame: PyFusedFrame,
    #[pyo3(get)]
    energy: PyDesireEnergy,
    #[pyo3(get)]
    telemetry: PyTelemetryReport,
    #[pyo3(get)]
    commands: HashMap<String, f64>,
    #[pyo3(get)]
    halted: bool,
}

#[pyclass(name = "RoboticsRuntime", module = "spiraltorch.robotics")]
pub(crate) struct PyRoboticsRuntime {
    inner: RoboticsRuntime,
    extractors: HashMap<String, Py<PyAny>>,
    policy: Option<Py<PyAny>>,
}

#[pymethods]
impl PyRoboticsRuntime {
    #[new]
    #[pyo3(signature = (sensors, desires, telemetry=None))]
    pub fn new(
        sensors: &PySensorFusionHub,
        desires: &PyDesireField,
        telemetry: Option<&PyPsiTelemetry>,
    ) -> Self {
        let telemetry_inner = telemetry
            .map(|value| value.inner.clone())
            .unwrap_or_else(PsiTelemetry::default);
        Self {
            inner: RoboticsRuntime::new(
                sensors.inner.clone(),
                desires.inner.clone(),
                telemetry_inner,
            ),
            extractors: sensors.extractors.clone(),
            policy: None,
        }
    }

    pub fn attach_policy_gradient(&mut self, policy: Py<PyAny>) {
        self.policy = Some(policy.clone());
        self.inner
            .attach_policy(Box::new(PyPolicyAdapter { policy }));
    }

    pub fn clear_policy_gradient(&mut self) {
        self.policy = None;
        self.inner.clear_policy();
    }

    pub fn step(&mut self, py: Python<'_>, payloads: &PyAny) -> PyResult<PyRuntimeStep> {
        let dict: &PyDict = payloads.downcast()?;
        let mut converted = HashMap::new();
        for (key, value) in dict.iter() {
            let name: String = key.extract()?;
            let data = if let Some(extractor) = self.extractors.get(&name) {
                let result = extractor.as_ref(py).call1((value,))?;
                coerce_payload(result.as_ref(py))?
            } else {
                coerce_payload(value)?
            };
            converted.insert(name, data);
        }
        let step = self.inner.step(&converted).map_err(robotics_err_to_py)?;
        Ok(PyRuntimeStep {
            frame: PyFusedFrame { inner: step.frame },
            energy: PyDesireEnergy { inner: step.energy },
            telemetry: PyTelemetryReport {
                inner: step.telemetry,
            },
            commands: step.commands,
            halted: step.halted,
        })
    }
}

fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "robotics")?;
    module.add_class::<PySensorFusionHub>()?;
    module.add_class::<PyFusedFrame>()?;
    module.add_class::<PyDesire>()?;
    module.add_class::<PyDesireField>()?;
    module.add_class::<PyDesireEnergy>()?;
    module.add_class::<PyPsiTelemetry>()?;
    module.add_class::<PyTelemetryReport>()?;
    module.add_class::<PyRoboticsRuntime>()?;
    module.add_class::<PyRuntimeStep>()?;
    module.add(
        "__all__",
        vec![
            "SensorFusionHub",
            "FusedFrame",
            "Desire",
            "DesireLagrangianField",
            "DesireEnergy",
            "PsiTelemetry",
            "TelemetryReport",
            "RoboticsRuntime",
            "RuntimeStep",
        ],
    )?;
    parent.add_submodule(&module)?;
    Ok(())
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    register_impl(py, parent)
}
