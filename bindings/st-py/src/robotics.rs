use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3::PyRef;
use std::collections::HashMap;
use std::time::UNIX_EPOCH;

use st_robotics::{
    Desire, DesireLagrangianField, EnergyReport, FusedFrame, PolicyGradientController,
    PsiTelemetry, RoboticsError, RoboticsRuntime, RuntimeStep, SensorFusionHub, TelemetryReport,
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

    #[pyo3(signature = (name, dimension))]
    pub fn register_channel(&mut self, name: &str, dimension: usize) -> PyResult<()> {
        self.inner
            .register_channel(name, dimension)
            .map_err(robotics_err_to_py)
    }

    #[pyo3(signature = (name, bias=None, scale=None, smoothing=None))]
    pub fn calibrate(
        &mut self,
        py: Python<'_>,
        name: &str,
        bias: Option<Py<PyAny>>,
        scale: Option<f32>,
        smoothing: Option<f32>,
    ) -> PyResult<()> {
        let bias_vec = match bias {
            Some(obj) => Some(obj.bind(py).extract::<Vec<f32>>()?),
            None => None,
        };
        self.inner
            .calibrate(name, bias_vec, scale, smoothing)
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
    pub fn new(mapping: &Bound<PyDict>) -> PyResult<Self> {
        let mut desires = HashMap::with_capacity(mapping.len());
        for (key, value) in mapping.iter() {
            let name: String = key.extract()?;
            let desire = value.extract::<PyRef<PyDesire>>()?;
            desires.insert(name, desire.inner.clone());
        }
        Ok(Self {
            inner: DesireLagrangianField::new(desires),
        })
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
}

#[pyclass(module = "spiraltorch.robotics", name = "PsiTelemetry")]
#[derive(Clone, Debug)]
pub(crate) struct PyPsiTelemetry {
    inner: PsiTelemetry,
}

#[pymethods]
impl PyPsiTelemetry {
    #[new]
    #[pyo3(signature = (window=8, stability_threshold=0.5, failure_energy=5.0, norm_limit=10.0))]
    pub fn new(
        window: usize,
        stability_threshold: f32,
        failure_energy: f32,
        norm_limit: f32,
    ) -> Self {
        Self {
            inner: PsiTelemetry::new(window, stability_threshold, failure_energy, norm_limit),
        }
    }

    pub fn observe(&mut self, frame: &PyFusedFrame, energy: &PyEnergyReport) -> PyTelemetryReport {
        PyTelemetryReport {
            inner: self.inner.observe(&frame.inner, &energy.inner),
        }
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

#[pyclass(module = "spiraltorch.robotics", name = "PolicyGradient")]
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
    module.add_class::<PyDesire>()?;
    module.add_class::<PyDesireLagrangianField>()?;
    module.add_class::<PyEnergyReport>()?;
    module.add_class::<PyPsiTelemetry>()?;
    module.add_class::<PyTelemetryReport>()?;
    module.add_class::<PyPolicyGradientController>()?;
    module.add_class::<PyRoboticsRuntime>()?;
    module.add_class::<PyRuntimeStep>()?;
    module.add("__doc__", "Robotics runtime bindings for SpiralTorch")?;
    parent.add_submodule(&module)?;
    Ok(())
}
