use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::{Bound, Py};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use st_core::telemetry::dashboard::{
    DashboardEvent, DashboardFrame, DashboardMetric, DashboardRing, EventSeverity,
};

fn severity_from_str(value: &str) -> PyResult<EventSeverity> {
    match value.to_ascii_lowercase().as_str() {
        "info" => Ok(EventSeverity::Info),
        "warning" => Ok(EventSeverity::Warning),
        "critical" => Ok(EventSeverity::Critical),
        other => Err(PyValueError::new_err(format!("unknown severity '{other}'"))),
    }
}

fn severity_to_str(severity: EventSeverity) -> &'static str {
    match severity {
        EventSeverity::Info => "info",
        EventSeverity::Warning => "warning",
        EventSeverity::Critical => "critical",
    }
}

fn timestamp_to_seconds(time: SystemTime) -> f64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs_f64(),
        Err(err) => {
            let duration = err.duration();
            -(duration.as_secs_f64())
        }
    }
}

fn seconds_to_timestamp(seconds: Option<f64>) -> SystemTime {
    match seconds {
        Some(value) if value >= 0.0 => UNIX_EPOCH + Duration::from_secs_f64(value),
        Some(value) => UNIX_EPOCH - Duration::from_secs_f64(value.abs()),
        None => SystemTime::now(),
    }
}

#[pyclass(module = "spiraltorch.telemetry", name = "DashboardMetric")]
#[derive(Clone)]
pub(crate) struct PyDashboardMetric {
    inner: DashboardMetric,
}

impl PyDashboardMetric {
    fn from_metric(metric: DashboardMetric) -> Self {
        Self { inner: metric }
    }
}

#[pymethods]
impl PyDashboardMetric {
    #[new]
    #[pyo3(signature = (name, value, unit=None, trend=None))]
    pub fn new(name: String, value: f64, unit: Option<String>, trend: Option<f64>) -> Self {
        let mut metric = DashboardMetric::new(name, value);
        if let Some(unit) = unit {
            metric = metric.with_unit(unit);
        }
        if let Some(trend) = trend {
            metric = metric.with_trend(trend);
        }
        Self { inner: metric }
    }

    #[getter]
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    pub fn value(&self) -> f64 {
        self.inner.value
    }

    #[getter]
    pub fn unit(&self) -> Option<&str> {
        self.inner.unit.as_deref()
    }

    #[getter]
    pub fn trend(&self) -> Option<f64> {
        self.inner.trend
    }
}

#[pyclass(module = "spiraltorch.telemetry", name = "DashboardEvent")]
#[derive(Clone)]
pub(crate) struct PyDashboardEvent {
    inner: DashboardEvent,
}

impl PyDashboardEvent {
    fn from_event(event: DashboardEvent) -> Self {
        Self { inner: event }
    }
}

#[pymethods]
impl PyDashboardEvent {
    #[new]
    pub fn new(message: String, severity: &str) -> PyResult<Self> {
        let severity = severity_from_str(severity)?;
        Ok(Self {
            inner: DashboardEvent { message, severity },
        })
    }

    #[getter]
    pub fn message(&self) -> &str {
        &self.inner.message
    }

    #[getter]
    pub fn severity(&self) -> &'static str {
        severity_to_str(self.inner.severity)
    }
}

#[pyclass(module = "spiraltorch.telemetry", name = "DashboardFrame")]
#[derive(Clone)]
pub(crate) struct PyDashboardFrame {
    pub(crate) inner: DashboardFrame,
}

impl PyDashboardFrame {
    fn from_frame(frame: DashboardFrame) -> Self {
        Self { inner: frame }
    }
}

#[pymethods]
impl PyDashboardFrame {
    #[new]
    #[pyo3(signature = (timestamp=None))]
    pub fn new(timestamp: Option<f64>) -> Self {
        let ts = seconds_to_timestamp(timestamp);
        Self {
            inner: DashboardFrame {
                timestamp: ts,
                metrics: Vec::new(),
                events: Vec::new(),
            },
        }
    }

    pub fn push_metric(&mut self, metric: &PyDashboardMetric) {
        self.inner.metrics.push(metric.inner.clone());
    }

    pub fn push_event(&mut self, event: &PyDashboardEvent) {
        self.inner.events.push(event.inner.clone());
    }

    #[getter]
    pub fn timestamp(&self) -> f64 {
        timestamp_to_seconds(self.inner.timestamp)
    }

    #[getter]
    pub fn metrics(&self) -> Vec<PyDashboardMetric> {
        self.inner
            .metrics
            .iter()
            .cloned()
            .map(PyDashboardMetric::from_metric)
            .collect()
    }

    #[getter]
    pub fn events(&self) -> Vec<PyDashboardEvent> {
        self.inner
            .events
            .iter()
            .cloned()
            .map(PyDashboardEvent::from_event)
            .collect()
    }
}

#[pyclass(module = "spiraltorch.telemetry", name = "DashboardRing", unsendable)]
pub(crate) struct PyDashboardRing {
    inner: DashboardRing,
}

impl PyDashboardRing {
    fn frames(&self) -> Vec<DashboardFrame> {
        self.inner.iter().cloned().collect()
    }
}

#[pymethods]
impl PyDashboardRing {
    #[new]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: DashboardRing::new(capacity),
        }
    }

    pub fn push(&mut self, frame: &PyDashboardFrame) {
        self.inner.push(frame.inner.clone());
    }

    pub fn latest(&self) -> Option<PyDashboardFrame> {
        self.inner
            .latest()
            .cloned()
            .map(PyDashboardFrame::from_frame)
    }

    pub fn iter(&self, py: Python<'_>) -> PyResult<Py<PyDashboardRingIter>> {
        Py::new(py, PyDashboardRingIter::new(self.frames()))
    }

    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyDashboardRingIter>> {
        slf.iter(slf.py())
    }
}

#[pyclass(
    module = "spiraltorch.telemetry",
    name = "DashboardRingIter",
    unsendable
)]
pub(crate) struct PyDashboardRingIter {
    frames: Vec<DashboardFrame>,
    index: usize,
}

impl PyDashboardRingIter {
    fn new(frames: Vec<DashboardFrame>) -> Self {
        Self { frames, index: 0 }
    }
}

#[pymethods]
impl PyDashboardRingIter {
    fn __iter__(slf: PyRef<'_, Self>) -> Py<PyDashboardRingIter> {
        slf.into()
    }

    fn __next__(&mut self) -> Option<PyDashboardFrame> {
        if self.index >= self.frames.len() {
            return None;
        }
        let frame = self.frames[self.index].clone();
        self.index += 1;
        Some(PyDashboardFrame::from_frame(frame))
    }
}

fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "telemetry")?;
    module.add("__doc__", "Telemetry dashboards and runtime metrics")?;
    module.add_class::<PyDashboardMetric>()?;
    module.add_class::<PyDashboardEvent>()?;
    module.add_class::<PyDashboardFrame>()?;
    module.add_class::<PyDashboardRing>()?;
    module.add_class::<PyDashboardRingIter>()?;
    module.add(
        "__all__",
        vec![
            "DashboardMetric",
            "DashboardEvent",
            "DashboardFrame",
            "DashboardRing",
        ],
    )?;
    parent.add_submodule(&module)?;
    parent.add("DashboardMetric", module.getattr("DashboardMetric")?)?;
    parent.add("DashboardEvent", module.getattr("DashboardEvent")?)?;
    parent.add("DashboardFrame", module.getattr("DashboardFrame")?)?;
    parent.add("DashboardRing", module.getattr("DashboardRing")?)?;
    Ok(())
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    register_impl(py, parent)
}
