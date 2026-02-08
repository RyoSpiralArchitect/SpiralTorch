use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;
use pyo3::{Bound, Py};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::introspect::PySoftlogicZFeedback;
use st_core::telemetry::atlas;
use st_core::telemetry::atlas::{AtlasFragment, AtlasFrame, AtlasMetric, AtlasRoute};
use st_core::telemetry::dashboard::{
    DashboardEvent, DashboardFrame, DashboardMetric, DashboardRing, EventSeverity,
};
use st_core::telemetry::hub;

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

#[pyclass(name = "DashboardMetric", module = "spiraltorch.telemetry")]
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

#[pyclass(name = "DashboardEvent", module = "spiraltorch.telemetry")]
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

#[pyclass(name = "DashboardFrame", module = "spiraltorch.telemetry")]
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

#[pyclass(name = "DashboardRing", module = "spiraltorch.telemetry", unsendable)]
pub(crate) struct PyDashboardRing {
    inner: DashboardRing,
}

impl PyDashboardRing {
    fn frames(&self) -> Vec<DashboardFrame> {
        self.inner.iter().cloned().collect()
    }
}

#[pyclass(name = "AtlasMetric", module = "spiraltorch.telemetry")]
#[derive(Clone)]
pub(crate) struct PyAtlasMetric {
    inner: AtlasMetric,
}

impl PyAtlasMetric {
    fn from_metric(metric: AtlasMetric) -> Self {
        Self { inner: metric }
    }
}

#[pymethods]
impl PyAtlasMetric {
    #[getter]
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    pub fn value(&self) -> f32 {
        self.inner.value
    }

    #[getter]
    pub fn district(&self) -> Option<&str> {
        self.inner.district.as_deref()
    }
}

#[pyclass(name = "AtlasFragment", module = "spiraltorch.telemetry")]
#[derive(Clone, Default)]
pub(crate) struct PyAtlasFragment {
    inner: AtlasFragment,
}

#[pymethods]
impl PyAtlasFragment {
    #[new]
    #[pyo3(signature = (timestamp=None))]
    pub fn new(timestamp: Option<f64>) -> PyResult<Self> {
        let mut fragment = AtlasFragment::new();
        if let Some(ts) = timestamp {
            let ts = ts as f32;
            if !ts.is_finite() {
                return Err(PyValueError::new_err("timestamp must be finite"));
            }
            fragment.timestamp = Some(ts);
        }
        Ok(Self { inner: fragment })
    }

    #[getter]
    pub fn timestamp(&self) -> Option<f32> {
        self.inner.timestamp
    }

    pub fn set_timestamp(&mut self, timestamp: f64) -> PyResult<()> {
        let ts = timestamp as f32;
        if !ts.is_finite() {
            return Err(PyValueError::new_err("timestamp must be finite"));
        }
        self.inner.timestamp = Some(ts);
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[pyo3(signature = (name, value, district=None))]
    pub fn push_metric(&mut self, name: String, value: f32, district: Option<String>) {
        if let Some(district) = district {
            self.inner.push_metric_with_district(name, value, district);
        } else {
            self.inner.push_metric(name, value);
        }
    }

    pub fn push_note(&mut self, note: String) {
        self.inner.push_note(note);
    }

    pub fn metrics(&self) -> Vec<PyAtlasMetric> {
        self.inner
            .metrics
            .iter()
            .cloned()
            .map(PyAtlasMetric::from_metric)
            .collect()
    }

    pub fn notes(&self) -> Vec<String> {
        self.inner.notes.clone()
    }

    pub fn to_frame(&self) -> Option<PyAtlasFrame> {
        AtlasFrame::from_fragment(self.inner.clone()).map(PyAtlasFrame::from_frame)
    }
}

#[pyclass(name = "AtlasFrame", module = "spiraltorch.telemetry")]
#[derive(Clone)]
pub(crate) struct PyAtlasFrame {
    inner: AtlasFrame,
}

impl PyAtlasFrame {
    pub(crate) fn from_frame(frame: AtlasFrame) -> Self {
        Self { inner: frame }
    }

    pub(crate) fn to_frame(&self) -> AtlasFrame {
        self.inner.clone()
    }
}

#[pymethods]
impl PyAtlasFrame {
    #[staticmethod]
    #[pyo3(signature = (metrics, *, timestamp=None))]
    pub fn from_metrics(
        metrics: std::collections::HashMap<String, f64>,
        timestamp: Option<f64>,
    ) -> PyResult<Self> {
        let mut fragment = AtlasFragment::new();
        if let Some(ts) = timestamp {
            let ts = ts as f32;
            if !ts.is_finite() {
                return Err(PyValueError::new_err("timestamp must be finite"));
            }
            fragment.timestamp = Some(ts);
        } else {
            fragment.timestamp = Some(timestamp_to_seconds(SystemTime::now()) as f32);
        }
        for (name, value) in metrics {
            let value = value as f32;
            fragment.push_metric(name, value);
        }
        AtlasFrame::from_fragment(fragment)
            .map(PyAtlasFrame::from_frame)
            .ok_or_else(|| PyValueError::new_err("empty atlas frame"))
    }

    #[getter]
    pub fn timestamp(&self) -> f32 {
        self.inner.timestamp
    }

    pub fn metric_value(&self, name: &str) -> Option<f32> {
        self.inner.metric_value(name)
    }

    pub fn metrics_with_prefix(&self, prefix: &str) -> Vec<PyAtlasMetric> {
        self.inner
            .metrics_with_prefix(prefix)
            .into_iter()
            .cloned()
            .map(PyAtlasMetric::from_metric)
            .collect()
    }

    pub fn metrics(&self) -> Vec<PyAtlasMetric> {
        self.inner
            .metrics
            .iter()
            .cloned()
            .map(PyAtlasMetric::from_metric)
            .collect()
    }

    pub fn notes(&self) -> Vec<String> {
        self.inner.notes.clone()
    }

    pub fn districts(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let mut out = Vec::new();
        for district in self.inner.districts() {
            let dict = PyDict::new_bound(py);
            dict.set_item("name", district.name)?;
            dict.set_item("mean", district.mean)?;
            dict.set_item("span", district.span)?;
            let metrics: Vec<PyAtlasMetric> = district
                .metrics
                .into_iter()
                .map(PyAtlasMetric::from_metric)
                .collect();
            dict.set_item("metrics", metrics.into_py(py))?;
            out.push(dict.into());
        }
        Ok(out)
    }
}

#[pyclass(name = "AtlasRoute", module = "spiraltorch.telemetry")]
#[derive(Clone, Default)]
pub(crate) struct PyAtlasRoute {
    inner: AtlasRoute,
}

#[pymethods]
impl PyAtlasRoute {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: AtlasRoute::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn latest_timestamp(&self) -> Option<f32> {
        self.inner.latest().map(|frame| frame.timestamp)
    }

    pub fn push_bounded(&mut self, frame: &PyAtlasFrame, bound: usize) {
        self.inner.push_bounded(frame.inner.clone(), bound);
    }

    pub fn summary(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = self.inner.summary();
        let dict = PyDict::new_bound(py);
        dict.set_item("frames", summary.frames)?;
        dict.set_item("latest_timestamp", summary.latest_timestamp)?;
        dict.set_item("mean_loop_support", summary.mean_loop_support)?;
        dict.set_item("loop_std", summary.loop_std)?;
        dict.set_item("loop_min", summary.loop_min)?;
        dict.set_item("loop_max", summary.loop_max)?;
        dict.set_item("loop_total", summary.loop_total)?;
        dict.set_item("loop_trend", summary.loop_trend)?;
        dict.set_item("latest_collapse_total", summary.latest_collapse_total)?;
        dict.set_item("collapse_trend", summary.collapse_trend)?;
        dict.set_item("latest_z_signal", summary.latest_z_signal)?;
        dict.set_item("z_signal_trend", summary.z_signal_trend)?;
        dict.set_item(
            "maintainer_status",
            summary.maintainer_status.map(|status| status.as_str()),
        )?;
        dict.set_item("maintainer_diagnostic", summary.maintainer_diagnostic)?;
        dict.set_item("suggested_max_scale", summary.suggested_max_scale)?;
        dict.set_item("suggested_pressure", summary.suggested_pressure)?;
        dict.set_item("script_hint", summary.script_hint)?;
        dict.set_item("total_notes", summary.total_notes)?;
        dict.set_item("latest_notes", summary.latest_notes)?;

        let districts = pyo3::types::PyList::empty_bound(py);
        for district in summary.districts {
            let district_dict = PyDict::new_bound(py);
            district_dict.set_item("name", district.name)?;
            district_dict.set_item("coverage", district.coverage)?;
            district_dict.set_item("mean", district.mean)?;
            district_dict.set_item("latest", district.latest)?;
            district_dict.set_item("min", district.min)?;
            district_dict.set_item("max", district.max)?;
            district_dict.set_item("delta", district.delta)?;
            district_dict.set_item("std_dev", district.std_dev)?;

            let focus = pyo3::types::PyList::empty_bound(py);
            for metric in district.focus {
                let metric_dict = PyDict::new_bound(py);
                metric_dict.set_item("name", metric.name)?;
                metric_dict.set_item("coverage", metric.coverage)?;
                metric_dict.set_item("mean", metric.mean)?;
                metric_dict.set_item("latest", metric.latest)?;
                metric_dict.set_item("delta", metric.delta)?;
                metric_dict.set_item("momentum", metric.momentum)?;
                metric_dict.set_item("std_dev", metric.std_dev)?;
                focus.append(metric_dict)?;
            }
            district_dict.set_item("focus", focus)?;
            districts.append(district_dict)?;
        }
        dict.set_item("districts", districts)?;

        let concept_pulses = pyo3::types::PyList::empty_bound(py);
        for pulse in summary.concept_pulses {
            let pulse_dict = PyDict::new_bound(py);
            pulse_dict.set_item("term", pulse.term)?;
            pulse_dict.set_item("sense", pulse.sense.label())?;
            pulse_dict.set_item("description", pulse.sense.description())?;
            pulse_dict.set_item("mentions", pulse.mentions)?;
            pulse_dict.set_item("last_rationale", pulse.last_rationale)?;
            concept_pulses.append(pulse_dict)?;
        }
        dict.set_item("concept_pulses", concept_pulses)?;

        Ok(dict.into())
    }

    #[pyo3(signature = (district, focus_prefixes=None))]
    pub fn perspective_for(
        &self,
        py: Python<'_>,
        district: &str,
        focus_prefixes: Option<Vec<String>>,
    ) -> PyResult<Option<Py<PyDict>>> {
        let summary = self.inner.summary();
        let perspective = match focus_prefixes.as_ref() {
            Some(prefixes) if !prefixes.is_empty() => {
                summary.perspective_for_with_focus(district, prefixes)
            }
            _ => summary.perspective_for(district),
        };
        let Some(perspective) = perspective else {
            return Ok(None);
        };

        let dict = PyDict::new_bound(py);
        dict.set_item("district", perspective.district)?;
        dict.set_item("coverage", perspective.coverage)?;
        dict.set_item("mean", perspective.mean)?;
        dict.set_item("latest", perspective.latest)?;
        dict.set_item("delta", perspective.delta)?;
        dict.set_item("momentum", perspective.momentum)?;
        dict.set_item("volatility", perspective.volatility)?;
        dict.set_item("stability", perspective.stability)?;
        dict.set_item("guidance", perspective.guidance)?;

        let focus = pyo3::types::PyList::empty_bound(py);
        for metric in perspective.focus {
            let metric_dict = PyDict::new_bound(py);
            metric_dict.set_item("name", metric.name)?;
            metric_dict.set_item("coverage", metric.coverage)?;
            metric_dict.set_item("mean", metric.mean)?;
            metric_dict.set_item("latest", metric.latest)?;
            metric_dict.set_item("delta", metric.delta)?;
            metric_dict.set_item("momentum", metric.momentum)?;
            metric_dict.set_item("std_dev", metric.std_dev)?;
            focus.append(metric_dict)?;
        }
        dict.set_item("focus", focus)?;

        Ok(Some(dict.into()))
    }

    #[pyo3(signature = (limit=8))]
    pub fn beacons(&self, py: Python<'_>, limit: usize) -> PyResult<Vec<Py<PyDict>>> {
        let summary = self.inner.summary();
        let mut out = Vec::new();
        for beacon in summary.beacons(limit) {
            let dict = PyDict::new_bound(py);
            dict.set_item("district", beacon.district)?;
            dict.set_item("metric", beacon.metric)?;
            dict.set_item("coverage", beacon.coverage)?;
            dict.set_item("mean", beacon.mean)?;
            dict.set_item("latest", beacon.latest)?;
            dict.set_item("delta", beacon.delta)?;
            dict.set_item("momentum", beacon.momentum)?;
            dict.set_item("volatility", beacon.volatility)?;
            dict.set_item("intensity", beacon.intensity)?;
            dict.set_item(
                "trend",
                match beacon.trend {
                    atlas::AtlasBeaconTrend::Rising => "rising",
                    atlas::AtlasBeaconTrend::Falling => "falling",
                    atlas::AtlasBeaconTrend::Steady => "steady",
                },
            )?;
            dict.set_item("narrative", beacon.narrative)?;
            out.push(dict.into());
        }
        Ok(out)
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
    module.add_class::<PyAtlasMetric>()?;
    module.add_class::<PyAtlasFragment>()?;
    module.add_class::<PyAtlasFrame>()?;
    module.add_class::<PyAtlasRoute>()?;
    module.add_function(wrap_pyfunction!(metric_root_token, &module)?)?;
    module.add_function(wrap_pyfunction!(infer_district, &module)?)?;
    module.add_function(wrap_pyfunction!(current, &module)?)?;
    module.add(
        "__all__",
        vec![
            "DashboardMetric",
            "DashboardEvent",
            "DashboardFrame",
            "DashboardRing",
            "AtlasMetric",
            "AtlasFragment",
            "AtlasFrame",
            "AtlasRoute",
            "metric_root_token",
            "infer_district",
            "current",
        ],
    )?;
    parent.add_submodule(&module)?;
    parent.add("DashboardMetric", module.getattr("DashboardMetric")?)?;
    parent.add("DashboardEvent", module.getattr("DashboardEvent")?)?;
    parent.add("DashboardFrame", module.getattr("DashboardFrame")?)?;
    parent.add("DashboardRing", module.getattr("DashboardRing")?)?;
    if let Ok(zspace) = parent.getattr("zspace") {
        if let Ok(feedback_cls) = zspace.getattr("SoftlogicZFeedback") {
            module.add("SoftlogicZFeedback", feedback_cls)?;
        }
        if let Ok(descriptor_cls) = zspace.getattr("ZSpaceRegionDescriptor") {
            module.add("ZSpaceRegionDescriptor", descriptor_cls)?;
        }
    }
    Ok(())
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    register_impl(py, parent)
}

#[pyfunction]
fn metric_root_token(name: &str) -> String {
    atlas::metric_root_token(name).to_string()
}

#[pyfunction]
fn infer_district(name: &str) -> &'static str {
    atlas::infer_district(name)
}

#[pyfunction]
fn current(py: Python<'_>) -> PyResult<Option<Py<PySoftlogicZFeedback>>> {
    hub::get_softlogic_z()
        .map(PySoftlogicZFeedback::from_feedback)
        .map(|feedback| Py::new(py, feedback))
        .transpose()
}
