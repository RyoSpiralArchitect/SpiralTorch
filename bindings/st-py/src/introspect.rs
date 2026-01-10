use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use st_core::telemetry::hub;
use st_core::telemetry::hub::{SoftlogicEllipticSample, SoftlogicZFeedback};
use st_core::telemetry::zspace_region::{
    ZSpaceRadiusBand, ZSpaceRegionDescriptor, ZSpaceRegionKey, ZSpaceSpinBand,
};
use st_core::theory::zpulse::{ZScale, ZSource};

type Tuple3 = (f32, f32, f32);
type Tuple3x3 = (Tuple3, Tuple3, Tuple3);

fn zsource_to_string(source: ZSource) -> &'static str {
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

fn scale_to_tuple(scale: Option<ZScale>) -> Option<(f32, f32)> {
    scale.map(|value| (value.physical_radius, value.log_radius))
}

fn rotor_to_tuple(values: [f32; 3]) -> Tuple3 {
    (values[0], values[1], values[2])
}

fn tensor3_to_tuple(values: [[f32; 3]; 3]) -> Tuple3x3 {
    (
        (values[0][0], values[0][1], values[0][2]),
        (values[1][0], values[1][1], values[1][2]),
        (values[2][0], values[2][1], values[2][2]),
    )
}

#[pyclass(module = "spiraltorch.zspace", name = "ZSpaceSpinBand")]
#[derive(Clone, Copy)]
pub(crate) struct PyZSpaceSpinBand {
    inner: ZSpaceSpinBand,
}

impl PyZSpaceSpinBand {
    pub(crate) fn from_band(inner: ZSpaceSpinBand) -> Self {
        Self { inner }
    }

    fn as_str(&self) -> &'static str {
        match self.inner {
            ZSpaceSpinBand::Leading => "leading",
            ZSpaceSpinBand::Neutral => "neutral",
            ZSpaceSpinBand::Trailing => "trailing",
        }
    }
}

#[pymethods]
impl PyZSpaceSpinBand {
    #[getter]
    pub fn label(&self) -> &'static str {
        self.inner.label()
    }

    #[getter]
    pub fn name(&self) -> &'static str {
        self.as_str()
    }

    fn __repr__(&self) -> String {
        format!("ZSpaceSpinBand('{}')", self.as_str())
    }
}

#[pyclass(module = "spiraltorch.zspace", name = "ZSpaceRadiusBand")]
#[derive(Clone, Copy)]
pub(crate) struct PyZSpaceRadiusBand {
    inner: ZSpaceRadiusBand,
}

impl PyZSpaceRadiusBand {
    pub(crate) fn from_band(inner: ZSpaceRadiusBand) -> Self {
        Self { inner }
    }

    fn as_str(&self) -> &'static str {
        match self.inner {
            ZSpaceRadiusBand::Core => "core",
            ZSpaceRadiusBand::Mantle => "mantle",
            ZSpaceRadiusBand::Edge => "edge",
        }
    }
}

#[pymethods]
impl PyZSpaceRadiusBand {
    #[getter]
    pub fn label(&self) -> &'static str {
        self.inner.label()
    }

    #[getter]
    pub fn name(&self) -> &'static str {
        self.as_str()
    }

    fn __repr__(&self) -> String {
        format!("ZSpaceRadiusBand('{}')", self.as_str())
    }
}

#[pyclass(module = "spiraltorch.zspace", name = "ZSpaceRegionKey")]
#[derive(Clone, Copy)]
pub(crate) struct PyZSpaceRegionKey {
    inner: ZSpaceRegionKey,
}

impl PyZSpaceRegionKey {
    pub(crate) fn from_key(inner: ZSpaceRegionKey) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyZSpaceRegionKey {
    #[getter]
    pub fn spin(&self) -> PyZSpaceSpinBand {
        PyZSpaceSpinBand::from_band(self.inner.spin)
    }

    #[getter]
    pub fn radius(&self) -> PyZSpaceRadiusBand {
        PyZSpaceRadiusBand::from_band(self.inner.radius)
    }

    #[getter]
    pub fn label(&self) -> String {
        self.inner.label()
    }

    fn __repr__(&self) -> String {
        format!("ZSpaceRegionKey(label='{}')", self.inner.label())
    }
}

#[pyclass(module = "spiraltorch.zspace", name = "ZSpaceRegionDescriptor")]
#[derive(Clone, Copy)]
pub(crate) struct PyZSpaceRegionDescriptor {
    inner: ZSpaceRegionDescriptor,
}

impl PyZSpaceRegionDescriptor {
    pub(crate) fn from_descriptor(inner: ZSpaceRegionDescriptor) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyZSpaceRegionDescriptor {
    #[new]
    #[pyo3(signature = (*, spin_alignment, normalized_radius, curvature_radius, geodesic_radius, sheet_index, sheet_count, topological_sector))]
    pub fn new(
        spin_alignment: f32,
        normalized_radius: f32,
        curvature_radius: f32,
        geodesic_radius: f32,
        sheet_index: u32,
        sheet_count: u32,
        topological_sector: u32,
    ) -> Self {
        Self {
            inner: ZSpaceRegionDescriptor {
                spin_alignment,
                normalized_radius,
                curvature_radius,
                geodesic_radius,
                sheet_index,
                sheet_count,
                topological_sector,
            },
        }
    }

    #[getter]
    pub fn spin_alignment(&self) -> f32 {
        self.inner.spin_alignment
    }

    #[getter]
    pub fn normalized_radius(&self) -> f32 {
        self.inner.normalized_radius
    }

    #[getter]
    pub fn curvature_radius(&self) -> f32 {
        self.inner.curvature_radius
    }

    #[getter]
    pub fn geodesic_radius(&self) -> f32 {
        self.inner.geodesic_radius
    }

    #[getter]
    pub fn sheet_index(&self) -> u32 {
        self.inner.sheet_index
    }

    #[getter]
    pub fn sheet_count(&self) -> u32 {
        self.inner.sheet_count
    }

    #[getter]
    pub fn topological_sector(&self) -> u32 {
        self.inner.topological_sector
    }

    #[getter]
    pub fn key(&self) -> PyZSpaceRegionKey {
        PyZSpaceRegionKey::from_key(self.inner.key())
    }

    #[getter]
    pub fn spin_band(&self) -> PyZSpaceSpinBand {
        PyZSpaceSpinBand::from_band(self.inner.key().spin)
    }

    #[getter]
    pub fn radius_band(&self) -> PyZSpaceRadiusBand {
        PyZSpaceRadiusBand::from_band(self.inner.key().radius)
    }

    #[getter]
    pub fn label(&self) -> String {
        self.inner.key().label()
    }

    fn __repr__(&self) -> String {
        format!(
            "ZSpaceRegionDescriptor(spin_alignment={:.3}, normalized_radius={:.3})",
            self.inner.spin_alignment, self.inner.normalized_radius
        )
    }
}

#[pyclass(module = "spiraltorch.zspace", name = "SoftlogicEllipticSample")]
#[derive(Clone, Copy)]
pub(crate) struct PySoftlogicEllipticSample {
    inner: SoftlogicEllipticSample,
}

impl PySoftlogicEllipticSample {
    pub(crate) fn from_sample(inner: SoftlogicEllipticSample) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySoftlogicEllipticSample {
    #[getter]
    pub fn curvature_radius(&self) -> f32 {
        self.inner.curvature_radius
    }

    #[getter]
    pub fn geodesic_radius(&self) -> f32 {
        self.inner.geodesic_radius
    }

    #[getter]
    pub fn normalized_radius(&self) -> f32 {
        self.inner.normalized_radius
    }

    #[getter]
    pub fn spin_alignment(&self) -> f32 {
        self.inner.spin_alignment
    }

    #[getter]
    pub fn sheet_index(&self) -> u32 {
        self.inner.sheet_index
    }

    #[getter]
    pub fn sheet_position(&self) -> f32 {
        self.inner.sheet_position
    }

    #[getter]
    pub fn normal_bias(&self) -> f32 {
        self.inner.normal_bias
    }

    #[getter]
    pub fn sheet_count(&self) -> u32 {
        self.inner.sheet_count
    }

    #[getter]
    pub fn topological_sector(&self) -> u32 {
        self.inner.topological_sector
    }

    #[getter]
    pub fn homology_index(&self) -> u32 {
        self.inner.homology_index
    }

    #[getter]
    pub fn rotor_field(&self) -> (f32, f32, f32) {
        rotor_to_tuple(self.inner.rotor_field)
    }

    #[getter]
    pub fn flow_vector(&self) -> (f32, f32, f32) {
        rotor_to_tuple(self.inner.flow_vector)
    }

    #[getter]
    pub fn curvature_tensor(&self) -> Tuple3x3 {
        tensor3_to_tuple(self.inner.curvature_tensor)
    }

    #[getter]
    pub fn resonance_heat(&self) -> f32 {
        self.inner.resonance_heat
    }

    #[getter]
    pub fn noise_density(&self) -> f32 {
        self.inner.noise_density
    }

    #[getter]
    pub fn quaternion(&self) -> (f32, f32, f32, f32) {
        (
            self.inner.quaternion[0],
            self.inner.quaternion[1],
            self.inner.quaternion[2],
            self.inner.quaternion[3],
        )
    }

    #[getter]
    pub fn rotation(&self) -> Tuple3x3 {
        (
            (
                self.inner.rotation[0],
                self.inner.rotation[1],
                self.inner.rotation[2],
            ),
            (
                self.inner.rotation[3],
                self.inner.rotation[4],
                self.inner.rotation[5],
            ),
            (
                self.inner.rotation[6],
                self.inner.rotation[7],
                self.inner.rotation[8],
            ),
        )
    }

    pub fn region_descriptor(&self) -> PyZSpaceRegionDescriptor {
        PyZSpaceRegionDescriptor::from_descriptor((&self.inner).into())
    }
}

#[pyclass(module = "spiraltorch.zspace", name = "SoftlogicZFeedback", unsendable)]
#[derive(Clone)]
pub(crate) struct PySoftlogicZFeedback {
    inner: SoftlogicZFeedback,
}

impl PySoftlogicZFeedback {
    pub(crate) fn from_feedback(inner: SoftlogicZFeedback) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySoftlogicZFeedback {
    #[getter]
    pub fn psi_total(&self) -> f32 {
        self.inner.psi_total
    }

    #[getter]
    pub fn weighted_loss(&self) -> f32 {
        self.inner.weighted_loss
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
    pub fn z_signal(&self) -> f32 {
        self.inner.z_signal
    }

    #[getter]
    pub fn scale(&self) -> Option<(f32, f32)> {
        scale_to_tuple(self.inner.scale)
    }

    #[getter]
    pub fn events(&self) -> Vec<String> {
        self.inner.events.clone()
    }

    #[getter]
    pub fn attributions(&self) -> Vec<(String, f32)> {
        self.inner
            .attributions
            .iter()
            .map(|(source, weight)| (zsource_to_string(*source).to_string(), *weight))
            .collect()
    }

    #[getter]
    pub fn elliptic(&self) -> Option<PySoftlogicEllipticSample> {
        self.inner
            .elliptic
            .map(PySoftlogicEllipticSample::from_sample)
    }

    #[getter]
    pub fn region(&self) -> Option<PyZSpaceRegionDescriptor> {
        self.inner
            .region_descriptor()
            .map(PyZSpaceRegionDescriptor::from_descriptor)
    }

    pub fn has_event(&self, tag: &str) -> bool {
        self.inner.has_event(tag)
    }

    pub fn spin_band(&self) -> Option<PyZSpaceSpinBand> {
        self.inner
            .region_descriptor()
            .map(|descriptor| PyZSpaceSpinBand::from_band(descriptor.key().spin))
    }

    pub fn radius_band(&self) -> Option<PyZSpaceRadiusBand> {
        self.inner
            .region_descriptor()
            .map(|descriptor| PyZSpaceRadiusBand::from_band(descriptor.key().radius))
    }

    pub fn label(&self) -> Option<String> {
        self.inner
            .region_descriptor()
            .map(|descriptor| descriptor.key().label())
    }
}

#[pyfunction]
pub(crate) fn zspace_snapshot(py: Python<'_>) -> PyResult<Option<Py<PyZSpaceRegionDescriptor>>> {
    hub::get_softlogic_z()
        .and_then(|feedback| feedback.region_descriptor())
        .map(PyZSpaceRegionDescriptor::from_descriptor)
        .map(|descriptor| Py::new(py, descriptor))
        .transpose()
}

#[pyfunction]
pub(crate) fn softlogic_feedback(py: Python<'_>) -> PyResult<Option<Py<PySoftlogicZFeedback>>> {
    hub::get_softlogic_z()
        .map(PySoftlogicZFeedback::from_feedback)
        .map(|feedback| Py::new(py, feedback))
        .transpose()
}

#[pyfunction]
#[pyo3(signature = (*, latest=true, feedback=false))]
pub(crate) fn describe_zspace(
    py: Python<'_>,
    latest: bool,
    feedback: bool,
) -> PyResult<Option<PyObject>> {
    if !latest {
        return Err(PyRuntimeError::new_err(
            "only the latest Z-space snapshot is currently available",
        ));
    }

    let Some(sample) = hub::get_softlogic_z() else {
        return Ok(None);
    };

    if feedback {
        let obj = Py::new(py, PySoftlogicZFeedback::from_feedback(sample))?;
        Ok(Some(obj.into_py(py)))
    } else if let Some(descriptor) = sample.region_descriptor() {
        let obj = Py::new(py, PyZSpaceRegionDescriptor::from_descriptor(descriptor))?;
        Ok(Some(obj.into_py(py)))
    } else {
        Ok(None)
    }
}

#[pyfunction]
pub(crate) fn softlogic_signal(py: Python<'_>) -> PyResult<Option<PyObject>> {
    let Some(sample) = hub::get_softlogic_z() else {
        return Ok(None);
    };

    let dict = PyDict::new_bound(py);
    dict.set_item("psi_total", sample.psi_total)?;
    dict.set_item("weighted_loss", sample.weighted_loss)?;
    dict.set_item("band_energy", sample.band_energy)?;
    dict.set_item("drift", sample.drift)?;
    dict.set_item("z_signal", sample.z_signal)?;
    dict.set_item("scale", scale_to_tuple(sample.scale))?;
    let events: Vec<String> = sample.events.clone();
    dict.set_item("events", events)?;
    let attributions: Vec<(String, f32)> = sample
        .attributions
        .iter()
        .map(|(source, weight)| (zsource_to_string(*source).to_string(), *weight))
        .collect();
    dict.set_item("attributions", attributions)?;
    if let Some(descriptor) = sample.region_descriptor() {
        let py_descriptor = Py::new(py, PyZSpaceRegionDescriptor::from_descriptor(descriptor))?;
        dict.set_item("region", py_descriptor)?;
    }
    Ok(Some(dict.into()))
}

pub(crate) fn register_top_level(py: Python<'_>, module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(zspace_snapshot, module)?)?;
    module.add_function(wrap_pyfunction!(softlogic_feedback, module)?)?;
    module.add_function(wrap_pyfunction!(describe_zspace, module)?)?;
    module.add_function(wrap_pyfunction!(softlogic_signal, module)?)?;
    let _ = py;
    Ok(())
}

pub(crate) fn register_submodule(_py: Python<'_>, module: &Bound<PyModule>) -> PyResult<()> {
    module.add("__doc__", "Z-space introspection helpers")?;
    module.add_class::<PyZSpaceSpinBand>()?;
    module.add_class::<PyZSpaceRadiusBand>()?;
    module.add_class::<PyZSpaceRegionKey>()?;
    module.add_class::<PyZSpaceRegionDescriptor>()?;
    module.add_class::<PySoftlogicEllipticSample>()?;
    module.add_class::<PySoftlogicZFeedback>()?;
    module.add_function(wrap_pyfunction!(zspace_snapshot, module)?)?;
    module.add_function(wrap_pyfunction!(softlogic_feedback, module)?)?;
    module.add_function(wrap_pyfunction!(describe_zspace, module)?)?;
    module.add_function(wrap_pyfunction!(softlogic_signal, module)?)?;
    let snapshot = module.getattr("zspace_snapshot")?;
    module.add("snapshot", snapshot)?;
    let feedback = module.getattr("softlogic_feedback")?;
    module.add("feedback", feedback)?;
    let describe = module.getattr("describe_zspace")?;
    module.add("describe", describe)?;
    module.add(
        "__all__",
        vec![
            "ZSpaceSpinBand",
            "ZSpaceRadiusBand",
            "ZSpaceRegionKey",
            "ZSpaceRegionDescriptor",
            "SoftlogicEllipticSample",
            "SoftlogicZFeedback",
            "zspace_snapshot",
            "softlogic_feedback",
            "describe_zspace",
            "snapshot",
            "feedback",
            "describe",
            "softlogic_signal",
        ],
    )?;
    Ok(())
}
