use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use st_core::theory::microlocal::{
    EllipticAnchoredLearningBatch, EllipticCausalLearningBatch, EllipticGatedCausalLearningBatch,
    EllipticLearningBatch, EllipticTelemetry, EllipticWarp,
};

type EllipticDifferential = (PyEllipticTelemetry, Vec<f32>, Vec<Vec<f32>>);

#[pyclass(name = "EllipticWarp", module = "spiraltorch")]
pub struct PyEllipticWarp {
    warp: EllipticWarp,
}

#[pyclass(name = "EllipticLearningBatch", module = "spiraltorch", frozen)]
pub struct PyEllipticLearningBatch {
    inner: EllipticLearningBatch,
}

#[pyclass(name = "EllipticCausalLearningBatch", module = "spiraltorch", frozen)]
pub struct PyEllipticCausalLearningBatch {
    inner: EllipticCausalLearningBatch,
}

#[pyclass(name = "EllipticAnchoredLearningBatch", module = "spiraltorch", frozen)]
pub struct PyEllipticAnchoredLearningBatch {
    inner: EllipticAnchoredLearningBatch,
}

#[pymethods]
impl PyEllipticAnchoredLearningBatch {
    #[getter]
    fn features(&self) -> Vec<f32> {
        self.inner.features().to_vec()
    }

    #[getter]
    fn mix(&self) -> f32 {
        self.inner.mix()
    }

    fn jvp(&self, py: Python<'_>, orientations: Vec<f32>, raw_mix: f32) -> PyResult<Vec<f32>> {
        py.detach(|| self.inner.jvp(&orientations, raw_mix))
            .map_err(value_error)
    }

    fn vjp(&self, py: Python<'_>, upstream: Vec<f32>) -> PyResult<(Vec<f32>, f32)> {
        py.detach(|| self.inner.vjp(&upstream))
            .map(|g| (g.orientations, g.raw_mix))
            .map_err(value_error)
    }
}

#[pyclass(
    name = "EllipticGatedCausalLearningBatch",
    module = "spiraltorch",
    frozen
)]
pub struct PyEllipticGatedCausalLearningBatch {
    inner: EllipticGatedCausalLearningBatch,
}

#[pymethods]
impl PyEllipticGatedCausalLearningBatch {
    #[getter]
    fn features(&self) -> Vec<f32> {
        self.inner.features().to_vec()
    }

    #[getter]
    fn mix(&self) -> f32 {
        self.inner.mix()
    }

    /// Returns (orientation gradient, sum-reduced shared raw-mix gradient).
    fn vjp(&self, py: Python<'_>, upstream: Vec<f32>) -> PyResult<(Vec<f32>, f32)> {
        py.detach(|| self.inner.vjp(&upstream))
            .map(|g| (g.orientations, g.raw_mix))
            .map_err(value_error)
    }
}

#[pymethods]
impl PyEllipticCausalLearningBatch {
    #[getter]
    fn features(&self) -> Vec<f32> {
        self.inner.features().to_vec()
    }

    fn vjp(&self, py: Python<'_>, upstream: Vec<f32>) -> PyResult<Vec<f32>> {
        py.detach(|| self.inner.vjp(&upstream)).map_err(value_error)
    }
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymethods]
impl PyEllipticLearningBatch {
    #[getter]
    fn features(&self) -> Vec<f32> {
        self.inner.features().to_vec()
    }

    fn telemetry(&self) -> Vec<PyEllipticTelemetry> {
        self.inner
            .telemetry()
            .iter()
            .cloned()
            .map(PyEllipticTelemetry::from)
            .collect()
    }

    fn vjp(&self, py: Python<'_>, upstream: Vec<f32>) -> PyResult<Vec<f32>> {
        py.detach(|| self.inner.vjp(&upstream)).map_err(value_error)
    }

    fn jvp(&self, py: Python<'_>, tangent: Vec<f32>) -> PyResult<Vec<f32>> {
        py.detach(|| self.inner.jvp(&tangent)).map_err(value_error)
    }
}

#[pyclass(name = "EllipticTelemetry", module = "spiraltorch")]
pub struct PyEllipticTelemetry {
    inner: EllipticTelemetry,
}

impl From<EllipticTelemetry> for PyEllipticTelemetry {
    fn from(inner: EllipticTelemetry) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEllipticWarp {
    #[new]
    #[pyo3(signature = (curvature_radius, sheet_count=None, spin_harmonics=None))]
    fn new(
        curvature_radius: f32,
        sheet_count: Option<usize>,
        spin_harmonics: Option<usize>,
    ) -> PyResult<Self> {
        let warp = EllipticWarp::for_learning(
            curvature_radius,
            sheet_count.unwrap_or(2),
            spin_harmonics.unwrap_or(1),
        )
        .map_err(value_error)?;
        Ok(Self { warp })
    }

    #[getter]
    fn curvature_radius(&self) -> f32 {
        self.warp.curvature_radius()
    }

    #[getter]
    fn sheet_count(&self) -> usize {
        self.warp.sheet_count()
    }

    #[getter]
    fn spin_harmonics(&self) -> usize {
        self.warp.spin_harmonics()
    }

    #[pyo3(signature = (sheet_count=None, spin_harmonics=None))]
    fn configure(
        &mut self,
        sheet_count: Option<usize>,
        spin_harmonics: Option<usize>,
    ) -> PyResult<()> {
        self.warp = EllipticWarp::for_learning(
            self.warp.curvature_radius(),
            sheet_count.unwrap_or(self.warp.sheet_count()),
            spin_harmonics.unwrap_or(self.warp.spin_harmonics()),
        )
        .map_err(value_error)?;
        Ok(())
    }

    #[pyo3(signature = (orientations, *, max_rows=65_536))]
    fn map_orientations_batch(
        &self,
        py: Python<'_>,
        orientations: Vec<f32>,
        max_rows: usize,
    ) -> PyResult<PyEllipticLearningBatch> {
        py.detach(|| self.warp.differentiate_batch(&orientations, max_rows))
            .map(|inner| PyEllipticLearningBatch { inner })
            .map_err(value_error)
    }

    #[pyo3(signature = (orientations, *, raw_mix, max_rows=65_536))]
    fn map_anchored_batch(
        &self,
        py: Python<'_>,
        orientations: Vec<f32>,
        raw_mix: f32,
        max_rows: usize,
    ) -> PyResult<PyEllipticAnchoredLearningBatch> {
        py.detach(|| {
            self.warp
                .differentiate_anchored_batch(&orientations, raw_mix, max_rows)
        })
        .map(|inner| PyEllipticAnchoredLearningBatch { inner })
        .map_err(value_error)
    }

    #[pyo3(signature = (orientations, *, batch_size, sequence_length, max_rows=65_536, max_pairs=1_048_576))]
    fn map_causal_batch(
        &self,
        py: Python<'_>,
        orientations: Vec<f32>,
        batch_size: usize,
        sequence_length: usize,
        max_rows: usize,
        max_pairs: usize,
    ) -> PyResult<PyEllipticCausalLearningBatch> {
        py.detach(|| {
            self.warp.differentiate_causal_batch(
                &orientations,
                batch_size,
                sequence_length,
                max_rows,
                max_pairs,
            )
        })
        .map(|inner| PyEllipticCausalLearningBatch { inner })
        .map_err(value_error)
    }

    #[allow(clippy::too_many_arguments)] // Keep the existing causal batch keyword API.
    #[pyo3(signature = (orientations, *, batch_size, sequence_length, raw_mix, max_rows=65_536, max_pairs=1_048_576))]
    fn map_gated_causal_batch(
        &self,
        py: Python<'_>,
        orientations: Vec<f32>,
        batch_size: usize,
        sequence_length: usize,
        raw_mix: f32,
        max_rows: usize,
        max_pairs: usize,
    ) -> PyResult<PyEllipticGatedCausalLearningBatch> {
        py.detach(|| {
            self.warp.differentiate_gated_causal_batch(
                &orientations,
                [batch_size, sequence_length],
                raw_mix,
                max_rows,
                max_pairs,
            )
        })
        .map(|inner| PyEllipticGatedCausalLearningBatch { inner })
        .map_err(value_error)
    }

    fn map_orientation(&self, orientation: Vec<f32>) -> PyResult<Option<PyEllipticTelemetry>> {
        Ok(self
            .warp
            .map_orientation(&orientation)
            .map(PyEllipticTelemetry::from))
    }

    fn map_orientation_differential(
        &self,
        orientation: Vec<f32>,
    ) -> PyResult<Option<EllipticDifferential>> {
        let Some((telemetry, diff)) = self.warp.map_orientation_with_differential(&orientation)
        else {
            return Ok(None);
        };
        let features = diff.feature_slice().to_vec();
        let jacobian = diff
            .jacobian()
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();
        Ok(Some((
            PyEllipticTelemetry::from(telemetry),
            features,
            jacobian,
        )))
    }
}

#[pymethods]
impl PyEllipticTelemetry {
    #[getter]
    fn curvature_radius(&self) -> f32 {
        self.inner.curvature_radius
    }

    #[getter]
    fn geodesic_radius(&self) -> f32 {
        self.inner.geodesic_radius
    }

    #[getter]
    fn normalized_radius(&self) -> f32 {
        self.inner.normalized_radius()
    }

    #[getter]
    fn spin_alignment(&self) -> f32 {
        self.inner.spin_alignment
    }

    #[getter]
    fn sheet_index(&self) -> usize {
        self.inner.sheet_index
    }

    #[getter]
    fn sheet_position(&self) -> f32 {
        self.inner.sheet_position
    }

    #[getter]
    fn normal_bias(&self) -> f32 {
        self.inner.normal_bias
    }

    #[getter]
    fn sheet_count(&self) -> usize {
        self.inner.sheet_count
    }

    #[getter]
    fn topological_sector(&self) -> u32 {
        self.inner.topological_sector
    }

    #[getter]
    fn homology_index(&self) -> u32 {
        self.inner.homology_index
    }

    #[getter]
    fn rotor_field(&self) -> [f32; 3] {
        self.inner.rotor_field
    }

    #[getter]
    fn flow_vector(&self) -> [f32; 3] {
        self.inner.flow_vector
    }

    #[getter]
    fn curvature_tensor(&self) -> [[f32; 3]; 3] {
        self.inner.curvature_tensor
    }

    #[getter]
    fn resonance_heat(&self) -> f32 {
        self.inner.resonance_heat
    }

    #[getter]
    fn noise_density(&self) -> f32 {
        self.inner.noise_density
    }

    #[getter]
    fn lie_log(&self) -> [f32; 3] {
        self.inner.lie_log
    }

    #[getter]
    fn rotor_transport(&self) -> [f32; 3] {
        self.inner.rotor_transport
    }

    fn lie_quaternion(&self) -> [f32; 4] {
        self.inner.lie_frame.quaternion()
    }

    fn lie_rotation(&self) -> [f32; 9] {
        self.inner.lie_frame.rotation_matrix()
    }

    fn event_tags(&self) -> Vec<String> {
        self.inner.event_tags().to_vec()
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("curvature_radius", self.inner.curvature_radius)?;
        dict.set_item("geodesic_radius", self.inner.geodesic_radius)?;
        dict.set_item("normalized_radius", self.inner.normalized_radius())?;
        dict.set_item("spin_alignment", self.inner.spin_alignment)?;
        dict.set_item("sheet_index", self.inner.sheet_index)?;
        dict.set_item("sheet_position", self.inner.sheet_position)?;
        dict.set_item("normal_bias", self.inner.normal_bias)?;
        dict.set_item("sheet_count", self.inner.sheet_count)?;
        dict.set_item("topological_sector", self.inner.topological_sector)?;
        dict.set_item("homology_index", self.inner.homology_index)?;
        dict.set_item("rotor_field", self.inner.rotor_field)?;
        dict.set_item("flow_vector", self.inner.flow_vector)?;
        dict.set_item("curvature_tensor", self.inner.curvature_tensor)?;
        dict.set_item("resonance_heat", self.inner.resonance_heat)?;
        dict.set_item("noise_density", self.inner.noise_density)?;
        dict.set_item("lie_log", self.inner.lie_log)?;
        dict.set_item("rotor_transport", self.inner.rotor_transport)?;
        dict.set_item("lie_quaternion", self.inner.lie_frame.quaternion())?;
        dict.set_item("lie_rotation", self.inner.lie_frame.rotation_matrix())?;
        dict.set_item("event_tags", self.inner.event_tags().to_vec())?;
        dict.into_py_any(py)
    }
}

pub fn register(py: Python<'_>, module: &Bound<PyModule>) -> PyResult<()> {
    module.add_class::<PyEllipticWarp>()?;
    module.add_class::<PyEllipticTelemetry>()?;
    module.add_class::<PyEllipticLearningBatch>()?;
    module.add_class::<PyEllipticAnchoredLearningBatch>()?;
    module.add_class::<PyEllipticCausalLearningBatch>()?;
    module.add_class::<PyEllipticGatedCausalLearningBatch>()?;
    module.add("__doc__", "Elliptic microlocal warp helpers")?;
    let _ = py;
    Ok(())
}
