use pyo3::prelude::*;
use pyo3::types::PyDict;
use st_core::theory::microlocal::{EllipticTelemetry, EllipticWarp};

#[pyclass(name = "EllipticWarp", module = "spiraltorch")]
pub struct PyEllipticWarp {
    warp: EllipticWarp,
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
    ) -> Self {
        let mut warp = EllipticWarp::new(curvature_radius);
        if let Some(sheets) = sheet_count {
            warp = warp.with_sheet_count(sheets);
        }
        if let Some(harmonics) = spin_harmonics {
            warp = warp.with_spin_harmonics(harmonics);
        }
        Self { warp }
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

    fn configure(&mut self, sheet_count: Option<usize>, spin_harmonics: Option<usize>) {
        if let Some(sheets) = sheet_count {
            self.warp = self.warp.clone().with_sheet_count(sheets);
        }
        if let Some(harmonics) = spin_harmonics {
            self.warp = self.warp.clone().with_spin_harmonics(harmonics);
        }
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
    ) -> PyResult<Option<(PyEllipticTelemetry, Vec<f32>, Vec<Vec<f32>>)>> {
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

    fn lie_quaternion(&self) -> [f32; 4] {
        self.inner.lie_frame.quaternion()
    }

    fn lie_rotation(&self) -> [f32; 9] {
        self.inner.lie_frame.rotation_matrix()
    }

    fn event_tags(&self) -> Vec<String> {
        self.inner.event_tags().to_vec()
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
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
        dict.set_item("lie_quaternion", self.inner.lie_frame.quaternion())?;
        dict.set_item("lie_rotation", self.inner.lie_frame.rotation_matrix())?;
        dict.set_item("event_tags", self.inner.event_tags().to_vec())?;
        Ok(dict.into_py(py))
    }
}

pub fn register(py: Python<'_>, module: &Bound<PyModule>) -> PyResult<()> {
    module.add_class::<PyEllipticWarp>()?;
    module.add_class::<PyEllipticTelemetry>()?;
    module.add("__doc__", "Elliptic microlocal warp helpers")?;
    let _ = py;
    Ok(())
}
