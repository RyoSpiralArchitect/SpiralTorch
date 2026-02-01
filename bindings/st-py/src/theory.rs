use crate::tensor::{tensor_err_to_py, tensor_to_torch, to_dlpack_impl, PyTensor};
use nalgebra::{DMatrix, Matrix4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use st_core::theory::renormalization_flow as rg_mellin;
use st_core::theory::rg_flow as rg_zspace;
use st_core::theory::general_relativity::{
    BoundaryCondition, BoundaryConditionKind, GeneralRelativityModel, InternalMetric,
    InternalPatch, InternalSpace, LorentzianMetric, MetricDerivatives, MetricError,
    MetricSecondDerivatives, MixedBlock, PhysicalConstants, ProductGeometry, ProductMetric,
    SymmetryAnsatz, Topology, WarpFactor, ZManifold, ZRelativityModel, ZRelativityTensorBundle,
};
use st_frac::mellin_types::{ComplexScalar, MellinError};
use st_tensor::Tensor;

fn metric_error(err: MetricError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn mellin_error(err: MellinError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn rg_mellin_error(err: rg_mellin::RGFlowError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn rg_zspace_error(err: rg_zspace::RgFlowError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn matrix4_from_py(data: Vec<Vec<f64>>) -> PyResult<Matrix4<f64>> {
    if data.len() != 4 {
        return Err(PyValueError::new_err(
            "expected 4 rows for a Lorentzian metric",
        ));
    }
    for row in &data {
        if row.len() != 4 {
            return Err(PyValueError::new_err(
                "expected 4 columns for a Lorentzian metric",
            ));
        }
    }
    let flat: Vec<f64> = data.into_iter().flatten().collect();
    Ok(Matrix4::from_row_slice(&flat))
}

fn matrix4_to_py(matrix: &Matrix4<f64>) -> Vec<Vec<f64>> {
    (0..4)
        .map(|i| (0..4).map(|j| matrix[(i, j)]).collect())
        .collect()
}

fn dmatrix_from_py(data: Vec<Vec<f64>>) -> PyResult<DMatrix<f64>> {
    let rows = data.len();
    if rows == 0 {
        return Err(PyValueError::new_err("matrix must have at least one row"));
    }
    let cols = data[0].len();
    if cols == 0 {
        return Err(PyValueError::new_err(
            "matrix must have at least one column",
        ));
    }
    for row in &data {
        if row.len() != cols {
            return Err(PyValueError::new_err(
                "matrix rows must all share the same length",
            ));
        }
    }
    let flat: Vec<f64> = data.into_iter().flatten().collect();
    Ok(DMatrix::from_row_slice(rows, cols, &flat))
}

fn dmatrix_to_py(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|i| (0..matrix.ncols()).map(|j| matrix[(i, j)]).collect())
        .collect()
}

fn tensor_to_py(py: Python<'_>, tensor: Tensor) -> PyResult<PyObject> {
    Ok(Py::new(py, PyTensor::from_tensor(tensor))?.into_py(py))
}

fn parse_metric_derivatives(data: Option<Vec<Vec<Vec<f64>>>>) -> PyResult<MetricDerivatives> {
    let Some(values) = data else {
        return Ok(MetricDerivatives::zero());
    };
    if values.len() != 4 {
        return Err(PyValueError::new_err(
            "first derivatives must have shape 4×4×4",
        ));
    }
    for plane in &values {
        if plane.len() != 4 {
            return Err(PyValueError::new_err(
                "first derivatives must have shape 4×4×4",
            ));
        }
        for row in plane {
            if row.len() != 4 {
                return Err(PyValueError::new_err(
                    "first derivatives must have shape 4×4×4",
                ));
            }
        }
    }
    Ok(MetricDerivatives::from_fn(move |rho, mu, nu| {
        values[rho][mu][nu]
    }))
}

fn parse_metric_second_derivatives(
    data: Option<Vec<Vec<Vec<Vec<f64>>>>>,
) -> PyResult<MetricSecondDerivatives> {
    let Some(values) = data else {
        return Ok(MetricSecondDerivatives::zero());
    };
    if values.len() != 4 {
        return Err(PyValueError::new_err(
            "second derivatives must have shape 4×4×4×4",
        ));
    }
    for cube in &values {
        if cube.len() != 4 {
            return Err(PyValueError::new_err(
                "second derivatives must have shape 4×4×4×4",
            ));
        }
        for plane in cube {
            if plane.len() != 4 {
                return Err(PyValueError::new_err(
                    "second derivatives must have shape 4×4×4×4",
                ));
            }
            for row in plane {
                if row.len() != 4 {
                    return Err(PyValueError::new_err(
                        "second derivatives must have shape 4×4×4×4",
                    ));
                }
            }
        }
    }
    Ok(MetricSecondDerivatives::from_fn(
        move |lambda, rho, mu, nu| values[lambda][rho][mu][nu],
    ))
}

fn parse_symmetry(symmetry: Option<String>) -> SymmetryAnsatz {
    match symmetry.as_deref() {
        Some("static_spherical") => SymmetryAnsatz::StaticSpherical,
        Some("homogeneous_isotropic") => SymmetryAnsatz::HomogeneousIsotropic,
        Some(text) if !text.is_empty() => SymmetryAnsatz::Custom(text.to_string()),
        _ => SymmetryAnsatz::Custom("unspecified Z-space symmetry".to_string()),
    }
}

fn parse_topology(topology: Option<String>) -> Topology {
    match topology.as_deref() {
        Some("r4") => Topology::R4,
        Some("r3xs1") => Topology::R3xS1,
        Some(text) if !text.is_empty() => Topology::Custom(text.to_string()),
        _ => Topology::R4,
    }
}

fn parse_boundary_conditions(boundaries: Option<Vec<String>>) -> Vec<BoundaryCondition> {
    boundaries
        .unwrap_or_default()
        .into_iter()
        .map(|value| {
            let kind = match value.as_str() {
                "asymptotically_flat" => BoundaryConditionKind::AsymptoticallyFlat,
                "regularity" => BoundaryConditionKind::Regularity,
                "periodic" => BoundaryConditionKind::Periodic,
                other => BoundaryConditionKind::Custom(other.to_string()),
            };
            BoundaryCondition {
                kind,
                location: None,
                notes: None,
            }
        })
        .collect()
}

#[pyclass(module = "spiraltorch.theory", name = "ZRelativityModel", unsendable)]
pub struct PyZRelativityModel {
    pub(crate) inner: ZRelativityModel,
}

#[pymethods]
impl PyZRelativityModel {
    pub fn as_tensor(&self) -> PyResult<PyTensor> {
        let tensor = self.inner.as_tensor().map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    pub fn to_dlpack(&self, py: Python<'_>) -> PyResult<PyObject> {
        let tensor = self.inner.as_tensor().map_err(tensor_err_to_py)?;
        to_dlpack_impl(py, &tensor)
    }

    pub fn effective_metric(&self) -> PyResult<PyTensor> {
        let tensor = self
            .inner
            .effective_metric_tensor()
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    pub fn internal_metric_components(&self) -> Vec<Vec<f64>> {
        dmatrix_to_py(self.inner.geometry.metric().internal().components())
    }

    pub fn gauge_tensor(&self) -> PyResult<PyTensor> {
        let tensor = self.inner.gauge_tensor().map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    pub fn scalar_moduli(&self) -> PyResult<PyTensor> {
        let tensor = self
            .inner
            .scalar_moduli_tensor()
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    pub fn field_equations(&self) -> PyResult<PyTensor> {
        let tensor = self
            .inner
            .field_equation_tensor()
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    pub fn tensor_bundle(&self, py: Python<'_>) -> PyResult<PyObject> {
        let bundle = self.inner.tensor_bundle().map_err(tensor_err_to_py)?;
        let ZRelativityTensorBundle {
            block_metric,
            effective_metric,
            gauge_field,
            scalar_moduli,
            field_equation,
            warp,
            internal_volume_density,
            field_prefactor,
        } = bundle;
        let dict = PyDict::new_bound(py);
        dict.set_item("block_metric", tensor_to_py(py, block_metric)?)?;
        dict.set_item("effective_metric", tensor_to_py(py, effective_metric)?)?;
        dict.set_item("gauge_field", tensor_to_py(py, gauge_field)?)?;
        dict.set_item("scalar_moduli", tensor_to_py(py, scalar_moduli)?)?;
        dict.set_item("field_equation", tensor_to_py(py, field_equation)?)?;
        if let Some(warp) = warp {
            dict.set_item("warp", tensor_to_py(py, warp)?)?;
        } else {
            dict.set_item("warp", py.None())?;
        }
        dict.set_item("internal_volume_density", internal_volume_density)?;
        dict.set_item("field_prefactor", field_prefactor)?;
        Ok(dict.into())
    }

    pub fn torch_bundle(&self, py: Python<'_>) -> PyResult<PyObject> {
        let bundle = self.inner.tensor_bundle().map_err(tensor_err_to_py)?;
        let ZRelativityTensorBundle {
            block_metric,
            effective_metric,
            gauge_field,
            scalar_moduli,
            field_equation,
            warp,
            internal_volume_density,
            field_prefactor,
        } = bundle;
        let dict = PyDict::new_bound(py);
        dict.set_item("block_metric", tensor_to_torch(py, &block_metric)?)?;
        dict.set_item("effective_metric", tensor_to_torch(py, &effective_metric)?)?;
        dict.set_item("gauge_field", tensor_to_torch(py, &gauge_field)?)?;
        dict.set_item("scalar_moduli", tensor_to_torch(py, &scalar_moduli)?)?;
        dict.set_item("field_equation", tensor_to_torch(py, &field_equation)?)?;
        if let Some(ref warp) = warp {
            dict.set_item("warp", tensor_to_torch(py, warp)?)?;
        } else {
            dict.set_item("warp", py.None())?;
        }
        dict.set_item("internal_volume_density", internal_volume_density)?;
        dict.set_item("field_prefactor", field_prefactor)?;
        Ok(dict.into())
    }

    pub fn reduction_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        let reduction = &self.inner.reduction;
        dict.set_item(
            "effective_metric",
            tensor_to_py(
                py,
                self.inner
                    .effective_metric_tensor()
                    .map_err(tensor_err_to_py)?,
            )?,
        )?;
        dict.set_item(
            "gauge_field",
            tensor_to_py(py, self.inner.gauge_tensor().map_err(tensor_err_to_py)?)?,
        )?;
        dict.set_item(
            "scalar_moduli",
            tensor_to_py(
                py,
                self.inner
                    .scalar_moduli_tensor()
                    .map_err(tensor_err_to_py)?,
            )?,
        )?;
        dict.set_item(
            "effective_newton_constant",
            reduction.effective_newton_constant(),
        )?;
        Ok(dict.into())
    }

    pub fn curvature_diagnostics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        let diagnostics = &self.inner.base_model.diagnostics;
        dict.set_item("scalar_curvature", diagnostics.scalar_curvature)?;
        dict.set_item("ricci_square", diagnostics.ricci_square)?;
        dict.set_item("kretschmann", diagnostics.kretschmann)?;
        dict.set_item("weyl_square", diagnostics.weyl_square)?;
        dict.set_item("weyl_dual_contraction", diagnostics.weyl_dual_contraction)?;
        dict.set_item("weyl_self_dual_squared", diagnostics.weyl_self_dual_squared)?;
        dict.set_item(
            "weyl_anti_self_dual_squared",
            diagnostics.weyl_anti_self_dual_squared,
        )?;
        dict.set_item(
            "weyl_self_dual_matrix",
            diagnostics
                .weyl_self_dual_matrix
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "weyl_anti_self_dual_matrix",
            diagnostics
                .weyl_anti_self_dual_matrix
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "weyl_self_dual_invariant_i",
            diagnostics.weyl_self_dual_invariant_i,
        )?;
        dict.set_item(
            "weyl_self_dual_invariant_j",
            diagnostics.weyl_self_dual_invariant_j,
        )?;
        dict.set_item(
            "weyl_self_dual_discriminant",
            diagnostics.weyl_self_dual_discriminant,
        )?;
        dict.set_item(
            "weyl_self_dual_eigenvalues",
            diagnostics.weyl_self_dual_eigenvalues.to_vec(),
        )?;
        Ok(dict.into())
    }

    pub fn learnable_flags(&self) -> (bool, bool, bool) {
        let flags = self.inner.learnable_flags();
        (flags.warp, flags.mixed, flags.internal)
    }

    pub fn warp_scale(&self) -> Option<f64> {
        self.inner.geometry.metric().warp().map(|warp| warp.scale())
    }

    pub fn internal_volume_density(&self) -> f64 {
        self.inner.geometry.internal_volume_density()
    }

    pub fn field_prefactor(&self) -> f64 {
        self.inner.field_equations.prefactor()
    }

    pub fn total_dimension(&self) -> usize {
        self.inner.geometry.total_dimension()
    }
}

#[pyfunction]
#[pyo3(signature = (components, scale))]
pub fn lorentzian_metric_scaled(
    py: Python<'_>,
    components: Vec<Vec<f64>>,
    scale: f64,
) -> PyResult<PyObject> {
    let matrix = matrix4_from_py(components)?;
    let metric = LorentzianMetric::try_scaled(matrix, scale).map_err(metric_error)?;
    let dict = PyDict::new_bound(py);
    dict.set_item("components", matrix4_to_py(metric.components()))?;
    dict.set_item("inverse", matrix4_to_py(metric.inverse()))?;
    dict.set_item("determinant", metric.determinant())?;
    dict.set_item("volume_element", metric.volume_element())?;
    dict.set_item(
        "signature",
        metric.signature().to_vec(),
    )?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (
    base_metric,
    internal_metric,
    *,
    mixed=None,
    warp=None,
    first_derivatives=None,
    second_derivatives=None,
    gravitational_constant,
    speed_of_light,
    internal_volume,
    cosmological_constant=0.0,
    symmetry=None,
    topology=None,
    boundary_conditions=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn assemble_zrelativity_model(
    py: Python<'_>,
    base_metric: Vec<Vec<f64>>,
    internal_metric: Vec<Vec<f64>>,
    mixed: Option<Vec<Vec<f64>>>,
    warp: Option<f64>,
    first_derivatives: Option<Vec<Vec<Vec<f64>>>>,
    second_derivatives: Option<Vec<Vec<Vec<Vec<f64>>>>>,
    gravitational_constant: f64,
    speed_of_light: f64,
    internal_volume: f64,
    cosmological_constant: f64,
    symmetry: Option<String>,
    topology: Option<String>,
    boundary_conditions: Option<Vec<String>>,
) -> PyResult<PyZRelativityModel> {
    let base = matrix4_from_py(base_metric)?;
    let base_metric = LorentzianMetric::try_new(base).map_err(metric_error)?;
    let internal_matrix = dmatrix_from_py(internal_metric)?;
    let internal_metric = InternalMetric::try_new(internal_matrix).map_err(metric_error)?;
    let internal_dim = internal_metric.dimension();

    let mixed_block = if let Some(values) = mixed {
        let matrix = dmatrix_from_py(values)?;
        Some(MixedBlock::new(matrix, 4, internal_dim).map_err(metric_error)?)
    } else {
        None
    };

    let warp_factor = if let Some(scale) = warp {
        Some(WarpFactor::from_multiplier(scale).map_err(metric_error)?)
    } else {
        None
    };

    let product_metric = ProductMetric::try_new(
        base_metric.clone(),
        internal_metric.clone(),
        mixed_block,
        warp_factor,
    )
    .map_err(metric_error)?;

    let mut coords = Vec::with_capacity(internal_dim);
    for idx in 0..internal_dim {
        coords.push(format!("z{idx}"));
    }
    let internal_space = InternalSpace::new("Z-internal", InternalPatch::new("chart", coords));
    let spacetime = ZManifold::canonical();
    let geometry = ProductGeometry::new(spacetime, internal_space, product_metric.clone());

    let constants = PhysicalConstants::new(gravitational_constant, speed_of_light);
    let first = parse_metric_derivatives(first_derivatives)?;
    let second = parse_metric_second_derivatives(second_derivatives)?;
    let symmetry = parse_symmetry(symmetry);
    let topology = parse_topology(topology);
    let boundaries = parse_boundary_conditions(boundary_conditions);

    let base_model = GeneralRelativityModel::new(
        geometry.spacetime().clone(),
        base_metric.clone(),
        first,
        second,
        symmetry,
        topology,
        boundaries,
    );
    let zr_model = ZRelativityModel::assemble(
        geometry.clone(),
        base_model.clone(),
        constants,
        internal_volume,
        cosmological_constant,
    )
    .map_err(metric_error)?;
    let _ = py;
    Ok(PyZRelativityModel { inner: zr_model })
}

#[pyclass(module = "spiraltorch.theory", name = "ResonanceProfile")]
#[derive(Clone)]
pub(crate) struct PyResonanceProfile {
    inner: rg_mellin::ResonanceProfile,
}

#[pymethods]
impl PyResonanceProfile {
    #[new]
    pub fn new(
        grid: &crate::frac::PyMellinLogGrid,
        pole: ComplexScalar,
        residue: ComplexScalar,
    ) -> Self {
        let inner = rg_mellin::ResonanceProfile::new(grid.inner.clone(), pole, residue);
        Self { inner }
    }

    pub fn evaluate(&self, log_scale: f32) -> PyResult<ComplexScalar> {
        self.inner.evaluate(log_scale).map_err(mellin_error)
    }
}

#[pyclass(module = "spiraltorch.theory", name = "RGOperator")]
#[derive(Clone)]
pub(crate) struct PyRGOperator {
    inner: rg_mellin::RGOperator,
}

#[pymethods]
impl PyRGOperator {
    #[new]
    #[pyo3(signature = (name, scaling_dimension, initial_coupling, nonlinear_feedback=0.0, resonance=None))]
    pub fn new(
        name: String,
        scaling_dimension: f32,
        initial_coupling: f32,
        nonlinear_feedback: f32,
        resonance: Option<&PyResonanceProfile>,
    ) -> Self {
        let mut inner = rg_mellin::RGOperator::new(name, scaling_dimension, initial_coupling)
            .with_nonlinear_feedback(nonlinear_feedback);
        if let Some(res) = resonance {
            inner = inner.with_resonance(res.inner.clone());
        }
        Self { inner }
    }

    #[getter]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub fn scaling_dimension(&self) -> f32 {
        self.inner.scaling_dimension
    }

    #[getter]
    pub fn initial_coupling(&self) -> f32 {
        self.inner.initial_coupling
    }

    #[getter]
    pub fn nonlinear_feedback(&self) -> f32 {
        self.inner.nonlinear_feedback
    }

    pub fn with_resonance(&self, resonance: &PyResonanceProfile) -> Self {
        Self {
            inner: self.inner.clone().with_resonance(resonance.inner.clone()),
        }
    }
}

#[pyclass(module = "spiraltorch.theory", name = "RGFlowTrajectory")]
#[derive(Clone)]
pub(crate) struct PyRGFlowTrajectory {
    inner: rg_mellin::RGFlowTrajectory,
}

#[pymethods]
impl PyRGFlowTrajectory {
    #[getter]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub fn values(&self) -> Vec<f32> {
        self.inner.values.clone()
    }

    pub fn at(&self, index: usize) -> Option<f32> {
        self.inner.at(index)
    }
}

#[pyclass(module = "spiraltorch.theory", name = "RGFlowSolution")]
#[derive(Clone)]
pub(crate) struct PyRGFlowSolution {
    inner: rg_mellin::RGFlowSolution,
}

#[pymethods]
impl PyRGFlowSolution {
    #[getter]
    pub fn lattice(&self) -> Vec<f32> {
        self.inner.lattice().to_vec()
    }

    #[getter]
    pub fn trajectories(&self) -> Vec<PyRGFlowTrajectory> {
        self.inner
            .iter()
            .cloned()
            .map(|inner| PyRGFlowTrajectory { inner })
            .collect()
    }

    pub fn trajectory(&self, name: &str) -> Option<PyRGFlowTrajectory> {
        self.inner
            .trajectory(name)
            .cloned()
            .map(|inner| PyRGFlowTrajectory { inner })
    }
}

#[pyclass(module = "spiraltorch.theory", name = "RGFlowModel")]
#[derive(Clone)]
pub(crate) struct PyRGFlowModel {
    inner: rg_mellin::RGFlowModel,
}

#[pymethods]
impl PyRGFlowModel {
    #[new]
    pub fn new(log_start: f32, log_step: f32, depth: usize) -> PyResult<Self> {
        let inner = rg_mellin::RGFlowModel::new(log_start, log_step, depth).map_err(rg_mellin_error)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn from_lattice(lattice: Vec<f32>) -> PyResult<Self> {
        let inner = rg_mellin::RGFlowModel::from_lattice(lattice).map_err(rg_mellin_error)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn narrative_depth(depth: usize, start: f32) -> PyResult<Self> {
        let inner = rg_mellin::RGFlowModel::narrative_depth(depth, start).map_err(rg_mellin_error)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn lattice(&self) -> Vec<f32> {
        self.inner.lattice().to_vec()
    }

    #[getter]
    pub fn damping(&self) -> f32 {
        self.inner.damping
    }

    #[setter]
    pub fn set_damping(&mut self, value: f32) {
        self.inner.damping = value.max(0.0);
    }

    pub fn register_operator(&mut self, operator: &PyRGOperator) {
        self.inner.register_operator(operator.inner.clone());
    }

    pub fn operators(&self) -> Vec<PyRGOperator> {
        self.inner
            .operators()
            .iter()
            .cloned()
            .map(|inner| PyRGOperator { inner })
            .collect()
    }

    pub fn propagate(&self) -> PyResult<PyRGFlowSolution> {
        let inner = self.inner.propagate().map_err(rg_mellin_error)?;
        Ok(PyRGFlowSolution { inner })
    }

    pub fn sample_beta(&self, coupling: f32, operator_index: usize, log_scale: f32) -> PyResult<f32> {
        self.inner
            .sample_beta(coupling, operator_index, log_scale)
            .map_err(rg_mellin_error)
    }
}

#[pyfunction]
#[pyo3(signature = (log_start, log_step, len, center, bandwidth, amplitude))]
fn gaussian_resonance(
    log_start: f32,
    log_step: f32,
    len: usize,
    center: f32,
    bandwidth: f32,
    amplitude: f32,
) -> PyResult<PyResonanceProfile> {
    rg_mellin::RGFlowModel::gaussian_resonance(log_start, log_step, len, center, bandwidth, amplitude)
        .map(|inner| PyResonanceProfile { inner })
        .map_err(mellin_error)
}

#[pyfunction]
#[pyo3(signature = (grid, pole, phase, strength))]
fn breathing_resonance(
    grid: &crate::frac::PyMellinLogGrid,
    pole: ComplexScalar,
    phase: f32,
    strength: f32,
) -> PyResult<PyResonanceProfile> {
    let shared = std::sync::Arc::new(grid.inner.clone());
    let inner = rg_mellin::RGFlowModel::breathing_resonance(shared, pole, phase, strength);
    Ok(PyResonanceProfile { inner })
}

#[pyclass(module = "spiraltorch.theory", name = "RgFlowNode")]
#[derive(Clone)]
pub(crate) struct PyRgFlowNode {
    inner: rg_zspace::RgFlowNode,
}

#[pymethods]
impl PyRgFlowNode {
    #[getter]
    pub fn log_scale(&self) -> f32 {
        self.inner.log_scale()
    }

    #[getter]
    pub fn scale(&self) -> f32 {
        self.inner.scale()
    }

    #[getter]
    pub fn narrative_depth(&self) -> f32 {
        self.inner.narrative_depth()
    }

    #[getter]
    pub fn couplings(&self) -> Vec<f32> {
        self.inner.couplings().to_vec()
    }

    #[getter]
    pub fn beta(&self) -> Vec<f32> {
        self.inner.beta().to_vec()
    }

    pub fn beta_norm(&self) -> f32 {
        self.inner.beta_norm()
    }
}

#[pyclass(module = "spiraltorch.theory", name = "ScaleFixedPoint")]
#[derive(Clone)]
pub(crate) struct PyScaleFixedPoint {
    inner: rg_zspace::ScaleFixedPoint,
}

#[pymethods]
impl PyScaleFixedPoint {
    #[getter]
    pub fn log_scale(&self) -> f32 {
        self.inner.log_scale()
    }

    #[getter]
    pub fn scale(&self) -> f32 {
        self.inner.scale()
    }

    #[getter]
    pub fn narrative_depth(&self) -> f32 {
        self.inner.narrative_depth()
    }

    #[getter]
    pub fn couplings(&self) -> Vec<f32> {
        self.inner.couplings().to_vec()
    }
}

#[pyclass(module = "spiraltorch.theory", name = "RgFlowLattice")]
#[derive(Clone)]
pub(crate) struct PyRgFlowLattice {
    inner: rg_zspace::RgFlowLattice,
}

#[pymethods]
impl PyRgFlowLattice {
    #[staticmethod]
    #[pyo3(signature = (log_start, log_step, steps, root_couplings, beta))]
    pub fn new_with_beta(
        log_start: f32,
        log_step: f32,
        steps: usize,
        root_couplings: Vec<f32>,
        beta: Py<PyAny>,
    ) -> PyResult<Self> {
        let callback = beta;
        let err_slot: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
        let mut beta_fn = |log_scale: f32, couplings: &[f32]| -> Vec<f32> {
            if err_slot.borrow().is_some() {
                return vec![0.0; couplings.len()];
            }
            Python::with_gil(|py| {
                let couplings_vec = couplings.to_vec();
                match callback.call1(py, (log_scale, couplings_vec)) {
                    Ok(value) => match value.extract::<Vec<f32>>(py) {
                        Ok(values) => values,
                        Err(err) => {
                            err_slot.replace(Some(err));
                            vec![0.0; couplings.len()]
                        }
                    },
                    Err(err) => {
                        err_slot.replace(Some(err));
                        vec![0.0; couplings.len()]
                    }
                }
            })
        };

        let lattice = rg_zspace::RgFlowLattice::new_with_beta(
            log_start,
            log_step,
            steps,
            root_couplings,
            &mut beta_fn,
        )
        .map_err(rg_zspace_error);

        if let Some(err) = err_slot.into_inner() {
            return Err(err);
        }
        Ok(Self { inner: lattice? })
    }

    #[staticmethod]
    #[pyo3(signature = (grid, root_couplings, beta))]
    pub fn from_mellin_grid(
        grid: &crate::frac::PyMellinLogGrid,
        root_couplings: Vec<f32>,
        beta: Py<PyAny>,
    ) -> PyResult<Self> {
        let callback = beta;
        let err_slot: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
        let mut beta_fn = |log_scale: f32, couplings: &[f32]| -> Vec<f32> {
            if err_slot.borrow().is_some() {
                return vec![0.0; couplings.len()];
            }
            Python::with_gil(|py| {
                let couplings_vec = couplings.to_vec();
                match callback.call1(py, (log_scale, couplings_vec)) {
                    Ok(value) => match value.extract::<Vec<f32>>(py) {
                        Ok(values) => values,
                        Err(err) => {
                            err_slot.replace(Some(err));
                            vec![0.0; couplings.len()]
                        }
                    },
                    Err(err) => {
                        err_slot.replace(Some(err));
                        vec![0.0; couplings.len()]
                    }
                }
            })
        };

        let lattice =
            rg_zspace::RgFlowLattice::from_mellin_grid(&grid.inner, root_couplings, &mut beta_fn)
                .map_err(rg_zspace_error);

        if let Some(err) = err_slot.into_inner() {
            return Err(err);
        }
        Ok(Self { inner: lattice? })
    }

    pub fn nodes(&self) -> Vec<PyRgFlowNode> {
        self.inner
            .nodes()
            .iter()
            .cloned()
            .map(|inner| PyRgFlowNode { inner })
            .collect()
    }

    pub fn mellin_series(&self) -> Option<Vec<ComplexScalar>> {
        self.inner.mellin_series().map(|series| series.to_vec())
    }

    pub fn evaluate_mellin(&self, s: ComplexScalar) -> PyResult<ComplexScalar> {
        self.inner.evaluate_mellin(s).map_err(rg_zspace_error)
    }

    pub fn evaluate_mellin_many(&self, s_values: Vec<ComplexScalar>) -> PyResult<Vec<ComplexScalar>> {
        self.inner
            .evaluate_mellin_many(&s_values)
            .map_err(rg_zspace_error)
    }

    pub fn fixed_points(&self, tolerance: f32) -> Vec<PyScaleFixedPoint> {
        self.inner
            .fixed_points(tolerance)
            .into_iter()
            .map(|inner| PyScaleFixedPoint { inner })
            .collect()
    }

    #[getter]
    pub fn log_step(&self) -> f32 {
        self.inner.log_step()
    }

    #[getter]
    pub fn log_start(&self) -> f32 {
        self.inner.log_start()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

pub fn register(_py: Python<'_>, module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lorentzian_metric_scaled, module)?)?;
    module.add_function(wrap_pyfunction!(assemble_zrelativity_model, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_resonance, module)?)?;
    module.add_function(wrap_pyfunction!(breathing_resonance, module)?)?;
    module.add_class::<PyZRelativityModel>()?;
    module.add_class::<PyResonanceProfile>()?;
    module.add_class::<PyRGOperator>()?;
    module.add_class::<PyRGFlowTrajectory>()?;
    module.add_class::<PyRGFlowSolution>()?;
    module.add_class::<PyRGFlowModel>()?;
    module.add_class::<PyRgFlowNode>()?;
    module.add_class::<PyScaleFixedPoint>()?;
    module.add_class::<PyRgFlowLattice>()?;
    Ok(())
}
