use crate::tensor::{tensor_err_to_py, tensor_to_torch, to_dlpack_impl, PyTensor};
use nalgebra::{DMatrix, Matrix4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use st_core::theory::general_relativity::{
    BoundaryCondition, BoundaryConditionKind, GeneralRelativityModel, InternalMetric,
    InternalPatch, InternalSpace, LorentzianMetric, MetricDerivatives, MetricError,
    MetricSecondDerivatives, MixedBlock, PhysicalConstants, ProductGeometry, ProductMetric,
    SymmetryAnsatz, Topology, WarpFactor, ZManifold, ZRelativityModel, ZRelativityTensorBundle,
};
use st_tensor::Tensor;

fn metric_error(err: MetricError) -> PyErr {
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

pub fn register(_py: Python<'_>, module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lorentzian_metric_scaled, module)?)?;
    module.add_function(wrap_pyfunction!(assemble_zrelativity_model, module)?)?;
    module.add_class::<PyZRelativityModel>()?;
    Ok(())
}
