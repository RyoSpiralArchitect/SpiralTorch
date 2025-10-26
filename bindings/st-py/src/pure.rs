use num_complex::Complex32 as PyComplex32;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::{wrap_pyfunction, Bound, PyRefMut};

use crate::tensor::{tensor_err_to_py, PyTensor};

use st_tensor::measure::{
    z_space_barycenter as z_space_barycenter_rs, BarycenterIntermediate, ZSpaceBarycenter,
};
use st_tensor::{
    AmegaHypergrad, Complex32 as StComplex32, ComplexTensor, GradientSummary, LanguageWaveEncoder,
    OpenCartesianTopos, Tensor, TensorBiome,
};

fn py_complex_to_st(values: Vec<PyComplex32>) -> Vec<StComplex32> {
    values
        .into_iter()
        .map(|value| StComplex32::new(value.re, value.im))
        .collect()
}

fn st_complex_to_py(values: &[StComplex32]) -> Vec<PyComplex32> {
    values
        .iter()
        .map(|value| PyComplex32::new(value.re, value.im))
        .collect()
}

#[pyclass(module = "spiraltorch", name = "ComplexTensor")]
#[derive(Clone)]
pub(crate) struct PyComplexTensor {
    pub(crate) inner: ComplexTensor,
}

#[pymethods]
impl PyComplexTensor {
    #[new]
    #[pyo3(signature = (rows, cols, data=None))]
    pub fn new(rows: usize, cols: usize, data: Option<Vec<PyComplex32>>) -> PyResult<Self> {
        let inner = match data {
            Some(values) => {
                let converted = py_complex_to_st(values);
                ComplexTensor::from_vec(rows, cols, converted)
            }
            None => ComplexTensor::zeros(rows, cols),
        }
        .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn zeros(rows: usize, cols: usize) -> PyResult<Self> {
        let inner = ComplexTensor::zeros(rows, cols).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn rows(&self) -> usize {
        self.inner.shape().0
    }

    #[getter]
    pub fn cols(&self) -> usize {
        self.inner.shape().1
    }

    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    pub fn to_tensor(&self) -> PyResult<PyTensor> {
        let tensor = self.inner.to_tensor().map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    pub fn data(&self) -> Vec<PyComplex32> {
        st_complex_to_py(self.inner.data())
    }

    pub fn matmul(&self, other: &PyComplexTensor) -> PyResult<Self> {
        let product = self.inner.matmul(&other.inner).map_err(tensor_err_to_py)?;
        Ok(Self { inner: product })
    }
}

#[pyclass(module = "spiraltorch", name = "OpenCartesianTopos")]
#[derive(Clone)]
pub(crate) struct PyOpenCartesianTopos {
    pub(crate) inner: OpenCartesianTopos,
}

impl PyOpenCartesianTopos {
    pub(crate) fn from_topos(inner: OpenCartesianTopos) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpenCartesianTopos {
    #[new]
    pub fn new(
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PyResult<Self> {
        let inner =
            OpenCartesianTopos::new(curvature, tolerance, saturation, max_depth, max_volume)
                .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    pub fn tolerance(&self) -> f32 {
        self.inner.tolerance()
    }

    pub fn saturation(&self) -> f32 {
        self.inner.saturation()
    }

    pub fn max_depth(&self) -> usize {
        self.inner.max_depth()
    }

    pub fn max_volume(&self) -> usize {
        self.inner.max_volume()
    }

    pub fn ensure_loop_free(&self, depth: usize) -> PyResult<()> {
        self.inner.ensure_loop_free(depth).map_err(tensor_err_to_py)
    }

    pub fn saturate(&self, value: f32) -> f32 {
        self.inner.saturate(value)
    }
}

#[pyclass(module = "spiraltorch", name = "LanguageWaveEncoder")]
#[derive(Clone)]
pub(crate) struct PyLanguageWaveEncoder {
    pub(crate) inner: LanguageWaveEncoder,
}

#[pymethods]
impl PyLanguageWaveEncoder {
    #[new]
    pub fn new(curvature: f32, temperature: f32) -> PyResult<Self> {
        let inner = LanguageWaveEncoder::new(curvature, temperature).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    pub fn temperature(&self) -> f32 {
        self.inner.temperature()
    }

    pub fn encode_wave(&self, text: &str) -> PyResult<PyComplexTensor> {
        let wave = self.inner.encode_wave(text).map_err(tensor_err_to_py)?;
        Ok(PyComplexTensor { inner: wave })
    }

    pub fn encode_z_space(&self, text: &str) -> PyResult<PyTensor> {
        let tensor = self.inner.encode_z_space(text).map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }
}

#[pyclass(module = "spiraltorch", name = "GradientSummary")]
#[derive(Clone, Copy)]
pub(crate) struct PyGradientSummary {
    inner: GradientSummary,
}

impl From<GradientSummary> for PyGradientSummary {
    fn from(inner: GradientSummary) -> Self {
        Self { inner }
    }
}

impl PyGradientSummary {
    pub(crate) fn as_inner(&self) -> GradientSummary {
        self.inner
    }
}

#[pymethods]
impl PyGradientSummary {
    pub fn l1(&self) -> f32 {
        self.inner.l1()
    }

    pub fn l2(&self) -> f32 {
        self.inner.l2()
    }

    pub fn linf(&self) -> f32 {
        self.inner.linf()
    }

    pub fn count(&self) -> usize {
        self.inner.count()
    }

    pub fn mean_abs(&self) -> f32 {
        self.inner.mean_abs()
    }

    pub fn rms(&self) -> f32 {
        self.inner.rms()
    }

    pub fn sum_squares(&self) -> f32 {
        self.inner.sum_squares()
    }
}

#[pyclass(module = "spiraltorch", name = "Hypergrad", unsendable)]
pub(crate) struct PyHypergrad {
    inner: AmegaHypergrad,
}

#[pymethods]
impl PyHypergrad {
    #[new]
    #[pyo3(signature = (curvature, learning_rate, rows, cols, topos=None))]
    pub fn new(
        curvature: f32,
        learning_rate: f32,
        rows: usize,
        cols: usize,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<Self> {
        let inner = match topos {
            Some(guard) => AmegaHypergrad::with_topos(
                curvature,
                learning_rate,
                rows,
                cols,
                guard.inner.clone(),
            ),
            None => AmegaHypergrad::new(curvature, learning_rate, rows, cols),
        }
        .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    pub fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    pub fn gradient(&self) -> Vec<f32> {
        self.inner.gradient().to_vec()
    }

    pub fn summary(&self) -> PyGradientSummary {
        self.inner.summary().into()
    }

    pub fn scale_learning_rate(&mut self, factor: f32) {
        self.inner.scale_learning_rate(factor);
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn retune(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        self.inner
            .retune(curvature, learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn accumulate_wave(&mut self, tensor: &PyTensor) -> PyResult<()> {
        self.inner
            .accumulate_wave(&tensor.inner)
            .map_err(tensor_err_to_py)
    }

    pub fn accumulate_complex_wave(&mut self, wave: &PyComplexTensor) -> PyResult<()> {
        self.inner
            .accumulate_complex_wave(&wave.inner)
            .map_err(tensor_err_to_py)
    }

    pub fn absorb_text(&mut self, encoder: &PyLanguageWaveEncoder, text: &str) -> PyResult<()> {
        self.inner
            .absorb_text(&encoder.inner, text)
            .map_err(tensor_err_to_py)
    }

    pub fn accumulate_pair(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<()> {
        self.inner
            .accumulate_pair(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)
    }

    pub fn apply(&mut self, mut weights: PyRefMut<'_, PyTensor>) -> PyResult<()> {
        self.inner
            .apply(&mut weights.inner)
            .map_err(tensor_err_to_py)
    }

    pub fn accumulate_barycenter_path(
        &mut self,
        intermediates: Vec<PyBarycenterIntermediate>,
    ) -> PyResult<()> {
        if intermediates.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "barycenter intermediates cannot be empty",
            ));
        }
        let stages: Vec<BarycenterIntermediate> =
            intermediates.into_iter().map(|stage| stage.inner).collect();
        self.inner
            .accumulate_barycenter_path(&stages)
            .map_err(tensor_err_to_py)
    }

    pub fn topos(&self) -> PyOpenCartesianTopos {
        PyOpenCartesianTopos::from_topos(self.inner.topos().clone())
    }
}

#[pyclass(module = "spiraltorch", name = "TensorBiome", unsendable)]
pub(crate) struct PyTensorBiome {
    inner: TensorBiome,
}

#[pyclass(module = "spiraltorch", name = "BarycenterIntermediate", unsendable)]
#[derive(Clone)]
pub(crate) struct PyBarycenterIntermediate {
    inner: BarycenterIntermediate,
}

impl From<BarycenterIntermediate> for PyBarycenterIntermediate {
    fn from(inner: BarycenterIntermediate) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "spiraltorch", name = "ZSpaceBarycenter", unsendable)]
pub(crate) struct PyZSpaceBarycenter {
    inner: ZSpaceBarycenter,
}

impl From<ZSpaceBarycenter> for PyZSpaceBarycenter {
    fn from(inner: ZSpaceBarycenter) -> Self {
        Self { inner }
    }
}

#[pyfunction]
#[pyo3(
    name = "z_space_barycenter",
    signature = (weights, densities, entropy_weight, beta_j, coupling=None)
)]
fn py_z_space_barycenter(
    py: Python<'_>,
    weights: Vec<f32>,
    densities: Vec<Py<PyTensor>>,
    entropy_weight: f32,
    beta_j: f32,
    coupling: Option<Py<PyTensor>>,
) -> PyResult<PyZSpaceBarycenter> {
    let density_clones: Vec<Tensor> = densities
        .into_iter()
        .map(|tensor| tensor.bind(py).borrow().inner.clone())
        .collect();
    let coupling_tensor = coupling.map(|tensor| tensor.bind(py).borrow().inner.clone());
    let barycenter = z_space_barycenter_rs(
        &weights,
        &density_clones,
        entropy_weight,
        beta_j,
        coupling_tensor.as_ref(),
    )
    .map_err(tensor_err_to_py)?;
    Ok(PyZSpaceBarycenter::from(barycenter))
}

#[pymethods]
impl PyTensorBiome {
    #[new]
    pub fn new(topos: &PyOpenCartesianTopos) -> Self {
        Self {
            inner: TensorBiome::new(topos.inner.clone()),
        }
    }

    pub fn topos(&self) -> PyOpenCartesianTopos {
        PyOpenCartesianTopos::from_topos(self.inner.topos().clone())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn total_weight(&self) -> f32 {
        self.inner.total_weight()
    }

    pub fn weights(&self) -> Vec<f32> {
        self.inner.weights().to_vec()
    }

    pub fn absorb(&mut self, tensor: &PyTensor) -> PyResult<()> {
        self.inner
            .absorb("py_tensor_biome_absorb", tensor.inner.clone())
            .map_err(tensor_err_to_py)
    }

    pub fn absorb_weighted(&mut self, tensor: &PyTensor, weight: f32) -> PyResult<()> {
        self.inner
            .absorb_weighted(
                "py_tensor_biome_absorb_weighted",
                tensor.inner.clone(),
                weight,
            )
            .map_err(tensor_err_to_py)
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    pub fn canopy(&self) -> PyResult<PyTensor> {
        let tensor = self.inner.canopy().map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }
}

#[pymethods]
impl PyBarycenterIntermediate {
    #[getter]
    fn interpolation(&self) -> f32 {
        self.inner.interpolation
    }

    #[getter]
    fn density(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.density.clone()))
    }

    #[getter]
    fn kl_energy(&self) -> f32 {
        self.inner.kl_energy
    }

    #[getter]
    fn entropy(&self) -> f32 {
        self.inner.entropy
    }

    #[getter]
    fn objective(&self) -> f32 {
        self.inner.objective
    }
}

#[pymethods]
impl PyZSpaceBarycenter {
    #[getter]
    fn density(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.density.clone()))
    }

    #[getter]
    fn kl_energy(&self) -> f32 {
        self.inner.kl_energy
    }

    #[getter]
    fn entropy(&self) -> f32 {
        self.inner.entropy
    }

    #[getter]
    fn coupling_energy(&self) -> f32 {
        self.inner.coupling_energy
    }

    #[getter]
    fn objective(&self) -> f32 {
        self.inner.objective
    }

    #[getter]
    fn effective_weight(&self) -> f32 {
        self.inner.effective_weight
    }

    fn intermediates(&self) -> Vec<PyBarycenterIntermediate> {
        self.inner
            .intermediates
            .iter()
            .cloned()
            .map(PyBarycenterIntermediate::from)
            .collect()
    }
}

pub(crate) fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyComplexTensor>()?;
    m.add_class::<PyOpenCartesianTopos>()?;
    m.add_class::<PyLanguageWaveEncoder>()?;
    m.add_class::<PyGradientSummary>()?;
    m.add_class::<PyHypergrad>()?;
    m.add_class::<PyTensorBiome>()?;
    m.add_class::<PyBarycenterIntermediate>()?;
    m.add_class::<PyZSpaceBarycenter>()?;
    m.add_function(wrap_pyfunction!(py_z_space_barycenter, m)?)?;
    Ok(())
}
