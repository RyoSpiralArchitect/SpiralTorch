mod sot;

use ndarray::{Array2, ArrayD, Ix2};
use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use pyo3::PyRefMut;
use st_backend_hip::{
    device_info as hip_device_info, hip_available as hip_runtime_available,
    DeviceInfo as HipDeviceInfo,
};
use st_core::backend::device_caps::{BackendKind, DeviceCaps};
use st_core::backend::unison_heuristics::RankKind;
use st_core::ops::rank_entry::{plan_rank, RankPlan};
use st_frac::fft::{fft_inplace as frac_fft_inplace, Complex32 as FracComplex32, FftError};
use st_frac::{
    fracdiff_gl_nd, fracdiff_gl_nd_backward, gl_coeffs as frac_gl_coeffs, FracErr, Pad as FracPad,
};
use st_nn::{
    Conv1d as NnConv1d, DifferentialTrace, Linear as NnLinear, Module, Sequential as NnSequential,
    SpiralSession, SpiralSessionBuilder, WaveRnn as NnWaveRnn,
};
use st_tensor::pure::{
    measure::{
        z_space_barycenter as rust_z_space_barycenter, BarycenterIntermediate, ZSpaceBarycenter,
    },
    topos::OpenCartesianTopos,
    AmegaHypergrad, Complex32, ComplexTensor, DifferentialResonance, LanguageWaveEncoder,
    PureResult, Tensor, TensorError,
};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn tensor_err(err: TensorError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn convert<T>(value: PureResult<T>) -> PyResult<T> {
    value.map_err(tensor_err)
}

fn convert_frac<T>(value: Result<T, FracErr>) -> PyResult<T> {
    value.map_err(|err| PyValueError::new_err(err.to_string()))
}

fn convert_fft<T>(value: Result<T, FftError>) -> PyResult<T> {
    value.map_err(|err| {
        PyValueError::new_err(match err {
            FftError::Empty => "signal must not be empty".to_string(),
            FftError::NonPowerOfTwo => {
                "signal length must be a power of two for the radix FFT".to_string()
            }
        })
    })
}

fn intern_label(label: &str) -> &'static str {
    static INTERNER: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let storage = INTERNER.get_or_init(|| Mutex::new(Vec::new()));
    {
        let guard = storage.lock().expect("intern labels lock poisoned");
        if let Some(&existing) = guard.iter().find(|&&item| item == label) {
            return existing;
        }
    }
    let leaked: &'static str = Box::leak(label.to_owned().into_boxed_str());
    let mut guard = storage.lock().expect("intern labels lock poisoned");
    guard.push(leaked);
    leaked
}

fn state_to_pydict(py: Python<'_>, state: HashMap<String, Tensor>) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    for (name, tensor) in state {
        let py_tensor = PyTensor::from_tensor(tensor);
        dict.set_item(name, py_tensor.into_py(py))?;
    }
    Ok(dict.into_py(py))
}

fn pydict_to_state(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, Tensor>> {
    let mut state = HashMap::new();
    for (key, value) in dict.iter() {
        let name: String = key.extract()?;
        let tensor: PyTensor = value.extract()?;
        state.insert(name, tensor.as_tensor().clone());
    }
    Ok(state)
}

fn py_device_info<'py>(py: Python<'py>, info: HipDeviceInfo) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", info.id)?;
    dict.set_item("name", info.name.as_ref())?;
    dict.set_item("multi_node", info.multi_node)?;
    Ok(dict)
}

fn backend_name(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Wgpu => "wgpu",
        BackendKind::Cuda => "cuda",
        BackendKind::Hip => "hip",
        BackendKind::Cpu => "cpu",
    }
}

fn tensor_to_array(tensor: &Tensor) -> PyResult<ArrayD<f32>> {
    let (rows, cols) = tensor.shape();
    Array2::from_shape_vec((rows, cols), tensor.data().to_vec())
        .map(|array| array.into_dyn())
        .map_err(|_| PyValueError::new_err("failed to view tensor as ndarray"))
}

fn array_to_tensor(array: ArrayD<f32>) -> PyResult<PyTensor> {
    let matrix: Array2<f32> = array
        .into_dimensionality::<Ix2>()
        .map_err(|_| PyValueError::new_err("fractional operators require 2D tensors"))?;
    let (rows, cols) = matrix.dim();
    let data = matrix.into_raw_vec();
    Ok(PyTensor::from_tensor(convert(Tensor::from_vec(
        rows, cols, data,
    ))?))
}

fn device_caps_dict<'py>(py: Python<'py>, caps: DeviceCaps) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("backend", backend_name(caps.backend))?;
    dict.set_item("lane_width", caps.lane_width)?;
    dict.set_item("max_workgroup", caps.max_workgroup)?;
    dict.set_item("subgroup", caps.subgroup)?;
    match caps.shared_mem_per_workgroup {
        Some(value) => dict.set_item("shared_mem_per_workgroup", value)?,
        None => dict.set_item("shared_mem_per_workgroup", py.None())?,
    }
    Ok(dict)
}

#[pyclass(module = "spiraltorch", name = "Tensor")]
#[derive(Clone, Debug)]
struct PyTensor {
    inner: Tensor,
}

impl PyTensor {
    fn from_tensor(tensor: Tensor) -> Self {
        Self { inner: tensor }
    }

    fn as_tensor(&self) -> &Tensor {
        &self.inner
    }

    fn as_tensor_mut(&mut self) -> &mut Tensor {
        &mut self.inner
    }

    fn into_tensor(self) -> Tensor {
        self.inner
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (rows, cols, data=None))]
    fn new(rows: usize, cols: usize, data: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let tensor = match data {
            Some(obj) => {
                let values: Vec<f32> = obj.extract()?;
                convert(Tensor::from_vec(rows, cols, values))?
            }
            None => convert(Tensor::zeros(rows, cols))?,
        };
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    fn zeros(rows: usize, cols: usize) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(Tensor::zeros(rows, cols))?))
    }

    #[pyo3(name = "clone")]
    fn clone_py(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.shape().0
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.shape().1
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn data(&self) -> Vec<f32> {
        self.inner.data().to_vec()
    }

    fn tolist(&self) -> Vec<Vec<f32>> {
        let (rows, cols) = self.inner.shape();
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let start = r * cols;
            let end = start + cols;
            out.push(self.inner.data()[start..end].to_vec());
        }
        out
    }

    fn matmul(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(
            self.inner.matmul(other.as_tensor()),
        )?))
    }

    fn add(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(
            self.inner.add(other.as_tensor()),
        )?))
    }

    fn sub(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(
            self.inner.sub(other.as_tensor()),
        )?))
    }

    fn scale(&self, value: f32) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(self.inner.scale(value))?))
    }

    fn hadamard(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(
            self.inner.hadamard(other.as_tensor()),
        )?))
    }

    fn add_scaled_inplace(&mut self, other: &PyTensor, scale: f32) -> PyResult<()> {
        convert(self.inner.add_scaled(other.as_tensor(), scale))
    }

    fn add_row_inplace(&mut self, bias: Vec<f32>) -> PyResult<()> {
        convert(self.inner.add_row_inplace(&bias))
    }

    fn transpose(&self) -> Self {
        Self::from_tensor(self.inner.transpose())
    }

    fn sum_axis0(&self) -> Vec<f32> {
        self.inner.sum_axis0()
    }

    fn squared_l2_norm(&self) -> f32 {
        self.inner.squared_l2_norm()
    }

    fn project_to_poincare(&self, curvature: f32) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(
            self.inner.project_to_poincare(curvature),
        )?))
    }

    fn hyperbolic_distance(&self, other: &PyTensor, curvature: f32) -> PyResult<f32> {
        convert(self.inner.hyperbolic_distance(other.as_tensor(), curvature))
    }

    fn __repr__(&self) -> PyResult<String> {
        let (rows, cols) = self.inner.shape();
        Ok(format!("Tensor(rows={rows}, cols={cols})"))
    }
}

#[pyclass(module = "spiraltorch", name = "ComplexTensor")]
#[derive(Clone, Debug)]
struct PyComplexTensor {
    inner: ComplexTensor,
}

impl PyComplexTensor {
    fn from_complex(inner: ComplexTensor) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "spiraltorch", name = "BarycenterIntermediate")]
#[derive(Clone, Debug)]
struct PyBarycenterIntermediate {
    inner: BarycenterIntermediate,
}

impl PyBarycenterIntermediate {
    fn from_stage(stage: BarycenterIntermediate) -> Self {
        Self { inner: stage }
    }
}

#[pyclass(module = "spiraltorch", name = "ZSpaceBarycenter")]
#[derive(Clone, Debug)]
struct PyZSpaceBarycenter {
    inner: ZSpaceBarycenter,
}

impl PyZSpaceBarycenter {
    fn from_result(result: ZSpaceBarycenter) -> Self {
        Self { inner: result }
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

    fn intermediates(&self, py: Python<'_>) -> PyResult<Vec<Py<PyBarycenterIntermediate>>> {
        let mut out = Vec::with_capacity(self.inner.intermediates.len());
        for stage in &self.inner.intermediates {
            out.push(Py::new(
                py,
                PyBarycenterIntermediate::from_stage(stage.clone()),
            )?);
        }
        Ok(out)
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "density",
            PyTensor::from_tensor(self.inner.density.clone()).into_py(py),
        )?;
        dict.set_item("kl_energy", self.inner.kl_energy)?;
        dict.set_item("entropy", self.inner.entropy)?;
        dict.set_item("coupling_energy", self.inner.coupling_energy)?;
        dict.set_item("objective", self.inner.objective)?;
        dict.set_item("effective_weight", self.inner.effective_weight)?;
        let py_intermediates = PyList::empty_bound(py);
        for stage in &self.inner.intermediates {
            py_intermediates
                .append(PyBarycenterIntermediate::from_stage(stage.clone()).into_py(py))?;
        }
        dict.set_item("intermediates", py_intermediates.into_py(py))?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ZSpaceBarycenter(objective={:.6}, entropy={:.6})",
            self.inner.objective, self.inner.entropy
        ))
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

    fn as_tuple(&self) -> PyResult<(f32, PyTensor, f32, f32, f32)> {
        Ok((
            self.inner.interpolation,
            PyTensor::from_tensor(self.inner.density.clone()),
            self.inner.kl_energy,
            self.inner.entropy,
            self.inner.objective,
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "BarycenterIntermediate(interpolation={:.2}, objective={:.6})",
            self.inner.interpolation, self.inner.objective
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "DifferentialResonance")]
#[derive(Clone)]
struct PyDifferentialResonance {
    inner: DifferentialResonance,
}

impl PyDifferentialResonance {
    fn from_resonance(inner: DifferentialResonance) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDifferentialResonance {
    #[getter]
    fn homotopy_flow(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.homotopy_flow.clone()))
    }

    #[getter]
    fn functor_linearisation(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(
            self.inner.functor_linearisation.clone(),
        ))
    }

    #[getter]
    fn recursive_objective(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(
            self.inner.recursive_objective.clone(),
        ))
    }

    #[getter]
    fn infinity_projection(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(
            self.inner.infinity_projection.clone(),
        ))
    }

    #[getter]
    fn infinity_energy(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.infinity_energy.clone()))
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "homotopy_flow",
            PyTensor::from_tensor(self.inner.homotopy_flow.clone()).into_py(py),
        )?;
        dict.set_item(
            "functor_linearisation",
            PyTensor::from_tensor(self.inner.functor_linearisation.clone()).into_py(py),
        )?;
        dict.set_item(
            "recursive_objective",
            PyTensor::from_tensor(self.inner.recursive_objective.clone()).into_py(py),
        )?;
        dict.set_item(
            "infinity_projection",
            PyTensor::from_tensor(self.inner.infinity_projection.clone()).into_py(py),
        )?;
        dict.set_item(
            "infinity_energy",
            PyTensor::from_tensor(self.inner.infinity_energy.clone()).into_py(py),
        )?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok("DifferentialResonance(...)".to_string())
    }
}

#[pyclass(module = "spiraltorch", name = "SpiralDifferentialTrace")]
struct PySpiralDifferentialTrace {
    trace: Option<DifferentialTrace>,
}

impl PySpiralDifferentialTrace {
    fn from_trace(trace: DifferentialTrace) -> Self {
        Self { trace: Some(trace) }
    }

    fn map_trace<F>(&mut self, f: F) -> PyResult<()>
    where
        F: FnOnce(DifferentialTrace) -> PureResult<DifferentialTrace>,
    {
        let trace = self
            .trace
            .take()
            .ok_or_else(|| PyValueError::new_err("trace has already been consumed"))?;
        let trace = convert(f(trace))?;
        self.trace = Some(trace);
        Ok(())
    }

    fn take_trace(&mut self) -> PyResult<DifferentialTrace> {
        self.trace
            .take()
            .ok_or_else(|| PyValueError::new_err("trace has already been consumed"))
    }
}

#[pymethods]
impl PySpiralDifferentialTrace {
    fn deform(&mut self, generator: &PyTensor, direction: &PyTensor) -> PyResult<()> {
        let generator = generator.as_tensor().clone();
        let direction = direction.as_tensor().clone();
        self.map_trace(move |trace| trace.deform(generator.clone(), direction.clone()))
    }

    fn across(&mut self, topos: &PyOpenTopos) -> PyResult<()> {
        let guard = topos.inner.clone();
        self.map_trace(move |trace| trace.across(guard.clone()))
    }

    #[pyo3(signature = (kernel, source=None))]
    fn via(&mut self, kernel: &PyTensor, source: Option<&PyTensor>) -> PyResult<()> {
        let kernel_tensor = kernel.as_tensor().clone();
        let source_tensor = source.map(|s| s.as_tensor().clone());
        self.map_trace(move |trace| {
            if let Some(ref src) = source_tensor {
                trace.via_with(kernel_tensor.clone(), src.clone())
            } else {
                trace.via(kernel_tensor.clone())
            }
        })
    }

    fn functor_step(&mut self, epsilon: f32) -> PyResult<()> {
        self.map_trace(move |trace| trace.functor_step(epsilon))
    }

    fn with_barycenter(&mut self, barycenter: &PyZSpaceBarycenter) -> PyResult<()> {
        let barycenter_clone = barycenter.inner.clone();
        self.map_trace(move |trace| trace.with_barycenter(&barycenter_clone))
    }

    fn with_barycenter_from(
        &mut self,
        weights: Vec<f32>,
        densities: Vec<PyTensor>,
    ) -> PyResult<()> {
        let tensors: Vec<Tensor> = densities.into_iter().map(PyTensor::into_tensor).collect();
        self.map_trace(move |trace| trace.with_barycenter_from(&weights, tensors.as_slice()))
    }

    #[pyo3(signature = (weights, densities, coupling=None))]
    fn with_barycenter_with(
        &mut self,
        weights: Vec<f32>,
        densities: Vec<PyTensor>,
        coupling: Option<&PyTensor>,
    ) -> PyResult<()> {
        let tensors: Vec<Tensor> = densities.into_iter().map(PyTensor::into_tensor).collect();
        let coupling_tensor = coupling.map(|tensor| tensor.as_tensor().clone());
        self.map_trace(move |trace| {
            let coupling_ref: Option<&Tensor> = coupling_tensor.as_ref().map(|tensor| tensor);
            trace.with_barycenter_with(&weights, tensors.as_slice(), coupling_ref)
        })
    }

    #[pyo3(signature = (levels, curvatures=None))]
    fn with_infinity(
        &mut self,
        levels: Vec<PyTensor>,
        curvatures: Option<Vec<f32>>,
    ) -> PyResult<()> {
        let tensors: Vec<Tensor> = levels.into_iter().map(PyTensor::into_tensor).collect();
        let curvatures = curvatures.unwrap_or_default();
        self.map_trace(move |trace| trace.with_infinity(tensors.clone(), curvatures.clone()))
    }

    fn resonate(&mut self) -> PyResult<PyDifferentialResonance> {
        let trace = self.take_trace()?;
        let resonance = convert(trace.resonate())?;
        Ok(PyDifferentialResonance::from_resonance(resonance))
    }

    fn resonate_with_hypergrad(
        &mut self,
        hypergrad: &mut PyHypergrad,
    ) -> PyResult<PyDifferentialResonance> {
        let trace = self.take_trace()?;
        let resonance = convert(trace.resonate_with_hypergrad(&mut hypergrad.inner))?;
        Ok(PyDifferentialResonance::from_resonance(resonance))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok("SpiralDifferentialTrace(...)".to_string())
    }
}

#[pymethods]
impl PyComplexTensor {
    #[new]
    #[pyo3(signature = (rows, cols, data=None))]
    fn new(rows: usize, cols: usize, data: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let tensor = match data {
            Some(obj) => {
                let raw: Vec<(f32, f32)> = obj.extract()?;
                let values = raw
                    .into_iter()
                    .map(|(re, im)| Complex32::new(re, im))
                    .collect();
                convert(ComplexTensor::from_vec(rows, cols, values))?
            }
            None => convert(ComplexTensor::zeros(rows, cols))?,
        };
        Ok(Self { inner: tensor })
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn data(&self) -> Vec<(f32, f32)> {
        self.inner
            .data()
            .iter()
            .map(|value| (value.re, value.im))
            .collect()
    }

    fn to_tensor(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(self.inner.to_tensor())?))
    }

    fn matmul(&self, other: &PyComplexTensor) -> PyResult<Self> {
        Ok(Self::from_complex(convert(
            self.inner.matmul(&other.inner),
        )?))
    }

    fn __repr__(&self) -> PyResult<String> {
        let (rows, cols) = self.inner.shape();
        Ok(format!("ComplexTensor(rows={rows}, cols={cols})"))
    }
}

#[pyclass(module = "spiraltorch", name = "OpenTopos")]
#[derive(Clone, Debug)]
struct PyOpenTopos {
    inner: OpenCartesianTopos,
}

impl PyOpenTopos {
    fn from_topos(topos: OpenCartesianTopos) -> Self {
        Self { inner: topos }
    }
}

#[pymethods]
impl PyOpenTopos {
    #[new]
    fn new(
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PyResult<Self> {
        Ok(Self::from_topos(convert(OpenCartesianTopos::new(
            curvature, tolerance, saturation, max_depth, max_volume,
        ))?))
    }

    fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    fn tolerance(&self) -> f32 {
        self.inner.tolerance()
    }

    fn saturation(&self) -> f32 {
        self.inner.saturation()
    }

    fn max_depth(&self) -> usize {
        self.inner.max_depth()
    }

    fn max_volume(&self) -> usize {
        self.inner.max_volume()
    }

    fn guard_tensor(&self, label: &str, tensor: &PyTensor) -> PyResult<()> {
        convert(
            self.inner
                .guard_tensor(intern_label(label), tensor.as_tensor()),
        )
    }

    fn saturate_scalar(&self, value: f32) -> f32 {
        self.inner.saturate(value)
    }

    fn saturate_tensor(&self, mut tensor: PyRefMut<'_, PyTensor>) -> PyResult<()> {
        let inner = tensor.as_tensor_mut();
        self.inner.saturate_slice(inner.data_mut());
        convert(self.inner.guard_tensor("tensor", inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "OpenTopos(curvature={}, tolerance={}, saturation={}, max_depth={}, max_volume={})",
            self.curvature(),
            self.tolerance(),
            self.saturation(),
            self.max_depth(),
            self.max_volume()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "LanguageWaveEncoder")]
#[derive(Clone, Debug)]
struct PyLanguageWaveEncoder {
    inner: LanguageWaveEncoder,
}

impl PyLanguageWaveEncoder {
    fn encode_wave_internal(&self, text: &str) -> PyResult<ComplexTensor> {
        convert(self.inner.encode_wave(text))
    }
}

#[pymethods]
impl PyLanguageWaveEncoder {
    #[new]
    fn new(curvature: f32, temperature: f32) -> PyResult<Self> {
        Ok(Self {
            inner: convert(LanguageWaveEncoder::new(curvature, temperature))?,
        })
    }

    fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    fn temperature(&self) -> f32 {
        self.inner.temperature()
    }

    fn encode_wave(&self, text: &str) -> PyResult<PyComplexTensor> {
        Ok(PyComplexTensor::from_complex(
            self.encode_wave_internal(text)?,
        ))
    }

    fn encode_z_space(&self, text: &str) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.inner.encode_z_space(text),
        )?))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "LanguageWaveEncoder(curvature={}, temperature={})",
            self.curvature(),
            self.temperature()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "Hypergrad")]
struct PyHypergrad {
    inner: AmegaHypergrad,
}

impl PyHypergrad {
    fn from_hypergrad(inner: AmegaHypergrad) -> Self {
        Self { inner }
    }

    fn ensure_shape(&self, tensor: &PyTensor) -> PyResult<()> {
        let shape = tensor.shape();
        if shape != self.inner.shape() {
            return Err(PyValueError::new_err(format!(
                "tensor shape {:?} does not match hypergrad {:?}",
                shape,
                self.inner.shape()
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PyHypergrad {
    #[new]
    #[pyo3(signature = (curvature, learning_rate, rows, cols, topos=None))]
    fn new(
        curvature: f32,
        learning_rate: f32,
        rows: usize,
        cols: usize,
        topos: Option<&PyOpenTopos>,
    ) -> PyResult<Self> {
        let inner = if let Some(topos) = topos {
            convert(AmegaHypergrad::with_topos(
                curvature,
                learning_rate,
                rows,
                cols,
                topos.inner.clone(),
            ))?
        } else {
            convert(AmegaHypergrad::new(curvature, learning_rate, rows, cols))?
        };
        Ok(Self { inner })
    }

    fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn gradient(&self) -> Vec<f32> {
        self.inner.gradient().to_vec()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn accumulate_wave(&mut self, tensor: &PyTensor) -> PyResult<()> {
        self.ensure_shape(tensor)?;
        convert(self.inner.accumulate_wave(tensor.as_tensor()))
    }

    fn accumulate_complex_wave(&mut self, wave: &PyComplexTensor) -> PyResult<()> {
        convert(self.inner.accumulate_complex_wave(&wave.inner))
    }

    fn accumulate_pair(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<()> {
        self.ensure_shape(prediction)?;
        convert(
            self.inner
                .accumulate_pair(prediction.as_tensor(), target.as_tensor()),
        )
    }

    fn absorb_text(&mut self, encoder: &PyLanguageWaveEncoder, text: &str) -> PyResult<()> {
        convert(self.inner.absorb_text(&encoder.inner, text))
    }

    #[pyo3(signature = (intermediates))]
    fn accumulate_barycenter_path(
        &mut self,
        py: Python<'_>,
        intermediates: Vec<Py<PyBarycenterIntermediate>>,
    ) -> PyResult<()> {
        if intermediates.is_empty() {
            return Err(PyValueError::new_err(
                "barycenter intermediates must not be empty",
            ));
        }
        let mut stages = Vec::with_capacity(intermediates.len());
        for stage in intermediates {
            let guard = stage.borrow(py);
            stages.push(guard.inner.clone());
        }
        convert(self.inner.accumulate_barycenter_path(&stages))
    }

    fn apply(&mut self, mut weights: PyRefMut<'_, PyTensor>) -> PyResult<()> {
        self.ensure_shape(&weights)?;
        convert(self.inner.apply(weights.as_tensor_mut()))
    }

    fn topos(&self) -> PyResult<PyOpenTopos> {
        Ok(PyOpenTopos::from_topos(self.inner.topos().clone()))
    }

    fn __repr__(&self) -> PyResult<String> {
        let (rows, cols) = self.shape();
        Ok(format!(
            "Hypergrad(curvature={}, learning_rate={}, rows={}, cols={})",
            self.curvature(),
            self.learning_rate(),
            rows,
            cols
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "SpiralSessionBuilder")]
struct PySpiralSessionBuilder {
    builder: Option<SpiralSessionBuilder>,
}

impl PySpiralSessionBuilder {
    fn from_builder(builder: SpiralSessionBuilder) -> Self {
        Self {
            builder: Some(builder),
        }
    }

    fn ensure_builder(&mut self) -> PyResult<&mut SpiralSessionBuilder> {
        self.builder
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("builder has already been consumed"))
    }
}

#[pymethods]
impl PySpiralSessionBuilder {
    #[new]
    #[pyo3(signature = (device=None))]
    fn new(device: Option<&str>) -> Self {
        let caps = caps_for(device);
        Self {
            builder: Some(SpiralSession::builder(caps)),
        }
    }

    fn curvature(&mut self, curvature: f32) -> PyResult<()> {
        self.ensure_builder()?.set_curvature(curvature);
        Ok(())
    }

    fn hyper_learning_rate(&mut self, learning_rate: f32) -> PyResult<()> {
        self.ensure_builder()?
            .set_hyper_learning_rate(learning_rate);
        Ok(())
    }

    fn fallback_learning_rate(&mut self, learning_rate: f32) -> PyResult<()> {
        self.ensure_builder()?
            .set_fallback_learning_rate(learning_rate);
        Ok(())
    }

    fn entropy_weight(&mut self, entropy_weight: f32) -> PyResult<()> {
        self.ensure_builder()?
            .set_barycenter_entropy(entropy_weight);
        Ok(())
    }

    fn beta_j(&mut self, beta_j: f32) -> PyResult<()> {
        self.ensure_builder()?.set_barycenter_beta_j(beta_j);
        Ok(())
    }

    #[pyo3(signature = (coupling=None))]
    fn coupling(&mut self, coupling: Option<PyTensor>) -> PyResult<()> {
        let tensor = coupling.map(PyTensor::into_tensor);
        self.ensure_builder()?.set_barycenter_coupling(tensor);
        Ok(())
    }

    fn topos_guard(&mut self, topos: &PyOpenTopos) -> PyResult<()> {
        self.ensure_builder()?.set_topos(Some(topos.inner.clone()));
        Ok(())
    }

    #[pyo3(signature = (curvature, tolerance, saturation, max_depth, max_volume))]
    fn topos(
        &mut self,
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PyResult<()> {
        self.ensure_builder()?
            .set_topos_from_params(curvature, tolerance, saturation, max_depth, max_volume)
            .map_err(tensor_err)?;
        Ok(())
    }

    fn clear_topos(&mut self) {
        if let Some(builder) = self.builder.as_mut() {
            builder.set_topos(None);
        }
    }

    fn build(&mut self) -> PyResult<PySpiralSession> {
        let builder = self
            .builder
            .take()
            .ok_or_else(|| PyValueError::new_err("builder has already been consumed"))?;
        let session = convert(builder.build())?;
        Ok(PySpiralSession::from_session(session))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok("SpiralSessionBuilder(...)".to_string())
    }
}

#[pyclass(module = "spiraltorch", name = "SpiralSession")]
#[derive(Clone)]
struct PySpiralSession {
    inner: SpiralSession,
}

impl PySpiralSession {
    fn from_session(inner: SpiralSession) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySpiralSession {
    #[new]
    #[pyo3(signature = (device=None, curvature=-1.0, hyper_learning_rate=0.05, fallback_learning_rate=0.01, entropy_weight=0.1, beta_j=0.0, topos=None, coupling=None))]
    fn new(
        device: Option<&str>,
        curvature: f32,
        hyper_learning_rate: f32,
        fallback_learning_rate: f32,
        entropy_weight: f32,
        beta_j: f32,
        topos: Option<&PyOpenTopos>,
        coupling: Option<&PyTensor>,
    ) -> PyResult<Self> {
        let caps = caps_for(device);
        let mut builder = SpiralSession::builder(caps);
        builder.set_curvature(curvature);
        builder.set_hyper_learning_rate(hyper_learning_rate);
        builder.set_fallback_learning_rate(fallback_learning_rate);
        builder.set_barycenter_entropy(entropy_weight);
        builder.set_barycenter_beta_j(beta_j);
        if let Some(topos) = topos {
            builder.set_topos(Some(topos.inner.clone()));
        }
        if let Some(coupling) = coupling {
            builder.set_barycenter_coupling(Some(coupling.as_tensor().clone()));
        }
        let session = convert(builder.build())?;
        Ok(Self { inner: session })
    }

    fn builder(&self) -> PySpiralSessionBuilder {
        PySpiralSessionBuilder::from_builder(self.inner.to_builder())
    }

    fn trace(&self, seed: &PyTensor) -> PyResult<PySpiralDifferentialTrace> {
        let trace = convert(self.inner.trace(seed.as_tensor().clone()))?;
        Ok(PySpiralDifferentialTrace::from_trace(trace))
    }

    #[getter]
    fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    #[getter]
    fn hyper_learning_rate(&self) -> f32 {
        self.inner.hyper_learning_rate()
    }

    #[getter]
    fn fallback_learning_rate(&self) -> f32 {
        self.inner.fallback_learning_rate()
    }

    #[getter]
    fn entropy_weight(&self) -> f32 {
        self.inner.barycenter_entropy_weight()
    }

    #[getter]
    fn beta_j(&self) -> f32 {
        self.inner.barycenter_beta_j()
    }

    #[getter]
    fn coupling(&self) -> Option<PyTensor> {
        self.inner
            .barycenter_coupling()
            .cloned()
            .map(PyTensor::from_tensor)
    }

    fn device_caps(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(device_caps_dict(py, self.inner.device_caps())?.into_py(py))
    }

    #[pyo3(signature = (kind, rows, cols, k))]
    fn plan(&self, py: Python<'_>, kind: &str, rows: u32, cols: u32, k: u32) -> PyResult<PyObject> {
        let rank_kind = parse_kind(kind)?;
        let plan = self.inner.plan_rank(rank_kind, rows, cols, k);
        let out = PyDict::new_bound(py);
        out.set_item("kind", kind.to_ascii_lowercase())?;
        out.set_item("rows", rows)?;
        out.set_item("cols", cols)?;
        out.set_item("k", k)?;
        out.set_item("choice", choice_dict(py, &plan)?.into_py(py))?;
        Ok(out.into_py(py))
    }

    fn hypergrad(&self, rows: usize, cols: usize) -> PyResult<PyHypergrad> {
        Ok(PyHypergrad::from_hypergrad(convert(
            self.inner.hypergrad(rows, cols),
        )?))
    }

    #[pyo3(signature = (densities, weights=None, entropy_weight=None, beta_j=None, coupling=None))]
    fn barycenter(
        &self,
        densities: Vec<PyTensor>,
        weights: Option<Vec<f32>>,
        entropy_weight: Option<f32>,
        beta_j: Option<f32>,
        coupling: Option<PyTensor>,
    ) -> PyResult<PyZSpaceBarycenter> {
        if densities.is_empty() {
            return Err(PyValueError::new_err("densities must not be empty"));
        }
        let tensors: Vec<Tensor> = densities.into_iter().map(PyTensor::into_tensor).collect();
        let weight_vec = weights.unwrap_or_else(|| vec![1.0; tensors.len()]);
        if weight_vec.len() != tensors.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} weights, received {}",
                tensors.len(),
                weight_vec.len()
            )));
        }
        let coupling_tensor = coupling.map(PyTensor::into_tensor);
        let coupling_ref = coupling_tensor.as_ref();
        let entropy = entropy_weight.unwrap_or_else(|| self.inner.barycenter_entropy_weight());
        let beta = beta_j.unwrap_or_else(|| self.inner.barycenter_beta_j());
        let result = self.inner.barycenter_with_parameters(
            &weight_vec,
            &tensors,
            entropy,
            beta,
            coupling_ref,
        );
        Ok(PyZSpaceBarycenter::from_result(convert(result)?))
    }

    fn align_hypergrad(
        &self,
        hypergrad: &mut PyHypergrad,
        barycenter: &PyZSpaceBarycenter,
    ) -> PyResult<()> {
        convert(
            self.inner
                .align_hypergrad(&mut hypergrad.inner, &barycenter.inner),
        )
    }

    #[getter]
    fn topos(&self) -> Option<PyOpenTopos> {
        self.inner.topos().cloned().map(PyOpenTopos::from_topos)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "SpiralSession(device={}, curvature={}, hyper_lr={}, fallback_lr={})",
            backend_name(self.inner.device_caps().backend),
            self.inner.curvature(),
            self.inner.hyper_learning_rate(),
            self.inner.fallback_learning_rate()
        ))
    }
}

#[pyclass(module = "spiraltorch.nn", name = "Linear")]
struct PyLinearModule {
    inner: Option<NnLinear>,
}

impl PyLinearModule {
    fn borrow(&self) -> PyResult<&NnLinear> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Linear module has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnLinear> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Linear module has been moved"))
    }

    fn take(&mut self) -> PyResult<NnLinear> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("Linear module has been moved"))
    }
}

#[pymethods]
impl PyLinearModule {
    #[new]
    #[pyo3(signature = (input_dim, output_dim, name=None))]
    fn new(input_dim: usize, output_dim: usize, name: Option<&str>) -> PyResult<Self> {
        let ident = name.unwrap_or("linear");
        let inner = convert(NnLinear::new(ident, input_dim, output_dim))?;
        Ok(Self { inner: Some(inner) })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let layer = self.borrow()?;
        Ok(PyTensor::from_tensor(convert(
            layer.forward(input.as_tensor()),
        )?))
    }

    fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let layer = self.borrow_mut()?;
        Ok(PyTensor::from_tensor(convert(
            layer.backward(input.as_tensor(), grad_output.as_tensor()),
        )?))
    }

    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        convert(
            self.borrow_mut()?
                .attach_hypergrad(curvature, learning_rate),
        )
    }

    fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        convert(self.borrow_mut()?.apply_step(fallback_lr))
    }

    fn zero_grad(&mut self) -> PyResult<()> {
        convert(self.borrow_mut()?.zero_accumulators())
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = convert(self.borrow()?.state_dict())?;
        state_to_pydict(py, state)
    }

    fn load_state_dict(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let state = pydict_to_state(dict)?;
        convert(self.borrow_mut()?.load_state_dict(&state))
    }
}

#[pyclass(module = "spiraltorch.nn", name = "Conv1d")]
struct PyConv1dModule {
    inner: Option<NnConv1d>,
}

impl PyConv1dModule {
    fn borrow(&self) -> PyResult<&NnConv1d> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Conv1d module has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnConv1d> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Conv1d module has been moved"))
    }

    fn take(&mut self) -> PyResult<NnConv1d> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("Conv1d module has been moved"))
    }
}

#[pymethods]
impl PyConv1dModule {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, name=None))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let ident = name.unwrap_or("conv1d");
        let inner = convert(NnConv1d::new(
            ident,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        ))?;
        Ok(Self { inner: Some(inner) })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.forward(input.as_tensor()),
        )?))
    }

    fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow_mut()?
                .backward(input.as_tensor(), grad_output.as_tensor()),
        )?))
    }

    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        convert(
            self.borrow_mut()?
                .attach_hypergrad(curvature, learning_rate),
        )
    }

    fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        convert(self.borrow_mut()?.apply_step(fallback_lr))
    }

    fn zero_grad(&mut self) -> PyResult<()> {
        convert(self.borrow_mut()?.zero_accumulators())
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = convert(self.borrow()?.state_dict())?;
        state_to_pydict(py, state)
    }

    fn load_state_dict(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let state = pydict_to_state(dict)?;
        convert(self.borrow_mut()?.load_state_dict(&state))
    }
}

#[pyclass(module = "spiraltorch.nn", name = "WaveRnn")]
struct PyWaveRnnModule {
    inner: Option<NnWaveRnn>,
}

impl PyWaveRnnModule {
    fn borrow(&self) -> PyResult<&NnWaveRnn> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("WaveRnn module has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnWaveRnn> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("WaveRnn module has been moved"))
    }

    fn take(&mut self) -> PyResult<NnWaveRnn> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("WaveRnn module has been moved"))
    }
}

#[pymethods]
impl PyWaveRnnModule {
    #[new]
    #[pyo3(signature = (in_channels, hidden_dim, kernel_size, stride=1, padding=0, curvature=-1.0, temperature=0.5, name=None))]
    fn new(
        in_channels: usize,
        hidden_dim: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        curvature: f32,
        temperature: f32,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let ident = name.unwrap_or("wave_rnn");
        let inner = convert(NnWaveRnn::new(
            ident,
            in_channels,
            hidden_dim,
            kernel_size,
            stride,
            padding,
            curvature,
            temperature,
        ))?;
        Ok(Self { inner: Some(inner) })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.forward(input.as_tensor()),
        )?))
    }

    fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow_mut()?
                .backward(input.as_tensor(), grad_output.as_tensor()),
        )?))
    }

    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        convert(
            self.borrow_mut()?
                .attach_hypergrad(curvature, learning_rate),
        )
    }

    fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        convert(self.borrow_mut()?.apply_step(fallback_lr))
    }

    fn zero_grad(&mut self) -> PyResult<()> {
        convert(self.borrow_mut()?.zero_accumulators())
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = convert(self.borrow()?.state_dict())?;
        state_to_pydict(py, state)
    }

    fn load_state_dict(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let state = pydict_to_state(dict)?;
        convert(self.borrow_mut()?.load_state_dict(&state))
    }
}

#[pyclass(module = "spiraltorch.nn", name = "Sequential", unsendable)]
struct PySequentialModule {
    inner: Option<NnSequential>,
}

impl PySequentialModule {
    fn borrow(&self) -> PyResult<&NnSequential> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Sequential has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnSequential> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Sequential has been moved"))
    }
}

#[pymethods]
impl PySequentialModule {
    #[new]
    fn new(py_layers: &Bound<'_, PyAny>) -> PyResult<Self> {
        let seq_iter = py_layers.iter()?;
        let mut seq = NnSequential::new();
        for item in seq_iter {
            let obj = item?;
            if let Ok(mut linear) = obj.extract::<PyRefMut<'_, PyLinearModule>>() {
                seq.push(linear.take()?);
            } else if let Ok(mut conv) = obj.extract::<PyRefMut<'_, PyConv1dModule>>() {
                seq.push(conv.take()?);
            } else if let Ok(mut wave) = obj.extract::<PyRefMut<'_, PyWaveRnnModule>>() {
                seq.push(wave.take()?);
            } else {
                return Err(PyValueError::new_err(
                    "Sequential expects Linear, Conv1d, or WaveRnn modules",
                ));
            }
        }
        Ok(Self { inner: Some(seq) })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.forward(input.as_tensor()),
        )?))
    }

    fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow_mut()?
                .backward(input.as_tensor(), grad_output.as_tensor()),
        )?))
    }

    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        convert(
            self.borrow_mut()?
                .attach_hypergrad(curvature, learning_rate),
        )
    }

    fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        convert(self.borrow_mut()?.apply_step(fallback_lr))
    }

    fn zero_grad(&mut self) -> PyResult<()> {
        convert(self.borrow_mut()?.zero_accumulators())
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = convert(self.borrow()?.state_dict())?;
        state_to_pydict(py, state)
    }

    fn load_state_dict(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let state = pydict_to_state(dict)?;
        convert(self.borrow_mut()?.load_state_dict(&state))
    }
}

fn parse_kind(kind: &str) -> PyResult<RankKind> {
    match kind.to_ascii_lowercase().as_str() {
        "topk" | "top" => Ok(RankKind::TopK),
        "midk" | "mid" => Ok(RankKind::MidK),
        "bottomk" | "bottom" => Ok(RankKind::BottomK),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported rank kind: {}",
            other
        ))),
    }
}

fn caps_for(device: Option<&str>) -> DeviceCaps {
    match device.map(|d| d.to_ascii_lowercase()) {
        Some(ref name) if name == "cuda" => DeviceCaps::cuda(32, 1024, Some(96 * 1024)),
        Some(ref name) if name == "hip" => DeviceCaps::hip(32, 1024, Some(64 * 1024)),
        Some(ref name) if name == "cpu" => DeviceCaps::cpu(),
        Some(ref name) if name == "mps" => DeviceCaps::wgpu(32, true, 256),
        Some(ref name) if name == "wgpu" => DeviceCaps::wgpu(32, true, 256),
        Some(ref name) if name == "auto" => DeviceCaps::wgpu(32, true, 256),
        Some(ref name) if name == "hip-real" => DeviceCaps::hip(32, 1024, Some(64 * 1024)),
        _ => DeviceCaps::wgpu(32, true, 256),
    }
}

fn choice_dict<'py>(py: Python<'py>, plan: &RankPlan) -> PyResult<Bound<'py, PyDict>> {
    let choice = PyDict::new_bound(py);
    choice.set_item("use_2ce", plan.choice.use_2ce)?;
    choice.set_item("workgroup", plan.choice.wg)?;
    choice.set_item("kl", plan.choice.kl)?;
    choice.set_item("channel_stride", plan.choice.ch)?;
    choice.set_item("merge_kind", plan.choice.mk)?;
    choice.set_item("merge_detail", plan.choice.mkd)?;
    choice.set_item("tile", plan.choice.tile)?;
    choice.set_item("compaction_tile", plan.choice.ctile)?;
    Ok(choice)
}

/// Inspect the unified heuristics for the requested rank family.
#[pyfunction]
#[pyo3(signature = (kind, rows, cols, k, device=None))]
fn plan(
    py: Python<'_>,
    kind: &str,
    rows: u32,
    cols: u32,
    k: u32,
    device: Option<&str>,
) -> PyResult<PyObject> {
    let rank_kind = parse_kind(kind)?;
    let caps = caps_for(device);
    let plan = plan_rank(rank_kind, rows, cols, k, caps);

    let out = PyDict::new_bound(py);
    out.set_item("kind", kind.to_ascii_lowercase())?;
    out.set_item("rows", rows)?;
    out.set_item("cols", cols)?;
    out.set_item("k", k)?;
    out.set_item("choice", choice_dict(py, &plan)?.into_py(py))?;
    Ok(out.into_py(py))
}

/// Compute the Z-space barycenter described by the weighted KL objective.
#[pyfunction(name = "z_space_barycenter")]
#[pyo3(signature = (densities, weights=None, entropy_weight=0.0, beta_j=0.0, coupling=None))]
fn z_space_barycenter_py(
    py: Python<'_>,
    densities: Vec<PyTensor>,
    weights: Option<Vec<f32>>,
    entropy_weight: f32,
    beta_j: f32,
    coupling: Option<PyTensor>,
) -> PyResult<PyObject> {
    if densities.is_empty() {
        return Err(PyValueError::new_err("densities must not be empty"));
    }
    let tensors: Vec<Tensor> = densities.into_iter().map(PyTensor::into_tensor).collect();
    let weight_vec = weights.unwrap_or_else(|| vec![1.0; tensors.len()]);
    if weight_vec.len() != tensors.len() {
        return Err(PyValueError::new_err(format!(
            "expected {} weights, received {}",
            tensors.len(),
            weight_vec.len()
        )));
    }
    let coupling_tensor = coupling.map(PyTensor::into_tensor);
    let coupling_ref = coupling_tensor.as_ref();
    let barycenter = convert(rust_z_space_barycenter(
        &weight_vec,
        &tensors,
        entropy_weight,
        beta_j,
        coupling_ref,
    ))?;
    PyZSpaceBarycenter::from_result(barycenter).as_dict(py)
}

fn parse_frac_pad(pad: &str) -> PyResult<FracPad> {
    match pad.to_ascii_lowercase().as_str() {
        "zero" => Ok(FracPad::Zero),
        "reflect" => Ok(FracPad::Reflect),
        other => Err(PyValueError::new_err(format!(
            "unsupported padding kind: {other}; expected 'zero' or 'reflect'"
        ))),
    }
}

#[pyfunction(name = "gl_coeffs")]
fn gl_coeffs_py(alpha: f32, len: usize) -> Vec<f32> {
    frac_gl_coeffs(alpha, len)
}

#[pyfunction(name = "fracdiff_gl")]
#[pyo3(signature = (tensor, alpha, axis, kernel_len, pad="zero", scale=None))]
fn fracdiff_gl_py(
    tensor: &PyTensor,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: &str,
    scale: Option<f32>,
) -> PyResult<PyTensor> {
    let array = tensor_to_array(tensor.as_tensor())?;
    let pad = parse_frac_pad(pad)?;
    let result = convert_frac(fracdiff_gl_nd(&array, alpha, axis, kernel_len, pad, scale))?;
    array_to_tensor(result)
}

#[pyfunction(name = "fracdiff_gl_backward")]
#[pyo3(signature = (tensor, alpha, axis, kernel_len, pad="zero", scale=None))]
fn fracdiff_gl_backward_py(
    tensor: &PyTensor,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: &str,
    scale: Option<f32>,
) -> PyResult<PyTensor> {
    let array = tensor_to_array(tensor.as_tensor())?;
    let pad = parse_frac_pad(pad)?;
    let result = convert_frac(fracdiff_gl_nd_backward(
        &array, alpha, axis, kernel_len, pad, scale,
    ))?;
    array_to_tensor(result)
}

#[pyfunction(name = "fft")]
#[pyo3(signature = (signal, inverse=false))]
fn frac_fft_py(signal: Vec<Complex64>, inverse: bool) -> PyResult<Vec<Complex64>> {
    let mut buffer: Vec<FracComplex32> = signal
        .into_iter()
        .map(|c| FracComplex32::new(c.re as f32, c.im as f32))
        .collect();
    convert_fft(frac_fft_inplace(&mut buffer, inverse))?;
    Ok(buffer
        .into_iter()
        .map(|c| Complex64::new(c.re as f64, c.im as f64))
        .collect())
}

#[pymodule]
fn nn(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinearModule>()?;
    m.add_class::<PyConv1dModule>()?;
    m.add_class::<PyWaveRnnModule>()?;
    m.add_class::<PySequentialModule>()?;
    m.setattr("__all__", vec!["Linear", "Conv1d", "WaveRnn", "Sequential"])?;
    m.setattr(
        "__doc__",
        "Rust-backed neural network modules: Linear, Conv1d, WaveRnn, Sequential.",
    )?;
    Ok(())
}

#[pymodule]
fn frac(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gl_coeffs_py, m)?)?;
    m.add_function(wrap_pyfunction!(fracdiff_gl_py, m)?)?;
    m.add_function(wrap_pyfunction!(fracdiff_gl_backward_py, m)?)?;
    m.add_function(wrap_pyfunction!(frac_fft_py, m)?)?;
    m.setattr(
        "__all__",
        vec!["gl_coeffs", "fracdiff_gl", "fracdiff_gl_backward", "fft"],
    )?;
    m.setattr(
        "__doc__",
        "Fractional calculus operators and FFT helpers used by SpiralTorch.",
    )?;
    Ok(())
}

/// Convenience helper for the TopK family.
#[pyfunction]
#[pyo3(signature = (rows, cols, k, device=None))]
fn plan_topk(
    py: Python<'_>,
    rows: u32,
    cols: u32,
    k: u32,
    device: Option<&str>,
) -> PyResult<PyObject> {
    plan(py, "topk", rows, cols, k, device)
}

/// Surface ROCm probing hints for Python callers.
#[pyfunction]
fn hip_probe(py: Python<'_>) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("available", hip_runtime_available())?;

    let devices = PyList::empty_bound(py);
    for info in hip_device_info() {
        devices.append(py_device_info(py, info)?.into_py(py))?;
    }
    out.set_item("devices", devices.into_py(py))?;

    Ok(out.into_py(py))
}

/// Return a basic capability template for the given device string.
#[pyfunction]
#[pyo3(signature = (device=None))]
fn describe_device(py: Python<'_>, device: Option<&str>) -> PyResult<PyObject> {
    let caps = caps_for(device);
    Ok(device_caps_dict(py, caps)?.into_py(py))
}

/// SpiralTorch Python module.
#[pymodule]
fn spiraltorch(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let nn_mod = PyModule::new_bound(_py, "nn")?;
    nn(_py, &nn_mod)?;
    m.add_submodule(&nn_mod)?;
    let frac_mod = PyModule::new_bound(_py, "frac")?;
    frac(_py, &frac_mod)?;
    m.add_submodule(&frac_mod)?;
    let sot_mod = PyModule::new_bound(_py, "sot")?;
    sot::module(_py, &sot_mod)?;
    m.add_submodule(&sot_mod)?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(plan_topk, m)?)?;
    m.add_function(wrap_pyfunction!(z_space_barycenter_py, m)?)?;
    m.add_function(wrap_pyfunction!(hip_probe, m)?)?;
    m.add_function(wrap_pyfunction!(describe_device, m)?)?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyComplexTensor>()?;
    m.add_class::<PyBarycenterIntermediate>()?;
    m.add_class::<PyZSpaceBarycenter>()?;
    m.add_class::<PyDifferentialResonance>()?;
    m.add_class::<PySpiralDifferentialTrace>()?;
    m.add_class::<PyOpenTopos>()?;
    m.add_class::<PyLanguageWaveEncoder>()?;
    m.add_class::<PyHypergrad>()?;
    m.add_class::<PySpiralSessionBuilder>()?;
    m.add_class::<PySpiralSession>()?;

    m.setattr(
        "__all__",
        vec![
            "plan",
            "plan_topk",
            "z_space_barycenter",
            "hip_probe",
            "describe_device",
            "Tensor",
            "ComplexTensor",
            "BarycenterIntermediate",
            "ZSpaceBarycenter",
            "DifferentialResonance",
            "SpiralDifferentialTrace",
            "OpenTopos",
            "LanguageWaveEncoder",
            "Hypergrad",
            "SpiralSessionBuilder",
            "SpiralSession",
            "nn",
            "frac",
            "sot",
        ],
    )?;
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;

    // Provide a tiny doc string that highlights the zero-shim approach.
    m.setattr(
        "__doc__",
        "Rust-first training primitives for SpiralTorch: tensors, hypergrads, and unified planners.",
    )?;

    Ok(())
}
