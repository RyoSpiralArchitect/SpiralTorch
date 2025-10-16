// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

mod sot;

use crate::sot::{PySoT3DPlan, Sot3DParams};

use ndarray::{Array2, ArrayD, Ix2};
use num_complex::Complex64;
use pyo3::exceptions::{PyImportError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule, PyTuple};
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use pyo3::PyRef;
use pyo3::PyRefMut;
use st_backend_hip::{
    device_info as hip_device_info, hip_available as hip_runtime_available,
    DeviceInfo as HipDeviceInfo,
};
use st_core::backend::device_caps::{BackendKind, DeviceCaps};
use st_core::backend::unison_heuristics::RankKind;
#[cfg(feature = "collapse")]
use st_core::engine::collapse_drive::DriveCmd;
use st_core::ops::rank_entry::{plan_rank, RankPlan};
use st_core::telemetry::atlas::{AtlasDistrict, AtlasFrame, AtlasMetric, AtlasRoute};
use st_core::telemetry::chrono::{
    ChronoFrame, ChronoHarmonics, ChronoLoopSignal, ChronoPeak, ChronoSummary,
};
use st_core::telemetry::hub;
use st_core::telemetry::maintainer::MaintainerReport;
use st_frac::fft::{fft_inplace as frac_fft_inplace, Complex32 as FracComplex32, FftError};
use st_frac::{
    fracdiff_gl_nd, fracdiff_gl_nd_backward, gl_coeffs as frac_gl_coeffs, FracErr, Pad as FracPad,
};
use st_nn::dataset::DataLoaderBatches as NnDataLoaderBatches;
use st_nn::dataset_from_vec as nn_dataset_from_vec;
use st_nn::{
    Conv1d as NnConv1d, DataLoader as NnDataLoader, DifferentialTrace, DistConfig, DistMode,
    EpochStats, LightningConfig as NnLightningConfig, Linear as NnLinear, Loss, MeanSquaredError,
    Module, ModuleTrainer, Relu as NnRelu, RoundtableConfig, RoundtableSchedule,
    Sequential as NnSequential, SpiralLightning as NnSpiralLightning, SpiralSession,
    SpiralSessionBuilder, WaveRnn as NnWaveRnn, ZSpaceProjector as NnZSpaceProjector,
};
use st_rec::{RatingTriple as RecRatingTriple, RecEpochReport, SpiralRecError, SpiralRecommender};
use st_rl::{EpisodeReport as RlEpisodeReport, SpiralPolicyGradient, SpiralRlError};
use st_tensor::backend::faer_dense;
#[cfg(feature = "wgpu")]
use st_tensor::backend::wgpu_dense as tensor_wgpu_dense;
use st_tensor::pure::{
    measure::{
        z_space_barycenter as rust_z_space_barycenter, BarycenterIntermediate, ZSpaceBarycenter,
    },
    topos::OpenCartesianTopos,
    AmegaHypergrad, Complex32, ComplexTensor, DifferentialResonance, LanguageWaveEncoder,
    MatmulBackend, PureResult, Tensor, TensorBiome, TensorError,
};
use st_text::{
    describe_atlas as text_describe_atlas, describe_frame as text_describe_frame,
    describe_resonance as text_describe_resonance, describe_timeline as text_describe_timeline,
    TextResonator,
};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::Path;
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceRoute {
    Wgpu,
    Cuda,
    Mps,
    Cpu,
}

impl DeviceRoute {
    fn as_str(self) -> &'static str {
        match self {
            DeviceRoute::Wgpu => "wgpu",
            DeviceRoute::Cuda => "cuda",
            DeviceRoute::Mps => "mps",
            DeviceRoute::Cpu => "cpu",
        }
    }
}

fn device_available(route: DeviceRoute) -> bool {
    match route {
        DeviceRoute::Wgpu => cfg!(feature = "wgpu-rt"),
        DeviceRoute::Cuda => cfg!(feature = "cuda"),
        DeviceRoute::Mps => cfg!(feature = "mps"),
        DeviceRoute::Cpu => true,
    }
}

fn choose_route(device: Option<&str>) -> PyResult<DeviceRoute> {
    let hint = device.unwrap_or("auto").to_ascii_lowercase();
    match hint.as_str() {
        "auto" => {
            for candidate in [
                DeviceRoute::Wgpu,
                DeviceRoute::Cuda,
                DeviceRoute::Mps,
                DeviceRoute::Cpu,
            ] {
                if device_available(candidate) {
                    return Ok(candidate);
                }
            }
            Ok(DeviceRoute::Cpu)
        }
        "wgpu" => Ok(if device_available(DeviceRoute::Wgpu) {
            DeviceRoute::Wgpu
        } else {
            DeviceRoute::Cpu
        }),
        "cuda" => Ok(if device_available(DeviceRoute::Cuda) {
            DeviceRoute::Cuda
        } else {
            DeviceRoute::Cpu
        }),
        "mps" => Ok(if device_available(DeviceRoute::Mps) {
            DeviceRoute::Mps
        } else {
            DeviceRoute::Cpu
        }),
        "cpu" => Ok(DeviceRoute::Cpu),
        other => Err(PyValueError::new_err(format!(
            "unsupported device hint: {other}"
        ))),
    }
}

fn topk_rows_cpu(
    data: &[f32],
    rows: usize,
    cols: usize,
    k: usize,
    largest: bool,
) -> (Vec<f32>, Vec<i32>) {
    use std::cmp::Ordering;

    let mut out_vals = vec![0.0f32; rows * k];
    let mut out_idx = vec![0i32; rows * k];
    let mut idx_buf: Vec<usize> = (0..cols).collect();

    for r in 0..rows {
        let row_offset = r * cols;
        for (slot, value) in idx_buf.iter_mut().enumerate() {
            *value = slot;
        }

        if k < cols {
            idx_buf.select_nth_unstable_by(k, |&a, &b| {
                let va = data[row_offset + a];
                let vb = data[row_offset + b];
                let ordering = va.partial_cmp(&vb).unwrap_or(Ordering::Equal);
                if largest {
                    ordering.reverse()
                } else {
                    ordering
                }
            });
        }

        let mut topk = idx_buf[..k].to_vec();
        topk.sort_unstable_by(|&a, &b| {
            let va = data[row_offset + a];
            let vb = data[row_offset + b];
            let ordering = va.partial_cmp(&vb).unwrap_or(Ordering::Equal);
            if largest {
                ordering.reverse()
            } else {
                ordering
            }
        });

        for (j, &col) in topk.iter().enumerate() {
            let target = r * k + j;
            out_vals[target] = data[row_offset + col];
            out_idx[target] = col as i32;
        }
    }

    (out_vals, out_idx)
}

fn tensor_err(err: TensorError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn convert<T>(value: PureResult<T>) -> PyResult<T> {
    value.map_err(tensor_err)
}

fn rl_err(err: SpiralRlError) -> PyErr {
    match err {
        SpiralRlError::Tensor(err) => tensor_err(err),
        SpiralRlError::EmptyEpisode => PyValueError::new_err(
            "no transitions recorded — call record_transition before finishing the episode",
        ),
        SpiralRlError::InvalidStateShape {
            expected,
            rows,
            cols,
        } => PyValueError::new_err(format!(
            "state tensor must be shaped as (1, {expected}) but received ({rows}, {cols})"
        )),
        SpiralRlError::InvalidAction { action, actions } => PyValueError::new_err(format!(
            "action index {action} exceeds configured action space {actions}"
        )),
        SpiralRlError::InvalidDiscount { discount } => PyValueError::new_err(format!(
            "discount factor must lie within [0, 1]; received {discount}"
        )),
    }
}

fn rec_err(err: SpiralRecError) -> PyErr {
    match err {
        SpiralRecError::Tensor(err) => tensor_err(err),
        SpiralRecError::OutOfBoundsRating { user, item } => PyValueError::new_err(format!(
            "rating ({user}, {item}) falls outside the configured recommender bounds"
        )),
        SpiralRecError::EmptyBatch => {
            PyValueError::new_err("at least one rating triple is required to train an epoch")
        }
    }
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

#[pyfunction]
fn describe_resonance(resonance: &PyDifferentialResonance) -> PyResult<String> {
    convert(text_describe_resonance(&resonance.inner))
}

#[pyfunction]
fn describe_frame(frame: &PyChronoFrame) -> String {
    text_describe_frame(frame.as_frame())
}

#[pyfunction]
fn describe_timeline(frames: Vec<PyChronoFrame>) -> PyResult<String> {
    let inner: Vec<ChronoFrame> = frames.into_iter().map(PyChronoFrame::into_frame).collect();
    convert(text_describe_timeline(&inner))
}

#[pyfunction]
fn describe_atlas(atlas: &PyAtlasFrame) -> PyResult<String> {
    convert(text_describe_atlas(&atlas.frame))
}

#[allow(clippy::too_many_arguments)]
fn build_roundtable_config(
    top_k: u32,
    mid_k: u32,
    bottom_k: u32,
    here_tolerance: f32,
    psychoid: bool,
    psychoid_log: bool,
    psi: bool,
    collapse: bool,
) -> RoundtableConfig {
    #[cfg(not(feature = "psychoid"))]
    {
        let _ = psychoid;
        let _ = psychoid_log;
    }
    #[cfg(not(feature = "psi"))]
    {
        let _ = psi;
    }
    #[cfg(not(feature = "collapse"))]
    {
        let _ = collapse;
    }

    let mut config = RoundtableConfig {
        top_k,
        mid_k,
        bottom_k,
        here_tolerance: here_tolerance.max(0.0),
        ..RoundtableConfig::default()
    };

    #[cfg(feature = "psychoid")]
    {
        if psychoid {
            config = if psychoid_log {
                config.enable_psychoid_with_log()
            } else {
                config.enable_psychoid()
            };
        }
    }

    #[cfg(feature = "psi")]
    {
        if psi {
            config = config.enable_psi();
        }
    }

    #[cfg(feature = "collapse")]
    {
        if collapse {
            config = config.enable_collapse();
        }
    }

    config
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

fn parse_matmul_backend(label: Option<&str>) -> PyResult<MatmulBackend> {
    match label {
        None => Ok(MatmulBackend::Auto),
        Some(value) => {
            let normalized = value.to_ascii_lowercase();
            match normalized.as_str() {
                "auto" => Ok(MatmulBackend::Auto),
                "faer" | "simd" => Ok(MatmulBackend::CpuFaer),
                "cpu" | "naive" => Ok(MatmulBackend::CpuNaive),
                #[cfg(feature = "wgpu")]
                "wgpu" | "gpu" => Ok(MatmulBackend::GpuWgpu),
                #[cfg(not(feature = "wgpu"))]
                "wgpu" | "gpu" => Err(PyValueError::new_err(
                    "WGPU backend requested but SpiralTorch was built without the `wgpu` feature",
                )),
                other => Err(PyValueError::new_err(format!(
                    "unknown matmul backend `{other}`; expected auto, faer, naive{}",
                    if cfg!(feature = "wgpu") {
                        ", or wgpu"
                    } else {
                        ""
                    }
                ))),
            }
        }
    }
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

    #[pyo3(signature = (other, backend=None))]
    fn matmul(&self, other: &PyTensor, backend: Option<&str>) -> PyResult<Self> {
        let backend = parse_matmul_backend(backend)?;
        Ok(Self::from_tensor(convert(
            self.inner.matmul_with_backend(other.as_tensor(), backend),
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

#[pyclass(module = "spiraltorch.dataset", name = "DataLoader", unsendable)]
#[derive(Clone)]
struct PyDataLoader {
    inner: NnDataLoader,
}

impl PyDataLoader {
    fn from_loader(inner: NnDataLoader) -> Self {
        Self { inner }
    }

    fn clone_inner(&self) -> NnDataLoader {
        self.inner.clone()
    }
}

#[pymethods]
impl PyDataLoader {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    #[getter]
    fn prefetch_depth(&self) -> usize {
        self.inner.prefetch_depth()
    }

    fn shuffle(&self, seed: u64) -> Self {
        Self::from_loader(self.inner.clone().shuffle(seed))
    }

    fn batched(&self, batch_size: usize) -> Self {
        Self::from_loader(self.inner.clone().batched(batch_size))
    }

    fn prefetch(&self, depth: usize) -> Self {
        Self::from_loader(self.inner.clone().prefetch(depth))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyDataLoaderIter>> {
        Py::new(
            slf.py(),
            PyDataLoaderIter {
                inner: Some(slf.clone_inner().into_iter()),
            },
        )
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "DataLoader(len={}, batch_size={})",
            self.inner.len(),
            self.inner.batch_size()
        ))
    }
}

#[pyclass(module = "spiraltorch.dataset", name = "DataLoaderIter", unsendable)]
struct PyDataLoaderIter {
    inner: Option<NnDataLoaderBatches>,
}

#[pymethods]
impl PyDataLoaderIter {
    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<(PyTensor, PyTensor)>> {
        if let Some(iter) = self.inner.as_mut() {
            match iter.next() {
                Some(Ok((input, target))) => {
                    return Ok(Some((
                        PyTensor::from_tensor(input),
                        PyTensor::from_tensor(target),
                    )));
                }
                Some(Err(err)) => return Err(tensor_err(err)),
                None => {
                    self.inner = None;
                    return Ok(None);
                }
            }
        }
        Ok(None)
    }
}

#[pyfunction(name = "from_vec")]
fn dataset_from_vec_py(samples: Vec<(PyTensor, PyTensor)>) -> PyResult<PyDataLoader> {
    let owned: Vec<(Tensor, Tensor)> = samples
        .into_iter()
        .map(|(input, target)| (input.into_tensor(), target.into_tensor()))
        .collect();
    Ok(PyDataLoader::from_loader(nn_dataset_from_vec(owned)))
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

#[pyclass(module = "spiraltorch", name = "ChronoFrame")]
#[derive(Clone)]
struct PyChronoFrame {
    frame: ChronoFrame,
}

impl PyChronoFrame {
    fn from_frame(frame: ChronoFrame) -> Self {
        Self { frame }
    }

    fn as_frame(&self) -> &ChronoFrame {
        &self.frame
    }

    fn into_frame(self) -> ChronoFrame {
        self.frame
    }
}

#[pyclass(module = "spiraltorch", name = "ChronoSummary")]
struct PyChronoSummary {
    summary: ChronoSummary,
}

impl PyChronoSummary {
    fn from_summary(summary: ChronoSummary) -> Self {
        Self { summary }
    }
}

#[pyclass(module = "spiraltorch", name = "AtlasMetric")]
#[derive(Clone)]
struct PyAtlasMetric {
    metric: AtlasMetric,
}

impl PyAtlasMetric {
    fn from_metric(metric: AtlasMetric) -> Self {
        Self { metric }
    }
}

#[pymethods]
impl PyAtlasMetric {
    #[getter]
    fn name(&self) -> &str {
        &self.metric.name
    }

    #[getter]
    fn value(&self) -> f32 {
        self.metric.value
    }

    #[getter]
    fn district(&self) -> Option<String> {
        self.metric.district().map(|name| name.to_string())
    }

    fn as_tuple(&self) -> (String, f32) {
        (self.metric.name.clone(), self.metric.value)
    }

    fn __repr__(&self) -> PyResult<String> {
        if let Some(district) = self.metric.district() {
            Ok(format!(
                "AtlasMetric(name={}, value={:.3}, district={})",
                self.metric.name, self.metric.value, district
            ))
        } else {
            Ok(format!(
                "AtlasMetric(name={}, value={:.3})",
                self.metric.name, self.metric.value
            ))
        }
    }
}

#[pyclass(module = "spiraltorch", name = "AtlasDistrict")]
#[derive(Clone)]
struct PyAtlasDistrict {
    district: AtlasDistrict,
}

impl PyAtlasDistrict {
    fn from_district(district: AtlasDistrict) -> Self {
        Self { district }
    }
}

#[pymethods]
impl PyAtlasDistrict {
    #[getter]
    fn name(&self) -> &str {
        &self.district.name
    }

    #[getter]
    fn mean(&self) -> f32 {
        self.district.mean
    }

    #[getter]
    fn span(&self) -> f32 {
        self.district.span
    }

    fn metrics(&self) -> Vec<PyAtlasMetric> {
        self.district
            .metrics
            .iter()
            .cloned()
            .map(PyAtlasMetric::from_metric)
            .collect()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "AtlasDistrict(name={}, mean={:.3}, span={:.3}, metrics={})",
            self.district.name,
            self.district.mean,
            self.district.span,
            self.district.metrics.len()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "AtlasFrame")]
#[derive(Clone)]
struct PyAtlasFrame {
    frame: AtlasFrame,
}

impl PyAtlasFrame {
    fn from_frame(frame: AtlasFrame) -> Self {
        Self { frame }
    }
}

#[pymethods]
impl PyAtlasFrame {
    #[getter]
    fn timestamp(&self) -> f32 {
        self.frame.timestamp
    }

    #[getter]
    fn loop_support(&self) -> f32 {
        self.frame.loop_support
    }

    #[getter]
    fn collapse_total(&self) -> Option<f32> {
        self.frame.collapse_total
    }

    #[getter]
    fn z_signal(&self) -> Option<f32> {
        self.frame.z_signal
    }

    #[getter]
    fn script_hint(&self) -> Option<String> {
        self.frame.script_hint.clone()
    }

    #[getter]
    fn maintainer_status(&self) -> Option<String> {
        self.frame
            .maintainer_status
            .map(|status| status.as_str().to_string())
    }

    #[getter]
    fn maintainer_diagnostic(&self) -> Option<String> {
        self.frame.maintainer_diagnostic.clone()
    }

    #[getter]
    fn suggested_max_scale(&self) -> Option<f32> {
        self.frame.suggested_max_scale
    }

    #[getter]
    fn suggested_pressure(&self) -> Option<f32> {
        self.frame.suggested_pressure
    }

    fn metrics(&self) -> Vec<PyAtlasMetric> {
        self.frame
            .metrics
            .iter()
            .cloned()
            .map(PyAtlasMetric::from_metric)
            .collect()
    }

    fn districts(&self) -> Vec<PyAtlasDistrict> {
        self.frame
            .districts()
            .into_iter()
            .map(PyAtlasDistrict::from_district)
            .collect()
    }

    fn notes(&self) -> Vec<String> {
        self.frame.notes.clone()
    }

    fn summary(&self) -> Option<PyChronoSummary> {
        self.frame
            .chrono_summary
            .clone()
            .map(PyChronoSummary::from_summary)
    }

    fn harmonics(&self) -> Option<PyChronoHarmonics> {
        self.frame
            .harmonics
            .clone()
            .map(PyChronoHarmonics::from_harmonics)
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("timestamp", self.frame.timestamp)?;
        dict.set_item("loop_support", self.frame.loop_support)?;
        dict.set_item("collapse_total", self.frame.collapse_total)?;
        dict.set_item("z_signal", self.frame.z_signal)?;
        dict.set_item("script_hint", self.frame.script_hint.clone())?;
        if let Some(status) = self.frame.maintainer_status {
            dict.set_item("maintainer_status", status.as_str())?;
        }
        dict.set_item(
            "maintainer_diagnostic",
            self.frame.maintainer_diagnostic.clone(),
        )?;
        dict.set_item("suggested_max_scale", self.frame.suggested_max_scale)?;
        dict.set_item("suggested_pressure", self.frame.suggested_pressure)?;
        if let Some(summary) = &self.frame.chrono_summary {
            dict.set_item(
                "summary",
                PyChronoSummary::from_summary(summary.clone()).into_py(py),
            )?;
        }
        if let Some(harmonics) = &self.frame.harmonics {
            dict.set_item(
                "harmonics",
                PyChronoHarmonics::from_harmonics(harmonics.clone()).into_py(py),
            )?;
        }
        let metrics: Vec<(String, f32)> = self
            .frame
            .metrics
            .iter()
            .map(|metric| (metric.name.clone(), metric.value))
            .collect();
        dict.set_item("metrics", metrics)?;
        dict.set_item("notes", self.frame.notes.clone())?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "AtlasFrame(t={:.3}, metrics={})",
            self.frame.timestamp,
            self.frame.metrics.len()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "AtlasRoute")]
#[derive(Clone)]
struct PyAtlasRoute {
    route: AtlasRoute,
}

impl PyAtlasRoute {
    fn from_route(route: AtlasRoute) -> Self {
        Self { route }
    }
}

#[pymethods]
impl PyAtlasRoute {
    #[getter]
    fn frames(&self) -> Vec<PyAtlasFrame> {
        self.route
            .frames
            .iter()
            .cloned()
            .map(PyAtlasFrame::from_frame)
            .collect()
    }

    #[getter]
    fn length(&self) -> usize {
        self.route.len()
    }

    fn is_empty(&self) -> bool {
        self.route.is_empty()
    }

    fn latest(&self) -> Option<PyAtlasFrame> {
        self.route.latest().cloned().map(PyAtlasFrame::from_frame)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "AtlasRoute(length={}, latest_ts={:.3})",
            self.route.len(),
            self.route
                .latest()
                .map(|frame| frame.timestamp)
                .unwrap_or(0.0)
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "ChronoPeak")]
#[derive(Clone)]
struct PyChronoPeak {
    peak: ChronoPeak,
}

impl PyChronoPeak {
    fn from_peak(peak: ChronoPeak) -> Self {
        Self { peak }
    }
}

#[pymethods]
impl PyChronoPeak {
    #[getter]
    fn frequency(&self) -> f32 {
        self.peak.frequency
    }

    #[getter]
    fn magnitude(&self) -> f32 {
        self.peak.magnitude
    }

    #[getter]
    fn phase(&self) -> f32 {
        self.peak.phase
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("frequency", self.peak.frequency)?;
        dict.set_item("magnitude", self.peak.magnitude)?;
        dict.set_item("phase", self.peak.phase)?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ChronoPeak(freq={:.3}, magnitude={:.3})",
            self.peak.frequency, self.peak.magnitude
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "ChronoHarmonics")]
struct PyChronoHarmonics {
    harmonics: ChronoHarmonics,
}

impl PyChronoHarmonics {
    fn from_harmonics(harmonics: ChronoHarmonics) -> Self {
        Self { harmonics }
    }
}

#[pymethods]
impl PyChronoHarmonics {
    #[getter]
    fn frames(&self) -> usize {
        self.harmonics.frames
    }

    #[getter]
    fn duration(&self) -> f32 {
        self.harmonics.duration
    }

    #[getter]
    fn sample_rate(&self) -> f32 {
        self.harmonics.sample_rate
    }

    #[getter]
    fn nyquist(&self) -> f32 {
        self.harmonics.nyquist
    }

    #[getter]
    fn drift_power(&self) -> Vec<f32> {
        self.harmonics.drift_power.clone()
    }

    #[getter]
    fn energy_power(&self) -> Vec<f32> {
        self.harmonics.energy_power.clone()
    }

    #[getter]
    fn dominant_drift(&self) -> Option<PyChronoPeak> {
        self.harmonics
            .dominant_drift
            .clone()
            .map(PyChronoPeak::from_peak)
    }

    #[getter]
    fn dominant_energy(&self) -> Option<PyChronoPeak> {
        self.harmonics
            .dominant_energy
            .clone()
            .map(PyChronoPeak::from_peak)
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("frames", self.harmonics.frames)?;
        dict.set_item("duration", self.harmonics.duration)?;
        dict.set_item("sample_rate", self.harmonics.sample_rate)?;
        dict.set_item("nyquist", self.harmonics.nyquist)?;
        dict.set_item("drift_power", self.harmonics.drift_power.clone())?;
        dict.set_item("energy_power", self.harmonics.energy_power.clone())?;
        let drift = if let Some(peak) = self.harmonics.dominant_drift.clone() {
            PyChronoPeak::from_peak(peak).as_dict(py)?
        } else {
            py.None()
        };
        dict.set_item("dominant_drift", drift)?;
        let energy = if let Some(peak) = self.harmonics.dominant_energy.clone() {
            PyChronoPeak::from_peak(peak).as_dict(py)?
        } else {
            py.None()
        };
        dict.set_item("dominant_energy", energy)?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ChronoHarmonics(frames={}, sample_rate={:.3})",
            self.harmonics.frames, self.harmonics.sample_rate
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "ChronoLoopSignal")]
#[derive(Clone)]
struct PyChronoLoopSignal {
    signal: ChronoLoopSignal,
}

impl PyChronoLoopSignal {
    fn from_signal(signal: ChronoLoopSignal) -> Self {
        Self { signal }
    }
}

#[pymethods]
impl PyChronoLoopSignal {
    #[getter]
    fn summary(&self) -> PyChronoSummary {
        PyChronoSummary::from_summary(self.signal.summary.clone())
    }

    #[getter]
    fn harmonics(&self) -> Option<PyChronoHarmonics> {
        self.signal
            .harmonics
            .clone()
            .map(PyChronoHarmonics::from_harmonics)
    }

    #[cfg(feature = "kdsl")]
    #[getter]
    fn spiralk_script(&self) -> Option<String> {
        self.signal.spiralk_script.clone()
    }

    #[cfg(not(feature = "kdsl"))]
    #[getter]
    fn spiralk_script(&self) -> Option<String> {
        None
    }

    #[cfg(feature = "kdsl")]
    #[getter]
    fn spiralk_hints(&self) -> Vec<String> {
        self.signal
            .spiralk_hints
            .iter()
            .map(|hint| {
                format!(
                    "soft({},{},{},{})",
                    hint.field, hint.value_expr, hint.weight_expr, hint.condition_expr
                )
            })
            .collect()
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "summary",
            PyChronoSummary::from_summary(self.signal.summary.clone()).into_py(py),
        )?;
        if let Some(harmonics) = self.signal.harmonics.clone() {
            dict.set_item(
                "harmonics",
                PyChronoHarmonics::from_harmonics(harmonics).into_py(py),
            )?;
        } else {
            dict.set_item("harmonics", py.None())?;
        }
        #[cfg(feature = "kdsl")]
        {
            dict.set_item("spiralk_script", self.signal.spiralk_script.clone())?;
            dict.set_item("spiralk_hints", self.spiralk_hints())?;
        }
        #[cfg(not(feature = "kdsl"))]
        {
            dict.set_item("spiralk_script", py.None())?;
        }
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ChronoLoopSignal(frames={}, energy_mean={:.3})",
            self.signal.summary.frames, self.signal.summary.mean_energy
        ))
    }
}

#[pymethods]
impl PyChronoSummary {
    #[getter]
    fn frames(&self) -> usize {
        self.summary.frames
    }

    #[getter]
    fn duration(&self) -> f32 {
        self.summary.duration
    }

    #[getter]
    fn latest_timestamp(&self) -> f32 {
        self.summary.latest_timestamp
    }

    #[getter]
    fn mean_drift(&self) -> f32 {
        self.summary.mean_drift
    }

    #[getter]
    fn mean_abs_drift(&self) -> f32 {
        self.summary.mean_abs_drift
    }

    #[getter]
    fn drift_std(&self) -> f32 {
        self.summary.drift_std
    }

    #[getter]
    fn mean_energy(&self) -> f32 {
        self.summary.mean_energy
    }

    #[getter]
    fn energy_std(&self) -> f32 {
        self.summary.energy_std
    }

    #[getter]
    fn mean_decay(&self) -> f32 {
        self.summary.mean_decay
    }

    #[getter]
    fn min_energy(&self) -> f32 {
        self.summary.min_energy
    }

    #[getter]
    fn max_energy(&self) -> f32 {
        self.summary.max_energy
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ChronoSummary(frames={}, duration={:.3}, energy={:.3}±{:.3})",
            self.summary.frames,
            self.summary.duration,
            self.summary.mean_energy,
            self.summary.energy_std
        ))
    }
}

#[pymethods]
impl PyChronoFrame {
    #[getter]
    fn step(&self) -> u64 {
        self.frame.step
    }

    #[getter]
    fn timestamp(&self) -> f32 {
        self.frame.timestamp
    }

    #[getter]
    fn dt(&self) -> f32 {
        self.frame.dt
    }

    #[getter]
    fn observed_curvature(&self) -> f32 {
        self.frame.observed_curvature
    }

    #[getter]
    fn curvature_drift(&self) -> f32 {
        self.frame.curvature_drift
    }

    #[getter]
    fn total_energy(&self) -> f32 {
        self.frame.total_energy
    }

    #[getter]
    fn energy_decay(&self) -> f32 {
        self.frame.energy_decay
    }

    #[getter]
    fn homotopy_energy(&self) -> f32 {
        self.frame.homotopy_energy
    }

    #[getter]
    fn functor_energy(&self) -> f32 {
        self.frame.functor_energy
    }

    #[getter]
    fn recursive_energy(&self) -> f32 {
        self.frame.recursive_energy
    }

    #[getter]
    fn projection_energy(&self) -> f32 {
        self.frame.projection_energy
    }

    #[getter]
    fn infinity_energy(&self) -> f32 {
        self.frame.infinity_energy
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("step", self.frame.step)?;
        dict.set_item("timestamp", self.frame.timestamp)?;
        dict.set_item("dt", self.frame.dt)?;
        dict.set_item("observed_curvature", self.frame.observed_curvature)?;
        dict.set_item("curvature_drift", self.frame.curvature_drift)?;
        dict.set_item("total_energy", self.frame.total_energy)?;
        dict.set_item("energy_decay", self.frame.energy_decay)?;
        dict.set_item("homotopy_energy", self.frame.homotopy_energy)?;
        dict.set_item("functor_energy", self.frame.functor_energy)?;
        dict.set_item("recursive_energy", self.frame.recursive_energy)?;
        dict.set_item("projection_energy", self.frame.projection_energy)?;
        dict.set_item("infinity_energy", self.frame.infinity_energy)?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ChronoFrame(step={}, t={:.3}, energy={:.3})",
            self.frame.step, self.frame.timestamp, self.frame.total_energy
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "MaintainerReport")]
#[derive(Clone)]
struct PyMaintainerReport {
    report: MaintainerReport,
}

impl PyMaintainerReport {
    fn from_report(report: MaintainerReport) -> Self {
        Self { report }
    }
}

#[pymethods]
impl PyMaintainerReport {
    #[getter]
    fn status(&self) -> &'static str {
        self.report.status.as_str()
    }

    #[getter]
    fn average_drift(&self) -> f32 {
        self.report.average_drift
    }

    #[getter]
    fn mean_energy(&self) -> f32 {
        self.report.mean_energy
    }

    #[getter]
    fn mean_decay(&self) -> f32 {
        self.report.mean_decay
    }

    #[getter]
    fn drift_peak(&self) -> Option<PyChronoPeak> {
        self.report.drift_peak.clone().map(PyChronoPeak::from_peak)
    }

    #[getter]
    fn energy_peak(&self) -> Option<PyChronoPeak> {
        self.report.energy_peak.clone().map(PyChronoPeak::from_peak)
    }

    #[getter]
    fn suggested_max_scale(&self) -> Option<f32> {
        self.report.suggested_max_scale
    }

    #[getter]
    fn suggested_pressure(&self) -> Option<f32> {
        self.report.suggested_pressure
    }

    #[getter]
    fn diagnostic(&self) -> &str {
        &self.report.diagnostic
    }

    #[cfg(feature = "kdsl")]
    #[getter]
    fn spiralk_script(&self) -> Option<String> {
        self.report.spiralk_script.clone()
    }

    fn should_rewrite(&self) -> bool {
        self.report.should_rewrite()
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("status", self.report.status.as_str())?;
        dict.set_item("average_drift", self.report.average_drift)?;
        dict.set_item("mean_energy", self.report.mean_energy)?;
        dict.set_item("mean_decay", self.report.mean_decay)?;
        let drift_peak = if let Some(peak) = self.report.drift_peak.clone() {
            PyChronoPeak::from_peak(peak).as_dict(py)?
        } else {
            py.None()
        };
        let energy_peak = if let Some(peak) = self.report.energy_peak.clone() {
            PyChronoPeak::from_peak(peak).as_dict(py)?
        } else {
            py.None()
        };
        dict.set_item("drift_peak", drift_peak)?;
        dict.set_item("energy_peak", energy_peak)?;
        dict.set_item("suggested_max_scale", self.report.suggested_max_scale)?;
        dict.set_item("suggested_pressure", self.report.suggested_pressure)?;
        dict.set_item("diagnostic", &self.report.diagnostic)?;
        dict.set_item("should_rewrite", self.report.should_rewrite())?;
        #[cfg(feature = "kdsl")]
        {
            dict.set_item("spiralk_script", self.report.spiralk_script.clone())?;
        }
        #[cfg(not(feature = "kdsl"))]
        {
            dict.set_item("spiralk_script", py.None())?;
        }
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        let peak = self
            .report
            .drift_peak
            .as_ref()
            .map(|p| format!(" {:.2}Hz", p.frequency))
            .unwrap_or_default();
        Ok(format!(
            "MaintainerReport(status={}, drift={:.3}{peak}, energy={:.3})",
            self.report.status.as_str(),
            self.report.average_drift,
            self.report.mean_energy
        ))
    }
}

#[cfg(feature = "collapse")]
#[pyclass(module = "spiraltorch", name = "CollapsePulse")]
#[derive(Clone)]
struct PyCollapsePulse {
    pulse: hub::CollapsePulse,
}

#[cfg(feature = "collapse")]
impl PyCollapsePulse {
    fn from_pulse(pulse: hub::CollapsePulse) -> Self {
        Self { pulse }
    }

    fn command_kind(&self) -> &'static str {
        match self.pulse.command {
            DriveCmd::Collapse { .. } => "collapse",
            DriveCmd::Bloom { .. } => "bloom",
            DriveCmd::None => "none",
        }
    }
}

#[cfg(feature = "collapse")]
#[pymethods]
impl PyCollapsePulse {
    #[getter]
    fn step(&self) -> u64 {
        self.pulse.step
    }

    #[getter]
    fn total(&self) -> f32 {
        self.pulse.total
    }

    #[getter]
    fn command(&self) -> &'static str {
        self.command_kind()
    }

    #[getter]
    fn loop_signal(&self) -> Option<PyChronoLoopSignal> {
        self.pulse
            .loop_signal
            .clone()
            .map(PyChronoLoopSignal::from_signal)
    }

    fn command_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        match self.pulse.command {
            DriveCmd::Collapse {
                grad_scale,
                max_norm,
                lr_decay,
            } => {
                dict.set_item("grad_scale", grad_scale)?;
                dict.set_item("max_norm", max_norm)?;
                dict.set_item("lr_decay", lr_decay)?;
            }
            DriveCmd::Bloom { lr_mul } => {
                dict.set_item("lr_mul", lr_mul)?;
            }
            DriveCmd::None => {}
        }
        Ok(dict.into_py(py))
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("step", self.pulse.step)?;
        dict.set_item("total", self.pulse.total)?;
        dict.set_item("command", self.command_kind())?;
        dict.set_item("params", self.command_params(py)?)?;
        if let Some(signal) = self.pulse.loop_signal.clone() {
            dict.set_item(
                "loop_signal",
                PyChronoLoopSignal::from_signal(signal).as_dict(py)?,
            )?;
        } else {
            dict.set_item("loop_signal", py.None())?;
        }
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CollapsePulse(step={}, total={:.3}, command={})",
            self.pulse.step,
            self.pulse.total,
            self.command_kind()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "SpiralDifferentialTrace")]
struct PySpiralDifferentialTrace {
    trace: Option<DifferentialTrace>,
    sot_plan: Option<PySoT3DPlan>,
}

impl PySpiralDifferentialTrace {
    fn from_trace_with_plan(trace: DifferentialTrace, plan: Option<PySoT3DPlan>) -> Self {
        Self {
            trace: Some(trace),
            sot_plan: plan,
        }
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
    #[getter]
    fn sot_plan(&self) -> Option<PySoT3DPlan> {
        self.sot_plan.clone()
    }

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

#[pyclass(module = "spiraltorch", name = "TensorBiome")]
#[derive(Clone, Debug)]
struct PyTensorBiome {
    inner: TensorBiome,
}

impl PyTensorBiome {
    fn from_biome(biome: TensorBiome) -> Self {
        Self { inner: biome }
    }

    fn total_weight_value(&self) -> f32 {
        self.inner.total_weight()
    }
}

#[pymethods]
impl PyTensorBiome {
    #[new]
    fn new(topos: &PyOpenTopos) -> Self {
        Self {
            inner: TensorBiome::new(topos.inner.clone()),
        }
    }

    fn topos(&self) -> PyOpenTopos {
        PyOpenTopos::from_topos(self.inner.topos().clone())
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn total_weight(&self) -> f32 {
        self.total_weight_value()
    }

    fn weights(&self) -> Vec<f32> {
        self.inner.weights().to_vec()
    }

    fn absorb(&mut self, label: &str, tensor: &PyTensor) -> PyResult<()> {
        convert(
            self.inner
                .absorb(intern_label(label), tensor.as_tensor().clone()),
        )
    }

    fn absorb_weighted(&mut self, label: &str, tensor: &PyTensor, weight: f32) -> PyResult<()> {
        convert(
            self.inner
                .absorb_weighted(intern_label(label), tensor.as_tensor().clone(), weight),
        )
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn canopy(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(self.inner.canopy())?))
    }

    fn stack(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(self.inner.stack())?))
    }

    fn shoots(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTensor>>> {
        self.inner
            .shoots()
            .iter()
            .cloned()
            .map(|tensor| Py::new(py, PyTensor::from_tensor(tensor)))
            .collect()
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.len())
    }

    fn __repr__(&self) -> PyResult<String> {
        let (rows, cols) = self
            .inner
            .shoots()
            .first()
            .map(|tensor| tensor.shape())
            .unwrap_or((0, 0));
        Ok(format!(
            "TensorBiome(len={}, shape=({}, {}), total_weight={:.3})",
            self.len(),
            rows,
            cols,
            self.total_weight_value()
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

    fn describe_resonance(&self, resonance: &PyDifferentialResonance) -> PyResult<String> {
        let resonator = TextResonator::with_encoder(self.inner.clone());
        Ok(resonator.describe_resonance(&resonance.inner).summary)
    }

    fn speak(&self, frames: Vec<PyChronoFrame>) -> PyResult<Vec<f32>> {
        let resonator = TextResonator::with_encoder(self.inner.clone());
        let frames: Vec<ChronoFrame> = frames.into_iter().map(|frame| frame.into_frame()).collect();
        convert(resonator.speak(&frames))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "LanguageWaveEncoder(curvature={}, temperature={})",
            self.curvature(),
            self.temperature()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "TextResonator")]
#[derive(Clone)]
struct PyTextResonator {
    inner: TextResonator,
}

impl PyTextResonator {
    fn from_inner(inner: TextResonator) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTextResonator {
    #[new]
    fn new(curvature: f32, temperature: f32) -> PyResult<Self> {
        Ok(Self::from_inner(convert(TextResonator::new(
            curvature,
            temperature,
        ))?))
    }

    fn curvature(&self) -> f32 {
        self.inner.encoder().curvature()
    }

    fn temperature(&self) -> f32 {
        self.inner.encoder().temperature()
    }

    fn describe_resonance(&self, resonance: &PyDifferentialResonance) -> PyResult<String> {
        Ok(self.inner.describe_resonance(&resonance.inner).summary)
    }

    fn describe_frame(&self, frame: &PyChronoFrame) -> PyResult<String> {
        Ok(self.inner.describe_frame(frame.as_frame()).summary)
    }

    fn speak(&self, frames: Vec<PyChronoFrame>) -> PyResult<Vec<f32>> {
        let frames: Vec<ChronoFrame> = frames.into_iter().map(|frame| frame.into_frame()).collect();
        convert(self.inner.speak(&frames))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "TextResonator(curvature={}, temperature={})",
            self.curvature(),
            self.temperature()
        ))
    }
}

#[pyclass(module = "spiraltorch.nn", name = "ZSpaceProjector")]
struct PyZSpaceProjector {
    inner: Option<NnZSpaceProjector>,
}

impl PyZSpaceProjector {
    fn borrow(&self) -> PyResult<&NnZSpaceProjector> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ZSpaceProjector has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnZSpaceProjector> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("ZSpaceProjector has been moved"))
    }

    fn take(&mut self) -> PyResult<NnZSpaceProjector> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("ZSpaceProjector has been moved"))
    }
}

#[pymethods]
impl PyZSpaceProjector {
    #[new]
    fn new(topos: &PyOpenTopos, encoder: &PyLanguageWaveEncoder) -> PyResult<Self> {
        let inner = convert(NnZSpaceProjector::new(
            topos.inner.clone(),
            encoder.inner.clone(),
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

    fn curvature(&self) -> PyResult<f32> {
        Ok(self.borrow()?.curvature())
    }

    fn topos(&self) -> PyResult<PyOpenTopos> {
        Ok(PyOpenTopos::from_topos(self.borrow()?.topos().clone()))
    }

    fn encode_text(&self, text: &str) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.encode_text(text),
        )?))
    }

    fn project_spiral(&self, plan: &PySoT3DPlan) -> PyResult<PyTensor> {
        let base = plan.positions_tensor().map_err(tensor_err)?;
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.forward(&base),
        )?))
    }

    fn reimport_biome(&self, biome: &PyTensorBiome) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.reimport_biome(&biome.inner),
        )?))
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

#[pyclass(module = "spiraltorch", name = "DistConfig")]
#[derive(Clone)]
struct PyDistConfig {
    inner: DistConfig,
}

impl PyDistConfig {
    fn from_config(inner: DistConfig) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDistConfig {
    #[new]
    #[pyo3(signature = (node_id=None, mode=None, push_interval=None, summary_window=None, meta_endpoints=None))]
    fn new(
        node_id: Option<String>,
        mode: Option<&str>,
        push_interval: Option<f32>,
        summary_window: Option<usize>,
        meta_endpoints: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut config = DistConfig::default();
        if let Some(id) = node_id {
            config.node_id = id;
        }
        if let Some(mode_str) = mode {
            config.mode = match mode_str {
                "local" | "local-only" => DistMode::LocalOnly,
                "periodic" | "periodic-meta" => DistMode::PeriodicMeta,
                "global" | "fully-global" => DistMode::FullyGlobal,
                other => {
                    return Err(PyValueError::new_err(format!(
                    "unknown dist mode '{}': expected local-only, periodic-meta, or fully-global",
                    other
                )))
                }
            };
        }
        if let Some(interval) = push_interval {
            if interval <= 0.0 {
                return Err(PyValueError::new_err(
                    "push_interval must be positive seconds",
                ));
            }
            config.push_interval = Duration::from_secs_f32(interval);
        }
        if let Some(window) = summary_window {
            config.summary_window = window.max(1);
        }
        if let Some(endpoints) = meta_endpoints {
            config.meta_endpoints = endpoints;
        }
        Ok(Self { inner: config })
    }

    #[getter]
    fn node_id(&self) -> &str {
        &self.inner.node_id
    }

    #[getter]
    fn mode(&self) -> &'static str {
        match self.inner.mode {
            DistMode::LocalOnly => "local-only",
            DistMode::PeriodicMeta => "periodic-meta",
            DistMode::FullyGlobal => "fully-global",
        }
    }

    #[getter]
    fn push_interval(&self) -> f32 {
        self.inner.push_interval.as_secs_f32()
    }

    #[getter]
    fn summary_window(&self) -> usize {
        self.inner.summary_window
    }

    #[getter]
    fn meta_endpoints(&self) -> Vec<String> {
        self.inner.meta_endpoints.clone()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "DistConfig(node_id='{}', mode='{}', push_interval={:.1}, summary_window={}, endpoints={:?})",
            self.node_id(),
            self.mode(),
            self.push_interval(),
            self.summary_window(),
            self.meta_endpoints(),
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "RoundtableSchedule", unsendable)]
struct PyRoundtableSchedule {
    inner: RoundtableSchedule,
}

impl PyRoundtableSchedule {
    fn from_schedule(inner: RoundtableSchedule) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRoundtableSchedule {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    #[getter]
    fn top_k(&self) -> u32 {
        self.inner.above().k
    }

    #[getter]
    fn mid_k(&self) -> u32 {
        self.inner.here().k
    }

    #[getter]
    fn bottom_k(&self) -> u32 {
        self.inner.beneath().k
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "RoundtableSchedule(top={}, mid={}, bottom={})",
            self.top_k(),
            self.mid_k(),
            self.bottom_k()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "EpochStats")]
#[derive(Clone, Copy)]
struct PyEpochStats {
    inner: EpochStats,
}

impl PyEpochStats {
    fn from_stats(inner: EpochStats) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEpochStats {
    #[getter]
    fn batches(&self) -> usize {
        self.inner.batches
    }

    #[getter]
    fn total_loss(&self) -> f32 {
        self.inner.total_loss
    }

    #[getter]
    fn average_loss(&self) -> f32 {
        self.inner.average_loss
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "EpochStats(batches={}, total_loss={:.6}, average_loss={:.6})",
            self.batches(),
            self.total_loss(),
            self.average_loss()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "ModuleTrainer", unsendable)]
struct PyModuleTrainer {
    inner: ModuleTrainer,
}

impl PyModuleTrainer {
    fn from_trainer(inner: ModuleTrainer) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyModuleTrainer {
    #[new]
    #[pyo3(signature = (device=None, curvature=-1.0, hyper_learning_rate=0.05, fallback_learning_rate=0.01))]
    fn new(
        device: Option<&str>,
        curvature: f32,
        hyper_learning_rate: f32,
        fallback_learning_rate: f32,
    ) -> Self {
        let caps = caps_for(device);
        let inner =
            ModuleTrainer::new(caps, curvature, hyper_learning_rate, fallback_learning_rate);
        Self { inner }
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

    #[pyo3(signature = (rows, cols, top_k=8, mid_k=8, bottom_k=8, here_tolerance=1e-5, psychoid=false, psychoid_log=false, psi=false, collapse=false, dist=None))]
    fn roundtable(
        &mut self,
        rows: u32,
        cols: u32,
        top_k: u32,
        mid_k: u32,
        bottom_k: u32,
        here_tolerance: f32,
        psychoid: bool,
        psychoid_log: bool,
        psi: bool,
        collapse: bool,
        dist: Option<PyDistConfig>,
    ) -> PyResult<PyRoundtableSchedule> {
        let config = build_roundtable_config(
            top_k,
            mid_k,
            bottom_k,
            here_tolerance,
            psychoid,
            psychoid_log,
            psi,
            collapse,
        );
        if let Some(dist_cfg) = dist {
            self.inner.configure_distribution(dist_cfg.inner.clone());
        } else {
            self.inner.clear_distribution();
        }
        Ok(PyRoundtableSchedule::from_schedule(
            self.inner.roundtable(rows, cols, config),
        ))
    }

    #[pyo3(signature = (threshold, participants=2))]
    fn install_meta_conductor(&mut self, threshold: f32, participants: usize) {
        self.inner.install_meta_conductor(threshold, participants);
    }

    #[pyo3(signature = (threshold, participants=2))]
    fn install_blackcat_moderator(&mut self, threshold: f32, participants: usize) {
        self.inner
            .install_blackcat_moderator(threshold, participants);
    }

    fn blackcat_minutes<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let minutes = self.inner.blackcat_minutes();
        let list = PyList::empty_bound(py);
        for minute in minutes {
            let entry = PyDict::new_bound(py);
            entry.set_item("plan_signature", minute.plan_signature.clone())?;
            entry.set_item("script_hint", minute.script_hint.clone())?;
            entry.set_item("winner", format!("{:?}", minute.winner))?;
            entry.set_item("support", minute.support)?;
            entry.set_item("mean_score", minute.mean_score)?;
            entry.set_item("mean_psi", minute.mean_psi)?;
            entry.set_item("confidence", (minute.confidence.0, minute.confidence.1))?;
            entry.set_item("reward", minute.reward)?;
            entry.set_item("notes", minute.notes.clone())?;
            entry.set_item(
                "issued_at",
                minute
                    .issued_at
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0),
            )?;
            let picks = PyDict::new_bound(py);
            for (k, v) in minute.picks.iter() {
                picks.set_item(k.clone(), v.clone())?;
            }
            entry.set_item("picks", picks.into_py(py))?;
            list.append(entry.into_py(py))?;
        }
        Ok(list.into_py(py))
    }

    #[pyo3(signature = (module, loss, batches, schedule))]
    fn train_epoch(
        &mut self,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        batches: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
    ) -> PyResult<PyEpochStats> {
        let stats =
            run_epoch_with_trainer(&mut self.inner, module, loss, batches, &schedule.inner)?;
        Ok(PyEpochStats::from_stats(stats))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ModuleTrainer(curvature={:.4}, hyper_lr={:.4}, fallback_lr={:.4})",
            self.curvature(),
            self.hyper_learning_rate(),
            self.fallback_learning_rate()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "SpiralLightning", unsendable)]
struct PySpiralLightning {
    inner: NnSpiralLightning,
}

impl PySpiralLightning {
    fn from_inner(inner: NnSpiralLightning) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySpiralLightning {
    #[new]
    #[pyo3(signature = (session, rows, cols, *, top_k=8, mid_k=8, bottom_k=8, here_tolerance=1e-5, auto_prepare=true, psychoid=false, psychoid_log=false, psi=false, collapse=false))]
    fn new(
        session: PySpiralSession,
        rows: u32,
        cols: u32,
        top_k: u32,
        mid_k: u32,
        bottom_k: u32,
        here_tolerance: f32,
        auto_prepare: bool,
        psychoid: bool,
        psychoid_log: bool,
        psi: bool,
        collapse: bool,
    ) -> Self {
        let roundtable = build_roundtable_config(
            top_k,
            mid_k,
            bottom_k,
            here_tolerance,
            psychoid,
            psychoid_log,
            psi,
            collapse,
        );
        let config = NnLightningConfig::builder(rows, cols)
            .roundtable(roundtable)
            .auto_prepare(auto_prepare)
            .build();
        let inner = NnSpiralLightning::with_config(session.inner.clone(), config);
        Self { inner }
    }

    #[getter]
    fn rows(&self) -> u32 {
        self.inner.config().rows()
    }

    #[getter]
    fn cols(&self) -> u32 {
        self.inner.config().cols()
    }

    #[getter]
    fn auto_prepare(&self) -> bool {
        self.inner.config().auto_prepare()
    }

    fn set_auto_prepare(&mut self, enabled: bool) {
        self.inner.set_auto_prepare(enabled);
    }

    fn schedule(&self) -> PyRoundtableSchedule {
        PyRoundtableSchedule::from_schedule(self.inner.schedule().clone())
    }

    #[pyo3(signature = (rows, cols, *, top_k=8, mid_k=8, bottom_k=8, here_tolerance=1e-5, auto_prepare=true, psychoid=false, psychoid_log=false, psi=false, collapse=false))]
    fn reconfigure(
        &mut self,
        rows: u32,
        cols: u32,
        top_k: u32,
        mid_k: u32,
        bottom_k: u32,
        here_tolerance: f32,
        auto_prepare: bool,
        psychoid: bool,
        psychoid_log: bool,
        psi: bool,
        collapse: bool,
    ) {
        let roundtable = build_roundtable_config(
            top_k,
            mid_k,
            bottom_k,
            here_tolerance,
            psychoid,
            psychoid_log,
            psi,
            collapse,
        );
        let config = NnLightningConfig::builder(rows, cols)
            .roundtable(roundtable)
            .auto_prepare(auto_prepare)
            .build();
        self.inner.reconfigure(config);
    }

    fn prepare_module(&mut self, module: &Bound<'_, PyAny>) -> PyResult<()> {
        prepare_module_for_lightning(&mut self.inner, module)
    }

    fn reset_prepared(&mut self) {
        self.inner.reset_prepared_modules();
    }

    fn train_epoch(
        &mut self,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        batches: &Bound<'_, PyAny>,
    ) -> PyResult<PyEpochStats> {
        let stats = run_epoch_with_lightning(&mut self.inner, module, loss, batches)?;
        Ok(PyEpochStats::from_stats(stats))
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        epochs: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyEpochStats>> {
        let epoch_objects: Vec<PyObject> = epochs.extract()?;
        let mut reports = Vec::with_capacity(epoch_objects.len());
        for epoch in epoch_objects {
            let bound = epoch.bind(py);
            let stats = run_epoch_with_lightning(&mut self.inner, module, loss, &bound)?;
            reports.push(PyEpochStats::from_stats(stats));
        }
        Ok(reports)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "SpiralLightning(rows={}, cols={}, auto_prepare={})",
            self.rows(),
            self.cols(),
            self.auto_prepare()
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

    #[pyo3(signature = (jitter_threshold=None, growth_threshold=None, energy_floor=None, clamp_min=None, clamp_max=None, pressure_step=None, window=None))]
    fn maintainer(
        &mut self,
        jitter_threshold: Option<f32>,
        growth_threshold: Option<f32>,
        energy_floor: Option<f32>,
        clamp_min: Option<f32>,
        clamp_max: Option<f32>,
        pressure_step: Option<f32>,
        window: Option<usize>,
    ) -> PyResult<()> {
        let builder = self.ensure_builder()?;
        let mut config = builder.maintainer_config().clone();
        if let Some(value) = jitter_threshold {
            config.jitter_threshold = value;
        }
        if let Some(value) = growth_threshold {
            config.growth_threshold = value;
        }
        if let Some(value) = energy_floor {
            config.energy_floor = value;
        }
        if let Some(value) = clamp_min {
            config.clamp_min = value;
        }
        if let Some(value) = clamp_max {
            config.clamp_max = value;
        }
        if let Some(value) = pressure_step {
            config.pressure_step = value;
        }
        if let Some(value) = window {
            config.window = value;
        }
        builder.set_maintainer_config(config);
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

    fn trainer(&self) -> PyModuleTrainer {
        PyModuleTrainer::from_trainer(self.inner.trainer())
    }

    fn resonate_over_time(
        &self,
        resonance: &PyDifferentialResonance,
        dt: f32,
    ) -> PyResult<PyChronoFrame> {
        let frame = convert(self.inner.resonate_over_time(&resonance.inner, dt))?;
        Ok(PyChronoFrame::from_frame(frame))
    }

    #[pyo3(signature = (timesteps=None))]
    fn timeline(&self, timesteps: Option<usize>) -> PyResult<Vec<PyChronoFrame>> {
        let frames = self.inner.chrono_frames();
        let limit = timesteps.unwrap_or(frames.len());
        let start = frames.len().saturating_sub(limit);
        Ok(frames[start..]
            .iter()
            .cloned()
            .map(PyChronoFrame::from_frame)
            .collect())
    }

    #[pyo3(signature = (timesteps=None))]
    fn animate_resonance(
        &self,
        timesteps: Option<usize>,
    ) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let frames = self.timeline(timesteps)?;
        let mut times = Vec::with_capacity(frames.len());
        let mut energies = Vec::with_capacity(frames.len());
        let mut drifts = Vec::with_capacity(frames.len());
        for frame in &frames {
            let inner = frame.as_frame();
            times.push(inner.timestamp);
            energies.push(inner.total_energy);
            drifts.push(inner.curvature_drift);
        }
        Ok((times, energies, drifts))
    }

    #[pyo3(signature = (timesteps=None))]
    fn timeline_summary(&self, timesteps: Option<usize>) -> Option<PyChronoSummary> {
        self.inner
            .timeline_summary(timesteps)
            .map(PyChronoSummary::from_summary)
    }

    fn atlas(&self) -> Option<PyAtlasFrame> {
        self.inner.atlas().map(PyAtlasFrame::from_frame)
    }

    #[pyo3(signature = (limit=None))]
    fn atlas_route(&self, limit: Option<usize>) -> PyAtlasRoute {
        PyAtlasRoute::from_route(self.inner.atlas_route(limit))
    }

    #[pyo3(signature = (timesteps=None, bins=16))]
    fn timeline_harmonics(
        &self,
        timesteps: Option<usize>,
        bins: usize,
    ) -> Option<PyChronoHarmonics> {
        self.inner
            .timeline_harmonics(timesteps, bins)
            .map(PyChronoHarmonics::from_harmonics)
    }

    #[pyo3(signature = (timesteps=None, bins=None))]
    fn loop_signal(
        &self,
        timesteps: Option<usize>,
        bins: Option<usize>,
    ) -> Option<PyChronoLoopSignal> {
        self.inner
            .loop_signal(timesteps, bins)
            .map(PyChronoLoopSignal::from_signal)
    }

    #[pyo3(signature = (timesteps=None, temperature=0.6))]
    fn timeline_story(
        &self,
        timesteps: Option<usize>,
        temperature: f32,
    ) -> PyResult<(String, Vec<String>)> {
        let narrative = self
            .inner
            .timeline_narrative(timesteps, temperature)
            .map_err(tensor_err)?;
        Ok((narrative.summary, narrative.highlights))
    }

    #[pyo3(signature = (temperature=0.6))]
    fn atlas_story(&self, temperature: f32) -> PyResult<Option<(String, Vec<String>)>> {
        let narrative = self
            .inner
            .atlas_narrative(temperature)
            .map_err(tensor_err)?;
        Ok(narrative.map(|story| (story.summary, story.highlights)))
    }

    #[pyo3(signature = (timesteps=None, temperature=0.6))]
    fn speak(&self, timesteps: Option<usize>, temperature: f32) -> PyResult<Vec<f32>> {
        convert(self.inner.speak(timesteps, temperature))
    }

    fn maintainer_config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cfg = self.inner.maintainer_config();
        let dict = PyDict::new_bound(py);
        dict.set_item("jitter_threshold", cfg.jitter_threshold)?;
        dict.set_item("growth_threshold", cfg.growth_threshold)?;
        dict.set_item("energy_floor", cfg.energy_floor)?;
        dict.set_item("clamp_min", cfg.clamp_min)?;
        dict.set_item("clamp_max", cfg.clamp_max)?;
        dict.set_item("pressure_step", cfg.pressure_step)?;
        dict.set_item("window", cfg.window)?;
        Ok(dict.into_py(py))
    }

    #[cfg(feature = "collapse")]
    fn collapse_pulse(&self) -> Option<PyCollapsePulse> {
        hub::get_collapse_pulse().map(PyCollapsePulse::from_pulse)
    }

    #[pyo3(signature = (jitter_threshold=None, growth_threshold=None, energy_floor=None, clamp_min=None, clamp_max=None, pressure_step=None, window=None))]
    fn configure_maintainer(
        &mut self,
        jitter_threshold: Option<f32>,
        growth_threshold: Option<f32>,
        energy_floor: Option<f32>,
        clamp_min: Option<f32>,
        clamp_max: Option<f32>,
        pressure_step: Option<f32>,
        window: Option<usize>,
    ) -> PyResult<()> {
        let mut config = self.inner.maintainer_config();
        if let Some(value) = jitter_threshold {
            config.jitter_threshold = value;
        }
        if let Some(value) = growth_threshold {
            config.growth_threshold = value;
        }
        if let Some(value) = energy_floor {
            config.energy_floor = value;
        }
        if let Some(value) = clamp_min {
            config.clamp_min = value;
        }
        if let Some(value) = clamp_max {
            config.clamp_max = value;
        }
        if let Some(value) = pressure_step {
            config.pressure_step = value;
        }
        if let Some(value) = window {
            config.window = value;
        }
        self.inner.set_maintainer_config(config);
        Ok(())
    }

    fn self_maintain(&self) -> PyMaintainerReport {
        PyMaintainerReport::from_report(self.inner.maintain())
    }

    #[pyo3(signature = (resonance=None, temperature=0.6))]
    fn describe(
        &self,
        resonance: Option<&PyDifferentialResonance>,
        temperature: f32,
    ) -> PyResult<String> {
        let resonance = resonance.map(|res| &res.inner);
        convert(self.inner.describe(resonance, temperature))
    }

    #[pyo3(signature = (rows, cols, top_k=8, mid_k=8, bottom_k=8, here_tolerance=1e-5, psychoid=false, psychoid_log=false, psi=false, collapse=false))]
    fn roundtable(
        &self,
        rows: u32,
        cols: u32,
        top_k: u32,
        mid_k: u32,
        bottom_k: u32,
        here_tolerance: f32,
        psychoid: bool,
        psychoid_log: bool,
        psi: bool,
        collapse: bool,
    ) -> PyRoundtableSchedule {
        let config = build_roundtable_config(
            top_k,
            mid_k,
            bottom_k,
            here_tolerance,
            psychoid,
            psychoid_log,
            psi,
            collapse,
        );
        PyRoundtableSchedule::from_schedule(self.inner.roundtable(rows, cols, config))
    }

    #[pyo3(signature = (module))]
    fn prepare_module(&self, module: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            return convert(self.inner.prepare_module(seq.borrow_mut()?));
        }
        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            return convert(self.inner.prepare_module(linear.borrow_mut()?));
        }
        if let Ok(mut relu) = module.extract::<PyRefMut<'_, PyReluModule>>() {
            return convert(self.inner.prepare_module(relu.borrow_mut()?));
        }
        if let Ok(mut conv) = module.extract::<PyRefMut<'_, PyConv1dModule>>() {
            return convert(self.inner.prepare_module(conv.borrow_mut()?));
        }
        if let Ok(mut wave) = module.extract::<PyRefMut<'_, PyWaveRnnModule>>() {
            return convert(self.inner.prepare_module(wave.borrow_mut()?));
        }
        if let Ok(mut projector) = module.extract::<PyRefMut<'_, PyZSpaceProjector>>() {
            return convert(self.inner.prepare_module(projector.borrow_mut()?));
        }

        Err(PyValueError::new_err(
            "prepare_module expects Linear, Relu, Conv1d, WaveRnn, ZSpaceProjector, or Sequential modules",
        ))
    }

    #[pyo3(signature = (*args))]
    fn train_epoch(&self, args: &Bound<'_, PyTuple>) -> PyResult<PyEpochStats> {
        match args.len() {
            5 => {
                let trainer = args.get_item(0)?;
                let mut trainer = trainer.extract::<PyRefMut<PyModuleTrainer>>()?;
                let module = args.get_item(1)?;
                let loss = args.get_item(2)?;
                let batches = args.get_item(3)?;
                let schedule = args.get_item(4)?.extract::<PyRef<PyRoundtableSchedule>>()?;
                trainer.train_epoch(&module, &loss, &batches, &*schedule)
            }
            4 => {
                let module = args.get_item(0)?;
                let loss = args.get_item(1)?;
                let batches = args.get_item(2)?;
                let schedule = args.get_item(3)?.extract::<PyRef<PyRoundtableSchedule>>()?;
                let mut trainer = self.inner.trainer();
                let stats = run_epoch_with_trainer(
                    &mut trainer,
                    &module,
                    &loss,
                    &batches,
                    &schedule.inner,
                )?;
                Ok(PyEpochStats::from_stats(stats))
            }
            _ => Err(PyValueError::new_err(
                "SpiralSession.train_epoch expects either (trainer, module, loss, batches, schedule) or (module, loss, batches, schedule)",
            )),
        }
    }

    #[pyo3(signature = (rows, cols, *, top_k=8, mid_k=8, bottom_k=8, here_tolerance=1e-5, auto_prepare=true, psychoid=false, psychoid_log=false, psi=false, collapse=false))]
    fn lightning(
        &self,
        rows: u32,
        cols: u32,
        top_k: u32,
        mid_k: u32,
        bottom_k: u32,
        here_tolerance: f32,
        auto_prepare: bool,
        psychoid: bool,
        psychoid_log: bool,
        psi: bool,
        collapse: bool,
    ) -> PySpiralLightning {
        let roundtable = build_roundtable_config(
            top_k,
            mid_k,
            bottom_k,
            here_tolerance,
            psychoid,
            psychoid_log,
            psi,
            collapse,
        );
        let config = NnLightningConfig::new(rows, cols)
            .with_roundtable(roundtable)
            .with_auto_prepare(auto_prepare);
        PySpiralLightning::from_inner(NnSpiralLightning::with_config(self.inner.clone(), config))
    }

    #[pyo3(signature = (seed, sot=None))]
    fn trace(
        &self,
        seed: &PyTensor,
        sot: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PySpiralDifferentialTrace> {
        let (rows, cols) = seed.as_tensor().shape();
        let default_steps = match rows.checked_mul(cols) {
            Some(0) | None => 1,
            Some(value) => value.max(1),
        };
        let mut plan_steps = default_steps;
        let mut params = Sot3DParams {
            base_radius: 1.0,
            radial_growth: 0.05,
            base_height: 1.0,
            meso_gain: 0.2,
            micro_gain: 0.05,
        };
        if let Some(cfg) = sot {
            if let Some(value) = cfg.get_item("steps")? {
                plan_steps = value.extract()?;
            }
            if let Some(value) = cfg.get_item("base_radius")? {
                params.base_radius = value.extract()?;
            }
            if let Some(value) = cfg.get_item("radial_growth")? {
                params.radial_growth = value.extract()?;
            }
            if let Some(value) = cfg.get_item("base_height")? {
                params.base_height = value.extract()?;
            }
            if let Some(value) = cfg.get_item("meso_gain")? {
                params.meso_gain = value.extract()?;
            }
            if let Some(value) = cfg.get_item("micro_gain")? {
                params.micro_gain = value.extract()?;
            }
        }

        let trace = convert(self.inner.trace(seed.as_tensor().clone()))?;
        let plan = if plan_steps == 0 {
            None
        } else {
            Some(crate::sot::generate_plan_with_params(plan_steps, params)?)
        };
        Ok(PySpiralDifferentialTrace::from_trace_with_plan(trace, plan))
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

#[pyclass(module = "spiraltorch.nn", name = "MeanSquaredError")]
struct PyMeanSquaredError {
    inner: MeanSquaredError,
}

impl PyMeanSquaredError {
    fn inner_mut(&mut self) -> &mut MeanSquaredError {
        &mut self.inner
    }
}

#[pymethods]
impl PyMeanSquaredError {
    #[new]
    fn new() -> Self {
        Self {
            inner: MeanSquaredError::new(),
        }
    }

    fn forward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.inner
                .forward(prediction.as_tensor(), target.as_tensor()),
        )?))
    }

    fn backward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.inner
                .backward(prediction.as_tensor(), target.as_tensor()),
        )?))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok("MeanSquaredError()".to_string())
    }
}

fn run_epoch_with_trainer(
    trainer: &mut ModuleTrainer,
    module: &Bound<'_, PyAny>,
    loss: &Bound<'_, PyAny>,
    batches: &Bound<'_, PyAny>,
    schedule: &RoundtableSchedule,
) -> PyResult<EpochStats> {
    if let Ok(loader) = batches.extract::<PyRef<PyDataLoader>>() {
        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats = convert(trainer.train_epoch(
                    seq.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    schedule,
                ))?;
                return Ok(stats);
            }
        }

        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats = convert(trainer.train_epoch(
                    linear.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    schedule,
                ))?;
                return Ok(stats);
            }
        }

        if let Ok(mut relu) = module.extract::<PyRefMut<'_, PyReluModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats = convert(trainer.train_epoch(
                    relu.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    schedule,
                ))?;
                return Ok(stats);
            }
        }
    }

    let dataset: Vec<(Tensor, Tensor)> = batches
        .extract::<Vec<(PyTensor, PyTensor)>>()?
        .into_iter()
        .map(|(input, target)| (input.into_tensor(), target.into_tensor()))
        .collect();

    if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
        if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
            let stats = convert(trainer.train_epoch(
                seq.borrow_mut()?,
                mse.inner_mut(),
                dataset.clone(),
                schedule,
            ))?;
            return Ok(stats);
        }
    }

    if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
        if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
            let stats = convert(trainer.train_epoch(
                linear.borrow_mut()?,
                mse.inner_mut(),
                dataset.clone(),
                schedule,
            ))?;
            return Ok(stats);
        }
    }

    if let Ok(mut relu) = module.extract::<PyRefMut<'_, PyReluModule>>() {
        if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
            let stats = convert(trainer.train_epoch(
                relu.borrow_mut()?,
                mse.inner_mut(),
                dataset,
                schedule,
            ))?;
            return Ok(stats);
        }
    }

    Err(PyValueError::new_err(
        "ModuleTrainer.train_epoch expects a Sequential, Linear, or Relu module and a supported loss",
    ))
}

fn prepare_module_for_lightning(
    lightning: &mut NnSpiralLightning,
    module: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
        convert(lightning.prepare_module(seq.borrow_mut()?))?;
        return Ok(());
    }
    if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
        convert(lightning.prepare_module(linear.borrow_mut()?))?;
        return Ok(());
    }
    if let Ok(mut relu) = module.extract::<PyRefMut<'_, PyReluModule>>() {
        convert(lightning.prepare_module(relu.borrow_mut()?))?;
        return Ok(());
    }
    if let Ok(mut conv) = module.extract::<PyRefMut<'_, PyConv1dModule>>() {
        convert(lightning.prepare_module(conv.borrow_mut()?))?;
        return Ok(());
    }
    if let Ok(mut wave) = module.extract::<PyRefMut<'_, PyWaveRnnModule>>() {
        convert(lightning.prepare_module(wave.borrow_mut()?))?;
        return Ok(());
    }
    if let Ok(mut projector) = module.extract::<PyRefMut<'_, PyZSpaceProjector>>() {
        convert(lightning.prepare_module(projector.borrow_mut()?))?;
        return Ok(());
    }
    Err(PyValueError::new_err(
        "SpiralLightning.prepare_module expects Linear, Relu, Conv1d, WaveRnn, ZSpaceProjector, or Sequential modules",
    ))
}

fn run_epoch_with_lightning(
    lightning: &mut NnSpiralLightning,
    module: &Bound<'_, PyAny>,
    loss: &Bound<'_, PyAny>,
    batches: &Bound<'_, PyAny>,
) -> PyResult<EpochStats> {
    if let Ok(loader) = batches.extract::<PyRef<PyDataLoader>>() {
        let dataset: Vec<(Tensor, Tensor)> = loader
            .clone_inner()
            .into_iter()
            .map(|batch| convert(batch))
            .collect::<PyResult<_>>()?;

        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats = convert(lightning.train_epoch(
                    seq.borrow_mut()?,
                    mse.inner_mut(),
                    dataset.clone(),
                ))?;
                return Ok(stats);
            }
        }

        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats = convert(lightning.train_epoch(
                    linear.borrow_mut()?,
                    mse.inner_mut(),
                    dataset.clone(),
                ))?;
                return Ok(stats);
            }
        }

        if let Ok(mut relu) = module.extract::<PyRefMut<'_, PyReluModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats =
                    convert(lightning.train_epoch(relu.borrow_mut()?, mse.inner_mut(), dataset))?;
                return Ok(stats);
            }
        }
    }

    let dataset: Vec<(Tensor, Tensor)> = batches
        .extract::<Vec<(PyTensor, PyTensor)>>()?
        .into_iter()
        .map(|(input, target)| (input.into_tensor(), target.into_tensor()))
        .collect();

    if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
        if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
            let stats = convert(lightning.train_epoch(
                seq.borrow_mut()?,
                mse.inner_mut(),
                dataset.clone(),
            ))?;
            return Ok(stats);
        }
    }

    if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
        if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
            let stats = convert(lightning.train_epoch(
                linear.borrow_mut()?,
                mse.inner_mut(),
                dataset.clone(),
            ))?;
            return Ok(stats);
        }
    }

    if let Ok(mut relu) = module.extract::<PyRefMut<'_, PyReluModule>>() {
        if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
            let stats =
                convert(lightning.train_epoch(relu.borrow_mut()?, mse.inner_mut(), dataset))?;
            return Ok(stats);
        }
    }

    Err(PyValueError::new_err(
        "SpiralLightning.train_epoch expects a Sequential, Linear, or Relu module and a supported loss",
    ))
}

#[pyclass(module = "spiraltorch.nn", name = "Relu")]
struct PyReluModule {
    inner: Option<NnRelu>,
}

impl PyReluModule {
    fn borrow(&self) -> PyResult<&NnRelu> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Relu module has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnRelu> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Relu module has been moved"))
    }

    fn take(&mut self) -> PyResult<NnRelu> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("Relu module has been moved"))
    }
}

#[pymethods]
impl PyReluModule {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(NnRelu::new()),
        }
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
        let state = PyDict::new_bound(py);
        Ok(state.into_py(py))
    }

    fn load_state_dict(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if dict.is_empty() {
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "Relu does not hold parameters and cannot load a state_dict",
            ))
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok("Relu()".to_string())
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
            } else if let Ok(mut relu) = obj.extract::<PyRefMut<'_, PyReluModule>>() {
                seq.push(relu.take()?);
            } else if let Ok(mut conv) = obj.extract::<PyRefMut<'_, PyConv1dModule>>() {
                seq.push(conv.take()?);
            } else if let Ok(mut wave) = obj.extract::<PyRefMut<'_, PyWaveRnnModule>>() {
                seq.push(wave.take()?);
            } else if let Ok(mut projector) = obj.extract::<PyRefMut<'_, PyZSpaceProjector>>() {
                seq.push(projector.take()?);
            } else {
                return Err(PyValueError::new_err(
                    "Sequential expects Linear, Relu, Conv1d, WaveRnn, or ZSpaceProjector modules",
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

#[pyfunction(name = "topk2d_tensor")]
#[pyo3(signature = (x, k, *, device=None, largest=true))]
fn topk2d_tensor_py(
    _py: Python<'_>,
    x: &PyTensor,
    k: usize,
    device: Option<&str>,
    largest: bool,
) -> PyResult<(PyTensor, PyTensor)> {
    let (rows, cols) = x.shape();
    if k == 0 || k > cols {
        return Err(PyValueError::new_err("invalid k for given tensor columns"));
    }

    let route = choose_route(device)?;
    if rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(PyValueError::new_err(
            "tensor dimensions exceed planner limits (u32::MAX)",
        ));
    }
    if k > u32::MAX as usize {
        return Err(PyValueError::new_err("k exceeds planner limits (u32::MAX)"));
    }
    let caps = caps_for(Some(route.as_str()));
    let plan = plan_rank(RankKind::TopK, rows as u32, cols as u32, k as u32, caps);
    debug_assert_eq!(plan.rows as usize, rows);
    debug_assert_eq!(plan.cols as usize, cols);

    let data = x.as_tensor().data();
    let (vals, idx) = match route {
        DeviceRoute::Cpu | DeviceRoute::Cuda | DeviceRoute::Mps | DeviceRoute::Wgpu => {
            topk_rows_cpu(data, rows, cols, k, largest)
        }
    };

    let vals_tensor = Tensor::from_vec(rows, k, vals).map_err(tensor_err)?;
    let idx_as_f32: Vec<f32> = idx.iter().map(|&value| value as f32).collect();
    let idx_tensor = Tensor::from_vec(rows, k, idx_as_f32).map_err(tensor_err)?;

    Ok((
        PyTensor::from_tensor(vals_tensor),
        PyTensor::from_tensor(idx_tensor),
    ))
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

#[pyfunction(name = "gemm")]
#[pyo3(signature = (lhs, rhs, backend=None))]
fn gemm_py(lhs: &PyTensor, rhs: &PyTensor, backend: Option<&str>) -> PyResult<PyTensor> {
    let backend = parse_matmul_backend(backend)?;
    Ok(PyTensor::from_tensor(convert(
        lhs.as_tensor()
            .matmul_with_backend(rhs.as_tensor(), backend),
    )?))
}

#[pyfunction(name = "available_backends")]
fn available_backends_py() -> Vec<&'static str> {
    let mut options = vec!["auto", "faer", "naive"];
    #[cfg(feature = "wgpu")]
    {
        options.push("wgpu");
    }
    options
}

#[pyfunction(name = "is_faer_available")]
fn is_faer_available_py() -> bool {
    faer_dense::is_available()
}

#[pyfunction(name = "is_wgpu_available")]
fn is_wgpu_available_py() -> bool {
    #[cfg(feature = "wgpu")]
    {
        return tensor_wgpu_dense::is_available();
    }
    #[cfg(not(feature = "wgpu"))]
    {
        false
    }
}

#[pyclass(module = "spiraltorch.rl", name = "EpisodeReport")]
struct PyRlEpisodeReport {
    inner: RlEpisodeReport,
}

#[pymethods]
impl PyRlEpisodeReport {
    #[getter]
    fn total_reward(&self) -> f32 {
        self.inner.total_reward
    }

    #[getter]
    fn mean_return(&self) -> f32 {
        self.inner.mean_return
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps
    }

    #[getter]
    fn hypergrad_applied(&self) -> bool {
        self.inner.hypergrad_applied
    }
}

#[pyclass(module = "spiraltorch.rl", name = "PolicyGradient", unsendable)]
struct PyPolicyGradient {
    inner: Mutex<SpiralPolicyGradient>,
}

#[pymethods]
impl PyPolicyGradient {
    #[new]
    #[pyo3(signature = (state_dim, action_dim, learning_rate=0.01, discount=0.99))]
    fn new(
        state_dim: usize,
        action_dim: usize,
        learning_rate: f32,
        discount: f32,
    ) -> PyResult<Self> {
        let inner = SpiralPolicyGradient::new(state_dim, action_dim, learning_rate, discount)
            .map_err(rl_err)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    fn enable_hypergrad(&self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        let mut guard = self.inner.lock().unwrap();
        guard
            .enable_hypergrad(curvature, learning_rate)
            .map_err(rl_err)
    }

    #[pyo3(signature = (state))]
    fn select_action(&self, state: &PyTensor) -> PyResult<(usize, Vec<f32>)> {
        let guard = self.inner.lock().unwrap();
        let probs = guard.policy(state.as_tensor()).map_err(rl_err)?;
        let (action, _) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap();
        Ok((action, probs))
    }

    fn record_transition(&self, state: &PyTensor, action: usize, reward: f32) -> PyResult<()> {
        let mut guard = self.inner.lock().unwrap();
        guard
            .record_transition(state.as_tensor().clone(), action, reward)
            .map_err(rl_err)
    }

    fn finish_episode(&self) -> PyResult<PyRlEpisodeReport> {
        let mut guard = self.inner.lock().unwrap();
        let report = guard.finish_episode().map_err(rl_err)?;
        Ok(PyRlEpisodeReport { inner: report })
    }

    fn reset(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.reset_episode();
    }

    fn weights(&self) -> PyResult<PyTensor> {
        let guard = self.inner.lock().unwrap();
        Ok(PyTensor::from_tensor(guard.weights().clone()))
    }

    fn bias(&self) -> PyResult<Vec<f32>> {
        let guard = self.inner.lock().unwrap();
        Ok(guard.bias().to_vec())
    }
}

#[pyclass(module = "spiraltorch.rec", name = "EpochReport")]
struct PyRecEpochReport {
    inner: RecEpochReport,
}

#[pymethods]
impl PyRecEpochReport {
    #[getter]
    fn rmse(&self) -> f32 {
        self.inner.rmse
    }

    #[getter]
    fn samples(&self) -> usize {
        self.inner.samples
    }

    #[getter]
    fn regularization_penalty(&self) -> f32 {
        self.inner.regularization_penalty
    }
}

#[pyclass(module = "spiraltorch.rec", name = "Recommender", unsendable)]
struct PyRecommender {
    inner: Mutex<SpiralRecommender>,
}

#[pymethods]
impl PyRecommender {
    #[new]
    #[pyo3(signature = (users, items, factors, learning_rate=0.01, regularization=0.001, curvature=-1.0))]
    fn new(
        users: usize,
        items: usize,
        factors: usize,
        learning_rate: f32,
        regularization: f32,
        curvature: f32,
    ) -> PyResult<Self> {
        let inner = SpiralRecommender::new(
            users,
            items,
            factors,
            learning_rate,
            regularization,
            curvature,
        )
        .map_err(rec_err)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    fn predict(&self, user: usize, item: usize) -> PyResult<f32> {
        let guard = self.inner.lock().unwrap();
        guard.predict(user, item).map_err(rec_err)
    }

    fn train_epoch(&self, ratings: Vec<(usize, usize, f32)>) -> PyResult<PyRecEpochReport> {
        let triples: Vec<RecRatingTriple> = ratings
            .into_iter()
            .map(|(u, i, r)| RecRatingTriple::new(u, i, r))
            .collect();
        let mut guard = self.inner.lock().unwrap();
        let report = guard.train_epoch(&triples).map_err(rec_err)?;
        Ok(PyRecEpochReport { inner: report })
    }

    fn user_embedding(&self, user: usize) -> PyResult<PyTensor> {
        let guard = self.inner.lock().unwrap();
        Ok(PyTensor::from_tensor(
            guard.user_embedding(user).map_err(rec_err)?,
        ))
    }

    fn item_embedding(&self, item: usize) -> PyResult<PyTensor> {
        let guard = self.inner.lock().unwrap();
        Ok(PyTensor::from_tensor(
            guard.item_embedding(item).map_err(rec_err)?,
        ))
    }

    #[getter]
    fn users(&self) -> PyResult<usize> {
        let guard = self.inner.lock().unwrap();
        Ok(guard.users())
    }

    #[getter]
    fn items(&self) -> PyResult<usize> {
        let guard = self.inner.lock().unwrap();
        Ok(guard.items())
    }

    #[getter]
    fn factors(&self) -> PyResult<usize> {
        let guard = self.inner.lock().unwrap();
        Ok(guard.factors())
    }
}

#[pymodule]
fn nn(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMeanSquaredError>()?;
    m.add_class::<PyLinearModule>()?;
    m.add_class::<PyReluModule>()?;
    m.add_class::<PyConv1dModule>()?;
    m.add_class::<PyWaveRnnModule>()?;
    m.add_class::<PyZSpaceProjector>()?;
    m.add_class::<PySequentialModule>()?;
    m.setattr(
        "__all__",
        vec!["Linear", "Conv1d", "WaveRnn", "Sequential", "Relu"],
    )?;
    m.setattr(
        "__doc__",
        "Rust-backed neural network modules: Linear, Relu, Conv1d, WaveRnn, ZSpaceProjector, Sequential.",
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

#[pymodule]
fn dataset(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dataset_from_vec_py, m)?)?;
    m.add_class::<PyDataLoader>()?;
    m.add_class::<PyDataLoaderIter>()?;
    m.setattr("__all__", vec!["from_vec", "DataLoader"])?;
    m.setattr(
        "__doc__",
        "Dataset helpers for SpiralTorch sessions: shuffle, batch, and prefetch in Rust.",
    )?;
    Ok(())
}

#[pymodule]
fn linalg(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gemm_py, m)?)?;
    m.add_function(wrap_pyfunction!(available_backends_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_faer_available_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_wgpu_available_py, m)?)?;
    m.setattr(
        "__all__",
        vec![
            "gemm",
            "available_backends",
            "is_faer_available",
            "is_wgpu_available",
        ],
    )?;
    m.setattr(
        "__doc__",
        "Dense linear algebra helpers accelerated by faer (SIMD) and WGPU compute GEMM.",
    )?;
    Ok(())
}

#[pymodule]
fn rl(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPolicyGradient>()?;
    m.add_class::<PyRlEpisodeReport>()?;
    m.setattr("__all__", vec!["PolicyGradient", "EpisodeReport"])?;
    m.setattr(
        "__doc__",
        "Policy gradient helpers that keep trajectories and hypergrad updates inside SpiralTorch Z-space.",
    )?;
    Ok(())
}

#[pymodule]
fn rec(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRecommender>()?;
    m.add_class::<PyRecEpochReport>()?;
    m.setattr("__all__", vec!["Recommender", "EpochReport"])?;
    m.setattr(
        "__doc__",
        "Recommender systems driven by SpiralTorch open-cartesian topos guards and factor models.",
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

#[pyfunction]
#[pyo3(signature = (tensor, k, *, largest=true))]
fn topk2d_py(tensor: &PyTensor, k: usize, largest: bool) -> PyResult<(PyTensor, Vec<Vec<usize>>)> {
    let (rows, cols) = tensor.as_tensor().shape();
    if k == 0 || k > cols {
        return Err(PyValueError::new_err(
            "k must be between 1 and the number of columns",
        ));
    }

    let mut values = Vec::with_capacity(rows * k);
    let mut indices = Vec::with_capacity(rows);
    let data = tensor.as_tensor().data();
    for row in 0..rows {
        let slice = &data[row * cols..(row + 1) * cols];
        let mut pairs: Vec<(usize, f32)> = slice.iter().copied().enumerate().collect();
        pairs.sort_unstable_by(|a, b| {
            let ord = a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal);
            if largest {
                ord.reverse()
            } else {
                ord
            }
        });
        let mut row_indices = Vec::with_capacity(k);
        for (idx, value) in pairs.into_iter().take(k) {
            values.push(value);
            row_indices.push(idx);
        }
        indices.push(row_indices);
    }

    let values_tensor = PyTensor::from_tensor(convert(Tensor::from_vec(rows, k, values))?);
    Ok((values_tensor, indices))
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

#[pyfunction]
fn get_psychoid_stats(py: Python<'_>) -> PyResult<Option<PyObject>> {
    #[cfg(feature = "psychoid")]
    {
        if let Some(reading) = hub::get_last_psychoid() {
            let dict = PyDict::new_bound(py);
            dict.set_item("step", reading.step)?;
            dict.set_item("cti", reading.cti)?;
            let raw = PyDict::new_bound(py);
            for (key, value) in reading.raw.iter() {
                raw.set_item(*key, value)?;
            }
            let z = PyDict::new_bound(py);
            for (key, value) in reading.z_scores.iter() {
                z.set_item(*key, value)?;
            }
            dict.set_item("raw", raw)?;
            dict.set_item("z", z)?;
            return Ok(Some(dict.into_py(py)));
        }
        Ok(None)
    }
    #[cfg(not(feature = "psychoid"))]
    {
        let _ = py;
        Ok(None)
    }
}

/// Return a basic capability template for the given device string.
#[pyfunction]
#[pyo3(signature = (device=None))]
fn describe_device(py: Python<'_>, device: Option<&str>) -> PyResult<PyObject> {
    let caps = caps_for(device);
    Ok(device_caps_dict(py, caps)?.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (model_name, serialized_file, export_path, handler, extra_files=None, config=None, version=None, requirements_file=None, force=false, archive_format=None))]
fn torchserve_archive(
    model_name: &str,
    serialized_file: &str,
    export_path: &str,
    handler: &str,
    extra_files: Option<Vec<String>>,
    config: Option<&str>,
    version: Option<&str>,
    requirements_file: Option<&str>,
    force: bool,
    archive_format: Option<&str>,
) -> PyResult<String> {
    let mut cmd = Command::new("torch-model-archiver");
    cmd.arg("--model-name").arg(model_name);
    cmd.arg("--serialized-file").arg(serialized_file);
    cmd.arg("--handler").arg(handler);
    cmd.arg("--export-path").arg(export_path);

    if let Some(extra) = extra_files {
        if !extra.is_empty() {
            cmd.arg("--extra-files").arg(extra.join(","));
        }
    }
    if let Some(config) = config {
        cmd.arg("--config-file").arg(config);
    }
    if let Some(version) = version {
        cmd.arg("--version").arg(version);
    }
    if let Some(req) = requirements_file {
        cmd.arg("--requirements-file").arg(req);
    }
    if let Some(fmt) = archive_format {
        cmd.arg("--archive-format").arg(fmt);
    }
    if force {
        cmd.arg("--force");
    }

    let output = cmd.output().map_err(|err| match err.kind() {
        ErrorKind::NotFound => PyRuntimeError::new_err(
            "torch-model-archiver CLI was not found in PATH; install TorchServe tooling first",
        ),
        _ => PyRuntimeError::new_err(format!("failed to execute torch-model-archiver: {err}")),
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PyRuntimeError::new_err(format!(
            "torch-model-archiver failed with status {}: {}",
            output.status,
            stderr.trim()
        )));
    }

    let base_name = format!("{model_name}.mar");
    let default_path = Path::new(export_path).join(&base_name);
    let archive_path = version
        .map(|v| Path::new(export_path).join(format!("{model_name}_v{v}.mar")))
        .filter(|candidate| candidate.exists())
        .unwrap_or(default_path);

    Ok(archive_path.to_string_lossy().into_owned())
}

#[pyfunction]
#[pyo3(signature = (model, name, signatures=None, labels=None, metadata=None, custom_objects=None, context=None, api_version=None))]
fn bentoml_save_model(
    py: Python<'_>,
    model: PyObject,
    name: &str,
    signatures: Option<&PyDict>,
    labels: Option<&PyDict>,
    metadata: Option<&PyDict>,
    custom_objects: Option<&PyDict>,
    context: Option<&PyDict>,
    api_version: Option<&str>,
) -> PyResult<PyObject> {
    let bentoml = py.import("bentoml").map_err(|err| {
        PyImportError::new_err(format!(
            "bentoml is required for BentoML integration but could not be imported: {err}"
        ))
    })?;
    let pytorch = bentoml.getattr("pytorch").map_err(|err| {
        PyImportError::new_err(format!("bentoml.pytorch backend is unavailable: {err}"))
    })?;

    let kwargs = PyDict::new_bound(py);
    if let Some(signatures) = signatures {
        kwargs.set_item("signatures", signatures)?;
    }
    if let Some(labels) = labels {
        kwargs.set_item("labels", labels)?;
    }
    if let Some(metadata) = metadata {
        kwargs.set_item("metadata", metadata)?;
    }
    if let Some(custom_objects) = custom_objects {
        kwargs.set_item("custom_objects", custom_objects)?;
    }
    if let Some(context) = context {
        kwargs.set_item("context", context)?;
    }
    if let Some(api_version) = api_version {
        kwargs.set_item("api_version", api_version)?;
    }

    let saved = pytorch.call_method("save_model", (name, model), Some(&kwargs))?;
    Ok(saved.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (objective, n_trials=None, timeout=None, study_name=None, storage=None, direction="minimize", sampler=None, pruner=None))]
fn optuna_optimize(
    py: Python<'_>,
    objective: PyObject,
    n_trials: Option<usize>,
    timeout: Option<f64>,
    study_name: Option<&str>,
    storage: Option<&str>,
    direction: &str,
    sampler: Option<PyObject>,
    pruner: Option<PyObject>,
) -> PyResult<PyObject> {
    let optuna = py.import("optuna").map_err(|err| {
        PyImportError::new_err(format!(
            "optuna is required for hyperparameter search but could not be imported: {err}"
        ))
    })?;

    let create_kwargs = PyDict::new_bound(py);
    create_kwargs.set_item("direction", direction)?;
    if let Some(name) = study_name {
        create_kwargs.set_item("study_name", name)?;
    }
    if let Some(storage) = storage {
        create_kwargs.set_item("storage", storage)?;
    }
    if let Some(sampler) = sampler {
        create_kwargs.set_item("sampler", sampler)?;
    }
    if let Some(pruner) = pruner {
        create_kwargs.set_item("pruner", pruner)?;
    }

    let study = optuna.call_method("create_study", (), Some(&create_kwargs))?;

    let optimize_kwargs = PyDict::new_bound(py);
    if let Some(n_trials) = n_trials {
        optimize_kwargs.set_item("n_trials", n_trials)?;
    }
    if let Some(timeout) = timeout {
        optimize_kwargs.set_item("timeout", timeout)?;
    }

    study.call_method("optimize", (objective,), Some(&optimize_kwargs))?;

    Ok(study.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (trainable, config=None, num_samples=None, resources_per_trial=None, metric=None, mode=None, name=None, local_dir=None, init=true))]
fn ray_tune_run(
    py: Python<'_>,
    trainable: PyObject,
    config: Option<&PyDict>,
    num_samples: Option<usize>,
    resources_per_trial: Option<&PyDict>,
    metric: Option<&str>,
    mode: Option<&str>,
    name: Option<&str>,
    local_dir: Option<&str>,
    init: bool,
) -> PyResult<PyObject> {
    let ray = py.import("ray").map_err(|err| {
        PyImportError::new_err(format!(
            "ray is required for Ray Tune integration but could not be imported: {err}"
        ))
    })?;

    if init {
        let initialized: bool = ray.getattr("is_initialized")?.call0()?.extract()?;
        if !initialized {
            let _ = ray.call_method0("init")?;
        }
    }

    let tune = ray.getattr("tune").map_err(|err| {
        PyImportError::new_err(format!("ray.tune namespace is unavailable: {err}"))
    })?;

    let kwargs = PyDict::new_bound(py);
    if let Some(config) = config {
        kwargs.set_item("config", config)?;
    }
    if let Some(num_samples) = num_samples {
        kwargs.set_item("num_samples", num_samples)?;
    }
    if let Some(resources_per_trial) = resources_per_trial {
        kwargs.set_item("resources_per_trial", resources_per_trial)?;
    }
    if let Some(metric) = metric {
        kwargs.set_item("metric", metric)?;
    }
    if let Some(mode) = mode {
        kwargs.set_item("mode", mode)?;
    }
    if let Some(name) = name {
        kwargs.set_item("name", name)?;
    }
    if let Some(local_dir) = local_dir {
        kwargs.set_item("local_dir", local_dir)?;
    }

    let result = tune.call_method("run", (trainable,), Some(&kwargs))?;
    Ok(result.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (model, example_input, export_path, opset_version=17, dynamic_axes=None, input_names=None, output_names=None, do_constant_folding=true))]
fn export_onnx(
    py: Python<'_>,
    model: PyObject,
    example_input: PyObject,
    export_path: &str,
    opset_version: i32,
    dynamic_axes: Option<&PyDict>,
    input_names: Option<Vec<String>>,
    output_names: Option<Vec<String>>,
    do_constant_folding: bool,
) -> PyResult<()> {
    let torch = py.import("torch").map_err(|err| {
        PyImportError::new_err(format!(
            "torch is required to export ONNX models but could not be imported: {err}"
        ))
    })?;
    let onnx = torch.getattr("onnx").map_err(|err| {
        PyRuntimeError::new_err(format!("torch.onnx export utility unavailable: {err}"))
    })?;

    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("opset_version", opset_version)?;
    kwargs.set_item("do_constant_folding", do_constant_folding)?;
    if let Some(dynamic_axes) = dynamic_axes {
        kwargs.set_item("dynamic_axes", dynamic_axes)?;
    }
    if let Some(input_names) = input_names {
        kwargs.set_item("input_names", input_names)?;
    }
    if let Some(output_names) = output_names {
        kwargs.set_item("output_names", output_names)?;
    }

    onnx.call_method("export", (model, example_input, export_path), Some(&kwargs))?;

    Ok(())
}

fn integrations(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(torchserve_archive, m)?)?;
    m.add_function(wrap_pyfunction!(bentoml_save_model, m)?)?;
    m.add_function(wrap_pyfunction!(optuna_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(ray_tune_run, m)?)?;
    m.add_function(wrap_pyfunction!(export_onnx, m)?)?;

    m.setattr(
        "__all__",
        vec![
            "torchserve_archive",
            "bentoml_save_model",
            "optuna_optimize",
            "ray_tune_run",
            "export_onnx",
        ],
    )?;

    Ok(())
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
    let dataset_mod = PyModule::new_bound(_py, "dataset")?;
    dataset(_py, &dataset_mod)?;
    m.add_submodule(&dataset_mod)?;
    let linalg_mod = PyModule::new_bound(_py, "linalg")?;
    linalg(_py, &linalg_mod)?;
    m.add_submodule(&linalg_mod)?;
    let rl_mod = PyModule::new_bound(_py, "rl")?;
    rl(_py, &rl_mod)?;
    m.add_submodule(&rl_mod)?;
    let rec_mod = PyModule::new_bound(_py, "rec")?;
    rec(_py, &rec_mod)?;
    m.add_submodule(&rec_mod)?;
    let sot_mod = PyModule::new_bound(_py, "sot")?;
    sot::module(_py, &sot_mod)?;
    m.add_submodule(&sot_mod)?;
    let integrations_mod = PyModule::new_bound(_py, "integrations")?;
    integrations(_py, &integrations_mod)?;
    m.add_submodule(&integrations_mod)?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(plan_topk, m)?)?;
    m.add_function(wrap_pyfunction!(topk2d_tensor_py, m)?)?;
    m.add_function(wrap_pyfunction!(z_space_barycenter_py, m)?)?;
    m.add_function(wrap_pyfunction!(hip_probe, m)?)?;
    m.add_function(wrap_pyfunction!(describe_device, m)?)?;
    m.add_function(wrap_pyfunction!(get_psychoid_stats, m)?)?;
    m.add_function(wrap_pyfunction!(describe_resonance, m)?)?;
    m.add_function(wrap_pyfunction!(describe_frame, m)?)?;
    m.add_function(wrap_pyfunction!(describe_timeline, m)?)?;
    m.add_function(wrap_pyfunction!(describe_atlas, m)?)?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyComplexTensor>()?;
    m.add_class::<PyBarycenterIntermediate>()?;
    m.add_class::<PyZSpaceBarycenter>()?;
    m.add_class::<PyDifferentialResonance>()?;
    m.add_class::<PyAtlasMetric>()?;
    m.add_class::<PyAtlasDistrict>()?;
    m.add_class::<PyAtlasFrame>()?;
    m.add_class::<PyAtlasRoute>()?;
    m.add_class::<PyChronoFrame>()?;
    m.add_class::<PyChronoSummary>()?;
    m.add_class::<PyChronoPeak>()?;
    m.add_class::<PyChronoHarmonics>()?;
    m.add_class::<PyChronoLoopSignal>()?;
    m.add_class::<PyMaintainerReport>()?;
    #[cfg(feature = "collapse")]
    m.add_class::<PyCollapsePulse>()?;
    m.add_class::<PySpiralDifferentialTrace>()?;
    m.add_class::<PyOpenTopos>()?;
    m.add_class::<PyTensorBiome>()?;
    m.add_class::<PyLanguageWaveEncoder>()?;
    m.add_class::<PyTextResonator>()?;
    m.add_class::<PyHypergrad>()?;
    m.add_class::<PyDistConfig>()?;
    m.add_class::<PyRoundtableSchedule>()?;
    m.add_class::<PyEpochStats>()?;
    m.add_class::<PyModuleTrainer>()?;
    m.add_class::<PySpiralLightning>()?;
    m.add_class::<PySpiralSessionBuilder>()?;
    m.add_class::<PySpiralSession>()?;

    m.setattr(
        "__all__",
        vec![
            "plan",
            "plan_topk",
            "topk2d_tensor",
            "z_space_barycenter",
            "hip_probe",
            "describe_device",
            "get_psychoid_stats",
            "describe_resonance",
            "describe_frame",
            "describe_timeline",
            "describe_atlas",
            "Tensor",
            "ComplexTensor",
            "BarycenterIntermediate",
            "ZSpaceBarycenter",
            "DifferentialResonance",
            "AtlasMetric",
            "AtlasFrame",
            "ChronoFrame",
            "ChronoSummary",
            "SpiralDifferentialTrace",
            "OpenTopos",
            "TensorBiome",
            "LanguageWaveEncoder",
            "TextResonator",
            "Hypergrad",
            "DistConfig",
            "RoundtableSchedule",
            "EpochStats",
            "ModuleTrainer",
            "SpiralLightning",
            "SpiralSessionBuilder",
            "SpiralSession",
            "nn",
            "frac",
            "dataset",
            "linalg",
            "rl",
            "rec",
            "sot",
            "integrations",
        ],
    )?;
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;

    // Provide a tiny doc string that highlights the zero-shim approach.
    m.setattr(
        "__doc__",
        "Rust-first training primitives for SpiralTorch: tensors, hypergrads, top-k planners, and unified datasets.",
    )?;

    Ok(())
}
