use pyo3::prelude::*;
#[cfg(feature = "nn")]
use pyo3::types::PyAny;
#[cfg(feature = "nn")]
use pyo3::types::PyDict;
use pyo3::types::PyModule;
#[cfg(feature = "nn")]
use pyo3::wrap_pyfunction;

#[cfg(feature = "nn")]
use pyo3::exceptions::PyTypeError;
#[cfg(feature = "nn")]
use pyo3::exceptions::PyValueError;

#[cfg(feature = "nn")]
use crate::json::json_to_py;
#[cfg(feature = "nn")]
use crate::planner::{build_caps, parse_backend, PyRankPlan};
#[cfg(feature = "nn")]
use crate::pure::{PyGradientSummary, PyOpenCartesianTopos};
#[cfg(feature = "nn")]
use crate::tensor::{tensor_err_to_py, tensor_to_torch, PyTensor};
#[cfg(feature = "nn")]
use crate::theory::PyZRelativityModel;

#[cfg(feature = "nn")]
use crate::spiralk::PyMaxwellPulse;
#[cfg(feature = "nn")]
use nalgebra::DVector;
#[cfg(feature = "nn")]
use pyo3::types::{PyIterator, PyList};
#[cfg(feature = "nn")]
use st_core::config::self_rewrite::SelfRewriteCfg;
#[cfg(feature = "nn")]
use st_core::{
    maxwell::MaxwellZPulse,
    theory::zpulse::ZScale,
    util::math::{ramanujan_pi, LeechProjector},
};
#[cfg(feature = "nn")]
use st_nn::loss::{ContrastiveLoss, FocalLoss, TripletLoss};
#[cfg(feature = "nn")]
use st_nn::trainer::{
    CurvatureDecision as RustCurvatureDecision, CurvatureScheduler as RustCurvatureScheduler,
    SoftLogicConfig as RustSoftLogicConfig,
};
#[cfg(feature = "nn")]
use st_nn::Loss;
#[cfg(feature = "nn")]
use st_nn::{
    constant,
    dataset::DataLoaderBatches,
    dataset_from_vec, io as nn_io,
    layers::{
        conv::{Conv2d, Conv6da},
        spiral_rnn::SpiralRnn as RustSpiralRnn,
        Dropout as RustDropout, Embedding as RustEmbedding, Identity, NonLiner, NonLinerActivation,
        NonLinerEllipticConfig, NonLinerGeometry, NonLinerHyperbolicConfig, Scaler,
        ZSpaceCoherenceScan as RustZSpaceCoherenceScan,
        ZSpaceCoherenceWaveBlock as RustZSpaceCoherenceWaveBlock,
        ZSpaceSoftmax as RustZSpaceSoftmax,
    },
    warmup,
    zspace_coherence::{
        is_swap_invariant as rust_is_swap_invariant, CoherenceDiagnostics, CoherenceLabel,
        CoherenceObservation, CoherenceSignature, LinguisticChannelReport, PreDiscardPolicy,
        PreDiscardSnapshot, PreDiscardTelemetry,
    },
    AvgPool2d, CategoricalCrossEntropy, ConceptHint, DataLoader, Dataset, DesireAutomation,
    DesireLagrangian, DesirePhase, DesirePipeline, DesireRoundtableBridge, DesireRoundtableSummary,
    DesireTelemetryBundle, DesireTrainerBridge, DesireWeights, EpochStats as RustEpochStats,
    HyperbolicCrossEntropy, Linear, MaxPool2d, MaxwellDesireBridge, MeanSquaredError, MellinBasis,
    ModuleTrainer as RustModuleTrainer, NarrativeHint, NarrativeSummary, Relu, RepressionField,
    RoundtableConfig as RustRoundtableConfig, RoundtableSchedule as RustRoundtableSchedule,
    SemanticBridge, Sequential, SparseKernel, SymbolGeometry, TemperatureController,
    TextInfusionEvery, TextInfusionMode, WaveGate, WaveRnn, ZRelativityModule,
    ZSpaceCoherenceSequencer, ZSpaceMixer, ZSpaceTextVae, ZSpaceTraceConfig, ZSpaceTraceRecorder,
    ZSpaceVae, ZSpaceVaeState, ZSpaceVaeStats,
};
#[cfg(feature = "nn")]
use st_nn::{Module, Parameter};
#[cfg(feature = "nn")]
use st_tensor::{OpenCartesianTopos, Tensor, TensorError};
#[cfg(feature = "nn")]
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "nn")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout2d {
    Nchw,
    Nhwc,
}

#[cfg(feature = "nn")]
impl Layout2d {
    fn parse(label: &str) -> PyResult<Self> {
        match label.to_ascii_uppercase().as_str() {
            "NCHW" => Ok(Self::Nchw),
            "NHWC" => Ok(Self::Nhwc),
            other => Err(PyValueError::new_err(format!(
                "unsupported layout '{other}', expected 'NCHW' or 'NHWC'"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Nchw => "NCHW",
            Self::Nhwc => "NHWC",
        }
    }
}

#[cfg(feature = "nn")]
#[derive(Clone, Copy, Debug)]
struct Spatial2d {
    channels: usize,
    height: usize,
    width: usize,
}

#[cfg(feature = "nn")]
impl Spatial2d {
    fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
        }
    }

    fn size(self) -> usize {
        self.channels * self.height * self.width
    }
}

#[cfg(feature = "nn")]
#[derive(Clone, Copy, Debug)]
struct Spatial3d {
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
}

#[cfg(feature = "nn")]
impl Spatial3d {
    fn new(channels: usize, depth: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            depth,
            height,
            width,
        }
    }

    fn size(self) -> usize {
        self.channels * self.depth * self.height * self.width
    }
}

#[cfg(feature = "nn")]
fn dvector_to_vec(vector: &DVector<f64>) -> Vec<f64> {
    vector.iter().copied().collect()
}

#[cfg(feature = "nn")]
#[derive(Clone, Copy, Debug)]
enum LayoutDirection {
    ToCanonical,
    FromCanonical,
}

#[cfg(feature = "nn")]
impl LayoutDirection {
    fn parse(label: &str) -> PyResult<Self> {
        match label.to_ascii_lowercase().as_str() {
            "to_canonical" | "to" => Ok(Self::ToCanonical),
            "from_canonical" | "from" => Ok(Self::FromCanonical),
            other => Err(PyValueError::new_err(format!(
                "unknown direction '{other}', expected 'to_canonical' or 'from_canonical'"
            ))),
        }
    }
}

#[cfg(feature = "nn")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout3d {
    Ncdhw,
    Ndhwc,
}

#[cfg(feature = "nn")]
impl Layout3d {
    fn parse(label: &str) -> PyResult<Self> {
        match label.to_ascii_uppercase().as_str() {
            "NCDHW" => Ok(Self::Ncdhw),
            "NDHWC" => Ok(Self::Ndhwc),
            other => Err(PyValueError::new_err(format!(
                "unsupported layout '{other}', expected 'NCDHW' or 'NDHWC'"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Ncdhw => "NCDHW",
            Self::Ndhwc => "NDHWC",
        }
    }
}

#[cfg(feature = "nn")]
#[derive(Clone, Copy, Debug)]
enum PoolMode {
    Max,
    Avg,
}

#[cfg(feature = "nn")]
impl PoolMode {
    fn parse(label: &str) -> PyResult<Self> {
        match label.to_ascii_lowercase().as_str() {
            "max" => Ok(Self::Max),
            "avg" | "average" => Ok(Self::Avg),
            other => Err(PyValueError::new_err(format!(
                "unknown pooling mode '{other}', expected 'max' or 'avg'"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Max => "max",
            Self::Avg => "avg",
        }
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "CurvatureDecision")]
#[derive(Clone, Copy)]
pub(crate) struct PyCurvatureDecision {
    inner: RustCurvatureDecision,
}

#[cfg(feature = "nn")]
impl From<RustCurvatureDecision> for PyCurvatureDecision {
    fn from(inner: RustCurvatureDecision) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCurvatureDecision {
    #[getter]
    pub fn raw_pressure(&self) -> f32 {
        self.inner.raw_pressure
    }

    #[getter]
    pub fn smoothed_pressure(&self) -> f32 {
        self.inner.smoothed_pressure
    }

    #[getter]
    pub fn curvature(&self) -> f32 {
        self.inner.curvature
    }

    #[getter]
    pub fn changed(&self) -> bool {
        self.inner.changed
    }

    fn __repr__(&self) -> String {
        format!(
            "CurvatureDecision(raw_pressure={:.4}, smoothed_pressure={:.4}, curvature={:.4}, changed={})",
            self.inner.raw_pressure,
            self.inner.smoothed_pressure,
            self.inner.curvature,
            self.inner.changed
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "CurvatureScheduler", unsendable)]
pub(crate) struct PyCurvatureScheduler {
    inner: RustCurvatureScheduler,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCurvatureScheduler {
    #[new]
    #[pyo3(
        signature = (
            *,
            initial=None,
            min_curvature=None,
            max_curvature=None,
            min=None,
            max=None,
            target_pressure=0.05,
            step=None,
            tolerance=None,
            smoothing=None
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        initial: Option<f32>,
        min_curvature: Option<f32>,
        max_curvature: Option<f32>,
        min: Option<f32>,
        max: Option<f32>,
        target_pressure: f32,
        step: Option<f32>,
        tolerance: Option<f32>,
        smoothing: Option<f32>,
    ) -> Self {
        let min_bound = min.or(min_curvature).unwrap_or(-1.5);
        let max_bound = max.or(max_curvature).unwrap_or(-0.2);
        let seed = initial.unwrap_or((min_bound + max_bound) * 0.5);
        let mut inner = RustCurvatureScheduler::new(seed, min_bound, max_bound, target_pressure);
        if let Some(value) = step {
            inner.set_step(value);
        }
        if let Some(value) = tolerance {
            inner.set_tolerance(value);
        }
        if let Some(value) = smoothing {
            inner.set_smoothing(value);
        }
        Self { inner }
    }

    #[getter]
    pub fn current(&self) -> f32 {
        self.inner.current()
    }

    #[getter]
    pub fn min_curvature(&self) -> f32 {
        self.inner.min_curvature()
    }

    #[getter]
    pub fn max_curvature(&self) -> f32 {
        self.inner.max_curvature()
    }

    #[getter]
    pub fn target_pressure(&self) -> f32 {
        self.inner.target_pressure()
    }

    #[getter]
    pub fn step(&self) -> f32 {
        self.inner.step_size()
    }

    #[getter]
    pub fn tolerance(&self) -> f32 {
        self.inner.tolerance()
    }

    #[getter]
    pub fn smoothing(&self) -> f32 {
        self.inner.smoothing()
    }

    #[getter]
    pub fn last_pressure(&self) -> Option<f32> {
        self.inner.last_pressure()
    }

    pub fn set_bounds(&mut self, min_curvature: f32, max_curvature: f32) {
        self.inner.set_bounds(min_curvature, max_curvature);
    }

    pub fn set_target_pressure(&mut self, target: f32) {
        self.inner.set_target_pressure(target);
    }

    pub fn set_step(&mut self, step: f32) {
        self.inner.set_step(step);
    }

    pub fn set_tolerance(&mut self, tolerance: f32) {
        self.inner.set_tolerance(tolerance);
    }

    pub fn set_smoothing(&mut self, smoothing: f32) {
        self.inner.set_smoothing(smoothing);
    }

    pub fn sync(&mut self, curvature: f32) {
        self.inner.sync(curvature);
    }

    pub fn observe(&mut self, summary: &PyGradientSummary) -> PyCurvatureDecision {
        PyCurvatureDecision::from(self.inner.observe(summary.as_inner()))
    }

    pub fn observe_pressure(&mut self, raw_pressure: f32) -> PyCurvatureDecision {
        PyCurvatureDecision::from(self.inner.observe_pressure(raw_pressure))
    }

    fn __repr__(&self) -> String {
        format!(
            "CurvatureScheduler(current={:.4}, min={:.4}, max={:.4}, target_pressure={:.4})",
            self.inner.current(),
            self.inner.min_curvature(),
            self.inner.max_curvature(),
            self.inner.target_pressure()
        )
    }
}

#[cfg(feature = "nn")]
enum PoolModule {
    Max(MaxPool2d),
    Avg(AvgPool2d),
}

#[cfg(feature = "nn")]
impl PoolModule {
    fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        match self {
            Self::Max(module) => module.forward(input),
            Self::Avg(module) => module.forward(input),
        }
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, TensorError> {
        match self {
            Self::Max(module) => module.backward(input, grad_output),
            Self::Avg(module) => module.backward(input, grad_output),
        }
    }
}

#[cfg(feature = "nn")]
fn ensure_feature_shape(tensor: &Tensor, dims: Spatial2d) -> Result<(), TensorError> {
    let cols = tensor.shape().1;
    let expected = dims.size();
    if cols != expected {
        return Err(TensorError::ShapeMismatch {
            left: (1, cols),
            right: (1, expected),
        });
    }
    Ok(())
}

#[cfg(feature = "nn")]
fn ensure_feature_shape_3d(tensor: &Tensor, dims: Spatial3d) -> Result<(), TensorError> {
    let cols = tensor.shape().1;
    let expected = dims.size();
    if cols != expected {
        return Err(TensorError::ShapeMismatch {
            left: (1, cols),
            right: (1, expected),
        });
    }
    Ok(())
}

#[cfg(feature = "nn")]
fn reorder_row_nhwc_to_canonical(row: &[f32], dst: &mut [f32], dims: Spatial2d) {
    let Spatial2d {
        channels,
        height,
        width,
    } = dims;
    for h in 0..height {
        for w in 0..width {
            for c in 0..channels {
                let nhwc_idx = ((h * width + w) * channels) + c;
                let canonical_idx = c * height * width + h * width + w;
                dst[canonical_idx] = row[nhwc_idx];
            }
        }
    }
}

#[cfg(feature = "nn")]
fn reorder_row_canonical_to_nhwc(row: &[f32], dst: &mut [f32], dims: Spatial2d) {
    let Spatial2d {
        channels,
        height,
        width,
    } = dims;
    for c in 0..channels {
        for h in 0..height {
            for w in 0..width {
                let canonical_idx = c * height * width + h * width + w;
                let nhwc_idx = ((h * width + w) * channels) + c;
                dst[nhwc_idx] = row[canonical_idx];
            }
        }
    }
}

#[cfg(feature = "nn")]
fn reorder_tensor_layout(
    tensor: &Tensor,
    dims: Spatial2d,
    layout: Layout2d,
    direction: LayoutDirection,
) -> Result<Tensor, TensorError> {
    ensure_feature_shape(tensor, dims)?;
    match layout {
        Layout2d::Nchw => Ok(tensor.clone()),
        Layout2d::Nhwc => {
            let (rows, cols) = tensor.shape();
            let mut buffer = vec![0.0f32; rows * cols];
            let src = tensor.data();
            for row_idx in 0..rows {
                let src_start = row_idx * cols;
                let src_end = src_start + cols;
                let dst_slice = &mut buffer[src_start..src_end];
                let src_slice = &src[src_start..src_end];
                match direction {
                    LayoutDirection::ToCanonical => {
                        reorder_row_nhwc_to_canonical(src_slice, dst_slice, dims)
                    }
                    LayoutDirection::FromCanonical => {
                        reorder_row_canonical_to_nhwc(src_slice, dst_slice, dims)
                    }
                }
            }
            Tensor::from_vec(rows, cols, buffer)
        }
    }
}

#[cfg(feature = "nn")]
fn reorder_row_ndhwc_to_canonical(row: &[f32], dst: &mut [f32], dims: Spatial3d) {
    let Spatial3d {
        channels,
        depth,
        height,
        width,
    } = dims;
    for d in 0..depth {
        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    let ndhwc_idx = (((d * height + h) * width + w) * channels) + c;
                    let canonical_idx =
                        c * depth * height * width + d * height * width + h * width + w;
                    dst[canonical_idx] = row[ndhwc_idx];
                }
            }
        }
    }
}

#[cfg(feature = "nn")]
fn reorder_row_canonical_to_ndhwc(row: &[f32], dst: &mut [f32], dims: Spatial3d) {
    let Spatial3d {
        channels,
        depth,
        height,
        width,
    } = dims;
    for c in 0..channels {
        for d in 0..depth {
            for h in 0..height {
                for w in 0..width {
                    let canonical_idx =
                        c * depth * height * width + d * height * width + h * width + w;
                    let ndhwc_idx = (((d * height + h) * width + w) * channels) + c;
                    dst[ndhwc_idx] = row[canonical_idx];
                }
            }
        }
    }
}

#[cfg(feature = "nn")]
fn reorder_tensor_layout_3d(
    tensor: &Tensor,
    dims: Spatial3d,
    layout: Layout3d,
    direction: LayoutDirection,
) -> Result<Tensor, TensorError> {
    ensure_feature_shape_3d(tensor, dims)?;
    match layout {
        Layout3d::Ncdhw => Ok(tensor.clone()),
        Layout3d::Ndhwc => {
            let (rows, cols) = tensor.shape();
            let mut buffer = vec![0.0f32; rows * cols];
            let src = tensor.data();
            for row_idx in 0..rows {
                let src_start = row_idx * cols;
                let src_end = src_start + cols;
                let dst_slice = &mut buffer[src_start..src_end];
                let src_slice = &src[src_start..src_end];
                match direction {
                    LayoutDirection::ToCanonical => {
                        reorder_row_ndhwc_to_canonical(src_slice, dst_slice, dims)
                    }
                    LayoutDirection::FromCanonical => {
                        reorder_row_canonical_to_ndhwc(src_slice, dst_slice, dims)
                    }
                }
            }
            Tensor::from_vec(rows, cols, buffer)
        }
    }
}

#[cfg(feature = "nn")]
fn dilated_extent(size: usize, dilation: usize) -> Result<usize, TensorError> {
    size.checked_sub(1)
        .and_then(|value| value.checked_mul(dilation))
        .and_then(|value| value.checked_add(1))
        .ok_or(TensorError::InvalidDimensions {
            rows: size,
            cols: dilation,
        })
}

#[cfg(feature = "nn")]
fn conv_output_hw(
    input: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<(usize, usize), TensorError> {
    let eff_h = dilated_extent(kernel.0, dilation.0)?;
    let eff_w = dilated_extent(kernel.1, dilation.1)?;
    let (in_h, in_w) = input;
    if in_h + 2 * padding.0 < eff_h || in_w + 2 * padding.1 < eff_w {
        return Err(TensorError::InvalidDimensions {
            rows: in_h + 2 * padding.0,
            cols: eff_h.max(eff_w),
        });
    }
    let out_h = (in_h + 2 * padding.0 - eff_h) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - eff_w) / stride.1 + 1;
    Ok((out_h, out_w))
}

#[cfg(feature = "nn")]
fn pool_output_hw(
    input: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<(usize, usize), TensorError> {
    let (in_h, in_w) = input;
    if in_h + 2 * padding.0 < kernel.0 || in_w + 2 * padding.1 < kernel.1 {
        return Err(TensorError::InvalidDimensions {
            rows: in_h + 2 * padding.0,
            cols: kernel.0.max(kernel.1),
        });
    }
    let out_h = (in_h + 2 * padding.0 - kernel.0) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - kernel.1) / stride.1 + 1;
    Ok((out_h, out_w))
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "Pool2d", unsendable)]
pub(crate) struct PyPool2d {
    inner: Option<PoolModule>,
    mode: PoolMode,
    layout: Layout2d,
    input_dims: Spatial2d,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

#[cfg(feature = "nn")]
impl PyPool2d {
    fn inner(&self) -> PyResult<&PoolModule> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("Pool2d was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut PoolModule> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("Pool2d was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<PoolModule> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("Pool2d was moved into a container and can no longer be used")
        })
    }

    fn output_dims(&self) -> Result<Spatial2d, TensorError> {
        let spatial_in = (self.input_dims.height, self.input_dims.width);
        let (out_h, out_w) = pool_output_hw(spatial_in, self.kernel, self.stride, self.padding)?;
        Ok(Spatial2d::new(self.input_dims.channels, out_h, out_w))
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyPool2d {
    #[new]
    #[pyo3(signature = (mode, channels, height, width, kernel, *, stride=None, padding=None, layout="NCHW"))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode: &str,
        channels: usize,
        height: usize,
        width: usize,
        kernel: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        layout: &str,
    ) -> PyResult<Self> {
        let stride = stride.unwrap_or(kernel);
        let padding = padding.unwrap_or((0, 0));
        let layout = Layout2d::parse(layout)?;
        let dims = Spatial2d::new(channels, height, width);
        let mode = PoolMode::parse(mode)?;
        let inner = match mode {
            PoolMode::Max => PoolModule::Max(
                MaxPool2d::new(channels, kernel, stride, padding, (height, width))
                    .map_err(tensor_err_to_py)?,
            ),
            PoolMode::Avg => PoolModule::Avg(
                AvgPool2d::new(channels, kernel, stride, padding, (height, width))
                    .map_err(tensor_err_to_py)?,
            ),
        };
        Ok(Self {
            inner: Some(inner),
            mode,
            layout,
            input_dims: dims,
            kernel,
            stride,
            padding,
        })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let canonical = reorder_tensor_layout(
            &input.inner,
            self.input_dims,
            self.layout,
            LayoutDirection::ToCanonical,
        )
        .map_err(tensor_err_to_py)?;
        let output = self
            .inner()?
            .forward(&canonical)
            .map_err(tensor_err_to_py)?;
        let out_dims = self.output_dims().map_err(tensor_err_to_py)?;
        let restored = reorder_tensor_layout(
            &output,
            out_dims,
            self.layout,
            LayoutDirection::FromCanonical,
        )
        .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(restored))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let canonical_input = reorder_tensor_layout(
            &input.inner,
            self.input_dims,
            self.layout,
            LayoutDirection::ToCanonical,
        )
        .map_err(tensor_err_to_py)?;
        let out_dims = self.output_dims().map_err(tensor_err_to_py)?;
        let canonical_grad = reorder_tensor_layout(
            &grad_output.inner,
            out_dims,
            self.layout,
            LayoutDirection::ToCanonical,
        )
        .map_err(tensor_err_to_py)?;
        let grad_input = self
            .inner_mut()?
            .backward(&canonical_input, &canonical_grad)
            .map_err(tensor_err_to_py)?;
        let restored = reorder_tensor_layout(
            &grad_input,
            self.input_dims,
            self.layout,
            LayoutDirection::FromCanonical,
        )
        .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(restored))
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    #[getter]
    pub fn mode(&self) -> &'static str {
        self.mode.as_str()
    }

    #[getter]
    pub fn layout(&self) -> &'static str {
        self.layout.as_str()
    }

    #[getter]
    pub fn kernel(&self) -> (usize, usize) {
        self.kernel
    }

    #[getter]
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    #[getter]
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    #[getter]
    pub fn input_shape(&self) -> (usize, usize, usize) {
        (
            self.input_dims.channels,
            self.input_dims.height,
            self.input_dims.width,
        )
    }

    #[getter]
    pub fn output_shape(&self) -> PyResult<(usize, usize, usize)> {
        let dims = self.output_dims().map_err(tensor_err_to_py)?;
        Ok((dims.channels, dims.height, dims.width))
    }

    pub fn set_layout(&mut self, layout: &str) -> PyResult<()> {
        self.inner()?;
        self.layout = Layout2d::parse(layout)?;
        Ok(())
    }
}

#[cfg(feature = "nn")]
#[pyfunction]
#[pyo3(signature = (tensor, channels, height, width, *, layout="NCHW", direction="to_canonical"))]
fn reorder_feature_tensor(
    tensor: &PyTensor,
    channels: usize,
    height: usize,
    width: usize,
    layout: &str,
    direction: &str,
) -> PyResult<PyTensor> {
    let layout = Layout2d::parse(layout)?;
    let direction = LayoutDirection::parse(direction)?;
    let dims = Spatial2d::new(channels, height, width);
    let reordered =
        reorder_tensor_layout(&tensor.inner, dims, layout, direction).map_err(tensor_err_to_py)?;
    Ok(PyTensor::from_tensor(reordered))
}

#[cfg(feature = "nn")]
#[pyfunction]
#[pyo3(signature = (
    input_hw,
    kernel_hw,
    *,
    stride=None,
    padding=None,
    dilation=None,
))]
fn conv_output_shape(
    input_hw: (usize, usize),
    kernel_hw: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    dilation: Option<(usize, usize)>,
) -> PyResult<(usize, usize)> {
    let stride = stride.unwrap_or((1, 1));
    let padding = padding.unwrap_or((0, 0));
    let dilation = dilation.unwrap_or((1, 1));
    conv_output_hw(input_hw, kernel_hw, stride, padding, dilation).map_err(tensor_err_to_py)
}

#[cfg(feature = "nn")]
#[pyfunction]
#[pyo3(signature = (input_hw, kernel_hw, *, stride=None, padding=None))]
fn pool_output_shape(
    input_hw: (usize, usize),
    kernel_hw: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
) -> PyResult<(usize, usize)> {
    let stride = stride.unwrap_or(kernel_hw);
    let padding = padding.unwrap_or((0, 0));
    pool_output_hw(input_hw, kernel_hw, stride, padding).map_err(tensor_err_to_py)
}

#[cfg(feature = "nn")]
fn convert_samples(
    samples: Vec<(PyTensor, PyTensor)>,
) -> Vec<(st_tensor::Tensor, st_tensor::Tensor)> {
    samples
        .into_iter()
        .map(|(input, target)| (input.inner.clone(), target.inner.clone()))
        .collect()
}

#[cfg(feature = "nn")]
fn parse_non_liner_activation(name: &str) -> PyResult<NonLinerActivation> {
    match name.to_ascii_lowercase().as_str() {
        "tanh" => Ok(NonLinerActivation::Tanh),
        "sigmoid" => Ok(NonLinerActivation::Sigmoid),
        "softsign" => Ok(NonLinerActivation::Softsign),
        other => Err(PyValueError::new_err(format!(
            "unknown activation '{other}', expected 'tanh', 'sigmoid', or 'softsign'"
        ))),
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "Identity", unsendable)]
pub(crate) struct PyIdentity {
    inner: Identity,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyIdentity {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Identity::new(),
        }
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self.inner.forward(&input.inner).map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "Linear", unsendable)]
pub(crate) struct PyLinear {
    inner: Option<Linear>,
}

#[cfg(feature = "nn")]
impl PyLinear {
    fn inner(&self) -> PyResult<&Linear> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("Linear was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut Linear> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("Linear was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<Linear> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("Linear was moved into a container and can no longer be used")
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyLinear {
    #[new]
    pub fn new(name: &str, input_dim: usize, output_dim: usize) -> PyResult<Self> {
        let inner = Linear::new(name, input_dim, output_dim).map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner_mut()?
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.inner_mut()?
                .attach_hypergrad(curvature, learning_rate)
                .map_err(tensor_err_to_py)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner_mut()?
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner_mut()?
            .zero_accumulators()
            .map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner_mut()?
            .apply_step(fallback_lr)
            .map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner()?
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner_mut()?
            .load_state_dict(&map)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "Embedding", unsendable)]
pub(crate) struct PyEmbedding {
    inner: Option<RustEmbedding>,
}

#[cfg(feature = "nn")]
impl PyEmbedding {
    fn inner(&self) -> PyResult<&RustEmbedding> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("Embedding was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut RustEmbedding> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("Embedding was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<RustEmbedding> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("Embedding was moved into a container and can no longer be used")
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyEmbedding {
    #[new]
    pub fn new(name: &str, vocab_size: usize, embed_dim: usize) -> PyResult<Self> {
        let inner = RustEmbedding::new(name, vocab_size, embed_dim).map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner_mut()?
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.inner_mut()?
                .attach_hypergrad(curvature, learning_rate)
                .map_err(tensor_err_to_py)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner_mut()?
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner_mut()?
            .zero_accumulators()
            .map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner_mut()?
            .apply_step(fallback_lr)
            .map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner()?
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner_mut()?
            .load_state_dict(&map)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[derive(Clone, Copy, Debug)]
struct FeatureReorder2d {
    dims: Spatial2d,
    layout: Layout2d,
    direction: LayoutDirection,
}

#[cfg(feature = "nn")]
impl FeatureReorder2d {
    fn opposite(self) -> Self {
        let direction = match self.direction {
            LayoutDirection::ToCanonical => LayoutDirection::FromCanonical,
            LayoutDirection::FromCanonical => LayoutDirection::ToCanonical,
        };
        Self { direction, ..self }
    }
}

#[cfg(feature = "nn")]
impl Module for FeatureReorder2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout(input, self.dims, self.layout, self.direction)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout(
            grad_output,
            self.dims,
            self.layout,
            self.opposite().direction,
        )
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&Parameter) -> Result<(), TensorError>,
    ) -> Result<(), TensorError> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut Parameter) -> Result<(), TensorError>,
    ) -> Result<(), TensorError> {
        Ok(())
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "FeatureReorder2d", unsendable)]
pub(crate) struct PyFeatureReorder2d {
    inner: Option<FeatureReorder2d>,
}

#[cfg(feature = "nn")]
impl PyFeatureReorder2d {
    fn inner(&self) -> PyResult<&FeatureReorder2d> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "FeatureReorder2d was moved into a container and can no longer be used",
            )
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut FeatureReorder2d> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err(
                "FeatureReorder2d was moved into a container and can no longer be used",
            )
        })
    }

    fn take_inner(&mut self) -> PyResult<FeatureReorder2d> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err(
                "FeatureReorder2d was moved into a container and can no longer be used",
            )
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyFeatureReorder2d {
    #[new]
    #[pyo3(signature = (channels, height, width, *, layout="NCHW", direction="to_canonical"))]
    pub fn new(
        channels: usize,
        height: usize,
        width: usize,
        layout: &str,
        direction: &str,
    ) -> PyResult<Self> {
        let layout = Layout2d::parse(layout)?;
        let direction = LayoutDirection::parse(direction)?;
        Ok(Self {
            inner: Some(FeatureReorder2d {
                dims: Spatial2d::new(channels, height, width),
                layout,
                direction,
            }),
        })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[getter]
    pub fn shape(&self) -> PyResult<(usize, usize, usize)> {
        let inner = self.inner()?;
        Ok((inner.dims.channels, inner.dims.height, inner.dims.width))
    }

    #[getter]
    pub fn layout(&self) -> PyResult<&'static str> {
        Ok(self.inner()?.layout.as_str())
    }

    #[getter]
    pub fn direction(&self) -> PyResult<&'static str> {
        Ok(match self.inner()?.direction {
            LayoutDirection::ToCanonical => "to_canonical",
            LayoutDirection::FromCanonical => "from_canonical",
        })
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    fn __repr__(&self) -> String {
        match self.inner.as_ref() {
            Some(inner) => format!(
                "FeatureReorder2d(shape=({}, {}, {}), layout='{}', direction='{}')",
                inner.dims.channels,
                inner.dims.height,
                inner.dims.width,
                inner.layout.as_str(),
                match inner.direction {
                    LayoutDirection::ToCanonical => "to_canonical",
                    LayoutDirection::FromCanonical => "from_canonical",
                }
            ),
            None => "FeatureReorder2d(moved)".to_string(),
        }
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "SpiralRnn", unsendable)]
pub(crate) struct PySpiralRnn {
    inner: Option<RustSpiralRnn>,
}

#[cfg(feature = "nn")]
impl PySpiralRnn {
    fn inner(&self) -> PyResult<&RustSpiralRnn> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("SpiralRnn was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut RustSpiralRnn> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("SpiralRnn was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<RustSpiralRnn> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("SpiralRnn was moved into a container and can no longer be used")
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PySpiralRnn {
    #[new]
    pub fn new(name: &str, input_dim: usize, hidden_dim: usize, steps: usize) -> PyResult<Self> {
        let inner =
            RustSpiralRnn::new(name, input_dim, hidden_dim, steps).map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner_mut()?
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.inner_mut()?
                .attach_hypergrad(curvature, learning_rate)
                .map_err(tensor_err_to_py)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner_mut()?
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner_mut()?
            .zero_accumulators()
            .map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner_mut()?
            .apply_step(fallback_lr)
            .map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner()?
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner_mut()?
            .load_state_dict(&map)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "WaveGate", unsendable)]
pub(crate) struct PyWaveGate {
    inner: Option<WaveGate>,
}

#[cfg(feature = "nn")]
impl PyWaveGate {
    fn inner(&self) -> PyResult<&WaveGate> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("WaveGate was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut WaveGate> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("WaveGate was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<WaveGate> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("WaveGate was moved into a container and can no longer be used")
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyWaveGate {
    #[new]
    pub fn new(name: &str, features: usize, curvature: f32, temperature: f32) -> PyResult<Self> {
        let inner =
            WaveGate::new(name, features, curvature, temperature).map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    pub fn infuse_text(&mut self, text: &str) -> PyResult<()> {
        self.inner_mut()?
            .infuse_text(text)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner_mut()?
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.inner_mut()?
                .attach_hypergrad(curvature, learning_rate)
                .map_err(tensor_err_to_py)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner_mut()?
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner_mut()?
            .zero_accumulators()
            .map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner_mut()?
            .apply_step(fallback_lr)
            .map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner()?
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner_mut()?
            .load_state_dict(&map)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "WaveRnn", unsendable)]
pub(crate) struct PyWaveRnn {
    inner: Option<WaveRnn>,
}

#[cfg(feature = "nn")]
impl PyWaveRnn {
    fn inner(&self) -> PyResult<&WaveRnn> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("WaveRnn was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut WaveRnn> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("WaveRnn was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<WaveRnn> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("WaveRnn was moved into a container and can no longer be used")
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyWaveRnn {
    #[new]
    #[pyo3(signature = (name, in_channels, hidden_dim, kernel_size, curvature, temperature, *, stride=1, padding=0))]
    pub fn new(
        name: &str,
        in_channels: usize,
        hidden_dim: usize,
        kernel_size: usize,
        curvature: f32,
        temperature: f32,
        stride: usize,
        padding: usize,
    ) -> PyResult<Self> {
        let inner = WaveRnn::new(
            name,
            in_channels,
            hidden_dim,
            kernel_size,
            stride,
            padding,
            curvature,
            temperature,
        )
        .map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    pub fn infuse_text(&mut self, text: &str) -> PyResult<()> {
        self.inner_mut()?
            .infuse_text(text)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner_mut()?
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.inner_mut()?
                .attach_hypergrad(curvature, learning_rate)
                .map_err(tensor_err_to_py)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner_mut()?
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner_mut()?
            .zero_accumulators()
            .map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner_mut()?
            .apply_step(fallback_lr)
            .map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner()?
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner_mut()?
            .load_state_dict(&map)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZSpaceMixer", unsendable)]
pub(crate) struct PyZSpaceMixer {
    inner: Option<ZSpaceMixer>,
}

#[cfg(feature = "nn")]
impl PyZSpaceMixer {
    fn inner(&self) -> PyResult<&ZSpaceMixer> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceMixer was moved into a container and can no longer be used",
            )
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut ZSpaceMixer> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceMixer was moved into a container and can no longer be used",
            )
        })
    }

    fn take_inner(&mut self) -> PyResult<ZSpaceMixer> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceMixer was moved into a container and can no longer be used",
            )
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZSpaceMixer {
    #[new]
    pub fn new(name: &str, features: usize) -> PyResult<Self> {
        let inner = ZSpaceMixer::new(name, features).map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner_mut()?
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.inner_mut()?
                .attach_hypergrad(curvature, learning_rate)
                .map_err(tensor_err_to_py)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner_mut()?
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner_mut()?
            .zero_accumulators()
            .map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner_mut()?
            .apply_step(fallback_lr)
            .map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner()?
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner_mut()?
            .load_state_dict(&map)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZSpaceSoftmax", unsendable)]
pub(crate) struct PyZSpaceSoftmax {
    inner: Option<RustZSpaceSoftmax>,
}

#[cfg(feature = "nn")]
impl PyZSpaceSoftmax {
    fn inner(&self) -> PyResult<&RustZSpaceSoftmax> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceSoftmax was moved into a container and can no longer be used",
            )
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut RustZSpaceSoftmax> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceSoftmax was moved into a container and can no longer be used",
            )
        })
    }

    fn take_inner(&mut self) -> PyResult<RustZSpaceSoftmax> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceSoftmax was moved into a container and can no longer be used",
            )
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZSpaceSoftmax {
    #[new]
    #[pyo3(signature = (curvature, temperature, *, entropy_target=None, entropy_tolerance=1e-4, entropy_gain=0.5, min_temperature=None, max_temperature=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        curvature: f32,
        temperature: f32,
        entropy_target: Option<f32>,
        entropy_tolerance: f32,
        entropy_gain: f32,
        min_temperature: Option<f32>,
        max_temperature: Option<f32>,
    ) -> PyResult<Self> {
        let mut inner = RustZSpaceSoftmax::new(curvature, temperature).map_err(tensor_err_to_py)?;
        if let Some(target) = entropy_target {
            inner = inner
                .with_entropy_target(target, entropy_tolerance, entropy_gain)
                .map_err(tensor_err_to_py)?;
        }
        if min_temperature.is_some() || max_temperature.is_some() {
            let min = min_temperature.ok_or_else(|| {
                PyValueError::new_err("min_temperature must be provided when setting bounds")
            })?;
            let max = max_temperature.ok_or_else(|| {
                PyValueError::new_err("max_temperature must be provided when setting bounds")
            })?;
            inner = inner
                .with_temperature_bounds(min, max)
                .map_err(tensor_err_to_py)?;
        }
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    pub fn reset_metrics(&mut self) -> PyResult<()> {
        self.inner_mut()?.reset_metrics();
        Ok(())
    }

    pub fn last_entropies(&self) -> PyResult<Vec<f32>> {
        Ok(self.inner()?.last_entropies())
    }

    pub fn last_temperatures(&self) -> PyResult<Vec<f32>> {
        Ok(self.inner()?.last_temperatures())
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        Ok(Vec::new())
    }

    pub fn load_state_dict(&mut self, _state: Vec<(String, PyTensor)>) -> PyResult<()> {
        Ok(())
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZSpaceCoherenceScan", unsendable)]
pub(crate) struct PyZSpaceCoherenceScan {
    inner: Option<RustZSpaceCoherenceScan>,
}

#[cfg(feature = "nn")]
impl PyZSpaceCoherenceScan {
    fn inner(&self) -> PyResult<&RustZSpaceCoherenceScan> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceCoherenceScan was moved into a container and can no longer be used",
            )
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut RustZSpaceCoherenceScan> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceCoherenceScan was moved into a container and can no longer be used",
            )
        })
    }

    fn take_inner(&mut self) -> PyResult<RustZSpaceCoherenceScan> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceCoherenceScan was moved into a container and can no longer be used",
            )
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZSpaceCoherenceScan {
    #[new]
    #[pyo3(signature = (dim, steps, memory, curvature, temperature))]
    pub fn new(
        dim: usize,
        steps: usize,
        memory: usize,
        curvature: f32,
        temperature: f32,
    ) -> PyResult<Self> {
        let inner = RustZSpaceCoherenceScan::new(dim, steps, memory, curvature, temperature)
            .map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        Ok(Vec::new())
    }

    pub fn load_state_dict(&mut self, _state: Vec<(String, PyTensor)>) -> PyResult<()> {
        Ok(())
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    #[getter]
    pub fn dim(&self) -> PyResult<usize> {
        Ok(self.inner()?.dim())
    }

    #[getter]
    pub fn steps(&self) -> PyResult<usize> {
        Ok(self.inner()?.steps())
    }

    #[getter]
    pub fn memory(&self) -> PyResult<usize> {
        Ok(self.inner()?.memory())
    }

    #[getter]
    pub fn curvature(&self) -> PyResult<f32> {
        Ok(self.inner()?.curvature())
    }

    #[getter]
    pub fn temperature(&self) -> PyResult<f32> {
        Ok(self.inner()?.temperature())
    }
}

#[cfg(feature = "nn")]
#[pyclass(
    module = "spiraltorch.nn",
    name = "ZSpaceCoherenceWaveBlock",
    unsendable
)]
pub(crate) struct PyZSpaceCoherenceWaveBlock {
    inner: Option<RustZSpaceCoherenceWaveBlock>,
}

#[cfg(feature = "nn")]
impl PyZSpaceCoherenceWaveBlock {
    fn inner(&self) -> PyResult<&RustZSpaceCoherenceWaveBlock> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceCoherenceWaveBlock was moved into a container and can no longer be used",
            )
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut RustZSpaceCoherenceWaveBlock> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceCoherenceWaveBlock was moved into a container and can no longer be used",
            )
        })
    }

    fn take_inner(&mut self) -> PyResult<RustZSpaceCoherenceWaveBlock> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err(
                "ZSpaceCoherenceWaveBlock was moved into a container and can no longer be used",
            )
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZSpaceCoherenceWaveBlock {
    #[new]
    #[pyo3(signature = (dim, steps, memory, curvature, temperature, *, kernel_size=3, dilations=None))]
    pub fn new(
        dim: usize,
        steps: usize,
        memory: usize,
        curvature: f32,
        temperature: f32,
        kernel_size: usize,
        dilations: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let dilations = dilations.unwrap_or_else(|| vec![1, 2]);
        let inner = RustZSpaceCoherenceWaveBlock::new(
            dim,
            steps,
            memory,
            curvature,
            temperature,
            kernel_size,
            dilations,
        )
        .map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    pub fn infuse_text(&mut self, text: &str) -> PyResult<()> {
        self.inner_mut()?
            .infuse_text(text)
            .map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner()?
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner_mut()?
            .load_state_dict(&map)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    #[getter]
    pub fn dim(&self) -> PyResult<usize> {
        Ok(self.inner()?.dim())
    }

    #[getter]
    pub fn steps(&self) -> PyResult<usize> {
        Ok(self.inner()?.steps())
    }

    #[getter]
    pub fn memory(&self) -> PyResult<usize> {
        Ok(self.inner()?.memory())
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "Relu", unsendable)]
pub(crate) struct PyRelu {
    inner: Relu,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyRelu {
    #[new]
    pub fn new() -> Self {
        Self { inner: Relu::new() }
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self.inner.forward(&input.inner).map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "Sequential", unsendable)]
pub(crate) struct PySequential {
    inner: Sequential,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PySequential {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Sequential::new(),
        }
    }

    pub fn add(&mut self, layer: &Bound<PyAny>) -> PyResult<()> {
        let py = layer.py();

        if let Ok(handle) = layer.extract::<Py<PyLinear>>() {
            let mut linear = handle.bind(py).borrow_mut();
            let inner = linear.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyEmbedding>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyFeatureReorder2d>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PySpiralRnn>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyWaveGate>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyWaveRnn>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyZSpaceMixer>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyZSpaceSoftmax>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyZSpaceCoherenceScan>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyZSpaceCoherenceWaveBlock>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyRelu>>() {
            let relu = handle.bind(py).borrow();
            self.inner.push(relu.inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyIdentity>>() {
            let identity = handle.bind(py).borrow();
            self.inner.push(identity.inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyNonLiner>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyScaler>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyDropout>>() {
            let mut layer = handle.bind(py).borrow_mut();
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyPool2d>>() {
            let mut layer = handle.bind(py).borrow_mut();
            if layer.layout != Layout2d::Nchw {
                return Err(PyValueError::new_err(
                    "Sequential.add only supports Pool2d with layout='NCHW'",
                ));
            }
            let inner = layer.take_inner()?;
            match inner {
                PoolModule::Max(module) => self.inner.push(module),
                PoolModule::Avg(module) => self.inner.push(module),
            }
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyZPooling>>() {
            let mut layer = handle.bind(py).borrow_mut();
            if layer.layout != Layout2d::Nchw {
                return Err(PyValueError::new_err(
                    "Sequential.add only supports ZPooling with layout='NCHW'",
                ));
            }
            let inner = layer.take_inner()?;
            match inner {
                PoolModule::Max(module) => self.inner.push(module),
                PoolModule::Avg(module) => self.inner.push(module),
            }
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyZConv>>() {
            let mut layer = handle.bind(py).borrow_mut();
            if layer.layout != Layout2d::Nchw {
                return Err(PyValueError::new_err(
                    "Sequential.add only supports ZConv with layout='NCHW'",
                ));
            }
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        if let Ok(handle) = layer.extract::<Py<PyZConv6DA>>() {
            let mut layer = handle.bind(py).borrow_mut();
            if layer.layout != Layout3d::Ncdhw {
                return Err(PyValueError::new_err(
                    "Sequential.add only supports ZConv6DA with layout='NCDHW'",
                ));
            }
            let inner = layer.take_inner()?;
            self.inner.push(inner);
            return Ok(());
        }

        Err(PyTypeError::new_err(
            "Sequential.add expects a spiraltorch.nn layer (supported: Linear, Embedding, FeatureReorder2d, SpiralRnn, WaveGate, WaveRnn, ZSpaceMixer, ZSpaceSoftmax, ZSpaceCoherenceScan, ZSpaceCoherenceWaveBlock, Relu, Identity, NonLiner, Scaler, Dropout, Pool2d, ZPooling, ZConv, ZConv6DA)",
        ))
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self.inner.forward(&input.inner).map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.inner
                .attach_hypergrad(curvature, learning_rate)
                .map_err(tensor_err_to_py)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner.zero_accumulators().map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner.apply_step(fallback_lr).map_err(tensor_err_to_py)
    }

    pub fn infuse_text(&mut self, text: &str) -> PyResult<()> {
        self.inner.infuse_text(text).map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner.load_state_dict(&map).map_err(tensor_err_to_py)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "MeanSquaredError", unsendable)]
pub(crate) struct PyMeanSquaredError {
    inner: MeanSquaredError,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyMeanSquaredError {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: MeanSquaredError::new(),
        }
    }

    pub fn forward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner
            .forward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner
            .backward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[pyo3(signature = (prediction, target))]
    pub fn __call__(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        self.forward(prediction, target)
    }
}

#[cfg(feature = "nn")]
#[pyclass(
    module = "spiraltorch.nn",
    name = "CategoricalCrossEntropy",
    unsendable
)]
pub(crate) struct PyCategoricalCrossEntropy {
    inner: CategoricalCrossEntropy,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCategoricalCrossEntropy {
    #[new]
    #[pyo3(signature = (*, epsilon=None))]
    pub fn new(epsilon: Option<f32>) -> Self {
        let mut inner = CategoricalCrossEntropy::new();
        if let Some(epsilon) = epsilon {
            inner = inner.with_epsilon(epsilon);
        }
        Self { inner }
    }

    pub fn forward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner
            .forward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner
            .backward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[getter]
    pub fn epsilon(&self) -> f32 {
        self.inner.epsilon()
    }

    #[pyo3(signature = (prediction, target))]
    pub fn __call__(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        self.forward(prediction, target)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "HyperbolicCrossEntropy", unsendable)]
pub(crate) struct PyHyperbolicCrossEntropy {
    inner: HyperbolicCrossEntropy,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyHyperbolicCrossEntropy {
    #[new]
    #[pyo3(signature = (curvature, *, epsilon=None))]
    pub fn new(curvature: f32, epsilon: Option<f32>) -> PyResult<Self> {
        let inner = if let Some(epsilon) = epsilon {
            HyperbolicCrossEntropy::with_epsilon(curvature, epsilon)
        } else {
            HyperbolicCrossEntropy::new(curvature)
        }
        .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn forward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner
            .forward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner
            .backward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[getter]
    pub fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    #[getter]
    pub fn epsilon(&self) -> f32 {
        self.inner.epsilon()
    }

    #[pyo3(signature = (prediction, target))]
    pub fn __call__(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        self.forward(prediction, target)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "FocalLoss", unsendable)]
pub(crate) struct PyFocalLoss {
    inner: FocalLoss,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyFocalLoss {
    #[new]
    #[pyo3(signature = (alpha=0.25, gamma=2.0, *, epsilon=None))]
    pub fn new(alpha: f32, gamma: f32, epsilon: Option<f32>) -> Self {
        let mut inner = FocalLoss::new(alpha, gamma);
        if let Some(epsilon) = epsilon {
            inner = inner.with_epsilon(epsilon);
        }
        Self { inner }
    }

    pub fn forward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner
            .forward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner
            .backward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[getter]
    pub fn alpha(&self) -> f32 {
        self.inner.alpha
    }

    #[getter]
    pub fn gamma(&self) -> f32 {
        self.inner.gamma
    }

    #[pyo3(signature = (prediction, target))]
    pub fn __call__(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        self.forward(prediction, target)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ContrastiveLoss", unsendable)]
pub(crate) struct PyContrastiveLoss {
    inner: ContrastiveLoss,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyContrastiveLoss {
    #[new]
    #[pyo3(signature = (margin=1.0))]
    pub fn new(margin: f32) -> Self {
        Self {
            inner: ContrastiveLoss::new(margin),
        }
    }

    pub fn forward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner
            .forward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner
            .backward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[getter]
    pub fn margin(&self) -> f32 {
        self.inner.margin
    }

    #[pyo3(signature = (prediction, target))]
    pub fn __call__(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        self.forward(prediction, target)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "TripletLoss", unsendable)]
pub(crate) struct PyTripletLoss {
    inner: TripletLoss,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyTripletLoss {
    #[new]
    #[pyo3(signature = (margin=1.0))]
    pub fn new(margin: f32) -> Self {
        Self {
            inner: TripletLoss::new(margin),
        }
    }

    pub fn forward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner
            .forward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner
            .backward(&prediction.inner, &target.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[getter]
    pub fn margin(&self) -> f32 {
        self.inner.margin
    }

    #[pyo3(signature = (prediction, target))]
    pub fn __call__(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        self.forward(prediction, target)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "RoundtableConfig", unsendable)]
pub(crate) struct PyRoundtableConfig {
    inner: RustRoundtableConfig,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyRoundtableConfig {
    #[new]
    #[pyo3(signature = (*, top_k=8, mid_k=8, bottom_k=8, here_tolerance=1e-5))]
    pub fn new(top_k: u32, mid_k: u32, bottom_k: u32, here_tolerance: f32) -> PyResult<Self> {
        if !here_tolerance.is_finite() {
            return Err(PyValueError::new_err("here_tolerance must be finite"));
        }
        if top_k == 0 || mid_k == 0 || bottom_k == 0 {
            return Err(PyValueError::new_err("top_k/mid_k/bottom_k must be >= 1"));
        }
        let mut config = RustRoundtableConfig::default();
        config.top_k = top_k;
        config.mid_k = mid_k;
        config.bottom_k = bottom_k;
        config.here_tolerance = here_tolerance.max(0.0);
        Ok(Self { inner: config })
    }

    #[getter]
    pub fn top_k(&self) -> u32 {
        self.inner.top_k
    }

    #[getter]
    pub fn mid_k(&self) -> u32 {
        self.inner.mid_k
    }

    #[getter]
    pub fn bottom_k(&self) -> u32 {
        self.inner.bottom_k
    }

    #[getter]
    pub fn here_tolerance(&self) -> f32 {
        self.inner.here_tolerance
    }

    fn __repr__(&self) -> String {
        format!(
            "RoundtableConfig(top_k={}, mid_k={}, bottom_k={}, here_tolerance={})",
            self.inner.top_k, self.inner.mid_k, self.inner.bottom_k, self.inner.here_tolerance
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "SoftLogicConfig", unsendable)]
pub(crate) struct PySoftLogicConfig {
    inner: RustSoftLogicConfig,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PySoftLogicConfig {
    #[new]
    #[pyo3(signature = (*, inertia=0.65, inertia_min=0.15, inertia_drift_k=0.6, inertia_z_k=0.2, drift_gain=0.25, psi_gain=0.5, loss_gain=0.35, floor=0.25, scale_gain=0.2, region_gain=0.15, region_factor_gain=0.35, energy_equalize_gain=0.0, mean_normalize_gain=0.0))]
    pub fn new(
        inertia: f32,
        inertia_min: f32,
        inertia_drift_k: f32,
        inertia_z_k: f32,
        drift_gain: f32,
        psi_gain: f32,
        loss_gain: f32,
        floor: f32,
        scale_gain: f32,
        region_gain: f32,
        region_factor_gain: f32,
        energy_equalize_gain: f32,
        mean_normalize_gain: f32,
    ) -> PyResult<Self> {
        let mut config = RustSoftLogicConfig {
            inertia,
            inertia_min,
            inertia_drift_k,
            inertia_z_k,
            drift_gain,
            psi_gain,
            loss_gain,
            floor,
            scale_gain,
            region_gain,
            region_factor_gain,
            energy_equalize_gain,
            mean_normalize_gain,
        };
        config.clamp_inplace();
        Ok(Self { inner: config })
    }

    #[staticmethod]
    pub fn from_env() -> Self {
        Self {
            inner: RustSoftLogicConfig::default().with_env_overrides(),
        }
    }

    #[getter]
    pub fn inertia(&self) -> f32 {
        self.inner.inertia
    }

    #[getter]
    pub fn inertia_min(&self) -> f32 {
        self.inner.inertia_min
    }

    #[getter]
    pub fn inertia_drift_k(&self) -> f32 {
        self.inner.inertia_drift_k
    }

    #[getter]
    pub fn inertia_z_k(&self) -> f32 {
        self.inner.inertia_z_k
    }

    #[getter]
    pub fn drift_gain(&self) -> f32 {
        self.inner.drift_gain
    }

    #[getter]
    pub fn psi_gain(&self) -> f32 {
        self.inner.psi_gain
    }

    #[getter]
    pub fn loss_gain(&self) -> f32 {
        self.inner.loss_gain
    }

    #[getter]
    pub fn floor(&self) -> f32 {
        self.inner.floor
    }

    #[getter]
    pub fn scale_gain(&self) -> f32 {
        self.inner.scale_gain
    }

    #[getter]
    pub fn region_gain(&self) -> f32 {
        self.inner.region_gain
    }

    #[getter]
    pub fn region_factor_gain(&self) -> f32 {
        self.inner.region_factor_gain
    }

    #[getter]
    pub fn energy_equalize_gain(&self) -> f32 {
        self.inner.energy_equalize_gain
    }

    #[getter]
    pub fn mean_normalize_gain(&self) -> f32 {
        self.inner.mean_normalize_gain
    }

    fn __repr__(&self) -> String {
        format!(
            "SoftLogicConfig(inertia={}, inertia_min={}, inertia_drift_k={}, inertia_z_k={}, drift_gain={}, psi_gain={}, loss_gain={}, floor={}, scale_gain={}, region_gain={}, region_factor_gain={}, energy_equalize_gain={}, mean_normalize_gain={})",
            self.inner.inertia,
            self.inner.inertia_min,
            self.inner.inertia_drift_k,
            self.inner.inertia_z_k,
            self.inner.drift_gain,
            self.inner.psi_gain,
            self.inner.loss_gain,
            self.inner.floor,
            self.inner.scale_gain,
            self.inner.region_gain,
            self.inner.region_factor_gain,
            self.inner.energy_equalize_gain,
            self.inner.mean_normalize_gain,
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "RoundtableSchedule", unsendable)]
pub(crate) struct PyRoundtableSchedule {
    inner: RustRoundtableSchedule,
}

#[cfg(feature = "nn")]
impl PyRoundtableSchedule {
    fn new(inner: RustRoundtableSchedule) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyRoundtableSchedule {
    pub fn above(&self) -> PyRankPlan {
        PyRankPlan::from_plan(self.inner.above().clone())
    }

    pub fn here(&self) -> PyRankPlan {
        PyRankPlan::from_plan(self.inner.here().clone())
    }

    pub fn beneath(&self) -> PyRankPlan {
        PyRankPlan::from_plan(self.inner.beneath().clone())
    }

    fn __repr__(&self) -> String {
        let above = self.inner.above();
        let here = self.inner.here();
        let beneath = self.inner.beneath();
        format!(
            "RoundtableSchedule(above_k={}, here_k={}, beneath_k={})",
            above.k, here.k, beneath.k
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "EpochStats", unsendable)]
pub(crate) struct PyEpochStats {
    inner: RustEpochStats,
}

#[cfg(feature = "nn")]
impl PyEpochStats {
    fn new(inner: RustEpochStats) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyEpochStats {
    #[getter]
    pub fn batches(&self) -> usize {
        self.inner.batches
    }

    #[getter]
    pub fn total_loss(&self) -> f32 {
        self.inner.total_loss
    }

    #[getter]
    pub fn average_loss(&self) -> f32 {
        self.inner.average_loss
    }

    fn __repr__(&self) -> String {
        format!(
            "EpochStats(batches={}, total_loss={}, average_loss={})",
            self.inner.batches, self.inner.total_loss, self.inner.average_loss
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ModuleTrainer", unsendable)]
pub(crate) struct PyNnModuleTrainer {
    inner: RustModuleTrainer,
}

#[cfg(feature = "nn")]
fn collect_batches(py: Python<'_>, batches: &Bound<'_, PyAny>) -> PyResult<Vec<(Tensor, Tensor)>> {
    let mut out = Vec::new();
    for item in PyIterator::from_bound_object(batches)? {
        let item = item?;
        let (inputs, targets): (Py<PyTensor>, Py<PyTensor>) = item.extract()?;
        let input = inputs.bind(py).borrow().inner.clone();
        let target = targets.bind(py).borrow().inner.clone();
        out.push((input, target));
    }
    Ok(out)
}

#[cfg(feature = "nn")]
fn with_module_ref<R>(
    module: &Bound<'_, PyAny>,
    f: impl FnOnce(&dyn Module) -> Result<R, TensorError>,
) -> PyResult<R> {
    let py = module.py();
    if let Ok(handle) = module.extract::<Py<PyLinear>>() {
        let model = handle.bind(py).borrow();
        let inner = model.inner()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyEmbedding>>() {
        let model = handle.bind(py).borrow();
        let inner = model.inner()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyFeatureReorder2d>>() {
        let model = handle.bind(py).borrow();
        let inner = model.inner()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PySpiralRnn>>() {
        let model = handle.bind(py).borrow();
        let inner = model.inner()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyWaveGate>>() {
        let model = handle.bind(py).borrow();
        let inner = model.inner()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyWaveRnn>>() {
        let model = handle.bind(py).borrow();
        let inner = model.inner()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZSpaceMixer>>() {
        let model = handle.bind(py).borrow();
        let inner = model.inner()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZSpaceSoftmax>>() {
        let model = handle.bind(py).borrow();
        let inner = model.inner()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZSpaceCoherenceScan>>() {
        let model = handle.bind(py).borrow();
        let inner = model.inner()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZSpaceCoherenceWaveBlock>>() {
        let model = handle.bind(py).borrow();
        let inner = model.inner()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PySequential>>() {
        let model = handle.bind(py).borrow();
        return f(&model.inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyIdentity>>() {
        let model = handle.bind(py).borrow();
        return f(&model.inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyRelu>>() {
        let model = handle.bind(py).borrow();
        return f(&model.inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyNonLiner>>() {
        let model = handle.bind(py).borrow();
        return f(model.inner()?).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyScaler>>() {
        let model = handle.bind(py).borrow();
        return f(model.inner()?).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyDropout>>() {
        let model = handle.bind(py).borrow();
        return f(model.inner()?).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZConv>>() {
        let model = handle.bind(py).borrow();
        return f(model.inner()?).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZConv6DA>>() {
        let model = handle.bind(py).borrow();
        return f(model.inner()?).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZRelativityModule>>() {
        let model = handle.bind(py).borrow();
        return f(&model.inner).map_err(tensor_err_to_py);
    }

    Err(PyTypeError::new_err(
        "module must be a spiraltorch.nn module (supported: Linear, Embedding, FeatureReorder2d, SpiralRnn, WaveGate, WaveRnn, ZSpaceMixer, ZSpaceSoftmax, ZSpaceCoherenceScan, ZSpaceCoherenceWaveBlock, Sequential, Identity, Relu, NonLiner, Scaler, Dropout, ZConv, ZConv6DA, ZRelativityModule)",
    ))
}

#[cfg(feature = "nn")]
fn with_module_mut<R>(
    module: &Bound<'_, PyAny>,
    f: impl FnOnce(&mut dyn Module) -> Result<R, TensorError>,
) -> PyResult<R> {
    let py = module.py();
    if let Ok(handle) = module.extract::<Py<PyLinear>>() {
        let mut model = handle.bind(py).borrow_mut();
        let inner = model.inner_mut()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyEmbedding>>() {
        let mut model = handle.bind(py).borrow_mut();
        let inner = model.inner_mut()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyFeatureReorder2d>>() {
        let mut model = handle.bind(py).borrow_mut();
        let inner = model.inner_mut()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PySpiralRnn>>() {
        let mut model = handle.bind(py).borrow_mut();
        let inner = model.inner_mut()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyWaveGate>>() {
        let mut model = handle.bind(py).borrow_mut();
        let inner = model.inner_mut()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyWaveRnn>>() {
        let mut model = handle.bind(py).borrow_mut();
        let inner = model.inner_mut()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZSpaceMixer>>() {
        let mut model = handle.bind(py).borrow_mut();
        let inner = model.inner_mut()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZSpaceSoftmax>>() {
        let mut model = handle.bind(py).borrow_mut();
        let inner = model.inner_mut()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZSpaceCoherenceScan>>() {
        let mut model = handle.bind(py).borrow_mut();
        let inner = model.inner_mut()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZSpaceCoherenceWaveBlock>>() {
        let mut model = handle.bind(py).borrow_mut();
        let inner = model.inner_mut()?;
        return f(inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PySequential>>() {
        let mut model = handle.bind(py).borrow_mut();
        return f(&mut model.inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyIdentity>>() {
        let mut model = handle.bind(py).borrow_mut();
        return f(&mut model.inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyRelu>>() {
        let mut model = handle.bind(py).borrow_mut();
        return f(&mut model.inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyNonLiner>>() {
        let mut model = handle.bind(py).borrow_mut();
        return f(model.inner_mut()?).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyScaler>>() {
        let mut model = handle.bind(py).borrow_mut();
        return f(model.inner_mut()?).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyDropout>>() {
        let mut model = handle.bind(py).borrow_mut();
        return f(model.inner_mut()?).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZConv>>() {
        let mut model = handle.bind(py).borrow_mut();
        return f(model.inner_mut()?).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZConv6DA>>() {
        let mut model = handle.bind(py).borrow_mut();
        return f(model.inner_mut()?).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = module.extract::<Py<PyZRelativityModule>>() {
        let mut model = handle.bind(py).borrow_mut();
        return f(&mut model.inner).map_err(tensor_err_to_py);
    }

    Err(PyTypeError::new_err(
        "module must be a spiraltorch.nn module (supported: Linear, Embedding, FeatureReorder2d, SpiralRnn, WaveGate, WaveRnn, ZSpaceMixer, ZSpaceSoftmax, ZSpaceCoherenceScan, ZSpaceCoherenceWaveBlock, Sequential, Identity, Relu, NonLiner, Scaler, Dropout, ZConv, ZConv6DA, ZRelativityModule)",
    ))
}

#[cfg(feature = "nn")]
fn with_loss_mut<R>(
    loss: &Bound<'_, PyAny>,
    f: impl FnOnce(&mut dyn Loss) -> Result<R, TensorError>,
) -> PyResult<R> {
    let py = loss.py();
    if let Ok(handle) = loss.extract::<Py<PyMeanSquaredError>>() {
        let mut inner = handle.bind(py).borrow_mut();
        return f(&mut inner.inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = loss.extract::<Py<PyCategoricalCrossEntropy>>() {
        let mut inner = handle.bind(py).borrow_mut();
        return f(&mut inner.inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = loss.extract::<Py<PyHyperbolicCrossEntropy>>() {
        let mut inner = handle.bind(py).borrow_mut();
        return f(&mut inner.inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = loss.extract::<Py<PyFocalLoss>>() {
        let mut inner = handle.bind(py).borrow_mut();
        return f(&mut inner.inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = loss.extract::<Py<PyContrastiveLoss>>() {
        let mut inner = handle.bind(py).borrow_mut();
        return f(&mut inner.inner).map_err(tensor_err_to_py);
    }
    if let Ok(handle) = loss.extract::<Py<PyTripletLoss>>() {
        let mut inner = handle.bind(py).borrow_mut();
        return f(&mut inner.inner).map_err(tensor_err_to_py);
    }

    Err(PyTypeError::new_err(
        "loss must be a spiraltorch.nn loss (supported: MeanSquaredError, CategoricalCrossEntropy, HyperbolicCrossEntropy, FocalLoss, ContrastiveLoss, TripletLoss)",
    ))
}

#[cfg(feature = "nn")]
fn extract_state_dict(
    py: Python<'_>,
    target: &Bound<'_, PyAny>,
) -> PyResult<Option<std::collections::HashMap<String, Tensor>>> {
    if let Ok(entries) = target.extract::<Vec<(String, Py<PyTensor>)>>() {
        let mut state = std::collections::HashMap::new();
        for (name, tensor) in entries {
            state.insert(name, tensor.bind(py).borrow().inner.clone());
        }
        return Ok(Some(state));
    }
    if let Ok(entries) = target.extract::<std::collections::HashMap<String, Py<PyTensor>>>() {
        let mut state = std::collections::HashMap::new();
        for (name, tensor) in entries {
            state.insert(name, tensor.bind(py).borrow().inner.clone());
        }
        return Ok(Some(state));
    }
    Ok(None)
}

#[cfg(feature = "nn")]
fn state_dict_to_pylist(
    py: Python<'_>,
    state: std::collections::HashMap<String, Tensor>,
) -> PyResult<PyObject> {
    let mut entries: Vec<_> = state.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let list = PyList::empty_bound(py);
    for (name, tensor) in entries {
        list.append((name, Py::new(py, PyTensor::from_tensor(tensor))?))?;
    }
    Ok(list.into_py(py))
}

#[cfg(feature = "nn")]
#[pyfunction]
#[pyo3(signature = (module, path))]
fn save_json(module: &Bound<'_, PyAny>, path: &str) -> PyResult<()> {
    let py = module.py();
    if let Some(state) = extract_state_dict(py, module)? {
        return nn_io::save_state_dict_json(&state, path).map_err(tensor_err_to_py);
    }
    with_module_ref(module, |inner| nn_io::save_json(inner, path))
}

#[cfg(feature = "nn")]
#[pyfunction]
#[pyo3(signature = (module, path))]
fn load_json(module: &Bound<'_, PyAny>, path: &str) -> PyResult<PyObject> {
    if module.is_none() {
        let state = nn_io::load_state_dict_json(path).map_err(tensor_err_to_py)?;
        return state_dict_to_pylist(module.py(), state);
    }
    with_module_mut(module, |inner| nn_io::load_json(inner, path))?;
    Ok(module.py().None())
}

#[cfg(feature = "nn")]
#[pyfunction]
#[pyo3(signature = (module, path))]
fn save_bincode(module: &Bound<'_, PyAny>, path: &str) -> PyResult<()> {
    let py = module.py();
    if let Some(state) = extract_state_dict(py, module)? {
        return nn_io::save_state_dict_bincode(&state, path).map_err(tensor_err_to_py);
    }
    with_module_ref(module, |inner| nn_io::save_bincode(inner, path))
}

#[cfg(feature = "nn")]
#[pyfunction]
#[pyo3(signature = (module, path))]
fn load_bincode(module: &Bound<'_, PyAny>, path: &str) -> PyResult<PyObject> {
    if module.is_none() {
        let state = nn_io::load_state_dict_bincode(path).map_err(tensor_err_to_py)?;
        return state_dict_to_pylist(module.py(), state);
    }
    with_module_mut(module, |inner| nn_io::load_bincode(inner, path))?;
    Ok(module.py().None())
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyNnModuleTrainer {
    #[new]
    #[pyo3(signature = (*, backend="cpu", curvature=-1.0, hyper_learning_rate=1e-2, fallback_learning_rate=1e-2, lane_width=None, subgroup=None, max_workgroup=None, shared_mem_per_workgroup=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: &str,
        curvature: f32,
        hyper_learning_rate: f32,
        fallback_learning_rate: f32,
        lane_width: Option<u32>,
        subgroup: Option<bool>,
        max_workgroup: Option<u32>,
        shared_mem_per_workgroup: Option<u32>,
    ) -> PyResult<Self> {
        if !curvature.is_finite() {
            return Err(PyValueError::new_err("curvature must be finite"));
        }
        if hyper_learning_rate <= 0.0 || !hyper_learning_rate.is_finite() {
            return Err(PyValueError::new_err(
                "hyper_learning_rate must be positive and finite",
            ));
        }
        if fallback_learning_rate <= 0.0 || !fallback_learning_rate.is_finite() {
            return Err(PyValueError::new_err(
                "fallback_learning_rate must be positive and finite",
            ));
        }
        let backend_kind = parse_backend(Some(backend))?;
        let caps = build_caps(
            backend_kind,
            lane_width,
            subgroup,
            max_workgroup,
            shared_mem_per_workgroup,
        );
        Ok(Self {
            inner: RustModuleTrainer::new(
                caps,
                curvature,
                hyper_learning_rate,
                fallback_learning_rate,
            ),
        })
    }

    #[pyo3(signature = (rows, cols, config=None))]
    pub fn roundtable(
        &mut self,
        rows: u32,
        cols: u32,
        config: Option<&PyRoundtableConfig>,
    ) -> PyRoundtableSchedule {
        let config = config.map(|cfg| cfg.inner).unwrap_or_default();
        PyRoundtableSchedule::new(self.inner.roundtable(rows, cols, config))
    }

    #[pyo3(signature = (module, loss, batches, schedule))]
    pub fn train_epoch(
        &mut self,
        module: &Bound<PyAny>,
        loss: &Bound<PyAny>,
        batches: &Bound<PyAny>,
        schedule: &PyRoundtableSchedule,
    ) -> PyResult<PyEpochStats> {
        let py = module.py();
        let batches = collect_batches(py, batches)?;
        let stats = with_loss_mut(loss, |loss_inner| {
            with_module_mut(module, |module_inner| {
                self.inner
                    .train_epoch(module_inner, loss_inner, batches, &schedule.inner)
            })
            .map_err(|err| TensorError::Generic(err.to_string()))
        })?;

        Ok(PyEpochStats::new(stats))
    }

    #[pyo3(signature = (text, *, every="epoch", mode="blend"))]
    pub fn set_text_infusion(&mut self, text: &str, every: &str, mode: &str) -> PyResult<()> {
        let every = match every.to_ascii_lowercase().as_str() {
            "once" => TextInfusionEvery::Once,
            "epoch" => TextInfusionEvery::Epoch,
            "batch" => TextInfusionEvery::Batch,
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid every={other}. Expected 'once', 'epoch', or 'batch'"
                )))
            }
        };
        let mode = match mode.to_ascii_lowercase().as_str() {
            "blend" => TextInfusionMode::Blend,
            "separate" => TextInfusionMode::Separate,
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid mode={other}. Expected 'blend' or 'separate'"
                )))
            }
        };
        self.inner
            .set_text_infusion(text, every, mode)
            .map_err(tensor_err_to_py)
    }

    pub fn clear_text_infusion(&mut self) {
        self.inner.clear_text_infusion();
    }

    pub fn softlogic_config(&self) -> PySoftLogicConfig {
        PySoftLogicConfig {
            inner: self.inner.softlogic_config(),
        }
    }

    #[pyo3(signature = (config=None))]
    pub fn set_softlogic_config(&mut self, config: Option<&PySoftLogicConfig>) {
        match config {
            Some(config) => self.inner.set_softlogic_config(config.inner),
            None => self.inner.reset_softlogic(),
        }
    }

    pub fn reset_softlogic(&mut self) {
        self.inner.reset_softlogic();
    }

    pub fn enable_desire_telemetry(&mut self, bundle: &PyDesireTelemetryBundle) -> PyResult<()> {
        let rust_bundle = bundle.to_rust_bundle();
        self.inner.enable_desire_telemetry(&rust_bundle);
        Ok(())
    }

    pub fn disable_desire_roundtable_bridge(&mut self) {
        self.inner.disable_desire_roundtable_bridge();
    }

    pub fn desire_roundtable_summary(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let summary = self.inner.desire_roundtable_summary();
        match summary {
            Some(summary) => Ok(Some(desire_roundtable_summary_to_py(py, &summary)?)),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        "ModuleTrainer(...)".to_string()
    }
}

#[cfg(feature = "nn")]
fn system_time_ms(timestamp: SystemTime) -> u128 {
    timestamp
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis()
}

#[cfg(feature = "nn")]
fn desire_phase_label(phase: DesirePhase) -> &'static str {
    match phase {
        DesirePhase::Observation => "observation",
        DesirePhase::Injection => "injection",
        DesirePhase::Integration => "integration",
    }
}

#[cfg(feature = "nn")]
fn desire_weights_to_py(py: Python<'_>, weights: &DesireWeights) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("alpha", weights.alpha)?;
    dict.set_item("beta", weights.beta)?;
    dict.set_item("gamma", weights.gamma)?;
    dict.set_item("lambda", weights.lambda)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "nn")]
fn desire_trainer_summary_to_py(
    py: Python<'_>,
    summary: &st_nn::DesireTrainerSummary,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("total", summary.total)?;
    dict.set_item("observation", summary.observation)?;
    dict.set_item("injection", summary.injection)?;
    dict.set_item("integration", summary.integration)?;
    dict.set_item("triggers", summary.triggers)?;
    dict.set_item("mean_entropy", summary.mean_entropy)?;
    dict.set_item("mean_temperature", summary.mean_temperature)?;
    dict.set_item("mean_penalty", summary.mean_penalty)?;
    dict.set_item("mean_alpha", summary.mean_alpha)?;
    dict.set_item("mean_beta", summary.mean_beta)?;
    dict.set_item("mean_gamma", summary.mean_gamma)?;
    dict.set_item("mean_lambda", summary.mean_lambda)?;
    dict.set_item("trigger_mean_penalty", summary.trigger_mean_penalty)?;
    dict.set_item("trigger_mean_entropy", summary.trigger_mean_entropy)?;
    dict.set_item("trigger_mean_temperature", summary.trigger_mean_temperature)?;
    dict.set_item("trigger_mean_samples", summary.trigger_mean_samples)?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "nn")]
fn desire_roundtable_summary_to_py(
    py: Python<'_>,
    summary: &DesireRoundtableSummary,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("steps", summary.steps)?;
    dict.set_item("triggers", summary.triggers)?;
    dict.set_item("mean_entropy", summary.mean_entropy)?;
    dict.set_item("mean_temperature", summary.mean_temperature)?;
    dict.set_item("mean_alpha", summary.mean_alpha)?;
    dict.set_item("mean_beta", summary.mean_beta)?;
    dict.set_item("mean_gamma", summary.mean_gamma)?;
    dict.set_item("mean_lambda", summary.mean_lambda)?;
    dict.set_item("mean_above", summary.mean_above)?;
    dict.set_item("mean_here", summary.mean_here)?;
    dict.set_item("mean_beneath", summary.mean_beneath)?;
    dict.set_item("mean_drift", summary.mean_drift)?;
    dict.set_item("last_timestamp_ms", system_time_ms(summary.last_timestamp))?;
    Ok(dict.into_py(py))
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "DesireTrainerBridge", unsendable)]
pub(crate) struct PyDesireTrainerBridge {
    inner: DesireTrainerBridge,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDesireTrainerBridge {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: DesireTrainerBridge::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn drain_summary(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let summary = self.inner.drain_summary().map_err(tensor_err_to_py)?;
        match summary {
            Some(summary) => Ok(Some(desire_trainer_summary_to_py(py, &summary)?)),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!("DesireTrainerBridge(len={})", self.inner.len())
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "DesireRoundtableBridge", unsendable)]
pub(crate) struct PyDesireRoundtableBridge {
    inner: DesireRoundtableBridge,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDesireRoundtableBridge {
    #[new]
    #[pyo3(signature = (*, blend=0.35, drift_gain=0.35))]
    pub fn new(blend: f32, drift_gain: f32) -> Self {
        let mut inner = DesireRoundtableBridge::new();
        inner.set_blend(blend);
        inner.set_drift_gain(drift_gain);
        Self { inner }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[getter]
    pub fn blend(&self) -> f32 {
        self.inner.blend()
    }

    #[setter]
    pub fn set_blend(&mut self, blend: f32) {
        self.inner.set_blend(blend);
    }

    #[getter]
    pub fn drift_gain(&self) -> f32 {
        self.inner.drift_gain()
    }

    #[setter]
    pub fn set_drift_gain(&mut self, drift_gain: f32) {
        self.inner.set_drift_gain(drift_gain);
    }

    pub fn impulse(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let impulse = self.inner.impulse().map_err(tensor_err_to_py)?;
        let Some(impulse) = impulse else {
            return Ok(None);
        };
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "multipliers",
            (
                impulse.multipliers.0,
                impulse.multipliers.1,
                impulse.multipliers.2,
            ),
        )?;
        dict.set_item("drift", impulse.drift)?;
        dict.set_item("timestamp_ms", system_time_ms(impulse.timestamp))?;
        Ok(Some(dict.into_py(py)))
    }

    pub fn drain_summary(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let summary = self.inner.drain_summary().map_err(tensor_err_to_py)?;
        match summary {
            Some(summary) => Ok(Some(desire_roundtable_summary_to_py(py, &summary)?)),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DesireRoundtableBridge(blend={}, drift_gain={}, len={})",
            self.inner.blend(),
            self.inner.drift_gain(),
            self.inner.len()
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "DesireTelemetryBundle", unsendable)]
pub(crate) struct PyDesireTelemetryBundle {
    trainer: Option<DesireTrainerBridge>,
    roundtable: Option<DesireRoundtableBridge>,
}

#[cfg(feature = "nn")]
impl PyDesireTelemetryBundle {
    fn to_rust_bundle(&self) -> DesireTelemetryBundle {
        let mut bundle = DesireTelemetryBundle::new();
        if let Some(bridge) = self.trainer.as_ref() {
            bundle = bundle.with_trainer_bridge(bridge);
        }
        if let Some(bridge) = self.roundtable.as_ref() {
            bundle = bundle.with_roundtable_bridge(bridge);
        }
        bundle
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDesireTelemetryBundle {
    #[new]
    #[pyo3(signature = (*, trainer=true, roundtable=true, blend=0.35, drift_gain=0.35))]
    pub fn new(trainer: bool, roundtable: bool, blend: f32, drift_gain: f32) -> Self {
        let trainer_bridge = trainer.then(DesireTrainerBridge::new);
        let roundtable_bridge = roundtable.then(|| {
            let mut bridge = DesireRoundtableBridge::new();
            bridge.set_blend(blend);
            bridge.set_drift_gain(drift_gain);
            bridge
        });
        Self {
            trainer: trainer_bridge,
            roundtable: roundtable_bridge,
        }
    }

    pub fn has_trainer(&self) -> bool {
        self.trainer.is_some()
    }

    pub fn has_roundtable(&self) -> bool {
        self.roundtable.is_some()
    }

    pub fn trainer_bridge(&self, py: Python<'_>) -> PyResult<Option<Py<PyDesireTrainerBridge>>> {
        match self.trainer.as_ref() {
            Some(bridge) => Ok(Some(Py::new(
                py,
                PyDesireTrainerBridge {
                    inner: bridge.clone(),
                },
            )?)),
            None => Ok(None),
        }
    }

    pub fn roundtable_bridge(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PyDesireRoundtableBridge>>> {
        match self.roundtable.as_ref() {
            Some(bridge) => Ok(Some(Py::new(
                py,
                PyDesireRoundtableBridge {
                    inner: bridge.clone(),
                },
            )?)),
            None => Ok(None),
        }
    }

    pub fn drain_trainer_summary(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let Some(bridge) = self.trainer.as_ref() else {
            return Ok(None);
        };
        let summary = bridge.drain_summary().map_err(tensor_err_to_py)?;
        match summary {
            Some(summary) => Ok(Some(desire_trainer_summary_to_py(py, &summary)?)),
            None => Ok(None),
        }
    }

    pub fn drain_roundtable_summary(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let Some(bridge) = self.roundtable.as_ref() else {
            return Ok(None);
        };
        let summary = bridge.drain_summary().map_err(tensor_err_to_py)?;
        match summary {
            Some(summary) => Ok(Some(desire_roundtable_summary_to_py(py, &summary)?)),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DesireTelemetryBundle(trainer={}, roundtable={})",
            self.trainer.is_some(),
            self.roundtable.is_some()
        )
    }
}

#[cfg(feature = "nn")]
fn build_simple_desire_automation(
    vocab_size: usize,
    concepts: usize,
    alpha_end: f32,
    beta_end: f32,
    gamma: f32,
    lambda_: f32,
    observation_horizon: Option<u64>,
    integration_horizon: Option<u64>,
    target_entropy: f32,
    eta: f32,
    temp_min: f32,
    temp_max: f32,
    score_thresh: f32,
    min_samples: usize,
    cooldown_sec: u64,
) -> Result<DesireAutomation, TensorError> {
    if vocab_size == 0 {
        return Err(TensorError::InvalidValue {
            label: "vocab_size must be >= 1",
        });
    }
    if concepts == 0 {
        return Err(TensorError::InvalidValue {
            label: "concepts must be >= 1",
        });
    }

    let epsilon = 1e-6;
    let syn = SparseKernel::from_rows((0..vocab_size).map(|i| vec![(i, 1.0)]).collect(), epsilon)?;
    let par = SparseKernel::from_rows((0..vocab_size).map(|i| vec![(i, 1.0)]).collect(), epsilon)?;
    let geometry = SymbolGeometry::new(syn, par)?;

    let repression = RepressionField::new(
        (0..vocab_size)
            .map(|i| 0.05 + (i % 7) as f32 * 0.01)
            .collect(),
    )?;

    let concept_kernel =
        SparseKernel::from_rows((0..concepts).map(|i| vec![(i, 1.0)]).collect(), epsilon)?;

    let primary: f32 = if concepts == 1 { 1.0 } else { 0.72 };
    let secondary: f32 = if concepts == 1 { 0.0 } else { 0.28 };
    let mut log_pi: Vec<Vec<(usize, f32)>> = Vec::with_capacity(vocab_size);
    let mut row_sums: Vec<f32> = Vec::with_capacity(vocab_size);
    let mut col_sums = vec![0.0f32; concepts];
    for token in 0..vocab_size {
        let mut row: Vec<(usize, f32)> = Vec::with_capacity(if concepts == 1 { 1 } else { 2 });
        let a = token % concepts;
        row.push((a, primary.max(f32::EPSILON).ln()));
        col_sums[a] += primary;
        if concepts > 1 {
            let b = (token + 1) % concepts;
            if b != a {
                row.push((b, secondary.max(f32::EPSILON).ln()));
                col_sums[b] += secondary;
            } else {
                col_sums[a] += secondary;
            }
        }
        log_pi.push(row);
        row_sums.push(1.0);
    }
    for sum in &mut col_sums {
        if *sum <= 0.0 {
            *sum = 1.0;
        }
    }

    let anchors = std::collections::HashSet::new();
    let semantics =
        SemanticBridge::new(log_pi, row_sums, col_sums, anchors, epsilon, concept_kernel)?;
    let controller = TemperatureController::new(1.0, target_entropy, eta, temp_min, temp_max);
    let desire = DesireLagrangian::new(geometry, repression, semantics, controller)?
        .with_alpha_schedule(warmup(0.0, alpha_end, 1))
        .with_beta_schedule(warmup(0.0, beta_end, 1))
        .with_gamma_schedule(constant(gamma))
        .with_lambda_schedule(constant(lambda_))
        .with_observation_horizon(observation_horizon)
        .with_integration_horizon(integration_horizon);

    let cfg = SelfRewriteCfg {
        score_thresh,
        min_samples: min_samples.max(1),
        cooldown_sec,
    };
    Ok(DesireAutomation::new(desire, cfg))
}

#[cfg(feature = "nn")]
fn maxwell_pulse_from_any(pulse: &Bound<'_, PyAny>) -> PyResult<MaxwellZPulse> {
    let py = pulse.py();

    if let Ok(handle) = pulse.extract::<Py<PyMaxwellPulse>>() {
        return Ok(handle.bind(py).borrow().to_pulse());
    }

    if let Ok(dict) = pulse.downcast::<PyDict>() {
        let get = |key: &str| -> PyResult<Bound<'_, PyAny>> {
            dict.get_item(key)?
                .ok_or_else(|| PyValueError::new_err(format!("pulse missing field '{key}'")))
        };
        let blocks: u64 = get("blocks")?.extract()?;
        let mean: f64 = get("mean")?.extract()?;
        let standard_error: f64 = get("standard_error")?.extract()?;
        let z_score: f64 = get("z_score")?.extract()?;
        let band_energy: (f32, f32, f32) = get("band_energy")?.extract()?;
        let z_bias: f32 = get("z_bias")?.extract()?;
        return Ok(MaxwellZPulse {
            blocks,
            mean,
            standard_error,
            z_score,
            band_energy,
            z_bias,
        });
    }

    let blocks = pulse.getattr("blocks").and_then(|v| v.extract::<u64>());
    let mean = pulse.getattr("mean").and_then(|v| v.extract::<f64>());
    let standard_error = pulse
        .getattr("standard_error")
        .and_then(|v| v.extract::<f64>());
    let z_score = pulse.getattr("z_score").and_then(|v| v.extract::<f64>());
    let band_energy = pulse
        .getattr("band_energy")
        .and_then(|v| v.extract::<(f32, f32, f32)>());
    let z_bias = pulse.getattr("z_bias").and_then(|v| v.extract::<f32>());

    match (blocks, mean, standard_error, z_score, band_energy, z_bias) {
        (Ok(blocks), Ok(mean), Ok(standard_error), Ok(z_score), Ok(band_energy), Ok(z_bias)) => {
            Ok(MaxwellZPulse {
                blocks,
                mean,
                standard_error,
                z_score,
                band_energy,
                z_bias,
            })
        }
        _ => Err(PyTypeError::new_err(concat!(
            "pulse must be spiraltorch.spiralk.MaxwellPulse, a mapping with keys ",
            "'blocks','mean','standard_error','z_score','band_energy','z_bias', ",
            "or an object exposing those attributes"
        ))),
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "NarrativeSummary")]
#[derive(Clone)]
pub(crate) struct PyNarrativeSummary {
    inner: NarrativeSummary,
}

#[cfg(feature = "nn")]
impl From<NarrativeSummary> for PyNarrativeSummary {
    fn from(inner: NarrativeSummary) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyNarrativeSummary {
    #[getter]
    pub fn channel(&self) -> &str {
        &self.inner.channel
    }

    #[getter]
    pub fn dominant_tag(&self) -> Option<String> {
        self.inner.dominant_tag.clone()
    }

    #[getter]
    pub fn tags(&self) -> Vec<String> {
        self.inner.tags.clone()
    }

    #[getter]
    pub fn intensity(&self) -> f32 {
        self.inner.intensity
    }

    #[getter]
    pub fn amplitude(&self) -> f32 {
        self.inner.amplitude
    }

    #[getter]
    pub fn phase(&self) -> f32 {
        self.inner.phase
    }

    #[getter]
    pub fn coherence(&self) -> f32 {
        self.inner.coherence
    }

    #[getter]
    pub fn decoherence(&self) -> f32 {
        self.inner.decoherence
    }

    #[getter]
    pub fn emphasis(&self) -> f32 {
        self.inner.emphasis
    }

    pub fn describe(&self) -> String {
        self.inner.describe()
    }

    fn __repr__(&self) -> String {
        format!("NarrativeSummary({})", self.inner.describe())
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "NarrativeHint")]
#[derive(Clone)]
pub(crate) struct PyNarrativeHint {
    inner: NarrativeHint,
}

#[cfg(feature = "nn")]
impl From<NarrativeHint> for PyNarrativeHint {
    fn from(inner: NarrativeHint) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyNarrativeHint {
    #[getter]
    pub fn channel(&self) -> &str {
        self.inner.channel()
    }

    #[getter]
    pub fn tags(&self) -> Vec<String> {
        self.inner.tags().to_vec()
    }

    pub fn dominant_tag(&self) -> Option<String> {
        self.inner.dominant_tag().map(|tag| tag.to_string())
    }

    #[getter]
    pub fn intensity(&self) -> f32 {
        self.inner.intensity()
    }

    #[getter]
    pub fn amplitude(&self) -> f32 {
        self.inner.amplitude()
    }

    #[getter]
    pub fn phase(&self) -> f32 {
        self.inner.phase()
    }

    #[getter]
    pub fn coherence(&self) -> f32 {
        self.inner.coherence()
    }

    #[getter]
    pub fn decoherence(&self) -> f32 {
        self.inner.decoherence()
    }

    pub fn collapsed_tag(&self) -> Option<String> {
        self.inner.collapsed_tag().map(|tag| tag.to_string())
    }

    pub fn quantum_emphasis(&self) -> f32 {
        self.inner.quantum_emphasis()
    }

    pub fn summary(&self) -> PyNarrativeSummary {
        PyNarrativeSummary::from(self.inner.summary())
    }

    pub fn describe(&self) -> String {
        self.inner.summary().describe()
    }

    fn __repr__(&self) -> String {
        format!("NarrativeHint({})", self.inner.summary().describe())
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "MaxwellDesireBridge", unsendable)]
pub(crate) struct PyMaxwellDesireBridge {
    inner: MaxwellDesireBridge,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyMaxwellDesireBridge {
    #[new]
    #[pyo3(signature = (*, smoothing=1e-4, magnitude_floor=0.0))]
    pub fn new(smoothing: f32, magnitude_floor: f32) -> Self {
        let mut bridge = MaxwellDesireBridge::new();
        bridge.set_smoothing(smoothing);
        bridge.set_magnitude_floor(magnitude_floor);
        Self { inner: bridge }
    }

    #[getter]
    pub fn smoothing(&self) -> f32 {
        self.inner.smoothing()
    }

    #[getter]
    pub fn magnitude_floor(&self) -> f32 {
        self.inner.magnitude_floor()
    }

    pub fn set_smoothing(&mut self, smoothing: f32) {
        self.inner.set_smoothing(smoothing);
    }

    pub fn set_magnitude_floor(&mut self, floor: f32) {
        self.inner.set_magnitude_floor(floor);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn contains(&self, channel: &str) -> bool {
        self.inner.contains(channel)
    }

    pub fn channel_names(&self) -> Vec<String> {
        self.inner.channel_names()
    }

    pub fn register_channel(&mut self, channel: &str, window: Vec<(usize, f32)>) -> PyResult<()> {
        self.inner
            .register_channel(channel.to_string(), window)
            .map_err(tensor_err_to_py)
    }

    pub fn register_channel_with_narrative(
        &mut self,
        channel: &str,
        window: Vec<(usize, f32)>,
        tags: Vec<String>,
    ) -> PyResult<()> {
        self.inner
            .register_channel_with_narrative(channel.to_string(), window, tags)
            .map_err(tensor_err_to_py)
    }

    pub fn set_narrative_tags(&mut self, channel: &str, tags: Vec<String>) -> PyResult<()> {
        self.inner
            .set_narrative_tags(channel, tags)
            .map_err(tensor_err_to_py)
    }

    pub fn narrative_tags(&self, channel: &str) -> Option<Vec<String>> {
        self.inner.narrative_tags(channel).map(|tags| tags.to_vec())
    }

    pub fn set_narrative_gain(&mut self, channel: &str, gain: f32) -> PyResult<()> {
        self.inner
            .set_narrative_gain(channel, gain)
            .map_err(tensor_err_to_py)
    }

    pub fn narrative_gain(&self, channel: &str) -> Option<f32> {
        self.inner.narrative_gain(channel)
    }

    pub fn hint_for(
        &self,
        channel: &str,
        pulse: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<(usize, f32)>>> {
        let pulse = maxwell_pulse_from_any(pulse)?;
        let Some(hint) = self.inner.hint_for(channel, &pulse) else {
            return Ok(None);
        };
        match hint {
            ConceptHint::Window(window) => Ok(Some(window)),
            ConceptHint::Distribution(_) => Err(PyValueError::new_err(
                "unexpected ConceptHint::Distribution from MaxwellDesireBridge",
            )),
        }
    }

    pub fn narrative_for(
        &self,
        py: Python<'_>,
        channel: &str,
        pulse: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyNarrativeHint>>> {
        let pulse = maxwell_pulse_from_any(pulse)?;
        match self.inner.narrative_for(channel, &pulse) {
            Some(hint) => Ok(Some(Py::new(py, PyNarrativeHint::from(hint))?)),
            None => Ok(None),
        }
    }

    pub fn emit(
        &self,
        py: Python<'_>,
        channel: &str,
        pulse: &Bound<'_, PyAny>,
    ) -> PyResult<Option<PyObject>> {
        let pulse = maxwell_pulse_from_any(pulse)?;
        let Some((hint, narrative)) = self.inner.emit(channel, &pulse) else {
            return Ok(None);
        };
        let dict = PyDict::new_bound(py);
        match hint {
            ConceptHint::Window(window) => {
                dict.set_item("window", window)?;
            }
            ConceptHint::Distribution(dist) => {
                dict.set_item("distribution", dist)?;
            }
        }
        if let Some(narrative) = narrative {
            dict.set_item("narrative", Py::new(py, PyNarrativeHint::from(narrative))?)?;
        } else {
            dict.set_item("narrative", py.None())?;
        }
        Ok(Some(dict.into_py(py)))
    }

    fn __repr__(&self) -> String {
        format!(
            "MaxwellDesireBridge(len={}, smoothing={:.2e}, magnitude_floor={:.2e})",
            self.inner.len(),
            self.inner.smoothing(),
            self.inner.magnitude_floor()
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "DesirePipeline", unsendable)]
pub(crate) struct PyDesirePipeline {
    inner: DesirePipeline,
    vocab_size: usize,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDesirePipeline {
    #[new]
    #[pyo3(signature = (
        vocab_size,
        *,
        concepts=3,
        alpha_end=0.2,
        beta_end=0.1,
        gamma=0.04,
        lambda_=0.02,
        observation_horizon=1,
        integration_horizon=2,
        target_entropy=0.8,
        eta=0.4,
        temp_min=0.4,
        temp_max=1.6,
        score_thresh=0.0,
        min_samples=2,
        cooldown_sec=0,
        bundle=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        concepts: usize,
        alpha_end: f32,
        beta_end: f32,
        gamma: f32,
        lambda_: f32,
        observation_horizon: u64,
        integration_horizon: u64,
        target_entropy: f32,
        eta: f32,
        temp_min: f32,
        temp_max: f32,
        score_thresh: f32,
        min_samples: usize,
        cooldown_sec: u64,
        bundle: Option<&PyDesireTelemetryBundle>,
    ) -> PyResult<Self> {
        let automation = build_simple_desire_automation(
            vocab_size,
            concepts,
            alpha_end,
            beta_end,
            gamma,
            lambda_,
            Some(observation_horizon),
            Some(integration_horizon),
            target_entropy,
            eta,
            temp_min,
            temp_max,
            score_thresh,
            min_samples,
            cooldown_sec,
        )
        .map_err(tensor_err_to_py)?;

        let mut builder = DesirePipeline::builder(automation);
        if let Some(bundle) = bundle {
            let rust_bundle = bundle.to_rust_bundle();
            builder = builder.with_telemetry_bundle(&rust_bundle);
        }
        let pipeline = builder.build();
        Ok(Self {
            inner: pipeline,
            vocab_size,
        })
    }

    pub fn sink_count(&self) -> usize {
        self.inner.sink_count()
    }

    #[pyo3(signature = (logits, previous_token, concept=None, window=None))]
    pub fn step(
        &mut self,
        py: Python<'_>,
        logits: Vec<f32>,
        previous_token: usize,
        concept: Option<Vec<f32>>,
        window: Option<Vec<(usize, f32)>>,
    ) -> PyResult<PyObject> {
        if logits.len() != self.vocab_size {
            return Err(PyValueError::new_err(format!(
                "logits length mismatch (expected {}, got {})",
                self.vocab_size,
                logits.len()
            )));
        }
        if previous_token >= self.vocab_size {
            return Err(PyValueError::new_err(format!(
                "previous_token out of range (expected < {}, got {})",
                self.vocab_size, previous_token
            )));
        }
        if concept.is_some() && window.is_some() {
            return Err(PyValueError::new_err(
                "concept and window are mutually exclusive; supply only one",
            ));
        }
        if let Some(window) = window.as_ref() {
            for (token, weight) in window {
                if *token >= self.vocab_size {
                    return Err(PyValueError::new_err(format!(
                        "window token out of range (expected < {}, got {})",
                        self.vocab_size, token
                    )));
                }
                if !weight.is_finite() || *weight < 0.0 {
                    return Err(PyValueError::new_err(
                        "window weights must be finite and non-negative",
                    ));
                }
            }
        }

        let concept_hint = match window {
            Some(window) => ConceptHint::Window(window),
            None => ConceptHint::Distribution(concept.unwrap_or_default()),
        };
        let step = self
            .inner
            .step_realtime(&logits, previous_token, &concept_hint)
            .map_err(tensor_err_to_py)?;

        let dict = PyDict::new_bound(py);
        dict.set_item("phase", desire_phase_label(step.solution.phase))?;
        dict.set_item("temperature", step.solution.temperature)?;
        dict.set_item("entropy", step.solution.entropy)?;
        dict.set_item("hypergrad_penalty", step.solution.hypergrad_penalty)?;
        dict.set_item("weights", desire_weights_to_py(py, &step.solution.weights)?)?;
        dict.set_item("indices", step.solution.indices.clone())?;
        dict.set_item("probabilities", step.solution.probabilities.clone())?;
        dict.set_item("logit_offsets", step.solution.logit_offsets.clone())?;
        dict.set_item("triggered", step.trigger.is_some())?;
        if let Some(trigger) = step.trigger.as_ref() {
            let trig = PyDict::new_bound(py);
            trig.set_item("mean_penalty", trigger.mean_penalty)?;
            trig.set_item("mean_entropy", trigger.mean_entropy)?;
            trig.set_item("temperature", trigger.temperature)?;
            trig.set_item("samples", trigger.samples)?;
            dict.set_item("trigger", trig)?;
        }
        Ok(dict.into_py(py))
    }

    pub fn flush(&mut self) -> PyResult<()> {
        self.inner.flush().map_err(tensor_err_to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "DesirePipeline(vocab_size={}, sinks={})",
            self.vocab_size,
            self.inner.sink_count()
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "NonLiner", unsendable)]
pub(crate) struct PyNonLiner {
    inner: Option<NonLiner>,
}

#[cfg(feature = "nn")]
impl PyNonLiner {
    fn inner(&self) -> PyResult<&NonLiner> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("NonLiner was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut NonLiner> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("NonLiner was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<NonLiner> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("NonLiner was moved into a container and can no longer be used")
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyNonLiner {
    #[new]
    #[pyo3(signature = (name, features, *, activation="tanh", slope=1.0, gain=1.0, bias=0.0, curvature=None, z_scale=None, retention=0.0))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        features: usize,
        activation: &str,
        slope: f32,
        gain: f32,
        bias: f32,
        curvature: Option<f32>,
        z_scale: Option<f32>,
        retention: f32,
    ) -> PyResult<Self> {
        let activation = parse_non_liner_activation(activation)?;
        let geometry = if let Some(curvature) = curvature {
            if curvature == 0.0 {
                return Err(PyValueError::new_err(
                    "curvature must be non-zero for non-Euclidean geometry",
                ));
            }
            let scale = match z_scale {
                Some(value) => ZScale::new(value)
                    .ok_or_else(|| PyValueError::new_err("z_scale must be positive and finite"))?,
                None => ZScale::ONE,
            };
            if curvature < 0.0 {
                let config = NonLinerHyperbolicConfig::new(curvature, scale, retention)
                    .map_err(tensor_err_to_py)?;
                NonLinerGeometry::hyperbolic(config)
            } else {
                let config = NonLinerEllipticConfig::new(curvature, scale, retention)
                    .map_err(tensor_err_to_py)?;
                NonLinerGeometry::elliptic(config)
            }
        } else {
            if z_scale.is_some() {
                return Err(PyValueError::new_err("z_scale requires curvature"));
            }
            if retention != 0.0 {
                return Err(PyValueError::new_err("retention requires curvature"));
            }
            NonLinerGeometry::Euclidean
        };
        let inner =
            NonLiner::with_geometry(name, features, activation, slope, gain, bias, geometry)
                .map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    pub fn reset_metrics(&self) -> PyResult<()> {
        self.inner()?.reset_metrics();
        Ok(())
    }

    #[pyo3(signature = (*, curvature=None, z_scale=None, retention=None))]
    pub fn configure_geometry(
        &mut self,
        curvature: Option<f32>,
        z_scale: Option<f32>,
        retention: Option<f32>,
    ) -> PyResult<()> {
        let geometry = if let Some(curvature) = curvature {
            if curvature == 0.0 {
                return Err(PyValueError::new_err(
                    "curvature must be non-zero for non-Euclidean geometry",
                ));
            }
            let base = self.inner()?.geometry();
            let scale = match z_scale {
                Some(value) => ZScale::new(value)
                    .ok_or_else(|| PyValueError::new_err("z_scale must be positive and finite"))?,
                None => base.z_scale().unwrap_or(ZScale::ONE),
            };
            let retention = retention.unwrap_or_else(|| base.retention().unwrap_or(0.0));
            if curvature < 0.0 {
                let config = NonLinerHyperbolicConfig::new(curvature, scale, retention)
                    .map_err(tensor_err_to_py)?;
                NonLinerGeometry::hyperbolic(config)
            } else {
                let config = NonLinerEllipticConfig::new(curvature, scale, retention)
                    .map_err(tensor_err_to_py)?;
                NonLinerGeometry::elliptic(config)
            }
        } else {
            if z_scale.is_some() {
                return Err(PyValueError::new_err("z_scale requires curvature"));
            }
            if let Some(retention) = retention {
                if retention != 0.0 {
                    return Err(PyValueError::new_err("retention requires curvature"));
                }
            }
            NonLinerGeometry::Euclidean
        };
        self.inner_mut()?.set_geometry(geometry);
        Ok(())
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner_mut()?
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.inner_mut()?
                .attach_hypergrad(curvature, learning_rate)
                .map_err(tensor_err_to_py)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner_mut()?
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner_mut()?
            .zero_accumulators()
            .map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner_mut()?
            .apply_step(fallback_lr)
            .map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner()?
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner_mut()?
            .load_state_dict(&map)
            .map_err(tensor_err_to_py)
    }

    #[getter]
    pub fn activation(&self) -> PyResult<String> {
        let activation = self.inner()?.activation();
        Ok(match activation {
            NonLinerActivation::Tanh => "tanh".to_string(),
            NonLinerActivation::Sigmoid => "sigmoid".to_string(),
            NonLinerActivation::Softsign => "softsign".to_string(),
        })
    }

    #[getter]
    pub fn curvature(&self) -> PyResult<Option<f32>> {
        Ok(self.inner()?.geometry().curvature())
    }

    #[getter]
    pub fn z_scale(&self) -> PyResult<Option<f32>> {
        Ok(self
            .inner()?
            .geometry()
            .z_scale()
            .map(|scale| scale.value()))
    }

    #[getter]
    pub fn retention(&self) -> PyResult<Option<f32>> {
        Ok(self.inner()?.geometry().retention())
    }

    #[getter]
    pub fn psi_drift(&self) -> PyResult<Option<f32>> {
        Ok(self.inner()?.psi_probe())
    }

    #[getter]
    pub fn last_hyperbolic_radius(&self) -> PyResult<Option<f32>> {
        Ok(self.inner()?.last_hyperbolic_radius())
    }

    #[getter]
    pub fn gain(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner()?.gain().value().clone()))
    }

    #[getter]
    pub fn slope(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner()?.slope().value().clone()))
    }

    #[getter]
    pub fn bias(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner()?.bias().value().clone()))
    }

    pub fn gradients(&self) -> PyResult<(Option<PyTensor>, Option<PyTensor>, Option<PyTensor>)> {
        let inner = self.inner()?;
        let gain = inner
            .gain()
            .gradient()
            .map(|g| PyTensor::from_tensor(g.clone()));
        let slope = inner
            .slope()
            .gradient()
            .map(|g| PyTensor::from_tensor(g.clone()));
        let bias = inner
            .bias()
            .gradient()
            .map(|g| PyTensor::from_tensor(g.clone()));
        Ok((gain, slope, bias))
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "Scaler", unsendable)]
pub(crate) struct PyScaler {
    inner: Option<Scaler>,
}

#[cfg(feature = "nn")]
impl PyScaler {
    fn inner(&self) -> PyResult<&Scaler> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("Scaler was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut Scaler> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("Scaler was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<Scaler> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("Scaler was moved into a container and can no longer be used")
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyScaler {
    #[new]
    pub fn new(name: &str, features: usize) -> PyResult<Self> {
        let inner = Scaler::new(name, features).map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    #[staticmethod]
    pub fn from_gain(name: &str, gain: &PyTensor) -> PyResult<Self> {
        let inner = Scaler::from_gain(name, gain.inner.clone()).map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    pub fn calibrate(&mut self, samples: &PyTensor, epsilon: f32) -> PyResult<()> {
        self.inner_mut()?
            .calibrate(&samples.inner, epsilon)
            .map_err(tensor_err_to_py)
    }

    pub fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        self.inner_mut()?
            .attach_hypergrad(curvature, learning_rate)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad_with_topos(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner_mut()?
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.attach_hypergrad(curvature, learning_rate)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner_mut()?
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner_mut()?
            .zero_accumulators()
            .map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner_mut()?
            .apply_step(fallback_lr)
            .map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner()?
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner_mut()?
            .load_state_dict(&map)
            .map_err(tensor_err_to_py)
    }

    #[getter]
    pub fn gain(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner()?.gain().value().clone()))
    }

    #[getter]
    pub fn baseline(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner()?.baseline().clone()))
    }

    pub fn gradient(&self) -> PyResult<Option<PyTensor>> {
        Ok(self
            .inner()?
            .gain()
            .gradient()
            .map(|g| PyTensor::from_tensor(g.clone())))
    }

    pub fn psi_probe(&self) -> PyResult<Option<f32>> {
        Ok(self.inner()?.psi_probe())
    }

    pub fn psi_components(&self) -> PyResult<Option<PyTensor>> {
        self.inner()?
            .psi_components()
            .map(|maybe| maybe.map(PyTensor::from_tensor))
            .map_err(tensor_err_to_py)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "Dropout", unsendable)]
pub(crate) struct PyDropout {
    inner: Option<RustDropout>,
}

#[cfg(feature = "nn")]
impl PyDropout {
    fn inner(&self) -> PyResult<&RustDropout> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("Dropout was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut RustDropout> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("Dropout was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<RustDropout> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("Dropout was moved into a container and can no longer be used")
        })
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDropout {
    #[new]
    #[pyo3(signature = (probability, *, seed=None, training=true))]
    pub fn new(probability: f32, seed: Option<u64>, training: bool) -> PyResult<Self> {
        let mut inner = RustDropout::with_seed(probability, seed).map_err(tensor_err_to_py)?;
        if !training {
            inner.eval();
        }
        Ok(Self { inner: Some(inner) })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self
            .inner()?
            .forward(&input.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner_mut()?
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    pub fn set_training(&mut self, training: bool) -> PyResult<()> {
        self.inner_mut()?.set_training(training);
        Ok(())
    }

    pub fn train(&mut self) -> PyResult<()> {
        self.inner_mut()?.train();
        Ok(())
    }

    pub fn eval(&mut self) -> PyResult<()> {
        self.inner_mut()?.eval();
        Ok(())
    }

    #[getter]
    pub fn training(&self) -> PyResult<bool> {
        Ok(self.inner()?.training())
    }

    #[getter]
    pub fn probability(&self) -> PyResult<f32> {
        Ok(self.inner()?.probability())
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZConv", unsendable)]
pub(crate) struct PyZConv {
    inner: Option<Conv2d>,
    layout: Layout2d,
    input_dims: Spatial2d,
    output_dims: Spatial2d,
    input_hw: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
}

#[cfg(feature = "nn")]
impl PyZConv {
    fn inner(&self) -> PyResult<&Conv2d> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("ZConv was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut Conv2d> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("ZConv was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<Conv2d> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("ZConv was moved into a container and can no longer be used")
        })
    }

    fn input_to_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout(
            tensor,
            self.input_dims,
            self.layout,
            LayoutDirection::ToCanonical,
        )
    }

    fn input_from_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout(
            tensor,
            self.input_dims,
            self.layout,
            LayoutDirection::FromCanonical,
        )
    }

    fn output_to_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout(
            tensor,
            self.output_dims,
            self.layout,
            LayoutDirection::ToCanonical,
        )
    }

    fn output_from_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout(
            tensor,
            self.output_dims,
            self.layout,
            LayoutDirection::FromCanonical,
        )
    }

    fn refresh_output_dims(&mut self) -> Result<(), TensorError> {
        let (out_h, out_w) = conv_output_hw(
            self.input_hw,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation,
        )?;
        self.output_dims = Spatial2d::new(self.output_dims.channels, out_h, out_w);
        Ok(())
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZConv {
    #[new]
    #[pyo3(signature = (name, in_channels, out_channels, input_hw, kernel, *, stride=None, padding=(0, 0), dilation=(1, 1), layout="NCHW"))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        in_channels: usize,
        out_channels: usize,
        input_hw: (usize, usize),
        kernel: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
        dilation: (usize, usize),
        layout: &str,
    ) -> PyResult<Self> {
        let stride = stride.unwrap_or((1, 1));
        let layout = Layout2d::parse(layout)?;
        let inner = Conv2d::new(
            name,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            input_hw,
        )
        .map_err(tensor_err_to_py)?;
        let (out_h, out_w) = conv_output_hw(input_hw, kernel, stride, padding, dilation)
            .map_err(tensor_err_to_py)?;
        Ok(Self {
            layout,
            input_dims: Spatial2d::new(in_channels, input_hw.0, input_hw.1),
            output_dims: Spatial2d::new(out_channels, out_h, out_w),
            input_hw,
            kernel,
            stride,
            padding,
            dilation,
            inner: Some(inner),
        })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let canonical_input = self
            .input_to_canonical(&input.inner)
            .map_err(tensor_err_to_py)?;
        let canonical_output = self
            .inner()?
            .forward(&canonical_input)
            .map_err(tensor_err_to_py)?;
        let output = self
            .output_from_canonical(&canonical_output)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let canonical_input = self
            .input_to_canonical(&input.inner)
            .map_err(tensor_err_to_py)?;
        let canonical_grad_output = self
            .output_to_canonical(&grad_output.inner)
            .map_err(tensor_err_to_py)?;
        let grad_input = self
            .inner_mut()?
            .backward(&canonical_input, &canonical_grad_output)
            .map_err(tensor_err_to_py)?;
        let restored = self
            .input_from_canonical(&grad_input)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(restored))
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner_mut()?
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.inner_mut()?
                .attach_hypergrad(curvature, learning_rate)
                .map_err(tensor_err_to_py)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner_mut()?
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner_mut()?
            .zero_accumulators()
            .map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner_mut()?
            .apply_step(fallback_lr)
            .map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner()?
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner_mut()?
            .load_state_dict(&map)
            .map_err(tensor_err_to_py)?;
        self.refresh_output_dims().map_err(tensor_err_to_py)
    }

    pub fn set_dilation(&mut self, dilation: (usize, usize)) -> PyResult<()> {
        self.inner_mut()?
            .set_dilation(dilation)
            .map_err(tensor_err_to_py)?;
        self.dilation = dilation;
        self.refresh_output_dims().map_err(tensor_err_to_py)
    }

    #[getter]
    pub fn layout(&self) -> &'static str {
        self.layout.as_str()
    }

    #[getter]
    pub fn in_channels(&self) -> usize {
        self.input_dims.channels
    }

    #[getter]
    pub fn out_channels(&self) -> usize {
        self.output_dims.channels
    }

    #[getter]
    pub fn input_hw(&self) -> (usize, usize) {
        self.input_hw
    }

    #[getter]
    pub fn output_hw(&self) -> (usize, usize) {
        (self.output_dims.height, self.output_dims.width)
    }

    #[getter]
    pub fn kernel(&self) -> (usize, usize) {
        self.kernel
    }

    #[getter]
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    #[getter]
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    #[getter]
    pub fn dilation(&self) -> (usize, usize) {
        self.dilation
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZConv6DA", unsendable)]
pub(crate) struct PyZConv6DA {
    inner: Option<Conv6da>,
    layout: Layout3d,
    input_dims: Spatial3d,
    output_dims: Spatial3d,
    grid: (usize, usize, usize),
    leech_rank: usize,
    leech_weight: f64,
    neighbor_offsets: Vec<(isize, isize, isize)>,
}

#[cfg(feature = "nn")]
impl PyZConv6DA {
    fn inner(&self) -> PyResult<&Conv6da> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("ZConv6DA was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut Conv6da> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("ZConv6DA was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<Conv6da> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("ZConv6DA was moved into a container and can no longer be used")
        })
    }

    fn input_to_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout_3d(
            tensor,
            self.input_dims,
            self.layout,
            LayoutDirection::ToCanonical,
        )
    }

    fn input_from_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout_3d(
            tensor,
            self.input_dims,
            self.layout,
            LayoutDirection::FromCanonical,
        )
    }

    fn output_to_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout_3d(
            tensor,
            self.output_dims,
            self.layout,
            LayoutDirection::ToCanonical,
        )
    }

    fn output_from_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout_3d(
            tensor,
            self.output_dims,
            self.layout,
            LayoutDirection::FromCanonical,
        )
    }

    fn leech_projector(&self) -> LeechProjector {
        LeechProjector::new(self.leech_rank, self.leech_weight)
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZConv6DA {
    #[new]
    #[pyo3(signature = (name, in_channels, out_channels, grid, *, leech_rank=24, leech_weight=1.0, layout="NCDHW", neighbors=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        in_channels: usize,
        out_channels: usize,
        grid: (usize, usize, usize),
        leech_rank: usize,
        leech_weight: f64,
        layout: &str,
        neighbors: Option<Vec<(isize, isize, isize)>>,
    ) -> PyResult<Self> {
        let layout = Layout3d::parse(layout)?;
        let inner = if let Some(ref offsets) = neighbors {
            Conv6da::with_neighbors(
                name,
                in_channels,
                out_channels,
                grid,
                leech_rank,
                leech_weight,
                offsets.as_slice(),
            )
        } else {
            Conv6da::new(
                name,
                in_channels,
                out_channels,
                grid,
                leech_rank,
                leech_weight,
            )
        }
        .map_err(tensor_err_to_py)?;
        let neighbor_offsets = inner.neighbor_offsets().to_vec();
        Ok(Self {
            layout,
            input_dims: Spatial3d::new(in_channels, grid.0, grid.1, grid.2),
            output_dims: Spatial3d::new(out_channels, grid.0, grid.1, grid.2),
            grid,
            leech_rank,
            leech_weight,
            neighbor_offsets,
            inner: Some(inner),
        })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let canonical_input = self
            .input_to_canonical(&input.inner)
            .map_err(tensor_err_to_py)?;
        let canonical_output = self
            .inner()?
            .forward(&canonical_input)
            .map_err(tensor_err_to_py)?;
        let output = self
            .output_from_canonical(&canonical_output)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let canonical_input = self
            .input_to_canonical(&input.inner)
            .map_err(tensor_err_to_py)?;
        let canonical_grad_output = self
            .output_to_canonical(&grad_output.inner)
            .map_err(tensor_err_to_py)?;
        let grad_input = self
            .inner_mut()?
            .backward(&canonical_input, &canonical_grad_output)
            .map_err(tensor_err_to_py)?;
        let restored = self
            .input_from_canonical(&grad_input)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(restored))
    }

    #[pyo3(signature = (curvature, learning_rate, *, topos=None))]
    pub fn attach_hypergrad(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<()> {
        if let Some(topos) = topos {
            self.inner_mut()?
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.inner_mut()?
                .attach_hypergrad(curvature, learning_rate)
                .map_err(tensor_err_to_py)
        }
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner_mut()?
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner_mut()?
            .zero_accumulators()
            .map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner_mut()?
            .apply_step(fallback_lr)
            .map_err(tensor_err_to_py)
    }

    pub fn state_dict(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let mut entries: Vec<_> = self
            .inner()?
            .state_dict()
            .map_err(tensor_err_to_py)?
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries
            .into_iter()
            .map(|(name, tensor)| (name, PyTensor::from_tensor(tensor)))
            .collect())
    }

    pub fn load_state_dict(&mut self, state: Vec<(String, PyTensor)>) -> PyResult<()> {
        let mut map = std::collections::HashMap::new();
        for (name, tensor) in state {
            map.insert(name, tensor.inner.clone());
        }
        self.inner_mut()?
            .load_state_dict(&map)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    #[getter]
    pub fn layout(&self) -> String {
        self.layout.as_str().to_string()
    }

    #[getter]
    pub fn grid(&self) -> (usize, usize, usize) {
        self.grid
    }

    #[getter]
    pub fn in_channels(&self) -> usize {
        self.input_dims.channels
    }

    #[getter]
    pub fn out_channels(&self) -> usize {
        self.output_dims.channels
    }

    #[getter]
    pub fn neighbor_count(&self) -> usize {
        self.neighbor_offsets.len()
    }

    #[getter]
    pub fn neighbor_offsets(&self) -> Vec<(isize, isize, isize)> {
        self.neighbor_offsets.clone()
    }

    #[getter]
    pub fn leech_rank(&self) -> usize {
        self.leech_rank
    }

    #[getter]
    pub fn leech_weight(&self) -> f64 {
        self.leech_weight
    }

    #[getter]
    pub fn input_shape(&self) -> (usize, usize, usize, usize) {
        (
            self.input_dims.channels,
            self.input_dims.depth,
            self.input_dims.height,
            self.input_dims.width,
        )
    }

    #[getter]
    pub fn output_shape(&self) -> (usize, usize, usize, usize) {
        (
            self.output_dims.channels,
            self.output_dims.depth,
            self.output_dims.height,
            self.output_dims.width,
        )
    }

    pub fn leech_enrich(&self, geodesic: f32) -> f32 {
        self.leech_projector().enrich(f64::from(geodesic)) as f32
    }

    #[getter]
    pub fn ramanujan_pi_delta(&self) -> f32 {
        (ramanujan_pi(6) - std::f64::consts::PI) as f32
    }

    #[staticmethod]
    pub fn ramanujan_pi_boost() -> f64 {
        ramanujan_pi(6)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZPooling", unsendable)]
pub(crate) struct PyZPooling {
    inner: Option<PoolModule>,
    mode: PoolMode,
    layout: Layout2d,
    input_dims: Spatial2d,
    output_dims: Spatial2d,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

#[cfg(feature = "nn")]
impl PyZPooling {
    fn inner(&self) -> PyResult<&PoolModule> {
        self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("ZPooling was moved into a container and can no longer be used")
        })
    }

    fn inner_mut(&mut self) -> PyResult<&mut PoolModule> {
        self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("ZPooling was moved into a container and can no longer be used")
        })
    }

    fn take_inner(&mut self) -> PyResult<PoolModule> {
        self.inner.take().ok_or_else(|| {
            PyValueError::new_err("ZPooling was moved into a container and can no longer be used")
        })
    }

    fn input_to_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout(
            tensor,
            self.input_dims,
            self.layout,
            LayoutDirection::ToCanonical,
        )
    }

    fn input_from_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout(
            tensor,
            self.input_dims,
            self.layout,
            LayoutDirection::FromCanonical,
        )
    }

    fn output_to_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout(
            tensor,
            self.output_dims,
            self.layout,
            LayoutDirection::ToCanonical,
        )
    }

    fn output_from_canonical(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        reorder_tensor_layout(
            tensor,
            self.output_dims,
            self.layout,
            LayoutDirection::FromCanonical,
        )
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZPooling {
    #[new]
    #[pyo3(signature = (channels, kernel, input_hw, *, stride=None, padding=(0,0), layout="NCHW", mode="max"))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels: usize,
        kernel: (usize, usize),
        input_hw: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
        layout: &str,
        mode: &str,
    ) -> PyResult<Self> {
        let layout = Layout2d::parse(layout)?;
        let mode = PoolMode::parse(mode)?;
        let stride = stride.unwrap_or(kernel);
        let inner = match mode {
            PoolMode::Max => PoolModule::Max(
                MaxPool2d::new(channels, kernel, stride, padding, input_hw)
                    .map_err(tensor_err_to_py)?,
            ),
            PoolMode::Avg => PoolModule::Avg(
                AvgPool2d::new(channels, kernel, stride, padding, input_hw)
                    .map_err(tensor_err_to_py)?,
            ),
        };
        let output_hw =
            pool_output_hw(input_hw, kernel, stride, padding).map_err(tensor_err_to_py)?;
        Ok(Self {
            inner: Some(inner),
            mode,
            layout,
            input_dims: Spatial2d::new(channels, input_hw.0, input_hw.1),
            output_dims: Spatial2d::new(channels, output_hw.0, output_hw.1),
            kernel,
            stride,
            padding,
        })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let canonical_input = self
            .input_to_canonical(&input.inner)
            .map_err(tensor_err_to_py)?;
        let canonical_output = self
            .inner()?
            .forward(&canonical_input)
            .map_err(tensor_err_to_py)?;
        let output = self
            .output_from_canonical(&canonical_output)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let canonical_input = self
            .input_to_canonical(&input.inner)
            .map_err(tensor_err_to_py)?;
        let canonical_grad_output = self
            .output_to_canonical(&grad_output.inner)
            .map_err(tensor_err_to_py)?;
        let grad_input = self
            .inner_mut()?
            .backward(&canonical_input, &canonical_grad_output)
            .map_err(tensor_err_to_py)?;
        let restored = self
            .input_from_canonical(&grad_input)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(restored))
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    #[getter]
    pub fn mode(&self) -> &'static str {
        self.mode.as_str()
    }

    #[getter]
    pub fn layout(&self) -> String {
        self.layout.as_str().to_string()
    }

    #[getter]
    pub fn channels(&self) -> usize {
        self.input_dims.channels
    }

    #[getter]
    pub fn input_hw(&self) -> (usize, usize) {
        (self.input_dims.height, self.input_dims.width)
    }

    #[getter]
    pub fn output_hw(&self) -> (usize, usize) {
        (self.output_dims.height, self.output_dims.width)
    }

    #[getter]
    pub fn kernel(&self) -> (usize, usize) {
        self.kernel
    }

    #[getter]
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    #[getter]
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "Dataset")]
pub(crate) struct PyDataset {
    inner: Dataset,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDataset {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Dataset::new(),
        }
    }

    #[staticmethod]
    pub fn from_pairs(samples: Vec<(PyTensor, PyTensor)>) -> Self {
        let converted = convert_samples(samples);
        Self {
            inner: Dataset::from_vec(converted),
        }
    }

    pub fn push(&mut self, input: &PyTensor, target: &PyTensor) {
        self.inner.push(input.inner.clone(), target.inner.clone());
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn loader(&self) -> PyDataLoader {
        PyDataLoader::from_loader(self.inner.loader())
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "DataLoader", unsendable)]
pub(crate) struct PyDataLoader {
    inner: DataLoader,
}

#[cfg(feature = "nn")]
impl PyDataLoader {
    fn from_loader(inner: DataLoader) -> Self {
        Self { inner }
    }

    fn iter_inner(&self) -> PyDataLoaderIter {
        PyDataLoaderIter::new(self.inner.clone().into_iter())
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDataLoader {
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    pub fn prefetch_depth(&self) -> usize {
        self.inner.prefetch_depth()
    }

    pub fn shuffle(&self, seed: u64) -> Self {
        Self::from_loader(self.inner.clone().shuffle(seed))
    }

    pub fn batched(&self, batch_size: usize) -> Self {
        Self::from_loader(self.inner.clone().batched(batch_size))
    }

    pub fn dynamic_batch_by_rows(&self, max_rows: usize) -> Self {
        Self::from_loader(self.inner.clone().dynamic_batch_by_rows(max_rows))
    }

    pub fn prefetch(&self, depth: usize) -> Self {
        Self::from_loader(self.inner.clone().prefetch(depth))
    }

    pub fn iter(&self, py: Python<'_>) -> PyResult<Py<PyDataLoaderIter>> {
        Py::new(py, self.iter_inner())
    }

    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyDataLoaderIter>> {
        slf.iter(slf.py())
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "DataLoaderIter", unsendable)]
pub(crate) struct PyDataLoaderIter {
    batches: Option<DataLoaderBatches>,
}

#[cfg(feature = "nn")]
impl PyDataLoaderIter {
    fn new(batches: DataLoaderBatches) -> Self {
        Self {
            batches: Some(batches),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyDataLoaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> Py<PyDataLoaderIter> {
        slf.into()
    }

    fn __next__(&mut self) -> PyResult<Option<(PyTensor, PyTensor)>> {
        let batches = match self.batches.as_mut() {
            Some(iter) => iter,
            None => return Ok(None),
        };
        match batches.next() {
            Some(Ok((input, target))) => Ok(Some((
                PyTensor::from_tensor(input),
                PyTensor::from_tensor(target),
            ))),
            Some(Err(err)) => Err(tensor_err_to_py(err)),
            None => {
                self.batches = None;
                Ok(None)
            }
        }
    }
}

#[cfg(feature = "nn")]
#[pyfunction]
#[pyo3(signature = (samples))]
fn from_samples(samples: Vec<(PyTensor, PyTensor)>) -> PyDataLoader {
    PyDataLoader::from_loader(dataset_from_vec(convert_samples(samples)))
}

#[cfg(feature = "nn")]
#[pyfunction(name = "is_swap_invariant")]
pub(crate) fn py_is_swap_invariant(arrangement: Vec<f32>) -> bool {
    rust_is_swap_invariant(&arrangement)
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "CoherenceChannelReport")]
pub(crate) struct PyCoherenceChannelReport {
    channel: usize,
    weight: f32,
    backend: String,
    dominant_concept: Option<String>,
    emphasis: f32,
    descriptor: Option<String>,
}

#[cfg(feature = "nn")]
impl PyCoherenceChannelReport {
    fn from_report(report: &LinguisticChannelReport) -> Self {
        Self {
            channel: report.channel(),
            weight: report.weight(),
            backend: report.backend().label().to_string(),
            dominant_concept: report
                .dominant_concept()
                .map(|concept| concept.label().to_string()),
            emphasis: report.emphasis(),
            descriptor: report.descriptor().map(|descriptor| descriptor.to_string()),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCoherenceChannelReport {
    #[getter]
    fn channel(&self) -> usize {
        self.channel
    }

    #[getter]
    fn weight(&self) -> f32 {
        self.weight
    }

    #[getter]
    fn backend(&self) -> &str {
        &self.backend
    }

    #[getter]
    fn dominant_concept(&self) -> Option<&str> {
        self.dominant_concept.as_deref()
    }

    #[getter]
    fn emphasis(&self) -> f32 {
        self.emphasis
    }

    #[getter]
    fn descriptor(&self) -> Option<&str> {
        self.descriptor.as_deref()
    }
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "CoherenceSignature", unsendable)]
pub(crate) struct PyCoherenceSignature {
    dominant_channel: Option<usize>,
    energy_ratio: f32,
    entropy: f32,
    mean_coherence: f32,
    swap_invariant: bool,
}

#[cfg(feature = "nn")]
impl PyCoherenceSignature {
    fn from_signature(signature: &CoherenceSignature) -> Self {
        Self {
            dominant_channel: signature.dominant_channel(),
            energy_ratio: signature.energy_ratio(),
            entropy: signature.entropy(),
            mean_coherence: signature.mean_coherence(),
            swap_invariant: signature.swap_invariant(),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCoherenceSignature {
    #[getter]
    fn dominant_channel(&self) -> Option<usize> {
        self.dominant_channel
    }

    #[getter]
    fn energy_ratio(&self) -> f32 {
        self.energy_ratio
    }

    #[getter]
    fn entropy(&self) -> f32 {
        self.entropy
    }

    #[getter]
    fn mean_coherence(&self) -> f32 {
        self.mean_coherence
    }

    #[getter]
    fn swap_invariant(&self) -> bool {
        self.swap_invariant
    }
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "CoherenceObservation", unsendable)]
pub(crate) struct PyCoherenceObservation {
    observation: CoherenceObservation,
    label: CoherenceLabel,
}

#[cfg(feature = "nn")]
impl PyCoherenceObservation {
    fn from_observation(observation: CoherenceObservation) -> Self {
        let label = observation.lift_to_label();
        Self { observation, label }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCoherenceObservation {
    #[getter]
    fn is_signature(&self) -> bool {
        matches!(self.observation, CoherenceObservation::Signature(_))
    }

    #[getter]
    fn label(&self) -> String {
        self.label.to_string()
    }

    #[getter]
    fn signature(&self) -> Option<PyCoherenceSignature> {
        match &self.observation {
            CoherenceObservation::Signature(signature) => {
                Some(PyCoherenceSignature::from_signature(signature))
            }
            CoherenceObservation::Undetermined => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.observation {
            CoherenceObservation::Undetermined => format!(
                "CoherenceObservation(label='{}', signature=None)",
                self.label
            ),
            CoherenceObservation::Signature(_) => format!(
                "CoherenceObservation(label='{}', signature=...)",
                self.label
            ),
        }
    }
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "CoherenceDiagnostics", unsendable)]
pub(crate) struct PyCoherenceDiagnostics {
    aggregated: PyTensor,
    coherence: Vec<f32>,
    channel_reports: Vec<PyCoherenceChannelReport>,
    pre_discard: Option<PyPreDiscardTelemetry>,
    observation: PyCoherenceObservation,
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "PreDiscardTelemetry", unsendable)]
pub(crate) struct PyPreDiscardTelemetry {
    dominance_ratio: f32,
    energy_floor: f32,
    discarded: usize,
    preserved: usize,
    fallback: bool,
    survivor_energy: f32,
    discarded_energy: f32,
    total_energy: f32,
    survivor_energy_ratio: f32,
    discarded_energy_ratio: f32,
    dominant_weight: f32,
}

#[cfg(feature = "nn")]
impl PyPreDiscardTelemetry {
    fn from_telemetry(telemetry: PreDiscardTelemetry) -> Self {
        Self {
            dominance_ratio: telemetry.dominance_ratio(),
            energy_floor: telemetry.energy_floor(),
            discarded: telemetry.discarded(),
            preserved: telemetry.preserved(),
            fallback: telemetry.used_fallback(),
            survivor_energy: telemetry.survivor_energy(),
            discarded_energy: telemetry.discarded_energy(),
            total_energy: telemetry.total_energy(),
            survivor_energy_ratio: telemetry.survivor_energy_ratio(),
            discarded_energy_ratio: telemetry.discarded_energy_ratio(),
            dominant_weight: telemetry.dominant_weight(),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyPreDiscardTelemetry {
    #[getter]
    fn dominance_ratio(&self) -> f32 {
        self.dominance_ratio
    }

    #[getter]
    fn energy_floor(&self) -> f32 {
        self.energy_floor
    }

    #[getter]
    fn discarded(&self) -> usize {
        self.discarded
    }

    #[getter]
    fn preserved(&self) -> usize {
        self.preserved
    }

    #[getter]
    fn used_fallback(&self) -> bool {
        self.fallback
    }

    #[getter]
    fn total(&self) -> usize {
        self.discarded + self.preserved
    }

    #[getter]
    fn preserved_ratio(&self) -> f32 {
        if self.discarded + self.preserved == 0 {
            0.0
        } else {
            (self.preserved as f32 / (self.discarded + self.preserved) as f32).clamp(0.0, 1.0)
        }
    }

    #[getter]
    fn discarded_ratio(&self) -> f32 {
        1.0 - self.preserved_ratio()
    }

    #[getter]
    fn survivor_energy(&self) -> f32 {
        self.survivor_energy
    }

    #[getter]
    fn discarded_energy(&self) -> f32 {
        self.discarded_energy
    }

    #[getter]
    fn total_energy(&self) -> f32 {
        self.total_energy
    }

    #[getter]
    fn survivor_energy_ratio(&self) -> f32 {
        self.survivor_energy_ratio
    }

    #[getter]
    fn discarded_energy_ratio(&self) -> f32 {
        self.discarded_energy_ratio
    }

    #[getter]
    fn dominant_weight(&self) -> f32 {
        self.dominant_weight
    }
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "PreDiscardSnapshot", unsendable)]
pub(crate) struct PyPreDiscardSnapshot {
    step: u64,
    telemetry: PyPreDiscardTelemetry,
    survivors: Vec<usize>,
    discarded: Vec<usize>,
    filtered: Vec<f32>,
}

#[cfg(feature = "nn")]
impl PyPreDiscardSnapshot {
    fn from_snapshot(snapshot: PreDiscardSnapshot) -> Self {
        Self {
            step: snapshot.step(),
            telemetry: PyPreDiscardTelemetry::from_telemetry(snapshot.telemetry().clone()),
            survivors: snapshot.survivors().to_vec(),
            discarded: snapshot.discarded().to_vec(),
            filtered: snapshot.filtered().to_vec(),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyPreDiscardSnapshot {
    #[getter]
    fn step(&self) -> u64 {
        self.step
    }

    #[getter]
    fn telemetry(&self) -> PyPreDiscardTelemetry {
        self.telemetry.clone()
    }

    #[getter]
    fn survivors(&self) -> Vec<usize> {
        self.survivors.clone()
    }

    #[getter]
    fn discarded(&self) -> Vec<usize> {
        self.discarded.clone()
    }

    #[getter]
    fn filtered(&self) -> Vec<f32> {
        self.filtered.clone()
    }
}

#[cfg(feature = "nn")]
#[derive(Clone)]
#[pyclass(module = "spiraltorch.nn", name = "PreDiscardPolicy", unsendable)]
pub(crate) struct PyPreDiscardPolicy {
    dominance_ratio: f32,
    energy_floor: f32,
    min_channels: usize,
}

#[cfg(feature = "nn")]
impl PyPreDiscardPolicy {
    fn from_policy(policy: &PreDiscardPolicy) -> Self {
        Self {
            dominance_ratio: policy.dominance_ratio(),
            energy_floor: policy.energy_floor(),
            min_channels: policy.min_channels(),
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyPreDiscardPolicy {
    #[new]
    #[pyo3(signature = (dominance_ratio, *, energy_floor=None, min_channels=None))]
    fn new(
        dominance_ratio: f32,
        energy_floor: Option<f32>,
        min_channels: Option<usize>,
    ) -> PyResult<Self> {
        let mut policy = PreDiscardPolicy::new(dominance_ratio).map_err(tensor_err_to_py)?;
        if let Some(floor) = energy_floor {
            policy = policy.with_energy_floor(floor).map_err(tensor_err_to_py)?;
        }
        if let Some(min_channels) = min_channels {
            policy = policy.with_min_channels(min_channels);
        }
        Ok(Self::from_policy(&policy))
    }

    #[getter]
    fn dominance_ratio(&self) -> f32 {
        self.dominance_ratio
    }

    #[getter]
    fn energy_floor(&self) -> f32 {
        self.energy_floor
    }

    #[getter]
    fn min_channels(&self) -> usize {
        self.min_channels
    }
}

#[cfg(feature = "nn")]
impl PyCoherenceDiagnostics {
    fn from_diagnostics(diagnostics: CoherenceDiagnostics) -> Self {
        let observation = PyCoherenceObservation::from_observation(diagnostics.observation());
        let (aggregated, coherence, channel_reports, pre_discard) = diagnostics.into_parts();
        let channel_reports = channel_reports
            .iter()
            .map(PyCoherenceChannelReport::from_report)
            .collect();
        Self {
            aggregated: PyTensor::from_tensor(aggregated),
            coherence,
            channel_reports,
            pre_discard: pre_discard.map(PyPreDiscardTelemetry::from_telemetry),
            observation,
        }
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyCoherenceDiagnostics {
    #[getter]
    fn aggregated(&self) -> PyTensor {
        self.aggregated.clone()
    }

    #[getter]
    fn coherence(&self) -> Vec<f32> {
        self.coherence.clone()
    }

    #[getter]
    fn channel_reports(&self) -> Vec<PyCoherenceChannelReport> {
        self.channel_reports.clone()
    }

    #[getter]
    fn preserved_channels(&self) -> usize {
        self.coherence.iter().filter(|value| **value > 0.0).count()
    }

    #[getter]
    fn discarded_channels(&self) -> usize {
        self.coherence
            .len()
            .saturating_sub(self.preserved_channels())
    }

    #[getter]
    fn pre_discard(&self) -> Option<PyPreDiscardTelemetry> {
        self.pre_discard.clone()
    }

    #[getter]
    fn observation(&self) -> PyCoherenceObservation {
        self.observation.clone()
    }
}

#[cfg(feature = "nn")]
#[pyclass(
    module = "spiraltorch.nn",
    name = "ZSpaceCoherenceSequencer",
    unsendable
)]
pub(crate) struct PyZSpaceCoherenceSequencer {
    inner: ZSpaceCoherenceSequencer,
    trace_recorder: Option<ZSpaceTraceRecorder>,
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZSpaceTraceRecorder", unsendable)]
pub(crate) struct PyZSpaceTraceRecorder {
    inner: ZSpaceTraceRecorder,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZSpaceTraceRecorder {
    pub fn snapshot(&self, py: Python<'_>) -> PyResult<PyObject> {
        let trace = self.inner.snapshot();
        let value =
            serde_json::to_value(&trace).map_err(|err| PyValueError::new_err(err.to_string()))?;
        json_to_py(py, &value)
    }

    pub fn clear(&self) {
        self.inner.clear();
    }

    pub fn write_jsonl(&self, path: &str) -> PyResult<()> {
        self.inner.write_jsonl(path).map_err(tensor_err_to_py)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "MellinBasis", unsendable)]
pub(crate) struct PyMellinBasis {
    inner: MellinBasis,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyMellinBasis {
    #[new]
    pub fn new(exponents: Vec<f64>) -> Self {
        Self {
            inner: MellinBasis::new(exponents),
        }
    }

    #[staticmethod]
    pub fn constant(dimension: usize, exponent: f64) -> Self {
        Self {
            inner: MellinBasis::constant(dimension, exponent),
        }
    }

    #[staticmethod]
    pub fn ramp(dimension: usize, start: f64, end: f64) -> Self {
        Self {
            inner: MellinBasis::ramp(dimension, start, end),
        }
    }

    pub fn project(&self, input: Vec<f64>) -> Vec<f64> {
        let vector = DVector::from_vec(input);
        let projected = self.inner.project(&vector);
        dvector_to_vec(&projected)
    }

    #[getter]
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn __repr__(&self) -> String {
        format!("MellinBasis(dim={})", self.inner.dimension())
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZSpaceVaeStats", unsendable)]
pub(crate) struct PyZSpaceVaeStats {
    inner: ZSpaceVaeStats,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZSpaceVaeStats {
    #[getter]
    pub fn recon_loss(&self) -> f64 {
        self.inner.recon_loss
    }

    #[getter]
    pub fn kl_loss(&self) -> f64 {
        self.inner.kl_loss
    }

    #[getter]
    pub fn evidence_lower_bound(&self) -> f64 {
        self.inner.evidence_lower_bound
    }

    #[getter]
    pub fn target(&self) -> Vec<f64> {
        dvector_to_vec(&self.inner.target)
    }

    fn __repr__(&self) -> String {
        format!(
            "ZSpaceVaeStats(recon_loss={:.6}, kl_loss={:.6}, elbo={:.6})",
            self.inner.recon_loss, self.inner.kl_loss, self.inner.evidence_lower_bound
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZSpaceVaeState", unsendable)]
pub(crate) struct PyZSpaceVaeState {
    inner: ZSpaceVaeState,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZSpaceVaeState {
    #[getter]
    pub fn latent(&self) -> Vec<f64> {
        dvector_to_vec(&self.inner.latent)
    }

    #[getter]
    pub fn reconstruction(&self) -> Vec<f64> {
        dvector_to_vec(&self.inner.reconstruction)
    }

    #[getter]
    pub fn mu(&self) -> Vec<f64> {
        dvector_to_vec(&self.inner.mu)
    }

    #[getter]
    pub fn logvar(&self) -> Vec<f64> {
        dvector_to_vec(&self.inner.logvar)
    }

    #[getter]
    pub fn stats(&self) -> PyZSpaceVaeStats {
        PyZSpaceVaeStats {
            inner: self.inner.stats.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ZSpaceVaeState(input_dim={}, latent_dim={})",
            self.inner.reconstruction.len(),
            self.inner.latent.len()
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZSpaceVae", unsendable)]
pub(crate) struct PyZSpaceVae {
    inner: ZSpaceVae,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZSpaceVae {
    #[new]
    #[pyo3(signature = (input_dim, latent_dim, seed=42))]
    pub fn new(input_dim: usize, latent_dim: usize, seed: u64) -> Self {
        Self {
            inner: ZSpaceVae::new(input_dim, latent_dim, seed),
        }
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let inner = ZSpaceVae::load(path).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(tensor_err_to_py)
    }

    #[getter]
    pub fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }

    #[getter]
    pub fn latent_dim(&self) -> usize {
        self.inner.latent_dim()
    }

    pub fn forward(&mut self, input: Vec<f64>) -> PyResult<PyZSpaceVaeState> {
        if input.len() != self.inner.input_dim() {
            return Err(PyValueError::new_err(format!(
                "input length mismatch (expected {}, got {})",
                self.inner.input_dim(),
                input.len()
            )));
        }
        let vec = DVector::from_vec(input);
        let state = self.inner.forward(&vec);
        Ok(PyZSpaceVaeState { inner: state })
    }

    pub fn encode(&self, input: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if input.len() != self.inner.input_dim() {
            return Err(PyValueError::new_err(format!(
                "input length mismatch (expected {}, got {})",
                self.inner.input_dim(),
                input.len()
            )));
        }
        let vec = DVector::from_vec(input);
        let (mu, logvar) = self.inner.encode(&vec);
        Ok((dvector_to_vec(&mu), dvector_to_vec(&logvar)))
    }

    pub fn encode_with_mellin(
        &self,
        input: Vec<f64>,
        basis: &PyMellinBasis,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if input.len() != self.inner.input_dim() {
            return Err(PyValueError::new_err(format!(
                "input length mismatch (expected {}, got {})",
                self.inner.input_dim(),
                input.len()
            )));
        }
        let vec = DVector::from_vec(input);
        let (mu, logvar) = self.inner.encode_with_mellin(&vec, &basis.inner);
        Ok((dvector_to_vec(&mu), dvector_to_vec(&logvar)))
    }

    pub fn sample_latent(&mut self, mu: Vec<f64>, logvar: Vec<f64>) -> PyResult<Vec<f64>> {
        if mu.len() != self.inner.latent_dim() || logvar.len() != self.inner.latent_dim() {
            return Err(PyValueError::new_err(format!(
                "latent length mismatch (expected {}, got mu={}, logvar={})",
                self.inner.latent_dim(),
                mu.len(),
                logvar.len()
            )));
        }
        let mu = DVector::from_vec(mu);
        let logvar = DVector::from_vec(logvar);
        let sample = self.inner.sample_latent(&mu, &logvar);
        Ok(dvector_to_vec(&sample))
    }

    pub fn mean_latent(&self, mu: Vec<f64>) -> PyResult<Vec<f64>> {
        if mu.len() != self.inner.latent_dim() {
            return Err(PyValueError::new_err(format!(
                "latent length mismatch (expected {}, got {})",
                self.inner.latent_dim(),
                mu.len()
            )));
        }
        let mu = DVector::from_vec(mu);
        let latent = self.inner.mean_latent(&mu);
        Ok(dvector_to_vec(&latent))
    }

    pub fn decode(&self, latent: Vec<f64>) -> PyResult<Vec<f64>> {
        if latent.len() != self.inner.latent_dim() {
            return Err(PyValueError::new_err(format!(
                "latent length mismatch (expected {}, got {})",
                self.inner.latent_dim(),
                latent.len()
            )));
        }
        let latent = DVector::from_vec(latent);
        let decoded = self.inner.decode(&latent);
        Ok(dvector_to_vec(&decoded))
    }

    pub fn decode_with_mellin(
        &self,
        latent: Vec<f64>,
        basis: &PyMellinBasis,
    ) -> PyResult<Vec<f64>> {
        if latent.len() != self.inner.latent_dim() {
            return Err(PyValueError::new_err(format!(
                "latent length mismatch (expected {}, got {})",
                self.inner.latent_dim(),
                latent.len()
            )));
        }
        let latent = DVector::from_vec(latent);
        let decoded = self.inner.decode_with_mellin(&latent, &basis.inner);
        Ok(dvector_to_vec(&decoded))
    }

    pub fn refine_decoder(&mut self, state: &PyZSpaceVaeState, learning_rate: f64) {
        self.inner.refine_decoder(&state.inner, learning_rate);
    }

    fn __repr__(&self) -> String {
        format!(
            "ZSpaceVae(input_dim={}, latent_dim={})",
            self.inner.input_dim(),
            self.inner.latent_dim()
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZSpaceTextVae", unsendable)]
pub(crate) struct PyZSpaceTextVae {
    inner: ZSpaceTextVae,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZSpaceTextVae {
    #[new]
    #[pyo3(signature = (window_chars, latent_dim, *, curvature=-1.0, temperature=1.0, seed=42))]
    pub fn new(
        window_chars: usize,
        latent_dim: usize,
        curvature: f32,
        temperature: f32,
        seed: u64,
    ) -> PyResult<Self> {
        let inner = ZSpaceTextVae::new(window_chars, latent_dim, curvature, temperature, seed)
            .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let inner = ZSpaceTextVae::load(path).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(tensor_err_to_py)
    }

    #[getter]
    pub fn window_chars(&self) -> usize {
        self.inner.window_chars()
    }

    #[getter]
    pub fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }

    #[getter]
    pub fn latent_dim(&self) -> usize {
        self.inner.latent_dim()
    }

    #[getter]
    pub fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    #[getter]
    pub fn temperature(&self) -> f32 {
        self.inner.temperature()
    }

    pub fn encode_text(&self, text: &str) -> PyResult<Vec<f64>> {
        let encoded = self.inner.encode_text(text).map_err(tensor_err_to_py)?;
        Ok(dvector_to_vec(&encoded))
    }

    pub fn encode_text_with_mellin(&self, text: &str, basis: &PyMellinBasis) -> PyResult<Vec<f64>> {
        let encoded = self
            .inner
            .encode_text_with_mellin(text, &basis.inner)
            .map_err(tensor_err_to_py)?;
        Ok(dvector_to_vec(&encoded))
    }

    pub fn forward_text(&mut self, text: &str) -> PyResult<PyZSpaceVaeState> {
        let state = self.inner.forward_text(text).map_err(tensor_err_to_py)?;
        Ok(PyZSpaceVaeState { inner: state })
    }

    pub fn forward_text_with_mellin(
        &mut self,
        text: &str,
        basis: &PyMellinBasis,
    ) -> PyResult<PyZSpaceVaeState> {
        let state = self
            .inner
            .forward_text_with_mellin(text, &basis.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyZSpaceVaeState { inner: state })
    }

    pub fn forward_encoded(&mut self, encoded: Vec<f64>) -> PyResult<PyZSpaceVaeState> {
        if encoded.len() != self.inner.input_dim() {
            return Err(PyValueError::new_err(format!(
                "encoded length mismatch (expected {}, got {})",
                self.inner.input_dim(),
                encoded.len()
            )));
        }
        let encoded = DVector::from_vec(encoded);
        let state = self
            .inner
            .forward_encoded(&encoded)
            .map_err(tensor_err_to_py)?;
        Ok(PyZSpaceVaeState { inner: state })
    }

    pub fn refine_decoder(&mut self, state: &PyZSpaceVaeState, learning_rate: f64) {
        self.inner.refine_decoder(&state.inner, learning_rate);
    }

    fn __repr__(&self) -> String {
        format!(
            "ZSpaceTextVae(window_chars={}, input_dim={}, latent_dim={})",
            self.inner.window_chars(),
            self.inner.input_dim(),
            self.inner.latent_dim()
        )
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZRelativityModule", unsendable)]
pub(crate) struct PyZRelativityModule {
    pub(crate) inner: ZRelativityModule,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZSpaceCoherenceSequencer {
    #[new]
    #[pyo3(signature = (dim, num_heads, curvature, *, topos=None))]
    pub fn new(
        dim: usize,
        num_heads: usize,
        curvature: f32,
        topos: Option<&PyOpenCartesianTopos>,
    ) -> PyResult<Self> {
        let topos = match topos {
            Some(guard) => guard.inner.clone(),
            None => OpenCartesianTopos::new(curvature, 1e-5, 10.0, 256, 8192)
                .map_err(tensor_err_to_py)?,
        };
        let inner = ZSpaceCoherenceSequencer::new(dim, num_heads, curvature, topos)
            .map_err(tensor_err_to_py)?;
        Ok(Self {
            inner,
            trace_recorder: None,
        })
    }

    pub fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let output = self.inner.forward(&x.inner).map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn forward_with_coherence(&self, x: &PyTensor) -> PyResult<(PyTensor, Vec<f32>)> {
        let (output, coherence) = self
            .inner
            .forward_with_coherence(&x.inner)
            .map_err(tensor_err_to_py)?;
        Ok((PyTensor::from_tensor(output), coherence))
    }

    pub fn forward_with_diagnostics(
        &self,
        x: &PyTensor,
    ) -> PyResult<(PyTensor, Vec<f32>, PyCoherenceDiagnostics)> {
        let (output, coherence, diagnostics) = self
            .inner
            .forward_with_diagnostics(&x.inner)
            .map_err(tensor_err_to_py)?;
        Ok((
            PyTensor::from_tensor(output),
            coherence,
            PyCoherenceDiagnostics::from_diagnostics(diagnostics),
        ))
    }

    #[pyo3(signature = (*, capacity=256, max_vector_len=256, publish_plugin_events=true))]
    pub fn install_trace_recorder(
        &mut self,
        capacity: usize,
        max_vector_len: usize,
        publish_plugin_events: bool,
    ) -> PyResult<PyZSpaceTraceRecorder> {
        if publish_plugin_events {
            st_core::plugin::init_plugin_system().map_err(tensor_err_to_py)?;
        }
        if self.trace_recorder.is_some() {
            self.inner.clear_plugins();
            self.trace_recorder = None;
        }
        let recorder = self.inner.install_trace_recorder(ZSpaceTraceConfig {
            capacity,
            max_vector_len,
            publish_plugin_events,
        });
        self.trace_recorder = Some(recorder.clone());
        Ok(PyZSpaceTraceRecorder { inner: recorder })
    }

    #[getter]
    pub fn trace_recorder(&self) -> Option<PyZSpaceTraceRecorder> {
        self.trace_recorder
            .as_ref()
            .map(|recorder| PyZSpaceTraceRecorder {
                inner: recorder.clone(),
            })
    }

    pub fn project_to_zspace(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let projected = self
            .inner
            .project_to_zspace(&x.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(projected))
    }

    pub fn diagnostics(&self, x: &PyTensor) -> PyResult<PyCoherenceDiagnostics> {
        let diagnostics = self.inner.diagnostics(&x.inner).map_err(tensor_err_to_py)?;
        Ok(PyCoherenceDiagnostics::from_diagnostics(diagnostics))
    }

    #[pyo3(signature = (dominance_ratio, *, energy_floor=None, min_channels=None))]
    pub fn configure_pre_discard(
        &mut self,
        dominance_ratio: f32,
        energy_floor: Option<f32>,
        min_channels: Option<usize>,
    ) -> PyResult<()> {
        let mut policy = PreDiscardPolicy::new(dominance_ratio).map_err(tensor_err_to_py)?;
        if let Some(floor) = energy_floor {
            policy = policy.with_energy_floor(floor).map_err(tensor_err_to_py)?;
        }
        if let Some(min_channels) = min_channels {
            policy = policy.with_min_channels(min_channels);
        }
        self.inner.enable_pre_discard(policy);
        Ok(())
    }

    pub fn disable_pre_discard(&mut self) {
        self.inner.disable_pre_discard();
    }

    pub fn configure_pre_discard_memory(&mut self, limit: usize) {
        self.inner.configure_pre_discard_memory(limit);
    }

    pub fn clear_pre_discard_snapshots(&self) {
        self.inner.clear_pre_discard_snapshots();
    }

    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.dim
    }

    #[getter]
    pub fn num_heads(&self) -> usize {
        self.inner.num_heads
    }

    #[getter]
    pub fn pre_discard_policy(&self) -> Option<PyPreDiscardPolicy> {
        self.inner
            .pre_discard_policy()
            .map(PyPreDiscardPolicy::from_policy)
    }

    #[getter]
    pub fn pre_discard_snapshots(&self) -> Vec<PyPreDiscardSnapshot> {
        self.inner
            .pre_discard_snapshots()
            .into_iter()
            .map(PyPreDiscardSnapshot::from_snapshot)
            .collect()
    }

    #[getter]
    pub fn curvature(&self) -> f32 {
        self.inner.curvature
    }

    pub fn maxwell_channels(&self) -> usize {
        self.inner.maxwell_channels()
    }

    pub fn topos(&self) -> PyOpenCartesianTopos {
        PyOpenCartesianTopos::from_topos(self.inner.topos().clone())
    }
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyZRelativityModule {
    #[new]
    pub fn new(model: &PyZRelativityModel) -> PyResult<Self> {
        let inner = ZRelativityModule::from_model(model.inner.clone()).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self.inner.forward(&input.inner).map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let grad = self
            .inner
            .backward(&input.inner, &grad_output.inner)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(grad))
    }

    pub fn parameter_tensor(&self) -> PyResult<PyTensor> {
        let seed = Tensor::zeros(1, 1).map_err(tensor_err_to_py)?;
        let output = self.inner.forward(&seed).map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(output))
    }

    pub fn torch_parameters(&self, py: Python<'_>) -> PyResult<PyObject> {
        let seed = Tensor::zeros(1, 1).map_err(tensor_err_to_py)?;
        let output = self.inner.forward(&seed).map_err(tensor_err_to_py)?;
        tensor_to_torch(py, &output)
    }

    pub fn parameter_dimension(&self) -> usize {
        self.inner.parameter_dimension()
    }

    pub fn model(&self) -> PyZRelativityModel {
        PyZRelativityModel {
            inner: self.inner.model().clone(),
        }
    }

    pub fn zero_accumulators(&mut self) -> PyResult<()> {
        self.inner.zero_accumulators().map_err(tensor_err_to_py)
    }

    pub fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        self.inner.apply_step(fallback_lr).map_err(tensor_err_to_py)
    }

    pub fn attach_realgrad(&mut self, learning_rate: f32) -> PyResult<()> {
        self.inner
            .attach_realgrad(learning_rate)
            .map_err(tensor_err_to_py)
    }

    pub fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        self.inner
            .attach_hypergrad(curvature, learning_rate)
            .map_err(tensor_err_to_py)
    }
}

#[cfg(feature = "nn")]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "nn")?;
    module.add("__doc__", "SpiralTorch neural network primitives")?;
    module.add_class::<PyIdentity>()?;
    module.add_class::<PyLinear>()?;
    module.add_class::<PyEmbedding>()?;
    module.add_class::<PyFeatureReorder2d>()?;
    module.add_class::<PySpiralRnn>()?;
    module.add_class::<PyWaveGate>()?;
    module.add_class::<PyWaveRnn>()?;
    module.add_class::<PyZSpaceMixer>()?;
    module.add_class::<PyZSpaceSoftmax>()?;
    module.add_class::<PyZSpaceCoherenceScan>()?;
    module.add_class::<PyZSpaceCoherenceWaveBlock>()?;
    module.add_class::<PyRelu>()?;
    module.add_class::<PySequential>()?;
    module.add_class::<PyMeanSquaredError>()?;
    module.add_class::<PyCategoricalCrossEntropy>()?;
    module.add_class::<PyHyperbolicCrossEntropy>()?;
    if let Ok(loss) = module.getattr("HyperbolicCrossEntropy") {
        module.add("CrossEntropy", loss)?;
    }
    module.add_class::<PyFocalLoss>()?;
    module.add_class::<PyContrastiveLoss>()?;
    module.add_class::<PyTripletLoss>()?;
    module.add_class::<PyRoundtableConfig>()?;
    module.add_class::<PySoftLogicConfig>()?;
    module.add_class::<PyRoundtableSchedule>()?;
    module.add_class::<PyEpochStats>()?;
    module.add_class::<PyNnModuleTrainer>()?;
    module.add_class::<PyDesireTrainerBridge>()?;
    module.add_class::<PyDesireRoundtableBridge>()?;
    module.add_class::<PyDesireTelemetryBundle>()?;
    module.add_class::<PyMaxwellDesireBridge>()?;
    module.add_class::<PyNarrativeHint>()?;
    module.add_class::<PyNarrativeSummary>()?;
    module.add_class::<PyDesirePipeline>()?;
    module.add_class::<PyNonLiner>()?;
    module.add_class::<PyScaler>()?;
    module.add_class::<PyZConv>()?;
    module.add_class::<PyZConv6DA>()?;
    module.add_class::<PyZPooling>()?;
    module.add_class::<PyDropout>()?;
    module.add_class::<PyPool2d>()?;
    module.add_class::<PyDataset>()?;
    module.add_class::<PyDataLoader>()?;
    module.add_class::<PyDataLoaderIter>()?;
    module.add_function(wrap_pyfunction!(py_is_swap_invariant, &module)?)?;
    module.add_function(wrap_pyfunction!(reorder_feature_tensor, &module)?)?;
    module.add_function(wrap_pyfunction!(conv_output_shape, &module)?)?;
    module.add_function(wrap_pyfunction!(pool_output_shape, &module)?)?;
    module.add_function(wrap_pyfunction!(save_json, &module)?)?;
    module.add_function(wrap_pyfunction!(load_json, &module)?)?;
    module.add_function(wrap_pyfunction!(save_bincode, &module)?)?;
    module.add_function(wrap_pyfunction!(load_bincode, &module)?)?;
    module.add_class::<PyCoherenceChannelReport>()?;
    module.add_class::<PyCoherenceSignature>()?;
    module.add_class::<PyCoherenceObservation>()?;
    module.add_class::<PyPreDiscardTelemetry>()?;
    module.add_class::<PyPreDiscardPolicy>()?;
    module.add_class::<PyPreDiscardSnapshot>()?;
    module.add_class::<PyCoherenceDiagnostics>()?;
    module.add_class::<PyZSpaceCoherenceSequencer>()?;
    module.add_class::<PyZSpaceTraceRecorder>()?;
    module.add_class::<PyMellinBasis>()?;
    module.add_class::<PyZSpaceVae>()?;
    module.add_class::<PyZSpaceVaeState>()?;
    module.add_class::<PyZSpaceVaeStats>()?;
    module.add_class::<PyZSpaceTextVae>()?;
    module.add_class::<PyZRelativityModule>()?;
    module.add_class::<PyCurvatureScheduler>()?;
    module.add_class::<PyCurvatureDecision>()?;
    module.add_function(wrap_pyfunction!(from_samples, &module)?)?;
    module.add(
        "__all__",
        vec![
            "Identity",
            "Linear",
            "Embedding",
            "FeatureReorder2d",
            "SpiralRnn",
            "WaveGate",
            "WaveRnn",
            "ZSpaceMixer",
            "ZSpaceSoftmax",
            "ZSpaceCoherenceScan",
            "ZSpaceCoherenceWaveBlock",
            "Relu",
            "Sequential",
            "MeanSquaredError",
            "CategoricalCrossEntropy",
            "HyperbolicCrossEntropy",
            "CrossEntropy",
            "FocalLoss",
            "ContrastiveLoss",
            "TripletLoss",
            "RoundtableConfig",
            "RoundtableSchedule",
            "EpochStats",
            "ModuleTrainer",
            "DesireTrainerBridge",
            "DesireRoundtableBridge",
            "DesireTelemetryBundle",
            "MaxwellDesireBridge",
            "NarrativeHint",
            "NarrativeSummary",
            "DesirePipeline",
            "save_json",
            "load_json",
            "save_bincode",
            "load_bincode",
            "NonLiner",
            "Scaler",
            "ZConv",
            "ZConv6DA",
            "ZPooling",
            "Dropout",
            "Pool2d",
            "Dataset",
            "DataLoader",
            "DataLoaderIter",
            "CoherenceChannelReport",
            "CoherenceDiagnostics",
            "PreDiscardTelemetry",
            "PreDiscardPolicy",
            "PreDiscardSnapshot",
            "ZSpaceCoherenceSequencer",
            "ZSpaceTraceRecorder",
            "MellinBasis",
            "ZSpaceVae",
            "ZSpaceVaeState",
            "ZSpaceVaeStats",
            "ZSpaceTextVae",
            "ZRelativityModule",
            "CurvatureScheduler",
            "CurvatureDecision",
            "from_samples",
            "reorder_feature_tensor",
            "conv_output_shape",
            "pool_output_shape",
        ],
    )?;
    parent.add_submodule(&module)?;
    if let Ok(identity) = module.getattr("Identity") {
        parent.add("Identity", identity)?;
    }
    if let Ok(linear) = module.getattr("Linear") {
        parent.add("Linear", linear)?;
    }
    if let Ok(relu) = module.getattr("Relu") {
        parent.add("Relu", relu)?;
    }
    if let Ok(sequential) = module.getattr("Sequential") {
        parent.add("Sequential", sequential)?;
    }
    if let Ok(mse) = module.getattr("MeanSquaredError") {
        parent.add("MeanSquaredError", mse)?;
    }
    if let Ok(non_liner) = module.getattr("NonLiner") {
        parent.add("NonLiner", non_liner)?;
    }
    if let Ok(scaler) = module.getattr("Scaler") {
        parent.add("Scaler", scaler)?;
    }
    if let Ok(zconv) = module.getattr("ZConv") {
        parent.add("ZConv", zconv)?;
    }
    if let Ok(zconv6da) = module.getattr("ZConv6DA") {
        parent.add("ZConv6DA", zconv6da)?;
    }
    if let Ok(zpool) = module.getattr("ZPooling") {
        parent.add("ZPooling", zpool)?;
    }
    if let Ok(reorder) = module.getattr("FeatureReorder2d") {
        parent.add("FeatureReorder2d", reorder)?;
    }
    if let Ok(wave_gate) = module.getattr("WaveGate") {
        parent.add("WaveGate", wave_gate)?;
    }
    if let Ok(wave_rnn) = module.getattr("WaveRnn") {
        parent.add("WaveRnn", wave_rnn)?;
    }
    if let Ok(mixer) = module.getattr("ZSpaceMixer") {
        parent.add("ZSpaceMixer", mixer)?;
    }
    if let Ok(dropout) = module.getattr("Dropout") {
        parent.add("Dropout", dropout)?;
    }
    if let Ok(pool2d) = module.getattr("Pool2d") {
        parent.add("Pool2d", pool2d)?;
    }
    if let Ok(sequencer) = module.getattr("ZSpaceCoherenceSequencer") {
        parent.add("ZSpaceCoherenceSequencer", sequencer)?;
    }
    if let Ok(recorder) = module.getattr("ZSpaceTraceRecorder") {
        parent.add("ZSpaceTraceRecorder", recorder)?;
    }
    if let Ok(basis) = module.getattr("MellinBasis") {
        parent.add("MellinBasis", basis)?;
    }
    if let Ok(vae) = module.getattr("ZSpaceVae") {
        parent.add("ZSpaceVae", vae)?;
    }
    if let Ok(text_vae) = module.getattr("ZSpaceTextVae") {
        parent.add("ZSpaceTextVae", text_vae)?;
    }
    if let Ok(scheduler) = module.getattr("CurvatureScheduler") {
        parent.add("CurvatureScheduler", scheduler)?;
    }
    if let Ok(decision) = module.getattr("CurvatureDecision") {
        parent.add("CurvatureDecision", decision)?;
    }
    if let Ok(channel_report) = module.getattr("CoherenceChannelReport") {
        parent.add("CoherenceChannelReport", channel_report)?;
    }
    if let Ok(signature) = module.getattr("CoherenceSignature") {
        parent.add("CoherenceSignature", signature)?;
    }
    if let Ok(observation) = module.getattr("CoherenceObservation") {
        parent.add("CoherenceObservation", observation)?;
    }
    if let Ok(telemetry) = module.getattr("PreDiscardTelemetry") {
        parent.add("PreDiscardTelemetry", telemetry)?;
    }
    if let Ok(policy) = module.getattr("PreDiscardPolicy") {
        parent.add("PreDiscardPolicy", policy)?;
    }
    if let Ok(snapshot) = module.getattr("PreDiscardSnapshot") {
        parent.add("PreDiscardSnapshot", snapshot)?;
    }
    if let Ok(diagnostics) = module.getattr("CoherenceDiagnostics") {
        parent.add("CoherenceDiagnostics", diagnostics)?;
    }
    if let Ok(zrelativity) = module.getattr("ZRelativityModule") {
        parent.add("ZRelativityModule", zrelativity)?;
    }
    if let Ok(is_swap_invariant) = module.getattr("is_swap_invariant") {
        parent.add("is_swap_invariant", is_swap_invariant)?;
    }
    Ok(())
}

#[cfg(not(feature = "nn"))]
fn register_impl(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "nn")?;
    module.add("__doc__", "SpiralTorch neural network primitives")?;
    parent.add_submodule(&module)?;
    Ok(())
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    register_impl(py, parent)
}
