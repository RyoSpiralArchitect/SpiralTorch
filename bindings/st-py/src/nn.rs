use pyo3::prelude::*;
use pyo3::types::PyModule;
#[cfg(feature = "nn")]
use pyo3::wrap_pyfunction;

#[cfg(feature = "nn")]
use pyo3::exceptions::PyValueError;

#[cfg(feature = "nn")]
use crate::pure::{PyGradientSummary, PyOpenCartesianTopos};
#[cfg(feature = "nn")]
use crate::tensor::{tensor_err_to_py, tensor_to_torch, PyTensor};
#[cfg(feature = "nn")]
use crate::theory::PyZRelativityModel;

#[cfg(feature = "nn")]
use st_core::{
    theory::zpulse::ZScale,
    util::math::{ramanujan_pi, LeechProjector},
};
#[cfg(feature = "nn")]
use st_nn::trainer::{
    CurvatureDecision as RustCurvatureDecision, CurvatureScheduler as RustCurvatureScheduler,
};
#[cfg(feature = "nn")]
use st_nn::Module;
#[cfg(feature = "nn")]
use st_nn::{
    dataset::DataLoaderBatches,
    dataset_from_vec,
    layers::{
        conv::{Conv2d, Conv6da},
        Dropout as RustDropout, Identity, NonLiner, NonLinerActivation, NonLinerEllipticConfig,
        NonLinerGeometry, NonLinerHyperbolicConfig, Scaler,
    },
    zspace_coherence::{
        is_swap_invariant as rust_is_swap_invariant, CoherenceDiagnostics, CoherenceLabel,
        CoherenceObservation, CoherenceSignature, LinguisticChannelReport, PreDiscardPolicy,
        PreDiscardSnapshot, PreDiscardTelemetry,
    },
    AvgPool2d, DataLoader, Dataset, MaxPool2d, ZRelativityModule, ZSpaceCoherenceSequencer,
};
#[cfg(feature = "nn")]
use st_tensor::{OpenCartesianTopos, Tensor, TensorError};

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
#[derive(Clone, Copy)]
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

    fn mode(&self) -> PoolMode {
        match self {
            Self::Max(_) => PoolMode::Max,
            Self::Avg(_) => PoolMode::Avg,
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
    inner: PoolModule,
    layout: Layout2d,
    input_dims: Spatial2d,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

#[cfg(feature = "nn")]
impl PyPool2d {
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
            inner,
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
        let output = self.inner.forward(&canonical).map_err(tensor_err_to_py)?;
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
            .inner
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
        self.inner.mode().as_str()
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
#[pyclass(module = "spiraltorch.nn", name = "NonLiner", unsendable)]
pub(crate) struct PyNonLiner {
    inner: NonLiner,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyNonLiner {
    #[new]
    #[pyo3(signature = (name, features, *, activation="tanh", slope=1.0, gain=1.0, bias=0.0, curvature=None, z_scale=None, retention=0.0))]
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

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    pub fn reset_metrics(&self) {
        self.inner.reset_metrics();
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
            let base = self.inner.geometry();
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
        self.inner.set_geometry(geometry);
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

    #[getter]
    pub fn activation(&self) -> String {
        match self.inner.activation() {
            NonLinerActivation::Tanh => "tanh".to_string(),
            NonLinerActivation::Sigmoid => "sigmoid".to_string(),
            NonLinerActivation::Softsign => "softsign".to_string(),
        }
    }

    #[getter]
    pub fn curvature(&self) -> Option<f32> {
        self.inner.geometry().curvature()
    }

    #[getter]
    pub fn z_scale(&self) -> Option<f32> {
        self.inner.geometry().z_scale().map(|scale| scale.value())
    }

    #[getter]
    pub fn retention(&self) -> Option<f32> {
        self.inner.geometry().retention()
    }

    #[getter]
    pub fn psi_drift(&self) -> Option<f32> {
        self.inner.psi_probe()
    }

    #[getter]
    pub fn last_hyperbolic_radius(&self) -> Option<f32> {
        self.inner.last_hyperbolic_radius()
    }

    #[getter]
    pub fn gain(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.gain().value().clone())
    }

    #[getter]
    pub fn slope(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.slope().value().clone())
    }

    #[getter]
    pub fn bias(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.bias().value().clone())
    }

    pub fn gradients(&self) -> (Option<PyTensor>, Option<PyTensor>, Option<PyTensor>) {
        let gain = self
            .inner
            .gain()
            .gradient()
            .map(|g| PyTensor::from_tensor(g.clone()));
        let slope = self
            .inner
            .slope()
            .gradient()
            .map(|g| PyTensor::from_tensor(g.clone()));
        let bias = self
            .inner
            .bias()
            .gradient()
            .map(|g| PyTensor::from_tensor(g.clone()));
        (gain, slope, bias)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "Scaler", unsendable)]
pub(crate) struct PyScaler {
    inner: Scaler,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyScaler {
    #[new]
    pub fn new(name: &str, features: usize) -> PyResult<Self> {
        let inner = Scaler::new(name, features).map_err(tensor_err_to_py)?;
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

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }

    pub fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        self.inner
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
            self.inner
                .attach_hypergrad_with_topos(curvature, learning_rate, topos.inner.clone())
                .map_err(tensor_err_to_py)
        } else {
            self.attach_hypergrad(curvature, learning_rate)
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

    #[getter]
    pub fn gain(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.gain().value().clone())
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "Dropout", unsendable)]
pub(crate) struct PyDropout {
    inner: RustDropout,
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

    pub fn set_training(&mut self, training: bool) {
        self.inner.set_training(training);
    }

    pub fn train(&mut self) {
        self.inner.train();
    }

    pub fn eval(&mut self) {
        self.inner.eval();
    }

    #[getter]
    pub fn training(&self) -> bool {
        self.inner.training()
    }

    #[getter]
    pub fn probability(&self) -> f32 {
        self.inner.probability()
    }

    #[pyo3(signature = (x))]
    pub fn __call__(&self, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(x)
    }
}

#[cfg(feature = "nn")]
#[pyclass(module = "spiraltorch.nn", name = "ZConv", unsendable)]
pub(crate) struct PyZConv {
    inner: Conv2d,
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
            inner,
        })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let canonical_input = self
            .input_to_canonical(&input.inner)
            .map_err(tensor_err_to_py)?;
        let canonical_output = self
            .inner
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
            .inner
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
        self.inner.load_state_dict(&map).map_err(tensor_err_to_py)?;
        self.refresh_output_dims().map_err(tensor_err_to_py)
    }

    pub fn set_dilation(&mut self, dilation: (usize, usize)) -> PyResult<()> {
        self.inner
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
    inner: Conv6da,
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
            inner,
        })
    }

    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let canonical_input = self
            .input_to_canonical(&input.inner)
            .map_err(tensor_err_to_py)?;
        let canonical_output = self
            .inner
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
            .inner
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
    inner: PoolModule,
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
            inner,
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
            .inner
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
            .inner
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
#[pyclass(module = "spiraltorch.nn")]
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
#[pyclass(module = "spiraltorch.nn", unsendable)]
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
#[pyclass(module = "spiraltorch.nn", unsendable)]
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
        Ok(Self { inner })
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
    module.add_class::<PyCoherenceChannelReport>()?;
    module.add_class::<PyCoherenceSignature>()?;
    module.add_class::<PyCoherenceObservation>()?;
    module.add_class::<PyPreDiscardTelemetry>()?;
    module.add_class::<PyPreDiscardPolicy>()?;
    module.add_class::<PyPreDiscardSnapshot>()?;
    module.add_class::<PyCoherenceDiagnostics>()?;
    module.add_class::<PyZSpaceCoherenceSequencer>()?;
    module.add_class::<PyZRelativityModule>()?;
    module.add_class::<PyCurvatureScheduler>()?;
    module.add_class::<PyCurvatureDecision>()?;
    module.add_function(wrap_pyfunction!(from_samples, &module)?)?;
    module.add(
        "__all__",
        vec![
            "Identity",
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
    if let Ok(dropout) = module.getattr("Dropout") {
        parent.add("Dropout", dropout)?;
    }
    if let Ok(pool2d) = module.getattr("Pool2d") {
        parent.add("Pool2d", pool2d)?;
    }
    if let Ok(sequencer) = module.getattr("ZSpaceCoherenceSequencer") {
        parent.add("ZSpaceCoherenceSequencer", sequencer)?;
    }
    if let Ok(scheduler) = module.getattr("CurvatureScheduler") {
        parent.add("CurvatureScheduler", scheduler)?;
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
