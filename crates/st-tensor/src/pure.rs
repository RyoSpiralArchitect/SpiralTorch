// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Pure Rust tensor, complex wave, and non-Euclidean learning primitives with
//! only lightweight external dependencies.
//!
//! The goal of this module is to offer a pragmatic starting point for
//! high-dimensional experimentation that **does not rely on PyTorch, NumPy, or
//! any other native bindings**. Everything here is written in safe Rust so it
//! can serve as a foundation for a fully independent learning stack that stays
//! responsive even when the surrounding platform is sandboxed.

pub mod differential;
pub mod fractal;
pub mod measure;
pub mod topos;
pub mod wasm_canvas;

pub use self::differential::{
    DifferentialResonance, FunctorDifferential, HomotopyDifferential, InfinityDifferential,
    RecursiveDifferential, SpiralDifferential,
};
pub use self::measure::{
    nirt_weight_update, tesla_tail_spectrum, z_space_barycenter, z_space_barycenter_guarded,
    BarycenterIntermediate, TeslaTail, TeslaTailLine, ZSpaceBarycenter,
};
pub use self::topos::{
    GraphGuardProfile, GraphGuardReport, LawvereTierneyGuard, ModalityProfile, MultiModalToposGuard,
    OpenCartesianTopos, RewardBoundary, RewardBoundarySignal, RewriteMonad, TensorBiome, ToposAtlas,
    ZBox, ZBoxSite,
};

use crate::backend::faer_dense;
#[cfg(feature = "wgpu")]
use crate::backend::wgpu_dense;
use crate::dlpack::{
    call_managed_deleter, drop_exported_state, DLDataType, DLDataTypeCode, DLDevice, DLDeviceType,
    DLManagedTensor, DLTensor, ExportData, ForeignTensor, ManagedTensorState,
};
use core::fmt;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;
#[allow(unused_imports)]
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::error::Error;
use std::f32::consts::PI;
use std::ffi::c_void;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;

/// Result alias used throughout the pure module.
pub type PureResult<T> = Result<T, TensorError>;

/// Errors emitted by tensor and wave utilities.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorError {
    /// A tensor constructor received an invalid shape.
    InvalidDimensions { rows: usize, cols: usize },
    /// Data provided to a constructor or operator does not match the tensor shape.
    DataLength { expected: usize, got: usize },
    /// An operator was asked to combine tensors of incompatible shapes.
    ShapeMismatch {
        left: (usize, usize),
        right: (usize, usize),
    },
    /// A value that must be strictly negative curvature was not.
    NonHyperbolicCurvature { curvature: f32 },
    /// Temperature must stay positive for wave encoders.
    NonPositiveTemperature { temperature: f32 },
    /// Learning rate must be positive for hypergrad optimizers.
    NonPositiveLearningRate { rate: f32 },
    /// Coherence weights that scale fractal relations must stay positive.
    NonPositiveCoherence { coherence: f32 },
    /// Tension weights that soften relations must stay positive.
    NonPositiveTension { tension: f32 },
    /// Topos tolerance must stay positive to avoid degeneracy.
    NonPositiveTolerance { tolerance: f32 },
    /// Topos saturation window must stay positive.
    NonPositiveSaturation { saturation: f32 },
    /// Tensor biome weights must stay positive when accumulating shoots.
    NonPositiveWeight { weight: f32 },
    /// Computation received an empty input which would otherwise trigger a panic.
    EmptyInput(&'static str),
    /// Weighted Z-space barycenter collapsed because the total KL + entropy temperature vanished.
    DegenerateBarycenter { effective_weight: f32 },
    /// Attempted to load or update a parameter that was missing from the state dict.
    MissingParameter { name: String },
    /// Wrapper around I/O failures when persisting or restoring tensors.
    IoError { message: String },
    /// Wrapper around serde failures when deserialising tensors.
    SerializationError { message: String },
    /// A helper expected matching curvature parameters but received different values.
    CurvatureMismatch { expected: f32, got: f32 },
    /// Numeric guard detected a non-finite value that would otherwise propagate NaNs.
    NonFiniteValue { label: &'static str, value: f32 },
    /// The requested tensor volume exceeds the configured open-cartesian topos boundary.
    TensorVolumeExceeded { volume: usize, max_volume: usize },
    /// Loop detection tripped for an open-cartesian topos traversal.
    LoopDetected { depth: usize, max_depth: usize },
    /// Conjugate gradient solver could not reach the requested tolerance.
    ConjugateGradientDiverged { residual: f32, tolerance: f32 },
    /// Execution failed on an accelerator backend.
    BackendFailure {
        backend: &'static str,
        message: String,
    },
    /// Generic configuration violation for pure-language helpers.
    InvalidValue { label: &'static str },
    /// Interoperability bridge encountered an unsupported or malformed DLPack tensor.
    DlpackError { message: String },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::InvalidDimensions { rows, cols } => {
                write!(
                    f,
                    "invalid tensor dimensions ({rows} x {cols}); both axes must be non-zero"
                )
            }
            TensorError::DataLength { expected, got } => {
                write!(f, "data length mismatch: expected {expected}, got {got}")
            }
            TensorError::ShapeMismatch { left, right } => {
                write!(
                    f,
                    "shape mismatch: left={:?}, right={:?} cannot be combined",
                    left, right
                )
            }
            TensorError::NonHyperbolicCurvature { curvature } => {
                write!(
                    f,
                    "hyperbolic operations require negative curvature; received {curvature}"
                )
            }
            TensorError::NonPositiveTemperature { temperature } => {
                write!(
                    f,
                    "language wave encoder temperature must be positive, got {temperature}"
                )
            }
            TensorError::NonPositiveLearningRate { rate } => {
                write!(f, "learning rate must be positive, got {rate}")
            }
            TensorError::NonPositiveCoherence { coherence } => {
                write!(f, "coherence must be positive, got {coherence}")
            }
            TensorError::NonPositiveTension { tension } => {
                write!(f, "tension must be positive, got {tension}")
            }
            TensorError::NonPositiveTolerance { tolerance } => {
                write!(f, "tolerance must be positive, got {tolerance}")
            }
            TensorError::NonPositiveSaturation { saturation } => {
                write!(f, "saturation window must be positive, got {saturation}")
            }
            TensorError::NonPositiveWeight { weight } => {
                write!(f, "tensor biome weight must be positive, got {weight}")
            }
            TensorError::EmptyInput(label) => {
                write!(f, "{label} must not be empty for this computation")
            }
            TensorError::DegenerateBarycenter { effective_weight } => {
                write!(
                    f,
                    "z-space barycenter degenerates when the total weight temperature ({effective_weight}) is non-positive"
                )
            }
            TensorError::CurvatureMismatch { expected, got } => {
                write!(
                    f,
                    "curvature mismatch: expected {expected} but pipeline reports {got}"
                )
            }
            TensorError::NonFiniteValue { label, value } => {
                write!(
                    f,
                    "non-finite value detected for {label}; rewrite monad absorbed {value}"
                )
            }
            TensorError::MissingParameter { name } => {
                write!(f, "missing parameter '{name}' while loading module state")
            }
            TensorError::IoError { message } => {
                write!(f, "i/o error while handling tensor data: {message}")
            }
            TensorError::SerializationError { message } => {
                write!(
                    f,
                    "serialization error while handling tensor data: {message}"
                )
            }
            TensorError::TensorVolumeExceeded { volume, max_volume } => {
                write!(
                    f,
                    "tensor volume {volume} exceeds open-cartesian capacity {max_volume}"
                )
            }
            TensorError::LoopDetected { depth, max_depth } => {
                write!(
                    f,
                    "topos traversal depth {depth} exceeded loop-free ceiling {max_depth}"
                )
            }
            TensorError::ConjugateGradientDiverged {
                residual,
                tolerance,
            } => {
                write!(
                    f,
                    "conjugate gradient residual {residual} failed to reach tolerance {tolerance}"
                )
            }
            TensorError::BackendFailure { backend, message } => {
                write!(f, "{backend} backend failure: {message}")
            }
            TensorError::InvalidValue { label } => {
                write!(f, "invalid value: {label}")
            }
            TensorError::DlpackError { message } => {
                write!(f, "dlpack error: {message}")
            }
        }
    }
}

impl Error for TensorError {}

/// Explicit matrix multiplication backend selection. `Auto` defers to the heuristics, while
/// `CpuFaer`/`GpuWgpu` force the respective accelerators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulBackend {
    /// Use heuristics to pick the best available backend.
    Auto,
    /// Force the SIMD-accelerated faer kernel.
    CpuFaer,
    /// Always fallback to the scalar implementation.
    CpuNaive,
    /// Force the compute-path GEMM running through WGPU.
    #[cfg(feature = "wgpu")]
    GpuWgpu,
}

impl MatmulBackend {
    fn label(self) -> &'static str {
        match self {
            MatmulBackend::Auto => "auto",
            MatmulBackend::CpuFaer => "faer",
            MatmulBackend::CpuNaive => "naive",
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => "wgpu",
        }
    }
}

impl fmt::Display for MatmulBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str((*self).label())
    }
}

#[derive(Clone, Debug)]
enum TensorBacking {
    Owned(Arc<Vec<f32>>),
    Foreign(ForeignTensor),
}

#[derive(Clone, Debug)]
pub(crate) struct TensorBuffer {
    backing: TensorBacking,
}

impl TensorBuffer {
    fn from_vec(data: Vec<f32>) -> Self {
        Self {
            backing: TensorBacking::Owned(Arc::new(data)),
        }
    }

    fn from_foreign(foreign: ForeignTensor) -> Self {
        Self {
            backing: TensorBacking::Foreign(foreign),
        }
    }

    fn as_slice(&self) -> &[f32] {
        match &self.backing {
            TensorBacking::Owned(vec) => vec.as_slice(),
            TensorBacking::Foreign(foreign) => foreign.as_slice(),
        }
    }

    fn make_mut_slice(&mut self) -> &mut [f32] {
        if let TensorBacking::Foreign(foreign) = &self.backing {
            let owned = foreign.to_vec();
            self.backing = TensorBacking::Owned(Arc::new(owned));
        }

        if let TensorBacking::Owned(vec) = &mut self.backing {
            Arc::make_mut(vec).as_mut_slice()
        } else {
            unreachable!()
        }
    }

    fn export_handle(&self) -> ExportData {
        match &self.backing {
            TensorBacking::Owned(vec) => ExportData::Owned(Arc::clone(vec)),
            TensorBacking::Foreign(foreign) => ExportData::Foreign(foreign.clone()),
        }
    }
}

impl Deref for TensorBuffer {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for TensorBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.make_mut_slice()
    }
}

impl PartialEq for TensorBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// A simple row-major 2D tensor backed by a reference-counted buffer.
#[derive(Clone, Debug)]
pub struct Tensor {
    data: Arc<TensorBuffer>,
    rows: usize,
    cols: usize,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self.data.as_slice() == other.data.as_slice()
    }
}

impl Tensor {
    fn seedable_rng(seed: Option<u64>) -> StdRng {
        match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => StdRng::from_entropy(),
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> PureResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        Ok(Self {
            data: Arc::new(TensorBuffer::from_vec(vec![0.0; rows * cols])),
            rows,
            cols,
        })
    }

    /// Create a tensor from raw data. The provided vector must match
    /// `rows * cols` elements.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> PureResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        let expected = rows * cols;
        if expected != data.len() {
            return Err(TensorError::DataLength {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            data: Arc::new(TensorBuffer::from_vec(data)),
            rows,
            cols,
        })
    }

    /// Construct a tensor by sampling a uniform distribution in `[min, max)`.
    ///
    /// When `seed` is provided the RNG becomes deterministic which makes tests
    /// and benchmarks reproducible. Otherwise entropy from the host is used.
    pub fn random_uniform(
        rows: usize,
        cols: usize,
        min: f32,
        max: f32,
        seed: Option<u64>,
    ) -> PureResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        if !(min < max) {
            return Err(TensorError::InvalidValue {
                label: "random_uniform_bounds",
            });
        }
        let mut rng = Self::seedable_rng(seed);
        let distribution = Uniform::new(min, max);
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(distribution.sample(&mut rng));
        }
        Self::from_vec(rows, cols, data)
    }

    /// Construct a tensor by sampling a normal distribution with the provided
    /// mean and standard deviation.
    pub fn random_normal(
        rows: usize,
        cols: usize,
        mean: f32,
        std: f32,
        seed: Option<u64>,
    ) -> PureResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        if std <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "random_normal_std",
            });
        }
        let mut rng = Self::seedable_rng(seed);
        let gaussian = StandardNormal;
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            let sample: f64 = gaussian.sample(&mut rng);
            data.push(mean + std * sample as f32);
        }
        Self::from_vec(rows, cols, data)
    }

    /// Construct a tensor by applying a generator function to each coordinate.
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> PureResult<Self>
    where
        F: FnMut(usize, usize) -> f32,
    {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self::from_vec(rows, cols, data)
    }

    /// Returns the `(rows, cols)` pair of the tensor.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns a read-only view of the underlying buffer.
    pub fn data(&self) -> &[f32] {
        self.data.as_slice()
    }

    /// Returns a mutable view of the underlying buffer.
    pub fn data_mut(&mut self) -> &mut [f32] {
        Arc::make_mut(&mut self.data).make_mut_slice()
    }

    /// Export the tensor as a managed DLPack tensor.
    pub fn to_dlpack(&self) -> PureResult<*mut DLManagedTensor> {
        let rows_i64 = i64::try_from(self.rows).map_err(|_| TensorError::DlpackError {
            message: "tensor rows exceed i64 range".to_string(),
        })?;
        let cols_i64 = i64::try_from(self.cols).map_err(|_| TensorError::DlpackError {
            message: "tensor cols exceed i64 range".to_string(),
        })?;

        let export = self.data.export_handle();
        let mut state = Box::new(ManagedTensorState::new(
            export,
            vec![rows_i64, cols_i64].into_boxed_slice(),
            vec![cols_i64, 1].into_boxed_slice(),
        ));

        let dl_tensor = DLTensor {
            data: state.data.as_ptr() as *mut c_void,
            device: DLDevice {
                device_type: DLDeviceType::Cpu as i32,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: DLDataTypeCode::Float as u8,
                bits: 32,
                lanes: 1,
            },
            shape: state.shape.as_mut_ptr(),
            strides: state.strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let manager_ctx = Box::into_raw(state) as *mut ManagedTensorState as *mut c_void;
        let managed = Box::new(DLManagedTensor {
            dl_tensor,
            manager_ctx,
            deleter: Some(drop_exported_state),
        });

        Ok(Box::into_raw(managed))
    }

    /// Construct a tensor from a managed DLPack tensor. The managed tensor is consumed.
    ///
    /// # Safety
    /// The caller must ensure `managed` points to a valid `DLManagedTensor`.
    pub unsafe fn from_dlpack(managed: *mut DLManagedTensor) -> PureResult<Self> {
        struct ManagedGuard {
            ptr: Option<NonNull<DLManagedTensor>>,
        }

        impl ManagedGuard {
            unsafe fn new(ptr: NonNull<DLManagedTensor>) -> Self {
                Self { ptr: Some(ptr) }
            }

            fn tensor(&self) -> &DLManagedTensor {
                unsafe { self.ptr.unwrap().as_ref() }
            }

            fn into_inner(mut self) -> NonNull<DLManagedTensor> {
                self.ptr.take().unwrap()
            }
        }

        impl Drop for ManagedGuard {
            fn drop(&mut self) {
                if let Some(ptr) = self.ptr.take() {
                    unsafe {
                        call_managed_deleter(ptr.as_ptr());
                    }
                }
            }
        }

        let managed_ptr = match NonNull::new(managed) {
            Some(ptr) => ptr,
            None => {
                return Err(TensorError::EmptyInput("dlpack tensor"));
            }
        };

        let guard = ManagedGuard::new(managed_ptr);
        let tensor = guard.tensor();
        let dl_tensor = &tensor.dl_tensor;

        if dl_tensor.ndim != 2 {
            return Err(TensorError::DlpackError {
                message: format!("expected 2 dimensions, got {}", dl_tensor.ndim),
            });
        }

        if dl_tensor.device.device_type != DLDeviceType::Cpu as i32 {
            return Err(TensorError::DlpackError {
                message: format!(
                    "unsupported device type {}; only CPU tensors are accepted",
                    dl_tensor.device.device_type
                ),
            });
        }

        if dl_tensor.dtype.code != DLDataTypeCode::Float as u8
            || dl_tensor.dtype.bits != 32
            || dl_tensor.dtype.lanes != 1
        {
            return Err(TensorError::DlpackError {
                message: "only f32 tensors are supported".to_string(),
            });
        }

        if dl_tensor.shape.is_null() {
            return Err(TensorError::DlpackError {
                message: "dlpack tensor provided a null shape pointer".to_string(),
            });
        }

        if tensor.deleter.is_none() {
            return Err(TensorError::DlpackError {
                message: "dlpack tensor is missing a deleter".to_string(),
            });
        }

        let ndim = usize::try_from(dl_tensor.ndim).map_err(|_| TensorError::DlpackError {
            message: format!("invalid ndim {}", dl_tensor.ndim),
        })?;

        let shape = slice::from_raw_parts(dl_tensor.shape, ndim);
        let (rows_i64, cols_i64) = match *shape {
            [rows, cols] => (rows, cols),
            _ => {
                return Err(TensorError::DlpackError {
                    message: "shape must contain exactly two dimensions".to_string(),
                })
            }
        };

        if rows_i64 <= 0 || cols_i64 <= 0 {
            return Err(TensorError::InvalidDimensions {
                rows: rows_i64.max(0) as usize,
                cols: cols_i64.max(0) as usize,
            });
        }

        let rows = usize::try_from(rows_i64).map_err(|_| TensorError::DlpackError {
            message: format!("rows {rows_i64} exceed usize range"),
        })?;
        let cols = usize::try_from(cols_i64).map_err(|_| TensorError::DlpackError {
            message: format!("cols {cols_i64} exceed usize range"),
        })?;

        let expected_len = rows
            .checked_mul(cols)
            .ok_or_else(|| TensorError::DlpackError {
                message: "tensor volume overflow".to_string(),
            })?;

        if dl_tensor.byte_offset % mem::size_of::<f32>() != 0 {
            return Err(TensorError::DlpackError {
                message: format!(
                    "byte offset {} is not aligned to f32 elements",
                    dl_tensor.byte_offset
                ),
            });
        }

        let offset = dl_tensor.byte_offset / mem::size_of::<f32>();
        if offset > expected_len {
            return Err(TensorError::DlpackError {
                message: format!("byte offset {offset} exceeds tensor length {expected_len}",),
            });
        }

        let base_ptr = match NonNull::new(dl_tensor.data as *mut f32) {
            Some(ptr) => ptr,
            None => {
                return Err(TensorError::EmptyInput("dlpack tensor data"));
            }
        };

        if !dl_tensor.strides.is_null() {
            let strides = slice::from_raw_parts(dl_tensor.strides, ndim);
            let expected_row = i64::try_from(cols).map_err(|_| TensorError::DlpackError {
                message: format!("cols {cols} exceed i64 range"),
            })?;
            if strides != [expected_row, 1] {
                return Err(TensorError::DlpackError {
                    message: format!(
                        "only contiguous row-major tensors are supported; received strides {:?}",
                        strides
                    ),
                });
            }
        }

        let data_ptr = unsafe { base_ptr.as_ptr().add(offset) };
        let data_ptr = match NonNull::new(data_ptr) {
            Some(ptr) => ptr,
            None => {
                return Err(TensorError::EmptyInput("dlpack tensor data"));
            }
        };

        let foreign = unsafe { ForeignTensor::new(guard.into_inner(), data_ptr, expected_len) };

        Ok(Self {
            data: Arc::new(TensorBuffer::from_foreign(foreign)),
            rows,
            cols,
        })
    }

    /// Matrix multiply (`self @ other`).
    pub fn matmul(&self, other: &Tensor) -> PureResult<Tensor> {
        self.matmul_with_backend(other, MatmulBackend::Auto)
    }

    /// Matrix multiply with an explicit backend selection.
    pub fn matmul_with_backend(
        &self,
        other: &Tensor,
        backend: MatmulBackend,
    ) -> PureResult<Tensor> {
        if self.cols != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }

        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;

        let data = match backend {
            MatmulBackend::Auto => self.matmul_auto(other, rows, inner, cols)?,
            MatmulBackend::CpuNaive => matmul_naive(self.data(), other.data(), rows, inner, cols),
            MatmulBackend::CpuFaer => matmul_faer(self.data(), other.data(), rows, inner, cols)?,
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => matmul_wgpu(self.data(), other.data(), rows, inner, cols)?,
        };

        Tensor::from_vec(rows, cols, data)
    }

    fn matmul_auto(
        &self,
        other: &Tensor,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<Vec<f32>> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                if let Ok(buffer) = wgpu_dense::matmul(self.data(), other.data(), rows, inner, cols)
                {
                    return Ok(buffer);
                }
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            if let Ok(buffer) = faer_dense::matmul(self.data(), other.data(), rows, inner, cols) {
                return Ok(buffer);
            }
        }

        Ok(matmul_naive(self.data(), other.data(), rows, inner, cols))
    }

    /// Matrix multiply using the WGPU backend when available.
    #[cfg(feature = "wgpu")]
    pub fn matmul_wgpu(&self, other: &Tensor) -> PureResult<Tensor> {
        if self.cols != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        let data = wgpu_dense::matmul(self.data(), other.data(), self.rows, other.cols, self.cols)
            .map_err(|message| TensorError::BackendFailure {
                backend: "wgpu",
                message,
            })?;
        Tensor::from_vec(self.rows, other.cols, data)
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Tensor) -> PureResult<Tensor> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            data.push(a + b);
        }
        Tensor::from_vec(self.rows, self.cols, data)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> PureResult<Tensor> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            data.push(a - b);
        }
        Tensor::from_vec(self.rows, self.cols, data)
    }

    /// Returns a new tensor where every element is scaled by `value`.
    pub fn scale(&self, value: f32) -> PureResult<Tensor> {
        let mut data = Vec::with_capacity(self.data.len());
        for &a in self.data.iter() {
            data.push(a * value);
        }
        Tensor::from_vec(self.rows, self.cols, data)
    }

    /// Element-wise product (Hadamard) between two tensors of identical shape.
    pub fn hadamard(&self, other: &Tensor) -> PureResult<Tensor> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            data.push(a * b);
        }
        Tensor::from_vec(self.rows, self.cols, data)
    }

    /// Add a scaled tensor to this tensor (`self += scale * other`).
    pub fn add_scaled(&mut self, other: &Tensor, scale: f32) -> PureResult<()> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        let data = Arc::make_mut(&mut self.data);
        for (a, b) in data.iter_mut().zip(other.data.iter()) {
            *a += scale * b;
        }
        Ok(())
    }

    /// Add the provided row vector to every row (`self[row] += bias`).
    pub fn add_row_inplace(&mut self, bias: &[f32]) -> PureResult<()> {
        if bias.len() != self.cols {
            return Err(TensorError::DataLength {
                expected: self.cols,
                got: bias.len(),
            });
        }
        let data = Arc::make_mut(&mut self.data);
        for r in 0..self.rows {
            let offset = r * self.cols;
            for c in 0..self.cols {
                data[offset + c] += bias[c];
            }
        }
        Ok(())
    }

    /// Returns the transpose of the tensor.
    pub fn transpose(&self) -> Tensor {
        let mut data = vec![0.0; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        Tensor {
            data: Arc::new(TensorBuffer::from_vec(data)),
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Returns a reshaped copy of the tensor when the requested dimensions are compatible.
    pub fn reshape(&self, rows: usize, cols: usize) -> PureResult<Tensor> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        if rows * cols != self.data.len() {
            return Err(TensorError::DataLength {
                expected: rows * cols,
                got: self.data.len(),
            });
        }
        Tensor::from_vec(rows, cols, self.data().to_vec())
    }

    /// Returns the sum over rows for each column.
    pub fn sum_axis0(&self) -> Vec<f32> {
        let mut sums = vec![0.0; self.cols];
        for r in 0..self.rows {
            let offset = r * self.cols;
            for c in 0..self.cols {
                sums[c] += self.data[offset + c];
            }
        }
        sums
    }

    /// Concatenates tensors row-wise producing a new tensor whose row count is the sum
    /// of the inputs while preserving the shared column dimension.
    pub fn cat_rows(tensors: &[Tensor]) -> PureResult<Tensor> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyInput("Tensor::cat_rows"));
        }
        let cols = tensors[0].cols;
        if cols == 0 {
            return Err(TensorError::InvalidDimensions { rows: 0, cols });
        }
        let mut total_rows = 0usize;
        for tensor in tensors {
            if tensor.cols != cols {
                return Err(TensorError::ShapeMismatch {
                    left: tensor.shape(),
                    right: (tensor.rows, cols),
                });
            }
            total_rows += tensor.rows;
        }
        let mut data = Vec::with_capacity(total_rows * cols);
        for tensor in tensors {
            data.extend_from_slice(tensor.data.as_slice());
        }
        Tensor::from_vec(total_rows, cols, data)
    }

    /// Computes the squared L2 norm of the tensor.
    pub fn squared_l2_norm(&self) -> f32 {
        self.data.iter().map(|v| v * v).sum()
    }

    /// Projects a flattened tensor onto the Poincaré ball.
    pub fn project_to_poincare(&self, curvature: f32) -> PureResult<Tensor> {
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        let scale = (-curvature).sqrt();
        let mut data = Vec::with_capacity(self.data.len());
        for r in 0..self.rows {
            let start = r * self.cols;
            let end = start + self.cols;
            let chunk = &self.data[start..end];
            let norm: f32 = chunk.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                let clip = (norm / scale).tanh();
                let factor = clip / norm;
                for v in chunk {
                    data.push(v * factor);
                }
            } else {
                data.extend_from_slice(chunk);
            }
        }
        Tensor::from_vec(self.rows, self.cols, data)
    }

    /// Estimates the hyperbolic distance between two flattened tensors treated as points.
    pub fn hyperbolic_distance(&self, other: &Tensor, curvature: f32) -> PureResult<f32> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        let scale = (-curvature).sqrt();
        let mut sum_norm = 0.0f32;
        let mut sum_inner = 0.0f32;
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            let pa = a / scale;
            let pb = b / scale;
            sum_norm += (pa - pb).powi(2);
            sum_inner += (1.0 - pa.powi(2)) * (1.0 - pb.powi(2));
        }
        let denom = sum_inner.max(1e-6).sqrt();
        Ok(2.0 * (1.0 + (sum_norm / denom)).acosh())
    }
}

/// Computes the mean squared error between `predictions` and `targets`.
pub fn mean_squared_error(predictions: &Tensor, targets: &Tensor) -> PureResult<f32> {
    if predictions.shape() != targets.shape() {
        return Err(TensorError::ShapeMismatch {
            left: predictions.shape(),
            right: targets.shape(),
        });
    }
    let mut sum = 0.0f32;
    for (p, t) in predictions.data().iter().zip(targets.data().iter()) {
        let diff = p - t;
        sum += diff * diff;
    }
    Ok(sum / (predictions.rows * predictions.cols) as f32)
}

/// A minimal fully-connected linear model.
///
/// The model keeps its weights and bias in plain Rust vectors so it can be
/// embedded in `no_std` friendly environments (alloc-only) and easily extended
/// with additional layers.
#[derive(Clone, Debug)]
pub struct LinearModel {
    weights: Tensor,
    bias: Vec<f32>,
}

impl LinearModel {
    /// Creates a new linear model with small deterministic parameters.
    pub fn new(input_dim: usize, output_dim: usize) -> PureResult<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: input_dim,
                cols: output_dim,
            });
        }
        let mut weights = Tensor::zeros(input_dim, output_dim)?;
        let mut scale = 0.01f32;
        for w in weights.data_mut().iter_mut() {
            *w = scale;
            scale += 0.01;
        }
        Ok(Self {
            weights,
            bias: vec![0.0; output_dim],
        })
    }

    /// Runs a forward pass: `inputs @ weights + bias`.
    pub fn forward(&self, inputs: &Tensor) -> PureResult<Tensor> {
        if inputs.shape().1 != self.weights.shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: inputs.shape(),
                right: self.weights.shape(),
            });
        }
        let mut out = inputs.matmul(&self.weights)?;
        out.add_row_inplace(&self.bias)?;
        Ok(out)
    }

    /// Performs a single batch of gradient descent and returns the batch loss.
    pub fn train_batch(
        &mut self,
        inputs: &Tensor,
        targets: &Tensor,
        learning_rate: f32,
    ) -> PureResult<f32> {
        if inputs.shape().0 != targets.shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: inputs.shape(),
                right: targets.shape(),
            });
        }
        if targets.shape().1 != self.weights.shape().1 {
            return Err(TensorError::ShapeMismatch {
                left: targets.shape(),
                right: self.weights.shape(),
            });
        }
        let batch_size = inputs.shape().0 as f32;
        let predictions = self.forward(inputs)?;
        let diff = predictions.sub(targets)?;
        let inputs_t = inputs.transpose();
        let grad_w = inputs_t.matmul(&diff)?.scale(1.0 / batch_size)?;
        let mut grad_b = diff.sum_axis0();
        for val in grad_b.iter_mut() {
            *val /= batch_size;
        }
        self.weights.add_scaled(&grad_w, -learning_rate)?;
        for (b, g) in self.bias.iter_mut().zip(grad_b.iter()) {
            *b -= learning_rate * g;
        }
        Ok(mean_squared_error_from_diff(&diff))
    }

    /// Returns a reference to the model weights.
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Returns a reference to the model bias.
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
}

fn mean_squared_error_from_diff(diff: &Tensor) -> f32 {
    let mut sum = 0.0f32;
    for v in diff.data() {
        sum += v * v;
    }
    sum / (diff.rows * diff.cols) as f32
}

/// Lightweight complex number for wave encodings without external crates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn modulus(self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn argument(self) -> f32 {
        self.im.atan2(self.re)
    }
}

impl core::ops::Add for Complex32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl core::ops::AddAssign for Complex32 {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl core::ops::Mul for Complex32 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl core::ops::Mul<f32> for Complex32 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl core::ops::Sub for Complex32 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

/// Complex tensor storing WGSL/FFT-friendly waveforms.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexTensor {
    data: Vec<Complex32>,
    rows: usize,
    cols: usize,
}

impl ComplexTensor {
    pub fn zeros(rows: usize, cols: usize) -> PureResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        Ok(Self {
            data: vec![Complex32::zero(); rows * cols],
            rows,
            cols,
        })
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<Complex32>) -> PureResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        let expected = rows * cols;
        if expected != data.len() {
            return Err(TensorError::DataLength {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            data: data.into(),
            rows,
            cols,
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn data(&self) -> &[Complex32] {
        &self.data
    }

    /// Converts the complex tensor into a real tensor by splitting real/imag parts.
    pub fn to_tensor(&self) -> PureResult<Tensor> {
        let mut data = Vec::with_capacity(self.data.len() * 2);
        for value in &self.data {
            data.push(value.re);
            data.push(value.im);
        }
        Tensor::from_vec(self.rows, self.cols * 2, data)
    }

    pub fn matmul(&self, other: &ComplexTensor) -> PureResult<Self> {
        if self.cols != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        let mut out = ComplexTensor::zeros(self.rows, other.cols)?;
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[r * self.cols + k];
                let row_offset = k * other.cols;
                for c in 0..other.cols {
                    out.data[r * other.cols + c] += a * other.data[row_offset + c];
                }
            }
        }
        Ok(out)
    }
}

fn discrete_fourier_transform(signal: &[f32]) -> Vec<Complex32> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let factor = -2.0 * PI / n as f32;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let mut accum = Complex32::zero();
        for (idx, &value) in signal.iter().enumerate() {
            let angle = factor * (k as f32) * (idx as f32);
            let basis = Complex32::new(angle.cos(), angle.sin());
            accum += basis * value;
        }
        out.push(accum);
    }
    out
}

/// Encodes language streams into complex Z-space waves without tokenization.
#[derive(Clone, Debug)]
pub struct LanguageWaveEncoder {
    curvature: f32,
    temperature: f32,
}

impl LanguageWaveEncoder {
    pub fn new(curvature: f32, temperature: f32) -> PureResult<Self> {
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if temperature <= 0.0 {
            return Err(TensorError::NonPositiveTemperature { temperature });
        }
        Ok(Self {
            curvature,
            temperature,
        })
    }

    /// Returns the curvature used for Z-space encodings.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the thermal scaling applied to the wavefront.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Convert a sentence directly into a complex wave on the Z-space manifold.
    pub fn encode_wave(&self, text: &str) -> PureResult<ComplexTensor> {
        let char_count = text.chars().count();
        if char_count == 0 {
            return Err(TensorError::EmptyInput("text"));
        }
        let mut signal = Vec::with_capacity(char_count);
        let denom = char_count as f32;
        for (idx, ch) in text.chars().enumerate() {
            let ordinal = ch as u32;
            let bucket = (ordinal % 1024) as f32 / 1024.0;
            let envelope = ((idx as f32 + 1.0) / denom).sin().abs();
            signal.push((bucket + envelope) * self.temperature);
        }
        let spectrum = discrete_fourier_transform(&signal);
        ComplexTensor::from_vec(1, spectrum.len(), spectrum)
    }

    /// Projects the encoded wave into a hyperbolic chart for transformer-friendly inference.
    pub fn encode_z_space(&self, text: &str) -> PureResult<Tensor> {
        let wave = self.encode_wave(text)?;
        if wave.data().is_empty() {
            return Err(TensorError::EmptyInput("wave"));
        }
        let mut coords = Vec::with_capacity(wave.data().len() * 2);
        let curvature_scale = (-self.curvature).sqrt();
        for complex in wave.data() {
            let radius = (complex.modulus() * curvature_scale).tanh();
            let angle = complex.argument();
            coords.push(radius * angle.cos());
            coords.push(radius * angle.sin());
        }
        let euclid = Tensor::from_vec(1, coords.len(), coords)?;
        euclid.project_to_poincare(self.curvature)
    }
}

/// Hyperbolic gradient accumulator for zero-traceback learning loops.
///
/// The optimiser keeps its own curvature-aligned gradient buffer and can
/// integrate Euclidean tensors, complex waves, or direct text streams emitted
/// by [`LanguageWaveEncoder`]. Every update is projected back onto the
/// Poincaré ball so state never escapes the non-Euclidean manifold.
pub struct AmegaHypergrad {
    curvature: f32,
    learning_rate: f32,
    rows: usize,
    cols: usize,
    gradient: Vec<f32>,
    topos: topos::OpenCartesianTopos,
}

/// Euclidean gradient accumulator that mirrors the hypergradient API while
/// staying entirely within flat-space optimisation loops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientSummary {
    l1: f32,
    l2: f32,
    linf: f32,
    count: usize,
}

impl GradientSummary {
    #[inline]
    pub fn from_slice(values: &[f32]) -> Self {
        let mut l1 = 0.0f32;
        let mut sum_squares = 0.0f32;
        let mut linf = 0.0f32;
        let mut count = 0usize;
        for &value in values {
            if !value.is_finite() {
                continue;
            }
            let abs = value.abs();
            l1 += abs;
            sum_squares += value * value;
            linf = linf.max(abs);
            count += 1;
        }
        let l2 = sum_squares.sqrt();
        Self {
            l1,
            l2,
            linf,
            count,
        }
    }

    /// Builds a summary directly from raw moment statistics. `l1` captures the
    /// sum of absolute values, `sum_squares` is the accumulated \(L_2^2\)
    /// energy, `linf` is the maximum absolute entry, and `count` indicates how
    /// many samples contributed to the summary.
    #[inline]
    pub fn from_moments(l1: f32, sum_squares: f32, linf: f32, count: usize) -> Self {
        let l1 = if l1.is_finite() { l1.max(0.0) } else { 0.0 };
        let sum_squares = if sum_squares.is_finite() {
            sum_squares.max(0.0)
        } else {
            0.0
        };
        let linf = if linf.is_finite() { linf.max(0.0) } else { 0.0 };
        Self {
            l1,
            l2: sum_squares.sqrt(),
            linf,
            count,
        }
    }

    #[inline]
    pub fn l1(&self) -> f32 {
        self.l1
    }

    #[inline]
    pub fn l2(&self) -> f32 {
        self.l2
    }

    #[inline]
    pub fn linf(&self) -> f32 {
        self.linf
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn mean_abs(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.l1 / self.count as f32
        }
    }

    #[inline]
    pub fn rms(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.l2 / (self.count as f32).sqrt()
        }
    }

    /// Returns the accumulated sum of squares captured by the summary.
    #[inline]
    pub fn sum_squares(&self) -> f32 {
        self.l2 * self.l2
    }
}

impl Default for GradientSummary {
    fn default() -> Self {
        Self {
            l1: 0.0,
            l2: 0.0,
            linf: 0.0,
            count: 0,
        }
    }
}

/// Interprets paired gradient summaries into high-level signals that can drive
/// Desire Lagrangian feedback loops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesireGradientInterpretation {
    hyper_pressure: f32,
    real_pressure: f32,
    balance: f32,
    stability: f32,
    saturation: f32,
}

impl DesireGradientInterpretation {
    const EPS: f32 = 1e-6;

    /// Analyse the hypergradient and Euclidean summaries returning a compact
    /// interpretation of their relative magnitudes and stability.
    #[inline]
    pub fn from_summaries(hyper: GradientSummary, real: GradientSummary) -> Self {
        let hyper_pressure = hyper.mean_abs();
        let real_pressure = real.mean_abs();
        let hyper_rms = hyper.rms();
        let real_rms = real.rms();
        let balance = if real_rms > Self::EPS {
            (hyper_rms / real_rms).clamp(0.0, 16.0)
        } else if hyper_rms > Self::EPS {
            16.0
        } else {
            1.0
        };
        let stability_raw = 1.0
            - (hyper_pressure - real_pressure).abs() / (hyper_pressure + real_pressure + Self::EPS);
        let stability = stability_raw.clamp(0.0, 1.0);
        let saturation = hyper.linf().max(real.linf());
        Self {
            hyper_pressure,
            real_pressure,
            balance,
            stability,
            saturation,
        }
    }

    /// Mean absolute magnitude of the hypergradient summary.
    #[inline]
    pub fn hyper_pressure(&self) -> f32 {
        self.hyper_pressure
    }

    /// Mean absolute magnitude of the Euclidean gradient summary.
    #[inline]
    pub fn real_pressure(&self) -> f32 {
        self.real_pressure
    }

    /// Ratio between the hypergradient and Euclidean RMS magnitudes.
    #[inline]
    pub fn balance(&self) -> f32 {
        self.balance
    }

    /// Stability score where `1` indicates matching mean-absolute gradients and
    /// `0` indicates divergent magnitudes.
    #[inline]
    pub fn stability(&self) -> f32 {
        self.stability
    }

    /// Maximum absolute value observed across both summaries.
    #[inline]
    pub fn saturation(&self) -> f32 {
        self.saturation
    }

    /// Gain factor for hypergradient penalties when the two tapes disagree.
    #[inline]
    pub fn penalty_gain(&self) -> f32 {
        let imbalance = (self.balance - 1.0).abs().min(2.0);
        let instability = (1.0 - self.stability).min(1.0);
        (1.0 + 0.5 * imbalance + 0.5 * instability).clamp(1.0, 2.5)
    }

    /// Mixing factor used when blending Desire bias updates – drops towards zero
    /// when the gradients disagree so the automation can tread lightly.
    #[inline]
    pub fn bias_mix(&self) -> f32 {
        (0.25 + 0.75 * self.stability).clamp(0.1, 1.0)
    }

    /// Gain used when accumulating avoidance reports during the observation
    /// phase. High saturation dampens the contribution to avoid runaway spikes.
    #[inline]
    pub fn observation_gain(&self) -> f32 {
        let saturation = self.saturation.tanh().clamp(0.0, 1.0);
        (0.5 + 0.5 * (1.0 - saturation)).clamp(0.25, 1.0)
    }

    /// Damping factor that can shrink epsilon-like tolerances when gradients
    /// spike.
    #[inline]
    pub fn damping(&self) -> f32 {
        (0.5 + 0.5 * self.saturation.tanh()).clamp(0.1, 1.0)
    }

    /// Collapse the interpretation into ready-to-apply control signals for
    /// Desire's automation layers. The returned structure encodes the tuned
    /// penalty, mixing, and learning-rate factors so downstream consumers can
    /// steer both the CPU and GPU loops without reimplementing the heuristics.
    #[inline]
    pub fn control(&self) -> DesireGradientControl {
        self.control_with_gain(1.0)
    }

    /// Collapse the interpretation into a control packet while scaling the
    /// adaptive heuristics by `gain`. Passing `0.0` retains the legacy neutral
    /// behaviour whereas `1.0` enables the full tuning guidance.
    #[inline]
    pub fn control_with_gain(&self, gain: f32) -> DesireGradientControl {
        DesireGradientControl::from_interpretation_with_gain(*self, gain)
    }
}

impl Default for DesireGradientInterpretation {
    fn default() -> Self {
        Self {
            hyper_pressure: 0.0,
            real_pressure: 0.0,
            balance: 1.0,
            stability: 1.0,
            saturation: 0.0,
        }
    }
}

/// Event bitflags describing notable actions suggested by
/// [`DesireGradientControl`]. The bitmask can be surfaced directly to telemetry
/// systems without additional allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DesireControlEvents {
    bits: u32,
}

impl DesireControlEvents {
    pub const NONE: Self = Self { bits: 0 };
    pub const LR_INCREASE: Self = Self { bits: 1 << 0 };
    pub const LR_DECREASE: Self = Self { bits: 1 << 1 };
    pub const LR_CLIPPED: Self = Self { bits: 1 << 2 };
    pub const TEMPERATURE_SUPPRESS: Self = Self { bits: 1 << 3 };
    pub const QUALITY_BOOST: Self = Self { bits: 1 << 4 };
    pub const QUALITY_SUPPRESS: Self = Self { bits: 1 << 5 };
    pub const Z_SUPPRESS: Self = Self { bits: 1 << 6 };
    pub const SLEW_LIMIT: Self = Self { bits: 1 << 7 };

    #[inline]
    pub const fn new(bits: u32) -> Self {
        Self { bits }
    }

    #[inline]
    pub const fn bits(self) -> u32 {
        self.bits
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    #[inline]
    pub const fn insert(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    pub fn labels(self) -> Vec<&'static str> {
        let mut labels = Vec::new();
        if self.contains(Self::LR_INCREASE) {
            labels.push("lr_increase");
        }
        if self.contains(Self::LR_DECREASE) {
            labels.push("lr_decrease");
        }
        if self.contains(Self::LR_CLIPPED) {
            labels.push("lr_clipped");
        }
        if self.contains(Self::TEMPERATURE_SUPPRESS) {
            labels.push("temperature_suppress");
        }
        if self.contains(Self::QUALITY_BOOST) {
            labels.push("quality_weight");
        }
        if self.contains(Self::QUALITY_SUPPRESS) {
            labels.push("quality_suppress");
        }
        if self.contains(Self::Z_SUPPRESS) {
            labels.push("z_suppress");
        }
        if self.contains(Self::SLEW_LIMIT) {
            labels.push("lr_slew_limit");
        }
        labels
    }
}

impl Default for DesireControlEvents {
    fn default() -> Self {
        Self::NONE
    }
}

/// Packaged feedback parameters derived from [`DesireGradientInterpretation`]
/// that automation layers can apply directly to gradient tapes, WGSL kernels,
/// and Desire's avoidance/bias mixers.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DesireGradientControl {
    penalty_gain: f32,
    bias_mix: f32,
    observation_gain: f32,
    damping: f32,
    hyper_rate_scale: f32,
    real_rate_scale: f32,
    operator_mix: f32,
    operator_gain: f32,
    #[serde(default = "DesireGradientControl::default_tuning_gain")]
    tuning_gain: f32,
    #[serde(default = "DesireGradientControl::default_target_entropy")]
    target_entropy: f32,
    #[serde(default = "DesireGradientControl::default_lr_eta")]
    learning_rate_eta: f32,
    #[serde(default = "DesireGradientControl::default_lr_min")]
    learning_rate_min: f32,
    #[serde(default = "DesireGradientControl::default_lr_max")]
    learning_rate_max: f32,
    #[serde(default = "DesireGradientControl::default_lr_slew")]
    learning_rate_slew: f32,
    #[serde(default = "DesireGradientControl::default_clip_norm")]
    clip_norm: f32,
    #[serde(default = "DesireGradientControl::default_clip_floor")]
    clip_floor: f32,
    #[serde(default = "DesireGradientControl::default_clip_ceiling")]
    clip_ceiling: f32,
    #[serde(default = "DesireGradientControl::default_clip_ema")]
    clip_ema: f32,
    #[serde(default = "DesireGradientControl::default_temperature_kappa")]
    temperature_kappa: f32,
    #[serde(default = "DesireGradientControl::default_temperature_slew")]
    temperature_slew: f32,
    #[serde(default = "DesireGradientControl::default_quality_gain")]
    quality_gain: f32,
    #[serde(default = "DesireGradientControl::default_quality_bias")]
    quality_bias: f32,
    #[serde(default = "DesireGradientControl::default_events")]
    events: DesireControlEvents,
}

impl DesireGradientControl {
    const fn default_tuning_gain() -> f32 {
        0.0
    }

    const fn default_target_entropy() -> f32 {
        3.5
    }

    const fn default_lr_eta() -> f32 {
        1.0
    }

    const fn default_lr_min() -> f32 {
        1e-4
    }

    const fn default_lr_max() -> f32 {
        3e-3
    }

    const fn default_lr_slew() -> f32 {
        0.2
    }

    const fn default_clip_norm() -> f32 {
        1.0
    }

    const fn default_clip_floor() -> f32 {
        0.1
    }

    const fn default_clip_ceiling() -> f32 {
        8.0
    }

    const fn default_clip_ema() -> f32 {
        0.5
    }

    const fn default_temperature_kappa() -> f32 {
        0.0
    }

    const fn default_temperature_slew() -> f32 {
        0.15
    }

    const fn default_quality_gain() -> f32 {
        4.0
    }

    const fn default_quality_bias() -> f32 {
        0.1
    }

    const fn default_events() -> DesireControlEvents {
        DesireControlEvents::NONE
    }

    /// Build a control packet from an interpretation, collapsing the
    /// higher-level descriptors into concrete gains and learning-rate scales.
    pub fn from_interpretation(interpretation: DesireGradientInterpretation) -> Self {
        Self::from_interpretation_with_gain(interpretation, 1.0)
    }

    /// Build a control packet while scaling the adaptive heuristics by `gain`.
    pub fn from_interpretation_with_gain(
        interpretation: DesireGradientInterpretation,
        gain: f32,
    ) -> Self {
        DesireControlBuilder::new()
            .with_interpretation(interpretation)
            .with_gain(gain)
            .finalise()
    }

    /// Hypergradient penalty amplification factor mirrored from the
    /// interpretation stage.
    pub fn penalty_gain(&self) -> f32 {
        self.penalty_gain
    }

    /// Desire bias blending factor for integration phases.
    pub fn bias_mix(&self) -> f32 {
        self.bias_mix
    }

    /// Observation accumulation gain applied to avoidance tracking.
    pub fn observation_gain(&self) -> f32 {
        self.observation_gain
    }

    /// Recommended damping factor for epsilon / tolerance schedules.
    pub fn damping(&self) -> f32 {
        self.damping
    }

    /// Multiplicative scale applied to the hypergradient learning rate.
    pub fn hyper_rate_scale(&self) -> f32 {
        self.hyper_rate_scale
    }

    /// Multiplicative scale applied to the Euclidean learning rate.
    pub fn real_rate_scale(&self) -> f32 {
        self.real_rate_scale
    }

    /// Suggested blend factor for GPU hypergradient kernels.
    pub fn operator_mix(&self) -> f32 {
        self.operator_mix
    }

    /// Suggested gain for GPU hypergradient kernels.
    pub fn operator_gain(&self) -> f32 {
        self.operator_gain
    }

    /// Strength applied to the adaptive heuristics. `0.0` disables the tuning
    /// feedback while `1.0` enables the recommended behaviour.
    pub fn tuning_gain(&self) -> f32 {
        self.tuning_gain
    }

    /// Target entropy that the learning-rate controller should chase.
    pub fn target_entropy(&self) -> f32 {
        self.target_entropy
    }

    /// Exponential learning-rate update coefficient.
    pub fn learning_rate_eta(&self) -> f32 {
        self.learning_rate_eta
    }

    /// Lower bound for the adaptive learning rate.
    pub fn learning_rate_min(&self) -> f32 {
        self.learning_rate_min
    }

    /// Upper bound for the adaptive learning rate.
    pub fn learning_rate_max(&self) -> f32 {
        self.learning_rate_max
    }

    /// Maximum relative change permitted between learning-rate updates.
    pub fn learning_rate_slew(&self) -> f32 {
        self.learning_rate_slew
    }

    /// Recommended gradient clipping norm.
    pub fn clip_norm(&self) -> f32 {
        self.clip_norm
    }

    /// Floor applied when smoothing the clipping window.
    pub fn clip_floor(&self) -> f32 {
        self.clip_floor
    }

    /// Ceiling applied when smoothing the clipping window.
    pub fn clip_ceiling(&self) -> f32 {
        self.clip_ceiling
    }

    /// Exponential moving-average factor for the clipping window.
    pub fn clip_ema(&self) -> f32 {
        self.clip_ema
    }

    /// Coupling factor applied when adjusting Desire temperature against the
    /// Z-order magnitude.
    pub fn temperature_kappa(&self) -> f32 {
        self.temperature_kappa
    }

    /// Maximum allowed change in Desire temperature per step.
    pub fn temperature_slew(&self) -> f32 {
        self.temperature_slew
    }

    /// Gain applied when incorporating external quality metrics.
    pub fn quality_gain(&self) -> f32 {
        self.quality_gain
    }

    /// Baseline offset for external quality metrics.
    pub fn quality_bias(&self) -> f32 {
        self.quality_bias
    }

    /// Event bitmask describing the adjustments suggested by the control.
    pub fn events(&self) -> DesireControlEvents {
        self.events
    }
}

impl Default for DesireGradientControl {
    fn default() -> Self {
        Self {
            penalty_gain: 1.0,
            bias_mix: 1.0,
            observation_gain: 1.0,
            damping: 1.0,
            hyper_rate_scale: 1.0,
            real_rate_scale: 1.0,
            operator_mix: 1.0,
            operator_gain: 1.0,
            tuning_gain: Self::default_tuning_gain(),
            target_entropy: Self::default_target_entropy(),
            learning_rate_eta: Self::default_lr_eta(),
            learning_rate_min: Self::default_lr_min(),
            learning_rate_max: Self::default_lr_max(),
            learning_rate_slew: Self::default_lr_slew(),
            clip_norm: Self::default_clip_norm(),
            clip_floor: Self::default_clip_floor(),
            clip_ceiling: Self::default_clip_ceiling(),
            clip_ema: Self::default_clip_ema(),
            temperature_kappa: Self::default_temperature_kappa(),
            temperature_slew: Self::default_temperature_slew(),
            quality_gain: Self::default_quality_gain(),
            quality_bias: Self::default_quality_bias(),
            events: Self::default_events(),
        }
    }
}

impl From<DesireGradientInterpretation> for DesireGradientControl {
    fn from(value: DesireGradientInterpretation) -> Self {
        Self::from_interpretation(value)
    }
}

/// Builder that assembles a [`DesireGradientControl`] packet from
/// interpretation metrics and live telemetry (entropy, Z magnitude, quality
/// estimates, etc.).
#[derive(Clone, Debug)]
pub struct DesireControlBuilder {
    control: DesireGradientControl,
    gain: f32,
    entropy: f32,
    target_entropy: f32,
    entropy_eta: f32,
    lr_min: f32,
    lr_max: f32,
    lr_slew: f32,
    hyper_base: f32,
    real_base: f32,
    operator_mix: f32,
    operator_gain: f32,
    clip_hint: f32,
    clip_floor: f32,
    clip_ceiling: f32,
    clip_ema: f32,
    z_magnitude: f32,
    z_kappa: f32,
    z_slew: f32,
    quality: f32,
    quality_bias: f32,
    quality_gain: f32,
    events: DesireControlEvents,
}

impl DesireControlBuilder {
    fn new() -> Self {
        Self {
            control: DesireGradientControl::default(),
            gain: 1.0,
            entropy: 0.8,
            target_entropy: DesireGradientControl::default_target_entropy(),
            entropy_eta: DesireGradientControl::default_lr_eta(),
            lr_min: DesireGradientControl::default_lr_min(),
            lr_max: DesireGradientControl::default_lr_max(),
            lr_slew: DesireGradientControl::default_lr_slew(),
            hyper_base: 1.0,
            real_base: 1.0,
            operator_mix: 1.0,
            operator_gain: 1.0,
            clip_hint: DesireGradientControl::default_clip_norm(),
            clip_floor: DesireGradientControl::default_clip_floor(),
            clip_ceiling: DesireGradientControl::default_clip_ceiling(),
            clip_ema: DesireGradientControl::default_clip_ema(),
            z_magnitude: 0.0,
            z_kappa: DesireGradientControl::default_temperature_kappa(),
            z_slew: DesireGradientControl::default_temperature_slew(),
            quality: 1.0,
            quality_bias: DesireGradientControl::default_quality_bias(),
            quality_gain: DesireGradientControl::default_quality_gain(),
            events: DesireControlEvents::NONE,
        }
    }

    pub fn with_interpretation(mut self, interpretation: DesireGradientInterpretation) -> Self {
        self.control.penalty_gain = interpretation.penalty_gain();
        self.control.bias_mix = interpretation.bias_mix();
        self.control.observation_gain = interpretation.observation_gain();
        self.control.damping = interpretation.damping();

        let imbalance = (interpretation.balance() - 1.0).clamp(-4.0, 4.0);
        let saturation = interpretation.saturation().tanh().clamp(0.0, 1.0);
        let stability = interpretation.stability().clamp(0.0, 1.0);
        let caution = (1.0 - stability).clamp(0.0, 1.0);

        let (hyper_base, real_base) = if imbalance >= 0.0 {
            (
                (1.0 / (1.0 + imbalance)).clamp(0.25, 1.0),
                (1.0 + 0.5 * imbalance).clamp(1.0, 1.8),
            )
        } else {
            (
                (1.0 - 0.5 * imbalance).clamp(1.0, 1.8),
                (1.0 / (1.0 - imbalance)).clamp(0.25, 1.0),
            )
        };

        let hyper_guard = (1.0 - 0.6 * caution - 0.4 * saturation).clamp(0.25, 1.0);
        let real_guard = (1.0 - 0.4 * caution - 0.25 * saturation).clamp(0.3, 1.0);
        self.hyper_base = (hyper_base * hyper_guard).clamp(0.25, 1.8);
        self.real_base = (real_base * real_guard).clamp(0.25, 1.8);

        self.operator_mix = (0.4 + 0.6 * stability).clamp(0.25, 1.0);
        self.operator_gain =
            (self.control.penalty_gain * (1.0 - 0.35 * saturation)).clamp(0.5, 1.5);

        self.target_entropy = 3.5 + 0.8 * caution;
        self.entropy_eta = 0.08 + 0.14 * caution;
        self.lr_slew = (0.25 - 0.15 * caution).clamp(0.05, 0.25);

        let clip_floor = (0.18 + 0.12 * caution).clamp(0.15, 0.3);
        let clip_target = interpretation
            .saturation()
            .max(interpretation.hyper_pressure() * 2.5)
            .max(interpretation.real_pressure() * 3.0);
        self.clip_floor = clip_floor;
        self.clip_hint = (clip_target * (0.6 + 0.4 * self.gain)).clamp(clip_floor, 32.0);
        self.clip_ceiling = (self.clip_hint * 1.6).max(self.clip_hint + 0.05);
        self.clip_ema = (0.25 + 0.35 * caution).clamp(0.2, 0.6);

        self.z_kappa = (0.02 + 0.08 * (1.0 - stability)).clamp(0.0, 0.12);
        self.z_slew = (0.22 - 0.1 * caution).clamp(0.05, 0.22);

        self.quality_gain = (0.6 + 0.4 * (1.0 - caution)).clamp(0.4, 1.0);

        self
    }

    pub fn with_gain(mut self, gain: f32) -> Self {
        self.gain = gain.clamp(0.0, 1.0);
        self
    }

    pub fn with_entropy(mut self, entropy: f32) -> Self {
        self.entropy = entropy.max(0.0);
        self
    }

    pub fn with_target_entropy(mut self, target: f32) -> Self {
        self.target_entropy = target.max(0.0);
        self
    }

    pub fn with_entropy_eta(mut self, eta: f32) -> Self {
        self.entropy_eta = eta.max(0.0);
        self
    }

    pub fn with_bounds(mut self, min: f32, max: f32) -> Self {
        self.lr_min = min.min(max).max(1e-8);
        self.lr_max = max.max(self.lr_min + 1e-8);
        self
    }

    pub fn with_slew_limit(mut self, slew: f32) -> Self {
        self.lr_slew = slew.max(0.0);
        self
    }

    pub fn with_clip_p95_hint(mut self, p95: f32) -> Self {
        self.clip_hint = p95.max(0.0);
        self
    }

    pub fn with_clip_parameters(mut self, floor: f32, ceiling: f32, ema: f32) -> Self {
        self.clip_floor = floor.max(0.0);
        self.clip_ceiling = ceiling.max(self.clip_floor + 1e-6);
        self.clip_ema = ema.clamp(0.0, 1.0);
        self
    }

    pub fn with_z_coupling(mut self, magnitude: f32) -> Self {
        self.z_magnitude = magnitude.abs();
        self
    }

    pub fn with_z_strength(mut self, kappa: f32) -> Self {
        self.z_kappa = kappa.max(0.0);
        self
    }

    pub fn with_quality(mut self, quality: f32) -> Self {
        self.quality = quality.max(0.0);
        self
    }

    pub fn with_quality_parameters(mut self, gain: f32, bias: f32) -> Self {
        self.quality_gain = gain.max(0.0);
        self.quality_bias = bias.clamp(0.0, 1.0);
        self
    }

    pub fn finalise(mut self) -> DesireGradientControl {
        self.control.tuning_gain = self.gain;
        self.control.target_entropy = self.target_entropy;
        self.control.learning_rate_eta = self.entropy_eta * self.gain;
        self.control.learning_rate_min = self.lr_min;
        self.control.learning_rate_max = self.lr_max;
        self.control.learning_rate_slew = self.lr_slew;

        self.control.clip_floor = self.clip_floor;
        self.control.clip_ceiling = self.clip_ceiling.max(self.clip_floor + 1e-6);
        self.control.clip_ema = self.clip_ema;
        let clip_norm = self
            .clip_hint
            .clamp(self.control.clip_floor, self.control.clip_ceiling);
        self.control.clip_norm = clip_norm;

        let z_kappa = self.z_kappa * self.gain;
        self.control.temperature_kappa = z_kappa;
        self.control.temperature_slew = self.z_slew;
        let z_suppress = (-z_kappa * self.z_magnitude).exp().clamp(0.05, 1.0);
        if z_kappa > 0.0 && self.z_magnitude > 1e-3 {
            self.events = self.events.insert(DesireControlEvents::Z_SUPPRESS);
            self.events = self
                .events
                .insert(DesireControlEvents::TEMPERATURE_SUPPRESS);
        }

        let entropy_error = (self.target_entropy - self.entropy).clamp(-8.0, 8.0);
        if entropy_error > 0.05 {
            self.events = self.events.insert(DesireControlEvents::LR_INCREASE);
        } else if entropy_error < -0.05 {
            self.events = self.events.insert(DesireControlEvents::LR_DECREASE);
        }

        if (clip_norm - self.control.clip_floor).abs() < 1e-4 {
            self.events = self.events.insert(DesireControlEvents::LR_CLIPPED);
        }

        if self.lr_slew < 0.1 {
            self.events = self.events.insert(DesireControlEvents::SLEW_LIMIT);
        }

        let quality_bias = self.quality_bias;
        self.control.quality_bias = quality_bias;
        let quality_gain = (self.quality_gain * self.gain).max(0.0);
        self.control.quality_gain = quality_gain;
        let q_delta = (self.quality - quality_bias).clamp(-4.0, 4.0);
        let quality_weight = 1.0 / (1.0 + (-quality_gain * q_delta).exp());
        if quality_weight > 0.55 {
            self.events = self.events.insert(DesireControlEvents::QUALITY_BOOST);
        } else if quality_weight < 0.45 {
            self.events = self.events.insert(DesireControlEvents::QUALITY_SUPPRESS);
        }

        let quality_scale = 0.4 + 0.9 * quality_weight;
        self.control.hyper_rate_scale =
            (self.hyper_base * z_suppress * quality_scale).clamp(0.1, 2.0);
        self.control.real_rate_scale =
            (self.real_base * z_suppress * quality_scale).clamp(0.1, 2.0);

        self.control.operator_mix = self.operator_mix;
        self.control.operator_gain = self.operator_gain;
        self.control.events = self.events;
        self.control
    }
}

impl DesireGradientControl {
    pub fn control_with_gain() -> DesireControlBuilder {
        DesireControlBuilder::new()
    }
}

pub struct AmegaRealgrad {
    learning_rate: f32,
    rows: usize,
    cols: usize,
    gradient: Vec<f32>,
}

impl AmegaHypergrad {
    /// Create a new hypergradient tape with the provided curvature and step size.
    pub fn new(curvature: f32, learning_rate: f32, rows: usize, cols: usize) -> PureResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if learning_rate <= 0.0 {
            return Err(TensorError::NonPositiveLearningRate {
                rate: learning_rate,
            });
        }
        let topos = topos::OpenCartesianTopos::new(
            curvature,
            1e-6,
            1e6,
            rows.saturating_mul(cols).saturating_mul(4).max(8),
            rows.saturating_mul(cols),
        )?;
        Ok(Self {
            curvature,
            learning_rate,
            rows,
            cols,
            gradient: vec![0.0; rows * cols],
            topos,
        })
    }

    /// Builds a hypergradient tape with a caller supplied open-cartesian topos.
    pub fn with_topos(
        curvature: f32,
        learning_rate: f32,
        rows: usize,
        cols: usize,
        topos: topos::OpenCartesianTopos,
    ) -> PureResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if learning_rate <= 0.0 {
            return Err(TensorError::NonPositiveLearningRate {
                rate: learning_rate,
            });
        }
        let capacity = rows.saturating_mul(cols);
        if capacity > topos.max_volume() {
            return Err(TensorError::TensorVolumeExceeded {
                volume: capacity,
                max_volume: topos.max_volume(),
            });
        }
        Ok(Self {
            curvature,
            learning_rate,
            rows,
            cols,
            gradient: vec![0.0; capacity],
            topos,
        })
    }

    /// Returns the hyperbolic curvature the tape is operating under.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the geometric learning rate used for each Riemannian step.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Returns the `(rows, cols)` dimensions captured by the tape.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Provides read-only access to the accumulated gradient buffer.
    pub fn gradient(&self) -> &[f32] {
        &self.gradient
    }

    /// Provides mutable access to the accumulated gradient buffer.
    pub fn gradient_mut(&mut self) -> &mut [f32] {
        &mut self.gradient
    }

    /// Summarise the accumulated gradient using basic norm statistics.
    pub fn summary(&self) -> GradientSummary {
        GradientSummary::from_slice(&self.gradient)
    }

    /// Returns the guard topos enforcing open-cartesian safety constraints.
    pub fn topos(&self) -> &topos::OpenCartesianTopos {
        &self.topos
    }

    /// Scales the learning rate used by subsequent hyperbolic updates.
    pub fn scale_learning_rate(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.learning_rate *= factor;
        }
    }

    /// Retunes the curvature and learning rate while rebuilding the guard topos.
    /// The accumulated gradient buffer is cleared to avoid mixing incompatible
    /// geometries across updates.
    pub fn retune(&mut self, curvature: f32, learning_rate: f32) -> PureResult<()> {
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if learning_rate <= 0.0 {
            return Err(TensorError::NonPositiveLearningRate {
                rate: learning_rate,
            });
        }
        let topos = topos::OpenCartesianTopos::new(
            curvature,
            self.topos.tolerance(),
            self.topos.saturation(),
            self.topos.max_depth(),
            self.topos.max_volume(),
        )?;
        self.curvature = curvature;
        self.learning_rate = learning_rate;
        self.topos = topos;
        self.reset();
        Ok(())
    }

    fn assert_tensor_shape(&self, tensor: &Tensor) -> PureResult<()> {
        if tensor.shape() != (self.rows, self.cols) {
            return Err(TensorError::ShapeMismatch {
                left: tensor.shape(),
                right: (self.rows, self.cols),
            });
        }
        Ok(())
    }

    /// Clears the accumulated gradient back to zero.
    pub fn reset(&mut self) {
        for g in self.gradient.iter_mut() {
            *g = 0.0;
        }
    }

    /// Accumulates a Euclidean tensor inside the hyperbolic tape using
    /// the standard conformal factor for the Poincaré ball.
    pub fn accumulate_wave(&mut self, tensor: &Tensor) -> PureResult<()> {
        self.assert_tensor_shape(tensor)?;
        self.topos.guard_tensor("hypergrad_wave", tensor)?;
        let tolerance = self.topos.tolerance();
        for (grad, value) in self.gradient.iter_mut().zip(tensor.data().iter()) {
            let denom = 1.0 - self.curvature * value * value;
            let update = *value / denom.abs().max(tolerance);
            *grad = self.topos.saturate(*grad + update);
        }
        Ok(())
    }

    /// Accumulates a complex wave by unfolding it into real and imaginary lanes.
    pub fn accumulate_complex_wave(&mut self, wave: &ComplexTensor) -> PureResult<()> {
        let tensor = wave.to_tensor()?;
        self.accumulate_wave(&tensor)
    }

    /// Absorbs text directly by encoding it into Z-space using the provided encoder.
    pub fn absorb_text(&mut self, encoder: &LanguageWaveEncoder, text: &str) -> PureResult<()> {
        if (encoder.curvature() - self.curvature).abs() > 1e-6 {
            return Err(TensorError::CurvatureMismatch {
                expected: self.curvature,
                got: encoder.curvature(),
            });
        }
        self.topos.ensure_loop_free(text.chars().count())?;
        let tensor = encoder.encode_z_space(text)?;
        self.accumulate_wave(&tensor)
    }

    /// Integrates a prediction/target pair to build a hyperbolic residual.
    pub fn accumulate_pair(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<()> {
        self.assert_tensor_shape(prediction)?;
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        self.topos
            .guard_tensor("hypergrad_prediction", prediction)?;
        self.topos.guard_tensor("hypergrad_target", target)?;
        for ((grad, pred), tgt) in self
            .gradient
            .iter_mut()
            .zip(prediction.data().iter())
            .zip(target.data().iter())
        {
            let delta = pred - tgt;
            *grad = self.topos.saturate(*grad + delta);
        }
        Ok(())
    }

    /// Applies the accumulated gradient to the provided tensor and reprojects it
    /// into the Poincaré ball. The gradient buffer is cleared afterwards so the
    /// tape can keep streaming samples without triggering a traceback.
    pub fn apply(&mut self, weights: &mut Tensor) -> PureResult<()> {
        self.assert_tensor_shape(weights)?;
        self.topos.guard_tensor("hypergrad_weights", weights)?;
        let tolerance = self.topos.tolerance();
        {
            let data = weights.data_mut();
            for (value, grad) in data.iter_mut().zip(self.gradient.iter()) {
                let denom = 1.0 - self.curvature * (*value) * (*value);
                let step = self.learning_rate / denom.abs().max(tolerance);
                let updated = *value - step * *grad;
                *value = self.topos.saturate(updated);
            }
        }
        let projected = weights.project_to_poincare(self.curvature)?;
        *weights = projected;
        self.topos
            .guard_tensor("hypergrad_weights_post_projection", weights)?;
        self.reset();
        Ok(())
    }

    /// Integrates the barycenter intermediate curve so the tape converges towards
    /// the Z-space minimiser through a loss-monotone path.
    pub fn accumulate_barycenter_path(
        &mut self,
        intermediates: &[BarycenterIntermediate],
    ) -> PureResult<()> {
        if intermediates.is_empty() {
            return Err(TensorError::EmptyInput("barycenter_intermediates"));
        }
        for stage in intermediates {
            self.assert_tensor_shape(&stage.density)?;
        }
        let mut stages = intermediates.iter();
        let first = stages.next().unwrap();
        self.accumulate_wave(&first.density)?;
        let mut previous = &first.density;
        for stage in stages {
            self.accumulate_pair(&stage.density, previous)?;
            previous = &stage.density;
        }
        Ok(())
    }
}

impl AmegaRealgrad {
    /// Create a new Euclidean gradient tape with the provided step size.
    pub fn new(learning_rate: f32, rows: usize, cols: usize) -> PureResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        if learning_rate <= 0.0 || !learning_rate.is_finite() {
            return Err(TensorError::NonPositiveLearningRate {
                rate: learning_rate,
            });
        }
        Ok(Self {
            learning_rate,
            rows,
            cols,
            gradient: vec![0.0; rows * cols],
        })
    }

    /// Returns the learning rate applied during [`apply`].
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Adjust the learning rate used for subsequent updates.
    pub fn scale_learning_rate(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.learning_rate *= factor;
        }
    }

    /// Dimensions of the gradient buffer.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Read-only view into the accumulated gradient.
    pub fn gradient(&self) -> &[f32] {
        &self.gradient
    }

    /// Mutable access to the gradient buffer.
    pub fn gradient_mut(&mut self) -> &mut [f32] {
        &mut self.gradient
    }

    /// Summarise the accumulated gradient using basic norm statistics.
    pub fn summary(&self) -> GradientSummary {
        GradientSummary::from_slice(&self.gradient)
    }

    fn assert_tensor_shape(&self, tensor: &Tensor) -> PureResult<()> {
        if tensor.shape() != (self.rows, self.cols) {
            return Err(TensorError::ShapeMismatch {
                left: tensor.shape(),
                right: (self.rows, self.cols),
            });
        }
        Ok(())
    }

    /// Reset the accumulated gradient to zero.
    pub fn reset(&mut self) {
        for value in &mut self.gradient {
            *value = 0.0;
        }
    }

    /// Accumulate an Euclidean tensor into the tape.
    pub fn accumulate_wave(&mut self, tensor: &Tensor) -> PureResult<()> {
        self.assert_tensor_shape(tensor)?;
        for (grad, value) in self.gradient.iter_mut().zip(tensor.data().iter()) {
            *grad += *value;
        }
        Ok(())
    }

    /// Accumulate a complex wave by flattening it into real/imag lanes.
    pub fn accumulate_complex_wave(&mut self, wave: &ComplexTensor) -> PureResult<()> {
        let tensor = wave.to_tensor()?;
        self.accumulate_wave(&tensor)
    }

    /// Encode text into Z-space and accumulate the resulting tensor.
    pub fn absorb_text(&mut self, encoder: &LanguageWaveEncoder, text: &str) -> PureResult<()> {
        let tensor = encoder.encode_z_space(text)?;
        self.accumulate_wave(&tensor)
    }

    /// Integrate a prediction/target pair as a residual update.
    pub fn accumulate_pair(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<()> {
        self.assert_tensor_shape(prediction)?;
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        for ((grad, pred), tgt) in self
            .gradient
            .iter_mut()
            .zip(prediction.data().iter())
            .zip(target.data().iter())
        {
            *grad += pred - tgt;
        }
        Ok(())
    }

    /// Apply the accumulated gradient to the provided weights and clear it.
    pub fn apply(&mut self, weights: &mut Tensor) -> PureResult<()> {
        self.assert_tensor_shape(weights)?;
        for (value, grad) in weights.data_mut().iter_mut().zip(self.gradient.iter()) {
            *value -= self.learning_rate * *grad;
        }
        self.reset();
        Ok(())
    }
}

fn matmul_naive(lhs: &[f32], rhs: &[f32], rows: usize, inner: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for k in 0..inner {
            let a = lhs[r * inner + k];
            let row_offset = k * cols;
            for c in 0..cols {
                out[r * cols + c] += a * rhs[row_offset + c];
            }
        }
    }
    out
}

fn matmul_faer(
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> PureResult<Vec<f32>> {
    faer_dense::matmul(lhs, rhs, rows, inner, cols).map_err(|message| TensorError::BackendFailure {
        backend: "faer",
        message,
    })
}

#[cfg(feature = "wgpu")]
fn matmul_wgpu(
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> PureResult<Vec<f32>> {
    wgpu_dense::matmul(lhs, rhs, rows, inner, cols).map_err(|message| TensorError::BackendFailure {
        backend: "wgpu",
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn tensor_roundtrip_dlpack_preserves_contents() {
        let tensor = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let managed = tensor.to_dlpack().unwrap();
        let restored = unsafe { Tensor::from_dlpack(managed).unwrap() };
        assert_eq!(tensor, restored);
        assert_eq!(tensor.data().as_ptr(), restored.data().as_ptr());
    }

    #[test]
    fn tensor_to_dlpack_shares_underlying_buffer() {
        let tensor = Tensor::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ptr = tensor.data.as_ptr();
        let managed = tensor.to_dlpack().unwrap();
        unsafe {
            let dl_tensor = &(*managed).dl_tensor;
            assert_eq!(dl_tensor.data as *mut f32, ptr as *mut f32);
        }
        unsafe {
            drop_exported_state(managed);
        }
    }

    #[test]
    fn tensor_matmul_and_add_work_without_panicking() {
        let a = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let product = a.matmul(&b).unwrap();
        assert_eq!(product.shape(), (2, 2));
        let expected = Tensor::from_vec(2, 2, vec![58.0, 64.0, 139.0, 154.0]).unwrap();
        assert_eq!(product, expected);

        let sum = product
            .add(&Tensor::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap())
            .unwrap();
        let expected_sum = Tensor::from_vec(2, 2, vec![59.0, 65.0, 140.0, 155.0]).unwrap();
        assert_eq!(sum, expected_sum);
    }

    #[test]
    fn tensor_hadamard_matches_manual_product() {
        let a = Tensor::from_vec(2, 2, vec![1.5, -2.0, 0.5, 3.0]).unwrap();
        let b = Tensor::from_vec(2, 2, vec![2.0, 4.0, -1.0, 0.5]).unwrap();
        let product = a.hadamard(&b).unwrap();
        let expected = Tensor::from_vec(2, 2, vec![3.0, -8.0, -0.5, 1.5]).unwrap();
        assert_eq!(product, expected);
    }

    #[test]
    fn linear_regression_converges_without_tracebacks() {
        let inputs = Tensor::from_vec(4, 1, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let targets = Tensor::from_vec(4, 1, vec![1.0, 3.0, 5.0, 7.0]).unwrap();
        let mut model = LinearModel::new(1, 1).unwrap();
        let mut loss = 0.0;
        for _ in 0..200 {
            loss = model.train_batch(&inputs, &targets, 0.1).unwrap();
        }
        assert!(loss < 1e-3, "loss should converge, got {loss}");

        let predictions = model.forward(&inputs).unwrap();
        let mse = mean_squared_error(&predictions, &targets).unwrap();
        assert!(mse < 1e-3, "model should fit the line, got {mse}");

        let weight = model.weights().data()[0];
        let bias = model.bias()[0];
        assert!((weight - 2.0).abs() < 1e-2, "weight too far: {weight}");
        assert!((bias - 1.0).abs() < 1e-2, "bias too far: {bias}");
    }

    #[test]
    fn language_wave_encoder_maps_text_without_tokens() {
        let encoder = LanguageWaveEncoder::new(-1.0, 0.5).unwrap();
        let wave = encoder.encode_wave("spiral").unwrap();
        assert_eq!(wave.shape(), (1, 6));
        let z = encoder.encode_z_space("spiral").unwrap();
        assert_eq!(z.shape().0, 1);
        assert_eq!(z.shape().1, 12);
    }

    #[test]
    fn amega_hypergrad_tracks_z_space_updates() {
        let encoder = LanguageWaveEncoder::new(-1.25, 0.9).unwrap();
        let z = encoder
            .encode_z_space("non-euclidean waves stay token free")
            .unwrap();
        let shape = z.shape();
        let mut hypergrad =
            AmegaHypergrad::new(encoder.curvature(), 0.05, shape.0, shape.1).unwrap();
        hypergrad.accumulate_wave(&z).unwrap();
        let mut weights = Tensor::zeros(shape.0, shape.1).unwrap();
        let targets = Tensor::zeros(shape.0, shape.1).unwrap();
        hypergrad.accumulate_pair(&z, &targets).unwrap();
        hypergrad.apply(&mut weights).unwrap();
        assert_eq!(weights.shape(), shape);
        assert!(weights.squared_l2_norm() > 0.0);
    }

    #[test]
    fn amega_hypergrad_absorbs_text_directly() {
        let encoder = LanguageWaveEncoder::new(-0.8, 0.7).unwrap();
        let z = encoder
            .encode_z_space("SpiralTorch dances in Z-space")
            .unwrap();
        let shape = z.shape();
        let mut hypergrad =
            AmegaHypergrad::new(encoder.curvature(), 0.02, shape.0, shape.1).unwrap();
        hypergrad
            .absorb_text(&encoder, "SpiralTorch dances in Z-space")
            .unwrap();
        assert!(hypergrad
            .gradient()
            .iter()
            .any(|value| value.abs() > f32::EPSILON));
    }

    #[test]
    fn desire_gradient_control_respects_balance() {
        let hyper = GradientSummary::from_slice(&[0.8, -0.6, 0.4, -0.2]);
        let real = GradientSummary::from_slice(&[0.1, -0.05, 0.02, -0.01]);
        let interpretation = DesireGradientInterpretation::from_summaries(hyper, real);
        let control = interpretation.control();
        assert!(control.penalty_gain() >= interpretation.penalty_gain());
        assert!(control.hyper_rate_scale() >= 0.1);
        assert!(control.hyper_rate_scale() <= 2.0);
        assert!(control.real_rate_scale() >= 0.1);
        assert!(control.real_rate_scale() <= 2.0);
        assert!(control.operator_mix() <= 1.0);
        assert!(control.clip_norm() >= control.clip_floor());
        assert!(control.clip_ceiling() >= control.clip_norm());
        assert!(control.learning_rate_min() >= 0.0);
        assert!(control.learning_rate_max() >= control.learning_rate_min());
        assert!(control.temperature_kappa() >= 0.0);
        assert!(control.events().bits() != 0);
    }

    #[test]
    fn desire_gradient_control_scales_with_gain() {
        let hyper = GradientSummary::from_slice(&[0.2, -0.1, 0.05, -0.02]);
        let real = GradientSummary::from_slice(&[0.2, -0.1, 0.05, -0.02]);
        let interpretation = DesireGradientInterpretation::from_summaries(hyper, real);
        let neutral = interpretation.control_with_gain(0.0);
        let tuned = interpretation.control_with_gain(1.0);
        assert!(neutral.learning_rate_eta() <= tuned.learning_rate_eta());
        assert!(neutral.temperature_kappa() <= tuned.temperature_kappa());
        assert!(neutral.quality_gain() <= tuned.quality_gain());
        assert!(tuned.events().bits() >= neutral.events().bits());
    }

    #[test]
    fn desire_control_events_report_labels() {
        let events = DesireControlEvents::LR_INCREASE
            .insert(DesireControlEvents::LR_CLIPPED)
            .insert(DesireControlEvents::QUALITY_SUPPRESS)
            .insert(DesireControlEvents::SLEW_LIMIT);
        let labels = events.labels();
        assert!(labels.contains(&"lr_increase"));
        assert!(labels.contains(&"lr_clipped"));
        assert!(labels.contains(&"quality_suppress"));
        assert!(labels.contains(&"lr_slew_limit"));
    }

    #[test]
    fn desire_control_builder_tracks_entropy_and_bounds() {
        let control = DesireGradientControl::control_with_gain()
            .with_entropy(0.5)
            .with_target_entropy(3.0)
            .with_bounds(1e-4, 5e-3)
            .with_slew_limit(0.05)
            .finalise();
        assert!(control.learning_rate_min() >= 1e-4);
        assert!((control.learning_rate_max() - 5e-3).abs() < 1e-8);
        assert!(control.events().contains(DesireControlEvents::LR_INCREASE));
        assert!(control.events().contains(DesireControlEvents::SLEW_LIMIT));
    }

    #[test]
    fn desire_control_builder_marks_z_suppression() {
        let control = DesireGradientControl::control_with_gain()
            .with_z_strength(0.6)
            .with_z_coupling(2.0)
            .finalise();
        assert!(control.temperature_kappa() > 0.0);
        assert!(control.hyper_rate_scale() < 1.0);
        assert!(control.events().contains(DesireControlEvents::Z_SUPPRESS));
        assert!(control
            .events()
            .contains(DesireControlEvents::TEMPERATURE_SUPPRESS));
    }

    #[test]
    fn desire_control_builder_quality_events() {
        let boosted = DesireGradientControl::control_with_gain()
            .with_quality_parameters(2.0, 0.1)
            .with_quality(0.9)
            .finalise();
        assert!(boosted
            .events()
            .contains(DesireControlEvents::QUALITY_BOOST));

        let suppressed = DesireGradientControl::control_with_gain()
            .with_quality_parameters(2.0, 0.6)
            .with_quality(0.05)
            .finalise();
        assert!(suppressed
            .events()
            .contains(DesireControlEvents::QUALITY_SUPPRESS));
        assert!(suppressed.real_rate_scale() <= boosted.real_rate_scale());
    }

    #[test]
    fn amega_hypergrad_tracks_barycenter_curve() {
        use crate::pure::measure::z_space_barycenter;

        let densities = vec![
            Tensor::from_vec(1, 2, vec![0.8, 0.2]).unwrap(),
            Tensor::from_vec(1, 2, vec![0.3, 0.7]).unwrap(),
        ];
        let weights = vec![1.0, 2.0];
        let result = z_space_barycenter(&weights, &densities, 0.1, 0.0, None).unwrap();
        let mut tape = AmegaHypergrad::new(-1.0, 0.05, 1, 2).unwrap();
        tape.accumulate_barycenter_path(&result.intermediates)
            .unwrap();
        let gradient = tape.gradient();
        assert!(gradient.iter().any(|value| value.abs() > 0.0));
    }

    #[test]
    fn amega_realgrad_accumulates_and_applies() {
        let mut tape = AmegaRealgrad::new(0.1, 1, 3).unwrap();
        let tensor = Tensor::from_vec(1, 3, vec![1.0, -2.0, 0.5]).unwrap();
        tape.accumulate_wave(&tensor).unwrap();
        let mut weights = Tensor::from_vec(1, 3, vec![0.0, 0.0, 0.0]).unwrap();
        tape.apply(&mut weights).unwrap();
        assert!((weights.data()[0] + 0.1).abs() < 1e-6);
        assert!((weights.data()[1] - 0.2).abs() < 1e-6);
        assert!((weights.data()[2] + 0.05).abs() < 1e-6);
    }

    #[test]
    fn amega_realgrad_absorbs_text() {
        let encoder = LanguageWaveEncoder::new(-1.0, 0.6).unwrap();
        let z = encoder.encode_z_space("spiral torch realgrad").unwrap();
        let mut tape = AmegaRealgrad::new(0.05, z.shape().0, z.shape().1).unwrap();
        tape.absorb_text(&encoder, "spiral torch realgrad").unwrap();
        assert!(tape
            .gradient()
            .iter()
            .any(|value| value.abs() > f32::EPSILON));
    }

    #[test]
    fn gradient_summary_reports_norms() {
        let summary = GradientSummary::from_slice(&[1.0, -2.0, 0.0, 3.0]);
        assert_eq!(summary.count(), 4);
        assert!((summary.l1() - 6.0).abs() < 1e-6);
        let expected_l2 = (1.0f32 + 4.0 + 0.0 + 9.0).sqrt();
        assert!((summary.l2() - expected_l2).abs() < 1e-6);
        assert!((summary.mean_abs() - 1.5).abs() < 1e-6);
        let expected_rms = expected_l2 / (4.0f32).sqrt();
        assert!((summary.rms() - expected_rms).abs() < 1e-6);
        assert_eq!(summary.linf(), 3.0);
    }

    #[test]
    fn gradient_summary_from_moments_matches_slice() {
        let from_slice = GradientSummary::from_slice(&[1.0, -2.0, 0.0, 3.0]);
        let from_moments = GradientSummary::from_moments(6.0, 14.0, 3.0, 4);
        assert!((from_slice.l1() - from_moments.l1()).abs() < 1e-6);
        assert!((from_slice.l2() - from_moments.l2()).abs() < 1e-6);
        assert_eq!(from_slice.count(), from_moments.count());
        assert_eq!(from_slice.linf(), from_moments.linf());
        assert!((from_slice.sum_squares() - from_moments.sum_squares()).abs() < 1e-6);
    }

    #[test]
    fn desire_gradient_interpretation_balances_metrics() {
        let hyper = GradientSummary::from_slice(&[1.0, -0.5, 0.25, 0.75]);
        let real = GradientSummary::from_slice(&[0.05, -0.1, 0.0, 0.1]);
        let interpretation = DesireGradientInterpretation::from_summaries(hyper, real);
        assert!(interpretation.balance() > 1.0);
        assert!(interpretation.penalty_gain() > 1.0);
        let stable = DesireGradientInterpretation::from_summaries(hyper, hyper);
        assert!(stable.stability() > interpretation.stability());
        assert!(stable.bias_mix() > interpretation.bias_mix());
        assert!(stable.observation_gain() >= interpretation.observation_gain());
    }

    #[test]
    fn gradient_tapes_surface_summary_metrics() {
        let tensor = Tensor::from_vec(1, 3, vec![1.0, -2.0, 4.0]).unwrap();
        let mut hypergrad = AmegaHypergrad::new(-1.0, 0.1, 1, 3).unwrap();
        hypergrad.accumulate_wave(&tensor).unwrap();
        let hyper_summary = hypergrad.summary();
        assert_eq!(hyper_summary.count(), 3);
        assert!(hyper_summary.l2() > 0.0);

        let mut realgrad = AmegaRealgrad::new(0.1, 1, 3).unwrap();
        realgrad.accumulate_wave(&tensor).unwrap();
        let real_summary = realgrad.summary();
        assert_eq!(real_summary.count(), 3);
        let expected_l1: f32 = tensor.data().iter().map(|value| value.abs()).sum();
        assert!((real_summary.l1() - expected_l1).abs() < 1e-6);
    }

    #[test]
    fn hypergrad_retune_updates_curvature_and_resets() {
        let mut tape = AmegaHypergrad::new(-1.0, 0.05, 1, 2).unwrap();
        let tensor = Tensor::from_vec(1, 2, vec![0.25, -0.4]).unwrap();
        tape.accumulate_wave(&tensor).unwrap();
        assert!(tape.gradient().iter().any(|value| value.abs() > 0.0));
        tape.retune(-0.5, 0.1).unwrap();
        assert!((tape.curvature() + 0.5).abs() < 1e-6);
        assert!((tape.learning_rate() - 0.1).abs() < 1e-6);
        assert!(tape.gradient().iter().all(|value| value.abs() < 1e-6));
        let guard = tape.topos();
        assert!((guard.curvature() + 0.5).abs() < 1e-6);
    }

    #[test]
    fn amega_realgrad_accumulates_pair() {
        let mut tape = AmegaRealgrad::new(0.01, 1, 2).unwrap();
        let prediction = Tensor::from_vec(1, 2, vec![0.5, -0.5]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![0.25, -0.75]).unwrap();
        tape.accumulate_pair(&prediction, &target).unwrap();
        assert!((tape.gradient()[0] - 0.25).abs() < 1e-6);
        assert!((tape.gradient()[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn random_uniform_respects_bounds_and_is_convertible_to_ndarray() {
        let tensor = Tensor::random_uniform(4, 3, -0.25, 0.75, Some(7)).unwrap();
        assert_eq!(tensor.shape(), (4, 3));
        assert!(tensor
            .data()
            .iter()
            .all(|value| (-0.25..0.75).contains(value)));
        let array = Array2::from_shape_vec((4, 3), tensor.data().to_vec()).unwrap();
        assert_eq!(array.dim(), (4, 3));
        assert_eq!(array[[0, 0]], tensor.data()[0]);
    }

    #[test]
    fn random_initialisers_are_deterministic_with_seed() {
        let left = Tensor::random_normal(2, 2, 0.0, 1.0, Some(42)).unwrap();
        let right = Tensor::random_normal(2, 2, 0.0, 1.0, Some(42)).unwrap();
        assert_eq!(left.data(), right.data());
    }

    #[test]
    fn random_uniform_rejects_invalid_bounds() {
        let err = Tensor::random_uniform(2, 2, 1.0, 1.0, None).unwrap_err();
        assert!(matches!(err, TensorError::InvalidValue { .. }));
    }
}
