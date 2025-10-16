// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Pure Rust tensor, complex wave, and non-Euclidean learning primitives with
//! zero external dependencies.
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

pub use self::differential::{
    DifferentialResonance, FunctorDifferential, HomotopyDifferential, InfinityDifferential,
    RecursiveDifferential, SpiralDifferential,
};
use self::measure::BarycenterIntermediate;
pub use self::topos::{OpenCartesianTopos, RewriteMonad, TensorBiome};

use crate::backend::faer_dense;
#[cfg(feature = "wgpu")]
use crate::backend::wgpu_dense;
use core::fmt;
use std::error::Error;
use std::f32::consts::PI;

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
    /// Weighted Z-space barycenter collapsed because the entropy weight cancelled the KL pull.
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
    /// External backend failed while executing an accelerated path.
    BackendFailure {
        backend: &'static str,
        message: String,
    },
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
                    "z-space barycenter degenerates when the effective weight {effective_weight} vanishes"
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
                write!(f, "backend {backend} failed: {message}")
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

/// A simple row-major 2D tensor backed by a `Vec<f32>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Tensor {
    /// Create a tensor filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> PureResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        Ok(Self {
            data: vec![0.0; rows * cols],
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
        Ok(Self { data, rows, cols })
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
        Ok(Self { data, rows, cols })
    }

    /// Returns the `(rows, cols)` pair of the tensor.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns a read-only view of the underlying buffer.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns a mutable view of the underlying buffer.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
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
        for a in &self.data {
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
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
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
        for r in 0..self.rows {
            let offset = r * self.cols;
            for c in 0..self.cols {
                self.data[offset + c] += bias[c];
            }
        }
        Ok(())
    }

    /// Returns the transpose of the tensor.
    pub fn transpose(&self) -> Tensor {
        let mut out = Tensor {
            data: vec![0.0; self.rows * self.cols],
            rows: self.cols,
            cols: self.rows,
        };
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        out
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
        Tensor::from_vec(rows, cols, self.data.clone())
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
            data.extend_from_slice(&tensor.data);
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
        Ok(Self { data, rows, cols })
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
}
