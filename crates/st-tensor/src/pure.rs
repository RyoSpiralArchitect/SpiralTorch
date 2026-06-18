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
pub mod python;
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
    GraphGuardProfile, GraphGuardReport, LawvereTierneyGuard, ModalityProfile, MultiModalAtlas,
    MultiModalBiome, MultiModalToposGuard, OpenCartesianTopos, RewardBoundary,
    RewardBoundarySignal, RewriteMonad, TensorBiome, ToposAtlas, ZBox, ZBoxSite,
};

#[cfg(feature = "hip")]
use crate::backend::hip_dense;
#[cfg(feature = "wgpu")]
use crate::backend::wgpu_dense;
use crate::backend::{cpu_dense, faer_dense};
use crate::dlpack::{
    call_managed_deleter, drop_exported_state, DLDataType, DLDataTypeCode, DLDevice, DLDeviceType,
    DLManagedTensor, DLTensor, ExportData, ForeignTensor, ManagedTensorState,
};
use crate::hardmax::{HardmaxBackend, HardmaxFusionPlan, HardmaxFusionResult, HardmaxMode};
use crate::memory::{
    aligned_from_slice, aligned_from_vec, aligned_with_capacity, aligned_zeroed, is_ptr_aligned,
    AlignedVec,
};
use core::fmt;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
#[allow(unused_imports)]
use rand_distr::StandardNormal;
use rayon::{current_num_threads, prelude::*};
use serde::{Deserialize, Serialize};
use spiral_config::determinism;
use std::cell::{Cell, RefCell};
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
    TensorVolumeExceeded {
        label: &'static str,
        volume: usize,
        max_volume: usize,
    },
    /// Loop detection tripped for an open-cartesian topos traversal.
    LoopDetected { depth: usize, max_depth: usize },
    /// Conjugate gradient solver could not reach the requested tolerance.
    ConjugateGradientDiverged { residual: f32, tolerance: f32 },
    /// Execution failed on an accelerator backend.
    BackendFailure {
        backend: &'static str,
        message: String,
    },
    /// Porosity parameters must stay within [0, 1].
    PorosityOutOfRange { porosity: f32 },
    /// Generic configuration violation for pure-language helpers.
    InvalidValue { label: &'static str },
    /// The tensor orientation is not supported by the requested computation.
    UnsupportedLayout { label: &'static str },
    /// Interoperability bridge encountered an unsupported or malformed DLPack tensor.
    DlpackError { message: String },
    /// Generic error with a custom message.
    Generic(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::InvalidDimensions { rows, cols } => {
                write!(f, "invalid tensor dimensions ({rows} x {cols})")
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
            TensorError::TensorVolumeExceeded {
                label,
                volume,
                max_volume,
            } => {
                write!(
                    f,
                    "tensor '{label}' volume {volume} exceeds open-cartesian capacity {max_volume}"
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
            TensorError::PorosityOutOfRange { porosity } => {
                write!(f, "porosity must lie in [0, 1]; received {porosity}")
            }
            TensorError::InvalidValue { label } => {
                write!(f, "invalid value: {label}")
            }
            TensorError::UnsupportedLayout { label } => {
                write!(
                    f,
                    "requested operation requires a different tensor layout ({label})"
                )
            }
            TensorError::DlpackError { message } => {
                write!(f, "dlpack error: {message}")
            }
            TensorError::Generic(message) => {
                write!(f, "{message}")
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
    /// Use the built-in tiled SIMD micro-kernel.
    CpuSimd,
    /// Always fallback to the scalar implementation.
    CpuNaive,
    /// Force the compute-path GEMM running through WGPU.
    #[cfg(feature = "wgpu")]
    GpuWgpu,
    /// Execute GEMM via the HIP backend when available.
    #[cfg(feature = "hip")]
    GpuHip,
}

impl MatmulBackend {
    fn label(self) -> &'static str {
        match self {
            MatmulBackend::Auto => "auto",
            MatmulBackend::CpuFaer => "faer",
            MatmulBackend::CpuSimd => "simd",
            MatmulBackend::CpuNaive => "naive",
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => "wgpu",
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => "hip",
        }
    }
}

impl fmt::Display for MatmulBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str((*self).label())
    }
}

#[cfg(any(feature = "wgpu", feature = "hip", test))]
fn strict_gpu_path() -> bool {
    std::env::var("SPIRALTORCH_STRICT_GPU")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

#[cfg(any(feature = "wgpu", feature = "hip"))]
fn strict_gpu_fallback_error(
    backend: &'static str,
    op: &'static str,
    message: String,
) -> TensorError {
    TensorError::BackendFailure {
        backend,
        message: format!("{op} {backend} path failed ({message}); fallback disabled"),
    }
}

#[cfg(feature = "wgpu")]
fn wgpu_runtime_unavailable(message: &str) -> bool {
    message.contains("no suitable WGPU adapter")
        || message.contains("failed to initialize WGPU")
        || message.contains("WGPU backend not available")
}

const WGPU_RUNTIME_FALLBACK_REASON: &str = "runtime_unavailable";

fn wgpu_runtime_fallback_meta(
    to_backend: &'static str,
    reason: &'static str,
    message: Option<&str>,
) -> serde_json::Value {
    let mut data = serde_json::Map::new();
    data.insert("from".to_string(), serde_json::json!("wgpu"));
    data.insert("to".to_string(), serde_json::json!(to_backend));
    data.insert("reason".to_string(), serde_json::json!(reason));
    if let Some(message) = message {
        data.insert("message".to_string(), serde_json::json!(message));
    }
    serde_json::Value::Object(data)
}

#[cfg(feature = "wgpu")]
fn wgpu_backend_runtime_unavailable(error: &TensorError) -> bool {
    matches!(
        error,
        TensorError::BackendFailure {
            backend: "wgpu",
            message,
        } if wgpu_runtime_unavailable(message)
    )
}

/// Explicit backend selection for row-wise softmax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftmaxBackend {
    /// Allow SpiralTorch to pick the most appropriate backend.
    Auto,
    /// Force the pure Rust implementation.
    Cpu,
    /// Execute on the WGPU accelerator backend when available.
    #[cfg(feature = "wgpu")]
    GpuWgpu,
}

impl SoftmaxBackend {
    fn label(self) -> &'static str {
        match self {
            SoftmaxBackend::Auto => "auto",
            SoftmaxBackend::Cpu => "cpu",
            #[cfg(feature = "wgpu")]
            SoftmaxBackend::GpuWgpu => "wgpu",
        }
    }
}

impl fmt::Display for SoftmaxBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

const SPIRAL_PROJECTOR_RANK: usize = 24;
const SPIRAL_PROJECTOR_WEIGHT: f64 = 0.75;
const SPIRAL_PROJECTOR_RAMANUJAN_ITERS: usize = 6;
const SPIRAL_LEECH_PACKING_DENSITY: f64 = 0.001_929_574_309_403_922_5;
const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;
const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_8;
const GOLDEN_RATIO_BIAS: f64 = 0.381_966_011_250_105_1;

/// Aggregate telemetry captured while building the spiral consensus weights.
#[derive(Clone, Copy, Debug)]
pub struct SpiralConsensusStats {
    /// The golden ratio φ used for the fusion.
    pub phi: f64,
    /// The conjugate of φ (1/φ) scaling the softmax contribution.
    pub phi_conjugate: f64,
    /// Complementary bias (1 - 1/φ) applied to the hardmax mask.
    pub phi_bias: f64,
    /// Ratio between π and the Ramanujan approximation employed.
    pub ramanujan_ratio: f64,
    /// Absolute deviation between the approximation and true π.
    pub ramanujan_delta: f64,
    /// Average enrichment contributed by the Leech lattice weighting.
    pub average_enrichment: f64,
    /// Mean Shannon entropy observed in the softmax rows.
    pub mean_entropy: f64,
    /// Mean cardinality of the hardmax mask per row.
    pub mean_hardmass: f64,
    /// Average per-row coherence balancing entropy, hardmass, and enrichment.
    pub spiral_coherence: f64,
}

impl Default for SpiralConsensusStats {
    fn default() -> Self {
        let approximation = ramanujan_pi(SPIRAL_PROJECTOR_RAMANUJAN_ITERS);
        let pi = std::f64::consts::PI;
        let (ramanujan_ratio, ramanujan_delta) =
            if approximation.is_finite() && approximation > f64::EPSILON {
                (pi / approximation, (approximation - pi).abs())
            } else {
                (1.0, pi.abs())
            };

        Self {
            phi: GOLDEN_RATIO,
            phi_conjugate: GOLDEN_RATIO_CONJUGATE,
            phi_bias: GOLDEN_RATIO_BIAS,
            ramanujan_ratio,
            ramanujan_delta,
            average_enrichment: 0.0,
            mean_entropy: 0.0,
            mean_hardmass: 0.0,
            spiral_coherence: 0.0,
        }
    }
}

/// Combined softmax, hardmax, and spiral consensus payload returned by
/// [`Tensor::row_softmax_hardmax_spiral`].
#[derive(Clone, Debug)]
pub struct SpiralSoftmaxHardmax {
    /// Softmax probabilities.
    pub softmax: Tensor,
    /// Hardmax mask (argmax indicators).
    pub hardmax: Tensor,
    /// Spiral consensus weights blending soft and hard responses.
    pub spiral: Tensor,
    /// Aggregated telemetry describing the consensus fusion.
    pub metrics: SpiralConsensusStats,
}

impl SpiralSoftmaxHardmax {
    /// Decomposes the payload into its constituent parts.
    pub fn into_parts(self) -> (Tensor, Tensor, Tensor, SpiralConsensusStats) {
        (self.softmax, self.hardmax, self.spiral, self.metrics)
    }
}

fn ramanujan_pi(iterations: usize) -> f64 {
    let iterations = iterations.max(1);
    let base = 396_f64.powi(4);
    let prefactor = (2.0 * 2.0_f64.sqrt()) / 9801.0;
    let mut sum = 0.0_f64;
    let mut factor = 1.0_f64;
    for k in 0..iterations {
        let kf = k as f64;
        sum += factor * (1103.0 + 26390.0 * kf);
        if k + 1 < iterations {
            let next = kf + 1.0;
            let numerator =
                (4.0 * next - 3.0) * (4.0 * next - 2.0) * (4.0 * next - 1.0) * (4.0 * next);
            let denominator = next.powi(4) * base;
            factor *= numerator / denominator;
        }
    }
    (prefactor * sum).recip()
}

pub(crate) fn spiral_softmax_hardmax_consensus(
    softmax: &[f32],
    hardmax: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, SpiralConsensusStats) {
    let expected = rows.saturating_mul(cols);
    if expected == 0 || softmax.len() != expected || hardmax.len() != expected {
        return (vec![0.0; expected], SpiralConsensusStats::default());
    }

    let approximation = ramanujan_pi(SPIRAL_PROJECTOR_RAMANUJAN_ITERS);
    let pi = std::f64::consts::PI;
    let ramanujan_ratio = if approximation > f64::EPSILON {
        pi / approximation
    } else {
        1.0
    };
    let ramanujan_delta = (approximation - pi).abs();
    let sqrt_rank = (SPIRAL_PROJECTOR_RANK as f64).sqrt();
    let leech_scale =
        SPIRAL_PROJECTOR_WEIGHT * SPIRAL_LEECH_PACKING_DENSITY * sqrt_rank * ramanujan_ratio;

    let mut fused = vec![0.0; expected];
    let mut total_entropy = 0.0_f64;
    let mut total_hardmass = 0.0_f64;
    let mut total_enrichment = 0.0_f64;
    let mut total_coherence = 0.0_f64;

    for row in 0..rows {
        let offset = row * cols;
        let row_soft = &softmax[offset..offset + cols];
        let row_hard = &hardmax[offset..offset + cols];

        let mut entropy = 0.0_f64;
        let mut hardmass = 0.0_f64;

        for (&prob, &mask) in row_soft.iter().zip(row_hard.iter()) {
            let p = f64::from(if prob.is_finite() { prob.max(0.0) } else { 0.0 });
            let m = f64::from(if mask.is_finite() { mask.max(0.0) } else { 0.0 });

            if p > 0.0 {
                entropy -= p * p.ln();
            }
            hardmass += m;
        }

        let geodesic = entropy * ramanujan_ratio + hardmass * GOLDEN_RATIO;
        let geodesic = if geodesic.is_finite() { geodesic } else { 0.0 };
        let enrichment = if geodesic > f64::EPSILON {
            leech_scale * geodesic
        } else {
            0.0
        };
        let enrichment = if enrichment.is_finite() {
            enrichment.max(0.0)
        } else {
            0.0
        };
        let raw_scale = 1.0 + enrichment;
        let scale = raw_scale.clamp(0.0, f32::MAX as f64) as f32;
        total_entropy += entropy;
        total_hardmass += hardmass;
        total_enrichment += enrichment;

        let entropy_norm = (entropy / (entropy + 1.0)).clamp(0.0, 1.0);
        let hardmass_norm = (hardmass / cols as f64).clamp(0.0, 1.0);
        let enrichment_norm = (enrichment / (1.0 + enrichment.abs())).clamp(0.0, 1.0);
        total_coherence += (entropy_norm + hardmass_norm + enrichment_norm) / 3.0;

        for (index, (&prob, &mask)) in row_soft.iter().zip(row_hard.iter()).enumerate() {
            let sanitized_prob = if prob.is_finite() { prob.max(0.0) } else { 0.0 };
            let sanitized_mask = if mask.is_finite() { mask.max(0.0) } else { 0.0 };

            let fused_value = (GOLDEN_RATIO_CONJUGATE as f32) * sanitized_prob
                + (GOLDEN_RATIO_BIAS as f32) * sanitized_mask;
            let candidate = scale * fused_value;
            fused[offset + index] = if candidate.is_finite() {
                candidate.max(0.0)
            } else {
                0.0
            };
        }
    }

    if rows == 0 {
        return (fused, SpiralConsensusStats::default());
    }

    let rows_f64 = rows as f64;
    let stats = SpiralConsensusStats {
        phi: GOLDEN_RATIO,
        phi_conjugate: GOLDEN_RATIO_CONJUGATE,
        phi_bias: GOLDEN_RATIO_BIAS,
        ramanujan_ratio,
        ramanujan_delta,
        average_enrichment: total_enrichment / rows_f64,
        mean_entropy: total_entropy / rows_f64,
        mean_hardmass: total_hardmass / rows_f64,
        spiral_coherence: total_coherence / rows_f64,
    };

    (fused, stats)
}

/// Explicit backend selection for fused attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionBackend {
    /// Allow SpiralTorch to pick the most appropriate backend.
    Auto,
    /// Force the pure Rust implementation.
    Cpu,
    /// Execute on the WGPU fused kernel when available.
    GpuWgpu,
}

impl AttentionBackend {
    fn label(self) -> &'static str {
        match self {
            AttentionBackend::Auto => "auto",
            AttentionBackend::Cpu => "cpu",
            AttentionBackend::GpuWgpu => "wgpu",
        }
    }
}

impl fmt::Display for AttentionBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Explicit backend selection for layer normalisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerNormBackend {
    /// Allow SpiralTorch to pick the most appropriate backend.
    Auto,
    /// Force the pure Rust implementation.
    Cpu,
    /// Execute on the WGPU fused kernel when available.
    GpuWgpu,
}

impl LayerNormBackend {
    fn label(self) -> &'static str {
        match self {
            LayerNormBackend::Auto => "auto",
            LayerNormBackend::Cpu => "cpu",
            LayerNormBackend::GpuWgpu => "wgpu",
        }
    }
}

impl fmt::Display for LayerNormBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Explicit backend selection for small tensor utility kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorUtilBackend {
    /// Keep the legacy tensor utility path.
    Auto,
    /// Force the pure Rust implementation.
    Cpu,
    /// Execute on WGPU utility kernels when available.
    GpuWgpu,
}

impl TensorUtilBackend {
    fn label(self) -> &'static str {
        match self {
            TensorUtilBackend::Auto => "auto",
            TensorUtilBackend::Cpu => "cpu",
            TensorUtilBackend::GpuWgpu => "wgpu",
        }
    }
}

impl fmt::Display for TensorUtilBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Clone, Debug)]
enum TensorBacking {
    Owned(Arc<AlignedVec>),
    Foreign(ForeignTensor),
}

#[derive(Clone, Debug)]
pub(crate) struct TensorBuffer {
    backing: TensorBacking,
}

impl TensorBuffer {
    fn from_aligned(data: AlignedVec) -> Self {
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
            let owned = aligned_from_slice(foreign.as_slice());
            self.backing = TensorBacking::Owned(Arc::new(owned));
        }

        if let TensorBacking::Owned(vec) = &mut self.backing {
            Arc::make_mut(vec).as_mut_slice()
        } else {
            unreachable!()
        }
    }

    fn as_ptr(&self) -> *const f32 {
        match &self.backing {
            TensorBacking::Owned(vec) => vec.as_ptr(),
            TensorBacking::Foreign(foreign) => foreign.as_ptr(),
        }
    }

    fn export_handle(&self) -> ExportData {
        match &self.backing {
            TensorBacking::Owned(vec) => ExportData::Owned(Arc::clone(vec)),
            TensorBacking::Foreign(foreign) => ExportData::Foreign(foreign.clone()),
        }
    }

    fn try_clone_owned(&self) -> Option<Arc<AlignedVec>> {
        match &self.backing {
            TensorBacking::Owned(vec) => Some(Arc::clone(vec)),
            TensorBacking::Foreign(_) => None,
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

/// Orientation tag for tensors and packed buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    ColMajor,
    Tiled { tm: u32, tn: u32, tk: u32 },
    Chimera { stripes: u32, tile: u32 },
}

impl Layout {
    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            Layout::RowMajor => "row_major",
            Layout::ColMajor => "col_major",
            Layout::Tiled { .. } => "tiled",
            Layout::Chimera { .. } => "chimera",
        }
    }

    #[inline]
    fn expect_row_major(self, label: &'static str) -> PureResult<()> {
        if matches!(self, Layout::RowMajor) {
            Ok(())
        } else {
            Err(TensorError::UnsupportedLayout { label })
        }
    }

    #[inline]
    fn to_dense(self, rows: usize, cols: usize) -> Result<faer_dense::DenseLayout, TensorError> {
        let _ = (rows, cols);
        match self {
            Layout::RowMajor => Ok(faer_dense::DenseLayout::RowMajor),
            Layout::ColMajor => Ok(faer_dense::DenseLayout::ColMajor),
            Layout::Tiled { .. } => Err(TensorError::UnsupportedLayout {
                label: "tiled layout is not yet supported by the dense kernels",
            }),
            Layout::Chimera { .. } => Err(TensorError::UnsupportedLayout {
                label: "chimera layout cannot be treated as dense",
            }),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_basic_tensor_op_meta<F>(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    output_rows: usize,
    output_cols: usize,
    layout: Layout,
    backend: &'static str,
    requested_backend: &'static str,
    kernel: &'static str,
    kind: &'static str,
    extra: F,
) where
    F: FnOnce(&mut serde_json::Map<String, serde_json::Value>),
{
    crate::emit_tensor_op_meta(op_name, || {
        let mut data = serde_json::Map::new();
        data.insert("backend".to_string(), serde_json::json!(backend));
        data.insert(
            "requested_backend".to_string(),
            serde_json::json!(requested_backend),
        );
        data.insert("kernel".to_string(), serde_json::json!(kernel));
        data.insert("kind".to_string(), serde_json::json!(kind));
        data.insert("rows".to_string(), serde_json::json!(rows));
        data.insert("cols".to_string(), serde_json::json!(cols));
        data.insert(
            "values".to_string(),
            serde_json::json!(rows.saturating_mul(cols)),
        );
        data.insert("output_rows".to_string(), serde_json::json!(output_rows));
        data.insert("output_cols".to_string(), serde_json::json!(output_cols));
        data.insert(
            "output_values".to_string(),
            serde_json::json!(output_rows.saturating_mul(output_cols)),
        );
        data.insert("layout".to_string(), serde_json::json!(layout.as_str()));
        data.insert(
            "empty".to_string(),
            serde_json::json!(rows == 0 || cols == 0 || output_rows == 0 || output_cols == 0),
        );
        extra(&mut data);
        serde_json::Value::Object(data)
    });
}

#[allow(clippy::too_many_arguments)]
fn emit_tensor_util_cpu_op_meta<F>(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    output_rows: usize,
    output_cols: usize,
    layout: Layout,
    requested_backend: &'static str,
    kind: &'static str,
    extra: F,
) where
    F: FnOnce(&mut serde_json::Map<String, serde_json::Value>),
{
    emit_basic_tensor_op_meta(
        op_name,
        rows,
        cols,
        output_rows,
        output_cols,
        layout,
        "cpu",
        requested_backend,
        "scalar",
        kind,
        |data| {
            extra(data);
            #[cfg(feature = "wgpu")]
            if requested_backend == "wgpu" && !data.contains_key("fallback") {
                data.insert(
                    "fallback".to_string(),
                    wgpu_runtime_fallback_meta("cpu", WGPU_RUNTIME_FALLBACK_REASON, None),
                );
            }
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_cpu_tensor_op_meta<F>(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    output_rows: usize,
    output_cols: usize,
    layout: Layout,
    kind: &'static str,
    extra: F,
) where
    F: FnOnce(&mut serde_json::Map<String, serde_json::Value>),
{
    emit_tensor_util_cpu_op_meta(
        op_name,
        rows,
        cols,
        output_rows,
        output_cols,
        layout,
        "auto",
        kind,
        extra,
    );
}

#[cfg(feature = "wgpu")]
fn row_l2_projection_stats(data: &[f32], rows: usize, cols: usize) -> (usize, f32) {
    let mut nonzero_rows = 0usize;
    let mut max_row_l2 = 0.0f32;
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let norm: f32 = data[start..end].iter().map(|v| v * v).sum::<f32>().sqrt();
        max_row_l2 = max_row_l2.max(norm);
        if norm > 0.0 {
            nonzero_rows = nonzero_rows.saturating_add(1);
        }
    }
    (nonzero_rows, max_row_l2)
}

fn porous_mix_value(value: f32, saturation: f32, porosity: f32) -> f32 {
    if !value.is_finite() || saturation <= 0.0 {
        return 0.0;
    }
    let limit = saturation.abs();
    let magnitude = value.abs();
    if magnitude <= limit {
        return value;
    }
    if porosity <= f32::EPSILON {
        return value.signum() * limit;
    }
    let bleed = (magnitude - limit) / (magnitude + limit);
    let absorb = (porosity * 0.25).min(1.0);
    let softened = limit * (1.0 - absorb * bleed.min(1.0)).max(0.0);
    value.signum() * softened
}

#[cfg(feature = "wgpu")]
#[allow(clippy::too_many_arguments)]
fn emit_wgpu_tensor_op_meta<F>(
    op_name: &'static str,
    requested_backend: &'static str,
    rows: usize,
    cols: usize,
    output_rows: usize,
    output_cols: usize,
    layout: Layout,
    kind: &'static str,
    kernel: &'static str,
    extra: F,
) where
    F: FnOnce(&mut serde_json::Map<String, serde_json::Value>),
{
    crate::emit_tensor_op_meta(op_name, || {
        let mut data = serde_json::Map::new();
        data.insert("backend".to_string(), serde_json::json!("wgpu_dense"));
        data.insert(
            "requested_backend".to_string(),
            serde_json::json!(requested_backend),
        );
        data.insert("kernel".to_string(), serde_json::json!(kernel));
        data.insert("kind".to_string(), serde_json::json!(kind));
        data.insert("rows".to_string(), serde_json::json!(rows));
        data.insert("cols".to_string(), serde_json::json!(cols));
        data.insert(
            "values".to_string(),
            serde_json::json!(rows.saturating_mul(cols)),
        );
        data.insert("output_rows".to_string(), serde_json::json!(output_rows));
        data.insert("output_cols".to_string(), serde_json::json!(output_cols));
        data.insert(
            "output_values".to_string(),
            serde_json::json!(output_rows.saturating_mul(output_cols)),
        );
        data.insert("layout".to_string(), serde_json::json!(layout.as_str()));
        data.insert(
            "empty".to_string(),
            serde_json::json!(rows == 0 || cols == 0 || output_rows == 0 || output_cols == 0),
        );
        extra(&mut data);
        serde_json::Value::Object(data)
    });
}

/// Tile configuration used when preparing packed matrices for matmul.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub tm: u32,
    pub tn: u32,
    pub tk: u32,
}

impl Tile {
    pub const fn new(tm: u32, tn: u32, tk: u32) -> Self {
        Self { tm, tn, tk }
    }

    pub const fn col_major() -> Self {
        Self::new(1, 1, 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedLayout {
    ColMajor,
    Tiled { tm: u32, tn: u32, tk: u32 },
}

impl PackedLayout {
    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            PackedLayout::ColMajor => "col_major",
            PackedLayout::Tiled { .. } => "tiled",
        }
    }

    fn to_dense(self) -> faer_dense::DenseLayout {
        match self {
            PackedLayout::ColMajor => faer_dense::DenseLayout::ColMajor,
            PackedLayout::Tiled { .. } => faer_dense::DenseLayout::ColMajor,
        }
    }
}

/// Prepacked representation of the right-hand side operand used by matmul.
#[derive(Clone, Debug)]
pub struct PackedB {
    cols: usize,
    inner: usize,
    tile: Tile,
    layout: PackedLayout,
    buf: Arc<AlignedVec>,
}

impl PackedB {
    pub fn from_tensor(tensor: &Tensor, tile: Tile) -> PureResult<Self> {
        match tensor.layout {
            Layout::RowMajor => Self::from_row_major(tensor, tile),
            Layout::ColMajor => Ok(Self::from_col_major(tensor, tile)),
            Layout::Tiled { .. } => Err(TensorError::UnsupportedLayout {
                label: "packing tiled tensors is not yet supported",
            }),
            Layout::Chimera { .. } => Err(TensorError::UnsupportedLayout {
                label: "convert chimera tensors to row major before packing",
            }),
        }
    }

    fn from_row_major(tensor: &Tensor, tile: Tile) -> PureResult<Self> {
        let rows = tensor.rows;
        let cols = tensor.cols;
        let mut packed = aligned_zeroed(rows * cols);
        let data = tensor.data();
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                packed[c * rows + r] = data[offset + c];
            }
        }
        Ok(Self {
            cols,
            inner: rows,
            tile,
            layout: PackedLayout::ColMajor,
            buf: Arc::new(packed),
        })
    }

    fn from_col_major(tensor: &Tensor, tile: Tile) -> Self {
        let rows = tensor.rows;
        let cols = tensor.cols;
        let buf = tensor
            .data
            .try_clone_owned()
            .unwrap_or_else(|| Arc::new(aligned_from_slice(tensor.data())));
        Self {
            cols,
            inner: rows,
            tile,
            layout: PackedLayout::ColMajor,
            buf,
        }
    }

    pub fn from_tensor_transpose(tensor: &Tensor, tile: Tile) -> PureResult<Self> {
        match tensor.layout {
            Layout::RowMajor => Self::from_row_major_transpose(tensor, tile),
            Layout::ColMajor => Self::from_col_major_transpose(tensor, tile),
            Layout::Tiled { .. } => Err(TensorError::UnsupportedLayout {
                label: "packing tiled tensors is not yet supported",
            }),
            Layout::Chimera { .. } => Err(TensorError::UnsupportedLayout {
                label: "convert chimera tensors to row major before packing",
            }),
        }
    }

    fn from_row_major_transpose(tensor: &Tensor, tile: Tile) -> PureResult<Self> {
        let rows = tensor.rows;
        let cols = tensor.cols;
        let buf = tensor
            .data
            .try_clone_owned()
            .unwrap_or_else(|| Arc::new(aligned_from_slice(tensor.data())));
        Ok(Self {
            cols: rows,
            inner: cols,
            tile,
            layout: PackedLayout::ColMajor,
            buf,
        })
    }

    fn from_col_major_transpose(tensor: &Tensor, tile: Tile) -> PureResult<Self> {
        let transposed = tensor.transpose_with_backend(TensorUtilBackend::Auto)?;
        let mut packed = PackedB::from_row_major(&transposed, tile)?;
        packed.cols = tensor.rows;
        packed.inner = tensor.cols;
        Ok(packed)
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn inner(&self) -> usize {
        self.inner
    }

    #[inline]
    pub fn tile(&self) -> Tile {
        self.tile
    }

    #[inline]
    pub fn layout(&self) -> PackedLayout {
        self.layout
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        self.buf.as_slice()
    }
}

/// A simple 2D tensor backed by a reference-counted buffer with explicit layout metadata.
#[derive(Clone, Debug)]
pub struct Tensor {
    data: Arc<TensorBuffer>,
    rows: usize,
    cols: usize,
    layout: Layout,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self.layout == other.layout
            && self.data.as_slice() == other.data.as_slice()
    }
}

impl Tensor {
    fn from_aligned(
        rows: usize,
        cols: usize,
        data: AlignedVec,
        layout: Layout,
    ) -> PureResult<Self> {
        let expected = rows * cols;
        if expected != data.len() {
            return Err(TensorError::DataLength {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            data: Arc::new(TensorBuffer::from_aligned(data)),
            rows,
            cols,
            layout,
        })
    }

    fn seedable_rng(seed: Option<u64>, label: &str) -> StdRng {
        determinism::rng_from_optional(seed, label)
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> PureResult<Self> {
        let data = aligned_zeroed(rows * cols);
        Self::from_aligned(rows, cols, data, Layout::RowMajor)
    }

    /// Create a tensor from raw data. The provided vector must match
    /// `rows * cols` elements.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> PureResult<Self> {
        let data = aligned_from_vec(data);
        Self::from_aligned(rows, cols, data, Layout::RowMajor)
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
        if min.partial_cmp(&max) != Some(std::cmp::Ordering::Less) {
            return Err(TensorError::InvalidValue {
                label: "random_uniform_bounds",
            });
        }
        let mut rng = Self::seedable_rng(seed, "st-tensor/tensor/uniform");
        let distribution = Uniform::new(min, max);
        let mut data = aligned_with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(distribution.sample(&mut rng));
        }
        Self::from_aligned(rows, cols, data, Layout::RowMajor)
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
        let mut rng = Self::seedable_rng(seed, "st-tensor/tensor/normal");
        let gaussian = StandardNormal;
        let mut data = aligned_with_capacity(rows * cols);
        for _ in 0..rows * cols {
            let sample: f64 = gaussian.sample(&mut rng);
            data.push(mean + std * sample as f32);
        }
        Self::from_aligned(rows, cols, data, Layout::RowMajor)
    }

    /// Construct a tensor by applying a generator function to each coordinate.
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> PureResult<Self>
    where
        F: FnMut(usize, usize) -> f32,
    {
        let mut data = aligned_with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self::from_aligned(rows, cols, data, Layout::RowMajor)
    }

    /// Returns the `(rows, cols)` pair of the tensor.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Total number of elements stored in the tensor.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the logical stride for advancing one row and one column.
    pub fn strides(&self) -> PureResult<(usize, usize)> {
        match self.layout {
            Layout::RowMajor => Ok((self.cols, 1)),
            Layout::ColMajor => Ok((1, self.rows)),
            Layout::Tiled { .. } => Err(TensorError::UnsupportedLayout {
                label: "tiled layout does not expose uniform strides",
            }),
            Layout::Chimera { .. } => Err(TensorError::UnsupportedLayout {
                label: "chimera layout does not have uniform row/col strides",
            }),
        }
    }

    /// Checks whether the tensor storage is contiguous for row- or column-major traversals.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        matches!(self.layout, Layout::RowMajor | Layout::ColMajor)
    }

    /// Returns `true` when the backing buffer is aligned to the requested byte boundary.
    #[inline]
    pub fn is_aligned_to(&self, alignment: usize) -> bool {
        is_ptr_aligned(self.data.as_ref().as_ptr(), alignment)
    }

    /// Returns `true` when the buffer is 16-byte aligned which enables `vec4` access on GPUs.
    #[inline]
    pub fn is_vec4_aligned(&self) -> bool {
        self.is_aligned_to(16)
    }

    /// Returns the layout descriptor attached to the tensor.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Returns a tensor whose buffer is reorganised to match the requested layout.
    pub fn to_layout(&self, layout: Layout) -> PureResult<Tensor> {
        if layout == self.layout {
            return Ok(self.clone());
        }

        match (self.layout, layout) {
            (Layout::RowMajor, Layout::ColMajor) => {
                let mut data = aligned_zeroed(self.len());
                let source = self.data();
                for r in 0..self.rows {
                    let offset = r * self.cols;
                    for c in 0..self.cols {
                        data[c * self.rows + r] = source[offset + c];
                    }
                }
                Tensor::from_aligned(self.rows, self.cols, data, Layout::ColMajor)
            }
            (Layout::ColMajor, Layout::RowMajor) => {
                let mut data = aligned_zeroed(self.len());
                let source = self.data();
                for c in 0..self.cols {
                    let offset = c * self.rows;
                    for r in 0..self.rows {
                        data[r * self.cols + c] = source[offset + r];
                    }
                }
                Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)
            }
            (Layout::RowMajor, Layout::Chimera { stripes, tile }) => {
                Self::convert_row_major_to_chimera(self, stripes, tile)
            }
            (Layout::Chimera { stripes, tile }, Layout::RowMajor) => {
                Self::convert_chimera_to_row_major(self, stripes, tile)
            }
            (Layout::Chimera { stripes, tile }, Layout::ColMajor) => {
                Self::convert_chimera_to_row_major(self, stripes, tile)?.to_layout(layout)
            }
            (Layout::ColMajor, Layout::Chimera { .. }) => {
                self.to_layout(Layout::RowMajor)?.to_layout(layout)
            }
            (
                Layout::Chimera {
                    stripes: src_stripes,
                    tile: src_tile,
                },
                Layout::Chimera {
                    stripes: dst_stripes,
                    tile: dst_tile,
                },
            ) => {
                if src_stripes == dst_stripes && src_tile == dst_tile {
                    Ok(self.clone())
                } else {
                    Self::convert_chimera_to_row_major(self, src_stripes, src_tile)?
                        .to_layout(layout)
                }
            }
            _ => Err(TensorError::UnsupportedLayout {
                label: "layout conversion requires row- or col-major tensors",
            }),
        }
    }

    fn convert_row_major_to_chimera(
        tensor: &Tensor,
        stripes: u32,
        tile: u32,
    ) -> PureResult<Tensor> {
        if stripes == 0 || tile == 0 {
            return Err(TensorError::UnsupportedLayout {
                label: "chimera layout requires positive stripes and tile",
            });
        }

        let stripes = stripes as usize;
        let tile = tile as usize;
        if stripes * tile != tensor.cols {
            return Err(TensorError::UnsupportedLayout {
                label: "chimera layout expects stripes * tile to equal the column count",
            });
        }

        let mut data = aligned_zeroed(tensor.len());
        let source = tensor.data();
        for r in 0..tensor.rows {
            let src_row_offset = r * tensor.cols;
            let dst_row_offset = r * tensor.cols;
            for c in 0..tensor.cols {
                let stripe = c / tile;
                let within = c % tile;
                let dst_index = dst_row_offset + within * stripes + stripe;
                data[dst_index] = source[src_row_offset + c];
            }
        }

        Tensor::from_aligned(
            tensor.rows,
            tensor.cols,
            data,
            Layout::Chimera {
                stripes: stripes as u32,
                tile: tile as u32,
            },
        )
    }

    fn convert_chimera_to_row_major(
        tensor: &Tensor,
        stripes: u32,
        tile: u32,
    ) -> PureResult<Tensor> {
        if stripes == 0 || tile == 0 {
            return Err(TensorError::UnsupportedLayout {
                label: "chimera layout requires positive stripes and tile",
            });
        }

        let stripes = stripes as usize;
        let tile = tile as usize;
        if stripes * tile != tensor.cols {
            return Err(TensorError::UnsupportedLayout {
                label: "chimera layout expects stripes * tile to equal the column count",
            });
        }

        let mut data = aligned_zeroed(tensor.len());
        let source = tensor.data();
        for r in 0..tensor.rows {
            let dst_row_offset = r * tensor.cols;
            let src_row_offset = r * tensor.cols;
            for stripe in 0..stripes {
                for within in 0..tile {
                    let c = stripe * tile + within;
                    let src_index = src_row_offset + within * stripes + stripe;
                    data[dst_row_offset + c] = source[src_index];
                }
            }
        }

        Tensor::from_aligned(tensor.rows, tensor.cols, data, Layout::RowMajor)
    }

    /// Returns a read-only view of the underlying buffer.
    pub fn data(&self) -> &[f32] {
        self.data.as_slice()
    }

    /// Returns a mutable view of the underlying buffer.
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.layout = Layout::RowMajor;
        Arc::make_mut(&mut self.data).make_mut_slice()
    }

    /// Return the DLPack device descriptor for this tensor.
    pub fn dlpack_device(&self) -> DLDevice {
        DLDevice {
            device_type: DLDeviceType::Cpu as i32,
            device_id: 0,
        }
    }

    /// Export the tensor as a managed DLPack tensor.
    pub fn to_dlpack(&self) -> PureResult<*mut DLManagedTensor> {
        self.layout.expect_row_major("dlpack export")?;
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

        let manager_ctx = Box::into_raw(state) as *mut c_void;
        let managed = Box::new(DLManagedTensor {
            dl_tensor,
            manager_ctx,
            deleter: Some(drop_exported_state),
        });

        Ok(Box::into_raw(managed))
    }

    /// Return a zero-copy view of the tensor with new row/column dimensions.
    pub fn view(&self, rows: usize, cols: usize) -> PureResult<Tensor> {
        if rows * cols != self.len() {
            return Err(TensorError::DataLength {
                expected: rows * cols,
                got: self.len(),
            });
        }
        self.layout
            .expect_row_major("Tensor::view requires row-major storage")?;
        Ok(Tensor {
            data: Arc::clone(&self.data),
            rows,
            cols,
            layout: Layout::RowMajor,
        })
    }

    /// Construct a tensor from a managed DLPack tensor. The managed tensor is consumed.
    ///
    /// # Safety
    /// - `managed` must point to a valid `DLManagedTensor` produced by a DLPack exporter.
    /// - This function takes ownership of `managed`; the pointer must not be used again after a
    ///   successful call.
    /// - The tensor must describe a contiguous 2D CPU `f32` buffer in row-major order, and
    ///   `dl_tensor.data` + `dl_tensor.byte_offset` must remain valid for `rows * cols` elements
    ///   until the DLPack deleter runs.
    /// - The managed tensor must include a non-null deleter that is safe to call exactly once.
    pub unsafe fn from_dlpack(managed: *mut DLManagedTensor) -> PureResult<Self> {
        struct ManagedGuard {
            ptr: NonNull<DLManagedTensor>,
            armed: bool,
        }

        impl ManagedGuard {
            fn new(ptr: NonNull<DLManagedTensor>) -> Self {
                Self { ptr, armed: true }
            }

            fn tensor(&self) -> &DLManagedTensor {
                unsafe { self.ptr.as_ref() }
            }

            fn into_inner(mut self) -> NonNull<DLManagedTensor> {
                self.armed = false;
                self.ptr
            }
        }

        impl Drop for ManagedGuard {
            fn drop(&mut self) {
                if self.armed {
                    unsafe { call_managed_deleter(self.ptr.as_ptr()) }
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

        if !(dl_tensor.shape as usize).is_multiple_of(mem::align_of::<i64>()) {
            return Err(TensorError::DlpackError {
                message: "dlpack tensor shape pointer is misaligned".to_string(),
            });
        }

        if tensor.deleter.is_none() {
            return Err(TensorError::DlpackError {
                message: "dlpack tensor is missing a deleter".to_string(),
            });
        }

        let shape = unsafe {
            // SAFETY: `dl_tensor.shape` is non-null and aligned, and `dl_tensor.ndim == 2` ensures
            // there are exactly two shape entries.
            slice::from_raw_parts(dl_tensor.shape, 2)
        };
        let rows_i64 = shape[0];
        let cols_i64 = shape[1];

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
        let _ = (dl_tensor.data as usize)
            .checked_add(dl_tensor.byte_offset)
            .ok_or_else(|| TensorError::DlpackError {
                message: "byte offset causes pointer overflow".to_string(),
            })?;

        let base_ptr = match NonNull::new(dl_tensor.data as *mut f32) {
            Some(ptr) => ptr,
            None => {
                return Err(TensorError::EmptyInput("dlpack tensor data"));
            }
        };

        if !dl_tensor.strides.is_null() {
            if !(dl_tensor.strides as usize).is_multiple_of(mem::align_of::<i64>()) {
                return Err(TensorError::DlpackError {
                    message: "dlpack tensor strides pointer is misaligned".to_string(),
                });
            }

            let strides = unsafe {
                // SAFETY: `dl_tensor.strides` is non-null and aligned, and `dl_tensor.ndim == 2`
                // ensures there are exactly two stride entries when provided.
                slice::from_raw_parts(dl_tensor.strides, 2)
            };
            let expected_row = i64::try_from(cols).map_err(|_| TensorError::DlpackError {
                message: format!("cols {cols} exceed i64 range"),
            })?;

            // Treat degenerate layouts as contiguous row-major:
            // * When there is a single row, the stride along that axis is irrelevant.
            // * When there is a single column, a stride of 1 still respects row-major
            //   assumptions and matches how column vectors are typically exported.
            let row_ok =
                strides[0] == expected_row || rows_i64 == 1 || (cols_i64 == 1 && strides[0] == 1);
            let col_ok = strides[1] == 1;

            if !(row_ok && col_ok) {
                return Err(TensorError::DlpackError {
                    message: format!(
                        "only contiguous row-major tensors are supported; received strides {:?}",
                        strides
                    ),
                });
            }
        }

        let data_ptr = base_ptr.as_ptr().wrapping_add(offset);
        let data_ptr = match NonNull::new(data_ptr) {
            Some(ptr) => ptr,
            None => {
                return Err(TensorError::EmptyInput("dlpack tensor data"));
            }
        };

        if !(data_ptr.as_ptr() as usize).is_multiple_of(mem::align_of::<f32>()) {
            return Err(TensorError::DlpackError {
                message: "dlpack tensor data pointer is misaligned for f32 elements".to_string(),
            });
        }

        let foreign = unsafe { ForeignTensor::new(guard.into_inner(), data_ptr, expected_len) };

        Ok(Self {
            data: Arc::new(TensorBuffer::from_foreign(foreign)),
            rows,
            cols,
            layout: Layout::RowMajor,
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
        let rows = self.rows;
        let cols = other.cols;
        let mut tensor = Tensor::zeros(rows, cols)?;
        self.matmul_into_with_backend(other, &mut tensor, backend)?;
        Ok(tensor)
    }

    /// Matrix multiply into an existing tensor buffer.
    pub fn matmul_into_with_backend(
        &self,
        other: &Tensor,
        dst: &mut Tensor,
        backend: MatmulBackend,
    ) -> PureResult<()> {
        if self.cols != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;

        if dst.rows != rows || dst.cols != cols {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: dst.shape(),
            });
        }

        if Arc::ptr_eq(&self.data, &dst.data) || Arc::ptr_eq(&other.data, &dst.data) {
            return Err(TensorError::InvalidValue {
                label: "matmul_out_alias",
            });
        }

        self.layout.expect_row_major("matmul lhs")?;
        dst.layout.expect_row_major("matmul destination")?;
        dst.layout = Layout::RowMajor;

        let lhs = self.data();
        let mut scratch = aligned_zeroed(rows * cols);
        let work_slice = scratch.as_mut_slice();
        #[cfg(feature = "wgpu")]
        let mut fallback_reason: Option<&'static str> = None;
        #[cfg(feature = "wgpu")]
        let mut fallback_message: Option<String> = None;
        #[cfg(not(feature = "wgpu"))]
        let fallback_reason: Option<&'static str> = None;
        #[cfg(not(feature = "wgpu"))]
        let fallback_message: Option<String> = None;

        let backend_used = match backend {
            MatmulBackend::Auto => self.matmul_auto_into(other, work_slice, rows, inner, cols)?,
            MatmulBackend::CpuSimd => {
                if !matches!(other.layout, Layout::RowMajor) {
                    return Err(TensorError::UnsupportedLayout {
                        label: "simd matmul expects row-major rhs",
                    });
                }
                cpu_dense::matmul_into(work_slice, lhs, other.data(), rows, inner, cols).map_err(
                    |message| TensorError::BackendFailure {
                        backend: "cpu_simd",
                        message,
                    },
                )?;
                "cpu_simd"
            }
            MatmulBackend::CpuNaive => {
                let packed = PackedB::from_tensor(other, Tile::col_major())?;
                matmul_naive_packed_into(work_slice, lhs, rows, inner, cols, &packed);
                "naive"
            }
            MatmulBackend::CpuFaer => {
                let packed = PackedB::from_tensor(other, Tile::col_major())?;
                let lhs_layout = self.layout.to_dense(rows, inner)?;
                let rhs_layout = packed.layout().to_dense();
                faer_dense::matmul_oriented_into(
                    work_slice,
                    lhs,
                    lhs_layout,
                    packed.as_slice(),
                    rhs_layout,
                    rows,
                    inner,
                    cols,
                )
                .map_err(|message| TensorError::BackendFailure {
                    backend: "faer",
                    message,
                })?;
                "faer"
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => {
                if !matches!(other.layout, Layout::RowMajor) {
                    return Err(TensorError::UnsupportedLayout {
                        label: "wgpu matmul expects row-major rhs",
                    });
                }
                let rhs = other.data();
                match matmul_wgpu(lhs, rhs, rows, inner, cols) {
                    Ok(buffer) => {
                        work_slice.copy_from_slice(&buffer);
                        "wgpu"
                    }
                    Err(error)
                        if !strict_gpu_path() && wgpu_backend_runtime_unavailable(&error) =>
                    {
                        let packed = PackedB::from_tensor(other, Tile::col_major())?;
                        matmul_naive_packed_into(work_slice, lhs, rows, inner, cols, &packed);
                        fallback_reason = Some(WGPU_RUNTIME_FALLBACK_REASON);
                        fallback_message = Some(error.to_string());
                        "naive"
                    }
                    Err(error) => return Err(error),
                }
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                if !matches!(other.layout, Layout::RowMajor) {
                    return Err(TensorError::UnsupportedLayout {
                        label: "hip matmul expects row-major rhs",
                    });
                }
                let rhs = other.data();
                hip_dense::matmul_into(lhs, rhs, work_slice, rows, inner, cols).map_err(
                    |message| TensorError::BackendFailure {
                        backend: "hip",
                        message,
                    },
                )?;
                "hip"
            }
        };
        Self::validate_finite_tensor_util_slice("matmul_output", scratch.as_slice())?;
        dst.data_mut().copy_from_slice(scratch.as_slice());

        crate::emit_tensor_op(
            "matmul",
            &[self.rows, self.cols, other.rows, other.cols],
            &[rows, cols],
        );
        crate::emit_tensor_op_meta("matmul", || {
            serde_json::json!({
                "backend": backend_used,
                "requested_backend": backend.label(),
                "rows": rows,
                "inner": inner,
                "cols": cols,
                "lhs_layout": self.layout.as_str(),
                "rhs_layout": other.layout.as_str(),
                "fallback": fallback_reason.map(|reason| {
                    wgpu_runtime_fallback_meta("naive", reason, fallback_message.as_deref())
                }),
            })
        });
        Ok(())
    }

    /// Matrix multiply followed by a scalar multiply, fused where the backend supports it.
    pub fn matmul_scaled_with_backend(
        &self,
        other: &Tensor,
        scale: f32,
        backend: MatmulBackend,
    ) -> PureResult<Tensor> {
        if self.cols != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        Self::validate_scale_factor("matmul_scaled_factor", scale)?;

        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;
        let mut tensor = Tensor::zeros(rows, cols)?;

        self.layout.expect_row_major("matmul lhs")?;
        tensor.layout.expect_row_major("matmul destination")?;
        tensor.layout = Layout::RowMajor;

        let lhs = self.data();
        let dst_slice = tensor.data_mut();
        #[cfg(feature = "wgpu")]
        let mut fallback_reason: Option<&'static str> = None;
        #[cfg(feature = "wgpu")]
        let mut fallback_message: Option<String> = None;
        #[cfg(not(feature = "wgpu"))]
        let fallback_reason: Option<&'static str> = None;
        #[cfg(not(feature = "wgpu"))]
        let fallback_message: Option<String> = None;

        let backend_used =
            match backend {
                MatmulBackend::Auto => {
                    self.matmul_scaled_auto_into(other, scale, dst_slice, rows, inner, cols)?
                }
                MatmulBackend::CpuSimd => {
                    if !matches!(other.layout, Layout::RowMajor) {
                        return Err(TensorError::UnsupportedLayout {
                            label: "simd matmul expects row-major rhs",
                        });
                    }
                    cpu_dense::matmul_into(dst_slice, lhs, other.data(), rows, inner, cols)
                        .map_err(|message| TensorError::BackendFailure {
                            backend: "cpu_simd",
                            message,
                        })?;
                    scale_inplace(dst_slice, scale);
                    "cpu_simd"
                }
                MatmulBackend::CpuNaive => {
                    let packed = PackedB::from_tensor(other, Tile::col_major())?;
                    matmul_naive_packed_into(dst_slice, lhs, rows, inner, cols, &packed);
                    scale_inplace(dst_slice, scale);
                    "naive"
                }
                MatmulBackend::CpuFaer => {
                    let packed = PackedB::from_tensor(other, Tile::col_major())?;
                    let lhs_layout = self.layout.to_dense(rows, inner)?;
                    let rhs_layout = packed.layout().to_dense();
                    faer_dense::matmul_oriented_into(
                        dst_slice,
                        lhs,
                        lhs_layout,
                        packed.as_slice(),
                        rhs_layout,
                        rows,
                        inner,
                        cols,
                    )
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "faer",
                        message,
                    })?;
                    scale_inplace(dst_slice, scale);
                    "faer"
                }
                #[cfg(feature = "wgpu")]
                MatmulBackend::GpuWgpu => {
                    if !matches!(other.layout, Layout::RowMajor) {
                        return Err(TensorError::UnsupportedLayout {
                            label: "wgpu matmul expects row-major rhs",
                        });
                    }
                    match matmul_scaled_wgpu(lhs, other.data(), rows, inner, cols, scale) {
                        Ok(buffer) => {
                            dst_slice.copy_from_slice(&buffer);
                            "wgpu"
                        }
                        Err(error)
                            if !strict_gpu_path() && wgpu_backend_runtime_unavailable(&error) =>
                        {
                            let packed = PackedB::from_tensor(other, Tile::col_major())?;
                            matmul_naive_packed_into(dst_slice, lhs, rows, inner, cols, &packed);
                            scale_inplace(dst_slice, scale);
                            fallback_reason = Some(WGPU_RUNTIME_FALLBACK_REASON);
                            fallback_message = Some(error.to_string());
                            "naive"
                        }
                        Err(error) => return Err(error),
                    }
                }
                #[cfg(feature = "hip")]
                MatmulBackend::GpuHip => {
                    if !matches!(other.layout, Layout::RowMajor) {
                        return Err(TensorError::UnsupportedLayout {
                            label: "hip matmul expects row-major rhs",
                        });
                    }
                    hip_dense::matmul_into(lhs, other.data(), dst_slice, rows, inner, cols)
                        .map_err(|message| TensorError::BackendFailure {
                            backend: "hip",
                            message,
                        })?;
                    scale_inplace(dst_slice, scale);
                    "hip"
                }
            };
        Self::validate_finite_tensor_util_slice("matmul_scaled_output", dst_slice)?;

        crate::emit_tensor_op(
            "matmul_scaled",
            &[self.rows, self.cols, other.rows, other.cols],
            &[rows, cols],
        );
        crate::emit_tensor_op_meta("matmul_scaled", || {
            serde_json::json!({
                "backend": backend_used,
                "requested_backend": backend.label(),
                "rows": rows,
                "inner": inner,
                "cols": cols,
                "lhs_layout": self.layout.as_str(),
                "rhs_layout": other.layout.as_str(),
                "scale": scale,
                "fallback": fallback_reason.map(|reason| {
                    wgpu_runtime_fallback_meta("naive", reason, fallback_message.as_deref())
                }),
            })
        });
        Ok(tensor)
    }

    /// Computes `self.T @ other * scale` without materialising `self.T`.
    pub fn matmul_lhs_transpose_scaled_with_backend(
        &self,
        other: &Tensor,
        scale: f32,
        backend: MatmulBackend,
    ) -> PureResult<Tensor> {
        if self.rows != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        Self::validate_scale_factor("matmul_lhs_transpose_scaled_factor", scale)?;

        let rows = self.cols;
        let inner = self.rows;
        let cols = other.cols;
        let mut tensor = Tensor::zeros(rows, cols)?;

        self.layout.expect_row_major("matmul lhs")?;
        other.layout.expect_row_major("matmul rhs")?;
        tensor.layout.expect_row_major("matmul destination")?;
        tensor.layout = Layout::RowMajor;

        let dst_slice = tensor.data_mut();
        #[cfg(feature = "wgpu")]
        let mut fallback_reason: Option<&'static str> = None;
        #[cfg(feature = "wgpu")]
        let mut fallback_message: Option<String> = None;
        #[cfg(not(feature = "wgpu"))]
        let fallback_reason: Option<&'static str> = None;
        #[cfg(not(feature = "wgpu"))]
        let fallback_message: Option<String> = None;
        let backend_used = match backend {
            MatmulBackend::Auto => self.matmul_lhs_transpose_scaled_auto_into(
                other, scale, dst_slice, rows, inner, cols,
            )?,
            MatmulBackend::CpuSimd | MatmulBackend::CpuNaive | MatmulBackend::CpuFaer => {
                matmul_lhs_transpose_scaled_naive_into(
                    dst_slice,
                    self.data(),
                    other.data(),
                    inner,
                    rows,
                    cols,
                    scale,
                );
                "naive"
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => {
                match matmul_lhs_transpose_scaled_wgpu(
                    self.data(),
                    other.data(),
                    inner,
                    rows,
                    cols,
                    scale,
                ) {
                    Ok(buffer) => {
                        dst_slice.copy_from_slice(&buffer);
                        "wgpu"
                    }
                    Err(error)
                        if !strict_gpu_path() && wgpu_backend_runtime_unavailable(&error) =>
                    {
                        matmul_lhs_transpose_scaled_naive_into(
                            dst_slice,
                            self.data(),
                            other.data(),
                            inner,
                            rows,
                            cols,
                            scale,
                        );
                        fallback_reason = Some(WGPU_RUNTIME_FALLBACK_REASON);
                        fallback_message = Some(error.to_string());
                        "naive"
                    }
                    Err(error) => return Err(error),
                }
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                matmul_lhs_transpose_scaled_naive_into(
                    dst_slice,
                    self.data(),
                    other.data(),
                    inner,
                    rows,
                    cols,
                    scale,
                );
                "naive"
            }
        };
        Self::validate_finite_tensor_util_slice("matmul_lhs_transpose_scaled_output", dst_slice)?;

        crate::emit_tensor_op(
            "matmul_lhs_transpose_scaled",
            &[self.rows, self.cols, other.rows, other.cols],
            &[rows, cols],
        );
        crate::emit_tensor_op_meta("matmul_lhs_transpose_scaled", || {
            serde_json::json!({
                "backend": backend_used,
                "requested_backend": backend.label(),
                "rows": rows,
                "inner": inner,
                "cols": cols,
                "lhs_layout": self.layout.as_str(),
                "rhs_layout": other.layout.as_str(),
                "lhs_transpose": true,
                "scale": scale,
                "fallback": fallback_reason.map(|reason| {
                    wgpu_runtime_fallback_meta("naive", reason, fallback_message.as_deref())
                }),
            })
        });
        Ok(tensor)
    }

    /// Matrix multiply using a prepacked right-hand side operand.
    pub fn matmul_prepacked(&self, packed: &PackedB) -> PureResult<Tensor> {
        self.matmul_prepacked_with_backend(packed, MatmulBackend::Auto)
    }

    /// Matrix multiply against a prepacked operand with an explicit backend selection.
    pub fn matmul_prepacked_with_backend(
        &self,
        packed: &PackedB,
        backend: MatmulBackend,
    ) -> PureResult<Tensor> {
        let rows = self.rows;
        let cols = packed.cols();
        let mut tensor = Tensor::zeros(rows, cols)?;
        self.matmul_prepacked_into_with_backend(packed, &mut tensor, backend)?;
        Ok(tensor)
    }

    /// Matrix multiply against a prepacked operand followed by row-wise bias addition.
    pub fn matmul_prepacked_bias(&self, packed: &PackedB, bias: &[f32]) -> PureResult<Tensor> {
        self.matmul_prepacked_bias_with_backend(packed, bias, MatmulBackend::Auto)
    }

    /// Matrix multiply against a prepacked operand with fused bias where the backend supports it.
    pub fn matmul_prepacked_bias_with_backend(
        &self,
        packed: &PackedB,
        bias: &[f32],
        backend: MatmulBackend,
    ) -> PureResult<Tensor> {
        if self.cols != packed.inner() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: (packed.inner(), packed.cols()),
            });
        }
        if bias.len() != packed.cols() {
            return Err(TensorError::DataLength {
                expected: packed.cols(),
                got: bias.len(),
            });
        }
        Self::validate_finite_tensor_util_slice("matmul_prepacked_bias_bias", bias)?;

        let rows = self.rows;
        let cols = packed.cols();
        let inner = packed.inner();
        let mut tensor = Tensor::zeros(rows, cols)?;

        self.layout.expect_row_major("matmul lhs")?;
        tensor.layout.expect_row_major("matmul destination")?;
        tensor.layout = Layout::RowMajor;

        let lhs = self.data();
        let dst_slice = tensor.data_mut();
        #[cfg(feature = "wgpu")]
        let mut fallback_reason: Option<&'static str> = None;
        #[cfg(feature = "wgpu")]
        let mut fallback_message: Option<String> = None;
        #[cfg(not(feature = "wgpu"))]
        let fallback_reason: Option<&'static str> = None;
        #[cfg(not(feature = "wgpu"))]
        let fallback_message: Option<String> = None;

        let backend_used = match backend {
            MatmulBackend::Auto => {
                self.matmul_prepacked_bias_auto_into(packed, bias, dst_slice, rows, inner, cols)?
            }
            MatmulBackend::CpuSimd => {
                if !matches!(packed.layout(), PackedLayout::ColMajor) {
                    return Err(TensorError::UnsupportedLayout {
                        label: "simd matmul expects col-major packed rhs",
                    });
                }
                cpu_dense::matmul_packed_into(dst_slice, lhs, packed.as_slice(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "cpu_simd",
                        message,
                    })?;
                add_bias_inplace(dst_slice, rows, cols, bias);
                "cpu_simd"
            }
            MatmulBackend::CpuNaive => {
                matmul_naive_packed_into(dst_slice, lhs, rows, inner, cols, packed);
                add_bias_inplace(dst_slice, rows, cols, bias);
                "naive"
            }
            MatmulBackend::CpuFaer => {
                let lhs_layout = self.layout.to_dense(rows, inner)?;
                let rhs_layout = packed.layout().to_dense();
                faer_dense::matmul_oriented_into(
                    dst_slice,
                    lhs,
                    lhs_layout,
                    packed.as_slice(),
                    rhs_layout,
                    rows,
                    inner,
                    cols,
                )
                .map_err(|message| TensorError::BackendFailure {
                    backend: "faer",
                    message,
                })?;
                add_bias_inplace(dst_slice, rows, cols, bias);
                "faer"
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => {
                match wgpu_dense::matmul_prepacked_bias(lhs, packed, bias, rows, inner, cols) {
                    Ok(buffer) => {
                        dst_slice.copy_from_slice(&buffer);
                        "wgpu"
                    }
                    Err(message) if !strict_gpu_path() && wgpu_runtime_unavailable(&message) => {
                        matmul_naive_packed_into(dst_slice, lhs, rows, inner, cols, packed);
                        add_bias_inplace(dst_slice, rows, cols, bias);
                        fallback_reason = Some(WGPU_RUNTIME_FALLBACK_REASON);
                        fallback_message = Some(message);
                        "naive"
                    }
                    Err(message) => {
                        return Err(TensorError::BackendFailure {
                            backend: "wgpu",
                            message,
                        });
                    }
                }
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                return Err(TensorError::BackendFailure {
                    backend: "hip",
                    message: "hip matmul does not yet support prepacked operands".into(),
                });
            }
        };
        Self::validate_finite_tensor_util_slice("matmul_prepacked_bias_output", dst_slice)?;

        crate::emit_tensor_op(
            "matmul_prepacked_bias",
            &[self.rows, self.cols, packed.inner(), packed.cols()],
            &[rows, cols],
        );
        crate::emit_tensor_op_meta("matmul_prepacked_bias", || {
            serde_json::json!({
                "backend": backend_used,
                "requested_backend": backend.label(),
                "rows": rows,
                "inner": inner,
                "cols": cols,
                "lhs_layout": self.layout.as_str(),
                "rhs_layout": "packed",
                "packed_layout": packed.layout().as_str(),
                "packed_tile": {
                    "tm": packed.tile().tm,
                    "tn": packed.tile().tn,
                    "tk": packed.tile().tk,
                },
                "fused_bias": true,
                "fallback": fallback_reason.map(|reason| {
                    wgpu_runtime_fallback_meta("naive", reason, fallback_message.as_deref())
                }),
            })
        });
        Ok(tensor)
    }

    /// Matrix multiply into an existing tensor buffer using a prepacked operand.
    pub fn matmul_prepacked_into_with_backend(
        &self,
        packed: &PackedB,
        dst: &mut Tensor,
        backend: MatmulBackend,
    ) -> PureResult<()> {
        if self.cols != packed.inner() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: (packed.inner(), packed.cols()),
            });
        }

        let rows = self.rows;
        let cols = packed.cols();
        let inner = packed.inner();

        if dst.rows != rows || dst.cols != cols {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: dst.shape(),
            });
        }

        if Arc::ptr_eq(&self.data, &dst.data) {
            return Err(TensorError::InvalidValue {
                label: "matmul_out_alias",
            });
        }

        self.layout.expect_row_major("matmul lhs")?;
        dst.layout.expect_row_major("matmul destination")?;
        dst.layout = Layout::RowMajor;

        let lhs = self.data();
        let mut scratch = aligned_zeroed(rows * cols);
        let work_slice = scratch.as_mut_slice();
        #[cfg(feature = "wgpu")]
        let mut fallback_reason: Option<&'static str> = None;
        #[cfg(feature = "wgpu")]
        let mut fallback_message: Option<String> = None;
        #[cfg(not(feature = "wgpu"))]
        let fallback_reason: Option<&'static str> = None;
        #[cfg(not(feature = "wgpu"))]
        let fallback_message: Option<String> = None;

        let backend_used = match backend {
            MatmulBackend::Auto => {
                self.matmul_prepacked_auto_into(packed, work_slice, rows, inner, cols)?
            }
            MatmulBackend::CpuSimd => {
                if !matches!(packed.layout(), PackedLayout::ColMajor) {
                    return Err(TensorError::UnsupportedLayout {
                        label: "simd matmul expects col-major packed rhs",
                    });
                }
                cpu_dense::matmul_packed_into(
                    work_slice,
                    lhs,
                    packed.as_slice(),
                    rows,
                    inner,
                    cols,
                )
                .map_err(|message| TensorError::BackendFailure {
                    backend: "cpu_simd",
                    message,
                })?;
                "cpu_simd"
            }
            MatmulBackend::CpuNaive => {
                matmul_naive_packed_into(work_slice, lhs, rows, inner, cols, packed);
                "naive"
            }
            MatmulBackend::CpuFaer => {
                let lhs_layout = self.layout.to_dense(rows, inner)?;
                let rhs_layout = packed.layout().to_dense();
                faer_dense::matmul_oriented_into(
                    work_slice,
                    lhs,
                    lhs_layout,
                    packed.as_slice(),
                    rhs_layout,
                    rows,
                    inner,
                    cols,
                )
                .map_err(|message| TensorError::BackendFailure {
                    backend: "faer",
                    message,
                })?;
                "faer"
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => {
                match wgpu_dense::matmul_prepacked(lhs, packed, rows, inner, cols) {
                    Ok(buffer) => {
                        work_slice.copy_from_slice(&buffer);
                        "wgpu"
                    }
                    Err(message) if !strict_gpu_path() && wgpu_runtime_unavailable(&message) => {
                        matmul_naive_packed_into(work_slice, lhs, rows, inner, cols, packed);
                        fallback_reason = Some(WGPU_RUNTIME_FALLBACK_REASON);
                        fallback_message = Some(message);
                        "naive"
                    }
                    Err(message) => {
                        return Err(TensorError::BackendFailure {
                            backend: "wgpu",
                            message,
                        });
                    }
                }
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                return Err(TensorError::BackendFailure {
                    backend: "hip",
                    message: "hip matmul does not yet support prepacked operands".into(),
                });
            }
        };
        Self::validate_finite_tensor_util_slice("matmul_prepacked_output", scratch.as_slice())?;
        dst.data_mut().copy_from_slice(scratch.as_slice());

        crate::emit_tensor_op(
            "matmul_prepacked",
            &[self.rows, self.cols, packed.inner(), packed.cols()],
            &[rows, cols],
        );
        crate::emit_tensor_op_meta("matmul_prepacked", || {
            serde_json::json!({
                "backend": backend_used,
                "requested_backend": backend.label(),
                "rows": rows,
                "inner": inner,
                "cols": cols,
                "lhs_layout": self.layout.as_str(),
                "rhs_layout": "packed",
                "packed_layout": packed.layout().as_str(),
                "packed_tile": {
                    "tm": packed.tile().tm,
                    "tn": packed.tile().tn,
                    "tk": packed.tile().tk,
                },
                "fallback": fallback_reason.map(|reason| {
                    wgpu_runtime_fallback_meta("naive", reason, fallback_message.as_deref())
                }),
            })
        });
        Ok(())
    }

    /// Matrix multiply writing into an existing tensor using automatic backend selection.
    pub fn matmul_into(&self, other: &Tensor, dst: &mut Tensor) -> PureResult<()> {
        self.matmul_into_with_backend(other, dst, MatmulBackend::Auto)
    }

    fn matmul_auto_into(
        &self,
        other: &Tensor,
        dst: &mut [f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<&'static str> {
        #[cfg(feature = "wgpu")]
        {
            if matches!(other.layout, Layout::RowMajor)
                && wgpu_dense::is_available()
                && wgpu_dense::should_use(rows, inner, cols)
            {
                match wgpu_dense::matmul(self.data(), other.data(), rows, inner, cols) {
                    Ok(buffer) => {
                        dst.copy_from_slice(&buffer);
                        return Ok("wgpu");
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "matmul", message));
                    }
                    Err(_) => {}
                }
            }
        }

        #[cfg(feature = "hip")]
        {
            if matches!(other.layout, Layout::RowMajor)
                && hip_dense::is_available()
                && hip_dense::should_use(rows, inner, cols)
            {
                match hip_dense::matmul_into(self.data(), other.data(), dst, rows, inner, cols) {
                    Ok(()) => return Ok("hip"),
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("hip", "matmul", message));
                    }
                    Err(_) => {}
                }
            }
        }

        if matches!(other.layout, Layout::RowMajor) && cpu_dense::should_use(rows, inner, cols) {
            if let Ok(()) =
                cpu_dense::matmul_into(dst, self.data(), other.data(), rows, inner, cols)
            {
                return Ok("cpu_simd");
            }
        }

        let packed = PackedB::from_tensor(other, Tile::col_major())?;
        self.matmul_prepacked_auto_into(&packed, dst, rows, inner, cols)
    }

    fn matmul_scaled_auto_into(
        &self,
        other: &Tensor,
        scale: f32,
        dst: &mut [f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<&'static str> {
        #[cfg(feature = "wgpu")]
        {
            if matches!(other.layout, Layout::RowMajor)
                && wgpu_dense::is_available()
                && wgpu_dense::should_use(rows, inner, cols)
            {
                match wgpu_dense::matmul_scaled(self.data(), other.data(), rows, inner, cols, scale)
                {
                    Ok(buffer) => {
                        dst.copy_from_slice(&buffer);
                        return Ok("wgpu");
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "matmul_scaled", message));
                    }
                    Err(_) => {}
                }
            }
        }

        #[cfg(feature = "hip")]
        {
            if matches!(other.layout, Layout::RowMajor)
                && hip_dense::is_available()
                && hip_dense::should_use(rows, inner, cols)
            {
                match hip_dense::matmul_into(self.data(), other.data(), dst, rows, inner, cols) {
                    Ok(()) => {
                        scale_inplace(dst, scale);
                        return Ok("hip");
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("hip", "matmul_scaled", message));
                    }
                    Err(_) => {}
                }
            }
        }

        if matches!(other.layout, Layout::RowMajor) && cpu_dense::should_use(rows, inner, cols) {
            if let Ok(()) =
                cpu_dense::matmul_into(dst, self.data(), other.data(), rows, inner, cols)
            {
                scale_inplace(dst, scale);
                return Ok("cpu_simd");
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            let packed = PackedB::from_tensor(other, Tile::col_major())?;
            if let Ok(()) = faer_dense::matmul_oriented_into(
                dst,
                self.data(),
                self.layout.to_dense(rows, inner)?,
                packed.as_slice(),
                packed.layout().to_dense(),
                rows,
                inner,
                cols,
            ) {
                scale_inplace(dst, scale);
                return Ok("faer");
            }
        }

        let packed = PackedB::from_tensor(other, Tile::col_major())?;
        matmul_naive_packed_into(dst, self.data(), rows, inner, cols, &packed);
        scale_inplace(dst, scale);
        Ok("naive")
    }

    fn matmul_lhs_transpose_scaled_auto_into(
        &self,
        other: &Tensor,
        scale: f32,
        dst: &mut [f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<&'static str> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                match wgpu_dense::matmul_lhs_transpose_scaled(
                    self.data(),
                    other.data(),
                    inner,
                    rows,
                    cols,
                    scale,
                ) {
                    Ok(buffer) => {
                        dst.copy_from_slice(&buffer);
                        return Ok("wgpu");
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "matmul_lhs_transpose_scaled",
                            message,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        matmul_lhs_transpose_scaled_naive_into(
            dst,
            self.data(),
            other.data(),
            inner,
            rows,
            cols,
            scale,
        );
        Ok("naive")
    }

    fn matmul_prepacked_auto_into(
        &self,
        packed: &PackedB,
        dst: &mut [f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<&'static str> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                match wgpu_dense::matmul_prepacked(self.data(), packed, rows, inner, cols) {
                    Ok(buffer) => {
                        dst.copy_from_slice(&buffer);
                        return Ok("wgpu");
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "matmul_prepacked",
                            message,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            if let Ok(()) = faer_dense::matmul_oriented_into(
                dst,
                self.data(),
                self.layout.to_dense(rows, inner)?,
                packed.as_slice(),
                packed.layout().to_dense(),
                rows,
                inner,
                cols,
            ) {
                return Ok("faer");
            }
        }

        matmul_naive_packed_into(dst, self.data(), rows, inner, cols, packed);
        Ok("naive")
    }

    fn matmul_prepacked_bias_auto_into(
        &self,
        packed: &PackedB,
        bias: &[f32],
        dst: &mut [f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<&'static str> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                match wgpu_dense::matmul_prepacked_bias(
                    self.data(),
                    packed,
                    bias,
                    rows,
                    inner,
                    cols,
                ) {
                    Ok(buffer) => {
                        dst.copy_from_slice(&buffer);
                        return Ok("wgpu");
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "matmul_prepacked_bias",
                            message,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            if let Ok(()) = faer_dense::matmul_oriented_into(
                dst,
                self.data(),
                self.layout.to_dense(rows, inner)?,
                packed.as_slice(),
                packed.layout().to_dense(),
                rows,
                inner,
                cols,
            ) {
                add_bias_inplace(dst, rows, cols, bias);
                return Ok("faer");
            }
        }

        matmul_naive_packed_into(dst, self.data(), rows, inner, cols, packed);
        add_bias_inplace(dst, rows, cols, bias);
        Ok("naive")
    }

    /// Row-wise softmax using automatic backend selection.
    pub fn row_softmax(&self) -> PureResult<Tensor> {
        self.row_softmax_with_backend(SoftmaxBackend::Auto)
    }

    /// Row-wise softmax with explicit backend control.
    pub fn row_softmax_with_backend(&self, backend: SoftmaxBackend) -> PureResult<Tensor> {
        let rows = self.rows;
        let cols = self.cols;
        Self::validate_finite_tensor_util_slice("row_softmax_input", self.data())?;

        let (output, backend_used, fallback_reason) = match backend {
            SoftmaxBackend::Auto => self
                .row_softmax_auto(rows, cols)
                .map(|(tensor, backend)| (tensor, backend, None::<&'static str>)),
            SoftmaxBackend::Cpu => {
                let buffer = row_softmax_cpu(self.data(), rows, cols);
                Tensor::from_vec(rows, cols, buffer)
                    .map(|tensor| (tensor, "cpu", None::<&'static str>))
            }
            #[cfg(feature = "wgpu")]
            SoftmaxBackend::GpuWgpu => self.row_softmax_wgpu_or_cpu(rows, cols),
        }?;
        Self::validate_finite_tensor_util_slice("row_softmax_output", output.data())?;

        crate::emit_tensor_op("row_softmax", &[rows, cols], &[rows, cols]);
        if backend_used != "wgpu_dense" {
            crate::emit_tensor_op_meta("row_softmax", || {
                serde_json::json!({
                    "backend": backend_used,
                    "requested_backend": backend.label(),
                    "rows": rows,
                    "cols": cols,
                    "layout": self.layout.as_str(),
                    "fallback": fallback_reason
                        .map(|reason| wgpu_runtime_fallback_meta("cpu", reason, None)),
                })
            });
        }
        Ok(output)
    }

    /// Row-wise softmax probabilities with accompanying hardmax mask using automatic backend selection.
    pub fn row_softmax_hardmax(&self) -> PureResult<(Tensor, Tensor)> {
        self.row_softmax_hardmax_with_backend(SoftmaxBackend::Auto)
    }

    /// Row-wise softmax and hardmax pair with explicit backend control.
    pub fn row_softmax_hardmax_with_backend(
        &self,
        backend: SoftmaxBackend,
    ) -> PureResult<(Tensor, Tensor)> {
        let rows = self.rows;
        let cols = self.cols;
        Self::validate_finite_tensor_util_slice("row_softmax_hardmax_input", self.data())?;

        let (output, backend_used, fallback_reason) = match backend {
            SoftmaxBackend::Auto => self
                .row_softmax_hardmax_auto(rows, cols)
                .map(|(pair, backend)| (pair, backend, None::<&'static str>)),
            SoftmaxBackend::Cpu => {
                let result = self.hardmax_fusion(
                    rows,
                    cols,
                    HardmaxBackend::Cpu,
                    HardmaxMode::SoftmaxAndMask,
                )?;
                let backend_used = result.backend;
                let fallback_reason = result.fallback_reason;
                self.fusion_pair_to_tensors(rows, cols, result)
                    .map(|pair| (pair, backend_used, fallback_reason))
            }
            #[cfg(feature = "wgpu")]
            SoftmaxBackend::GpuWgpu => {
                let result =
                    self.hardmax_fusion_wgpu_or_cpu(rows, cols, HardmaxMode::SoftmaxAndMask)?;
                let backend_used = result.backend;
                let fallback_reason = result.fallback_reason;
                self.fusion_pair_to_tensors(rows, cols, result)
                    .map(|pair| (pair, backend_used, fallback_reason))
            }
        }?;
        Self::validate_finite_tensor_util_slice("row_softmax_hardmax_softmax", output.0.data())?;
        Self::validate_finite_tensor_util_slice("row_softmax_hardmax_hardmax", output.1.data())?;

        crate::emit_tensor_op("row_softmax_hardmax", &[rows, cols], &[rows, cols]);
        if backend_used != "wgpu_dense" {
            crate::emit_tensor_op_meta("row_softmax_hardmax", || {
                serde_json::json!({
                    "backend": backend_used,
                    "requested_backend": backend.label(),
                    "rows": rows,
                    "cols": cols,
                    "layout": self.layout.as_str(),
                    "fallback": fallback_reason
                        .map(|reason| wgpu_runtime_fallback_meta("cpu", reason, None)),
                })
            });
        }
        Ok(output)
    }

    /// Row-wise softmax, hardmax, and spiral consensus payload using automatic backend selection.
    pub fn row_softmax_hardmax_spiral(&self) -> PureResult<SpiralSoftmaxHardmax> {
        self.row_softmax_hardmax_spiral_with_backend(SoftmaxBackend::Auto)
    }

    /// Row-wise softmax/hardmax/spiral payload with explicit backend control.
    pub fn row_softmax_hardmax_spiral_with_backend(
        &self,
        backend: SoftmaxBackend,
    ) -> PureResult<SpiralSoftmaxHardmax> {
        let rows = self.rows;
        let cols = self.cols;
        Self::validate_finite_tensor_util_slice("row_softmax_hardmax_spiral_input", self.data())?;

        let (output, backend_used, fallback_reason) = match backend {
            SoftmaxBackend::Auto => self
                .row_softmax_hardmax_spiral_auto(rows, cols)
                .map(|(spiral, backend)| (spiral, backend, None::<&'static str>)),
            SoftmaxBackend::Cpu => {
                let result = self.hardmax_fusion(
                    rows,
                    cols,
                    HardmaxBackend::Cpu,
                    HardmaxMode::SoftmaxAndMask,
                )?;
                let backend_used = result.backend;
                let fallback_reason = result.fallback_reason;
                self.fusion_pair_to_spiral(rows, cols, result)
                    .map(|spiral| (spiral, backend_used, fallback_reason))
            }
            #[cfg(feature = "wgpu")]
            SoftmaxBackend::GpuWgpu => {
                let result =
                    self.hardmax_fusion_wgpu_or_cpu(rows, cols, HardmaxMode::SoftmaxAndMask)?;
                let backend_used = result.backend;
                let fallback_reason = result.fallback_reason;
                self.fusion_pair_to_spiral(rows, cols, result)
                    .map(|spiral| (spiral, backend_used, fallback_reason))
            }
        }?;
        Self::validate_finite_tensor_util_slice(
            "row_softmax_hardmax_spiral_softmax",
            output.softmax.data(),
        )?;
        Self::validate_finite_tensor_util_slice(
            "row_softmax_hardmax_spiral_hardmax",
            output.hardmax.data(),
        )?;
        Self::validate_finite_tensor_util_slice(
            "row_softmax_hardmax_spiral_spiral",
            output.spiral.data(),
        )?;

        crate::emit_tensor_op("row_softmax_hardmax_spiral", &[rows, cols], &[rows, cols]);
        if backend_used != "wgpu_dense" {
            crate::emit_tensor_op_meta("row_softmax_hardmax_spiral", || {
                serde_json::json!({
                    "backend": backend_used,
                    "requested_backend": backend.label(),
                    "rows": rows,
                    "cols": cols,
                    "layout": self.layout.as_str(),
                    "fallback": fallback_reason
                        .map(|reason| wgpu_runtime_fallback_meta("cpu", reason, None)),
                })
            });
        }
        Ok(output)
    }

    /// Row-wise hardmax using automatic backend selection.
    pub fn row_hardmax(&self) -> PureResult<Tensor> {
        self.row_hardmax_with_backend(HardmaxBackend::Auto)
    }

    /// Row-wise hardmax with explicit backend control.
    pub fn row_hardmax_with_backend(&self, backend: HardmaxBackend) -> PureResult<Tensor> {
        let rows = self.rows;
        let cols = self.cols;
        Self::validate_finite_tensor_util_slice("row_hardmax_input", self.data())?;

        let (output, backend_used, fallback_reason) = match backend {
            HardmaxBackend::Auto => self
                .row_hardmax_auto(rows, cols)
                .map(|(tensor, backend)| (tensor, backend, None::<&'static str>)),
            HardmaxBackend::Cpu => {
                let result =
                    self.hardmax_fusion(rows, cols, HardmaxBackend::Cpu, HardmaxMode::MaskOnly)?;
                let backend_used = result.backend;
                let fallback_reason = result.fallback_reason;
                Tensor::from_vec(rows, cols, result.hardmax)
                    .map(|tensor| (tensor, backend_used, fallback_reason))
            }
            #[cfg(feature = "wgpu")]
            HardmaxBackend::GpuWgpu => {
                let result = self.hardmax_fusion_wgpu_or_cpu(rows, cols, HardmaxMode::MaskOnly)?;
                let backend_used = result.backend;
                let fallback_reason = result.fallback_reason;
                Tensor::from_vec(rows, cols, result.hardmax)
                    .map(|tensor| (tensor, backend_used, fallback_reason))
            }
        }?;
        Self::validate_finite_tensor_util_slice("row_hardmax_output", output.data())?;

        crate::emit_tensor_op("row_hardmax", &[rows, cols], &[rows, cols]);
        if backend_used != "wgpu_dense" {
            crate::emit_tensor_op_meta("row_hardmax", || {
                serde_json::json!({
                    "backend": backend_used,
                    "requested_backend": backend.label(),
                    "rows": rows,
                    "cols": cols,
                    "layout": self.layout.as_str(),
                    "fallback": fallback_reason
                        .map(|reason| wgpu_runtime_fallback_meta("cpu", reason, None)),
                })
            });
        }
        Ok(output)
    }

    #[cfg(feature = "wgpu")]
    fn row_softmax_wgpu_or_cpu(
        &self,
        rows: usize,
        cols: usize,
    ) -> PureResult<(Tensor, &'static str, Option<&'static str>)> {
        match wgpu_dense::row_softmax(self.data(), rows, cols, self.layout) {
            Ok(data) => {
                Tensor::from_vec(rows, cols, data).map(|tensor| (tensor, "wgpu_dense", None))
            }
            Err(message) if !strict_gpu_path() && wgpu_runtime_unavailable(&message) => {
                let buffer = row_softmax_cpu(self.data(), rows, cols);
                Tensor::from_vec(rows, cols, buffer)
                    .map(|tensor| (tensor, "cpu", Some(WGPU_RUNTIME_FALLBACK_REASON)))
            }
            Err(message) => Err(TensorError::BackendFailure {
                backend: "wgpu",
                message,
            }),
        }
    }

    #[cfg(feature = "wgpu")]
    fn hardmax_fusion_wgpu_or_cpu(
        &self,
        rows: usize,
        cols: usize,
        mode: HardmaxMode,
    ) -> PureResult<HardmaxFusionResult> {
        match self.hardmax_fusion(rows, cols, HardmaxBackend::GpuWgpu, mode) {
            Ok(result) => Ok(result),
            Err(error) if !strict_gpu_path() && wgpu_backend_runtime_unavailable(&error) => {
                let mut result = self.hardmax_fusion(rows, cols, HardmaxBackend::Cpu, mode)?;
                result.fallback_reason = Some(WGPU_RUNTIME_FALLBACK_REASON);
                Ok(result)
            }
            Err(error) => Err(error),
        }
    }

    fn row_softmax_auto(&self, rows: usize, cols: usize) -> PureResult<(Tensor, &'static str)> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::supports_row_softmax(rows, cols) {
                match wgpu_dense::row_softmax(self.data(), rows, cols, self.layout) {
                    Ok(buffer) => {
                        return Tensor::from_vec(rows, cols, buffer)
                            .map(|tensor| (tensor, "wgpu_dense"));
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "row_softmax", message));
                    }
                    Err(_) => {}
                }
            }
        }

        let buffer = row_softmax_cpu(self.data(), rows, cols);
        Tensor::from_vec(rows, cols, buffer).map(|tensor| (tensor, "cpu"))
    }

    fn row_softmax_hardmax_auto(
        &self,
        rows: usize,
        cols: usize,
    ) -> PureResult<((Tensor, Tensor), &'static str)> {
        let result = self.hardmax_fusion(
            rows,
            cols,
            HardmaxBackend::Auto,
            HardmaxMode::SoftmaxAndMask,
        )?;
        let backend_used = result.backend;
        self.fusion_pair_to_tensors(rows, cols, result)
            .map(|pair| (pair, backend_used))
    }

    fn row_softmax_hardmax_spiral_auto(
        &self,
        rows: usize,
        cols: usize,
    ) -> PureResult<(SpiralSoftmaxHardmax, &'static str)> {
        let result = self.hardmax_fusion(
            rows,
            cols,
            HardmaxBackend::Auto,
            HardmaxMode::SoftmaxAndMask,
        )?;
        let backend_used = result.backend;
        self.fusion_pair_to_spiral(rows, cols, result)
            .map(|spiral| (spiral, backend_used))
    }

    fn row_hardmax_auto(&self, rows: usize, cols: usize) -> PureResult<(Tensor, &'static str)> {
        let result =
            self.hardmax_fusion(rows, cols, HardmaxBackend::Auto, HardmaxMode::MaskOnly)?;
        let backend_used = result.backend;
        Tensor::from_vec(rows, cols, result.hardmax).map(|tensor| (tensor, backend_used))
    }

    fn hardmax_fusion(
        &self,
        rows: usize,
        cols: usize,
        backend: HardmaxBackend,
        mode: HardmaxMode,
    ) -> PureResult<HardmaxFusionResult> {
        HardmaxFusionPlan::new(self.data(), rows, cols)
            .layout(self.layout)
            .backend(backend)
            .mode(mode)
            .execute()
    }

    fn fusion_pair_to_tensors(
        &self,
        rows: usize,
        cols: usize,
        result: HardmaxFusionResult,
    ) -> PureResult<(Tensor, Tensor)> {
        let soft = result.softmax.ok_or_else(|| TensorError::BackendFailure {
            backend: "hardmax",
            message: "fusion result missing softmax payload".into(),
        })?;
        let soft_tensor = Tensor::from_vec(rows, cols, soft)?;
        let hard_tensor = Tensor::from_vec(rows, cols, result.hardmax)?;
        Ok((soft_tensor, hard_tensor))
    }

    fn fusion_pair_to_spiral(
        &self,
        rows: usize,
        cols: usize,
        result: HardmaxFusionResult,
    ) -> PureResult<SpiralSoftmaxHardmax> {
        let HardmaxFusionResult {
            backend: _,
            fallback_reason: _,
            softmax,
            hardmax,
            dp_reductions,
            einsum,
            fused_ops,
        } = result;

        let soft = softmax.ok_or_else(|| TensorError::BackendFailure {
            backend: "hardmax",
            message: "fusion result missing softmax payload".into(),
        })?;
        let hard = hardmax;
        let (spiral, mut metrics) = spiral_softmax_hardmax_consensus(&soft, &hard, rows, cols);

        if rows > 0 && cols > 0 {
            let total = (rows as f64) * (cols as f64);
            if total.is_finite() && total > 0.0 {
                let dp_ratio = (dp_reductions as f64) / total;
                let blended = (metrics.spiral_coherence + dp_ratio.clamp(0.0, 1.0)) * 0.5;
                metrics.spiral_coherence = blended;

                let fused_bias = if fused_ops.contains("wgpu") {
                    0.25
                } else if fused_ops.contains("par") {
                    0.18
                } else {
                    0.1
                };
                metrics.average_enrichment *= 1.0 + fused_bias * dp_ratio;

                if einsum.contains("argmax") {
                    metrics.mean_hardmass = metrics.mean_hardmass.max(dp_ratio);
                }
            }
        }

        let soft_tensor = Tensor::from_vec(rows, cols, soft)?;
        let hard_tensor = Tensor::from_vec(rows, cols, hard)?;
        let spiral_tensor = Tensor::from_vec(rows, cols, spiral)?;
        Ok(SpiralSoftmaxHardmax {
            softmax: soft_tensor,
            hardmax: hard_tensor,
            spiral: spiral_tensor,
            metrics,
        })
    }

    /// Layer normalisation over the last dimension (`cols`) with affine parameters.
    pub fn layer_norm_affine(
        &self,
        gamma: &Tensor,
        beta: &Tensor,
        epsilon: f32,
    ) -> PureResult<Tensor> {
        self.layer_norm_affine_with_backend(gamma, beta, epsilon, LayerNormBackend::Auto)
    }

    /// Layer normalisation with explicit backend control.
    pub fn layer_norm_affine_with_backend(
        &self,
        gamma: &Tensor,
        beta: &Tensor,
        epsilon: f32,
        backend: LayerNormBackend,
    ) -> PureResult<Tensor> {
        self.layer_norm_affine_add_impl(None, gamma, beta, epsilon, backend)
    }

    /// Layer normalisation over `self + residual` with affine parameters.
    pub fn layer_norm_affine_add(
        &self,
        residual: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        epsilon: f32,
    ) -> PureResult<Tensor> {
        self.layer_norm_affine_add_with_backend(
            residual,
            gamma,
            beta,
            epsilon,
            LayerNormBackend::Auto,
        )
    }

    /// Layer normalisation over `self + residual` with explicit backend control.
    pub fn layer_norm_affine_add_with_backend(
        &self,
        residual: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        epsilon: f32,
        backend: LayerNormBackend,
    ) -> PureResult<Tensor> {
        self.layer_norm_affine_add_impl(Some(residual), gamma, beta, epsilon, backend)
    }

    fn layer_norm_affine_add_impl(
        &self,
        residual: Option<&Tensor>,
        gamma: &Tensor,
        beta: &Tensor,
        epsilon: f32,
        backend: LayerNormBackend,
    ) -> PureResult<Tensor> {
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "layernorm_epsilon",
                value: epsilon,
            });
        }

        let (rows, cols) = self.shape();
        if cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        self.layout.expect_row_major("layer_norm input")?;

        if gamma.shape() != (1, cols) {
            return Err(TensorError::ShapeMismatch {
                left: gamma.shape(),
                right: (1, cols),
            });
        }
        if beta.shape() != (1, cols) {
            return Err(TensorError::ShapeMismatch {
                left: beta.shape(),
                right: (1, cols),
            });
        }
        gamma.layout.expect_row_major("layer_norm gamma")?;
        beta.layout.expect_row_major("layer_norm beta")?;
        Self::validate_finite_tensor_util_slice("layer_norm_input", self.data())?;
        Self::validate_finite_tensor_util_slice("layer_norm_gamma", gamma.data())?;
        Self::validate_finite_tensor_util_slice("layer_norm_beta", beta.data())?;

        let residual_data = if let Some(residual) = residual {
            if residual.shape() != (rows, cols) {
                return Err(TensorError::ShapeMismatch {
                    left: residual.shape(),
                    right: (rows, cols),
                });
            }
            residual.layout.expect_row_major("layer_norm residual")?;
            Self::validate_finite_tensor_util_slice("layer_norm_residual", residual.data())?;
            Some(residual.data())
        } else {
            None
        };

        if rows == 0 {
            let tensor = Tensor::zeros(rows, cols)?;
            crate::emit_tensor_op("layer_norm", &[rows, cols], &[rows, cols]);
            crate::emit_tensor_op_meta("layer_norm", || {
                serde_json::json!({
                    "backend": "cpu",
                    "requested_backend": backend.label(),
                    "rows": rows,
                    "cols": cols,
                    "epsilon": epsilon,
                    "flags": {
                        "use_residual": residual_data.is_some(),
                        "empty": true,
                    }
                })
            });
            return Ok(tensor);
        }

        let volume = rows
            .checked_mul(cols)
            .ok_or_else(|| TensorError::TensorVolumeExceeded {
                label: "layer_norm",
                volume: rows,
                max_volume: usize::MAX / cols.max(1),
            })?;

        let gamma_slice = gamma.data();
        let beta_slice = beta.data();

        let (tensor, backend_used): (Tensor, &'static str) = match backend {
            LayerNormBackend::Auto => {
                let prefer_wgpu = volume >= 4096;
                #[cfg(feature = "wgpu")]
                {
                    if prefer_wgpu
                        && wgpu_dense::is_available()
                        && wgpu_dense::supports_layer_norm(rows, cols)
                    {
                        match Self::layer_norm_wgpu(
                            self.data(),
                            residual_data,
                            gamma_slice,
                            beta_slice,
                            rows,
                            cols,
                            epsilon,
                        ) {
                            Ok(tensor) => (tensor, "wgpu_dense"),
                            Err(message) if strict_gpu_path() => {
                                return Err(strict_gpu_fallback_error(
                                    "wgpu",
                                    "layer_norm",
                                    message,
                                ));
                            }
                            Err(_) => (
                                self.layer_norm_cpu(
                                    residual_data,
                                    gamma_slice,
                                    beta_slice,
                                    epsilon,
                                )?,
                                "cpu",
                            ),
                        }
                    } else {
                        (
                            self.layer_norm_cpu(residual_data, gamma_slice, beta_slice, epsilon)?,
                            "cpu",
                        )
                    }
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    let _ = prefer_wgpu;
                    (
                        self.layer_norm_cpu(residual_data, gamma_slice, beta_slice, epsilon)?,
                        "cpu",
                    )
                }
            }
            LayerNormBackend::Cpu => (
                self.layer_norm_cpu(residual_data, gamma_slice, beta_slice, epsilon)?,
                "cpu",
            ),
            LayerNormBackend::GpuWgpu => {
                #[cfg(feature = "wgpu")]
                {
                    if !wgpu_dense::supports_layer_norm(rows, cols) {
                        return Err(TensorError::BackendFailure {
                            backend: "wgpu",
                            message: format!(
                                "layer_norm does not support shape {}x{} on the WGPU backend",
                                rows, cols
                            ),
                        });
                    }
                    let tensor = Self::layer_norm_wgpu(
                        self.data(),
                        residual_data,
                        gamma_slice,
                        beta_slice,
                        rows,
                        cols,
                        epsilon,
                    )
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "wgpu",
                        message,
                    })?;
                    (tensor, "wgpu_dense")
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    return Err(TensorError::BackendFailure {
                        backend: "wgpu",
                        message: "wgpu backend disabled at compile time".into(),
                    });
                }
            }
        };

        crate::emit_tensor_op("layer_norm", &[rows, cols], &[rows, cols]);
        if backend_used != "wgpu_dense" {
            crate::emit_tensor_op_meta("layer_norm", || {
                serde_json::json!({
                    "backend": backend_used,
                    "requested_backend": backend.label(),
                    "rows": rows,
                    "cols": cols,
                    "epsilon": epsilon,
                    "flags": {
                        "use_residual": residual_data.is_some(),
                    }
                })
            });
        }
        Ok(tensor)
    }

    #[cfg(feature = "wgpu")]
    fn layer_norm_wgpu(
        input: &[f32],
        residual: Option<&[f32]>,
        gamma: &[f32],
        beta: &[f32],
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) -> Result<Tensor, String> {
        let buffer = match residual {
            Some(residual_slice) => wgpu_dense::layer_norm_affine_add(
                input,
                residual_slice,
                gamma,
                beta,
                rows,
                cols,
                epsilon,
            ),
            None => wgpu_dense::layer_norm_affine(input, gamma, beta, rows, cols, epsilon),
        }?;
        Tensor::validate_finite_tensor_util_slice("layer_norm_output", buffer.as_slice())
            .map_err(|error| error.to_string())?;
        Tensor::from_vec(rows, cols, buffer).map_err(|error| error.to_string())
    }

    fn layer_norm_cpu(
        &self,
        residual: Option<&[f32]>,
        gamma: &[f32],
        beta: &[f32],
        epsilon: f32,
    ) -> PureResult<Tensor> {
        let (rows, cols) = self.shape();
        let cols_f = cols as f32;
        let mut output = vec![0.0f32; rows * cols];
        let input = self.data();

        for r in 0..rows {
            let offset = r * cols;
            let mut sum = 0.0f32;
            let mut sumsq = 0.0f32;
            for c in 0..cols {
                let idx = offset + c;
                let mut v = input[idx];
                if let Some(residual) = residual {
                    v += residual[idx];
                }
                Self::validate_finite_tensor_util_value("layer_norm_value", v)?;
                sum += v;
                Self::validate_finite_tensor_util_value("layer_norm_sum", sum)?;
                sumsq += v * v;
                Self::validate_finite_tensor_util_value("layer_norm_sumsq", sumsq)?;
            }
            let mean = sum / cols_f;
            Self::validate_finite_tensor_util_value("layer_norm_mean", mean)?;
            let raw_var = sumsq / cols_f - mean * mean;
            Self::validate_finite_tensor_util_value("layer_norm_variance", raw_var)?;
            let var = raw_var.max(0.0);
            let denom = (var + epsilon).sqrt();
            Self::validate_finite_tensor_util_value("layer_norm_denom", denom)?;
            for c in 0..cols {
                let idx = offset + c;
                let mut v = input[idx];
                if let Some(residual) = residual {
                    v += residual[idx];
                }
                Self::validate_finite_tensor_util_value("layer_norm_value", v)?;
                let normed = (v - mean) / denom;
                Self::validate_finite_tensor_util_value("layer_norm_normed", normed)?;
                output[idx] = normed * gamma[c] + beta[c];
                Self::validate_finite_tensor_util_value("layer_norm_output", output[idx])?;
            }
        }

        Tensor::from_vec(rows, cols, output)
    }

    /// Scaled dot-product attention using automatic backend selection.
    pub fn scaled_dot_attention(
        &self,
        keys: &Tensor,
        values: &Tensor,
        contexts: usize,
        sequence: usize,
        scale: f32,
    ) -> PureResult<Tensor> {
        self.scaled_dot_attention_with_backend(
            keys,
            values,
            contexts,
            sequence,
            scale,
            None,
            None,
            AttentionBackend::Auto,
        )
    }

    /// Scaled dot-product attention with optional biases and backend override.
    #[allow(clippy::too_many_arguments)]
    pub fn scaled_dot_attention_with_backend(
        &self,
        keys: &Tensor,
        values: &Tensor,
        contexts: usize,
        sequence: usize,
        scale: f32,
        z_bias: Option<&Tensor>,
        attn_bias: Option<&Tensor>,
        backend: AttentionBackend,
    ) -> PureResult<Tensor> {
        let expected_rows =
            contexts
                .checked_mul(sequence)
                .ok_or_else(|| TensorError::TensorVolumeExceeded {
                    label: "attention contexts*sequence",
                    volume: contexts,
                    max_volume: usize::MAX / sequence.max(1),
                })?;

        if self.rows != expected_rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: (expected_rows, self.cols),
            });
        }
        if keys.rows != expected_rows || keys.cols != self.cols {
            return Err(TensorError::ShapeMismatch {
                left: keys.shape(),
                right: (expected_rows, self.cols),
            });
        }
        if values.rows != expected_rows || values.cols != self.cols {
            return Err(TensorError::ShapeMismatch {
                left: values.shape(),
                right: (expected_rows, self.cols),
            });
        }

        let z_bias_slice = if let Some(bias) = z_bias {
            let (rows, cols) = bias.shape();
            if rows != contexts || cols != sequence {
                return Err(TensorError::ShapeMismatch {
                    left: bias.shape(),
                    right: (contexts, sequence),
                });
            }
            Some(bias.data())
        } else {
            None
        };

        let attn_bias_slice = if let Some(bias) = attn_bias {
            let (rows, cols) = bias.shape();
            if rows != expected_rows || cols != sequence {
                return Err(TensorError::ShapeMismatch {
                    left: bias.shape(),
                    right: (expected_rows, sequence),
                });
            }
            Some(bias.data())
        } else {
            None
        };

        Self::validate_scale_factor("scaled_dot_attention_scale", scale)?;
        Self::validate_finite_tensor_util_slice("scaled_dot_attention_query", self.data())?;
        Self::validate_finite_tensor_util_slice("scaled_dot_attention_key", keys.data())?;
        Self::validate_finite_tensor_util_slice("scaled_dot_attention_value", values.data())?;
        if let Some(bias) = z_bias_slice {
            Self::validate_finite_tensor_util_slice("scaled_dot_attention_z_bias", bias)?;
        }
        if let Some(bias) = attn_bias_slice {
            Self::validate_finite_tensor_util_slice("scaled_dot_attention_attn_bias", bias)?;
        }

        let head_dim = self.cols;
        if expected_rows == 0 || head_dim == 0 {
            let tensor = Tensor::zeros(expected_rows, head_dim)?;
            crate::emit_tensor_op(
                "scaled_dot_attention",
                &[expected_rows, head_dim],
                &[expected_rows, head_dim],
            );
            crate::emit_tensor_op_meta("scaled_dot_attention", || {
                serde_json::json!({
                    "backend": "cpu",
                    "requested_backend": backend.label(),
                    "contexts": contexts,
                    "sequence": sequence,
                    "head_dim": head_dim,
                    "scale": scale,
                    "flags": {
                        "use_z_bias": z_bias_slice.is_some(),
                        "use_attn_bias": attn_bias_slice.is_some(),
                        "empty": true,
                    }
                })
            });
            return Ok(tensor);
        }
        let queries = self.data();
        let keys_data = keys.data();
        let values_data = values.data();

        let make_tensor = |buffer: Vec<f32>| {
            Self::validate_finite_tensor_util_slice(
                "scaled_dot_attention_output",
                buffer.as_slice(),
            )?;
            Tensor::from_vec(expected_rows, head_dim, buffer)
        };

        let (tensor, backend_used): (Tensor, &'static str) = match backend {
            AttentionBackend::Auto => {
                let wgpu_tensor: Option<Tensor> = {
                    #[cfg(feature = "wgpu")]
                    {
                        if wgpu_dense::is_available()
                            && wgpu_dense::supports_fused_attention(contexts, sequence, head_dim)
                        {
                            match wgpu_dense::fused_attention(
                                queries,
                                keys_data,
                                values_data,
                                contexts,
                                sequence,
                                head_dim,
                                scale,
                                z_bias_slice,
                                attn_bias_slice,
                            ) {
                                Ok(buffer) => Some(make_tensor(buffer)?),
                                Err(message) if strict_gpu_path() => {
                                    return Err(strict_gpu_fallback_error(
                                        "wgpu",
                                        "scaled_dot_attention",
                                        message,
                                    ));
                                }
                                Err(_) => None,
                            }
                        } else {
                            None
                        }
                    }
                    #[cfg(not(feature = "wgpu"))]
                    {
                        None
                    }
                };

                if let Some(tensor) = wgpu_tensor {
                    Ok((tensor, "wgpu_dense"))
                } else {
                    let buffer = fused_attention_cpu(
                        queries,
                        keys_data,
                        values_data,
                        contexts,
                        sequence,
                        head_dim,
                        scale,
                        z_bias_slice,
                        attn_bias_slice,
                    )?;
                    make_tensor(buffer).map(|tensor| (tensor, "cpu"))
                }
            }
            AttentionBackend::Cpu => {
                let buffer = fused_attention_cpu(
                    queries,
                    keys_data,
                    values_data,
                    contexts,
                    sequence,
                    head_dim,
                    scale,
                    z_bias_slice,
                    attn_bias_slice,
                )?;
                make_tensor(buffer).map(|tensor| (tensor, "cpu"))
            }
            #[cfg(feature = "wgpu")]
            AttentionBackend::GpuWgpu => {
                let data = wgpu_dense::fused_attention(
                    queries,
                    keys_data,
                    values_data,
                    contexts,
                    sequence,
                    head_dim,
                    scale,
                    z_bias_slice,
                    attn_bias_slice,
                )
                .map_err(|message| TensorError::BackendFailure {
                    backend: "wgpu",
                    message,
                })?;
                make_tensor(data).map(|tensor| (tensor, "wgpu_dense"))
            }
            #[cfg(not(feature = "wgpu"))]
            AttentionBackend::GpuWgpu => Err(TensorError::BackendFailure {
                backend: "wgpu",
                message: "wgpu backend disabled at compile time".into(),
            }),
        }?;

        crate::emit_tensor_op(
            "scaled_dot_attention",
            &[expected_rows, head_dim],
            &[expected_rows, head_dim],
        );
        if backend_used != "wgpu_dense" {
            crate::emit_tensor_op_meta("scaled_dot_attention", || {
                serde_json::json!({
                    "backend": backend_used,
                    "requested_backend": backend.label(),
                    "contexts": contexts,
                    "sequence": sequence,
                    "head_dim": head_dim,
                    "scale": scale,
                    "flags": {
                        "use_z_bias": z_bias_slice.is_some(),
                        "use_attn_bias": attn_bias_slice.is_some(),
                    }
                })
            });
        }
        Ok(tensor)
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
        let data = wgpu_dense::matmul(self.data(), other.data(), self.rows, self.cols, other.cols)
            .map_err(|message| TensorError::BackendFailure {
                backend: "wgpu",
                message,
            })?;
        Tensor::from_vec(self.rows, other.cols, data)
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_fused_matmul_meta(
        &self,
        op_name: &'static str,
        other: &Tensor,
        requested_backend: MatmulBackend,
        backend_used: &'static str,
        rows: usize,
        inner: usize,
        cols: usize,
    ) {
        crate::emit_tensor_op(
            op_name,
            &[self.rows, self.cols, other.rows, other.cols],
            &[rows, cols],
        );
        crate::emit_tensor_op_meta(op_name, || {
            serde_json::json!({
                "backend": backend_used,
                "requested_backend": requested_backend.label(),
                "rows": rows,
                "inner": inner,
                "cols": cols,
                "lhs_layout": self.layout.as_str(),
                "rhs_layout": other.layout.as_str(),
            })
        });
    }

    /// Matrix multiply followed by bias addition and ReLU activation.
    pub fn matmul_bias_relu(&self, other: &Tensor, bias: &[f32]) -> PureResult<Tensor> {
        self.matmul_bias_relu_with_backend(other, bias, MatmulBackend::Auto)
    }

    /// Matrix multiply followed by bias addition and ReLU activation with explicit backend control.
    pub fn matmul_bias_relu_with_backend(
        &self,
        other: &Tensor,
        bias: &[f32],
        backend: MatmulBackend,
    ) -> PureResult<Tensor> {
        if self.cols != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        if other.cols != bias.len() {
            return Err(TensorError::DataLength {
                expected: other.cols,
                got: bias.len(),
            });
        }

        let rows = self.rows;
        let cols = other.cols;
        let mut tensor = Tensor::zeros(rows, cols)?;
        self.matmul_bias_relu_into_with_backend(other, bias, &mut tensor, backend)?;
        Ok(tensor)
    }

    pub fn matmul_bias_relu_into_with_backend(
        &self,
        other: &Tensor,
        bias: &[f32],
        dst: &mut Tensor,
        backend: MatmulBackend,
    ) -> PureResult<()> {
        if self.cols != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        if dst.rows != self.rows || dst.cols != other.cols {
            return Err(TensorError::ShapeMismatch {
                left: (self.rows, other.cols),
                right: dst.shape(),
            });
        }
        if other.cols != bias.len() {
            return Err(TensorError::DataLength {
                expected: other.cols,
                got: bias.len(),
            });
        }
        Self::validate_finite_tensor_util_slice("matmul_bias_relu_bias", bias)?;
        if Arc::ptr_eq(&self.data, &dst.data) || Arc::ptr_eq(&other.data, &dst.data) {
            return Err(TensorError::InvalidValue {
                label: "matmul_out_alias",
            });
        }

        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;
        let mut scratch = aligned_zeroed(rows * cols);
        let work_slice = scratch.as_mut_slice();

        let backend_used = match backend {
            MatmulBackend::Auto => {
                self.matmul_bias_relu_into_auto(other, bias, work_slice, rows, inner, cols)?
            }
            MatmulBackend::CpuSimd => {
                cpu_dense::matmul_into(work_slice, self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "cpu_simd",
                        message,
                    })?;
                add_bias_relu_inplace(work_slice, rows, cols, bias);
                "cpu_simd"
            }
            MatmulBackend::CpuNaive => {
                matmul_naive_into(work_slice, self.data(), other.data(), rows, inner, cols);
                add_bias_relu_inplace(work_slice, rows, cols, bias);
                "naive"
            }
            MatmulBackend::CpuFaer => {
                faer_dense::matmul_into(work_slice, self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                    backend: "faer",
                    message,
                })?;
                add_bias_relu_inplace(work_slice, rows, cols, bias);
                "faer"
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => {
                let data = wgpu_dense::matmul_bias_relu(
                    self.data(),
                    other.data(),
                    bias,
                    rows,
                    inner,
                    cols,
                )
                .map_err(|message| TensorError::BackendFailure {
                    backend: "wgpu",
                    message,
                })?;
                work_slice.copy_from_slice(&data);
                "wgpu"
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                hip_dense::matmul_into(self.data(), other.data(), work_slice, rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "hip",
                        message,
                    })?;
                add_bias_relu_inplace(work_slice, rows, cols, bias);
                "hip"
            }
        };
        Self::validate_finite_tensor_util_slice("matmul_bias_relu_output", scratch.as_slice())?;
        dst.data_mut().copy_from_slice(scratch.as_slice());

        self.emit_fused_matmul_meta(
            "matmul_bias_relu",
            other,
            backend,
            backend_used,
            rows,
            inner,
            cols,
        );
        Ok(())
    }

    fn matmul_bias_relu_into_auto(
        &self,
        other: &Tensor,
        bias: &[f32],
        dst: &mut [f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<&'static str> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                match wgpu_dense::matmul_bias_relu(
                    self.data(),
                    other.data(),
                    bias,
                    rows,
                    inner,
                    cols,
                ) {
                    Ok(buffer) => {
                        dst.copy_from_slice(&buffer);
                        return Ok("wgpu");
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "matmul_bias_relu",
                            message,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        #[cfg(feature = "hip")]
        {
            if hip_dense::is_available() && hip_dense::should_use(rows, inner, cols) {
                match hip_dense::matmul_into(self.data(), other.data(), dst, rows, inner, cols) {
                    Ok(()) => {
                        add_bias_relu_inplace(dst, rows, cols, bias);
                        return Ok("hip");
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "hip",
                            "matmul_bias_relu",
                            message,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        if cpu_dense::should_use(rows, inner, cols) {
            if let Ok(()) =
                cpu_dense::matmul_into(dst, self.data(), other.data(), rows, inner, cols)
            {
                add_bias_relu_inplace(dst, rows, cols, bias);
                return Ok("cpu_simd");
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            if let Ok(()) =
                faer_dense::matmul_into(dst, self.data(), other.data(), rows, inner, cols)
            {
                add_bias_relu_inplace(dst, rows, cols, bias);
                return Ok("faer");
            }
        }

        matmul_naive_into(dst, self.data(), other.data(), rows, inner, cols);
        add_bias_relu_inplace(dst, rows, cols, bias);
        Ok("naive")
    }

    /// Matrix multiply followed by bias addition and GELU activation.
    pub fn matmul_bias_gelu(&self, other: &Tensor, bias: &[f32]) -> PureResult<Tensor> {
        self.matmul_bias_gelu_with_backend(other, bias, MatmulBackend::Auto)
    }

    /// Matrix multiply followed by bias addition and GELU activation with explicit backend control.
    pub fn matmul_bias_gelu_with_backend(
        &self,
        other: &Tensor,
        bias: &[f32],
        backend: MatmulBackend,
    ) -> PureResult<Tensor> {
        if self.cols != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        if other.cols != bias.len() {
            return Err(TensorError::DataLength {
                expected: other.cols,
                got: bias.len(),
            });
        }
        Self::validate_finite_tensor_util_slice("matmul_bias_gelu_bias", bias)?;

        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;

        let (data, backend_used) = match backend {
            MatmulBackend::Auto => self.matmul_bias_gelu_auto(other, bias, rows, inner, cols)?,
            MatmulBackend::CpuSimd => {
                let mut buffer = vec![0.0; rows * cols];
                cpu_dense::matmul_into(&mut buffer, self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                    backend: "cpu_simd",
                    message,
                })?;
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                (buffer, "cpu_simd")
            }
            MatmulBackend::CpuNaive => {
                let mut buffer = matmul_naive(self.data(), other.data(), rows, inner, cols);
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                (buffer, "naive")
            }
            MatmulBackend::CpuFaer => {
                let mut buffer = faer_dense::matmul(self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                    backend: "faer",
                    message,
                })?;
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                (buffer, "faer")
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => {
                let buffer = wgpu_dense::matmul_bias_gelu(
                    self.data(),
                    other.data(),
                    bias,
                    rows,
                    inner,
                    cols,
                )
                .map_err(|message| TensorError::BackendFailure {
                    backend: "wgpu",
                    message,
                })?;
                (buffer, "wgpu")
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                let mut buffer = hip_dense::matmul(self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "hip",
                        message,
                    })?;
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                (buffer, "hip")
            }
        };

        Self::validate_finite_tensor_util_slice("matmul_bias_gelu_output", data.as_slice())?;
        let tensor = Tensor::from_vec(rows, cols, data)?;
        self.emit_fused_matmul_meta(
            "matmul_bias_gelu",
            other,
            backend,
            backend_used,
            rows,
            inner,
            cols,
        );
        Ok(tensor)
    }

    fn matmul_bias_gelu_auto(
        &self,
        other: &Tensor,
        bias: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<(Vec<f32>, &'static str)> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                match wgpu_dense::matmul_bias_gelu(
                    self.data(),
                    other.data(),
                    bias,
                    rows,
                    inner,
                    cols,
                ) {
                    Ok(buffer) => return Ok((buffer, "wgpu")),
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "matmul_bias_gelu",
                            message,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        #[cfg(feature = "hip")]
        {
            if hip_dense::is_available() && hip_dense::should_use(rows, inner, cols) {
                match hip_dense::matmul(self.data(), other.data(), rows, inner, cols) {
                    Ok(mut buffer) => {
                        add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                        return Ok((buffer, "hip"));
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "hip",
                            "matmul_bias_gelu",
                            message,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        if cpu_dense::should_use(rows, inner, cols) {
            let mut buffer = vec![0.0; rows * cols];
            if cpu_dense::matmul_into(&mut buffer, self.data(), other.data(), rows, inner, cols)
                .is_ok()
            {
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                return Ok((buffer, "cpu_simd"));
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            if let Ok(mut buffer) = faer_dense::matmul(self.data(), other.data(), rows, inner, cols)
            {
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                return Ok((buffer, "faer"));
            }
        }

        let mut buffer = matmul_naive(self.data(), other.data(), rows, inner, cols);
        add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
        Ok((buffer, "naive"))
    }

    /// Matrix multiply with bias, residual addition, and ReLU activation.
    pub fn matmul_bias_add_relu(
        &self,
        other: &Tensor,
        bias: &[f32],
        residual: &Tensor,
    ) -> PureResult<Tensor> {
        self.matmul_bias_add_relu_with_backend(other, bias, residual, MatmulBackend::Auto)
    }

    /// Matrix multiply with bias, residual addition, and ReLU activation with explicit backend control.
    pub fn matmul_bias_add_relu_with_backend(
        &self,
        other: &Tensor,
        bias: &[f32],
        residual: &Tensor,
        backend: MatmulBackend,
    ) -> PureResult<Tensor> {
        if self.cols != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        if other.cols != bias.len() {
            return Err(TensorError::DataLength {
                expected: other.cols,
                got: bias.len(),
            });
        }
        if residual.shape() != (self.rows, other.cols) {
            return Err(TensorError::ShapeMismatch {
                left: residual.shape(),
                right: (self.rows, other.cols),
            });
        }

        let rows = self.rows;
        let cols = other.cols;
        let mut tensor = Tensor::zeros(rows, cols)?;
        self.matmul_bias_add_relu_into_with_backend(other, bias, residual, &mut tensor, backend)?;
        Ok(tensor)
    }

    pub fn matmul_bias_add_relu_into_with_backend(
        &self,
        other: &Tensor,
        bias: &[f32],
        residual: &Tensor,
        dst: &mut Tensor,
        backend: MatmulBackend,
    ) -> PureResult<()> {
        if self.cols != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        if dst.rows != self.rows || dst.cols != other.cols {
            return Err(TensorError::ShapeMismatch {
                left: (self.rows, other.cols),
                right: dst.shape(),
            });
        }
        if other.cols != bias.len() {
            return Err(TensorError::DataLength {
                expected: other.cols,
                got: bias.len(),
            });
        }
        Self::validate_finite_tensor_util_slice("matmul_bias_add_relu_bias", bias)?;
        if residual.shape() != (self.rows, other.cols) {
            return Err(TensorError::ShapeMismatch {
                left: residual.shape(),
                right: (self.rows, other.cols),
            });
        }
        Self::validate_finite_tensor_util_slice("matmul_bias_add_relu_residual", residual.data())?;
        if Arc::ptr_eq(&self.data, &dst.data)
            || Arc::ptr_eq(&other.data, &dst.data)
            || Arc::ptr_eq(&residual.data, &dst.data)
        {
            return Err(TensorError::InvalidValue {
                label: "matmul_out_alias",
            });
        }

        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;
        let mut scratch = aligned_zeroed(rows * cols);
        let work_slice = scratch.as_mut_slice();

        let backend_used = match backend {
            MatmulBackend::Auto => self.matmul_bias_add_relu_into_auto(
                other, bias, residual, work_slice, rows, inner, cols,
            )?,
            MatmulBackend::CpuSimd => {
                cpu_dense::matmul_into(work_slice, self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "cpu_simd",
                        message,
                    })?;
                add_bias_residual_relu_inplace(work_slice, rows, cols, bias, residual.data());
                "cpu_simd"
            }
            MatmulBackend::CpuNaive => {
                matmul_naive_into(work_slice, self.data(), other.data(), rows, inner, cols);
                add_bias_residual_relu_inplace(work_slice, rows, cols, bias, residual.data());
                "naive"
            }
            MatmulBackend::CpuFaer => {
                faer_dense::matmul_into(work_slice, self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                    backend: "faer",
                    message,
                })?;
                add_bias_residual_relu_inplace(work_slice, rows, cols, bias, residual.data());
                "faer"
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => {
                let data = wgpu_dense::matmul_bias_add_relu(
                    self.data(),
                    other.data(),
                    bias,
                    residual.data(),
                    rows,
                    inner,
                    cols,
                )
                .map_err(|message| TensorError::BackendFailure {
                    backend: "wgpu",
                    message,
                })?;
                work_slice.copy_from_slice(&data);
                "wgpu"
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                hip_dense::matmul_into(self.data(), other.data(), work_slice, rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "hip",
                        message,
                    })?;
                add_bias_residual_relu_inplace(work_slice, rows, cols, bias, residual.data());
                "hip"
            }
        };
        Self::validate_finite_tensor_util_slice("matmul_bias_add_relu_output", scratch.as_slice())?;
        dst.data_mut().copy_from_slice(scratch.as_slice());

        self.emit_fused_matmul_meta(
            "matmul_bias_add_relu",
            other,
            backend,
            backend_used,
            rows,
            inner,
            cols,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_bias_add_relu_into_auto(
        &self,
        other: &Tensor,
        bias: &[f32],
        residual: &Tensor,
        dst: &mut [f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<&'static str> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                match wgpu_dense::matmul_bias_add_relu(
                    self.data(),
                    other.data(),
                    bias,
                    residual.data(),
                    rows,
                    inner,
                    cols,
                ) {
                    Ok(buffer) => {
                        dst.copy_from_slice(&buffer);
                        return Ok("wgpu");
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "matmul_bias_add_relu",
                            message,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        #[cfg(feature = "hip")]
        {
            if hip_dense::is_available() && hip_dense::should_use(rows, inner, cols) {
                match hip_dense::matmul_into(self.data(), other.data(), dst, rows, inner, cols) {
                    Ok(()) => {
                        add_bias_residual_relu_inplace(dst, rows, cols, bias, residual.data());
                        return Ok("hip");
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "hip",
                            "matmul_bias_add_relu",
                            message,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        if cpu_dense::should_use(rows, inner, cols) {
            if let Ok(()) =
                cpu_dense::matmul_into(dst, self.data(), other.data(), rows, inner, cols)
            {
                add_bias_residual_relu_inplace(dst, rows, cols, bias, residual.data());
                return Ok("cpu_simd");
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            if let Ok(()) =
                faer_dense::matmul_into(dst, self.data(), other.data(), rows, inner, cols)
            {
                add_bias_residual_relu_inplace(dst, rows, cols, bias, residual.data());
                return Ok("faer");
            }
        }

        matmul_naive_into(dst, self.data(), other.data(), rows, inner, cols);
        add_bias_residual_relu_inplace(dst, rows, cols, bias, residual.data());
        Ok("naive")
    }

    /// Matrix multiply with bias, residual addition, and GELU activation.
    pub fn matmul_bias_add_gelu(
        &self,
        other: &Tensor,
        bias: &[f32],
        residual: &Tensor,
    ) -> PureResult<Tensor> {
        self.matmul_bias_add_gelu_with_backend(other, bias, residual, MatmulBackend::Auto)
    }

    /// Matrix multiply with bias, residual addition, and GELU activation with explicit backend control.
    pub fn matmul_bias_add_gelu_with_backend(
        &self,
        other: &Tensor,
        bias: &[f32],
        residual: &Tensor,
        backend: MatmulBackend,
    ) -> PureResult<Tensor> {
        if self.cols != other.rows {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        if other.cols != bias.len() {
            return Err(TensorError::DataLength {
                expected: other.cols,
                got: bias.len(),
            });
        }
        Self::validate_finite_tensor_util_slice("matmul_bias_add_gelu_bias", bias)?;
        if residual.shape() != (self.rows, other.cols) {
            return Err(TensorError::ShapeMismatch {
                left: residual.shape(),
                right: (self.rows, other.cols),
            });
        }
        Self::validate_finite_tensor_util_slice("matmul_bias_add_gelu_residual", residual.data())?;

        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;

        let (data, backend_used) = match backend {
            MatmulBackend::Auto => {
                self.matmul_bias_add_gelu_auto(other, bias, residual, rows, inner, cols)?
            }
            MatmulBackend::CpuSimd => {
                let mut buffer = vec![0.0; rows * cols];
                cpu_dense::matmul_into(&mut buffer, self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                    backend: "cpu_simd",
                    message,
                })?;
                add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                (buffer, "cpu_simd")
            }
            MatmulBackend::CpuNaive => {
                let mut buffer = matmul_naive(self.data(), other.data(), rows, inner, cols);
                add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                (buffer, "naive")
            }
            MatmulBackend::CpuFaer => {
                let mut buffer = faer_dense::matmul(self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                    backend: "faer",
                    message,
                })?;
                add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                (buffer, "faer")
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => {
                let buffer = wgpu_dense::matmul_bias_add_gelu(
                    self.data(),
                    other.data(),
                    bias,
                    residual.data(),
                    rows,
                    inner,
                    cols,
                )
                .map_err(|message| TensorError::BackendFailure {
                    backend: "wgpu",
                    message,
                })?;
                (buffer, "wgpu")
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                let mut buffer = hip_dense::matmul(self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "hip",
                        message,
                    })?;
                add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                (buffer, "hip")
            }
        };

        Self::validate_finite_tensor_util_slice("matmul_bias_add_gelu_output", data.as_slice())?;
        let tensor = Tensor::from_vec(rows, cols, data)?;
        self.emit_fused_matmul_meta(
            "matmul_bias_add_gelu",
            other,
            backend,
            backend_used,
            rows,
            inner,
            cols,
        );
        Ok(tensor)
    }

    fn matmul_bias_add_gelu_auto(
        &self,
        other: &Tensor,
        bias: &[f32],
        residual: &Tensor,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<(Vec<f32>, &'static str)> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                match wgpu_dense::matmul_bias_add_gelu(
                    self.data(),
                    other.data(),
                    bias,
                    residual.data(),
                    rows,
                    inner,
                    cols,
                ) {
                    Ok(buffer) => return Ok((buffer, "wgpu")),
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "matmul_bias_add_gelu",
                            message,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        #[cfg(feature = "hip")]
        {
            if hip_dense::is_available() && hip_dense::should_use(rows, inner, cols) {
                match hip_dense::matmul(self.data(), other.data(), rows, inner, cols) {
                    Ok(mut buffer) => {
                        add_bias_residual_gelu_inplace(
                            &mut buffer,
                            rows,
                            cols,
                            bias,
                            residual.data(),
                        );
                        return Ok((buffer, "hip"));
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "hip",
                            "matmul_bias_add_gelu",
                            message,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        if cpu_dense::should_use(rows, inner, cols) {
            let mut buffer = vec![0.0; rows * cols];
            if cpu_dense::matmul_into(&mut buffer, self.data(), other.data(), rows, inner, cols)
                .is_ok()
            {
                add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                return Ok((buffer, "cpu_simd"));
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            if let Ok(mut buffer) = faer_dense::matmul(self.data(), other.data(), rows, inner, cols)
            {
                add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                return Ok((buffer, "faer"));
            }
        }

        let mut buffer = matmul_naive(self.data(), other.data(), rows, inner, cols);
        add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
        Ok((buffer, "naive"))
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Tensor) -> PureResult<Tensor> {
        self.add_with_backend(other, TensorUtilBackend::Auto)
    }

    /// Element-wise addition with an explicit utility backend selection.
    pub fn add_with_backend(
        &self,
        other: &Tensor,
        _backend: TensorUtilBackend,
    ) -> PureResult<Tensor> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::add(self.data(), other.data(), self.rows, self.cols) {
                    Ok(buffer) => {
                        for &output in &buffer {
                            Self::validate_finite_tensor_util_value("add_output", output)?;
                        }
                        let output = Tensor::from_vec(self.rows, self.cols, buffer)?;
                        crate::emit_tensor_op(
                            "add",
                            &[self.rows, self.cols, other.rows, other.cols],
                            &[self.rows, self.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "add",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            self.rows,
                            self.cols,
                            self.layout,
                            "elementwise",
                            "tensor_util.add",
                            |data| {
                                data.insert("rhs_rows".to_string(), serde_json::json!(other.rows));
                                data.insert("rhs_cols".to_string(), serde_json::json!(other.cols));
                                data.insert(
                                    "rhs_layout".to_string(),
                                    serde_json::json!(other.layout.as_str()),
                                );
                            },
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "add", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut data = aligned_with_capacity(self.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            let output = a + b;
            Self::validate_finite_tensor_util_value("add_output", output)?;
            data.push(output);
        }
        let output = Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)?;
        crate::emit_tensor_op(
            "add",
            &[self.rows, self.cols, other.rows, other.cols],
            &[self.rows, self.cols],
        );
        emit_tensor_util_cpu_op_meta(
            "add",
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.layout,
            _backend.label(),
            "elementwise",
            |data| {
                data.insert("rhs_rows".to_string(), serde_json::json!(other.rows));
                data.insert("rhs_cols".to_string(), serde_json::json!(other.cols));
                data.insert(
                    "rhs_layout".to_string(),
                    serde_json::json!(other.layout.as_str()),
                );
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(output)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> PureResult<Tensor> {
        self.sub_with_backend(other, TensorUtilBackend::Auto)
    }

    /// Element-wise subtraction with an explicit utility backend selection.
    pub fn sub_with_backend(
        &self,
        other: &Tensor,
        _backend: TensorUtilBackend,
    ) -> PureResult<Tensor> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::sub(self.data(), other.data(), self.rows, self.cols) {
                    Ok(buffer) => {
                        for &output in &buffer {
                            Self::validate_finite_tensor_util_value("sub_output", output)?;
                        }
                        let output = Tensor::from_vec(self.rows, self.cols, buffer)?;
                        crate::emit_tensor_op(
                            "sub",
                            &[self.rows, self.cols, other.rows, other.cols],
                            &[self.rows, self.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "sub",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            self.rows,
                            self.cols,
                            self.layout,
                            "elementwise",
                            "tensor_util.sub",
                            |data| {
                                data.insert("rhs_rows".to_string(), serde_json::json!(other.rows));
                                data.insert("rhs_cols".to_string(), serde_json::json!(other.cols));
                                data.insert(
                                    "rhs_layout".to_string(),
                                    serde_json::json!(other.layout.as_str()),
                                );
                            },
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "sub", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut data = aligned_with_capacity(self.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            let output = a - b;
            Self::validate_finite_tensor_util_value("sub_output", output)?;
            data.push(output);
        }
        let output = Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)?;
        crate::emit_tensor_op(
            "sub",
            &[self.rows, self.cols, other.rows, other.cols],
            &[self.rows, self.cols],
        );
        emit_tensor_util_cpu_op_meta(
            "sub",
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.layout,
            _backend.label(),
            "elementwise",
            |data| {
                data.insert("rhs_rows".to_string(), serde_json::json!(other.rows));
                data.insert("rhs_cols".to_string(), serde_json::json!(other.cols));
                data.insert(
                    "rhs_layout".to_string(),
                    serde_json::json!(other.layout.as_str()),
                );
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(output)
    }

    /// Returns a new tensor where every element is scaled by `value`.
    pub fn scale(&self, value: f32) -> PureResult<Tensor> {
        self.scale_with_backend(value, TensorUtilBackend::Auto)
    }

    fn validate_finite_tensor_util_value(label: &'static str, value: f32) -> PureResult<()> {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue { label, value });
        }
        Ok(())
    }

    fn validate_finite_tensor_util_slice(label: &'static str, values: &[f32]) -> PureResult<()> {
        for &value in values {
            Self::validate_finite_tensor_util_value(label, value)?;
        }
        Ok(())
    }

    fn validate_scale_factor(label: &'static str, scale: f32) -> PureResult<()> {
        if !scale.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label,
                value: scale,
            });
        }
        Ok(())
    }

    fn checked_scale_output(input: f32, scale: f32) -> PureResult<f32> {
        Self::validate_scale_factor("scale_factor", scale)?;
        let output = input * scale;
        Self::validate_finite_tensor_util_value("scale_output", output)?;
        Ok(output)
    }

    fn validate_add_scaled_outputs(lhs: &[f32], rhs: &[f32], scale: f32) -> PureResult<()> {
        Self::validate_scale_factor("add_scaled_factor", scale)?;
        for (&a, &b) in lhs.iter().zip(rhs.iter()) {
            let delta = scale * b;
            Self::validate_finite_tensor_util_value("add_scaled_delta", delta)?;
            let output = a + delta;
            Self::validate_finite_tensor_util_value("add_scaled_output", output)?;
        }
        Ok(())
    }

    fn validate_add_row_outputs(
        rows: usize,
        cols: usize,
        lhs: &[f32],
        bias: &[f32],
    ) -> PureResult<()> {
        for &value in bias {
            Self::validate_finite_tensor_util_value("add_row_bias", value)?;
        }
        for row in 0..rows {
            let offset = row * cols;
            for col in 0..cols {
                let output = lhs[offset + col] + bias[col];
                Self::validate_finite_tensor_util_value("add_row_output", output)?;
            }
        }
        Ok(())
    }

    fn validate_mul_row_outputs(
        rows: usize,
        cols: usize,
        lhs: &[f32],
        row: &[f32],
    ) -> PureResult<()> {
        for &value in row {
            Self::validate_finite_tensor_util_value("mul_row_rhs", value)?;
        }
        for row_idx in 0..rows {
            let offset = row_idx * cols;
            for col in 0..cols {
                let output = lhs[offset + col] * row[col];
                Self::validate_finite_tensor_util_value("mul_row_output", output)?;
            }
        }
        Ok(())
    }

    fn validate_row_affine_outputs(
        rows: usize,
        cols: usize,
        lhs: &[f32],
        scale_row: &[f32],
        bias_row: &[f32],
    ) -> PureResult<()> {
        for &value in scale_row {
            Self::validate_finite_tensor_util_value("row_affine_scale", value)?;
        }
        for &value in bias_row {
            Self::validate_finite_tensor_util_value("row_affine_bias", value)?;
        }
        for row_idx in 0..rows {
            let offset = row_idx * cols;
            for col in 0..cols {
                let output = lhs[offset + col] * scale_row[col] + bias_row[col];
                Self::validate_finite_tensor_util_value("row_affine_output", output)?;
            }
        }
        Ok(())
    }

    /// Returns a scaled tensor with an explicit utility backend selection.
    pub fn scale_with_backend(
        &self,
        value: f32,
        _backend: TensorUtilBackend,
    ) -> PureResult<Tensor> {
        Self::validate_scale_factor("scale_factor", value)?;
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::scale(self.data(), self.rows, self.cols, value) {
                    Ok(buffer) => {
                        for &output in &buffer {
                            Self::validate_finite_tensor_util_value("scale_output", output)?;
                        }
                        let output = Tensor::from_vec(self.rows, self.cols, buffer)?;
                        crate::emit_tensor_op(
                            "scale",
                            &[self.rows, self.cols],
                            &[self.rows, self.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "scale",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            self.rows,
                            self.cols,
                            self.layout,
                            "elementwise",
                            "tensor_util.scale",
                            |data| {
                                data.insert("scale".to_string(), serde_json::json!(value));
                            },
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "scale", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut data = aligned_with_capacity(self.len());
        for &a in self.data.iter() {
            data.push(Self::checked_scale_output(a, value)?);
        }
        let output = Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)?;
        crate::emit_tensor_op("scale", &[self.rows, self.cols], &[self.rows, self.cols]);
        emit_tensor_util_cpu_op_meta(
            "scale",
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.layout,
            _backend.label(),
            "elementwise",
            |data| {
                data.insert("scale".to_string(), serde_json::json!(value));
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(output)
    }

    /// Element-wise product (Hadamard) between two tensors of identical shape.
    pub fn hadamard(&self, other: &Tensor) -> PureResult<Tensor> {
        self.hadamard_with_backend(other, TensorUtilBackend::Auto)
    }

    /// Element-wise product with an explicit utility backend selection.
    pub fn hadamard_with_backend(
        &self,
        other: &Tensor,
        _backend: TensorUtilBackend,
    ) -> PureResult<Tensor> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::hadamard(self.data(), other.data(), self.rows, self.cols) {
                    Ok(buffer) => {
                        for &output in &buffer {
                            Self::validate_finite_tensor_util_value("hadamard_output", output)?;
                        }
                        let output = Tensor::from_vec(self.rows, self.cols, buffer)?;
                        crate::emit_tensor_op(
                            "hadamard",
                            &[self.rows, self.cols, other.rows, other.cols],
                            &[self.rows, self.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "hadamard",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            self.rows,
                            self.cols,
                            self.layout,
                            "elementwise",
                            "tensor_util.hadamard",
                            |data| {
                                data.insert("rhs_rows".to_string(), serde_json::json!(other.rows));
                                data.insert("rhs_cols".to_string(), serde_json::json!(other.cols));
                                data.insert(
                                    "rhs_layout".to_string(),
                                    serde_json::json!(other.layout.as_str()),
                                );
                            },
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "hadamard", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut data = aligned_with_capacity(self.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            let output = a * b;
            Self::validate_finite_tensor_util_value("hadamard_output", output)?;
            data.push(output);
        }
        let output = Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)?;
        crate::emit_tensor_op(
            "hadamard",
            &[self.rows, self.cols, other.rows, other.cols],
            &[self.rows, self.cols],
        );
        emit_tensor_util_cpu_op_meta(
            "hadamard",
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.layout,
            _backend.label(),
            "elementwise",
            |data| {
                data.insert("rhs_rows".to_string(), serde_json::json!(other.rows));
                data.insert("rhs_cols".to_string(), serde_json::json!(other.cols));
                data.insert(
                    "rhs_layout".to_string(),
                    serde_json::json!(other.layout.as_str()),
                );
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(output)
    }

    /// Multiplies every tensor row by a row vector.
    pub fn mul_row(&self, row: &[f32]) -> PureResult<Tensor> {
        self.mul_row_with_backend(row, TensorUtilBackend::Auto)
    }

    /// Multiplies every tensor row by a row vector with an explicit utility backend.
    pub fn mul_row_with_backend(
        &self,
        row: &[f32],
        _backend: TensorUtilBackend,
    ) -> PureResult<Tensor> {
        if row.len() != self.cols {
            return Err(TensorError::DataLength {
                expected: self.cols,
                got: row.len(),
            });
        }
        Self::validate_mul_row_outputs(self.rows, self.cols, self.data(), row)?;
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::mul_row(self.data(), row, self.rows, self.cols) {
                    Ok(buffer) => {
                        for &output in &buffer {
                            Self::validate_finite_tensor_util_value("mul_row_output", output)?;
                        }
                        let output = Tensor::from_vec(self.rows, self.cols, buffer)?;
                        crate::emit_tensor_op(
                            "mul_row",
                            &[self.rows, self.cols, 1, row.len()],
                            &[self.rows, self.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "mul_row",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            self.rows,
                            self.cols,
                            self.layout,
                            "broadcast",
                            "tensor_util.mul_row",
                            |data| {
                                data.insert("rhs_cols".to_string(), serde_json::json!(row.len()));
                            },
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "mul_row", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut data = aligned_with_capacity(self.len());
        for row_idx in 0..self.rows {
            let offset = row_idx * self.cols;
            for (col, gain) in row.iter().copied().enumerate().take(self.cols) {
                let value = self.data()[offset + col];
                let output = value * gain;
                Self::validate_finite_tensor_util_value("mul_row_output", output)?;
                data.push(output);
            }
        }
        let output = Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)?;
        crate::emit_tensor_op(
            "mul_row",
            &[self.rows, self.cols, 1, row.len()],
            &[self.rows, self.cols],
        );
        emit_tensor_util_cpu_op_meta(
            "mul_row",
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.layout,
            _backend.label(),
            "broadcast",
            |data| {
                data.insert("rhs_cols".to_string(), serde_json::json!(row.len()));
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(output)
    }

    /// Applies a row-wise affine transform: `output[row, col] = self[row, col] * scale[col] + bias[col]`.
    pub fn row_affine(&self, scale_row: &[f32], bias_row: &[f32]) -> PureResult<Tensor> {
        self.row_affine_with_backend(scale_row, bias_row, TensorUtilBackend::Auto)
    }

    /// Applies a row-wise affine transform with an explicit utility backend.
    pub fn row_affine_with_backend(
        &self,
        scale_row: &[f32],
        bias_row: &[f32],
        _backend: TensorUtilBackend,
    ) -> PureResult<Tensor> {
        if scale_row.len() != self.cols {
            return Err(TensorError::DataLength {
                expected: self.cols,
                got: scale_row.len(),
            });
        }
        if bias_row.len() != self.cols {
            return Err(TensorError::DataLength {
                expected: self.cols,
                got: bias_row.len(),
            });
        }
        Self::validate_row_affine_outputs(self.rows, self.cols, self.data(), scale_row, bias_row)?;
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::row_affine(self.data(), scale_row, bias_row, self.rows, self.cols)
                {
                    Ok(buffer) => {
                        for &output in &buffer {
                            Self::validate_finite_tensor_util_value("row_affine_output", output)?;
                        }
                        let output = Tensor::from_vec(self.rows, self.cols, buffer)?;
                        crate::emit_tensor_op(
                            "row_affine",
                            &[self.rows, self.cols, 1, scale_row.len(), 1, bias_row.len()],
                            &[self.rows, self.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "row_affine",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            self.rows,
                            self.cols,
                            self.layout,
                            "broadcast",
                            "tensor_util.row_affine",
                            |data| {
                                data.insert(
                                    "scale_cols".to_string(),
                                    serde_json::json!(scale_row.len()),
                                );
                                data.insert(
                                    "bias_cols".to_string(),
                                    serde_json::json!(bias_row.len()),
                                );
                            },
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "row_affine", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut data = aligned_with_capacity(self.len());
        for row_idx in 0..self.rows {
            let offset = row_idx * self.cols;
            for col in 0..self.cols {
                let value = self.data()[offset + col];
                let output = value * scale_row[col] + bias_row[col];
                Self::validate_finite_tensor_util_value("row_affine_output", output)?;
                data.push(output);
            }
        }
        let output = Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)?;
        crate::emit_tensor_op(
            "row_affine",
            &[self.rows, self.cols, 1, scale_row.len(), 1, bias_row.len()],
            &[self.rows, self.cols],
        );
        emit_tensor_util_cpu_op_meta(
            "row_affine",
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.layout,
            _backend.label(),
            "broadcast",
            |data| {
                data.insert("scale_cols".to_string(), serde_json::json!(scale_row.len()));
                data.insert("bias_cols".to_string(), serde_json::json!(bias_row.len()));
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(output)
    }

    /// Add a scaled tensor to this tensor (`self += scale * other`).
    pub fn add_scaled(&mut self, other: &Tensor, scale: f32) -> PureResult<()> {
        self.add_scaled_with_backend(other, scale, TensorUtilBackend::Auto)
    }

    /// Add a scaled tensor in-place with an explicit utility backend selection.
    pub fn add_scaled_with_backend(
        &mut self,
        other: &Tensor,
        scale: f32,
        _backend: TensorUtilBackend,
    ) -> PureResult<()> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        Self::validate_add_scaled_outputs(self.data(), other.data(), scale)?;
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                let input_layout = self.layout;
                match wgpu_dense::add_scaled(self.data(), other.data(), self.rows, self.cols, scale)
                {
                    Ok(buffer) => {
                        for &output in &buffer {
                            Self::validate_finite_tensor_util_value("add_scaled_output", output)?;
                        }
                        self.data = Arc::new(TensorBuffer::from_aligned(aligned_from_vec(buffer)));
                        self.layout = Layout::RowMajor;
                        crate::emit_tensor_op(
                            "add_scaled",
                            &[self.rows, self.cols, other.rows, other.cols],
                            &[self.rows, self.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "add_scaled",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            self.rows,
                            self.cols,
                            input_layout,
                            "elementwise_inplace",
                            "tensor_util.add_scaled",
                            |data| {
                                data.insert("rhs_rows".to_string(), serde_json::json!(other.rows));
                                data.insert("rhs_cols".to_string(), serde_json::json!(other.cols));
                                data.insert(
                                    "rhs_layout".to_string(),
                                    serde_json::json!(other.layout.as_str()),
                                );
                                data.insert("scale".to_string(), serde_json::json!(scale));
                            },
                        );
                        return Ok(());
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "add_scaled", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let data = Arc::make_mut(&mut self.data);
        for (a, b) in data.iter_mut().zip(other.data.iter()) {
            *a += scale * b;
        }
        crate::emit_tensor_op(
            "add_scaled",
            &[self.rows, self.cols, other.rows, other.cols],
            &[self.rows, self.cols],
        );
        emit_tensor_util_cpu_op_meta(
            "add_scaled",
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.layout,
            _backend.label(),
            "elementwise_inplace",
            |data| {
                data.insert("rhs_rows".to_string(), serde_json::json!(other.rows));
                data.insert("rhs_cols".to_string(), serde_json::json!(other.cols));
                data.insert(
                    "rhs_layout".to_string(),
                    serde_json::json!(other.layout.as_str()),
                );
                data.insert("scale".to_string(), serde_json::json!(scale));
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(())
    }

    /// Add the provided row vector to every row (`self[row] += bias`).
    pub fn add_row_inplace(&mut self, bias: &[f32]) -> PureResult<()> {
        self.add_row_inplace_with_backend(bias, TensorUtilBackend::Auto)
    }

    /// Add a row vector in-place with an explicit utility backend selection.
    pub fn add_row_inplace_with_backend(
        &mut self,
        bias: &[f32],
        _backend: TensorUtilBackend,
    ) -> PureResult<()> {
        if bias.len() != self.cols {
            return Err(TensorError::DataLength {
                expected: self.cols,
                got: bias.len(),
            });
        }
        Self::validate_add_row_outputs(self.rows, self.cols, self.data(), bias)?;
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                let input_layout = self.layout;
                match wgpu_dense::add_row(self.data(), bias, self.rows, self.cols) {
                    Ok(buffer) => {
                        for &output in &buffer {
                            Self::validate_finite_tensor_util_value("add_row_output", output)?;
                        }
                        self.data = Arc::new(TensorBuffer::from_aligned(aligned_from_vec(buffer)));
                        self.layout = Layout::RowMajor;
                        crate::emit_tensor_op(
                            "add_row_inplace",
                            &[self.rows, self.cols, 1, bias.len()],
                            &[self.rows, self.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "add_row_inplace",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            self.rows,
                            self.cols,
                            input_layout,
                            "broadcast_inplace",
                            "tensor_util.add_row",
                            |data| {
                                data.insert("bias_cols".to_string(), serde_json::json!(bias.len()));
                            },
                        );
                        return Ok(());
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "add_row_inplace",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let data = Arc::make_mut(&mut self.data);
        for r in 0..self.rows {
            let offset = r * self.cols;
            for c in 0..self.cols {
                data[offset + c] += bias[c];
            }
        }
        crate::emit_tensor_op(
            "add_row_inplace",
            &[self.rows, self.cols, 1, bias.len()],
            &[self.rows, self.cols],
        );
        emit_tensor_util_cpu_op_meta(
            "add_row_inplace",
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.layout,
            _backend.label(),
            "broadcast_inplace",
            |data| {
                data.insert("bias_cols".to_string(), serde_json::json!(bias.len()));
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(())
    }

    /// Apply the ReLU activation in-place (`self[i] = max(self[i], 0)`).
    pub fn relu_inplace(&mut self) {
        let data = Arc::make_mut(&mut self.data);
        for value in data.iter_mut() {
            if *value < 0.0 {
                *value = 0.0;
            }
        }
        crate::emit_tensor_op(
            "relu_inplace",
            &[self.rows, self.cols],
            &[self.rows, self.cols],
        );
        emit_cpu_tensor_op_meta(
            "relu_inplace",
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.layout,
            "activation_inplace",
            |_| {},
        );
    }

    /// Return a ReLU-activated tensor (`output[i] = max(self[i], 0)`).
    pub fn relu(&self) -> PureResult<Tensor> {
        self.relu_with_backend(TensorUtilBackend::Auto)
    }

    /// Return a ReLU-activated tensor with an explicit utility backend selection.
    pub fn relu_with_backend(&self, _backend: TensorUtilBackend) -> PureResult<Tensor> {
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::relu(self.data(), self.rows, self.cols) {
                    Ok(buffer) => {
                        for &output in &buffer {
                            Self::validate_finite_tensor_util_value("relu_output", output)?;
                        }
                        let output = Tensor::from_vec(self.rows, self.cols, buffer)?;
                        crate::emit_tensor_op(
                            "relu",
                            &[self.rows, self.cols],
                            &[self.rows, self.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "relu",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            self.rows,
                            self.cols,
                            self.layout,
                            "activation",
                            "tensor_util.relu",
                            |_| {},
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "relu", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut data = aligned_with_capacity(self.len());
        for &value in self.data.iter() {
            let output = value.max(0.0);
            Self::validate_finite_tensor_util_value("relu_output", output)?;
            data.push(output);
        }
        let output = Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)?;
        crate::emit_tensor_op("relu", &[self.rows, self.cols], &[self.rows, self.cols]);
        emit_tensor_util_cpu_op_meta(
            "relu",
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.layout,
            _backend.label(),
            "activation",
            |_meta| {
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    _meta.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(output)
    }

    /// Apply the GELU activation in-place (`self[i] = GELU(self[i])`).
    pub fn gelu_inplace(&mut self) {
        let data = Arc::make_mut(&mut self.data);
        for value in data.iter_mut() {
            *value = gelu(*value);
        }
        crate::emit_tensor_op(
            "gelu_inplace",
            &[self.rows, self.cols],
            &[self.rows, self.cols],
        );
        emit_cpu_tensor_op_meta(
            "gelu_inplace",
            self.rows,
            self.cols,
            self.rows,
            self.cols,
            self.layout,
            "activation_inplace",
            |_| {},
        );
    }

    /// Applies the derivative of GELU to the provided gradient tensor.
    pub fn gelu_backward(&self, grad_output: &Tensor) -> PureResult<Tensor> {
        self.gelu_backward_with_backend(grad_output, TensorUtilBackend::Auto)
    }

    /// Applies the derivative of GELU to the provided gradient tensor with explicit backend control.
    pub fn gelu_backward_with_backend(
        &self,
        grad_output: &Tensor,
        backend: TensorUtilBackend,
    ) -> PureResult<Tensor> {
        if self.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, cols) = self.shape();

        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        match backend {
            TensorUtilBackend::Auto =>
            {
                #[cfg(feature = "wgpu")]
                if wgpu_dense::is_available() {
                    match wgpu_dense::gelu_backward(self.data(), grad_output.data(), rows, cols) {
                        Ok(buffer) => {
                            let output = Tensor::from_vec(rows, cols, buffer)?;
                            crate::emit_tensor_op(
                                "gelu_backward",
                                &[self.rows, self.cols, grad_output.rows, grad_output.cols],
                                &[rows, cols],
                            );
                            crate::emit_tensor_op_meta("gelu_backward", || {
                                serde_json::json!({
                                    "backend": "wgpu_dense",
                                    "requested_backend": backend.label(),
                                    "rows": rows,
                                    "cols": cols,
                                })
                            });
                            return Ok(output);
                        }
                        Err(message) if strict_gpu_path() => {
                            return Err(strict_gpu_fallback_error(
                                "wgpu",
                                "gelu_backward",
                                message,
                            ));
                        }
                        Err(message) => {
                            wgpu_failure = Some(message);
                        }
                    }
                }
            }
            TensorUtilBackend::Cpu => {}
            TensorUtilBackend::GpuWgpu => {
                #[cfg(feature = "wgpu")]
                {
                    match wgpu_dense::gelu_backward(self.data(), grad_output.data(), rows, cols) {
                        Ok(buffer) => {
                            let output = Tensor::from_vec(rows, cols, buffer)?;
                            crate::emit_tensor_op(
                                "gelu_backward",
                                &[self.rows, self.cols, grad_output.rows, grad_output.cols],
                                &[rows, cols],
                            );
                            crate::emit_tensor_op_meta("gelu_backward", || {
                                serde_json::json!({
                                    "backend": "wgpu_dense",
                                    "requested_backend": backend.label(),
                                    "rows": rows,
                                    "cols": cols,
                                })
                            });
                            return Ok(output);
                        }
                        Err(message) => {
                            return Err(TensorError::BackendFailure {
                                backend: "wgpu",
                                message,
                            });
                        }
                    }
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    return Err(TensorError::BackendFailure {
                        backend: "wgpu",
                        message: "wgpu backend disabled at compile time".into(),
                    });
                }
            }
        }

        let mut data = Vec::with_capacity(rows * cols);
        for (z, g) in self.data().iter().zip(grad_output.data().iter()) {
            data.push(gelu_prime(*z) * g);
        }
        let output = Tensor::from_vec(rows, cols, data)?;
        #[cfg(feature = "wgpu")]
        let (fallback_from, fallback_message) = if let Some(message) = wgpu_failure.as_deref() {
            (Some("wgpu"), Some(message))
        } else {
            (None, None)
        };
        #[cfg(not(feature = "wgpu"))]
        let (fallback_from, fallback_message): (Option<&str>, Option<&str>) = (None, None);

        crate::emit_tensor_op(
            "gelu_backward",
            &[self.rows, self.cols, grad_output.rows, grad_output.cols],
            &[rows, cols],
        );
        crate::emit_tensor_op_meta("gelu_backward", || {
            serde_json::json!({
                "backend": "cpu",
                "requested_backend": backend.label(),
                "rows": rows,
                "cols": cols,
                "fallback": {
                    "from": fallback_from,
                    "message": fallback_message,
                }
            })
        });
        Ok(output)
    }

    /// Returns the transpose of the tensor.
    pub fn transpose(&self) -> Tensor {
        self.transpose_with_backend(TensorUtilBackend::Auto)
            .expect("CPU transpose is infallible")
    }

    /// Returns the transpose of the tensor with an explicit utility backend selection.
    pub fn transpose_with_backend(&self, _backend: TensorUtilBackend) -> PureResult<Tensor> {
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::transpose(self.data(), self.rows, self.cols) {
                    Ok(buffer) => {
                        let output = Tensor::from_vec(self.cols, self.rows, buffer)?;
                        crate::emit_tensor_op(
                            "transpose",
                            &[self.rows, self.cols],
                            &[output.rows, output.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "transpose",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            output.rows,
                            output.cols,
                            self.layout,
                            "layout",
                            "tensor_util.transpose",
                            |_| {},
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "transpose", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut data = aligned_zeroed(self.len());
        for r in 0..self.rows {
            for c in 0..self.cols {
                data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        let output = Tensor {
            data: Arc::new(TensorBuffer::from_aligned(data)),
            rows: self.cols,
            cols: self.rows,
            layout: Layout::RowMajor,
        };
        crate::emit_tensor_op(
            "transpose",
            &[self.rows, self.cols],
            &[output.rows, output.cols],
        );
        emit_tensor_util_cpu_op_meta(
            "transpose",
            self.rows,
            self.cols,
            output.rows,
            output.cols,
            self.layout,
            _backend.label(),
            "layout",
            |_data| {
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    _data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(output)
    }

    /// Returns a reshaped tensor when the requested dimensions are compatible.
    pub fn reshape(&self, rows: usize, cols: usize) -> PureResult<Tensor> {
        if rows * cols != self.len() {
            return Err(TensorError::DataLength {
                expected: rows * cols,
                got: self.len(),
            });
        }

        let zero_copy = matches!(self.layout, Layout::RowMajor);
        let output = if zero_copy {
            self.view(rows, cols)?
        } else {
            let row_major = self.to_layout(Layout::RowMajor)?;
            row_major.view(rows, cols)?
        };
        crate::emit_tensor_op("reshape", &[self.rows, self.cols], &[rows, cols]);
        emit_basic_tensor_op_meta(
            "reshape",
            self.rows,
            self.cols,
            rows,
            cols,
            self.layout,
            if zero_copy { "view" } else { "cpu" },
            "auto",
            if zero_copy { "metadata" } else { "scalar" },
            "layout",
            |data| {
                data.insert("zero_copy".to_string(), serde_json::json!(zero_copy));
            },
        );
        Ok(output)
    }

    /// Returns the sum over rows for each column.
    pub fn sum_axis0(&self) -> Vec<f32> {
        self.sum_axis0_with_backend(TensorUtilBackend::Auto)
    }

    /// Returns the sum over rows for each column with an explicit utility backend.
    pub fn sum_axis0_with_backend(&self, _backend: TensorUtilBackend) -> Vec<f32> {
        self.try_sum_axis0_with_backend(_backend)
            .unwrap_or_else(|_| self.sum_axis0_unchecked())
    }

    fn sum_axis0_unchecked(&self) -> Vec<f32> {
        let mut sums = vec![0.0; self.cols];
        if self.cols == 0 {
            return sums;
        }
        for row in self.data().chunks(self.cols) {
            for (sum, value) in sums.iter_mut().zip(row.iter()) {
                *sum += *value;
            }
        }
        sums
    }

    /// Fallible sum over rows for each column with finite-output validation.
    pub fn try_sum_axis0(&self) -> PureResult<Vec<f32>> {
        self.try_sum_axis0_with_backend(TensorUtilBackend::Auto)
    }

    /// Fallible sum over rows for each column with an explicit utility backend.
    pub fn try_sum_axis0_with_backend(&self, _backend: TensorUtilBackend) -> PureResult<Vec<f32>> {
        let mut sums = vec![0.0; self.cols];
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;
        if self.cols == 0 {
            crate::emit_tensor_op("sum_axis0", &[self.rows, self.cols], &[1, self.cols]);
            emit_tensor_util_cpu_op_meta(
                "sum_axis0",
                self.rows,
                self.cols,
                1,
                self.cols,
                self.layout,
                _backend.label(),
                "reduction",
                |data| {
                    data.insert("axis".to_string(), serde_json::json!(0));
                },
            );
            return Ok(sums);
        }
        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::sum_axis0(self.data(), self.rows, self.cols) {
                    Ok(buffer) => {
                        for &output in &buffer {
                            Self::validate_finite_tensor_util_value("sum_axis0_output", output)?;
                        }
                        crate::emit_tensor_op(
                            "sum_axis0",
                            &[self.rows, self.cols],
                            &[1, self.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "sum_axis0",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            1,
                            self.cols,
                            self.layout,
                            "reduction",
                            "tensor_util.sum_axis0",
                            |data| {
                                data.insert("axis".to_string(), serde_json::json!(0));
                            },
                        );
                        return Ok(buffer);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "sum_axis0", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }
        for row in self.data().chunks(self.cols) {
            for (sum, value) in sums.iter_mut().zip(row.iter()) {
                Self::validate_finite_tensor_util_value("sum_axis0_input", *value)?;
                *sum += *value;
                Self::validate_finite_tensor_util_value("sum_axis0_output", *sum)?;
            }
        }
        crate::emit_tensor_op("sum_axis0", &[self.rows, self.cols], &[1, self.cols]);
        emit_tensor_util_cpu_op_meta(
            "sum_axis0",
            self.rows,
            self.cols,
            1,
            self.cols,
            self.layout,
            _backend.label(),
            "reduction",
            |data| {
                data.insert("axis".to_string(), serde_json::json!(0));
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(sums)
    }

    /// Returns the sum over rows for each column multiplied by `scale`.
    pub fn sum_axis0_scaled_with_backend(
        &self,
        scale: f32,
        _backend: TensorUtilBackend,
    ) -> Vec<f32> {
        self.try_sum_axis0_scaled_with_backend(scale, _backend)
            .unwrap_or_else(|_| self.sum_axis0_scaled_unchecked(scale))
    }

    fn sum_axis0_scaled_unchecked(&self, scale: f32) -> Vec<f32> {
        let mut sums = vec![0.0; self.cols];
        if self.cols == 0 {
            return sums;
        }
        for row in self.data().chunks(self.cols) {
            for (sum, value) in sums.iter_mut().zip(row.iter()) {
                *sum += *value;
            }
        }
        for sum in &mut sums {
            *sum *= scale;
        }
        sums
    }

    /// Fallible sum over rows multiplied by `scale` with finite-output validation.
    pub fn try_sum_axis0_scaled_with_backend(
        &self,
        scale: f32,
        _backend: TensorUtilBackend,
    ) -> PureResult<Vec<f32>> {
        Self::validate_scale_factor("sum_axis0_scaled_factor", scale)?;
        let mut sums = vec![0.0; self.cols];
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;
        if self.cols == 0 {
            crate::emit_tensor_op("sum_axis0_scaled", &[self.rows, self.cols], &[1, self.cols]);
            emit_tensor_util_cpu_op_meta(
                "sum_axis0_scaled",
                self.rows,
                self.cols,
                1,
                self.cols,
                self.layout,
                _backend.label(),
                "reduction",
                |data| {
                    data.insert("axis".to_string(), serde_json::json!(0));
                    data.insert("scale".to_string(), serde_json::json!(scale));
                },
            );
            return Ok(sums);
        }
        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::sum_axis0_scaled(self.data(), self.rows, self.cols, scale) {
                    Ok(buffer) => {
                        for &output in &buffer {
                            Self::validate_finite_tensor_util_value(
                                "sum_axis0_scaled_output",
                                output,
                            )?;
                        }
                        crate::emit_tensor_op(
                            "sum_axis0_scaled",
                            &[self.rows, self.cols],
                            &[1, self.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "sum_axis0_scaled",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            1,
                            self.cols,
                            self.layout,
                            "reduction",
                            "tensor_util.sum_axis0_scaled",
                            |data| {
                                data.insert("axis".to_string(), serde_json::json!(0));
                                data.insert("scale".to_string(), serde_json::json!(scale));
                            },
                        );
                        return Ok(buffer);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "sum_axis0_scaled",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }
        for row in self.data().chunks(self.cols) {
            for (sum, value) in sums.iter_mut().zip(row.iter()) {
                Self::validate_finite_tensor_util_value("sum_axis0_scaled_input", *value)?;
                *sum += *value;
                Self::validate_finite_tensor_util_value("sum_axis0_scaled_accumulator", *sum)?;
            }
        }
        for sum in &mut sums {
            *sum *= scale;
            Self::validate_finite_tensor_util_value("sum_axis0_scaled_output", *sum)?;
        }
        crate::emit_tensor_op("sum_axis0_scaled", &[self.rows, self.cols], &[1, self.cols]);
        emit_tensor_util_cpu_op_meta(
            "sum_axis0_scaled",
            self.rows,
            self.cols,
            1,
            self.cols,
            self.layout,
            _backend.label(),
            "reduction",
            |data| {
                data.insert("axis".to_string(), serde_json::json!(0));
                data.insert("scale".to_string(), serde_json::json!(scale));
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(sums)
    }

    /// Returns the sum over columns for each row.
    pub fn sum_axis1(&self) -> Vec<f32> {
        self.try_sum_axis1().unwrap_or_else(|_| {
            let mut sums = vec![0.0; self.rows];
            if self.cols == 0 {
                return sums;
            }
            for (slot, row) in sums.iter_mut().zip(self.data().chunks(self.cols)) {
                *slot = row.iter().copied().sum();
            }
            sums
        })
    }

    /// Fallible sum over columns for each row with finite-output validation.
    pub fn try_sum_axis1(&self) -> PureResult<Vec<f32>> {
        let mut sums = vec![0.0; self.rows];
        if self.cols == 0 {
            crate::emit_tensor_op("sum_axis1", &[self.rows, self.cols], &[self.rows, 1]);
            emit_cpu_tensor_op_meta(
                "sum_axis1",
                self.rows,
                self.cols,
                self.rows,
                1,
                self.layout,
                "reduction",
                |data| {
                    data.insert("axis".to_string(), serde_json::json!(1));
                },
            );
            return Ok(sums);
        }
        for (slot, row) in sums.iter_mut().zip(self.data().chunks(self.cols)) {
            for &value in row {
                Self::validate_finite_tensor_util_value("sum_axis1_input", value)?;
                *slot += value;
                Self::validate_finite_tensor_util_value("sum_axis1_output", *slot)?;
            }
        }
        crate::emit_tensor_op("sum_axis1", &[self.rows, self.cols], &[self.rows, 1]);
        emit_cpu_tensor_op_meta(
            "sum_axis1",
            self.rows,
            self.cols,
            self.rows,
            1,
            self.layout,
            "reduction",
            |data| {
                data.insert("axis".to_string(), serde_json::json!(1));
            },
        );
        Ok(sums)
    }

    /// Concatenates tensors row-wise producing a new tensor whose row count is the sum
    /// of the inputs while preserving the shared column dimension.
    pub fn cat_rows(tensors: &[Tensor]) -> PureResult<Tensor> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyInput("Tensor::cat_rows"));
        }
        let cols = tensors[0].cols;
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
        let mut data = aligned_with_capacity(total_rows * cols);
        for tensor in tensors {
            data.extend_from_slice(tensor.data.as_slice());
        }
        let output = Tensor::from_aligned(total_rows, cols, data, Layout::RowMajor)?;
        crate::emit_tensor_op(
            "cat_rows",
            &[tensors.len(), total_rows, cols],
            &[total_rows, cols],
        );
        emit_cpu_tensor_op_meta(
            "cat_rows",
            total_rows,
            cols,
            total_rows,
            cols,
            Layout::RowMajor,
            "copy",
            |data| {
                data.insert("inputs".to_string(), serde_json::json!(tensors.len()));
            },
        );
        Ok(output)
    }

    /// Computes the squared L2 norm of the tensor.
    pub fn squared_l2_norm(&self) -> f32 {
        self.squared_l2_norm_cpu(TensorUtilBackend::Auto.label(), None)
    }

    fn squared_l2_norm_cpu(&self, requested_backend: &'static str, fallback: Option<&str>) -> f32 {
        let sum_squares = self.data.iter().map(|v| v * v).sum();
        crate::emit_tensor_op("squared_l2_norm", &[self.rows, self.cols], &[1, 1]);
        emit_basic_tensor_op_meta(
            "squared_l2_norm",
            self.rows,
            self.cols,
            1,
            1,
            self.layout,
            "cpu",
            requested_backend,
            "scalar",
            "diagnostic_reduction",
            |data| {
                data.insert("reduction".to_string(), serde_json::json!("sum_squares"));
                data.insert("result".to_string(), serde_json::json!(sum_squares));
                if let Some(message) = fallback {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        sum_squares
    }

    /// Computes the squared L2 norm with an explicit utility backend.
    pub fn squared_l2_norm_with_backend(&self, _backend: TensorUtilBackend) -> PureResult<f32> {
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::sum_squares(self.data(), self.rows, self.cols) {
                    Ok(sum_squares) => {
                        Self::validate_finite_tensor_util_value(
                            "squared_l2_norm_output",
                            sum_squares,
                        )?;
                        crate::emit_tensor_op("squared_l2_norm", &[self.rows, self.cols], &[1, 1]);
                        emit_wgpu_tensor_op_meta(
                            "squared_l2_norm",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            1,
                            1,
                            self.layout,
                            "diagnostic_reduction",
                            "tensor_util.sum_squares",
                            |data| {
                                data.insert(
                                    "reduction".to_string(),
                                    serde_json::json!("sum_squares"),
                                );
                                data.insert("result".to_string(), serde_json::json!(sum_squares));
                            },
                        );
                        return Ok(sum_squares);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "squared_l2_norm",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        Ok(self.squared_l2_norm_cpu(_backend.label(), {
            #[cfg(feature = "wgpu")]
            {
                wgpu_failure.as_deref()
            }
            #[cfg(not(feature = "wgpu"))]
            {
                None
            }
        }))
    }

    /// Computes the L1 sum of absolute values.
    pub fn sum_abs(&self) -> f32 {
        self.sum_abs_cpu(TensorUtilBackend::Auto.label(), None)
    }

    fn sum_abs_cpu(&self, requested_backend: &'static str, fallback: Option<&str>) -> f32 {
        let sum_abs = self.data.iter().map(|value| value.abs()).sum();
        crate::emit_tensor_op("sum_abs", &[self.rows, self.cols], &[1, 1]);
        emit_basic_tensor_op_meta(
            "sum_abs",
            self.rows,
            self.cols,
            1,
            1,
            self.layout,
            "cpu",
            requested_backend,
            "scalar",
            "diagnostic_reduction",
            |data| {
                data.insert("reduction".to_string(), serde_json::json!("sum_abs"));
                data.insert("result".to_string(), serde_json::json!(sum_abs));
                if let Some(message) = fallback {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        sum_abs
    }

    /// Computes the L1 sum of absolute values with an explicit utility backend.
    pub fn sum_abs_with_backend(&self, _backend: TensorUtilBackend) -> PureResult<f32> {
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::sum_abs(self.data(), self.rows, self.cols) {
                    Ok(sum_abs) => {
                        Self::validate_finite_tensor_util_value("sum_abs_output", sum_abs)?;
                        crate::emit_tensor_op("sum_abs", &[self.rows, self.cols], &[1, 1]);
                        emit_wgpu_tensor_op_meta(
                            "sum_abs",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            1,
                            1,
                            self.layout,
                            "diagnostic_reduction",
                            "tensor_util.sum_abs",
                            |data| {
                                data.insert("reduction".to_string(), serde_json::json!("sum_abs"));
                                data.insert("result".to_string(), serde_json::json!(sum_abs));
                            },
                        );
                        return Ok(sum_abs);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error("wgpu", "sum_abs", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        Ok(self.sum_abs_cpu(_backend.label(), {
            #[cfg(feature = "wgpu")]
            {
                wgpu_failure.as_deref()
            }
            #[cfg(not(feature = "wgpu"))]
            {
                None
            }
        }))
    }

    /// Projects a flattened tensor onto the Poincaré ball.
    pub fn project_to_poincare(&self, curvature: f32) -> PureResult<Tensor> {
        self.project_to_poincare_with_backend(curvature, TensorUtilBackend::Auto)
    }

    /// Projects onto the Poincaré ball with an explicit utility backend.
    pub fn project_to_poincare_with_backend(
        &self,
        curvature: f32,
        _backend: TensorUtilBackend,
    ) -> PureResult<Tensor> {
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        let scale = (-curvature).sqrt();
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::project_to_poincare(self.data(), self.rows, self.cols, curvature)
                {
                    Ok(buffer) => {
                        let output = Tensor::from_vec(self.rows, self.cols, buffer)?;
                        let (nonzero_rows, max_row_l2) =
                            row_l2_projection_stats(self.data(), self.rows, self.cols);
                        crate::emit_tensor_op(
                            "project_to_poincare",
                            &[self.rows, self.cols],
                            &[output.rows, output.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "project_to_poincare",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            output.rows,
                            output.cols,
                            self.layout,
                            "hyperbolic_projection",
                            "tensor_util.project_to_poincare",
                            |data| {
                                data.insert("curvature".to_string(), serde_json::json!(curvature));
                                data.insert("scale".to_string(), serde_json::json!(scale));
                                data.insert(
                                    "nonzero_rows".to_string(),
                                    serde_json::json!(nonzero_rows),
                                );
                                data.insert(
                                    "max_row_l2".to_string(),
                                    serde_json::json!(max_row_l2),
                                );
                            },
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "project_to_poincare",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut data = aligned_with_capacity(self.len());
        let mut nonzero_rows = 0usize;
        let mut max_row_l2 = 0.0f32;
        for r in 0..self.rows {
            let start = r * self.cols;
            let end = start + self.cols;
            let chunk = &self.data[start..end];
            let norm: f32 = chunk.iter().map(|v| v * v).sum::<f32>().sqrt();
            max_row_l2 = max_row_l2.max(norm);
            if norm > 0.0 {
                nonzero_rows = nonzero_rows.saturating_add(1);
                let clip = (norm / scale).tanh();
                let factor = clip / norm;
                for v in chunk {
                    data.push(v * factor);
                }
            } else {
                data.extend_from_slice(chunk);
            }
        }
        let output = Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)?;
        crate::emit_tensor_op(
            "project_to_poincare",
            &[self.rows, self.cols],
            &[output.rows, output.cols],
        );
        emit_tensor_util_cpu_op_meta(
            "project_to_poincare",
            self.rows,
            self.cols,
            output.rows,
            output.cols,
            self.layout,
            _backend.label(),
            "hyperbolic_projection",
            |data| {
                data.insert("curvature".to_string(), serde_json::json!(curvature));
                data.insert("scale".to_string(), serde_json::json!(scale));
                data.insert("nonzero_rows".to_string(), serde_json::json!(nonzero_rows));
                data.insert("max_row_l2".to_string(), serde_json::json!(max_row_l2));
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(output)
    }

    /// Applies a row-wise affine gate and projects the result onto the Poincaré ball.
    #[allow(clippy::too_many_arguments)]
    pub fn wave_gate_project_with_backend(
        &self,
        gate: &[f32],
        bias: &[f32],
        curvature: f32,
        saturation: f32,
        porosity: f32,
        _backend: TensorUtilBackend,
    ) -> PureResult<Tensor> {
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if gate.len() != self.cols {
            return Err(TensorError::DataLength {
                expected: self.cols,
                got: gate.len(),
            });
        }
        if bias.len() != self.cols {
            return Err(TensorError::DataLength {
                expected: self.cols,
                got: bias.len(),
            });
        }
        let scale = (-curvature).sqrt();
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(_backend, TensorUtilBackend::GpuWgpu)
                && self.rows > 0
                && self.cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::wave_gate_project(
                    self.data(),
                    gate,
                    bias,
                    self.rows,
                    self.cols,
                    curvature,
                    saturation,
                    porosity,
                ) {
                    Ok(buffer) => {
                        let output = Tensor::from_vec(self.rows, self.cols, buffer)?;
                        crate::emit_tensor_op(
                            "wave_gate_project",
                            &[self.rows, self.cols, 1, gate.len(), 1, bias.len()],
                            &[output.rows, output.cols],
                        );
                        emit_wgpu_tensor_op_meta(
                            "wave_gate_project",
                            _backend.label(),
                            self.rows,
                            self.cols,
                            output.rows,
                            output.cols,
                            self.layout,
                            "fused_hyperbolic_projection",
                            "tensor_util.wave_gate_project",
                            |data| {
                                data.insert("curvature".to_string(), serde_json::json!(curvature));
                                data.insert("scale".to_string(), serde_json::json!(scale));
                                data.insert(
                                    "saturation".to_string(),
                                    serde_json::json!(saturation),
                                );
                                data.insert("porosity".to_string(), serde_json::json!(porosity));
                                data.insert("gate_cols".to_string(), serde_json::json!(gate.len()));
                                data.insert("bias_cols".to_string(), serde_json::json!(bias.len()));
                            },
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "wave_gate_project",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut gated = aligned_with_capacity(self.len());
        for r in 0..self.rows {
            let offset = r * self.cols;
            for c in 0..self.cols {
                let value = self.data[offset + c] * gate[c] + bias[c];
                gated.push(porous_mix_value(value, saturation, porosity));
            }
        }
        let mut data = aligned_with_capacity(self.len());
        for r in 0..self.rows {
            let start = r * self.cols;
            let end = start + self.cols;
            let chunk = &gated[start..end];
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
        let output = Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)?;
        crate::emit_tensor_op(
            "wave_gate_project",
            &[self.rows, self.cols, 1, gate.len(), 1, bias.len()],
            &[output.rows, output.cols],
        );
        emit_tensor_util_cpu_op_meta(
            "wave_gate_project",
            self.rows,
            self.cols,
            output.rows,
            output.cols,
            self.layout,
            _backend.label(),
            "fused_hyperbolic_projection",
            |data| {
                data.insert("curvature".to_string(), serde_json::json!(curvature));
                data.insert("scale".to_string(), serde_json::json!(scale));
                data.insert("saturation".to_string(), serde_json::json!(saturation));
                data.insert("porosity".to_string(), serde_json::json!(porosity));
                data.insert("gate_cols".to_string(), serde_json::json!(gate.len()));
                data.insert("bias_cols".to_string(), serde_json::json!(bias.len()));
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
            },
        );
        Ok(output)
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
        let distance = 2.0 * (1.0 + (sum_norm / denom)).acosh();
        crate::emit_tensor_op(
            "hyperbolic_distance",
            &[self.rows, self.cols, other.rows, other.cols],
            &[1, 1],
        );
        emit_cpu_tensor_op_meta(
            "hyperbolic_distance",
            self.rows,
            self.cols,
            1,
            1,
            self.layout,
            "hyperbolic_distance",
            |data| {
                data.insert("rhs_rows".to_string(), serde_json::json!(other.rows));
                data.insert("rhs_cols".to_string(), serde_json::json!(other.cols));
                data.insert(
                    "rhs_layout".to_string(),
                    serde_json::json!(other.layout.as_str()),
                );
                data.insert("curvature".to_string(), serde_json::json!(curvature));
                data.insert("scale".to_string(), serde_json::json!(scale));
                data.insert("sum_norm".to_string(), serde_json::json!(sum_norm));
                data.insert("sum_inner".to_string(), serde_json::json!(sum_inner));
                data.insert("denom".to_string(), serde_json::json!(denom));
                data.insert("distance".to_string(), serde_json::json!(distance));
            },
        );
        Ok(distance)
    }
}

/// Computes the mean squared error between `predictions` and `targets`.
pub fn mean_squared_error(predictions: &Tensor, targets: &Tensor) -> PureResult<f32> {
    mean_squared_error_with_backend(predictions, targets, TensorUtilBackend::Auto)
}

/// Computes mean squared error with an explicit tensor utility backend for the reduction tail.
pub fn mean_squared_error_with_backend(
    predictions: &Tensor,
    targets: &Tensor,
    backend: TensorUtilBackend,
) -> PureResult<f32> {
    if predictions.shape() != targets.shape() {
        return Err(TensorError::ShapeMismatch {
            left: predictions.shape(),
            right: targets.shape(),
        });
    }
    crate::emit_tensor_op(
        "mean_squared_error",
        &[
            predictions.rows,
            predictions.cols,
            targets.rows,
            targets.cols,
        ],
        &[1, 1],
    );
    emit_basic_tensor_op_meta(
        "mean_squared_error",
        predictions.rows,
        predictions.cols,
        1,
        1,
        predictions.layout,
        "composite",
        backend.label(),
        "tensor_util.mean_squared_error",
        "reduction",
        |data| {
            data.insert("rhs_rows".to_string(), serde_json::json!(targets.rows));
            data.insert("rhs_cols".to_string(), serde_json::json!(targets.cols));
            data.insert(
                "rhs_layout".to_string(),
                serde_json::json!(targets.layout.as_str()),
            );
            data.insert("reduction".to_string(), serde_json::json!("mean"));
            data.insert(
                "reduction_backend".to_string(),
                serde_json::json!(backend.label()),
            );
        },
    );
    let values = predictions.rows.saturating_mul(predictions.cols);
    if values == 0 {
        return Ok(0.0);
    }
    Tensor::validate_finite_tensor_util_slice("mse_prediction", predictions.data())?;
    Tensor::validate_finite_tensor_util_slice("mse_target", targets.data())?;
    let diff =
        relabel_tensor_non_finite(predictions.sub_with_backend(targets, backend), "mse_diff")?;
    mean_squared_error_from_diff_labels(
        &diff,
        "mse_diff",
        "mse_squared_diff",
        "mse_column_mean",
        "mean_squared_error",
        backend,
    )
}

fn relabel_tensor_non_finite<T>(result: PureResult<T>, label: &'static str) -> PureResult<T> {
    match result {
        Err(TensorError::NonFiniteValue { value, .. }) => {
            Err(TensorError::NonFiniteValue { label, value })
        }
        other => other,
    }
}

fn mean_squared_error_from_diff_labels(
    diff: &Tensor,
    diff_label: &'static str,
    squared_label: &'static str,
    column_mean_label: &'static str,
    mean_label: &'static str,
    backend: TensorUtilBackend,
) -> PureResult<f32> {
    let values = diff.data().len();
    if values == 0 {
        return Ok(0.0);
    }
    Tensor::validate_finite_tensor_util_slice(diff_label, diff.data())?;
    let squared =
        relabel_tensor_non_finite(diff.hadamard_with_backend(diff, backend), squared_label)?;
    let column_means = relabel_tensor_non_finite(
        squared.try_sum_axis0_scaled_with_backend(1.0 / values as f32, backend),
        column_mean_label,
    )?;
    let mut mean = 0.0f32;
    for value in column_means {
        Tensor::validate_finite_tensor_util_value(column_mean_label, value)?;
        mean += value;
        Tensor::validate_finite_tensor_util_value(mean_label, mean)?;
    }
    Ok(mean)
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
    packed_weights: RefCell<Option<PackedB>>,
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
            packed_weights: RefCell::new(None),
        })
    }

    /// Runs a forward pass: `inputs @ weights + bias`.
    pub fn forward(&self, inputs: &Tensor) -> PureResult<Tensor> {
        self.forward_with_backend(inputs, MatmulBackend::Auto)
    }

    /// Runs a forward pass with an explicit matmul backend.
    pub fn forward_with_backend(
        &self,
        inputs: &Tensor,
        backend: MatmulBackend,
    ) -> PureResult<Tensor> {
        if inputs.shape().1 != self.weights.shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: inputs.shape(),
                right: self.weights.shape(),
            });
        }
        let pack = self.ensure_packed_weights()?;
        inputs.matmul_prepacked_bias_with_backend(&pack, &self.bias, backend)
    }

    /// Performs a single batch of gradient descent and returns the batch loss.
    pub fn train_batch(
        &mut self,
        inputs: &Tensor,
        targets: &Tensor,
        learning_rate: f32,
    ) -> PureResult<f32> {
        self.train_batch_with_backend(
            inputs,
            targets,
            learning_rate,
            MatmulBackend::Auto,
            TensorUtilBackend::Auto,
        )
    }

    /// Performs a single batch of gradient descent with explicit compute backends.
    pub fn train_batch_with_backend(
        &mut self,
        inputs: &Tensor,
        targets: &Tensor,
        learning_rate: f32,
        matmul_backend: MatmulBackend,
        tensor_backend: TensorUtilBackend,
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
        Tensor::validate_finite_tensor_util_value("linear_model_learning_rate", learning_rate)?;
        let batch_size = inputs.shape().0 as f32;
        let pack = self.ensure_packed_weights()?;
        let predictions =
            inputs.matmul_prepacked_bias_with_backend(&pack, &self.bias, matmul_backend)?;
        let diff = relabel_tensor_non_finite(
            predictions.sub_with_backend(targets, tensor_backend),
            "linear_model_mse_diff",
        )?;
        let batch_loss = mean_squared_error_from_diff_with_backend(&diff, tensor_backend)?;
        let grad_w = inputs.matmul_lhs_transpose_scaled_with_backend(
            &diff,
            1.0 / batch_size,
            matmul_backend,
        )?;
        let grad_b = relabel_tensor_non_finite(
            diff.try_sum_axis0_scaled_with_backend(1.0 / batch_size, tensor_backend),
            "linear_model_bias_gradient",
        )?;
        for val in &grad_b {
            Tensor::validate_finite_tensor_util_value("linear_model_bias_gradient", *val)?;
        }
        let mut next_bias = self.bias.clone();
        for (b, g) in next_bias.iter_mut().zip(grad_b.iter()) {
            let delta = learning_rate * g;
            Tensor::validate_finite_tensor_util_value("linear_model_bias_delta", delta)?;
            *b -= delta;
            Tensor::validate_finite_tensor_util_value("linear_model_bias", *b)?;
        }
        self.weights
            .add_scaled_with_backend(&grad_w, -learning_rate, tensor_backend)?;
        self.bias = next_bias;
        self.packed_weights.borrow_mut().take();
        Ok(batch_loss)
    }

    /// Returns a reference to the model weights.
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Returns a reference to the model bias.
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    fn ensure_packed_weights(&self) -> PureResult<PackedB> {
        if let Some(existing) = self.packed_weights.borrow().clone() {
            return Ok(existing);
        }
        let pack = PackedB::from_tensor(&self.weights, Tile::col_major())?;
        *self.packed_weights.borrow_mut() = Some(pack.clone());
        Ok(pack)
    }
}

fn mean_squared_error_from_diff_with_backend(
    diff: &Tensor,
    backend: TensorUtilBackend,
) -> PureResult<f32> {
    mean_squared_error_from_diff_labels(
        diff,
        "linear_model_mse_diff",
        "linear_model_mse_squared_diff",
        "linear_model_mse_column_mean",
        "linear_model_mse",
        backend,
    )
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HypergradTelemetry {
    summary: GradientSummary,
    curvature: f32,
    learning_rate: f32,
    saturation: f32,
    porosity: f32,
    tolerance: f32,
    max_depth: usize,
    max_volume: usize,
    finite_count: usize,
    non_finite_count: usize,
    non_finite_ratio: f32,
    rows: usize,
    cols: usize,
}

impl HypergradTelemetry {
    #[inline]
    pub fn summary(&self) -> GradientSummary {
        self.summary
    }

    #[inline]
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    #[inline]
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    #[inline]
    pub fn saturation(&self) -> f32 {
        self.saturation
    }

    #[inline]
    pub fn porosity(&self) -> f32 {
        self.porosity
    }

    #[inline]
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    #[inline]
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    #[inline]
    pub fn max_volume(&self) -> usize {
        self.max_volume
    }

    #[inline]
    pub fn finite_count(&self) -> usize {
        self.finite_count
    }

    #[inline]
    pub fn non_finite_count(&self) -> usize {
        self.non_finite_count
    }

    #[inline]
    pub fn non_finite_ratio(&self) -> f32 {
        self.non_finite_ratio
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn volume(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }
}

pub struct AmegaHypergrad {
    curvature: f32,
    learning_rate: f32,
    rows: usize,
    cols: usize,
    gradient: Vec<f32>,
    summary: Cell<GradientSummary>,
    summary_dirty: Cell<bool>,
    min_dirty: Cell<bool>,
    max_dirty: Cell<bool>,
    linf_dirty: Cell<bool>,
    topos: topos::OpenCartesianTopos,
}

/// Euclidean gradient accumulator that mirrors the hypergradient API while
/// staying entirely within flat-space optimisation loops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientSummary {
    l1: f32,
    l2: f32,
    linf: f32,
    sum: f32,
    sum_squares: f32,
    sum_cubes: f32,
    sum_quartic: f32,
    count: usize,
    min: f32,
    max: f32,
    positive_count: usize,
    negative_count: usize,
    near_zero_count: usize,
}

impl GradientSummary {
    const NEAR_ZERO_EPS: f32 = 1e-5;

    #[inline]
    pub fn from_slice(values: &[f32]) -> Self {
        let mut l1 = 0.0f64;
        let mut sum = 0.0f64;
        let mut sum_squares = 0.0f64;
        let mut sum_cubes = 0.0f64;
        let mut sum_quartic = 0.0f64;
        let mut linf = 0.0f64;
        let mut count = 0usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut positive_count = 0usize;
        let mut negative_count = 0usize;
        let mut near_zero_count = 0usize;
        for &value in values {
            if !value.is_finite() {
                continue;
            }
            let value = value as f64;
            let abs = value.abs();
            l1 += abs;
            sum += value;
            let square = value * value;
            sum_squares += square;
            sum_cubes += square * value;
            sum_quartic += square * square;
            linf = linf.max(abs);
            count += 1;
            let value_f32 = value as f32;
            min = min.min(value_f32);
            max = max.max(value_f32);
            if value_f32 > 0.0 {
                positive_count += 1;
            } else if value_f32 < 0.0 {
                negative_count += 1;
            }
            if abs as f32 <= Self::NEAR_ZERO_EPS {
                near_zero_count += 1;
            }
        }
        if count == 0 {
            min = 0.0;
            max = 0.0;
        }
        Self {
            l1: l1 as f32,
            l2: (sum_squares as f32).sqrt(),
            linf: linf as f32,
            sum: sum as f32,
            sum_squares: sum_squares as f32,
            sum_cubes: sum_cubes as f32,
            sum_quartic: sum_quartic as f32,
            count,
            min,
            max,
            positive_count,
            negative_count,
            near_zero_count,
        }
    }

    /// Builds a summary directly from raw moment statistics. `l1` captures the
    /// sum of absolute values, `sum_squares` is the accumulated \(L_2^2\)
    /// energy, `linf` is the maximum absolute entry, and `count` indicates how
    /// many samples contributed to the summary.
    #[inline]
    pub fn from_moments(l1: f32, sum_squares: f32, linf: f32, count: usize) -> Self {
        Self::from_extended_moments(l1, 0.0, sum_squares, 0.0, 0.0, linf, count)
    }

    /// Builds a summary from the provided raw power sums. `sum` is the first
    /// raw moment, `sum_squares`/`sum_cubes`/`sum_quartic` capture the higher
    /// power accumulators.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn from_extended_moments(
        l1: f32,
        sum: f32,
        sum_squares: f32,
        sum_cubes: f32,
        sum_quartic: f32,
        linf: f32,
        count: usize,
    ) -> Self {
        let l1 = if l1.is_finite() { l1.max(0.0) } else { 0.0 };
        let sum = if sum.is_finite() { sum } else { 0.0 };
        let sum_squares = if sum_squares.is_finite() {
            sum_squares.max(0.0)
        } else {
            0.0
        };
        let sum_cubes = if sum_cubes.is_finite() {
            sum_cubes
        } else {
            0.0
        };
        let sum_quartic = if sum_quartic.is_finite() {
            sum_quartic.max(0.0)
        } else {
            0.0
        };
        let linf = if linf.is_finite() { linf.max(0.0) } else { 0.0 };
        Self {
            l1,
            l2: sum_squares.sqrt(),
            linf,
            sum,
            sum_squares,
            sum_cubes,
            sum_quartic,
            count,
            min: 0.0,
            max: 0.0,
            positive_count: 0,
            negative_count: 0,
            near_zero_count: 0,
        }
    }

    /// Attach support and sign statistics to an existing summary. When the
    /// summary was constructed from aggregated power sums this method can be
    /// used to backfill the additional metrics without reprocessing the raw
    /// gradient samples.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn with_support(
        mut self,
        min: f32,
        max: f32,
        positive_count: usize,
        negative_count: usize,
        near_zero_count: usize,
    ) -> Self {
        if self.count == 0 {
            self.min = 0.0;
            self.max = 0.0;
            self.positive_count = 0;
            self.negative_count = 0;
            self.near_zero_count = 0;
            return self;
        }

        let mut min = if min.is_finite() { min } else { 0.0 };
        let mut max = if max.is_finite() { max } else { 0.0 };
        if max < min {
            mem::swap(&mut min, &mut max);
        }

        let mut positive_count = positive_count.min(self.count);
        let mut negative_count = negative_count.min(self.count - positive_count);
        let near_zero_count = near_zero_count.min(self.count);

        if positive_count + negative_count > self.count {
            let overflow = positive_count + negative_count - self.count;
            if negative_count >= overflow {
                negative_count -= overflow;
            } else {
                positive_count = positive_count.saturating_sub(overflow - negative_count);
                negative_count = 0;
            }
        }

        self.min = min;
        self.max = max;
        self.positive_count = positive_count;
        self.negative_count = negative_count;
        self.near_zero_count = near_zero_count;
        self
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

    /// Returns the signed sum of the gradient entries.
    #[inline]
    pub fn sum(&self) -> f32 {
        self.sum
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
        self.sum_squares
    }

    /// Returns the accumulated sum of cubes captured by the summary.
    #[inline]
    pub fn sum_cubes(&self) -> f32 {
        self.sum_cubes
    }

    /// Returns the accumulated sum of fourth powers captured by the summary.
    #[inline]
    pub fn sum_quartic(&self) -> f32 {
        self.sum_quartic
    }

    /// Minimum gradient value captured by the summary.
    #[inline]
    pub fn min(&self) -> f32 {
        self.min
    }

    /// Maximum gradient value captured by the summary.
    #[inline]
    pub fn max(&self) -> f32 {
        self.max
    }

    /// Difference between the maximum and minimum gradient value.
    #[inline]
    pub fn support_width(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.max - self.min
        }
    }

    /// Number of positive entries observed when constructing the summary.
    #[inline]
    pub fn positive_count(&self) -> usize {
        self.positive_count
    }

    /// Number of negative entries observed when constructing the summary.
    #[inline]
    pub fn negative_count(&self) -> usize {
        self.negative_count
    }

    /// Number of entries that landed within the near-zero guard threshold.
    #[inline]
    pub fn near_zero_count(&self) -> usize {
        self.near_zero_count
    }

    /// Number of exact zeros recorded by the summary.
    #[inline]
    pub fn zero_count(&self) -> usize {
        self.count
            .saturating_sub(self.positive_count + self.negative_count)
    }

    /// Population mean of the gradient distribution.
    #[inline]
    pub fn mean(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f32
        }
    }

    /// Population variance of the gradient distribution.
    #[inline]
    pub fn variance(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let n = self.count as f32;
        let mean = self.mean();
        let mean_square = self.sum_squares / n;
        (mean_square - mean * mean).max(0.0)
    }

    /// Standard deviation derived from the population variance.
    #[inline]
    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }

    /// Fisher-Pearson coefficient of skewness derived from the raw moments.
    #[inline]
    pub fn skewness(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f32;
        let mean = self.mean();
        let mean_square = self.sum_squares / n;
        let variance = (mean_square - mean * mean).max(0.0);
        if variance <= 1e-12 {
            return 0.0;
        }
        let mean_cube = self.sum_cubes / n;
        let central_moment3 = mean_cube - 3.0 * mean * mean_square + 2.0 * mean * mean * mean;
        (central_moment3 / variance.powf(1.5)).clamp(-1e6, 1e6)
    }

    /// Kurtosis (non-excess) derived from the raw moments.
    #[inline]
    pub fn kurtosis(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f32;
        let mean = self.mean();
        let mean_square = self.sum_squares / n;
        let variance = (mean_square - mean * mean).max(0.0);
        if variance <= 1e-12 {
            return 0.0;
        }
        let mean_cube = self.sum_cubes / n;
        let mean_quartic = self.sum_quartic / n;
        let central_moment4 = mean_quartic - 4.0 * mean * mean_cube
            + 6.0 * mean * mean * mean_square
            - 3.0 * mean.powi(4);
        if central_moment4 <= 0.0 {
            0.0
        } else {
            (central_moment4 / (variance * variance)).max(0.0)
        }
    }

    /// Fraction of entries that were positive when constructing the summary.
    #[inline]
    pub fn positive_fraction(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.positive_count as f32 / self.count as f32
        }
    }

    /// Fraction of entries that were negative when constructing the summary.
    #[inline]
    pub fn negative_fraction(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.negative_count as f32 / self.count as f32
        }
    }

    /// Fraction of entries that landed within the near-zero guard threshold.
    #[inline]
    pub fn near_zero_fraction(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.near_zero_count.min(self.count)) as f32 / self.count as f32
        }
    }

    /// Fraction of exact zero entries captured by the summary.
    #[inline]
    pub fn zero_fraction(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.zero_count() as f32 / self.count as f32
        }
    }

    /// Activity score indicating how many entries escaped the near-zero band.
    #[inline]
    pub fn activation(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            1.0 - self.near_zero_fraction()
        }
    }

    /// Signed imbalance between positive and negative entries in `[-1, 1]`.
    #[inline]
    pub fn sign_lean(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.positive_count as f32 - self.negative_count as f32) / self.count as f32
        }
    }

    /// Shannon entropy of the sign distribution normalised to `[0, 1]`.
    #[inline]
    pub fn sign_entropy(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let total = self.count as f32;
        let bins = [
            self.positive_count as f32 / total,
            self.negative_count as f32 / total,
            self.zero_count() as f32 / total,
        ];
        let mut non_zero_bins = 0usize;
        let entropy = bins.iter().fold(0.0, |acc, &p| {
            if p > 0.0 {
                non_zero_bins += 1;
                acc - p * p.ln()
            } else {
                acc
            }
        });
        if non_zero_bins <= 1 {
            0.0
        } else {
            let norm = (non_zero_bins as f32).ln();
            if norm > 0.0 {
                (entropy / norm).clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
    }
}

fn insert_gradient_summary_meta(
    data: &mut serde_json::Map<String, serde_json::Value>,
    prefix: &str,
    summary: GradientSummary,
) {
    data.insert(
        format!("{prefix}_finite_values"),
        serde_json::json!(summary.count()),
    );
    data.insert(format!("{prefix}_l1"), serde_json::json!(summary.l1()));
    data.insert(format!("{prefix}_l2"), serde_json::json!(summary.l2()));
    data.insert(format!("{prefix}_linf"), serde_json::json!(summary.linf()));
    data.insert(format!("{prefix}_rms"), serde_json::json!(summary.rms()));
    data.insert(
        format!("{prefix}_mean_abs"),
        serde_json::json!(summary.mean_abs()),
    );
    data.insert(
        format!("{prefix}_near_zero_values"),
        serde_json::json!(summary.near_zero_count()),
    );
}

impl Default for GradientSummary {
    fn default() -> Self {
        Self {
            l1: 0.0,
            l2: 0.0,
            linf: 0.0,
            sum: 0.0,
            sum_squares: 0.0,
            sum_cubes: 0.0,
            sum_quartic: 0.0,
            count: 0,
            min: 0.0,
            max: 0.0,
            positive_count: 0,
            negative_count: 0,
            near_zero_count: 0,
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
    hyper_std: f32,
    real_std: f32,
    sharpness: f32,
    activation: f32,
    sign_alignment: f32,
    sign_entropy: f32,
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
        let hyper_std = hyper.std();
        let real_std = real.std();
        let hyper_kurtosis = hyper.kurtosis();
        let sharpness = if hyper_std > Self::EPS {
            (hyper_kurtosis - 3.0).abs().min(4.0)
        } else {
            0.0
        };
        let stability_raw = 1.0
            - (hyper_pressure - real_pressure).abs() / (hyper_pressure + real_pressure + Self::EPS);
        let stability = stability_raw.clamp(0.0, 1.0);
        let saturation = hyper.linf().max(real.linf());
        let activation = 0.5 * (hyper.activation() + real.activation());
        let hyper_lean = hyper.sign_lean();
        let real_lean = real.sign_lean();
        let sign_alignment = (1.0 - 0.5 * (hyper_lean - real_lean).abs()).clamp(0.0, 1.0);
        let sign_entropy = 0.5 * (hyper.sign_entropy() + real.sign_entropy());
        Self {
            hyper_pressure,
            real_pressure,
            balance,
            stability,
            saturation,
            hyper_std,
            real_std,
            sharpness,
            activation,
            sign_alignment,
            sign_entropy: sign_entropy.clamp(0.0, 1.0),
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

    /// Standard deviation of the hypergradient summary.
    #[inline]
    pub fn hyper_std(&self) -> f32 {
        self.hyper_std
    }

    /// Standard deviation of the Euclidean gradient summary.
    #[inline]
    pub fn real_std(&self) -> f32 {
        self.real_std
    }

    /// Magnitude of the hypergradient's excess kurtosis.
    #[inline]
    pub fn sharpness(&self) -> f32 {
        self.sharpness
    }

    /// Average activation of the paired summaries (values escaping the near-zero band).
    #[inline]
    pub fn activation(&self) -> f32 {
        self.activation
    }

    /// Alignment score describing how closely the hyper and Euclidean signs agree.
    #[inline]
    pub fn sign_alignment(&self) -> f32 {
        self.sign_alignment
    }

    /// Average Shannon entropy of the sign distribution across the paired summaries.
    #[inline]
    pub fn sign_entropy(&self) -> f32 {
        self.sign_entropy
    }

    /// Gain factor for hypergradient penalties when the two tapes disagree.
    #[inline]
    pub fn penalty_gain(&self) -> f32 {
        let imbalance = (self.balance - 1.0).abs().min(2.0);
        let instability = (1.0 - self.stability).min(1.0);
        let sharpness = (self.sharpness * 0.25).min(1.0);
        let misalignment = (1.0 - self.sign_alignment).min(1.0);
        let dormancy = (1.0 - self.activation).min(1.0);
        (1.0 + 0.35 * imbalance
            + 0.35 * instability
            + 0.2 * sharpness
            + 0.2 * misalignment
            + 0.1 * dormancy)
            .clamp(1.0, 3.5)
    }

    /// Mixing factor used when blending Desire bias updates – drops towards zero
    /// when the gradients disagree so the automation can tread lightly.
    #[inline]
    pub fn bias_mix(&self) -> f32 {
        let softness = (1.0 / (1.0 + 0.5 * self.sharpness)).clamp(0.25, 1.0);
        let alignment = (0.4 + 0.6 * self.sign_alignment).clamp(0.25, 1.0);
        let activation = (0.3 + 0.7 * self.activation).clamp(0.2, 1.0);
        (0.2 + 0.8 * self.stability * softness * alignment * activation).clamp(0.1, 1.0)
    }

    /// Gain used when accumulating avoidance reports during the observation
    /// phase. High saturation dampens the contribution to avoid runaway spikes.
    #[inline]
    pub fn observation_gain(&self) -> f32 {
        let saturation = self.saturation.tanh().clamp(0.0, 1.0);
        let sharpness = (self.sharpness * 0.25).min(1.0);
        let activation = (0.4 + 0.6 * self.activation).clamp(0.2, 1.0);
        let entropy = (0.5 + 0.5 * self.sign_entropy).clamp(0.25, 1.0);
        (0.4 + 0.6 * (1.0 - saturation) * (1.0 - 0.25 * sharpness) * activation * entropy)
            .clamp(0.25, 1.0)
    }

    /// Damping factor that can shrink epsilon-like tolerances when gradients
    /// spike.
    #[inline]
    pub fn damping(&self) -> f32 {
        let saturation = self.saturation.tanh().clamp(0.0, 1.0);
        let dormancy = (1.0 - self.activation).clamp(0.0, 1.0);
        (0.4 + 0.4 * saturation + 0.2 * dormancy).clamp(0.1, 1.0)
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
            hyper_std: 0.0,
            real_std: 0.0,
            sharpness: 0.0,
            activation: 0.0,
            sign_alignment: 1.0,
            sign_entropy: 0.0,
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
        let activation = interpretation.activation().clamp(0.0, 1.0);
        let dormancy = (1.0 - activation).clamp(0.0, 1.0);
        let alignment = interpretation.sign_alignment().clamp(0.0, 1.0);
        let misalignment = (1.0 - alignment).clamp(0.0, 1.0);
        let sign_entropy = interpretation.sign_entropy().clamp(0.0, 1.0);

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

        let hyper_guard =
            (1.0 - 0.6 * caution - 0.4 * saturation - 0.25 * misalignment).clamp(0.2, 1.0);
        let real_guard =
            (1.0 - 0.4 * caution - 0.25 * saturation - 0.2 * dormancy).clamp(0.25, 1.0);
        self.hyper_base = (hyper_base * hyper_guard).clamp(0.25, 1.8);
        self.real_base = (real_base * real_guard).clamp(0.25, 1.8);

        self.operator_mix =
            (0.25 + 0.45 * stability + 0.2 * alignment + 0.1 * sign_entropy).clamp(0.2, 1.0);
        self.operator_gain =
            (self.control.penalty_gain * (1.0 - 0.35 * saturation) * (0.7 + 0.3 * activation))
                .clamp(0.5, 1.6);

        self.target_entropy = 3.5 + 0.8 * caution + 0.6 * (1.0 - sign_entropy);
        self.entropy_eta = (0.08 + 0.14 * caution + 0.05 * misalignment).clamp(0.05, 0.3);
        self.lr_slew = (0.25 - 0.15 * caution - 0.05 * dormancy).clamp(0.05, 0.25);

        let clip_floor = (0.18 + 0.12 * caution + 0.05 * misalignment).clamp(0.15, 0.35);
        let clip_target = interpretation
            .saturation()
            .max(interpretation.hyper_pressure() * (2.5 + 0.5 * misalignment))
            .max(interpretation.real_pressure() * (3.0 + dormancy));
        self.clip_floor = clip_floor;
        let clip_gain = (0.6 + 0.4 * self.gain) * (1.0 + 0.2 * misalignment);
        self.clip_hint = (clip_target * clip_gain).clamp(clip_floor, 32.0);
        self.clip_ceiling = (self.clip_hint * 1.6).max(self.clip_hint + 0.05);
        self.clip_ema = (0.25 + 0.35 * caution + 0.1 * misalignment).clamp(0.2, 0.65);

        self.z_kappa = (0.02 + 0.08 * (1.0 - stability) + 0.04 * misalignment).clamp(0.0, 0.16);
        self.z_slew = (0.22 - 0.1 * caution - 0.05 * dormancy).clamp(0.05, 0.22);

        self.quality_gain = (0.6 + 0.3 * (1.0 - caution) + 0.1 * alignment).clamp(0.4, 1.0);

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
        if curvature >= 0.0 || !curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if learning_rate <= 0.0 || !learning_rate.is_finite() {
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
        let gradient = vec![0.0; rows * cols];
        let summary = GradientSummary::from_slice(&gradient);
        Ok(Self {
            curvature,
            learning_rate,
            rows,
            cols,
            gradient,
            summary: Cell::new(summary),
            summary_dirty: Cell::new(false),
            min_dirty: Cell::new(false),
            max_dirty: Cell::new(false),
            linf_dirty: Cell::new(false),
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
        if curvature >= 0.0 || !curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if learning_rate <= 0.0 || !learning_rate.is_finite() {
            return Err(TensorError::NonPositiveLearningRate {
                rate: learning_rate,
            });
        }
        let capacity = rows.saturating_mul(cols);
        if capacity > topos.max_volume() {
            return Err(TensorError::TensorVolumeExceeded {
                label: "hypergradient_tape",
                volume: capacity,
                max_volume: topos.max_volume(),
            });
        }
        let gradient = vec![0.0; capacity];
        let summary = GradientSummary::from_slice(&gradient);
        Ok(Self {
            curvature,
            learning_rate,
            rows,
            cols,
            gradient,
            summary: Cell::new(summary),
            summary_dirty: Cell::new(false),
            min_dirty: Cell::new(false),
            max_dirty: Cell::new(false),
            linf_dirty: Cell::new(false),
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
        self.summary_dirty.set(true);
        self.min_dirty.set(true);
        self.max_dirty.set(true);
        self.linf_dirty.set(true);
        &mut self.gradient
    }

    /// Number of finite entries currently tracked by the hypergradient cache.
    pub fn finite_count(&self) -> usize {
        self.summary().count()
    }

    /// Number of non-finite (NaN or Inf) entries present in the gradient.
    pub fn non_finite_count(&self) -> usize {
        let finite = self.summary().count();
        self.gradient.len().saturating_sub(finite)
    }

    /// Fraction of the gradient buffer containing non-finite values.
    pub fn non_finite_ratio(&self) -> f32 {
        let volume = self.gradient.len();
        if volume == 0 {
            0.0
        } else {
            let finite = self.summary().count();
            let non_finite = volume.saturating_sub(finite);
            non_finite as f32 / volume as f32
        }
    }

    /// Returns whether the gradient currently holds any non-finite values.
    pub fn has_non_finite(&self) -> bool {
        self.summary().count() < self.gradient.len()
    }

    /// Summarise the accumulated gradient using basic norm statistics.
    pub fn summary(&self) -> GradientSummary {
        self.ensure_summary_extrema()
    }

    #[inline]
    fn ensure_summary_extrema(&self) -> GradientSummary {
        if self.summary_dirty.get() {
            self.rebuild_summary();
        }
        if self.min_dirty.get() || self.max_dirty.get() || self.linf_dirty.get() {
            let mut summary = self.summary.get();
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut linf = 0.0f32;
            let mut any_finite = false;
            for &value in &self.gradient {
                if !value.is_finite() {
                    continue;
                }
                if !any_finite {
                    any_finite = true;
                    min = value;
                    max = value;
                    linf = value.abs();
                } else {
                    min = min.min(value);
                    max = max.max(value);
                    linf = linf.max(value.abs());
                }
            }
            if !any_finite {
                min = 0.0;
                max = 0.0;
                linf = 0.0;
            }
            summary.min = min;
            summary.max = max;
            summary.linf = linf;
            self.summary.set(summary);
            self.min_dirty.set(false);
            self.max_dirty.set(false);
            self.linf_dirty.set(false);
        }
        self.summary.get()
    }

    #[inline]
    fn rebuild_summary(&self) {
        let summary = GradientSummary::from_slice(&self.gradient);
        self.summary.set(summary);
        self.summary_dirty.set(false);
        self.min_dirty.set(false);
        self.max_dirty.set(false);
        self.linf_dirty.set(false);
    }

    fn record_transition(&self, old: f32, new: f32) {
        if old.to_bits() == new.to_bits() {
            return;
        }
        if self.summary_dirty.get() {
            self.rebuild_summary();
            return;
        }
        let old_finite = old.is_finite();
        let new_finite = new.is_finite();
        if !old_finite && !new_finite {
            return;
        }
        if !old_finite && new_finite {
            let mut summary = self.summary.get();
            let new64 = new as f64;
            let new_abs = new64.abs();
            summary.l1 = (summary.l1 as f64 + new_abs).max(0.0) as f32;
            summary.sum = (summary.sum as f64 + new64) as f32;
            let new_sq = new64 * new64;
            let mut sum_squares = summary.sum_squares as f64 + new_sq;
            if sum_squares < 0.0 {
                sum_squares = 0.0;
            }
            summary.sum_squares = sum_squares as f32;
            summary.l2 = summary.sum_squares.sqrt();
            summary.sum_cubes = (summary.sum_cubes as f64 + new_sq * new64) as f32;
            let mut sum_quartic = summary.sum_quartic as f64 + new_sq * new_sq;
            if sum_quartic < 0.0 {
                sum_quartic = 0.0;
            }
            summary.sum_quartic = sum_quartic as f32;
            let capacity = self.gradient.len();
            summary.count = summary.count.saturating_add(1).min(capacity);
            if new > 0.0 {
                summary.positive_count =
                    summary.positive_count.saturating_add(1).min(summary.count);
            } else if new < 0.0 {
                summary.negative_count =
                    summary.negative_count.saturating_add(1).min(summary.count);
            }
            if new_abs as f32 <= GradientSummary::NEAR_ZERO_EPS {
                summary.near_zero_count =
                    summary.near_zero_count.saturating_add(1).min(summary.count);
            }
            if summary.count == 1 {
                summary.min = new;
                summary.max = new;
                summary.linf = new.abs();
                self.min_dirty.set(false);
                self.max_dirty.set(false);
                self.linf_dirty.set(false);
            } else {
                if !self.min_dirty.get() && new < summary.min {
                    summary.min = new;
                }
                if !self.max_dirty.get() && new > summary.max {
                    summary.max = new;
                }
                if !self.linf_dirty.get() {
                    let new_abs_f32 = new.abs();
                    if new_abs_f32 > summary.linf {
                        summary.linf = new_abs_f32;
                    }
                }
            }
            self.summary.set(summary);
            self.summary_dirty.set(false);
            return;
        }
        if old_finite && !new_finite {
            let mut summary = self.summary.get();
            let old64 = old as f64;
            let old_abs = old64.abs();
            let mut l1 = summary.l1 as f64 - old_abs;
            if l1 < 0.0 {
                l1 = 0.0;
            }
            summary.l1 = l1 as f32;
            summary.sum = (summary.sum as f64 - old64) as f32;
            let old_sq = old64 * old64;
            let mut sum_squares = summary.sum_squares as f64 - old_sq;
            if sum_squares < 0.0 {
                sum_squares = 0.0;
            }
            summary.sum_squares = sum_squares as f32;
            summary.l2 = summary.sum_squares.sqrt();
            summary.sum_cubes = (summary.sum_cubes as f64 - old_sq * old64) as f32;
            let mut sum_quartic = summary.sum_quartic as f64 - old_sq * old_sq;
            if sum_quartic < 0.0 {
                sum_quartic = 0.0;
            }
            summary.sum_quartic = sum_quartic as f32;
            if old > 0.0 {
                summary.positive_count = summary.positive_count.saturating_sub(1);
            } else if old < 0.0 {
                summary.negative_count = summary.negative_count.saturating_sub(1);
            }
            if old.abs() <= GradientSummary::NEAR_ZERO_EPS {
                summary.near_zero_count = summary.near_zero_count.saturating_sub(1);
            }
            summary.count = summary.count.saturating_sub(1);
            if summary.count == 0 {
                summary.min = 0.0;
                summary.max = 0.0;
                summary.linf = 0.0;
                summary.positive_count = 0;
                summary.negative_count = 0;
                summary.near_zero_count = 0;
                self.min_dirty.set(false);
                self.max_dirty.set(false);
                self.linf_dirty.set(false);
            } else {
                self.min_dirty.set(true);
                self.max_dirty.set(true);
                self.linf_dirty.set(true);
            }
            self.summary.set(summary);
            self.summary_dirty.set(false);
            return;
        }
        let mut summary = self.summary.get();
        let old64 = old as f64;
        let new64 = new as f64;
        let mut l1 = summary.l1 as f64 + new64.abs() - old64.abs();
        if l1 < 0.0 {
            l1 = 0.0;
        }
        summary.l1 = l1 as f32;
        summary.sum = (summary.sum as f64 + new64 - old64) as f32;
        let old_sq = old64 * old64;
        let new_sq = new64 * new64;
        let mut sum_squares = summary.sum_squares as f64 + new_sq - old_sq;
        if sum_squares < 0.0 {
            sum_squares = 0.0;
        }
        summary.sum_squares = sum_squares as f32;
        summary.l2 = summary.sum_squares.sqrt();
        let old_cubic = old_sq * old64;
        let new_cubic = new_sq * new64;
        summary.sum_cubes = (summary.sum_cubes as f64 + new_cubic - old_cubic) as f32;
        let old_quartic = old_sq * old_sq;
        let new_quartic = new_sq * new_sq;
        let mut sum_quartic = summary.sum_quartic as f64 + new_quartic - old_quartic;
        if sum_quartic < 0.0 {
            sum_quartic = 0.0;
        }
        summary.sum_quartic = sum_quartic as f32;

        if old > 0.0 {
            summary.positive_count = summary.positive_count.saturating_sub(1);
        } else if old < 0.0 {
            summary.negative_count = summary.negative_count.saturating_sub(1);
        }
        if new > 0.0 {
            summary.positive_count = summary.positive_count.saturating_add(1).min(summary.count);
        } else if new < 0.0 {
            summary.negative_count = summary.negative_count.saturating_add(1).min(summary.count);
        }
        if old.abs() <= GradientSummary::NEAR_ZERO_EPS {
            summary.near_zero_count = summary.near_zero_count.saturating_sub(1);
        }
        if new.abs() <= GradientSummary::NEAR_ZERO_EPS {
            summary.near_zero_count = summary.near_zero_count.saturating_add(1).min(summary.count);
        }
        if new < summary.min {
            summary.min = new;
        } else if old <= summary.min && new > old {
            self.min_dirty.set(true);
        }
        if new > summary.max {
            summary.max = new;
        } else if old >= summary.max && new < old {
            self.max_dirty.set(true);
        }
        let new_abs = new.abs();
        if new_abs > summary.linf {
            summary.linf = new_abs;
        } else {
            let old_abs = old.abs();
            if old_abs >= summary.linf && new_abs < old_abs {
                self.linf_dirty.set(true);
            }
        }
        self.summary.set(summary);
        self.summary_dirty.set(false);
    }

    fn telemetry_snapshot(&self) -> HypergradTelemetry {
        let guard = self.topos();
        let summary = self.summary();
        let volume = self.gradient.len();
        let finite_count = summary.count();
        let non_finite_count = volume.saturating_sub(finite_count);
        let non_finite_ratio = if volume == 0 {
            0.0
        } else {
            non_finite_count as f32 / volume as f32
        };
        HypergradTelemetry {
            summary,
            curvature: self.curvature,
            learning_rate: self.learning_rate,
            saturation: guard.saturation(),
            porosity: guard.porosity(),
            tolerance: guard.tolerance(),
            max_depth: guard.max_depth(),
            max_volume: guard.max_volume(),
            finite_count,
            non_finite_count,
            non_finite_ratio,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Returns a telemetry bundle that mirrors the tape's guard envelope and
    /// accumulated statistics. Automation layers use the payload to translate
    /// hypergrad magnitudes into Desire feedback and WGSL operator hints.
    pub fn telemetry(&self) -> HypergradTelemetry {
        self.telemetry_snapshot()
    }

    /// Builds a Desire gradient interpretation by pairing the hypergrad tape's
    /// summary with the provided Euclidean gradient statistics.
    pub fn desire_interpretation(&self, real: GradientSummary) -> DesireGradientInterpretation {
        DesireGradientInterpretation::from_summaries(self.summary(), real)
    }

    /// Derives a Desire control packet scaled by `gain` using the tape's
    /// telemetry and the supplied Euclidean gradient summary.
    pub fn desire_control_with_gain(
        &self,
        real: GradientSummary,
        gain: f32,
    ) -> DesireGradientControl {
        DesireGradientControl::from_interpretation_with_gain(self.desire_interpretation(real), gain)
    }

    /// Convenience wrapper over [`desire_control_with_gain`] that applies the
    /// recommended gain of `1.0`.
    pub fn desire_control(&self, real: GradientSummary) -> DesireGradientControl {
        self.desire_control_with_gain(real, 1.0)
    }

    /// Returns the guard topos enforcing open-cartesian safety constraints.
    pub fn topos(&self) -> &topos::OpenCartesianTopos {
        &self.topos
    }

    /// Scales the learning rate used by subsequent hyperbolic updates.
    pub fn scale_learning_rate(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            let next = self.learning_rate * factor;
            if next.is_finite() && next > 0.0 {
                self.learning_rate = next;
            }
        }
    }

    /// Scales the accumulated gradient by `factor` while keeping the cached
    /// summary in sync. Non-finite scaling factors reset the tape to a clean
    /// state to avoid propagating NaNs or infinities through downstream
    /// telemetry.
    pub fn scale_gradient(&mut self, factor: f32) {
        if !factor.is_finite() {
            self.reset();
            return;
        }
        if (factor - 1.0).abs() <= f32::EPSILON {
            return;
        }
        for idx in 0..self.gradient.len() {
            let old = self.gradient[idx];
            let candidate = old * factor;
            let new = self.topos.saturate(candidate);
            if old.to_bits() != new.to_bits() {
                self.gradient[idx] = new;
                self.record_transition(old, new);
            } else {
                self.gradient[idx] = new;
            }
        }
    }

    /// Scales the accumulated gradient through a tensor utility backend before
    /// applying the tape's topos saturation and summary accounting.
    pub fn scale_gradient_with_backend(
        &mut self,
        factor: f32,
        backend: TensorUtilBackend,
    ) -> PureResult<()> {
        if !factor.is_finite() {
            self.reset();
            return Ok(());
        }
        if (factor - 1.0).abs() <= f32::EPSILON {
            return Ok(());
        }
        let gradient = Tensor::from_vec(self.rows, self.cols, self.gradient.clone())?;
        let scaled = gradient.scale_with_backend(factor, backend)?;
        let mut next_gradient = self.gradient.clone();
        for (idx, &candidate) in scaled.data().iter().enumerate() {
            next_gradient[idx] =
                self.checked_saturated_gradient("hypergrad_scaled_gradient", candidate)?;
        }
        self.commit_gradient(next_gradient);
        Ok(())
    }

    /// Rescales the gradient so its root-mean-square matches `target_rms` and
    /// returns the factor that was applied. When the gradient is dormant the
    /// tape stays untouched and the method returns ``0.0`` to signal no-op
    /// scaling.
    pub fn rescale_rms(&mut self, target_rms: f32) -> f32 {
        if !target_rms.is_finite() {
            return 1.0;
        }
        if target_rms <= 0.0 {
            self.reset();
            return 0.0;
        }
        let summary = self.summary();
        let current_rms = summary.rms();
        if current_rms <= 1e-12 {
            return 0.0;
        }
        let factor = target_rms / current_rms;
        self.scale_gradient(factor);
        factor
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

    fn checked_saturated_gradient(&self, label: &'static str, candidate: f32) -> PureResult<f32> {
        if !candidate.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label,
                value: candidate,
            });
        }
        Ok(self.topos.saturate(candidate))
    }

    fn commit_gradient(&mut self, next_gradient: Vec<f32>) {
        for (idx, new) in next_gradient.into_iter().enumerate() {
            let old = self.gradient[idx];
            self.gradient[idx] = new;
            if old.to_bits() != new.to_bits() {
                self.record_transition(old, new);
            }
        }
    }

    fn updated_weights_tensor(&self, weights: &Tensor) -> PureResult<Tensor> {
        let tolerance = self.topos.tolerance();
        let mut updated = weights.clone();
        for (value, grad) in updated.data_mut().iter_mut().zip(self.gradient.iter()) {
            let original = *value;
            let denom = 1.0 - self.curvature * original * original;
            let step = self.learning_rate / denom.abs().max(tolerance);
            if !step.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "hypergrad_step",
                    value: step,
                });
            }
            let delta = step * *grad;
            if !delta.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "hypergrad_delta",
                    value: delta,
                });
            }
            let next = original - delta;
            if !next.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "hypergrad_update",
                    value: next,
                });
            }
            *value = self.topos.saturate(next);
        }
        Ok(updated)
    }

    /// Clears the accumulated gradient back to zero.
    pub fn reset(&mut self) {
        for idx in 0..self.gradient.len() {
            let old = self.gradient[idx];
            let new = 0.0f32;
            self.gradient[idx] = new;
            if old.to_bits() != new.to_bits() {
                self.record_transition(old, new);
            }
        }
    }

    /// Accumulates a Euclidean tensor inside the hyperbolic tape using
    /// the standard conformal factor for the Poincaré ball.
    pub fn accumulate_wave(&mut self, tensor: &Tensor) -> PureResult<()> {
        self.accumulate_wave_with_backend(tensor, TensorUtilBackend::Auto)
    }

    /// Accumulates a Euclidean tensor using an explicit tensor utility backend
    /// for the conformal update while keeping CPU-side guards and summaries.
    pub fn accumulate_wave_with_backend(
        &mut self,
        tensor: &Tensor,
        backend: TensorUtilBackend,
    ) -> PureResult<()> {
        self.assert_tensor_shape(tensor)?;
        self.topos.guard_tensor("hypergrad_wave", tensor)?;
        let tolerance = self.topos.tolerance();
        let values = tensor.data();
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(backend, TensorUtilBackend::GpuWgpu)
                && !values.is_empty()
                && wgpu_dense::is_available()
            {
                match wgpu_dense::hypergrad_accumulate_wave(
                    &self.gradient,
                    values,
                    self.rows,
                    self.cols,
                    self.curvature,
                    tolerance,
                    self.topos.saturation(),
                    self.topos.porosity(),
                ) {
                    Ok(next_gradient) => {
                        for &candidate in &next_gradient {
                            if !candidate.is_finite() {
                                return Err(TensorError::NonFiniteValue {
                                    label: "hypergrad_accumulate_wave",
                                    value: candidate,
                                });
                            }
                        }
                        self.commit_gradient(next_gradient);
                        crate::emit_tensor_op(
                            "hypergrad_accumulate_wave",
                            &[tensor.rows, tensor.cols],
                            &[self.rows, self.cols],
                        );
                        emit_basic_tensor_op_meta(
                            "hypergrad_accumulate_wave",
                            tensor.rows,
                            tensor.cols,
                            self.rows,
                            self.cols,
                            tensor.layout,
                            "wgpu_dense",
                            backend.label(),
                            "hypergrad.wgpu_accumulate_wave",
                            "gradient_tape_accumulate",
                            |data| {
                                data.insert(
                                    "curvature".to_string(),
                                    serde_json::json!(self.curvature),
                                );
                                data.insert(
                                    "learning_rate".to_string(),
                                    serde_json::json!(self.learning_rate),
                                );
                                data.insert("tolerance".to_string(), serde_json::json!(tolerance));
                                data.insert(
                                    "saturation".to_string(),
                                    serde_json::json!(self.topos.saturation()),
                                );
                                data.insert(
                                    "porosity".to_string(),
                                    serde_json::json!(self.topos.porosity()),
                                );
                                data.insert(
                                    "input_non_finite_values".to_string(),
                                    serde_json::json!(values
                                        .iter()
                                        .filter(|value| !value.is_finite())
                                        .count()),
                                );
                                insert_gradient_summary_meta(data, "gradient", self.summary());
                            },
                        );
                        return Ok(());
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(strict_gpu_fallback_error(
                            "wgpu",
                            "hypergrad_accumulate_wave",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut next_gradient = self.gradient.clone();
        for (idx, &value) in values.iter().enumerate() {
            let denom = 1.0 - self.curvature * value * value;
            let update = value / denom.abs().max(tolerance);
            let old = self.gradient[idx];
            let candidate = old + update;
            next_gradient[idx] =
                self.checked_saturated_gradient("hypergrad_accumulate_wave", candidate)?;
        }
        self.commit_gradient(next_gradient);
        crate::emit_tensor_op(
            "hypergrad_accumulate_wave",
            &[tensor.rows, tensor.cols],
            &[self.rows, self.cols],
        );
        emit_basic_tensor_op_meta(
            "hypergrad_accumulate_wave",
            tensor.rows,
            tensor.cols,
            self.rows,
            self.cols,
            tensor.layout,
            "cpu",
            backend.label(),
            "scalar",
            "gradient_tape_accumulate",
            |data| {
                data.insert("curvature".to_string(), serde_json::json!(self.curvature));
                data.insert(
                    "learning_rate".to_string(),
                    serde_json::json!(self.learning_rate),
                );
                data.insert("tolerance".to_string(), serde_json::json!(tolerance));
                data.insert(
                    "saturation".to_string(),
                    serde_json::json!(self.topos.saturation()),
                );
                data.insert(
                    "input_non_finite_values".to_string(),
                    serde_json::json!(values.iter().filter(|value| !value.is_finite()).count()),
                );
                #[cfg(feature = "wgpu")]
                if let Some(message) = wgpu_failure.as_deref() {
                    data.insert(
                        "fallback".to_string(),
                        wgpu_runtime_fallback_meta(
                            "cpu",
                            WGPU_RUNTIME_FALLBACK_REASON,
                            Some(message),
                        ),
                    );
                }
                insert_gradient_summary_meta(data, "gradient", self.summary());
            },
        );
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
        let prediction_data = prediction.data();
        let target_data = target.data();
        let mut next_gradient = self.gradient.clone();
        for idx in 0..self.gradient.len() {
            let delta = prediction_data[idx] - target_data[idx];
            let old = self.gradient[idx];
            let candidate = old + delta;
            next_gradient[idx] =
                self.checked_saturated_gradient("hypergrad_accumulate_pair", candidate)?;
        }
        self.commit_gradient(next_gradient);
        crate::emit_tensor_op(
            "hypergrad_accumulate_pair",
            &[prediction.rows, prediction.cols, target.rows, target.cols],
            &[self.rows, self.cols],
        );
        emit_cpu_tensor_op_meta(
            "hypergrad_accumulate_pair",
            prediction.rows,
            prediction.cols,
            self.rows,
            self.cols,
            prediction.layout,
            "gradient_tape_accumulate",
            |data| {
                data.insert("curvature".to_string(), serde_json::json!(self.curvature));
                data.insert(
                    "learning_rate".to_string(),
                    serde_json::json!(self.learning_rate),
                );
                data.insert("rhs_rows".to_string(), serde_json::json!(target.rows));
                data.insert("rhs_cols".to_string(), serde_json::json!(target.cols));
                data.insert(
                    "rhs_layout".to_string(),
                    serde_json::json!(target.layout.as_str()),
                );
                data.insert(
                    "prediction_non_finite_values".to_string(),
                    serde_json::json!(prediction_data
                        .iter()
                        .filter(|value| !value.is_finite())
                        .count()),
                );
                data.insert(
                    "target_non_finite_values".to_string(),
                    serde_json::json!(target_data
                        .iter()
                        .filter(|value| !value.is_finite())
                        .count()),
                );
                insert_gradient_summary_meta(data, "gradient", self.summary());
            },
        );
        Ok(())
    }

    /// Applies the accumulated gradient to the provided tensor and reprojects it
    /// into the Poincaré ball. The gradient buffer is cleared afterwards so the
    /// tape can keep streaming samples without triggering a traceback.
    pub fn apply(&mut self, weights: &mut Tensor) -> PureResult<()> {
        self.apply_with_backend(weights, TensorUtilBackend::Auto)
    }

    /// Applies the accumulated gradient using an explicit tensor utility backend
    /// for the final Poincaré projection.
    pub fn apply_with_backend(
        &mut self,
        weights: &mut Tensor,
        backend: TensorUtilBackend,
    ) -> PureResult<()> {
        self.assert_tensor_shape(weights)?;
        self.topos.guard_tensor("hypergrad_weights", weights)?;
        let updated_weights = self.updated_weights_tensor(weights)?;
        crate::emit_tensor_op(
            "hypergrad_apply_update",
            &[weights.rows, weights.cols, self.rows, self.cols],
            &[weights.rows, weights.cols],
        );
        emit_cpu_tensor_op_meta(
            "hypergrad_apply_update",
            weights.rows,
            weights.cols,
            weights.rows,
            weights.cols,
            weights.layout,
            "gradient_tape_update",
            |data| {
                data.insert("curvature".to_string(), serde_json::json!(self.curvature));
                data.insert(
                    "learning_rate".to_string(),
                    serde_json::json!(self.learning_rate),
                );
                data.insert(
                    "projection_requested_backend".to_string(),
                    serde_json::json!(backend.label()),
                );
                data.insert(
                    "weight_non_finite_values".to_string(),
                    serde_json::json!(weights
                        .data()
                        .iter()
                        .filter(|value| !value.is_finite())
                        .count()),
                );
                insert_gradient_summary_meta(data, "gradient", self.summary());
            },
        );
        let projected =
            updated_weights.project_to_poincare_with_backend(self.curvature, backend)?;
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
        let first = stages
            .next()
            .ok_or(TensorError::EmptyInput("barycenter_intermediates"))?;
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
            let next = self.learning_rate * factor;
            if next.is_finite() && next > 0.0 {
                self.learning_rate = next;
            }
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

    /// Scale the accumulated gradient through a tensor utility backend.
    pub fn scale_gradient_with_backend(
        &mut self,
        factor: f32,
        backend: TensorUtilBackend,
    ) -> PureResult<()> {
        if !factor.is_finite() {
            return Ok(());
        }
        if (factor - 1.0).abs() <= f32::EPSILON {
            return Ok(());
        }
        for value in self.gradient.iter().copied() {
            let scaled = value * factor;
            if !scaled.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "realgrad_scaled_gradient",
                    value: scaled,
                });
            }
        }
        let gradient = Tensor::from_vec(self.rows, self.cols, self.gradient.clone())?;
        let scaled = gradient.scale_with_backend(factor, backend)?;
        self.gradient.clear();
        self.gradient.extend_from_slice(scaled.data());
        Ok(())
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

    fn validate_update(&self, weights: &Tensor) -> PureResult<()> {
        for (&weight, &grad) in weights.data().iter().zip(self.gradient.iter()) {
            let delta = self.learning_rate * grad;
            if !delta.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "realgrad_delta",
                    value: delta,
                });
            }
            let next = weight - delta;
            if !next.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "realgrad_update",
                    value: next,
                });
            }
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
        crate::emit_tensor_op(
            "realgrad_accumulate_wave",
            &[tensor.rows, tensor.cols],
            &[self.rows, self.cols],
        );
        emit_cpu_tensor_op_meta(
            "realgrad_accumulate_wave",
            tensor.rows,
            tensor.cols,
            self.rows,
            self.cols,
            tensor.layout,
            "gradient_tape_accumulate",
            |data| {
                data.insert(
                    "learning_rate".to_string(),
                    serde_json::json!(self.learning_rate),
                );
                data.insert(
                    "input_non_finite_values".to_string(),
                    serde_json::json!(tensor
                        .data()
                        .iter()
                        .filter(|value| !value.is_finite())
                        .count()),
                );
                insert_gradient_summary_meta(data, "gradient", self.summary());
            },
        );
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
        crate::emit_tensor_op(
            "realgrad_accumulate_pair",
            &[prediction.rows, prediction.cols, target.rows, target.cols],
            &[self.rows, self.cols],
        );
        emit_cpu_tensor_op_meta(
            "realgrad_accumulate_pair",
            prediction.rows,
            prediction.cols,
            self.rows,
            self.cols,
            prediction.layout,
            "gradient_tape_accumulate",
            |data| {
                data.insert(
                    "learning_rate".to_string(),
                    serde_json::json!(self.learning_rate),
                );
                data.insert("rhs_rows".to_string(), serde_json::json!(target.rows));
                data.insert("rhs_cols".to_string(), serde_json::json!(target.cols));
                data.insert(
                    "rhs_layout".to_string(),
                    serde_json::json!(target.layout.as_str()),
                );
                data.insert(
                    "prediction_non_finite_values".to_string(),
                    serde_json::json!(prediction
                        .data()
                        .iter()
                        .filter(|value| !value.is_finite())
                        .count()),
                );
                data.insert(
                    "target_non_finite_values".to_string(),
                    serde_json::json!(target
                        .data()
                        .iter()
                        .filter(|value| !value.is_finite())
                        .count()),
                );
                insert_gradient_summary_meta(data, "gradient", self.summary());
            },
        );
        Ok(())
    }

    /// Apply the accumulated gradient to the provided weights and clear it.
    pub fn apply(&mut self, weights: &mut Tensor) -> PureResult<()> {
        self.apply_with_backend(weights, TensorUtilBackend::Auto)
    }

    /// Apply the accumulated gradient using an explicit tensor utility backend.
    pub fn apply_with_backend(
        &mut self,
        weights: &mut Tensor,
        backend: TensorUtilBackend,
    ) -> PureResult<()> {
        self.assert_tensor_shape(weights)?;
        self.validate_update(weights)?;
        let gradient = Tensor::from_vec(self.rows, self.cols, self.gradient.clone())?;
        weights.add_scaled_with_backend(&gradient, -self.learning_rate, backend)?;
        self.reset();
        Ok(())
    }
}

const PARALLEL_GEMM_THRESHOLD: usize = 32 * 32 * 32;
const MATMUL_BLOCK_INNER: usize = 64;
const MATMUL_BLOCK_COLS: usize = 128;
const PACKED_MATMUL_COL_BLOCK: usize = 4;

#[inline]
fn should_parallelize(rows: usize, inner: usize, cols: usize) -> bool {
    if current_num_threads() <= 1 {
        return false;
    }
    rows.saturating_mul(inner).saturating_mul(cols) >= PARALLEL_GEMM_THRESHOLD
}

#[inline]
fn fused_axpy(dst: &mut [f32], rhs: &[f32], scale: f32) {
    if scale == 0.0 {
        return;
    }
    debug_assert_eq!(dst.len(), rhs.len());
    let len = dst.len();
    let chunk_len = len - (len % 4);
    let (dst_head, dst_tail) = dst.split_at_mut(chunk_len);
    let (rhs_head, rhs_tail) = rhs.split_at(chunk_len);

    for (dst_chunk, rhs_chunk) in dst_head.chunks_exact_mut(4).zip(rhs_head.chunks_exact(4)) {
        dst_chunk[0] += scale * rhs_chunk[0];
        dst_chunk[1] += scale * rhs_chunk[1];
        dst_chunk[2] += scale * rhs_chunk[2];
        dst_chunk[3] += scale * rhs_chunk[3];
    }

    for (dst_value, &rhs_value) in dst_tail.iter_mut().zip(rhs_tail.iter()) {
        *dst_value += scale * rhs_value;
    }
}

#[inline]
fn matmul_row(dst_row: &mut [f32], lhs_row: &[f32], rhs: &[f32], inner: usize, cols: usize) {
    dst_row.fill(0.0);
    let col_block = MATMUL_BLOCK_COLS.max(1);
    let inner_block = MATMUL_BLOCK_INNER.max(1);
    let mut col_start = 0;
    while col_start < cols {
        let col_end = (col_start + col_block).min(cols);
        let dst_block = &mut dst_row[col_start..col_end];
        let mut k_start = 0;
        while k_start < inner {
            let k_end = (k_start + inner_block).min(inner);
            for k in k_start..k_end {
                let scale = lhs_row[k];
                if scale == 0.0 {
                    continue;
                }
                let rhs_slice = &rhs[k * cols + col_start..k * cols + col_end];
                fused_axpy(dst_block, rhs_slice, scale);
            }
            k_start = k_end;
        }
        col_start = col_end;
    }
}

#[inline]
fn matmul_naive_serial(dst: &mut [f32], lhs: &[f32], rhs: &[f32], inner: usize, cols: usize) {
    for (dst_row, lhs_row) in dst.chunks_mut(cols).zip(lhs.chunks(inner)) {
        matmul_row(dst_row, lhs_row, rhs, inner, cols);
    }
}

#[inline]
fn matmul_naive_parallel(dst: &mut [f32], lhs: &[f32], rhs: &[f32], inner: usize, cols: usize) {
    dst.par_chunks_mut(cols)
        .zip(lhs.par_chunks(inner))
        .for_each(|(dst_row, lhs_row)| matmul_row(dst_row, lhs_row, rhs, inner, cols));
}

fn matmul_lhs_transpose_scaled_naive_into(
    dst: &mut [f32],
    lhs: &[f32],
    rhs: &[f32],
    lhs_rows: usize,
    lhs_cols: usize,
    rhs_cols: usize,
    scale: f32,
) {
    debug_assert_eq!(dst.len(), lhs_cols * rhs_cols);
    debug_assert_eq!(lhs.len(), lhs_rows * lhs_cols);
    debug_assert_eq!(rhs.len(), lhs_rows * rhs_cols);
    for out_row in 0..lhs_cols {
        let dst_offset = out_row * rhs_cols;
        for out_col in 0..rhs_cols {
            let mut acc = 0.0f32;
            for k in 0..lhs_rows {
                acc += lhs[k * lhs_cols + out_row] * rhs[k * rhs_cols + out_col];
            }
            dst[dst_offset + out_col] = acc * scale;
        }
    }
}

#[inline]
fn dot_unrolled(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let chunks = lhs.len() / 4;
    for i in 0..chunks {
        let base = i * 4;
        acc0 += lhs[base] * rhs[base];
        acc1 += lhs[base + 1] * rhs[base + 1];
        acc2 += lhs[base + 2] * rhs[base + 2];
        acc3 += lhs[base + 3] * rhs[base + 3];
    }

    let mut sum = acc0 + acc1 + acc2 + acc3;
    for i in chunks * 4..lhs.len() {
        sum += lhs[i] * rhs[i];
    }
    sum
}

#[inline]
fn matmul_naive_packed_row(
    dst_row: &mut [f32],
    lhs_row: &[f32],
    inner: usize,
    cols: usize,
    rhs: &[f32],
) {
    let mut col = 0;
    while col + PACKED_MATMUL_COL_BLOCK <= cols {
        let base0 = col * inner;
        let base1 = base0 + inner;
        let base2 = base1 + inner;
        let base3 = base2 + inner;
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        let mut k = 0;
        while k + 4 <= inner {
            let l0 = lhs_row[k];
            let l1 = lhs_row[k + 1];
            let l2 = lhs_row[k + 2];
            let l3 = lhs_row[k + 3];

            acc0 += l0 * rhs[base0 + k];
            acc1 += l0 * rhs[base1 + k];
            acc2 += l0 * rhs[base2 + k];
            acc3 += l0 * rhs[base3 + k];

            acc0 += l1 * rhs[base0 + k + 1];
            acc1 += l1 * rhs[base1 + k + 1];
            acc2 += l1 * rhs[base2 + k + 1];
            acc3 += l1 * rhs[base3 + k + 1];

            acc0 += l2 * rhs[base0 + k + 2];
            acc1 += l2 * rhs[base1 + k + 2];
            acc2 += l2 * rhs[base2 + k + 2];
            acc3 += l2 * rhs[base3 + k + 2];

            acc0 += l3 * rhs[base0 + k + 3];
            acc1 += l3 * rhs[base1 + k + 3];
            acc2 += l3 * rhs[base2 + k + 3];
            acc3 += l3 * rhs[base3 + k + 3];

            k += 4;
        }

        while k < inner {
            let lhs_val = lhs_row[k];
            acc0 += lhs_val * rhs[base0 + k];
            acc1 += lhs_val * rhs[base1 + k];
            acc2 += lhs_val * rhs[base2 + k];
            acc3 += lhs_val * rhs[base3 + k];
            k += 1;
        }

        dst_row[col] = acc0;
        dst_row[col + 1] = acc1;
        dst_row[col + 2] = acc2;
        dst_row[col + 3] = acc3;
        col += PACKED_MATMUL_COL_BLOCK;
    }
    while col < cols {
        let base = col * inner;
        let column = &rhs[base..base + inner];
        dst_row[col] = dot_unrolled(lhs_row, column);
        col += 1;
    }
}

#[inline]
fn matmul_naive_packed_serial(
    dst: &mut [f32],
    lhs: &[f32],
    inner: usize,
    cols: usize,
    rhs: &[f32],
) {
    for (dst_row, lhs_row) in dst.chunks_mut(cols).zip(lhs.chunks(inner)) {
        matmul_naive_packed_row(dst_row, lhs_row, inner, cols, rhs);
    }
}

#[inline]
fn matmul_naive_packed_parallel(
    dst: &mut [f32],
    lhs: &[f32],
    inner: usize,
    cols: usize,
    rhs: &[f32],
) {
    dst.par_chunks_mut(cols)
        .zip(lhs.par_chunks(inner))
        .for_each(|(dst_row, lhs_row)| {
            matmul_naive_packed_row(dst_row, lhs_row, inner, cols, rhs);
        });
}

fn matmul_naive_into(
    dst: &mut [f32],
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) {
    debug_assert_eq!(dst.len(), rows * cols);
    if rows == 0 || cols == 0 || inner == 0 {
        dst.fill(0.0);
        return;
    }
    let parallel = should_parallelize(rows, inner, cols) && !determinism::lock_reduction_order();
    if parallel {
        matmul_naive_parallel(dst, lhs, rhs, inner, cols);
    } else {
        matmul_naive_serial(dst, lhs, rhs, inner, cols);
    }
}

fn matmul_naive_packed_into(
    dst: &mut [f32],
    lhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
    packed: &PackedB,
) {
    debug_assert_eq!(dst.len(), rows * cols);
    debug_assert_eq!(inner, packed.inner());
    if rows == 0 || cols == 0 || inner == 0 {
        dst.fill(0.0);
        return;
    }

    let rhs = packed.as_slice();
    match packed.layout() {
        PackedLayout::ColMajor | PackedLayout::Tiled { .. } => {
            let parallel =
                should_parallelize(rows, inner, cols) && !determinism::lock_reduction_order();
            if parallel {
                matmul_naive_packed_parallel(dst, lhs, inner, cols, rhs);
            } else {
                matmul_naive_packed_serial(dst, lhs, inner, cols, rhs);
            }
        }
    }
}

fn matmul_naive(lhs: &[f32], rhs: &[f32], rows: usize, inner: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    matmul_naive_into(&mut out, lhs, rhs, rows, inner, cols);
    out
}

#[allow(clippy::too_many_arguments)]
fn fused_attention_cpu(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    contexts: usize,
    sequence: usize,
    head_dim: usize,
    scale: f32,
    z_bias: Option<&[f32]>,
    attn_bias: Option<&[f32]>,
) -> PureResult<Vec<f32>> {
    let total = contexts * sequence * head_dim;
    let mut output = vec![0.0f32; total];
    let mut accum = vec![0.0f32; head_dim];

    for context in 0..contexts {
        let context_offset = context * sequence;
        for query_idx in 0..sequence {
            accum.iter_mut().for_each(|value| *value = 0.0);
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            let query_row = context_offset + query_idx;
            let query_offset = query_row * head_dim;

            for key_idx in 0..sequence {
                let key_row = context_offset + key_idx;
                let key_offset = key_row * head_dim;
                let mut dot = 0.0f32;
                for dim in 0..head_dim {
                    dot += queries[query_offset + dim] * keys[key_offset + dim];
                    Tensor::validate_finite_tensor_util_value("scaled_dot_attention_dot", dot)?;
                }

                let mut logit = dot * scale;
                Tensor::validate_finite_tensor_util_value("scaled_dot_attention_logit", logit)?;
                if let Some(bias) = z_bias {
                    logit += bias[context_offset + key_idx];
                    Tensor::validate_finite_tensor_util_value("scaled_dot_attention_logit", logit)?;
                }
                if let Some(bias) = attn_bias {
                    logit += bias[query_row * sequence + key_idx];
                    Tensor::validate_finite_tensor_util_value("scaled_dot_attention_logit", logit)?;
                }

                let new_max = running_max.max(logit);
                let scaled_sum = if running_sum > 0.0 {
                    running_sum * (running_max - new_max).exp()
                } else {
                    0.0
                };
                let exp_curr = (logit - new_max).exp();
                let denom = scaled_sum + exp_curr;
                Tensor::validate_finite_tensor_util_value("scaled_dot_attention_denom", denom)?;
                let alpha = if denom > 0.0 { scaled_sum / denom } else { 0.0 };
                let weight = if denom > 0.0 { exp_curr / denom } else { 0.0 };
                Tensor::validate_finite_tensor_util_value("scaled_dot_attention_weight", alpha)?;
                Tensor::validate_finite_tensor_util_value("scaled_dot_attention_weight", weight)?;
                running_max = new_max;
                running_sum = denom;

                for dim in 0..head_dim {
                    accum[dim] = accum[dim] * alpha + weight * values[key_offset + dim];
                    Tensor::validate_finite_tensor_util_value(
                        "scaled_dot_attention_accumulator",
                        accum[dim],
                    )?;
                }
            }

            output[query_offset..query_offset + head_dim].copy_from_slice(&accum);
        }
    }

    Ok(output)
}

fn cpu_row_softmax_hardmax(data: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let expected = rows.saturating_mul(cols);
    if data.len() != expected || expected == 0 {
        return (vec![0.0; expected], vec![0.0; expected]);
    }

    let mut softmax = vec![0.0f32; expected];
    let mut hardmax = vec![0.0f32; expected];

    for row in 0..rows {
        let offset = row * cols;
        let row_slice = &data[offset..offset + cols];

        let mut max_value = f32::NEG_INFINITY;
        let mut max_index = 0usize;
        let mut found = false;
        for (index, &value) in row_slice.iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            if !found || value > max_value {
                max_value = value;
                max_index = index;
                found = true;
            }
        }

        let base = if found && max_value.is_finite() {
            max_value
        } else {
            0.0
        };

        let mut denom = 0.0f32;
        for (index, &value) in row_slice.iter().enumerate() {
            let weight = if !found || value.is_nan() {
                0.0
            } else {
                (value - base).exp()
            };
            softmax[offset + index] = weight;
            denom += weight;
        }

        if found {
            if denom.is_finite() && denom > f32::EPSILON {
                let inv = denom.recip();
                for value in &mut softmax[offset..offset + cols] {
                    *value *= inv;
                }
            } else {
                for value in &mut softmax[offset..offset + cols] {
                    *value = 0.0;
                }
                softmax[offset + max_index] = 1.0;
            }
            hardmax[offset + max_index] = 1.0;
        }
    }

    (softmax, hardmax)
}

fn row_softmax_cpu(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    cpu_row_softmax_hardmax(data, rows, cols).0
}

#[allow(dead_code)]
fn row_hardmax_cpu(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    cpu_row_softmax_hardmax(data, rows, cols).1
}

fn scale_inplace(data: &mut [f32], scale: f32) {
    for value in data {
        *value *= scale;
    }
}

fn add_bias_inplace(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    assert_eq!(data.len(), rows * cols);
    assert_eq!(bias.len(), cols);
    if cols == 0 {
        return;
    }
    for row in data.chunks_mut(cols) {
        for (value, &bias) in row.iter_mut().zip(bias.iter()) {
            *value += bias;
        }
    }
}

fn add_bias_relu_inplace(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    assert_eq!(data.len(), rows * cols);
    assert_eq!(bias.len(), cols);
    if cols == 0 {
        return;
    }
    for row in data.chunks_mut(cols) {
        for (value, &bias) in row.iter_mut().zip(bias.iter()) {
            let sum = *value + bias;
            *value = if sum > 0.0 { sum } else { 0.0 };
        }
    }
}

fn gelu(x: f32) -> f32 {
    const COEFF: f32 = 0.044_715;
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let x_cubed = x * x * x;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x_cubed)).tanh())
}

fn gelu_prime(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const KAPPA: f32 = 0.044_715;
    let x2 = x * x;
    let inner = SQRT_2_OVER_PI * (x + KAPPA * x * x2);
    let t = inner.tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * KAPPA * x2)
}

fn add_bias_gelu_inplace(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    assert_eq!(data.len(), rows * cols);
    assert_eq!(bias.len(), cols);
    if cols == 0 {
        return;
    }
    for row in data.chunks_mut(cols) {
        for (value, &bias) in row.iter_mut().zip(bias.iter()) {
            *value = gelu(*value + bias);
        }
    }
}

fn add_bias_residual_relu_inplace(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    bias: &[f32],
    residual: &[f32],
) {
    assert_eq!(data.len(), rows * cols);
    assert_eq!(bias.len(), cols);
    assert_eq!(residual.len(), rows * cols);
    if cols == 0 {
        return;
    }
    for (row, residual_row) in data.chunks_mut(cols).zip(residual.chunks(cols)) {
        for ((value, &bias), &residual) in row.iter_mut().zip(bias.iter()).zip(residual_row.iter())
        {
            let sum = *value + bias + residual;
            *value = if sum > 0.0 { sum } else { 0.0 };
        }
    }
}

fn add_bias_residual_gelu_inplace(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    bias: &[f32],
    residual: &[f32],
) {
    assert_eq!(data.len(), rows * cols);
    assert_eq!(bias.len(), cols);
    assert_eq!(residual.len(), rows * cols);
    if cols == 0 {
        return;
    }
    for (row, residual_row) in data.chunks_mut(cols).zip(residual.chunks(cols)) {
        for ((value, &bias), &residual) in row.iter_mut().zip(bias.iter()).zip(residual_row.iter())
        {
            *value = gelu(*value + bias + residual);
        }
    }
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

#[cfg(feature = "wgpu")]
fn matmul_scaled_wgpu(
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
    scale: f32,
) -> PureResult<Vec<f32>> {
    wgpu_dense::matmul_scaled(lhs, rhs, rows, inner, cols, scale).map_err(|message| {
        TensorError::BackendFailure {
            backend: "wgpu",
            message,
        }
    })
}

#[cfg(feature = "wgpu")]
fn matmul_lhs_transpose_scaled_wgpu(
    lhs: &[f32],
    rhs: &[f32],
    lhs_rows: usize,
    lhs_cols: usize,
    rhs_cols: usize,
    scale: f32,
) -> PureResult<Vec<f32>> {
    wgpu_dense::matmul_lhs_transpose_scaled(lhs, rhs, lhs_rows, lhs_cols, rhs_cols, scale).map_err(
        |message| TensorError::BackendFailure {
            backend: "wgpu",
            message,
        },
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop, clippy::useless_vec)]

    use super::*;
    use ndarray::Array2;
    use std::ffi::OsString;
    use std::ptr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};

    struct DlpackTestCtx {
        drops: Arc<AtomicUsize>,
        data: Box<[f32]>,
        shape: Box<[i64]>,
        strides: Option<Box<[i64]>>,
    }

    unsafe extern "C" fn dlpack_test_deleter(ptr: *mut DLManagedTensor) {
        if ptr.is_null() {
            return;
        }
        let mut boxed = Box::from_raw(ptr);
        if !boxed.manager_ctx.is_null() {
            let ctx = Box::from_raw(boxed.manager_ctx as *mut DlpackTestCtx);
            ctx.drops.fetch_add(1, Ordering::SeqCst);
            drop(ctx);
            boxed.manager_ctx = ptr::null_mut();
        }
        boxed.deleter = None;
        drop(boxed);
    }

    #[track_caller]
    fn unwrap_ok<T, E: core::fmt::Debug>(result: Result<T, E>) -> T {
        match result {
            Ok(value) => value,
            Err(error) => panic!("expected Ok(..), got Err({error:?})"),
        }
    }

    #[track_caller]
    fn unwrap_err<T, E: core::fmt::Debug>(result: Result<T, E>) -> E {
        match result {
            Ok(_) => panic!("expected Err(..), got Ok(..)"),
            Err(error) => error,
        }
    }

    #[cfg(feature = "wgpu")]
    #[track_caller]
    fn assert_tensor_close(lhs: &Tensor, rhs: &Tensor, tolerance: f32) {
        assert_eq!(lhs.shape(), rhs.shape());
        for (idx, (&left, &right)) in lhs.data().iter().zip(rhs.data()).enumerate() {
            let delta = (left - right).abs();
            assert!(
                delta <= tolerance,
                "tensor mismatch at {idx}: left={left} right={right} delta={delta} tolerance={tolerance}"
            );
        }
    }

    #[cfg(feature = "wgpu")]
    fn wgpu_unavailable(error: &TensorError) -> bool {
        matches!(error, TensorError::BackendFailure { backend, .. } if *backend == "wgpu")
    }

    #[cfg(feature = "wgpu")]
    fn run_wgpu_runtime_tests() -> bool {
        std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock available")
    }

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        static OBSERVER_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        OBSERVER_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("observer lock available")
    }

    struct EnvVarRestore {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarRestore {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                previous: std::env::var_os(key),
            }
        }
    }

    impl Drop for EnvVarRestore {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                unsafe { std::env::set_var(self.key, previous) };
            } else {
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    fn with_strict_gpu_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _lock = env_lock();
        let _restore = EnvVarRestore::capture("SPIRALTORCH_STRICT_GPU");
        if let Some(value) = value {
            unsafe { std::env::set_var("SPIRALTORCH_STRICT_GPU", value) };
        } else {
            unsafe { std::env::remove_var("SPIRALTORCH_STRICT_GPU") };
        }
        f()
    }

    #[test]
    fn strict_gpu_env_matches_rankk_truthy_contract() {
        with_strict_gpu_env(None, || assert!(!strict_gpu_path()));
        with_strict_gpu_env(Some("1"), || assert!(strict_gpu_path()));
        with_strict_gpu_env(Some("true"), || assert!(strict_gpu_path()));
        with_strict_gpu_env(Some("TRUE"), || assert!(strict_gpu_path()));
        with_strict_gpu_env(Some("0"), || assert!(!strict_gpu_path()));
        with_strict_gpu_env(Some("false"), || assert!(!strict_gpu_path()));
        with_strict_gpu_env(Some("yes"), || assert!(!strict_gpu_path()));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_runtime_tests_env_is_opt_in() {
        let _lock = env_lock();
        let _restore = EnvVarRestore::capture("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS");

        unsafe { std::env::remove_var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") };
        assert!(!run_wgpu_runtime_tests());
        unsafe { std::env::set_var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS", "1") };
        assert!(run_wgpu_runtime_tests());
        unsafe { std::env::set_var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS", "true") };
        assert!(run_wgpu_runtime_tests());
        unsafe { std::env::set_var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS", "TRUE") };
        assert!(run_wgpu_runtime_tests());
        unsafe { std::env::set_var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS", "0") };
        assert!(!run_wgpu_runtime_tests());
    }

    #[test]
    fn spiral_consensus_sanitises_non_finite_values() {
        let softmax = [0.5_f32, f32::NAN, -0.2, 1.2];
        let hardmax = [1.0_f32, f32::NAN, -3.0, 0.0];

        let (fused, stats) = spiral_softmax_hardmax_consensus(&softmax, &hardmax, 2, 2);

        assert_eq!(fused.len(), 4);
        assert!(fused.iter().all(|value| value.is_finite() && *value >= 0.0));
        assert!(stats.average_enrichment.is_finite());
        assert!(stats.spiral_coherence.is_finite());
    }

    #[test]
    fn spiral_consensus_bounds_extreme_values() {
        let huge = f32::powi(10.0, 30);
        let softmax = [huge, huge, huge, huge];
        let hardmax = [huge, huge, huge, huge];

        let (fused, stats) = spiral_softmax_hardmax_consensus(&softmax, &hardmax, 2, 2);

        assert_eq!(fused.len(), 4);
        assert!(fused.iter().all(|value| value.is_finite() && *value >= 0.0));
        assert!(stats.average_enrichment.is_finite());
        assert!(stats.spiral_coherence.is_finite());
    }

    #[test]
    fn spiral_consensus_default_metrics_have_constants() {
        let stats = SpiralConsensusStats::default();

        assert!((stats.phi - GOLDEN_RATIO).abs() < 1e-12);
        assert!((stats.phi_conjugate - GOLDEN_RATIO_CONJUGATE).abs() < 1e-12);
        assert!((stats.phi_bias - GOLDEN_RATIO_BIAS).abs() < 1e-12);
        assert!(stats.ramanujan_ratio.is_finite());
        assert!(stats.ramanujan_delta.is_finite());
        assert_eq!(stats.average_enrichment, 0.0);
        assert_eq!(stats.mean_entropy, 0.0);
        assert_eq!(stats.mean_hardmass, 0.0);
        assert_eq!(stats.spiral_coherence, 0.0);
    }

    fn assert_summary_close(actual: GradientSummary, expected: GradientSummary) {
        const EPS: f32 = 1e-5;
        assert!((actual.l1() - expected.l1()).abs() < EPS);
        assert!((actual.l2() - expected.l2()).abs() < EPS);
        assert!((actual.linf() - expected.linf()).abs() < EPS);
        assert!((actual.sum() - expected.sum()).abs() < EPS);
        assert!((actual.sum_squares() - expected.sum_squares()).abs() < EPS);
        assert!((actual.sum_cubes() - expected.sum_cubes()).abs() < EPS);
        assert!((actual.sum_quartic() - expected.sum_quartic()).abs() < EPS);
        assert_eq!(actual.count(), expected.count());
        assert_eq!(actual.positive_count(), expected.positive_count());
        assert_eq!(actual.negative_count(), expected.negative_count());
        assert_eq!(actual.near_zero_count(), expected.near_zero_count());
        assert!((actual.min() - expected.min()).abs() < EPS);
        assert!((actual.max() - expected.max()).abs() < EPS);
    }

    #[test]
    fn matmul_bias_relu_matches_scalar_pipeline() {
        let lhs = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![1.0, -2.0, 0.5, 0.25, 1.5, -0.75],
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            3,
            2,
            vec![0.5, -1.0, 2.0, 0.25, -0.5, 1.0],
        ));
        let bias = vec![0.5, -0.25];

        let fused = unwrap_ok(lhs.matmul_bias_relu(&rhs, &bias));

        let mut reference = unwrap_ok(lhs.matmul(&rhs));
        unwrap_ok(reference.add_row_inplace(&bias));
        reference.relu_inplace();

        assert_eq!(fused.shape(), reference.shape());
        for (a, b) in fused.data().iter().zip(reference.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn matmul_bias_relu_rejects_non_finite_bias_on_empty_output() {
        let lhs = unwrap_ok(Tensor::from_vec(0, 1, Vec::new()));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![1.0]));

        let err = unwrap_err(lhs.matmul_bias_relu_with_backend(
            &rhs,
            &[f32::NAN],
            MatmulBackend::CpuNaive,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_bias_relu_bias",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn matmul_bias_relu_into_rejects_overflowing_output_without_mutating_destination() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));
        let mut dst = unwrap_ok(Tensor::from_vec(1, 1, vec![7.0]));
        let before = dst.clone();

        let err = unwrap_err(lhs.matmul_bias_relu_into_with_backend(
            &rhs,
            &[0.0],
            &mut dst,
            MatmulBackend::CpuNaive,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_bias_relu_output",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(dst, before);
    }

    #[test]
    fn matmul_bias_gelu_matches_scalar_pipeline() {
        let lhs = unwrap_ok(Tensor::from_vec(
            2,
            4,
            vec![1.0, -1.5, 0.75, 2.0, -0.25, 0.5, 1.25, -0.75],
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            4,
            3,
            vec![
                0.5, -0.25, 1.0, 1.5, 0.75, -1.0, -0.5, 0.33, 0.8, -0.2, 1.2, 0.6,
            ],
        ));
        let bias = vec![0.1, -0.05, 0.2];

        let fused = unwrap_ok(lhs.matmul_bias_gelu(&rhs, &bias));

        let mut reference = unwrap_ok(lhs.matmul(&rhs));
        unwrap_ok(reference.add_row_inplace(&bias));
        reference.gelu_inplace();

        assert_eq!(fused.shape(), reference.shape());
        for (a, b) in fused.data().iter().zip(reference.data().iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn matmul_bias_gelu_rejects_overflowing_output() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));

        let err =
            unwrap_err(lhs.matmul_bias_gelu_with_backend(&rhs, &[0.0], MatmulBackend::CpuNaive));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_bias_gelu_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn matmul_bias_gelu_zero_axes_preserve_learning_semantics() {
        let empty_rows = unwrap_ok(Tensor::from_vec(0, 2, Vec::new()));
        let rhs = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![0.5, -0.25, 1.0, 1.5, 0.75, -1.0],
        ));
        let bias = vec![0.1, -0.05, 0.2];
        let output = unwrap_ok(empty_rows.matmul_bias_gelu_with_backend(
            &rhs,
            &bias,
            MatmulBackend::CpuNaive,
        ));
        assert_eq!(output.shape(), (0, 3));
        assert!(output.data().is_empty());

        let zero_inner_lhs = unwrap_ok(Tensor::from_vec(2, 0, Vec::new()));
        let zero_inner_rhs = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let output = unwrap_ok(zero_inner_lhs.matmul_bias_gelu_with_backend(
            &zero_inner_rhs,
            &bias,
            MatmulBackend::CpuNaive,
        ));
        assert_eq!(output.shape(), (2, 3));
        let expected = [gelu(bias[0]), gelu(bias[1]), gelu(bias[2])];
        for row in output.data().chunks_exact(3) {
            for (observed, expected) in row.iter().zip(expected.iter()) {
                assert!((observed - expected).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn matmul_bias_add_relu_matches_scalar_pipeline() {
        let lhs = unwrap_ok(Tensor::from_vec(
            3,
            4,
            vec![
                1.0, -0.5, 0.25, 2.0, 0.75, -1.25, 1.5, -0.75, 0.33, 0.5, -0.25, 1.0,
            ],
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            4,
            3,
            vec![
                0.5, 1.25, -0.75, -1.0, 0.75, 0.5, 1.5, -0.25, 0.33, -0.66, 0.25, 0.8,
            ],
        ));
        let bias = vec![0.2, -0.1, 0.05];
        let residual = unwrap_ok(Tensor::from_vec(
            3,
            3,
            vec![0.1, 0.2, -0.3, -0.4, 0.5, 0.6, 0.0, -0.2, 0.3],
        ));

        let fused = unwrap_ok(lhs.matmul_bias_add_relu(&rhs, &bias, &residual));

        let mut reference = unwrap_ok(lhs.matmul(&rhs));
        unwrap_ok(reference.add_row_inplace(&bias));
        let mut reference = unwrap_ok(reference.add(&residual));
        reference.relu_inplace();

        assert_eq!(fused.shape(), reference.shape());
        for (a, b) in fused.data().iter().zip(reference.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn matmul_bias_add_relu_into_rejects_overflowing_output_without_mutating_destination() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![0.0]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![0.0]));
        let residual = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let mut dst = unwrap_ok(Tensor::from_vec(1, 1, vec![7.0]));
        let before = dst.clone();

        let err = unwrap_err(lhs.matmul_bias_add_relu_into_with_backend(
            &rhs,
            &[f32::MAX],
            &residual,
            &mut dst,
            MatmulBackend::CpuNaive,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_bias_add_relu_output",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(dst, before);
    }

    #[test]
    fn matmul_bias_add_relu_zero_axes_preserve_residual_semantics() {
        let empty_rows = unwrap_ok(Tensor::from_vec(0, 2, Vec::new()));
        let rhs = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![0.5, 1.25, -0.75, -1.0, 0.75, 0.5],
        ));
        let bias = vec![0.2, -0.1, 0.05];
        let empty_residual = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let output = unwrap_ok(empty_rows.matmul_bias_add_relu_with_backend(
            &rhs,
            &bias,
            &empty_residual,
            MatmulBackend::CpuNaive,
        ));
        assert_eq!(output.shape(), (0, 3));
        assert!(output.data().is_empty());

        let zero_inner_lhs = unwrap_ok(Tensor::from_vec(2, 0, Vec::new()));
        let zero_inner_rhs = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let residual = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![0.1, -0.25, 0.0, -0.4, 0.5, 0.2],
        ));
        let output = unwrap_ok(zero_inner_lhs.matmul_bias_add_relu_with_backend(
            &zero_inner_rhs,
            &bias,
            &residual,
            MatmulBackend::CpuNaive,
        ));
        assert_eq!(output.shape(), (2, 3));
        let expected = [
            (bias[0] + 0.1f32).max(0.0),
            (bias[1] - 0.25f32).max(0.0),
            (bias[2] + 0.0f32).max(0.0),
            (bias[0] - 0.4f32).max(0.0),
            (bias[1] + 0.5f32).max(0.0),
            (bias[2] + 0.2f32).max(0.0),
        ];
        for (observed, expected) in output.data().iter().zip(expected.iter()) {
            assert!((observed - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn tensor_gelu_backward_matches_manual() {
        let input = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![-1.0, 0.25, 0.75, -0.5, 0.0, 1.25],
        ));
        let grad = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![0.5, -0.75, 0.3, -0.2, 0.4, 0.1],
        ));
        let result = unwrap_ok(input.gelu_backward(&grad));
        for ((z, g), value) in input
            .data()
            .iter()
            .zip(grad.data().iter())
            .zip(result.data().iter())
        {
            let expected = gelu_prime(*z) * g;
            assert!((expected - value).abs() < 1e-6);
        }
    }

    #[test]
    fn gelu_backward_observer_meta_reports_backend() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = unwrap_ok(Tensor::from_vec(1, 3, vec![-0.75, 0.0, 0.75]));
        let grad = unwrap_ok(Tensor::from_vec(1, 3, vec![0.2, -0.5, 0.3]));
        let _ = unwrap_ok(input.gelu_backward(&grad));
        crate::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let gelu_backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "gelu_backward" && data["rows"] == 1 && data["cols"] == 3
            })
            .expect("gelu backward metadata event");
        assert_eq!(gelu_backward.1["requested_backend"], "auto");
        assert!(gelu_backward.1["backend"].as_str().is_some());
    }

    #[test]
    fn gelu_backward_with_cpu_backend_reports_cpu() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = unwrap_ok(Tensor::from_vec(1, 3, vec![-0.75, 0.0, 0.75]));
        let grad = unwrap_ok(Tensor::from_vec(1, 3, vec![0.2, -0.5, 0.3]));
        let _ = unwrap_ok(input.gelu_backward_with_backend(&grad, TensorUtilBackend::Cpu));
        crate::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let gelu_backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "gelu_backward" && data["rows"] == 1 && data["cols"] == 3
            })
            .expect("gelu backward metadata event");
        assert_eq!(gelu_backward.1["requested_backend"], "cpu");
        assert_eq!(gelu_backward.1["backend"], "cpu");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_fused_gelu_backward_matches_cpu() {
        if !wgpu_dense::is_available() {
            return;
        }
        let rows = 4;
        let cols = 5;
        let z: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.1) - 1.0).collect();
        let grad: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.2)
            .collect();
        let residual: Vec<f32> = vec![0.05; rows * cols];
        let (gz, dr, db) = unwrap_ok(wgpu_dense::fused_gelu_backward(
            &z,
            &grad,
            Some(&residual),
            rows,
            cols,
        ));

        for ((z_val, g_val), gz_val) in z.iter().zip(grad.iter()).zip(gz.iter()) {
            let expected = gelu_prime(*z_val) * g_val;
            assert!((expected - gz_val).abs() < 1e-5);
        }

        for ((gz_val, residual_val), dr_val) in gz.iter().zip(residual.iter()).zip(dr.iter()) {
            let expected = gz_val + residual_val;
            assert!((expected - dr_val).abs() < 1e-5);
        }

        for c in 0..cols {
            let mut expected = 0.0f32;
            for r in 0..rows {
                expected += gz[r * cols + c];
            }
            assert!((expected - db[c]).abs() < 1e-4);
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_tensor_utils_match_cpu_reference_on_sequence_shapes() {
        if !wgpu_dense::is_available() {
            return;
        }
        let rows = 7;
        let cols = 5;
        let input: Vec<f32> = (0..rows * cols)
            .map(|idx| ((idx as f32 * 0.137).sin() * 0.7) - ((idx % 4) as f32 * 0.03))
            .collect();
        let rhs: Vec<f32> = (0..rows * cols)
            .map(|idx| ((idx as f32 * 0.071).cos() * 0.4) + ((idx % 3) as f32 * 0.02))
            .collect();
        let bias: Vec<f32> = (0..cols).map(|idx| idx as f32 * 0.041 - 0.08).collect();

        let scaled = unwrap_ok(wgpu_dense::scale(&input, rows, cols, -0.25));
        for (idx, value) in scaled.iter().enumerate() {
            let expected = input[idx] * -0.25;
            assert!((expected - value).abs() < 1e-6);
        }

        let subbed = unwrap_ok(wgpu_dense::sub(&input, &rhs, rows, cols));
        for (idx, value) in subbed.iter().enumerate() {
            let expected = input[idx] - rhs[idx];
            assert!((expected - value).abs() < 1e-6);
        }

        let transposed = unwrap_ok(wgpu_dense::transpose(&input, rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                let expected = input[r * cols + c];
                assert!((expected - transposed[c * rows + r]).abs() < 1e-6);
            }
        }

        let token_indices = vec![0.0, 2.0, 1.0, 2.0, 4.0, 0.0];
        let vocab_size = 5;
        let embed_dim = 3;
        let embedding_weights: Vec<f32> = (0..vocab_size * embed_dim)
            .map(|idx| idx as f32 * 0.07 - 0.2)
            .collect();
        let gathered = unwrap_ok(wgpu_dense::embedding_gather(
            &token_indices,
            &embedding_weights,
            token_indices.len(),
            embed_dim,
        ));
        for (token_offset, &token) in token_indices.iter().enumerate() {
            let token = token as usize;
            for dim in 0..embed_dim {
                let expected = embedding_weights[token * embed_dim + dim];
                assert!((expected - gathered[token_offset * embed_dim + dim]).abs() < 1e-6);
            }
        }

        let hidden_dim = 3;
        let gates: Vec<f32> = (0..hidden_dim * 4)
            .map(|idx| (idx as f32 * 0.19).sin() * 0.8 - 0.2)
            .collect();
        let cell_prev: Vec<f32> = (0..hidden_dim)
            .map(|idx| (idx as f32 * 0.31).cos() * 0.3)
            .collect();
        let gate_step = unwrap_ok(wgpu_dense::lstm_forward_gate_step(
            &gates, &cell_prev, hidden_dim,
        ));
        assert_eq!(gate_step.len(), hidden_dim * 6);
        for unit in 0..hidden_dim {
            let i = 1.0 / (1.0 + (-gates[unit]).exp());
            let f = 1.0 / (1.0 + (-gates[hidden_dim + unit]).exp());
            let g = gates[2 * hidden_dim + unit].tanh();
            let o = 1.0 / (1.0 + (-gates[3 * hidden_dim + unit]).exp());
            let cell = f * cell_prev[unit] + i * g;
            let hidden = o * cell.tanh();
            let expected = [i, f, g, o, cell, hidden];
            for (segment, expected_value) in expected.into_iter().enumerate() {
                let actual = gate_step[segment * hidden_dim + unit];
                assert!((actual - expected_value).abs() < 1e-5);
            }
        }

        let grad_output: Vec<f32> = (0..token_indices.len() * embed_dim)
            .map(|idx| idx as f32 * 0.03 - 0.1)
            .collect();
        let scattered = unwrap_ok(wgpu_dense::embedding_scatter_add(
            &token_indices,
            &grad_output,
            token_indices.len(),
            vocab_size,
            embed_dim,
        ));
        for vocab in 0..vocab_size {
            for dim in 0..embed_dim {
                let mut expected = 0.0f32;
                for (token_offset, &token) in token_indices.iter().enumerate() {
                    if token as usize == vocab {
                        expected += grad_output[token_offset * embed_dim + dim];
                    }
                }
                assert!((expected - scattered[vocab * embed_dim + dim]).abs() < 1e-6);
            }
        }

        let seq_batch = 2;
        let seq_steps = 4;
        let seq_features = 3;
        let seq_values: Vec<f32> = (0..seq_batch * seq_steps * seq_features)
            .map(|idx| idx as f32 * 0.031 - 0.2)
            .collect();
        let last_step = unwrap_ok(wgpu_dense::sequence_last_step_gather(
            &seq_values,
            seq_batch,
            seq_steps,
            seq_features,
        ));
        for b in 0..seq_batch {
            for f in 0..seq_features {
                let expected =
                    seq_values[b * seq_steps * seq_features + (seq_steps - 1) * seq_features + f];
                assert!((expected - last_step[b * seq_features + f]).abs() < 1e-6);
            }
        }
        let scattered_last = unwrap_ok(wgpu_dense::sequence_last_step_scatter(
            &last_step,
            seq_batch,
            seq_steps,
            seq_features,
        ));
        for b in 0..seq_batch {
            for step in 0..seq_steps {
                for f in 0..seq_features {
                    let idx = b * seq_steps * seq_features + step * seq_features + f;
                    let expected = if step + 1 == seq_steps {
                        last_step[b * seq_features + f]
                    } else {
                        0.0
                    };
                    assert!((expected - scattered_last[idx]).abs() < 1e-6);
                }
            }
        }

        let coherence_batch = 2;
        let coherence_steps = 4;
        let coherence_dim = 3;
        let coherence_memory = 3;
        let curvature = -1.0f32;
        let temperature = 1.0f32;
        let self_score_scale = 0.25f32;
        let query_residual_scale = 0.5f32;
        let coherence_values: Vec<f32> = (0..coherence_batch * coherence_steps * coherence_dim)
            .map(|idx| ((idx as f32 * 0.173).sin() * 0.6) - 0.15)
            .collect();
        let (coherence_context, coherence_weights) =
            unwrap_ok(wgpu_dense::zspace_coherence_scan_forward(
                &coherence_values,
                coherence_batch,
                coherence_steps,
                coherence_dim,
                coherence_memory,
                curvature,
                temperature,
                self_score_scale,
                query_residual_scale,
            ));
        let start_step = coherence_steps - coherence_memory;
        let order = 1.0 + (-curvature).sqrt().min(4.0);
        let score_pair = |query: &[f32], value: &[f32]| {
            let mse = query
                .iter()
                .zip(value.iter())
                .map(|(&q, &v)| {
                    let diff = q - v;
                    diff * diff
                })
                .sum::<f32>();
            let dist = ((mse / coherence_dim as f32).sqrt() * (-curvature).sqrt() / temperature)
                .max(1.0e-6);
            let score = 1.0 / (dist.powf(order) + 1.0e-12);
            if score.is_finite() {
                score
            } else {
                0.0
            }
        };
        let fallback_weight = |step: usize| {
            if step < start_step {
                return 0.0;
            }
            if self_score_scale > 0.0 {
                return 1.0 / coherence_memory as f32;
            }
            if step + 1 == coherence_steps && coherence_memory > 1 {
                return 0.0;
            }
            1.0 / coherence_memory.saturating_sub(1).max(1) as f32
        };
        for b in 0..coherence_batch {
            let base = b * coherence_steps * coherence_dim;
            let query_start = base + (coherence_steps - 1) * coherence_dim;
            let query = &coherence_values[query_start..query_start + coherence_dim];
            let mut scores = vec![0.0f32; coherence_steps];
            let mut total = 0.0f32;
            for step in start_step..coherence_steps {
                let value_start = base + step * coherence_dim;
                let value = &coherence_values[value_start..value_start + coherence_dim];
                let mut score = score_pair(query, value);
                if step + 1 == coherence_steps {
                    score *= self_score_scale;
                }
                scores[step] = score;
                total += score;
            }
            let total_valid = total.is_finite() && total > 0.0;
            for step in 0..coherence_steps {
                let expected = if total_valid && step >= start_step {
                    scores[step] / total
                } else {
                    fallback_weight(step)
                };
                assert!(
                    (expected - coherence_weights[b * coherence_steps + step]).abs() < 1e-4,
                    "coherence weight mismatch batch={b} step={step}"
                );
            }
            for dim in 0..coherence_dim {
                let mut expected = 0.0f32;
                for step in start_step..coherence_steps {
                    let weight = if total_valid {
                        scores[step] / total
                    } else {
                        fallback_weight(step)
                    };
                    expected += weight * coherence_values[base + step * coherence_dim + dim];
                }
                expected += query_residual_scale * query[dim];
                assert!(
                    (expected - coherence_context[b * coherence_dim + dim]).abs() < 1e-4,
                    "coherence context mismatch batch={b} dim={dim}"
                );
            }
        }

        let coherence_grad_output: Vec<f32> = (0..coherence_batch * coherence_dim)
            .map(|idx| idx as f32 * 0.041 - 0.19)
            .collect();
        let coherence_grad = unwrap_ok(wgpu_dense::zspace_coherence_scan_backward(
            &coherence_values,
            &coherence_grad_output,
            &coherence_weights,
            coherence_batch,
            coherence_steps,
            coherence_dim,
            coherence_memory,
            curvature,
            temperature,
            self_score_scale,
            query_residual_scale,
        ));
        let score_pair_gradient_common =
            |query: &[f32], value: &[f32], score_scale: f32| -> Option<f32> {
                if score_scale == 0.0 {
                    return None;
                }
                let mse = query
                    .iter()
                    .zip(value.iter())
                    .map(|(&q, &v)| {
                        let diff = q - v;
                        diff * diff
                    })
                    .sum::<f32>();
                let dim = coherence_dim as f32;
                let mean = mse / dim.max(1.0);
                if !mean.is_finite() || mean <= 0.0 {
                    return None;
                }
                let sqrt_mean = mean.sqrt();
                let alpha = (-curvature).sqrt() / temperature;
                let unclamped_dist = sqrt_mean * alpha;
                if !unclamped_dist.is_finite() || unclamped_dist <= 1.0e-6 {
                    return None;
                }
                let dist_pow = unclamped_dist.powf(order);
                let denom = dist_pow + 1.0e-12;
                if !denom.is_finite() || denom <= 0.0 {
                    return None;
                }
                let dscore_ddist = -order * unclamped_dist.powf(order - 1.0) / (denom * denom);
                let common = score_scale * dscore_ddist * alpha / (dim * sqrt_mean);
                common.is_finite().then_some(common)
            };
        let mut expected_grad = vec![0.0f32; coherence_values.len()];
        for b in 0..coherence_batch {
            let base = b * coherence_steps * coherence_dim;
            let go_offset = b * coherence_dim;
            for step in 0..coherence_steps {
                let weight = coherence_weights[b * coherence_steps + step];
                let grad_offset = base + step * coherence_dim;
                for dim in 0..coherence_dim {
                    expected_grad[grad_offset + dim] +=
                        weight * coherence_grad_output[go_offset + dim];
                }
            }
            let query_step = coherence_steps - 1;
            let query_offset = base + query_step * coherence_dim;
            for dim in 0..coherence_dim {
                expected_grad[query_offset + dim] +=
                    query_residual_scale * coherence_grad_output[go_offset + dim];
            }

            let query = &coherence_values[query_offset..query_offset + coherence_dim];
            let grad_row = &coherence_grad_output[go_offset..go_offset + coherence_dim];
            let mut scores = vec![0.0f32; coherence_steps];
            let mut total = 0.0f32;
            for step in start_step..coherence_steps {
                let value_offset = base + step * coherence_dim;
                let value = &coherence_values[value_offset..value_offset + coherence_dim];
                let mut score = score_pair(query, value);
                if step == query_step {
                    score *= self_score_scale;
                }
                scores[step] = score;
                total += score;
            }
            if total.is_finite() && total > 0.0 {
                let mut weighted_dot = 0.0f32;
                for step in start_step..coherence_steps {
                    let weight = coherence_weights[b * coherence_steps + step];
                    if weight == 0.0 {
                        continue;
                    }
                    let value_offset = base + step * coherence_dim;
                    let value = &coherence_values[value_offset..value_offset + coherence_dim];
                    let dot = grad_row
                        .iter()
                        .zip(value.iter())
                        .map(|(&grad, &src)| grad * src)
                        .sum::<f32>();
                    weighted_dot += weight * dot;
                }

                for step in start_step..coherence_steps {
                    let score = scores[step];
                    if score == 0.0 {
                        continue;
                    }
                    let value_offset = base + step * coherence_dim;
                    let value = &coherence_values[value_offset..value_offset + coherence_dim];
                    let dot = grad_row
                        .iter()
                        .zip(value.iter())
                        .map(|(&grad, &src)| grad * src)
                        .sum::<f32>();
                    let dloss_dscore = (dot - weighted_dot) / total;
                    if !dloss_dscore.is_finite() || dloss_dscore == 0.0 {
                        continue;
                    }
                    let score_scale = if step == query_step {
                        self_score_scale
                    } else {
                        1.0
                    };
                    let Some(common) = score_pair_gradient_common(query, value, score_scale) else {
                        continue;
                    };
                    for dim in 0..coherence_dim {
                        let delta = query[dim] - value[dim];
                        let contribution = dloss_dscore * common * delta;
                        if contribution.is_finite() {
                            expected_grad[query_offset + dim] += contribution;
                            expected_grad[value_offset + dim] -= contribution;
                        }
                    }
                }
            }
        }
        for (idx, (expected, actual)) in expected_grad.iter().zip(coherence_grad.iter()).enumerate()
        {
            assert!(
                (expected - actual).abs() < 2.0e-3,
                "coherence backward mismatch idx={idx}: expected={expected}, actual={actual}"
            );
        }

        let sum_squares = unwrap_ok(wgpu_dense::sum_squares(&input, rows, cols));
        let expected_sum_squares = input.iter().map(|value| value * value).sum::<f32>();
        assert!((expected_sum_squares - sum_squares).abs() < 1e-5);

        let sum_abs = unwrap_ok(wgpu_dense::sum_abs(&input, rows, cols));
        let expected_sum_abs = input.iter().map(|value| value.abs()).sum::<f32>();
        assert!((expected_sum_abs - sum_abs).abs() < 1e-5);

        let relu = unwrap_ok(wgpu_dense::relu(&input, rows, cols));
        for (expected, actual) in input.iter().map(|value| value.max(0.0)).zip(relu.iter()) {
            assert!((expected - actual).abs() < 1e-6);
        }
        let relu_backward = unwrap_ok(wgpu_dense::relu_backward(&input, &rhs, rows, cols));
        for ((source, grad), actual) in input.iter().zip(rhs.iter()).zip(relu_backward.iter()) {
            let expected = if *source > 0.0 { *grad } else { 0.0 };
            assert!((expected - actual).abs() < 1e-6);
        }

        let row_gate: Vec<f32> = (0..cols).map(|idx| 0.75 + idx as f32 * 0.11).collect();
        let mul_row = unwrap_ok(wgpu_dense::mul_row(&input, &row_gate, rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                let expected = input[r * cols + c] * row_gate[c];
                assert!((expected - mul_row[r * cols + c]).abs() < 1e-6);
            }
        }
        let row_bias: Vec<f32> = (0..cols).map(|idx| idx as f32 * -0.07 + 0.13).collect();
        let row_affine = unwrap_ok(wgpu_dense::row_affine(
            &input, &row_gate, &row_bias, rows, cols,
        ));
        for r in 0..rows {
            for c in 0..cols {
                let expected = input[r * cols + c] * row_gate[c] + row_bias[c];
                assert!((expected - row_affine[r * cols + c]).abs() < 1e-6);
            }
        }

        let kg_mass: Vec<f32> = (0..cols).map(|idx| 0.05 + idx as f32 * 0.017).collect();
        let kg_spin: Vec<f32> = (0..cols)
            .map(|idx| (idx as f32 * 0.19).sin() * 0.1)
            .collect();
        let kg_time_step = 0.2;
        let kg_damping = 0.1;
        let kg_forward = unwrap_ok(wgpu_dense::dynamic_klein_gordon_forward(
            &input,
            &kg_mass,
            &kg_spin,
            rows,
            cols,
            kg_time_step,
            kg_damping,
        ));
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let wave = input[idx];
                let amplitude = wave.tanh();
                let kg_coeff =
                    1.0 - kg_time_step * kg_damping - kg_time_step * kg_time_step * kg_mass[c];
                let dirac_coeff = kg_time_step * kg_spin[c];
                let expected = wave * kg_coeff + dirac_coeff * amplitude;
                assert!((expected - kg_forward[idx]).abs() < 1e-6);
            }
        }
        let kg_grad = unwrap_ok(wgpu_dense::dynamic_klein_gordon_backward(
            &input,
            &rhs,
            &kg_mass,
            &kg_spin,
            rows,
            cols,
            kg_time_step,
            kg_damping,
        ));
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let wave = input[idx];
                let amplitude = wave.tanh();
                let sech2 = 1.0 - amplitude * amplitude;
                let kg_coeff =
                    1.0 - kg_time_step * kg_damping - kg_time_step * kg_time_step * kg_mass[c];
                let dirac_coeff = kg_time_step * kg_spin[c];
                let expected = rhs[idx] * (kg_coeff + dirac_coeff * sech2);
                assert!((expected - kg_grad.0[idx]).abs() < 1e-6);
            }
        }
        for c in 0..cols {
            let mut expected_mass = 0.0f32;
            let mut expected_spin = 0.0f32;
            for r in 0..rows {
                let idx = r * cols + c;
                expected_mass += rhs[idx] * (-kg_time_step * kg_time_step * input[idx]);
                expected_spin += rhs[idx] * (kg_time_step * input[idx].tanh());
            }
            assert!((expected_mass - kg_grad.1[c]).abs() < 1e-5);
            assert!((expected_spin - kg_grad.2[c]).abs() < 1e-5);
        }

        let hj_potential: Vec<f32> = (0..cols)
            .map(|idx| ((idx as f32 + 1.0) * 0.13).sin().abs() + 0.1)
            .collect();
        let hj_step = 0.1;
        let hj_forward = unwrap_ok(wgpu_dense::dynamic_hamilton_jacobi_forward(
            &input,
            &hj_potential,
            rows,
            cols,
            hj_step,
        ));
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let current = input[idx];
                let prev = if r > 0 { input[idx - cols] } else { current };
                let next = if r + 1 < rows {
                    input[idx + cols]
                } else {
                    current
                };
                let grad = (2.0 * current - prev - next) + hj_potential[c] * current;
                let expected = current - hj_step * grad;
                assert!((expected - hj_forward[idx]).abs() < 1e-6);
            }
        }
        let hj_grad = unwrap_ok(wgpu_dense::dynamic_hamilton_jacobi_backward(
            &input,
            &rhs,
            &hj_potential,
            rows,
            cols,
            hj_step,
        ));
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let mut factor = 1.0 - hj_step * (2.0 + hj_potential[c]);
                if rows == 1 {
                    factor = 1.0 - hj_step * hj_potential[c];
                } else if r == 0 || r + 1 == rows {
                    factor = 1.0 - hj_step * (1.0 + hj_potential[c]);
                }
                let mut expected = rhs[idx] * factor;
                if r > 0 {
                    expected += rhs[idx - cols] * hj_step;
                }
                if r + 1 < rows {
                    expected += rhs[idx + cols] * hj_step;
                }
                assert!((expected - hj_grad.0[idx]).abs() < 1e-6);
            }
        }
        for c in 0..cols {
            let mut expected = 0.0f32;
            for r in 0..rows {
                let idx = r * cols + c;
                expected += -hj_step * input[idx] * rhs[idx];
            }
            assert!((expected - hj_grad.1[c]).abs() < 1e-5);
        }

        let sch_rate = 0.4;
        let sch_coherence: Vec<f32> = (0..cols)
            .map(|idx| (0.85 - idx as f32 * 0.03).max(0.1))
            .collect();
        let sch_forward = unwrap_ok(wgpu_dense::dynamic_schrodinger_forward(
            &input,
            &sch_coherence,
            rows,
            cols,
            sch_rate,
        ));
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let amp = input[idx].tanh();
                let deco = 1.0 / (1.0 + sch_rate * amp.abs());
                let expected = amp * sch_coherence[c] * deco;
                assert!((expected - sch_forward.0[idx]).abs() < 1e-6);
                assert!((amp - sch_forward.1[idx]).abs() < 1e-6);
                assert!((deco - sch_forward.2[idx]).abs() < 1e-6);
            }
        }
        let sch_grad = unwrap_ok(wgpu_dense::dynamic_schrodinger_backward(
            &sch_forward.1,
            &sch_forward.2,
            &rhs,
            &sch_coherence,
            rows,
            cols,
            sch_rate,
        ));
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let amp = sch_forward.1[idx];
                let deco = sch_forward.2[idx];
                let denom = 1.0 + sch_rate * amp.abs();
                let sign = if amp >= 0.0 { 1.0 } else { -1.0 };
                let d_deco_d_amp = -sch_rate * sign / (denom * denom);
                let base = deco + amp * d_deco_d_amp;
                let d_amp_d_input = 1.0 - amp * amp;
                let expected = rhs[idx] * sch_coherence[c] * base * d_amp_d_input;
                assert!((expected - sch_grad.0[idx]).abs() < 1e-6);
            }
        }
        for c in 0..cols {
            let mut expected = 0.0f32;
            for r in 0..rows {
                let idx = r * cols + c;
                expected += rhs[idx] * sch_forward.1[idx] * sch_forward.2[idx];
            }
            assert!((expected - sch_grad.1[c]).abs() < 1e-5);
        }

        let pool_input = vec![
            1.0, 2.0, 5.0, 4.0, 3.0, 6.0, 7.0, 8.0, 0.5, -1.0, 9.0, 2.0, 3.0, 1.5, 4.5, 0.0, 0.2,
            -0.4, 1.1, 0.9, 2.2, 1.7, 0.3, -0.8, 1.4, 3.3, 2.8, 0.6, -1.2, 0.5, 1.8, 2.4,
        ];
        let (pool_values, pool_indices) = unwrap_ok(wgpu_dense::max_pool2d_forward(
            &pool_input,
            1,
            2,
            4,
            4,
            2,
            2,
            2,
            2,
            2,
            2,
            0,
            0,
        ));
        let expected_pool_values = vec![6.0, 8.0, 3.0, 9.0, 2.2, 1.1, 3.3, 2.8];
        let expected_pool_indices = vec![5, 7, 12, 10, 20, 18, 25, 26];
        assert_eq!(pool_indices, expected_pool_indices);
        for (expected, actual) in expected_pool_values.iter().zip(pool_values.iter()) {
            assert!((expected - actual).abs() < 1e-6);
        }
        let pool_grad = vec![0.5, -0.25, 1.0, 0.75, -0.5, 0.2, 0.3, -0.1];
        let pool_grad_input = unwrap_ok(wgpu_dense::max_pool2d_backward(
            &pool_input,
            &pool_grad,
            1,
            2,
            4,
            4,
            2,
            2,
            2,
            2,
            2,
            2,
            0,
            0,
        ));
        for (idx, actual) in pool_grad_input.iter().enumerate() {
            let expected = expected_pool_indices
                .iter()
                .zip(pool_grad.iter())
                .filter_map(|(&winner, &grad)| (winner == idx).then_some(grad))
                .sum::<f32>();
            assert!((expected - actual).abs() < 1e-6);
        }

        let avg_pool_values = unwrap_ok(wgpu_dense::avg_pool2d_forward(
            &pool_input,
            1,
            2,
            4,
            4,
            2,
            2,
            2,
            2,
            2,
            2,
            0,
            0,
        ));
        let expected_avg_pool_values = vec![3.0, 6.0, 1.0, 3.875, 0.925, 0.375, 1.0, 1.9];
        for (expected, actual) in expected_avg_pool_values.iter().zip(avg_pool_values.iter()) {
            assert!((expected - actual).abs() < 1e-6);
        }
        let avg_pool_grad_input = unwrap_ok(wgpu_dense::avg_pool2d_backward(
            &pool_input,
            &pool_grad,
            1,
            2,
            4,
            4,
            2,
            2,
            2,
            2,
            2,
            2,
            0,
            0,
        ));
        let expected_avg_pool_grad_input = vec![
            0.125, 0.125, -0.0625, -0.0625, 0.125, 0.125, -0.0625, -0.0625, 0.25, 0.25, 0.1875,
            0.1875, 0.25, 0.25, 0.1875, 0.1875, -0.125, -0.125, 0.05, 0.05, -0.125, -0.125, 0.05,
            0.05, 0.075, 0.075, -0.025, -0.025, 0.075, 0.075, -0.025, -0.025,
        ];
        for (expected, actual) in expected_avg_pool_grad_input
            .iter()
            .zip(avg_pool_grad_input.iter())
        {
            assert!((expected - actual).abs() < 1e-6);
        }

        let mse_prediction = vec![0.5, -0.5, 1.0, 0.25];
        let mse_target = vec![0.0, 0.0, 1.5, -0.75];
        let mse_loss = unwrap_ok(wgpu_dense::mse_loss_forward(
            &mse_prediction,
            &mse_target,
            2,
            2,
        ));
        let expected_mse = mse_prediction
            .iter()
            .zip(mse_target.iter())
            .map(|(prediction, target)| {
                let diff = *prediction - *target;
                diff * diff
            })
            .sum::<f32>()
            / mse_prediction.len() as f32;
        assert!((expected_mse - mse_loss).abs() < 1e-6);
        let mse_grad = unwrap_ok(wgpu_dense::mse_loss_backward(
            &mse_prediction,
            &mse_target,
            2,
            2,
        ));
        for ((prediction, target), actual) in mse_prediction
            .iter()
            .zip(mse_target.iter())
            .zip(mse_grad.iter())
        {
            let expected = 2.0 * (*prediction - *target) / mse_prediction.len() as f32;
            assert!((expected - actual).abs() < 1e-6);
        }

        let ce_prediction = vec![0.1, 0.6, 0.3, 0.8, 0.1, 0.1];
        let ce_target = vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let ce_loss = unwrap_ok(wgpu_dense::categorical_cross_entropy_forward(
            &ce_prediction,
            &ce_target,
            2,
            3,
            1e-9,
        ));
        let expected_ce_loss = (-0.6_f32.ln() - 0.8_f32.ln()) / 2.0;
        assert!((expected_ce_loss - ce_loss).abs() < 1e-6);
        let ce_grad = unwrap_ok(wgpu_dense::categorical_cross_entropy_backward(
            &ce_prediction,
            &ce_target,
            2,
            3,
            1e-9,
        ));
        let expected_ce_grad = vec![0.0, -1.0 / (0.6 * 2.0), 0.0, -1.0 / (0.8 * 2.0), 0.0, 0.0];
        for (expected, actual) in expected_ce_grad.iter().zip(ce_grad.iter()) {
            assert!((expected - actual).abs() < 1e-6);
        }

        let hce_prediction = vec![0.25, -0.5, 0.9, -1.25];
        let hce_target = vec![1.0, 0.0, 0.75, 0.1];
        let hce_curvature = -1.44f32;
        let hce_epsilon = 1e-5f32;
        let hce_scale = (-hce_curvature).sqrt();
        let hce_loss = unwrap_ok(wgpu_dense::hyperbolic_cross_entropy_forward(
            &hce_prediction,
            &hce_target,
            2,
            2,
            hce_curvature,
            hce_epsilon,
        ));
        let stable_softplus = |value: f32| {
            if value > 0.0 {
                value + (-value).exp().ln_1p()
            } else {
                value.exp().ln_1p()
            }
        };
        let expected_hce = hce_prediction
            .iter()
            .zip(hce_target.iter())
            .map(|(prediction, target)| {
                let target = target.clamp(hce_epsilon, 1.0 - hce_epsilon);
                let scaled = prediction * hce_scale;
                let log_sigmoid_pos = -stable_softplus(-scaled);
                let log_sigmoid_neg = -stable_softplus(scaled);
                -target * log_sigmoid_pos - (1.0 - target) * log_sigmoid_neg
            })
            .sum::<f32>()
            / hce_prediction.len() as f32;
        assert!((expected_hce - hce_loss).abs() < 1e-5);
        let hce_grad = unwrap_ok(wgpu_dense::hyperbolic_cross_entropy_backward(
            &hce_prediction,
            &hce_target,
            2,
            2,
            hce_curvature,
            hce_epsilon,
        ));
        for ((prediction, target), actual) in hce_prediction
            .iter()
            .zip(hce_target.iter())
            .zip(hce_grad.iter())
        {
            let target = target.clamp(hce_epsilon, 1.0 - hce_epsilon);
            let scaled = prediction * hce_scale;
            let sigmoid = if scaled >= 0.0 {
                1.0 / (1.0 + (-scaled).exp())
            } else {
                let exp_value = scaled.exp();
                exp_value / (1.0 + exp_value)
            };
            let expected = hce_scale * (sigmoid - target) / hce_prediction.len() as f32;
            assert!((expected - actual).abs() < 1e-6);
        }

        let softmax_input = vec![0.2, -0.1, 0.3, 0.5, 0.0, -0.25];
        let softmax_grad = vec![0.05, -0.02, 0.1, -0.15, 0.2, 0.03];
        let softmax_factor = 0.75;
        let softmax_backward = unwrap_ok(wgpu_dense::zspace_softmax_backward_fixed(
            &softmax_input,
            &softmax_grad,
            2,
            3,
            softmax_factor,
        ));
        for row in 0..2 {
            let offset = row * 3;
            let logits = &softmax_input[offset..offset + 3];
            let grad = &softmax_grad[offset..offset + 3];
            let max_logit = logits
                .iter()
                .map(|value| value * softmax_factor)
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_values = logits
                .iter()
                .map(|value| (value * softmax_factor - max_logit).exp())
                .collect::<Vec<_>>();
            let sum = exp_values.iter().sum::<f32>();
            let probs = exp_values
                .iter()
                .map(|value| value / sum)
                .collect::<Vec<_>>();
            let dot = grad
                .iter()
                .zip(probs.iter())
                .map(|(g, p)| g * p)
                .sum::<f32>();
            for col in 0..3 {
                let expected = softmax_factor * probs[col] * (grad[col] - dot);
                assert!((expected - softmax_backward[offset + col]).abs() < 2e-5);
            }
        }

        let gradient: Vec<f32> = (0..rows * cols)
            .map(|idx| idx as f32 * 0.011 - 0.2)
            .collect();
        let hyper_wave: Vec<f32> = (0..rows * cols)
            .map(|idx| ((idx as f32 * 0.097).sin() * 0.5).clamp(-0.75, 0.75))
            .collect();
        let hyper_next = unwrap_ok(wgpu_dense::hypergrad_accumulate_wave(
            &gradient,
            &hyper_wave,
            rows,
            cols,
            -1.0,
            1e-6,
            1_000_000.0,
            0.2,
        ));
        for idx in 0..gradient.len() {
            let value = hyper_wave[idx];
            let denom = 1.0 + value * value;
            let update = value / denom.abs().max(1e-6);
            let expected = porous_mix_value(gradient[idx] + update, 1_000_000.0, 0.2);
            assert!((expected - hyper_next[idx]).abs() < 1e-5);
        }

        let elementwise_added = unwrap_ok(wgpu_dense::add(&input, &rhs, rows, cols));
        for (idx, value) in elementwise_added.iter().enumerate() {
            let expected = input[idx] + rhs[idx];
            assert!((expected - value).abs() < 1e-6);
        }

        let multiplied = unwrap_ok(wgpu_dense::hadamard(&input, &rhs, rows, cols));
        for (idx, value) in multiplied.iter().enumerate() {
            let expected = input[idx] * rhs[idx];
            assert!((expected - value).abs() < 1e-6);
        }

        let scaled_accum = unwrap_ok(wgpu_dense::add_scaled(&input, &rhs, rows, cols, -0.25));
        for (idx, value) in scaled_accum.iter().enumerate() {
            let expected = input[idx] + rhs[idx] * -0.25;
            assert!((expected - value).abs() < 1e-6);
        }

        let added = unwrap_ok(wgpu_dense::add_row(&input, &bias, rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let expected = input[idx] + bias[c];
                assert!((expected - added[idx]).abs() < 1e-6);
            }
        }

        let summed = unwrap_ok(wgpu_dense::sum_axis0(&input, rows, cols));
        for c in 0..cols {
            let mut expected = 0.0f32;
            for r in 0..rows {
                expected += input[r * cols + c];
            }
            assert!((expected - summed[c]).abs() < 1e-5);
        }

        let summed_scaled = unwrap_ok(wgpu_dense::sum_axis0_scaled(&input, rows, cols, 0.25));
        for c in 0..cols {
            let mut expected = 0.0f32;
            for r in 0..rows {
                expected += input[r * cols + c];
            }
            assert!((expected * 0.25 - summed_scaled[c]).abs() < 1e-5);
        }

        let projected = unwrap_ok(wgpu_dense::project_to_poincare(&input, rows, cols, -1.0));
        for r in 0..rows {
            let start = r * cols;
            let end = start + cols;
            let row = &input[start..end];
            let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
            let factor = if norm > 0.0 { norm.tanh() / norm } else { 1.0 };
            for c in 0..cols {
                let expected = input[start + c] * factor;
                assert!((expected - projected[start + c]).abs() < 2e-5);
            }
        }

        let gate: Vec<f32> = (0..cols).map(|idx| 0.2 + idx as f32 * 0.03).collect();
        let wave_bias: Vec<f32> = (0..cols).map(|idx| idx as f32 * 0.05 - 0.1).collect();
        let wave = unwrap_ok(wgpu_dense::wave_gate_project(
            &input, &gate, &wave_bias, rows, cols, -1.0, 0.45, 0.2,
        ));
        for r in 0..rows {
            let start = r * cols;
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(porous_mix_value(
                    input[start + c] * gate[c] + wave_bias[c],
                    0.45,
                    0.2,
                ));
            }
            let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
            let factor = if norm > 0.0 { norm.tanh() / norm } else { 1.0 };
            for c in 0..cols {
                let expected = row[c] * factor;
                assert!((expected - wave[start + c]).abs() < 2e-5);
            }
        }
    }

    #[test]
    fn mul_row_cpu_backend_matches_manual_rows() {
        let tensor = unwrap_ok(Tensor::from_vec(2, 3, vec![1.0, -2.0, 0.5, 3.0, 4.0, -1.5]));
        let output =
            unwrap_ok(tensor.mul_row_with_backend(&[0.5, -1.0, 2.0], TensorUtilBackend::Cpu));
        let expected = [0.5, 2.0, 1.0, 1.5, -4.0, -3.0];

        assert_eq!(output.shape(), (2, 3));
        for (expected, actual) in expected.iter().zip(output.data()) {
            assert!((expected - actual).abs() < 1.0e-6);
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn tensor_utility_methods_emit_wgpu_backend_when_available() {
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let tensor = unwrap_ok(Tensor::from_vec(
            3,
            4,
            vec![
                0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 0.1, 0.2, -0.3, 0.4,
            ],
        ));
        let _ = unwrap_ok(tensor.scale_with_backend(0.5, TensorUtilBackend::GpuWgpu));
        let rhs = unwrap_ok(Tensor::from_vec(
            3,
            4,
            vec![
                -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, 0.9, -1.0, 1.1, -1.2,
            ],
        ));
        let _ = unwrap_ok(tensor.add_with_backend(&rhs, TensorUtilBackend::GpuWgpu));
        let _ = unwrap_ok(tensor.hadamard_with_backend(&rhs, TensorUtilBackend::GpuWgpu));
        let _ = unwrap_ok(tensor.relu_with_backend(TensorUtilBackend::GpuWgpu));
        let _ = unwrap_ok(
            tensor.mul_row_with_backend(&[0.2, -0.4, 0.6, -0.8], TensorUtilBackend::GpuWgpu),
        );
        let _ = unwrap_ok(tensor.row_affine_with_backend(
            &[0.2, -0.4, 0.6, -0.8],
            &[0.05, -0.1, 0.15, -0.2],
            TensorUtilBackend::GpuWgpu,
        ));
        let _ = unwrap_ok(tensor.sub_with_backend(&rhs, TensorUtilBackend::GpuWgpu));
        let _ = unwrap_ok(tensor.transpose_with_backend(TensorUtilBackend::GpuWgpu));
        let mut accum = tensor.clone();
        unwrap_ok(accum.add_scaled_with_backend(&rhs, -0.25, TensorUtilBackend::GpuWgpu));
        let mut biased = tensor.clone();
        unwrap_ok(
            biased
                .add_row_inplace_with_backend(&[0.1, -0.2, 0.3, -0.4], TensorUtilBackend::GpuWgpu),
        );
        let _ = tensor.sum_axis0_with_backend(TensorUtilBackend::GpuWgpu);
        let _ = tensor.sum_axis0_scaled_with_backend(0.25, TensorUtilBackend::GpuWgpu);
        let _ = unwrap_ok(tensor.squared_l2_norm_with_backend(TensorUtilBackend::GpuWgpu));
        let _ = unwrap_ok(tensor.sum_abs_with_backend(TensorUtilBackend::GpuWgpu));
        let _ =
            unwrap_ok(tensor.project_to_poincare_with_backend(-1.0, TensorUtilBackend::GpuWgpu));
        let _ = unwrap_ok(tensor.wave_gate_project_with_backend(
            &[0.1, 0.2, 0.3, 0.4],
            &[0.05, -0.05, 0.1, -0.1],
            -1.0,
            1.0,
            0.2,
            TensorUtilBackend::GpuWgpu,
        ));
        crate::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        for op_name in [
            "scale",
            "add",
            "hadamard",
            "relu",
            "mul_row",
            "row_affine",
            "sub",
            "transpose",
            "add_scaled",
            "add_row_inplace",
            "sum_axis0",
            "sum_axis0_scaled",
            "squared_l2_norm",
            "sum_abs",
            "project_to_poincare",
            "wave_gate_project",
        ] {
            let (_, data) = events
                .iter()
                .find(|(observed, _)| *observed == op_name)
                .unwrap_or_else(|| panic!("{op_name} metadata event"));
            assert_eq!(data["backend"], "wgpu_dense");
            assert_eq!(data["requested_backend"], "wgpu");
            assert!(data["kernel"]
                .as_str()
                .unwrap_or_default()
                .starts_with("tensor_util."));
        }
    }

    #[test]
    fn matmul_bias_add_gelu_matches_scalar_pipeline() {
        let lhs = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![0.5, -1.0, 1.5, 0.25, 0.75, -0.5],
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            3,
            3,
            vec![1.0, -0.75, 0.5, -0.5, 0.33, 1.25, 0.8, -0.2, 0.4],
        ));
        let bias = vec![0.2, -0.1, 0.05];
        let residual = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![0.1, -0.05, 0.0, -0.2, 0.3, 0.4],
        ));

        let fused = unwrap_ok(lhs.matmul_bias_add_gelu(&rhs, &bias, &residual));

        let mut reference = unwrap_ok(lhs.matmul(&rhs));
        unwrap_ok(reference.add_row_inplace(&bias));
        let mut reference = unwrap_ok(reference.add(&residual));
        reference.gelu_inplace();

        assert_eq!(fused.shape(), reference.shape());
        for (a, b) in fused.data().iter().zip(reference.data().iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn matmul_bias_add_gelu_rejects_non_finite_residual() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![0.0]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![0.0]));
        let residual = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::NAN]));

        let err = unwrap_err(lhs.matmul_bias_add_gelu_with_backend(
            &rhs,
            &[0.0],
            &residual,
            MatmulBackend::CpuNaive,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_bias_add_gelu_residual",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn matmul_bias_add_gelu_rejects_overflowing_output() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![0.0]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![0.0]));
        let residual = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));

        let err = unwrap_err(lhs.matmul_bias_add_gelu_with_backend(
            &rhs,
            &[f32::MAX],
            &residual,
            MatmulBackend::CpuNaive,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_bias_add_gelu_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn matmul_bias_fused_forced_wgpu_matches_cpu_reference_on_edge_tiles() {
        let lhs = unwrap_ok(Tensor::from_vec(
            13,
            9,
            (0..117)
                .map(|idx| ((idx as f32 * 0.071).sin() * 0.8) - ((idx % 11) as f32 * 0.01))
                .collect(),
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            9,
            7,
            (0..63)
                .map(|idx| ((idx as f32 * 0.113).cos() * 0.6) + ((idx % 5) as f32 * 0.02))
                .collect(),
        ));
        let bias: Vec<f32> = (0..7).map(|idx| idx as f32 * 0.031 - 0.08).collect();
        let residual = unwrap_ok(Tensor::from_vec(
            13,
            7,
            (0..91)
                .map(|idx| ((idx as f32 * 0.037).sin() * 0.15) - 0.03)
                .collect(),
        ));

        let cpu_relu =
            unwrap_ok(lhs.matmul_bias_relu_with_backend(&rhs, &bias, MatmulBackend::CpuNaive));
        let wgpu_relu = match lhs.matmul_bias_relu_with_backend(&rhs, &bias, MatmulBackend::GpuWgpu)
        {
            Ok(value) => value,
            Err(error) if wgpu_unavailable(&error) => return,
            Err(error) => panic!("forced WGPU matmul_bias_relu failed: {error:?}"),
        };
        assert_tensor_close(&cpu_relu, &wgpu_relu, 2e-4);

        let cpu_gelu =
            unwrap_ok(lhs.matmul_bias_gelu_with_backend(&rhs, &bias, MatmulBackend::CpuNaive));
        let wgpu_gelu = match lhs.matmul_bias_gelu_with_backend(&rhs, &bias, MatmulBackend::GpuWgpu)
        {
            Ok(value) => value,
            Err(error) if wgpu_unavailable(&error) => return,
            Err(error) => panic!("forced WGPU matmul_bias_gelu failed: {error:?}"),
        };
        assert_tensor_close(&cpu_gelu, &wgpu_gelu, 2e-4);

        let cpu_add_relu = unwrap_ok(lhs.matmul_bias_add_relu_with_backend(
            &rhs,
            &bias,
            &residual,
            MatmulBackend::CpuNaive,
        ));
        let wgpu_add_relu = match lhs.matmul_bias_add_relu_with_backend(
            &rhs,
            &bias,
            &residual,
            MatmulBackend::GpuWgpu,
        ) {
            Ok(value) => value,
            Err(error) if wgpu_unavailable(&error) => return,
            Err(error) => panic!("forced WGPU matmul_bias_add_relu failed: {error:?}"),
        };
        assert_tensor_close(&cpu_add_relu, &wgpu_add_relu, 2e-4);

        let cpu_add_gelu = unwrap_ok(lhs.matmul_bias_add_gelu_with_backend(
            &rhs,
            &bias,
            &residual,
            MatmulBackend::CpuNaive,
        ));
        let wgpu_add_gelu = match lhs.matmul_bias_add_gelu_with_backend(
            &rhs,
            &bias,
            &residual,
            MatmulBackend::GpuWgpu,
        ) {
            Ok(value) => value,
            Err(error) if wgpu_unavailable(&error) => return,
            Err(error) => panic!("forced WGPU matmul_bias_add_gelu failed: {error:?}"),
        };
        assert_tensor_close(&cpu_add_gelu, &wgpu_add_gelu, 2e-4);
    }

    #[test]
    fn tensor_roundtrip_dlpack_preserves_contents() {
        let tensor = unwrap_ok(Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let managed = unwrap_ok(tensor.to_dlpack());
        let restored = unsafe { unwrap_ok(Tensor::from_dlpack(managed)) };
        assert_eq!(tensor, restored);
        assert_eq!(tensor.data().as_ptr(), restored.data().as_ptr());
    }

    #[test]
    fn tensor_from_dlpack_calls_deleter_on_error() {
        let drops = Arc::new(AtomicUsize::new(0));
        let ctx = Box::new(DlpackTestCtx {
            drops: Arc::clone(&drops),
            data: vec![0.0; 4].into_boxed_slice(),
            shape: vec![2, 2].into_boxed_slice(),
            strides: Some(vec![2, 1].into_boxed_slice()),
        });
        let data_ptr = ctx.data.as_ptr() as *mut f32;
        let shape_ptr = ctx.shape.as_ptr() as *mut i64;
        let strides_ptr = match ctx.strides.as_ref() {
            Some(strides) => strides.as_ptr() as *mut i64,
            None => ptr::null_mut(),
        };

        let managed = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr as *mut c_void,
                device: DLDevice {
                    device_type: DLDeviceType::Cuda as i32,
                    device_id: 0,
                },
                ndim: 2,
                dtype: DLDataType {
                    code: DLDataTypeCode::Float as u8,
                    bits: 32,
                    lanes: 1,
                },
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: 0,
            },
            manager_ctx: Box::into_raw(ctx) as *mut c_void,
            deleter: Some(dlpack_test_deleter),
        });
        let managed_ptr = Box::into_raw(managed);

        let err = unsafe { unwrap_err(Tensor::from_dlpack(managed_ptr)) };
        assert!(matches!(err, TensorError::DlpackError { .. }));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn tensor_from_dlpack_respects_byte_offset() {
        let drops = Arc::new(AtomicUsize::new(0));
        let data: Vec<f32> = (0..16).map(|value| value as f32).collect();
        let ctx = Box::new(DlpackTestCtx {
            drops: Arc::clone(&drops),
            data: data.into_boxed_slice(),
            shape: vec![2, 2].into_boxed_slice(),
            strides: Some(vec![2, 1].into_boxed_slice()),
        });
        let data_ptr = ctx.data.as_ptr() as *mut f32;
        let shape_ptr = ctx.shape.as_ptr() as *mut i64;
        let strides_ptr = match ctx.strides.as_ref() {
            Some(strides) => strides.as_ptr() as *mut i64,
            None => ptr::null_mut(),
        };
        let byte_offset = mem::size_of::<f32>() * 8;
        let expected_ptr = data_ptr.wrapping_add(8) as *const f32;

        let managed = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr as *mut c_void,
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
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset,
            },
            manager_ctx: Box::into_raw(ctx) as *mut c_void,
            deleter: Some(dlpack_test_deleter),
        });
        let managed_ptr = Box::into_raw(managed);

        let tensor = unsafe { unwrap_ok(Tensor::from_dlpack(managed_ptr)) };
        assert_eq!(tensor.data(), &[8.0, 9.0, 10.0, 11.0]);
        assert_eq!(tensor.data().as_ptr(), expected_ptr);
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        drop(tensor);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn tensor_from_dlpack_rejects_misaligned_byte_offset() {
        let drops = Arc::new(AtomicUsize::new(0));
        let ctx = Box::new(DlpackTestCtx {
            drops: Arc::clone(&drops),
            data: vec![0.0; 4].into_boxed_slice(),
            shape: vec![2, 2].into_boxed_slice(),
            strides: Some(vec![2, 1].into_boxed_slice()),
        });
        let data_ptr = ctx.data.as_ptr() as *mut f32;
        let shape_ptr = ctx.shape.as_ptr() as *mut i64;
        let strides_ptr = match ctx.strides.as_ref() {
            Some(strides) => strides.as_ptr() as *mut i64,
            None => ptr::null_mut(),
        };

        let managed = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr as *mut c_void,
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
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: 2,
            },
            manager_ctx: Box::into_raw(ctx) as *mut c_void,
            deleter: Some(dlpack_test_deleter),
        });
        let managed_ptr = Box::into_raw(managed);

        let err = unsafe { unwrap_err(Tensor::from_dlpack(managed_ptr)) };
        assert!(matches!(err, TensorError::DlpackError { .. }));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn tensor_to_dlpack_shares_underlying_buffer() {
        let tensor = unwrap_ok(Tensor::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]));
        let ptr = tensor.data.as_ptr();
        let managed = unwrap_ok(tensor.to_dlpack());
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
        let a = unwrap_ok(Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let b = unwrap_ok(Tensor::from_vec(
            3,
            2,
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ));
        let product = unwrap_ok(a.matmul(&b));
        assert_eq!(product.shape(), (2, 2));
        let expected = unwrap_ok(Tensor::from_vec(2, 2, vec![58.0, 64.0, 139.0, 154.0]));
        assert_eq!(product, expected);

        let addend = unwrap_ok(Tensor::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]));
        let sum = unwrap_ok(product.add(&addend));
        let expected_sum = unwrap_ok(Tensor::from_vec(2, 2, vec![59.0, 65.0, 140.0, 155.0]));
        assert_eq!(sum, expected_sum);
    }

    #[test]
    fn tensor_preserves_zero_sized_axes_for_basic_shape_ops() {
        let empty_rows = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        assert_eq!(empty_rows.shape(), (0, 3));
        assert_eq!(empty_rows.len(), 0);
        assert!(empty_rows.is_empty());
        assert!(empty_rows.data().is_empty());

        let zero_cols = unwrap_ok(Tensor::zeros(2, 0));
        assert_eq!(zero_cols.shape(), (2, 0));
        assert!(zero_cols.data().is_empty());

        let generated = unwrap_ok(Tensor::from_fn(0, 5, |_, _| panic!("generator not called")));
        assert_eq!(generated.shape(), (0, 5));

        let reshaped = unwrap_ok(empty_rows.reshape(3, 0));
        assert_eq!(reshaped.shape(), (3, 0));
        let viewed = unwrap_ok(reshaped.view(0, 3));
        assert_eq!(viewed.shape(), (0, 3));

        let transposed = empty_rows.transpose();
        assert_eq!(transposed.shape(), (3, 0));
        assert!(transposed.data().is_empty());
    }

    #[test]
    fn tensor_zero_sized_axes_flow_through_basic_math() {
        let empty_a = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let empty_b = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        assert_eq!(unwrap_ok(empty_a.add(&empty_b)).shape(), (0, 3));
        assert_eq!(unwrap_ok(empty_a.sub(&empty_b)).shape(), (0, 3));
        assert_eq!(unwrap_ok(empty_a.scale(2.0)).shape(), (0, 3));
        assert_eq!(unwrap_ok(mean_squared_error(&empty_a, &empty_b)), 0.0);
        assert_eq!(unwrap_ok(empty_a.row_softmax()).shape(), (0, 3));
        let (soft, hard) = unwrap_ok(empty_a.row_softmax_hardmax());
        assert_eq!(soft.shape(), (0, 3));
        assert_eq!(hard.shape(), (0, 3));
        assert!(soft.data().is_empty());
        assert!(hard.data().is_empty());
        let spiral = unwrap_ok(empty_a.row_softmax_hardmax_spiral());
        assert_eq!(spiral.softmax.shape(), (0, 3));
        assert_eq!(spiral.hardmax.shape(), (0, 3));
        assert_eq!(spiral.spiral.shape(), (0, 3));
        assert_eq!(spiral.metrics.average_enrichment, 0.0);
        assert_eq!(spiral.metrics.mean_entropy, 0.0);
        assert_eq!(spiral.metrics.mean_hardmass, 0.0);
        assert_eq!(spiral.metrics.spiral_coherence, 0.0);

        let zero_cols = unwrap_ok(Tensor::from_vec(2, 0, Vec::new()));
        let (soft, hard) = unwrap_ok(zero_cols.row_softmax_hardmax());
        assert_eq!(soft.shape(), (2, 0));
        assert_eq!(hard.shape(), (2, 0));
        assert!(soft.data().is_empty());
        assert!(hard.data().is_empty());
        let spiral = unwrap_ok(zero_cols.row_softmax_hardmax_spiral());
        assert_eq!(spiral.softmax.shape(), (2, 0));
        assert_eq!(spiral.hardmax.shape(), (2, 0));
        assert_eq!(spiral.spiral.shape(), (2, 0));
        assert_eq!(spiral.metrics.average_enrichment, 0.0);
        assert_eq!(spiral.metrics.mean_entropy, 0.0);
        assert_eq!(spiral.metrics.mean_hardmass, 0.0);
        assert_eq!(spiral.metrics.spiral_coherence, 0.0);

        let rhs = unwrap_ok(Tensor::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let product = unwrap_ok(empty_a.matmul(&rhs));
        assert_eq!(product.shape(), (0, 2));
        assert!(product.data().is_empty());

        let lhs = unwrap_ok(Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let empty_rhs = unwrap_ok(Tensor::from_vec(3, 0, Vec::new()));
        let product = unwrap_ok(lhs.matmul(&empty_rhs));
        assert_eq!(product.shape(), (2, 0));
        assert!(product.data().is_empty());

        let zero_inner_lhs = unwrap_ok(Tensor::from_vec(2, 0, Vec::new()));
        let zero_inner_rhs = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let product = unwrap_ok(zero_inner_lhs.matmul(&zero_inner_rhs));
        assert_eq!(product.shape(), (2, 3));
        assert_eq!(product.data(), &[0.0; 6]);
    }

    #[test]
    fn tensor_cpu_elementwise_and_reduction_meta_reports_backend() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let a = unwrap_ok(Tensor::from_vec(2, 2, vec![1.0, -2.0, 3.0, -4.0]));
        let b = unwrap_ok(Tensor::from_vec(2, 2, vec![0.5, 1.5, -2.0, 0.25]));
        let _ = unwrap_ok(a.add(&b));
        let _ = unwrap_ok(a.sub(&b));
        let _ = unwrap_ok(a.scale(0.5));
        let _ = unwrap_ok(a.hadamard(&b));
        let mut inplace = a.clone();
        unwrap_ok(inplace.add_scaled(&b, -0.25));
        unwrap_ok(inplace.add_row_inplace(&[0.1, -0.2]));
        inplace.relu_inplace();
        inplace.gelu_inplace();
        let _ = inplace.transpose();
        let _ = unwrap_ok(inplace.reshape(1, 4));
        let _ = inplace.sum_axis0();
        let _ = inplace.sum_axis1();
        let _ = unwrap_ok(Tensor::cat_rows(&[a.clone(), b.clone()]));
        let _ = unwrap_ok(mean_squared_error(&a, &b));
        crate::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let find = |op_name: &'static str| {
            events
                .iter()
                .find(|(observed, _)| *observed == op_name)
                .unwrap_or_else(|| panic!("{op_name} metadata event"))
        };
        for op_name in [
            "add",
            "sub",
            "scale",
            "hadamard",
            "add_scaled",
            "add_row_inplace",
            "relu_inplace",
            "gelu_inplace",
            "transpose",
            "reshape",
            "sum_axis0",
            "sum_axis0_scaled",
            "sum_axis1",
            "cat_rows",
            "mean_squared_error",
        ] {
            let (_, data) = find(op_name);
            if op_name == "reshape" {
                assert_eq!(data["backend"], "view");
                assert_eq!(data["kernel"], "metadata");
                assert_eq!(data["zero_copy"], true);
            } else if op_name == "mean_squared_error" {
                assert_eq!(data["backend"], "composite");
                assert_eq!(data["kernel"], "tensor_util.mean_squared_error");
            } else {
                assert_eq!(data["backend"], "cpu");
                assert_eq!(data["kernel"], "scalar");
            }
            assert_eq!(data["requested_backend"], "auto");
        }

        let (_, scale) = find("scale");
        assert_eq!(scale["kind"], "elementwise");
        assert_eq!(scale["scale"], 0.5);
        let (_, add_scaled) = find("add_scaled");
        assert_eq!(add_scaled["kind"], "elementwise_inplace");
        assert_eq!(add_scaled["scale"], -0.25);
        let (_, sum_axis0) = find("sum_axis0");
        assert_eq!(sum_axis0["kind"], "reduction");
        assert_eq!(sum_axis0["axis"], 0);
        assert_eq!(sum_axis0["output_rows"], 1);
        assert_eq!(sum_axis0["output_cols"], 2);
        let (_, sum_axis0_scaled) = find("sum_axis0_scaled");
        assert_eq!(sum_axis0_scaled["kind"], "reduction");
        assert_eq!(sum_axis0_scaled["axis"], 0);
        assert_eq!(sum_axis0_scaled["scale"], 0.25);
        let (_, cat_rows) = find("cat_rows");
        assert_eq!(cat_rows["kind"], "copy");
        assert_eq!(cat_rows["inputs"], 2);
        assert_eq!(cat_rows["output_rows"], 4);
        let (_, mse) = find("mean_squared_error");
        assert_eq!(mse["kind"], "reduction");
        assert_eq!(mse["reduction"], "mean");
        assert_eq!(mse["reduction_backend"], "auto");
        assert_eq!(mse["output_values"], 1);
    }

    #[test]
    fn tensor_add_rejects_overflowing_output() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));

        let err = lhs.add(&rhs).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "add_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn tensor_sub_rejects_overflowing_output() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![-f32::MAX]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));

        let err = lhs.sub(&rhs).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "sub_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn tensor_hadamard_rejects_overflowing_output() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));

        let err = lhs.hadamard(&rhs).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "hadamard_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn tensor_scale_rejects_non_finite_factor_on_empty_tensor() {
        let tensor = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let err = tensor.scale(f32::INFINITY).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "scale_factor",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn tensor_scale_rejects_overflowing_output() {
        let tensor = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));
        let err = tensor.scale(f32::MAX).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "scale_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn tensor_add_scaled_rejects_non_finite_factor_on_empty_tensor() {
        let mut lhs = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let rhs = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let err = lhs.add_scaled(&rhs, f32::NAN).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "add_scaled_factor",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn tensor_add_scaled_rejects_overflowing_delta_without_mutating_lhs() {
        let mut lhs = unwrap_ok(Tensor::zeros(1, 1));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));
        let before = lhs.clone();

        let err = lhs.add_scaled(&rhs, f32::MAX).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "add_scaled_delta",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(lhs, before);
    }

    #[test]
    fn tensor_add_scaled_rejects_overflowing_output_without_mutating_lhs() {
        let mut lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![0.5]));
        let before = lhs.clone();

        let err = lhs.add_scaled(&rhs, f32::MAX).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "add_scaled_output",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(lhs, before);
    }

    #[test]
    fn tensor_add_row_rejects_non_finite_bias_on_empty_tensor() {
        let mut tensor = unwrap_ok(Tensor::from_vec(0, 2, Vec::new()));
        let err = tensor.add_row_inplace(&[0.0, f32::NAN]).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "add_row_bias",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn tensor_add_row_rejects_overflowing_output_without_mutating_lhs() {
        let mut tensor = unwrap_ok(Tensor::from_vec(1, 2, vec![f32::MAX, 1.0]));
        let before = tensor.clone();

        let err = tensor.add_row_inplace(&[f32::MAX, 0.0]).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "add_row_output",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(tensor, before);
    }

    #[test]
    fn tensor_mean_squared_error_rejects_non_finite_prediction() {
        let prediction = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::INFINITY]));
        let target = unwrap_ok(Tensor::zeros(1, 1));

        let err = mean_squared_error(&prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "mse_prediction",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn tensor_mean_squared_error_rejects_overflowing_square() {
        let prediction = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let target = unwrap_ok(Tensor::zeros(1, 1));

        let err = mean_squared_error(&prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "mse_squared_diff",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn linear_model_train_batch_rejects_overflowing_loss() {
        let mut model = unwrap_ok(LinearModel::new(1, 1));
        let input = unwrap_ok(Tensor::zeros(1, 1));
        let target = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let weights_before = model.weights().clone();
        let bias_before = model.bias().to_vec();

        let err = model.train_batch(&input, &target, 0.0).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "linear_model_mse_squared_diff",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(*model.weights(), weights_before);
        assert_eq!(model.bias(), bias_before.as_slice());
    }

    #[test]
    fn linear_model_train_batch_with_backend_emits_explicit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let inputs = unwrap_ok(Tensor::from_vec(2, 1, vec![0.0, 1.0]));
        let targets = unwrap_ok(Tensor::from_vec(2, 1, vec![1.0, 3.0]));
        let mut model = unwrap_ok(LinearModel::new(1, 1));
        let loss = unwrap_ok(model.train_batch_with_backend(
            &inputs,
            &targets,
            0.1,
            MatmulBackend::CpuNaive,
            TensorUtilBackend::Cpu,
        ));
        crate::set_tensor_op_meta_observer(previous);

        assert!(loss.is_finite());
        let events = events.lock().unwrap();
        let prepacked_bias = events
            .iter()
            .find(|(op_name, _)| *op_name == "matmul_prepacked_bias")
            .expect("linear model forward prepacked-bias metadata event");
        assert_eq!(prepacked_bias.1["requested_backend"], "naive");
        let squared = events
            .iter()
            .find(|(op_name, _)| *op_name == "hadamard")
            .expect("linear model squared-diff metadata event");
        assert_eq!(squared.1["backend"], "cpu");
        assert_eq!(squared.1["requested_backend"], "cpu");
        let grad_w = events
            .iter()
            .find(|(op_name, _)| *op_name == "matmul_lhs_transpose_scaled")
            .expect("linear model weight-gradient metadata event");
        assert_eq!(grad_w.1["requested_backend"], "naive");
        let grad_b = events
            .iter()
            .find(|(op_name, _)| *op_name == "sum_axis0_scaled")
            .expect("linear model bias-gradient metadata event");
        assert_eq!(grad_b.1["backend"], "cpu");
        assert_eq!(grad_b.1["requested_backend"], "cpu");
        let weight_update = events
            .iter()
            .find(|(op_name, _)| *op_name == "add_scaled")
            .expect("linear model weight-update metadata event");
        assert_eq!(weight_update.1["backend"], "cpu");
        assert_eq!(weight_update.1["requested_backend"], "cpu");
    }

    #[test]
    fn tensor_try_sum_axis0_rejects_overflowing_output() {
        let tensor = unwrap_ok(Tensor::from_vec(2, 1, vec![f32::MAX, f32::MAX]));

        let err = tensor.try_sum_axis0().unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "sum_axis0_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn tensor_try_sum_axis0_scaled_rejects_non_finite_factor_on_empty_tensor() {
        let tensor = unwrap_ok(Tensor::from_vec(0, 2, Vec::new()));

        let err = tensor
            .try_sum_axis0_scaled_with_backend(f32::NAN, TensorUtilBackend::Cpu)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "sum_axis0_scaled_factor",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn tensor_try_sum_axis0_scaled_rejects_overflowing_output() {
        let tensor = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));

        let err = tensor
            .try_sum_axis0_scaled_with_backend(f32::MAX, TensorUtilBackend::Cpu)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "sum_axis0_scaled_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn tensor_try_sum_axis1_rejects_overflowing_output() {
        let tensor = unwrap_ok(Tensor::from_vec(1, 2, vec![f32::MAX, f32::MAX]));

        let err = tensor.try_sum_axis1().unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "sum_axis1_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn tensor_hyperbolic_diagnostics_meta_reports_backend() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let tensor = unwrap_ok(Tensor::from_vec(2, 2, vec![0.25, -0.5, 0.75, 0.0]));
        let _ = tensor.squared_l2_norm();
        let _ = tensor.sum_abs();
        let projected = unwrap_ok(tensor.project_to_poincare(-1.0));
        let _ = unwrap_ok(projected.hyperbolic_distance(&projected, -1.0));
        crate::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let find = |op_name: &'static str| {
            events
                .iter()
                .find(|(observed, _)| *observed == op_name)
                .unwrap_or_else(|| panic!("{op_name} metadata event"))
        };

        let (_, l2) = find("squared_l2_norm");
        assert_eq!(l2["backend"], "cpu");
        assert_eq!(l2["requested_backend"], "auto");
        assert_eq!(l2["kind"], "diagnostic_reduction");
        assert_eq!(l2["reduction"], "sum_squares");
        assert_eq!(l2["output_rows"], 1);
        assert_eq!(l2["output_cols"], 1);
        assert!(l2["result"].as_f64().unwrap_or(0.0) > 0.0);

        let (_, l1) = find("sum_abs");
        assert_eq!(l1["backend"], "cpu");
        assert_eq!(l1["requested_backend"], "auto");
        assert_eq!(l1["kind"], "diagnostic_reduction");
        assert_eq!(l1["reduction"], "sum_abs");
        assert_eq!(l1["output_rows"], 1);
        assert_eq!(l1["output_cols"], 1);
        assert!(l1["result"].as_f64().unwrap_or(0.0) > 0.0);

        let (_, projection) = find("project_to_poincare");
        assert_eq!(projection["backend"], "cpu");
        assert_eq!(projection["kind"], "hyperbolic_projection");
        assert_eq!(projection["curvature"], -1.0);
        assert_eq!(projection["nonzero_rows"], 2);
        assert_eq!(projection["output_rows"], 2);
        assert_eq!(projection["output_cols"], 2);
        assert!(projection["max_row_l2"].as_f64().unwrap_or(0.0) > 0.0);

        let (_, distance) = find("hyperbolic_distance");
        assert_eq!(distance["backend"], "cpu");
        assert_eq!(distance["kind"], "hyperbolic_distance");
        assert_eq!(distance["rhs_rows"], 2);
        assert_eq!(distance["rhs_cols"], 2);
        assert_eq!(distance["output_values"], 1);
        assert!(distance["distance"].as_f64().unwrap_or(-1.0) >= 0.0);
    }

    #[test]
    fn scaled_dot_attention_zero_sequence_preserves_shape_and_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let queries = unwrap_ok(Tensor::from_vec(0, 4, Vec::new()));
        let keys = unwrap_ok(Tensor::from_vec(0, 4, Vec::new()));
        let values = unwrap_ok(Tensor::from_vec(0, 4, Vec::new()));
        let z_bias = unwrap_ok(Tensor::from_vec(2, 0, Vec::new()));
        let attn_bias = unwrap_ok(Tensor::from_vec(0, 0, Vec::new()));
        let output = unwrap_ok(queries.scaled_dot_attention_with_backend(
            &keys,
            &values,
            2,
            0,
            0.5,
            Some(&z_bias),
            Some(&attn_bias),
            AttentionBackend::Cpu,
        ));
        crate::set_tensor_op_meta_observer(previous);

        assert_eq!(output.shape(), (0, 4));
        assert!(output.data().is_empty());
        let events = events.lock().unwrap();
        let attention = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "scaled_dot_attention"
                    && data["contexts"] == 2
                    && data["sequence"] == 0
                    && data["head_dim"] == 4
            })
            .expect("empty scaled attention metadata event");
        assert_eq!(attention.1["backend"], "cpu");
        assert_eq!(attention.1["requested_backend"], "cpu");
        assert_eq!(attention.1["flags"]["empty"], true);
        assert_eq!(attention.1["flags"]["use_z_bias"], true);
        assert_eq!(attention.1["flags"]["use_attn_bias"], true);
    }

    #[test]
    fn scaled_dot_attention_zero_head_dim_preserves_shape() {
        let queries = unwrap_ok(Tensor::from_vec(3, 0, Vec::new()));
        let keys = unwrap_ok(Tensor::from_vec(3, 0, Vec::new()));
        let values = unwrap_ok(Tensor::from_vec(3, 0, Vec::new()));

        let output = unwrap_ok(queries.scaled_dot_attention_with_backend(
            &keys,
            &values,
            1,
            3,
            1.0,
            None,
            None,
            AttentionBackend::Auto,
        ));

        assert_eq!(output.shape(), (3, 0));
        assert!(output.data().is_empty());
    }

    #[test]
    fn scaled_dot_attention_rejects_non_finite_scale_on_empty_output() {
        let queries = unwrap_ok(Tensor::from_vec(0, 4, Vec::new()));
        let keys = unwrap_ok(Tensor::from_vec(0, 4, Vec::new()));
        let values = unwrap_ok(Tensor::from_vec(0, 4, Vec::new()));

        let err = unwrap_err(queries.scaled_dot_attention_with_backend(
            &keys,
            &values,
            2,
            0,
            f32::NAN,
            None,
            None,
            AttentionBackend::Cpu,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "scaled_dot_attention_scale",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn scaled_dot_attention_rejects_non_finite_bias_on_empty_head_dim() {
        let queries = unwrap_ok(Tensor::from_vec(2, 0, Vec::new()));
        let keys = unwrap_ok(Tensor::from_vec(2, 0, Vec::new()));
        let values = unwrap_ok(Tensor::from_vec(2, 0, Vec::new()));
        let z_bias = unwrap_ok(Tensor::from_vec(1, 2, vec![0.0, f32::NAN]));

        let err = unwrap_err(queries.scaled_dot_attention_with_backend(
            &keys,
            &values,
            1,
            2,
            1.0,
            Some(&z_bias),
            None,
            AttentionBackend::Cpu,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "scaled_dot_attention_z_bias",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn scaled_dot_attention_rejects_non_finite_attention_bias() {
        let queries = unwrap_ok(Tensor::from_vec(1, 1, vec![0.0]));
        let keys = unwrap_ok(Tensor::from_vec(1, 1, vec![0.0]));
        let values = unwrap_ok(Tensor::from_vec(1, 1, vec![0.0]));
        let attn_bias = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::INFINITY]));

        let err = unwrap_err(queries.scaled_dot_attention_with_backend(
            &keys,
            &values,
            1,
            1,
            1.0,
            None,
            Some(&attn_bias),
            AttentionBackend::Cpu,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "scaled_dot_attention_attn_bias",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn scaled_dot_attention_rejects_overflowing_cpu_dot_product() {
        let queries = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let keys = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));
        let values = unwrap_ok(Tensor::from_vec(1, 1, vec![1.0]));

        let err = unwrap_err(queries.scaled_dot_attention_with_backend(
            &keys,
            &values,
            1,
            1,
            1.0,
            None,
            None,
            AttentionBackend::Cpu,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "scaled_dot_attention_dot",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn layer_norm_empty_rows_preserve_shape_and_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let residual = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let gamma = unwrap_ok(Tensor::from_vec(1, 3, vec![1.0, 1.0, 1.0]));
        let beta = unwrap_ok(Tensor::zeros(1, 3));
        let output = unwrap_ok(input.layer_norm_affine_add_with_backend(
            &residual,
            &gamma,
            &beta,
            1.0e-5,
            LayerNormBackend::Cpu,
        ));
        crate::set_tensor_op_meta_observer(previous);

        assert_eq!(output.shape(), (0, 3));
        assert!(output.data().is_empty());
        let events = events.lock().unwrap();
        let layer_norm = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "layer_norm" && data["rows"] == 0 && data["cols"] == 3
            })
            .expect("empty layer_norm metadata event");
        assert_eq!(layer_norm.1["backend"], "cpu");
        assert_eq!(layer_norm.1["requested_backend"], "cpu");
        assert_eq!(layer_norm.1["flags"]["empty"], true);
        assert_eq!(layer_norm.1["flags"]["use_residual"], true);
    }

    #[test]
    fn layer_norm_rejects_non_finite_affine_on_empty_rows() {
        let input = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let gamma = unwrap_ok(Tensor::from_vec(1, 3, vec![1.0, f32::NAN, 1.0]));
        let beta = unwrap_ok(Tensor::zeros(1, 3));

        let err = unwrap_err(input.layer_norm_affine_with_backend(
            &gamma,
            &beta,
            1.0e-5,
            LayerNormBackend::Cpu,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "layer_norm_gamma",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn layer_norm_rejects_non_finite_residual() {
        let input = unwrap_ok(Tensor::from_vec(1, 2, vec![0.0, 1.0]));
        let residual = unwrap_ok(Tensor::from_vec(1, 2, vec![0.0, f32::INFINITY]));
        let gamma = unwrap_ok(Tensor::from_vec(1, 2, vec![1.0, 1.0]));
        let beta = unwrap_ok(Tensor::zeros(1, 2));

        let err = unwrap_err(input.layer_norm_affine_add_with_backend(
            &residual,
            &gamma,
            &beta,
            1.0e-5,
            LayerNormBackend::Cpu,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "layer_norm_residual",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn layer_norm_rejects_overflowing_cpu_variance() {
        let input = unwrap_ok(Tensor::from_vec(1, 2, vec![f32::MAX, -f32::MAX]));
        let gamma = unwrap_ok(Tensor::from_vec(1, 2, vec![1.0, 1.0]));
        let beta = unwrap_ok(Tensor::zeros(1, 2));

        let err = unwrap_err(input.layer_norm_affine_with_backend(
            &gamma,
            &beta,
            1.0e-5,
            LayerNormBackend::Cpu,
        ));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "layer_norm_sumsq",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn tensor_cat_rows_preserves_empty_batches() {
        let empty = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let rows = unwrap_ok(Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let combined = unwrap_ok(Tensor::cat_rows(&[empty.clone(), rows.clone()]));
        assert_eq!(combined.shape(), (2, 3));
        assert_eq!(combined.data(), rows.data());

        let all_empty = unwrap_ok(Tensor::cat_rows(&[empty.clone(), empty]));
        assert_eq!(all_empty.shape(), (0, 3));
        assert!(all_empty.data().is_empty());
    }

    #[test]
    fn matmul_prepacked_matches_standard() {
        let lhs = unwrap_ok(Tensor::from_vec(
            4,
            3,
            vec![
                1.0, -0.5, 2.0, 0.25, 1.5, -1.25, 0.75, 0.5, -0.75, 1.0, -1.5, 0.33,
            ],
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            3,
            5,
            vec![
                0.5, -1.0, 0.25, 1.5, -0.75, 1.0, 0.5, -0.5, 0.75, -1.25, 0.66, 0.8, -0.2, 1.2,
                -0.4,
            ],
        ));
        let packed = unwrap_ok(PackedB::from_tensor(&rhs, Tile::col_major()));
        let standard = unwrap_ok(lhs.matmul(&rhs));
        let prepacked = unwrap_ok(lhs.matmul_prepacked(&packed));
        assert_eq!(standard, prepacked);
    }

    #[test]
    fn matmul_into_rejects_overflowing_output_without_mutating_destination() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));
        let mut dst = unwrap_ok(Tensor::from_vec(1, 1, vec![7.0]));
        let before = dst.clone();

        let err = lhs
            .matmul_into_with_backend(&rhs, &mut dst, MatmulBackend::CpuNaive)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_output",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(dst, before);
    }

    #[test]
    fn matmul_prepacked_into_rejects_overflowing_output_without_mutating_destination() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));
        let packed = unwrap_ok(PackedB::from_tensor(&rhs, Tile::col_major()));
        let mut dst = unwrap_ok(Tensor::from_vec(1, 1, vec![7.0]));
        let before = dst.clone();

        let err = lhs
            .matmul_prepacked_into_with_backend(&packed, &mut dst, MatmulBackend::CpuNaive)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_prepacked_output",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(dst, before);
    }

    #[test]
    fn matmul_scaled_matches_standard_pipeline() {
        let lhs = unwrap_ok(Tensor::from_vec(
            4,
            3,
            vec![
                1.0, -0.5, 2.0, 0.25, 1.5, -1.25, 0.75, 0.5, -0.75, 1.0, -1.5, 0.33,
            ],
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            3,
            5,
            vec![
                0.5, -1.0, 0.25, 1.5, -0.75, 1.0, 0.5, -0.5, 0.75, -1.25, 0.66, 0.8, -0.2, 1.2,
                -0.4,
            ],
        ));
        let standard = unwrap_ok(unwrap_ok(lhs.matmul(&rhs)).scale(0.25));
        let scaled = unwrap_ok(lhs.matmul_scaled_with_backend(&rhs, 0.25, MatmulBackend::CpuNaive));
        assert_eq!(standard, scaled);
    }

    #[test]
    fn matmul_lhs_transpose_scaled_matches_standard_pipeline() {
        let lhs = unwrap_ok(Tensor::from_vec(
            4,
            3,
            vec![
                1.0, -0.5, 2.0, 0.25, 1.5, -1.25, 0.75, 0.5, -0.75, 1.0, -1.5, 0.33,
            ],
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            4,
            5,
            vec![
                0.5, -1.0, 0.25, 1.5, -0.75, 1.0, 0.5, -0.5, 0.75, -1.25, 0.66, 0.8, -0.2, 1.2,
                -0.4, 0.2, -0.1, 0.9, -0.3, 0.7,
            ],
        ));
        let standard = unwrap_ok(unwrap_ok(lhs.transpose().matmul(&rhs)).scale(0.25));
        let scaled = unwrap_ok(lhs.matmul_lhs_transpose_scaled_with_backend(
            &rhs,
            0.25,
            MatmulBackend::CpuNaive,
        ));
        assert_eq!(standard, scaled);
    }

    #[test]
    fn matmul_scaled_rejects_non_finite_scale_on_empty_output() {
        let lhs = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let rhs = unwrap_ok(Tensor::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

        let err = lhs
            .matmul_scaled_with_backend(&rhs, f32::NAN, MatmulBackend::CpuNaive)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_scaled_factor",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn matmul_scaled_rejects_overflowing_output() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));

        let err = lhs
            .matmul_scaled_with_backend(&rhs, 1.0, MatmulBackend::CpuNaive)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_scaled_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn matmul_lhs_transpose_scaled_rejects_non_finite_scale_on_empty_inner() {
        let lhs = unwrap_ok(Tensor::from_vec(0, 3, Vec::new()));
        let rhs = unwrap_ok(Tensor::from_vec(0, 2, Vec::new()));

        let err = lhs
            .matmul_lhs_transpose_scaled_with_backend(&rhs, f32::INFINITY, MatmulBackend::CpuNaive)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_lhs_transpose_scaled_factor",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn matmul_lhs_transpose_scaled_rejects_overflowing_output() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));

        let err = lhs
            .matmul_lhs_transpose_scaled_with_backend(&rhs, 1.0, MatmulBackend::CpuNaive)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_lhs_transpose_scaled_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn matmul_prepacked_bias_matches_standard_pipeline() {
        let lhs = unwrap_ok(Tensor::from_vec(
            4,
            3,
            vec![
                1.0, -0.5, 2.0, 0.25, 1.5, -1.25, 0.75, 0.5, -0.75, 1.0, -1.5, 0.33,
            ],
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            3,
            5,
            vec![
                0.5, -1.0, 0.25, 1.5, -0.75, 1.0, 0.5, -0.5, 0.75, -1.25, 0.66, 0.8, -0.2, 1.2,
                -0.4,
            ],
        ));
        let bias = vec![0.1, -0.2, 0.05, 0.3, -0.15];
        let packed = unwrap_ok(PackedB::from_tensor(&rhs, Tile::col_major()));
        let mut standard = unwrap_ok(lhs.matmul(&rhs));
        unwrap_ok(standard.add_row_inplace(&bias));
        let prepacked = unwrap_ok(lhs.matmul_prepacked_bias(&packed, &bias));
        assert_eq!(standard, prepacked);
    }

    #[test]
    fn matmul_prepacked_bias_rejects_non_finite_bias_on_empty_output() {
        let lhs = unwrap_ok(Tensor::from_vec(0, 1, Vec::new()));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![1.0]));
        let packed = unwrap_ok(PackedB::from_tensor(&rhs, Tile::col_major()));

        let err = lhs
            .matmul_prepacked_bias_with_backend(&packed, &[f32::NAN], MatmulBackend::CpuNaive)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_prepacked_bias_bias",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn matmul_prepacked_bias_rejects_overflowing_output() {
        let lhs = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let rhs = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));
        let packed = unwrap_ok(PackedB::from_tensor(&rhs, Tile::col_major()));

        let err = lhs
            .matmul_prepacked_bias_with_backend(&packed, &[0.0], MatmulBackend::CpuNaive)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "matmul_prepacked_bias_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn matmul_prepacked_forced_wgpu_matches_cpu_reference_on_edge_tiles() {
        let lhs = unwrap_ok(Tensor::from_vec(
            12,
            8,
            (0..96)
                .map(|idx| ((idx as f32 * 0.137).sin() * 0.7) + ((idx % 5) as f32 * 0.01))
                .collect(),
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            8,
            6,
            (0..48)
                .map(|idx| ((idx as f32 * 0.173).cos() * 0.5) - ((idx % 7) as f32 * 0.015))
                .collect(),
        ));
        let packed = unwrap_ok(PackedB::from_tensor(&rhs, Tile::col_major()));
        let cpu = unwrap_ok(lhs.matmul_prepacked_with_backend(&packed, MatmulBackend::CpuNaive));
        let wgpu = match lhs.matmul_prepacked_with_backend(&packed, MatmulBackend::GpuWgpu) {
            Ok(value) => value,
            Err(error) if wgpu_unavailable(&error) => return,
            Err(error) => panic!("forced WGPU prepacked matmul failed: {error:?}"),
        };

        assert_tensor_close(&cpu, &wgpu, 1e-4);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn matmul_scaled_forced_wgpu_matches_cpu_reference_on_edge_tiles() {
        let lhs = unwrap_ok(Tensor::from_vec(
            12,
            8,
            (0..96)
                .map(|idx| ((idx as f32 * 0.137).sin() * 0.7) + ((idx % 5) as f32 * 0.01))
                .collect(),
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            8,
            6,
            (0..48)
                .map(|idx| ((idx as f32 * 0.173).cos() * 0.5) - ((idx % 7) as f32 * 0.015))
                .collect(),
        ));
        let cpu = unwrap_ok(lhs.matmul_scaled_with_backend(&rhs, 0.125, MatmulBackend::CpuNaive));
        let wgpu = match lhs.matmul_scaled_with_backend(&rhs, 0.125, MatmulBackend::GpuWgpu) {
            Ok(value) => value,
            Err(error) if wgpu_unavailable(&error) => return,
            Err(error) => panic!("forced WGPU scaled matmul failed: {error:?}"),
        };

        assert_tensor_close(&cpu, &wgpu, 1e-4);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn matmul_lhs_transpose_scaled_forced_wgpu_matches_cpu_reference_on_edge_tiles() {
        let lhs = unwrap_ok(Tensor::from_vec(
            12,
            8,
            (0..96)
                .map(|idx| ((idx as f32 * 0.137).sin() * 0.7) + ((idx % 5) as f32 * 0.01))
                .collect(),
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            12,
            6,
            (0..72)
                .map(|idx| ((idx as f32 * 0.173).cos() * 0.5) - ((idx % 7) as f32 * 0.015))
                .collect(),
        ));
        let cpu = unwrap_ok(lhs.matmul_lhs_transpose_scaled_with_backend(
            &rhs,
            0.125,
            MatmulBackend::CpuNaive,
        ));
        let wgpu =
            match lhs.matmul_lhs_transpose_scaled_with_backend(&rhs, 0.125, MatmulBackend::GpuWgpu)
            {
                Ok(value) => value,
                Err(error) if wgpu_unavailable(&error) => return,
                Err(error) => panic!("forced WGPU lhs-transpose scaled matmul failed: {error:?}"),
            };

        assert_tensor_close(&cpu, &wgpu, 1e-4);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn matmul_prepacked_bias_forced_wgpu_matches_cpu_reference_on_edge_tiles() {
        let lhs = unwrap_ok(Tensor::from_vec(
            12,
            8,
            (0..96)
                .map(|idx| ((idx as f32 * 0.137).sin() * 0.7) + ((idx % 5) as f32 * 0.01))
                .collect(),
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            8,
            6,
            (0..48)
                .map(|idx| ((idx as f32 * 0.173).cos() * 0.5) - ((idx % 7) as f32 * 0.015))
                .collect(),
        ));
        let bias: Vec<f32> = (0..6).map(|idx| idx as f32 * 0.021 - 0.06).collect();
        let packed = unwrap_ok(PackedB::from_tensor(&rhs, Tile::col_major()));
        let cpu = unwrap_ok(lhs.matmul_prepacked_bias_with_backend(
            &packed,
            &bias,
            MatmulBackend::CpuNaive,
        ));
        let wgpu =
            match lhs.matmul_prepacked_bias_with_backend(&packed, &bias, MatmulBackend::GpuWgpu) {
                Ok(value) => value,
                Err(error) if wgpu_unavailable(&error) => return,
                Err(error) => panic!("forced WGPU prepacked bias matmul failed: {error:?}"),
            };

        assert_tensor_close(&cpu, &wgpu, 1e-4);
    }

    #[test]
    fn wgpu_runtime_fallback_meta_reports_route_reason_and_message() {
        let data = wgpu_runtime_fallback_meta(
            "naive",
            "runtime_unavailable",
            Some("no suitable WGPU adapter found"),
        );

        assert_eq!(data["from"], "wgpu");
        assert_eq!(data["to"], "naive");
        assert_eq!(data["reason"], "runtime_unavailable");
        assert_eq!(data["message"], "no suitable WGPU adapter found");
    }

    #[test]
    fn matmul_observer_meta_reports_selected_backend() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let lhs = unwrap_ok(Tensor::from_vec(
            4,
            3,
            vec![
                1.0, -0.5, 2.0, 0.25, 1.5, -1.25, 0.75, 0.5, -0.75, 1.0, -1.5, 0.33,
            ],
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            3,
            5,
            vec![
                0.5, -1.0, 0.25, 1.5, -0.75, 1.0, 0.5, -0.5, 0.75, -1.25, 0.66, 0.8, -0.2, 1.2,
                -0.4,
            ],
        ));
        let packed = unwrap_ok(PackedB::from_tensor(&rhs, Tile::col_major()));

        let _ = unwrap_ok(lhs.matmul(&rhs));
        let _ = unwrap_ok(lhs.matmul_prepacked(&packed));
        crate::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let matmul = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "matmul" && data["rows"] == 4 && data["inner"] == 3 && data["cols"] == 5
            })
            .expect("matmul metadata event");
        assert_eq!(matmul.1["requested_backend"], "auto");
        assert!(matmul.1["backend"].as_str().is_some());
        assert_eq!(matmul.1["rows"], 4);
        assert_eq!(matmul.1["inner"], 3);
        assert_eq!(matmul.1["cols"], 5);

        let prepacked = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "matmul_prepacked"
                    && data["rows"] == 4
                    && data["inner"] == 3
                    && data["cols"] == 5
            })
            .expect("prepacked matmul metadata event");
        assert_eq!(prepacked.1["requested_backend"], "auto");
        assert!(prepacked.1["backend"].as_str().is_some());
        assert_eq!(prepacked.1["rhs_layout"], "packed");
        assert_eq!(prepacked.1["packed_layout"], "col_major");
    }

    #[test]
    fn softmax_observer_meta_reports_cpu_backend() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let logits = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.0],
        ));
        let _ = unwrap_ok(logits.row_softmax_with_backend(SoftmaxBackend::Cpu));
        crate::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let softmax = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "row_softmax"
                    && data["rows"] == 2
                    && data["cols"] == 3
                    && data["backend"] == "cpu"
            })
            .expect("row_softmax metadata event");
        assert_eq!(softmax.1["requested_backend"], "cpu");
        assert_eq!(softmax.1["layout"], "row_major");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn forced_wgpu_softmax_reports_cpu_fallback_when_runtime_missing() {
        if !run_wgpu_runtime_tests() {
            eprintln!(
                "skipping runtime WGPU fallback test; set SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1"
            );
            return;
        }
        if wgpu_dense::is_available() {
            return;
        }

        with_strict_gpu_env(None, || {
            let _lock = observer_lock();
            let events = Arc::new(Mutex::new(Vec::new()));
            let captured = events.clone();
            let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
                captured
                    .lock()
                    .unwrap()
                    .push((event.op_name, event.data.clone()));
            })));

            let logits = unwrap_ok(Tensor::from_vec(
                2,
                3,
                vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.0],
            ));
            let _ = unwrap_ok(logits.row_softmax_with_backend(SoftmaxBackend::GpuWgpu));
            let _ = unwrap_ok(logits.row_softmax_hardmax_with_backend(SoftmaxBackend::GpuWgpu));
            let _ = unwrap_ok(logits.row_hardmax_with_backend(HardmaxBackend::GpuWgpu));
            crate::set_tensor_op_meta_observer(previous);

            let events = events.lock().unwrap();
            for op_name in ["row_softmax", "row_softmax_hardmax", "row_hardmax"] {
                let event = events
                    .iter()
                    .find(|(observed, data)| {
                        *observed == op_name
                            && data["requested_backend"] == "wgpu"
                            && data["backend"] == "cpu"
                    })
                    .unwrap_or_else(|| panic!("{op_name} WGPU-to-CPU fallback metadata event"));
                assert_eq!(event.1["rows"], 2);
                assert_eq!(event.1["cols"], 3);
                assert_eq!(event.1["fallback"]["from"], "wgpu");
                assert_eq!(event.1["fallback"]["to"], "cpu");
                assert_eq!(event.1["fallback"]["reason"], WGPU_RUNTIME_FALLBACK_REASON);
            }
        });
    }

    #[test]
    fn row_softmax_rejects_non_finite_logits() {
        let logits = unwrap_ok(Tensor::from_vec(1, 3, vec![0.0, f32::NAN, 1.0]));

        let err = unwrap_err(logits.row_softmax_with_backend(SoftmaxBackend::Cpu));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "row_softmax_input",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn row_softmax_hardmax_rejects_non_finite_logits() {
        let logits = unwrap_ok(Tensor::from_vec(1, 3, vec![0.0, f32::INFINITY, 1.0]));

        let err = unwrap_err(logits.row_softmax_hardmax_with_backend(SoftmaxBackend::Cpu));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "row_softmax_hardmax_input",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn row_softmax_hardmax_spiral_rejects_non_finite_logits() {
        let logits = unwrap_ok(Tensor::from_vec(1, 3, vec![0.0, f32::NEG_INFINITY, 1.0]));

        let err = unwrap_err(logits.row_softmax_hardmax_spiral_with_backend(SoftmaxBackend::Cpu));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "row_softmax_hardmax_spiral_input",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn row_hardmax_rejects_non_finite_logits() {
        let logits = unwrap_ok(Tensor::from_vec(1, 3, vec![0.0, f32::NAN, 1.0]));

        let err = unwrap_err(logits.row_hardmax_with_backend(HardmaxBackend::Cpu));

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "row_hardmax_input",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn fused_matmul_observer_meta_reports_selected_backend() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let lhs = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![0.5, -1.0, 1.5, 0.25, 0.75, -0.5],
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            3,
            2,
            vec![1.0, -0.75, -0.5, 0.33, 0.8, -0.2],
        ));
        let bias = vec![0.2, -0.1];
        let _ = unwrap_ok(lhs.matmul_bias_gelu_with_backend(&rhs, &bias, MatmulBackend::CpuNaive));
        crate::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let fused = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "matmul_bias_gelu"
                    && data["rows"] == 2
                    && data["inner"] == 3
                    && data["cols"] == 2
            })
            .expect("fused matmul metadata event");
        assert_eq!(fused.1["backend"], "naive");
        assert_eq!(fused.1["requested_backend"], "naive");
    }

    #[test]
    fn matmul_prepacked_transpose_matches_standard() {
        let lhs = unwrap_ok(Tensor::from_vec(
            4,
            3,
            vec![
                0.2, -0.4, 0.6, 1.1, -0.9, 0.7, 0.3, -0.2, 0.5, -1.3, 0.8, -0.1,
            ],
        ));
        let rhs = unwrap_ok(Tensor::from_vec(
            5,
            3,
            vec![
                0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -1.1, 1.2, -1.3, 1.4, -1.5,
            ],
        ));
        let rhs_t = rhs.transpose();
        let packed_t = unwrap_ok(PackedB::from_tensor_transpose(&rhs, Tile::col_major()));
        let standard = unwrap_ok(lhs.matmul(&rhs_t));
        let prepacked = unwrap_ok(lhs.matmul_prepacked(&packed_t));
        assert_eq!(standard, prepacked);
    }

    #[test]
    fn tensor_hadamard_matches_manual_product() {
        let a = unwrap_ok(Tensor::from_vec(2, 2, vec![1.5, -2.0, 0.5, 3.0]));
        let b = unwrap_ok(Tensor::from_vec(2, 2, vec![2.0, 4.0, -1.0, 0.5]));
        let product = unwrap_ok(a.hadamard(&b));
        let expected = unwrap_ok(Tensor::from_vec(2, 2, vec![3.0, -8.0, -0.5, 1.5]));
        assert_eq!(product, expected);
    }

    #[test]
    fn linear_regression_converges_without_tracebacks() {
        let inputs = unwrap_ok(Tensor::from_vec(4, 1, vec![0.0, 1.0, 2.0, 3.0]));
        let targets = unwrap_ok(Tensor::from_vec(4, 1, vec![1.0, 3.0, 5.0, 7.0]));
        let mut model = unwrap_ok(LinearModel::new(1, 1));
        let mut loss = 0.0;
        for _ in 0..200 {
            loss = unwrap_ok(model.train_batch(&inputs, &targets, 0.1));
        }
        assert!(loss < 1e-3, "loss should converge, got {loss}");

        let predictions = unwrap_ok(model.forward(&inputs));
        let mse = unwrap_ok(mean_squared_error(&predictions, &targets));
        assert!(mse < 1e-3, "model should fit the line, got {mse}");

        let weight = model.weights().data()[0];
        let bias = model.bias()[0];
        assert!((weight - 2.0).abs() < 1e-2, "weight too far: {weight}");
        assert!((bias - 1.0).abs() < 1e-2, "bias too far: {bias}");
    }

    #[test]
    fn language_wave_encoder_maps_text_without_tokens() {
        let encoder = unwrap_ok(LanguageWaveEncoder::new(-1.0, 0.5));
        let wave = unwrap_ok(encoder.encode_wave("spiral"));
        assert_eq!(wave.shape(), (1, 6));
        let z = unwrap_ok(encoder.encode_z_space("spiral"));
        assert_eq!(z.shape().0, 1);
        assert_eq!(z.shape().1, 12);
    }

    #[test]
    fn amega_hypergrad_tracks_z_space_updates() {
        let encoder = unwrap_ok(LanguageWaveEncoder::new(-1.25, 0.9));
        let z = unwrap_ok(encoder.encode_z_space("non-euclidean waves stay token free"));
        let shape = z.shape();
        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(
            encoder.curvature(),
            0.05,
            shape.0,
            shape.1,
        ));
        unwrap_ok(hypergrad.accumulate_wave(&z));
        let mut weights = unwrap_ok(Tensor::zeros(shape.0, shape.1));
        let targets = unwrap_ok(Tensor::zeros(shape.0, shape.1));
        unwrap_ok(hypergrad.accumulate_pair(&z, &targets));
        unwrap_ok(hypergrad.apply(&mut weights));
        assert_eq!(weights.shape(), shape);
        assert!(weights.squared_l2_norm() > 0.0);
    }

    #[test]
    fn amega_hypergrad_absorbs_text_directly() {
        let encoder = unwrap_ok(LanguageWaveEncoder::new(-0.8, 0.7));
        let z = unwrap_ok(encoder.encode_z_space("SpiralTorch dances in Z-space"));
        let shape = z.shape();
        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(
            encoder.curvature(),
            0.02,
            shape.0,
            shape.1,
        ));
        unwrap_ok(hypergrad.absorb_text(&encoder, "SpiralTorch dances in Z-space"));
        assert!(hypergrad
            .gradient()
            .iter()
            .any(|value| value.abs() > f32::EPSILON));
    }

    #[test]
    fn amega_hypergrad_telemetry_reports_guard_state() {
        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(-1.1, 0.03, 1, 4));
        let tensor = unwrap_ok(Tensor::from_vec(1, 4, vec![0.5, -0.25, 0.75, -0.125]));
        unwrap_ok(hypergrad.accumulate_wave(&tensor));
        let telemetry = hypergrad.telemetry();
        assert_eq!(telemetry.shape(), (1, 4));
        assert_eq!(telemetry.volume(), 4);
        assert_eq!(telemetry.curvature(), -1.1);
        assert_eq!(telemetry.learning_rate(), 0.03);
        let summary = telemetry.summary();
        assert_eq!(summary.count(), 4);
        assert!(summary.l1() > 0.0);
        assert!(telemetry.saturation() > 0.0);
        assert!(telemetry.tolerance() > 0.0);
        assert!(telemetry.max_volume() >= telemetry.volume());
        assert!(telemetry.max_depth() > 0);
    }

    #[test]
    fn amega_hypergrad_desire_control_matches_gain() {
        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(-0.95, 0.04, 1, 3));
        let tensor = unwrap_ok(Tensor::from_vec(1, 3, vec![0.8, -0.4, 0.2]));
        unwrap_ok(hypergrad.accumulate_wave(&tensor));
        let real = GradientSummary::from_slice(&[0.4, -0.2, 0.1]);
        let interpretation = hypergrad.desire_interpretation(real);
        assert!(interpretation.hyper_pressure() > interpretation.real_pressure());
        let neutral = hypergrad.desire_control(real);
        let tempered = hypergrad.desire_control_with_gain(real, 0.5);
        assert!(neutral.penalty_gain() >= tempered.penalty_gain());
        assert!(neutral.hyper_rate_scale() >= tempered.hyper_rate_scale());
        assert!(neutral.events().bits() != 0);
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
            unwrap_ok(Tensor::from_vec(1, 2, vec![0.8, 0.2])),
            unwrap_ok(Tensor::from_vec(1, 2, vec![0.3, 0.7])),
        ];
        let weights = vec![1.0, 2.0];
        let result = unwrap_ok(z_space_barycenter(&weights, &densities, 0.1, 0.0, None));
        let mut tape = unwrap_ok(AmegaHypergrad::new(-1.0, 0.05, 1, 2));
        unwrap_ok(tape.accumulate_barycenter_path(&result.intermediates));
        let gradient = tape.gradient();
        assert!(gradient.iter().any(|value| value.abs() > 0.0));
    }

    #[test]
    fn amega_realgrad_accumulates_and_applies() {
        let mut tape = unwrap_ok(AmegaRealgrad::new(0.1, 1, 3));
        let tensor = unwrap_ok(Tensor::from_vec(1, 3, vec![1.0, -2.0, 0.5]));
        unwrap_ok(tape.accumulate_wave(&tensor));
        let mut weights = unwrap_ok(Tensor::from_vec(1, 3, vec![0.0, 0.0, 0.0]));
        unwrap_ok(tape.apply(&mut weights));
        assert!((weights.data()[0] + 0.1).abs() < 1e-6);
        assert!((weights.data()[1] - 0.2).abs() < 1e-6);
        assert!((weights.data()[2] + 0.05).abs() < 1e-6);
    }

    #[test]
    fn amega_realgrad_absorbs_text() {
        let encoder = unwrap_ok(LanguageWaveEncoder::new(-1.0, 0.6));
        let z = unwrap_ok(encoder.encode_z_space("spiral torch realgrad"));
        let mut tape = unwrap_ok(AmegaRealgrad::new(0.05, z.shape().0, z.shape().1));
        unwrap_ok(tape.absorb_text(&encoder, "spiral torch realgrad"));
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
        assert!((summary.sum() - 2.0).abs() < 1e-6);
        assert!((summary.mean() - 0.5).abs() < 1e-6);
        assert!(summary.variance() > 0.0);
        assert!(summary.std() > 0.0);
        assert!(summary.kurtosis() >= 0.0);
        assert_eq!(summary.min(), -2.0);
        assert_eq!(summary.max(), 3.0);
        assert_eq!(summary.positive_count(), 2);
        assert_eq!(summary.negative_count(), 1);
        assert_eq!(summary.zero_count(), 1);
        assert_eq!(summary.near_zero_count(), 1);
        assert!((summary.activation() - 0.75).abs() < 1e-6);
        assert!((summary.sign_lean() - 0.25).abs() < 1e-6);
        let entropy = summary.sign_entropy();
        assert!(entropy > 0.0 && entropy < 1.0);
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
    fn gradient_summary_extended_moments_matches_slice() {
        let values: [f32; 4] = [1.0, -2.0, 0.0, 3.0];
        let from_slice = GradientSummary::from_slice(&values);
        let sum: f32 = values.iter().sum();
        let sum_squares: f32 = values.iter().map(|v| v * v).sum();
        let sum_cubes: f32 = values.iter().map(|v| v * v * v).sum();
        let sum_quartic: f32 = values.iter().map(|v| v * v * v * v).sum();
        let from_moments = GradientSummary::from_extended_moments(
            6.0,
            sum,
            sum_squares,
            sum_cubes,
            sum_quartic,
            3.0,
            values.len(),
        )
        .with_support(
            from_slice.min(),
            from_slice.max(),
            from_slice.positive_count(),
            from_slice.negative_count(),
            from_slice.near_zero_count(),
        );
        assert!((from_slice.sum() - from_moments.sum()).abs() < 1e-6);
        assert!((from_slice.skewness() - from_moments.skewness()).abs() < 1e-6);
        assert!((from_slice.kurtosis() - from_moments.kurtosis()).abs() < 1e-6);
        assert_eq!(from_slice.positive_count(), from_moments.positive_count());
        assert_eq!(from_slice.negative_count(), from_moments.negative_count());
        assert_eq!(from_slice.near_zero_count(), from_moments.near_zero_count());
        assert!((from_slice.sign_entropy() - from_moments.sign_entropy()).abs() < 1e-6);
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
        assert!(interpretation.hyper_std() > 0.0);
        assert!(interpretation.sharpness() >= 0.0);
        assert!(interpretation.activation() > 0.0);
        assert!(interpretation.sign_alignment() >= 0.0);
        assert!(interpretation.sign_entropy() > 0.0);
        assert!(stable.sign_alignment() >= interpretation.sign_alignment());
    }

    #[test]
    fn gradient_tapes_surface_summary_metrics() {
        let tensor = unwrap_ok(Tensor::from_vec(1, 3, vec![1.0, -2.0, 4.0]));
        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(-1.0, 0.1, 1, 3));
        unwrap_ok(hypergrad.accumulate_wave(&tensor));
        let hyper_summary = hypergrad.summary();
        assert_eq!(hyper_summary.count(), 3);
        assert!(hyper_summary.l2() > 0.0);

        let mut realgrad = unwrap_ok(AmegaRealgrad::new(0.1, 1, 3));
        unwrap_ok(realgrad.accumulate_wave(&tensor));
        let real_summary = realgrad.summary();
        assert_eq!(real_summary.count(), 3);
        let expected_l1: f32 = tensor.data().iter().map(|value| value.abs()).sum();
        assert!((real_summary.l1() - expected_l1).abs() < 1e-6);
    }

    #[test]
    fn hypergrad_summary_stays_in_sync_with_updates() {
        let mut tape = unwrap_ok(AmegaHypergrad::new(-1.0, 0.05, 2, 2));
        let wave = unwrap_ok(Tensor::from_vec(2, 2, vec![0.5, -0.25, 1.2, -0.9]));
        unwrap_ok(tape.accumulate_wave(&wave));
        let after_wave = tape.summary();
        let expected_wave = GradientSummary::from_slice(tape.gradient());
        assert_summary_close(after_wave, expected_wave);

        let prediction = unwrap_ok(Tensor::from_vec(2, 2, vec![0.8, -0.6, 0.3, -0.1]));
        let target = unwrap_ok(Tensor::from_vec(2, 2, vec![0.2, -0.3, 0.05, -0.15]));
        unwrap_ok(tape.accumulate_pair(&prediction, &target));
        let after_pair = tape.summary();
        let expected_pair = GradientSummary::from_slice(tape.gradient());
        assert_summary_close(after_pair, expected_pair);

        let mut weights = unwrap_ok(Tensor::from_vec(2, 2, vec![0.05, -0.05, 0.1, -0.1]));
        unwrap_ok(tape.apply(&mut weights));
        let reset = tape.summary();
        assert_eq!(reset.count(), 4);
        assert!(reset.l1() <= 1e-5);
        assert!(reset.l2() <= 1e-5);
        assert_eq!(reset.near_zero_count(), reset.count());
        assert!(reset.min().abs() <= 1e-6);
        assert!(reset.max().abs() <= 1e-6);
    }

    #[test]
    fn gradient_tapes_scale_with_backend_preserve_summaries() {
        let wave = unwrap_ok(Tensor::from_vec(2, 2, vec![0.5, -0.25, 1.2, -0.9]));
        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(-1.0, 0.05, 2, 2));
        unwrap_ok(hypergrad.accumulate_wave(&wave));
        unwrap_ok(hypergrad.scale_gradient_with_backend(0.5, TensorUtilBackend::Cpu));
        let hyper_summary = hypergrad.summary();
        let expected_hyper = GradientSummary::from_slice(hypergrad.gradient());
        assert_summary_close(hyper_summary, expected_hyper);

        let mut realgrad = unwrap_ok(AmegaRealgrad::new(0.05, 2, 2));
        unwrap_ok(realgrad.accumulate_wave(&wave));
        unwrap_ok(realgrad.scale_gradient_with_backend(0.5, TensorUtilBackend::Cpu));
        for (scaled, original) in realgrad.gradient().iter().zip(wave.data().iter()) {
            assert!((*scaled - original * 0.5).abs() < 1e-6);
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn hypergrad_accumulate_wave_can_emit_wgpu_backend_when_forced() {
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let wave = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![0.5, -0.25, 0.75, -0.4, 0.2, 0.1],
        ));
        let mut tape = unwrap_ok(AmegaHypergrad::new(-1.0, 0.05, 2, 3));
        unwrap_ok(tape.accumulate_wave_with_backend(&wave, TensorUtilBackend::GpuWgpu));
        crate::set_tensor_op_meta_observer(previous);

        let expected = GradientSummary::from_slice(tape.gradient());
        assert_summary_close(tape.summary(), expected);
        let events = events.lock().unwrap();
        let (_, data) = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "hypergrad_accumulate_wave" && data["backend"] == "wgpu_dense"
            })
            .expect("wgpu hypergrad accumulate metadata event");
        assert_eq!(data["requested_backend"], "wgpu");
        assert_eq!(data["kernel"], "hypergrad.wgpu_accumulate_wave");
        assert_eq!(data["gradient_finite_values"], 6);
    }

    #[test]
    fn gradient_tapes_ignore_overflow_learning_rate_scale() {
        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(-1.0, f32::MAX, 1, 1));
        hypergrad.scale_learning_rate(2.0);
        assert_eq!(hypergrad.learning_rate(), f32::MAX);

        let mut realgrad = unwrap_ok(AmegaRealgrad::new(f32::MAX, 1, 1));
        realgrad.scale_learning_rate(2.0);
        assert_eq!(realgrad.learning_rate(), f32::MAX);
    }

    #[test]
    fn hypergrad_rejects_non_finite_constructor_scalars() {
        let err = match AmegaHypergrad::new(-1.0, f32::INFINITY, 1, 1) {
            Ok(_) => panic!("non-finite hypergrad learning rate should be rejected"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            TensorError::NonPositiveLearningRate { rate } if rate.is_infinite()
        ));

        let err = match AmegaHypergrad::new(f32::NAN, 0.1, 1, 1) {
            Ok(_) => panic!("non-finite hypergrad curvature should be rejected"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            TensorError::NonHyperbolicCurvature { curvature } if curvature.is_nan()
        ));
    }

    #[test]
    fn hypergrad_rejects_overflowing_apply_without_mutating_state() {
        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(-1.0, f32::MAX, 1, 1));
        hypergrad.gradient_mut()[0] = 2.0;
        let gradient_before = hypergrad.gradient().to_vec();
        let mut weights = unwrap_ok(Tensor::zeros(1, 1));
        let weights_before = weights.clone();

        let err = hypergrad
            .apply_with_backend(&mut weights, TensorUtilBackend::Cpu)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "hypergrad_delta",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(weights, weights_before);
        assert_eq!(hypergrad.gradient(), gradient_before.as_slice());
    }

    #[test]
    fn hypergrad_rejects_overflowing_gradient_scale_without_mutating_state() {
        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(-1.0, 0.1, 1, 1));
        hypergrad.gradient_mut()[0] = 2.0;
        let gradient_before = hypergrad.gradient().to_vec();

        let err = hypergrad
            .scale_gradient_with_backend(f32::MAX, TensorUtilBackend::Cpu)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "hypergrad_scaled_gradient",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(hypergrad.gradient(), gradient_before.as_slice());
    }

    #[test]
    fn hypergrad_rejects_overflowing_accumulate_pair_without_mutating_state() {
        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(-1.0, 0.1, 1, 1));
        hypergrad.gradient_mut()[0] = f32::MAX;
        let gradient_before = hypergrad.gradient().to_vec();
        let prediction = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let target = unwrap_ok(Tensor::zeros(1, 1));

        let err = hypergrad.accumulate_pair(&prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "hypergrad_accumulate_pair",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(hypergrad.gradient(), gradient_before.as_slice());
    }

    #[test]
    fn realgrad_rejects_overflowing_apply_without_mutating_state() {
        let mut realgrad = unwrap_ok(AmegaRealgrad::new(f32::MAX, 1, 1));
        let wave = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));
        unwrap_ok(realgrad.accumulate_wave(&wave));
        let mut weights = unwrap_ok(Tensor::zeros(1, 1));
        let weights_before = weights.clone();
        let gradient_before = realgrad.gradient().to_vec();

        let err = realgrad
            .apply_with_backend(&mut weights, TensorUtilBackend::Cpu)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "realgrad_delta",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(weights, weights_before);
        assert_eq!(realgrad.gradient(), gradient_before.as_slice());
    }

    #[test]
    fn realgrad_rejects_overflowing_gradient_scale_without_mutating_state() {
        let mut realgrad = unwrap_ok(AmegaRealgrad::new(0.1, 1, 1));
        let wave = unwrap_ok(Tensor::from_vec(1, 1, vec![2.0]));
        unwrap_ok(realgrad.accumulate_wave(&wave));
        let gradient_before = realgrad.gradient().to_vec();

        let err = realgrad
            .scale_gradient_with_backend(f32::MAX, TensorUtilBackend::Cpu)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "realgrad_scaled_gradient",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(realgrad.gradient(), gradient_before.as_slice());
    }

    #[test]
    fn realgrad_rejects_overflowing_next_weight_without_mutating_state() {
        let mut realgrad = unwrap_ok(AmegaRealgrad::new(f32::MAX, 1, 1));
        let wave = unwrap_ok(Tensor::from_vec(1, 1, vec![-0.5]));
        unwrap_ok(realgrad.accumulate_wave(&wave));
        let mut weights = unwrap_ok(Tensor::from_vec(1, 1, vec![f32::MAX]));
        let weights_before = weights.clone();
        let gradient_before = realgrad.gradient().to_vec();

        let err = realgrad
            .apply_with_backend(&mut weights, TensorUtilBackend::Cpu)
            .unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "realgrad_update",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(weights, weights_before);
        assert_eq!(realgrad.gradient(), gradient_before.as_slice());
    }

    #[test]
    fn gradient_tapes_apply_with_backend_clear_accumulators() {
        let wave = unwrap_ok(Tensor::from_vec(1, 3, vec![0.5, -0.25, 0.75]));

        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(-1.0, 0.05, 1, 3));
        unwrap_ok(hypergrad.accumulate_wave(&wave));
        let mut hyper_weights = unwrap_ok(Tensor::zeros(1, 3));
        unwrap_ok(hypergrad.apply_with_backend(&mut hyper_weights, TensorUtilBackend::Cpu));
        assert!(hyper_weights.squared_l2_norm() > 0.0);
        assert!(hypergrad.gradient().iter().all(|value| value.abs() <= 1e-6));

        let mut realgrad = unwrap_ok(AmegaRealgrad::new(0.1, 1, 3));
        unwrap_ok(realgrad.accumulate_wave(&wave));
        let mut real_weights = unwrap_ok(Tensor::zeros(1, 3));
        unwrap_ok(realgrad.apply_with_backend(&mut real_weights, TensorUtilBackend::Cpu));
        for (weight, grad) in real_weights.data().iter().zip(wave.data().iter()) {
            assert!((*weight + grad * 0.1).abs() < 1e-6);
        }
        assert!(realgrad.gradient().iter().all(|value| value.abs() <= 1e-6));
    }

    #[test]
    fn gradient_tapes_emit_cpu_metadata_for_accumulate_and_update() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = crate::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let wave = unwrap_ok(Tensor::from_vec(1, 3, vec![0.5, -0.25, 0.75]));
        let prediction = unwrap_ok(Tensor::from_vec(1, 3, vec![0.6, -0.1, 0.4]));
        let target = unwrap_ok(Tensor::from_vec(1, 3, vec![0.1, 0.2, -0.2]));

        let mut hypergrad = unwrap_ok(AmegaHypergrad::new(-1.0, 0.05, 1, 3));
        unwrap_ok(hypergrad.accumulate_wave(&wave));
        unwrap_ok(hypergrad.accumulate_pair(&prediction, &target));
        let mut weights = unwrap_ok(Tensor::zeros(1, 3));
        unwrap_ok(hypergrad.apply_with_backend(&mut weights, TensorUtilBackend::Cpu));

        let mut realgrad = unwrap_ok(AmegaRealgrad::new(0.1, 1, 3));
        unwrap_ok(realgrad.accumulate_wave(&wave));
        unwrap_ok(realgrad.accumulate_pair(&prediction, &target));

        crate::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let find = |op_name: &'static str| {
            events
                .iter()
                .find(|(observed, _)| *observed == op_name)
                .unwrap_or_else(|| panic!("{op_name} metadata event"))
        };

        let (_, hyper_wave) = find("hypergrad_accumulate_wave");
        assert_eq!(hyper_wave["backend"], "cpu");
        assert_eq!(hyper_wave["kind"], "gradient_tape_accumulate");
        assert_eq!(hyper_wave["curvature"], -1.0);
        assert_eq!(hyper_wave["gradient_finite_values"], 3);
        assert!(hyper_wave["gradient_l2"].as_f64().unwrap_or(0.0) > 0.0);

        let (_, hyper_pair) = find("hypergrad_accumulate_pair");
        assert_eq!(hyper_pair["backend"], "cpu");
        assert_eq!(hyper_pair["rhs_cols"], 3);
        assert_eq!(hyper_pair["target_non_finite_values"], 0);

        let (_, hyper_update) = find("hypergrad_apply_update");
        assert_eq!(hyper_update["backend"], "cpu");
        assert_eq!(hyper_update["kind"], "gradient_tape_update");
        assert_eq!(hyper_update["projection_requested_backend"], "cpu");
        assert_eq!(hyper_update["gradient_finite_values"], 3);

        let (_, real_wave) = find("realgrad_accumulate_wave");
        assert_eq!(real_wave["backend"], "cpu");
        assert_eq!(real_wave["kind"], "gradient_tape_accumulate");
        assert_eq!(real_wave["gradient_finite_values"], 3);

        let (_, real_pair) = find("realgrad_accumulate_pair");
        assert_eq!(real_pair["backend"], "cpu");
        assert_eq!(real_pair["prediction_non_finite_values"], 0);
        assert_eq!(real_pair["target_non_finite_values"], 0);
    }

    #[test]
    fn hypergrad_summary_rebuilds_after_manual_mutation() {
        let mut tape = unwrap_ok(AmegaHypergrad::new(-1.0, 0.05, 2, 3));
        let wave = unwrap_ok(Tensor::from_vec(
            2,
            3,
            vec![0.4, -0.3, 0.2, 0.1, -0.5, 0.75],
        ));
        unwrap_ok(tape.accumulate_wave(&wave));
        {
            let gradient = tape.gradient_mut();
            gradient.copy_from_slice(&[0.8, -0.1, 0.0, -0.25, 0.6, -0.9]);
        }
        let summary = tape.summary();
        let expected = GradientSummary::from_slice(tape.gradient());
        assert_summary_close(summary, expected);
    }

    #[test]
    fn hypergrad_summary_recovers_from_non_finite_entries_incrementally() {
        let mut tape = unwrap_ok(AmegaHypergrad::new(-1.0, 0.05, 1, 4));
        {
            let gradient = tape.gradient_mut();
            gradient.copy_from_slice(&[1.0, -2.0, 3.0, -4.0]);
        }
        let baseline = tape.summary();
        assert_eq!(baseline.count(), 4);
        assert!((baseline.l1() - 10.0).abs() < 1e-6);
        assert_eq!(tape.non_finite_count(), 0);
        assert!(!tape.has_non_finite());
        assert!(tape.non_finite_ratio().abs() < 1e-6);

        {
            let gradient = tape.gradient_mut();
            gradient[1] = f32::NAN;
            gradient[2] = f32::INFINITY;
        }
        let pre_repair = tape.summary();
        assert_eq!(pre_repair.count(), 2);
        assert!((pre_repair.l1() - 5.0).abs() < 1e-6);
        assert_eq!(pre_repair.positive_count(), 1);
        assert_eq!(pre_repair.negative_count(), 1);
        assert_eq!(tape.non_finite_count(), 2);
        assert!(tape.has_non_finite());
        assert!((tape.non_finite_ratio() - 0.5).abs() < 1e-6);

        let zeros = unwrap_ok(Tensor::from_vec(1, 4, vec![0.0f32; 4]));
        unwrap_ok(tape.accumulate_wave(&zeros));

        let repaired = tape.summary();
        assert_eq!(repaired.count(), 4);
        assert!((repaired.l1() - 5.0).abs() < 1e-6);
        assert_eq!(repaired.positive_count(), 1);
        assert_eq!(repaired.negative_count(), 1);
        assert_eq!(repaired.near_zero_count(), 2);
        assert!(repaired.min() <= -4.0);
        assert!(repaired.max() >= 1.0);
        assert_eq!(tape.non_finite_count(), 0);
        assert!(!tape.has_non_finite());
        assert!(tape.non_finite_ratio().abs() < 1e-6);
    }

    #[test]
    fn hypergrad_retune_updates_curvature_and_resets() {
        let mut tape = unwrap_ok(AmegaHypergrad::new(-1.0, 0.05, 1, 2));
        let tensor = unwrap_ok(Tensor::from_vec(1, 2, vec![0.25, -0.4]));
        unwrap_ok(tape.accumulate_wave(&tensor));
        assert!(tape.gradient().iter().any(|value| value.abs() > 0.0));
        unwrap_ok(tape.retune(-0.5, 0.1));
        assert!((tape.curvature() + 0.5).abs() < 1e-6);
        assert!((tape.learning_rate() - 0.1).abs() < 1e-6);
        assert!(tape.gradient().iter().all(|value| value.abs() < 1e-6));
        let guard = tape.topos();
        assert!((guard.curvature() + 0.5).abs() < 1e-6);
    }

    #[test]
    fn amega_realgrad_accumulates_pair() {
        let mut tape = unwrap_ok(AmegaRealgrad::new(0.01, 1, 2));
        let prediction = unwrap_ok(Tensor::from_vec(1, 2, vec![0.5, -0.5]));
        let target = unwrap_ok(Tensor::from_vec(1, 2, vec![0.25, -0.75]));
        unwrap_ok(tape.accumulate_pair(&prediction, &target));
        assert!((tape.gradient()[0] - 0.25).abs() < 1e-6);
        assert!((tape.gradient()[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn random_uniform_respects_bounds_and_is_convertible_to_ndarray() {
        let tensor = unwrap_ok(Tensor::random_uniform(4, 3, -0.25, 0.75, Some(7)));
        assert_eq!(tensor.shape(), (4, 3));
        assert!(tensor
            .data()
            .iter()
            .all(|value| (-0.25..0.75).contains(value)));
        let array = unwrap_ok(Array2::from_shape_vec((4, 3), tensor.data().to_vec()));
        assert_eq!(array.dim(), (4, 3));
        assert_eq!(array[[0, 0]], tensor.data()[0]);
    }

    #[test]
    fn random_initialisers_are_deterministic_with_seed() {
        let left = unwrap_ok(Tensor::random_normal(2, 2, 0.0, 1.0, Some(42)));
        let right = unwrap_ok(Tensor::random_normal(2, 2, 0.0, 1.0, Some(42)));
        assert_eq!(left.data(), right.data());
    }

    #[test]
    fn random_uniform_rejects_invalid_bounds() {
        let err = unwrap_err(Tensor::random_uniform(2, 2, 1.0, 1.0, None));
        assert!(matches!(err, TensorError::InvalidValue { .. }));
    }
}
