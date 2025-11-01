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
const GOLDEN_RATIO: f64 = 1.618_033_988_749_894_8;
const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_8;
const GOLDEN_RATIO_BIAS: f64 = 0.381_966_011_250_105_1;

/// Aggregate telemetry captured while building the spiral consensus weights.
#[derive(Clone, Copy, Debug, Default)]
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
            let p = f64::from(prob.max(0.0));
            if p > 0.0 {
                entropy -= p * p.ln();
            }
            hardmass += f64::from(mask.max(0.0));
        }

        let geodesic = entropy * ramanujan_ratio + hardmass * GOLDEN_RATIO;
        let enrichment = if geodesic > f64::EPSILON {
            leech_scale * geodesic
        } else {
            0.0
        };
        let scale = (1.0 + enrichment) as f32;
        total_entropy += entropy;
        total_hardmass += hardmass;
        total_enrichment += enrichment;

        let entropy_norm = (entropy / (entropy + 1.0)).clamp(0.0, 1.0);
        let hardmass_norm = (hardmass / cols as f64).clamp(0.0, 1.0);
        let enrichment_norm = (enrichment / (1.0 + enrichment.abs())).clamp(0.0, 1.0);
        total_coherence += (entropy_norm + hardmass_norm + enrichment_norm) / 3.0;

        for (index, (&prob, &mask)) in row_soft.iter().zip(row_hard.iter()).enumerate() {
            let fused_value =
                (GOLDEN_RATIO_CONJUGATE as f32) * prob + (GOLDEN_RATIO_BIAS as f32) * mask;
            fused[offset + index] = scale * fused_value;
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
        let transposed = tensor.transpose();
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
        if !(min < max) {
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
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
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

        let manager_ctx = Box::into_raw(state) as *mut ManagedTensorState as *mut c_void;
        let managed = Box::new(DLManagedTensor {
            dl_tensor,
            manager_ctx,
            deleter: Some(drop_exported_state),
        });

        Ok(Box::into_raw(managed))
    }

    /// Return a zero-copy view of the tensor with new row/column dimensions.
    pub fn view(&self, rows: usize, cols: usize) -> PureResult<Tensor> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
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
        let dst_slice = dst.data_mut();

        match backend {
            MatmulBackend::Auto => self.matmul_auto_into(other, dst_slice, rows, inner, cols)?,
            MatmulBackend::CpuSimd => {
                if !matches!(other.layout, Layout::RowMajor) {
                    return Err(TensorError::UnsupportedLayout {
                        label: "simd matmul expects row-major rhs",
                    });
                }
                cpu_dense::matmul_into(dst_slice, lhs, other.data(), rows, inner, cols).map_err(
                    |message| TensorError::BackendFailure {
                        backend: "cpu_simd",
                        message,
                    },
                )?;
            }
            MatmulBackend::CpuNaive => {
                let packed = PackedB::from_tensor(other, Tile::col_major())?;
                matmul_naive_packed_into(dst_slice, lhs, rows, inner, cols, &packed);
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
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => {
                if !matches!(other.layout, Layout::RowMajor) {
                    return Err(TensorError::UnsupportedLayout {
                        label: "wgpu matmul expects row-major rhs",
                    });
                }
                let rhs = other.data();
                let buffer = matmul_wgpu(lhs, rhs, rows, inner, cols)?;
                dst_slice.copy_from_slice(&buffer);
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                if !matches!(other.layout, Layout::RowMajor) {
                    return Err(TensorError::UnsupportedLayout {
                        label: "hip matmul expects row-major rhs",
                    });
                }
                let rhs = other.data();
                hip_dense::matmul_into(lhs, rhs, dst_slice, rows, inner, cols).map_err(
                    |message| TensorError::BackendFailure {
                        backend: "hip",
                        message,
                    },
                )?;
            }
        }

        Ok(())
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
        let dst_slice = dst.data_mut();

        match backend {
            MatmulBackend::Auto => {
                self.matmul_prepacked_auto_into(packed, dst_slice, rows, inner, cols)?;
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
            }
            MatmulBackend::CpuNaive => {
                matmul_naive_packed_into(dst_slice, lhs, rows, inner, cols, packed);
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
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => {
                return Err(TensorError::UnsupportedLayout {
                    label: "wgpu matmul does not accept prepacked operands",
                });
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                return Err(TensorError::BackendFailure {
                    backend: "hip",
                    message: "hip matmul does not yet support prepacked operands".into(),
                });
            }
        }

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
    ) -> PureResult<()> {
        #[cfg(feature = "wgpu")]
        {
            if matches!(other.layout, Layout::RowMajor)
                && wgpu_dense::is_available()
                && wgpu_dense::should_use(rows, inner, cols)
            {
                if let Ok(buffer) = wgpu_dense::matmul(self.data(), other.data(), rows, inner, cols)
                {
                    dst.copy_from_slice(&buffer);
                    return Ok(());
                }
            }
        }

        #[cfg(feature = "hip")]
        {
            if matches!(other.layout, Layout::RowMajor)
                && hip_dense::is_available()
                && hip_dense::should_use(rows, inner, cols)
            {
                if hip_dense::matmul_into(self.data(), other.data(), dst, rows, inner, cols).is_ok()
                {
                    return Ok(());
                }
            }
        }

        if matches!(other.layout, Layout::RowMajor) && cpu_dense::should_use(rows, inner, cols) {
            if let Ok(()) =
                cpu_dense::matmul_into(dst, self.data(), other.data(), rows, inner, cols)
            {
                return Ok(());
            }
        }

        let packed = PackedB::from_tensor(other, Tile::col_major())?;
        self.matmul_prepacked_auto_into(&packed, dst, rows, inner, cols)
    }

    fn matmul_prepacked_auto_into(
        &self,
        packed: &PackedB,
        dst: &mut [f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<()> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                if let Ok(buffer) =
                    wgpu_dense::matmul_prepacked(self.data(), packed, rows, inner, cols)
                {
                    dst.copy_from_slice(&buffer);
                    return Ok(());
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
                return Ok(());
            }
        }

        matmul_naive_packed_into(dst, self.data(), rows, inner, cols, packed);
        Ok(())
    }

    /// Row-wise softmax using automatic backend selection.
    pub fn row_softmax(&self) -> PureResult<Tensor> {
        self.row_softmax_with_backend(SoftmaxBackend::Auto)
    }

    /// Row-wise softmax with explicit backend control.
    pub fn row_softmax_with_backend(&self, backend: SoftmaxBackend) -> PureResult<Tensor> {
        let rows = self.rows;
        let cols = self.cols;

        match backend {
            SoftmaxBackend::Auto => self.row_softmax_auto(rows, cols),
            SoftmaxBackend::Cpu => {
                let buffer = row_softmax_cpu(self.data(), rows, cols);
                Tensor::from_vec(rows, cols, buffer)
            }
            #[cfg(feature = "wgpu")]
            SoftmaxBackend::GpuWgpu => {
                let data = wgpu_dense::row_softmax(self.data(), rows, cols, self.layout).map_err(
                    |message| TensorError::BackendFailure {
                        backend: "wgpu",
                        message,
                    },
                )?;
                Tensor::from_vec(rows, cols, data)
            }
        }
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

        match backend {
            SoftmaxBackend::Auto => self.row_softmax_hardmax_auto(rows, cols),
            SoftmaxBackend::Cpu => {
                let result = self.hardmax_fusion(
                    rows,
                    cols,
                    HardmaxBackend::Cpu,
                    HardmaxMode::SoftmaxAndMask,
                )?;
                self.fusion_pair_to_tensors(rows, cols, result)
            }
            #[cfg(feature = "wgpu")]
            SoftmaxBackend::GpuWgpu => {
                let result = self.hardmax_fusion(
                    rows,
                    cols,
                    HardmaxBackend::GpuWgpu,
                    HardmaxMode::SoftmaxAndMask,
                )?;
                self.fusion_pair_to_tensors(rows, cols, result)
            }
        }
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

        match backend {
            SoftmaxBackend::Auto => self.row_softmax_hardmax_spiral_auto(rows, cols),
            SoftmaxBackend::Cpu => {
                let result = self.hardmax_fusion(
                    rows,
                    cols,
                    HardmaxBackend::Cpu,
                    HardmaxMode::SoftmaxAndMask,
                )?;
                self.fusion_pair_to_spiral(rows, cols, result)
            }
            #[cfg(feature = "wgpu")]
            SoftmaxBackend::GpuWgpu => {
                let result = self.hardmax_fusion(
                    rows,
                    cols,
                    HardmaxBackend::GpuWgpu,
                    HardmaxMode::SoftmaxAndMask,
                )?;
                self.fusion_pair_to_spiral(rows, cols, result)
            }
        }
    }

    /// Row-wise hardmax using automatic backend selection.
    pub fn row_hardmax(&self) -> PureResult<Tensor> {
        self.row_hardmax_with_backend(HardmaxBackend::Auto)
    }

    /// Row-wise hardmax with explicit backend control.
    pub fn row_hardmax_with_backend(&self, backend: HardmaxBackend) -> PureResult<Tensor> {
        let rows = self.rows;
        let cols = self.cols;

        match backend {
            HardmaxBackend::Auto => self.row_hardmax_auto(rows, cols),
            HardmaxBackend::Cpu => {
                let result =
                    self.hardmax_fusion(rows, cols, HardmaxBackend::Cpu, HardmaxMode::MaskOnly)?;
                Tensor::from_vec(rows, cols, result.hardmax)
            }
            #[cfg(feature = "wgpu")]
            HardmaxBackend::GpuWgpu => {
                let result = self.hardmax_fusion(
                    rows,
                    cols,
                    HardmaxBackend::GpuWgpu,
                    HardmaxMode::MaskOnly,
                )?;
                Tensor::from_vec(rows, cols, result.hardmax)
            }
        }
    }

    fn row_softmax_auto(&self, rows: usize, cols: usize) -> PureResult<Tensor> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::supports_row_softmax(rows, cols) {
                if let Ok(buffer) = wgpu_dense::row_softmax(self.data(), rows, cols, self.layout) {
                    return Tensor::from_vec(rows, cols, buffer);
                }
            }
        }

        let buffer = row_softmax_cpu(self.data(), rows, cols);
        Tensor::from_vec(rows, cols, buffer)
    }

    fn row_softmax_hardmax_auto(&self, rows: usize, cols: usize) -> PureResult<(Tensor, Tensor)> {
        let result = self.hardmax_fusion(
            rows,
            cols,
            HardmaxBackend::Auto,
            HardmaxMode::SoftmaxAndMask,
        )?;
        self.fusion_pair_to_tensors(rows, cols, result)
    }

    fn row_softmax_hardmax_spiral_auto(
        &self,
        rows: usize,
        cols: usize,
    ) -> PureResult<SpiralSoftmaxHardmax> {
        let result = self.hardmax_fusion(
            rows,
            cols,
            HardmaxBackend::Auto,
            HardmaxMode::SoftmaxAndMask,
        )?;
        self.fusion_pair_to_spiral(rows, cols, result)
    }

    fn row_hardmax_auto(&self, rows: usize, cols: usize) -> PureResult<Tensor> {
        let result =
            self.hardmax_fusion(rows, cols, HardmaxBackend::Auto, HardmaxMode::MaskOnly)?;
        Tensor::from_vec(rows, cols, result.hardmax)
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
        if contexts == 0 || sequence == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: contexts,
                cols: sequence,
            });
        }
        if self.cols == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: self.rows,
                cols: self.cols,
            });
        }

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

        let head_dim = self.cols;
        let queries = self.data();
        let keys_data = keys.data();
        let values_data = values.data();

        let make_tensor = |buffer: Vec<f32>| Tensor::from_vec(expected_rows, head_dim, buffer);

        match backend {
            AttentionBackend::Auto => {
                #[cfg(feature = "wgpu")]
                {
                    if wgpu_dense::is_available()
                        && wgpu_dense::supports_fused_attention(contexts, sequence, head_dim)
                    {
                        if let Ok(buffer) = wgpu_dense::fused_attention(
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
                            return make_tensor(buffer);
                        }
                    }
                }

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
                );
                make_tensor(buffer)
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
                );
                make_tensor(buffer)
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
                make_tensor(data)
            }
            #[cfg(not(feature = "wgpu"))]
            AttentionBackend::GpuWgpu => Err(TensorError::BackendFailure {
                backend: "wgpu",
                message: "wgpu backend disabled at compile time".into(),
            }),
        }
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
        if Arc::ptr_eq(&self.data, &dst.data) || Arc::ptr_eq(&other.data, &dst.data) {
            return Err(TensorError::InvalidValue {
                label: "matmul_out_alias",
            });
        }

        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;
        let dst_slice = dst.data_mut();

        match backend {
            MatmulBackend::Auto => {
                self.matmul_bias_relu_into_auto(other, bias, dst_slice, rows, inner, cols)?;
            }
            MatmulBackend::CpuSimd => {
                cpu_dense::matmul_into(dst_slice, self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "cpu_simd",
                        message,
                    })?;
                add_bias_relu_inplace(dst_slice, rows, cols, bias);
            }
            MatmulBackend::CpuNaive => {
                matmul_naive_into(dst_slice, self.data(), other.data(), rows, inner, cols);
                add_bias_relu_inplace(dst_slice, rows, cols, bias);
            }
            MatmulBackend::CpuFaer => {
                faer_dense::matmul_into(dst_slice, self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "faer",
                        message,
                    })?;
                add_bias_relu_inplace(dst_slice, rows, cols, bias);
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
                dst_slice.copy_from_slice(&data);
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                hip_dense::matmul_into(self.data(), other.data(), dst_slice, rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "hip",
                        message,
                    })?;
                add_bias_relu_inplace(dst_slice, rows, cols, bias);
            }
        }

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
    ) -> PureResult<()> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                if let Ok(buffer) =
                    wgpu_dense::matmul_bias_relu(self.data(), other.data(), bias, rows, inner, cols)
                {
                    dst.copy_from_slice(&buffer);
                    return Ok(());
                }
            }
        }

        #[cfg(feature = "hip")]
        {
            if hip_dense::is_available() && hip_dense::should_use(rows, inner, cols) {
                if hip_dense::matmul_into(self.data(), other.data(), dst, rows, inner, cols).is_ok()
                {
                    add_bias_relu_inplace(dst, rows, cols, bias);
                    return Ok(());
                }
            }
        }

        if cpu_dense::should_use(rows, inner, cols) {
            if let Ok(()) =
                cpu_dense::matmul_into(dst, self.data(), other.data(), rows, inner, cols)
            {
                add_bias_relu_inplace(dst, rows, cols, bias);
                return Ok(());
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            if let Ok(()) =
                faer_dense::matmul_into(dst, self.data(), other.data(), rows, inner, cols)
            {
                add_bias_relu_inplace(dst, rows, cols, bias);
                return Ok(());
            }
        }

        matmul_naive_into(dst, self.data(), other.data(), rows, inner, cols);
        add_bias_relu_inplace(dst, rows, cols, bias);
        Ok(())
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

        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;

        let data = match backend {
            MatmulBackend::Auto => self.matmul_bias_gelu_auto(other, bias, rows, inner, cols)?,
            MatmulBackend::CpuSimd => {
                let mut buffer = vec![0.0; rows * cols];
                cpu_dense::matmul_into(&mut buffer, self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                    backend: "cpu_simd",
                    message,
                })?;
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                buffer
            }
            MatmulBackend::CpuNaive => {
                let mut buffer = matmul_naive(self.data(), other.data(), rows, inner, cols);
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                buffer
            }
            MatmulBackend::CpuFaer => {
                let mut buffer = faer_dense::matmul(self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                    backend: "faer",
                    message,
                })?;
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                buffer
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => {
                wgpu_dense::matmul_bias_gelu(self.data(), other.data(), bias, rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "wgpu",
                        message,
                    })?
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                let mut buffer = hip_dense::matmul(self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "hip",
                        message,
                    })?;
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                buffer
            }
        };

        Tensor::from_vec(rows, cols, data)
    }

    fn matmul_bias_gelu_auto(
        &self,
        other: &Tensor,
        bias: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<Vec<f32>> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                if let Ok(buffer) =
                    wgpu_dense::matmul_bias_gelu(self.data(), other.data(), bias, rows, inner, cols)
                {
                    return Ok(buffer);
                }
            }
        }

        #[cfg(feature = "hip")]
        {
            if hip_dense::is_available() && hip_dense::should_use(rows, inner, cols) {
                if let Ok(mut buffer) =
                    hip_dense::matmul(self.data(), other.data(), rows, inner, cols)
                {
                    add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                    return Ok(buffer);
                }
            }
        }

        if cpu_dense::should_use(rows, inner, cols) {
            let mut buffer = vec![0.0; rows * cols];
            if cpu_dense::matmul_into(&mut buffer, self.data(), other.data(), rows, inner, cols)
                .is_ok()
            {
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                return Ok(buffer);
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            if let Ok(mut buffer) = faer_dense::matmul(self.data(), other.data(), rows, inner, cols)
            {
                add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
                return Ok(buffer);
            }
        }

        let mut buffer = matmul_naive(self.data(), other.data(), rows, inner, cols);
        add_bias_gelu_inplace(&mut buffer, rows, cols, bias);
        Ok(buffer)
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
        if residual.shape() != (self.rows, other.cols) {
            return Err(TensorError::ShapeMismatch {
                left: residual.shape(),
                right: (self.rows, other.cols),
            });
        }
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
        let dst_slice = dst.data_mut();

        match backend {
            MatmulBackend::Auto => {
                self.matmul_bias_add_relu_into_auto(
                    other, bias, residual, dst_slice, rows, inner, cols,
                )?;
            }
            MatmulBackend::CpuSimd => {
                cpu_dense::matmul_into(dst_slice, self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "cpu_simd",
                        message,
                    })?;
                add_bias_residual_relu_inplace(dst_slice, rows, cols, bias, residual.data());
            }
            MatmulBackend::CpuNaive => {
                matmul_naive_into(dst_slice, self.data(), other.data(), rows, inner, cols);
                add_bias_residual_relu_inplace(dst_slice, rows, cols, bias, residual.data());
            }
            MatmulBackend::CpuFaer => {
                faer_dense::matmul_into(dst_slice, self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "faer",
                        message,
                    })?;
                add_bias_residual_relu_inplace(dst_slice, rows, cols, bias, residual.data());
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
                dst_slice.copy_from_slice(&data);
            }
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                hip_dense::matmul_into(self.data(), other.data(), dst_slice, rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "hip",
                        message,
                    })?;
                add_bias_residual_relu_inplace(dst_slice, rows, cols, bias, residual.data());
            }
        }

        Ok(())
    }

    fn matmul_bias_add_relu_into_auto(
        &self,
        other: &Tensor,
        bias: &[f32],
        residual: &Tensor,
        dst: &mut [f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<()> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                if let Ok(buffer) = wgpu_dense::matmul_bias_add_relu(
                    self.data(),
                    other.data(),
                    bias,
                    residual.data(),
                    rows,
                    inner,
                    cols,
                ) {
                    dst.copy_from_slice(&buffer);
                    return Ok(());
                }
            }
        }

        #[cfg(feature = "hip")]
        {
            if hip_dense::is_available() && hip_dense::should_use(rows, inner, cols) {
                if hip_dense::matmul_into(self.data(), other.data(), dst, rows, inner, cols).is_ok()
                {
                    add_bias_residual_relu_inplace(dst, rows, cols, bias, residual.data());
                    return Ok(());
                }
            }
        }

        if cpu_dense::should_use(rows, inner, cols) {
            if let Ok(()) =
                cpu_dense::matmul_into(dst, self.data(), other.data(), rows, inner, cols)
            {
                add_bias_residual_relu_inplace(dst, rows, cols, bias, residual.data());
                return Ok(());
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            if let Ok(()) =
                faer_dense::matmul_into(dst, self.data(), other.data(), rows, inner, cols)
            {
                add_bias_residual_relu_inplace(dst, rows, cols, bias, residual.data());
                return Ok(());
            }
        }

        matmul_naive_into(dst, self.data(), other.data(), rows, inner, cols);
        add_bias_residual_relu_inplace(dst, rows, cols, bias, residual.data());
        Ok(())
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
        if residual.shape() != (self.rows, other.cols) {
            return Err(TensorError::ShapeMismatch {
                left: residual.shape(),
                right: (self.rows, other.cols),
            });
        }

        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;

        let data = match backend {
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
                buffer
            }
            MatmulBackend::CpuNaive => {
                let mut buffer = matmul_naive(self.data(), other.data(), rows, inner, cols);
                add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                buffer
            }
            MatmulBackend::CpuFaer => {
                let mut buffer = faer_dense::matmul(self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                    backend: "faer",
                    message,
                })?;
                add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                buffer
            }
            #[cfg(feature = "wgpu")]
            MatmulBackend::GpuWgpu => wgpu_dense::matmul_bias_add_gelu(
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
            })?,
            #[cfg(feature = "hip")]
            MatmulBackend::GpuHip => {
                let mut buffer = hip_dense::matmul(self.data(), other.data(), rows, inner, cols)
                    .map_err(|message| TensorError::BackendFailure {
                        backend: "hip",
                        message,
                    })?;
                add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                buffer
            }
        };

        Tensor::from_vec(rows, cols, data)
    }

    fn matmul_bias_add_gelu_auto(
        &self,
        other: &Tensor,
        bias: &[f32],
        residual: &Tensor,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> PureResult<Vec<f32>> {
        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() && wgpu_dense::should_use(rows, inner, cols) {
                if let Ok(buffer) = wgpu_dense::matmul_bias_add_gelu(
                    self.data(),
                    other.data(),
                    bias,
                    residual.data(),
                    rows,
                    inner,
                    cols,
                ) {
                    return Ok(buffer);
                }
            }
        }

        #[cfg(feature = "hip")]
        {
            if hip_dense::is_available() && hip_dense::should_use(rows, inner, cols) {
                if let Ok(mut buffer) =
                    hip_dense::matmul(self.data(), other.data(), rows, inner, cols)
                {
                    add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                    return Ok(buffer);
                }
            }
        }

        if cpu_dense::should_use(rows, inner, cols) {
            let mut buffer = vec![0.0; rows * cols];
            if cpu_dense::matmul_into(&mut buffer, self.data(), other.data(), rows, inner, cols)
                .is_ok()
            {
                add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                return Ok(buffer);
            }
        }

        if faer_dense::is_available() && faer_dense::should_use(rows, inner, cols) {
            if let Ok(mut buffer) = faer_dense::matmul(self.data(), other.data(), rows, inner, cols)
            {
                add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
                return Ok(buffer);
            }
        }

        let mut buffer = matmul_naive(self.data(), other.data(), rows, inner, cols);
        add_bias_residual_gelu_inplace(&mut buffer, rows, cols, bias, residual.data());
        Ok(buffer)
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Tensor) -> PureResult<Tensor> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        let mut data = aligned_with_capacity(self.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            data.push(a + b);
        }
        Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> PureResult<Tensor> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        let mut data = aligned_with_capacity(self.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            data.push(a - b);
        }
        Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)
    }

    /// Returns a new tensor where every element is scaled by `value`.
    pub fn scale(&self, value: f32) -> PureResult<Tensor> {
        let mut data = aligned_with_capacity(self.len());
        for &a in self.data.iter() {
            data.push(a * value);
        }
        Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)
    }

    /// Element-wise product (Hadamard) between two tensors of identical shape.
    pub fn hadamard(&self, other: &Tensor) -> PureResult<Tensor> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        let mut data = aligned_with_capacity(self.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            data.push(a * b);
        }
        Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)
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

    /// Apply the ReLU activation in-place (`self[i] = max(self[i], 0)`).
    pub fn relu_inplace(&mut self) {
        let data = Arc::make_mut(&mut self.data);
        for value in data.iter_mut() {
            if *value < 0.0 {
                *value = 0.0;
            }
        }
    }

    /// Apply the GELU activation in-place (`self[i] = GELU(self[i])`).
    pub fn gelu_inplace(&mut self) {
        let data = Arc::make_mut(&mut self.data);
        for value in data.iter_mut() {
            *value = gelu(*value);
        }
    }

    /// Applies the derivative of GELU to the provided gradient tensor.
    pub fn gelu_backward(&self, grad_output: &Tensor) -> PureResult<Tensor> {
        if self.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, cols) = self.shape();

        #[cfg(feature = "wgpu")]
        {
            if wgpu_dense::is_available() {
                if let Ok(buffer) =
                    wgpu_dense::gelu_backward(self.data(), grad_output.data(), rows, cols)
                {
                    return Tensor::from_vec(rows, cols, buffer);
                }
            }
        }

        let mut data = Vec::with_capacity(rows * cols);
        for (z, g) in self.data().iter().zip(grad_output.data().iter()) {
            data.push(gelu_prime(*z) * g);
        }
        Tensor::from_vec(rows, cols, data)
    }

    /// Returns the transpose of the tensor.
    pub fn transpose(&self) -> Tensor {
        let mut data = aligned_zeroed(self.len());
        for r in 0..self.rows {
            for c in 0..self.cols {
                data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        Tensor {
            data: Arc::new(TensorBuffer::from_aligned(data)),
            rows: self.cols,
            cols: self.rows,
            layout: Layout::RowMajor,
        }
    }

    /// Returns a reshaped copy of the tensor when the requested dimensions are compatible.
    pub fn reshape(&self, rows: usize, cols: usize) -> PureResult<Tensor> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        if rows * cols != self.len() {
            return Err(TensorError::DataLength {
                expected: rows * cols,
                got: self.len(),
            });
        }

        if matches!(self.layout, Layout::RowMajor) {
            return self.view(rows, cols);
        }

        let row_major = self.to_layout(Layout::RowMajor)?;
        row_major.view(rows, cols)
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

    /// Returns the sum over columns for each row.
    pub fn sum_axis1(&self) -> Vec<f32> {
        let mut sums = vec![0.0; self.rows];
        for r in 0..self.rows {
            let offset = r * self.cols;
            let mut total = 0.0f32;
            for c in 0..self.cols {
                total += self.data[offset + c];
            }
            sums[r] = total;
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
        let mut data = aligned_with_capacity(total_rows * cols);
        for tensor in tensors {
            data.extend_from_slice(tensor.data.as_slice());
        }
        Tensor::from_aligned(total_rows, cols, data, Layout::RowMajor)
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
        let mut data = aligned_with_capacity(self.len());
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
        Tensor::from_aligned(self.rows, self.cols, data, Layout::RowMajor)
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
        if inputs.shape().1 != self.weights.shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: inputs.shape(),
                right: self.weights.shape(),
            });
        }
        let pack = self.ensure_packed_weights()?;
        let mut out = inputs.matmul_prepacked(&pack)?;
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
        let pack = self.ensure_packed_weights()?;
        let mut predictions = inputs.matmul_prepacked(&pack)?;
        predictions.add_row_inplace(&self.bias)?;
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
        self.packed_weights.borrow_mut().take();
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

    fn ensure_packed_weights(&self) -> PureResult<PackedB> {
        if let Some(existing) = self.packed_weights.borrow().clone() {
            return Ok(existing);
        }
        let pack = PackedB::from_tensor(&self.weights, Tile::col_major())?;
        *self.packed_weights.borrow_mut() = Some(pack.clone());
        Ok(pack)
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
        let summary = Self {
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
        };

        summary
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
        if !old.is_finite() || !new.is_finite() {
            self.rebuild_summary();
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
        HypergradTelemetry {
            summary: self.summary(),
            curvature: self.curvature,
            learning_rate: self.learning_rate,
            saturation: guard.saturation(),
            porosity: guard.porosity(),
            tolerance: guard.tolerance(),
            max_depth: guard.max_depth(),
            max_volume: guard.max_volume(),
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
            self.learning_rate *= factor;
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

    /// Clears the accumulated gradient back to zero.
    pub fn reset(&mut self) {
        for idx in 0..self.gradient.len() {
            let old = self.gradient[idx];
            let new = 0.0f32;
            if old.to_bits() != new.to_bits() {
                self.gradient[idx] = new;
                self.record_transition(old, new);
            } else {
                self.gradient[idx] = new;
            }
        }
    }

    /// Accumulates a Euclidean tensor inside the hyperbolic tape using
    /// the standard conformal factor for the Poincaré ball.
    pub fn accumulate_wave(&mut self, tensor: &Tensor) -> PureResult<()> {
        self.assert_tensor_shape(tensor)?;
        self.topos.guard_tensor("hypergrad_wave", tensor)?;
        let tolerance = self.topos.tolerance();
        let values = tensor.data();
        for idx in 0..self.gradient.len() {
            let value = values[idx];
            let denom = 1.0 - self.curvature * value * value;
            let update = value / denom.abs().max(tolerance);
            let old = self.gradient[idx];
            let candidate = old + update;
            let new = self.topos.saturate(candidate);
            if old.to_bits() != new.to_bits() {
                self.gradient[idx] = new;
                self.record_transition(old, new);
            } else {
                self.gradient[idx] = new;
            }
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
        let prediction_data = prediction.data();
        let target_data = target.data();
        for idx in 0..self.gradient.len() {
            let delta = prediction_data[idx] - target_data[idx];
            let old = self.gradient[idx];
            let candidate = old + delta;
            let new = self.topos.saturate(candidate);
            if old.to_bits() != new.to_bits() {
                self.gradient[idx] = new;
                self.record_transition(old, new);
            } else {
                self.gradient[idx] = new;
            }
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
) -> Vec<f32> {
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
                }

                let mut logit = dot * scale;
                if let Some(bias) = z_bias {
                    logit += bias[context_offset + key_idx];
                }
                if let Some(bias) = attn_bias {
                    logit += bias[query_row * sequence + key_idx];
                }

                let new_max = running_max.max(logit);
                let scaled_sum = if running_sum > 0.0 {
                    running_sum * (running_max - new_max).exp()
                } else {
                    0.0
                };
                let exp_curr = (logit - new_max).exp();
                let denom = scaled_sum + exp_curr;
                let alpha = if denom > 0.0 { scaled_sum / denom } else { 0.0 };
                let weight = if denom > 0.0 { exp_curr / denom } else { 0.0 };
                running_max = new_max;
                running_sum = denom;

                for dim in 0..head_dim {
                    accum[dim] = accum[dim] * alpha + weight * values[key_offset + dim];
                }
            }

            output[query_offset..query_offset + head_dim].copy_from_slice(&accum);
        }
    }

    output
}

fn cpu_row_softmax_hardmax(
    data: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    let expected = rows.saturating_mul(cols);
    let mut softmax = vec![0.0; expected];
    let mut hardmax = vec![0.0; expected];

    if expected == 0 || data.len() != expected {
        return (softmax, hardmax);
    }

    for row in 0..rows {
        let offset = row * cols;
        let input_row = &data[offset..offset + cols];
        let soft_row = &mut softmax[offset..offset + cols];
        let hard_row = &mut hardmax[offset..offset + cols];

        soft_row.fill(0.0);
        hard_row.fill(0.0);

        let mut row_max = f32::NEG_INFINITY;
        let mut argmax_index: Option<usize> = None;
        let mut finite_values = 0usize;

        for (index, &value) in input_row.iter().enumerate() {
            if value.is_nan() {
                continue;
            }

            finite_values += 1;
            if value > row_max || argmax_index.is_none() {
                row_max = value;
                argmax_index = Some(index);
            }
        }

        if finite_values == 0 {
            continue;
        }

        let mut sum = 0.0f32;
        for (prob_slot, &value) in soft_row.iter_mut().zip(input_row.iter()) {
            if value.is_nan() {
                *prob_slot = 0.0;
                continue;
            }

            let shifted = if row_max.is_infinite() {
                if value == row_max { 1.0 } else { 0.0 }
            } else {
                (value - row_max).exp()
            };

            sum += shifted;
            *prob_slot = shifted;
        }

        let inv_sum = if sum.is_finite() && sum > 0.0 {
            sum.recip()
        } else {
            0.0
        };

        if inv_sum > 0.0 {
            for prob in soft_row.iter_mut() {
                *prob *= inv_sum;
            }
        } else {
            soft_row.fill(0.0);
        }

        if let Some(idx) = argmax_index {
            if let Some(slot) = hard_row.get_mut(idx) {
                *slot = 1.0;
            }
        }
    }

    (softmax, hardmax)
}

fn row_softmax_cpu(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    cpu_row_softmax_hardmax(data, rows, cols).0
}

#[cfg_attr(feature = "wgpu", allow(dead_code))]
fn row_hardmax_cpu(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    cpu_row_softmax_hardmax(data, rows, cols).1
}

fn add_bias_relu_inplace(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    for r in 0..rows {
        let offset = r * cols;
        for c in 0..cols {
            let index = offset + c;
            let sum = data[index] + bias[c];
            data[index] = if sum > 0.0 { sum } else { 0.0 };
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
    const SQRT_2_OVER_PI: f32 = 0.797_884_560_802_865_4;
    const KAPPA: f32 = 0.044_715;
    let x2 = x * x;
    let inner = SQRT_2_OVER_PI * (x + KAPPA * x * x2);
    let t = inner.tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * KAPPA * x2)
}

fn add_bias_gelu_inplace(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    for r in 0..rows {
        let offset = r * cols;
        for c in 0..cols {
            let index = offset + c;
            let sum = data[index] + bias[c];
            data[index] = gelu(sum);
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
    for r in 0..rows {
        let offset = r * cols;
        for c in 0..cols {
            let index = offset + c;
            let sum = data[index] + bias[c] + residual[index];
            data[index] = if sum > 0.0 { sum } else { 0.0 };
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
    for r in 0..rows {
        let offset = r * cols;
        for c in 0..cols {
            let index = offset + c;
            let sum = data[index] + bias[c] + residual[index];
            data[index] = gelu(sum);
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

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
        let lhs = Tensor::from_vec(2, 3, vec![1.0, -2.0, 0.5, 0.25, 1.5, -0.75]).unwrap();
        let rhs = Tensor::from_vec(3, 2, vec![0.5, -1.0, 2.0, 0.25, -0.5, 1.0]).unwrap();
        let bias = vec![0.5, -0.25];

        let fused = lhs.matmul_bias_relu(&rhs, &bias).unwrap();

        let mut reference = lhs.matmul(&rhs).unwrap();
        reference.add_row_inplace(&bias).unwrap();
        reference.relu_inplace();

        assert_eq!(fused.shape(), reference.shape());
        for (a, b) in fused.data().iter().zip(reference.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn matmul_bias_gelu_matches_scalar_pipeline() {
        let lhs =
            Tensor::from_vec(2, 4, vec![1.0, -1.5, 0.75, 2.0, -0.25, 0.5, 1.25, -0.75]).unwrap();
        let rhs = Tensor::from_vec(
            4,
            3,
            vec![
                0.5, -0.25, 1.0, 1.5, 0.75, -1.0, -0.5, 0.33, 0.8, -0.2, 1.2, 0.6,
            ],
        )
        .unwrap();
        let bias = vec![0.1, -0.05, 0.2];

        let fused = lhs.matmul_bias_gelu(&rhs, &bias).unwrap();

        let mut reference = lhs.matmul(&rhs).unwrap();
        reference.add_row_inplace(&bias).unwrap();
        reference.gelu_inplace();

        assert_eq!(fused.shape(), reference.shape());
        for (a, b) in fused.data().iter().zip(reference.data().iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn matmul_bias_add_relu_matches_scalar_pipeline() {
        let lhs = Tensor::from_vec(
            3,
            4,
            vec![
                1.0, -0.5, 0.25, 2.0, 0.75, -1.25, 1.5, -0.75, 0.33, 0.5, -0.25, 1.0,
            ],
        )
        .unwrap();
        let rhs = Tensor::from_vec(
            4,
            3,
            vec![
                0.5, 1.25, -0.75, -1.0, 0.75, 0.5, 1.5, -0.25, 0.33, -0.66, 0.25, 0.8,
            ],
        )
        .unwrap();
        let bias = vec![0.2, -0.1, 0.05];
        let residual =
            Tensor::from_vec(3, 3, vec![0.1, 0.2, -0.3, -0.4, 0.5, 0.6, 0.0, -0.2, 0.3]).unwrap();

        let fused = lhs.matmul_bias_add_relu(&rhs, &bias, &residual).unwrap();

        let mut reference = lhs.matmul(&rhs).unwrap();
        reference.add_row_inplace(&bias).unwrap();
        let mut reference = reference.add(&residual).unwrap();
        reference.relu_inplace();

        assert_eq!(fused.shape(), reference.shape());
        for (a, b) in fused.data().iter().zip(reference.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn tensor_gelu_backward_matches_manual() {
        let input = Tensor::from_vec(2, 3, vec![-1.0, 0.25, 0.75, -0.5, 0.0, 1.25]).unwrap();
        let grad = Tensor::from_vec(2, 3, vec![0.5, -0.75, 0.3, -0.2, 0.4, 0.1]).unwrap();
        let result = input.gelu_backward(&grad).unwrap();
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
        let (gz, dr, db) =
            wgpu_dense::fused_gelu_backward(&z, &grad, Some(&residual), rows, cols).unwrap();

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

    #[test]
    fn matmul_bias_add_gelu_matches_scalar_pipeline() {
        let lhs = Tensor::from_vec(2, 3, vec![0.5, -1.0, 1.5, 0.25, 0.75, -0.5]).unwrap();
        let rhs = Tensor::from_vec(
            3,
            3,
            vec![1.0, -0.75, 0.5, -0.5, 0.33, 1.25, 0.8, -0.2, 0.4],
        )
        .unwrap();
        let bias = vec![0.2, -0.1, 0.05];
        let residual = Tensor::from_vec(2, 3, vec![0.1, -0.05, 0.0, -0.2, 0.3, 0.4]).unwrap();

        let fused = lhs.matmul_bias_add_gelu(&rhs, &bias, &residual).unwrap();

        let mut reference = lhs.matmul(&rhs).unwrap();
        reference.add_row_inplace(&bias).unwrap();
        let mut reference = reference.add(&residual).unwrap();
        reference.gelu_inplace();

        assert_eq!(fused.shape(), reference.shape());
        for (a, b) in fused.data().iter().zip(reference.data().iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

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
    fn matmul_prepacked_matches_standard() {
        let lhs = Tensor::from_vec(
            4,
            3,
            vec![
                1.0, -0.5, 2.0, 0.25, 1.5, -1.25, 0.75, 0.5, -0.75, 1.0, -1.5, 0.33,
            ],
        )
        .unwrap();
        let rhs = Tensor::from_vec(
            3,
            5,
            vec![
                0.5, -1.0, 0.25, 1.5, -0.75, 1.0, 0.5, -0.5, 0.75, -1.25, 0.66, 0.8, -0.2, 1.2,
                -0.4,
            ],
        )
        .unwrap();
        let packed = PackedB::from_tensor(&rhs, Tile::col_major()).unwrap();
        let standard = lhs.matmul(&rhs).unwrap();
        let prepacked = lhs.matmul_prepacked(&packed).unwrap();
        assert_eq!(standard, prepacked);
    }

    #[test]
    fn matmul_prepacked_transpose_matches_standard() {
        let lhs = Tensor::from_vec(
            4,
            3,
            vec![
                0.2, -0.4, 0.6, 1.1, -0.9, 0.7, 0.3, -0.2, 0.5, -1.3, 0.8, -0.1,
            ],
        )
        .unwrap();
        let rhs = Tensor::from_vec(
            5,
            3,
            vec![
                0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -1.1, 1.2, -1.3, 1.4, -1.5,
            ],
        )
        .unwrap();
        let rhs_t = rhs.transpose();
        let packed_t = PackedB::from_tensor_transpose(&rhs, Tile::col_major()).unwrap();
        let standard = lhs.matmul(&rhs_t).unwrap();
        let prepacked = lhs.matmul_prepacked(&packed_t).unwrap();
        assert_eq!(standard, prepacked);
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
    fn amega_hypergrad_telemetry_reports_guard_state() {
        let mut hypergrad = AmegaHypergrad::new(-1.1, 0.03, 1, 4).unwrap();
        let tensor = Tensor::from_vec(1, 4, vec![0.5, -0.25, 0.75, -0.125]).unwrap();
        hypergrad.accumulate_wave(&tensor).unwrap();
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
        let mut hypergrad = AmegaHypergrad::new(-0.95, 0.04, 1, 3).unwrap();
        let tensor = Tensor::from_vec(1, 3, vec![0.8, -0.4, 0.2]).unwrap();
        hypergrad.accumulate_wave(&tensor).unwrap();
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
    fn hypergrad_summary_stays_in_sync_with_updates() {
        let mut tape = AmegaHypergrad::new(-1.0, 0.05, 2, 2).unwrap();
        let wave = Tensor::from_vec(2, 2, vec![0.5, -0.25, 1.2, -0.9]).unwrap();
        tape.accumulate_wave(&wave).unwrap();
        let after_wave = tape.summary();
        let expected_wave = GradientSummary::from_slice(tape.gradient());
        assert_summary_close(after_wave, expected_wave);

        let prediction = Tensor::from_vec(2, 2, vec![0.8, -0.6, 0.3, -0.1]).unwrap();
        let target = Tensor::from_vec(2, 2, vec![0.2, -0.3, 0.05, -0.15]).unwrap();
        tape.accumulate_pair(&prediction, &target).unwrap();
        let after_pair = tape.summary();
        let expected_pair = GradientSummary::from_slice(tape.gradient());
        assert_summary_close(after_pair, expected_pair);

        let mut weights = Tensor::from_vec(2, 2, vec![0.05, -0.05, 0.1, -0.1]).unwrap();
        tape.apply(&mut weights).unwrap();
        let reset = tape.summary();
        assert_eq!(reset.count(), 4);
        assert!(reset.l1() <= 1e-5);
        assert!(reset.l2() <= 1e-5);
        assert_eq!(reset.near_zero_count(), reset.count());
        assert!(reset.min().abs() <= 1e-6);
        assert!(reset.max().abs() <= 1e-6);
    }

    #[test]
    fn hypergrad_summary_rebuilds_after_manual_mutation() {
        let mut tape = AmegaHypergrad::new(-1.0, 0.05, 2, 3).unwrap();
        let wave = Tensor::from_vec(2, 3, vec![0.4, -0.3, 0.2, 0.1, -0.5, 0.75]).unwrap();
        tape.accumulate_wave(&wave).unwrap();
        {
            let gradient = tape.gradient_mut();
            gradient.copy_from_slice(&[0.8, -0.1, 0.0, -0.25, 0.6, -0.9]);
        }
        let summary = tape.summary();
        let expected = GradientSummary::from_slice(tape.gradient());
        assert_summary_close(summary, expected);
    }

    #[test]
    fn hypergrad_scale_gradient_stays_in_sync() {
        let mut tape = AmegaHypergrad::new(-0.9, 0.05, 2, 2).unwrap();
        let wave = Tensor::from_vec(2, 2, vec![0.45, -0.3, 0.15, 0.2]).unwrap();
        tape.accumulate_wave(&wave).unwrap();
        let before = tape.gradient().to_vec();
        tape.scale_gradient(-0.5);
        let after = tape.gradient();
        for (prev, current) in before.iter().zip(after.iter()) {
            assert!((current + 0.5 * prev).abs() <= 1e-6);
        }
        let summary = tape.summary();
        let expected = GradientSummary::from_slice(after);
        assert_summary_close(summary, expected);
    }

    #[test]
    fn hypergrad_rescale_rms_matches_target() {
        let mut tape = AmegaHypergrad::new(-0.85, 0.05, 2, 3).unwrap();
        let tensor = Tensor::from_vec(2, 3, vec![0.5, -0.4, 0.25, 0.6, -0.3, 0.1]).unwrap();
        tape.accumulate_wave(&tensor).unwrap();
        let initial = tape.summary();
        let target = initial.rms() * 0.4;
        let factor = tape.rescale_rms(target);
        assert!((factor - 0.4).abs() <= 5e-3);
        let summary = tape.summary();
        assert!((summary.rms() - target).abs() <= 5e-3);
        let expected = GradientSummary::from_slice(tape.gradient());
        assert_summary_close(summary, expected);
    }

    #[test]
    fn hypergrad_rescale_rms_on_dormant_gradient_is_noop() {
        let mut tape = AmegaHypergrad::new(-0.95, 0.05, 1, 2).unwrap();
        assert_eq!(tape.summary().rms(), 0.0);
        let factor = tape.rescale_rms(0.25);
        assert_eq!(factor, 0.0);
        let summary = tape.summary();
        assert_eq!(summary.rms(), 0.0);
        assert_eq!(summary.l1(), 0.0);
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
