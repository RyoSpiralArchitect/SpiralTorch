// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use num_complex::Complex;
use thiserror::Error;

/// Primary floating-point scalar for Mellin tooling.
pub type Scalar = f32;

/// Complex companion to the Mellin scalar.
pub type ComplexScalar = Complex<Scalar>;

#[derive(Debug, Error)]
pub enum ZSpaceError {
    #[error("at least one sample is required")]
    EmptySamples,
    #[error("weights must match samples (samples={samples}, weights={weights})")]
    WeightLengthMismatch { samples: usize, weights: usize },
    #[error("series must not be empty")]
    EmptySeries,
    #[error("z-values must not be empty")]
    EmptyZValues,
    // New: explicit finiteness checks for robustness
    #[error("weight[{index}] is not finite: {value}")]
    NonFiniteWeight { index: usize, value: Scalar },
    #[error("sample[{index}] coefficient is not finite")]
    NonFiniteSampleCoeff { index: usize },
}

#[derive(Debug, Error)]
pub enum MellinError {
    #[error("integration bounds must be positive, finite, and ordered")]
    InvalidRange,
    #[error("log_step must be positive and finite")]
    InvalidLogStep,
    #[error("log_start must be finite")]
    InvalidLogStart,
    #[error("samples must not be empty")]
    EmptySamples,
    #[error("at least two samples required for trapezoidal rule")]
    InsufficientSamples,
    #[error("function evaluated to a non-finite value at x={x}")]
    NonFiniteFunctionValue { x: Scalar },
    #[error("sample {index} produced a non-finite value")]
    NonFiniteSample { index: usize },
    #[error("mellin lattices do not match")]
    LatticeMismatch,
    #[error("hilbert inner product became negative: {value}")]
    NegativeInnerProduct { value: ComplexScalar },
    // New: explicit invalid z
    #[error("z value is not finite: re={re}, im={im}")]
    NonFiniteZ { re: Scalar, im: Scalar },
    #[error(transparent)]
    ZSpace(#[from] ZSpaceError),
    #[cfg(feature = "wgpu")]
    #[error(transparent)]
    Gpu(#[from] crate::mellin_wgpu::MellinGpuError),
}

pub type MellinResult<T> = Result<T, MellinError>;
