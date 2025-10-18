// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Real gradient projection helpers that blend the Reach lattice sketch with the
//! existing Leech lattice tooling.
//!
//! The helper intentionally stays dependency free so we can drive the julia-like
//! transliteration from notebooks, but now reuses the shared Ramanujan π series
//! and Leech density projector that already live inside the repository.

use core::f32::consts::PI;
use core::fmt;

use crate::theory::zpulse::{ZPulse, ZSource};
use crate::util::math::{ramanujan_pi, LeechProjector};

const DEFAULT_RANK: usize = 24;
const DEFAULT_WEIGHT: f64 = 1.0;
const DEFAULT_THRESHOLD: f32 = 0.005;

/// Discrete Fourier transform backend used by [`RealGradKernel`].
pub trait SpectralEngine {
    /// Computes the complex DFT of the provided real input.
    ///
    /// The output slice must have the same length as `input`. Each entry is
    /// expressed as `(re, im)`.
    fn dft(&mut self, input: &[f32], out: &mut [(f32, f32)]);

    /// Returns the length supported by the cached plan.
    fn len(&self) -> usize;
}

/// Naive O(N²) reference implementation of [`SpectralEngine`].
#[derive(Debug, Default, Clone)]
pub struct CpuNaive {
    len: usize,
    twiddle_cos: Vec<f32>,
    twiddle_sin: Vec<f32>,
}

impl CpuNaive {
    fn ensure_capacity(&mut self, len: usize) {
        if self.len == len {
            return;
        }
        self.len = len;
        self.twiddle_cos.resize(len, 0.0);
        self.twiddle_sin.resize(len, 0.0);
        if len == 0 {
            return;
        }
        let base = -2.0f64 * core::f64::consts::PI / len as f64;
        for idx in 0..len {
            let angle = base * idx as f64;
            let (sin, cos) = angle.sin_cos();
            self.twiddle_cos[idx] = cos as f32;
            self.twiddle_sin[idx] = sin as f32;
        }
    }
}

impl SpectralEngine for CpuNaive {
    fn dft(&mut self, input: &[f32], out: &mut [(f32, f32)]) {
        let len = input.len();
        assert_eq!(out.len(), len, "output length must match input length");
        self.ensure_capacity(len);
        if len == 0 {
            return;
        }

        let delta_x = 1.0f64 / len as f64;
        for (xi_idx, slot) in out.iter_mut().enumerate() {
            let mut acc_re = 0.0f64;
            let mut acc_im = 0.0f64;
            let step_idx = xi_idx % self.len;
            let cos_step = if step_idx == 0 {
                1.0f64
            } else {
                self.twiddle_cos[step_idx] as f64
            };
            let sin_step = if step_idx == 0 {
                0.0f64
            } else {
                self.twiddle_sin[step_idx] as f64
            };
            let mut cos = 1.0f64;
            let mut sin = 0.0f64;
            for &value in input {
                let value = value as f64;
                acc_re = f64::mul_add(value * cos, delta_x, acc_re);
                acc_im = f64::mul_add(-value * sin, delta_x, acc_im);
                if step_idx != 0 {
                    let next_cos = cos * cos_step - sin * sin_step;
                    let next_sin = cos * sin_step + sin * cos_step;
                    cos = next_cos;
                    sin = next_sin;
                }
            }
            *slot = (acc_re as f32, acc_im as f32);
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Result emitted by [`project_realgrad`].
#[derive(Debug, Clone, PartialEq)]
pub struct RealGradProjection {
    /// Scaled gradient that lives in the concrete Z-space tape.
    pub realgrad: Vec<f32>,
    /// Leech-projected Z-space magnitudes.
    pub z_space: Vec<f32>,
    /// Residual magnitudes that were too large to keep and were therefore sent
    /// to the "monad biome" side channel for dedicated treatment.
    pub monad_biome: Vec<Residual>,
    /// Lebesgue-style integral (L¹ norm) used to stabilise the projection.
    pub lebesgue_measure: f32,
    /// Ramanujan π estimate used for the projection.
    pub ramanujan_pi: f32,
}

impl Default for RealGradProjection {
    fn default() -> Self {
        Self {
            realgrad: Vec::new(),
            z_space: Vec::new(),
            monad_biome: Vec::new(),
            lebesgue_measure: 0.0,
            ramanujan_pi: 0.0,
        }
    }
}

/// Reason why a RealGrad residual was routed to the monad biome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResidualReason {
    /// Magnitude exceeded the configured residual threshold.
    OverThreshold,
    /// Tempered sequence failed to converge under the configured tolerance.
    NonConvergent,
}

/// Metadata attached to a residual bin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Residual {
    /// Frequency bin responsible for the residual.
    pub bin: usize,
    /// Magnitude that overflowed.
    pub magnitude: f32,
    /// Reason why the magnitude was routed to the monad biome.
    pub reason: ResidualReason,
}

/// Cached projector state that can be reused across multiple RealGrad invocations.
pub struct RealGradKernel {
    config: RealGradConfig,
    projector: LeechProjector,
    ramanujan_pi: f32,
    engine: Box<dyn SpectralEngine + Send>,
    spectrum: Vec<(f32, f32)>,
    z_buf: Vec<f32>,
    residuals: Vec<Residual>,
}

impl fmt::Debug for RealGradKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealGradKernel")
            .field("config", &self.config)
            .field("projector", &self.projector)
            .field("ramanujan_pi", &self.ramanujan_pi)
            .field("engine_len", &self.engine.len())
            .field("spectrum_len", &self.spectrum.len())
            .finish()
    }
}

/// Scratch buffers that can be reused across RealGrad projections.
#[derive(Debug, Clone, Default)]
pub struct RealGradProjectionScratch {
    projection: RealGradProjection,
}

impl RealGradProjectionScratch {
    /// Creates an empty scratch projection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns an immutable view over the projection stored in the scratch buffers.
    pub fn as_projection(&self) -> &RealGradProjection {
        &self.projection
    }

    /// Returns a mutable view over the projection stored in the scratch buffers.
    pub fn as_projection_mut(&mut self) -> &mut RealGradProjection {
        &mut self.projection
    }

    /// Consumes the scratch buffers, yielding the owned projection.
    pub fn into_projection(self) -> RealGradProjection {
        self.projection
    }

    /// Clears the projection contents while preserving the allocated capacity.
    pub fn clear(&mut self) {
        self.projection.clear();
    }
}

/// Projection outcome for a tempered (distribution-like) RealGrad sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct TemperedRealGradProjection {
    /// RealGrad projection computed from the approximating Schwartz sequence.
    pub projection: RealGradProjection,
    /// Whether every term of the sequence stayed within the provided dominator.
    pub dominated: bool,
    /// Final L¹-style error between the last two projections in the sequence.
    pub convergence_error: f32,
    /// Number of sequence elements that were processed.
    pub iterations: usize,
}

impl TemperedRealGradProjection {
    /// Returns `true` when the tempered projection converged under the provided tolerance.
    pub fn converged(&self, tolerance: f32) -> bool {
        self.convergence_error <= tolerance.max(0.0)
    }
}

impl RealGradProjection {
    /// Returns `true` when the projection yielded any non-zero residuals.
    pub fn has_residuals(&self) -> bool {
        !self.monad_biome.is_empty()
    }

    /// Returns the total magnitude routed to the monad biome.
    pub fn residual_energy(&self) -> f32 {
        self.monad_biome
            .iter()
            .map(|residual| residual.magnitude)
            .sum()
    }

    /// Returns the accumulated energy of the Z-space field.
    pub fn z_energy(&self) -> f32 {
        self.z_space.iter().map(|value| value.abs()).sum()
    }

    /// Clears the projection buffers while keeping the allocated capacity.
    pub fn clear(&mut self) {
        self.realgrad.clear();
        self.z_space.clear();
        self.monad_biome.clear();
        self.lebesgue_measure = 0.0;
        self.ramanujan_pi = 0.0;
    }
}

impl RealGradKernel {
    /// Creates a reusable RealGrad kernel with a sanitised configuration and cached state.
    pub fn new(config: RealGradConfig) -> Self {
        let config = config.sanitised();
        let projector = LeechProjector::new(config.z_rank, config.z_weight);
        let ramanujan_pi = ramanujan_pi(config.ramanujan_iterations) as f32;
        Self {
            config,
            projector,
            ramanujan_pi,
            engine: Box::new(CpuNaive::default()),
            spectrum: Vec::new(),
            z_buf: Vec::new(),
            residuals: Vec::new(),
        }
    }

    /// Replaces the spectral engine used by the kernel.
    pub fn with_engine(mut self, engine: Box<dyn SpectralEngine + Send>) -> Self {
        self.engine = engine;
        self
    }

    /// Returns the effective configuration used by the kernel.
    pub fn config(&self) -> RealGradConfig {
        self.config
    }

    /// Returns the cached Ramanujan π estimate used by the kernel.
    pub fn ramanujan_pi(&self) -> f32 {
        self.ramanujan_pi
    }

    fn residual_threshold(&self) -> f32 {
        self.config.residual_threshold
    }

    /// Projects the provided samples into the RealGrad field using the cached projector.
    pub fn project(&mut self, values: &[f32]) -> RealGradProjection {
        let mut scratch = RealGradProjectionScratch::default();
        self.project_into(values, &mut scratch);
        scratch.into_projection()
    }

    /// Projects the provided samples into the RealGrad field using the cached projector,
    /// storing the results inside the supplied scratch buffers.
    pub fn project_into(&mut self, values: &[f32], out: &mut RealGradProjectionScratch) {
        out.clear();
        if values.is_empty() {
            out.as_projection_mut().ramanujan_pi = self.ramanujan_pi;
            return;
        }

        let len = values.len();
        self.spectrum.resize(len, (0.0, 0.0));
        self.engine.dft(values, &mut self.spectrum);

        let scale = self.config.spectrum_norm.scale(len);
        if (scale - 1.0).abs() > f64::EPSILON {
            for (re, im) in &mut self.spectrum {
                *re = (*re as f64 * scale) as f32;
                *im = (*im as f64 * scale) as f32;
            }
        }

        self.z_buf.resize(len, 0.0);
        self.residuals.clear();
        let projection = out.as_projection_mut();
        projection.realgrad.resize(len, 0.0);
        projection.z_space.resize(len, 0.0);
        for (idx, &(re, im)) in self.spectrum.iter().enumerate() {
            let magnitude = (re * re + im * im).sqrt();
            if magnitude > self.residual_threshold() {
                self.residuals.push(Residual {
                    bin: idx,
                    magnitude,
                    reason: ResidualReason::OverThreshold,
                });
            }
            let enriched = self.projector.enrich(magnitude as f64) as f32;
            self.z_buf[idx] = enriched;
            projection.z_space[idx] = enriched;
        }

        let lebesgue_measure = values.iter().map(|v| v.abs()).sum::<f32>();
        let normaliser = if lebesgue_measure > 0.0 {
            self.ramanujan_pi / lebesgue_measure
        } else {
            0.0
        };

        for idx in 0..len {
            projection.realgrad[idx] = values[idx] * normaliser + self.z_buf[idx];
        }

        projection.monad_biome.extend_from_slice(&self.residuals);
        projection.lebesgue_measure = lebesgue_measure;
        projection.ramanujan_pi = self.ramanujan_pi;
    }

    /// Projects a Schwartz sequence using the cached projector and returns the tempered outcome.
    pub fn project_tempered(
        &mut self,
        sequence: &SchwartzSequence,
        tolerance: f32,
    ) -> TemperedRealGradProjection {
        let tolerance = tolerance.max(0.0);

        if sequence.is_empty() {
            return TemperedRealGradProjection {
                projection: self.project(&[]),
                dominated: true,
                convergence_error: 0.0,
                iterations: 0,
            };
        }

        if !sequence.lengths_consistent() {
            return TemperedRealGradProjection {
                projection: self.project(&[]),
                dominated: false,
                convergence_error: f32::INFINITY,
                iterations: 0,
            };
        }

        let dominated = sequence.is_dominated();
        let mut previous: Option<RealGradProjection> = None;
        let mut overflow_residuals: Vec<Residual> = Vec::new();
        let mut convergence_error = f32::INFINITY;
        let mut iterations = 0usize;

        for member in sequence.iter() {
            iterations += 1;
            let projection = self.project(member);
            if let Some(prev) = &previous {
                let diff = projection
                    .realgrad
                    .iter()
                    .zip(prev.realgrad.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    + (projection.lebesgue_measure - prev.lebesgue_measure).abs();
                convergence_error = diff;
                if diff > tolerance {
                    overflow_residuals.push(Residual {
                        bin: 0,
                        magnitude: diff - tolerance,
                        reason: ResidualReason::NonConvergent,
                    });
                }
            } else {
                convergence_error = 0.0;
            }
            previous = Some(projection);
        }

        let mut projection = previous.unwrap_or_else(|| self.project(&[]));
        if !overflow_residuals.is_empty() {
            projection.monad_biome.extend(overflow_residuals);
        }
        if !dominated && convergence_error.is_finite() && convergence_error > 0.0 {
            projection.monad_biome.push(Residual {
                bin: 0,
                magnitude: convergence_error,
                reason: ResidualReason::NonConvergent,
            });
        }

        TemperedRealGradProjection {
            projection,
            dominated,
            convergence_error,
            iterations,
        }
    }
}

/// Normalisation applied to the DFT spectrum prior to enrichment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumNorm {
    /// Applies a `1/√N` scaling to preserve Parseval symmetry.
    Unitary,
    /// Leaves the forward transform in its natural `1/N` scaling.
    Forward,
    /// Applies a `1/N²` scaling matching the backward transform convention.
    Backward,
}

impl SpectrumNorm {
    fn scale(self, len: usize) -> f64 {
        let n = len.max(1) as f64;
        match self {
            SpectrumNorm::Unitary => n.sqrt(),
            SpectrumNorm::Forward => n,
            SpectrumNorm::Backward => 1.0,
        }
    }
}

/// Configuration for [`project_realgrad`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealGradConfig {
    /// Number of iterations used for the Ramanujan π approximation.
    pub ramanujan_iterations: usize,
    /// Target Z-space rank for the Leech projector.
    pub z_rank: usize,
    /// Weight applied when enriching magnitudes with the Leech projector.
    pub z_weight: f64,
    /// Threshold beyond which magnitudes are routed to the monad biome.
    pub residual_threshold: f32,
    /// Normalisation applied to the DFT spectrum.
    pub spectrum_norm: SpectrumNorm,
}

/// Discrete representation of a Schwartz sequence used to approximate a tempered function.
#[derive(Debug, Clone, PartialEq)]
pub struct SchwartzSequence {
    members: Vec<Vec<f32>>,
    dominator: Vec<f32>,
}

impl SchwartzSequence {
    /// Creates a new Schwartz sequence from the provided members and dominator.
    pub fn new(members: Vec<Vec<f32>>, dominator: Vec<f32>) -> Self {
        Self { members, dominator }
    }

    /// Returns `true` when the sequence has no members.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Returns the number of functions contained in the sequence.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns the length of the sampled lattice, if the sequence is non-empty.
    pub fn sample_len(&self) -> Option<usize> {
        self.members.first().map(Vec::len)
    }

    fn lengths_consistent(&self) -> bool {
        if let Some(expected) = self.sample_len() {
            self.members.iter().all(|member| member.len() == expected)
                && self.dominator.len() == expected
        } else {
            self.dominator.is_empty()
        }
    }

    /// Returns an iterator over the members as immutable slices.
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        self.members.iter().map(Vec::as_slice)
    }

    /// Returns `true` when every term is bounded by the dominator (dominated convergence).
    pub fn is_dominated(&self) -> bool {
        if !self.lengths_consistent() {
            return false;
        }
        if self.dominator.iter().any(|value| !value.is_finite()) {
            return false;
        }
        self.iter().all(|member| {
            member
                .iter()
                .zip(self.dominator.iter())
                .all(|(value, bound)| value.abs() <= bound.abs() + f32::EPSILON)
        })
    }
}

impl Default for RealGradConfig {
    fn default() -> Self {
        Self {
            ramanujan_iterations: 4,
            z_rank: DEFAULT_RANK,
            z_weight: DEFAULT_WEIGHT,
            residual_threshold: DEFAULT_THRESHOLD,
            spectrum_norm: SpectrumNorm::Backward,
        }
    }
}

impl RealGradConfig {
    /// Ensures the configuration is numerically stable.
    fn sanitised(self) -> Self {
        Self {
            ramanujan_iterations: self.ramanujan_iterations.max(1),
            z_rank: self.z_rank.max(1),
            z_weight: self.z_weight.max(0.0),
            residual_threshold: self.residual_threshold.max(0.0),
            spectrum_norm: self.spectrum_norm,
        }
    }

    /// Calibrates the configuration against a previous projection and returns
    /// the tuned configuration alongside the tuning diagnostics.
    pub fn calibrate(self, projection: &RealGradProjection) -> (Self, RealGradTuning) {
        let base = self.sanitised();
        let monad_energy = projection.residual_energy();
        let z_energy = projection.z_energy();
        let lebesgue = projection.lebesgue_measure.max(f32::EPSILON);
        let total_energy = (monad_energy + z_energy).max(f32::EPSILON);
        let residual_ratio = monad_energy / total_energy;
        let lebesgue_ratio = monad_energy / lebesgue;
        let pi_multiplier = if PI > 0.0 {
            (projection.ramanujan_pi / PI).max(0.0)
        } else {
            1.0
        };
        let target_ratio = 0.1f32;
        let raw_adjustment = 1.0 + (residual_ratio - target_ratio) * pi_multiplier + lebesgue_ratio;
        let adjustment_factor = raw_adjustment.clamp(0.25, 4.0);
        let suggested_threshold = (base.residual_threshold * adjustment_factor).max(0.0);

        let tuned_config = RealGradConfig {
            residual_threshold: suggested_threshold,
            ..base
        };

        let diagnostics = RealGradTuning {
            monad_energy,
            z_energy,
            residual_ratio,
            lebesgue_ratio,
            pi_multiplier,
            adjustment_factor,
            suggested_threshold,
        };

        (tuned_config, diagnostics)
    }
}

/// Diagnostics emitted by [`RealGradConfig::calibrate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealGradTuning {
    /// Total magnitude rerouted to the monad biome in the previous projection.
    pub monad_energy: f32,
    /// Total magnitude retained in the Z-space field.
    pub z_energy: f32,
    /// Share of the overall energy captured by the monad biome.
    pub residual_ratio: f32,
    /// Ratio between the monad biome energy and the Lebesgue integral.
    pub lebesgue_ratio: f32,
    /// Multiplier derived from the Ramanujan π estimate relative to π.
    pub pi_multiplier: f32,
    /// Factor applied to the previous residual threshold.
    pub adjustment_factor: f32,
    /// Suggested residual threshold for the next projection.
    pub suggested_threshold: f32,
}

/// Projects RealGrad output into a canonical [`ZPulse`].
#[derive(Clone, Debug)]
pub struct RealGradZProjector {
    projector: LeechProjector,
    bias_gain: f32,
    min_energy: f32,
}

impl RealGradZProjector {
    /// Creates a new projector with the provided enrichment parameters.
    pub fn new(projector: LeechProjector, bias_gain: f32, min_energy: f32) -> Self {
        Self {
            projector,
            bias_gain,
            min_energy: min_energy.max(0.0),
        }
    }

    /// Converts a [`RealGradProjection`] into a Z-space pulse compatible with the conductor.
    pub fn project(&self, projection: &RealGradProjection) -> ZPulse {
        let here = projection.lebesgue_measure.max(0.0);
        let mut above = 0.0f32;
        let mut beneath = 0.0f32;
        for &value in &projection.z_space {
            if value >= 0.0 {
                above += value.abs();
            } else {
                beneath += value.abs();
            }
        }
        let drift = above - beneath;
        let z_energy = projection.z_energy();
        let magnitude = drift.abs() as f64;
        let enriched = self.projector.enrich(magnitude) as f32;
        let z_bias = if z_energy >= self.min_energy {
            enriched.copysign(drift) * self.bias_gain
        } else {
            0.0
        };

        ZPulse {
            source: ZSource::Other("RealGrad"),
            ts: 0,
            band_energy: (above, here, beneath),
            drift,
            z_bias,
            support: z_energy,
            quality: 0.0,
            stderr: 0.0,
            latency_ms: 0.0,
        }
    }
}

/// Adaptive residual threshold tuner built on an exponential moving average.
#[derive(Clone, Debug)]
pub struct RealGradAutoTuner {
    ema_ratio: f32,
    ema_alpha: f32,
}

impl RealGradAutoTuner {
    /// Creates a new auto tuner using the provided EMA coefficient.
    pub fn new(ema_alpha: f32) -> Self {
        Self {
            ema_ratio: 0.0,
            ema_alpha: ema_alpha.clamp(0.0, 1.0),
        }
    }

    /// Returns the current EMA coefficient.
    pub fn ema_alpha(&self) -> f32 {
        self.ema_alpha
    }

    /// Returns the tracked residual ratio.
    pub fn residual_ratio(&self) -> f32 {
        self.ema_ratio
    }

    /// Updates the residual threshold based on the projection statistics.
    pub fn update(&mut self, projection: &RealGradProjection, config: &mut RealGradConfig) {
        let monad = projection.residual_energy();
        let z_energy = projection.z_energy();
        let total = (monad + z_energy).max(1.0e-9);
        let ratio = monad / total;
        let alpha = self.ema_alpha.clamp(0.0, 1.0);
        let complement = 1.0 - alpha;
        self.ema_ratio = self.ema_ratio * complement + ratio * alpha;

        let target = 0.1f32;
        let adjustment = (1.0 + (self.ema_ratio - target)).clamp(0.5, 1.5);
        config.residual_threshold = (config.residual_threshold * adjustment).max(0.0);
    }
}

/// Projects a Euclidean gradient into the Reach lattice "Realgrad" tape.
///
/// The routine performs a discrete Fourier transform so that harmonics can be
/// scheduled independently, scales the resulting real-valued field using a
/// Lebesgue-style normaliser, enriches the magnitudes through the Leech lattice,
/// and finally hands excessively energetic modes to the "monad biome" output
/// vector.
///
/// The discrete transform directly mirrors the classical continuous model
/// `\hat f(\xi) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i x \xi} \, dx` by
/// sampling the signal on an even lattice, weighting each contribution with the
/// Ramanujan π approximation, and accumulating the real and imaginary channels
/// accordingly.
pub fn project_realgrad(values: &[f32], config: RealGradConfig) -> RealGradProjection {
    RealGradKernel::new(config).project(values)
}

/// Projects a Schwartz sequence that approximates a tempered distribution.
///
/// Each member of the sequence is individually projected using [`project_realgrad`].
/// Differences between successive projections act as the discrete analogue of the
/// dominated convergence theorem and are used to route non-convergent magnitude to
/// the monad biome.
pub fn project_tempered_realgrad(
    sequence: &SchwartzSequence,
    config: RealGradConfig,
    tolerance: f32,
) -> TemperedRealGradProjection {
    RealGradKernel::new(config).project_tempered(sequence, tolerance)
}

#[cfg(test)]
mod tests {
    use super::{
        project_realgrad, project_tempered_realgrad, RealGradAutoTuner, RealGradConfig,
        RealGradKernel, RealGradProjectionScratch, RealGradZProjector, SchwartzSequence,
        SpectrumNorm, DEFAULT_THRESHOLD,
    };
    use crate::theory::zpulse::ZSource;
    use crate::util::math::{LeechProjector, LEECH_PACKING_DENSITY};

    #[test]
    fn projection_handles_empty_input() {
        let projection = project_realgrad(&[], RealGradConfig::default());
        assert!(projection.realgrad.is_empty());
        assert!(projection.z_space.is_empty());
        assert!(projection.monad_biome.is_empty());
        assert_eq!(projection.lebesgue_measure, 0.0);
        assert!(projection.ramanujan_pi > 3.14);
    }

    #[test]
    fn kernel_caches_ramanujan_series() {
        let config = RealGradConfig::default();
        let mut kernel = RealGradKernel::new(config);
        assert!(kernel.ramanujan_pi() > 3.14);
        let empty = kernel.project(&[]);
        assert_eq!(empty.ramanujan_pi, kernel.ramanujan_pi());
        let data = [0.5f32, -0.25];
        let projection = kernel.project(&data);
        assert_eq!(projection.ramanujan_pi, kernel.ramanujan_pi());
    }

    #[test]
    fn project_into_reuses_scratch_buffers() {
        let mut kernel = RealGradKernel::new(RealGradConfig::default());
        let mut scratch = RealGradProjectionScratch::new();
        kernel.project_into(&[0.25f32, -0.5], &mut scratch);
        let initial_capacity = scratch.as_projection().realgrad.capacity();
        kernel.project_into(&[0.5, -0.75], &mut scratch);
        assert_eq!(
            initial_capacity,
            scratch.as_projection().realgrad.capacity()
        );
        assert_eq!(scratch.as_projection().ramanujan_pi, kernel.ramanujan_pi());
    }

    #[test]
    fn projection_respects_l1_measure() {
        let data = [1.0f32, -2.0, 3.0, -4.0];
        let projection = project_realgrad(&data, RealGradConfig::default());
        assert_eq!(projection.lebesgue_measure, 10.0);
        assert_eq!(projection.realgrad.len(), data.len());
        assert_eq!(projection.z_space.len(), data.len());
        assert!(projection.monad_biome.len() <= data.len());
        assert!(projection.realgrad.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn projection_routes_large_modes_to_monad_biome() {
        let mut data = vec![0.0f32; 8];
        data[0] = DEFAULT_THRESHOLD * 16.0;
        let projection = project_realgrad(&data, RealGradConfig::default());
        assert!(projection.has_residuals());
        assert!(!projection.monad_biome.is_empty());
    }

    #[test]
    fn projection_reports_energy_levels() {
        let data = [0.25f32; 4];
        let projection = project_realgrad(&data, RealGradConfig::default());
        assert!(projection.z_energy() > 0.0);
        assert_eq!(
            projection.residual_energy(),
            projection
                .monad_biome
                .iter()
                .map(|residual| residual.magnitude)
                .sum::<f32>()
        );
    }

    #[test]
    fn projection_respects_classical_constant_transform() {
        let config = RealGradConfig {
            residual_threshold: 2.0,
            ..RealGradConfig::default()
        };
        let data = [1.0f32; 8];
        let projection = project_realgrad(&data, config);
        assert!(projection.monad_biome.is_empty());
        let expected = (config.z_weight as f32)
            * (LEECH_PACKING_DENSITY as f32)
            * (config.z_rank as f32).sqrt();
        let first_mode = projection.z_space[0];
        assert!((first_mode - expected).abs() < 1.0e-3);
        for mode in projection.z_space.iter().skip(1) {
            assert!(mode.abs() < expected);
        }
    }

    #[test]
    fn calibration_adjusts_threshold_based_on_residual_ratio() {
        let data = vec![DEFAULT_THRESHOLD * 1024.0; 16];
        let config = RealGradConfig::default();
        let projection = project_realgrad(&data, config);
        let (tuned, tuning) = config.calibrate(&projection);
        assert!(tuning.monad_energy > 0.0);
        assert!(tuning.lebesgue_ratio > 0.0);
        assert!(tuning.adjustment_factor > 1.0);
        assert!(tuned.residual_threshold > config.residual_threshold);

        let small_data = vec![DEFAULT_THRESHOLD * 0.1; 16];
        let small_projection = project_realgrad(&small_data, config);
        let (tuned_small, tuning_small) = config.calibrate(&small_projection);
        assert!(tuning_small.adjustment_factor < 1.0);
        assert!(tuned_small.residual_threshold < config.residual_threshold);
    }

    #[test]
    fn spectrum_norm_variants_affect_scaling() {
        let mut config = RealGradConfig::default();
        config.spectrum_norm = SpectrumNorm::Backward;
        let mut backward = RealGradKernel::new(config);
        let backward_proj = backward.project(&[1.0f32, 0.0, 0.0, 0.0]);

        config.spectrum_norm = SpectrumNorm::Forward;
        let mut forward = RealGradKernel::new(config);
        let forward_proj = forward.project(&[1.0f32, 0.0, 0.0, 0.0]);

        assert_ne!(backward_proj.z_space[0], forward_proj.z_space[0]);
    }

    #[test]
    fn schwartz_sequence_detects_dominated_terms() {
        let dominator = vec![1.0f32; 8];
        let mut members = Vec::new();
        for scale in [1.0f32, 2.0, 4.0] {
            let mut entry = Vec::with_capacity(8);
            for (idx, bound) in dominator.iter().enumerate() {
                let x = idx as f32 - 4.0;
                let value = (-x * x / scale).exp();
                entry.push(value.min(*bound));
            }
            members.push(entry);
        }
        let sequence = SchwartzSequence::new(members, dominator);
        assert!(sequence.is_dominated());
        assert_eq!(sequence.len(), 3);
        assert_eq!(sequence.sample_len(), Some(8));
    }

    #[test]
    fn tempered_projection_routes_non_convergent_mass() {
        let len = 16;
        let constant = 0.75f32;
        let dominator = vec![constant; len];
        let mut members = Vec::new();
        let center = (len as f32 - 1.0) * 0.5;
        for scale in [1.5f32, 3.0, 6.0, 12.0] {
            let mut entry = Vec::with_capacity(len);
            for idx in 0..len {
                let x = (idx as f32 - center) / scale;
                let value = constant * (-x * x).exp();
                entry.push(value);
            }
            members.push(entry);
        }
        let sequence = SchwartzSequence::new(members, dominator);
        let config = RealGradConfig {
            residual_threshold: 5.0,
            ..RealGradConfig::default()
        };
        let tempered = project_tempered_realgrad(&sequence, config, 1.0e-3);
        assert!(tempered.dominated);
        assert_eq!(tempered.iterations, 4);
        assert!(tempered.convergence_error >= 0.0);
        assert!(tempered.projection.lebesgue_measure > 0.0);
        assert!(tempered
            .projection
            .monad_biome
            .iter()
            .all(|value| value.magnitude.is_finite()));
        assert!(
            tempered.converged(4.0),
            "err {}",
            tempered.convergence_error
        );
    }

    #[test]
    fn z_projector_emits_realgrad_pulse() {
        let config = RealGradConfig::default();
        let mut kernel = RealGradKernel::new(config);
        let projection = kernel.project(&[0.5f32, -0.25, 0.75, -0.5]);
        let projector = RealGradZProjector::new(
            LeechProjector::new(config.z_rank, config.z_weight),
            1.0,
            0.0,
        );
        let pulse = projector.project(&projection);
        assert!(matches!(pulse.source, ZSource::Other("RealGrad")));
        assert!(pulse.support >= 0.0);
        assert!(pulse.band_energy.0 >= 0.0);
    }

    #[test]
    fn auto_tuner_tracks_residual_ratio() {
        let mut tuner = RealGradAutoTuner::new(0.2);
        let mut config = RealGradConfig::default();
        let projection = project_realgrad(&vec![DEFAULT_THRESHOLD * 64.0; 8], config);
        let previous = config.residual_threshold;
        tuner.update(&projection, &mut config);
        assert!(tuner.residual_ratio() > 0.0);
        assert!(config.residual_threshold > previous);
    }
}
