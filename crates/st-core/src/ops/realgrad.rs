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

use crate::util::math::{ramanujan_pi, LeechProjector};

const DEFAULT_RANK: usize = 24;
const DEFAULT_WEIGHT: f64 = 1.0;
const DEFAULT_THRESHOLD: f32 = 0.005;

/// Result emitted by [`project_realgrad`].
#[derive(Debug, Clone, PartialEq)]
pub struct RealGradProjection {
    /// Scaled gradient that lives in the concrete Z-space tape.
    pub realgrad: Vec<f32>,
    /// Leech-projected Z-space magnitudes.
    pub z_space: Vec<f32>,
    /// Residual magnitudes that were too large to keep and were therefore sent
    /// to the "monad biome" side channel for dedicated treatment.
    pub monad_biome: Vec<f32>,
    /// Lebesgue-style integral (L¹ norm) used to stabilise the projection.
    pub lebesgue_measure: f32,
    /// Ramanujan π estimate used for the projection.
    pub ramanujan_pi: f32,
}

/// Cached projector state that can be reused across multiple RealGrad invocations.
#[derive(Debug, Clone)]
pub struct RealGradKernel {
    config: RealGradConfig,
    projector: LeechProjector,
    ramanujan_pi: f32,
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
        self.monad_biome.iter().copied().sum()
    }

    /// Returns the accumulated energy of the Z-space field.
    pub fn z_energy(&self) -> f32 {
        self.z_space.iter().map(|value| value.abs()).sum()
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
        }
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
    pub fn project(&self, values: &[f32]) -> RealGradProjection {
        if values.is_empty() {
            return RealGradProjection {
                realgrad: Vec::new(),
                z_space: Vec::new(),
                monad_biome: Vec::new(),
                lebesgue_measure: 0.0,
                ramanujan_pi: self.ramanujan_pi,
            };
        }

        let len = values.len();
        let delta_x = 1.0f32 / len as f32;
        let mut spectrum: Vec<(f32, f32)> = vec![(0.0, 0.0); len];

        for (xi_idx, slot) in spectrum.iter_mut().enumerate() {
            let mut acc_re = 0.0f32;
            let mut acc_im = 0.0f32;
            let xi = xi_idx as f32;
            for (n_idx, &value) in values.iter().enumerate() {
                let x = n_idx as f32 * delta_x;
                let angle = -2.0 * self.ramanujan_pi * x * xi;
                let (sin, cos) = angle.sin_cos();
                acc_re += value * cos * delta_x;
                acc_im -= value * sin * delta_x;
            }
            *slot = (acc_re, acc_im);
        }

        let mut z_space = Vec::with_capacity(len);
        let mut monad_biome = Vec::new();
        for &(re, im) in &spectrum {
            let magnitude = (re * re + im * im).sqrt();
            if magnitude > self.residual_threshold() {
                monad_biome.push(magnitude);
            }
            let enriched = self.projector.enrich(magnitude as f64) as f32;
            z_space.push(enriched);
        }

        let lebesgue_measure = values.iter().map(|v| v.abs()).sum::<f32>();
        let normaliser = if lebesgue_measure > 0.0 {
            self.ramanujan_pi / lebesgue_measure
        } else {
            0.0
        };

        let realgrad = values
            .iter()
            .zip(z_space.iter())
            .map(|(&value, &z)| value * normaliser + z)
            .collect();

        RealGradProjection {
            realgrad,
            z_space,
            monad_biome,
            lebesgue_measure,
            ramanujan_pi: self.ramanujan_pi,
        }
    }

    /// Projects a Schwartz sequence using the cached projector and returns the tempered outcome.
    pub fn project_tempered(
        &self,
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
        let mut overflow_residuals: Vec<f32> = Vec::new();
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
                    overflow_residuals.push(diff - tolerance);
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
            projection.monad_biome.push(convergence_error);
        }

        TemperedRealGradProjection {
            projection,
            dominated,
            convergence_error,
            iterations,
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
        project_realgrad, project_tempered_realgrad, RealGradConfig, RealGradKernel,
        SchwartzSequence, DEFAULT_THRESHOLD,
    };
    use crate::util::math::LEECH_PACKING_DENSITY;

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
        let kernel = RealGradKernel::new(config);
        assert!(kernel.ramanujan_pi() > 3.14);
        let empty = kernel.project(&[]);
        assert_eq!(empty.ramanujan_pi, kernel.ramanujan_pi());
        let data = [0.5f32, -0.25];
        let projection = kernel.project(&data);
        assert_eq!(projection.ramanujan_pi, kernel.ramanujan_pi());
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
            projection.monad_biome.iter().sum::<f32>()
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
            .all(|value| value.is_finite()));
        assert!(
            tempered.converged(4.0),
            "err {}",
            tempered.convergence_error
        );
    }
}
