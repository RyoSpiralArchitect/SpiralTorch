// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

/// Memoises successive approximations of the Ramanujan π series so repeated
/// callers only pay for the additional iterations they request.
struct RamanujanCache {
    approximations: Vec<f64>,
    sum: f64,
    factor: f64,
    base: f64,
    prefactor: f64,
}

impl RamanujanCache {
    fn new() -> Self {
        Self {
            approximations: Vec::new(),
            sum: 0.0,
            factor: 1.0,
            base: 396_f64.powi(4),
            prefactor: (2.0 * 2.0_f64.sqrt()) / 9801.0,
        }
    }

    fn ensure_iterations(&mut self, iterations: usize) {
        let target = iterations.max(1);
        while self.approximations.len() < target {
            let k = self.approximations.len() as f64;
            self.sum += self.factor * (1103.0 + 26390.0 * k);
            let value = (self.prefactor * self.sum).recip();
            self.approximations.push(value);

            let k1 = self.approximations.len() as f64;
            let numerator = (4.0 * k1 - 3.0) * (4.0 * k1 - 2.0) * (4.0 * k1 - 1.0) * (4.0 * k1);
            let denominator = k1.powi(4) * self.base;
            self.factor *= numerator / denominator;
        }
    }

    fn value_at(&mut self, iterations: usize) -> f64 {
        self.ensure_iterations(iterations);
        self.approximations[iterations.max(1) - 1]
    }
}

impl Default for RamanujanCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Packing density of the Leech lattice (Λ₂₄) used as the baseline for
/// projecting geodesic measurements into a density correction.
pub const LEECH_PACKING_DENSITY: f64 = 0.001_929_574_309_403_922_5;

/// Golden ratio φ used when blending softmax and hardmax consensus weights.
pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_894_8;
/// Conjugate of φ (1/φ) that balances the softmax contribution in consensus blending.
pub const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_8;
/// Complement of the conjugate that controls the hardmax contribution (1 - 1/φ).
pub const GOLDEN_RATIO_BIAS: f64 = 0.381_966_011_250_105_1;

static RAMANUJAN_CACHE: OnceLock<Mutex<RamanujanCache>> = OnceLock::new();

/// Returns the Ramanujan π approximation using the classical rapidly
/// converging series. Results are memoised per iteration count so repeated
/// callers across the ecosystem share the cached values.
pub fn ramanujan_pi(iterations: usize) -> f64 {
    let iterations = iterations.max(1);
    let cache = RAMANUJAN_CACHE.get_or_init(|| Mutex::new(RamanujanCache::new()));
    cache.lock().unwrap().value_at(iterations)
}

/// Computes the Ramanujan π approximation while adaptively increasing the
/// iteration count until two successive approximations differ by at most the
/// provided tolerance. The iteration count is capped by `max_iterations` to
/// avoid unbounded work in pathological cases. The resulting approximation and
/// the iteration count that produced it are returned as a tuple.
pub fn ramanujan_pi_with_tolerance(tolerance: f64, max_iterations: usize) -> (f64, usize) {
    let tolerance = tolerance.max(f64::EPSILON);
    let max_iterations = max_iterations.max(1);
    let cache = RAMANUJAN_CACHE.get_or_init(|| Mutex::new(RamanujanCache::new()));
    let mut cache = cache.lock().unwrap();
    let mut previous = cache.value_at(1);
    for iterations in 2..=max_iterations {
        let current = cache.value_at(iterations);
        if (current - previous).abs() <= tolerance || (current - PI).abs() <= tolerance {
            return (current, iterations);
        }
        previous = current;
    }
    (previous, max_iterations)
}

/// Lightweight projector that turns geodesic magnitudes into Leech lattice
/// density corrections. The expensive square root of the target rank is cached
/// within the struct so scaling becomes a single fused multiply operation.
#[derive(Clone, Copy, Debug)]
pub struct LeechProjector {
    sqrt_rank: f64,
    weight: f64,
    ramanujan_iterations: usize,
    ramanujan_pi: f64,
    ramanujan_normalizer: f64,
}

impl LeechProjector {
    /// Creates a projector for the provided Z-space rank and weighting factor.
    pub fn new(rank: usize, weight: f64) -> Self {
        Self::with_ramanujan_iterations(rank, weight, 0)
    }

    /// Creates a projector that incorporates the provided Ramanujan π
    /// approximation order as a normalising factor. An iteration count of zero
    /// disables the Ramanujan weighting and behaves identically to [`Self::new`].
    pub fn with_ramanujan_iterations(rank: usize, weight: f64, iterations: usize) -> Self {
        let sqrt_rank = (rank.max(1) as f64).sqrt();
        let weight = weight.max(0.0);
        let iterations = iterations;
        let (ramanujan_pi, ramanujan_normalizer) = if iterations == 0 {
            (PI, 1.0)
        } else {
            let approximation = ramanujan_pi(iterations).max(f64::EPSILON);
            let ratio = PI / approximation;
            (approximation, ratio)
        };
        Self {
            sqrt_rank,
            weight,
            ramanujan_iterations: iterations,
            ramanujan_pi,
            ramanujan_normalizer,
        }
    }

    /// Returns the number of Ramanujan iterations baked into this projector.
    pub fn ramanujan_iterations(&self) -> usize {
        self.ramanujan_iterations
    }

    /// Returns the Ramanujan π approximation associated with this projector.
    pub fn ramanujan_pi(&self) -> f64 {
        self.ramanujan_pi
    }

    /// Returns the multiplicative normaliser derived from the Ramanujan π
    /// approximation (π / approximation). When Ramanujan weighting is disabled
    /// this value is `1.0`.
    pub fn ramanujan_ratio(&self) -> f64 {
        self.ramanujan_normalizer
    }

    /// Returns the absolute deviation of the Ramanujan approximation from π.
    pub fn ramanujan_delta(&self) -> f64 {
        (self.ramanujan_pi - PI).abs()
    }

    /// Projects the provided geodesic magnitude into a density correction using
    /// the Leech lattice baseline blended with the Ramanujan normaliser. When
    /// the projector weight or geodesic magnitude are effectively zero the
    /// result is short-circuited to avoid unnecessary floating point work.
    pub fn enrich(&self, geodesic: f64) -> f64 {
        if self.weight <= f64::EPSILON || geodesic <= f64::EPSILON {
            0.0
        } else {
            self.weight
                * LEECH_PACKING_DENSITY
                * geodesic
                * self.sqrt_rank
                * self.ramanujan_normalizer
        }
    }
}

/// Aggregate telemetry captured when fusing softmax and hardmax outputs into a
/// spiral consensus weighting. The fields are intentionally exposed so callers
/// across the stack (Rust, Python, telemetry exporters) can stitch the values
/// into their preferred diagnostics pipelines.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpiralConsensusStats {
    /// The golden ratio φ used for the fusion.
    pub phi: f64,
    /// The conjugate of φ (1/φ) that scales the softmax contribution.
    pub phi_conjugate: f64,
    /// The complementary bias (1 - 1/φ) applied to the hardmax mask.
    pub phi_bias: f64,
    /// Ratio between π and the Ramanujan approximation baked into the projector.
    pub ramanujan_ratio: f64,
    /// Absolute deviation of the Ramanujan approximation from π.
    pub ramanujan_delta: f64,
    /// Average enrichment factor injected by the Leech projector across rows.
    pub average_enrichment: f64,
    /// Mean Shannon entropy of the softmax distributions prior to fusion.
    pub mean_entropy: f64,
    /// Mean hardmax mass (number of winning entries) observed per row.
    pub mean_hardmass: f64,
    /// Average per-row coherence between entropy, hardmass, and enrichment.
    pub spiral_coherence: f64,
}

/// Computes a spiral consensus weighting that blends softmax probabilities,
/// hardmax masks, Ramanujan π normalisation, and Leech lattice enrichment.
///
/// The returned vector matches the input dimensionality and can be surfaced
/// directly as a tensor while the accompanying [`SpiralConsensusStats`]
/// captures aggregated telemetry useful for debugging or adaptive scheduling.
pub fn spiral_softmax_hardmax_consensus(
    softmax: &[f32],
    hardmax: &[f32],
    rows: usize,
    cols: usize,
    projector: &LeechProjector,
) -> (Vec<f32>, SpiralConsensusStats) {
    let expected = rows.saturating_mul(cols);
    if expected == 0 || softmax.len() != expected || hardmax.len() != expected {
        return (vec![0.0; expected], SpiralConsensusStats::default());
    }

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

        let geodesic = entropy * projector.ramanujan_ratio() + hardmass * GOLDEN_RATIO;
        let enrichment = projector.enrich(geodesic.abs());
        let scale = (1.0 + enrichment) as f32;
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
        ramanujan_ratio: projector.ramanujan_ratio(),
        ramanujan_delta: projector.ramanujan_delta(),
        average_enrichment: total_enrichment / rows_f64,
        mean_entropy: total_entropy / rows_f64,
        mean_hardmass: total_hardmass / rows_f64,
        spiral_coherence: total_coherence / rows_f64,
    };

    (fused, stats)
}

#[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_abs_diff_eq;

    #[test]
    fn ramanujan_cache_reuses_values() {
        let first = ramanujan_pi(3);
        let second = ramanujan_pi(3);
        assert_abs_diff_eq!(first, second);
    }

    #[test]
    fn ramanujan_pi_with_tolerance_converges() {
        let (value, iterations) = ramanujan_pi_with_tolerance(1e-12, 8);
        assert!(iterations >= 1 && iterations <= 8);
        assert_abs_diff_eq!(value, std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn leech_projector_scales_geodesic() {
        let projector = LeechProjector::new(24, 0.5);
        let enriched = projector.enrich(10.0);
        assert!(enriched > 0.0);
        let zero_weight = LeechProjector::new(24, 0.0);
        assert_eq!(zero_weight.enrich(10.0), 0.0);
    }

    #[test]
    fn leech_projector_ramanujan_weighting_scales() {
        let vanilla = LeechProjector::new(24, 1.0);
        let weighted = LeechProjector::with_ramanujan_iterations(24, 1.0, 4);
        assert!(weighted.ramanujan_ratio() >= 1.0);
        assert!(weighted.enrich(1.0) >= vanilla.enrich(1.0));
        assert!(weighted.ramanujan_delta() >= 0.0);
    }

    #[test]
    fn spiral_softmax_hardmax_consensus_sanitises_non_finite_values() {
        let projector = LeechProjector::new(24, 0.75);
        let softmax = [0.5_f32, f32::NAN, -0.2, 1.2];
        let hardmax = [1.0_f32, f32::NAN, -3.0, 0.0];

        let (fused, stats) = spiral_softmax_hardmax_consensus(&softmax, &hardmax, 2, 2, &projector);

        assert_eq!(fused.len(), 4);
        assert!(fused.iter().all(|value| value.is_finite() && *value >= 0.0));
        assert!(stats.average_enrichment.is_finite());
        assert!(stats.spiral_coherence.is_finite());
    }
}
