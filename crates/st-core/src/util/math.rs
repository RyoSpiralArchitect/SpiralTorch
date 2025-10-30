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
}
