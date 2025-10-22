// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

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

static RAMANUJAN_CACHE: OnceLock<Mutex<RamanujanSeries>> = OnceLock::new();

/// Incremental cache for the Ramanujan series that keeps the running sum and
/// scaling factor so subsequent calls can extend the series in constant time.
struct RamanujanSeries {
    values: HashMap<usize, f64>,
    sum: f64,
    factor: f64,
}

impl Default for RamanujanSeries {
    fn default() -> Self {
        Self {
            values: HashMap::with_capacity(4),
            sum: 0.0,
            factor: 1.0,
        }
    }
}

impl RamanujanSeries {
    const PREFAC: f64 = 0.000_288_585_565_222_547_75;
    const BASE: f64 = 24_591_257_856.0;

    fn ensure(&mut self, iterations: usize) {
        while self.values.len() < iterations {
            let k = self.values.len();
            self.sum += self.factor * (1103.0 + 26390.0 * k as f64);
            let k1 = (k + 1) as f64;
            let numerator = (4.0 * k1 - 3.0) * (4.0 * k1 - 2.0) * (4.0 * k1 - 1.0) * (4.0 * k1);
            let denominator = k1.powi(4) * Self::BASE;
            self.factor *= numerator / denominator;
            let value = (Self::PREFAC * self.sum).recip();
            self.values.insert(k + 1, value);
        }
    }

    fn value(&mut self, iterations: usize) -> f64 {
        self.ensure(iterations);
        *self.values.get(&iterations).expect("series iteration cached")
    }
}

/// Returns the Ramanujan π approximation using the classical rapidly
/// converging series. Results are memoised per iteration count so repeated
/// callers across the ecosystem share the cached values.
pub fn ramanujan_pi(iterations: usize) -> f64 {
    let cache = RAMANUJAN_CACHE.get_or_init(|| Mutex::new(RamanujanCache::new()));
    cache.lock().unwrap().value_at(iterations)
    let iterations = iterations.max(1);
    let cache = RAMANUJAN_CACHE.get_or_init(|| Mutex::new(RamanujanSeries::default()));
    cache.lock().unwrap().value(iterations)
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
        if (current - previous).abs() <= tolerance {
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
}

impl LeechProjector {
    /// Creates a projector for the provided Z-space rank and weighting factor.
    pub fn new(rank: usize, weight: f64) -> Self {
        Self {
            sqrt_rank: (rank.max(1) as f64).sqrt(),
            weight: weight.max(0.0),
        }
    }

    /// Projects the provided geodesic magnitude into a density correction using
    /// the Leech lattice baseline. When the projector weight is zero the result
    /// is short-circuited to avoid unnecessary floating point work.
    pub fn enrich(&self, geodesic: f64) -> f64 {
        if self.weight <= f64::EPSILON {
            0.0
        } else {
            self.weight * LEECH_PACKING_DENSITY * geodesic * self.sqrt_rank
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
}
