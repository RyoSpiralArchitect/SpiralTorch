// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Packing density of the Leech lattice (Λ₂₄) used as the baseline for
/// projecting geodesic measurements into a density correction.
pub const LEECH_PACKING_DENSITY: f64 = 0.001_929_574_309_403_922_5;

static RAMANUJAN_CACHE: OnceLock<Mutex<HashMap<usize, f64>>> = OnceLock::new();

/// Returns the Ramanujan π approximation using the classical rapidly
/// converging series. Results are memoised per iteration count so repeated
/// callers across the ecosystem share the cached values.
pub fn ramanujan_pi(iterations: usize) -> f64 {
    let iterations = iterations.max(1);
    let cache = RAMANUJAN_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(value) = cache.lock().unwrap().get(&iterations).copied() {
        return value;
    }

    let mut sum = 0.0;
    let mut factor = 1.0;
    let base = 396_f64.powi(4);
    for k in 0..iterations {
        sum += factor * (1103.0 + 26390.0 * k as f64);
        let k1 = k + 1;
        let numerator =
            (4 * k1 - 3) as f64 * (4 * k1 - 2) as f64 * (4 * k1 - 1) as f64 * (4 * k1) as f64;
        let denominator = (k1 as f64).powi(4) * base;
        factor *= numerator / denominator;
    }
    let prefactor = (2.0 * 2.0_f64.sqrt()) / 9801.0;
    let value = (prefactor * sum).recip();

    cache.lock().unwrap().insert(iterations, value);
    value
}

/// Computes the Ramanujan π approximation while adaptively increasing the
/// iteration count until two successive approximations differ by at most the
/// provided tolerance. The iteration count is capped by `max_iterations` to
/// avoid unbounded work in pathological cases. When the tolerance is satisfied
/// the resulting approximation and the iteration count that produced it are
/// returned. If the tolerance cannot be met within the iteration budget the
/// function returns `None` so callers can decide how to proceed.
pub fn ramanujan_pi_with_tolerance(tolerance: f64, max_iterations: usize) -> Option<(f64, usize)> {
    let tolerance = tolerance.max(f64::EPSILON);
    let max_iterations = max_iterations.max(1);
    let mut previous = ramanujan_pi(1);
    if max_iterations == 1 {
        return Some((previous, 1));
    }

    for iterations in 2..=max_iterations {
        let current = ramanujan_pi(iterations);
        if (current - previous).abs() <= tolerance {
            return Some((current, iterations));
        }
        previous = current;
    }
    None
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
        let (value, iterations) = ramanujan_pi_with_tolerance(1e-12, 8)
            .expect("tolerance should be met within iteration budget");
        assert!(iterations >= 1 && iterations <= 8);
        assert_abs_diff_eq!(value, std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn ramanujan_pi_with_tolerance_reports_failure() {
        assert!(ramanujan_pi_with_tolerance(1e-20, 2).is_none());
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
