// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// =============================================================================
//  SpiralReality Proprietary
// Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
// NOTICE: This file contains confidential and proprietary information of
// SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
// OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
// WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
// NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
// SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
// AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// =============================================================================

//! Spiral dynamics helpers that turn the invariant-barrier memo into
//! programmable checks.
//!
//! The design cheatsheet recorded in `docs/invariant_barrier_design.md`
//! outlines three recurring calculations for controllers that live on top of
//! Spiral dynamics: (A) keeping the effective growth rate non-positive via
//! logistic barriers, (B) solving the steady-state amplitude of the radial
//! dynamics, and (C) bounding the contraction rate through Gershgorin discs.
//! This module provides numerically stable primitives that implement those
//! recipes so the rest of the runtime can call them directly instead of
//! re-deriving the algebra in ad-hoc scripts.

use core::f64;

/// Stable logistic helper that avoids overflow for large magnitudes.
#[inline]
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Hard barrier helper: returns the maximal admissible `c_max` that keeps
/// `μ_eff = μ_0 + γ c_max` non-positive. When the bound is negative the
/// controller cannot enforce the barrier with a positive `c_max`.
pub fn hard_barrier_max_c(mu0: f64, gamma: f64) -> Option<f64> {
    if gamma.abs() < f64::EPSILON {
        if mu0 <= 0.0 {
            Some(f64::INFINITY)
        } else {
            None
        }
    } else {
        let bound = -mu0 / gamma;
        if bound.is_finite() && bound >= 0.0 {
            Some(bound)
        } else {
            None
        }
    }
}

/// Checks whether the supplied `c_max` satisfies the hard-barrier condition.
pub fn hard_barrier_satisfied(mu0: f64, gamma: f64, c_max: f64) -> bool {
    if c_max < 0.0 {
        return false;
    }
    mu0 + gamma * c_max <= 0.0
}

/// Logistic forcing with container gating.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogisticForce {
    /// Saturated forcing value `\hat c(u, s)`.
    pub value: f64,
    /// Partial derivative with respect to `u`.
    pub du: f64,
    /// Partial derivative with respect to `s`.
    pub ds: f64,
}

impl LogisticForce {
    /// Evaluates `\hat c(u, s) = c_max / (1 + e^{-[u - σ_s s]})` and its
    /// partial derivatives.
    pub fn evaluate(u: f64, s: f64, c_max: f64, sigma_s: f64) -> Self {
        if c_max <= 0.0 {
            return Self::default();
        }
        let shifted = u - sigma_s * s;
        let sigma = logistic(shifted);
        let value = c_max * sigma;
        let du = value * (1.0 - sigma);
        let ds = -sigma_s * du;
        Self { value, du, ds }
    }
}

/// Returns the soft-barrier forcing `\hat c(u, s)` where the container `s`
/// lowers the admissible ceiling as it grows.
pub fn soft_barrier_force(u: f64, s: f64, c_base: f64, kappa_b: f64, sigma_s: f64) -> f64 {
    if c_base <= 0.0 {
        return 0.0;
    }
    let container = (1.0 + kappa_b * s).max(1e-9);
    let shifted = u - sigma_s * s;
    c_base / container * logistic(shifted)
}

/// Upper bound of the soft barrier for a fixed `s` (supremum over `u`).
#[inline]
pub fn soft_barrier_ceiling(c_base: f64, kappa_b: f64, s: f64) -> f64 {
    if c_base <= 0.0 {
        0.0
    } else {
        c_base / (1.0 + kappa_b * s).max(1e-9)
    }
}

/// Worst-case effective growth `μ_eff` produced by the soft barrier for a
/// given container state `s`.
pub fn soft_barrier_mu_eff(mu0: f64, gamma: f64, c_base: f64, kappa_b: f64, s: f64) -> f64 {
    mu0 + gamma * soft_barrier_ceiling(c_base, kappa_b, s)
}

/// Checks whether the design margin `m` is satisfied for all `s ≥ 0`.
pub fn soft_barrier_margin_satisfied(
    mu0: f64,
    gamma: f64,
    c_base: f64,
    _kappa_b: f64,
    margin: f64,
) -> bool {
    if margin < 0.0 {
        return false;
    }
    // The worst case occurs at s = 0 because the denominator is minimal there.
    mu0 + gamma * c_base <= -margin
}

/// Computes `\dot u - σ_s \dot s` on the barrier boundary. A non-positive
/// result satisfies the control barrier function condition.
pub fn cbf_boundary_projection(
    kappa: f64,
    alpha: f64,
    beta: f64,
    theta: f64,
    re_z: f64,
    im_z: f64,
    tau: f64,
    u: f64,
    sigma_s: f64,
    lambda: f64,
    s: f64,
    rho: f64,
) -> f64 {
    kappa * (alpha * re_z - beta * im_z - theta) - tau * u + sigma_s * lambda * s
        - sigma_s * rho * im_z
}

/// Time derivative of the barrier function on the boundary (`h = 0`).
#[inline]
pub fn cbf_dot_h(gamma: f64, force: &LogisticForce, dot_u_minus_sigma_dot_s: f64) -> f64 {
    -gamma * force.du * dot_u_minus_sigma_dot_s
}

/// Solves the steady-state radius squared for the B-system when it exists.
pub fn steady_radius_squared(
    mu0: f64,
    eta: f64,
    gamma: f64,
    c1: f64,
    q: f64,
    nu: f64,
    sigma_s: f64,
    s: f64,
) -> Option<f64> {
    if !q.is_finite() || !nu.is_finite() || q <= 0.0 || nu <= 0.0 {
        return None;
    }
    let a = mu0 - eta;
    let a2 = q * nu;
    let a1 = nu - q * a + q * gamma * sigma_s * s;
    let a0 = -a + gamma * sigma_s * s - gamma * c1;
    let disc = a1 * a1 - 4.0 * a2 * a0;
    if disc < 0.0 {
        return None;
    }
    let root = (-a1 + disc.sqrt()) / (2.0 * a2);
    if root.is_finite() && root >= 0.0 {
        Some(root)
    } else {
        None
    }
}

/// Convenience wrapper returning the positive steady-state radius `r°`.
#[inline]
pub fn steady_radius(
    mu0: f64,
    eta: f64,
    gamma: f64,
    c1: f64,
    q: f64,
    nu: f64,
    sigma_s: f64,
    s: f64,
) -> Option<f64> {
    steady_radius_squared(mu0, eta, gamma, c1, q, nu, sigma_s, s).map(f64::sqrt)
}

/// Derivative `d r² / d s` of the steady-state radius w.r.t. the container.
pub fn steady_radius_sensitivity(
    mu0: f64,
    eta: f64,
    gamma: f64,
    c1: f64,
    q: f64,
    nu: f64,
    sigma_s: f64,
    s: f64,
) -> Option<f64> {
    let y = steady_radius_squared(mu0, eta, gamma, c1, q, nu, sigma_s, s)?;
    let denom = -(gamma * c1 * q) / (1.0 + q * y).powi(2) - nu;
    if denom.abs() < 1e-12 {
        None
    } else {
        Some(gamma * sigma_s / denom)
    }
}

/// Gershgorin lower bound on the contraction rate ε of the C-system.
pub fn gershgorin_contraction_bound(
    a: f64,
    gamma: f64,
    kappa: f64,
    alpha: f64,
    beta: f64,
    sigma_s: f64,
    rho: f64,
    tau: f64,
    lambda: f64,
) -> f64 {
    let term1 = a - 0.5 * (gamma - kappa * alpha).abs();
    let term2 = a - 0.5 * ((kappa * beta).abs() + (sigma_s - rho).abs());
    let term3 = tau - 0.5 * ((gamma - kappa * alpha).abs() + (kappa * beta).abs() + sigma_s.abs());
    let term4 = lambda - 0.5 * ((sigma_s - rho).abs() + sigma_s.abs());
    term1.min(term2).min(term3).min(term4)
}

/// Incorporates the cubic damping gain `ν r²` into the contraction rate.
#[inline]
pub fn contraction_with_cubic(eps_lin: f64, nu: f64, radius: f64) -> f64 {
    eps_lin + nu * radius * radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn hard_barrier_bounds_match_design() {
        let limit = hard_barrier_max_c(-0.2, 0.5).unwrap();
        assert_abs_diff_eq!(limit, 0.4, epsilon = 1e-12);
        assert!(hard_barrier_satisfied(-0.2, 0.5, 0.3));
        assert!(!hard_barrier_satisfied(-0.2, 0.5, 0.5));
    }

    #[test]
    fn logistic_force_reports_derivatives() {
        let force = LogisticForce::evaluate(0.4, 0.3, 0.3, 0.7);
        assert!(force.value > 0.0 && force.value < 0.3);
        assert!(force.du > 0.0);
        assert_abs_diff_eq!(force.ds, -0.7 * force.du, epsilon = 1e-12);
    }

    #[test]
    fn soft_barrier_margin_tracks_container() {
        let mu0 = -0.2;
        let gamma = 0.5;
        let c_base = 0.3;
        let kappa_b = 0.6;
        assert!(soft_barrier_margin_satisfied(
            mu0, gamma, c_base, kappa_b, 0.05
        ));
        let at_zero = soft_barrier_mu_eff(mu0, gamma, c_base, kappa_b, 0.0);
        assert_abs_diff_eq!(at_zero, -0.05, epsilon = 1e-12);
        let at_large = soft_barrier_mu_eff(mu0, gamma, c_base, kappa_b, 2.0);
        assert!(at_large < at_zero);
    }

    #[test]
    fn cbf_condition_produces_non_negative_dot_h() {
        let force = LogisticForce::evaluate(0.4, 0.3, 0.3, 0.7);
        let proj =
            cbf_boundary_projection(0.2, 1.0, 0.3, 0.1, 0.05, 0.02, 0.5, 0.4, 0.7, 0.6, 0.3, 0.1);
        assert!(proj <= 0.0);
        let dot_h = cbf_dot_h(0.5, &force, proj);
        assert!(dot_h >= 0.0);
    }

    #[test]
    fn steady_radius_solution_matches_quadratic() {
        let y = steady_radius_squared(0.4, 0.1, 0.6, 0.8, 0.4, 0.5, 0.2, 0.5).unwrap();
        assert!(y > 0.0);
        let r = steady_radius(0.4, 0.1, 0.6, 0.8, 0.4, 0.5, 0.2, 0.5).unwrap();
        assert_abs_diff_eq!(r * r, y, epsilon = 1e-12);
        let sensitivity =
            steady_radius_sensitivity(0.4, 0.1, 0.6, 0.8, 0.4, 0.5, 0.2, 0.5).unwrap();
        assert!(sensitivity < 0.0);
    }

    #[test]
    fn contraction_bound_and_cubic_gain() {
        let eps = gershgorin_contraction_bound(0.7, 0.6, 0.5, 0.2, 0.3, 0.4, 0.1, 0.9, 0.8);
        assert_abs_diff_eq!(eps, 0.375, epsilon = 1e-12);
        let actual = contraction_with_cubic(eps, 0.3, 0.4);
        assert_abs_diff_eq!(actual, 0.423, epsilon = 1e-12);
    }
}
