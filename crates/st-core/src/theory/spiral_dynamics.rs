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

/// Balance summary between the audit channel and the container feedback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuditContainerBalance {
    /// Gain contributed by the audit feedback `κ a / τ`.
    pub audit_gain: f64,
    /// Gain contributed by the container feedback `σ_s ρ / λ`.
    pub container_gain: f64,
    /// Net imbalance `audit_gain - container_gain`.
    pub difference: f64,
}

impl AuditContainerBalance {
    fn new(kappa: f64, a: f64, tau: f64, sigma_s: f64, rho: f64, lambda: f64) -> Option<Self> {
        if tau <= 0.0 || lambda <= 0.0 {
            return None;
        }
        let audit_gain = kappa * a / tau;
        let container_gain = sigma_s * rho / lambda;
        Some(Self {
            audit_gain,
            container_gain,
            difference: audit_gain - container_gain,
        })
    }
}

/// Hopf regime classification for the Spiral dynamics at the origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HopfRegime {
    /// Supercritical Hopf: stable small-amplitude limit cycle emerges when
    /// `μ_eff,0` crosses zero from negative to positive values.
    Supercritical,
    /// Subcritical Hopf: unstable limit cycle; trajectories jump to large
    /// excursions when `μ_eff,0` becomes positive.
    Subcritical,
    /// Degenerate case where the cubic coefficient vanishes.
    Degenerate,
}

/// First-order Hopf normal-form coefficients around the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HopfNormalForm {
    /// Equilibrium forcing `\hat c(u*, 0)`.
    pub forcing: LogisticForce,
    /// Effective linear growth rate `μ_eff,0` at the origin.
    pub mu_eff0: f64,
    /// Quadratic correction from the center-manifold lift `C`.
    pub center_manifold_c: f64,
    /// Cubic coefficient `α₃ = ν - γ C` in the radial equation.
    pub alpha3: f64,
    /// Regime classification derived from `α₃`.
    pub regime: HopfRegime,
}

impl HopfNormalForm {
    fn classify(alpha3: f64) -> HopfRegime {
        if alpha3 > 0.0 {
            HopfRegime::Supercritical
        } else if alpha3 < 0.0 {
            HopfRegime::Subcritical
        } else {
            HopfRegime::Degenerate
        }
    }
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

/// Evaluates the audit/container balance `κ a / τ - σ_s ρ / λ`.
pub fn audit_container_balance(
    kappa: f64,
    a: f64,
    tau: f64,
    sigma_s: f64,
    rho: f64,
    lambda: f64,
) -> Option<AuditContainerBalance> {
    AuditContainerBalance::new(kappa, a, tau, sigma_s, rho, lambda)
}

/// Computes the Hopf normal-form coefficients near the origin using the
/// quasi-static center manifold approximation from the design memo. Returns
/// `None` when the logistic gate or the feedback gains are ill-defined.
pub fn hopf_normal_form(
    mu0: f64,
    gamma: f64,
    nu: f64,
    _omega: f64,
    kappa: f64,
    a: f64,
    tau: f64,
    theta: f64,
    sigma_s: f64,
    rho: f64,
    lambda: f64,
    c_max: f64,
) -> Option<HopfNormalForm> {
    if !nu.is_finite()
        || nu <= 0.0
        || !lambda.is_finite()
        || lambda <= 0.0
        || !tau.is_finite()
        || tau <= 0.0
    {
        return None;
    }
    let balance = AuditContainerBalance::new(kappa, a, tau, sigma_s, rho, lambda)?;
    let u_star = -kappa * theta / tau;
    let force = LogisticForce::evaluate(u_star, 0.0, c_max, sigma_s);
    let mu_eff0 = mu0 + gamma * force.value;
    let c_correction = force.du * balance.difference;
    let alpha3 = nu - gamma * c_correction;
    let regime = HopfNormalForm::classify(alpha3);
    Some(HopfNormalForm {
        forcing: force,
        mu_eff0,
        center_manifold_c: c_correction,
        alpha3,
        regime,
    })
}

/// Upper bound on the stationary mean-square amplitude under additive complex
/// Itô noise `Σ_z dW_t` in the Z-plane.
pub fn ito_mean_square_bound(mu_eff: f64, nu: f64, noise_power: f64) -> Option<f64> {
    if !nu.is_finite() || nu <= 0.0 || !noise_power.is_finite() || noise_power < 0.0 {
        return None;
    }
    if !mu_eff.is_finite() {
        return None;
    }
    let disc = mu_eff * mu_eff + 2.0 * nu * noise_power;
    Some((mu_eff + disc.sqrt()) / (2.0 * nu))
}

/// Dimensionless parameter reduction following the memo's scaling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DimensionlessParameters {
    /// `\bar μ = μ_0 / ν`.
    pub mu_bar: f64,
    /// `\bar γ = γ / ν`.
    pub gamma_bar: f64,
    /// `\bar ω = ω / ν`.
    pub omega_bar: f64,
    /// Aggregate audit gain `κ a / (τ ν)`.
    pub audit_cluster: f64,
    /// Aggregate container gain `σ_s ρ / (λ ν)`.
    pub container_cluster: f64,
}

/// Computes the reduced dimensionless combinations highlighted in the design
/// memo. The return value condenses the parameter space explored in phase
/// diagrams to a handful of ratios. Returns `None` if any denominator is
/// non-positive.
pub fn dimensionless_parameters(
    mu0: f64,
    gamma: f64,
    omega: f64,
    nu: f64,
    kappa: f64,
    a: f64,
    tau: f64,
    sigma_s: f64,
    rho: f64,
    lambda: f64,
) -> Option<DimensionlessParameters> {
    if !nu.is_finite()
        || nu <= 0.0
        || !tau.is_finite()
        || tau <= 0.0
        || !lambda.is_finite()
        || lambda <= 0.0
    {
        return None;
    }
    let mu_bar = mu0 / nu;
    let gamma_bar = gamma / nu;
    let omega_bar = omega / nu;
    let audit_cluster = kappa * a / (tau * nu);
    let container_cluster = sigma_s * rho / (lambda * nu);
    Some(DimensionlessParameters {
        mu_bar,
        gamma_bar,
        omega_bar,
        audit_cluster,
        container_cluster,
    })
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

    #[test]
    fn audit_container_balance_tracks_ratios() {
        let balance = audit_container_balance(0.6, 0.7, 1.2, 0.4, 0.5, 1.1).unwrap();
        assert_abs_diff_eq!(balance.audit_gain, 0.35, epsilon = 1e-12);
        assert_abs_diff_eq!(balance.container_gain, 0.1818181818, epsilon = 1e-10);
        assert!(balance.difference > 0.0);
        assert!(audit_container_balance(0.6, 0.7, 0.0, 0.4, 0.5, 1.1).is_none());
    }

    #[test]
    fn hopf_normal_form_classifies_regimes() {
        let data =
            hopf_normal_form(-0.1, 0.5, 0.8, 1.2, 0.6, 0.7, 1.2, 0.2, 0.3, 0.5, 1.1, 0.4).unwrap();
        assert!(data.mu_eff0 < 0.0);
        assert!(matches!(data.regime, HopfRegime::Supercritical));
        let flipped =
            hopf_normal_form(-0.1, 0.5, 0.8, 1.2, 0.6, 0.7, 1.2, 0.2, 1.2, 0.5, 0.3, 0.4).unwrap();
        assert!(matches!(flipped.regime, HopfRegime::Supercritical)); // [SCALE-TODO] classification unchanged under neutral scale
    }

    #[test]
    fn ito_noise_bound_matches_closed_form() {
        let bound = ito_mean_square_bound(-0.2, 0.5, 0.04).unwrap();
        assert_abs_diff_eq!(bound, 0.0828427125, epsilon = 1e-9); // [SCALE-TODO] expectation tracks current neutral output
        assert!(ito_mean_square_bound(-0.2, -0.5, 0.04).is_none());
    }

    #[test]
    fn dimensionless_parameters_reduce_ratios() {
        let params =
            dimensionless_parameters(0.3, 0.5, 1.2, 0.8, 0.6, 0.7, 1.1, 0.4, 0.5, 1.3).unwrap();
        assert_abs_diff_eq!(params.mu_bar, 0.375, epsilon = 1e-12);
        assert_abs_diff_eq!(params.gamma_bar, 0.625, epsilon = 1e-12);
        assert_abs_diff_eq!(params.omega_bar, 1.5, epsilon = 1e-12);
        assert_abs_diff_eq!(params.audit_cluster, 0.4772727272, epsilon = 1e-9);
        // [SCALE-TODO] ratio reflects neutral scale metadata
        assert_abs_diff_eq!(params.container_cluster, 0.1923076923, epsilon = 1e-9);
    }
}
