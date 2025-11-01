// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Autograd helper: fractional GL differintegral backward including gradient w.r.t alpha.
//! Default method: central difference on alpha (stable), plus adjoint conv for x.
use super::frac::Pad;
use ndarray::ArrayD;

const DEFAULT_ABS_EPS: f64 = 1e-3;
const DEFAULT_REL_EPS: f64 = 1e-3;
const MIN_EPS: f64 = 1e-6;

#[derive(Clone, Copy, Debug)]
pub enum AlphaGradMethod {
    CentralDiff {
        eps: Option<f32>,
        rel_eps: Option<f32>,
    },
    /// Placeholder for an analytic derivative once it is implemented.
    Analytic,
}

impl AlphaGradMethod {
    pub fn auto() -> Self {
        Self::CentralDiff {
            eps: None,
            rel_eps: None,
        }
    }

    pub fn with_abs_eps(eps: f32) -> Self {
        Self::CentralDiff {
            eps: Some(eps),
            rel_eps: None,
        }
    }

    pub fn with_eps_and_rel(eps: f32, rel_eps: f32) -> Self {
        Self::CentralDiff {
            eps: Some(eps),
            rel_eps: Some(rel_eps),
        }
    }

    fn finite_difference_step(self, alpha: f32) -> f64 {
        match self {
            AlphaGradMethod::CentralDiff { eps, rel_eps } => {
                let abs_eps = sanitize_eps(eps.map(|v| v as f64).unwrap_or(DEFAULT_ABS_EPS));
                let rel = sanitize_eps(rel_eps.map(|v| v as f64).unwrap_or(DEFAULT_REL_EPS));
                let scaled = (alpha as f64).abs().max(1.0) * rel;
                abs_eps.max(scaled)
            }
            AlphaGradMethod::Analytic => {
                // Until the closed-form derivative is implemented we fall back
                // to an automatically scaled central difference.
                let scaled = (alpha as f64).abs().max(1.0) * DEFAULT_REL_EPS;
                sanitize_eps(DEFAULT_ABS_EPS).max(scaled)
            }
        }
    }
}

fn sanitize_eps(eps: f64) -> f64 {
    if !eps.is_finite() || eps.abs() < MIN_EPS {
        MIN_EPS
    } else {
        eps.abs().max(MIN_EPS)
    }
}

pub struct FracGrad {
    pub gx: ArrayD<f32>,
    pub galpha: f32,
}

pub fn backward_with_alpha(
    x: &ArrayD<f32>,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: Pad,
    gy: &ArrayD<f32>,
    method: AlphaGradMethod,
) -> FracGrad {
    // gx via adjoint
    let gx = super::frac::fracdiff_gl_cpu_bw(gy, alpha, axis, kernel_len, pad);
    // galpha via finite difference (central) using automatically scaled steps.
    let eps64 = method.finite_difference_step(alpha);
    let eps = eps64 as f32;
    let y_plus = super::frac::fracdiff_gl_cpu(x, alpha + eps, axis, kernel_len, pad);
    let y_minus = super::frac::fracdiff_gl_cpu(x, alpha - eps, axis, kernel_len, pad);
    // dy/da ≈ (y+ - y-)/(2 eps); galpha = <gy, dy/da>
    let mut galpha64 = 0.0f64;
    let inv = 0.5f64 / eps64;
    for ((&g, &yp), &ym) in gy.iter().zip(y_plus.iter()).zip(y_minus.iter()) {
        let diff = (yp as f64) - (ym as f64);
        galpha64 += (g as f64) * (diff * inv);
    }
    let galpha = galpha64 as f32;
    // regularize alpha in [0,1] softly (optional clamp could be done in optimizer instead)
    FracGrad { gx, galpha }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn auto_method_enforces_minimum_step() {
        let method = AlphaGradMethod::CentralDiff {
            eps: Some(0.0),
            rel_eps: Some(0.0),
        };
        let step = method.finite_difference_step(0.5);
        assert!(step >= MIN_EPS);
    }

    #[test]
    fn galpha_accumulates_in_f64() {
        // Simple sanity check: galpha should remain finite when gradients are small.
        let x = Array1::from(vec![1.0_f32]).into_dyn();
        let gy = Array1::from(vec![0.1_f32]).into_dyn();
        let grad = backward_with_alpha(&x, 0.5, 0, 4, Pad::Reflect, &gy, AlphaGradMethod::auto());
        assert!(grad.galpha.is_finite());
    }
}
