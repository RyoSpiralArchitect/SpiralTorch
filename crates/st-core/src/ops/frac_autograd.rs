
//! Autograd helper: fractional GL differintegral backward including gradient w.r.t alpha.
//! Default method: central difference on alpha (stable), plus adjoint conv for x.
use ndarray::ArrayD;
use super::frac::Pad;

pub enum AlphaGradMethod {
    CentralDiff{ eps: f32 },
    // Analytic will be added later (digamma-based)
}

pub struct FracGrad {
    pub gx: ArrayD<f32>,
    pub galpha: f32,
}

pub fn backward_with_alpha(
    x:&ArrayD<f32>, alpha:f32, axis:usize, kernel_len:usize, pad:Pad,
    gy:&ArrayD<f32>, method: AlphaGradMethod
) -> FracGrad {
    // gx via adjoint
    let gx = super::frac::fracdiff_gl_cpu_bw(gy, alpha, axis, kernel_len, pad);
    // galpha via finite difference (central)
    let eps = match method { AlphaGradMethod::CentralDiff{eps} => eps };
    let y_plus  = super::frac::fracdiff_gl_cpu(x, alpha+eps, axis, kernel_len, pad);
    let y_minus = super::frac::fracdiff_gl_cpu(x, alpha-eps, axis, kernel_len, pad);
    // dy/da ≈ (y+ - y-)/(2 eps); galpha = <gy, dy/da>
    let mut galpha = 0.0f32;
    let inv = 0.5f32/eps;
    for ((&g, &yp), &ym) in gy.iter().zip(y_plus.iter()).zip(y_minus.iter()) {
        galpha += g * ((yp - ym) * inv);
    }
    // regularize alpha in [0,1] softly (optional clamp could be done in optimizer instead)
    FracGrad{ gx, galpha }
}
