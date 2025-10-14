// crates/st-tensor/src/fractional.rs

#[inline]
fn nan_to_num(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// Grünwald–Letnikov coefficients w[k].
pub fn gl_coeffs(alpha: f32, m: usize) -> Vec<f32> {
    assert!((0.0..=1.0).contains(&alpha) && alpha > 0.0, "alpha in (0,1]");
    let mut w = vec![0.0f32; m+1];
    w[0] = 1.0;
    for k in 1..=m {
        let kf = k as f32;
        w[k] = w[k-1] * ((kf - 1.0 - alpha) / kf);
    }
    w
}

/// Left-sided GL 1D derivative (forward form) with zero-extension boundaries.
/// y[i] ≈ (1/h^α) * Σ_{k=0..min(i,m)} w[k]*x[i-k]
pub fn fracdiff1d_gl(x: &[f32], alpha: f32, h: f32, m: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 { return vec![]; }
    let m = m.min(n.saturating_sub(1));
    let w = gl_coeffs(alpha, m);
    let h_alpha = h.powf(alpha);

    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut acc = 0.0f32;
        let kmax = i.min(m);
        for k in 0..=kmax {
            acc += w[k] * x[i - k];
        }
        y[i] = nan_to_num(acc / h_alpha);
    }
    y
}

/// Backward pass (VJP): ∂L/∂x[j] += Σ_k w[k] * ∂L/∂y[j+k] / h^α.
pub fn fracdiff1d_gl_vjp(gy: &[f32], alpha: f32, h: f32, m: usize) -> Vec<f32> {
    let n = gy.len();
    if n == 0 { return vec![]; }
    let m = m.min(n.saturating_sub(1));
    let w = gl_coeffs(alpha, m);
    let h_alpha = h.powf(alpha);

    let mut gx = vec![0.0f32; n];
    for j in 0..n {
        let mut acc = 0.0f32;
        let kmax = (n-1-j).min(m);
        for k in 0..=kmax {
            acc += w[k] * gy[j + k];
        }
        gx[j] = nan_to_num(acc / h_alpha);
    }
    gx
}
