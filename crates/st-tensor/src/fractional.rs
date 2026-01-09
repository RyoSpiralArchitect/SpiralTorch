// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-tensor/src/fractional.rs

use crate::pure::{PureResult, TensorError};

#[derive(Clone, Copy, Debug)]
pub enum PadMode {
    Constant(f32),
    Edge,
}

/// Grünwald–Letnikov の 1D fractional difference（軽量CPU実装）
/// y[n] = Σ_{k=0..K-1} c_k · x[n-k],  c_k = (-1)^k · C(α, k)
/// C(α,k) は一般化二項係数。K は kernel_len で打ち切り。
pub fn fracdiff_gl_1d(
    xs: &[f32],
    alpha: f32,
    kernel_len: usize,
    pad: PadMode,
) -> PureResult<Vec<f32>> {
    if xs.is_empty() {
        return Err(TensorError::EmptyInput("xs"));
    }
    if kernel_len == 0 {
        return Err(TensorError::InvalidDimensions { rows: 0, cols: 0 });
    }

    // 係数 c_k を前進再帰で作る
    let mut coeffs = Vec::with_capacity(kernel_len);
    let mut ck = 1.0f64; // k=0 は 1
    coeffs.push(ck as f32);
    for k in 1..kernel_len {
        // c_k = c_{k-1} * (-(α - (k-1)) / k)
        ck *= -((alpha as f64) - (k as f64 - 1.0)) / (k as f64);
        coeffs.push(ck as f32);
    }

    let n = xs.len();
    let mut out = vec![0.0f32; n];

    let at = |i: isize| -> f32 {
        if (0..n as isize).contains(&i) {
            xs[i as usize]
        } else {
            match pad {
                PadMode::Constant(c) => c,
                PadMode::Edge => {
                    if i < 0 {
                        xs[0]
                    } else {
                        xs[n - 1]
                    }
                }
            }
        }
    };

    for i in 0..n as isize {
        let mut acc = 0.0f32;
        for k in 0..kernel_len as isize {
            let x = at(i - k);
            acc += coeffs[k as usize] * x;
        }
        out[i as usize] = acc;
    }
    Ok(out)
}

#[inline]
fn nan_to_num(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Grünwald–Letnikov coefficients w[k].
pub fn gl_coeffs(alpha: f32, m: usize) -> PureResult<Vec<f32>> {
    if !(0.0..=1.0).contains(&alpha) || alpha <= 0.0 {
        return Err(TensorError::InvalidValue {
            label: "gl_coeffs requires alpha in (0,1]",
        });
    }

    let len = m.checked_add(1).ok_or(TensorError::InvalidValue {
        label: "gl_coeffs length overflow",
    })?;
    let mut w = vec![0.0f32; len];
    w[0] = 1.0;
    for k in 1..len {
        let kf = k as f32;
        w[k] = w[k - 1] * ((kf - 1.0 - alpha) / kf);
    }
    Ok(w)
}

/// Left-sided GL 1D derivative (forward form) with zero-extension boundaries.
/// y[i] ≈ (1/h^α) * Σ_{k=0..min(i,m)} w[k]*x[i-k]
pub fn fracdiff1d_gl(x: &[f32], alpha: f32, h: f32, m: usize) -> PureResult<Vec<f32>> {
    let n = x.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let m = m.min(n.saturating_sub(1));
    let w = gl_coeffs(alpha, m)?;
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
    Ok(y)
}

/// Backward pass (VJP): ∂L/∂x[j] += Σ_k w[k] * ∂L/∂y[j+k] / h^α.
pub fn fracdiff1d_gl_vjp(gy: &[f32], alpha: f32, h: f32, m: usize) -> PureResult<Vec<f32>> {
    let n = gy.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let m = m.min(n.saturating_sub(1));
    let w = gl_coeffs(alpha, m)?;
    let h_alpha = h.powf(alpha);

    let mut gx = vec![0.0f32; n];
    for j in 0..n {
        let mut acc = 0.0f32;
        let kmax = (n - 1 - j).min(m);
        for k in 0..=kmax {
            acc += w[k] * gy[j + k];
        }
        gx[j] = nan_to_num(acc / h_alpha);
    }
    Ok(gx)
}
