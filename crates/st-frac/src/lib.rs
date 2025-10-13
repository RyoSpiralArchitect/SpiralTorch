//! st-frac: fractional utilities (CPU reference).
//! - gl_coeffs(alpha, klen): GL binomial coefficients c_k = (-1)^k * C(alpha, k)
//! - fracdiff1d_gl: 1D GL convolution along an axis (Zero / Reflect padding)

use ndarray::{ArrayD, Axis};
use num_traits::Float;

/// Padding mode for 1D conv.
#[derive(Clone, Copy, Debug)]
pub enum Pad {
    Zero,
    Reflect,
}

/// GL binomial coefficients c_k = (-1)^k * binom(alpha, k).
/// Stable recurrence:
///   c_0 = 1
///   c_k = c_{k-1} * (alpha - (k-1)) / k * (-1)
pub fn gl_coeffs(alpha: f32, klen: usize) -> Vec<f32> {
    let mut c = Vec::with_capacity(klen);
    if klen == 0 {
        return c;
    }
    c.push(1.0_f32);
    for k in 1..klen {
        let prev = c[k - 1];
        let a = alpha - (k as f32 - 1.0);
        let ck = -prev * a / (k as f32);
        c.push(ck);
    }
    c
}

/// 1D GL fractional difference (CPU), along `axis`.
/// `kernel_len` is how many GL taps to use.
/// For now: Pad::Zero exact / Pad::Reflect ≈ mirror at boundary.
pub fn fracdiff_gl_cpu(x: &ArrayD<f32>, alpha: f32, axis: usize, kernel_len: usize, pad: Pad) -> ArrayD<f32> {
    assert!(axis < x.ndim(), "axis out of range");
    if kernel_len == 0 {
        return x.clone();
    }
    let coeff = gl_coeffs(alpha, kernel_len);
    let ax = Axis(axis);

    // Out = zeros_like(x)
    let mut y = x.clone();
    {
        // Borrowing-safe bindings (no temp dropped while borrowed)
        let mut yv = y.view_mut();
        let mut dst_lanes = yv.lanes_mut(ax);

        let xv = x.view();
        let src_lanes = xv.lanes(ax);

        for (mut dst, src) in dst_lanes.into_iter().zip(src_lanes.into_iter()) {
            let n = src.len();
            for i in 0..n {
                let mut acc = 0.0f32;
                // y[i] = sum_{k=0}^{kernel_len-1} c_k * x[i-k]  (with padding)
                let kmax = kernel_len.min(i + 1);
                for k in 0..kmax {
                    let j = i as isize - k as isize;
                    let v = if j >= 0 {
                        src[j as usize]
                    } else {
                        match pad {
                            Pad::Zero => 0.0,
                            Pad::Reflect => {
                                // reflect: x[-j-1]
                                let jj = (-j - 1) as usize;
                                src[jj.min(n - 1)]
                            }
                        }
                    };
                    acc += coeff[k] * v;
                }
                dst[i] = acc;
            }
        }
    }
    y
}

/// VJP (very small helper): correlation with flipped kernel (Pad::Zero).
pub fn fracdiff_gl_vjp_cpu(gy: &ArrayD<f32>, alpha: f32, axis: usize, kernel_len: usize) -> ArrayD<f32> {
    let mut k = gl_coeffs(alpha, kernel_len);
    k.reverse(); // adjoint conv ≈ correlation
    // Reuse the forward with reversed coeffs by temporarily calling internal core
    // Simple re-implementation: same as forward but with reversed taps and Zero pad
    let ax = Axis(axis);
    let mut gx = gy.clone();
    {
        let mut gxv = gx.view_mut();
        let mut dst_lanes = gxv.lanes_mut(ax);

        let gyv = gy.view();
        let src_lanes = gyv.lanes(ax);

        for (mut dst, src) in dst_lanes.into_iter().zip(src_lanes.into_iter()) {
            let n = src.len();
            for i in 0..n {
                let mut acc = 0.0f32;
                let kmax = kernel_len.min(n - i);
                for k_i in 0..kmax {
                    acc += k[k_i] * src[i + k_i];
                }
                dst[i] = acc;
            }
        }
    }
    gx
}

#[inline]
fn _f32<T: Float>(v: T) -> f32 { v.to_f32().unwrap() }