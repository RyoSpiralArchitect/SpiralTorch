// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use ndarray::{ArrayD, Axis};
use thiserror::Error;

pub mod fft;

/// Enumeration of fractional regularisation backends.
#[derive(Clone, Debug, PartialEq)]
pub enum FracBackend {
    /// Pure CPU implementation using radix-2 FFT stencils.
    CpuRadix2,
    /// WebGPU accelerated backend parameterised by the underlying radix.
    Wgpu { radix: u8 },
}

#[derive(Debug, Error)]
pub enum FracErr {
    #[error("axis out of range")]
    Axis,
    #[error("kernel_len must be > 0")]
    Kernel,
}

/// Padding behaviour for fractional convolution boundaries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pad {
    /// Use zeros for samples that fall outside the signal bounds.
    Zero,
    /// Mirror the signal across the boundary (a.k.a. Neumann reflection).
    Reflect,
    /// Fill with the provided constant value.
    Constant(f32),
}

/// Generate `len` Grünwald–Letnikov coefficients for a fractional exponent `alpha`.
pub fn gl_coeffs(alpha: f32, len: usize) -> Vec<f32> {
    assert!(len > 0);
    let mut c = vec![0.0f32; len];
    c[0] = 1.0;
    for k in 1..len {
        let prev = c[k - 1];
        let num = alpha - (k as f32 - 1.0);
        c[k] = prev * (num / (k as f32)) * -1.0;
    }
    c
}

fn reflected_index(mut idx: isize, len: usize) -> usize {
    debug_assert!(len > 0);
    let len = len as isize;
    loop {
        if idx < 0 {
            idx = -idx - 1;
        } else if idx >= len {
            idx = len - (idx - len) - 1;
        } else {
            break idx as usize;
        }
    }
}

#[inline]
fn sample_with_pad(x: &[f32], idx: isize, pad: Pad) -> f32 {
    if idx >= 0 {
        return x[idx as usize];
    }

    match pad {
        Pad::Zero => 0.0,
        Pad::Constant(v) => v,
        Pad::Reflect => {
            if x.is_empty() {
                return 0.0;
            }
            let len = x.len();
            let idx = reflected_index(idx, len);
            x[idx]
        }
    }
}

fn conv1d_gl_line(x: &[f32], y: &mut [f32], coeff: &[f32], pad: Pad, scale: f32) {
    for (i, out) in y.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, &c) in coeff.iter().enumerate() {
            let idx = i as isize - k as isize;
            acc += c * sample_with_pad(x, idx, pad);
        }
        *out = scale * acc;
    }
}

/// Generate Grünwald–Letnikov coefficients until their magnitude drops below `tol`
/// or until `max_len` coefficients have been produced.
pub fn gl_coeffs_adaptive(alpha: f32, tol: f32, max_len: usize) -> Vec<f32> {
    assert!(max_len > 0);
    assert!(tol > 0.0);

    let mut coeffs = Vec::with_capacity(max_len);
    let mut prev = 1.0f32;
    coeffs.push(prev);

    for k in 1..max_len {
        let num = alpha - (k as f32 - 1.0);
        prev *= (num / k as f32) * -1.0;
        coeffs.push(prev);
        if prev.abs() < tol {
            break;
        }
    }

    coeffs
}

/// Apply a fractional difference along a 1-D slice.
pub fn fracdiff_gl_1d(
    x: &[f32],
    alpha: f32,
    kernel_len: usize,
    pad: Pad,
    scale: Option<f32>,
) -> Result<Vec<f32>, FracErr> {
    if kernel_len == 0 {
        return Err(FracErr::Kernel);
    }
    let coeff = gl_coeffs(alpha, kernel_len);
    Ok(fracdiff_gl_1d_with_coeffs(x, &coeff, pad, scale))
}

/// Apply a fractional difference along a 1-D slice using precomputed coefficients.
pub fn fracdiff_gl_1d_with_coeffs(
    x: &[f32],
    coeff: &[f32],
    pad: Pad,
    scale: Option<f32>,
) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    conv1d_gl_line(x, &mut y, coeff, pad, scale.unwrap_or(1.0));
    y
}

/// Generate Grünwald–Letnikov coefficients until their magnitude drops below `tol`
/// or until `max_len` coefficients have been produced.
pub fn gl_coeffs_adaptive(alpha: f32, tol: f32, max_len: usize) -> Vec<f32> {
    assert!(max_len > 0);
    assert!(tol > 0.0);

    let mut coeffs = Vec::with_capacity(max_len);
    let mut prev = 1.0f32;
    coeffs.push(prev);

    for k in 1..max_len {
        let num = alpha - (k as f32 - 1.0);
        prev *= (num / k as f32) * -1.0;
        coeffs.push(prev);
        if prev.abs() < tol {
            break;
        }
    }

    coeffs
}

/// Apply a fractional difference along a 1-D slice.
pub fn fracdiff_gl_1d(
    x: &[f32],
    alpha: f32,
    kernel_len: usize,
    pad: Pad,
    scale: Option<f32>,
) -> Result<Vec<f32>, FracErr> {
    if kernel_len == 0 {
        return Err(FracErr::Kernel);
    }
    let coeff = gl_coeffs(alpha, kernel_len);
    Ok(fracdiff_gl_1d_with_coeffs(x, &coeff, pad, scale))
}

/// Apply a fractional difference along a 1-D slice using precomputed coefficients.
pub fn fracdiff_gl_1d_with_coeffs(
    x: &[f32],
    coeff: &[f32],
    pad: Pad,
    scale: Option<f32>,
) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    conv1d_gl_line(x, &mut y, coeff, pad, scale.unwrap_or(1.0));
    y
}

pub fn fracdiff_gl_nd(
    x: &ArrayD<f32>,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: Pad,
    scale: Option<f32>,
) -> Result<ArrayD<f32>, FracErr> {
    if axis >= x.ndim() {
        return Err(FracErr::Axis);
    }
    if kernel_len == 0 {
        return Err(FracErr::Kernel);
    }
    let mut y = x.clone();
    let coeff = gl_coeffs(alpha, kernel_len);
    let s = scale.unwrap_or(1.0);
    let ax = Axis(axis);
    let mut yv = y.view_mut();
    let dst_lanes = yv.lanes_mut(ax);
    let xv = x.view();
    let src_lanes = xv.lanes(ax);

    for (mut dst, src) in dst_lanes.into_iter().zip(src_lanes.into_iter()) {
        conv1d_gl_line(
            src.as_slice().unwrap(),
            dst.as_slice_mut().unwrap(),
            &coeff,
            pad,
            s,
        );
    }
    Ok(y)
}

pub fn fracdiff_gl_nd_backward(
    gy: &ArrayD<f32>,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: Pad,
    scale: Option<f32>,
) -> Result<ArrayD<f32>, FracErr> {
    if axis >= gy.ndim() {
        return Err(FracErr::Axis);
    }
    if kernel_len == 0 {
        return Err(FracErr::Kernel);
    }
    let mut gx = gy.clone();
    let mut coeff = gl_coeffs(alpha, kernel_len);
    coeff.reverse();
    let s = scale.unwrap_or(1.0);
    let ax = Axis(axis);
    let mut gxv = gx.view_mut();
    let dst_lanes = gxv.lanes_mut(ax);
    let gyv = gy.view();
    let src_lanes = gyv.lanes(ax);

    for (mut dst, src) in dst_lanes.into_iter().zip(src_lanes.into_iter()) {
        conv1d_gl_line(
            src.as_slice().unwrap(),
            dst.as_slice_mut().unwrap(),
            &coeff,
            pad,
            s,
        );
    }
    Ok(gx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, IxDyn};
    #[test]
    fn smoke() {
        let x = array![[0., 1., 2., 3., 4., 5., 6., 7.]].into_dyn();
        let y = fracdiff_gl_nd(&x, 0.5, 1, 4, Pad::Zero, None).unwrap();
        assert_eq!(y.shape(), &[1, 8]);
    }

    #[test]
    fn adaptive_coeffs_truncates() {
        let coeffs = gl_coeffs_adaptive(0.3, 1e-4, 64);
        assert!(!coeffs.is_empty());
        assert!(coeffs.len() <= 64);
        if coeffs.len() < 64 {
            assert!(coeffs.last().unwrap().abs() < 1e-4);
        } else {
            assert!(coeffs.last().unwrap().abs() >= 1e-4);
        }
    }

    #[test]
    fn fracdiff_1d_matches_nd() {
        let x = vec![0., 1., 2., 3.];
        let coeff = gl_coeffs(0.7, 4);
        let line = fracdiff_gl_1d_with_coeffs(&x, &coeff, Pad::Zero, Some(1.0));

        let arr = ArrayD::from_shape_vec(IxDyn(&[1, 4]), x.clone()).unwrap();
        let nd = fracdiff_gl_nd(&arr, 0.7, 1, 4, Pad::Zero, Some(1.0)).unwrap();

        assert_eq!(line.len(), nd.len());
        for (a, b) in line.iter().zip(nd.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn constant_pad_behaves() {
        let x = vec![1.0, 2.0];
        let coeff = gl_coeffs(0.4, 3);
        let y = fracdiff_gl_1d_with_coeffs(&x, &coeff, Pad::Constant(5.0), None);
        assert_eq!(y.len(), 2);
        // When padding with 5, the first element should only see the padded value
        // except for the zeroth coefficient which stays 1.
        assert!((y[0] - (coeff[0] * 1.0 + coeff[1] * 5.0 + coeff[2] * 5.0)).abs() < 1e-6);
    }

    #[test]
    fn reflect_pad_mirrors_left_edge() {
        let x = vec![10.0, 20.0, 30.0];
        let coeff = vec![1.0, 1.0, 1.0, 1.0];
        let y = fracdiff_gl_1d_with_coeffs(&x, &coeff, Pad::Reflect, Some(1.0));

        assert_eq!(y[0], 10.0 + 10.0 + 20.0 + 30.0);
        assert_eq!(y[1], 20.0 + 10.0 + 10.0 + 20.0);
    }
}
