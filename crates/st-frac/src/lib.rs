// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

pub mod fft;
pub mod mellin;
pub mod mellin_types;
#[cfg(feature = "wgpu")]
pub mod mellin_wgpu;
pub mod zspace;

use ndarray::{ArrayD, Axis};
use thiserror::Error;

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
    /// Clamp to the edge value (a.k.a. Dirichlet boundary condition).
    Edge,
    /// Wrap around the signal length (periodic boundary condition).
    Wrap,
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
        c[k] = -(prev * (num / k as f32));
    }
    c
}

fn wrap_index(idx: isize, len: usize) -> usize {
    debug_assert!(len > 0);
    let len = len as isize;
    idx.rem_euclid(len) as usize
}

/// Reflect `idx` into the valid `[0, len)` range using the same "edge repeating"
/// semantics as the previous iterative implementation but without potentially
/// overflowing when `idx` is at the limits of `isize`.
fn reflected_index(idx: isize, len: usize) -> usize {
    debug_assert!(len > 0);
    if len == 1 {
        return 0;
    }

    let len_i128 = len as i128;
    let period = len_i128 * 2;
    let mut m = idx as i128 % period;
    if m < 0 {
        m += period;
    }
    if m >= len_i128 {
        m = period - m - 1;
    }

    m as usize
}

fn clamped_edge_index(idx: isize, len: usize) -> usize {
    debug_assert!(len > 0);
    if idx < 0 {
        0
    } else if idx as usize >= len {
        len - 1
    } else {
        idx as usize
    }
}

#[inline]
fn sample_with_pad(x: &[f32], idx: isize, pad: Pad) -> f32 {
    let len = x.len() as isize;

    if len == 0 {
        return match pad {
            Pad::Zero | Pad::Reflect | Pad::Edge | Pad::Wrap => 0.0,
            Pad::Constant(v) => v,
        };
    }

    if (0..len).contains(&idx) {
        return x[idx as usize];
    }

    match pad {
        Pad::Zero => 0.0,
        Pad::Constant(v) => v,
        Pad::Reflect => x[reflected_index(idx, x.len())],
        Pad::Edge => x[clamped_edge_index(idx, x.len())],
        Pad::Wrap => x[wrap_index(idx, x.len())],
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
fn gl_coeffs_adaptive_forward(alpha: f32, tol: f32, max_len: usize) -> Vec<f32> {
    gl_coeffs_adaptive_impl(alpha, tol, max_len)
}

#[inline]
fn gl_coeffs_adaptive_impl(alpha: f32, tol: f32, max_len: usize) -> Vec<f32> {
    assert!(max_len > 0);
    assert!(tol > 0.0);

    let mut coeffs = Vec::with_capacity(max_len);
    let mut prev = 1.0f32;
    coeffs.push(prev);

    for k in 1..max_len {
        let num = alpha - (k as f32 - 1.0);
        prev *= -(num / k as f32);
        coeffs.push(prev);
        if prev.abs() < tol {
            break;
        }
    }

    coeffs
}

/// Apply a fractional difference along a 1-D slice.
fn fracdiff_gl_1d_forward(
    x: &[f32],
    alpha: f32,
    kernel_len: usize,
    pad: Pad,
    scale: Option<f32>,
) -> Result<Vec<f32>, FracErr> {
    fracdiff_gl_1d_impl(x, alpha, kernel_len, pad, scale)
}

#[inline]
fn fracdiff_gl_1d_impl(
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
    Ok(fracdiff_gl_1d_with_coeffs_impl(x, &coeff, pad, scale))
}

/// Apply a fractional difference along a 1-D slice using precomputed coefficients.
fn fracdiff_gl_1d_with_coeffs_forward(
    x: &[f32],
    coeff: &[f32],
    pad: Pad,
    scale: Option<f32>,
) -> Vec<f32> {
    fracdiff_gl_1d_with_coeffs_impl(x, coeff, pad, scale)
}

#[inline]
fn fracdiff_gl_1d_with_coeffs_impl(
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
    gl_coeffs_adaptive_forward(alpha, tol, max_len)
}

/// Apply a fractional difference along a 1-D slice.
pub fn fracdiff_gl_1d(
    x: &[f32],
    alpha: f32,
    kernel_len: usize,
    pad: Pad,
    scale: Option<f32>,
) -> Result<Vec<f32>, FracErr> {
    fracdiff_gl_1d_forward(x, alpha, kernel_len, pad, scale)
}

pub fn fracdiff_gl_1d_with_coeffs(
    x: &[f32],
    coeff: &[f32],
    pad: Pad,
    scale: Option<f32>,
) -> Result<Vec<f32>, FracErr> {
    Ok(fracdiff_gl_1d_with_coeffs_forward(x, coeff, pad, scale))
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
        let line = fracdiff_gl_1d_with_coeffs(&x, &coeff, Pad::Zero, Some(1.0)).unwrap();

        let arr = ArrayD::from_shape_vec(IxDyn(&[1, 4]), x.clone()).unwrap();
        let nd = fracdiff_gl_nd(&arr, 0.7, 1, 4, Pad::Zero, Some(1.0)).unwrap();

        assert_eq!(line.len(), nd.len());
        for (a, b) in line.iter().zip(nd.iter()) {
            assert!((*a - *b).abs() < 1e-6f32);
        }
    }

    #[test]
    fn constant_pad_behaves() {
        let x = vec![1.0, 2.0];
        let coeff = gl_coeffs(0.4, 3);
        let y = fracdiff_gl_1d_with_coeffs(&x, &coeff, Pad::Constant(5.0), None).unwrap();
        assert_eq!(y.len(), 2);
        // When padding with 5, the first element should only see the padded value
        // except for the zeroth coefficient which stays 1.
        assert!((y[0] - (coeff[0] * 1.0 + coeff[1] * 5.0 + coeff[2] * 5.0)).abs() < 1e-6f32);
    }

    #[test]
    fn reflect_pad_mirrors_left_edge() {
        let x = vec![10.0, 20.0, 30.0];
        let coeff = vec![1.0, 1.0, 1.0, 1.0];
        let y = fracdiff_gl_1d_with_coeffs(&x, &coeff, Pad::Reflect, Some(1.0)).unwrap();

        assert_eq!(y[0], 10.0 + 10.0 + 20.0 + 30.0);
        assert_eq!(y[1], 20.0 + 10.0 + 10.0 + 20.0);
    }

    #[test]
    fn constant_pad_applies_to_right_edge() {
        let x = vec![1.0, 2.0];
        let coeff = vec![1.0, 0.5, 0.25];
        let y = fracdiff_gl_1d_with_coeffs(&x, &coeff, Pad::Constant(5.0), Some(1.0)).unwrap();

        assert_eq!(y[1], 2.0 * 1.0 + 1.0 * 0.5 + 5.0 * 0.25);
    }

    #[test]
    fn edge_pad_clamps_to_boundary() {
        let x = vec![3.0, 7.0, 11.0];
        let coeff = vec![1.0, 1.0, 1.0];
        let y = fracdiff_gl_1d_with_coeffs(&x, &coeff, Pad::Edge, Some(1.0)).unwrap();

        assert_eq!(y[0], 3.0 + 3.0 + 3.0);
    }

    #[test]
    fn wrap_pad_cycles_through_signal() {
        let x = vec![2.0, 4.0, 6.0];
        let coeff = vec![1.0, 1.0, 1.0, 1.0];
        let y = fracdiff_gl_1d_with_coeffs(&x, &coeff, Pad::Wrap, Some(1.0)).unwrap();

        assert_eq!(y[0], 2.0 + 6.0 + 4.0 + 2.0);
        assert_eq!(y[1], 4.0 + 2.0 + 6.0 + 4.0);
    }

    #[test]
    fn reflected_index_wraps_right_edge() {
        assert_eq!(super::reflected_index(3, 3), 2);
        assert_eq!(super::reflected_index(4, 3), 1);
    }

    #[test]
    fn reflected_index_handles_far_offsets() {
        assert_eq!(super::reflected_index(-7, 4), 1);
        assert_eq!(super::reflected_index(9, 4), 1);
        assert_eq!(super::reflected_index(-1, 1), 0);
    }

    #[test]
    fn wrap_index_handles_large_displacements() {
        assert_eq!(super::wrap_index(10, 3), 1);
        assert_eq!(super::wrap_index(-8, 5), 2);
    }

    #[test]
    fn reflected_index_matches_naive_small_ranges() {
        fn naive(idx: isize, len: usize) -> usize {
            if len == 1 {
                return 0;
            }
            let len_i128 = len as i128;
            let mut idx = idx as i128;
            while idx < 0 || idx >= len_i128 {
                if idx < 0 {
                    idx = -idx - 1;
                } else {
                    idx = len_i128 - (idx - len_i128) - 1;
                }
            }
            idx as usize
        }

        for len in 1..6 {
            for idx in -32..=32 {
                assert_eq!(super::reflected_index(idx, len), naive(idx, len));
            }
        }
    }

    #[test]
    fn sample_with_pad_reflect_handles_extreme_indices() {
        let x = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(super::sample_with_pad(&x, isize::MIN, Pad::Reflect), 10.0);
        assert_eq!(
            super::sample_with_pad(&x, isize::MAX - 1, Pad::Reflect),
            20.0
        );
    }

    #[test]
    fn clamped_edge_index_clamps_bounds() {
        assert_eq!(super::clamped_edge_index(-3, 4), 0);
        assert_eq!(super::clamped_edge_index(8, 4), 3);
        assert_eq!(super::clamped_edge_index(2, 4), 2);
    }

    #[test]
    fn sample_with_pad_handles_right_constant() {
        let x = [1.0, 2.0];
        assert_eq!(super::sample_with_pad(&x, 2, Pad::Constant(5.0)), 5.0);
    }

    #[test]
    fn sample_with_pad_handles_edge() {
        let x = [1.0, 2.0, 3.0];
        assert_eq!(super::sample_with_pad(&x, -1, Pad::Edge), 1.0);
        assert_eq!(super::sample_with_pad(&x, 5, Pad::Edge), 3.0);
    }

    #[test]
    fn sample_with_pad_handles_wrap() {
        let x = [1.0, 2.0, 3.0];
        assert_eq!(super::sample_with_pad(&x, -1, Pad::Wrap), 3.0);
        assert_eq!(super::sample_with_pad(&x, 4, Pad::Wrap), 2.0);
        assert_eq!(super::sample_with_pad(&x, 10, Pad::Wrap), 2.0);
    }
}
