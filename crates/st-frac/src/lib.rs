// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

pub mod cosmology;
pub mod fft;
pub mod fractal_field;
pub mod mellin;
pub mod mellin_types;
#[cfg(feature = "wgpu")]
pub mod mellin_wgpu;
pub mod scale_stack;
pub mod zspace;

pub use fractal_field::{FractalFieldError, FractalFieldGenerator, FractalFieldResult};

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

#[derive(Debug, Error, PartialEq)]
pub enum FracErr {
    /// The requested axis does not exist in the input tensor.
    #[error("axis out of range")]
    Axis,
    /// A GL operator requires at least one coefficient.
    #[error("kernel_len must be > 0")]
    Kernel,
    /// Adaptive coefficient truncation requires a finite positive tolerance.
    #[error("tol must be > 0")]
    Tol,
    /// Fractional differentiation requires a finite positive order.
    #[error("alpha must be finite and > 0, got {alpha}")]
    Alpha { alpha: f32 },
    /// The sample spacing used by a fractional derivative was invalid.
    #[error("fractional step h must be finite and > 0, got {h}")]
    Step { h: f32 },
    /// A scalar operator parameter was not finite.
    #[error("{label} must be finite, got {value}")]
    NonFiniteParameter { label: &'static str, value: f32 },
    /// An input signal or coefficient contained a non-finite value.
    #[error("{label}[{index}] must be finite, got {value}")]
    NonFiniteSample {
        label: &'static str,
        index: usize,
        value: f32,
    },
    /// An intermediate cannot be represented as a finite f32 value.
    #[error("{label} cannot be represented as f32 (value={value})")]
    NumericOverflow { label: &'static str, value: f64 },
    /// A requested coefficient or output buffer cannot be allocated safely.
    #[error("{label} capacity {requested} cannot be allocated")]
    Capacity {
        label: &'static str,
        requested: usize,
    },
}

/// Padding behaviour for fractional convolution boundaries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pad {
    /// Use zeros for samples that fall outside the signal bounds.
    Zero,
    /// Mirror the signal across the boundary (a.k.a. Neumann reflection).
    Reflect,
    /// Clamp to the nearest edge sample (constant extension).
    Edge,
    /// Wrap around the signal length (periodic boundary condition).
    Wrap,
    /// Fill with the provided constant value.
    Constant(f32),
}

/// Validated powers of a GL sample spacing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlStepScale {
    h_alpha: f32,
    inverse_h_alpha: f32,
}

impl GlStepScale {
    /// Return `h^alpha`, the denominator used by a GL derivative kernel.
    pub fn h_alpha(self) -> f32 {
        self.h_alpha
    }

    /// Return `1 / h^alpha`, the multiplier used by CPU GL kernels.
    pub fn inverse_h_alpha(self) -> f32 {
        self.inverse_h_alpha
    }
}

fn validate_alpha(alpha: f32) -> Result<f64, FracErr> {
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(FracErr::Alpha { alpha });
    }
    Ok(f64::from(alpha))
}

fn validate_slice(label: &'static str, values: &[f32]) -> Result<(), FracErr> {
    if let Some((index, &value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(FracErr::NonFiniteSample {
            label,
            index,
            value,
        });
    }
    Ok(())
}

fn validate_pad(pad: Pad) -> Result<(), FracErr> {
    if let Pad::Constant(value) = pad {
        if !value.is_finite() {
            return Err(FracErr::NonFiniteParameter {
                label: "pad constant",
                value,
            });
        }
    }
    Ok(())
}

fn validate_scale(scale: Option<f32>) -> Result<f64, FracErr> {
    let scale = scale.unwrap_or(1.0);
    if !scale.is_finite() {
        return Err(FracErr::NonFiniteParameter {
            label: "fractional scale",
            value: scale,
        });
    }
    Ok(f64::from(scale))
}

fn checked_f32(label: &'static str, value: f64) -> Result<f32, FracErr> {
    if !value.is_finite() || value > f64::from(f32::MAX) || value < f64::from(f32::MIN) {
        return Err(FracErr::NumericOverflow { label, value });
    }
    Ok(value as f32)
}

fn zeroed_vec<T: Clone + Default>(label: &'static str, len: usize) -> Result<Vec<T>, FracErr> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| FracErr::Capacity {
            label,
            requested: len,
        })?;
    values.resize(len, T::default());
    Ok(values)
}

/// Validate `alpha` and sample spacing `h`, returning representable GL scales.
pub fn gl_step_scale(alpha: f32, h: f32) -> Result<GlStepScale, FracErr> {
    let alpha = validate_alpha(alpha)?;
    if !h.is_finite() || h <= 0.0 {
        return Err(FracErr::Step { h });
    }
    let h_alpha_f64 = f64::from(h).powf(alpha);
    let inverse_h_alpha_f64 = h_alpha_f64.recip();
    let h_alpha = checked_f32("h^alpha", h_alpha_f64)?;
    let inverse_h_alpha = checked_f32("1 / h^alpha", inverse_h_alpha_f64)?;
    if h_alpha == 0.0 || inverse_h_alpha == 0.0 {
        return Err(FracErr::NumericOverflow {
            label: "fractional step scale",
            value: if h_alpha == 0.0 {
                h_alpha_f64
            } else {
                inverse_h_alpha_f64
            },
        });
    }
    Ok(GlStepScale {
        h_alpha,
        inverse_h_alpha,
    })
}

/// Generate `len` Grünwald–Letnikov coefficients for a positive fractional exponent `alpha`.
pub fn gl_coeffs(alpha: f32, len: usize) -> Result<Vec<f32>, FracErr> {
    if len == 0 {
        return Err(FracErr::Kernel);
    }
    let alpha = validate_alpha(alpha)?;
    let mut c = zeroed_vec("GL coefficient", len)?;
    c[0] = 1.0;
    let mut previous = 1.0f64;
    for (index, coefficient) in c.iter_mut().enumerate().skip(1) {
        let order = index as f64;
        previous *= -((alpha - (order - 1.0)) / order);
        *coefficient = checked_f32("GL coefficient", previous)?;
    }
    Ok(c)
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

fn source_index_with_pad(idx: isize, len: usize, pad: Pad) -> Option<usize> {
    if len == 0 {
        return None;
    }
    if (0..len as isize).contains(&idx) {
        return Some(idx as usize);
    }
    match pad {
        Pad::Zero | Pad::Constant(_) => None,
        Pad::Reflect => Some(reflected_index(idx, len)),
        Pad::Edge => Some(clamped_edge_index(idx, len)),
        Pad::Wrap => Some(wrap_index(idx, len)),
    }
}

fn conv1d_gl_line(
    x: &[f32],
    y: &mut [f32],
    coeff: &[f32],
    pad: Pad,
    scale: f64,
) -> Result<(), FracErr> {
    validate_slice("fractional input", x)?;
    validate_slice("GL coefficient", coeff)?;
    validate_pad(pad)?;
    if coeff.is_empty() {
        return Err(FracErr::Kernel);
    }
    for (i, out) in y.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (k, &c) in coeff.iter().enumerate() {
            let idx = i as isize - k as isize;
            acc += f64::from(c) * f64::from(sample_with_pad(x, idx, pad));
        }
        *out = checked_f32("fractional output", scale * acc)?;
    }
    Ok(())
}

fn vjp1d_gl_line(
    gy: &[f32],
    gx: &mut [f32],
    coeff: &[f32],
    pad: Pad,
    scale: f64,
) -> Result<(), FracErr> {
    validate_slice("fractional output gradient", gy)?;
    validate_slice("GL coefficient", coeff)?;
    validate_pad(pad)?;
    if coeff.is_empty() {
        return Err(FracErr::Kernel);
    }
    debug_assert_eq!(gy.len(), gx.len());

    let mut accumulators = zeroed_vec("fractional VJP accumulator", gx.len())?;
    for (output_index, &gradient) in gy.iter().enumerate() {
        for (lag, &coefficient) in coeff.iter().enumerate() {
            let source_index = output_index as isize - lag as isize;
            if let Some(source_index) = source_index_with_pad(source_index, gx.len(), pad) {
                accumulators[source_index] += scale * f64::from(coefficient) * f64::from(gradient);
            }
        }
    }
    for (output, value) in gx.iter_mut().zip(accumulators) {
        *output = checked_f32("fractional input gradient", value)?;
    }
    Ok(())
}

/// Generate Grünwald–Letnikov coefficients until their magnitude drops below `tol`
/// or until `max_len` coefficients have been produced.
fn gl_coeffs_adaptive_forward(alpha: f32, tol: f32, max_len: usize) -> Result<Vec<f32>, FracErr> {
    gl_coeffs_adaptive_impl(alpha, tol, max_len)
}

#[inline]
fn gl_coeffs_adaptive_impl(alpha: f32, tol: f32, max_len: usize) -> Result<Vec<f32>, FracErr> {
    if max_len == 0 {
        return Err(FracErr::Kernel);
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(FracErr::Tol);
    }
    let alpha = validate_alpha(alpha)?;

    let mut coeffs = Vec::new();
    coeffs
        .try_reserve_exact(max_len)
        .map_err(|_| FracErr::Capacity {
            label: "adaptive GL coefficient",
            requested: max_len,
        })?;
    let mut previous = 1.0f64;
    coeffs.push(1.0);

    for index in 1..max_len {
        let order = index as f64;
        previous *= -((alpha - (order - 1.0)) / order);
        coeffs.push(checked_f32("adaptive GL coefficient", previous)?);
        if previous.abs() < f64::from(tol) {
            break;
        }
    }

    Ok(coeffs)
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
    validate_pad(pad)?;
    let coeff = gl_coeffs(alpha, kernel_len)?;
    fracdiff_gl_1d_with_coeffs_impl(x, &coeff, pad, scale)
}

/// Apply a fractional difference along a 1-D slice using precomputed coefficients.
fn fracdiff_gl_1d_with_coeffs_forward(
    x: &[f32],
    coeff: &[f32],
    pad: Pad,
    scale: Option<f32>,
) -> Result<Vec<f32>, FracErr> {
    fracdiff_gl_1d_with_coeffs_impl(x, coeff, pad, scale)
}

#[inline]
fn fracdiff_gl_1d_with_coeffs_impl(
    x: &[f32],
    coeff: &[f32],
    pad: Pad,
    scale: Option<f32>,
) -> Result<Vec<f32>, FracErr> {
    let scale = validate_scale(scale)?;
    let mut y = zeroed_vec("fractional output", x.len())?;
    conv1d_gl_line(x, &mut y, coeff, pad, scale)?;
    Ok(y)
}

/// Generate Grünwald–Letnikov coefficients until their magnitude drops below `tol`
/// or until `max_len` coefficients have been produced.
pub fn gl_coeffs_adaptive(alpha: f32, tol: f32, max_len: usize) -> Result<Vec<f32>, FracErr> {
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

/// Apply a fractional difference with caller-supplied GL coefficients.
pub fn fracdiff_gl_1d_with_coeffs(
    x: &[f32],
    coeff: &[f32],
    pad: Pad,
    scale: Option<f32>,
) -> Result<Vec<f32>, FracErr> {
    fracdiff_gl_1d_with_coeffs_forward(x, coeff, pad, scale)
}

/// Apply the exact vector-Jacobian product of [`fracdiff_gl_1d`].
///
/// Boundary samples from `Reflect`, `Edge`, and `Wrap` padding are accumulated
/// back into the source indices that produced them. A constant pad contributes
/// no gradient because it is independent of the input signal.
pub fn fracdiff_gl_1d_vjp(
    gy: &[f32],
    alpha: f32,
    kernel_len: usize,
    pad: Pad,
    scale: Option<f32>,
) -> Result<Vec<f32>, FracErr> {
    let coeff = gl_coeffs(alpha, kernel_len)?;
    fracdiff_gl_1d_vjp_with_coeffs(gy, &coeff, pad, scale)
}

/// Apply the exact vector-Jacobian product of [`fracdiff_gl_1d_with_coeffs`].
pub fn fracdiff_gl_1d_vjp_with_coeffs(
    gy: &[f32],
    coeff: &[f32],
    pad: Pad,
    scale: Option<f32>,
) -> Result<Vec<f32>, FracErr> {
    let scale = validate_scale(scale)?;
    let mut gx = zeroed_vec("fractional input gradient", gy.len())?;
    vjp1d_gl_line(gy, &mut gx, coeff, pad, scale)?;
    Ok(gx)
}

/// Apply a GL fractional difference along one axis of an n-dimensional array.
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
    validate_pad(pad)?;
    let mut y = x.clone();
    let coeff = gl_coeffs(alpha, kernel_len)?;
    let scale = validate_scale(scale)?;
    let axis_len = x.len_of(Axis(axis));
    let mut scratch_src = zeroed_vec("fractional source scratch", axis_len)?;
    let mut scratch_dst = zeroed_vec("fractional destination scratch", axis_len)?;
    let ax = Axis(axis);
    let mut yv = y.view_mut();
    let dst_lanes = yv.lanes_mut(ax);
    let xv = x.view();
    let src_lanes = xv.lanes(ax);

    for (mut dst, src) in dst_lanes.into_iter().zip(src_lanes) {
        match (src.as_slice(), dst.as_slice_mut()) {
            (Some(src_slice), Some(dst_slice)) => {
                conv1d_gl_line(src_slice, dst_slice, &coeff, pad, scale)?;
            }
            _ => {
                for (slot, &value) in scratch_src.iter_mut().zip(src.iter()) {
                    *slot = value;
                }
                conv1d_gl_line(&scratch_src, &mut scratch_dst, &coeff, pad, scale)?;
                for (dst_value, &value) in dst.iter_mut().zip(scratch_dst.iter()) {
                    *dst_value = value;
                }
            }
        }
    }
    Ok(y)
}

/// Apply the exact VJP of [`fracdiff_gl_nd`] along the selected axis.
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
    validate_pad(pad)?;
    let mut gx = gy.clone();
    let coeff = gl_coeffs(alpha, kernel_len)?;
    let scale = validate_scale(scale)?;
    let axis_len = gy.len_of(Axis(axis));
    let mut scratch_src = zeroed_vec("fractional VJP source scratch", axis_len)?;
    let mut scratch_dst = zeroed_vec("fractional VJP destination scratch", axis_len)?;
    let ax = Axis(axis);
    let mut gxv = gx.view_mut();
    let dst_lanes = gxv.lanes_mut(ax);
    let gyv = gy.view();
    let src_lanes = gyv.lanes(ax);

    for (mut dst, src) in dst_lanes.into_iter().zip(src_lanes) {
        match (src.as_slice(), dst.as_slice_mut()) {
            (Some(src_slice), Some(dst_slice)) => {
                vjp1d_gl_line(src_slice, dst_slice, &coeff, pad, scale)?;
            }
            _ => {
                for (slot, &value) in scratch_src.iter_mut().zip(src.iter()) {
                    *slot = value;
                }
                vjp1d_gl_line(&scratch_src, &mut scratch_dst, &coeff, pad, scale)?;
                for (dst_value, &value) in dst.iter_mut().zip(scratch_dst.iter()) {
                    *dst_value = value;
                }
            }
        }
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
        let coeffs = gl_coeffs_adaptive(0.3, 1e-4, 64).unwrap();
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
        let coeff = gl_coeffs(0.7, 4).unwrap();
        let line = fracdiff_gl_1d_with_coeffs(&x, &coeff, Pad::Zero, Some(1.0)).unwrap();

        let arr = ArrayD::from_shape_vec(IxDyn(&[1, 4]), x.clone()).unwrap();
        let nd = fracdiff_gl_nd(&arr, 0.7, 1, 4, Pad::Zero, Some(1.0)).unwrap();

        assert_eq!(line.len(), nd.len());
        for (a, b) in line.iter().zip(nd.iter()) {
            assert!((*a - *b).abs() < 1e-6f32);
        }
    }

    #[test]
    fn fracdiff_nd_handles_strided_lanes() {
        let x = ArrayD::from_shape_vec(
            IxDyn(&[3, 4]),
            vec![
                0.0, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 7.0, //
                8.0, 9.0, 10.0, 11.0,
            ],
        )
        .unwrap();

        // Axis 0 lanes are strided (columns) in a standard row-major layout.
        let x_view = x.view();
        let first_lane = x_view.lanes(Axis(0)).into_iter().next().unwrap();
        assert!(first_lane.as_slice().is_none());

        let alpha = 0.35;
        let kernel_len = 4;
        let pad = Pad::Zero;
        let scale = Some(0.8);

        let y = fracdiff_gl_nd(&x, alpha, 0, kernel_len, pad, scale).unwrap();
        let coeff = gl_coeffs(alpha, kernel_len).unwrap();

        for col in 0..4 {
            let lane: Vec<f32> = (0..3).map(|row| x[[row, col]]).collect();
            let expected = fracdiff_gl_1d_with_coeffs(&lane, &coeff, pad, scale).unwrap();
            for row in 0..3 {
                assert!((y[[row, col]] - expected[row]).abs() < 1e-6f32);
            }
        }
    }

    #[test]
    fn fracdiff_nd_backward_handles_strided_lanes() {
        let gy = ArrayD::from_shape_vec(
            IxDyn(&[3, 4]),
            vec![
                0.0, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 7.0, //
                8.0, 9.0, 10.0, 11.0,
            ],
        )
        .unwrap();

        // Axis 0 lanes are strided (columns) in a standard row-major layout.
        let gy_view = gy.view();
        let first_lane = gy_view.lanes(Axis(0)).into_iter().next().unwrap();
        assert!(first_lane.as_slice().is_none());

        let alpha = 0.6;
        let kernel_len = 3;
        let pad = Pad::Edge;
        let scale = Some(0.25);

        let gx = fracdiff_gl_nd_backward(&gy, alpha, 0, kernel_len, pad, scale).unwrap();
        let coeff = gl_coeffs(alpha, kernel_len).unwrap();

        for col in 0..4 {
            let lane: Vec<f32> = (0..3).map(|row| gy[[row, col]]).collect();
            let expected = fracdiff_gl_1d_vjp_with_coeffs(&lane, &coeff, pad, scale).unwrap();
            for row in 0..3 {
                assert!((gx[[row, col]] - expected[row]).abs() < 1e-6f32);
            }
        }
    }

    #[test]
    fn vjp_is_the_exact_adjoint_for_every_padding_mode() {
        let x = [0.25, -1.5, 2.0, 0.75];
        let gy = [1.25, -0.5, 0.75, 2.0];
        let zero = [0.0; 4];
        let coeff = gl_coeffs(0.6, 6).unwrap();
        let scale = Some(0.7);
        let pads = [
            Pad::Zero,
            Pad::Reflect,
            Pad::Edge,
            Pad::Wrap,
            Pad::Constant(2.5),
        ];

        for pad in pads {
            let y = fracdiff_gl_1d_with_coeffs(&x, &coeff, pad, scale).unwrap();
            let baseline = fracdiff_gl_1d_with_coeffs(&zero, &coeff, pad, scale).unwrap();
            let gx = fracdiff_gl_1d_vjp_with_coeffs(&gy, &coeff, pad, scale).unwrap();
            let lhs: f64 = y
                .iter()
                .zip(&baseline)
                .zip(&gy)
                .map(|((&value, &offset), &gradient)| {
                    f64::from(value - offset) * f64::from(gradient)
                })
                .sum();
            let rhs: f64 = x
                .iter()
                .zip(&gx)
                .map(|(&value, &gradient)| f64::from(value) * f64::from(gradient))
                .sum();
            assert!(
                (lhs - rhs).abs() < 2e-6,
                "pad={pad:?}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[test]
    fn first_order_gl_matches_the_backward_difference() {
        let coeff = gl_coeffs(1.0, 4).unwrap();
        assert_eq!(coeff, vec![1.0, -1.0, 0.0, 0.0]);
        let output = fracdiff_gl_1d(&[1.0, 2.0, 4.0], 1.0, 4, Pad::Zero, None).unwrap();
        assert_eq!(output, vec![1.0, 1.0, 2.0]);
    }

    #[test]
    fn gl_contract_rejects_invalid_parameters_and_samples() {
        assert!(matches!(gl_coeffs(0.0, 2), Err(FracErr::Alpha { .. })));
        assert!(matches!(gl_coeffs(f32::NAN, 2), Err(FracErr::Alpha { .. })));
        assert!(matches!(gl_step_scale(0.5, 0.0), Err(FracErr::Step { .. })));
        assert!(matches!(
            fracdiff_gl_1d(&[1.0, f32::NAN], 0.5, 2, Pad::Zero, None),
            Err(FracErr::NonFiniteSample { .. })
        ));
        assert!(matches!(
            fracdiff_gl_1d(&[1.0], 0.5, 2, Pad::Constant(f32::INFINITY), None),
            Err(FracErr::NonFiniteParameter { .. })
        ));
        assert!(matches!(
            fracdiff_gl_1d_with_coeffs(&[1.0], &[], Pad::Zero, None),
            Err(FracErr::Kernel)
        ));
        assert!(matches!(
            fracdiff_gl_1d_with_coeffs(&[1.0], &[f32::INFINITY], Pad::Zero, None),
            Err(FracErr::NonFiniteSample { .. })
        ));
        assert!(matches!(
            fracdiff_gl_1d(&[1.0], 0.5, 2, Pad::Zero, Some(f32::NAN)),
            Err(FracErr::NonFiniteParameter { .. })
        ));
    }

    #[test]
    fn gl_step_scale_is_shared_and_checked() {
        let scale = gl_step_scale(0.5, 0.25).unwrap();
        assert_eq!(scale.h_alpha(), 0.5);
        assert_eq!(scale.inverse_h_alpha(), 2.0);
        assert!(matches!(
            gl_step_scale(f32::MAX, f32::MIN_POSITIVE),
            Err(FracErr::NumericOverflow { .. })
        ));
    }

    #[test]
    fn finite_inputs_cannot_silently_overflow_to_zero() {
        let error = fracdiff_gl_1d_with_coeffs(&[f32::MAX], &[f32::MAX], Pad::Zero, None);
        assert!(matches!(error, Err(FracErr::NumericOverflow { .. })));
    }

    #[test]
    fn coefficient_capacity_overflow_is_an_error() {
        assert!(matches!(
            gl_coeffs(0.5, usize::MAX),
            Err(FracErr::Capacity { .. })
        ));
        assert!(matches!(
            gl_coeffs_adaptive(0.5, 1e-6, usize::MAX),
            Err(FracErr::Capacity { .. })
        ));
    }

    #[test]
    fn constant_pad_behaves() {
        let x = vec![1.0, 2.0];
        let coeff = gl_coeffs(0.4, 3).unwrap();
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
