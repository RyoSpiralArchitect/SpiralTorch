// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Lightweight FFT helpers tailored for fractional calculus experiments.
//!
//! The design favours clarity over absolute peak performance so the routines can
//! be ported to GPU kernels or auto-generated SpiralK programs.  The module
//! offers primitive radix-2 and radix-4 butterflies and an in-place Cooley–Tukey
//! driver that selects the best mix for the signal length.  All code is `no_std`
//! compatible (alloc-only) and keeps allocations outside of the hot paths.

use core::f32::consts::PI;
use core::fmt;

/// Minimal complex number implementation to avoid pulling in external crates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    #[inline]
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn scale(self, rhs: f32) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }

    #[inline]
    #[allow(
        clippy::should_implement_trait,
        reason = "Minimal complex helper keeps explicit method names for clarity"
    )]
    pub fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }

    #[inline]
    #[allow(
        clippy::should_implement_trait,
        reason = "Minimal complex helper keeps explicit method names for clarity"
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }

    #[inline]
    #[allow(
        clippy::should_implement_trait,
        reason = "Minimal complex helper keeps explicit method names for clarity"
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
}

/// Radix-2 butterfly. Returns `(top, bottom)`.
#[inline]
pub fn radix2(a: Complex32, b: Complex32, twiddle: Complex32) -> (Complex32, Complex32) {
    let t = b.mul(twiddle);
    (a.add(t), a.sub(t))
}

/// Radix-4 butterfly returning `[y0, y1, y2, y3]`.
#[inline]
pub fn radix4(values: [Complex32; 4], twiddles: [Complex32; 3]) -> [Complex32; 4] {
    let a0 = values[0];
    let a1 = values[1].mul(twiddles[0]);
    let a2 = values[2].mul(twiddles[1]);
    let a3 = values[3].mul(twiddles[2]);

    let t0 = a0.add(a2);
    let t1 = a0.sub(a2);
    let t2 = a1.add(a3);
    let t3 = a1.sub(a3);

    [
        t0.add(t2),
        Complex32::new(t1.re - t3.im, t1.im + t3.re),
        t0.sub(t2),
        Complex32::new(t1.re + t3.im, t1.im - t3.re),
    ]
}

/// Error raised when the FFT driver receives unsupported parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftError {
    /// Signal length was zero.
    Empty,
    /// Length was not a power of two.
    NonPowerOfTwo,
}

impl fmt::Display for FftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FftError::Empty => f.write_str("FFT signal cannot be empty"),
            FftError::NonPowerOfTwo => {
                f.write_str("FFT length must be a power of two for radix-2/4 pipeline")
            }
        }
    }
}

/// In-place iterative FFT using a radix-4 kernel whenever possible and falling
/// back to radix-2 stages for the tail factors.  The function accepts forward
/// (`inverse = false`) and inverse (`inverse = true`) transforms.
pub fn fft_inplace(signal: &mut [Complex32], inverse: bool) -> Result<(), FftError> {
    let n = signal.len();
    if n == 0 {
        return Err(FftError::Empty);
    }
    if !n.is_power_of_two() {
        return Err(FftError::NonPowerOfTwo);
    }

    bit_reverse_permute(signal);
    let mut m = 1;
    while m < n {
        if m * 4 <= n {
            radix4_stage(signal, m, inverse);
            m *= 4;
        } else {
            radix2_stage(signal, m, inverse);
            m *= 2;
        }
    }

    if inverse {
        let scale = 1.0 / n as f32;
        for v in signal.iter_mut() {
            v.re *= scale;
            v.im *= scale;
        }
    }
    Ok(())
}

fn radix2_stage(buf: &mut [Complex32], half_stride: usize, inverse: bool) {
    let step = half_stride * 2;
    let sign = if inverse { 1.0 } else { -1.0 };
    for k in (0..buf.len()).step_by(step) {
        for j in 0..half_stride {
            let tw = twiddle(j, step, sign);
            let a = buf[k + j];
            let b = buf[k + j + half_stride];
            let (top, bottom) = radix2(a, b, tw);
            buf[k + j] = top;
            buf[k + j + half_stride] = bottom;
        }
    }
}

fn radix4_stage(buf: &mut [Complex32], quarter_stride: usize, inverse: bool) {
    let step = quarter_stride * 4;
    let sign = if inverse { 1.0 } else { -1.0 };
    for k in (0..buf.len()).step_by(step) {
        for j in 0..quarter_stride {
            let base = k + j;
            let vals = [
                buf[base],
                buf[base + quarter_stride],
                buf[base + 2 * quarter_stride],
                buf[base + 3 * quarter_stride],
            ];
            let tw = [
                twiddle(j, step, sign),
                twiddle(2 * j, step, sign),
                twiddle(3 * j, step, sign),
            ];
            let out = radix4(vals, tw);
            buf[base] = out[0];
            buf[base + quarter_stride] = out[1];
            buf[base + 2 * quarter_stride] = out[2];
            buf[base + 3 * quarter_stride] = out[3];
        }
    }
}

fn twiddle(index: usize, size: usize, sign: f32) -> Complex32 {
    let angle = 2.0 * PI * index as f32 / size as f32 * sign;
    Complex32::new(angle.cos(), angle.sin())
}

fn bit_reverse_permute(buf: &mut [Complex32]) {
    let n = buf.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let rev = i.reverse_bits() >> (usize::BITS - bits);
        if i < rev {
            buf.swap(i, rev);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn impulse(n: usize) -> Vec<Complex32> {
        let mut v = vec![Complex32::default(); n];
        v[0] = Complex32::new(1.0, 0.0);
        v
    }

    #[test]
    fn fft_roundtrip_radix4() {
        let mut data = impulse(16);
        fft_inplace(&mut data, false).unwrap();
        fft_inplace(&mut data, true).unwrap();
        for (i, v) in data.iter().enumerate() {
            let expected = if i == 0 { 1.0 } else { 0.0 };
            assert!((v.re - expected).abs() < 1e-5, "index {i}");
            assert!(v.im.abs() < 1e-5);
        }
    }

    #[test]
    fn fft_mixed_radix2_radix4() {
        let mut data = impulse(8);
        fft_inplace(&mut data, false).unwrap();
        // Energy must be conserved.
        let energy: f32 = data.iter().map(|c| c.re * c.re + c.im * c.im).sum();
        assert!((energy - 8.0).abs() < 1e-4);
    }
}
