// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use num_complex::Complex32;

/// Change-of-variable helper for Mellin integrals.
///
/// Given a positive interval `(a, b)` we operate in the logarithmic domain by
/// setting `x = exp(t)`. The integral bounds in the log domain then become
/// `(ln a, ln b)`, and the Mellin kernel simplifies to `exp(s * t)`.
#[inline]
fn map_range_to_log(range: (f32, f32)) -> (f32, f32) {
    let (a, b) = range;
    assert!(a.is_finite() && b.is_finite(), "integration bounds must be finite");
    assert!(a > 0.0 && b > 0.0, "integration bounds must be positive");
    assert!(a < b, "integration bounds must be ordered");
    (a.ln(), b.ln())
}

/// Numerically integrate a locally integrable function over `(0, ∞)` to obtain
/// its Mellin transform at `s`.
///
/// The caller supplies a truncation range `(a, b)` that captures the support of
/// the function (or the region where it is numerically relevant) together with
/// the number of steps to use for the composite trapezoidal rule in the log
/// domain.  The integral is approximated as
///
/// ```text
/// \int_a^b x^{s-1} f(x) dx = \int_{\ln a}^{\ln b} e^{s t} f(e^{t}) dt
/// ```
///
/// Using the logarithmic domain provides stable behaviour for rapidly decaying
/// functions and mirrors the Hilbert space setting for `L^2((0, \infty), dx/x)`.
pub fn mellin_transform<F>(
    f: F,
    s: Complex32,
    range: (f32, f32),
    steps: usize,
) -> Complex32
where
    F: Fn(f32) -> Complex32,
{
    assert!(steps >= 2, "at least two steps required for trapezoidal rule");
    let (log_a, log_b) = map_range_to_log(range);
    let h = (log_b - log_a) / steps as f32;

    let mut acc = Complex32::new(0.0, 0.0);
    for i in 0..=steps {
        let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
        let t = log_a + h * i as f32;
        let x = t.exp();
        let kernel = (s * Complex32::new(t, 0.0)).exp();
        let val = f(x);
        if !(val.re.is_finite() && val.im.is_finite()) {
            panic!("function produced non-finite value at x={}", x);
        }
        acc += kernel * val * weight;
    }

    acc * Complex32::new(h, 0.0)
}

/// Evaluate the Mellin transform from log-spaced samples.
///
/// The samples must correspond to the points `x_k = exp(log_start + k * log_step)`
/// for `k = 0..n-1`.  This layout makes the Mellin integral a standard
/// trapezoidal rule in the logarithmic domain, which aligns with the
/// Parseval identity on the Hilbert space `L^2((0, \infty), dx/x)`.
pub fn mellin_transform_log_samples(
    log_start: f32,
    log_step: f32,
    samples: &[Complex32],
    s: Complex32,
) -> Complex32 {
    assert!(log_step.is_finite() && log_step > 0.0, "log_step must be positive");
    assert!(!samples.is_empty(), "samples must not be empty");

    let mut acc = Complex32::new(0.0, 0.0);
    for (idx, &sample) in samples.iter().enumerate() {
        let weight = if idx == 0 || idx + 1 == samples.len() { 0.5 } else { 1.0 };
        let t = log_start + log_step * idx as f32;
        let kernel = (s * Complex32::new(t, 0.0)).exp();
        if !(sample.re.is_finite() && sample.im.is_finite()) {
            panic!("sample {} produced non-finite value", idx);
        }
        acc += kernel * sample * weight;
    }

    acc * Complex32::new(log_step, 0.0)
}

/// Inner product associated with the Hilbert space `L^2((0, \infty), dx/x)`.
///
/// This is the natural domain for the Mellin transform where it acts as a
/// unitary operator (Plancherel theorem).  Numerically we again employ the log
/// domain so that the weight `dx/x` becomes the standard Lebesgue measure in the
/// `t` variable.
pub fn mellin_l2_inner_product<F, G>(
    f: F,
    g: G,
    range: (f32, f32),
    steps: usize,
) -> Complex32
where
    F: Fn(f32) -> Complex32,
    G: Fn(f32) -> Complex32,
{
    assert!(steps >= 2, "at least two steps required for trapezoidal rule");
    let (log_a, log_b) = map_range_to_log(range);
    let h = (log_b - log_a) / steps as f32;

    let mut acc = Complex32::new(0.0, 0.0);
    for i in 0..=steps {
        let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
        let t = log_a + h * i as f32;
        let x = t.exp();
        let val = f(x).conj() * g(x);
        if !(val.re.is_finite() && val.im.is_finite()) {
            panic!("inner product integrand not finite at x={}", x);
        }
        acc += val * weight;
    }

    acc * Complex32::new(h, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use libm::tgammaf;

    fn exp_decay(x: f32) -> Complex32 {
        Complex32::new((-x).exp(), 0.0)
    }

    #[test]
    fn mellin_transform_matches_gamma_on_real_axis() {
        // For f(x) = exp(-x), the Mellin transform on the real axis is Gamma(s).
        let s = Complex32::new(2.5, 0.0);
        let approx = mellin_transform(exp_decay, s, (1e-4, 40.0), 16_384);
        let expected = tgammaf(2.5);
        assert!((approx.re - expected).abs() < 1e-3, "approx={} expected={}", approx, expected);
        assert!(approx.im.abs() < 1e-3);
    }

    #[test]
    fn log_samples_agree_with_function_integration() {
        let s = Complex32::new(1.3, 0.5);
        let log_start = -5.0f32;
        // sample from x in [exp(log_start), exp(log_start + log_step*(n-1))]
        let log_step = 0.005;
        let n = 2200;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = log_start + log_step * i as f32;
            let x = t.exp();
            samples.push(exp_decay(x));
        }
        let discrete = mellin_transform_log_samples(log_start, log_step, &samples, s);
        let continuous = mellin_transform(exp_decay, s, ((log_start).exp(), (log_start + log_step * (n - 1) as f32).exp()), n - 1);
        let diff = (discrete - continuous).norm();
        assert!(diff < 5e-3, "diff={} discrete={} continuous={}", diff, discrete, continuous);
    }

    #[test]
    fn hilbert_inner_product_positive_definite() {
        let range = (1e-4, 60.0);
        let ip = mellin_l2_inner_product(exp_decay, exp_decay, range, 8_192);
        assert!(ip.im.abs() < 1e-4);
        assert!(ip.re > 0.0);

        // Symmetry check: <f,g> = conj(<g,f>)
        let g = |x: f32| Complex32::new(x.powf(0.5) * (-x).exp(), 0.0);
        let fg = mellin_l2_inner_product(exp_decay, g, range, 8_192);
        let gf = mellin_l2_inner_product(g, exp_decay, range, 8_192);
        let diff = (fg - gf.conj()).norm();
        assert!(diff < 1e-4, "diff={} fg={} gf={} ", diff, fg, gf);
    }
}
