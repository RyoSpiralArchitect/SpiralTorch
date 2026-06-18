// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Minimal helpers for projecting slow-roll inflation backgrounds onto the
//! logarithmic Z-space lattice described in the design notes.
//!
//! The module focuses on three primitives that mirror the implementation plan
//! outlined in the user guide:
//! - [`LogLattice`] generates uniformly spaced samples of the e-folding time
//!   \(\tau = \ln a\) together with trapezoidal weights and optional Hann
//!   tapers so that Mellin / Z transforms remain numerically stable on a finite
//!   window.
//! - [`z_transform`] evaluates a weighted Z transform using Horner's method,
//!   ensuring well-conditioned accumulation even when the evaluation point is
//!   close to a pole.
//! - [`assemble_primordial_spectrum`] combines the transformed background
//!   channels into the slow-roll estimate of the scalar power spectrum
//!   \(\mathcal{P}_\zeta(k) = H^2/(8\pi^2 \epsilon M_{\rm Pl}^2)\).
//!
//! These pieces are intentionally minimal—they provide the scaffolding required
//! to run the Z-space workflow without dictating higher-level orchestration.

use std::f64::consts::PI;

use num_complex::Complex64;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

/// Bundles the Mellin/Z-plane evaluations of the slow-roll background
/// together with the assembled primordial power spectrum.
#[derive(Clone, Debug, PartialEq)]
pub struct PrimordialProjection {
    /// Starting value of the logarithmic lattice (\(\tau_0\)).
    pub log_start: f64,
    /// Step size of the logarithmic lattice (\(\Delta\tau\)).
    pub log_step: f64,
    /// Number of samples used when constructing the lattice in the time domain.
    pub lattice_len: usize,
    /// Mellin abscissa evaluation points.
    pub s_values: Vec<Complex64>,
    /// Corresponding points on the Z plane \(z = e^{s\Delta\tau}\).
    pub z_points: Vec<Complex64>,
    /// Z-transform of the Hubble parameter.
    pub h_z: Vec<Complex64>,
    /// Z-transform of the first slow-roll parameter.
    pub epsilon_z: Vec<Complex64>,
    /// Assembled primordial curvature power spectrum values.
    pub spectrum: Vec<f64>,
}

impl PrimordialProjection {
    /// Creates a new projection from the supplied Mellin/Z-plane channels.
    ///
    /// The Z points and background channels must already be evaluated at the
    /// same set of Mellin abscissae.  The primordial spectrum is assembled on
    /// creation so that downstream consumers can access the full bundle without
    /// recomputing it.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        log_start: f64,
        log_step: f64,
        lattice_len: usize,
        s_values: Vec<Complex64>,
        z_points: Vec<Complex64>,
        h_z: Vec<Complex64>,
        epsilon_z: Vec<Complex64>,
        planck_mass: f64,
    ) -> Self {
        assert!(log_step.is_sign_positive(), "log_step must be positive");
        assert!(log_start.is_finite(), "log_start must be finite");
        assert!(lattice_len >= 2, "lattice length must be at least 2");
        let len = s_values.len();
        assert_eq!(len, z_points.len(), "s_values/z_points length mismatch");
        assert_eq!(len, h_z.len(), "s_values/H(z) length mismatch");
        assert_eq!(len, epsilon_z.len(), "s_values/epsilon(z) length mismatch");

        let spectrum = assemble_primordial_spectrum(&z_points, &h_z, &epsilon_z, planck_mass);
        emit_primordial_projection_meta(
            log_start,
            log_step,
            lattice_len,
            &s_values,
            &z_points,
            &h_z,
            &epsilon_z,
            &spectrum,
            planck_mass,
        );

        Self {
            log_start,
            log_step,
            lattice_len,
            s_values,
            z_points,
            h_z,
            epsilon_z,
            spectrum,
        }
    }

    /// Returns the number of Mellin/Z evaluation points.
    pub fn len(&self) -> usize {
        self.s_values.len()
    }

    /// Indicates whether the projection is empty.
    pub fn is_empty(&self) -> bool {
        self.s_values.is_empty()
    }
}

fn finite_meta_f64(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn complex_abs_mean(values: &[Complex64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|value| value.norm()).sum::<f64>() / values.len() as f64
}

fn complex_abs_max(values: &[Complex64]) -> f64 {
    values
        .iter()
        .map(|value| value.norm())
        .fold(0.0f64, f64::max)
}

fn real_min(values: &[Complex64]) -> f64 {
    values
        .iter()
        .map(|value| value.re)
        .fold(f64::INFINITY, f64::min)
}

fn spectrum_stats(spectrum: &[f64]) -> (f64, f64, f64, usize) {
    if spectrum.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut non_finite = 0usize;
    for value in spectrum {
        if value.is_finite() {
            min = min.min(*value);
            max = max.max(*value);
            sum += *value;
        } else {
            non_finite += 1;
        }
    }
    let finite_count = spectrum.len().saturating_sub(non_finite).max(1);
    (min, max, sum / finite_count as f64, non_finite)
}

#[allow(clippy::too_many_arguments)]
fn emit_primordial_projection_meta(
    log_start: f64,
    log_step: f64,
    lattice_len: usize,
    s_values: &[Complex64],
    z_points: &[Complex64],
    h_z: &[Complex64],
    epsilon_z: &[Complex64],
    spectrum: &[f64],
    planck_mass: f64,
) {
    let (spectrum_min, spectrum_max, spectrum_mean, spectrum_non_finite) = spectrum_stats(spectrum);
    let epsilon_min = real_min(epsilon_z);
    let epsilon_clamp_count = epsilon_z.iter().filter(|value| value.re < 1e-20).count();

    emit_tensor_op(
        "inflaton_primordial_projection",
        &[lattice_len, s_values.len()],
        &[spectrum.len().max(1), 3],
    );
    emit_tensor_op_meta("inflaton_primordial_projection", || {
        let mut payload = serde_json::Map::new();
        payload.insert("backend".into(), "cpu".into());
        payload.insert("requested_backend".into(), "auto".into());
        payload.insert(
            "kind".into(),
            "st_core_inflaton_primordial_projection".into(),
        );
        payload.insert("log_start".into(), finite_meta_f64(log_start).into());
        payload.insert("log_step".into(), finite_meta_f64(log_step).into());
        payload.insert("lattice_len".into(), lattice_len.into());
        payload.insert("evaluation_count".into(), s_values.len().into());
        payload.insert("z_point_count".into(), z_points.len().into());
        payload.insert("spectrum_count".into(), spectrum.len().into());
        payload.insert("planck_mass".into(), finite_meta_f64(planck_mass).into());
        payload.insert(
            "h_abs_mean".into(),
            finite_meta_f64(complex_abs_mean(h_z)).into(),
        );
        payload.insert(
            "h_abs_max".into(),
            finite_meta_f64(complex_abs_max(h_z)).into(),
        );
        payload.insert(
            "epsilon_abs_mean".into(),
            finite_meta_f64(complex_abs_mean(epsilon_z)).into(),
        );
        payload.insert(
            "epsilon_abs_max".into(),
            finite_meta_f64(complex_abs_max(epsilon_z)).into(),
        );
        payload.insert(
            "epsilon_real_min".into(),
            finite_meta_f64(epsilon_min).into(),
        );
        payload.insert("epsilon_clamp_count".into(), epsilon_clamp_count.into());
        payload.insert("spectrum_min".into(), finite_meta_f64(spectrum_min).into());
        payload.insert("spectrum_max".into(), finite_meta_f64(spectrum_max).into());
        payload.insert(
            "spectrum_mean".into(),
            finite_meta_f64(spectrum_mean).into(),
        );
        payload.insert("spectrum_non_finite".into(), spectrum_non_finite.into());
        payload.into()
    });
}

/// Discrete sampling of the logarithmic scale factor \(\tau = \ln a\).
#[derive(Clone, Debug, PartialEq)]
pub struct LogLattice {
    /// Starting value of \(\tau\).
    pub tau0: f64,
    /// Spacing between consecutive samples.
    pub delta_tau: f64,
    /// Recorded samples of the observable.
    pub samples: Vec<f64>,
    /// Quadrature weights (trapezoidal rule with optional tapering).
    pub weights: Vec<f64>,
}

impl LogLattice {
    /// Creates a new lattice with trapezoidal weights normalised to unit L1 norm.
    ///
    /// The `samples` vector is copied into the lattice.  The weights are
    /// initialised to a unit-sum trapezoidal rule so that integrating a constant
    /// sequence returns the constant value.
    pub fn from_samples(tau0: f64, delta_tau: f64, samples: Vec<f64>) -> Self {
        assert!(
            delta_tau.is_finite() && delta_tau > 0.0,
            "delta_tau must be positive"
        );
        let n = samples.len();
        assert!(
            n >= 2,
            "at least two samples are required for a trapezoidal rule"
        );

        let mut weights = vec![1.0; n];
        weights[0] = 0.5;
        weights[n - 1] = 0.5;

        let norm: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= norm;
        }

        Self {
            tau0,
            delta_tau,
            samples,
            weights,
        }
    }

    /// Applies a Hann window to the weights and re-normalises them.
    ///
    /// Windowing suppresses spectral leakage when the sampled sequence does not
    /// vanish at the edges of the time interval.
    pub fn apply_hann_window(&mut self) {
        let n = self.weights.len();
        if n <= 1 {
            return;
        }

        for (i, w) in self.weights.iter_mut().enumerate() {
            let window = hann_coefficient(i, n);
            *w *= window;
        }

        let norm: f64 = self.weights.iter().sum();
        if norm > 0.0 {
            for w in &mut self.weights {
                *w /= norm;
            }
        }
    }

    /// Returns the length of the lattice.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` when the lattice contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Iterator over `(sample, weight)` pairs.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (f64, f64)> + '_ {
        self.samples
            .iter()
            .copied()
            .zip(self.weights.iter().copied())
    }
}

fn hann_coefficient(index: usize, len: usize) -> f64 {
    if len <= 1 {
        return 1.0;
    }
    let n = len as f64;
    let i = index as f64;
    0.5 * (1.0 - (2.0 * PI * i / (n - 1.0)).cos())
}

/// Weighted Z transform evaluated with Horner's method.
///
/// The transform is defined as
/// \[
/// F(z) = \sum_{k=0}^{N-1} w_k x_k z^k,
/// \]
/// where the weights and samples originate from a [`LogLattice`].
pub fn z_transform(weights: &[f64], samples: &[f64], z: Complex64) -> Complex64 {
    assert_eq!(
        weights.len(),
        samples.len(),
        "weights and samples must match"
    );
    weights
        .iter()
        .zip(samples)
        .rev()
        .fold(Complex64::new(0.0, 0.0), |acc, (&w, &x)| {
            acc * z + Complex64::new(w * x, 0.0)
        })
}

/// Combines the Z-domain background channels into the slow-roll power spectrum.
///
/// Each entry of `z_points` corresponds to an evaluation point \(z = e^{s\Delta\tau}\).
/// `h_z` and `epsilon_z` are precomputed Z transforms of the Hubble parameter and
/// the first slow-roll parameter evaluated at the same points.
///
/// The function returns the real-valued estimate of \(\mathcal{P}_\zeta(k)\) for
/// each point.  The epsilon channel is clamped away from zero to avoid
/// catastrophic amplification.
pub fn assemble_primordial_spectrum(
    z_points: &[Complex64],
    h_z: &[Complex64],
    epsilon_z: &[Complex64],
    planck_mass: f64,
) -> Vec<f64> {
    assert_eq!(z_points.len(), h_z.len(), "H(z) channel length mismatch");
    assert_eq!(
        z_points.len(),
        epsilon_z.len(),
        "epsilon(z) channel length mismatch"
    );
    assert!(
        planck_mass.is_sign_positive(),
        "Planck mass must be positive"
    );

    z_points
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            let h_real = h_z[idx].re;
            let eps_real = epsilon_z[idx].re.max(1e-20);
            (h_real * h_real) / (8.0 * PI * PI * eps_real * planck_mass * planck_mass)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::sync::{Arc, Mutex};

    #[test]
    fn trapezoid_weights_sum_to_one() {
        let lattice = LogLattice::from_samples(0.0, 0.25, vec![1.0, 2.0, 3.0, 4.0]);
        let sum: f64 = lattice.weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn hann_window_reduces_edge_weight() {
        let mut lattice = LogLattice::from_samples(0.0, 0.25, vec![1.0, 1.0, 1.0, 1.0]);
        let before = lattice.weights.clone();
        lattice.apply_hann_window();
        assert!(lattice.weights[0] < before[0]);
        assert!(lattice.weights[1] > 0.0);
        let sum: f64 = lattice.weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn horner_matches_direct_sum() {
        let samples = vec![1.0, 2.0, 4.0, 8.0];
        let lattice = LogLattice::from_samples(0.0, 0.5, samples.clone());
        let z = Complex64::new(0.8, 0.2);

        let direct = lattice
            .weights
            .iter()
            .zip(samples.iter())
            .enumerate()
            .fold(Complex64::new(0.0, 0.0), |acc, (k, (&w, &x))| {
                acc + Complex64::new(w * x, 0.0) * z.powu(k as u32)
            });
        let horner = z_transform(&lattice.weights, &lattice.samples, z);
        assert_relative_eq!(horner.re, direct.re, epsilon = 1e-12);
        assert_relative_eq!(horner.im, direct.im, epsilon = 1e-12);
    }

    #[test]
    fn assemble_returns_scale_invariant_spectrum() {
        let z_points = vec![Complex64::new(0.5, 0.0), Complex64::new(0.25, 0.0)];
        let h_z = vec![Complex64::new(10.0, 0.0); 2];
        let epsilon_z = vec![Complex64::new(0.01, 0.0); 2];
        let spectrum = assemble_primordial_spectrum(&z_points, &h_z, &epsilon_z, 2.435e18);

        assert_eq!(spectrum.len(), 2);
        assert_relative_eq!(spectrum[0], spectrum[1], epsilon = 1e-24);
        let expected = (10.0 * 10.0) / (8.0 * PI * PI * 0.01 * 2.435e18 * 2.435e18);
        assert_relative_eq!(spectrum[0], expected, epsilon = 1e-24);
    }

    #[test]
    fn primordial_projection_bundles_channels() {
        let log_start = 0.0;
        let log_step = 0.25;
        let lattice_len = 4;
        let s_values = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.5),
            Complex64::new(1.0, -0.5),
        ];
        let z_points: Vec<Complex64> = s_values
            .iter()
            .map(|s| (s * Complex64::new(log_step, 0.0)).exp())
            .collect();
        let h_z = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.1),
            Complex64::new(2.0, -0.1),
        ];
        let epsilon_z = vec![
            Complex64::new(0.01, 0.0),
            Complex64::new(0.02, 0.0),
            Complex64::new(0.015, 0.0),
        ];
        let projection = PrimordialProjection::new(
            log_start,
            log_step,
            lattice_len,
            s_values.clone(),
            z_points.clone(),
            h_z.clone(),
            epsilon_z.clone(),
            2.435e18,
        );

        assert_eq!(projection.log_start, log_start);
        assert_eq!(projection.log_step, log_step);
        assert_eq!(projection.lattice_len, lattice_len);
        assert_eq!(projection.s_values, s_values);
        assert_eq!(projection.z_points, z_points);
        assert_eq!(projection.h_z, h_z);
        assert_eq!(projection.epsilon_z, epsilon_z);
        assert_eq!(projection.len(), s_values.len());
        assert!(!projection.is_empty());
        assert_eq!(projection.spectrum.len(), s_values.len());
    }

    #[test]
    fn primordial_projection_emits_backend_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let log_start = 0.0;
        let log_step = 0.25;
        let lattice_len = 4;
        let s_values = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.5)];
        let z_points: Vec<Complex64> = s_values
            .iter()
            .map(|s| (s * Complex64::new(log_step, 0.0)).exp())
            .collect();
        let h_z = vec![Complex64::new(2.0, 0.0), Complex64::new(2.5, 0.1)];
        let epsilon_z = vec![Complex64::new(0.01, 0.0), Complex64::new(0.02, 0.0)];
        let projection = PrimordialProjection::new(
            log_start,
            log_step,
            lattice_len,
            s_values,
            z_points,
            h_z,
            epsilon_z,
            2.435e18,
        );
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(projection.len(), 2);
        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "inflaton_primordial_projection"
                    && data["kind"] == "st_core_inflaton_primordial_projection"
                    && data["lattice_len"] == 4
                    && data["evaluation_count"] == 2
            })
            .expect("inflaton_primordial_projection metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["requested_backend"], "auto");
        assert_eq!(meta.1["spectrum_count"], 2);
        assert_eq!(meta.1["epsilon_clamp_count"], 0);
        assert!(meta.1["h_abs_mean"].as_f64().unwrap_or(0.0) > 0.0);
        assert!(meta.1["epsilon_abs_mean"].as_f64().unwrap_or(0.0) > 0.0);
        assert!(meta.1["spectrum_mean"].as_f64().unwrap_or(0.0) > 0.0);
        assert_eq!(meta.1["spectrum_non_finite"], 0);
    }
}
