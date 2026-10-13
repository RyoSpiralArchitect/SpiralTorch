// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex projective chord-squared bias, not Fubini-Study geodesic distance.
//!
//! Real/imaginary chart pairs encode `u = (1,z) / sqrt(1 + |z|^2)`.
//! The squared chord is `1 - |u^H v|^2`, evaluated as a sum of squared
//! complex wedges to avoid cancellation near identical states. The fixed
//! homogeneous anchor makes the zero chart regular. Both endpoints and
//! softplus head gains participate in the pullback.

use crate::pair_bias::softplus_gain;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ProjectiveError {
    #[error("projective bias needs nonempty [batch,time,even coordinates] and heads")]
    Shape,
    #[error("projective dimensions exceed portable addressing")]
    Overflow,
    #[error("projective coordinate, gain, state or cotangent length mismatch")]
    Length,
    #[error("projective input, score or returned gradient is non-finite")]
    NonFinite,
    #[error("a homogeneous projective state must have nonzero norm")]
    ZeroState,
}

fn mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}
fn wedge(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> [f64; 2] {
    let ab = mul(a, b);
    let cd = mul(c, d);
    [ab[0] - cd[0], ab[1] - cd[1]]
}
fn pair(row: &[f64], i: usize) -> [f64; 2] {
    [row[i], row[i + 1]]
}
fn norm(z: [f64; 2]) -> f64 {
    z[0] * z[0] + z[1] * z[1]
}

/// Squared chord between arbitrary homogeneous complex states, interleaved
/// `[re,im,...]`. Invariant to a separate nonzero complex scale on each state.
/// This is half the squared Frobenius distance between normalized projectors.
pub fn complex_projective_chord_squared(x: &[f64], y: &[f64]) -> Result<f64, ProjectiveError> {
    if x.len() != y.len() {
        return Err(ProjectiveError::Length);
    }
    if x.is_empty() || !x.len().is_multiple_of(2) {
        return Err(ProjectiveError::Shape);
    }
    let normalized = |row: &[f64]| -> Result<Vec<f64>, ProjectiveError> {
        if row.iter().any(|v| !v.is_finite()) {
            return Err(ProjectiveError::NonFinite);
        }
        let scale = row.iter().fold(0f64, |m, v| m.max(v.abs()));
        if scale == 0. {
            return Err(ProjectiveError::ZeroState);
        }
        Ok(row.iter().map(|v| v / scale).collect())
    };
    let x = normalized(x)?;
    let y = normalized(y)?;
    let nx: f64 = x.iter().map(|v| v * v).sum();
    let ny: f64 = y.iter().map(|v| v * v).sum();
    let mut numerator = 0.;
    for i in (0..x.len()).step_by(2) {
        for j in ((i + 2)..x.len()).step_by(2) {
            numerator += norm(wedge(pair(&x, i), pair(&y, j), pair(&x, j), pair(&y, i)));
        }
    }
    Ok(numerator / (nx * ny))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectiveBiasSpec {
    shape: [usize; 3],
    heads: usize,
    coordinates: usize,
    pairs: usize,
    scores: usize,
}

impl ProjectiveBiasSpec {
    pub fn new(shape: [usize; 3], heads: usize) -> Result<Self, ProjectiveError> {
        if shape.contains(&0) || !shape[2].is_multiple_of(2) || heads == 0 {
            return Err(ProjectiveError::Shape);
        }
        let count = |dims: &[usize]| {
            let n = dims
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or(ProjectiveError::Overflow)?;
            u32::try_from(n).map_err(|_| ProjectiveError::Overflow)?;
            Ok(n)
        };
        Ok(Self {
            shape,
            heads,
            coordinates: count(&shape)?,
            pairs: count(&[shape[0], shape[1], shape[1]])?,
            scores: count(&[shape[0], heads, shape[1], shape[1]])?,
        })
    }
    pub fn shape(self) -> [usize; 3] {
        self.shape
    }
    pub fn heads(self) -> usize {
        self.heads
    }
    pub fn coordinates_len(self) -> usize {
        self.coordinates
    }
    pub fn pairs_len(self) -> usize {
        self.pairs
    }
    pub fn scores_len(self) -> usize {
        self.scores
    }
    pub fn score_shape(self) -> [usize; 4] {
        [self.shape[0], self.heads, self.shape[1], self.shape[1]]
    }
}

// f32 chart components have enough exponent headroom in f64 even for quartic
// wedges. Do not narrow the distance or reduced cotangent before their product.
fn chart_pair(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let a = 1. + x.iter().map(|v| v * v).sum::<f64>();
    let b = 1. + y.iter().map(|v| v * v).sum::<f64>();
    let mut n = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
    for i in (0..x.len()).step_by(2) {
        for j in ((i + 2)..x.len()).step_by(2) {
            n += norm(wedge(pair(x, i), pair(y, j), pair(x, j), pair(y, i)));
        }
    }
    (n / (a * b), a, b)
}

#[derive(Clone, Copy, Default)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, x: f64) {
        let total = self.value + x;
        self.correction += if self.value.abs() >= x.abs() {
            (self.value - total) + x
        } else {
            (x - total) + self.value
        };
        self.value = total;
    }
    fn total(self) -> f64 {
        self.value + self.correction
    }
}

fn chart_derivative(x: &[f64], y: &[f64], a: f64, b: f64, out: &mut [f64], seed: f64) {
    let mut dot = [Sum::default(); 2];
    for j in (0..x.len()).step_by(2) {
        // Each product of two f32 chart components is exact in f64; do not
        // round their complex sum before compensation can retain its low term.
        dot[0].add(x[j] * y[j]);
        dot[0].add(x[j + 1] * y[j + 1]);
        dot[1].add(x[j] * y[j + 1]);
        dot[1].add(-x[j + 1] * y[j]);
    }
    // Include the anchor before collapsing the correction: both 1+L^2-L^2
    // and 1-1+epsilon^2 must retain their recoverable overlap.
    dot[0].add(1.);
    let overlap = [dot[0].total(), dot[1].total()];
    for i in (0..x.len()).step_by(2) {
        // Factor the quotient derivative as conj(C) * (C*x - A*y).
        // The wedge expansion avoids subtracting two almost-unit terms when
        // the distance saturates, retaining small but recoverable derivatives.
        let mut terms = [Sum::default(); 2];
        for j in (0..x.len()).step_by(2) {
            if i == j {
                continue;
            }
            let w = wedge(pair(x, i), pair(y, j), pair(x, j), pair(y, i));
            let t = mul(w, [x[j], -x[j + 1]]);
            terms[0].add(t[0]);
            terms[1].add(t[1]);
        }
        // Preserve each endpoint before the wedge terms cancel its high part.
        terms[0].add(x[i]);
        terms[0].add(-y[i]);
        terms[1].add(x[i + 1]);
        terms[1].add(-y[i + 1]);
        let tangent = [terms[0].total(), terms[1].total()];
        let derivative = mul([overlap[0], -overlap[1]], tangent);
        for c in 0..2 {
            out[i + c] += seed * 2. * derivative[c] / (a * a * b);
        }
    }
}

fn narrow(v: f64) -> Result<f32, ProjectiveError> {
    let f = v as f32;
    f.is_finite().then_some(f).ok_or(ProjectiveError::NonFinite)
}

#[derive(Clone, Debug)]
pub struct ProjectiveBiasForward {
    spec: ProjectiveBiasSpec,
    coordinates: Vec<f64>,
    gains: Vec<(f64, f64)>,
    scores: Vec<f32>,
}

#[derive(Debug)]
pub struct ProjectiveBiasVjp {
    pub coordinates: Vec<f32>,
    pub raw_gain: Vec<f32>,
}

impl ProjectiveBiasForward {
    pub fn new(
        spec: ProjectiveBiasSpec,
        coordinates: &[f32],
        raw_gain: &[f32],
    ) -> Result<Self, ProjectiveError> {
        if coordinates.len() != spec.coordinates || raw_gain.len() != spec.heads {
            return Err(ProjectiveError::Length);
        }
        if coordinates.iter().chain(raw_gain).any(|v| !v.is_finite()) {
            return Err(ProjectiveError::NonFinite);
        }
        let coordinates: Vec<_> = coordinates.iter().copied().map(f64::from).collect();
        let gains: Vec<_> = raw_gain.iter().map(|&r| softplus_gain(r)).collect();
        let [batch, time, cols] = spec.shape;
        let mut scores = vec![0.; spec.scores];
        for b in 0..batch {
            for q in 0..time {
                for k in 0..=q {
                    let qi = (b * time + q) * cols;
                    let ki = (b * time + k) * cols;
                    let d = chart_pair(&coordinates[qi..qi + cols], &coordinates[ki..ki + cols]).0;
                    for h in 0..spec.heads {
                        scores[((b * spec.heads + h) * time + q) * time + k] =
                            narrow(-gains[h].0 * d)?;
                    }
                }
            }
        }
        Ok(Self {
            spec,
            coordinates,
            gains,
            scores,
        })
    }
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }
    pub fn backward(&self, seed: &[f32]) -> Result<ProjectiveBiasVjp, ProjectiveError> {
        let s = self.spec;
        if seed.len() != s.scores {
            return Err(ProjectiveError::Length);
        }
        if seed.iter().any(|v| !v.is_finite()) {
            return Err(ProjectiveError::NonFinite);
        }
        let [batch, time, cols] = s.shape;
        let mut dx = vec![0f64; s.coordinates];
        let mut dg = vec![0f64; s.heads];
        for b in 0..batch {
            for q in 0..time {
                for k in 0..=q {
                    let qi = (b * time + q) * cols;
                    let ki = (b * time + k) * cols;
                    let x = &self.coordinates[qi..qi + cols];
                    let y = &self.coordinates[ki..ki + cols];
                    let (d, a, b_norm) = chart_pair(x, y);
                    let mut reduced = 0.;
                    for h in 0..s.heads {
                        let g = f64::from(seed[((b * s.heads + h) * time + q) * time + k]);
                        reduced -= g * self.gains[h].0;
                        dg[h] -= g * self.gains[h].1 * d;
                    }
                    chart_derivative(x, y, a, b_norm, &mut dx[qi..qi + cols], reduced);
                    chart_derivative(y, x, b_norm, a, &mut dx[ki..ki + cols], reduced);
                }
            }
        }
        Ok(ProjectiveBiasVjp {
            coordinates: dx.into_iter().map(narrow).collect::<Result<_, _>>()?,
            raw_gain: dg.into_iter().map(narrow).collect::<Result<_, _>>()?,
        })
    }
}

#[cfg(test)]
mod tests;
