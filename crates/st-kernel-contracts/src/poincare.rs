// SPDX-License-Identifier: AGPL-3.0-or-later
//! Causal Poincare squared-distance score bias and its Euclidean pullback.
//! Coordinates must already lie in the open ball. This is not a retraction,
//! Riemannian optimizer, tokenizer or replacement for Attention's causal mask.

use crate::pair_bias::softplus_gain as gain;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum PoincareError {
    #[error("Poincare bias needs nonempty [batch,time,coordinates] and heads")]
    Shape,
    #[error("Poincare curvature must be finite and strictly negative")]
    Curvature,
    #[error("Poincare dimensions exceed portable addressing")]
    Overflow,
    #[error("Poincare coordinate, gain or cotangent length mismatch")]
    Length,
    #[error("Poincare coordinates must lie strictly inside the curvature ball")]
    Domain,
    #[error("Poincare input, score or returned gradient is non-finite")]
    NonFinite,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoincareBiasSpec {
    shape: [usize; 3],
    heads: usize,
    curvature: f32,
    coordinates: usize,
    pairs: usize,
    scores: usize,
}

impl PoincareBiasSpec {
    pub fn new(shape: [usize; 3], heads: usize, curvature: f32) -> Result<Self, PoincareError> {
        if shape.contains(&0) || heads == 0 {
            return Err(PoincareError::Shape);
        }
        if !curvature.is_finite() || curvature >= 0. {
            return Err(PoincareError::Curvature);
        }
        let count = |dims: &[usize]| {
            let n = dims
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or(PoincareError::Overflow)?;
            u32::try_from(n).map_err(|_| PoincareError::Overflow)?;
            Ok(n)
        };
        Ok(Self {
            shape,
            heads,
            curvature,
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
    pub fn curvature(self) -> f32 {
        self.curvature
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

fn narrow(v: f64) -> Result<f32, PoincareError> {
    let f = v as f32;
    f.is_finite().then_some(f).ok_or(PoincareError::NonFinite)
}

fn ball_margin(row: &[f32], curvature_magnitude: f64) -> f64 {
    // f32 squares are exact in f64. Preserve the residual of multiplying by
    // curvature, then sum signed terms before narrowing the tiny ball margin.
    let mut partials = vec![1f64];
    for &x in row {
        let square = f64::from(x) * f64::from(x);
        let product = curvature_magnitude * square;
        let residual = curvature_magnitude.mul_add(square, -product);
        for mut value in [-product, -residual] {
            let mut kept = 0;
            for i in 0..partials.len() {
                let mut other = partials[i];
                if value.abs() < other.abs() {
                    std::mem::swap(&mut value, &mut other);
                }
                let sum = value + other;
                let error = other - (sum - value);
                if error != 0. {
                    partials[kept] = error;
                    kept += 1;
                }
                value = sum;
            }
            partials.truncate(kept);
            partials.push(value);
        }
    }
    partials.iter().sum()
}

#[derive(Clone, Debug)]
pub struct PoincareBiasForward {
    spec: PoincareBiasSpec,
    coordinates: Vec<f32>,
    gains: Vec<(f64, f64)>,
    // d^2, 8*K/(A*B), 8*K*v/A, 8*K*v/B. Never round before contractions.
    pairs: Vec<[f64; 4]>,
    scores: Vec<f32>,
}

#[derive(Debug)]
pub struct PoincareBiasVjp {
    pub coordinates: Vec<f32>,
    pub raw_gain: Vec<f32>,
}

impl PoincareBiasForward {
    pub fn new(
        spec: PoincareBiasSpec,
        coordinates: &[f32],
        raw_gain: &[f32],
    ) -> Result<Self, PoincareError> {
        if coordinates.len() != spec.coordinates || raw_gain.len() != spec.heads {
            return Err(PoincareError::Length);
        }
        if coordinates.iter().chain(raw_gain).any(|x| !x.is_finite()) {
            return Err(PoincareError::NonFinite);
        }
        let [batch, time, cols] = spec.shape;
        let c = -f64::from(spec.curvature);
        let margins: Vec<_> = coordinates
            .chunks_exact(cols)
            .map(|row| ball_margin(row, c))
            .collect();
        if margins.iter().any(|&a| a <= 0.) {
            return Err(PoincareError::Domain);
        }
        let gains: Vec<_> = raw_gain.iter().map(|&r| gain(r)).collect();
        let mut pairs = vec![[0.; 4]; spec.pairs];
        let mut scores = vec![0.; spec.scores];
        for b in 0..batch {
            for q in 0..time {
                for k in 0..=q {
                    let (qi, ki) = ((b * time + q) * cols, (b * time + k) * cols);
                    let (a, d) = (margins[b * time + q], margins[b * time + k]);
                    let square: f64 = coordinates[qi..qi + cols]
                        .iter()
                        .zip(&coordinates[ki..ki + cols])
                        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                        .sum();
                    let v = c * square / (a * d);
                    let root = v.sqrt();
                    let angle = root.asinh();
                    let kernel = if v == 0. {
                        1.
                    } else {
                        angle / (root * (1. + v).sqrt())
                    };
                    let distance = 4. * angle * angle / c;
                    pairs[(b * time + q) * time + k] = [
                        distance,
                        8. * kernel / (a * d),
                        8. * kernel * v / a,
                        8. * kernel * v / d,
                    ];
                    for (h, &(g, _)) in gains.iter().enumerate() {
                        scores[((b * spec.heads + h) * time + q) * time + k] =
                            narrow(-g * distance)?;
                    }
                }
            }
        }
        Ok(Self {
            spec,
            coordinates: coordinates.to_vec(),
            gains,
            pairs,
            scores,
        })
    }
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }
    pub fn spec(&self) -> PoincareBiasSpec {
        self.spec
    }

    /// Sum both endpoint roles and all heads. Future scores have zero VJP;
    /// nevertheless every supplied cotangent must be finite.
    pub fn backward(&self, seed: &[f32]) -> Result<PoincareBiasVjp, PoincareError> {
        let s = self.spec;
        if seed.len() != s.scores {
            return Err(PoincareError::Length);
        }
        if seed.iter().any(|v| !v.is_finite()) {
            return Err(PoincareError::NonFinite);
        }
        let [batch, time, cols] = s.shape;
        let mut coordinates = vec![0f64; s.coordinates];
        let mut raw_gain = vec![0f64; s.heads];
        for b in 0..batch {
            for q in 0..time {
                for k in 0..=q {
                    let [distance, base, rx, ry] = self.pairs[(b * time + q) * time + k];
                    let mut scale = 0.;
                    for (h, (&(gain, derivative), grad)) in
                        self.gains.iter().zip(&mut raw_gain).enumerate()
                    {
                        let cotangent = f64::from(seed[((b * s.heads + h) * time + q) * time + k]);
                        scale -= cotangent * gain;
                        *grad -= cotangent * distance * derivative;
                    }
                    let (qi, ki) = ((b * time + q) * cols, (b * time + k) * cols);
                    for i in 0..cols {
                        let (x, y) = (
                            f64::from(self.coordinates[qi + i]),
                            f64::from(self.coordinates[ki + i]),
                        );
                        coordinates[qi + i] += scale * (base * (x - y) + rx * x);
                        coordinates[ki + i] += scale * (base * (y - x) + ry * y);
                    }
                }
            }
        }
        Ok(PoincareBiasVjp {
            coordinates: coordinates
                .into_iter()
                .map(narrow)
                .collect::<Result<_, _>>()?,
            raw_gain: raw_gain.into_iter().map(narrow).collect::<Result<_, _>>()?,
        })
    }
}

#[cfg(test)]
mod tests;
