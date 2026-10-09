// SPDX-License-Identifier: AGPL-3.0-or-later
//! Causal, scaled squared-Euclidean distance bias with learned softplus gains.
//! This operates on the supplied coordinates; it performs no ball projection.
use crate::pair_bias::softplus_gain;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum EuclideanBiasError {
    #[error("Euclidean bias needs nonempty [batch,time,coordinates] and heads")]
    Shape,
    #[error("Euclidean squared-distance scale must be positive and finite")]
    Scale,
    #[error("Euclidean bias dimensions exceed portable addressing")]
    Overflow,
    #[error("Euclidean bias coordinate, gain or cotangent length mismatch")]
    Length,
    #[error("Euclidean bias input, score or returned gradient is non-finite")]
    NonFinite,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EuclideanBiasSpec {
    shape: [usize; 3],
    heads: usize,
    scale: f32,
    coordinates: usize,
    pairs: usize,
    scores: usize,
}

impl EuclideanBiasSpec {
    pub fn new(shape: [usize; 3], heads: usize, scale: f32) -> Result<Self, EuclideanBiasError> {
        if shape.contains(&0) || heads == 0 {
            return Err(EuclideanBiasError::Shape);
        }
        if !scale.is_finite() || scale <= 0. {
            return Err(EuclideanBiasError::Scale);
        }
        let count = |dims: &[usize]| {
            let n = dims
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or(EuclideanBiasError::Overflow)?;
            u32::try_from(n).map_err(|_| EuclideanBiasError::Overflow)?;
            Ok(n)
        };
        Ok(Self {
            shape,
            heads,
            scale,
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
    pub fn scale(self) -> f32 {
        self.scale
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

fn narrow(v: f64) -> Result<f32, EuclideanBiasError> {
    let f = v as f32;
    f.is_finite()
        .then_some(f)
        .ok_or(EuclideanBiasError::NonFinite)
}

#[derive(Clone, Debug)]
pub struct EuclideanBiasForward {
    spec: EuclideanBiasSpec,
    coordinates: Vec<f32>,
    gains: Vec<(f64, f64)>,
    distances: Vec<f64>,
    scores: Vec<f32>,
}

#[derive(Debug)]
pub struct EuclideanBiasVjp {
    pub coordinates: Vec<f32>,
    pub raw_gain: Vec<f32>,
}

impl EuclideanBiasForward {
    pub fn new(
        spec: EuclideanBiasSpec,
        coordinates: &[f32],
        raw_gain: &[f32],
    ) -> Result<Self, EuclideanBiasError> {
        if coordinates.len() != spec.coordinates || raw_gain.len() != spec.heads {
            return Err(EuclideanBiasError::Length);
        }
        if coordinates.iter().chain(raw_gain).any(|v| !v.is_finite()) {
            return Err(EuclideanBiasError::NonFinite);
        }
        let [batch, time, cols] = spec.shape;
        let gains: Vec<_> = raw_gain.iter().map(|&r| softplus_gain(r)).collect();
        let mut distances = vec![0.; spec.pairs];
        let mut scores = vec![0.; spec.scores];
        for b in 0..batch {
            for q in 0..time {
                for k in 0..=q {
                    let distance = f64::from(spec.scale)
                        * (0..cols)
                            .map(|i| {
                                (f64::from(coordinates[(b * time + q) * cols + i])
                                    - f64::from(coordinates[(b * time + k) * cols + i]))
                                .powi(2)
                            })
                            .sum::<f64>();
                    distances[(b * time + q) * time + k] = distance;
                    for h in 0..spec.heads {
                        scores[((b * spec.heads + h) * time + q) * time + k] =
                            narrow(-gains[h].0 * distance)?;
                    }
                }
            }
        }
        Ok(Self {
            spec,
            coordinates: coordinates.to_vec(),
            gains,
            distances,
            scores,
        })
    }
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }
    pub fn backward(&self, seed: &[f32]) -> Result<EuclideanBiasVjp, EuclideanBiasError> {
        let s = self.spec;
        if seed.len() != s.scores {
            return Err(EuclideanBiasError::Length);
        }
        if seed.iter().any(|v| !v.is_finite()) {
            return Err(EuclideanBiasError::NonFinite);
        }
        let [batch, time, cols] = s.shape;
        let mut dx = vec![0f64; s.coordinates];
        let mut dg = vec![0f64; s.heads];
        for b in 0..batch {
            for q in 0..time {
                for k in 0..=q {
                    let distance = self.distances[(b * time + q) * time + k];
                    let mut scale = 0.;
                    for h in 0..s.heads {
                        let g = f64::from(seed[((b * s.heads + h) * time + q) * time + k]);
                        scale -= g * self.gains[h].0;
                        dg[h] -= g * self.gains[h].1 * distance;
                    }
                    for i in 0..cols {
                        let qi = (b * time + q) * cols + i;
                        let ki = (b * time + k) * cols + i;
                        let derivative = scale
                            * 2.
                            * f64::from(s.scale)
                            * (f64::from(self.coordinates[qi]) - f64::from(self.coordinates[ki]));
                        dx[qi] += derivative;
                        dx[ki] -= derivative;
                    }
                }
            }
        }
        Ok(EuclideanBiasVjp {
            coordinates: dx.into_iter().map(narrow).collect::<Result<_, _>>()?,
            raw_gain: dg.into_iter().map(narrow).collect::<Result<_, _>>()?,
        })
    }
}

#[cfg(test)]
mod tests;
