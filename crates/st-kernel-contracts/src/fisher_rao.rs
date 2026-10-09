// SPDX-License-Identifier: AGPL-3.0-or-later
//! Categorical Fisher-Rao squared distance on rowwise softmax coordinates.
//! Scores are causal negative distances with learned positive softplus gains.
use crate::pair_bias::softplus_gain;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum FisherRaoError {
    #[error("Fisher-Rao bias needs nonempty [batch,time,categories] and heads")]
    Shape,
    #[error("Fisher-Rao dimensions exceed portable addressing")]
    Overflow,
    #[error("Fisher-Rao logit, gain or cotangent length mismatch")]
    Length,
    #[error("Fisher-Rao input, score or returned gradient is non-finite")]
    NonFinite,
    #[error("normalized nonnegative root-chord squared must be in [0,2]")]
    RootChord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FisherRaoBiasSpec {
    shape: [usize; 3],
    heads: usize,
    coordinates: usize,
    pairs: usize,
    scores: usize,
}

impl FisherRaoBiasSpec {
    pub fn new(shape: [usize; 3], heads: usize) -> Result<Self, FisherRaoError> {
        if shape.contains(&0) || heads == 0 {
            return Err(FisherRaoError::Shape);
        }
        let count = |dims: &[usize]| {
            let n = dims
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or(FisherRaoError::Overflow)?;
            u32::try_from(n).map_err(|_| FisherRaoError::Overflow)?;
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

/// Return squared radius-two spherical distance and its derivative with respect
/// to `t = sum((sqrt(p) - sqrt(q))^2)`, for normalized distributions.
/// This equals `4 * acos(sum(sqrt(p*q)))^2`, without cancellation at identity.
/// Callers may cap roundoff above two after establishing normalization.
pub fn squared_distance_from_root_chord(t: f64) -> Result<(f64, f64), FisherRaoError> {
    if !t.is_finite() || !(0.0..=2.0).contains(&t) {
        return Err(FisherRaoError::RootChord);
    }
    let z = t * 0.25;
    if z < 1e-4 {
        let ratio = 1. + z * (1. / 3. + z * (8. / 45. + z * (4. / 35. + z * 128. / 1575.)));
        let slope = 1. + z * (2. / 3. + z * (8. / 15. + z * (16. / 35. + z * 128. / 315.)));
        Ok((4. * t * ratio, 4. * slope))
    } else {
        let root = z.sqrt();
        let angle = root.asin();
        Ok((16. * angle * angle, 4. * angle / (root * (1. - z).sqrt())))
    }
}

fn narrow(v: f64) -> Result<f32, FisherRaoError> {
    let f = v as f32;
    f.is_finite().then_some(f).ok_or(FisherRaoError::NonFinite)
}

#[derive(Clone, Debug)]
pub struct FisherRaoBiasForward {
    spec: FisherRaoBiasSpec,
    roots: Vec<f64>,
    gains: Vec<(f64, f64)>,
    distances: Vec<(f64, f64)>,
    scores: Vec<f32>,
}

#[derive(Debug)]
pub struct FisherRaoBiasVjp {
    /// Cotangent of the supplied logits, including the simplex-map pullback.
    pub coordinates: Vec<f32>,
    pub raw_gain: Vec<f32>,
}

impl FisherRaoBiasForward {
    pub fn new(
        spec: FisherRaoBiasSpec,
        logits: &[f32],
        raw_gain: &[f32],
    ) -> Result<Self, FisherRaoError> {
        if logits.len() != spec.coordinates || raw_gain.len() != spec.heads {
            return Err(FisherRaoError::Length);
        }
        if logits.iter().chain(raw_gain).any(|x| !x.is_finite()) {
            return Err(FisherRaoError::NonFinite);
        }
        let [batch, time, cols] = spec.shape;
        let mut roots = Vec::with_capacity(spec.coordinates);
        for row in logits.chunks_exact(cols) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let start = roots.len();
            roots.extend(
                row.iter()
                    .map(|&x| ((f64::from(x) - f64::from(max)) * 0.5).exp()),
            );
            let norm = roots[start..].iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in &mut roots[start..] {
                *x /= norm;
            }
        }
        let gains: Vec<_> = raw_gain.iter().map(|&r| softplus_gain(r)).collect();
        let mut distances = vec![(0., 0.); spec.pairs];
        let mut scores = vec![0.; spec.scores];
        for b in 0..batch {
            for q in 0..time {
                for k in 0..=q {
                    let chord = (0..cols)
                        .map(|i| {
                            (roots[(b * time + q) * cols + i] - roots[(b * time + k) * cols + i])
                                .powi(2)
                        })
                        .sum::<f64>();
                    let distance = squared_distance_from_root_chord(chord.min(2.))?;
                    distances[(b * time + q) * time + k] = distance;
                    for h in 0..spec.heads {
                        scores[((b * spec.heads + h) * time + q) * time + k] =
                            narrow(-gains[h].0 * distance.0)?;
                    }
                }
            }
        }
        Ok(Self {
            spec,
            roots,
            gains,
            distances,
            scores,
        })
    }
    pub fn spec(&self) -> FisherRaoBiasSpec {
        self.spec
    }
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    pub fn backward(&self, seed: &[f32]) -> Result<FisherRaoBiasVjp, FisherRaoError> {
        let s = self.spec;
        if seed.len() != s.scores {
            return Err(FisherRaoError::Length);
        }
        if seed.iter().any(|x| !x.is_finite()) {
            return Err(FisherRaoError::NonFinite);
        }
        let [batch, time, cols] = s.shape;
        let mut dr = vec![0.; s.coordinates];
        let mut dg = vec![0.; s.heads];
        for b in 0..batch {
            for q in 0..time {
                for k in 0..=q {
                    let (distance, slope) = self.distances[(b * time + q) * time + k];
                    let mut scale = 0.;
                    for h in 0..s.heads {
                        let g = f64::from(seed[((b * s.heads + h) * time + q) * time + k]);
                        scale -= g * self.gains[h].0;
                        dg[h] -= g * self.gains[h].1 * distance;
                    }
                    for i in 0..cols {
                        let qi = (b * time + q) * cols + i;
                        let ki = (b * time + k) * cols + i;
                        let d = scale * 2. * slope * (self.roots[qi] - self.roots[ki]);
                        dr[qi] += d;
                        dr[ki] -= d;
                    }
                }
            }
        }
        let mut dx = Vec::with_capacity(s.coordinates);
        for (roots, gradient) in self.roots.chunks_exact(cols).zip(dr.chunks_exact(cols)) {
            let dot = roots.iter().zip(gradient).map(|(r, g)| r * g).sum::<f64>();
            for (&r, &g) in roots.iter().zip(gradient) {
                dx.push(narrow(0.5 * r * (g - r * dot))?);
            }
        }
        Ok(FisherRaoBiasVjp {
            coordinates: dx,
            raw_gain: dg.into_iter().map(narrow).collect::<Result<_, _>>()?,
        })
    }
}

#[cfg(test)]
mod tests;
