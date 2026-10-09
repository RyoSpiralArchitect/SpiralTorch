// SPDX-License-Identifier: AGPL-3.0-or-later
//! Causal complex-state filter and a bounded Poincare-ball coordinate chart.
//! The chart is not an exponential map or a Riemannian optimizer. Raw decay and
//! phase are ordinary differentiable parameters; state is explicit, not global.

use thiserror::Error;

pub const MAX_DECAY: f32 = 0.99;
pub const INTERIOR_RATIO: f32 = 0.95;

#[derive(Debug, Error, PartialEq)]
pub enum CausalWaveError {
    #[error("causal wave needs nonempty [batch,time,2*complex_channels]")]
    Shape,
    #[error("causal wave curvature must be finite, negative and have a representable radius")]
    Curvature,
    #[error("causal wave dimensions exceed portable addressing")]
    Overflow,
    #[error("causal wave input, state, parameters or cotangent shape mismatch")]
    Length,
    #[error("causal wave input, intermediate or derivative is non-finite")]
    NonFinite,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CausalWaveSpec {
    shape: [usize; 3],
    curvature: f32,
    radius: f32,
    len: usize,
}

impl CausalWaveSpec {
    pub fn new(shape: [usize; 3], curvature: f32) -> Result<Self, CausalWaveError> {
        if shape.contains(&0) || !shape[2].is_multiple_of(2) {
            return Err(CausalWaveError::Shape);
        }
        let len = shape
            .iter()
            .try_fold(1usize, |n, &s| n.checked_mul(s))
            .ok_or(CausalWaveError::Overflow)?;
        u32::try_from(len).map_err(|_| CausalWaveError::Overflow)?;
        let radius = INTERIOR_RATIO / (-curvature).sqrt();
        if !curvature.is_finite() || curvature >= 0. || !radius.is_finite() || radius <= 0. {
            return Err(CausalWaveError::Curvature);
        }
        Ok(Self {
            shape,
            curvature,
            radius,
            len,
        })
    }
    pub fn shape(self) -> [usize; 3] {
        self.shape
    }
    pub fn curvature(self) -> f32 {
        self.curvature
    }
    pub fn radius(self) -> f32 {
        self.radius
    }
    pub fn len(self) -> usize {
        self.len
    }
    pub fn is_empty(self) -> bool {
        false
    }
    pub fn pairs(self) -> usize {
        self.shape[2] / 2
    }
    pub fn state_len(self) -> usize {
        self.shape[0] * self.shape[2]
    }
    pub fn validate_lengths(
        self,
        drive: usize,
        decay: usize,
        phase: usize,
        state: usize,
    ) -> Result<(), CausalWaveError> {
        if drive != self.len
            || decay != self.pairs()
            || phase != self.pairs()
            || state != self.state_len()
        {
            return Err(CausalWaveError::Length);
        }
        Ok(())
    }
}

fn finite(v: f32) -> Result<f32, CausalWaveError> {
    v.is_finite().then_some(v).ok_or(CausalWaveError::NonFinite)
}

fn coefficients(decay: f32, phase: f32) -> Result<[f32; 5], CausalWaveError> {
    finite(decay)?;
    finite(phase)?;
    let sigmoid = if decay >= 0. {
        1. / (1. + (-decay).exp())
    } else {
        let e = decay.exp();
        e / (1. + e)
    };
    let t = phase.tanh();
    let theta = std::f32::consts::PI * t;
    Ok([
        MAX_DECAY * sigmoid,
        theta.cos(),
        theta.sin(),
        MAX_DECAY * sigmoid * (1. - sigmoid),
        std::f32::consts::PI * (1. - t * t),
    ])
}

/// Stable evaluation of s/sqrt(1+||s||^2), without squaring unscaled states.
fn chart_unit(state: &[f32]) -> Result<(Vec<f32>, f32), CausalWaveError> {
    let scale = state.iter().fold(1f32, |a, v| a.max(v.abs()));
    let inv = 1. / scale;
    let mut squared = inv * inv;
    for &v in state {
        squared = finite(squared + (v / scale) * (v / scale))?;
    }
    let root = squared.sqrt();
    Ok((
        state.iter().map(|v| (v / scale) / root).collect(),
        inv / root,
    ))
}

// Keep extended adjoints through the rotation contraction. Narrowing chart
// components first destroys proportionality and invents phase derivatives.
fn chart_pullback(state: &[f32], seed: &[f32], radius: f32) -> Vec<f64> {
    let mut pivot = 0;
    for (i, s) in state.iter().enumerate() {
        if s.abs() > state[pivot].abs() {
            pivot = i;
        }
    }
    let s: Vec<_> = state.iter().map(|&s| f64::from(s)).collect();
    let g: Vec<_> = seed.iter().map(|&g| f64::from(g)).collect();
    let radius = f64::from(radius);
    if s[pivot] == 0. {
        return g.iter().map(|g| radius * g).collect();
    }
    let residual: Vec<_> = s
        .iter()
        .zip(&g)
        .map(|(&v, &h)| (h * s[pivot] - g[pivot] * v) / s[pivot])
        .collect();
    let q = 1. + s.iter().map(|s| s * s).sum::<f64>();
    let root = q.sqrt();
    let dot: f64 = s.iter().zip(&residual).map(|(s, r)| s * r).sum();
    s.iter()
        .zip(&residual)
        .map(|(&v, &r)| {
            radius / root * (r - v * dot / q) + radius / root / q * (g[pivot] / s[pivot]) * v
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct CausalWaveForward {
    spec: CausalWaveSpec,
    drive: Vec<f32>,
    initial: Vec<f32>,
    coefficients: Vec<[f32; 5]>,
    states: Vec<f32>,
    features: Vec<f32>,
    final_state: Vec<f32>,
}

#[derive(Debug)]
pub struct CausalWaveVjp {
    pub drive: Vec<f32>,
    pub raw_decay: Vec<f32>,
    pub raw_phase: Vec<f32>,
    pub initial_state: Vec<f32>,
}

impl CausalWaveForward {
    /// rho = .99*sigmoid(raw_decay), theta = pi*tanh(raw_phase).
    /// s_t = rho*rotate(theta,s_(t-1)) + (1-rho)*drive_t.
    pub fn new(
        spec: CausalWaveSpec,
        drive: &[f32],
        decay: &[f32],
        phase: &[f32],
        initial: &[f32],
    ) -> Result<Self, CausalWaveError> {
        spec.validate_lengths(drive.len(), decay.len(), phase.len(), initial.len())?;
        for &v in drive.iter().chain(initial) {
            finite(v)?;
        }
        let coefficients = decay
            .iter()
            .zip(phase)
            .map(|(&d, &p)| coefficients(d, p))
            .collect::<Result<Vec<_>, _>>()?;
        let [batch, steps, cols] = spec.shape;
        let mut states = vec![0.; spec.len];
        let mut features = vec![0.; spec.len];
        let mut final_state = vec![0.; spec.state_len()];
        for b in 0..batch {
            for t in 0..steps {
                let base = (b * steps + t) * cols;
                for (p, &[rho, cos, sin, _, _]) in coefficients.iter().enumerate() {
                    let i = base + 2 * p;
                    let prev = if t == 0 {
                        &initial[b * cols..(b + 1) * cols]
                    } else {
                        &states[base - cols..base]
                    };
                    let (x, y) = (prev[2 * p], prev[2 * p + 1]);
                    let rx = finite(finite(cos * x)? - finite(sin * y)?)?;
                    let ry = finite(finite(sin * x)? + finite(cos * y)?)?;
                    states[i] = finite(finite(rho * rx)? + finite((1. - rho) * drive[i])?)?;
                    states[i + 1] = finite(finite(rho * ry)? + finite((1. - rho) * drive[i + 1])?)?;
                }
                let (unit, _) = chart_unit(&states[base..base + cols])?;
                for c in 0..cols {
                    features[base + c] = finite(spec.radius * unit[c])?;
                }
            }
            final_state[b * cols..(b + 1) * cols]
                .copy_from_slice(&states[((b + 1) * steps - 1) * cols..(b + 1) * steps * cols]);
        }
        Ok(Self {
            spec,
            drive: drive.to_vec(),
            initial: initial.to_vec(),
            coefficients,
            states,
            features,
            final_state,
        })
    }
    pub fn features(&self) -> &[f32] {
        &self.features
    }
    pub fn final_state(&self) -> &[f32] {
        &self.final_state
    }
    pub fn states(&self) -> &[f32] {
        &self.states
    }

    /// Exact pullback of both returned paths. Terminal-state seeds permit
    /// differentiating chunked execution without truncating BPTT at a boundary.
    pub fn backward(
        &self,
        features: &[f32],
        terminal: &[f32],
    ) -> Result<CausalWaveVjp, CausalWaveError> {
        let spec = self.spec;
        if features.len() != spec.len || terminal.len() != spec.state_len() {
            return Err(CausalWaveError::Length);
        }
        for &v in features.iter().chain(terminal) {
            finite(v)?;
        }
        let [batch, steps, cols] = spec.shape;
        let mut state_grad = vec![0.; spec.len];
        for row in 0..batch * steps {
            let start = row * cols;
            state_grad[start..start + cols].copy_from_slice(&chart_pullback(
                &self.states[start..start + cols],
                &features[start..start + cols],
                spec.radius,
            ));
        }
        let mut result = CausalWaveVjp {
            drive: vec![0.; spec.len],
            raw_decay: vec![0.; spec.pairs()],
            raw_phase: vec![0.; spec.pairs()],
            initial_state: vec![0.; spec.state_len()],
        };
        for b in 0..batch {
            for (p, &[rho, cos, sin, dr, dt]) in self.coefficients.iter().enumerate() {
                let mut gx = f64::from(terminal[b * cols + 2 * p]);
                let mut gy = f64::from(terminal[b * cols + 2 * p + 1]);
                let mut gd = 0.;
                let mut gp = 0.;
                for t in (0..steps).rev() {
                    let i = (b * steps + t) * cols + 2 * p;
                    gx += state_grad[i];
                    gy += state_grad[i + 1];
                    result.drive[i] = finite((f64::from(1. - rho) * gx) as f32)?;
                    result.drive[i + 1] = finite((f64::from(1. - rho) * gy) as f32)?;
                    let prev = if t == 0 {
                        &self.initial[b * cols..(b + 1) * cols]
                    } else {
                        &self.states[(b * steps + t - 1) * cols..(b * steps + t) * cols]
                    };
                    let (x, y) = (prev[2 * p], prev[2 * p + 1]);
                    let rx = finite(finite(cos * x)? - finite(sin * y)?)?;
                    let ry = finite(finite(sin * x)? + finite(cos * y)?)?;
                    let decay_x = gx * (f64::from(rx) - f64::from(self.drive[i]));
                    let decay_y = gy * (f64::from(ry) - f64::from(self.drive[i + 1]));
                    gd += (decay_x + decay_y) * f64::from(dr);
                    let rotation = gy * f64::from(rx) - gx * f64::from(ry);
                    gp += f64::from(rho) * rotation * f64::from(dt);
                    let next_x = f64::from(rho) * (f64::from(cos) * gx + f64::from(sin) * gy);
                    let next_y = f64::from(rho) * (f64::from(cos) * gy - f64::from(sin) * gx);
                    gx = next_x;
                    gy = next_y;
                }
                result.initial_state[b * cols + 2 * p] = finite(gx as f32)?;
                result.initial_state[b * cols + 2 * p + 1] = finite(gy as f32)?;
                result.raw_decay[p] = finite(result.raw_decay[p] + finite(gd as f32)?)?;
                result.raw_phase[p] = finite(result.raw_phase[p] + finite(gp as f32)?)?;
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
