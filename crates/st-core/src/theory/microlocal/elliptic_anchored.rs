//! Pointwise, signed correction toward the fixed chart anchor phi(1, 0, 0).
//! The anchor uses the same warp; this is not a geodesic interpolation.

use super::{EllipticLearningBatch, EllipticLearningError, EllipticWarp};

#[derive(Clone, Debug)]
pub struct EllipticAnchoredLearningBatch {
    local: EllipticLearningBatch,
    anchor: Vec<f32>,
    mix: f64,
    features: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EllipticAnchoredGradients {
    pub orientations: Vec<f32>,
    /// Sum over all rows and features, not an average.
    pub raw_mix: f32,
}

fn checked_gradient(value: f64) -> Result<f32, EllipticLearningError> {
    let value = value as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EllipticLearningError::NonFiniteGradient)
    }
}

impl EllipticAnchoredLearningBatch {
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    pub fn mix(&self) -> f32 {
        self.mix as f32
    }

    /// The fixed anchor has no trainable coordinates. Both input and shared
    /// raw-gate derivatives are computed from this immutable forward snapshot.
    pub fn vjp(
        &self,
        upstream: &[f32],
    ) -> Result<EllipticAnchoredGradients, EllipticLearningError> {
        if upstream.len() != self.features.len() || upstream.iter().any(|v| !v.is_finite()) {
            return Err(EllipticLearningError::InvalidUpstream);
        }
        let raw_mix = upstream
            .iter()
            .zip(self.local.features())
            .zip(self.anchor.iter().cycle())
            .map(|((&u, &local), &anchor)| f64::from(u) * (f64::from(anchor) - f64::from(local)))
            .sum::<f64>()
            * (1.0 - self.mix * self.mix);
        let orientations = if self.mix == 0.0 {
            self.local.vjp(upstream)?
        } else {
            let scaled = upstream
                .iter()
                .map(|&u| checked_gradient((1.0 - self.mix) * f64::from(u)))
                .collect::<Result<Vec<_>, _>>()?;
            self.local.vjp(&scaled)?
        };
        Ok(EllipticAnchoredGradients {
            orientations,
            raw_mix: checked_gradient(raw_mix)?,
        })
    }
}

impl EllipticWarp {
    /// y = (1 - tanh(raw_mix)) * phi(x) + tanh(raw_mix) * phi(1, 0, 0).
    /// Rows are independent; only the raw gate is shared. No token mixing,
    /// pair budget, mask or sequence state is introduced. Zero is exactly local.
    pub fn differentiate_anchored_batch(
        &self,
        orientations: &[f32],
        raw_mix: f32,
        max_rows: usize,
    ) -> Result<EllipticAnchoredLearningBatch, EllipticLearningError> {
        if !raw_mix.is_finite() {
            return Err(EllipticLearningError::Configuration);
        }
        let local = self.differentiate_batch(orientations, max_rows)?;
        let anchor = self
            .differentiate_batch(&[1.0, 0.0, 0.0], 1)?
            .features()
            .to_vec();
        let mix = f64::from(raw_mix).tanh();
        let features = if mix == 0.0 {
            local.features().to_vec()
        } else {
            local
                .features()
                .iter()
                .zip(anchor.iter().cycle())
                .map(|(&f, &a)| {
                    let value = ((1.0 - mix) * f64::from(f) + mix * f64::from(a)) as f32;
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(EllipticLearningError::NonFiniteOutput)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(EllipticAnchoredLearningBatch {
            local,
            anchor,
            mix,
            features,
        })
    }
}
