//! Causal attention on the existing nine elliptic/Lie features, not a new metric.
//! Q = K = V = phi(orientation); softmax scores are phi_i dot phi_j / sqrt(9).

use super::{EllipticLearningBatch, EllipticLearningError, EllipticWarp};
use st_kernel_contracts::attention::{
    attention_reference, attention_vjp_reference, AttentionMask, AttentionSpec,
};

#[derive(Clone, Debug)]
pub struct EllipticCausalLearningBatch {
    local: EllipticLearningBatch,
    spec: AttentionSpec,
    features: Vec<f32>,
}

impl EllipticCausalLearningBatch {
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    /// Includes all three tied Q/K/V paths before the original Rust chart VJP.
    pub fn vjp(&self, upstream: &[f32]) -> Result<Vec<f32>, EllipticLearningError> {
        self.local.vjp(&self.feature_vjp(upstream)?)
    }

    fn feature_vjp(&self, upstream: &[f32]) -> Result<Vec<f32>, EllipticLearningError> {
        let f = self.local.features();
        let g = attention_vjp_reference(self.spec, f, f, f, None, None, upstream)?;
        Ok(g.query
            .iter()
            .zip(&g.key)
            .zip(&g.value)
            .map(|((&q, &k), &v)| (f64::from(q) + f64::from(k) + f64::from(v)) as f32)
            .collect())
    }
}

/// Local features plus a bounded, signed contextual correction. This is an
/// ambient feature-space residual, not an interpolation on the elliptic manifold.
#[derive(Clone, Debug)]
pub struct EllipticGatedCausalLearningBatch {
    causal: EllipticCausalLearningBatch,
    mix: f64,
    features: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EllipticGatedCausalGradients {
    pub orientations: Vec<f32>,
    /// Sum over every batch/token/feature contribution; not a mean.
    pub raw_mix: f32,
}

fn gradient_f32(value: f64) -> Result<f32, EllipticLearningError> {
    let value = value as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EllipticLearningError::NonFiniteGradient)
    }
}

impl EllipticGatedCausalLearningBatch {
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    pub fn mix(&self) -> f32 {
        self.mix as f32
    }

    pub fn vjp(
        &self,
        upstream: &[f32],
    ) -> Result<EllipticGatedCausalGradients, EllipticLearningError> {
        if upstream.len() != self.features.len() || upstream.iter().any(|v| !v.is_finite()) {
            return Err(EllipticLearningError::InvalidUpstream);
        }
        let local = self.causal.local.features();
        let raw_mix = upstream
            .iter()
            .zip(local)
            .zip(self.causal.features())
            .map(|((&u, &a), &b)| f64::from(u) * (f64::from(b) - f64::from(a)))
            .sum::<f64>()
            * (1.0 - self.mix * self.mix);
        // Exact pointwise orientation VJP at initialization, without losing the
        // nonzero gate derivative that lets the contextual path start learning.
        let orientations = if self.mix == 0.0 {
            self.causal.local.vjp(upstream)?
        } else {
            let scaled = upstream
                .iter()
                .map(|&u| gradient_f32(f64::from(u) * self.mix))
                .collect::<Result<Vec<_>, _>>()?;
            let context = self.causal.feature_vjp(&scaled)?;
            let combined = context
                .iter()
                .zip(upstream)
                .map(|(&g, &u)| gradient_f32(f64::from(g) + (1.0 - self.mix) * f64::from(u)))
                .collect::<Result<Vec<_>, _>>()?;
            self.causal.local.vjp(&combined)?
        };
        Ok(EllipticGatedCausalGradients {
            orientations,
            raw_mix: gradient_f32(raw_mix)?,
        })
    }
}

impl EllipticWarp {
    /// y = (1 - tanh(raw_mix)) * phi(x) + tanh(raw_mix) * causal(phi(x)).
    /// A scalar gate is shared across full unpadded [batch, sequence] contexts.
    /// Zero is exactly pointwise; negative coefficients are signed corrections.
    /// Attention is still evaluated at zero to obtain the gate derivative.
    pub fn differentiate_gated_causal_batch(
        &self,
        orientations: &[f32],
        shape: [usize; 2],
        raw_mix: f32,
        max_rows: usize,
        max_pairs: usize,
    ) -> Result<EllipticGatedCausalLearningBatch, EllipticLearningError> {
        if !raw_mix.is_finite() {
            return Err(EllipticLearningError::Configuration);
        }
        let causal =
            self.differentiate_causal_batch(orientations, shape[0], shape[1], max_rows, max_pairs)?;
        let mix = f64::from(raw_mix).tanh();
        let features = if mix == 0.0 {
            causal.local.features().to_vec()
        } else {
            causal
                .local
                .features()
                .iter()
                .zip(causal.features())
                .map(|(&local, &context)| {
                    let value = ((1.0 - mix) * f64::from(local) + mix * f64::from(context)) as f32;
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(EllipticLearningError::NonFiniteOutput)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(EllipticGatedCausalLearningBatch {
            causal,
            mix,
            features,
        })
    }

    /// Full unpadded [batch, sequence, 3] contexts. No KV cache, dropout or
    /// implicit cross-sequence flattening. The budget counts B*T*T potential
    /// pairs (conservatively including masked pairs), not a score allocation.
    pub fn differentiate_causal_batch(
        &self,
        orientations: &[f32],
        batch: usize,
        sequence: usize,
        max_rows: usize,
        max_pairs: usize,
    ) -> Result<EllipticCausalLearningBatch, EllipticLearningError> {
        if batch == 0 || sequence == 0 || max_pairs == 0 {
            return Err(EllipticLearningError::Configuration);
        }
        let rows = batch
            .checked_mul(sequence)
            .ok_or(EllipticLearningError::Shape)?;
        if rows.checked_mul(3) != Some(orientations.len()) {
            return Err(EllipticLearningError::Shape);
        }
        let pairs = rows
            .checked_mul(sequence)
            .ok_or(EllipticLearningError::Shape)?;
        if pairs > max_pairs {
            return Err(EllipticLearningError::PairBudget);
        }
        let shape = [batch, 1, sequence, 9];
        let spec = AttentionSpec::new(
            &shape,
            &shape,
            &shape,
            1. / 3.,
            AttentionMask::Causal { query_offset: 0 },
        )?;
        let local = self.differentiate_batch(orientations, max_rows)?;
        let f = local.features();
        let features = attention_reference(spec, f, f, f, None, None)?;
        Ok(EllipticCausalLearningBatch {
            local,
            spec,
            features,
        })
    }
}
