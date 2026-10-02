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
        let f = self.local.features();
        let g = attention_vjp_reference(self.spec, f, f, f, None, None, upstream)?;
        let feature_gradients = g
            .query
            .iter()
            .zip(&g.key)
            .zip(&g.value)
            .map(|((&q, &k), &v)| (f64::from(q) + f64::from(k) + f64::from(v)) as f32)
            .collect::<Vec<_>>();
        self.local.vjp(&feature_gradients)
    }
}

impl EllipticWarp {
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
