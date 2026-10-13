//! Query-local convex routing. Each batch/head/query has its own logits;
//! the selected mixture is shared only over that query's keys.
use super::{HeadwiseMixtureForward, HeadwiseMixtureSpec, MixtureError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuerywiseMixtureSpec {
    shape: [usize; 5],
    grouped: HeadwiseMixtureSpec,
}

impl QuerywiseMixtureSpec {
    /// Components [batch,heads,queries,keys,components]; logits [B,H,Q,K].
    /// This operation does not impose a causal mask or validate the provenance
    /// of the logits. The attention mask and causal router remain caller-owned.
    pub fn new(shape: [usize; 5]) -> Result<Self, MixtureError> {
        if shape.contains(&0) {
            return Err(MixtureError::Shape);
        }
        let [batch, heads, queries, keys, components] = shape;
        let groups = batch
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(queries))
            .ok_or(MixtureError::Overflow)?;
        let grouped = HeadwiseMixtureSpec::new([1, groups, keys, components])?;
        Ok(Self { shape, grouped })
    }

    pub fn shape(self) -> [usize; 5] {
        self.shape
    }
    /// Equivalent contiguous [1,B*H*Q,S,K] contract; no numerical reinterpretation.
    pub fn grouped_spec(self) -> HeadwiseMixtureSpec {
        self.grouped
    }
    pub fn len(self) -> usize {
        self.grouped.len()
    }
    pub fn is_empty(self) -> bool {
        false
    }
    pub fn output_len(self) -> usize {
        self.grouped.output_len()
    }
    pub fn router_len(self) -> usize {
        self.grouped.router_len()
    }
    pub fn validate_lengths(self, input: usize, logits: usize) -> Result<(), MixtureError> {
        self.grouped.validate_lengths(input, logits)
    }
}

#[derive(Clone, Debug)]
pub struct QuerywiseMixtureForward {
    inner: HeadwiseMixtureForward,
}

#[derive(Debug)]
pub struct QuerywiseMixtureVjp {
    pub input: Vec<f32>,
    pub logits: Vec<f32>,
}

impl QuerywiseMixtureForward {
    pub fn new(
        spec: QuerywiseMixtureSpec,
        input: &[f32],
        logits: &[f32],
    ) -> Result<Self, MixtureError> {
        Ok(Self {
            inner: HeadwiseMixtureForward::new(spec.grouped_spec(), input, logits)?,
        })
    }
    pub fn output(&self) -> &[f32] {
        self.inner.output()
    }
    pub fn weights(&self) -> &[f32] {
        self.inner.weights()
    }
    /// Component pullback [B,H,Q,S,K] and local-logit pullback [B,H,Q,K].
    /// Reduction is over keys only, never other queries, heads or batches.
    pub fn backward(&self, seed: &[f32]) -> Result<QuerywiseMixtureVjp, MixtureError> {
        let g = self.inner.backward(seed)?;
        Ok(QuerywiseMixtureVjp {
            input: g.input,
            logits: g.logits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_shapes_and_exact_router_coverage() {
        let s = QuerywiseMixtureSpec::new([2, 3, 4, 5, 3]).unwrap();
        assert_eq!(s.shape(), [2, 3, 4, 5, 3]);
        assert_eq!((s.len(), s.output_len(), s.router_len()), (360, 120, 72));
        assert_eq!(s.grouped_spec().shape(), [1, 24, 5, 3]);
        assert!(!s.is_empty());
        assert!(s.validate_lengths(360, 72).is_ok());
        assert!(s.validate_lengths(360, 9).is_err());
        for dim in 0..5 {
            let mut shape = [2, 3, 4, 5, 3];
            shape[dim] = 0;
            assert_eq!(QuerywiseMixtureSpec::new(shape), Err(MixtureError::Shape));
        }
        for k in [1, 9] {
            assert_eq!(
                QuerywiseMixtureSpec::new([1, 1, 1, 1, k]),
                Err(MixtureError::Shape)
            );
        }
        assert_eq!(
            QuerywiseMixtureSpec::new([usize::MAX, 2, 2, 1, 2]),
            Err(MixtureError::Overflow)
        );
        assert_eq!(
            QuerywiseMixtureSpec::new([1, 1, 1, u32::MAX as usize, 2]),
            Err(MixtureError::Overflow)
        );
    }

    #[test]
    fn isolated_query_has_no_other_batch_head_or_query_pullback() {
        let s = QuerywiseMixtureSpec::new([2, 3, 4, 5, 3]).unwrap();
        let x: Vec<_> = (0..s.len()).map(|i| (i % 17) as f32 * 0.2).collect();
        let logits: Vec<_> = (0..s.router_len())
            .map(|i| (i as f32 * 0.7).sin())
            .collect();
        let f = QuerywiseMixtureForward::new(s, &x, &logits).unwrap();
        let (batch, head, query) = (1, 2, 1);
        let selected = (batch * 3 + head) * 4 + query;
        let mut seed = vec![0.; s.output_len()];
        seed[selected * 5..(selected + 1) * 5].copy_from_slice(&[0.1, -0.2, 0.4, 0.3, -0.1]);
        let g = f.backward(&seed).unwrap();
        for (group, grad) in g.logits.as_chunks::<3>().0.iter().enumerate() {
            if group != selected {
                assert_eq!(grad, &[0.; 3]);
            } else {
                assert!(grad.iter().any(|v| v.abs() > 1e-5));
                assert!(grad.iter().sum::<f32>().abs() < 1e-6);
            }
        }
        for (group, grad) in g.input.as_chunks::<{ 5 * 3 }>().0.iter().enumerate() {
            if group != selected {
                assert!(grad.iter().all(|&v| v == 0.));
            }
        }
        // Check the mapping against an explicit per-query dot product.
        for group in 0..24 {
            let l = &logits[group * 3..(group + 1) * 3];
            let denom: f32 = l.iter().map(|x| x.exp()).sum();
            for key in 0..5 {
                let row = (group * 5 + key) * 3;
                let expected: f32 = (0..3).map(|k| x[row + k] * l[k].exp() / denom).sum();
                assert!((f.output()[group * 5 + key] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn query_logit_vjp_matches_finite_differences() {
        let s = QuerywiseMixtureSpec::new([2, 2, 3, 4, 3]).unwrap();
        let x: Vec<_> = (0..s.len()).map(|i| (i as f32 * 0.4).cos()).collect();
        let logits: Vec<_> = (0..s.router_len())
            .map(|i| (i as f32 * 0.2).sin())
            .collect();
        let seed: Vec<_> = (0..s.output_len())
            .map(|i| (i as f32 * 0.3).cos())
            .collect();
        let g = QuerywiseMixtureForward::new(s, &x, &logits)
            .unwrap()
            .backward(&seed)
            .unwrap();
        let loss = |l: &[f32]| -> f64 {
            QuerywiseMixtureForward::new(s, &x, l)
                .unwrap()
                .output()
                .iter()
                .zip(&seed)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum()
        };
        for (i, &actual) in g.logits.iter().enumerate() {
            let mut plus = logits.clone();
            let mut minus = logits.clone();
            plus[i] += 0.002;
            minus[i] -= 0.002;
            let numeric = (loss(&plus) - loss(&minus)) / 0.004;
            assert!((numeric - f64::from(actual)).abs() < 1e-4);
        }
    }
}
