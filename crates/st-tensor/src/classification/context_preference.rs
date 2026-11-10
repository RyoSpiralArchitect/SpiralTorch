//! Prefer the same full-vocabulary token under the first context over the second.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ContextTokenPreference {
    pub target_id: usize,
    pub vocabulary: usize,
    pub margin: f64,
    pub preferred_nll: f64,
    pub other_nll: f64,
    /// log p(target | preferred) - log p(target | other), before rounding either NLL.
    pub log_probability_gap: f64,
    pub loss: f64,
    /// Multiplier on (preferred CE gradient - other CE gradient); no row-mean divisor.
    pub gradient_weight: f64,
}

impl ContextTokenPreference {
    pub fn validate_domain(vocabulary: usize, target: usize, margin: f64) -> PureResult<()> {
        TargetClassScore::validate_domain(vocabulary, target)?;
        if !margin.is_finite() || margin < 0. {
            return Err(TensorError::InvalidValue {
                label: "context preference needs a finite nonnegative log margin",
            });
        }
        Ok(())
    }
}

// Keep short nonoverlapping residuals: subtracting already-rounded NLLs can erase
// the entire log-ratio when the target is very unlikely in both contexts.
fn expansion_sum<const N: usize>(terms: [f64; N]) -> f64 {
    let mut parts = [0.; N];
    let mut n = 0;
    for mut x in terms {
        let mut kept = 0;
        for j in 0..n {
            let mut y: f64 = parts[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0. {
                parts[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        parts[kept] = x;
        n = kept + 1;
    }
    parts[..n].iter().sum()
}

impl Tensor {
    fn context_preference_rows(
        &self,
        target: usize,
        margin: f64,
    ) -> PureResult<(Self, [RowPartition; 2], ContextTokenPreference)> {
        let (rows, cols) = self.shape();
        ContextTokenPreference::validate_domain(cols, target, margin)?;
        if rows != 2 {
            return Err(TensorError::InvalidValue {
                label: "context preference requires exactly two logit rows",
            });
        }
        let input = finite_row_major(self, "context preference logits")?;
        let (a, b) = input.data().split_at(cols);
        let (p, q) = (RowPartition::new(a), RowPartition::new(b));
        let (x, y) = (f64::from(a[target]), f64::from(b[target]));
        let (u, v) = (p.log_partition(), q.log_partition());
        let gap = expansion_sum([q.maximum, -y, v, -p.maximum, x, -u]);
        let d = expansion_sum([p.maximum, -x, u, -q.maximum, y, -v, margin]);
        let tail = (-d.abs()).exp();
        let report = ContextTokenPreference {
            target_id: target,
            vocabulary: cols,
            margin,
            preferred_nll: (p.maximum - x) + u,
            other_nll: (q.maximum - y) + v,
            log_probability_gap: gap,
            loss: d.max(0.) + tail.ln_1p(),
            gradient_weight: if d >= 0. {
                1. / (1. + tail)
            } else {
                tail / (1. + tail)
            },
        };
        Ok((input, [p, q], report))
    }

    pub fn context_token_preference(
        &self,
        target: usize,
        margin: f64,
    ) -> PureResult<ContextTokenPreference> {
        Ok(self.context_preference_rows(target, margin)?.2)
    }

    pub fn context_token_preference_gradient(
        &self,
        target: usize,
        margin: f64,
    ) -> PureResult<(ContextTokenPreference, Self)> {
        let (input, parts, report) = self.context_preference_rows(target, margin)?;
        let mut seed = Vec::with_capacity(input.len());
        for ((row, norm), sign) in input
            .data()
            .chunks_exact(report.vocabulary)
            .zip(parts)
            .zip([1., -1.])
        {
            for (i, &z) in row.iter().enumerate() {
                let ce = if i != target {
                    norm.probability(z)
                } else if i == norm.maximum_index {
                    -norm.tail_probability()
                } else {
                    norm.probability(z) - 1.
                };
                seed.push((sign * report.gradient_weight * ce) as f32);
            }
        }
        let out = Self::from_vec(2, report.vocabulary, seed)?;
        emit_classification_op("context_token_preference_backward", self, &out, None);
        Ok((report, out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_preference_both_rows_match_independent_finite_differences() {
        let input = [1.1f32, -0.25, 0.5, -0.75, 1.3, 0.4];
        let direct = |j: usize, h: f64| {
            let mut v = input.map(f64::from);
            v[j] += h;
            let ce = |r: &[f64]| r.iter().map(|z| z.exp()).sum::<f64>().ln() - r[1];
            (0.3 + ce(&v[..3]) - ce(&v[3..])).exp().ln_1p()
        };
        let x = Tensor::from_vec(2, 3, input.to_vec()).unwrap();
        let (r, g) = x.context_token_preference_gradient(1, 0.3).unwrap();
        assert_eq!(r, x.context_token_preference(1, 0.3).unwrap());
        assert!((r.loss - direct(0, 0.)).abs() < 1e-12);
        for (j, &v) in g.data().iter().enumerate() {
            assert!((f64::from(v) - (direct(j, 1e-5) - direct(j, -1e-5)) / 2e-5).abs() < 4e-8);
        }
        assert!(g.data()[1] < 0. && g.data()[4] > 0.);
        let (s, h) = x
            .to_layout(Layout::ColMajor)
            .unwrap()
            .context_token_preference_gradient(1, 0.3)
            .unwrap();
        assert_eq!(r, s);
        assert_eq!(g.data(), h.data());
    }

    #[test]
    fn context_preference_recovers_gap_hidden_by_huge_common_cost() {
        let m = f32::MAX;
        let x = Tensor::from_vec(2, 3, vec![m, m, -m, m, 0., -m]).unwrap();
        let (r, g) = x.context_token_preference_gradient(2, 0.).unwrap();
        assert_eq!(r.preferred_nll, r.other_nll);
        assert!((r.log_probability_gap + 2f64.ln()).abs() < 1e-14);
        assert!((r.loss - 3f64.ln()).abs() < 1e-14);
        assert!((r.gradient_weight - 2. / 3.).abs() < 1e-14);
        assert_eq!(g.data()[2], -g.data()[5]);
    }

    #[test]
    fn context_preference_retains_confident_target_tails_and_rejects_invalid_domains() {
        let x = Tensor::from_vec(2, 2, vec![0., -40., 0., -40.]).unwrap();
        let (r, g) = x.context_token_preference_gradient(0, 0.).unwrap();
        assert_eq!(r.loss, 2f64.ln());
        assert!(g.data()[0] < 0. && g.data()[0] == -g.data()[1] && g.data()[2] == -g.data()[0]);
        let saturated = Tensor::from_vec(2, 2, vec![0., 0., -1200., 0.])
            .unwrap()
            .context_token_preference(0, 0.)
            .unwrap();
        assert!(saturated.loss.is_finite() && saturated.loss < 1e-300);
        for margin in [-1., f64::NAN, f64::INFINITY] {
            assert!(x.context_token_preference(0, margin).is_err());
        }
        assert!(x.context_token_preference(2, 0.).is_err());
        assert!(Tensor::zeros(1, 2)
            .unwrap()
            .context_token_preference(0, 0.)
            .is_err());
        assert!(Tensor::from_vec(2, 2, vec![0., 1., 0., f32::NAN])
            .unwrap()
            .context_token_preference(0, 0.)
            .is_err());
    }
}
