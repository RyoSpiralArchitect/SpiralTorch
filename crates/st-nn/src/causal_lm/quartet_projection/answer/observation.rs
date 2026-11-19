use super::*;
use crate::causal_lm::scalar_output::logistic;

/// Read-only decomposition of a full-vocabulary answer loss at one projection.
#[derive(Clone, Debug, Serialize)]
pub struct CausalLmBindingGradientObservation {
    pub projection_id: String,
    /// The original CE, quartet interaction, and their actual weighted mixture.
    pub training: CausalLmAnswerProjectionStep,
    /// Negative log probability of either candidate, with its own pullback.
    pub candidate_mass: CausalLmLossGradient,
    /// Gold CE conditional on those candidates, from the contrast pullback.
    pub conditional: CausalLmLossGradient,
    pub prompt_forwards: usize,
    pub projection_pullbacks: usize,
}

pub(super) struct PairObservation {
    mass_loss: sum::Sum,
    pair_loss: sum::Sum,
    mass: Vec<sum::Sum>,
    pair: Vec<sum::Sum>,
}

impl PairObservation {
    pub(super) fn new(width: usize) -> Self {
        let zeros = || (0..width).map(|_| sum::Sum::default()).collect();
        Self {
            mass_loss: sum::Sum::default(),
            pair_loss: sum::Sum::default(),
            mass: zeros(),
            pair: zeros(),
        }
    }

    pub(super) fn add(
        &mut self,
        mass_loss: f64,
        mass_grad: Vec<f64>,
        gap: f64,
        sign: f64,
        gap_grad: &[f64],
    ) {
        self.mass_loss.add(mass_loss);
        let (loss, slope) = logistic(gap * sign);
        self.pair_loss.add(loss);
        for (s, g) in self.mass.iter_mut().zip(mass_grad) {
            s.add(g);
        }
        for (s, &g) in self.pair.iter_mut().zip(gap_grad) {
            s.add(slope * sign * g);
        }
    }

    pub(super) fn finish(
        self,
        projection_id: String,
        training: CausalLmAnswerProjectionStep,
    ) -> CausalLmResult<CausalLmBindingGradientObservation> {
        let n = training.prompt_forwards;
        let mean = |loss: sum::Sum, sums: Vec<sum::Sum>| {
            CausalLmLossGradient::new(
                loss.value() / n as f64,
                sums.into_iter().map(|s| s.value() / n as f64).collect(),
            )
        };
        Ok(CausalLmBindingGradientObservation {
            projection_id,
            candidate_mass: mean(self.mass_loss, self.mass)?,
            conditional: mean(self.pair_loss, self.pair)?,
            training,
            prompt_forwards: n,
            projection_pullbacks: n * 3,
        })
    }
}

pub(super) struct CandidateMassSeed {
    pub(super) loss: f64,
    pub(super) outside_share: f64,
    outside: Vec<f64>,
    outside_den: f64,
    pair: [usize; 2],
    second: f64,
}

fn add_weight_difference(sum: &mut sum::Sum, w: f64, base: f64, scale: f64) {
    let difference = w - base;
    // TwoDiff retains the low component before weighting and cancellation.
    let virtual_base = w - difference;
    let residual = (w - (difference + virtual_base)) + (virtual_base - base);
    sum.add(scale * difference);
    if residual != 0. {
        sum.add(scale * residual);
    }
}

impl CandidateMassSeed {
    pub(super) fn readout_upstream(&self, model: &NativeCausalLm) -> CausalLmResult<Tensor> {
        // The native GPT-2 loader packs each vocabulary row as one contiguous column.
        let head = &model.head;
        let width = head.inner();
        if self.outside.len() != head.cols()
            || self.pair.iter().any(|&i| i >= head.cols())
            || width == 0
        {
            return Err("invalid binding observation readout".into());
        }
        if self.outside_den == 0. {
            return Ok(Tensor::zeros(1, width)?);
        }
        let [first, second] = self
            .pair
            .map(|i| &head.as_slice()[i * width..(i + 1) * width]);
        let mut sums: Vec<_> = (0..width).map(|_| sum::Sum::default()).collect();
        // Average all outside-inside differences, normalizing only after the
        // numerator. Rounded probabilities need not sum to one.
        for (&outside, column) in self.outside.iter().zip(head.as_slice().chunks_exact(width)) {
            if outside == 0. {
                continue;
            }
            for (((s, &w), &a), &b) in sums.iter_mut().zip(column).zip(first).zip(second) {
                let w = f64::from(w);
                let mut pair = sum::Sum::default();
                add_weight_difference(&mut pair, w, f64::from(a), 1.);
                if self.second != 0. {
                    add_weight_difference(&mut pair, w, f64::from(b), self.second);
                }
                s.add_scaled(&pair, outside);
            }
        }
        let den = self.outside_den * (1. + self.second);
        let upstream: Vec<_> = sums.iter().map(|s| (s.value() / den) as f32).collect();
        if upstream.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite binding observation readout adjoint".into());
        }
        Ok(Tensor::from_vec(1, width, upstream)?)
    }
}

/// Factor the gradient of -ln(P(a)+P(b)) into a normalized VJP and an
/// outside share. Apply the share in f64 after the decoder pullback.
pub(super) fn candidate_mass_seed(
    z: &[f32],
    [a, b]: [usize; 2],
) -> CausalLmResult<CandidateMassSeed> {
    if z.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite binding observation logits".into());
    }
    let [a, b] = if z[a] > z[b] || (z[a] == z[b] && a < b) {
        [a, b]
    } else {
        [b, a]
    };
    let second = (f64::from(z[b]) - f64::from(z[a])).exp();
    let outside_max = z
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != a && i != b)
        .map(|(_, &v)| f64::from(v))
        .fold(f64::NEG_INFINITY, f64::max);
    if outside_max == f64::NEG_INFINITY {
        return Ok(CandidateMassSeed {
            loss: 0.,
            outside_share: 0.,
            outside: vec![0.; z.len()],
            outside_den: 0.,
            pair: [a, b],
            second,
        });
    }
    let outside: Vec<_> = z
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i == a || i == b {
                0.
            } else {
                (f64::from(v) - outside_max).exp()
            }
        })
        .collect();
    let mut accumulator = sum::Sum::default();
    for &value in &outside {
        accumulator.add(value);
    }
    let outside_den = accumulator.value();
    let outside_vs_pair = (outside_max - f64::from(z[a])) + outside_den.ln() - second.ln_1p();
    let (loss, negative_share) = logistic(-outside_vs_pair);
    Ok(CandidateMassSeed {
        loss,
        outside_share: -negative_share,
        outside,
        outside_den,
        pair: [a, b],
        second,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_readout_retains_small_differences_regardless_of_candidate_order() {
        let mut m = crate::causal_lm::tests::load();
        let large = 2f32.powi(60);
        for (logits, weights, expected) in [
            (vec![-60., 0., 0.], vec![large, 1., 2.], 1.),
            (vec![0., -60., 0., 0.], vec![large, 1., 2. * large, 2.], 1.),
            // The outside normalizer rounds to 1, but the two inside
            // contributions have to cancel for every individual outsider.
            (vec![0., 0., 0., -40.], vec![large, -large, 0., 0.], 0.),
        ] {
            let head = Tensor::from_vec(weights.len(), 1, weights).unwrap();
            m.head = PackedB::from_tensor_transpose(&head, st_tensor::Tile::col_major()).unwrap();
            for ids in [[0, 1], [1, 0]] {
                let seed = candidate_mass_seed(&logits, ids).unwrap();
                let upstream = seed.readout_upstream(&m).unwrap();
                assert_eq!(upstream.data(), &[expected], "{logits:?}, {ids:?}");
            }
        }
    }

    #[test]
    fn saturated_pair_mass_keeps_the_outside_probability_and_seed() {
        let seed = candidate_mass_seed(&[45., 44., 0.], [0, 1]).unwrap();
        assert!(seed.loss > 0. && seed.loss < 1e-19);
        assert_eq!(seed.outside, [0., 0., 1.]);
        assert_eq!(seed.pair, [0, 1]);
        assert_eq!(seed.second, (-1_f64).exp());
        let empty = candidate_mass_seed(&[45., 44.], [1, 0]).unwrap();
        assert_eq!(empty.loss, 0.);
        assert_eq!(empty.outside_share, 0.);
        assert_eq!(empty.outside, [0., 0.]);
        assert_eq!(empty.outside_den, 0.);
        assert!(candidate_mass_seed(&[0., f32::NAN, 0.], [0, 1]).is_err());
    }

    #[test]
    fn outside_share_can_be_below_f32_range_without_losing_the_seed() {
        let seed = candidate_mass_seed(&[0., 0., -104.], [0, 1]).unwrap();
        assert!(seed.loss > 0. && seed.outside_share > 0.);
        assert_eq!(seed.outside_share as f32, 0.);
        assert_eq!(seed.outside, [0., 0., 1.]);
        assert_eq!(seed.second, 1.);
    }
}
