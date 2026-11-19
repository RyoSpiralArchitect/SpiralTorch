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

struct CandidateMassSeed {
    loss: f64,
    outside_share: f64,
    normalized: Tensor,
}

struct CandidateMassTerms {
    loss: f64,
    outside_share: f64,
    outside_max: f64,
    outside_den: f64,
    pair_weights: [f64; 2],
}

impl NativeCausalLmRankOneProjection {
    pub(super) fn candidate_mass_gradient(
        &self,
        row: projection_probe::ProjectionLastRow,
        pair: [usize; 2],
    ) -> CausalLmResult<(f64, Vec<f64>)> {
        let seed = candidate_mass_seed(row.logits.data(), pair)?;
        let mut gradient =
            self.current
                .fixed_left_seed(row, &seed.normalized, self.spec, &self.left)?;
        for g in &mut gradient {
            *g *= seed.outside_share;
        }
        Ok((seed.loss, gradient))
    }
}

pub(super) fn candidate_mass_loss(z: &[f32], pair: [usize; 2]) -> CausalLmResult<f64> {
    Ok(candidate_mass_terms(z, pair)?.loss)
}

/// Factor the gradient of -ln(P(a)+P(b)) before rounding the seed to f32.
/// Apply the outside share in f64 after the linear pullback.
fn candidate_mass_seed(z: &[f32], [a, b]: [usize; 2]) -> CausalLmResult<CandidateMassSeed> {
    let terms = candidate_mass_terms(z, [a, b])?;
    let seed = if terms.outside_max == f64::NEG_INFINITY {
        vec![0.; z.len()]
    } else {
        z.iter()
            .enumerate()
            .map(|(i, &v)| {
                let g = match i {
                    i if i == a => -terms.pair_weights[0],
                    i if i == b => -terms.pair_weights[1],
                    _ => (f64::from(v) - terms.outside_max).exp() / terms.outside_den,
                };
                g as f32
            })
            .collect()
    };
    Ok(CandidateMassSeed {
        loss: terms.loss,
        outside_share: terms.outside_share,
        normalized: Tensor::from_vec(1, z.len(), seed)?,
    })
}

fn candidate_mass_terms(z: &[f32], [a, b]: [usize; 2]) -> CausalLmResult<CandidateMassTerms> {
    if z.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite binding observation logits".into());
    }
    let outside = || {
        z.iter()
            .enumerate()
            .filter(|&(i, _)| i != a && i != b)
            .map(|(_, &v)| f64::from(v))
    };
    let outside_max = outside().fold(f64::NEG_INFINITY, f64::max);
    let pair_max = f64::from(z[a].max(z[b]));
    let ga = (f64::from(z[a]) - pair_max).exp();
    let gb = (f64::from(z[b]) - pair_max).exp();
    let pair_den = ga + gb;
    let (outside_den, loss, outside_share) = if outside_max == f64::NEG_INFINITY {
        (1., 0., 0.)
    } else {
        let mut tail = sum::Sum::default();
        let mut skipped_max = false;
        for v in outside() {
            if !skipped_max && v == outside_max {
                skipped_max = true;
                continue;
            }
            tail.add((v - outside_max).exp());
        }
        let tail = tail.value();
        let outside_vs_pair = (outside_max - pair_max) + tail.ln_1p() - ga.min(gb).ln_1p();
        let (loss, negative_share) = logistic(-outside_vs_pair);
        (1. + tail, loss, -negative_share)
    };
    Ok(CandidateMassTerms {
        loss,
        outside_share,
        outside_max,
        outside_den,
        pair_weights: [ga / pair_den, gb / pair_den],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturated_pair_mass_keeps_the_outside_probability_and_seed() {
        let seed = candidate_mass_seed(&[45., 44., 0.], [0, 1]).unwrap();
        assert!(seed.loss > 0. && seed.loss < 1e-19);
        let [a, b, outside] = seed.normalized.data() else {
            panic!("expected a three-token seed");
        };
        assert!(*a < 0. && *b < 0. && *outside == 1.);
        assert!((f64::from(*a) + f64::from(*b) + 1.).abs() < 1e-7);
        let empty = candidate_mass_seed(&[45., 44.], [1, 0]).unwrap();
        assert_eq!(empty.loss, 0.);
        assert_eq!(empty.outside_share, 0.);
        assert_eq!(empty.normalized.data(), &[0., 0.]);
        assert!(candidate_mass_seed(&[0., f32::NAN, 0.], [0, 1]).is_err());
    }

    #[test]
    fn outside_share_can_be_below_f32_range_without_losing_the_seed() {
        let seed = candidate_mass_seed(&[0., 0., -104.], [0, 1]).unwrap();
        assert!(seed.loss > 0. && seed.outside_share > 0.);
        assert_eq!(seed.outside_share as f32, 0.);
        assert_eq!(seed.normalized.data(), &[-0.5, -0.5, 1.]);
    }

    #[test]
    fn accumulated_outside_tail_changes_pair_mass_in_either_vocab_order() {
        let loss = |tail, max_first| {
            let mut z = vec![tail; 10_003];
            z[0] = 0.;
            z[1] = -1.;
            z[if max_first { 2 } else { 10_002 }] = 0.;
            candidate_mass_loss(&z, [0, 1]).unwrap()
        };
        let before = loss(-40., true);
        let after = loss(-38., true);
        assert_eq!(before, loss(-40., false));
        assert_eq!(after, loss(-38., false));
        assert!(after - before > 1e-14);
    }
}
