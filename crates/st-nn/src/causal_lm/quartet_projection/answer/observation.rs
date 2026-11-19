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

/// Gradient of -ln(P(a)+P(b)). Factoring out the outside mass avoids
/// subtracting two nearly equal probabilities for the inside candidates.
pub(super) fn candidate_mass_seed(z: &[f32], [a, b]: [usize; 2]) -> CausalLmResult<(f64, Tensor)> {
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
    if outside_max == f64::NEG_INFINITY {
        return Ok((0., Tensor::from_vec(1, z.len(), vec![0.; z.len()])?));
    }
    let outside_den = outside().map(|v| (v - outside_max).exp()).sum::<f64>();
    let pair_max = f64::from(z[a].max(z[b]));
    let ga = (f64::from(z[a]) - pair_max).exp();
    let gb = (f64::from(z[b]) - pair_max).exp();
    let pair_den = ga + gb;
    let outside_vs_pair = (outside_max - pair_max) + outside_den.ln() - pair_den.ln();
    let (loss, negative_share) = logistic(-outside_vs_pair);
    let share = -negative_share;
    let scale = share / outside_den;
    let seed = z
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let g = match i {
                i if i == a => -share * ga / pair_den,
                i if i == b => -share * gb / pair_den,
                _ => scale * (f64::from(v) - outside_max).exp(),
            };
            g as f32
        })
        .collect();
    Ok((loss, Tensor::from_vec(1, z.len(), seed)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturated_pair_mass_keeps_the_outside_probability_and_seed() {
        let (loss, seed) = candidate_mass_seed(&[45., 44., 0.], [0, 1]).unwrap();
        assert!(loss > 0. && loss < 1e-19);
        assert!(seed.data()[0] < 0. && seed.data()[1] < 0. && seed.data()[2] > 0.);
        assert!(f64::from(seed.data().iter().sum::<f32>()).abs() < 1e-26);
        let (loss, seed) = candidate_mass_seed(&[45., 44.], [1, 0]).unwrap();
        assert_eq!(loss, 0.);
        assert_eq!(seed.data(), &[0., 0.]);
        assert!(candidate_mass_seed(&[0., f32::NAN, 0.], [0, 1]).is_err());
    }
}
