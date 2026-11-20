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

struct CandidateMassTerms {
    loss: f64,
    log_odds: Option<f64>,
    outside_share: f64,
    outside_max: f64,
    outside_den: f64,
    pair: [usize; 2],
    second: f64,
}

#[derive(Clone, Copy)]
pub(super) struct CandidateMassMeasure {
    pub(super) loss: f64,
    pub(super) log_loss: Option<f64>,
    pub(super) log_outside: Option<f64>,
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

impl CandidateMassTerms {
    fn measure(&self) -> CandidateMassMeasure {
        CandidateMassMeasure {
            loss: self.loss,
            log_outside: self.log_odds.map(|odds| -logistic(odds).0),
            // In this tail, log(softplus(x)) rounds to x before softplus itself underflows.
            log_loss: self
                .log_odds
                .map(|odds| if odds < -40. { odds } else { self.loss.ln() }),
        }
    }

    fn readout_upstream(&self, model: &NativeCausalLm, logits: &[f32]) -> CausalLmResult<Vec<f64>> {
        // The native GPT-2 loader packs each vocabulary row as one contiguous column.
        let head = &model.head;
        let width = head.inner();
        if logits.len() != head.cols() || self.pair.iter().any(|&i| i >= head.cols()) || width == 0
        {
            return Err("invalid binding observation readout".into());
        }
        if self.outside_den == 0. {
            return Ok(vec![0.; width]);
        }
        let [first, second] = self
            .pair
            .map(|i| &head.as_slice()[i * width..(i + 1) * width]);
        let mut sums: Vec<_> = (0..width).map(|_| sum::Sum::default()).collect();
        // Average all outside-inside differences before dividing by the normalizer.
        for (i, (&logit, column)) in logits
            .iter()
            .zip(head.as_slice().chunks_exact(width))
            .enumerate()
        {
            if self.pair.contains(&i) {
                continue;
            }
            let weight = (f64::from(logit) - self.outside_max).exp();
            if weight == 0. {
                continue;
            }
            for (((s, &w), &a), &b) in sums.iter_mut().zip(column).zip(first).zip(second) {
                let w = f64::from(w);
                let mut pair = sum::Sum::default();
                add_weight_difference(&mut pair, w, f64::from(a), 1.);
                if self.second != 0. {
                    add_weight_difference(&mut pair, w, f64::from(b), self.second);
                }
                s.add_scaled(&pair, weight);
            }
        }
        let den = self.outside_den * (1. + self.second);
        let upstream: Vec<_> = sums.iter().map(|s| s.value() / den).collect();
        if upstream.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite binding observation readout adjoint".into());
        }
        Ok(upstream)
    }
}

impl NativeCausalLmRankOneProjection {
    pub(super) fn candidate_mass_gradient(
        &self,
        row: projection_probe::ProjectionLastRow,
        pair: [usize; 2],
    ) -> CausalLmResult<(f64, Vec<f64>)> {
        self.candidate_mass_measured_gradient(row, pair)
            .map(|(measure, gradient)| (measure.loss, gradient))
    }

    pub(super) fn candidate_mass_measured_gradient(
        &self,
        row: projection_probe::ProjectionLastRow,
        pair: [usize; 2],
    ) -> CausalLmResult<(CandidateMassMeasure, Vec<f64>)> {
        let logits = row.logits.data();
        let terms = candidate_mass_terms(logits, pair)?;
        let upstream = terms.readout_upstream(&self.current, logits)?;
        let (mut gradient, readout_scale) = self
            .current
            .fixed_left_wide_upstream(row, &upstream, self.spec, &self.left)?;
        // Apply the outside probability and restore the readout scale in f64.
        for g in &mut gradient {
            *g *= terms.outside_share * readout_scale;
        }
        Ok((terms.measure(), gradient))
    }
}

pub(super) fn candidate_mass_measure(
    z: &[f32],
    pair: [usize; 2],
) -> CausalLmResult<CandidateMassMeasure> {
    Ok(candidate_mass_terms(z, pair)?.measure())
}

fn candidate_mass_terms(z: &[f32], [a, b]: [usize; 2]) -> CausalLmResult<CandidateMassTerms> {
    if z.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite binding observation logits".into());
    }
    let [a, b] = if z[a] > z[b] || (z[a] == z[b] && a < b) {
        [a, b]
    } else {
        [b, a]
    };
    let pair_max = f64::from(z[a]);
    let second = (f64::from(z[b]) - pair_max).exp();
    let outside = || {
        z.iter()
            .enumerate()
            .filter(|&(i, _)| i != a && i != b)
            .map(|(_, &v)| f64::from(v))
    };
    let outside_max = outside().fold(f64::NEG_INFINITY, f64::max);
    let (outside_den, loss, outside_share, log_odds) = if outside_max == f64::NEG_INFINITY {
        (0., 0., 0., None)
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
        let outside_vs_pair = (outside_max - pair_max) + tail.ln_1p() - second.ln_1p();
        let (loss, negative_share) = logistic(-outside_vs_pair);
        (1. + tail, loss, -negative_share, Some(outside_vs_pair))
    };
    Ok(CandidateMassTerms {
        loss,
        log_odds,
        outside_share,
        outside_max,
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
                let terms = candidate_mass_terms(&logits, ids).unwrap();
                let upstream = terms.readout_upstream(&m, &logits).unwrap();
                assert_eq!(upstream.len(), 1);
                assert_eq!(upstream[0] as f32, expected, "{logits:?}, {ids:?}");
            }
        }
    }

    #[test]
    fn finite_scaled_adjoint_does_not_reject_opposite_extreme_weights() {
        let mut model = crate::causal_lm::tests::load();
        let head = Tensor::from_vec(3, 1, vec![f32::MAX, f32::MAX, -f32::MAX]).unwrap();
        model.head = PackedB::from_tensor_transpose(&head, st_tensor::Tile::col_major()).unwrap();
        for tail in [-2., -104.] {
            let logits = [0., 0., tail];
            let terms = candidate_mass_terms(&logits, [0, 1]).unwrap();
            let scaled_expected = -2. * f64::from(f32::MAX) * terms.outside_share;
            assert!(scaled_expected.is_finite() && (scaled_expected as f32).is_normal());
            if tail == -104. {
                assert_eq!(terms.outside_share as f32, 0.);
            }
            let upstream = terms.readout_upstream(&model, &logits).unwrap();
            assert_eq!(upstream[0] * terms.outside_share, scaled_expected);
        }
    }

    #[test]
    fn extreme_readout_with_small_outside_mass_reaches_decoder_projection() {
        let model = crate::causal_lm::tests::load_constant_hidden_with_extreme_readout();
        let projection = NativeCausalLmRankOneProjection::new(
            &model,
            1,
            CausalLmProjectionSite::Output,
            &[1., 0., 0., 0.],
            &[0.; 4],
        )
        .unwrap();
        let row = model.projection_last_row(&[0], projection.spec).unwrap();
        assert_eq!(row.logits.data(), &[0., -1., -104.]);
        let share = candidate_mass_terms(row.logits.data(), [0, 1])
            .unwrap()
            .outside_share;
        assert!(share > 0. && (share as f32) == 0.);
        let factor =
            -2. * f64::from(f32::MAX) * share / f64::from(model.config.norm_epsilon).sqrt();
        let (_, values) = projection
            .candidate_mass_measured_gradient(row, [0, 1])
            .unwrap();
        for (actual, centre) in values.iter().zip([0.75, -0.25, -0.25, -0.25]) {
            let expected = factor * centre;
            assert!(
                (actual / expected - 1.).abs() < 2e-6,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn mixed_scale_readout_keeps_the_small_coordinate_amplified_by_final_norm() {
        let mut model = crate::causal_lm::tests::load_constant_hidden_with_extreme_readout();
        model.final_norm.gamma = Tensor::from_vec(1, 4, vec![0., 2f32.powi(100), 0., 0.]).unwrap();
        model.final_norm.beta = Tensor::from_vec(1, 4, vec![0., 0., 1., 0.]).unwrap();
        let mut head = vec![0.; 12];
        head[6] = -1.;
        head[8] = 2f32.powi(80);
        head[9] = 2f32.powi(-110);
        head[10] = -104.;
        let head = Tensor::from_vec(3, 4, head).unwrap();
        model.head = PackedB::from_tensor_transpose(&head, st_tensor::Tile::col_major()).unwrap();
        let projection = NativeCausalLmRankOneProjection::new(
            &model,
            1,
            CausalLmProjectionSite::Output,
            &[1., 0., 0., 0.],
            &[0.; 4],
        )
        .unwrap();
        let row = model.projection_last_row(&[0], projection.spec).unwrap();
        assert_eq!(row.logits.data(), &[0., -1., -104.]);
        let outside = (-104_f64).exp();
        let share = outside / (1. + (-1_f64).exp() + outside);
        let expected = 0.75 * 2f64.powi(-10) * share / f64::from(model.config.norm_epsilon).sqrt();
        let (_, values) = projection
            .candidate_mass_measured_gradient(row, [0, 1])
            .unwrap();
        assert!(
            (values[1] / expected - 1.).abs() < 2e-6,
            "{} != {expected}",
            values[1]
        );
    }

    #[test]
    fn saturated_pair_mass_keeps_the_outside_probability_and_seed() {
        let seed = candidate_mass_terms(&[45., 44., 0.], [0, 1]).unwrap();
        assert!(seed.loss > 0. && seed.loss < 1e-19);
        assert_eq!(seed.outside_den, 1.);
        assert_eq!(seed.pair, [0, 1]);
        assert_eq!(seed.second, (-1_f64).exp());
        let empty = candidate_mass_terms(&[45., 44.], [1, 0]).unwrap();
        assert_eq!(empty.loss, 0.);
        assert_eq!(empty.outside_share, 0.);
        assert_eq!(empty.outside_den, 0.);
        assert!(candidate_mass_terms(&[0., f32::NAN, 0.], [0, 1]).is_err());
    }

    #[test]
    fn outside_share_can_be_below_f32_range_without_losing_the_seed() {
        let seed = candidate_mass_terms(&[0., 0., -104.], [0, 1]).unwrap();
        assert!(seed.loss > 0. && seed.outside_share > 0.);
        assert_eq!(seed.outside_share as f32, 0.);
        assert_eq!(seed.outside_den, 1.);
        assert_eq!(seed.second, 1.);
    }

    #[test]
    fn accumulated_outside_tail_changes_pair_mass_in_either_vocab_order() {
        let loss = |tail, max_first| {
            let mut z = vec![tail; 10_003];
            z[0] = 0.;
            z[1] = -1.;
            z[if max_first { 2 } else { 10_002 }] = 0.;
            candidate_mass_measure(&z, [0, 1]).unwrap().loss
        };
        let before = loss(-40., true);
        let after = loss(-38., true);
        assert_eq!(before, loss(-40., false));
        assert_eq!(after, loss(-38., false));
        assert!(after - before > 1e-14);
    }
}
