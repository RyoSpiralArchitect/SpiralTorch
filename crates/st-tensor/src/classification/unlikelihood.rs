//! Sparse negative-token loss; planning and strength application belong to callers.
use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct UnlikelihoodPosition {
    pub row: usize,
    pub candidates: Vec<usize>,
}

impl Tensor {
    /// Mean over active rows of each row's mean `-ln(1-min(p,1-epsilon))`.
    /// Returns the loss and its unit-seed VJP. Empty plans produce exact zeros.
    /// f32 storage uses stable f64 normalization; a clamped candidate has zero
    /// derivative above the cap (at equality the unclamped derivative is used).
    pub fn unlikelihood_with_logits(
        &self,
        positions: &[UnlikelihoodPosition],
        epsilon: f64,
    ) -> PureResult<(f32, Tensor)> {
        let (rows, classes) = self.shape();
        if classes == 0
            || !epsilon.is_finite()
            || !(0.0..1.0).contains(&epsilon)
            || epsilon == 0.0
            || 1.0 - epsilon == 1.0
        {
            return Err(TensorError::InvalidValue {
                label: "invalid unlikelihood classes or epsilon",
            });
        }
        let mut seen = BTreeSet::new();
        for position in positions {
            if position.row >= rows
                || !seen.insert(position.row)
                || position.candidates.is_empty()
                || position.candidates.len() > classes
            {
                return Err(TensorError::InvalidValue {
                    label: "invalid unlikelihood active row",
                });
            }
            let mut candidates = BTreeSet::new();
            for &id in &position.candidates {
                if id >= classes || !candidates.insert(id) {
                    return Err(TensorError::InvalidValue {
                        label: "invalid unlikelihood candidate",
                    });
                }
            }
        }
        let input = finite_row_major(self, "unlikelihood_logits")?;
        let mut gradient = vec![0.; input.len()];
        let mut loss = 0.;
        for position in positions {
            let offset = position.row * classes;
            let row = &input.data()[offset..offset + classes];
            let partition = RowPartition::new(row);
            let normalizer = positions.len() as f64 * position.candidates.len() as f64;
            let mut coefficients = Vec::with_capacity(position.candidates.len());
            let mut total = 0.;
            for &candidate in &position.candidates {
                let probability = partition.probability(row[candidate]);
                loss -= (-probability.min(1.0 - epsilon)).ln_1p() / normalizer;
                let coefficient = if probability <= 1.0 - epsilon {
                    probability / (1.0 - probability) / normalizer
                } else {
                    0.
                };
                coefficients.push((candidate, coefficient));
                total += coefficient;
            }
            coefficients.sort_unstable_by_key(|&(id, _)| id);
            let mut coefficients = coefficients.into_iter().peekable();
            for (index, &value) in row.iter().enumerate() {
                let own = if coefficients.peek().is_some_and(|&(id, _)| id == index) {
                    coefficients.next().unwrap().1
                } else {
                    0.
                };
                gradient[offset + index] = checked_f32(
                    own - partition.probability(value) * total,
                    "unlikelihood_gradient",
                )?;
            }
        }
        let loss = checked_f32(loss, "unlikelihood_loss")?;
        let gradient = Tensor::from_vec(rows, classes, gradient)?;
        emit_tensor_op(
            "unlikelihood_with_logits",
            &[rows, classes],
            &[rows, classes],
        );
        Ok((loss, gradient))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlikelihood_uses_separate_row_and_candidate_means_and_clamp_sides() {
        let logits = Tensor::zeros(3, 4).unwrap();
        let (loss, gradient) = logits
            .unlikelihood_with_logits(
                &[
                    UnlikelihoodPosition {
                        row: 0,
                        candidates: vec![1],
                    },
                    UnlikelihoodPosition {
                        row: 2,
                        candidates: vec![0, 3],
                    },
                ],
                1e-6,
            )
            .unwrap();
        assert!((loss - (4f32 / 3.).ln()).abs() < 1e-7);
        for (actual, expected) in gradient.data().iter().zip([
            -1. / 24.,
            1. / 8.,
            -1. / 24.,
            -1. / 24.,
            0.,
            0.,
            0.,
            0.,
            1. / 24.,
            -1. / 24.,
            -1. / 24.,
            1. / 24.,
        ]) {
            assert!((actual - expected).abs() < 1e-8);
        }
        let positions = [UnlikelihoodPosition {
            row: 0,
            candidates: vec![0],
        }];
        for (logit, active) in [(13., true), (14., false), (f32::MAX, false)] {
            let (_, grad) = Tensor::from_vec(1, 2, vec![logit, 0.])
                .unwrap()
                .unlikelihood_with_logits(&positions, 1e-6)
                .unwrap();
            assert_eq!(grad.data()[0] > 0.9, active);
            assert!(grad.data().iter().all(|v| v.is_finite()));
        }
        let (_, grad) = Tensor::zeros(1, 1)
            .unwrap()
            .unlikelihood_with_logits(&positions, 1e-6)
            .unwrap();
        assert_eq!(grad.data(), &[0.]);
    }

    #[test]
    fn unlikelihood_vjp_matches_finite_differences_and_layouts() {
        let values = vec![0.2, -0.1, 0.7, -0.4, 0.8, 0.1, 0., 0., 0.];
        let positions = vec![
            UnlikelihoodPosition {
                row: 0,
                candidates: vec![0, 2],
            },
            UnlikelihoodPosition {
                row: 1,
                candidates: vec![1],
            },
        ];
        let input = Tensor::from_vec(3, 3, values.clone()).unwrap();
        let (loss, gradient) = input.unlikelihood_with_logits(&positions, 1e-6).unwrap();
        let (column_loss, column_gradient) = input
            .to_layout(Layout::ColMajor)
            .unwrap()
            .unlikelihood_with_logits(&positions, 1e-6)
            .unwrap();
        assert_eq!(loss, column_loss);
        assert_eq!(gradient.data(), column_gradient.data());
        for index in 0..values.len() {
            let mut plus = values.clone();
            let mut minus = values.clone();
            plus[index] += 0.001;
            minus[index] -= 0.001;
            let f = |v| {
                Tensor::from_vec(3, 3, v)
                    .unwrap()
                    .unlikelihood_with_logits(&positions, 1e-6)
                    .unwrap()
                    .0
            };
            let numerical = (f(plus) - f(minus)) / 0.002;
            assert!((numerical - gradient.data()[index]).abs() < 4e-5);
        }
        assert_eq!(&gradient.data()[6..], &[0., 0., 0.]);
    }

    #[test]
    fn unlikelihood_empty_clamp_and_invalid_inputs() {
        let input = Tensor::from_vec(1, 2, vec![100., -100.]).unwrap();
        let (loss, zero) = input.unlikelihood_with_logits(&[], 1e-6).unwrap();
        assert_eq!(loss, 0.);
        assert_eq!(zero.data(), &[0., 0.]);
        let active = vec![UnlikelihoodPosition {
            row: 0,
            candidates: vec![0],
        }];
        let (loss, zero) = input.unlikelihood_with_logits(&active, 1e-6).unwrap();
        assert!((loss + (1e-6f32).ln()).abs() < 1e-5);
        assert_eq!(zero.data(), &[0., 0.]);
        for epsilon in [0., 1., -1., f64::NAN, f64::EPSILON / 4.] {
            assert!(input.unlikelihood_with_logits(&active, epsilon).is_err());
        }
        for positions in [
            vec![UnlikelihoodPosition {
                row: 1,
                candidates: vec![0],
            }],
            vec![UnlikelihoodPosition {
                row: 0,
                candidates: vec![],
            }],
            vec![UnlikelihoodPosition {
                row: 0,
                candidates: vec![0, 0],
            }],
            vec![UnlikelihoodPosition {
                row: 0,
                candidates: vec![2],
            }],
            vec![active[0].clone(), active[0].clone()],
        ] {
            assert!(input.unlikelihood_with_logits(&positions, 1e-6).is_err());
        }
        assert!(Tensor::from_vec(1, 2, vec![0., f32::NAN])
            .unwrap()
            .unlikelihood_with_logits(&[], 1e-6)
            .is_err());
    }
}
