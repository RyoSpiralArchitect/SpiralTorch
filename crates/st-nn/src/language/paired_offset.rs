// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fit one input-independent offset to a two-token logit contrast.
//! No paired input, labels or second model are needed when using the fitted offset.

use std::fmt;

/// Each row holds the same two candidate logits, in the same order. In row 0 the
/// first candidate is desired; in row 1 the second candidate is desired.
pub type PairedTokenLogits = [[f32; 2]; 2];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OffsetError {
    EmptyCalibration,
    UnmatchedRows,
    NonFinite,
}

impl fmt::Display for OffsetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::EmptyCalibration => "at least one calibration pair is required",
            Self::UnmatchedRows => "source and reference must cover the same pairs",
            Self::NonFinite => "logits and offset must be finite",
        })
    }
}

impl std::error::Error for OffsetError {}

fn gap([a, b]: [f32; 2]) -> Result<f64, OffsetError> {
    if a.is_finite() && b.is_finite() {
        Ok(f64::from(a) - f64::from(b))
    } else {
        Err(OffsetError::NonFinite)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PairedTokenOffset(f64);

impl PairedTokenOffset {
    pub fn new(value: f64) -> Result<Self, OffsetError> {
        if value.is_finite() {
            Ok(Self(if value == 0. { 0. } else { value }))
        } else {
            Err(OffsetError::NonFinite)
        }
    }

    pub fn value(self) -> f64 {
        self.0
    }

    /// Add to the *contrast* in f64. This does not modify vocabulary logits or probabilities.
    pub fn adjusted_gap(self, logits: [f32; 2]) -> Result<f64, OffsetError> {
        Ok(gap(logits)? + self.0)
    }

    pub fn both_correct(self, pair: PairedTokenLogits) -> Result<bool, OffsetError> {
        let [p, n] = pair;
        let (p, n) = (self.adjusted_gap(p)?, self.adjusted_gap(n)?);
        Ok(p > 0. && n < 0.)
    }

    /// Least-squares constant for predicting the reference's contrasts from the source's.
    /// Caller must supply matched rows; this control never uses desired token labels.
    pub fn match_reference(
        source: &[PairedTokenLogits],
        reference: &[PairedTokenLogits],
    ) -> Result<Self, OffsetError> {
        if source.len() != reference.len() {
            return Err(OffsetError::UnmatchedRows);
        }
        if source.is_empty() {
            return Err(OffsetError::EmptyCalibration);
        }
        let (mut sum, mut correction) = (0_f64, 0_f64);
        for (s, r) in source.iter().zip(reference) {
            for (&s, &r) in s.iter().zip(r) {
                // Accumulate the two signed gaps separately. Forming a rounded
                // difference first can erase a small term before compensation.
                for term in [gap(r)?, -gap(s)?] {
                    let next = sum + term;
                    correction += if sum.abs() >= term.abs() {
                        (sum - next) + term
                    } else {
                        (term - next) + sum
                    };
                    sum = next;
                }
            }
        }
        Self::new((sum + correction) / (2. * source.len() as f64))
    }

    /// Maximize strict both-correct coverage in O(n log n). Prefer zero if optimal;
    /// otherwise take the midpoint of the widest maximum-coverage elementary interval.
    /// Equal widths favor the center nearest zero, then the lesser signed center.
    pub fn fit_coverage(pairs: &[PairedTokenLogits]) -> Result<OffsetCoverage, OffsetError> {
        if pairs.is_empty() {
            return Err(OffsetError::EmptyCalibration);
        }
        let mut events = Vec::new();
        let mut zero_hits = 0;
        let mut separable = 0;
        for &[p, n] in pairs {
            let (p, n) = (gap(p)?, gap(n)?);
            if p > n {
                separable += 1;
                zero_hits += usize::from(p > 0. && n < 0.);
                events.push((-p, true));
                events.push((-n, false));
            }
        }
        events.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let (mut i, mut active) = (0, 0_usize);
        let mut best = Choice {
            hits: 0,
            width: 0.,
            offset: 0.,
        };
        while i < events.len() {
            let low = events[i].0;
            let (mut starts, mut ends) = (0, 0);
            while i < events.len() && events[i].0 == low {
                if events[i].1 {
                    starts += 1
                } else {
                    ends += 1
                }
                i += 1;
            }
            // Open endpoints: intervals ending here no longer count, and new ones do not count yet.
            active -= ends;
            best.consider(active, 0., low);
            active += starts;
            if let Some(&(high, _)) = events.get(i) {
                let width = high - low;
                let mid = low + width * 0.5;
                if low < mid && mid < high {
                    best.consider(active, width, mid);
                }
            }
        }
        Ok(OffsetCoverage {
            offset: Self::new(if best.hits == zero_hits {
                0.
            } else {
                best.offset
            })?,
            pairs: pairs.len(),
            separable,
            both_correct: best.hits,
        })
    }
}

struct Choice {
    hits: usize,
    width: f64,
    offset: f64,
}

impl Choice {
    fn consider(&mut self, hits: usize, width: f64, offset: f64) {
        if hits > self.hits
            || (hits == self.hits
                && (width > self.width
                    || (width == self.width
                        && (offset.abs() < self.offset.abs()
                            || (offset.abs() == self.offset.abs() && offset < self.offset)))))
        {
            *self = Self {
                hits,
                width,
                offset,
            };
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OffsetCoverage {
    pub offset: PairedTokenOffset,
    pub pairs: usize,
    /// Pairs with the correct contrast ordering before any scalar shift.
    pub separable: usize,
    pub both_correct: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(left: f32, right: f32) -> PairedTokenLogits {
        [[0., left], [0., right]]
    }

    #[test]
    fn strict_endpoints_and_shared_boundaries_are_not_hits() {
        let rows = [
            interval(1., 3.),
            interval(3., 5.),
            interval(2., 4.),
            interval(5., 5.),
            interval(7., 6.),
        ];
        let fit = PairedTokenOffset::fit_coverage(&rows).unwrap();
        assert_eq!(
            (fit.offset.value(), fit.both_correct, fit.separable),
            (2.5, 2, 3)
        );
        let at_three = PairedTokenOffset::new(3.).unwrap();
        assert_eq!(
            rows.into_iter()
                .filter(|&r| at_three.both_correct(r).unwrap())
                .count(),
            1
        );
    }

    #[test]
    fn zero_first_then_widest_then_nearest_then_signed() {
        for (rows, v) in [
            (vec![interval(-1., 1.), interval(2., 8.)], 0.),
            (vec![interval(1., 2.), interval(3., 7.)], 5.),
            (vec![interval(-5., -3.), interval(1., 3.)], 2.),
            (vec![interval(-3., -1.), interval(1., 3.)], -2.),
            (vec![interval(2., 1.)], 0.),
        ] {
            assert_eq!(
                PairedTokenOffset::fit_coverage(&rows)
                    .unwrap()
                    .offset
                    .value(),
                v
            );
        }
    }

    #[test]
    fn sweep_agrees_with_independent_exhaustive_thresholds() {
        let inputs: Vec<_> = (-2..=2)
            .flat_map(|l| (-2..=2).map(move |r| interval(l as f32, r as f32)))
            .collect();
        for a in &inputs {
            for b in &inputs {
                for c in &inputs {
                    let rows = [*a, *b, *c];
                    let count = |x| {
                        rows.iter()
                            .filter(|r| PairedTokenOffset(x).both_correct(**r).unwrap())
                            .count()
                    };
                    let fit = PairedTokenOffset::fit_coverage(&rows).unwrap();
                    assert_eq!(
                        fit.both_correct,
                        (-5..=5).map(|i| count(f64::from(i) * 0.5)).max().unwrap()
                    );
                    assert_eq!(fit.both_correct, count(fit.offset.value()));
                }
            }
        }
    }

    #[test]
    fn handles_subnormal_and_extreme_finite_f32_without_narrowing_the_offset() {
        let u = f32::from_bits(1);
        let fit = PairedTokenOffset::fit_coverage(&[interval(u, 2. * u)]).unwrap();
        assert_eq!(fit.offset.value(), 1.5 * f64::from(u));
        let row = [[-f32::MAX, 0.], [-f32::MAX, f32::MAX]];
        let fit = PairedTokenOffset::fit_coverage(&[row]).unwrap();
        assert!(fit.offset.value() > f64::from(f32::MAX) && fit.offset.both_correct(row).unwrap());
    }

    #[test]
    fn reference_mean_and_score_have_the_named_sign_and_keep_small_terms() {
        let src = [[[0., 0.]; 2]; 2];
        let reference = [[[1e25, 0.], [1., 0.]], [[-1e25, 0.], [3., 0.]]];
        let fit = PairedTokenOffset::match_reference(&src, &reference).unwrap();
        assert_eq!(fit.value(), 1.);
        assert_eq!(fit.adjusted_gap([2., 4.]), Ok(-1.));
        assert_eq!(
            PairedTokenOffset::match_reference(&reference, &src)
                .unwrap()
                .value(),
            -1.
        );
        let src = [[[1., 0.], [0., 0.]]];
        let reference = [[[1e25, 0.], [-1e25, 0.]]];
        assert_eq!(
            PairedTokenOffset::match_reference(&src, &reference)
                .unwrap()
                .value(),
            -0.5
        );
    }

    #[test]
    fn adjacent_f64_endpoints_do_not_offer_a_representable_offset() {
        let pair = [[-1., 0.], [-1., 2_f32.powi(-52)]];
        let fit = PairedTokenOffset::fit_coverage(&[pair]).unwrap();
        assert_eq!(
            (fit.both_correct, fit.separable, fit.offset.value()),
            (0, 1, 0.)
        );
    }

    #[test]
    fn rejects_invalid_data_in_either_lane_even_if_first_lane_is_wrong() {
        assert_eq!(
            PairedTokenOffset::fit_coverage(&[]),
            Err(OffsetError::EmptyCalibration)
        );
        assert_eq!(
            PairedTokenOffset::match_reference(&[], &[interval(1., 2.)]),
            Err(OffsetError::UnmatchedRows)
        );
        assert!(PairedTokenOffset::new(f64::INFINITY).is_err());
        let bad = [[0., 1.], [f32::NAN, 0.]];
        assert!(PairedTokenOffset::default().both_correct(bad).is_err());
        assert!(PairedTokenOffset::fit_coverage(&[bad]).is_err());
        assert!(PairedTokenOffset::match_reference(&[bad], &[interval(1., 2.)]).is_err());
    }
}
