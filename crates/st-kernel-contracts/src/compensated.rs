// SPDX-License-Identifier: AGPL-3.0-or-later
//! Private finite reductions that retain small terms across cancellation.

#[derive(Clone, Default)]
pub(crate) struct Sum {
    partials: Vec<f64>,
}

impl Sum {
    pub(crate) fn add(&mut self, mut x: f64) {
        // A non-overlapping expansion, not one uncompensated correction.
        // Keep every lost term until later cancellation has participated.
        let mut count = 0;
        for i in 0..self.partials.len() {
            let mut y = self.partials[i];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let high = x + y;
            let low = y - (high - x);
            if low != 0. {
                self.partials[count] = low;
                count += 1;
            }
            x = high;
        }
        self.partials.truncate(count);
        if x != 0. {
            self.partials.push(x);
        }
    }

    pub(crate) fn product(&mut self, a: f64, b: f64) {
        let product = a * b;
        self.add(product);
        self.add(a.mul_add(b, -product));
    }

    pub(crate) fn total(&self) -> f64 {
        self.partials.iter().sum()
    }

    /// Round the finite expansion directly, including subnormal/overflow ties.
    /// Rounding through f64 first can discard the sign of a tiny midpoint tail.
    pub(crate) fn into_f32(mut self) -> f32 {
        let leading = self.total();
        let candidate = (leading as f32).clamp(-f32::MAX, f32::MAX);
        self.add(-f64::from(candidate));
        let residual = self.partials.last().copied().unwrap_or(0.);
        if residual == 0. {
            return candidate;
        }
        let positive = residual > 0.;
        let bits = candidate.to_bits();
        let next_bits = if bits & 0x7fff_ffff == 0 {
            if positive {
                1
            } else {
                0x8000_0001
            }
        } else if positive == (bits >> 31 == 0) {
            bits + 1
        } else {
            bits - 1
        };
        let next = f32::from_bits(next_bits);
        let half_gap = if next.is_infinite() {
            if positive {
                2f64.powi(103)
            } else {
                -2f64.powi(103)
            }
        } else {
            0.5 * (f64::from(next) - f64::from(candidate))
        };
        self.add(-half_gap);
        let comparison = self.partials.last().copied().unwrap_or(0.);
        if (comparison != 0. && (comparison > 0.) == positive)
            || (comparison == 0. && bits & 1 != 0)
        {
            next
        } else if candidate == 0. && !positive {
            -0.
        } else {
            candidate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_f32_rounding_retains_midpoint_tails() {
        for sign in [1., -1.] {
            let sign_bit = if sign < 0. { 0x8000_0000 } else { 0 };
            for tail in [-1., 0., 1.] {
                for (base, half, bits) in [
                    (1., 2f64.powi(-24), 1f32.to_bits()),
                    (0., 2f64.powi(-150), 0),
                    (2f64.powi(-148), 2f64.powi(-150), 2),
                ] {
                    let mut sum = Sum::default();
                    for value in [base, half, tail * 2f64.powi(-500)] {
                        sum.add(sign * value);
                    }
                    assert_eq!(
                        sum.into_f32().to_bits(),
                        (bits + u32::from(tail > 0.)) | sign_bit
                    );
                }
                let mut sum = Sum::default();
                for value in [f64::from(f32::MAX), 2f64.powi(103), tail] {
                    sum.add(sign * value);
                }
                let bits = if tail < 0. {
                    f32::MAX.to_bits()
                } else {
                    f32::INFINITY.to_bits()
                };
                assert_eq!(sum.into_f32().to_bits(), bits | sign_bit);
            }
        }
    }

    #[test]
    fn nested_cancellation_and_overlapping_retained_parts_round_correctly() {
        for small in [1., f64::from(0.2475f32), 2f64.powi(-149)] {
            let mut sum = Sum::default();
            for value in [
                2f64.powi(300),
                2f64.powi(150),
                small,
                -2f64.powi(150),
                -2f64.powi(300),
            ] {
                sum.add(value);
            }
            assert_eq!(sum.into_f32(), small as f32);
        }
        let mut sum = Sum::default();
        for value in [2f64.powi(46), f64::from(0.2475f32), -2f64.powi(46)] {
            sum.add(value);
        }
        assert_eq!(sum.into_f32(), 0.2475f32);
    }
}
