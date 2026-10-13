// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact reduction of seed * router_weight * component_weight * component.
//!
//! Four finite f32 factors have quantum 2^-596. The two weights are in `[0,1]`,
//! so each product is below 2^256. A checked u32 input length permits fewer
//! than 2^33 signed terms: their absolute sum is below 2^289. The signed
//! [-608,320) bit window therefore loses neither tails nor intermediate sums.
//! This is specific to the mixture contract, not a general f32 product sum.

#[derive(Clone, Default)]
pub(super) struct RouterSum {
    words: [u32; 29],
}

impl RouterSum {
    pub(super) fn product(&mut self, factors: [f32; 4], subtract: bool) {
        debug_assert!(factors.iter().all(|x| x.is_finite()));
        debug_assert!((0.0..=1.0).contains(&factors[1]));
        debug_assert!((0.0..=1.0).contains(&factors[2]));
        let mut negative = subtract;
        let mut exponent = 608i32;
        let mut product = 1u128;
        for factor in factors {
            let bits = factor.to_bits();
            negative ^= bits >> 31 != 0;
            let encoded = (bits >> 23) & 255;
            let mut mantissa = bits & 0x7fffff;
            exponent += if encoded == 0 {
                -149
            } else {
                encoded as i32 - 150
            };
            if encoded != 0 {
                mantissa |= 0x800000;
            }
            if mantissa == 0 {
                return;
            }
            product *= u128::from(mantissa);
        }
        let shift = exponent as usize;
        for chunk in 0..3 {
            let digit = (product >> (32 * chunk)) as u32;
            let offset = shift + 32 * chunk;
            self.word(offset / 32, digit << (offset % 32), negative);
            if !offset.is_multiple_of(32) {
                self.word(offset / 32 + 1, digit >> (32 - offset % 32), negative);
            }
        }
    }

    fn word(&mut self, mut index: usize, mut value: u32, subtract: bool) {
        while value != 0 && index < self.words.len() {
            let (next, carry) = if subtract {
                self.words[index].overflowing_sub(value)
            } else {
                self.words[index].overflowing_add(value)
            };
            self.words[index] = next;
            value = u32::from(carry);
            index += 1;
        }
    }

    pub(super) fn total(&self) -> f32 {
        let sign = self.words[28] & 0x80000000;
        let mut magnitude = self.words;
        if sign != 0 {
            let mut carry = true;
            for word in &mut magnitude {
                let (value, next) = (!*word).overflowing_add(u32::from(carry));
                *word = value;
                carry = next;
            }
        }
        let Some(index) = magnitude.iter().rposition(|&word| word != 0) else {
            return 0.0;
        };
        let highest = index * 32 + 31 - magnitude[index].leading_zeros() as usize;
        let mut exponent = highest as i32 - 608;
        if exponent > 127 {
            return f32::from_bits(sign | 0x7f800000);
        }
        // Round once, directly to f32; routing through f64 can double-round a
        // midpoint whose only tie-breaking bit is a remote cancellation tail.
        let shift = highest.saturating_sub(23).max(608 - 149);
        let mut mantissa = magnitude[shift / 32] >> (shift % 32);
        if !shift.is_multiple_of(32) && shift / 32 + 1 < magnitude.len() {
            mantissa |= magnitude[shift / 32 + 1] << (32 - shift % 32);
        }
        let guard = shift - 1;
        let guard_set = magnitude[guard / 32] & (1 << (guard % 32)) != 0;
        let sticky = magnitude[..guard / 32].iter().any(|&word| word != 0)
            || magnitude[guard / 32] & ((1 << (guard % 32)) - 1) != 0;
        if guard_set && (sticky || mantissa & 1 != 0) {
            mantissa += 1;
        }
        if exponent < -126 {
            return f32::from_bits(sign | mantissa);
        }
        if mantissa == 0x1000000 {
            mantissa >>= 1;
            exponent += 1;
        }
        if exponent > 127 {
            return f32::from_bits(sign | 0x7f800000);
        }
        f32::from_bits(sign | ((exponent + 127) as u32) << 23 | (mantissa & 0x7fffff))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_tail_breaks_normal_and_subnormal_ties() {
        for negative in [false, true] {
            for direction in [-1, 0, 1] {
                for (base, half_ulp, expected) in [
                    (1.0, [2f32.powi(-24), 1., 1., 1.], 1.0f32.to_bits()),
                    (0., [f32::from_bits(1), 0.5, 1., 1.], 0),
                    (f32::from_bits(2), [f32::from_bits(1), 0.5, 1., 1.], 2),
                ] {
                    let mut sum = RouterSum::default();
                    sum.product([base, 1., 1., 1.], negative);
                    sum.product(half_ulp, negative);
                    if direction != 0 {
                        sum.product([f32::from_bits(1); 4], negative ^ (direction < 0));
                    }
                    let bits = (expected + u32::from(direction > 0)) | (u32::from(negative) << 31);
                    assert_eq!(sum.total().to_bits(), bits);
                }
            }
        }
    }

    #[test]
    fn overflow_midpoint_observes_remote_sticky_bits() {
        for direction in [-1, 0, 1] {
            let mut sum = RouterSum::default();
            sum.product([f32::MAX, 1., 1., 1.], false);
            sum.product([2f32.powi(103), 1., 1., 1.], false);
            if direction != 0 {
                sum.product([f32::from_bits(1); 4], direction < 0);
            }
            assert_eq!(
                sum.total(),
                if direction < 0 {
                    f32::MAX
                } else {
                    f32::INFINITY
                }
            );
        }
    }

    #[test]
    fn products_retain_all_96_bits_and_cancel_before_overflow() {
        // (1+u)^4 - 1 - 4u - 6u^2 - 4u^3 = u^4, u=2^-23.
        // The final bit is 92 places below the leading bit of the first term.
        let mut polynomial = RouterSum::default();
        let a = f32::from_bits(1.0f32.to_bits() + 1);
        polynomial.product([4. * a, 0.5 * a, 0.5 * a, a], false);
        for term in [1., 2f32.powi(-21), 6. * 2f32.powi(-46), 2f32.powi(-67)] {
            polynomial.product([term, 1., 1., 1.], true);
        }
        assert_eq!(polynomial.total(), 2f32.powi(-92));

        let mut sum = RouterSum::default();
        sum.product([f32::MAX, 1., 1., f32::MAX], false);
        sum.product([f32::from_bits(1), 0.5, 0.5, 4.], false);
        sum.product([f32::MAX, 1., 1., f32::MAX], true);
        assert_eq!(sum.total().to_bits(), 1);
        sum.product([f32::MAX, 1., 1., 2.], false);
        assert_eq!(sum.total(), f32::INFINITY);
    }

    #[test]
    fn exact_zero_and_underflow_keep_the_expected_sign() {
        let mut sum = RouterSum::default();
        sum.product([f32::from_bits(1); 4], true);
        assert_eq!(sum.total().to_bits(), (-0.0f32).to_bits());
        sum.product([f32::from_bits(1); 4], false);
        assert_eq!(sum.total().to_bits(), 0);
    }
}
