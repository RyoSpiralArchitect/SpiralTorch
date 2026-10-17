// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exactly accumulate finite f32 terms, then round once to nearest, ties to even.
//!
//! Every finite f32 is an integer multiple of 2^-149. Its magnitude in those
//! units is less than 2^277, so at most u32::MAX terms fit in a signed 320-bit
//! accumulator. No floating-point compensation or intermediate rounding is
//! involved in that accumulator. Exact zero is canonicalized to positive zero.
//!
//! For compatibility with guarded training, a nonfinite sequential f32 prefix
//! is still rejected, even if later cancellation would make the exact sum fit.
use thiserror::Error;

/// WGSL arithmetic for finite terms and at most u32::MAX additions. The caller
/// must validate terms, sequential prefixes and the rounded result separately.
pub const WGSL: &str = include_str!("f32_sum.wgsl");

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SumError {
    #[error("f32 sum exceeds u32::MAX terms")]
    TooManyTerms,
    #[error("nonfinite f32 sum input at index {index}")]
    NonFiniteInput { index: usize },
    #[error("nonfinite sequential f32 sum prefix at index {index}")]
    NonFinitePrefix { index: usize },
    #[error("exact sum rounds outside finite f32")]
    NonFiniteSum,
}

/// Correctly rounded sum with the existing sequential-prefix safety guard.
pub fn checked_sum_f32(values: &[f32]) -> Result<f32, SumError> {
    u32::try_from(values.len()).map_err(|_| SumError::TooManyTerms)?;
    let mut words = [0u32; 10];
    let mut prefix = 0f32;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(SumError::NonFiniteInput { index });
        }
        prefix += value;
        if !prefix.is_finite() {
            return Err(SumError::NonFinitePrefix { index });
        }
        add(&mut words, value);
    }
    let result = rounded(words);
    if !result.is_finite() {
        return Err(SumError::NonFiniteSum);
    }
    Ok(result)
}

fn add(words: &mut [u32; 10], value: f32) {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 255;
    let mantissa = (bits & 0x7f_ffff) | if exponent == 0 { 0 } else { 0x80_0000 };
    let shift = exponent.saturating_sub(1);
    let first = (shift / 32) as usize;
    let offset = shift % 32;
    let low = mantissa << offset;
    let high = if offset == 0 {
        0
    } else {
        mantissa >> (32 - offset)
    };
    let negative = bits >> 31 != 0;
    let mut carry = false;
    for (i, word) in words.iter_mut().enumerate().skip(first) {
        let part = if i == first {
            low
        } else if i == first + 1 {
            high
        } else {
            0
        };
        let (sum, c0) = if negative {
            word.overflowing_sub(part)
        } else {
            word.overflowing_add(part)
        };
        let (sum, c1) = if negative {
            sum.overflowing_sub(u32::from(carry))
        } else {
            sum.overflowing_add(u32::from(carry))
        };
        *word = sum;
        carry = c0 || c1;
        if i > first && !carry {
            break;
        }
    }
}

fn rounded(mut magnitude: [u32; 10]) -> f32 {
    let sign = magnitude[9] & 0x8000_0000;
    if sign != 0 {
        let mut carry = true;
        for word in &mut magnitude {
            let (value, overflow) = (!*word).overflowing_add(u32::from(carry));
            *word = value;
            carry = overflow;
        }
    }
    let Some(last) = magnitude.iter().rposition(|&word| word != 0) else {
        return 0.;
    };
    let mut top = last as u32 * 32 + 31 - magnitude[last].leading_zeros();
    if top < 23 {
        return f32::from_bits(sign | magnitude[0]);
    }
    if top >= 277 {
        return f32::from_bits(sign | 0x7f80_0000);
    }
    let shift = top - 23;
    let first = (shift / 32) as usize;
    let offset = shift % 32;
    let mut mantissa = magnitude[first] >> offset;
    if offset != 0 {
        mantissa |= magnitude[first + 1] << (32 - offset);
    }
    if shift != 0 {
        let round_word = ((shift - 1) / 32) as usize;
        let round_bit = (shift - 1) % 32;
        let round = (magnitude[round_word] >> round_bit) & 1 != 0;
        let sticky = magnitude[..round_word].iter().any(|&word| word != 0)
            || magnitude[round_word] & ((1u32 << round_bit) - 1) != 0;
        if round && (sticky || mantissa & 1 != 0) {
            mantissa += 1;
        }
    }
    if mantissa == 1 << 24 {
        mantissa >>= 1;
        top += 1;
    }
    let exponent = top - 22;
    if exponent >= 255 {
        return f32::from_bits(sign | 0x7f80_0000);
    }
    f32::from_bits(sign | (exponent << 23) | (mantissa & 0x7f_ffff))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_zero_subnormals_and_ties() {
        assert_eq!(checked_sum_f32(&[]).unwrap().to_bits(), 0);
        assert_eq!(checked_sum_f32(&[-0., 0.]).unwrap().to_bits(), 0);
        for sign in [1., -1.] {
            for small in [1., 2f32.powi(-25), f32::from_bits(1)] {
                assert_eq!(
                    checked_sum_f32(&[sign * f32::MAX, sign * small, -sign * f32::MAX]).unwrap(),
                    sign * small
                );
            }
            for (base, expected) in [
                (1f32, 1f32),
                (f32::from_bits(0x3f80_0001), f32::from_bits(0x3f80_0002)),
            ] {
                assert_eq!(
                    checked_sum_f32(&[sign * base, sign * 2f32.powi(-24)]).unwrap(),
                    sign * expected
                );
            }
            assert_eq!(
                checked_sum_f32(&[sign * f32::from_bits(0x7f_ffff), sign * f32::from_bits(1)])
                    .unwrap(),
                sign * f32::MIN_POSITIVE
            );
        }
    }

    #[test]
    fn finite_maximum_tail_is_not_prematurely_rounded() {
        for sign in [1., -1.] {
            let values =
                [f32::MAX, 2f32.powi(102), 2f32.powi(102), -2f32.powi(77)].map(|v| sign * v);
            assert_eq!(checked_sum_f32(&values), Ok(sign * f32::MAX));
            assert_eq!(checked_sum_f32(&values[..3]), Err(SumError::NonFiniteSum));
        }
    }

    #[test]
    fn rejects_nonfinite_terms_prefixes_and_final_sum() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                checked_sum_f32(&[1., value]),
                Err(SumError::NonFiniteInput { index: 1 })
            );
        }
        assert_eq!(
            checked_sum_f32(&[f32::MAX, f32::MAX, -f32::MAX]),
            Err(SumError::NonFinitePrefix { index: 1 })
        );
    }

    #[test]
    fn full_u32_term_bound_and_long_carries_fit() {
        // Exact sum of u32::MAX copies of the largest finite value, in 2^-149
        // units. This tests the high limb/sign bound without allocating terms.
        let product = u64::from(u32::MAX) * 0xff_ffff;
        let mut words = [0u32; 10];
        words[7] = (product << 29) as u32;
        words[8] = (product >> 3) as u32;
        words[9] = (product >> 35) as u32;
        assert_eq!(words[9] & 0x8000_0000, 0);
        assert_eq!(rounded(words), f32::INFINITY);
        let mut words = [0u32; 10];
        add(&mut words, -f32::from_bits(1));
        assert_eq!(words, [u32::MAX; 10]);
        add(&mut words, f32::from_bits(1));
        assert_eq!(words, [0; 10]);
    }

    #[test]
    fn matches_independent_integer_oracle() {
        let fixture = include_str!("../tests/fixtures/f32_sum.txt");
        let mut count = 0;
        for line in fixture.lines().filter(|line| !line.starts_with('#')) {
            let mut fields = line.split_whitespace();
            let name = fields.next().unwrap();
            let expected = fields.next().unwrap();
            let values: Vec<_> = fields
                .map(|s| f32::from_bits(u32::from_str_radix(s, 16).unwrap()))
                .collect();
            let result = checked_sum_f32(&values);
            match expected {
                "input" => assert!(
                    matches!(result, Err(SumError::NonFiniteInput { .. })),
                    "{name}: {result:?}"
                ),
                "prefix" => assert!(
                    matches!(result, Err(SumError::NonFinitePrefix { .. })),
                    "{name}: {result:?}"
                ),
                "sum" => assert_eq!(result, Err(SumError::NonFiniteSum), "{name}"),
                bits => assert_eq!(
                    result.unwrap().to_bits(),
                    u32::from_str_radix(bits, 16).unwrap(),
                    "{name}"
                ),
            }
            count += 1;
        }
        assert_eq!(count, 92);
    }
}
