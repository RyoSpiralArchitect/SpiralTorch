use super::*;

// Dense pre-optimization oracles deliberately visit every declared coefficient.
pub(crate) fn dense_line(input: &[f32], coeff: &[f32], pad: Pad, scale: f64) -> Vec<f32> {
    (0..input.len())
        .map(|time| {
            let value = coeff.iter().enumerate().fold(0.0f64, |sum, (lag, &c)| {
                sum + f64::from(c)
                    * f64::from(sample_with_pad(input, time as isize - lag as isize, pad))
            });
            checked_f32("dense test output", scale * value).unwrap()
        })
        .collect()
}

pub(crate) fn dense_vjp(gradient: &[f32], coeff: &[f32], pad: Pad, scale: f64) -> Vec<f32> {
    let mut result = vec![0.0f64; gradient.len()];
    for (time, &g) in gradient.iter().enumerate() {
        for (lag, &c) in coeff.iter().enumerate() {
            if let Some(index) =
                source_index_with_pad(time as isize - lag as isize, gradient.len(), pad)
            {
                result[index] += scale * f64::from(c) * f64::from(g);
            }
        }
    }
    result.into_iter().map(|v| v as f32).collect()
}

pub(crate) fn bits(values: impl IntoIterator<Item = f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

#[test]
fn exact_support_does_not_prune_tiny_or_interior_taps() {
    assert_eq!(nonzero_lags(&[0., -0., 0.]), 0..0);
    assert_eq!(nonzero_lags(&[0., f32::from_bits(1), 0.]), 1..2);
    assert_eq!(nonzero_lags(&[0., 1., 0., 2., -0.]), 1..4);
    assert_eq!(nonzero_lags(&[1., 0.]), 0..1);
}

#[test]
fn sparse_line_maps_are_bit_exact_with_all_padding_and_signed_scales() {
    let patterns = [
        vec![0.; 16],
        vec![-0.; 16],
        vec![0., 0., 0.25, -0.5, 0., 0.],
        vec![0., f32::from_bits(1), 0., -f32::from_bits(1), 0.],
        vec![0., 1., 0., 0., -2., 0.],
        vec![0., 0., 0., 0., 0., 0., 0., 1.],
        vec![1.],
    ];
    for length in [0, 1, 3, 9, 65] {
        let input: Vec<_> = [0., -0., f32::from_bits(1), -f32::from_bits(1), 0.75, -0.4]
            .into_iter()
            .cycle()
            .take(length)
            .collect();
        for coeff in &patterns {
            for pad in [
                Pad::Zero,
                Pad::Constant(-0.),
                Pad::Constant(0.7),
                Pad::Reflect,
                Pad::Edge,
                Pad::Wrap,
            ] {
                for scale in [0., -0., 0.7, -1.4] {
                    let actual =
                        fracdiff_gl_1d_with_coeffs(&input, coeff, pad, Some(scale)).unwrap();
                    let expected = dense_line(&input, coeff, pad, f64::from(scale));
                    assert_eq!(bits(actual), bits(expected));
                    let actual =
                        fracdiff_gl_1d_vjp_with_coeffs(&input, coeff, pad, Some(scale)).unwrap();
                    assert_eq!(
                        bits(actual),
                        bits(dense_vjp(&input, coeff, pad, f64::from(scale)))
                    );
                }
            }
        }
    }
}

#[test]
fn zero_support_keeps_validation_and_overflow_checks() {
    for operation in [fracdiff_gl_1d_with_coeffs, fracdiff_gl_1d_vjp_with_coeffs] {
        assert!(operation(&[f32::NAN], &[0.; 3], Pad::Zero, None).is_err());
        assert!(operation(&[1.], &[0.; 3], Pad::Constant(f32::NAN), None).is_err());
        assert!(operation(&[1.], &[0.; 3], Pad::Zero, Some(f32::INFINITY)).is_err());
        assert!(operation(&[1.], &[0., f32::NAN, 0.], Pad::Zero, None).is_err());
        assert!(operation(&[1.], &[], Pad::Zero, None).is_err());
        assert!(operation(&[f32::MAX; 3], &[0., f32::MAX, 0.], Pad::Zero, None).is_err());
    }
}
