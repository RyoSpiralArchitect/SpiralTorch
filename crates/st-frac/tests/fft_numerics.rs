use st_frac::fft::{fft_inplace, radix4, Complex32, FftError};

fn signal(n: usize) -> Vec<Complex32> {
    let mut state = 0x1234_5678u32;
    let mut sample = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / 16_777_216.0 - 0.5
    };
    (0..n).map(|_| Complex32::new(sample(), sample())).collect()
}

// Independent f64 definition, deliberately not another radix implementation.
fn dft(input: &[Complex32], inverse: bool) -> Vec<(f64, f64)> {
    let n = input.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    let scale = if inverse { 1.0 / n as f64 } else { 1.0 };
    (0..n)
        .map(|k| {
            let mut re = 0.0;
            let mut im = 0.0;
            for (j, value) in input.iter().enumerate() {
                let angle = sign * std::f64::consts::TAU * (j * k) as f64 / n as f64;
                let (sine, cosine) = angle.sin_cos();
                re += f64::from(value.re) * cosine - f64::from(value.im) * sine;
                im += f64::from(value.re) * sine + f64::from(value.im) * cosine;
            }
            (re * scale, im * scale)
        })
        .collect()
}

fn assert_close(actual: &[Complex32], expected: &[(f64, f64)], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (value, &(re, im))) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(value.re) - re).hypot(f64::from(value.im) - im);
        assert!(
            error <= tolerance,
            "bin {index}: {value:?} != ({re}, {im}); error={error}"
        );
    }
}

#[test]
fn forward_and_inverse_match_independent_dft_for_mixed_radices() {
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
        for inverse in [false, true] {
            let input = signal(n);
            let expected = dft(&input, inverse);
            let mut actual = input;
            fft_inplace(&mut actual, inverse).unwrap();
            assert_close(&actual, &expected, 2e-6 * n as f64);
        }
    }
}

#[test]
fn forward_four_point_ramp_has_standard_frequency_order_and_sign() {
    let mut input = (1..=4)
        .map(|x| Complex32::new(x as f32, 0.0))
        .collect::<Vec<_>>();
    fft_inplace(&mut input, false).unwrap();
    assert_close(
        &input,
        &[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
        0.0,
    );
}

#[test]
fn standalone_radix4_uses_forward_sign_and_input_order() {
    let values: [Complex32; 4] = signal(4).try_into().unwrap();
    let twiddles = [
        Complex32::new(0.0, 1.0),
        Complex32::new(-0.5, 0.25),
        Complex32::new(0.75, -0.125),
    ];
    let weighted = [
        values[0],
        values[1].mul(twiddles[0]),
        values[2].mul(twiddles[1]),
        values[3].mul(twiddles[2]),
    ];
    assert_close(&radix4(values, twiddles), &dft(&weighted, false), 2e-7);
}

#[test]
fn singleton_is_identity_in_both_directions() {
    for inverse in [false, true] {
        let mut value = [Complex32::new(3.25, -1.5)];
        fft_inplace(&mut value, inverse).unwrap();
        assert_eq!(value, [Complex32::new(3.25, -1.5)]);
    }
}

#[test]
fn arbitrary_complex_roundtrips_preserve_every_sample() {
    for n in [2, 4, 8, 16, 32, 128, 1024, 4096] {
        let input = signal(n);
        let mut actual = input.clone();
        fft_inplace(&mut actual, false).unwrap();
        fft_inplace(&mut actual, true).unwrap();
        let expected = input
            .iter()
            .map(|x| (f64::from(x.re), f64::from(x.im)))
            .collect::<Vec<_>>();
        assert_close(&actual, &expected, 2e-6);
    }
}

#[test]
fn shifted_impulses_and_complex_tones_keep_their_phase() {
    for n in [4, 8, 16, 32, 64] {
        let mut impulse = vec![Complex32::default(); n];
        impulse[n / 2 - 1] = Complex32::new(1.0, -0.5);
        let expected = dft(&impulse, false);
        fft_inplace(&mut impulse, false).unwrap();
        assert_close(&impulse, &expected, 2e-6 * n as f64);

        let mut tone = (0..n)
            .map(|j| {
                let angle = std::f64::consts::TAU * 3.0 * j as f64 / n as f64;
                Complex32::new(angle.cos() as f32, angle.sin() as f32)
            })
            .collect::<Vec<_>>();
        fft_inplace(&mut tone, false).unwrap();
        let mut expected = vec![(0.0, 0.0); n];
        expected[3] = (n as f64, 0.0);
        assert_close(&tone, &expected, 2e-6 * n as f64);
    }
}

#[test]
fn circular_convolution_agrees_with_direct_time_domain_filtering() {
    let input = signal(32);
    let mut kernel = vec![Complex32::default(); input.len()];
    kernel[0] = Complex32::new(0.5, 0.0);
    kernel[1] = Complex32::new(0.25, 0.125);
    kernel[3] = Complex32::new(-0.125, 0.25);
    let expected = (0..input.len())
        .map(|k| {
            let value = kernel
                .iter()
                .enumerate()
                .fold(Complex32::default(), |sum, (j, weight)| {
                    sum.add(input[(k + input.len() - j) % input.len()].mul(*weight))
                });
            (f64::from(value.re), f64::from(value.im))
        })
        .collect::<Vec<_>>();
    let mut spectrum = input;
    fft_inplace(&mut spectrum, false).unwrap();
    fft_inplace(&mut kernel, false).unwrap();
    for (value, weight) in spectrum.iter_mut().zip(kernel) {
        *value = value.mul(weight);
    }
    fft_inplace(&mut spectrum, true).unwrap();
    assert_close(&spectrum, &expected, 2e-6);
}

#[test]
fn invalid_lengths_are_rejected_without_mutation() {
    for n in [0, 3, 5, 6, 15, 31] {
        for inverse in [false, true] {
            let mut input = signal(n);
            let before = input.clone();
            let expected = if n == 0 {
                FftError::Empty
            } else {
                FftError::NonPowerOfTwo
            };
            assert_eq!(fft_inplace(&mut input, inverse), Err(expected));
            assert_eq!(input, before);
        }
    }
}
