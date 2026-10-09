use super::*;

fn scores(batch: usize, start: usize, factors: &[f32]) -> Vec<f32> {
    (start..start + batch)
        .flat_map(|b| {
            factors.iter().flat_map(move |&scale| {
                (0..3).flat_map(move |q: usize| {
                    (0..3).map(move |k| {
                        if k <= q {
                            -scale * (b + 1) as f32 * ((q - k) * (q - k)) as f32
                        } else {
                            10000.
                        }
                    })
                })
            })
        })
        .collect()
}

fn moments(factor: f32) -> CausalBiasMoments {
    CausalBiasMoments::from_scores([1, 1, 3, 3], &scores(1, 0, &[factor])).unwrap()
}

#[test]
fn centered_rms_is_per_head_causal_and_row_shift_invariant() {
    let mut values = scores(1, 0, &[1., 2.]);
    let original = CausalBiasMoments::from_scores([1, 2, 3, 3], &values).unwrap();
    assert_eq!(original.valid_pairs_per_head(), 6);
    assert!((original.rms()[0] - 55f64.sqrt() / 6.).abs() < 1e-14);
    assert_eq!(original.rms()[1], original.rms()[0] * 2.);
    for (i, row) in values.chunks_exact_mut(3).enumerate() {
        for (k, value) in row.iter_mut().enumerate() {
            *value += (16 * (i + 1)) as f32;
            if k > i % 3 {
                *value = -999999.;
            }
        }
    }
    let shifted = CausalBiasMoments::from_scores([1, 2, 3, 3], &values).unwrap();
    assert_eq!(shifted.rms(), original.rms());
    let permuted = CausalBiasMoments::from_scores([1, 2, 3, 3], &scores(1, 0, &[2., 1.])).unwrap();
    assert_eq!(permuted.rms(), vec![original.rms()[1], original.rms()[0]]);
}

#[test]
fn merges_weight_unequal_batches_by_causal_pair_count() {
    let mut first = CausalBiasMoments::from_scores([1, 2, 3, 3], &scores(1, 0, &[1., 3.])).unwrap();
    let rest = CausalBiasMoments::from_scores([2, 2, 3, 3], &scores(2, 1, &[1., 3.])).unwrap();
    let whole = CausalBiasMoments::from_scores([3, 2, 3, 3], &scores(3, 0, &[1., 3.])).unwrap();
    first.merge(&rest).unwrap();
    assert_eq!(first.valid_pairs_per_head(), 18);
    for (actual, expected) in first.rms().iter().zip(whole.rms()) {
        assert!((actual - expected).abs() < 1e-14);
    }
}

#[test]
fn invalid_shapes_lengths_and_masked_nan_are_not_hidden() {
    for shape in [
        [0, 1, 2, 2],
        [1, 0, 2, 2],
        [1, 1, 2, 3],
        [usize::MAX, 1, 2, 2],
    ] {
        assert_eq!(
            CausalBiasMoments::from_scores(shape, &[]).unwrap_err(),
            CausalBiasScaleError::Shape
        );
    }
    assert_eq!(
        CausalBiasMoments::from_scores([1, 1, 2, 2], &[]).unwrap_err(),
        CausalBiasScaleError::Length
    );
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            CausalBiasMoments::from_scores([1, 1, 2, 2], &[0., value, -1., 0.]).unwrap_err(),
            CausalBiasScaleError::NonFinite
        );
    }
}

#[test]
fn failed_merges_are_transactional_including_count_overflow() {
    let mut m = moments(1.);
    let before = m.rms();
    let other = CausalBiasMoments::from_scores([1, 1, 2, 2], &[0., 0., -1., 0.]).unwrap();
    assert_eq!(m.merge(&other).unwrap_err(), CausalBiasScaleError::Coverage);
    assert_eq!(m.rms(), before);
    let mut overflow = false;
    for _ in 0..64 {
        let old = m.clone();
        if let Err(error) = m.merge(&old) {
            assert_eq!(error, CausalBiasScaleError::Overflow);
            assert_eq!(m.valid_pairs, old.valid_pairs);
            assert_eq!(m.squared, old.squared);
            overflow = true;
            break;
        }
    }
    assert!(overflow);
}

#[test]
fn identity_preserves_raw_float_bits_even_at_extremes() {
    let values = scores(1, 0, &[1.; 6]);
    let m = CausalBiasMoments::from_scores([1, 6, 3, 3], &values).unwrap();
    let raw = [-0., 0., -f32::MAX, f32::MAX, -1000., 1000.];
    let fit = CausalBiasScaleMatch::new(&m, &m, &raw, 0.).unwrap();
    assert_eq!(
        fit.raw_gains()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        raw.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(fit.realized_scales(), &[1.; 6]);
    assert_eq!(fit.relative_errors(), &[0.; 6]);
}

#[test]
fn scale_match_inverts_softplus_without_clipping_or_exp_overflow() {
    let candidate = moments(1.);
    for raw in [-100., -20., -2., 0., 4., 100., 1e5] {
        for scale in [0.25, 0.75, 1., 2., 8.] {
            let fit = CausalBiasScaleMatch::new(&moments(scale), &candidate, &[raw], 1e-5).unwrap();
            let gain = |r: f64| r.max(0.) + (-r.abs()).exp().ln_1p();
            let ratio = gain(f64::from(fit.raw_gains()[0])) / gain(f64::from(raw));
            assert!(
                (ratio / f64::from(scale) - 1.).abs() <= 1e-5,
                "{raw} {scale}"
            );
            assert!((fit.requested_scales()[0] - f64::from(scale)).abs() < 1e-14);
            assert!((fit.realized_scales()[0] / ratio - 1.).abs() < 1e-12);
        }
    }
}

#[test]
fn fitting_distinct_heads_commutes_with_joint_permutation() {
    let moment = |factors: &[f32]| {
        CausalBiasMoments::from_scores([2, 3, 3, 3], &scores(2, 0, factors)).unwrap()
    };
    let target = moment(&[1., 9., 2.]);
    let candidate = moment(&[4., 3., 1.]);
    let raw = [-2., 0.25, 7.];
    let fit = CausalBiasScaleMatch::new(&target, &candidate, &raw, 1e-5).unwrap();
    for (index, expected) in [0.25, 3., 2.].iter().enumerate() {
        assert!((fit.requested_scales()[index] - expected).abs() < 1e-14);
        assert!((fit.realized_scales()[index] / expected - 1.).abs() <= 1e-5);
    }
    let permuted = CausalBiasScaleMatch::new(
        &moment(&[2., 1., 9.]),
        &moment(&[1., 4., 3.]),
        &[raw[2], raw[0], raw[1]],
        1e-5,
    )
    .unwrap();
    for (index, original) in [2, 0, 1].iter().enumerate() {
        assert_eq!(
            permuted.raw_gains()[index].to_bits(),
            fit.raw_gains()[*original].to_bits()
        );
        assert_eq!(
            permuted.relative_errors()[index],
            fit.relative_errors()[*original]
        );
    }
}

#[test]
fn fit_rejects_degenerate_mismatched_and_unrepresentable_controls() {
    let m = moments(1.);
    let zero = moments(0.);
    for (a, b) in [(&m, &zero), (&zero, &m), (&zero, &zero)] {
        assert_eq!(
            CausalBiasScaleMatch::new(a, b, &[0.], 1e-5).unwrap_err(),
            CausalBiasScaleError::NoSignal
        );
    }
    let single = CausalBiasMoments::from_scores([1, 1, 1, 1], &[17.]).unwrap();
    assert_eq!(
        CausalBiasScaleMatch::new(&single, &single, &[0.], 1e-5).unwrap_err(),
        CausalBiasScaleError::NoSignal
    );
    for tolerance in [-1., 1., f64::NAN, f64::INFINITY] {
        assert_eq!(
            CausalBiasScaleMatch::new(&m, &m, &[0.], tolerance).unwrap_err(),
            CausalBiasScaleError::Tolerance
        );
    }
    for raw in [vec![], vec![0., 0.], vec![f32::NAN]] {
        assert_eq!(
            CausalBiasScaleMatch::new(&m, &m, &raw, 1e-5).unwrap_err(),
            CausalBiasScaleError::Gain
        );
    }
    let mut more = m.clone();
    more.merge(&m).unwrap();
    assert_eq!(
        CausalBiasScaleMatch::new(&m, &more, &[0.], 1e-5).unwrap_err(),
        CausalBiasScaleError::Coverage
    );
    for raw in [f32::MAX, -f32::MAX] {
        assert_eq!(
            CausalBiasScaleMatch::new(&moments(2.), &m, &[raw], 1e-5).unwrap_err(),
            CausalBiasScaleError::Unrepresentable
        );
    }
}

#[test]
fn finite_f32_extremes_do_not_overflow_moment_accumulation() {
    let m = CausalBiasMoments::from_scores([1, 1, 2, 2], &[0., 0., -f32::MAX, f32::MAX]).unwrap();
    assert!((m.rms()[0] / f64::from(f32::MAX) - (2f64 / 3.).sqrt()).abs() < 1e-14);
    let tiny = f32::from_bits(1);
    let m = CausalBiasMoments::from_scores([1, 1, 2, 2], &[0., 0., -tiny, tiny]).unwrap();
    assert!(m.rms()[0] > 0.);
}
