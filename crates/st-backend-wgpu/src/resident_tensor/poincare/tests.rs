use super::*;
use st_kernel_contracts::euclidean::{EuclideanBiasForward, EuclideanBiasSpec};
use st_kernel_contracts::fisher_rao::{FisherRaoBiasForward, FisherRaoBiasSpec};
use st_kernel_contracts::poincare::PoincareBiasForward;
use st_kernel_contracts::projective::{ProjectiveBiasForward, ProjectiveBiasSpec};

#[test]
fn swapped_nearby_fisher_retains_second_order_pivot_gradients() {
    let Some(d) = device() else { return };
    let e = f32::from_bits(1);
    let gain = 2f32.powi(127);
    let f = d
        .upload(&[1, 2, 3], &[0., e, -e, 0., 0., 0.])
        .unwrap()
        .causal_fisher_rao_bias(&d.upload(&[1], &[gain]).unwrap())
        .unwrap();
    let gradient = read(
        f.backward(&d.upload(&[1, 1, 2, 2], &[0., 0., gain, 0.]).unwrap())
            .unwrap()
            .coordinates(),
    );
    let scale = (f64::from(gain).powi(2) * f64::from(e).powi(2)) as f32;
    for (index, expected) in [(0, scale / 3.), (3, -scale / 9.)] {
        assert!(
            (gradient[index] / expected - 1.).abs() < 2e-5,
            "{index}: {} != {expected}",
            gradient[index]
        );
    }
}

#[test]
fn fisher_mixed_close_and_far_gradients_share_only_the_final_sum() {
    let Some(d) = device() else { return };
    let x = [0., 0., 0., 0., 0.125, -0.125, 1., -2., 0.5];
    let seed = [0., 0., 0., 0.3, 0., 0., -0.4, 0.7, 0.];
    let f = d
        .upload(&[1, 3, 3], &x)
        .unwrap()
        .causal_fisher_rao_bias(&d.upload(&[1], &[0.2]).unwrap())
        .unwrap();
    let backward = |s: &[f32]| f.backward(&d.upload(&[1, 1, 3, 3], s).unwrap()).unwrap();
    let all = backward(&seed);
    let close_only = backward(&[0., 0., 0., 0.3, 0., 0., 0., 0., 0.]);
    let far_only = backward(&[0., 0., 0., 0., 0., 0., -0.4, 0.7, 0.]);
    let sum: Vec<_> = read(close_only.coordinates())
        .into_iter()
        .zip(read(far_only.coordinates()))
        .map(|(a, b)| a + b)
        .collect();
    close(&read(all.coordinates()), &sum);
    let oracle =
        FisherRaoBiasForward::new(FisherRaoBiasSpec::new([1, 3, 3], 1).unwrap(), &x, &[0.2])
            .unwrap()
            .backward(&seed)
            .unwrap();
    close(&read(all.coordinates()), &oracle.coordinates);
    close(&read(all.raw_gain()), &oracle.raw_gain);
}

#[test]
fn nearby_fisher_retains_second_order_pivot_gradients() {
    let Some(d) = device() else { return };
    let e = f32::from_bits(1);
    let gain = 2f32.powi(127);
    let f = d
        .upload(&[1, 2, 3], &[0., 0., 0., 0., e, -e])
        .unwrap()
        .causal_fisher_rao_bias(&d.upload(&[1], &[gain]).unwrap())
        .unwrap();
    let g = f
        .backward(&d.upload(&[1, 1, 2, 2], &[0., 0., gain, 0.]).unwrap())
        .unwrap();
    let gradient = read(g.coordinates());
    let scale = (f64::from(gain).powi(2) * f64::from(e).powi(2)) as f32;
    for (index, expected) in [(0, -scale / 9.), (3, scale / 3.)] {
        assert!(
            (gradient[index] / expected - 1.).abs() < 2e-5,
            "index={index}: {} != {expected}",
            gradient[index]
        );
    }
}

#[test]
fn fisher_close_chart_transition_and_retained_backward_match_cpu() {
    let Some(d) = device() else { return };
    let spec = FisherRaoBiasSpec::new([1, 2, 3], 1).unwrap();
    let seed = d.upload(&[1, 1, 2, 2], &[0., 0., 1., 0.]).unwrap();
    for perturbation in [0.001, 0.2499, 0.25, 0.2501, 0.5] {
        let x = [0., -0.3, -0.6, 0., -0.3, -0.6 + perturbation];
        let cpu = FisherRaoBiasForward::new(spec, &x, &[0.2]).unwrap();
        let expected = cpu.backward(&[0., 0., 1., 0.]).unwrap();
        let f = strided(&d, &[1, 2, 3], &x)
            .causal_fisher_rao_bias(&d.upload(&[1], &[0.2]).unwrap())
            .unwrap();
        close(&read(f.scores()), cpu.scores());
        let g = f.backward(&seed).unwrap();
        close(&read(g.coordinates()), &expected.coordinates);
        close(&read(g.raw_gain()), &expected.raw_gain);
        assert_eq!(
            read(g.coordinates()),
            read(f.backward(&seed).unwrap().coordinates())
        );
    }
}

#[test]
fn nearby_fisher_rows_preserve_rescaled_subnormal_logit_signals() {
    let Some(d) = device() else { return };
    let gain = 2f32.powi(120);
    for exponent in [30, 60, 90, 120, 149] {
        let e = 2f64.powi(-exponent) as f32;
        let seed = 2f32.powi(exponent - 120);
        for common in [0., 2f32.powi(100)] {
            let f = d
                .upload(&[1, 2, 2], &[common, common, e, -e])
                .unwrap()
                .causal_fisher_rao_bias(&d.upload(&[1], &[gain]).unwrap())
                .unwrap();
            let score = read(f.scores())[2];
            let expected = (-f64::from(gain) * f64::from(e).powi(2)) as f32;
            assert!(
                (score - expected).abs() <= 2e-5 * expected.abs(),
                "e=2^-{exponent}, common={common}: {score} != {expected}"
            );
            let g = f
                .backward(&d.upload(&[1, 1, 2, 2], &[0., 0., seed, 0.]).unwrap())
                .unwrap();
            for (actual, expected) in read(g.coordinates()).into_iter().zip([1., -1., -1., 1.]) {
                assert!(
                    (actual - expected).abs() < 2e-5,
                    "e=2^-{exponent}: {actual} != {expected}"
                );
            }
        }
    }
    let f = d
        .upload(&[1, 2, 2], &[1., 0., 1., f32::from_bits(1)])
        .unwrap()
        .causal_fisher_rao_bias(&d.upload(&[1], &[gain]).unwrap())
        .unwrap();
    let g = f
        .backward(
            &d.upload(&[1, 1, 2, 2], &[0., 0., 2f32.powi(29), 0.])
                .unwrap(),
        )
        .unwrap();
    let p = 1. / (1. + 1f64.exp());
    let expected = (2. * p * (1. - p)) as f32;
    for (actual, sign) in read(g.coordinates()).into_iter().zip([-1., 1., 1., -1.]) {
        assert!(
            (actual - sign * expected).abs() < 2e-5,
            "{actual} != {}",
            sign * expected
        );
    }
}

#[test]
fn nearby_simplex_rows_retain_a_recoverable_fisher_distance() {
    let Some(d) = device() else { return };
    let e = 2f32.powi(-30);
    let f = d
        .upload(&[1, 2, 2], &[0., 0., e, -e])
        .unwrap()
        .causal_fisher_rao_bias(&d.upload(&[1], &[2f32.powi(60)]).unwrap())
        .unwrap();
    let score = read(f.scores())[2];
    assert!((score + 1.).abs() < 2e-5, "{score} != -1");
    let g = f
        .backward(&d.upload(&[1, 1, 2, 2], &[0., 0., e, 0.]).unwrap())
        .unwrap();
    for (actual, expected) in read(g.coordinates()).into_iter().zip([1., -1., -1., 1.]) {
        assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
    }
}

#[test]
fn dynamic_range_projective_wedge_retains_primitive_product_residual() {
    let Some(d) = device() else { return };
    for power in [30, 62] {
        let l = 2f32.powi(power);
        let f = d
            .upload(&[1, 2, 4], &[l, 1., l, 2., l, 2., l, 3.])
            .unwrap()
            .causal_projective_bias(&d.upload(&[1], &[2f32.powi(120)]).unwrap())
            .unwrap();
        assert_eq!(
            read(f.scores())[2],
            (-0.75 * 2f64.powi(120 - 4 * power)) as f32
        );
    }
}

#[test]
fn dynamic_range_fisher_roots_keep_recoverable_logit_gradients() {
    let Some(d) = device() else { return };
    let shape = [1, 2, 2];
    for gap in [180f32, 240., 400.] {
        let x = [0., -gap, -gap, 0.];
        let raw = [1e30];
        let seed = [0., 0., 1e30, 0.];
        let cpu = FisherRaoBiasForward::new(FisherRaoBiasSpec::new(shape, 1).unwrap(), &x, &raw)
            .unwrap()
            .backward(&seed)
            .unwrap();
        let expected = (2.
            * std::f64::consts::PI
            * f64::from(raw[0])
            * f64::from(seed[2])
            * (-0.5 * f64::from(gap)).exp()) as f32;
        let f = d
            .upload(&shape, &x)
            .unwrap()
            .causal_fisher_rao_bias(&d.upload(&[1], &raw).unwrap())
            .unwrap();
        let g = f
            .backward(&d.upload(&[1, 1, 2, 2], &seed).unwrap())
            .unwrap();
        for (i, actual) in read(g.coordinates()).into_iter().enumerate() {
            let want = if i == 0 || i == 3 {
                -expected
            } else {
                expected
            };
            assert!(
                (cpu.coordinates[i] - want).abs() <= 2e-5 * expected,
                "CPU {i}: {} != {want}",
                cpu.coordinates[i]
            );
            assert!(
                (actual - want).abs() <= 2e-5 * expected,
                "GPU gap={gap}, {i}: {actual} != {want}"
            );
        }
    }
}

#[test]
fn projective_base_subtraction_preserves_recoverable_low_endpoint() {
    let Some(device) = device() else { return };
    let e = 2f32.powi(-121);
    let l = 2f32.powi(122);
    let f = device
        .upload(&[1, 2, 4], &[e, 0., 1., 0., 1., 0., l, 0.])
        .unwrap()
        .causal_projective_bias(&device.upload(&[1], &[2f32.powi(60)]).unwrap())
        .unwrap();
    let g = f
        .backward(
            &device
                .upload(&[1, 1, 2, 2], &[0., 0., 2f32.powi(63), 0.])
                .unwrap(),
        )
        .unwrap();
    assert_eq!(read(g.coordinates()), [-e, 0., l, 0., e, 0., -e, 0.]);
}

#[test]
fn projective_cancelled_overlap_matches_analytic_not_cpu_only() {
    let Some(device) = device() else { return };
    let l = 2f32.powi(70);
    let gain = 2f32.powi(120);
    let d = 2f32.powi(29);
    let e = 2f32.powi(-100);
    for (cols, x, expected) in [
        (
            4,
            vec![1., 0., e, 0., -1., 0., e, 0.],
            vec![
                -2f32.powi(39),
                0.,
                2f32.powi(-61),
                0.,
                2f32.powi(39),
                0.,
                2f32.powi(-61),
                0.,
            ],
        ),
        (
            4,
            vec![l, 32., l, 0., l, 32., -l, 0.],
            vec![
                1025. * d,
                1025. * 2f32.powi(-36),
                -1025. * d,
                0.,
                1025. * d,
                1025. * 2f32.powi(-36),
                1025. * d,
                0.,
            ],
        ),
        (
            4,
            vec![l, 0., l, 0., l, 0., -l, 0.],
            vec![d, 0., -d, 0., d, 0., d, 0.],
        ),
        (
            6,
            vec![l, 0., 32., 0., l, 0., l, 0., 32., 0., -l, 0.],
            vec![
                1025. * d,
                0.,
                1025. * 2f32.powi(-36),
                0.,
                -1025. * d,
                0.,
                1025. * d,
                0.,
                1025. * 2f32.powi(-36),
                0.,
                1025. * d,
                0.,
            ],
        ),
    ] {
        let f = device
            .upload(&[1, 2, cols], &x)
            .unwrap()
            .causal_projective_bias(&device.upload(&[1], &[gain]).unwrap())
            .unwrap();
        let g = f
            .backward(&device.upload(&[1, 1, 2, 2], &[0., 0., gain, 0.]).unwrap())
            .unwrap();
        for (a, b) in read(g.coordinates()).iter().zip(expected) {
            if b == 0. {
                assert_eq!(*a, 0.);
            } else {
                assert!(
                    (f64::from(*a) / f64::from(b) - 1.).abs() < 8e-5,
                    "{a} != {b}"
                );
            }
        }
    }
}

#[test]
fn projective_strided_complex_scores_both_endpoints_and_gains_match_cpu() {
    let Some(d) = device() else { return };
    for shape in [[1, 1, 2], [2, 3, 4], [2, 7, 6], [1, 17, 2]] {
        let s = ProjectiveBiasSpec::new(shape, 3).unwrap();
        let x: Vec<_> = (0..s.coordinates_len())
            .map(|i| ((i * 7 % 29) as f32 - 14.) * 0.125)
            .collect();
        let gains = [-7., 0.2, 4.];
        let seed: Vec<_> = (0..s.scores_len())
            .map(|i| ((i * 11 % 23) as f32 - 11.) * 0.07)
            .collect();
        let cpu = ProjectiveBiasForward::new(s, &x, &gains).unwrap();
        let want = cpu.backward(&seed).unwrap();
        let f = strided(&d, &shape, &x)
            .causal_projective_bias(&strided(&d, &[3], &gains))
            .unwrap();
        let g = f.backward(&strided(&d, &s.score_shape(), &seed)).unwrap();
        assert_eq!(f.spec(), s);
        close(&read(f.scores()), cpu.scores());
        close(&read(g.coordinates()), &want.coordinates);
        close(&read(g.raw_gain()), &want.raw_gain);
    }
}

#[test]
fn projective_tiny_and_saturated_chart_retain_recoverable_gradients() {
    let Some(d) = device() else { return };
    for (value, gain, cotangent) in [(1e-30, 1e30, 1.), (1e20, 0., 1e30), (1e-25, 0., 1e30)] {
        let s = ProjectiveBiasSpec::new([1, 2, 2], 1).unwrap();
        let x = [0., 0., value, 0.];
        let seed = [0., 0., cotangent, 0.];
        let cpu = ProjectiveBiasForward::new(s, &x, &[gain])
            .unwrap()
            .backward(&seed)
            .unwrap();
        let f = d
            .upload(&s.shape(), &x)
            .unwrap()
            .causal_projective_bias(&d.upload(&[1], &[gain]).unwrap())
            .unwrap();
        let g = f
            .backward(&d.upload(&s.score_shape(), &seed).unwrap())
            .unwrap();
        for (i, (a, b)) in read(g.coordinates())
            .iter()
            .zip(cpu.coordinates)
            .enumerate()
        {
            if i % 2 == 0 {
                assert!(
                    *a != 0. && (f64::from(*a) / f64::from(b) - 1.).abs() < 8e-5,
                    "{a} != {b}"
                );
            } else {
                assert_eq!(*a, 0.);
            }
        }
    }
}

#[test]
fn projective_identity_extremes_guards_and_retained_tapes() {
    let Some(d) = device() else { return };
    let gain = d.upload(&[1], &[0.]).unwrap();
    let seed = d.upload(&[1, 1, 2, 2], &[1.; 4]).unwrap();
    for x in [
        [0.; 8],
        [0.1; 8],
        [f32::MAX, 0., 0., 0., 0., 0., f32::MAX, 0.],
    ] {
        let cpu =
            ProjectiveBiasForward::new(ProjectiveBiasSpec::new([1, 2, 4], 1).unwrap(), &x, &[0.])
                .unwrap();
        let f = d
            .upload(&[1, 2, 4], &x)
            .unwrap()
            .causal_projective_bias(&gain)
            .unwrap();
        close(&read(f.scores()), cpu.scores());
        close(
            &read(f.backward(&seed).unwrap().coordinates()),
            &cpu.backward(&[1.; 4]).unwrap().coordinates,
        );
    }
    assert!(d
        .upload(&[1, 2, 3], &[0.; 6])
        .unwrap()
        .causal_projective_bias(&gain)
        .is_err());
    let x = d.upload(&[1, 2, 2], &[0.2, -0.4, 0.7, 0.1]).unwrap();
    let f = x.causal_projective_bias(&gain).unwrap();
    let g = f.backward(&seed).unwrap();
    let saved = read(g.coordinates());
    let overflow = d
        .upload(&[1, 1, 2, 2], &[0., f32::MAX, 0., 0.])
        .unwrap()
        .mul(&d.upload(&[1, 1, 2, 2], &[2.; 4]).unwrap())
        .unwrap();
    let bad = f.backward(&overflow).unwrap();
    for value in [bad.coordinates(), bad.raw_gain()] {
        assert!(matches!(
            value.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    let bad_x = d.guard_together(&[&x, &overflow]).unwrap().remove(0);
    assert!(matches!(
        bad_x
            .causal_projective_bias(&gain)
            .unwrap()
            .scores()
            .snapshot()
            .unwrap()
            .read(),
        Err(TensorError::NonFinite)
    ));
    assert_eq!(read(f.backward(&seed).unwrap().coordinates()), saved);
    assert_eq!(read(g.coordinates()), saved);
    let s = ProjectiveBiasSpec::new([2, 7, 4], 2).unwrap();
    assert_eq!(sizes(s, &wgpu::Limits::default()).unwrap(), (1568, 450));
    assert!(sizes(
        s,
        &wgpu::Limits {
            max_storage_buffer_binding_size: 64,
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn fisher_strided_logits_both_endpoints_and_gains_match_cpu() {
    let Some(d) = device() else { return };
    for shape in [[1, 1, 1], [2, 3, 4], [2, 7, 3], [1, 17, 2]] {
        let s = FisherRaoBiasSpec::new(shape, 3).unwrap();
        let x: Vec<_> = (0..s.coordinates_len())
            .map(|i| ((i * 7 % 29) as f32 - 14.) * 0.125)
            .collect();
        let gains = [-7., 0.2, 4.];
        let seed: Vec<_> = (0..s.scores_len())
            .map(|i| ((i * 11 % 23) as f32 - 11.) * 0.07)
            .collect();
        let cpu = FisherRaoBiasForward::new(s, &x, &gains).unwrap();
        let want = cpu.backward(&seed).unwrap();
        let f = strided(&d, &shape, &x)
            .causal_fisher_rao_bias(&strided(&d, &[3], &gains))
            .unwrap();
        let g = f.backward(&strided(&d, &s.score_shape(), &seed)).unwrap();
        assert_eq!(f.spec(), s);
        close(&read(f.scores()), cpu.scores());
        close(&read(g.coordinates()), &want.coordinates);
        close(&read(g.raw_gain()), &want.raw_gain);
        for row in read(g.coordinates()).chunks_exact(shape[2]) {
            assert!(row.iter().map(|&v| f64::from(v)).sum::<f64>().abs() < 3e-5);
        }
    }
}

#[test]
fn fisher_identity_neighbours_extremes_and_guard_propagation() {
    let Some(d) = device() else { return };
    let gain = d.upload(&[1], &[0.]).unwrap();
    let seed = d.upload(&[1, 1, 2, 2], &[1.; 4]).unwrap();
    for x in [[0.1; 4], [f32::MAX, -f32::MAX, -f32::MAX, f32::MAX]] {
        let cpu =
            FisherRaoBiasForward::new(FisherRaoBiasSpec::new([1, 2, 2], 1).unwrap(), &x, &[0.])
                .unwrap();
        let f = d
            .upload(&[1, 2, 2], &x)
            .unwrap()
            .causal_fisher_rao_bias(&gain)
            .unwrap();
        close(&read(f.scores()), cpu.scores());
        close(
            &read(f.backward(&seed).unwrap().coordinates()),
            &cpu.backward(&[1.; 4]).unwrap().coordinates,
        );
    }
    let x = d.upload(&[1, 2, 2], &[0., 0., 0.001, -0.001]).unwrap();
    let f = x.causal_fisher_rao_bias(&gain).unwrap();
    let g = f.backward(&seed).unwrap();
    assert!(read(f.scores())[2] < 0.);
    assert!(read(g.coordinates()).iter().any(|v| *v != 0.));
    let saved = read(g.coordinates());
    let overflow = d
        .upload(&[1, 1, 2, 2], &[0., f32::MAX, 0., 0.])
        .unwrap()
        .mul(&d.upload(&[1, 1, 2, 2], &[2.; 4]).unwrap())
        .unwrap();
    let guarded = d.guard_together(&[&seed, &overflow]).unwrap().remove(0);
    let bad = f.backward(&guarded).unwrap();
    for value in [bad.coordinates(), bad.raw_gain()] {
        assert!(matches!(
            value.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    let invalid_x = d.guard_together(&[&x, &overflow]).unwrap().remove(0);
    let bad = invalid_x.causal_fisher_rao_bias(&gain).unwrap();
    assert!(matches!(
        bad.scores().snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    assert_eq!(read(f.backward(&seed).unwrap().coordinates()), saved);
}

#[test]
fn fisher_large_intermediate_has_representable_final_pullback() {
    let Some(d) = device() else { return };
    let s = FisherRaoBiasSpec::new([1, 2, 2], 1).unwrap();
    let x = [0.5, -0.5, -0.5, 0.5];
    let seed = [0., 0., 3e38, 0.];
    let cpu = FisherRaoBiasForward::new(s, &x, &[0.])
        .unwrap()
        .backward(&seed)
        .unwrap();
    assert!(cpu
        .coordinates
        .iter()
        .chain(&cpu.raw_gain)
        .all(|v| v.is_finite()));
    let f = d
        .upload(&s.shape(), &x)
        .unwrap()
        .causal_fisher_rao_bias(&d.upload(&[1], &[0.]).unwrap())
        .unwrap();
    let g = f
        .backward(&d.upload(&s.score_shape(), &seed).unwrap())
        .unwrap();
    close(&read(g.coordinates()), &cpu.coordinates);
    close(&read(g.raw_gain()), &cpu.raw_gain);
}

#[test]
fn fisher_chart_shader_validates_without_optional_capabilities() {
    let source = fisher_chart::source();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let s = FisherRaoBiasSpec::new([2, 7, 3], 2).unwrap();
    assert_eq!(sizes(s, &wgpu::Limits::default()).unwrap(), (1568, 772));
    let wide = FisherRaoBiasSpec::new([1, 2, 120], 2).unwrap();
    assert!(sizes(
        wide,
        &wgpu::Limits {
            max_storage_buffer_binding_size: 2000,
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn euclidean_strided_both_endpoints_gains_and_wide_values() {
    let Some(d) = device() else { return };
    for shape in [[1, 1, 1], [2, 3, 4], [2, 7, 3], [1, 17, 2]] {
        for scale in [1., 4.] {
            let s = EuclideanBiasSpec::new(shape, 3, scale).unwrap();
            let x: Vec<_> = (0..s.coordinates_len())
                .map(|i| ((i * 7 % 29) as f32 - 14.) * 0.25)
                .collect();
            let gains = [-7., 0.2, 4.];
            let seed: Vec<_> = (0..s.scores_len())
                .map(|i| ((i * 11 % 23) as f32 - 11.) * 0.07)
                .collect();
            let cpu = EuclideanBiasForward::new(s, &x, &gains).unwrap();
            let want = cpu.backward(&seed).unwrap();
            let f = strided(&d, &shape, &x)
                .causal_euclidean_bias(&strided(&d, &[3], &gains), scale)
                .unwrap();
            let g = f.backward(&strided(&d, &s.score_shape(), &seed)).unwrap();
            assert_eq!(f.spec(), s);
            close(&read(f.scores()), cpu.scores());
            close(&read(g.coordinates()), &want.coordinates);
            close(&read(g.raw_gain()), &want.raw_gain);
        }
    }
    for (x, raw, seed) in [(1e-25, 0., 1e30), (2e14, -110., 1.), (1e-20, 1e20, 1.)] {
        let s = EuclideanBiasSpec::new([1, 2, 1], 1, 4.).unwrap();
        let cpu = EuclideanBiasForward::new(s, &[0., x], &[raw])
            .unwrap()
            .backward(&[0., 0., seed, 0.])
            .unwrap();
        let f = d
            .upload(&[1, 2, 1], &[0., x])
            .unwrap()
            .causal_euclidean_bias(&d.upload(&[1], &[raw]).unwrap(), 4.)
            .unwrap();
        let g = f
            .backward(&d.upload(&[1, 1, 2, 2], &[0., 0., seed, 0.]).unwrap())
            .unwrap();
        for (a, b) in [
            (read(g.coordinates()), cpu.coordinates),
            (read(g.raw_gain()), cpu.raw_gain),
        ] {
            for (a, b) in a.iter().zip(b) {
                assert!(
                    *a != 0. && (f64::from(*a) / f64::from(b) - 1.).abs() < 8e-5,
                    "{a} != {b}"
                );
            }
        }
    }
}

#[test]
fn euclidean_guards_and_retained_tapes_remain_atomic() {
    let Some(d) = device() else { return };
    let x = d.upload(&[1, 2, 1], &[0., 2.]).unwrap();
    let gain = d.upload(&[1], &[0.]).unwrap();
    for scale in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(x.causal_euclidean_bias(&gain, scale).is_err());
    }
    let f = x.causal_euclidean_bias(&gain, 4.).unwrap();
    let seed = d.upload(&[1, 1, 2, 2], &[0., 0., 1., 0.]).unwrap();
    let g = f.backward(&seed).unwrap();
    let saved = read(g.coordinates());
    assert!(matches!(
        d.upload(&[1, 1, 2, 2], &[0., f32::NAN, 0., 0.]),
        Err(TensorError::NonFinite)
    ));
    let masked_overflow = d
        .upload(&[1, 1, 2, 2], &[0., f32::MAX, 0., 0.])
        .unwrap()
        .mul(&d.upload(&[1, 1, 2, 2], &[2.; 4]).unwrap())
        .unwrap();
    let late_overflow = d.upload(&[1, 1, 2, 2], &[0., 0., f32::MAX, 0.]).unwrap();
    for seed in [masked_overflow, late_overflow] {
        let bad = f.backward(&seed).unwrap();
        for v in [bad.coordinates(), bad.raw_gain()] {
            assert!(matches!(
                v.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
        }
    }
    let finite = d.upload(&[1, 2, 1], &[0., 0.]).unwrap();
    let overflow = d
        .upload(&[1, 2, 1], &[f32::MAX; 2])
        .unwrap()
        .mul(&d.upload(&[1, 2, 1], &[2.; 2]).unwrap())
        .unwrap();
    let invalid = d.guard_together(&[&finite, &overflow]).unwrap().remove(0);
    assert!(invalid.shares_storage_with(&finite));
    assert_eq!(read(&finite), [0., 0.]);
    let bad = invalid.causal_euclidean_bias(&gain, 4.).unwrap();
    assert!(matches!(
        bad.scores().snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    assert_eq!(read(g.coordinates()), saved);
    assert_eq!(read(f.backward(&seed).unwrap().coordinates()), saved);
}

#[test]
fn shader_and_preflight_validate_without_optional_features() {
    let source = source();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let s = PoincareBiasSpec::new([2, 7, 3], 2, -1.).unwrap();
    assert_eq!(sizes(s, &wgpu::Limits::default()).unwrap(), (1568, 436));
    // This shape is limited by packed gradients, not the forward pair cache.
    let wide = PoincareBiasSpec::new([1, 2, 120], 2, -1.).unwrap();
    assert!(sizes(
        wide,
        &wgpu::Limits {
            max_storage_buffer_binding_size: 1000,
            ..Default::default()
        }
    )
    .is_err());
    for limits in [
        wgpu::Limits {
            max_storage_buffers_per_shader_stage: 5,
            ..Default::default()
        },
        wgpu::Limits {
            max_bindings_per_bind_group: 6,
            ..Default::default()
        },
        wgpu::Limits {
            max_uniform_buffer_binding_size: 63,
            ..Default::default()
        },
        wgpu::Limits {
            max_uniform_buffers_per_shader_stage: 0,
            ..Default::default()
        },
        wgpu::Limits {
            max_storage_buffer_binding_size: 2048,
            ..Default::default()
        },
        wgpu::Limits {
            max_compute_workgroups_per_dimension: 0,
            ..Default::default()
        },
    ] {
        assert!(sizes(s, &limits).is_err());
    }
}

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("poincare.tests").unwrap();
    Some(TensorDevice::new(runtime).unwrap())
}
fn read(t: &ResidentTensor) -> Vec<f32> {
    t.snapshot().unwrap().read().unwrap()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.is_finite()
                && y.is_finite()
                && (f64::from(x) - f64::from(y)).abs() <= 3e-6 + 8e-5 * f64::from(y).abs(),
            "{i}: {x} != {y}"
        );
    }
}
fn strided(d: &TensorDevice, s: &[usize], v: &[f32]) -> ResidentTensor {
    let mut shape = s.to_vec();
    shape.push(2);
    d.upload(
        &shape,
        &v.iter().flat_map(|&v| [0.123, v]).collect::<Vec<_>>(),
    )
    .unwrap()
    .select(s.len(), 1)
    .unwrap()
}

#[test]
fn strided_scores_both_endpoints_and_gains_match_cpu() {
    let Some(d) = device() else { return };
    for shape in [[1, 1, 1], [2, 3, 4], [2, 7, 3], [1, 17, 2]] {
        for c in [-0.01, -0.75, -1., -100.] {
            let s = PoincareBiasSpec::new(shape, 3, c).unwrap();
            let x: Vec<_> = (0..s.coordinates_len())
                .map(|i| ((i * 7 % 29) as f32 - 14.) * 0.025 / (-c).sqrt())
                .collect();
            let gains = [-7., 0.2, 4.];
            let seed: Vec<_> = (0..s.scores_len())
                .map(|i| ((i * 11 % 23) as f32 - 11.) * 0.07)
                .collect();
            let cpu = PoincareBiasForward::new(s, &x, &gains).unwrap();
            let expected = cpu.backward(&seed).unwrap();
            let f = strided(&d, &shape, &x)
                .causal_poincare_bias(&strided(&d, &[3], &gains), c)
                .unwrap();
            let g = f.backward(&strided(&d, &s.score_shape(), &seed)).unwrap();
            assert_eq!(f.scores().layout.shape(), s.score_shape());
            assert_eq!(g.coordinates().layout.shape(), shape);
            assert_eq!(g.raw_gain().layout.shape(), [3]);
            close(&read(f.scores()), cpu.scores());
            close(&read(g.coordinates()), &expected.coordinates);
            close(&read(g.raw_gain()), &expected.raw_gain);
        }
    }
}

#[test]
fn tiny_values_keep_recoverable_products_and_gradient() {
    let Some(d) = device() else { return };
    for (c, x, gain, seed) in [
        (-1., [0., 1e-25], 0., 1e30),
        (-1e-30, [0., 2e14], -110., 1.),
        (-1., [0., 1e-20], 1e20, 1.),
    ] {
        let s = PoincareBiasSpec::new([1, 2, 1], 1, c).unwrap();
        let cpu = PoincareBiasForward::new(s, &x, &[gain]).unwrap();
        let expected = cpu.backward(&[0., 0., seed, 0.]).unwrap();
        let f = d
            .upload(&[1, 2, 1], &x)
            .unwrap()
            .causal_poincare_bias(&d.upload(&[1], &[gain]).unwrap(), c)
            .unwrap();
        let g = f
            .backward(&d.upload(&[1, 1, 2, 2], &[0., 0., seed, 0.]).unwrap())
            .unwrap();
        for (got, want) in [
            (read(g.coordinates()), expected.coordinates),
            (read(g.raw_gain()), expected.raw_gain),
        ] {
            for (a, b) in got.iter().zip(want) {
                assert!(
                    *a != 0. && (f64::from(*a) / f64::from(b) - 1.).abs() < 8e-5,
                    "{a} != {b}"
                );
            }
        }
        close(&read(f.scores()), cpu.scores());
    }
}

#[test]
fn near_boundary_margin_and_large_v_remain_representable() {
    let Some(d) = device() else { return };
    let u = 2f32.powi(-24);
    let x = [
        1. - u,
        2f32.powi(-12) * (1. - u),
        2f32.powi(-12),
        u * (1. - u),
    ];
    let points: Vec<_> = x.into_iter().chain(x.map(|x| -x)).collect();
    let gain = d.upload(&[1], &[0.]).unwrap();
    let f = d
        .upload(&[1, 2, 4], &points)
        .unwrap()
        .causal_poincare_bias(&gain, -1.)
        .unwrap();
    let a = f64::from(u).powi(3) * (1. - f64::from(u));
    let distance = 4. * (4. * (1. - a) / (a * a)).sqrt().asinh().powi(2);
    let scores = read(f.scores());
    assert!((f64::from(scores[2]) / (-2f64.ln() * distance) - 1.).abs() < 8e-5);
    let g = f
        .backward(&d.upload(&[1, 1, 2, 2], &[0., 0., 1., 0.]).unwrap())
        .unwrap();
    assert!(read(g.coordinates())
        .iter()
        .all(|x| x.is_finite() && *x != 0.));
}

#[test]
fn domain_and_late_gradient_failures_do_not_poison_retained_tapes() {
    let Some(d) = device() else { return };
    let gain = d.upload(&[1], &[0.]).unwrap();
    let x = d.upload(&[1, 2, 1], &[0.1, 0.8]).unwrap();
    let f = x.causal_poincare_bias(&gain, -1.).unwrap();
    let seed = d.upload(&[1, 1, 2, 2], &[0., 0., 1., 0.]).unwrap();
    let g = f.backward(&seed).unwrap();
    let saved = read(g.coordinates());
    for outside in [1., 1.01] {
        let bad = d
            .upload(&[1, 2, 1], &[0., outside])
            .unwrap()
            .causal_poincare_bias(&gain, -1.)
            .unwrap();
        assert!(matches!(
            bad.scores().snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
        let bad = bad.backward(&seed).unwrap();
        for t in [bad.coordinates(), bad.raw_gain()] {
            assert!(matches!(
                t.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
        }
    }
    let bad = f
        .backward(&d.upload(&[1, 1, 2, 2], &[0., 0., f32::MAX, 0.]).unwrap())
        .unwrap();
    for t in [bad.coordinates(), bad.raw_gain()] {
        assert!(matches!(
            t.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    assert_eq!(read(g.coordinates()), saved);
    assert_eq!(read(f.backward(&seed).unwrap().coordinates()), saved);
}
