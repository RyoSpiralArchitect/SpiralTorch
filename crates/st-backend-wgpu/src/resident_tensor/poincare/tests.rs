use super::*;
use st_kernel_contracts::euclidean::{EuclideanBiasForward, EuclideanBiasSpec};
use st_kernel_contracts::fisher_rao::{FisherRaoBiasForward, FisherRaoBiasSpec};
use st_kernel_contracts::poincare::PoincareBiasForward;

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
    assert_eq!(sizes(s, &wgpu::Limits::default()).unwrap(), (1568, 604));
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
