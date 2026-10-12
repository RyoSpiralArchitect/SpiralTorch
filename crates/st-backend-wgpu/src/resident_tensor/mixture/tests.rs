use super::*;
use st_kernel_contracts::mixture::HeadwiseMixtureForward;

#[test]
fn mixture_shaders_validate_without_a_device() {
    for template in [
        include_str!("../shaders/mixture.wgsl"),
        include_str!("../shaders/mixture_backward.wgsl"),
    ] {
        let s = source(template);
        let module =
            naga::front::wgsl::parse_str(&s).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&s)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn mixture_limits_are_preflighted() {
    let s = HeadwiseMixtureSpec::new([2, 3, 7, 3]).unwrap();
    preflight(s, &wgpu::Limits::default()).unwrap();
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
            max_uniform_buffers_per_shader_stage: 0,
            ..Default::default()
        },
        wgpu::Limits {
            max_uniform_buffer_binding_size: 31,
            ..Default::default()
        },
        wgpu::Limits {
            max_buffer_size: 32,
            ..Default::default()
        },
        wgpu::Limits {
            max_compute_workgroups_per_dimension: 0,
            ..Default::default()
        },
    ] {
        assert!(preflight(s, &limits).is_err());
    }
}

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("mixture.tests").unwrap();
    Some(TensorDevice::new(runtime).unwrap())
}
fn read(t: &ResidentTensor) -> Vec<f32> {
    t.snapshot().unwrap().read().unwrap()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!(
            a.is_finite() && (a - b).abs() <= 3e-6 + 5e-5 * b.abs(),
            "{a} != {b}"
        );
    }
}

#[test]
fn mixture_native_forward_and_vjps_support_strided_inputs_and_repeated_backward() {
    let Some(d) = device() else { return };
    for shape in [[1, 1, 1, 2], [2, 3, 7, 3], [2, 2, 17, 8]] {
        let s = HeadwiseMixtureSpec::new(shape).unwrap();
        let [b, h, p, k] = shape;
        let x: Vec<_> = (0..s.len())
            .map(|i| ((i * 7 % 31) as f32 - 15.) * 0.11)
            .collect();
        let l: Vec<_> = (0..s.router_len())
            .map(|i| (i as f32 * 0.3).cos())
            .collect();
        let seed: Vec<_> = (0..s.output_len())
            .map(|i| (i as f32 * 0.7).sin())
            .collect();
        let padded = |v: &[f32]| {
            v.chunks_exact(k)
                .flat_map(|row| row.iter().copied().chain([99.]))
                .collect::<Vec<_>>()
        };
        let x_gpu = d
            .upload(&[b, h, p, k + 1], &padded(&x))
            .unwrap()
            .narrow(3, 0, k)
            .unwrap();
        let l_gpu = d
            .upload(&[h, k + 1], &padded(&l))
            .unwrap()
            .narrow(1, 0, k)
            .unwrap();
        let seed_gpu = d
            .upload(
                &[b + 1, h, p],
                &vec![vec![99.; h * p], seed.clone()].concat(),
            )
            .unwrap()
            .narrow(0, 1, b)
            .unwrap();
        let oracle = HeadwiseMixtureForward::new(s, &x, &l).unwrap();
        let f = x_gpu.headwise_mixture(&l_gpu).unwrap();
        close(&read(f.output()), oracle.output());
        close(&read(f.weights()), oracle.weights());
        let expected = oracle.backward(&seed).unwrap();
        let g = f.backward(&seed_gpu).unwrap();
        close(&read(g.components()), &expected.input);
        close(&read(g.logits()), &expected.logits);
        let again = f.backward(&seed_gpu).unwrap();
        assert_eq!(read(g.logits()), read(again.logits()));
        assert_eq!(read(g.components()), read(again.components()));
    }
}

#[test]
fn mixture_extreme_logits_do_not_hide_inherited_failures() {
    let Some(d) = device() else { return };
    let x = d.upload(&[2, 1, 1, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    let l = d.upload(&[1, 3], &[-f32::MAX, f32::MAX, 0.]).unwrap();
    let f = x.headwise_mixture(&l).unwrap();
    assert_eq!(read(f.weights()), [0., 1., 0.]);
    assert_eq!(read(f.output()), [2., 5.]);
    let large = d.upload(&[1], &[f32::MAX]).unwrap();
    let invalid = large.add(&large).unwrap();
    let guarded = d.guard_together(&[&x, &invalid]).unwrap();
    let broken = guarded[0].headwise_mixture(&l).unwrap();
    for v in [broken.weights(), broken.output()] {
        assert!(matches!(
            v.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    let s = d.upload(&[2, 1, 1], &[2., 3.]).unwrap();
    let g = broken.backward(&s).unwrap();
    for v in [g.components(), g.logits()] {
        assert!(matches!(
            v.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    let g = f
        .backward(&d.guard_together(&[&s, &invalid]).unwrap()[0])
        .unwrap();
    for v in [g.components(), g.logits()] {
        assert!(matches!(
            v.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
}

#[test]
fn mixture_wide_centering_accepts_finite_final_derivatives_only() {
    let Some(d) = device() else { return };
    let logits = d.upload(&[1, 2], &[0., 0.]).unwrap();
    let seed = d.upload(&[1, 1, 1], &[f32::MAX]).unwrap();
    let mixed = d
        .upload(&[1, 1, 1, 2], &[1., 2.])
        .unwrap()
        .headwise_mixture(&logits)
        .unwrap();
    assert_eq!(read(mixed.output()), [1.5]);
    let gradient = mixed.backward(&seed).unwrap();
    close(&read(gradient.components()), &[f32::MAX / 2.; 2]);
    close(&read(gradient.logits()), &[-f32::MAX / 4., f32::MAX / 4.]);
    let same = d
        .upload(&[2, 1, 1, 3], &[f32::MAX; 6])
        .unwrap()
        .headwise_mixture(&d.upload(&[1, 3], &[-0.3, 0.2, 0.7]).unwrap())
        .unwrap();
    assert_eq!(
        read(
            same.backward(&d.upload(&[2, 1, 1], &[f32::MAX; 2]).unwrap())
                .unwrap()
                .logits()
        ),
        [0.; 3]
    );
    let bad = d
        .upload(&[1, 1, 1, 2], &[1., 10.])
        .unwrap()
        .headwise_mixture(&logits)
        .unwrap()
        .backward(&seed)
        .unwrap();
    for tensor in [bad.components(), bad.logits()] {
        assert!(matches!(
            tensor.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
}

#[test]
fn mixture_centering_precedes_position_and_batch_reduction() {
    let Some(d) = device() else { return };
    let logits = d.upload(&[1, 2], &[0., 0.]).unwrap();
    for shape in [[1, 1, 2, 2], [2, 1, 1, 2]] {
        let forward = d
            .upload(&shape, &[f32::MAX, f32::MAX, 1., 2.])
            .unwrap()
            .headwise_mixture(&logits)
            .unwrap();
        assert_eq!(read(forward.output()), [f32::MAX, 1.5]);
        let gradient = forward
            .backward(&d.upload(&shape[..3], &[1., 1.]).unwrap())
            .unwrap();
        assert_eq!(read(gradient.components()), [0.5; 4]);
        assert_eq!(read(gradient.logits()), [-0.25, 0.25]);
    }
}
