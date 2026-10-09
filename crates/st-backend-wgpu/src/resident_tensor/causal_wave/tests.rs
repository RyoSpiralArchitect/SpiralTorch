use super::*;
use st_kernel_contracts::causal_wave::CausalWaveForward;

#[test]
fn shaders_validate_without_a_device() {
    for template in [
        include_str!("../shaders/causal_wave.wgsl"),
        include_str!("../shaders/causal_wave_backward.wgsl"),
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
fn storage_uniform_and_grid_limits_are_checked_before_execution() {
    let spec = CausalWaveSpec::new([2, 7, 4], -1.).unwrap();
    assert_eq!(
        preflight_sizes(spec, &wgpu::Limits::default()).unwrap(),
        (74, 76)
    );
    for limits in [
        wgpu::Limits {
            max_storage_buffers_per_shader_stage: 7,
            ..Default::default()
        },
        wgpu::Limits {
            max_bindings_per_bind_group: 8,
            ..Default::default()
        },
        wgpu::Limits {
            max_uniform_buffers_per_shader_stage: 0,
            ..Default::default()
        },
        wgpu::Limits {
            max_uniform_buffer_binding_size: 63,
            ..Default::default()
        },
        wgpu::Limits {
            max_buffer_size: 128,
            ..Default::default()
        },
        wgpu::Limits {
            max_compute_workgroups_per_dimension: 0,
            ..Default::default()
        },
    ] {
        assert!(preflight_sizes(spec, &limits).is_err());
    }
    // Wide chart adjoints need 16 bytes/value, even when the packed f32
    // forward and gradient buffers fit in the same binding limit.
    let small_binding = wgpu::Limits {
        max_storage_buffer_binding_size: 1024,
        max_buffer_size: 1024,
        ..Default::default()
    };
    assert!(preflight_sizes(
        CausalWaveSpec::new([1, 64, 2], -1.).unwrap(),
        &small_binding
    )
    .is_err());
}

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("causal_wave.tests").unwrap();
    Some(TensorDevice::new(runtime).unwrap())
}
fn read(t: &ResidentTensor) -> Vec<f32> {
    t.snapshot().unwrap().read().unwrap()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            a.is_finite() && (a - b).abs() <= 3e-6 + 8e-5 * b.abs(),
            "{a} != {b}"
        );
    }
}

#[test]
fn native_values_and_four_vjps_match_shared_contract_with_strided_views() {
    let Some(device) = device() else { return };
    for shape in [[1, 1, 2], [2, 3, 4], [2, 7, 6], [1, 31, 8]] {
        let [batch, steps, cols] = shape;
        let n = batch * steps * cols;
        let x: Vec<_> = (0..n).map(|i| ((i * 7 % 29) as f32 - 14.) * 0.04).collect();
        let raw: Vec<_> = (0..cols / 2).map(|i| i as f32 * 0.2 - 0.5).collect();
        let phase: Vec<_> = (0..cols / 2).map(|i| i as f32 * 0.3 - 0.2).collect();
        let initial = vec![0.1; batch * cols];
        let seed = vec![0.2; n];
        let end = vec![-0.1; batch * cols];
        let spec = CausalWaveSpec::new(shape, -0.75).unwrap();
        let cpu = CausalWaveForward::new(spec, &x, &raw, &phase, &initial).unwrap();
        let reference = cpu.backward(&seed, &end).unwrap();
        let mut strided = vec![0.; n];
        for b in 0..batch {
            for t in 0..steps {
                for c in 0..cols {
                    strided[(b * cols + c) * steps + t] = x[(b * steps + t) * cols + c];
                }
            }
        }
        let input = device
            .upload(&[batch, cols, steps], &strided)
            .unwrap()
            .permute(&[0, 2, 1])
            .unwrap();
        let decay = device.upload(&[cols / 2], &raw).unwrap();
        let phase = device.upload(&[cols / 2], &phase).unwrap();
        let state = device.upload(&[batch, cols], &initial).unwrap();
        let output = input
            .causal_zspace_wave(&decay, &phase, &state, -0.75)
            .unwrap();
        close(&read(output.features()), cpu.features());
        close(&read(output.final_state()), cpu.final_state());
        let cot = device.upload(&shape, &seed).unwrap();
        let terminal = device.upload(&[batch, cols], &end).unwrap();
        let g = output.backward(&cot, &terminal).unwrap();
        for (tensor, expected) in [
            (g.drive(), &reference.drive),
            (g.raw_decay(), &reference.raw_decay),
            (g.raw_phase(), &reference.raw_phase),
            (g.initial_state(), &reference.initial_state),
        ] {
            close(&read(tensor), expected);
        }
        let frozen = read(g.drive());
        let huge = device.upload(&[1], &[f32::MAX]).unwrap();
        let invalid = huge.mul(&huge).unwrap();
        let bad = device.guard_together(&[&cot, &invalid]).unwrap().remove(0);
        let failed = output.backward(&bad, &terminal).unwrap();
        for tensor in [
            failed.drive(),
            failed.raw_decay(),
            failed.raw_phase(),
            failed.initial_state(),
        ] {
            assert!(matches!(
                tensor.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
        }
        assert_eq!(read(g.drive()), frozen);
        let fresh = output.backward(&cot, &terminal).unwrap();
        close(&read(fresh.drive()), &frozen);
        let other = device.upload(&shape, &vec![0.7; n]).unwrap();
        let changed = other
            .causal_zspace_wave(&decay, &phase, &state, -0.75)
            .unwrap();
        assert_ne!(read(changed.features()), read(output.features()));
        close(&read(output.features()), cpu.features());
    }
}

#[test]
fn every_forward_operand_guard_and_late_batch_reduction_failure_propagate() {
    let Some(device) = device() else { return };
    let values = [
        device.upload(&[1, 2, 2], &[0.2; 4]).unwrap(),
        device.upload(&[1], &[0.]).unwrap(),
        device.upload(&[1], &[0.]).unwrap(),
        device.upload(&[1, 2], &[0.; 2]).unwrap(),
    ];
    let huge = device.upload(&[1], &[f32::MAX]).unwrap();
    let failed = huge.mul(&huge).unwrap();
    for slot in 0..4 {
        let mut inputs = values.clone();
        inputs[slot] = device
            .guard_together(&[&inputs[slot], &failed])
            .unwrap()
            .remove(0);
        let f = inputs[0]
            .causal_zspace_wave(&inputs[1], &inputs[2], &inputs[3], -1.)
            .unwrap();
        for tensor in [f.features(), f.final_state()] {
            assert!(matches!(
                tensor.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
        }
    }
    for batch in [1, 16] {
        let x = device.upload(&[batch, 1, 2], &vec![0.; batch * 2]).unwrap();
        let init = device.upload(&[batch, 2], &[1., 0.].repeat(batch)).unwrap();
        let f = x
            .causal_zspace_wave(&values[1], &values[2], &init, -1.)
            .unwrap();
        let cot = device.upload(&[batch, 1, 2], &vec![0.; batch * 2]).unwrap();
        let end = device
            .upload(&[batch, 2], &[1e38, 0.].repeat(batch))
            .unwrap();
        let g = f.backward(&cot, &end).unwrap();
        for t in [g.drive(), g.raw_decay(), g.raw_phase(), g.initial_state()] {
            let r = t.snapshot().unwrap().read();
            if batch == 1 {
                assert!(r.is_ok());
            } else {
                assert!(matches!(r, Err(TensorError::NonFinite)));
            }
        }
    }
}
