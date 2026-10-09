use super::*;
use st_kernel_contracts::attention::{attention_vjp_reference, AttentionGradients};

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) =
        runtime::ensure_default_runtime_blocking("tensor.attention.vjp.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    eprintln!(
        "resident attention VJP adapter: {:?}",
        runtime.adapter_info()
    );
    Some(TensorDevice::new(runtime).unwrap())
}

fn read(value: &ResidentTensor) -> Vec<f32> {
    value.snapshot().unwrap().read().unwrap()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite()
                && e.is_finite()
                && (f64::from(a) - f64::from(e)).abs() <= 3e-6 + 5e-5 * f64::from(e).abs(),
            "[{index}] {a} != {e}"
        );
    }
}

fn compare(actual: &ResidentAttentionGradients, expected: &AttentionGradients) {
    close(&read(&actual.query), &expected.query);
    close(&read(&actual.key), &expected.key);
    close(&read(&actual.value), &expected.value);
    for (a, e) in [
        (&actual.z_bias, &expected.z_bias),
        (&actual.pair_bias, &expected.pair_bias),
    ] {
        assert_eq!(a.is_some(), e.is_some());
        if let (Some(a), Some(e)) = (a, e) {
            close(&read(a), e);
        }
    }
}

fn data(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (i as f32 * 0.17 + phase).sin() * 0.5)
        .collect()
}

#[test]
fn shader_abi_and_workgroup_storage_validate_without_an_adapter() {
    let source = shader_source();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    assert_eq!(std::mem::size_of::<VjpParams>(), 256);
    let ty = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("VjpParams"))
        .unwrap()
        .1;
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("uniform struct")
    };
    assert_eq!(*span, 256);
    for (name, offset) in [
        ("query", std::mem::offset_of!(VjpParams, query)),
        ("upstream", std::mem::offset_of!(VjpParams, upstream)),
    ] {
        assert_eq!(
            members
                .iter()
                .find(|m| m.name.as_deref() == Some(name))
                .unwrap()
                .offset as usize,
            offset
        );
    }
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).unwrap();
    let storage: u32 = module
        .global_variables
        .iter()
        .filter(|(_, v)| v.space == naga::AddressSpace::WorkGroup)
        .map(|(_, v)| layouter[v.ty].size)
        .sum();
    assert!(storage <= 10_240, "{storage} exceeds preflight bound");
    assert_eq!(module.entry_points.len(), 3);
}

#[test]
fn complete_allocation_and_both_grids_are_preflighted() {
    let qs = [1, 1, 2, 3];
    let ks = [1, 1, 7, 3];
    let spec = AttentionSpec::new(&qs, &ks, &ks, 1., AttentionMask::None).unwrap();
    let mut limits = wgpu::Limits::default();
    let packed = PackedLayout::new(spec, true, true, &limits).unwrap();
    assert_eq!(packed.total, 18 + 6 + 21 + 21 + 7 + 14);
    assert_eq!(packed.offsets, [18, 24, 45, 66, 73]);
    limits.max_storage_buffer_binding_size = (packed.total * 4 - 1) as u32;
    assert!(validate_vjp_limits(spec, true, true, &limits).is_err());
    limits.max_storage_buffer_binding_size += 1;
    assert!(validate_vjp_limits(spec, true, true, &limits).is_ok());
    limits.max_compute_workgroups_per_dimension = 2;
    assert!(validate_vjp_limits(spec, false, false, &limits).is_err());
    limits.max_compute_workgroups_per_dimension = 3;
    assert_eq!(
        PackedLayout::new(spec, false, false, &limits)
            .unwrap()
            .key_grid,
        [3, 3]
    );
    for bad in [
        wgpu::Limits {
            max_storage_buffers_per_shader_stage: 7,
            ..Default::default()
        },
        wgpu::Limits {
            max_bindings_per_bind_group: 8,
            ..Default::default()
        },
        wgpu::Limits {
            max_uniform_buffer_binding_size: 255,
            ..Default::default()
        },
        wgpu::Limits {
            max_compute_workgroup_storage_size: 10_239,
            ..Default::default()
        },
        wgpu::Limits {
            max_compute_invocations_per_workgroup: 63,
            ..Default::default()
        },
    ] {
        assert!(validate_vjp_limits(spec, false, false, &bad).is_err());
    }
}

#[test]
fn gradients_match_shared_reference_for_biases_masks_batches_and_dimension_tails() {
    let Some(device) = device() else { return };
    for d in [1, 7, 65, 256] {
        let qs = [2, 2, 3, d];
        let ks = [2, 2, 5, d];
        let q = data(12 * d, 0.1);
        let k = data(20 * d, 1.2);
        let v = data(20 * d, -0.7);
        let u = data(12 * d, 2.1);
        let z = data(20, 0.4);
        let pair = data(60, -0.5);
        let qg = device.upload(&qs, &q).unwrap();
        let kg = device.upload(&ks, &k).unwrap();
        let vg = device.upload(&ks, &v).unwrap();
        let ug = device.upload(&qs, &u).unwrap();
        let zg = device.upload(&[2, 2, 5], &z).unwrap();
        let pg = device.upload(&[2, 2, 3, 5], &pair).unwrap();
        for mask in [
            AttentionMask::None,
            AttentionMask::Causal { query_offset: 0 },
            AttentionMask::Causal { query_offset: 2 },
        ] {
            for mode in 0..4 {
                let spec = AttentionSpec::new(&qs, &ks, &ks, 0.375, mask).unwrap();
                let expected = attention_vjp_reference(
                    spec,
                    &q,
                    &k,
                    &v,
                    (mode & 1 != 0).then_some(z.as_slice()),
                    (mode & 2 != 0).then_some(pair.as_slice()),
                    &u,
                )
                .unwrap();
                let actual = qg
                    .scaled_dot_attention_vjp(
                        &kg,
                        &vg,
                        &ug,
                        0.375,
                        mask,
                        (mode & 1 != 0).then_some(&zg),
                        (mode & 2 != 0).then_some(&pg),
                    )
                    .unwrap();
                compare(&actual, &expected);
                assert_eq!(actual.query.layout.shape(), qs);
                assert_eq!(actual.key.layout.shape(), ks);
                if let (AttentionMask::Causal { query_offset }, Some(pair)) =
                    (mask, &actual.pair_bias)
                {
                    let values = read(pair);
                    let (rows, remainder) = values.as_chunks::<5>();
                    assert!(remainder.is_empty());
                    for (index, row) in rows.iter().enumerate() {
                        assert!(row[query_offset + index % 3 + 1..].iter().all(|&v| v == 0.));
                    }
                }
            }
        }
    }
}

#[test]
fn all_strided_inputs_and_broadcast_biases_keep_logical_gradients_owned() {
    let Some(device) = device() else { return };
    let make = |len, phase| {
        device
            .upload(&[1, 2, 7, len + 2], &data(14 * (len + 2), phase))
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap()
            .narrow(2, 1, len)
            .unwrap()
            .broadcast_to(&[2, 2, len, 7])
            .unwrap()
    };
    let q = make(3, 0.1);
    let k = make(5, 0.2);
    let v = make(5, 0.3);
    let u = make(3, 0.4);
    let z = device
        .upload(&[7], &data(7, 0.5))
        .unwrap()
        .narrow(0, 1, 5)
        .unwrap()
        .broadcast_to(&[2, 2, 5])
        .unwrap();
    let pair = device
        .upload(&[5, 3], &data(15, 0.6))
        .unwrap()
        .permute(&[1, 0])
        .unwrap()
        .broadcast_to(&[2, 2, 3, 5])
        .unwrap();
    let mask = AttentionMask::Causal { query_offset: 1 };
    let spec = AttentionSpec::new(
        q.layout.shape(),
        k.layout.shape(),
        v.layout.shape(),
        -0.25,
        mask,
    )
    .unwrap();
    let expected = attention_vjp_reference(
        spec,
        &read(&q),
        &read(&k),
        &read(&v),
        Some(&read(&z)),
        Some(&read(&pair)),
        &read(&u),
    )
    .unwrap();
    let actual = q
        .scaled_dot_attention_vjp(&k, &v, &u, -0.25, mask, Some(&z), Some(&pair))
        .unwrap();
    let later = q
        .scaled_dot_attention_vjp(&k, &v, &u, 0., mask, None, None)
        .unwrap();
    close(&read(&later.query), &vec![0.; expected.query.len()]);
    let consumed = actual.query.relu().unwrap();
    drop((q, k, v, u, z, pair, later));
    compare(&actual, &expected);
    close(
        &read(&consumed),
        &expected.query.iter().map(|v| v.max(0.)).collect::<Vec<_>>(),
    );
}

#[test]
fn wide_intermediates_and_underflowing_probabilities_keep_finite_gradients() {
    let Some(device) = device() else { return };
    let qs = [1, 1, 1, 1];
    let ks = [1, 1, 2, 1];
    let q = device.upload(&qs, &[1e-38]).unwrap();
    let k = device.upload(&ks, &[-1e-38, 1e-38]).unwrap();
    let v = device.upload(&ks, &[-1e38, 1e38]).unwrap();
    let u = device.upload(&qs, &[1e38]).unwrap();
    let spec = AttentionSpec::new(&qs, &ks, &ks, 0.5, AttentionMask::None).unwrap();
    let expected =
        attention_vjp_reference(spec, &read(&q), &read(&k), &read(&v), None, None, &read(&u))
            .unwrap();
    let actual = q
        .scaled_dot_attention_vjp(&k, &v, &u, 0.5, AttentionMask::None, None, None)
        .unwrap();
    compare(&actual, &expected);
    let z = device.upload(&[1, 1, 2], &[0., 0.]).unwrap();
    let requested_overflow = q
        .scaled_dot_attention_vjp(&k, &v, &u, 0.5, AttentionMask::None, Some(&z), None)
        .unwrap();
    for output in [
        &requested_overflow.query,
        &requested_overflow.key,
        &requested_overflow.value,
        requested_overflow.z_bias.as_ref().unwrap(),
    ] {
        assert!(matches!(
            output.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    let q = device.upload(&qs, &[0.]).unwrap();
    let k = device.upload(&ks, &[1., 0.]).unwrap();
    let v = device.upload(&ks, &[1e38, 0.]).unwrap();
    let pair = device.upload(&[1, 1, 1, 2], &[-120., 0.]).unwrap();
    let spec = AttentionSpec::new(&qs, &ks, &ks, 1., AttentionMask::None).unwrap();
    let expected = attention_vjp_reference(
        spec,
        &[0.],
        &[1., 0.],
        &[1e38, 0.],
        None,
        Some(&[-120., 0.]),
        &[1e38],
    )
    .unwrap();
    assert!(expected.query[0] > 1e20);
    let actual = q
        .scaled_dot_attention_vjp(&k, &v, &u, 1., AttentionMask::None, None, Some(&pair))
        .unwrap();
    compare(&actual, &expected);
}

#[test]
fn vjp_scores_preserve_the_forward_key_tile_reduction_under_cancellation() {
    let Some(device) = device() else { return };
    let d = 24;
    for count in [127, 128, 131] {
        let q = device.upload(&[1, 1, 1, d], &vec![1.; d]).unwrap();
        let mut keys = vec![0.; count * d];
        keys[0] = 16_777_216.;
        keys[8] = 1.;
        keys[16] = -16_777_216.;
        let mut values = vec![0.; count * d];
        values[0] = 1.;
        let mut seed = vec![0.; d];
        seed[0] = 1.;
        let k = device.upload(&[1, 1, count, d], &keys).unwrap();
        let v = device.upload(&[1, 1, count, d], &values).unwrap();
        let u = device.upload(&[1, 1, 1, d], &seed).unwrap();
        let probability = read(
            &q.scaled_dot_attention(&k, &v, 1., AttentionMask::None, None, None)
                .unwrap(),
        )[0];
        let expected = if count >= 128 {
            1. / count as f32
        } else {
            1f32.exp() / (count as f32 - 1. + 1f32.exp())
        };
        close(&[probability], &[expected]);
        let result = q
            .scaled_dot_attention_vjp(&k, &v, &u, 1., AttentionMask::None, None, None)
            .unwrap();
        close(&[read(&result.value)[0]], &[probability]);
    }
}

#[test]
fn empty_queries_return_zero_key_gradients_and_validate_shape_before_dispatch() {
    let Some(device) = device() else { return };
    for qs in [[0, 2, 3, 7], [2, 0, 3, 7], [2, 2, 0, 7]] {
        let ks = [qs[0], qs[1], 5, 7];
        let q = device.upload(&qs, &[]).unwrap();
        let kv = device.upload(&ks, &vec![1.; ks.iter().product()]).unwrap();
        let actual = q
            .scaled_dot_attention_vjp(&kv, &kv, &q, 1., AttentionMask::None, None, None)
            .unwrap();
        assert!(read(&actual.query).is_empty());
        close(&read(&actual.key), &vec![0.; ks.iter().product()]);
        close(&read(&actual.value), &vec![0.; ks.iter().product()]);
    }
    let q = device.upload(&[1, 1, 1, 3], &[0.; 3]).unwrap();
    let bad = device.upload(&[3], &[0.; 3]).unwrap();
    assert!(q
        .scaled_dot_attention_vjp(&q, &q, &bad, 1., AttentionMask::None, None, None)
        .is_err());
    assert!(q
        .scaled_dot_attention_vjp(&q, &q, &q, f32::NAN, AttentionMask::None, None, None)
        .is_err());
    assert!(q
        .scaled_dot_attention_vjp(
            &q,
            &q,
            &q,
            1.,
            AttentionMask::Causal { query_offset: 1 },
            None,
            None
        )
        .is_err());
}

#[test]
fn every_cropped_failed_operand_taints_all_gradients_even_when_empty() {
    let Some(device) = device() else { return };
    let good = device.upload(&[1], &[0.]).unwrap();
    let failed = device
        .upload(&[2], &[0., f32::MAX])
        .unwrap()
        .mul(&device.upload(&[2], &[1., 2.]).unwrap())
        .unwrap()
        .narrow(0, 0, 1)
        .unwrap();
    for queries in [0, 1] {
        for slot in 0..6 {
            let input = |which, shape: &[usize]| {
                (if slot == which { &failed } else { &good })
                    .broadcast_to(shape)
                    .unwrap()
            };
            let q = input(0, &[2, 2, queries, 1]);
            let k = input(1, &[2, 2, 3, 1]);
            let v = input(2, &[2, 2, 3, 1]);
            let z = input(3, &[2, 2, 3]);
            let pair = input(4, &[2, 2, queries, 3]);
            let u = input(5, &[2, 2, queries, 1]);
            let result = q
                .scaled_dot_attention_vjp(
                    &k,
                    &v,
                    &u,
                    1.,
                    AttentionMask::Causal { query_offset: 0 },
                    Some(&z),
                    Some(&pair),
                )
                .unwrap();
            for output in [
                &result.query,
                &result.key,
                &result.value,
                result.z_bias.as_ref().unwrap(),
                result.pair_bias.as_ref().unwrap(),
            ] {
                assert!(matches!(
                    output.snapshot().unwrap().read(),
                    Err(TensorError::NonFinite)
                ));
                assert!(matches!(
                    output.relu().unwrap().snapshot().unwrap().read(),
                    Err(TensorError::NonFinite)
                ));
            }
        }
    }
}
