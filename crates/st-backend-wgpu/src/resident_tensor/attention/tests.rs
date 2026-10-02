use super::*;
use st_kernel_contracts::attention::{attention_reference, AttentionError};

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("tensor.attention.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    eprintln!("resident attention adapter: {:?}", runtime.adapter_info());
    Some(TensorDevice::new(runtime).unwrap())
}

fn read(tensor: &ResidentTensor) -> Vec<f32> {
    tensor.snapshot().unwrap().read().unwrap()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 3e-6 + 3e-5 * b.abs(),
            "[{i}] {a} != {b}"
        );
    }
}

fn data(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (i as f32 * 0.17 + phase).sin() * 0.5)
        .collect()
}

#[test]
fn shader_parses_and_validates_without_a_runtime() {
    let module = naga::front::wgsl::parse_str(SHADER_SOURCE).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(std::mem::size_of::<Params>(), 32);
}

#[test]
fn capabilities_and_portable_grid_fail_before_pipeline_creation() {
    let shape = [1, 1, 1, 257];
    let spec = AttentionSpec::new(&shape, &shape, &shape, 1., AttentionMask::None).unwrap();
    let mut limits = wgpu::Limits::default();
    assert!(preflight(spec, &limits).is_err());
    let shape = [1, 1, 5, 3];
    let spec = AttentionSpec::new(&shape, &shape, &shape, 1., AttentionMask::None).unwrap();
    limits.max_compute_workgroups_per_dimension = 2;
    assert!(preflight(spec, &limits).is_err());
    limits.max_compute_workgroups_per_dimension = 3;
    assert_eq!(preflight(spec, &limits).unwrap(), [3, 2]);
    limits.max_bindings_per_bind_group = 7;
    assert!(preflight(spec, &limits).is_err());
}

#[test]
fn plain_and_biased_attention_match_rust_across_heads_and_dimension_tails() {
    let Some(device) = device() else { return };
    for d in [1, 7, 64, 65, 127, 256] {
        let q_shape = [2, 2, 3, d];
        let k_shape = [2, 2, 5, d];
        let q = data(12 * d, 0.1);
        let k = data(20 * d, 1.2);
        let v = data(20 * d, -0.7);
        let z = data(20, 0.4);
        let pair = data(60, -0.5);
        let q_gpu = device.upload(&q_shape, &q).unwrap();
        let k_gpu = device.upload(&k_shape, &k).unwrap();
        let v_gpu = device.upload(&k_shape, &v).unwrap();
        let z_gpu = device.upload(&[2, 2, 5], &z).unwrap();
        let pair_gpu = device.upload(&[2, 2, 3, 5], &pair).unwrap();
        for mask in [
            AttentionMask::None,
            AttentionMask::Causal { query_offset: 0 },
            AttentionMask::Causal { query_offset: 2 },
        ] {
            for mode in 0..4 {
                let spec = AttentionSpec::new(&q_shape, &k_shape, &k_shape, 0.375, mask).unwrap();
                let expected = attention_reference(
                    spec,
                    &q,
                    &k,
                    &v,
                    (mode & 1 != 0).then_some(z.as_slice()),
                    (mode & 2 != 0).then_some(pair.as_slice()),
                )
                .unwrap();
                let output = q_gpu
                    .scaled_dot_attention(
                        &k_gpu,
                        &v_gpu,
                        0.375,
                        mask,
                        (mode & 1 != 0).then_some(&z_gpu),
                        (mode & 2 != 0).then_some(&pair_gpu),
                    )
                    .unwrap();
                close(&read(&output), &expected);
            }
        }
    }
}

#[test]
fn strided_qkv_and_broadcast_bias_are_packed_on_gpu_and_outputs_are_owned() {
    let Some(device) = device() else { return };
    let q = device
        .upload(&[1, 3, 2, 3], &data(18, 0.1))
        .unwrap()
        .permute(&[0, 2, 1, 3])
        .unwrap()
        .narrow(2, 1, 2)
        .unwrap();
    let k = device
        .upload(&[1, 3, 2, 3], &data(18, 0.7))
        .unwrap()
        .permute(&[0, 2, 1, 3])
        .unwrap();
    let v = device
        .upload(&[1, 3, 2, 3], &data(18, 1.2))
        .unwrap()
        .permute(&[0, 2, 1, 3])
        .unwrap();
    let z = device
        .upload(&[3], &[0.2, -0.5, 0.1])
        .unwrap()
        .broadcast_to(&[1, 2, 3])
        .unwrap();
    let pair = device
        .upload(&[2, 3], &data(6, -0.3))
        .unwrap()
        .broadcast_to(&[1, 2, 2, 3])
        .unwrap();
    let mask = AttentionMask::Causal { query_offset: 1 };
    let spec = AttentionSpec::new(
        q.layout.shape(),
        k.layout.shape(),
        v.layout.shape(),
        0.5,
        mask,
    )
    .unwrap();
    let expected = attention_reference(
        spec,
        &read(&q),
        &read(&k),
        &read(&v),
        Some(&read(&z)),
        Some(&read(&pair)),
    )
    .unwrap();
    let output = q
        .scaled_dot_attention(&k, &v, 0.5, mask, Some(&z), Some(&pair))
        .unwrap();
    let next = output.gelu().unwrap();
    drop((q, k, v, z, pair));
    close(&read(&output), &expected);
    assert!(read(&next).iter().all(|value| value.is_finite()));
    close(&read(&output), &expected);
}

#[test]
fn future_bias_cannot_unmask_and_empty_query_has_no_dispatch() {
    let Some(device) = device() else { return };
    let q = device.upload(&[1, 1, 2, 1], &[0., 0.]).unwrap();
    let v = device.upload(&[1, 1, 2, 1], &[2., 10.]).unwrap();
    let z = device.upload(&[1, 1, 2], &[0., 3f32.ln()]).unwrap();
    let pair = device
        .upload(&[1, 1, 2, 2], &[0., f32::MAX, 0., 0.])
        .unwrap();
    let output = q
        .scaled_dot_attention(
            &q,
            &v,
            1.,
            AttentionMask::Causal { query_offset: 0 },
            Some(&z),
            Some(&pair),
        )
        .unwrap();
    close(&read(&output), &[2., 8.]);
    let empty = device.upload(&[1, 1, 0, 1], &[]).unwrap();
    let output = empty
        .scaled_dot_attention(
            &q,
            &v,
            1.,
            AttentionMask::Causal { query_offset: 2 },
            None,
            None,
        )
        .unwrap();
    assert!(read(&output).is_empty());
    assert!(matches!(
        q.scaled_dot_attention(
            &q,
            &v,
            1.,
            AttentionMask::Causal { query_offset: 1 },
            None,
            None
        ),
        Err(TensorError::Attention(AttentionError::QueryOffset))
    ));
}

#[test]
fn score_overflow_and_all_upstream_guards_propagate_even_through_masked_bias() {
    let Some(device) = device() else { return };
    let x = device.upload(&[1, 1, 1, 1], &[f32::MAX]).unwrap();
    let two = device.upload(&[1, 1, 1, 1], &[2.]).unwrap();
    let good = device.upload(&[1, 1, 1, 1], &[0.]).unwrap();
    let failed = x.mul(&two).unwrap();
    let score_overflow = x
        .scaled_dot_attention(&two, &good, 1., AttentionMask::None, None, None)
        .unwrap();
    let mut outputs = vec![score_overflow];
    for slot in 0..5 {
        let z = failed.reshape(&[1, 1, 1]).unwrap();
        let q = if slot == 0 { &failed } else { &good };
        let k = if slot == 1 { &failed } else { &good };
        let v = if slot == 2 { &failed } else { &good };
        outputs.push(
            q.scaled_dot_attention(
                k,
                v,
                1.,
                AttentionMask::None,
                (slot == 3).then_some(&z),
                (slot == 4).then_some(&failed),
            )
            .unwrap(),
        );
    }
    for output in outputs {
        assert!(matches!(
            output.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
        assert!(matches!(
            output.gelu().unwrap().snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    let kv = device.upload(&[1, 1, 2, 1], &[0., 0.]).unwrap();
    let z = device.upload(&[1, 1, 2], &[0., f32::MAX]).unwrap();
    let z = z
        .mul(&device.upload(&[1, 1, 2], &[1., 2.]).unwrap())
        .unwrap();
    let output = good
        .scaled_dot_attention(
            &kv,
            &kv,
            1.,
            AttentionMask::Causal { query_offset: 0 },
            Some(&z),
            None,
        )
        .unwrap();
    assert!(matches!(
        output.snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
}

#[test]
fn device_ownership_shape_and_bias_admission_are_explicit() {
    let Some(device) = device() else { return };
    let q = device.upload(&[1, 1, 2, 3], &[0.; 6]).unwrap();
    let wrong_bias = device.upload(&[2], &[0.; 2]).unwrap();
    assert!(matches!(
        q.scaled_dot_attention(&q, &q, 1., AttentionMask::None, Some(&wrong_bias), None),
        Err(TensorError::Attention(AttentionError::BiasShape))
    ));
    assert!(matches!(
        q.scaled_dot_attention(&q, &q, f32::NAN, AttentionMask::None, None, None),
        Err(TensorError::Attention(AttentionError::Scale))
    ));
    let other_device = TensorDevice::new(
        pollster::block_on(WgpuRuntime::request_headless("attention.other_device")).unwrap(),
    )
    .unwrap();
    let other = other_device.upload(&[1, 1, 2, 3], &[0.; 6]).unwrap();
    assert!(matches!(
        q.scaled_dot_attention(&other, &q, 1., AttentionMask::None, None, None),
        Err(TensorError::DeviceMismatch)
    ));
    let shared_device = TensorDevice::new(device.runtime().clone()).unwrap();
    let shared = shared_device.upload(&[1, 1, 2, 3], &[1.; 6]).unwrap();
    close(
        &read(
            &q.scaled_dot_attention(&q, &shared, 1., AttentionMask::None, None, None)
                .unwrap(),
        ),
        &[1.; 6],
    );
}

#[test]
fn zero_bias_is_identity_and_large_finite_scores_remain_stable() {
    let Some(device) = device() else { return };
    let shape = [1, 1, 3, 1];
    let q = device.upload(&shape, &[1.; 3]).unwrap();
    let k = device.upload(&shape, &[-1e30, 0., 1e30]).unwrap();
    let v = device.upload(&shape, &[1., 2., 4.]).unwrap();
    let z = device.upload(&[1, 1, 3], &[0.; 3]).unwrap();
    let pair = device.upload(&[1, 1, 3, 3], &[0.; 9]).unwrap();
    for (scale, expected) in [(1., 4.), (-1., 1.), (0., 7. / 3.)] {
        let plain = q
            .scaled_dot_attention(&k, &v, scale, AttentionMask::None, None, None)
            .unwrap();
        let zero_biased = q
            .scaled_dot_attention(&k, &v, scale, AttentionMask::None, Some(&z), Some(&pair))
            .unwrap();
        close(&read(&plain), &[expected; 3]);
        close(&read(&zero_biased), &read(&plain));
    }
}

#[test]
fn pytorch_sdpa_fixture_matches_rust_and_resident_outputs() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/resident_attention_torch.json"
    ))
    .unwrap();
    assert_eq!(fixture["schema"], "spiraltorch.resident_attention_torch.v1");
    let device = device();
    let floats = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let shape = |v: &serde_json::Value| -> Vec<usize> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect()
    };
    for case in fixture["cases"].as_array().unwrap() {
        let q_shape = shape(&case["query_shape"]);
        let k_shape = shape(&case["key_shape"]);
        let q = floats(&case["query"]);
        let k = floats(&case["key"]);
        let v = floats(&case["value"]);
        let z = (!case["z_bias"].is_null()).then(|| floats(&case["z_bias"]));
        let pair = (!case["pair_bias"].is_null()).then(|| floats(&case["pair_bias"]));
        let expected = floats(&case["expected"]);
        let scale = case["scale"].as_f64().unwrap() as f32;
        let mask = case["query_offset"]
            .as_u64()
            .map_or(AttentionMask::None, |offset| AttentionMask::Causal {
                query_offset: offset as usize,
            });
        let spec = AttentionSpec::new(&q_shape, &k_shape, &k_shape, scale, mask).unwrap();
        let reference =
            attention_reference(spec, &q, &k, &v, z.as_deref(), pair.as_deref()).unwrap();
        close(&reference, &expected);
        if let Some(device) = &device {
            let q_gpu = device.upload(&q_shape, &q).unwrap();
            let k_gpu = device.upload(&k_shape, &k).unwrap();
            let v_gpu = device.upload(&k_shape, &v).unwrap();
            let z_gpu = z
                .as_ref()
                .map(|z| device.upload(&spec.z_bias_shape(), z).unwrap());
            let pair_gpu = pair
                .as_ref()
                .map(|p| device.upload(&spec.pair_bias_shape(), p).unwrap());
            let output = q_gpu
                .scaled_dot_attention(
                    &k_gpu,
                    &v_gpu,
                    scale,
                    mask,
                    z_gpu.as_ref(),
                    pair_gpu.as_ref(),
                )
                .unwrap();
            let actual = read(&output);
            close(&actual, &expected);
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("{} max_abs_error={max_error:e}", case["name"]);
        }
    }
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 20);
}
