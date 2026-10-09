use super::*;
use serde_json::Value;

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("attention.training.tests")
            .unwrap();
    assert_ne!(format!("{:?}", runtime.adapter_info().device_type), "Cpu");
    eprintln!(
        "resident attention training adapter: {:?}",
        runtime.adapter_info()
    );
    Some(TensorDevice::new(runtime).unwrap())
}

fn fixture() -> Value {
    let value: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/resident_attention_training_torch.json"
    )))
    .unwrap();
    assert_eq!(
        value["schema"],
        "spiraltorch.resident_attention_training_torch.v1"
    );
    assert_eq!(value["cases"].as_array().unwrap().len(), 30);
    value
}
fn data(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn shape(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}
fn read(t: &ResidentTensor) -> Vec<f32> {
    t.snapshot().unwrap().read().unwrap()
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite()
                && e.is_finite()
                && (f64::from(a) - f64::from(e)).abs() <= 3e-6 + 5e-5 * f64::from(e).abs(),
            "[{i}] {a} != {e}"
        );
    }
}
fn plan(case: &Value) -> AttentionInferencePlan {
    let parameters: Vec<_> = case["projections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let dims = shape(&p["weight_shape"]);
            (
                Tensor::from_vec(dims[0], dims[1], data(&p["weight"])).unwrap(),
                Tensor::from_vec(1, dims[1], data(&p["bias"])).unwrap(),
            )
        })
        .collect();
    AttentionInferencePlan::from_parameters(
        NdLayout::contiguous(&shape(&case["input_shape"])).unwrap(),
        case["heads"].as_u64().unwrap() as usize,
        if case["causal"].as_bool().unwrap() {
            AttentionMask::Causal { query_offset: 0 }
        } else {
            AttentionMask::None
        },
        std::array::from_fn(|i| (&parameters[i].0, &parameters[i].1)),
    )
    .unwrap()
}
fn biases(device: &TensorDevice, case: &Value) -> [Option<ResidentTensor>; 2] {
    let input = shape(&case["input_shape"]);
    let dims = [
        input[0],
        case["heads"].as_u64().unwrap() as usize,
        input[1],
        input[1],
    ];
    ["z_bias", "pair_bias"].map(|name| {
        (!case[name].is_null()).then(|| {
            device
                .upload(
                    if name == "z_bias" { &dims[..3] } else { &dims },
                    &data(&case[name]),
                )
                .unwrap()
        })
    })
}

#[test]
fn projections_and_geometry_match_independent_torch_in_thirty_conditions() {
    let Some(device) = device() else {
        return;
    };
    let fixture = fixture();
    for case in fixture["cases"].as_array().unwrap() {
        eprintln!("attention projection case: {}", case["name"]);
        let plan = plan(case);
        let mut model = plan
            .compile_training_wgpu(device.runtime().clone())
            .unwrap();
        let input = device
            .upload(plan.input_layout().shape(), &data(&case["input"]))
            .unwrap();
        let upstream = device
            .upload(plan.output_layout().shape(), &data(&case["upstream"]))
            .unwrap();
        let [z, pair] = biases(&device, case);
        let forward = model.forward(&input, z.as_ref(), pair.as_ref()).unwrap();
        close(&read(forward.prediction()), &data(&case["expected"]));
        let gradients = model.backward(&forward, &upstream).unwrap();
        close(
            &read(gradients.input_gradient()),
            &data(&case["input_gradient"]),
        );
        assert_eq!(gradients.parameter_gradients().len(), 4);
        for (actual, expected) in gradients
            .parameter_gradients()
            .iter()
            .zip(case["parameter_gradients"].as_array().unwrap())
        {
            close(&read(actual), &data(expected));
        }
        for (actual, name) in [
            (gradients.z_bias_gradient(), "z_bias_gradient"),
            (gradients.pair_bias_gradient(), "pair_bias_gradient"),
        ] {
            assert_eq!(actual.is_none(), case[name].is_null());
            if let Some(actual) = actual {
                close(&read(actual), &data(&case[name]));
            }
        }
        drop((model, input, z, pair, upstream));
        close(&read(forward.prediction()), &data(&case["expected"]));
        close(
            &read(gradients.input_gradient()),
            &data(&case["input_gradient"]),
        );
    }
}

#[test]
fn sixteen_resident_projection_updates_match_torch_without_intermediate_readback() {
    let Some(device) = device() else {
        return;
    };
    let fixture = fixture();
    let case = &fixture["training"];
    let plan = plan(case);
    let mut model = plan
        .compile_training_wgpu(device.runtime().clone())
        .unwrap();
    let input = device
        .upload(plan.input_layout().shape(), &data(&case["input"]))
        .unwrap();
    let target = device
        .upload(plan.output_layout().shape(), &data(&case["target"]))
        .unwrap();
    let [z, pair] = biases(&device, case);
    let initial = model.parameter_snapshot();
    let mut losses = Vec::new();
    let mut updates = Vec::new();
    for _ in 0..16 {
        let forward = model.forward(&input, z.as_ref(), pair.as_ref()).unwrap();
        let loss = forward.prediction().mean_squared_error(&target).unwrap();
        let gradients = model
            .backward(&forward, loss.prediction_gradient())
            .unwrap();
        losses.push(loss.value().clone());
        updates.push(
            model
                .sgd(&gradients, case["rate"].as_f64().unwrap() as f32)
                .unwrap(),
        );
    }
    assert_eq!(model.parameter_snapshot().revision(), 16);
    for (i, (loss, update)) in losses.iter().zip(&updates).enumerate() {
        assert_eq!(update.snapshot().unwrap().read().unwrap(), (i + 1) as u64);
        close(&read(loss), &[case["losses"][i].as_f64().unwrap() as f32]);
    }
    let final_parameters = model.parameter_snapshot();
    for (actual, expected) in final_parameters
        .values()
        .iter()
        .zip(case["final_parameters"].as_array().unwrap())
    {
        close(&read(actual), &data(expected));
    }
    let forward = model.forward(&input, z.as_ref(), pair.as_ref()).unwrap();
    close(
        &read(forward.prediction()),
        &data(&case["final_prediction"]),
    );
    let mut original = plan
        .compile_training_wgpu(device.runtime().clone())
        .unwrap();
    let original_forward = original.forward(&input, z.as_ref(), pair.as_ref()).unwrap();
    close(
        &read(original_forward.prediction()),
        &data(&case["expected"]),
    );
    for (before, unchanged) in initial
        .values()
        .iter()
        .zip(original.parameter_snapshot().values())
    {
        assert_eq!(read(before), read(unchanged));
    }
}

#[test]
fn stale_foreign_and_failed_updates_cannot_partially_change_projections() {
    let Some(device) = device() else {
        return;
    };
    let fixture = fixture();
    let case = &fixture["training"];
    let plan = plan(case);
    let mut model = plan
        .compile_training_wgpu(device.runtime().clone())
        .unwrap();
    let mut foreign = plan
        .compile_training_wgpu(device.runtime().clone())
        .unwrap();
    let input = device
        .upload(plan.input_layout().shape(), &data(&case["input"]))
        .unwrap();
    let seed = device
        .upload(plan.output_layout().shape(), &data(&case["upstream"]))
        .unwrap();
    let [z, pair] = biases(&device, case);
    let forward = model.forward(&input, z.as_ref(), pair.as_ref()).unwrap();
    let _ = foreign.forward(&input, z.as_ref(), pair.as_ref()).unwrap();
    assert!(matches!(
        foreign.backward(&forward, &seed),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ));
    let good = model.backward(&forward, &seed).unwrap();
    assert!(matches!(
        foreign.sgd(&good, 0.03),
        Err(InferenceError::Training(TrainingError::ParameterVersion))
    ));
    let before = model.parameter_snapshot();
    let huge = device
        .upload(
            plan.output_layout().shape(),
            &vec![f32::MAX; plan.output_layout().len()],
        )
        .unwrap();
    let invalid = huge.mul(&huge).unwrap();
    let bad = model.backward(&forward, &invalid).unwrap();
    let rejected = model.sgd(&bad, 0.03).unwrap();
    assert!(matches!(
        rejected.snapshot().unwrap().read(),
        Err(TrainingError::Rejected { .. })
    ));
    assert_eq!(model.parameter_snapshot().revision(), 1);
    for (a, b) in before
        .values()
        .iter()
        .zip(model.parameter_snapshot().values())
    {
        assert_eq!(
            read(a).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            read(b).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    assert!(matches!(
        model.sgd(&good, 0.03),
        Err(InferenceError::Training(TrainingError::ParameterVersion))
    ));
    assert!(matches!(
        model.backward(&forward, &seed),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ));
    let valid = model.forward(&input, z.as_ref(), pair.as_ref()).unwrap();
    close(&read(valid.prediction()), &data(&case["expected"]));
    assert!(model
        .backward(&valid, &device.upload(&[1], &[1.]).unwrap())
        .is_err());
    let good = model.backward(&valid, &seed).unwrap();
    assert!(model.sgd(&good, f32::NAN).is_err());
    let zero = model.sgd(&good, 0.).unwrap();
    assert_eq!(zero.snapshot().unwrap().read().unwrap(), 2);
    for (a, b) in before
        .values()
        .iter()
        .zip(model.parameter_snapshot().values())
    {
        assert_eq!(read(a), read(b));
    }
    let valid = model.forward(&input, z.as_ref(), pair.as_ref()).unwrap();
    let latest = model.forward(&input, z.as_ref(), pair.as_ref()).unwrap();
    assert!(matches!(
        model.backward(&valid, &seed),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ));
    assert!(model.backward(&latest, &seed).is_ok());
}

#[test]
fn one_projection_overflow_rejects_changes_to_both_projection_groups() {
    let Some(device) = device() else {
        return;
    };
    // With one token, y = (x * Wv + bv) * Wo + bo. These finite
    // gradients overflow only one projection's candidate at rate MAX.
    for (value_weight, output_weight, failing_group) in [(0., 2., 0), (2., 0.5, 1)] {
        let weights = [0., 0., value_weight, output_weight]
            .map(|value| Tensor::from_vec(1, 1, vec![value]).unwrap());
        let bias = Tensor::from_vec(1, 1, vec![0.]).unwrap();
        let plan = AttentionInferencePlan::from_parameters(
            NdLayout::contiguous(&[1, 1, 1]).unwrap(),
            1,
            AttentionMask::None,
            std::array::from_fn(|i| (&weights[i], &bias)),
        )
        .unwrap();
        let mut model = plan
            .compile_training_wgpu(device.runtime().clone())
            .unwrap();
        let input = device.upload(&[1, 1, 1], &[1.]).unwrap();
        let forward = model.forward(&input, None, None).unwrap();
        let gradients = model.backward(&forward, &input).unwrap();
        let before = model.parameter_snapshot();
        let mut overflow = [false; 2];
        let mut would_change = [false; 2];
        for (i, (parameter, gradient)) in before
            .values()
            .iter()
            .zip(gradients.parameter_gradients())
            .enumerate()
        {
            for (p, g) in read(parameter).into_iter().zip(read(gradient)) {
                assert!(g.is_finite());
                let candidate = p - f32::MAX * g;
                overflow[i / 2] |= !candidate.is_finite();
                would_change[i / 2] |= p.to_bits() != candidate.to_bits();
            }
        }
        assert!(overflow[failing_group]);
        assert!(!overflow[1 - failing_group]);
        assert!(would_change[1 - failing_group]);
        let rejected = model.sgd(&gradients, f32::MAX).unwrap();
        assert!(matches!(
            rejected.snapshot().unwrap().read(),
            Err(TrainingError::Rejected { .. })
        ));
        assert_eq!(model.parameter_snapshot().revision(), 1);
        for (a, b) in before
            .values()
            .iter()
            .zip(model.parameter_snapshot().values())
        {
            assert!(read(a)
                .iter()
                .map(|v| v.to_bits())
                .eq(read(b).iter().map(|v| v.to_bits())));
        }
    }
}

#[test]
fn held_gradients_survive_a_distinct_backward_and_remain_applicable() {
    let Some(device) = device() else {
        return;
    };
    let fixture = fixture();
    let case = &fixture["training"];
    let plan = plan(case);
    let mut model = plan
        .compile_training_wgpu(device.runtime().clone())
        .unwrap();
    let input = device
        .upload(plan.input_layout().shape(), &data(&case["input"]))
        .unwrap();
    let seed = device
        .upload(plan.output_layout().shape(), &data(&case["upstream"]))
        .unwrap();
    let negative = device
        .upload(
            plan.output_layout().shape(),
            &data(&case["upstream"])
                .into_iter()
                .map(|v| -v)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let [z, pair] = biases(&device, case);
    let forward = model.forward(&input, z.as_ref(), pair.as_ref()).unwrap();
    let original = model.backward(&forward, &seed).unwrap();
    let later = model.backward(&forward, &negative).unwrap();
    close(
        &read(original.input_gradient()),
        &data(&case["input_gradient"]),
    );
    for (gradient, name) in [
        (original.z_bias_gradient(), "z_bias_gradient"),
        (original.pair_bias_gradient(), "pair_bias_gradient"),
    ] {
        close(&read(gradient.unwrap()), &data(&case[name]));
    }
    let before = model.parameter_snapshot();
    let mut expected_update = Vec::new();
    for (i, reference) in case["parameter_gradients"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let expected = data(reference);
        close(&read(&original.parameter_gradients()[i]), &expected);
        close(
            &read(&later.parameter_gradients()[i]),
            &expected.iter().map(|v| -v).collect::<Vec<_>>(),
        );
        expected_update.push(
            read(&before.values()[i])
                .into_iter()
                .zip(expected)
                .map(|(p, g)| p - 0.03 * g)
                .collect::<Vec<_>>(),
        );
    }
    let accepted = model.sgd(&original, 0.03).unwrap();
    assert_eq!(accepted.snapshot().unwrap().read().unwrap(), 1);
    for (actual, expected) in model
        .parameter_snapshot()
        .values()
        .iter()
        .zip(expected_update)
    {
        close(&read(actual), &expected);
    }
}
