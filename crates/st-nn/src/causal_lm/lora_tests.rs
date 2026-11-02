use super::*;

fn record(model: &NativeCausalLm) -> serde_json::Value {
    serde_json::from_str(&model.lora_to_json().unwrap()).unwrap()
}

fn nonzero_adapter(base: &NativeCausalLm, layer: usize) -> NativeCausalLm {
    let mut value = record(&base.with_lora(layer, 2, 3.).unwrap());
    value["a"] = json!([0.2, -0.1, 0.3, 0.15, -0.2, 0.1, 0.05, 0.3]);
    value["b"] = json!([0.1, -0.2, 0.15, 0.05, -0.1, 0.2, 0.1, -0.15]);
    base.with_lora_json(&value.to_string()).unwrap()
}

// Independent f64 CE reduction avoids an f32 loss quantization floor in finite
// differences. Decoder forward still follows the explicitly selected policy.
fn objective(model: &NativeCausalLm, windows: &[usize], shape: [usize; 2]) -> f64 {
    let prepared = model
        .prepare_objective(windows, shape, None, CausalLmObjective::default())
        .unwrap();
    let logits = model
        .logits(&prepared.input, prepared.shape, false)
        .unwrap();
    logits
        .data()
        .chunks_exact(model.config.vocabulary)
        .zip(&prepared.labels)
        .map(|(row, &label)| {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            maximum - f64::from(row[label as usize])
                + row
                    .iter()
                    .map(|&v| (f64::from(v) - maximum).exp())
                    .sum::<f64>()
                    .ln()
        })
        .sum::<f64>()
        / prepared.labels.len() as f64
}

#[test]
fn zero_b_matches_unadapted_model_at_each_supported_layer() {
    for accumulator in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F64,
    ] {
        let base = load().with_readout_accumulator(accumulator);
        for layer in 0..base.config.layers {
            let adapted = base.with_lora(layer, 2, 3.).unwrap();
            for last_only in [false, true] {
                assert_eq!(
                    base.logits(&[1, 2, 3, 4, 5, 6], [2, 3], last_only).unwrap(),
                    adapted
                        .logits(&[1, 2, 3, 4, 5, 6], [2, 3], last_only)
                        .unwrap()
                );
            }
            assert_eq!(base.base_model_identity(), adapted.base_model_identity());
            assert!(base.lora_artifact_id().is_none());
            assert!(adapted.lora_artifact_id().is_some());
            assert!(adapted.geometry_artifact_id().is_none());
        }
    }
}

#[test]
fn native_lora_ce_pullback_matches_a_and_b_finite_differences() {
    let windows = [1, 2, 3, 4, 5, 6, 7, 8];
    for accumulator in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F64,
    ] {
        let base = load().with_readout_accumulator(accumulator);
        for layer in 0..base.config.layers {
            let model = nonzero_adapter(&base, layer);
            let gradients = model.lora_loss_gradients(&windows, [2, 4]).unwrap();
            assert_eq!(
                gradients.loss(),
                model.next_token_loss(&windows, [2, 4]).unwrap()
            );
            assert_eq!(gradients.window_shape(), [2, 4]);
            assert_eq!(gradients.target_tokens(), 6);
            assert_eq!(gradients.base_model_identity(), base.base_model_identity());
            assert_eq!(Some(gradients.lora_artifact_id()), model.lora_artifact_id());
            for (name, analytic) in [("a", gradients.a()), ("b", gradients.b())] {
                for (index, &expected) in analytic.iter().enumerate() {
                    let epsilon = 0.002;
                    let perturbed = |sign: f32| {
                        let mut value = record(&model);
                        let original = value[name][index].as_f64().unwrap() as f32;
                        value[name][index] = json!(original + sign * epsilon);
                        base.with_lora_json(&value.to_string()).unwrap()
                    };
                    let numeric = (objective(&perturbed(1.), &windows, [2, 4])
                        - objective(&perturbed(-1.), &windows, [2, 4]))
                        / (2. * f64::from(epsilon));
                    assert!((numeric - f64::from(expected)).abs() < 8e-5,
                        "{accumulator:?} layer {layer} {name}[{index}]: numeric={numeric}, analytic={expected}");
                }
            }
            // CE means target tokens, not another batch reduction in LoRA.
            let first = model.lora_loss_gradients(&windows[..4], [1, 4]).unwrap();
            let second = model.lora_loss_gradients(&windows[4..], [1, 4]).unwrap();
            for (joint, left, right) in [
                (gradients.a(), first.a(), second.a()),
                (gradients.b(), first.b(), second.b()),
            ] {
                for ((&value, &a), &b) in joint.iter().zip(left).zip(right) {
                    assert!((value - (a + b) / 2.).abs() < 1e-7);
                }
            }
        }
    }
}

#[test]
fn real_sgd_updates_only_adapter_and_preserves_original_snapshot() {
    let base = load();
    let ids = [1, 2, 3];
    let before = base.logits(&ids, [1, 3], false).unwrap();
    let model = base.with_lora(0, 2, 3.).unwrap();
    let initial = model.lora_to_json().unwrap();
    let g0 = model.lora_loss_gradients(&[1, 2, 3, 4], [1, 4]).unwrap();
    assert!(g0.a().iter().all(|&v| v == 0.));
    assert!(g0.b().iter().any(|&v| v != 0.));
    let first = model.step_lora_sgd(&g0, 1.).unwrap();
    assert_eq!(record(&first)["a"], record(&model)["a"]);
    assert_ne!(record(&first)["b"], record(&model)["b"]);
    let g1 = first.lora_loss_gradients(&[1, 2, 3, 4], [1, 4]).unwrap();
    assert!(g1.a().iter().any(|&v| v != 0.));
    let second = first.step_lora_sgd(&g1, 1.).unwrap();
    assert_ne!(record(&second)["a"], record(&first)["a"]);
    assert_ne!(record(&second)["b"], record(&first)["b"]);
    assert_eq!(model.lora_to_json().unwrap(), initial);
    assert_eq!(base.logits(&ids, [1, 3], false).unwrap(), before);
    assert_eq!(second.base_model_identity(), base.base_model_identity());
    assert_eq!(second.token, base.token);
    assert_eq!(second.position, base.position);
    // Removing only the adapter on a test clone must expose the unchanged base.
    let mut stripped = second;
    stripped.lora = None;
    assert_eq!(stripped.logits(&ids, [1, 3], false).unwrap(), before);
}

#[test]
fn adapter_round_trip_continues_stateless_sgd_exactly() {
    for accumulator in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F64,
    ] {
        let base = load().with_readout_accumulator(accumulator);
        let mut continuous = base.with_lora(1, 2, 4.).unwrap();
        let mut resumed = base.with_lora(1, 2, 4.).unwrap();
        let steps = [
            ([1, 2, 3, 4], 0.5),
            ([4, 3, 2, 1], 0.25),
            ([5, 6, 7, 8], 0.125),
        ];
        for (windows, rate) in steps {
            continuous = continuous
                .step_lora_sgd(
                    &continuous.lora_loss_gradients(&windows, [1, 4]).unwrap(),
                    rate,
                )
                .unwrap();
            // Reconstruct base and adapter each step. Stateless SGD has no
            // moments; callers explicitly retain data, rate and readout policy.
            resumed = load()
                .with_readout_accumulator(accumulator)
                .with_lora_json(&resumed.lora_to_json().unwrap())
                .unwrap();
            resumed = resumed
                .step_lora_sgd(
                    &resumed.lora_loss_gradients(&windows, [1, 4]).unwrap(),
                    rate,
                )
                .unwrap();
            assert_eq!(
                continuous.lora_to_json().unwrap(),
                resumed.lora_to_json().unwrap()
            );
            assert_eq!(continuous.lora_artifact_id(), resumed.lora_artifact_id());
            assert_eq!(
                continuous.logits(&[1, 2, 3], [1, 3], false).unwrap(),
                resumed.logits(&[1, 2, 3], [1, 3], false).unwrap()
            );
        }
    }
}

#[test]
fn lora_rejects_invalid_configuration_and_snapshot_without_mutating_owner() {
    let base = load();
    for (layer, rank, alpha) in [
        (2, 1, 1.),
        (usize::MAX, 1, 1.),
        (0, 0, 1.),
        (0, 5, 1.),
        (0, 1, 0.),
        (0, 1, -1.),
        (0, 1, f32::NAN),
        (0, 1, f32::INFINITY),
    ] {
        assert!(base.with_lora(layer, rank, alpha).is_err());
    }
    assert!(base.lora_to_json().is_err());
    assert!(base.lora_loss_gradients(&[1, 2], [1, 2]).is_err());
    let model = base.with_lora(0, 2, 2.).unwrap();
    let snapshot = model.lora_to_json().unwrap();
    assert!(model.with_lora(1, 1, 1.).is_err());
    for (key, invalid) in [
        ("schema", json!("unknown")),
        ("base_model_identity", json!("foreign")),
        ("width", json!(3)),
        ("layer", json!(2)),
        ("rank", json!(0)),
        ("rank", json!(5)),
        ("alpha", json!(0)),
        ("a", json!([0.])),
        ("b", json!([])),
        ("extra", json!(true)),
        ("alpha", serde_json::Value::Null),
    ] {
        let mut value = record(&model);
        value[key] = invalid;
        assert!(
            model.with_lora_json(&value.to_string()).is_err(),
            "accepted invalid {key}"
        );
        assert_eq!(model.lora_to_json().unwrap(), snapshot);
    }
    let mut value = record(&model);
    value["a"][0] = serde_json::Value::Null;
    assert!(model.with_lora_json(&value.to_string()).is_err());
    for invalid in ["{}", "null", "{", "NaN", "{\"alpha\":1e999}"] {
        assert!(model.with_lora_json(invalid).is_err());
    }
    assert!(base.lora_artifact_id().is_none());
}

#[test]
fn lora_seals_gradient_owner_and_readout_policy() {
    let base = load();
    let model = base.with_lora(0, 2, 2.).unwrap();
    let gradient = model.lora_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    let updated = model.step_lora_sgd(&gradient, 0.5).unwrap();
    assert!(updated.step_lora_sgd(&gradient, 0.5).is_err());
    assert!(base.step_lora_sgd(&gradient, 0.5).is_err());
    assert!(base
        .with_lora(1, 2, 2.)
        .unwrap()
        .step_lora_sgd(&gradient, 0.5)
        .is_err());
    let wide = model.with_readout_accumulator(CausalLmReadoutAccumulator::F64);
    assert!(wide.step_lora_sgd(&gradient, 0.5).is_err());
    let wide_gradient = wide.lora_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    assert!(model.step_lora_sgd(&wide_gradient, 0.5).is_err());
    for rate in [0., -0.1, f32::NAN, f32::INFINITY] {
        assert!(model.step_lora_sgd(&gradient, rate).is_err());
    }
    assert_eq!(model.lora_artifact_id(), Some(gradient.lora_artifact_id()));
}

#[test]
fn lora_admits_work_before_token_extraction_and_rejects_excess_budgets() {
    let mut base = load();
    base.limits.max_parameter_elements = 15; // width 4 * rank 2 * A/B = 16.
    assert!(base.with_lora(0, 2, 2.).is_err());
    let mut model = load().with_lora(0, 2, 2.).unwrap();
    model.limits.max_training_checkpoint_elements = 23; // 3 checkpoints * 2 rows * 4.
    assert!(model.lora_gradient_input_count([1, 3]).is_err());
    assert!(model.lora_loss_gradients(&[1, 2, 3], [1, 3]).is_err());
    model.limits.max_training_checkpoint_elements = 24;
    assert_eq!(model.lora_gradient_input_count([1, 3]).unwrap(), 3);
    for shape in [[0, 3], [1, 0], [1, 1], [1, 10], [usize::MAX, 3]] {
        assert!(
            model.lora_gradient_input_count(shape).is_err(),
            "accepted {shape:?}"
        );
    }
    assert!(model.lora_loss_gradients(&[1, 2], [1, 3]).is_err());
    assert!(model.lora_loss_gradients(&[1, 2, 11], [1, 3]).is_err());
}

#[test]
fn composed_models_require_joint_owner_api_and_preserve_geometry_learning() {
    let base = load();
    let mut record = artifact(&base, "transformer.h.0.attn", 0.2)
        .record()
        .clone();
    record.training = true;
    record.placements[0].training = true;
    record.placements[0].tensors[0].requires_grad = true;
    let artifact = GeometryAdapterArtifact::create(record).unwrap();
    let geometry = base.with_geometry(artifact.clone()).unwrap();
    let before = geometry
        .geometry_loss_gradients(&[1, 2, 3], [1, 3])
        .unwrap();
    let lora = base.with_lora(0, 2, 2.).unwrap();
    assert!(geometry.with_lora(0, 2, 2.).is_ok());
    assert!(geometry
        .with_lora_json(&lora.lora_to_json().unwrap())
        .is_ok());
    assert!(lora.with_geometry(artifact).is_ok());
    assert!(lora.geometry_loss_gradients(&[1, 2, 3], [1, 3]).is_err());
    let after = geometry
        .geometry_loss_gradients(&[1, 2, 3], [1, 3])
        .unwrap();
    assert_eq!(before.loss(), after.loss());
    assert_eq!(before.parameters(), after.parameters());
    assert!(geometry.step_geometry_sgd(&after, 0.1).is_ok());
}

#[test]
fn native_parameter_pullback_stops_before_unused_overflowing_input_gradient() {
    let mut weights = parameters();
    weights
        .get_mut("h.0.attn.c_attn.weight")
        .unwrap()
        .1
        .fill(0.);
    weights.get_mut("h.0.attn.c_attn.bias").unwrap().1.fill(0.);
    let base = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&weights),
        Default::default(),
    )
    .unwrap();
    let mut value = record(&base.with_lora(0, 1, 1.).unwrap());
    value["a"] = json!([1e30, 1e30, 1e30, 1e30]);
    value["b"] = json!([1e20, -1e20, 1e20, -1e20]);
    let model = base.with_lora_json(&value.to_string()).unwrap();
    let gradient = model.lora_loss_gradients(&[1, 2, 3], [1, 3]).unwrap();
    assert!(gradient.loss().is_finite());
    assert!(gradient.a().iter().chain(gradient.b()).all(|&v| v == 0.));
}

#[test]
fn seeded_initialization_exercises_requested_rank_beyond_two() {
    let base = load();
    for rank in [3, 4] {
        let model = base.with_lora(0, rank, 4.).unwrap();
        let snapshot = record(&model);
        assert_eq!(
            model.lora_to_json().unwrap(),
            base.with_lora(0, rank, 4.).unwrap().lora_to_json().unwrap()
        );
        let values: Vec<f64> = snapshot["a"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        // Modified Gram-Schmidt on the requested columns detects the old
        // additive row/column initializer's absent third and fourth directions.
        let mut basis: Vec<Vec<f64>> = Vec::new();
        for column in 0..rank {
            let mut vector: Vec<_> = (0..4).map(|row| values[row * rank + column]).collect();
            for previous in &basis {
                let projection: f64 = vector.iter().zip(previous).map(|(a, b)| a * b).sum();
                for (v, p) in vector.iter_mut().zip(previous) {
                    *v -= projection * p;
                }
            }
            let norm = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                norm > 1e-5,
                "rank {rank}, column {column} is degenerate: {norm}"
            );
            for v in &mut vector {
                *v /= norm;
            }
            basis.push(vector);
        }
        assert_eq!(
            model.logits(&[1, 2, 3], [1, 3], false).unwrap(),
            base.logits(&[1, 2, 3], [1, 3], false).unwrap()
        );
    }
}
