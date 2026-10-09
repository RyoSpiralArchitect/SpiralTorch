use super::super::tests::{fixture, saved, valid};
use super::*;

fn source() -> Value {
    let mut value = fixture();
    value["schema"] = json!("spiraltorch.byte_corpus.request.v3");
    value["cases"][0]["geometry_update"] = json!("train");
    let geometric = value["cases"][1].clone();
    value["cases"].as_array_mut().unwrap().truncate(1);
    for metric in ["poincare_squared.v1", "euclidean_chord_squared.v1"] {
        for update in ["train", "frozen"] {
            let mut case = geometric.clone();
            case["name"] = json!(format!("{metric}-{update}"));
            case["geometry_update"] = json!(update);
            case["pair_metric"] = json!(metric);
            value["cases"].as_array_mut().unwrap().push(case);
        }
    }
    value
}

fn prepared() -> Value {
    let prep =
        ByteCorpusBiasCalibration::from_json(source().to_string().as_bytes(), &[0, 1]).unwrap();
    serde_json::from_str(
        &prep
            .materialize(&BTreeMap::from([(7, vec![vec![0.25]])]))
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn seven_explicit_arms_are_order_independent_and_only_gains_can_differ() {
    let mut value = prepared();
    for _ in 0..7 {
        assert!(valid(&value));
        value["cases"].as_array_mut().unwrap().rotate_left(1);
    }
    for case in 1..7 {
        for slot in [0, 2, 3, 4, 5, 6] {
            let mut value = prepared();
            value["cases"][case]["parameters"][slot]["values"][0] = json!(0.75);
            assert!(!valid(&value), "one-arm mutation at {case}/{slot}");
        }
    }
    for field in ["pair_metric", "geometry_update", "bias_initialization"] {
        let mut value = prepared();
        value["cases"][5].as_object_mut().unwrap().remove(field);
        assert!(!valid(&value));
        value["cases"][5][field] = Value::Null;
        assert!(!valid(&value));
    }
    let mut value = prepared();
    value["cases"][5]["bias_initialization"] = json!("original");
    assert!(!valid(&value));
    // Matching the first geometric arm must not exempt the original group.
    let mut value = prepared();
    value["cases"].as_array_mut().unwrap().rotate_right(2);
    value["cases"][3]["parameters"][6]["values"][0] = json!(0.75);
    assert!(!valid(&value));
}

#[test]
fn calibration_metadata_and_training_selection_are_explicit_and_bounded() {
    for indices in [vec![], vec![0, 0], vec![2], vec![0; 17]] {
        assert!(
            ByteCorpusBiasCalibration::from_json(source().to_string().as_bytes(), &indices)
                .is_err()
        );
    }
    for (field, bad) in [
        ("source_request_sha256", json!("a".repeat(63))),
        ("source_request_sha256", json!("A".repeat(64))),
        ("relative_tolerance", json!(1e-4)),
        ("train_batch_indices", json!([])),
        ("train_batch_indices", json!([0, 0])),
        ("train_batch_indices", json!([2])),
        ("validation_batch_indices", json!([0])),
    ] {
        let mut value = prepared();
        value["bias_calibration"][field] = bad;
        assert!(!valid(&value));
    }
    for value in [Value::Null, json!({})] {
        let mut request = prepared();
        request["bias_calibration"] = value;
        assert!(!valid(&request));
    }
    let mut request = prepared();
    request.as_object_mut().unwrap().remove("bias_calibration");
    assert!(!valid(&request));
    for mut legacy in [fixture(), source()] {
        legacy["bias_calibration"] = prepared()["bias_calibration"].clone();
        assert!(!valid(&legacy));
        legacy["bias_calibration"] = Value::Null;
        assert!(!valid(&legacy));
        legacy.as_object_mut().unwrap().remove("bias_calibration");
        legacy["cases"][0]["bias_initialization"] = json!("original");
        assert!(!valid(&legacy));
    }
    assert!(ByteCorpusBiasCalibration::from_json(fixture().to_string().as_bytes(), &[0]).is_err());
}

#[test]
fn materialization_preserves_source_bits_and_binds_exact_source_bytes() {
    let mut source = source();
    for case in source["cases"].as_array_mut().unwrap() {
        case["parameters"][0]["values"][0] = json!(-0.0);
        case["parameters"][0]["values"][1] = json!(f32::from_bits(0x3e99999a));
    }
    let bytes = source.to_string();
    let prep = ByteCorpusBiasCalibration::from_json(bytes.as_bytes(), &[1, 0]).unwrap();
    let text = prep
        .materialize(&BTreeMap::from([(7, vec![vec![-0.0]])]))
        .unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        value["bias_calibration"]["source_request_sha256"],
        json!(format!("{:x}", Sha256::digest(bytes.as_bytes())))
    );
    for index in 0..5 {
        let mut case = value["cases"][index].clone();
        case.as_object_mut().unwrap().remove("bias_initialization");
        assert_eq!(case, source["cases"][index]);
    }
    let study = ByteCorpusStudy::from_json(text.as_bytes()).unwrap();
    assert_eq!(
        study.request.cases[5].parameters[6].values[0].to_bits(),
        (-0.0f32).to_bits()
    );
    assert!(policy::same_bits(
        &study.request.cases[5].parameters[6].values,
        &study.request.cases[6].parameters[6].values
    ));
}

#[test]
fn three_complete_groups_fit_but_a_fourth_is_rejected() {
    let mut request = prepared();
    let group = request["cases"].as_array().unwrap().clone();
    for seed in [8, 9, 10] {
        for case in &group {
            let mut copy = case.clone();
            copy["name"] = json!(format!("{}-{seed}", case["name"].as_str().unwrap()));
            copy["seed"] = json!(seed);
            request["cases"].as_array_mut().unwrap().push(copy);
        }
        assert_eq!(valid(&request), seed != 10);
    }
}

#[test]
fn materialization_preserves_distinct_gains_for_every_block_and_head() {
    let mut value = source();
    value["config"]["heads"] = json!(2);
    value["config"]["blocks"] = json!([false, false]);
    for case in value["cases"].as_array_mut().unwrap() {
        let geometry = case["geometry"].as_bool().unwrap();
        let parameters = case["parameters"].as_array_mut().unwrap();
        let second_block = parameters
            .iter()
            .filter_map(|p| {
                let name = p["name"].as_str().unwrap();
                name.strip_prefix("block.0.").map(|suffix| {
                    let mut copy = p.clone();
                    copy["name"] = json!(format!("block.1.{suffix}"));
                    copy
                })
            })
            .collect::<Vec<_>>();
        let head = parameters
            .iter()
            .position(|p| p["name"] == "head.gain")
            .unwrap();
        parameters.splice(head..head, second_block);
        if geometry {
            parameters[6]["shape"] = json!([2]);
            parameters[6]["values"] = json!([0.1, 0.2]);
            let mut gain = parameters[6].clone();
            gain["name"] = json!("geometry.raw_gain.1");
            gain["values"] = json!([-0.3, 0.4]);
            parameters.insert(7, gain);
        }
    }
    assert!(valid(&value));
    let preparation =
        ByteCorpusBiasCalibration::from_json(value.to_string().as_bytes(), &[0, 1]).unwrap();
    let gains = vec![vec![0.21, 0.32], vec![-0.43, 0.54]];
    let text = preparation
        .materialize(&BTreeMap::from([(7, gains.clone())]))
        .unwrap();
    let study = ByteCorpusStudy::from_json(text.as_bytes()).unwrap();
    for index in [5, 6] {
        for (block, values) in gains.iter().enumerate() {
            assert!(policy::same_bits(
                &study.request.cases[index].parameters[6 + block].values,
                values
            ));
        }
        let plan = plan(&study.request.config, &study.request.cases[index]).unwrap();
        assert_eq!(plan.block_count(), 2);
        assert_eq!(plan.causal_geometry().unwrap().raw_gains(), &gains);
    }
}

#[test]
fn checkpoint_binds_actual_fitted_initials_frozen_values_and_calibration_recipe() {
    let mut request = prepared();
    let batches = request["train_batches"].as_array().unwrap().clone();
    request["train_batches"] = json!(batches.into_iter().cycle().take(128).collect::<Vec<_>>());
    request["checkpoint_every"] = json!(64);
    let study = ByteCorpusStudy::from_json(request.to_string().as_bytes()).unwrap();
    assert_eq!(
        study.request.report_schema(true),
        "spiraltorch.byte_corpus.result.v4"
    );
    assert_eq!(
        study.request.report_schema(false),
        "spiraltorch.byte_corpus.partial.v4"
    );
    for cursor in [0, 37, 64, 128] {
        let checkpoint = saved(&study, cursor);
        assert_eq!(checkpoint.schema, "spiraltorch.byte_corpus.checkpoint.v4");
        study.validate_segment(Some(&checkpoint), cursor).unwrap();
        assert!(study.checkpoint_size_bound().unwrap() >= checkpoint.to_json().unwrap().len());
        for i in [5, 6] {
            let mut changed = checkpoint.clone();
            let mut model: Value = serde_json::from_str(&changed.cases[i].model_json).unwrap();
            model["model"]["geometry"]["raw_gains"][0][0] = json!(0.1);
            changed.cases[i].model_json = model.to_string();
            assert_eq!(study.checked_models(&changed).is_ok(), cursor > 0 && i == 5);
        }
    }
    let saved = saved(&study, 37).to_json().unwrap();
    for (field, value) in [
        ("train_batch_indices", json!([1, 0])),
        ("source_request_sha256", json!("f".repeat(64))),
    ] {
        let mut changed = request.clone();
        changed["bias_calibration"][field] = value;
        let other = ByteCorpusStudy::from_json(changed.to_string().as_bytes()).unwrap();
        assert!(other.checkpoint_from_json(saved.as_bytes()).is_err());
    }
}

#[test]
fn realized_rms_gate_rejects_zero_mismatches_and_incomplete_coverage() {
    let a = CausalBiasMoments::from_scores([1, 1, 2, 2], &[0., 0., -1., 0.]).unwrap();
    let b = CausalBiasMoments::from_scores([1, 1, 2, 2], &[0., 0., -2., 0.]).unwrap();
    let zero = CausalBiasMoments::from_scores([1, 1, 2, 2], &[0.; 4]).unwrap();
    assert!(checked_errors(std::slice::from_ref(&a), std::slice::from_ref(&a)).is_ok());
    assert!(checked_errors(&[a.clone()], &[b]).is_err());
    assert!(checked_errors(&[zero.clone()], &[zero]).is_err());
    assert!(checked_errors(&[a], &[]).is_err());
    let two_heads =
        CausalBiasMoments::from_scores([1, 2, 2, 2], &[0., 0., -1., 0., 0., 0., -2., 0.]).unwrap();
    let changed_last_head =
        CausalBiasMoments::from_scores([1, 2, 2, 2], &[0., 0., -1., 0., 0., 0., -2.1, 0.]).unwrap();
    let reference = [two_heads.clone(), two_heads.clone()];
    assert!(checked_errors(&reference, &reference).is_ok());
    assert!(checked_errors(&reference, &[two_heads.clone(), changed_last_head]).is_err());
    let mut more_pairs = two_heads.clone();
    more_pairs.merge(&two_heads).unwrap();
    assert!(checked_errors(&reference, &[two_heads, more_pairs]).is_err());
}
