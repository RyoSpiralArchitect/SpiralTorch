use super::super::tests::{fixture, saved, valid};
use super::*;

pub(in crate::resident::byte_study) fn source() -> Value {
    let mut value = fixture();
    value["schema"] = json!("spiraltorch.byte_corpus.request.v5");
    value["config"]["heads"] = json!(2);
    value["bias_calibration"] = json!({"source_request_sha256":"a".repeat(64),
        "train_batch_indices":[0,1],"relative_tolerance":1e-5});
    let mut geometry = value["cases"][1].clone();
    for p in geometry["parameters"].as_array_mut().unwrap() {
        if p["name"] == "geometry.raw_gain.0" {
            p["shape"] = json!([2]);
            p["values"] = json!([-0.3, -0.4]);
        }
    }
    value["cases"].as_array_mut().unwrap().truncate(1);
    value["cases"][0]["geometry_update"] = json!("train");
    value["cases"][0]["bias_initialization"] = json!("original");
    for (metric, matched, gains) in [
        (METRICS[0], false, [-0.3, -0.4]),
        (METRICS[1], false, [-0.3, -0.4]),
        (METRICS[1], true, [2., 3.]),
        (METRICS[2], true, [4., 5.]),
    ] {
        for update in ["train", "frozen"] {
            let mut c = geometry.clone();
            c["name"] = json!(format!("{metric:?}-{matched}-{update}"));
            c["pair_metric"] = json!(metric);
            c["geometry_update"] = json!(update);
            c["bias_initialization"] = json!(if matched {
                "matched_poincare_rms"
            } else {
                "original"
            });
            for p in c["parameters"].as_array_mut().unwrap() {
                if p["name"] == "geometry.raw_gain.0" {
                    p["values"] = json!(gains);
                }
            }
            value["cases"].as_array_mut().unwrap().push(c);
        }
    }
    assert!(valid(&value));
    value
}

pub(in crate::resident::byte_study) fn prepared() -> Value {
    serde_json::from_str(
        &ByteCorpusHeadwiseControls::from_json(source().to_string().as_bytes())
            .unwrap()
            .materialize()
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn headwise_expansion_preserves_all_nine_arms_and_checks_each_selected_gain() {
    let source = source();
    let mut value = prepared();
    assert_eq!(
        &value["cases"].as_array().unwrap()[..9],
        source["cases"].as_array().unwrap()
    );
    assert_eq!(value["cases"].as_array().unwrap().len(), 21);
    for _ in 0..21 {
        assert!(valid(&value));
        value["cases"].as_array_mut().unwrap().rotate_left(1);
    }
    for index in 9..21 {
        for head in 0..2 {
            let mut v = prepared();
            v["cases"][index]["parameters"][6]["values"][head] = json!(99.);
            assert!(!valid(&v));
        }
    }
}

#[test]
fn headwise_requires_complete_exclusive_routes_and_v6_metadata() {
    for route in [
        json!([]),
        json!([[METRICS[0]]]),
        json!([[METRICS[0], METRICS[0]]]),
        json!([[METRICS[0], METRICS[2]]]),
        json!([["unknown", METRICS[1]]]),
        Value::Null,
    ] {
        let mut v = prepared();
        v["cases"][9]["head_pair_metrics"] = route;
        assert!(!valid(&v));
    }
    for field in [
        "head_pair_metrics",
        "geometry_update",
        "bias_initialization",
    ] {
        let mut v = prepared();
        v["cases"][9].as_object_mut().unwrap().remove(field);
        assert!(!valid(&v));
    }
    let mut v = prepared();
    v["cases"][9]["pair_metric"] = json!(METRICS[0]);
    assert!(!valid(&v));
    for schema in 1..=5 {
        let mut v = prepared();
        v["schema"] = json!(format!("spiraltorch.byte_corpus.request.v{schema}"));
        assert!(!valid(&v));
    }
    for bad in [
        Value::Null,
        json!({}),
        json!({"source_request_sha256":"Z".repeat(64)}),
    ] {
        let mut v = prepared();
        v["headwise_controls"] = bad;
        assert!(!valid(&v));
    }
    let mut v = source();
    v["cases"][1]["head_pair_metrics"] = json!([[METRICS[0], METRICS[1]]]);
    assert!(!valid(&v));
}

#[test]
fn headwise_checkpoint_binds_routes_and_frozen_values() {
    let study = ByteCorpusStudy::from_json(prepared().to_string().as_bytes()).unwrap();
    assert_eq!(
        study.request.report_schema(true),
        "spiraltorch.byte_corpus.result.v6"
    );
    assert_eq!(
        study.request.report_schema(false),
        "spiraltorch.byte_corpus.partial.v6"
    );
    for cursor in 0..=2 {
        let checkpoint = saved(&study, cursor);
        assert_eq!(checkpoint.schema, "spiraltorch.byte_corpus.checkpoint.v6");
        study.validate_segment(Some(&checkpoint), cursor).unwrap();
        for index in [9, 10] {
            let mut changed = checkpoint.clone();
            let mut model: Value = serde_json::from_str(&changed.cases[index].model_json).unwrap();
            assert_eq!(model["schema"], "spiraltorch.nn.byte_decoder_checkpoint.v3");
            model["model"]["geometry"]["head_pair_metrics"] = json!([[METRICS[1], METRICS[0]]]);
            changed.cases[index].model_json = model.to_string();
            assert!(study.checked_models(&changed).is_err());
            let mut changed = checkpoint.clone();
            let mut model: Value = serde_json::from_str(&changed.cases[index].model_json).unwrap();
            model["model"]["geometry"]["raw_gains"][0][1] = json!(99.);
            changed.cases[index].model_json = model.to_string();
            assert_eq!(
                study.checked_models(&changed).is_ok(),
                cursor > 0 && index == 9
            );
        }
    }
}

#[test]
fn headwise_signed_zero_gain_inheritance_is_bitwise() {
    let mut value = source();
    for case in value["cases"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .filter(|c| c["geometry"] == true && c["bias_initialization"] == "original")
    {
        case["parameters"][6]["values"][0] = json!(-0.0);
    }
    let preparation = ByteCorpusHeadwiseControls::from_json(value.to_string().as_bytes()).unwrap();
    let mut prepared: Value = serde_json::from_str(&preparation.materialize().unwrap()).unwrap();
    assert!(valid(&prepared));
    for index in [9, 10] {
        prepared["cases"][index]["parameters"][6]["values"][0] = json!(0.0);
    }
    assert!(!valid(&prepared));
}

#[test]
fn headwise_three_seed_budget_does_not_accept_a_fourth() {
    let mut value = prepared();
    let group = value["cases"].as_array().unwrap().clone();
    for seed in [8, 9, 10] {
        for case in &group {
            let mut copy = case.clone();
            copy["seed"] = json!(seed);
            copy["name"] = json!(format!("{}-{seed}", case["name"].as_str().unwrap()));
            value["cases"].as_array_mut().unwrap().push(copy);
        }
        assert_eq!(valid(&value), seed != 10);
    }
}

#[test]
fn headwise_preparation_rejects_expanded_history_budget_before_gpu() {
    let mut value = source();
    value["train_batches"] = json!(vec![value["train_batches"][0].clone(); 1024]);
    value["validation_batches"] = json!(vec![value["validation_batches"][0].clone(); 128]);
    value["checkpoint_every"] = json!(1);
    let input = value.to_string();
    ByteCorpusStudy::from_json(input.as_bytes())
        .unwrap()
        .validate_segment(None, 0)
        .unwrap();
    assert!(ByteCorpusHeadwiseControls::from_json(input.as_bytes()).is_err());
}
