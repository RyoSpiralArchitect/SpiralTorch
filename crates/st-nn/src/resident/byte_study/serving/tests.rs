use super::super::tests::{fixture, saved};
use super::*;

pub(super) fn gate_value() -> Value {
    json!({"schema":GATE_SCHEMA,"documents":[[7,8,9,10,11]],
        "batches":[[[0,0]],[[0,2]]],"steps":2,
        "limits":{"max_mean_ce":10.,"max_mean_ce_regression":1.,
            "max_batch_ce_regression":1.,"max_geometry_centered_rms":100.,
            "max_router_mass_error":0.00001,"min_router_normalized_entropy":0.}})
}

pub(super) fn study() -> ByteCorpusStudy {
    ByteCorpusStudy::from_json(fixture().to_string().as_bytes()).unwrap()
}

pub(super) fn cohort_gate_value() -> Value {
    let legacy = gate_value();
    json!({"schema":"spiraltorch.byte_corpus.serving_gate.v2",
    "documents":legacy["documents"], "steps":legacy["steps"],
    "cohorts":[
        {"name":"prefix","batches":[[[0,0]]],"limits":legacy["limits"]},
        {"name":"suffix","batches":[[[0,2]],[[0,2]]],"limits":legacy["limits"]}
    ]})
}

#[test]
fn gate_preserves_exact_data_threshold_identity_without_a_device() {
    let raw = gate_value().to_string();
    let gate = ByteCorpusServingGate::from_json(raw.as_bytes()).unwrap();
    assert_eq!(gate.to_json(), raw);
    assert_eq!(gate.batch_count(), 2);
    assert_eq!(gate.batch_shape(), [1, 2]);
    assert_eq!(gate.schema_version(), 1);
    assert_eq!(gate.cohort_names(), ["default"]);
    assert_eq!(gate.cohorts[0].batches[1].input_bytes(), &[9, 10]);
    assert_eq!(gate.cohorts[0].batches[1].target_bytes(), &[10, 11]);
    // Serde's original v1 struct also admitted positional JSON arrays.
    let value = gate_value();
    let positional = json!([
        value["schema"],
        value["documents"],
        value["batches"],
        value["steps"],
        value["limits"]
    ])
    .to_string();
    let legacy = ByteCorpusServingGate::from_json(positional.as_bytes()).unwrap();
    assert_eq!(legacy.to_json(), positional);
    assert_eq!(legacy.schema_version(), 1);
    assert_eq!(legacy.batch_count(), gate.batch_count());
    assert_eq!(legacy.cohorts[0].batches[1].target_bytes(), &[10, 11]);
    let other = ByteCorpusServingGate::from_json(format!("{raw} ").as_bytes()).unwrap();
    assert_ne!(other.request_sha256(), gate.request_sha256());
    let mut edited = gate_value();
    edited["limits"]["max_mean_ce_regression"] = json!(0.);
    assert_ne!(
        ByteCorpusServingGate::from_json(edited.to_string().as_bytes())
            .unwrap()
            .request_sha256(),
        gate.request_sha256()
    );
}

#[test]
fn gate_rejects_ambiguous_limits_shapes_and_cross_document_windows() {
    for (key, value) in [
        ("schema", json!("unknown")),
        ("steps", json!(0)),
        ("steps", json!(513)),
        ("steps", json!(true)),
        ("documents", json!([])),
        ("batches", json!([])),
        ("batches", json!([[]])),
        ("batches", json!([[[0, 0]], [[0, 1], [0, 2]]])),
        ("batches", json!([[[0, 4]]])),
        ("batches", json!([[[1, 0]]])),
        ("limits", Value::Null),
        ("extra", json!(true)),
    ] {
        let mut invalid = gate_value();
        invalid[key] = value;
        assert!(
            ByteCorpusServingGate::from_json(invalid.to_string().as_bytes()).is_err(),
            "{key}"
        );
    }
    for (key, value) in [
        ("max_mean_ce", json!(-1)),
        ("max_geometry_centered_rms", json!(-1)),
        ("max_mean_ce_regression", Value::Null),
        ("max_batch_ce_regression", json!(true)),
        ("max_router_mass_error", json!(1.)),
        ("max_router_mass_error", json!(-1.)),
        ("min_router_normalized_entropy", json!(1.01)),
        ("unknown", json!(0.)),
    ] {
        let mut invalid = gate_value();
        invalid["limits"][key] = value;
        assert!(
            ByteCorpusServingGate::from_json(invalid.to_string().as_bytes()).is_err(),
            "{key}"
        );
    }
    let mut missing = gate_value();
    missing["limits"]
        .as_object_mut()
        .unwrap()
        .remove("max_mean_ce");
    assert!(ByteCorpusServingGate::from_json(missing.to_string().as_bytes()).is_err());
    assert!(
        ByteCorpusServingGate::from_json(&vec![b' '; BYTE_CORPUS_STUDY_MAX_BYTES + 1]).is_err()
    );
    assert!(ByteCorpusServingGate::from_json(b"\xff").is_err());
}

#[test]
fn negative_regression_limits_explicitly_require_improvement() {
    let mut v = gate_value();
    v["limits"]["max_mean_ce_regression"] = json!(-0.1);
    v["limits"]["max_batch_ce_regression"] = json!(-0.01);
    let gate = ByteCorpusServingGate::from_json(v.to_string().as_bytes()).unwrap();
    let mut limits = gate.cohorts[0].limits.clone();
    limits.max_mean_ce_regression = f64::NAN;
    assert!(limits.validate().is_err());
    limits.max_mean_ce_regression = 0.;
    limits.max_batch_ce_regression = f64::INFINITY;
    assert!(limits.validate().is_err());
}

#[test]
fn named_cohorts_preserve_order_windows_limits_and_identity() {
    let mut value = cohort_gate_value();
    value["cohorts"][1]["name"] = json!("\u{65e5}\u{672c}\u{8a9e}");
    value["cohorts"][1]["limits"]["max_mean_ce_regression"] = json!(-0.25);
    let raw = value.to_string();
    let gate = ByteCorpusServingGate::from_json(raw.as_bytes()).unwrap();
    assert_eq!(gate.to_json(), raw);
    assert_eq!(gate.schema_version(), 2);
    assert_eq!(gate.batch_count(), 3);
    assert_eq!(gate.batch_shape(), [1, 2]);
    assert_eq!(gate.cohort_names(), ["prefix", "\u{65e5}\u{672c}\u{8a9e}"]);
    assert_eq!(gate.cohorts[1].batches[0].target_bytes(), &[10, 11]);
    assert_eq!(gate.cohorts[1].limits.max_mean_ce_regression, -0.25);
    value["cohorts"].as_array_mut().unwrap().reverse();
    assert_ne!(
        ByteCorpusServingGate::from_json(value.to_string().as_bytes())
            .unwrap()
            .request_sha256(),
        gate.request_sha256()
    );
}

#[test]
fn cohorts_require_explicit_limits_unique_names_and_bounded_common_shapes() {
    for name in [
        "",
        " prefix",
        "prefix ",
        "suffix",
        "bad\nname",
        "bad\0name",
        &"x".repeat(129),
    ] {
        let mut value = cohort_gate_value();
        value["cohorts"][0]["name"] = json!(name);
        assert!(
            ByteCorpusServingGate::from_json(value.to_string().as_bytes()).is_err(),
            "{name:?}"
        );
    }
    for (key, changed) in [
        ("batches", json!([])),
        ("batches", json!([[]])),
        ("batches", json!([[[0, 0], [0, 2]]])),
        ("batches", json!([[[0, 4]]])),
        ("batches", json!([[[1, 0]]])),
        ("limits", Value::Null),
        ("extra", json!(1)),
    ] {
        let mut value = cohort_gate_value();
        value["cohorts"][1][key] = changed;
        assert!(
            ByteCorpusServingGate::from_json(value.to_string().as_bytes()).is_err(),
            "{key}"
        );
    }
    for (key, changed) in [
        ("limits", gate_value()["limits"].clone()),
        ("batches", json!([[[0, 0]]])),
        ("cohorts", json!([])),
    ] {
        let mut value = cohort_gate_value();
        value[key] = changed;
        assert!(ByteCorpusServingGate::from_json(value.to_string().as_bytes()).is_err());
    }
    let mut value = cohort_gate_value();
    value["cohorts"][1]["limits"]
        .as_object_mut()
        .unwrap()
        .remove("max_mean_ce");
    assert!(ByteCorpusServingGate::from_json(value.to_string().as_bytes()).is_err());
    let mut value = cohort_gate_value();
    let template = value["cohorts"][0].clone();
    value["cohorts"] = json!((0..32)
        .map(|i| {
            let mut c = template.clone();
            c["name"] = json!(format!("cohort-{i}"));
            c["batches"] = json!(vec![vec![(0, 0)]; 4]);
            c
        })
        .collect::<Vec<_>>());
    assert_eq!(
        ByteCorpusServingGate::from_json(value.to_string().as_bytes())
            .unwrap()
            .batch_count(),
        128
    );
    value["cohorts"][31]["batches"]
        .as_array_mut()
        .unwrap()
        .push(json!([[0, 0]]));
    assert!(ByteCorpusServingGate::from_json(value.to_string().as_bytes()).is_err());
    for cohort in value["cohorts"].as_array_mut().unwrap() {
        cohort["batches"] = json!([[[0, 0]]]);
    }
    value["cohorts"].as_array_mut().unwrap().push(template);
    assert!(ByteCorpusServingGate::from_json(value.to_string().as_bytes()).is_err());
    let mut value = gate_value();
    value["cohorts"] = cohort_gate_value()["cohorts"].clone();
    assert!(ByteCorpusServingGate::from_json(value.to_string().as_bytes()).is_err());
    value["cohorts"] = Value::Null;
    assert!(ByteCorpusServingGate::from_json(value.to_string().as_bytes()).is_err());
    for key in ["batches", "limits"] {
        let mut value = cohort_gate_value();
        value[key] = Value::Null;
        assert!(ByteCorpusServingGate::from_json(value.to_string().as_bytes()).is_err());
    }
}

#[test]
fn artifact_window_is_checked_before_restore() {
    let study = study();
    let artifact = study
        .case_artifact(
            &saved(&study, 0),
            "geometry",
            ByteCorpusInferencePolicy::FixedWindowFrozenV1,
        )
        .unwrap();
    ByteCorpusServingGate::from_json(gate_value().to_string().as_bytes())
        .unwrap()
        .validate_artifact(&artifact)
        .unwrap();
    let mut wrong = gate_value();
    wrong["steps"] = json!(1);
    assert!(
        ByteCorpusServingGate::from_json(wrong.to_string().as_bytes())
            .unwrap()
            .validate_artifact(&artifact)
            .is_err()
    );
}
