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

#[test]
fn gate_preserves_exact_data_threshold_identity_without_a_device() {
    let raw = gate_value().to_string();
    let gate = ByteCorpusServingGate::from_json(raw.as_bytes()).unwrap();
    assert_eq!(gate.to_json(), raw);
    assert_eq!(gate.batch_count(), 2);
    assert_eq!(gate.batch_shape(), [1, 2]);
    assert_eq!(gate.batches[1].input_bytes(), &[9, 10]);
    assert_eq!(gate.batches[1].target_bytes(), &[10, 11]);
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
    let mut limits = gate.limits.clone();
    limits.max_mean_ce_regression = f64::NAN;
    assert!(limits.validate().is_err());
    limits.max_mean_ce_regression = 0.;
    limits.max_batch_ce_regression = f64::INFINITY;
    assert!(limits.validate().is_err());
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
