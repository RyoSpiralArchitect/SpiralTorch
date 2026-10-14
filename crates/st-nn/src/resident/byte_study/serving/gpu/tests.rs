use super::super::super::tests::saved;
use super::super::tests::{gate_value, study};
use super::*;

fn gate(value: &Value) -> ByteCorpusServingGate {
    ByteCorpusServingGate::from_json(value.to_string().as_bytes()).unwrap()
}

fn artifact(
    study: &ByteCorpusStudy,
    cp: &ByteCorpusStudyCheckpoint,
    name: &str,
) -> ByteCorpusModelArtifact {
    study
        .case_artifact(cp, name, ByteCorpusInferencePolicy::FixedWindowFrozenV1)
        .unwrap()
}

#[test]
fn observed_router_health_handles_uniform_one_hot_and_invalid_masses() {
    let uniform = router_health(2, &[0.5, 0.5, 0.5, 0.5]).unwrap();
    assert_eq!(uniform.rows, 2);
    assert_eq!(uniform.max_mass_error, 0.);
    assert_eq!(uniform.min_normalized_entropy, 1.);
    assert_eq!(
        router_health(2, &[1., 0.]).unwrap().min_normalized_entropy,
        0.
    );
    let unnormalized = router_health(2, &[0.5, 0.25]).unwrap();
    assert!((unnormalized.max_mass_error - 0.25).abs() < 1e-12);
    assert!((unnormalized.min_normalized_entropy - 0.9182958340544896).abs() < 1e-12);
    for values in [
        vec![],
        vec![0.1],
        vec![f32::NAN, 0.],
        vec![1.1, -0.1],
        vec![0., 0.],
    ] {
        assert!(router_health(2, &values).is_err());
    }
    assert!(router_health(1, &[1.]).is_err());
}

#[test]
fn mean_and_worst_window_regression_are_independent() {
    let observation = |batch_ce: Vec<f64>| Observation {
        artifact_id: "test".into(),
        revision: 0,
        mean_ce: batch_ce.iter().sum::<f64>() / batch_ce.len() as f64,
        batch_ce,
        target_bytes: 4,
        logits_f32le_sha256: vec![],
        geometry_centered_rms: vec![],
        router: vec![],
    };
    let reference = observation(vec![2., 2.]);
    let mut limits = gate(&gate_value()).limits;
    limits.max_mean_ce_regression = 0.5;
    limits.max_batch_ce_regression = 2.;
    assert_eq!(
        observation(vec![3., 3.]).assess(&reference, &limits),
        (vec!["mean_ce_regression"], 1., 1.)
    );
    limits.max_mean_ce_regression = 0.;
    limits.max_batch_ce_regression = 0.25;
    assert_eq!(
        observation(vec![1., 2.5]).assess(&reference, &limits),
        (vec!["batch_ce_regression"], -0.25, 0.5)
    );
    assert_eq!(
        observation(vec![1., 2.25]).assess(&reference, &limits),
        (vec![], -0.375, 0.25)
    );
    limits.max_mean_ce_regression = -0.5;
    limits.max_batch_ce_regression = -0.5;
    assert_eq!(
        observation(vec![2., 2.]).assess(&reference, &limits).0,
        vec!["mean_ce_regression", "batch_ce_regression"]
    );

    let mut candidate = observation(vec![2., 2.]);
    candidate
        .router
        .push(router_health(2, &[0.5, 0.25]).unwrap());
    limits.max_router_mass_error = 0.3;
    limits.min_router_normalized_entropy = 0.95;
    assert_eq!(candidate.failures(&limits), vec!["router_entropy"]);
}

#[test]
fn every_absolute_gate_has_an_independent_failure_reason() {
    let limits = gate(&gate_value()).limits;
    let mut value = Observation {
        artifact_id: "test".into(),
        revision: 0,
        batch_ce: vec![1.],
        mean_ce: 1.,
        target_bytes: 2,
        logits_f32le_sha256: vec![],
        geometry_centered_rms: vec![vec![0.1]],
        router: vec![router_health(2, &[0.5, 0.5]).unwrap()],
    };
    assert!(value.failures(&limits).is_empty());
    value.mean_ce = 11.;
    value.geometry_centered_rms[0][0] = 101.;
    value.router[0].max_mass_error = 0.1;
    let mut limits = limits;
    limits.min_router_normalized_entropy = 0.5;
    value.router[0].min_normalized_entropy = 0.;
    assert_eq!(
        value.failures(&limits),
        vec![
            "mean_ce",
            "geometry_centered_rms",
            "router_mass_error",
            "router_entropy"
        ]
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn live_slot_switches_trained_models_rejects_stale_permits_and_rolls_back_exactly() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.serving.test").unwrap();
    let study = study();
    let initial = artifact(&study, &saved(&study, 0), "geometry");
    let trained =
        pollster::block_on(study.advance(runtime.clone(), None, study.total_updates())).unwrap();
    let candidate = artifact(&study, &trained.checkpoint, "geometry");
    let mut serving = pollster::block_on(ResidentByteServing::bootstrap(
        runtime.clone(),
        initial.clone(),
        gate(&gate_value()),
    ))
    .unwrap();
    let batch = serving
        .prepare_batch(&ByteLmBatch::from_windows(&[&[7, 8, 9]]).unwrap())
        .unwrap();
    let retained = serving.forward(&batch).unwrap();
    let before = retained
        .forward()
        .prediction()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    let prepared = pollster::block_on(serving.prepare_candidate(candidate.clone())).unwrap();
    assert!(prepared.accepted());
    let report: Value = serde_json::from_str(&prepared.to_json().unwrap()).unwrap();
    assert_eq!(report["incumbent"]["artifact_id"], initial.artifact_id());
    assert_eq!(report["candidate"]["artifact_id"], candidate.artifact_id());
    assert_eq!(report["candidate"]["target_bytes"], 4);
    assert!(
        report["candidate"]["geometry_centered_rms"][0][0]
            .as_f64()
            .unwrap()
            > 0.
    );
    let stale = pollster::block_on(serving.prepare_candidate(candidate.clone())).unwrap();
    let mut foreign = pollster::block_on(ResidentByteServing::bootstrap(
        runtime.clone(),
        initial.clone(),
        gate(&gate_value()),
    ))
    .unwrap();
    assert!(foreign.activate(stale).is_err());
    assert_eq!(foreign.generation(), 0);
    let stale = pollster::block_on(serving.prepare_candidate(candidate.clone())).unwrap();
    let event = serving.activate(prepared).unwrap();
    assert_eq!(event["to"], candidate.artifact_id());
    assert_eq!(serving.generation(), 1);
    assert_eq!(serving.previous_artifact_id(), Some(initial.artifact_id()));
    assert!(serving.activate(stale).is_err());
    let after = serving.forward(&batch).unwrap();
    assert_eq!(after.artifact_id(), candidate.artifact_id());
    assert_ne!(
        after
            .forward()
            .prediction()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        before
    );
    let stale_after_rollback =
        pollster::block_on(serving.prepare_candidate(initial.clone())).unwrap();
    assert!(serving.rollback(0).is_err());
    serving.rollback(1).unwrap();
    assert_eq!(serving.generation(), 2);
    assert_eq!(serving.artifact_id(), initial.artifact_id());
    assert_eq!(serving.previous_artifact_id(), None);
    assert!(serving.activate(stale_after_rollback).is_err());
    assert!(serving.rollback(2).is_err());
    assert_eq!(
        serving
            .forward(&batch)
            .unwrap()
            .forward()
            .prediction()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        before
    );
    assert_eq!(retained.artifact_id(), initial.artifact_id());
    assert_eq!(
        retained
            .forward()
            .prediction()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        before
    );
    drop(serving);
    assert_eq!(after.artifact_id(), candidate.artifact_id());

    // Deliberately impossible improvement is a deterministic policy rejection,
    // not evidence about useful production thresholds or language quality.
    let mut strict = gate_value();
    strict["limits"]["max_mean_ce_regression"] = json!(-100.);
    strict["limits"]["max_batch_ce_regression"] = json!(-100.);
    let mut serving = pollster::block_on(ResidentByteServing::bootstrap(
        runtime.clone(),
        initial.clone(),
        gate(&strict),
    ))
    .unwrap();
    let rejected = pollster::block_on(serving.prepare_candidate(candidate.clone())).unwrap();
    assert!(!rejected.accepted());
    let report: Value = serde_json::from_str(&rejected.to_json().unwrap()).unwrap();
    assert_eq!(
        report["failed_checks"],
        json!(["mean_ce_regression", "batch_ce_regression"])
    );
    assert!(serving.activate(rejected).is_err());
    assert_eq!(serving.artifact_id(), initial.artifact_id());
    assert_eq!(serving.generation(), 0);
    assert!(serving.previous_artifact_id().is_none());
    assert!(pollster::block_on(serving.prepare_candidate(initial.clone())).is_err());
    let mut strict = gate_value();
    strict["limits"]["max_mean_ce"] = json!(0.);
    assert!(pollster::block_on(ResidentByteServing::bootstrap(
        runtime.clone(),
        initial.clone(),
        gate(&strict)
    ))
    .is_err());
    let mut strict = gate_value();
    strict["limits"]["max_geometry_centered_rms"] = json!(0.);
    assert!(pollster::block_on(ResidentByteServing::bootstrap(
        runtime,
        initial,
        gate(&strict)
    ))
    .is_err());
}
