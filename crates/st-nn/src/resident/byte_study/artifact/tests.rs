use super::super::tests::{fixture, saved};
use super::*;

fn setup() -> (ByteCorpusStudy, ByteCorpusStudyCheckpoint) {
    let study = ByteCorpusStudy::from_json(&serde_json::to_vec(&fixture()).unwrap()).unwrap();
    let checkpoint = saved(&study, 1);
    (study, checkpoint)
}

fn artifact(
    study: &ByteCorpusStudy,
    checkpoint: &ByteCorpusStudyCheckpoint,
) -> ByteCorpusModelArtifact {
    study
        .case_artifact(
            checkpoint,
            "geometry",
            ByteCorpusInferencePolicy::FixedWindowFrozenV1,
        )
        .unwrap()
}

fn self_pinned(record: &Value) -> Result<ByteCorpusModelArtifact> {
    let bytes = serde_json::to_vec(record).unwrap();
    ByteCorpusModelArtifact::from_json(&bytes, &format!("sha256:{}", sha256(&bytes)))
}

#[test]
fn artifact_binds_all_companions_policy_and_exact_model_bytes() {
    let (study, checkpoint) = setup();
    let before = checkpoint.to_json().unwrap();
    let artifact = artifact(&study, &checkpoint);
    assert_eq!(artifact.policy().as_str(), "fixed_window_frozen.v1");
    assert_eq!(artifact.request_sha256(), study.request_sha256());
    assert_eq!(
        artifact.study_checkpoint_sha256(),
        sha256(before.as_bytes())
    );
    assert_eq!(artifact.checkpoint().attempted_revision(), 1);
    assert_eq!(
        artifact.checkpoint().to_json().unwrap(),
        study
            .case_checkpoint(&checkpoint, "geometry")
            .unwrap()
            .to_json()
            .unwrap()
    );
    let restored =
        ByteCorpusModelArtifact::from_json(artifact.to_json().as_bytes(), artifact.artifact_id())
            .unwrap();
    assert_eq!(restored.to_json(), artifact.to_json());
    assert_eq!(restored.artifact_id(), artifact.artifact_id());
    study.validate_artifact(&checkpoint, &restored).unwrap();
    assert_eq!(checkpoint.to_json().unwrap(), before);
}

#[test]
fn identity_is_exact_transport_not_decoded_json_equivalence() {
    let (study, checkpoint) = setup();
    let artifact = artifact(&study, &checkpoint);
    let edited = format!("{}\n", artifact.to_json());
    assert!(ByteCorpusModelArtifact::from_json(edited.as_bytes(), artifact.artifact_id()).is_err());
    for id in ["", "sha256:abc", "not-an-identity"] {
        assert!(ByteCorpusModelArtifact::from_json(artifact.to_json().as_bytes(), id).is_err());
    }
    // A separately pinned transport spelling is preserved, not silently rewritten.
    let new_id = format!("sha256:{}", sha256(edited.as_bytes()));
    let restored = ByteCorpusModelArtifact::from_json(edited.as_bytes(), &new_id).unwrap();
    assert_eq!(restored.to_json(), edited);
    study.validate_artifact(&checkpoint, &restored).unwrap();
}

#[test]
fn invalid_envelopes_reject_even_with_recomputed_hashes() {
    let (study, checkpoint) = setup();
    let artifact = artifact(&study, &checkpoint);
    let record: Value = serde_json::from_str(artifact.to_json()).unwrap();
    for (key, value) in [
        ("schema", json!("unknown")),
        ("policy", json!("mutable")),
        ("policy", Value::Null),
        ("request_sha256", json!("A".repeat(64))),
        ("study_checkpoint_sha256", json!("bad")),
        ("case_name", Value::Null),
        ("completed_updates", json!(0)),
        ("total_updates", json!(0)),
        ("total_updates", json!(1025)),
        ("model_json", json!("{}")),
        ("unrecognized", json!(true)),
    ] {
        let mut invalid = record.clone();
        invalid[key] = value;
        assert!(self_pinned(&invalid).is_err(), "accepted {key}");
    }
    for bytes in [b"{}".as_slice(), b"null", b"\xff"] {
        assert!(
            ByteCorpusModelArtifact::from_json(bytes, &format!("sha256:{}", sha256(bytes)))
                .is_err()
        );
    }
    assert!(ByteCorpusModelArtifact::from_json(
        &vec![b' '; BYTE_CORPUS_STUDY_MAX_BYTES + 1],
        "invalid"
    )
    .is_err());
}

#[test]
fn digest_integrity_alone_is_not_source_attestation() {
    let (study, checkpoint) = setup();
    let artifact = artifact(&study, &checkpoint);
    let mut record: Value = serde_json::from_str(artifact.to_json()).unwrap();
    record["request_sha256"] = json!("0".repeat(64));
    let forged = self_pinned(&record).unwrap();
    assert_ne!(forged.artifact_id(), artifact.artifact_id());
    assert!(study.validate_artifact(&checkpoint, &forged).is_err());
}

#[test]
fn unselected_case_and_exact_request_remain_bound() {
    let (study, checkpoint) = setup();
    let artifact = artifact(&study, &checkpoint);
    let mut changed = checkpoint.clone();
    changed.cases[0].training[0].ce = 0.2;
    study.checked_models(&changed).unwrap();
    assert!(study.validate_artifact(&changed, &artifact).is_err());
    changed.cases[0].training.clear();
    assert!(study
        .case_artifact(&changed, "geometry", artifact.policy())
        .is_err());
    let other = ByteCorpusStudy::from_json(format!("{} ", fixture()).as_bytes()).unwrap();
    assert!(other.validate_artifact(&checkpoint, &artifact).is_err());
    assert!(study
        .case_artifact(&checkpoint, "unknown", artifact.policy())
        .is_err());
}

#[test]
fn every_cursor_remains_explicit_without_promotion() {
    let (study, _) = setup();
    for cursor in 0..=study.total_updates() {
        let checkpoint = saved(&study, cursor);
        let artifact = artifact(&study, &checkpoint);
        let loaded = ByteCorpusModelArtifact::from_json(
            artifact.to_json().as_bytes(),
            artifact.artifact_id(),
        )
        .unwrap();
        study.validate_artifact(&checkpoint, &loaded).unwrap();
        assert_eq!(loaded.checkpoint().attempted_revision(), cursor as u64);
    }
}

#[test]
fn exact_empty_case_name_round_trips_like_other_study_names() {
    let mut request = fixture();
    request["cases"][1]["name"] = json!("");
    let study = ByteCorpusStudy::from_json(&serde_json::to_vec(&request).unwrap()).unwrap();
    let checkpoint = saved(&study, 1);
    let artifact = study
        .case_artifact(
            &checkpoint,
            "",
            ByteCorpusInferencePolicy::FixedWindowFrozenV1,
        )
        .unwrap();
    let loaded =
        ByteCorpusModelArtifact::from_json(artifact.to_json().as_bytes(), artifact.artifact_id())
            .unwrap();
    assert_eq!(loaded.case_name(), "");
    study.validate_artifact(&checkpoint, &loaded).unwrap();
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
fn inference_owner_retains_trained_geometry_without_mutable_parameter_escape() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte_artifact.test").unwrap();
    let (study, _) = setup();
    let trained =
        pollster::block_on(study.advance(runtime.clone(), None, study.total_updates())).unwrap();
    let artifact = artifact(&study, &trained.checkpoint);
    let mut active = artifact.restore_wgpu(runtime.clone()).unwrap();
    let mut separate = artifact.checkpoint().restore_wgpu(runtime).unwrap();
    let input = ByteLmBatch::from_windows(&[&[7, 8, 9]]).unwrap();
    let batch = active.prepare_batch(&input).unwrap();
    let retained = active.forward(&batch).unwrap();
    assert_eq!(retained.artifact_id(), artifact.artifact_id());
    let before = retained
        .forward()
        .prediction()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    let forward = separate.forward(&batch).unwrap();
    assert_eq!(
        before,
        forward.prediction().snapshot().unwrap().read().unwrap()
    );
    assert!(forward.geometry_pair_bias(0).is_some());
    let loss = forward
        .next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.).unwrap())
        .unwrap();
    let vjp = separate
        .backward(&forward, loss.prediction_gradient())
        .unwrap();
    separate
        .sgd(&vjp, 0.2)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    assert_ne!(
        separate
            .checkpoint_snapshot()
            .unwrap()
            .read()
            .unwrap()
            .to_json()
            .unwrap(),
        artifact.checkpoint().to_json().unwrap()
    );
    let corrupted = format!("{} ", artifact.to_json());
    assert!(
        ByteCorpusModelArtifact::from_json(corrupted.as_bytes(), artifact.artifact_id()).is_err()
    );
    assert_eq!(
        before,
        active
            .forward(&batch)
            .unwrap()
            .forward()
            .prediction()
            .snapshot()
            .unwrap()
            .read()
            .unwrap()
    );
    assert_eq!(active.artifact_id(), artifact.artifact_id());
    assert_eq!(active.attempted_revision(), study.total_updates() as u64);
    let wrong_shape = ByteLmBatch::from_windows(&[&[1, 2, 3, 4]]).unwrap();
    assert!(active.prepare_batch(&wrong_shape).is_err());
    drop(active);
    assert_eq!(retained.artifact_id(), artifact.artifact_id());
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
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
fn immutable_inference_omits_gpu_moments_but_artifact_keeps_resume_history() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte_artifact.adamw").unwrap();
    let mut request = fixture();
    request["design_schema"] = request["schema"].clone();
    request["schema"] = json!("spiraltorch.byte_corpus.request.v9");
    request["optimizer"] = json!({"kind":"adamw_zero_rate_freeze.v1", "beta1":0.9,
        "beta2":0.999, "epsilon":1e-8, "weight_decay":0.01, "geometry_weight_decay":0.0});
    let study = ByteCorpusStudy::from_json(request.to_string().as_bytes()).unwrap();
    let trained = pollster::block_on(study.advance(runtime.clone(), None, 1)).unwrap();
    let artifact = artifact(&study, &trained.checkpoint);
    let before = artifact.checkpoint().to_json().unwrap();
    assert!(artifact.checkpoint().adamw_state().is_some());
    let mut active = artifact.restore_wgpu(runtime.clone()).unwrap();
    assert!(active.model.adamw_config().is_none());
    let mut training = artifact.checkpoint().restore_wgpu(runtime).unwrap();
    assert!(training.adamw_config().is_some());
    assert_eq!(
        active.attempted_revision(),
        training.parameter_snapshot().revision()
    );
    let batch = active
        .prepare_batch(&ByteLmBatch::from_windows(&[&[7, 8, 9]]).unwrap())
        .unwrap();
    let predicted = active.forward(&batch).unwrap();
    let reference = training.forward(&batch).unwrap();
    assert_eq!(
        predicted
            .forward()
            .prediction()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        reference.prediction().snapshot().unwrap().read().unwrap()
    );
    assert_eq!(before, artifact.checkpoint().to_json().unwrap());
}
