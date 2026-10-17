use super::*;
use crate::resident::byte_study::tests::{fixture, saved};

const POLICY: ByteCorpusInferencePolicy = ByteCorpusInferencePolicy::FixedWindowFrozenV1;

fn study() -> ByteCorpusStudy {
    let mut request = fixture();
    request["train_batches"] = json!([[[0, 0]], [[0, 2]], [[0, 0]], [[0, 2]]]);
    ByteCorpusStudy::from_json(&serde_json::to_vec(&request).unwrap()).unwrap()
}

fn snapshots(study: &ByteCorpusStudy) -> Vec<ByteCorpusStudyCheckpoint> {
    [0, 2, 4]
        .into_iter()
        .map(|cursor| {
            let mut checkpoint = saved(study, cursor);
            for point in &mut checkpoint.cases[0].validation {
                point.batch_losses = vec![match point.revision {
                    0 => 3.,
                    2 => 1.,
                    _ => 2.,
                }];
            }
            checkpoint
        })
        .collect()
}

#[test]
fn chooses_actual_best_saved_artifact_not_final_cursor() {
    let study = study();
    let candidates = snapshots(&study);
    let refs: Vec<_> = candidates.iter().collect();
    let selected = study.select_checkpoint(&refs, "plain", POLICY).unwrap();
    assert_eq!(selected.report["selected"]["candidate_index"], 1);
    assert_eq!(selected.report["selected"]["completed_updates"], 2);
    assert_eq!(selected.artifact.checkpoint().attempted_revision(), 2);
    study
        .validate_artifact(&candidates[1], &selected.artifact)
        .unwrap();
    assert!(study
        .validate_artifact(&candidates[2], &selected.artifact)
        .is_err());
    assert_eq!(
        selected.artifact.to_json(),
        study
            .case_artifact(&candidates[1], "plain", POLICY)
            .unwrap()
            .to_json()
    );
    assert_eq!(selected.report["candidates"].as_array().unwrap().len(), 3);
    assert!(!selected.report.to_string().contains("model_json"));
    assert!(!selected.report.to_string().contains("parameters"));
}

#[test]
fn ties_keep_earliest_and_unsaved_minima_are_not_selected() {
    let study = study();
    let candidates = snapshots(&study);
    let refs: Vec<_> = candidates.iter().collect();
    let tied = study.select_checkpoint(&refs, "geometry", POLICY).unwrap();
    assert_eq!(tied.report["selected"]["completed_updates"], 0);
    let sparse = study
        .select_checkpoint(&[&candidates[0], &candidates[2]], "plain", POLICY)
        .unwrap();
    assert_eq!(sparse.report["selected"]["completed_updates"], 4);
    let single = study
        .select_checkpoint(&[&candidates[1]], "plain", POLICY)
        .unwrap();
    assert_eq!(single.report["selected"]["completed_updates"], 2);
}

#[test]
fn rejects_unbound_unscheduled_duplicate_reversed_and_excess_candidates() {
    let study = study();
    let candidates = snapshots(&study);
    let middle = saved(&study, 1);
    for refs in [
        vec![],
        vec![&middle],
        vec![&candidates[0]; 33],
        vec![&candidates[1], &candidates[0]],
        vec![&candidates[1]; 2],
    ] {
        assert!(study.select_checkpoint(&refs, "plain", POLICY).is_err());
    }
    assert!(study
        .select_checkpoint(&[&candidates[0]], " Plain", POLICY)
        .is_err());
    let mut wrong = candidates[1].clone();
    wrong.request_sha256 = "0".repeat(64);
    assert!(study.select_checkpoint(&[&wrong], "plain", POLICY).is_err());
    let mut corrupt = candidates[1].clone();
    corrupt.cases[1].model_json = "{}".into();
    assert!(study
        .select_checkpoint(&[&corrupt], "plain", POLICY)
        .is_err());
}

#[test]
fn rejects_forked_all_case_histories_even_for_unselected_cases() {
    let study = study();
    for evaluation in [false, true] {
        let mut candidates = snapshots(&study);
        if evaluation {
            candidates[2].cases[1].validation[0].batch_losses[0] = 0.2;
        } else {
            candidates[2].cases[1].training[0].ce = 0.2;
        }
        // Every snapshot alone is valid; joining divergent histories is not.
        study.checked_models(&candidates[2]).unwrap();
        assert!(study
            .select_checkpoint(&[&candidates[1], &candidates[2]], "plain", POLICY)
            .is_err());
    }
    let mut candidates = snapshots(&study);
    candidates[1].cases[1].training[0].ce = 0.0;
    candidates[2].cases[1].training[0].ce = -0.0;
    assert!(study
        .select_checkpoint(&[&candidates[1], &candidates[2]], "plain", POLICY)
        .is_err());
}
