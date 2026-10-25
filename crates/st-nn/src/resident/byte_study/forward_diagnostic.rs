//! One explicitly selected forward boundary; no implicit readback or retry.
use super::*;
#[cfg(feature = "wgpu")]
use crate::resident::ByteDecoderGraphDiagnosticReadback;
use crate::resident::BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_BYTES;

/// The instrumented study outcome and an independently consumable graph copy.
/// Save `run_outcome` before reading `diagnostic`: mapping can still fail.
#[cfg(feature = "wgpu")]
pub struct ByteCorpusForwardDiagnosticObservation {
    pub run_outcome: Value,
    pub diagnostic: Option<ByteDecoderGraphDiagnosticReadback>,
}

#[cfg(any(feature = "wgpu", test))]
enum ForwardCapture<T> {
    NotReached,
    Captured(T),
    Failed(String),
}

#[cfg(any(feature = "wgpu", test))]
fn resume_digest(resume: Option<&ByteCorpusStudyCheckpoint>) -> Result<Option<String>> {
    Ok(resume
        .map(ByteCorpusStudyCheckpoint::to_json)
        .transpose()?
        .map(|json| format!("{:x}", Sha256::digest(json.as_bytes()))))
}

#[cfg(any(feature = "wgpu", test))]
impl<T> ForwardCapture<T> {
    fn observe(
        &mut self,
        name: &str,
        update: usize,
        selected: (&str, usize),
        snapshot: impl FnOnce() -> Result<T>,
    ) -> Result<()> {
        if (name, update) != selected {
            return Ok(());
        }
        if !matches!(self, Self::NotReached) {
            return Err("forward diagnostic boundary invoked more than once".into());
        }
        // A diagnostic failure does not replace the observed learning outcome.
        // Do not retry; earlier device copies may already have been submitted.
        *self = match snapshot() {
            Ok(snapshot) => Self::Captured(snapshot),
            Err(error) => Self::Failed(error.to_string()),
        };
        Ok(())
    }

    fn status(&self) -> &'static str {
        match self {
            Self::NotReached => "not_reached",
            Self::Captured(_) => "copies_submitted",
            Self::Failed(_) => "error",
        }
    }

    fn into_snapshot(self) -> Option<T> {
        match self {
            Self::Captured(snapshot) => Some(snapshot),
            Self::NotReached | Self::Failed(_) => None,
        }
    }
}

impl ByteCorpusStudy {
    /// CPU-only selection validation, usable before requesting any GPU runtime.
    /// The selected update is absolute and one-based, including after resume.
    /// Actual graph sizes and tokens are preflighted at the selected boundary,
    /// before the first diagnostic copy, not by this request-level check.
    pub fn validate_forward_diagnostic(
        &self,
        resume: Option<&ByteCorpusStudyCheckpoint>,
        case_name: &str,
        before_update: usize,
        max_bytes: usize,
    ) -> Result<()> {
        if max_bytes == 0 || max_bytes > BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_BYTES {
            return Err("forward diagnostic byte budget must be in 1..=64 MiB".into());
        }
        let start = resume.map_or(0, ByteCorpusStudyCheckpoint::completed_updates);
        if before_update <= start || before_update > self.total_updates() {
            return Err(
                "diagnostic update must be after the saved cursor and within this study".into(),
            );
        }
        if !self.request.cases.iter().any(|c| c.name == case_name) {
            return Err("diagnostic case does not belong to this study".into());
        }
        self.validate_segment(resume, self.total_updates())?;
        Ok(())
    }

    /// Execute all cases in their original order, copying one selected forward.
    /// `resume` restores all cases; None starts from the request initialization.
    /// This is an explicitly instrumented run, never ordinary qualification.
    ///
    /// The copy is submitted after forward and loss submission and before that
    /// update's backward. It includes only the model's graph diagnostic scope,
    /// not owner weights, optimizer state, targets, loss or full non-graph tapes.
    /// Copying changes scheduling and resource lifetimes. No mapping, wait,
    /// fallback or retry is added by this observer.
    ///
    /// Selection errors return before execution. Once execution starts, both
    /// learning errors and diagnostic errors are returned as separate fields in
    /// `run_outcome`. A diagnostic error does not abort learning or imply that
    /// no copies ran. Save the outcome before consuming the optional readback.
    #[cfg(feature = "wgpu")]
    pub async fn run_with_forward_diagnostic(
        &self,
        runtime: WgpuRuntime,
        resume: Option<&ByteCorpusStudyCheckpoint>,
        case_name: &str,
        before_update: usize,
        max_bytes: usize,
    ) -> Result<ByteCorpusForwardDiagnosticObservation> {
        self.validate_forward_diagnostic(resume, case_name, before_update, max_bytes)?;
        let start = resume.map_or(0, ByteCorpusStudyCheckpoint::completed_updates);
        let resume_sha256 = resume_digest(resume)?;
        let mut capture = ForwardCapture::NotReached;
        let result = self
            .execute(
                runtime,
                resume,
                self.total_updates(),
                false,
                None::<UpdateObserverFn>,
                Some(
                    |name: &str,
                     update,
                     model: &ResidentByteDecoder,
                     tape: &ResidentByteDecoderForward| {
                        capture.observe(name, update, (case_name, before_update), || {
                            Ok(model.graph_diagnostic_snapshot(tape, max_bytes)?)
                        })
                    },
                ),
            )
            .await
            .map(|(report, _)| report);
        let run_outcome = self.forward_diagnostic_outcome(
            case_name,
            (start, resume_sha256.as_deref()),
            before_update,
            max_bytes,
            result,
            &capture,
        );
        Ok(ByteCorpusForwardDiagnosticObservation {
            run_outcome,
            diagnostic: capture.into_snapshot(),
        })
    }

    #[cfg(any(feature = "wgpu", test))]
    fn forward_diagnostic_outcome<T>(
        &self,
        case_name: &str,
        origin: (usize, Option<&str>),
        before_update: usize,
        max_bytes: usize,
        result: Result<Value>,
        capture: &ForwardCapture<T>,
    ) -> Value {
        let mut outcome = json!({
            "schema": "spiraltorch.byte_forward_graph_diagnostic.v1",
            "request_sha256": self.request_sha256(),
            "case": case_name,
            "start_after_update": origin.0,
            "resume_checkpoint_json_sha256": origin.1,
            "resume_checkpoint_hash_encoding": "ByteCorpusStudyCheckpoint::to_json",
            "requested_before_update": before_update,
            "boundary": "after_forward_and_loss_submission_before_backward",
            "capture_status": capture.status(),
            "max_capture_bytes": max_bytes,
            "diagnostic_read_policy": "explicit_after_run_returns",
            "release_ready": false,
            "scope": "Instrumented graph-only observation: copies change scheduling and \
                resource lifetimes. No repair, acceptance, determinism or release claim."
        });
        if let ForwardCapture::Failed(error) = capture {
            outcome["diagnostic_error"] = json!(error);
        }
        match result {
            Ok(report) => {
                outcome["status"] = json!("completed");
                outcome["all_cases_completed"] = json!(true);
                outcome["report"] = report;
            }
            Err(error) => {
                outcome["status"] = json!("error");
                outcome["all_cases_completed"] = json!(false);
                outcome["error"] = json!(error.to_string());
            }
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resident::byte_study::tests::{fixture, saved};

    fn study() -> ByteCorpusStudy {
        ByteCorpusStudy::from_json(fixture().to_string().as_bytes()).unwrap()
    }

    #[test]
    fn forward_diagnostic_preflight_binds_case_cursor_and_budget() {
        let study = study();
        let first = saved(&study, 1);
        let max = BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_BYTES;
        study
            .validate_forward_diagnostic(None, "plain", 1, 1)
            .unwrap();
        study
            .validate_forward_diagnostic(None, "plain", 2, max)
            .unwrap();
        study
            .validate_forward_diagnostic(Some(&first), "plain", 2, max)
            .unwrap();
        for update in [0, 3, usize::MAX] {
            assert!(study
                .validate_forward_diagnostic(None, "plain", update, max)
                .is_err());
        }
        for bytes in [0, max + 1, usize::MAX] {
            assert!(study
                .validate_forward_diagnostic(None, "plain", 2, bytes)
                .is_err());
        }
        assert!(study
            .validate_forward_diagnostic(None, "missing", 2, max)
            .is_err());
        assert!(study
            .validate_forward_diagnostic(Some(&first), "plain", 1, max)
            .is_err());
        assert!(study
            .validate_forward_diagnostic(Some(&saved(&study, 2)), "plain", 2, max)
            .is_err());
    }

    #[test]
    fn forward_diagnostic_resume_validation_covers_the_whole_checkpoint() {
        let study = study();
        let max = BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_BYTES;
        let mut foreign = saved(&study, 1);
        foreign.request_sha256.push('0');
        assert!(study
            .validate_forward_diagnostic(Some(&foreign), "plain", 2, max)
            .is_err());
        let mut corrupt_other = saved(&study, 1);
        corrupt_other.cases[1].model_json = "{}".into();
        assert!(study
            .validate_forward_diagnostic(Some(&corrupt_other), "plain", 2, max)
            .is_err());
    }

    #[test]
    fn forward_diagnostic_resume_digest_binds_the_serialized_checkpoint() {
        let study = study();
        let first = saved(&study, 1);
        assert_eq!(resume_digest(None).unwrap(), None);
        assert_eq!(
            resume_digest(Some(&first)).unwrap(),
            Some(format!(
                "{:x}",
                Sha256::digest(first.to_json().unwrap().as_bytes())
            ))
        );
        let mut changed = first.clone();
        changed.cases[0].training[0].ce += 1.0;
        assert_ne!(
            resume_digest(Some(&first)).unwrap(),
            resume_digest(Some(&changed)).unwrap()
        );
    }

    #[test]
    fn forward_diagnostic_only_attempts_the_selected_boundary_once() {
        let mut capture = ForwardCapture::NotReached;
        let mut calls = 0;
        for (name, update) in [("plain", 1), ("other", 2)] {
            capture
                .observe(name, update, ("plain", 2), || {
                    calls += 1;
                    Ok(17)
                })
                .unwrap();
        }
        assert_eq!(calls, 0);
        assert_eq!(capture.status(), "not_reached");
        capture
            .observe("plain", 2, ("plain", 2), || {
                calls += 1;
                Ok(17)
            })
            .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(capture.status(), "copies_submitted");
        assert!(capture
            .observe("plain", 2, ("plain", 2), || {
                calls += 1;
                Ok(18)
            })
            .is_err());
        assert_eq!(calls, 1);
        assert_eq!(capture.into_snapshot(), Some(17));
    }

    #[test]
    fn forward_diagnostic_copy_failure_is_recorded_without_retry_or_run_abort() {
        let mut capture = ForwardCapture::<()>::NotReached;
        capture
            .observe("plain", 2, ("plain", 2), || Err("copy failed".into()))
            .unwrap();
        assert_eq!(capture.status(), "error");
        assert!(capture
            .observe("plain", 2, ("plain", 2), || panic!("must not retry"))
            .is_err());
        capture
            .observe("later-case", 2, ("plain", 2), || {
                panic!("must not capture another case")
            })
            .unwrap();
        assert!(capture.into_snapshot().is_none());
    }

    #[test]
    fn forward_diagnostic_outcome_preserves_learning_and_capture_status_separately() {
        let study = study();
        for capture in [
            ForwardCapture::NotReached,
            ForwardCapture::Captured(()),
            ForwardCapture::Failed("copy failed".into()),
        ] {
            let synthetic = json!({"synthetic": "serialization fixture, not GPU evidence"});
            let completed = study.forward_diagnostic_outcome(
                "plain",
                (1, Some("synthetic-checkpoint-digest")),
                2,
                1024,
                Ok(synthetic.clone()),
                &capture,
            );
            let failed = study.forward_diagnostic_outcome(
                "plain",
                (1, Some("synthetic-checkpoint-digest")),
                2,
                1024,
                Err("learning failed".into()),
                &capture,
            );
            for outcome in [&completed, &failed] {
                assert_eq!(outcome["request_sha256"], study.request_sha256());
                assert_eq!(outcome["start_after_update"], 1);
                assert_eq!(
                    outcome["resume_checkpoint_json_sha256"],
                    "synthetic-checkpoint-digest"
                );
                assert_eq!(outcome["requested_before_update"], 2);
                assert_eq!(outcome["max_capture_bytes"], 1024);
                assert_eq!(outcome["capture_status"], capture.status());
                assert_eq!(outcome["release_ready"], false);
                if matches!(&capture, ForwardCapture::Failed(_)) {
                    assert_eq!(outcome["diagnostic_error"], "copy failed");
                } else {
                    assert!(outcome.get("diagnostic_error").is_none());
                }
            }
            assert_eq!(completed["status"], "completed");
            assert_eq!(completed["all_cases_completed"], true);
            assert_eq!(completed["report"], synthetic);
            assert!(completed.get("error").is_none());
            assert_eq!(failed["status"], "error");
            assert_eq!(failed["all_cases_completed"], false);
            assert_eq!(failed["error"], "learning failed");
            assert!(failed.get("report").is_none());
        }
    }
}
