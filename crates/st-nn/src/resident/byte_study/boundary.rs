//! Explicit diagnostics, separate from normal learning and release acceptance.
use super::*;
#[cfg(feature = "wgpu")]
use crate::resident::ByteDecoderCheckpointHandles;

/// The original study outcome and a separately readable diagnostic boundary.
/// Save the outcome before consuming the handles: copying/mapping can fail.
/// Neither handle retention nor this record establishes update acceptance.
#[cfg(feature = "wgpu")]
pub struct ByteCorpusBoundaryObservation {
    pub run_outcome: Value,
    pub boundary: Option<ByteDecoderCheckpointHandles>,
}

impl ByteCorpusStudy {
    /// CPU-only preflight for a single update-1 diagnostic. The reference is
    /// validated against this entire request, not used to initialize the run.
    pub fn validate_retained_boundary_reference(
        &self,
        first: &ByteCorpusStudyCheckpoint,
        case_name: &str,
    ) -> Result<()> {
        if self.total_updates() < 2 || first.completed_updates() != 1 {
            return Err("requires at least two updates and an update-1 reference".into());
        }
        self.validate_segment(None, self.total_updates())?;
        self.case_checkpoint(first, case_name)?;
        Ok(())
    }

    /// Run all cases in order, retaining one first-update boundary in this
    /// process. No diagnostic GPU copy/map is submitted by this method.
    ///
    /// Validation errors return Err before execution. Once execution starts,
    /// ordinary study errors are retained in `run_outcome`, independently of
    /// boundary availability. Save that outcome before `boundary.snapshot()`.
    /// Resource lifetimes and host work change; this is not an unperturbed run.
    #[cfg(feature = "wgpu")]
    pub async fn run_with_retained_boundary(
        &self,
        runtime: WgpuRuntime,
        first: &ByteCorpusStudyCheckpoint,
        case_name: &str,
    ) -> Result<ByteCorpusBoundaryObservation> {
        self.validate_retained_boundary_reference(first, case_name)?;
        let mut boundary = None;
        let result = self
            .run_with_update_observer(runtime, |name, step, model| {
                if name == case_name && step == 1 {
                    if boundary.is_some() {
                        return Err("diagnostic boundary was already retained".into());
                    }
                    boundary = Some(model.retain_checkpoint()?);
                }
                Ok(())
            })
            .await;
        let run_outcome = self.retained_boundary_outcome(case_name, result, boundary.is_some());
        Ok(ByteCorpusBoundaryObservation {
            run_outcome,
            boundary,
        })
    }

    #[cfg(any(feature = "wgpu", test))]
    fn retained_boundary_outcome(
        &self,
        case_name: &str,
        result: Result<Value>,
        captured: bool,
    ) -> Value {
        let mut outcome = json!({
            "schema": "spiraltorch.byte_retained_boundary.v1",
            "request_sha256": self.request_sha256(),
            "case": case_name,
            "requested_capture_after_update": 1,
            "capture_mode": "retained",
            "boundary_captured": captured,
            "boundary_read_policy": "explicit_after_run_returns",
            "release_ready": false,
            "scope": "Instrumented observation: handle retention changes host work and \
                resource lifetimes. No repair, determinism or release claim."
        });
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
    fn retained_boundary_preflight_is_request_case_and_cursor_bound() {
        let study = study();
        let first = saved(&study, 1);
        study
            .validate_retained_boundary_reference(&first, "plain")
            .unwrap();
        assert!(study
            .validate_retained_boundary_reference(&first, "missing")
            .is_err());
        for cursor in [0, 2] {
            assert!(study
                .validate_retained_boundary_reference(&saved(&study, cursor), "plain")
                .is_err());
        }
        let mut foreign = first.clone();
        foreign.request_sha256.push('0');
        assert!(study
            .validate_retained_boundary_reference(&foreign, "plain")
            .is_err());
        let mut corrupt_other_case = first;
        corrupt_other_case.cases[1].model_json = "{}".into();
        assert!(study
            .validate_retained_boundary_reference(&corrupt_other_case, "plain")
            .is_err());
        let mut one_update = fixture();
        one_update["train_batches"]
            .as_array_mut()
            .unwrap()
            .truncate(1);
        let short = ByteCorpusStudy::from_json(one_update.to_string().as_bytes()).unwrap();
        assert!(short
            .validate_retained_boundary_reference(&saved(&short, 1), "plain")
            .is_err());
    }

    #[test]
    fn study_outcome_is_independent_of_boundary_availability() {
        let study = study();
        for captured in [false, true] {
            let failed =
                study.retained_boundary_outcome("plain", Err("study failed".into()), captured);
            assert_eq!(failed["status"], "error");
            assert_eq!(failed["error"], "study failed");
            assert_eq!(failed["all_cases_completed"], false);
            assert_eq!(failed["boundary_captured"], captured);
            assert_eq!(failed["release_ready"], false);
            assert!(failed.get("report").is_none());
            let report = json!({"synthetic": "outcome serialization test, not runtime evidence"});
            let completed = study.retained_boundary_outcome("plain", Ok(report.clone()), captured);
            assert_eq!(completed["report"], report);
            assert_eq!(completed["status"], "completed");
            assert_eq!(completed["all_cases_completed"], true);
            assert_eq!(completed["boundary_captured"], captured);
            assert_eq!(completed["release_ready"], false);
            assert!(completed.get("error").is_none());
        }
    }
}
