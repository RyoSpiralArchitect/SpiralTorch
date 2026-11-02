//! One explicitly authorized selected-case experiment, not ordinary qualification.
use super::*;
#[cfg(feature = "wgpu")]
use crate::resident::{ByteDecoderCheckpointHandles, ByteDecoderGraphDiagnosticReadback};

const CASE: &str = "mixture-11-3-train-query-frozen";
#[cfg(feature = "wgpu")]
const MAX_GRAPH_BYTES: usize = 64 * 1024 * 1024;

#[cfg(any(feature = "wgpu", test))]
pub(super) fn selected(selection: Option<&str>, name: &str) -> bool {
    selection.is_none_or(|selected| selected == name)
}

#[cfg(feature = "wgpu")]
pub struct ByteCorpusCase24Observation {
    pub outcome: Value,
    pub first: Option<ByteDecoderCheckpointHandles>,
    pub final_checkpoint: Option<ByteDecoderCheckpointHandles>,
    pub graph: Option<ByteDecoderGraphDiagnosticReadback>,
}

impl ByteCorpusStudy {
    /// Validate the complete original request and checkpoint without GPU work.
    /// Selection never changes the request, initialization, rates or saved state.
    pub fn validate_case24_diagnostic(&self, first: &ByteCorpusStudyCheckpoint) -> Result<()> {
        self.validate_retained_boundary_reference(first, CASE)?;
        if self.request.cases.len() != 25
            || self.request.cases[24].name != CASE
            || self.total_updates() != 2
            || self.request.checkpoint_every != 2
        {
            return Err("case24 diagnostic requires original 25-case, two-update request".into());
        }
        Ok(())
    }

    /// Execute only case24, either its two continuous updates or update2 from
    /// the retained original split checkpoint. No other model is compiled or
    /// executed; calibrated values are preserved, not remeasured or refitted.
    /// The caller saves outcome before consuming any captures. No diagnostic
    /// readback is performed here. Retention/copies perturb execution timing.
    #[cfg(feature = "wgpu")]
    pub async fn run_case24_diagnostic(
        &self,
        runtime: WgpuRuntime,
        first: &ByteCorpusStudyCheckpoint,
        resumed: bool,
    ) -> Result<ByteCorpusCase24Observation> {
        self.validate_case24_diagnostic(first)?;
        let mut boundary = None;
        let mut final_checkpoint = None;
        let mut graph = None;
        let mut capture_errors = Vec::new();
        let result = self
            .execute(
                runtime,
                Some(CASE),
                resumed.then_some(first),
                2,
                false,
                Some(|name: &str, step, model: &ResidentByteDecoder| {
                    if name != CASE {
                        return Err("unexpected diagnostic case".into());
                    }
                    if step == 1 {
                        boundary = Some(model.retain_checkpoint()?);
                    }
                    if step == 2 {
                        final_checkpoint = Some(model.retain_checkpoint()?);
                    }
                    Ok(())
                }),
                Some(
                    |name: &str,
                     step,
                     model: &ResidentByteDecoder,
                     tape: &ResidentByteDecoderForward| {
                        if name != CASE {
                            return Err("unexpected diagnostic case".into());
                        }
                        if step == 2 {
                            match model.graph_diagnostic_snapshot(tape, MAX_GRAPH_BYTES) {
                                Ok(value) => graph = Some(value),
                                Err(error) => capture_errors.push(error.to_string()),
                            }
                        }
                        Ok(())
                    },
                ),
            )
            .await;
        let mut outcome = json!({
            "schema":"spiraltorch.case24_diagnostic.v1",
            "case":CASE,"request_sha256":self.request_sha256(),
            "trajectory":if resumed {"original_split_resume"} else {"continuous"},
            "original_first_checkpoint_sha256":format!("{:x}",Sha256::digest(first.to_json()?.as_bytes())),
            "start_after_update":if resumed {1} else {0},
            "stop_after_update":2,"executed_case_count":1,
            "calibration_gpu_verification":"not_run_selected_case_only",
            "first_boundary_retained":boundary.is_some(),
            "final_boundary_retained":final_checkpoint.is_some(),
            "forward_graph_copied":graph.is_some(),"capture_errors":capture_errors,
            "release_ready":false,
            "scope":"Selected-case instrumentation changes resource lifetimes and scheduling; not ordinary full-study replay or determinism qualification. No retries."
        });
        match result {
            Ok((report, _)) => {
                outcome["status"] = json!("completed");
                outcome["selected_report"] = report;
            }
            Err(error) => {
                outcome["status"] = json!("error");
                outcome["error"] = json!(error.to_string());
            }
        }
        Ok(ByteCorpusCase24Observation {
            outcome,
            first: boundary,
            final_checkpoint,
            graph,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_is_exact_and_default_path_is_unchanged() {
        assert!(selected(None, "plain"));
        assert!(selected(Some(CASE), CASE));
        assert!(!selected(Some(CASE), "mixture-11-3-train-query-train"));
        assert!(!selected(Some(CASE), ""));
    }
    #[test]
    fn unrelated_study_is_rejected_before_gpu() {
        let study =
            ByteCorpusStudy::from_json(super::super::tests::fixture().to_string().as_bytes())
                .unwrap();
        let saved = super::super::tests::saved(&study, 1);
        assert!(study.validate_case24_diagnostic(&saved).is_err());
    }
}
