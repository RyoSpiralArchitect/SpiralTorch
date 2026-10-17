//! Development-loss selection of actual saved models, never implicit promotion.
use super::*;

/// Shared ingress bound for native, Python and WASM saved-model selection.
pub const BYTE_CORPUS_CHECKPOINT_SELECTION_MAX_CANDIDATES: usize = 32;

/// A selected immutable artifact and its bounded, weight-free decision record.
/// The artifact keeps the original model/geometry/optimizer and source binding.
pub struct ByteCorpusCheckpointSelection {
    pub artifact: ByteCorpusModelArtifact,
    pub report: Value,
}

fn extends(previous: &CaseCheckpoint, next: &CaseCheckpoint) -> bool {
    previous.training.len() <= next.training.len()
        && previous.validation.len() <= next.validation.len()
        && previous
            .training
            .iter()
            .zip(&next.training)
            .all(|(a, b)| a.revision == b.revision && a.ce.to_bits() == b.ce.to_bits())
        && previous
            .validation
            .iter()
            .zip(&next.validation)
            .all(|(a, b)| {
                a.revision == b.revision && policy::same_bits(&a.batch_losses, &b.batch_losses)
            })
}

impl ByteCorpusStudy {
    /// Choose one case's lowest development CE among 1..32 supplied snapshots.
    /// Candidates must be chronological, at evaluation boundaries, from this
    /// exact request, with unchanged all-case history prefixes. Ties keep the
    /// earliest supplied revision. An unsaved historical minimum is ineligible.
    ///
    /// Validation windows are development data once used for selection. This is
    /// neither held-out test performance, a promotion gate nor proof of training.
    /// Candidates' combined canonical JSON is bounded to 64 MiB, without a GPU.
    pub fn select_checkpoint(
        &self,
        checkpoints: &[&ByteCorpusStudyCheckpoint],
        case_name: &str,
        policy: ByteCorpusInferencePolicy,
    ) -> Result<ByteCorpusCheckpointSelection> {
        if !(1..=BYTE_CORPUS_CHECKPOINT_SELECTION_MAX_CANDIDATES).contains(&checkpoints.len()) {
            return Err("selection requires 1..32 saved checkpoints".into());
        }
        let case_index = self
            .request
            .cases
            .iter()
            .position(|case| case.name == case_name)
            .ok_or("selection case name is not in the request")?;
        let mut candidates = Vec::with_capacity(checkpoints.len());
        let mut total_bytes = 0usize;
        let mut best = 0usize;
        let mut best_ce = f64::INFINITY;
        for (index, checkpoint) in checkpoints.iter().enumerate() {
            let json = checkpoint.to_json()?;
            total_bytes = total_bytes
                .checked_add(json.len())
                .filter(|&n| n <= BYTE_CORPUS_STUDY_MAX_BYTES)
                .ok_or("selection checkpoints exceed 64 MiB combined")?;
            self.validate_segment(Some(checkpoint), checkpoint.completed_updates())?;
            if !self.evaluation_boundary(checkpoint.completed_updates()) {
                return Err("selection needs a saved evaluation-boundary model".into());
            }
            if index > 0 {
                let previous = checkpoints[index - 1];
                if previous.completed_updates >= checkpoint.completed_updates
                    || previous
                        .cases
                        .iter()
                        .zip(&checkpoint.cases)
                        .any(|(a, b)| !extends(a, b))
                {
                    return Err("selection needs increasing cursors and unchanged all-case history prefixes".into());
                }
            }
            let point = checkpoint.cases[case_index]
                .validation
                .last()
                .filter(|p| p.revision == checkpoint.completed_updates)
                .ok_or("selection score does not describe the saved model revision")?;
            let mean_ce = point
                .batch_losses
                .iter()
                .map(|&v| f64::from(v))
                .sum::<f64>()
                / point.batch_losses.len() as f64;
            if mean_ce < best_ce {
                best = index;
                best_ce = mean_ce;
            }
            candidates.push(json!({
                "completed_updates":checkpoint.completed_updates,
                "study_checkpoint_sha256":format!("{:x}", Sha256::digest(json.as_bytes())),
                "mean_ce":mean_ce, "bits_per_byte":mean_ce / std::f64::consts::LN_2,
                "target_bytes":self.request.config.batch * self.request.config.steps * point.batch_losses.len()
            }));
        }
        let artifact = self.case_artifact(checkpoints[best], case_name, policy)?;
        let report = json!({
            "schema":"spiraltorch.byte_corpus.checkpoint_selection.v1",
            "request_sha256":self.request_sha256, "case_name":case_name,
            "rule":"minimum_validation_ce_earliest_saved.v1",
            "selection_data":"request_validation_windows",
            "candidates":candidates,
            "selected":{"candidate_index":best,
                "completed_updates":checkpoints[best].completed_updates,
                "artifact_id":artifact.artifact_id()},
            "scope":"Saved-model development selection only; not an unsaved minimum, training attestation, test result or serving promotion"
        });
        Ok(ByteCorpusCheckpointSelection { artifact, report })
    }
}

#[cfg(test)]
mod tests;
