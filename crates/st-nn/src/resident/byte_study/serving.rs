//! Fixed evaluation policy and process-local, measured candidate activation.
use super::*;

const GATE_SCHEMA: &str = "spiraltorch.byte_corpus.serving_gate.v1";

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Limits {
    max_mean_ce: f64,
    max_mean_ce_regression: f64,
    max_batch_ce_regression: f64,
    max_geometry_centered_rms: f64,
    max_router_mass_error: f64,
    min_router_normalized_entropy: f64,
}

impl Limits {
    fn validate(&self) -> Result<()> {
        if [self.max_mean_ce, self.max_geometry_centered_rms]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.)
            || !self.max_mean_ce_regression.is_finite()
            || !self.max_batch_ce_regression.is_finite()
            || !self.max_router_mass_error.is_finite()
            || !(0. ..1.).contains(&self.max_router_mass_error)
            || !self.min_router_normalized_entropy.is_finite()
            || !(0. ..=1.).contains(&self.min_router_normalized_entropy)
        {
            return Err("invalid explicit serving limits".into());
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GateRequest {
    schema: String,
    documents: Vec<Vec<u8>>,
    batches: Vec<Vec<(usize, usize)>>,
    steps: usize,
    limits: Limits,
}

/// Exact-byte-pinned evaluation windows and limits, fixed for one serving slot.
/// The caller owns held-outness, rights and split selection. Windows may overlap;
/// neither the byte hash nor a gate result proves statistical independence.
#[derive(Clone)]
pub struct ByteCorpusServingGate {
    json: String,
    request_sha256: String,
    batches: Vec<ByteLmBatch>,
    limits: Limits,
}

impl ByteCorpusServingGate {
    /// CPU preflight; no default thresholds, model loading or device acquisition.
    pub fn from_json(input: &[u8]) -> Result<Self> {
        if input.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
            return Err("serving gate exceeds 64 MiB".into());
        }
        let request: GateRequest = serde_json::from_slice(input)?;
        request.limits.validate()?;
        if request.schema != GATE_SCHEMA
            || !(1..=512).contains(&request.steps)
            || !(1..=128).contains(&request.batches.len())
            || !(1..=1024).contains(&request.documents.len())
        {
            return Err("invalid bounded serving evaluation".into());
        }
        let count = request.batches[0].len();
        if !(1..=32).contains(&count) || request.batches.iter().any(|b| b.len() != count) {
            return Err("serving batches need the same nonzero row count".into());
        }
        let docs: Vec<_> = request.documents.iter().map(Vec::as_slice).collect();
        let batches = request
            .batches
            .iter()
            .map(|b| ByteLmBatch::from_documents(&docs, b, request.steps))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            json: std::str::from_utf8(input)?.to_owned(),
            request_sha256: format!("{:x}", Sha256::digest(input)),
            batches,
            limits: request.limits,
        })
    }

    pub fn to_json(&self) -> &str {
        &self.json
    }
    pub fn request_sha256(&self) -> &str {
        &self.request_sha256
    }
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }
    pub fn batch_shape(&self) -> [usize; 2] {
        self.batches[0].shape()
    }

    /// Exact model window and operating policy, not corpus provenance validation.
    pub fn validate_artifact(&self, artifact: &ByteCorpusModelArtifact) -> Result<()> {
        let shape = artifact.checkpoint().plan().input_layout().shape();
        if shape[..2] != self.batch_shape()
            || artifact.policy() != ByteCorpusInferencePolicy::FixedWindowFrozenV1
        {
            return Err("artifact differs from the serving gate's fixed window/policy".into());
        }
        Ok(())
    }
}

#[cfg(feature = "wgpu")]
mod gpu;
#[cfg(feature = "wgpu")]
pub use gpu::{ResidentByteCandidate, ResidentByteServing};

#[cfg(test)]
mod tests;
