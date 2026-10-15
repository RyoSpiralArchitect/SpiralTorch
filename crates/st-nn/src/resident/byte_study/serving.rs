//! Fixed evaluation policy and process-local, measured candidate activation.
use super::*;

#[cfg(test)]
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
    #[serde(default, deserialize_with = "present")]
    batches: Option<Vec<Vec<(usize, usize)>>>,
    steps: usize,
    #[serde(default, deserialize_with = "present")]
    limits: Option<Limits>,
    #[serde(default, deserialize_with = "present")]
    cohorts: Option<Vec<CohortRequest>>,
}

// Missing is distinct from null. Decode bytes directly: an internally tagged
// enum would first buffer every corpus byte as generic Serde Content.
fn present<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CohortRequest {
    name: String,
    batches: Vec<Vec<(usize, usize)>>,
    limits: Limits,
}

#[derive(Clone)]
struct Cohort {
    name: String,
    batches: Vec<ByteLmBatch>,
    limits: Limits,
}

/// Exact-byte-pinned evaluation windows and limits, fixed for one serving slot.
/// The caller owns held-outness, rights and split selection. Windows may overlap;
/// neither the byte hash nor a gate result proves statistical independence.
#[derive(Clone)]
pub struct ByteCorpusServingGate {
    json: String,
    request_sha256: String,
    version: u8,
    cohorts: Vec<Cohort>,
}

impl ByteCorpusServingGate {
    /// CPU preflight; no default thresholds, model loading or device acquisition.
    pub fn from_json(input: &[u8]) -> Result<Self> {
        if input.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
            return Err("serving gate exceeds 64 MiB".into());
        }
        let GateRequest {
            schema,
            documents,
            steps,
            batches,
            limits,
            cohorts,
        } = serde_json::from_slice(input)?;
        let (version, requests) = match (schema.as_str(), batches, limits, cohorts) {
            ("spiraltorch.byte_corpus.serving_gate.v1", Some(batches), Some(limits), None) => (
                1,
                vec![CohortRequest {
                    name: "default".into(),
                    batches,
                    limits,
                }],
            ),
            ("spiraltorch.byte_corpus.serving_gate.v2", None, None, Some(cohorts)) => (2, cohorts),
            _ => {
                return Err(
                    "unsupported schema or mixed/missing version-specific serving fields".into(),
                )
            }
        };
        if !(1..=512).contains(&steps)
            || !(1..=1024).contains(&documents.len())
            || !(1..=32).contains(&requests.len())
        {
            return Err("invalid bounded serving evaluation".into());
        }
        let mut names = std::collections::BTreeSet::new();
        let mut total = 0usize;
        for request in &requests {
            request.limits.validate()?;
            if request.name.is_empty()
                || request.name.len() > 128
                || request.name.trim() != request.name
                || request.name.chars().any(char::is_control)
                || !names.insert(&request.name)
                || request.batches.is_empty()
            {
                return Err(
                    "cohorts need unique bounded names, explicit limits and nonempty batches"
                        .into(),
                );
            }
            total = total
                .checked_add(request.batches.len())
                .ok_or("cohort batch count overflow")?;
            if total > 128 {
                return Err("serving gate exceeds 128 total batches".into());
            }
        }
        let count = requests[0].batches[0].len();
        if !(1..=32).contains(&count)
            || requests
                .iter()
                .flat_map(|c| &c.batches)
                .any(|b| b.len() != count)
        {
            return Err(
                "serving batches need the same nonzero row count across all cohorts".into(),
            );
        }
        let docs: Vec<_> = documents.iter().map(Vec::as_slice).collect();
        let mut cohorts = Vec::with_capacity(requests.len());
        for request in requests {
            let batches = request
                .batches
                .iter()
                .map(|b| ByteLmBatch::from_documents(&docs, b, steps))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            cohorts.push(Cohort {
                name: request.name,
                batches,
                limits: request.limits,
            });
        }
        Ok(Self {
            json: std::str::from_utf8(input)?.to_owned(),
            request_sha256: format!("{:x}", Sha256::digest(input)),
            version,
            cohorts,
        })
    }

    pub fn to_json(&self) -> &str {
        &self.json
    }
    pub fn request_sha256(&self) -> &str {
        &self.request_sha256
    }
    pub fn batch_count(&self) -> usize {
        self.cohorts.iter().map(|c| c.batches.len()).sum()
    }
    pub fn batch_shape(&self) -> [usize; 2] {
        self.cohorts[0].batches[0].shape()
    }
    pub fn schema_version(&self) -> u8 {
        self.version
    }
    /// Ordered, caller-declared labels, not language detection or split proofs.
    /// Legacy v1 has one implicit cohort named `default`.
    pub fn cohort_names(&self) -> Vec<&str> {
        self.cohorts.iter().map(|c| c.name.as_str()).collect()
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
