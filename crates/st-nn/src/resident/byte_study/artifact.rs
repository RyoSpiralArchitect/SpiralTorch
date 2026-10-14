//! Content-bound study models with an executable, inference-only policy.
use super::*;

pub const BYTE_CORPUS_MODEL_ARTIFACT_SCHEMA: &str = "spiraltorch.byte_corpus.model_artifact.v1";

/// An operating policy, not a quality/promotion decision or training-resume rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ByteCorpusInferencePolicy {
    /// Fixed model window, immutable weights/geometry, no external score biases.
    /// Each forward resets positions and causal geometry to the window origin.
    #[serde(rename = "fixed_window_frozen.v1")]
    FixedWindowFrozenV1,
}

impl ByteCorpusInferencePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FixedWindowFrozenV1 => "fixed_window_frozen.v1",
        }
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema: String,
    policy: ByteCorpusInferencePolicy,
    request_sha256: String,
    study_checkpoint_sha256: String,
    case_name: String,
    completed_updates: u32,
    total_updates: u32,
    model_json: String,
}

/// An opaque, hash-pinned executable model, including learned geometry/router
/// values. The full request and study remain separate private resume companions.
/// Hash integrity is not authentication or evidence of past training/quality.
#[derive(Clone)]
pub struct ByteCorpusModelArtifact {
    json: String,
    artifact_id: String,
    record: Record,
    model: ByteDecoderCheckpoint,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
}

impl ByteCorpusModelArtifact {
    /// Check the caller-pinned SHA256 of exact UTF-8 artifact bytes before any
    /// decoding or GPU allocation. Obtain expected_id from a trusted selection,
    /// not from the same untrusted payload. Whitespace changes the identity.
    /// This validates the envelope/model; validate_artifact additionally checks
    /// source request/study companions when those private records are available.
    pub fn from_json(input: &[u8], expected_id: &str) -> Result<Self> {
        if input.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
            return Err("model artifact exceeds 64 MiB".into());
        }
        let artifact_id = format!("sha256:{}", sha256(input));
        if artifact_id != expected_id {
            return Err("model artifact identity differs from the pinned identity".into());
        }
        let record: Record = serde_json::from_slice(input)?;
        if record.schema != BYTE_CORPUS_MODEL_ARTIFACT_SCHEMA
            || !is_digest(&record.request_sha256)
            || !is_digest(&record.study_checkpoint_sha256)
            || !(1..=1024).contains(&record.total_updates)
            || record.completed_updates > record.total_updates
        {
            return Err("invalid model artifact schema, provenance or cursor".into());
        }
        let model = ByteDecoderCheckpoint::from_json(&record.model_json)?;
        if model.attempted_revision() != u64::from(record.completed_updates) {
            return Err("model artifact revision differs from the study cursor".into());
        }
        Ok(Self {
            json: std::str::from_utf8(input)?.to_owned(),
            artifact_id,
            record,
            model,
        })
    }

    /// Preserve exact bytes; clients must not reserialize a decoded dictionary.
    pub fn to_json(&self) -> &str {
        &self.json
    }

    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    pub fn policy(&self) -> ByteCorpusInferencePolicy {
        self.record.policy
    }

    pub fn case_name(&self) -> &str {
        &self.record.case_name
    }

    pub fn request_sha256(&self) -> &str {
        &self.record.request_sha256
    }

    /// SHA256 of the source study checkpoint's Rust to_json(), not the original
    /// file's arbitrary outer whitespace. Request identity remains exact bytes.
    pub fn study_checkpoint_sha256(&self) -> &str {
        &self.record.study_checkpoint_sha256
    }

    /// Low-level escape hatch: creating a separate training owner does not
    /// mutate this artifact, but that owner's later state is NOT this artifact.
    pub fn checkpoint(&self) -> &ByteDecoderCheckpoint {
        &self.model
    }

    /// Restore the policy-constrained owner, not a mutable training handle.
    #[cfg(feature = "wgpu")]
    pub fn restore_wgpu(
        &self,
        runtime: WgpuRuntime,
    ) -> std::result::Result<ResidentByteInference, crate::resident::InferenceError> {
        match self.policy() {
            ByteCorpusInferencePolicy::FixedWindowFrozenV1 => Ok(ResidentByteInference {
                artifact_id: self.artifact_id.clone(),
                model: self.model.restore_wgpu(runtime)?,
            }),
        }
    }
}

impl ByteCorpusStudy {
    /// Validate all cases, then bind the selected model, source companions and
    /// explicit inference policy. Partial studies are allowed, not promoted.
    pub fn case_artifact(
        &self,
        checkpoint: &ByteCorpusStudyCheckpoint,
        case_name: &str,
        policy: ByteCorpusInferencePolicy,
    ) -> Result<ByteCorpusModelArtifact> {
        let model = self.case_checkpoint(checkpoint, case_name)?;
        let record = Record {
            schema: BYTE_CORPUS_MODEL_ARTIFACT_SCHEMA.to_owned(),
            policy,
            request_sha256: self.request_sha256.clone(),
            study_checkpoint_sha256: sha256(checkpoint.to_json()?.as_bytes()),
            case_name: case_name.to_owned(),
            completed_updates: checkpoint.completed_updates,
            total_updates: self.total_updates() as u32,
            model_json: model.to_json()?,
        };
        let json = serde_json::to_string(&record)?;
        if json.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
            return Err("model artifact exceeds 64 MiB".into());
        }
        Ok(ByteCorpusModelArtifact {
            artifact_id: format!("sha256:{}", sha256(json.as_bytes())),
            json,
            record,
            model,
        })
    }

    /// Verify every source case and compare the complete model/provenance/policy
    /// binding. Changing even an unselected case changes the companion digest.
    pub fn validate_artifact(
        &self,
        checkpoint: &ByteCorpusStudyCheckpoint,
        artifact: &ByteCorpusModelArtifact,
    ) -> Result<()> {
        let expected = self.case_artifact(checkpoint, artifact.case_name(), artifact.policy())?;
        if expected.record != artifact.record {
            return Err("model artifact differs from its source companions".into());
        }
        Ok(())
    }
}

/// A policy-constrained owner: no SGD, parameter handles or external-bias API.
/// It reuses the existing resident forward (including retained forward tapes);
/// this interface is not a specialized inference kernel or a KV-cache decoder.
#[cfg(feature = "wgpu")]
pub struct ResidentByteInference {
    artifact_id: String,
    model: ResidentByteDecoder,
}

/// A retained forward bound to its originating artifact even after the owner
/// is dropped or the caller selects a different artifact.
#[cfg(feature = "wgpu")]
pub struct ResidentByteInferenceForward {
    artifact_id: String,
    forward: crate::resident::ResidentByteDecoderForward,
}

#[cfg(feature = "wgpu")]
impl ResidentByteInferenceForward {
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }
    pub fn forward(&self) -> &crate::resident::ResidentByteDecoderForward {
        &self.forward
    }
    pub fn into_parts(self) -> (String, crate::resident::ResidentByteDecoderForward) {
        (self.artifact_id, self.forward)
    }
}

#[cfg(feature = "wgpu")]
impl ResidentByteInference {
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    pub fn attempted_revision(&self) -> u64 {
        self.model.parameter_snapshot().revision()
    }

    pub fn tensor_device(&self) -> &st_backend_wgpu::resident_tensor::TensorDevice {
        self.model.tensor_device()
    }

    pub fn input_layout(&self) -> &NdLayout {
        self.model.input_layout()
    }

    pub fn output_layout(&self) -> &NdLayout {
        self.model.output_layout()
    }

    pub fn prepare_batch(
        &self,
        batch: &ByteLmBatch,
    ) -> std::result::Result<crate::resident::ResidentByteBatch, crate::resident::InferenceError>
    {
        self.model.prepare_batch(batch)
    }

    pub fn forward(
        &mut self,
        batch: &crate::resident::ResidentByteBatch,
    ) -> std::result::Result<ResidentByteInferenceForward, crate::resident::InferenceError> {
        Ok(ResidentByteInferenceForward {
            artifact_id: self.artifact_id.clone(),
            forward: self.model.forward(batch)?,
        })
    }
}

#[cfg(test)]
mod tests;
