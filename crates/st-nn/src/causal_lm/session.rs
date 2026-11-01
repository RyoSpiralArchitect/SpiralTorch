//! Rust-owned, document-safe SGD sessions for LoRA, geometry or their composition.
//! The decoder owns derivatives; this module owns only data order and continuation.
use super::training::{Corpus, MAX_TRAINING_TOKENS};
use super::*;
use crate::language_windows::{CausalDocumentWindows, MAX_DOCUMENTS};
use sha2::{Digest, Sha256};

const SCHEMA: &str = "spiraltorch.native_causal_lm_training_session.v1";
const MAX_CHECKPOINT_BYTES: usize = 128 * 1024 * 1024;

/// One disjoint full document window per SGD update. No hidden scheduler,
/// shuffling, tokenizer, EOS insertion, gradient accumulation or optimizer moments.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CausalLmTrainingSessionConfig {
    pub context_length: u32,
    pub epochs: u32,
    pub learning_rate: f32,
    pub mode: CausalLmAdaptationMode,
}

impl CausalLmTrainingSessionConfig {
    pub fn validate(self) -> CausalLmResult<()> {
        if self.context_length < 2
            || self.epochs == 0
            || self.epochs > 1_048_576
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
        {
            return Err("invalid native training session context, epochs or learning rate".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmTrainingSessionStep {
    pub update: u64,
    pub epoch: u32,
    pub window_start: usize,
    pub window_end: usize,
    pub target_tokens: usize,
    pub loss: f32,
    pub mode: CausalLmAdaptationMode,
    pub pre_lora_artifact_id: Option<String>,
    pub post_lora_artifact_id: Option<String>,
    pub pre_geometry_artifact_id: Option<String>,
    pub post_geometry_artifact_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct NativeCausalLmTrainingSession {
    model: NativeCausalLm,
    corpus: Arc<Corpus>,
    config: CausalLmTrainingSessionConfig,
    updates: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema: String,
    base_model_identity: String,
    data_id: String,
    config: CausalLmTrainingSessionConfig,
    updates: u64,
    readout_accumulator: CausalLmReadoutAccumulator,
    geometry_json: Option<String>,
    geometry_id: Option<String>,
    lora_json: Option<String>,
    lora_id: Option<String>,
}

/// Artifact/config owners verified before a binding reads foreign corpus values.
pub struct CausalLmTrainingSessionResume {
    saved: Checkpoint,
    model: NativeCausalLm,
}

impl CausalLmTrainingSessionResume {
    pub fn config(&self) -> CausalLmTrainingSessionConfig {
        self.saved.config
    }
    pub fn model(&self) -> &NativeCausalLm {
        &self.model
    }
    pub fn resume(
        self,
        documents: Vec<Vec<usize>>,
    ) -> CausalLmResult<NativeCausalLmTrainingSession> {
        let mut restored =
            NativeCausalLmTrainingSession::new(self.model, documents, self.saved.config)?;
        if restored.data_id() != self.saved.data_id {
            return Err("training session document content or plan changed".into());
        }
        if self.saved.updates > restored.total_updates() {
            return Err("training session cursor exceeds its planned updates".into());
        }
        restored.updates = self.saved.updates;
        Ok(restored)
    }
}

fn identity(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

impl NativeCausalLmTrainingSession {
    pub const MAX_TOKENS: usize = MAX_TRAINING_TOKENS;

    /// Admit lengths/work before bindings eagerly copy nested token arrays.
    pub fn document_plan(
        model: &NativeCausalLm,
        config: CausalLmTrainingSessionConfig,
        lengths: &[usize],
    ) -> CausalLmResult<CausalDocumentWindows> {
        config.validate()?;
        model.validate_adaptation_mode(config.mode)?;
        if lengths.len() > MAX_DOCUMENTS {
            return Err("document count exceeds budget".into());
        }
        let count = lengths
            .iter()
            .try_fold(0usize, |a, &b| a.checked_add(b))
            .filter(|&n| n <= MAX_TRAINING_TOKENS)
            .ok_or("training token count exceeds budget")?;
        if count == 0 {
            return Err("no full document windows".into());
        }
        // Each length fits u32 because the aggregate is already bounded.
        let plan = CausalDocumentWindows::from_lengths(
            lengths.iter().map(|&n| n as u32).collect(),
            config.context_length,
        )?;
        if plan.windows().is_empty() {
            return Err("no full document windows".into());
        }
        model.adaptation_gradient_input_count([1, config.context_length as usize])?;
        Ok(plan)
    }

    pub fn new(
        model: NativeCausalLm,
        documents: Vec<Vec<usize>>,
        config: CausalLmTrainingSessionConfig,
    ) -> CausalLmResult<Self> {
        let lengths: Vec<_> = documents.iter().map(Vec::len).collect();
        let plan = Self::document_plan(&model, config, &lengths)?;
        if documents
            .iter()
            .flatten()
            .any(|&id| id >= model.config.vocabulary)
        {
            return Err("invalid training document token ID".into());
        }
        let corpus = Arc::new(Corpus::new(documents, None, plan)?);
        Ok(Self {
            model,
            corpus,
            config,
            updates: 0,
        })
    }
    pub fn model(&self) -> &NativeCausalLm {
        &self.model
    }
    pub fn config(&self) -> CausalLmTrainingSessionConfig {
        self.config
    }
    pub fn data_id(&self) -> &str {
        &self.corpus.id
    }
    pub fn plan(&self) -> &CausalDocumentWindows {
        &self.corpus.plan
    }
    pub fn completed_updates(&self) -> u64 {
        self.updates
    }
    pub fn updates_per_epoch(&self) -> u64 {
        self.plan().windows().len() as u64
    }
    pub fn total_updates(&self) -> u64 {
        self.updates_per_epoch() * u64::from(self.config.epochs)
    }
    pub fn exhausted(&self) -> bool {
        self.updates == self.total_updates()
    }
    pub fn epoch(&self) -> u32 {
        (self.updates / self.updates_per_epoch()) as u32
    }
    pub fn next_window(&self) -> usize {
        if self.exhausted() {
            self.plan().windows().len()
        } else {
            (self.updates % self.updates_per_epoch()) as usize
        }
    }

    /// Compute both groups from the same pre-update snapshot. No session field
    /// changes until every selected parameter update has validated successfully.
    pub fn step(&mut self) -> CausalLmResult<Option<CausalLmTrainingSessionStep>> {
        if self.exhausted() {
            return Ok(None);
        }
        let start = self.next_window();
        let (ids, _) = self.corpus.batch(&self.plan().windows()[start..start + 1]);
        let gradients = self
            .model
            .adaptation_loss_gradients(&ids, [1, self.config.context_length as usize])?;
        let model = self.model.step_adaptation_sgd(
            &gradients,
            self.config.mode,
            self.config.learning_rate,
        )?;
        let report = CausalLmTrainingSessionStep {
            update: self.updates + 1,
            epoch: self.epoch(),
            window_start: start,
            window_end: start + 1,
            target_tokens: gradients.target_tokens(),
            loss: gradients.loss(),
            mode: self.config.mode,
            pre_lora_artifact_id: self.model.lora_artifact_id().map(str::to_owned),
            post_lora_artifact_id: model.lora_artifact_id().map(str::to_owned),
            pre_geometry_artifact_id: self.model.geometry_artifact_id().map(str::to_owned),
            post_geometry_artifact_id: model.geometry_artifact_id().map(str::to_owned),
        };
        self.model = model;
        self.updates += 1;
        Ok(Some(report))
    }

    pub fn to_json(&self) -> CausalLmResult<String> {
        let checkpoint = Checkpoint {
            schema: SCHEMA.into(),
            base_model_identity: self.model.base_id.clone(),
            data_id: self.data_id().into(),
            config: self.config,
            updates: self.updates,
            readout_accumulator: self.model.readout_accumulator,
            geometry_json: self
                .model
                .geometry_artifact()
                .map(|a| a.to_json().to_owned()),
            geometry_id: self.model.geometry_artifact_id().map(str::to_owned),
            lora_json: self
                .model
                .lora
                .as_ref()
                .map(|_| self.model.lora_to_json())
                .transpose()?,
            lora_id: self.model.lora_artifact_id().map(str::to_owned),
        };
        let json = serde_json::to_string(&checkpoint)?;
        if json.len() > MAX_CHECKPOINT_BYTES {
            return Err("training session checkpoint exceeds byte budget".into());
        }
        Ok(json)
    }
    pub fn checkpoint_id(&self) -> CausalLmResult<String> {
        Ok(identity(self.to_json()?.as_bytes()))
    }

    /// Verify a caller-pinned byte identity and recover the saved arithmetic
    /// policy and both parameter groups on an unadapted matching base. The hash
    /// detects mismatched bytes; it is not a signature or training provenance proof.
    pub fn prepare_restore(
        base: NativeCausalLm,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<CausalLmTrainingSessionResume> {
        if bytes.len() > MAX_CHECKPOINT_BYTES {
            return Err("training session checkpoint exceeds byte budget".into());
        }
        if identity(bytes) != expected_id {
            return Err("training session checkpoint differs from pinned identity".into());
        }
        let saved: Checkpoint = serde_json::from_slice(bytes)?;
        if saved.schema != SCHEMA || saved.base_model_identity != base.base_id {
            return Err("training session checkpoint schema or base mismatch".into());
        }
        if base.geometry.is_some() || base.lora.is_some() {
            return Err("training session restore requires an unadapted base".into());
        }
        saved.config.validate()?;
        let mut model = base.with_readout_accumulator(saved.readout_accumulator);
        match (&saved.geometry_json, &saved.geometry_id) {
            (Some(json), Some(id)) => {
                let artifact =
                    GeometryAdapterArtifact::from_json(json.as_bytes(), id, &model.base_id)?;
                model = model.with_geometry(artifact)?;
            }
            (None, None) => {}
            _ => return Err("incomplete training session geometry snapshot".into()),
        }
        match (&saved.lora_json, &saved.lora_id) {
            (Some(json), Some(id)) => {
                model = model.with_lora_json(json)?;
                if model.lora_artifact_id() != Some(id.as_str()) {
                    return Err("training session LoRA identity mismatch".into());
                }
            }
            (None, None) => {}
            _ => return Err("incomplete training session LoRA snapshot".into()),
        }
        model.validate_adaptation_mode(saved.config.mode)?;
        Ok(CausalLmTrainingSessionResume { saved, model })
    }
    pub fn from_json(
        base: NativeCausalLm,
        documents: Vec<Vec<usize>>,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<Self> {
        Self::prepare_restore(base, bytes, expected_id)?.resume(documents)
    }
}
