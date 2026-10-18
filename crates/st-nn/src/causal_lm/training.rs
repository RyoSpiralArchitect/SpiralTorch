//! Ordered document-safe training. The existing accumulator and optimizer own math.
use super::*;
use crate::geometry_runtime::{AdamWStep, NativeGeometryAdamW};
use crate::language_windows::{CausalDocumentWindow, CausalDocumentWindows, MAX_DOCUMENTS};
use sha2::{Digest, Sha256};

const SCHEMA: &str = "spiraltorch.native_causal_lm_trainer.v1";
const MAX_CHECKPOINT_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_TRAINING_TOKENS: usize = 16_777_216;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CausalLmTrainingConfig {
    pub context_length: u32,
    pub microbatch_windows: usize,
    pub accumulation_steps: usize,
    pub epochs: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub label_smoothing: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition: Option<CausalLmRepetition>,
}

impl CausalLmTrainingConfig {
    pub fn validate(self) -> CausalLmResult<()> {
        if self.context_length < 2
            || self.microbatch_windows == 0
            || self.microbatch_windows > 1_048_576
            || self.accumulation_steps == 0
            || self.accumulation_steps > 1024
            || self.epochs == 0
            || self.epochs > 1_048_576
        {
            return Err(
                "invalid native training context, batch, accumulation or epoch count".into(),
            );
        }
        AdamWStep::new(self.learning_rate, self.weight_decay)?;
        CausalLmObjective {
            label_smoothing: self.label_smoothing,
            repetition: self.repetition,
        }
        .validate()?;
        Ok(())
    }

    fn group_windows(self) -> usize {
        // Both bounds fit the portable 32-bit address range.
        self.microbatch_windows * self.accumulation_steps
    }
}

#[derive(Clone, Debug)]
struct Corpus {
    documents: Vec<Vec<usize>>,
    labels: Option<Vec<Vec<i64>>>,
    plan: CausalDocumentWindows,
    id: String,
}

impl Corpus {
    fn batch(&self, windows: &[CausalDocumentWindow]) -> (Vec<usize>, Option<Vec<i64>>) {
        let count = windows.iter().map(|w| w.length as usize).sum();
        let mut ids = Vec::with_capacity(count);
        let mut labels = self.labels.as_ref().map(|_| Vec::with_capacity(count));
        for w in windows {
            let span = w.start as usize..(w.start + w.length) as usize;
            ids.extend_from_slice(&self.documents[w.document as usize][span.clone()]);
            if let Some(targets) = &mut labels {
                targets
                    .extend_from_slice(&self.labels.as_ref().unwrap()[w.document as usize][span]);
            }
        }
        (ids, labels)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmTrainingStep {
    pub update: u64,
    pub epoch: u32,
    pub window_start: usize,
    pub window_end: usize,
    pub microbatches: u64,
    pub target_tokens: u64,
    pub loss: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_positions: Option<u64>,
    pub pre_geometry_artifact_id: String,
    pub post_geometry_artifact_id: String,
}

/// Fixed-order, unpadded document windows; no implicit shuffling, EOS or scheduler.
/// A successful step atomically advances one model/optimizer pair and its cursor.
#[derive(Clone, Debug)]
pub struct NativeCausalLmTrainer {
    model: NativeCausalLm,
    optimizer: NativeGeometryAdamW,
    corpus: Arc<Corpus>,
    config: CausalLmTrainingConfig,
    updates: u64,
    initial_optimizer_revision: u64,
    initial_optimizer_steps: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema: String,
    base_model_identity: String,
    data_id: String,
    config: CausalLmTrainingConfig,
    updates: u64,
    initial_optimizer_revision: u64,
    initial_optimizer_steps: Vec<u32>,
    geometry_json: String,
    geometry_id: String,
    optimizer_json: String,
    optimizer_id: String,
    #[serde(default, skip_serializing_if = "CausalLmReadoutAccumulator::is_f32")]
    readout_accumulator: CausalLmReadoutAccumulator,
}

/// Verified checkpoint owners, before the separately retained corpus is supplied.
pub struct CausalLmTrainingResume {
    saved: Checkpoint,
    model: NativeCausalLm,
    optimizer: NativeGeometryAdamW,
}

impl CausalLmTrainingResume {
    pub fn config(&self) -> CausalLmTrainingConfig {
        self.saved.config
    }
    pub fn model(&self) -> &NativeCausalLm {
        &self.model
    }

    pub fn resume(
        self,
        documents: Vec<Vec<usize>>,
        labels: Option<Vec<Vec<i64>>>,
    ) -> CausalLmResult<NativeCausalLmTrainer> {
        let saved = self.saved;
        let mut restored = NativeCausalLmTrainer::new(
            self.model,
            self.optimizer,
            documents,
            labels,
            saved.config,
        )?;
        if restored.data_id() != saved.data_id {
            return Err("training document content or label identity changed".into());
        }
        if saved.updates > restored.total_updates() {
            return Err("training cursor and optimizer revision are inconsistent".into());
        }
        restored.updates = saved.updates;
        restored.initial_optimizer_revision = saved.initial_optimizer_revision;
        restored.initial_optimizer_steps = saved.initial_optimizer_steps;
        Ok(restored)
    }
}

fn identity(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

impl NativeCausalLmTrainer {
    pub const MAX_TOKENS: usize = MAX_TRAINING_TOKENS;

    pub fn validate_owners(
        model: &NativeCausalLm,
        optimizer: &NativeGeometryAdamW,
        config: CausalLmTrainingConfig,
    ) -> CausalLmResult<()> {
        config.validate()?;
        if let Some(recipe) = config.repetition {
            recipe.admit_shape([1, config.context_length as usize], model.config.vocabulary)?;
        }
        if Some(optimizer.geometry_artifact_id()) != model.geometry_artifact_id() {
            return Err("training optimizer belongs to another geometry snapshot".into());
        }
        Ok(())
    }

    /// Admission for foreign clients before copying any nested token/label values.
    pub fn document_plan(
        model: &NativeCausalLm,
        config: CausalLmTrainingConfig,
        lengths: &[usize],
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        config.validate()?;
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
        let plan = CausalDocumentWindows::from_lengths(
            lengths.iter().map(|&n| n as u32).collect(),
            config.context_length,
        )?;
        if plan.windows().is_empty() {
            return Err("no full document windows".into());
        }
        if let Some(recipe) = config.repetition {
            recipe.admit_shape(
                [
                    config.microbatch_windows.min(plan.windows().len()),
                    config.context_length as usize,
                ],
                model.config.vocabulary,
            )?;
        }
        model.geometry_objective_input_count(
            [
                config.microbatch_windows.min(plan.windows().len()),
                config.context_length as usize,
            ],
            has_labels,
        )?;
        Ok(plan)
    }

    pub fn new(
        model: NativeCausalLm,
        optimizer: NativeGeometryAdamW,
        documents: Vec<Vec<usize>>,
        labels: Option<Vec<Vec<i64>>>,
        config: CausalLmTrainingConfig,
    ) -> CausalLmResult<Self> {
        Self::validate_owners(&model, &optimizer, config)?;
        let lengths: Vec<_> = documents.iter().map(Vec::len).collect();
        let plan = Self::document_plan(&model, config, &lengths, labels.is_some())?;
        if documents
            .iter()
            .flatten()
            .any(|&id| id >= model.config.vocabulary)
        {
            return Err("invalid training document token ID".into());
        }
        if let Some(labels) = &labels {
            if labels.len() != documents.len()
                || labels.iter().zip(&lengths).any(|(row, &n)| row.len() != n)
                || labels
                    .iter()
                    .flatten()
                    .any(|&v| v != -100 && (v < 0 || v as u64 >= model.config.vocabulary as u64))
            {
                return Err("invalid training document labels".into());
            }
            if config.repetition.is_some_and(|r| r.config().strength > 0.)
                && labels
                    .iter()
                    .flatten()
                    .zip(documents.iter().flatten())
                    .any(|(&label, &id)| label != -100 && label != id as i64)
            {
                return Err("repetition labels must equal input IDs or -100".into());
            }
        }
        let mut hash = Sha256::new();
        hash.update(b"spiraltorch.native_causal_training_data.v1\0");
        hash.update(plan.plan_id()?.as_bytes());
        hash.update([u8::from(labels.is_some())]);
        for (index, document) in documents.iter().enumerate() {
            hash.update((document.len() as u64).to_le_bytes());
            for &id in document {
                hash.update((id as u64).to_le_bytes());
            }
            if let Some(labels) = &labels {
                for &label in &labels[index] {
                    hash.update(label.to_le_bytes());
                }
            }
        }
        let corpus = Arc::new(Corpus {
            documents,
            labels,
            plan,
            id: format!("sha256:{:x}", hash.finalize()),
        });
        let result = Self {
            initial_optimizer_revision: optimizer.revision(),
            initial_optimizer_steps: optimizer.state().slots.iter().map(|s| s.steps).collect(),
            model,
            optimizer,
            corpus,
            config,
            updates: 0,
        };
        // Reject zero-target or over-budget microbatches before the first update.
        for windows in result
            .corpus
            .plan
            .windows()
            .chunks(config.microbatch_windows)
        {
            let active = windows
                .iter()
                .map(|w| {
                    result
                        .corpus
                        .labels
                        .as_ref()
                        .map_or(w.length as usize - 1, |labels| {
                            labels[w.document as usize]
                                [w.start as usize + 1..(w.start + w.length) as usize]
                                .iter()
                                .filter(|&&v| v != -100)
                                .count()
                        })
                })
                .sum();
            if active == 0 {
                return Err("training microbatch has no selected targets".into());
            }
            result.model.logit_count(active)?;
            if let Some(recipe) = config.repetition.filter(|r| r.config().strength > 0.) {
                let (ids, labels) = result.corpus.batch(windows);
                // Plan/work bounds depend on the complete mask, not just shape.
                // Admit every batch now, not after earlier updates have committed.
                super::repetition::RepetitionWindow::prepare(
                    recipe,
                    &ids,
                    [windows.len(), config.context_length as usize],
                    labels.as_deref(),
                    result.model.config.vocabulary,
                )?;
            }
        }
        Ok(result)
    }

    pub fn model(&self) -> &NativeCausalLm {
        &self.model
    }
    pub fn optimizer(&self) -> &NativeGeometryAdamW {
        &self.optimizer
    }
    pub fn config(&self) -> CausalLmTrainingConfig {
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
        self.plan()
            .windows()
            .len()
            .div_ceil(self.config.group_windows()) as u64
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
            return self.plan().windows().len();
        }
        (self.updates % self.updates_per_epoch()) as usize * self.config.group_windows()
    }

    pub fn step(&mut self) -> CausalLmResult<Option<CausalLmTrainingStep>> {
        if self.exhausted() {
            return Ok(None);
        }
        let start = self.next_window();
        let end = (start + self.config.group_windows()).min(self.plan().windows().len());
        let objective = CausalLmObjective {
            label_smoothing: self.config.label_smoothing,
            repetition: self.config.repetition,
        };
        let mut accumulator = self.model.geometry_accumulator_with_objective(objective)?;
        for windows in self.plan().windows()[start..end].chunks(self.config.microbatch_windows) {
            let (ids, labels) = self.corpus.batch(windows);
            accumulator.accumulate_loss_with_labels(
                &self.model,
                &ids,
                [windows.len(), self.config.context_length as usize],
                labels.as_deref(),
            )?;
        }
        let (model, optimizer) = accumulator.step_adamw(
            &self.model,
            &self.optimizer,
            AdamWStep::new(self.config.learning_rate, self.config.weight_decay)?,
        )?;
        let report = CausalLmTrainingStep {
            update: self.updates + 1,
            epoch: self.epoch(),
            window_start: start,
            window_end: end,
            microbatches: accumulator.microbatches(),
            target_tokens: accumulator.target_tokens(),
            loss: accumulator.loss().unwrap(),
            repetition_loss: self
                .config
                .repetition
                .map(|_| accumulator.repetition_loss().unwrap()),
            repetition_positions: self
                .config
                .repetition
                .map(|_| accumulator.repetition_positions()),
            pre_geometry_artifact_id: self.model.geometry_artifact_id().unwrap().into(),
            post_geometry_artifact_id: model.geometry_artifact_id().unwrap().into(),
        };
        self.model = model;
        self.optimizer = optimizer;
        self.updates += 1;
        Ok(Some(report))
    }

    pub fn to_json(&self) -> CausalLmResult<String> {
        let geometry = self.model.geometry_artifact().unwrap();
        let checkpoint = Checkpoint {
            schema: SCHEMA.into(),
            base_model_identity: self.model.base_id.clone(),
            data_id: self.data_id().into(),
            config: self.config,
            updates: self.updates,
            initial_optimizer_revision: self.initial_optimizer_revision,
            initial_optimizer_steps: self.initial_optimizer_steps.clone(),
            geometry_json: geometry.to_json().into(),
            geometry_id: geometry.artifact_id().into(),
            optimizer_json: self.optimizer.to_json()?,
            optimizer_id: self.optimizer.checkpoint_id()?,
            readout_accumulator: self.model.readout_accumulator,
        };
        let json = serde_json::to_string(&checkpoint)?;
        if json.len() > MAX_CHECKPOINT_BYTES {
            return Err("training checkpoint exceeds byte budget".into());
        }
        Ok(json)
    }

    pub fn checkpoint_id(&self) -> CausalLmResult<String> {
        Ok(identity(self.to_json()?.as_bytes()))
    }

    fn checkpoint(
        bytes: &[u8],
        expected_id: &str,
        base: &NativeCausalLm,
    ) -> CausalLmResult<Checkpoint> {
        if bytes.len() > MAX_CHECKPOINT_BYTES {
            return Err("training checkpoint exceeds byte budget".into());
        }
        if identity(bytes) != expected_id {
            return Err("training checkpoint differs from pinned identity".into());
        }
        let checkpoint: Checkpoint = serde_json::from_slice(bytes)?;
        if checkpoint.schema != SCHEMA
            || checkpoint.base_model_identity != base.base_id
            || checkpoint.readout_accumulator != base.readout_accumulator
        {
            return Err("training checkpoint schema, base or readout accumulator mismatch".into());
        }
        checkpoint.config.validate()?;
        Ok(checkpoint)
    }

    /// Bindings verify the checkpoint before reading any foreign document values.
    pub fn prepare_restore(
        base: NativeCausalLm,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<CausalLmTrainingResume> {
        let saved = Self::checkpoint(bytes, expected_id, &base)?;
        let geometry = GeometryAdapterArtifact::from_json(
            saved.geometry_json.as_bytes(),
            &saved.geometry_id,
            &base.base_id,
        )?;
        let optimizer = NativeGeometryAdamW::from_json(
            saved.optimizer_json.as_bytes(),
            &saved.optimizer_id,
            &geometry,
        )?;
        optimizer.validate_uniform_continuation(
            saved.initial_optimizer_revision,
            &saved.initial_optimizer_steps,
            if saved.config.learning_rate > 0. {
                saved.updates
            } else {
                0
            },
        )?;
        let model = base.with_geometry(geometry)?;
        Ok(CausalLmTrainingResume {
            saved,
            model,
            optimizer,
        })
    }

    pub fn from_json(
        base: NativeCausalLm,
        documents: Vec<Vec<usize>>,
        labels: Option<Vec<Vec<i64>>>,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<Self> {
        Self::prepare_restore(base, bytes, expected_id)?.resume(documents, labels)
    }
}
