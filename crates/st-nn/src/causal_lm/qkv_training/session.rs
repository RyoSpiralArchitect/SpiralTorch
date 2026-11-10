//! Data-bound ordered QKV training; shared windows and optimizer own semantics.
use super::super::training::{Corpus, MAX_TRAINING_TOKENS};
use super::*;
use crate::language_windows::{CausalDocumentWindow, CausalDocumentWindows, MAX_DOCUMENTS};

const SCHEMA: &str = "spiraltorch.native_causal_lm_qkv_trainer.v1";
const HEADER_LIMIT: usize = 16384;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CausalLmQkvTrainingConfig {
    pub context_length: u32,
    pub batch_windows: usize,
    pub max_rows: usize,
    pub epochs: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub label_smoothing: f64,
}

impl CausalLmQkvTrainingConfig {
    pub fn validate(self) -> CausalLmResult<()> {
        if self.context_length < 2
            || self.batch_windows == 0
            || self.batch_windows > NativeCausalLmQkvAdamW::MAX_CAUSAL_DOCUMENTS
            || self.max_rows == 0
            || self.max_rows > NativeCausalLmQkvAdamW::MAX_CAUSAL_DOCUMENTS
            || self.epochs == 0
            || self.epochs > 1_048_576
            || self.learning_rate <= 0.
        {
            return Err("invalid QKV trainer context, batch, epochs or rate".into());
        }
        AdamWStep::new(self.learning_rate, self.weight_decay)?;
        CausalLmObjective {
            label_smoothing: self.label_smoothing,
            ..Default::default()
        }
        .validate()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmQkvTrainingStep {
    pub update: u64,
    /// Zero-based epoch and half-open indices in the active window plan.
    pub epoch: u32,
    pub window_start: usize,
    pub window_end: usize,
    pub optimizer: CausalLmQkvDocumentStep,
}

#[derive(Clone, Debug)]
pub struct NativeCausalLmQkvTrainer {
    optimizer: NativeCausalLmQkvAdamW,
    corpus: Arc<Corpus>,
    windows: Arc<[CausalDocumentWindow]>,
    config: CausalLmQkvTrainingConfig,
    updates: u64,
    initial_steps: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema: String,
    data_id: String,
    config: CausalLmQkvTrainingConfig,
    updates: u64,
    initial_steps: u32,
    optimizer_id: String,
    has_labels: bool,
}

/// Verified model/optimizer owners and policy before a client supplies corpus IDs.
pub struct CausalLmQkvTrainingResume {
    saved: Record,
    optimizer: NativeCausalLmQkvAdamW,
}

impl CausalLmQkvTrainingResume {
    pub fn document_plan(
        &self,
        lengths: &[usize],
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        if has_labels != self.saved.has_labels {
            return Err("QKV trainer label presence changed".into());
        }
        NativeCausalLmQkvTrainer::document_plan(
            &self.optimizer,
            self.saved.config,
            lengths,
            has_labels,
        )
    }

    pub fn resume(
        self,
        documents: Vec<Vec<usize>>,
        labels: Option<Vec<Vec<i64>>>,
    ) -> CausalLmResult<NativeCausalLmQkvTrainer> {
        if labels.is_some() != self.saved.has_labels {
            return Err("QKV trainer label presence changed".into());
        }
        let mut trainer = NativeCausalLmQkvTrainer::from_corpus(
            self.optimizer,
            documents,
            labels,
            self.saved.config,
            self.saved.initial_steps,
        )?;
        if trainer.data_id() != self.saved.data_id || self.saved.updates > trainer.total_updates() {
            return Err("QKV trainer corpus, plan or cursor changed".into());
        }
        trainer.updates = self.saved.updates;
        Ok(trainer)
    }
}

fn identity(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

impl NativeCausalLmQkvTrainer {
    pub const MAX_TOKENS: usize = MAX_TRAINING_TOKENS;
    pub const MAX_CHECKPOINT_INTERVAL: u64 = super::super::checkpoint::MAX_INTERVAL;

    pub fn save_checkpoint_new(&self, path: impl AsRef<std::path::Path>) -> CausalLmResult<String> {
        super::super::checkpoint::admit_new(path.as_ref())?;
        super::super::checkpoint::save(path.as_ref(), self.checkpoint()?)
    }

    /// Commit the live owner only after the complete segment has been published.
    pub fn advance_to_checkpoint_new(
        &mut self,
        stop: u64,
        path: impl AsRef<std::path::Path>,
    ) -> CausalLmResult<(Vec<CausalLmQkvTrainingStep>, String)> {
        let count = super::super::checkpoint::count(self.updates(), self.total_updates(), stop)?;
        super::super::checkpoint::advance_new(
            self,
            count,
            path.as_ref(),
            Self::step,
            Self::checkpoint,
        )
    }
    pub const MAX_CHECKPOINT_BYTES: usize =
        4 + HEADER_LIMIT + NativeCausalLmQkvAdamW::MAX_CHECKPOINT_BYTES;

    /// Cover within-document targets once, retaining short tails. This bounds
    /// transport; actual live groups are admitted after label inspection.
    pub fn document_plan(
        optimizer: &NativeCausalLmQkvAdamW,
        config: CausalLmQkvTrainingConfig,
        lengths: &[usize],
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        config.validate()?;
        if lengths.is_empty()
            || lengths.len() > MAX_DOCUMENTS
            || config.context_length as usize - 1 > optimizer.model.config.positions
        {
            return Err("invalid QKV training corpus or context".into());
        }
        let count = lengths
            .iter()
            .try_fold(0usize, |a, &b| a.checked_add(b))
            .filter(|&n| n <= Self::MAX_TOKENS)
            .ok_or("QKV training corpus exceeds token budget")?;
        if count == 0 {
            return Err("empty QKV training corpus".into());
        }
        let plan = CausalDocumentWindows::covering_targets(
            lengths.iter().map(|&n| n as u32).collect(),
            config.context_length,
        )?;
        if plan.windows().is_empty() {
            return Err("QKV training corpus has no target windows".into());
        }
        let mut widths = std::collections::HashSet::new();
        for window in plan.windows() {
            if widths.insert(window.length) {
                optimizer.loss_input_count([1, window.length as usize], has_labels)?;
            }
        }
        Ok(plan)
    }

    pub fn new(
        optimizer: NativeCausalLmQkvAdamW,
        documents: Vec<Vec<usize>>,
        labels: Option<Vec<Vec<i64>>>,
        config: CausalLmQkvTrainingConfig,
    ) -> CausalLmResult<Self> {
        let initial_steps = optimizer.steps();
        Self::from_corpus(optimizer, documents, labels, config, initial_steps)
    }

    fn from_corpus(
        optimizer: NativeCausalLmQkvAdamW,
        documents: Vec<Vec<usize>>,
        labels: Option<Vec<Vec<i64>>>,
        config: CausalLmQkvTrainingConfig,
        initial_steps: u32,
    ) -> CausalLmResult<Self> {
        let lengths = documents.iter().map(Vec::len).collect::<Vec<_>>();
        let plan = Self::document_plan(&optimizer, config, &lengths, labels.is_some())?;
        if documents
            .iter()
            .flatten()
            .any(|&id| id >= optimizer.model.config.vocabulary)
        {
            return Err("invalid QKV training token".into());
        }
        if let Some(rows) = &labels {
            Corpus::validate_labels(&documents, rows, optimizer.model.config.vocabulary)?;
        }
        let has_labels = labels.is_some();
        let corpus = Corpus::new(documents, labels, plan)?;
        let windows = corpus
            .plan
            .windows()
            .iter()
            .filter(|w| corpus.active_targets(std::slice::from_ref(w)) > 0)
            .cloned()
            .collect::<Vec<_>>();
        if windows.is_empty() {
            return Err("QKV training corpus has no selected targets".into());
        }
        // Admit every actual batch, with no decoder execution or retained copies.
        for batch in windows.chunks(config.batch_windows) {
            let lengths = batch.iter().map(|w| w.length as usize).collect::<Vec<_>>();
            optimizer.document_input_count(&lengths, config.max_rows, has_labels)?;
            for group in crate::language_windows::equal_width_groups(batch, config.max_rows) {
                optimizer
                    .model
                    .logit_count(corpus.active_targets(&batch[group]))?;
            }
        }
        let total = windows.len().div_ceil(config.batch_windows) as u64 * u64::from(config.epochs);
        if total > u64::from(u32::MAX - initial_steps) {
            return Err("QKV training plan exceeds optimizer clock".into());
        }
        Ok(Self {
            optimizer,
            corpus: Arc::new(corpus),
            windows: windows.into(),
            config,
            updates: 0,
            initial_steps,
        })
    }

    pub fn config(&self) -> CausalLmQkvTrainingConfig {
        self.config
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    pub fn data_id(&self) -> &str {
        &self.corpus.id
    }
    pub fn active_windows(&self) -> usize {
        self.windows.len()
    }
    pub fn total_updates(&self) -> u64 {
        self.windows.len().div_ceil(self.config.batch_windows) as u64
            * u64::from(self.config.epochs)
    }
    pub fn optimizer(&self) -> &NativeCausalLmQkvAdamW {
        &self.optimizer
    }

    pub fn step(&mut self) -> CausalLmResult<Option<CausalLmQkvTrainingStep>> {
        if self.updates == self.total_updates() {
            return Ok(None);
        }
        let batches = self.windows.len().div_ceil(self.config.batch_windows) as u64;
        let epoch = (self.updates / batches) as u32;
        let start = (self.updates % batches) as usize * self.config.batch_windows;
        let end = (start + self.config.batch_windows).min(self.windows.len());
        let (ids, labels) = self.corpus.window_rows(&self.windows[start..end]);
        let report = self.optimizer.step_documents(
            &ids,
            labels.as_deref(),
            self.config.max_rows,
            self.config.label_smoothing,
            AdamWStep::new(self.config.learning_rate, self.config.weight_decay)?,
        )?;
        self.updates += 1;
        Ok(Some(CausalLmQkvTrainingStep {
            update: self.updates,
            epoch,
            window_start: start,
            window_end: end,
            optimizer: report,
        }))
    }

    pub fn checkpoint(&self) -> CausalLmResult<(Vec<u8>, String)> {
        let (payload, pin) = self.optimizer.checkpoint()?;
        let saved = Record {
            schema: SCHEMA.into(),
            data_id: self.data_id().into(),
            config: self.config,
            updates: self.updates,
            initial_steps: self.initial_steps,
            optimizer_id: pin,
            has_labels: self.corpus.has_labels(),
        };
        let header = serde_json::to_vec(&saved)?;
        if header.len() > HEADER_LIMIT {
            return Err("QKV trainer header exceeds bound".into());
        }
        let mut out = Vec::with_capacity(4 + header.len() + payload.len());
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&payload);
        let pin = identity(&out);
        Ok((out, pin))
    }

    pub fn prepare_resume(
        base: &NativeCausalLm,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<CausalLmQkvTrainingResume> {
        if bytes.len() < 4
            || bytes.len() > Self::MAX_CHECKPOINT_BYTES
            || identity(bytes) != expected_id
        {
            return Err("QKV trainer checkpoint length or identity mismatch".into());
        }
        let length = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
        if length > HEADER_LIMIT || length > bytes.len() - 4 {
            return Err("invalid QKV trainer header length".into());
        }
        let saved: Record = serde_json::from_slice(&bytes[4..4 + length])?;
        saved.config.validate()?;
        if saved.schema != SCHEMA || !crate::geometry_artifact::is_sha256_id(&saved.data_id) {
            return Err("invalid QKV trainer schema or data identity".into());
        }
        let optimizer =
            NativeCausalLmQkvAdamW::restore(base, &bytes[4 + length..], &saved.optimizer_id)?;
        if u64::from(saved.initial_steps).checked_add(saved.updates)
            != Some(u64::from(optimizer.steps()))
        {
            return Err("QKV trainer cursor and optimizer clock differ".into());
        }
        Ok(CausalLmQkvTrainingResume { saved, optimizer })
    }
}
