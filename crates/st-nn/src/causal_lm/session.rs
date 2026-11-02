//! Rust-owned, document-safe SGD sessions for LoRA, geometry or their composition.
//! The decoder owns derivatives; this module owns only data order and continuation.
use super::training::{Corpus, MAX_TRAINING_TOKENS};
use super::*;
use crate::language_windows::{
    equal_width_groups, unique_group_shapes, CausalDocumentWindows, MAX_DOCUMENTS, MAX_WINDOWS,
};
use sha2::{Digest, Sha256};

const SCHEMA: &str = "spiraltorch.native_causal_lm_training_session.v1";
const BATCH_SCHEMA: &str = "spiraltorch.native_causal_lm_training_session.v2";
const DATA_SCHEMA: &str = "spiraltorch.native_causal_lm_training_session.v3";
const MAX_CHECKPOINT_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmSessionWindowPolicy {
    #[default]
    Disjoint,
    CoveringTargets,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct DataPolicy {
    window_policy: CausalLmSessionWindowPolicy,
    has_labels: bool,
}

/// Fixed-order document windows. Batching is opt-in via `new_batched`;
/// there is no hidden scheduler, shuffling, gradient accumulation or momentum.
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
    batch_windows: usize,
    updates: u64,
    data_policy: DataPolicy,
    // v1/v2 keep their implicit fixed-width schedule. v3 stores live ranges
    // only, derived from the pinned corpus rather than trusted from a checkpoint.
    live_ranges: Option<Arc<[(u32, u32)]>>,
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
    #[serde(
        default,
        deserialize_with = "declared_batch",
        skip_serializing_if = "Option::is_none"
    )]
    batch_windows: Option<u32>,
    #[serde(
        default,
        deserialize_with = "declared_data",
        skip_serializing_if = "Option::is_none"
    )]
    data_policy: Option<DataPolicy>,
}

fn declared_data<'de, D: serde::Deserializer<'de>>(
    input: D,
) -> Result<Option<DataPolicy>, D::Error> {
    DataPolicy::deserialize(input).map(Some)
}

// Missing means v1. Explicit null must not silently downgrade a malformed v2 policy.
fn declared_batch<'de, D: serde::Deserializer<'de>>(input: D) -> Result<Option<u32>, D::Error> {
    u32::deserialize(input).map(Some)
}

/// Artifact/config owners verified before a binding reads foreign corpus values.
pub struct CausalLmTrainingSessionResume {
    saved: Checkpoint,
    model: NativeCausalLm,
}

impl CausalLmTrainingSessionResume {
    fn validate_recorded_boundary(&self) -> CausalLmResult<()> {
        let context = self.saved.config.context_length as usize;
        let epochs = u64::from(self.saved.config.epochs);
        let covering = self.window_policy() == CausalLmSessionWindowPolicy::CoveringTargets;
        let max_batches = if covering {
            // Ragged windows can require singleton batches even with a large row cap.
            MAX_WINDOWS.min(MAX_TRAINING_TOKENS as u64)
        } else {
            MAX_WINDOWS
                .min((MAX_TRAINING_TOKENS / context) as u64)
                .div_ceil(self.batch_windows() as u64)
        };
        if !crate::geometry_artifact::is_sha256_id(&self.saved.data_id)
            || context - 1 > self.model.config.positions
            || max_batches == 0
            || self.saved.updates > max_batches * epochs
        {
            return Err("invalid recorded training session boundary".into());
        }
        // Without the corpus we can only prove a minimum shape. More than one
        // update per epoch requires a full batch, even if the final batch was short.
        let min_rows = if !covering && self.saved.updates > epochs {
            self.batch_windows()
        } else {
            1
        };
        if let Some(runtime) = &self.model.geometry {
            for p in &runtime.artifact().record().placements {
                runtime.sequence_dimensions(
                    &p.path,
                    [min_rows, if covering { 1 } else { context - 1 }],
                )?;
            }
        }
        Ok(())
    }

    pub fn config(&self) -> CausalLmTrainingSessionConfig {
        self.saved.config
    }
    pub fn model(&self) -> &NativeCausalLm {
        &self.model
    }
    pub fn batch_windows(&self) -> usize {
        self.saved.batch_windows.unwrap_or(1) as usize
    }
    pub fn window_policy(&self) -> CausalLmSessionWindowPolicy {
        self.saved.data_policy.unwrap_or_default().window_policy
    }
    pub fn has_labels(&self) -> bool {
        self.saved.data_policy.unwrap_or_default().has_labels
    }
    pub fn resume(
        self,
        documents: Vec<Vec<usize>>,
    ) -> CausalLmResult<NativeCausalLmTrainingSession> {
        self.resume_with_labels(documents, None)
    }

    pub fn resume_with_labels(
        self,
        documents: Vec<Vec<usize>>,
        labels: Option<Vec<Vec<i64>>>,
    ) -> CausalLmResult<NativeCausalLmTrainingSession> {
        if labels.is_some() != self.has_labels() {
            return Err("session checkpoint label presence differs".into());
        }
        let batch_windows = self.batch_windows();
        let policy = self.window_policy();
        let mut restored = NativeCausalLmTrainingSession::new_with_policy(
            self.model,
            documents,
            labels,
            self.saved.config,
            batch_windows,
            policy,
        )?;
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

impl NativeCausalLm {
    /// Open a pinned SGD session for inference, without training documents.
    /// Both optional owners must match exactly: None means absent, not unchecked.
    /// A recorded data ID is not verification of a corpus or proof of exhaustion.
    pub fn from_session_checkpoint_at(
        base: Self,
        bytes: &[u8],
        expected_id: &str,
        update: u64,
        recorded_data_id: &str,
        post_lora_id: Option<&str>,
        post_geometry_id: Option<&str>,
    ) -> CausalLmResult<Self> {
        let state = NativeCausalLmTrainingSession::prepare_restore(base, bytes, expected_id)?;
        let saved = &state.saved;
        if saved.data_id != recorded_data_id
            || saved.updates != update
            || state.model.lora_artifact_id() != post_lora_id
            || state.model.geometry_artifact_id() != post_geometry_id
        {
            return Err(
                "session checkpoint does not match a valid requested inference boundary".into(),
            );
        }
        Ok(state.model)
    }
}

impl NativeCausalLmTrainingSession {
    pub const MAX_TOKENS: usize = MAX_TRAINING_TOKENS;
    pub const MAX_PAYLOAD_BYTES: usize = MAX_CHECKPOINT_BYTES;
    /// Bound the number of per-update reports retained for one disk commit.
    pub const MAX_CHECKPOINT_INTERVAL: u64 = checkpoint::MAX_INTERVAL;

    /// Admit lengths/work before bindings eagerly copy nested token arrays.
    pub fn document_plan(
        model: &NativeCausalLm,
        config: CausalLmTrainingSessionConfig,
        lengths: &[usize],
    ) -> CausalLmResult<CausalDocumentWindows> {
        Self::document_plan_batched(model, config, lengths, 1)
    }

    pub fn validate_policy(
        model: &NativeCausalLm,
        config: CausalLmTrainingSessionConfig,
        batch_windows: usize,
    ) -> CausalLmResult<()> {
        config.validate()?;
        model.validate_adaptation_mode(config.mode)?;
        if batch_windows == 0 || batch_windows > crate::language_windows::MAX_WINDOWS as usize {
            return Err("training session batch_windows exceeds its valid range".into());
        }
        Ok(())
    }

    pub fn document_plan_batched(
        model: &NativeCausalLm,
        config: CausalLmTrainingSessionConfig,
        lengths: &[usize],
        batch_windows: usize,
    ) -> CausalLmResult<CausalDocumentWindows> {
        Self::document_plan_with_policy(
            model,
            config,
            lengths,
            batch_windows,
            CausalLmSessionWindowPolicy::Disjoint,
            false,
        )
    }

    pub fn document_plan_with_policy(
        model: &NativeCausalLm,
        config: CausalLmTrainingSessionConfig,
        lengths: &[usize],
        batch_windows: usize,
        window_policy: CausalLmSessionWindowPolicy,
        has_labels: bool,
    ) -> CausalLmResult<CausalDocumentWindows> {
        Self::validate_policy(model, config, batch_windows)?;
        if config.context_length as usize - 1 > model.config.positions {
            return Err("training context exceeds model positions".into());
        }
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
        let lengths = lengths.iter().map(|&n| n as u32).collect();
        let plan = match window_policy {
            CausalLmSessionWindowPolicy::Disjoint => {
                CausalDocumentWindows::from_lengths(lengths, config.context_length)?
            }
            CausalLmSessionWindowPolicy::CoveringTargets => {
                CausalDocumentWindows::covering_targets(lengths, config.context_length)?
            }
        };
        if plan.windows().is_empty() {
            return Err(match window_policy {
                CausalLmSessionWindowPolicy::Disjoint => "no full document windows",
                CausalLmSessionWindowPolicy::CoveringTargets => "no within-document targets",
            }
            .into());
        }
        if window_policy == CausalLmSessionWindowPolicy::CoveringTargets {
            for shape in unique_group_shapes(plan.windows(), batch_windows) {
                model.adaptation_labeled_input_count(shape, has_labels)?;
            }
            return Ok(plan);
        }
        let windows = plan.windows().len();
        let rows = batch_windows.min(windows);
        model.adaptation_labeled_input_count([rows, config.context_length as usize], has_labels)?;
        let tail = windows % batch_windows;
        if tail != 0 && tail != rows {
            model.adaptation_labeled_input_count(
                [tail, config.context_length as usize],
                has_labels,
            )?;
        }
        Ok(plan)
    }

    pub fn new(
        model: NativeCausalLm,
        documents: Vec<Vec<usize>>,
        config: CausalLmTrainingSessionConfig,
    ) -> CausalLmResult<Self> {
        Self::new_batched(model, documents, config, 1)
    }

    /// One target-mean SGD update on up to `batch_windows` independent rows.
    /// The last batch of each epoch can be short; it never borrows from the next epoch.
    pub fn new_batched(
        model: NativeCausalLm,
        documents: Vec<Vec<usize>>,
        config: CausalLmTrainingSessionConfig,
        batch_windows: usize,
    ) -> CausalLmResult<Self> {
        Self::new_with_policy(
            model,
            documents,
            None,
            config,
            batch_windows,
            CausalLmSessionWindowPolicy::Disjoint,
        )
    }

    /// Labels are unshifted per document; -100 masks the target, not its context.
    pub fn new_with_policy(
        model: NativeCausalLm,
        documents: Vec<Vec<usize>>,
        labels: Option<Vec<Vec<i64>>>,
        config: CausalLmTrainingSessionConfig,
        batch_windows: usize,
        window_policy: CausalLmSessionWindowPolicy,
    ) -> CausalLmResult<Self> {
        if documents.len() > MAX_DOCUMENTS {
            return Err("document count exceeds budget".into());
        }
        let lengths: Vec<_> = documents.iter().map(Vec::len).collect();
        let data_policy = DataPolicy {
            window_policy,
            has_labels: labels.is_some(),
        };
        let plan = Self::document_plan_with_policy(
            &model,
            config,
            &lengths,
            batch_windows,
            window_policy,
            data_policy.has_labels,
        )?;
        if documents
            .iter()
            .flatten()
            .any(|&id| id >= model.config.vocabulary)
        {
            return Err("invalid training document token ID".into());
        }
        if let Some(labels) = &labels {
            Corpus::validate_labels(&documents, labels, model.config.vocabulary)?;
        }
        let corpus = Arc::new(Corpus::new(documents, labels, plan)?);
        let live_ranges = if data_policy == DataPolicy::default() {
            None
        } else {
            let windows = corpus.plan.windows();
            let mut live = Vec::new();
            for range in equal_width_groups(windows, batch_windows) {
                let active = corpus.active_targets(&windows[range.clone()]);
                if active > 0 {
                    model.logit_count(active)?;
                    live.push((range.start as u32, range.end as u32));
                }
            }
            if live.is_empty() {
                return Err("no supervised targets in training windows".into());
            }
            Some(Arc::from(live))
        };
        Ok(Self {
            model,
            corpus,
            config,
            batch_windows,
            updates: 0,
            data_policy,
            live_ranges,
        })
    }
    pub fn model(&self) -> &NativeCausalLm {
        &self.model
    }
    pub fn config(&self) -> CausalLmTrainingSessionConfig {
        self.config
    }
    pub fn batch_windows(&self) -> usize {
        self.batch_windows
    }
    pub fn window_policy(&self) -> CausalLmSessionWindowPolicy {
        self.data_policy.window_policy
    }
    pub fn has_labels(&self) -> bool {
        self.data_policy.has_labels
    }
    fn declared_data_policy(&self) -> Option<DataPolicy> {
        (self.data_policy != DataPolicy::default()).then_some(self.data_policy)
    }
    pub fn config_json(&self) -> CausalLmResult<String> {
        #[derive(Serialize)]
        struct Policy {
            #[serde(flatten)]
            config: CausalLmTrainingSessionConfig,
            #[serde(skip_serializing_if = "Option::is_none")]
            batch_windows: Option<u32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            data_policy: Option<DataPolicy>,
        }
        Ok(serde_json::to_string(&Policy {
            config: self.config,
            batch_windows: self.declared_batch(),
            data_policy: self.declared_data_policy(),
        })?)
    }
    fn declared_batch(&self) -> Option<u32> {
        (self.batch_windows > 1).then_some(self.batch_windows as u32)
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
        self.live_ranges.as_ref().map_or_else(
            || self.plan().windows().len().div_ceil(self.batch_windows),
            |ranges| ranges.len(),
        ) as u64
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
            self.next_range().start
        }
    }
    fn next_range(&self) -> std::ops::Range<usize> {
        let index = (self.updates % self.updates_per_epoch()) as usize;
        if let Some(ranges) = &self.live_ranges {
            let (start, end) = ranges[index];
            start as usize..end as usize
        } else {
            let start = index * self.batch_windows;
            start..(start + self.batch_windows).min(self.plan().windows().len())
        }
    }

    /// Compute both groups from the same pre-update snapshot. No session field
    /// changes until every selected parameter update has validated successfully.
    pub fn step(&mut self) -> CausalLmResult<Option<CausalLmTrainingSessionStep>> {
        if self.exhausted() {
            return Ok(None);
        }
        let std::ops::Range { start, end } = self.next_range();
        let windows = &self.plan().windows()[start..end];
        let (ids, labels) = self.corpus.batch(windows);
        let gradients = self.model.adaptation_loss_gradients_with_labels(
            &ids,
            [end - start, windows[0].length as usize],
            labels.as_deref(),
            Some(self.config.mode),
        )?;
        let model = self.model.step_adaptation_sgd(
            &gradients,
            self.config.mode,
            self.config.learning_rate,
        )?;
        let report = CausalLmTrainingSessionStep {
            update: self.updates + 1,
            epoch: self.epoch(),
            window_start: start,
            window_end: end,
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
            schema: if self.declared_data_policy().is_some() {
                DATA_SCHEMA
            } else if self.batch_windows == 1 {
                SCHEMA
            } else {
                BATCH_SCHEMA
            }
            .into(),
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
            batch_windows: self.declared_batch(),
            data_policy: self.declared_data_policy(),
        };
        let json = serde_json::to_string(&checkpoint)?;
        if json.len() > MAX_CHECKPOINT_BYTES {
            return Err("training session checkpoint exceeds byte budget".into());
        }
        Ok(json)
    }
    pub fn checkpoint_id(&self) -> CausalLmResult<String> {
        Ok(self.checkpoint_with_id()?.1)
    }

    /// Serialize the two adapter owners and cursor once and pin exactly these bytes.
    pub fn checkpoint_with_id(&self) -> CausalLmResult<(String, String)> {
        let json = self.to_json()?;
        let id = identity(json.as_bytes());
        Ok((json, id))
    }

    /// Stage the exact checkpoint bytes and return their pin after no-clobber
    /// publication. Saving never advances the cursor or changes either adapter.
    pub fn save_checkpoint_new(&self, path: impl AsRef<std::path::Path>) -> CausalLmResult<String> {
        checkpoint::save(path.as_ref(), self.checkpoint_with_id()?)
    }

    /// Run the ordinary SGD steps up to an absolute cursor on a private session
    /// snapshot. Commit the live cursor and both owners only after publishing
    /// the exact resume checkpoint at a new name. No per-step disk I/O.
    /// On a late filesystem error a complete file can remain; never auto-retry.
    pub fn advance_to_checkpoint_new(
        &mut self,
        stop_after: u64,
        path: impl AsRef<std::path::Path>,
    ) -> CausalLmResult<(Vec<CausalLmTrainingSessionStep>, String)> {
        let count = checkpoint::count(self.updates, self.total_updates(), stop_after)?;
        checkpoint::advance_new(
            self,
            count,
            path.as_ref(),
            Self::step,
            Self::checkpoint_with_id,
        )
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
        let valid_batch = |n: u32| (2..=MAX_WINDOWS).contains(&u64::from(n));
        let valid_policy = match (
            saved.schema.as_str(),
            saved.batch_windows,
            saved.data_policy,
        ) {
            (SCHEMA, None, None) => true,
            (BATCH_SCHEMA, Some(n), None) => valid_batch(n),
            (DATA_SCHEMA, n, Some(policy)) => {
                policy != DataPolicy::default() && n.map_or(true, valid_batch)
            }
            _ => false,
        };
        if !valid_policy || saved.base_model_identity != base.base_id {
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
        let restored = CausalLmTrainingSessionResume { saved, model };
        restored.validate_recorded_boundary()?;
        Ok(restored)
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

#[cfg(all(test, not(target_arch = "wasm32")))]
#[test]
fn failed_publication_after_sgd_preserves_session_and_older_checkpoint() {
    let model = super::tests::load().with_lora(0, 2, 3.).unwrap();
    let config = CausalLmTrainingSessionConfig {
        context_length: 4,
        epochs: 3,
        learning_rate: 0.2,
        mode: CausalLmAdaptationMode::Lora,
    };
    let mut s = NativeCausalLmTrainingSession::new(model, vec![vec![1, 2, 3, 4]], config).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let older = dir.path().join("initial.json");
    let pin = s.save_checkpoint_new(&older).unwrap();
    let before = std::fs::read(&older).unwrap();
    for late in [false, true] {
        let path = dir.path().join("next.json");
        let failure = checkpoint::transaction::<_, ()>(&mut s, |candidate| {
            checkpoint::steps(candidate, 2, NativeCausalLmTrainingSession::step)?;
            assert_eq!(candidate.completed_updates(), 2);
            assert_ne!(candidate.checkpoint_id()?, pin);
            if late {
                candidate.save_checkpoint_new(&path)?;
            }
            Err(std::io::Error::from(std::io::ErrorKind::StorageFull).into())
        })
        .unwrap_err();
        assert_eq!(
            failure.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::StorageFull
        );
        assert_eq!(path.exists(), late);
        assert_eq!(s.checkpoint_id().unwrap(), pin);
        assert_eq!(s.completed_updates(), 0);
        assert_eq!(std::fs::read(&older).unwrap(), before);
    }
}
