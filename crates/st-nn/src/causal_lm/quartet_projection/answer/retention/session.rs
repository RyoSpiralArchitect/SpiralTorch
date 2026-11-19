//! Fixed ordered batch schedule over an immutable conditional prefix bank.
use super::super::conditional::{conditional_bank_input_values, MAX_PREPARED_PREFIX_VALUES};
use super::checkpoint::{canonical_digest, fingerprint};
use super::*;
use crate::causal_lm::checkpoint as files;
use std::path::{Path, PathBuf};

const LEGACY_SCHEMA: &str = "spiraltorch.native_conditional_answer_session.v3";
const LEGACY_PHASE_SCHEMA: &str = "spiraltorch.native_conditional_answer_session.v4";
const SCHEMA: &str = "spiraltorch.native_conditional_answer_session.start_bound.v1";
const HEADER_LIMIT: usize = 4096;

/// A complete checkpoint was installed but publication did not finish.
/// The live cursor did not advance. Verify any retained file with this pin
/// before using it for an explicit recovery.
#[derive(Debug)]
pub struct ConditionalCheckpointPublicationUncertain {
    checkpoint_id: String,
    source: st_core::runtime::AtomicPublishError,
}

impl ConditionalCheckpointPublicationUncertain {
    pub fn checkpoint_id(&self) -> &str {
        &self.checkpoint_id
    }
}

impl std::fmt::Display for ConditionalCheckpointPublicationUncertain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "conditional checkpoint installed but publication incomplete (pin {}): {}",
            self.checkpoint_id, self.source
        )
    }
}

impl std::error::Error for ConditionalCheckpointPublicationUncertain {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

fn publish_checkpoint(path: &Path, (bytes, pin): (Vec<u8>, String)) -> CausalLmResult<String> {
    match st_core::runtime::atomic_create_new_with_status(path, &bytes) {
        Ok(()) => Ok(pin),
        Err(source) if source.destination_installed() => {
            Err(Box::new(ConditionalCheckpointPublicationUncertain {
                checkpoint_id: pin,
                source,
            }))
        }
        Err(source) => Err(source.into_io_error().into()),
    }
}

/// Each scheduled occurrence has an ordered pair, even when indices repeat.
pub type CausalLmConditionalAnswerBatch = (Vec<usize>, Vec<[usize; 2]>);

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmConditionalAnswerSessionStep {
    /// One-based consumed batch position; a refused proposal also consumes a slot.
    pub position: u64,
    pub accepted_updates: u64,
    pub projection_id: String,
    pub report: CausalLmRetainedConditionalAnswerStep,
}

#[derive(Clone)]
pub struct NativeCausalLmConditionalAnswerSession {
    projection: NativeCausalLmRankOneProjection,
    guard: CausalLmConditionalMarginRetention,
    pool: Arc<super::super::CausalLmConditionalAnswerPrefixBank>,
    batches: Arc<[CausalLmConditionalAnswerBatch]>,
    epochs: u32,
    policy: CausalLmConditionalAnswerConfig,
    max_prefix_values: usize,
    plan_id: String,
    start_projection_id: Option<String>,
    predecessor_checkpoint_id: Option<String>,
    restored_checkpoint: Option<PinnedCheckpoint>,
    position: u64,
    accepted_updates: u64,
}

/// A staged interval. Dropping it leaves the session and filesystem unchanged.
pub struct CausalLmPreparedConditionalAnswerInterval<'a> {
    live: &'a mut NativeCausalLmConditionalAnswerSession,
    next: NativeCausalLmConditionalAnswerSession,
    reports: Vec<CausalLmConditionalAnswerSessionStep>,
    checkpoint: (Vec<u8>, String),
    path: PathBuf,
}

impl CausalLmPreparedConditionalAnswerInterval<'_> {
    pub fn reports(&self) -> &[CausalLmConditionalAnswerSessionStep] {
        &self.reports
    }

    pub fn checkpoint_id(&self) -> &str {
        &self.checkpoint.1
    }

    /// Publish without clobbering the destination, then advance the live session.
    pub fn publish(self) -> CausalLmResult<(Vec<CausalLmConditionalAnswerSessionStep>, String)> {
        let pin = publish_checkpoint(&self.path, self.checkpoint)?;
        *self.live = self.next;
        Ok((self.reports, pin))
    }
}

#[derive(Clone)]
struct PinnedCheckpoint {
    bytes: Arc<[u8]>,
    pin: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema: String,
    plan_id: String,
    guard_pin: String,
    pool_rows: usize,
    total_batches: u64,
    epochs: u32,
    position: u64,
    accepted_updates: u64,
    length_bits: u64,
    max_trials: usize,
    mass_inward_cosine_bits: Option<u64>,
    max_prefix_values: usize,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "read_present_id"
    )]
    start_projection_id: Option<String>,
    #[serde(
        default,
        deserialize_with = "read_present_id",
        skip_serializing_if = "Option::is_none"
    )]
    predecessor_checkpoint_id: Option<String>,
}

/// Byte-admitted restore handle, with no training or prompt evaluation yet performed.
pub struct CausalLmConditionalAnswerSessionRestore {
    record: Record,
    policy: CausalLmConditionalAnswerConfig,
    retained: CausalLmConditionalRetentionRestore,
    batches_per_epoch: usize,
    checkpoint: PinnedCheckpoint,
}

fn read_present_id<'de, D: serde::Deserializer<'de>>(input: D) -> Result<Option<String>, D::Error> {
    // An omitted field is distinct from an explicitly supplied null.
    String::deserialize(input).map(Some)
}

fn phase_plan_id(schedule: String, predecessor: Option<&str>) -> String {
    let Some(predecessor) = predecessor else {
        return schedule;
    };
    let mut hash = Sha256::new();
    hash.update(b"spiraltorch.conditional_answer_phase.v2\0");
    hash.update(predecessor.as_bytes());
    hash.update(schedule.as_bytes());
    format!("sha256:{:x}", hash.finalize())
}

/// Read-only weights from a pinned session, without its training roster or guard.
pub struct NativeCausalLmConditionalServingSnapshot {
    model: NativeCausalLm,
    projection_id: String,
    checkpoint_id: String,
    position: u64,
    total_batches: u64,
}

impl NativeCausalLmConditionalServingSnapshot {
    pub fn projection_id(&self) -> &str {
        &self.projection_id
    }
    pub fn checkpoint_id(&self) -> &str {
        &self.checkpoint_id
    }
    pub fn position(&self) -> u64 {
        self.position
    }
    pub fn total_batches(&self) -> u64 {
        self.total_batches
    }
    pub fn prefill_input_count(
        &self,
        steps: usize,
        max_cache_values: usize,
    ) -> CausalLmResult<usize> {
        self.model.context_prefill_count(steps, max_cache_values)
    }
    pub fn prefill(
        &self,
        prompt: Vec<usize>,
        max_cache_values: usize,
    ) -> CausalLmResult<NativeCausalLmProjectionPrefix> {
        NativeCausalLmProjectionPrefix::from_snapshot(
            &self.model,
            self.projection_id.clone(),
            prompt,
            max_cache_values,
        )
    }
}

fn plan_id(
    prompts: &[Vec<usize>],
    batches: &[CausalLmConditionalAnswerBatch],
    epochs: u32,
) -> String {
    fn put(h: &mut Sha256, n: usize) {
        h.update((n as u64).to_le_bytes());
    }
    let mut h = Sha256::new();
    h.update(b"spiraltorch.conditional_answer_schedule.v3\0");
    h.update(epochs.to_le_bytes());
    put(&mut h, prompts.len());
    for row in prompts {
        put(&mut h, row.len());
        for &id in row {
            put(&mut h, id);
        }
    }
    put(&mut h, batches.len());
    for (indices, pairs) in batches {
        put(&mut h, indices.len());
        for (&i, &[gold, rival]) in indices.iter().zip(pairs) {
            for n in [i, gold, rival] {
                put(&mut h, n);
            }
        }
    }
    format!("sha256:{:x}", h.finalize())
}

fn validate_schedule(
    model: &NativeCausalLm,
    pool_steps: &[usize],
    batches: &[CausalLmConditionalAnswerBatch],
    policy: CausalLmConditionalAnswerConfig,
) -> CausalLmResult<()> {
    let width = model.config.width;
    for (indices, pairs) in batches {
        if indices.len() != pairs.len() || !(1..=MAX_OUTPUT_CONTRAST_ROWS).contains(&indices.len())
        {
            return Err("invalid conditional session batch length".into());
        }
        product(
            &[indices.len(), width],
            output_margin::MAX_BANK_VALUES,
            "conditional session roster",
        )?;
        let mut tokens = 0usize;
        for (&i, &[gold, rival]) in indices.iter().zip(pairs) {
            model.check_contrast_candidates(gold, rival)?;
            tokens = tokens
                .checked_add(
                    *pool_steps
                        .get(i)
                        .ok_or("conditional session index outside pool")?,
                )
                .ok_or("conditional session token overflow")?;
        }
        product(
            &[
                tokens,
                width,
                1 + usize::from(policy.mass_inward_cosine.is_some()),
            ],
            output_margin::MAX_CAPTURE_TERMS,
            "conditional session capture work",
        )?;
    }
    Ok(())
}

impl NativeCausalLmConditionalAnswerSession {
    pub const MAX_BATCHES: usize = 1024;
    pub const MAX_TOTAL_BATCHES: u64 = 1 << 20;
    pub const MAX_CHECKPOINT_BYTES: usize =
        4 + HEADER_LIMIT + NativeCausalLmRankOneProjection::MAX_RETENTION_CHECKPOINT_BYTES;
    pub const MAX_CHECKPOINT_INTERVAL: u64 = files::MAX_INTERVAL;

    /// Admit the repeated schedule size before reading a foreign prompt pool.
    pub fn scheduled_positions(batches_per_epoch: usize, epochs: u32) -> CausalLmResult<u64> {
        if !(1..=Self::MAX_BATCHES).contains(&batches_per_epoch) {
            return Err("invalid conditional session schedule length".into());
        }
        let total = batches_per_epoch as u64 * u64::from(epochs);
        if !(1..=Self::MAX_TOTAL_BATCHES).contains(&total) {
            return Err("conditional session repeat count outside total batch budget".into());
        }
        Ok(total)
    }

    /// Admit schedule indices, labels, and work from bounded prompt lengths.
    pub fn admit_schedule_inputs(
        projection: &NativeCausalLmRankOneProjection,
        pool_steps: &[usize],
        batches: &[CausalLmConditionalAnswerBatch],
        policy: CausalLmConditionalAnswerConfig,
        max_prefix_values: usize,
        epochs: u32,
    ) -> CausalLmResult<usize> {
        Self::scheduled_positions(batches.len(), epochs)?;
        policy.validate()?;
        let values = projection.conditional_answer_bank_values(pool_steps, max_prefix_values)?;
        validate_schedule(&projection.current, pool_steps, batches, policy)?;
        Ok(values)
    }

    /// Take a snapshot of a matching projection and guard, without rebasing the guard.
    pub fn new(
        projection: &NativeCausalLmRankOneProjection,
        guard: &CausalLmConditionalMarginRetention,
        prompts: &[Vec<usize>],
        batches: Vec<CausalLmConditionalAnswerBatch>,
        policy: CausalLmConditionalAnswerConfig,
        max_prefix_values: usize,
    ) -> CausalLmResult<Self> {
        Self::new_repeated(
            projection,
            guard,
            prompts,
            batches,
            policy,
            max_prefix_values,
            1,
        )
    }

    /// Repeat the exact same ordered batch plan without expanding it or rebuilding
    /// the prefix bank between epochs. The original guard is never reset.
    pub fn new_repeated(
        projection: &NativeCausalLmRankOneProjection,
        guard: &CausalLmConditionalMarginRetention,
        prompts: &[Vec<usize>],
        batches: Vec<CausalLmConditionalAnswerBatch>,
        policy: CausalLmConditionalAnswerConfig,
        max_prefix_values: usize,
        epochs: u32,
    ) -> CausalLmResult<Self> {
        guard.verify_checkpoint_owner(projection)?;
        if !(1..=MAX_CONDITIONAL_BANK_ROWS).contains(&prompts.len()) {
            return Err("invalid conditional session pool length".into());
        }
        let max_prefix_values = max_prefix_values.min(MAX_PREPARED_PREFIX_VALUES);
        Self::admit_schedule_inputs(
            projection,
            &prompts.iter().map(Vec::len).collect::<Vec<_>>(),
            &batches,
            policy,
            max_prefix_values,
            epochs,
        )?;
        let pool = projection.prepare_conditional_answer_prefix_bank(prompts, max_prefix_values)?;
        Ok(Self {
            projection: projection.clone(),
            guard: guard.clone(),
            pool: Arc::new(pool),
            plan_id: plan_id(prompts, &batches, epochs),
            predecessor_checkpoint_id: None,
            restored_checkpoint: None,
            batches: batches.into(),
            epochs,
            policy,
            max_prefix_values,
            start_projection_id: Some(projection.projection_id()),
            position: 0,
            accepted_updates: 0,
        })
    }

    /// Start a fixed schedule after exhaustion without rebasing the original guard.
    /// Only this new phase's counters are reset.
    pub fn next_phase(
        &self,
        prompts: &[Vec<usize>],
        batches: Vec<CausalLmConditionalAnswerBatch>,
        policy: CausalLmConditionalAnswerConfig,
        max_prefix_values: usize,
        epochs: u32,
    ) -> CausalLmResult<Self> {
        if !self.is_exhausted() {
            return Err("conditional session must be exhausted before starting a new phase".into());
        }
        let (_, predecessor) = self.checkpoint()?;
        let mut next = Self::new_repeated(
            &self.projection,
            &self.guard,
            prompts,
            batches,
            policy,
            max_prefix_values,
            epochs,
        )?;
        next.plan_id = phase_plan_id(next.plan_id, Some(&predecessor));
        next.predecessor_checkpoint_id = Some(predecessor);
        Ok(next)
    }

    pub fn projection(&self) -> &NativeCausalLmRankOneProjection {
        &self.projection
    }
    pub fn plan_id(&self) -> &str {
        &self.plan_id
    }
    /// Digest of the immediate predecessor, not a proof of its history.
    pub fn predecessor_checkpoint_id(&self) -> Option<&str> {
        self.predecessor_checkpoint_id.as_deref()
    }
    pub fn position(&self) -> u64 {
        self.position
    }
    pub fn total_batches(&self) -> u64 {
        self.batches_per_epoch() as u64 * u64::from(self.epochs)
    }
    pub fn epochs(&self) -> u32 {
        self.epochs
    }
    pub fn batches_per_epoch(&self) -> usize {
        self.batches.len()
    }
    pub fn accepted_updates(&self) -> u64 {
        self.accepted_updates
    }

    pub fn is_exhausted(&self) -> bool {
        self.position == self.total_batches()
    }

    pub fn step(&mut self) -> CausalLmResult<Option<CausalLmConditionalAnswerSessionStep>> {
        if self.is_exhausted() {
            return Ok(None);
        }
        let index = (self.position % self.batches_per_epoch() as u64) as usize;
        let (indices, pairs) = &self.batches[index];
        let report = self.projection.step_indexed_conditional_answers_retaining(
            &self.pool,
            indices,
            pairs,
            self.policy,
            &mut self.guard,
        )?;
        self.position += 1;
        self.accepted_updates += u64::from(report.step.committed);
        self.restored_checkpoint = None;
        Ok(Some(CausalLmConditionalAnswerSessionStep {
            position: self.position,
            accepted_updates: self.accepted_updates,
            projection_id: self.projection.projection_id(),
            report,
        }))
    }

    pub fn checkpoint(&self) -> CausalLmResult<(Vec<u8>, String)> {
        if let Some(checkpoint) = &self.restored_checkpoint {
            return Ok((checkpoint.bytes.to_vec(), checkpoint.pin.clone()));
        }
        let (retained, guard_pin) = self.projection.checkpoint_with_retention(&self.guard)?;
        let header = serde_json::to_vec(&Record {
            schema: if self.start_projection_id.is_some() {
                SCHEMA
            } else if self.predecessor_checkpoint_id.is_some() {
                LEGACY_PHASE_SCHEMA
            } else {
                LEGACY_SCHEMA
            }
            .into(),
            plan_id: self.plan_id.clone(),
            guard_pin,
            pool_rows: self.pool.len(),
            total_batches: self.total_batches(),
            epochs: self.epochs,
            position: self.position,
            accepted_updates: self.accepted_updates,
            length_bits: self.policy.length.to_bits(),
            max_trials: self.policy.max_trials,
            mass_inward_cosine_bits: self.policy.mass_inward_cosine.map(f64::to_bits),
            max_prefix_values: self.max_prefix_values,
            start_projection_id: self.start_projection_id.clone(),
            predecessor_checkpoint_id: self.predecessor_checkpoint_id.clone(),
        })?;
        if header.len() > HEADER_LIMIT {
            return Err("conditional session header exceeds limit".into());
        }
        let mut bytes = Vec::with_capacity(4 + header.len() + retained.len());
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend(header);
        bytes.extend(retained);
        let pin = fingerprint(&bytes);
        Ok((bytes, pin))
    }

    pub fn save_checkpoint_new(&self, path: impl AsRef<Path>) -> CausalLmResult<String> {
        files::admit_new(path.as_ref())?;
        publish_checkpoint(path.as_ref(), self.checkpoint()?)
    }

    /// Publish a complete interval before advancing the live session.
    pub fn advance_to_checkpoint_new(
        &mut self,
        stop: u64,
        path: impl AsRef<Path>,
    ) -> CausalLmResult<(Vec<CausalLmConditionalAnswerSessionStep>, String)> {
        self.prepare_checkpoint_interval(stop, path)?.publish()
    }

    /// Stage updates, reports and checkpoint bytes without publishing or advancing.
    /// A binding may prepare its return values before committing the interval.
    pub fn prepare_checkpoint_interval(
        &mut self,
        stop: u64,
        path: impl AsRef<Path>,
    ) -> CausalLmResult<CausalLmPreparedConditionalAnswerInterval<'_>> {
        let count = files::count(self.position(), self.total_batches(), stop)?;
        let path = std::path::absolute(path.as_ref())?;
        files::admit_new(&path)?;
        let mut next = self.clone();
        let reports = files::steps(&mut next, count, Self::step)?;
        let checkpoint = next.checkpoint()?;
        Ok(CausalLmPreparedConditionalAnswerInterval {
            live: self,
            next,
            reports,
            checkpoint,
            path,
        })
    }

    /// Reconstruct pinned weights without training data or guard replay.
    /// This verifies bytes and source identity, not the history of the weights.
    pub fn serving_from_checkpoint(
        source: &NativeCausalLm,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<NativeCausalLmConditionalServingSnapshot> {
        let (record, _, retained, _) = Self::admit_checkpoint(source, bytes, expected_id)?;
        let projection = retained.serving_endpoint()?;
        Ok(NativeCausalLmConditionalServingSnapshot {
            projection_id: projection.projection_id(),
            model: projection.current,
            checkpoint_id: expected_id.to_owned(),
            position: record.position,
            total_batches: record.total_batches,
        })
    }

    pub fn prepare_resume(
        source: &NativeCausalLm,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<CausalLmConditionalAnswerSessionRestore> {
        let (record, policy, retained, batches_per_epoch) =
            Self::admit_checkpoint(source, bytes, expected_id)?;
        Ok(CausalLmConditionalAnswerSessionRestore {
            record,
            policy,
            retained,
            batches_per_epoch,
            checkpoint: PinnedCheckpoint {
                bytes: bytes.into(),
                pin: expected_id.to_owned(),
            },
        })
    }

    fn admit_checkpoint(
        source: &NativeCausalLm,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<(
        Record,
        CausalLmConditionalAnswerConfig,
        CausalLmConditionalRetentionRestore,
        usize,
    )> {
        if !(4..=Self::MAX_CHECKPOINT_BYTES).contains(&bytes.len())
            || !canonical_digest(expected_id)
            || fingerprint(bytes) != expected_id
        {
            return Err("conditional session checkpoint length or digest mismatch".into());
        }
        let size = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
        if size > HEADER_LIMIT || size > bytes.len() - 4 {
            return Err("invalid conditional session checkpoint frame".into());
        }
        let (header, retained) = bytes[4..].split_at(size);
        let record: Record = serde_json::from_slice(header)?;
        match (
            record.schema.as_str(),
            record.start_projection_id.as_deref(),
            record.predecessor_checkpoint_id.as_deref(),
        ) {
            (LEGACY_SCHEMA, None, None) => {}
            (LEGACY_PHASE_SCHEMA, None, Some(parent)) if canonical_digest(parent) => {}
            (SCHEMA, Some(start), parent)
                if canonical_digest(start) && parent.is_none_or(canonical_digest) => {}
            _ => return Err("invalid conditional session checkpoint schema or ancestry".into()),
        }
        if record.epochs == 0 {
            return Err("invalid conditional session checkpoint repeat policy".into());
        }
        let epochs = record.epochs;
        let batches_per_epoch = usize::try_from(record.total_batches / u64::from(epochs))?;
        if Self::scheduled_positions(batches_per_epoch, epochs)? != record.total_batches
            || !(1..=MAX_CONDITIONAL_BANK_ROWS).contains(&record.pool_rows)
            || !(1..=MAX_PREPARED_PREFIX_VALUES).contains(&record.max_prefix_values)
            || record.position > record.total_batches
            || record.accepted_updates > record.position
            || !canonical_digest(&record.plan_id)
        {
            return Err("invalid conditional session checkpoint policy or cursor".into());
        }
        let policy = CausalLmConditionalAnswerConfig {
            length: f64::from_bits(record.length_bits),
            max_trials: record.max_trials,
            mass_inward_cosine: record.mass_inward_cosine_bits.map(f64::from_bits),
        };
        policy.validate()?;
        let retained = NativeCausalLmRankOneProjection::admit_retained_checkpoint(
            source,
            retained,
            &record.guard_pin,
        )?;
        if record.accepted_updates == 0
            && record
                .start_projection_id
                .as_deref()
                .is_some_and(|id| id != retained.projection_id())
        {
            return Err("conditional session projection changed with zero accepted updates".into());
        }
        Ok((record, policy, retained, batches_per_epoch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_schedule_budget_is_separate_from_interval_limit() {
        let positions = NativeCausalLmConditionalAnswerSession::scheduled_positions;
        assert_eq!(positions(1024, 1024).unwrap(), 1 << 20);
        assert_eq!(positions(1, 1 << 20).unwrap(), 1 << 20);
        for (batches, epochs) in [(0, 1), (1025, 1), (1024, 1025), (1, 0), (1, u32::MAX)] {
            assert!(positions(batches, epochs).is_err());
        }
        assert!(files::count(0, 2048, 1025).is_err());
        assert_eq!(files::count(1024, 2048, 2048).unwrap(), 1024);
    }

    #[test]
    fn later_step_error_discards_the_staged_interval() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("failed.bin");
        let mut cursor = 0;
        let result = files::advance_new_with(
            &mut cursor,
            2,
            &path,
            |next| {
                if *next == 1 {
                    return Err("injected second batch error".into());
                }
                *next += 1;
                Ok(Some(*next))
            },
            |_| -> CausalLmResult<(Vec<u8>, String)> { panic!("invalid interval snapshot") },
            publish_checkpoint,
        );
        assert!(result.unwrap_err().to_string().contains("second batch"));
        assert_eq!(cursor, 0);
        assert!(!path.exists());
    }
}

impl CausalLmConditionalAnswerSessionRestore {
    pub fn guard_plan(&self) -> &CausalLmConditionalRetentionRestore {
        &self.retained
    }
    /// Counts of the external prompt pool and ordered batches per epoch.
    pub fn schedule_shape(&self) -> (usize, usize) {
        (self.record.pool_rows, self.batches_per_epoch)
    }
    pub fn pool_input_values(
        &self,
        lengths: &[usize],
        batches: &[CausalLmConditionalAnswerBatch],
    ) -> CausalLmResult<usize> {
        if (lengths.len(), batches.len()) != self.schedule_shape() {
            return Err("conditional session schedule shape changed".into());
        }
        let values = conditional_bank_input_values(
            &self.retained.source,
            self.retained.spec,
            lengths.len(),
            lengths.iter().copied(),
            self.record.max_prefix_values,
        )?;
        validate_schedule(&self.retained.source, lengths, batches, self.policy)?;
        Ok(values)
    }

    pub fn resume(
        self,
        prompts: &[Vec<usize>],
        batches: Vec<CausalLmConditionalAnswerBatch>,
        guard_prompts: &[Vec<usize>],
        guard_pairs: &[[usize; 2]],
    ) -> CausalLmResult<NativeCausalLmConditionalAnswerSession> {
        if (prompts.len(), batches.len()) != self.schedule_shape() {
            return Err("conditional session schedule shape changed".into());
        }
        self.pool_input_values(&prompts.iter().map(Vec::len).collect::<Vec<_>>(), &batches)?;
        let id = phase_plan_id(
            plan_id(prompts, &batches, self.record.epochs),
            self.record.predecessor_checkpoint_id.as_deref(),
        );
        if id != self.record.plan_id {
            return Err("conditional session training pool or schedule changed".into());
        }
        let legacy = self.record.start_projection_id.is_none();
        let (projection, guard) =
            self.retained
                .restore_session(guard_prompts, guard_pairs, legacy)?;
        let pool = projection
            .prepare_conditional_answer_prefix_bank(prompts, self.record.max_prefix_values)?;
        Ok(NativeCausalLmConditionalAnswerSession {
            projection,
            guard,
            pool: Arc::new(pool),
            batches: batches.into(),
            epochs: self.record.epochs,
            policy: self.policy,
            plan_id: id,
            predecessor_checkpoint_id: self.record.predecessor_checkpoint_id,
            restored_checkpoint: Some(self.checkpoint),
            max_prefix_values: self.record.max_prefix_values,
            start_projection_id: self.record.start_projection_id,
            position: self.record.position,
            accepted_updates: self.record.accepted_updates,
        })
    }
}
