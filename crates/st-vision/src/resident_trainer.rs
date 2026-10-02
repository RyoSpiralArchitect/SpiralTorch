//! Rust-owned resident classification, settlement and restart. No host mapping
//! occurs during submission; a pending update must settle before reuse or save.

use crate::models::{
    ConvNeXtClassifier, ConvNeXtClassifierCheckpoint, ConvNeXtClassifierCheckpointSnapshot,
    ConvNeXtConfig, ResidentConvNeXtClassifier,
};
use crate::{DataLoader, DataLoaderCheckpoint, VisionDataset};
use serde::{Deserialize, Serialize};
use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice},
    resident_training::{parameters::ResidentParameterUpdate, TrainingError},
};
use st_core::runtime::{
    trainer_checkpoint::payload_sha256,
    trainer_optimizer::TRAINER_OPTIMIZER_MAX_SAFE_INTEGER,
    zspace_optimizer::{
        zspace_parameter_control_from_report, zspace_parameter_control_from_value,
        ZSpaceMetaOptimizerStepReport, ZSpaceParameterControl,
    },
    zspace_optimizer_feedback::ZSpaceOptimizerFeedbackConfig,
};
use st_kernel_contracts::sgd::SgdStep;
use st_nn::{
    loss::{CrossEntropyWithLogits, Loss},
    optim::{
        WarmupCosineScheduler, WarmupCosineSchedulerState, ZSpaceParameterControlReceipt,
        ZSpaceParameterControlState, ZSpaceParameterFeedbackState,
    },
    resident::InferenceError,
};
use std::sync::Arc;

mod clients;
pub use clients::ResidentVisionTrainerConfig;

const SCHEMA: &str = "spiraltorch.vision.training_checkpoint.v1";
const CONTROLLED_SCHEMA: &str = "spiraltorch.vision.training_checkpoint.v2";
const FEEDBACK_SCHEMA: &str = "spiraltorch.vision.training_checkpoint.v3";
const MAX_JSON_BYTES: usize = 545 * 1024 * 1024;
const MAX_CONTROL_JSON_BYTES: usize = 16 * 1024 * 1024;

fn invalid(message: &'static str) -> InferenceError {
    InferenceError::ModuleUpdate(message)
}

/// The shared Rust scheduler advances on accepted updates, not input batches.
/// Constant rates preserve the plain-SGD comparison, including a zero-rate probe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResidentLearningRate {
    Constant { rate: f32 },
    WarmupCosine { state: WarmupCosineSchedulerState },
}

impl ResidentLearningRate {
    fn validate(&self, accepted: u64) -> Result<(), InferenceError> {
        match self {
            Self::Constant { rate } => {
                SgdStep::new(*rate).map_err(|_| TrainingError::LearningRate)?;
            }
            Self::WarmupCosine { state } => {
                WarmupCosineScheduler::from_state(*state)?;
                if state.step != accepted.min(u64::from(u32::MAX)) as u32 {
                    return Err(invalid("scheduler clock differs from accepted updates"));
                }
            }
        }
        Ok(())
    }

    fn next(&self) -> Result<(f32, Self), InferenceError> {
        match self {
            Self::Constant { rate } => Ok((*rate, self.clone())),
            Self::WarmupCosine { state } => {
                let scheduler = WarmupCosineScheduler::from_state(*state)?;
                let (rate, next) = scheduler.preview_step();
                SgdStep::new(rate).map_err(|_| TrainingError::LearningRate)?;
                Ok((rate, Self::WarmupCosine { state: next }))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidentVisionTrainingState {
    epoch: u64,
    accepted_updates: u64,
    rejected_updates: u64,
    learning_rate: ResidentLearningRate,
    #[serde(
        default,
        skip_serializing_if = "ZSpaceParameterControlState::is_default"
    )]
    parameter_control: ZSpaceParameterControlState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    optimizer_feedback: Option<ZSpaceParameterFeedbackState>,
}

impl ResidentVisionTrainingState {
    /// Zero-based epoch containing the most recently submitted batch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn accepted_updates(&self) -> u64 {
        self.accepted_updates
    }
    pub fn rejected_updates(&self) -> u64 {
        self.rejected_updates
    }
    pub fn learning_rate(&self) -> &ResidentLearningRate {
        &self.learning_rate
    }

    pub fn parameter_control(&self) -> &ZSpaceParameterControlState {
        &self.parameter_control
    }

    pub fn optimizer_feedback(&self) -> Option<&ZSpaceParameterFeedbackState> {
        self.optimizer_feedback.as_ref()
    }

    fn schema(&self) -> &'static str {
        if self.optimizer_feedback.is_some() {
            FEEDBACK_SCHEMA
        } else if self.parameter_control.is_default() {
            SCHEMA
        } else {
            CONTROLLED_SCHEMA
        }
    }

    fn attempted(&self) -> Result<u64, InferenceError> {
        self.accepted_updates
            .checked_add(self.rejected_updates)
            .filter(|&n| n <= TRAINER_OPTIMIZER_MAX_SAFE_INTEGER)
            .ok_or_else(|| invalid("training clock exhausted"))
    }

    fn validate(&self, revision: u64, input: &DataLoaderCheckpoint) -> Result<(), InferenceError> {
        self.learning_rate.validate(self.accepted_updates)?;
        // A future rate may need retuning; never make an already settled state
        // unsavable. Submission checks the effective rate before consuming input.
        self.parameter_control.validate()?;
        if let Some(feedback) = &self.optimizer_feedback {
            feedback.validate()?;
            if feedback.state().control_step != self.attempted()?
                || feedback.state().observation_count != self.accepted_updates
            {
                return Err(invalid("feedback and settled trainer clocks differ"));
            }
        }
        let len = input.dataset_len();
        let batch = input.batch_size();
        if len == 0 || batch == 0 || !len.is_multiple_of(batch) {
            return Err(invalid("resident trainer requires nonempty full batches"));
        }
        let consumed = self
            .epoch
            .checked_mul((len / batch) as u64)
            .and_then(|n| n.checked_add((input.position() / batch) as u64));
        if self.attempted()? != revision || consumed != Some(revision) {
            return Err(invalid("model, input and trainer clocks differ"));
        }
        Ok(())
    }
}

/// Independently versioned payloads bound at one settled training boundary.
/// Hashes detect corruption/mixing; they do not authenticate a dataset or writer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisionTrainingCheckpoint {
    schema: String,
    model: ConvNeXtClassifierCheckpoint,
    input: DataLoaderCheckpoint,
    trainer: ResidentVisionTrainingState,
    model_sha256: String,
    input_sha256: String,
    trainer_sha256: String,
}

fn digest<T: Serialize>(value: &T) -> Result<String, InferenceError> {
    payload_sha256(value).map_err(|_| invalid("checkpoint encoding failed"))
}

impl VisionTrainingCheckpoint {
    fn new(
        model: ConvNeXtClassifierCheckpoint,
        input: DataLoaderCheckpoint,
        trainer: ResidentVisionTrainingState,
    ) -> Result<Self, InferenceError> {
        let value = Self {
            schema: trainer.schema().into(),
            model_sha256: digest(&model)?,
            input_sha256: digest(&input)?,
            trainer_sha256: digest(&trainer)?,
            model,
            input,
            trainer,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), InferenceError> {
        // Reuse component validation, including bounds, instead of approximating it.
        self.model.to_json()?;
        self.input.to_json()?;
        if self.schema != self.trainer.schema()
            || self.model.batch_size() != self.input.batch_size()
            || self.model_sha256 != digest(&self.model)?
            || self.input_sha256 != digest(&self.input)?
            || self.trainer_sha256 != digest(&self.trainer)?
        {
            return Err(invalid(
                "training checkpoint schema, batch or integrity mismatch",
            ));
        }
        self.trainer
            .validate(self.model.attempted_updates(), &self.input)
    }

    pub fn to_json(&self) -> Result<String, InferenceError> {
        self.validate()?;
        let json = serde_json::to_string(self)?;
        if json.len() > MAX_JSON_BYTES {
            return Err(invalid("training checkpoint size limit"));
        }
        Ok(json)
    }
    pub fn from_json(json: &str) -> Result<Self, InferenceError> {
        if json.len() > MAX_JSON_BYTES {
            return Err(invalid("training checkpoint size limit"));
        }
        let value: Self = serde_json::from_str(json)?;
        value.validate()?;
        Ok(value)
    }
    pub fn model(&self) -> &ConvNeXtClassifierCheckpoint {
        &self.model
    }
    pub fn input(&self) -> &DataLoaderCheckpoint {
        &self.input
    }
    pub fn trainer(&self) -> &ResidentVisionTrainingState {
        &self.trainer
    }
}

pub struct VisionTrainingCheckpointSnapshot {
    model: ConvNeXtClassifierCheckpointSnapshot,
    input: DataLoaderCheckpoint,
    trainer: ResidentVisionTrainingState,
}

impl VisionTrainingCheckpointSnapshot {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<VisionTrainingCheckpoint, InferenceError> {
        VisionTrainingCheckpoint::new(self.model.read()?, self.input, self.trainer)
    }
    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<VisionTrainingCheckpoint, InferenceError> {
        VisionTrainingCheckpoint::new(self.model.read_async().await?, self.input, self.trainer)
    }
}

/// Frozen observation handles; retaining them cannot advance or mutate training.
pub struct ResidentVisionSubmission {
    pub attempted_revision: u64,
    pub epoch: u64,
    pub learning_rate: f32,
    pub labels: Vec<Option<String>>,
    pub images: ResidentTensor,
    pub loss: ResidentTensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentVisionStepOutcome {
    pub attempted_revision: u64,
    pub accepted: bool,
}

struct PendingStep {
    update: ResidentParameterUpdate,
    next_rate: ResidentLearningRate,
    feedback: Option<PendingFeedback>,
}

struct PendingFeedback {
    state: ZSpaceParameterFeedbackState,
    loss: ResidentTensor,
    learning_rate: f32,
}

/// Owns one classifier, fixed-size dataset batches, RNGs and rate schedule.
/// Host preparation failures do not consume input. Successful update submission
/// consumes it even when numerical acceptance later fails. Unknown readback
/// failures retain the pending step and prohibit checkpointing/another submission.
pub struct ResidentVisionTrainer<D: VisionDataset> {
    model: ResidentConvNeXtClassifier,
    loader: DataLoader<D>,
    dataset_sha256: String,
    config: ConvNeXtConfig,
    classes: usize,
    state: ResidentVisionTrainingState,
    pending: Option<PendingStep>,
}

impl<D: VisionDataset> ResidentVisionTrainer<D> {
    pub fn new(
        model: &ConvNeXtClassifier,
        device: TensorDevice,
        loader: DataLoader<D>,
        dataset_sha256: &str,
        learning_rate: ResidentLearningRate,
    ) -> Result<Self, InferenceError> {
        let input = loader.checkpoint(dataset_sha256)?;
        let state = ResidentVisionTrainingState {
            epoch: 0,
            accepted_updates: 0,
            rejected_updates: 0,
            learning_rate,
            parameter_control: ZSpaceParameterControlState::default(),
            optimizer_feedback: None,
        };
        state.validate(0, &input)?;
        let resident = model.compile_resident_training(device, loader.batch_size)?;
        Ok(Self {
            model: resident,
            loader,
            dataset_sha256: dataset_sha256.into(),
            config: model.config().clone(),
            classes: model.num_classes(),
            state,
            pending: None,
        })
    }

    pub fn from_checkpoint(
        device: TensorDevice,
        mut loader: DataLoader<D>,
        dataset_sha256: &str,
        checkpoint: &VisionTrainingCheckpoint,
    ) -> Result<Self, InferenceError> {
        checkpoint.validate()?;
        loader.restore_checkpoint(dataset_sha256, &checkpoint.input)?;
        let model = checkpoint.model.restore_resident(device)?;
        Ok(Self {
            model,
            loader,
            dataset_sha256: dataset_sha256.into(),
            config: checkpoint.model.config().clone(),
            classes: checkpoint.model.num_classes(),
            state: checkpoint.trainer.clone(),
            pending: None,
        })
    }

    pub fn state(&self) -> &ResidentVisionTrainingState {
        &self.state
    }
    pub fn has_pending_update(&self) -> bool {
        self.pending.is_some()
    }

    /// Opt in before the first submission. Reconfiguration cannot silently erase
    /// the feedback history; continuation restores it from a bound checkpoint.
    pub fn enable_zspace_optimizer_feedback(
        &mut self,
        config: ZSpaceOptimizerFeedbackConfig,
    ) -> Result<(), InferenceError> {
        self.ensure_settled()?;
        if self.state.attempted()? != 0 || self.state.optimizer_feedback.is_some() {
            return Err(invalid("configure feedback only once, before training"));
        }
        self.state.optimizer_feedback = Some(ZSpaceParameterFeedbackState::new(config)?);
        Ok(())
    }
    fn ensure_settled(&self) -> Result<(), InferenceError> {
        if self.has_pending_update() {
            return Err(invalid("settle pending update before reuse or checkpoint"));
        }
        Ok(())
    }

    fn copy_input(&self) -> DataLoader<D> {
        DataLoader {
            dataset: Arc::clone(&self.loader.dataset),
            batch_size: self.loader.batch_size,
            order: self.loader.order.clone(),
            position: self.loader.position,
            shuffle: self.loader.shuffle,
            shuffle_rng: self.loader.shuffle_rng.clone(),
            pipeline: self.loader.pipeline.clone(),
        }
    }

    /// Validate/allocate a complete replacement before changing this owner.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &VisionTrainingCheckpoint,
    ) -> Result<(), InferenceError> {
        self.ensure_settled()?;
        if self.config != *checkpoint.model.config()
            || self.classes != checkpoint.model.num_classes()
        {
            return Err(invalid("training checkpoint architecture differs"));
        }
        let restored = Self::from_checkpoint(
            self.model.tensor_device().clone(),
            self.copy_input(),
            &self.dataset_sha256,
            checkpoint,
        )?;
        *self = restored;
        Ok(())
    }

    pub fn submit_next(&mut self) -> Result<ResidentVisionSubmission, InferenceError> {
        self.ensure_settled()?;
        if self.state.attempted()? == TRAINER_OPTIMIZER_MAX_SAFE_INTEGER {
            return Err(invalid("training clock exhausted"));
        }
        let (nominal_rate, next_rate) = self.state.learning_rate.next()?;
        let (next_feedback, rate) = match &self.state.optimizer_feedback {
            Some(feedback) => {
                let (next, rate) = feedback.preview(
                    nominal_rate,
                    self.state.parameter_control.absolute_learning_rate_scale(),
                )?;
                (Some(next), rate)
            }
            None => (
                None,
                self.state
                    .parameter_control
                    .effective_learning_rate(nominal_rate)?,
            ),
        };
        // Copy the order only at epoch boundaries, not on every training step.
        let mut next_epoch = if self.loader.position == self.loader.order.len() {
            let mut next = self.copy_input();
            next.reset();
            Some(next)
        } else {
            None
        };
        let input = next_epoch.as_ref().unwrap_or(&self.loader);
        let prepared = input
            .prepare_resident_batch(self.model.tensor_device())?
            .ok_or_else(|| invalid("resident trainer has no batch"))?;
        let targets = prepared.batch.upload_targets()?;
        let forward = self.model.forward(&prepared.batch.images)?;
        let loss =
            CrossEntropyWithLogits::default().evaluate_resident(forward.prediction(), &targets)?;
        let gradients = self.model.backward(&forward, loss.prediction_gradient())?;
        let update = self.model.sgd(&gradients, rate)?;
        // Nothing fallible follows the point where the parameter owner advances.
        if let Some(loader) = next_epoch.take() {
            self.loader = loader;
            self.state.epoch += 1;
        }
        let batch = self.loader.commit_resident_batch(prepared);
        let submission = ResidentVisionSubmission {
            attempted_revision: update.revision(),
            epoch: self.state.epoch,
            learning_rate: rate,
            labels: batch.labels,
            images: batch.images,
            loss: loss.value().clone(),
        };
        self.pending = Some(PendingStep {
            update,
            next_rate,
            feedback: next_feedback.map(|state| PendingFeedback {
                state,
                loss: loss.value().clone(),
                learning_rate: rate,
            }),
        });
        Ok(submission)
    }

    /// Commit a parameter-side control at a settled boundary. This does not
    /// advance model/input/schedule clocks or own the producer's latent state.
    pub fn apply_zspace_parameter_control(
        &mut self,
        control: &ZSpaceParameterControl,
    ) -> Result<ZSpaceParameterControlReceipt, InferenceError> {
        self.ensure_settled()?;
        let (next, receipt) = self.state.parameter_control.preview(control)?;
        let nominal_rate = self.state.learning_rate.next()?.0;
        if let Some(feedback) = &self.state.optimizer_feedback {
            feedback.preview(nominal_rate, next.absolute_learning_rate_scale())?;
        } else {
            next.effective_learning_rate(nominal_rate)?;
        }
        self.state.parameter_control = next;
        receipt.emit("resident_vision_trainer");
        Ok(receipt)
    }

    pub fn apply_zspace_meta_optimizer_report(
        &mut self,
        report: &ZSpaceMetaOptimizerStepReport,
    ) -> Result<ZSpaceParameterControlReceipt, InferenceError> {
        self.ensure_settled()?;
        let control = zspace_parameter_control_from_report(report)
            .map_err(|e| st_tensor::TensorError::Generic(e.to_string()))?;
        self.apply_zspace_parameter_control(&control)
    }

    /// Both bindings pass the complete report to this shared validation path.
    pub fn apply_zspace_meta_optimizer_report_json(
        &mut self,
        json: &str,
    ) -> Result<ZSpaceParameterControlReceipt, InferenceError> {
        self.ensure_settled()?;
        if json.len() > MAX_CONTROL_JSON_BYTES {
            return Err(invalid("Z-space parameter control report size limit"));
        }
        let control = zspace_parameter_control_from_value(serde_json::from_str(json)?)
            .map_err(|e| st_tensor::TensorError::Generic(e.to_string()))?;
        self.apply_zspace_parameter_control(&control)
    }

    fn finish_settlement(
        &mut self,
        result: Result<u64, TrainingError>,
        feedback_loss: Option<f32>,
    ) -> Result<ResidentVisionStepOutcome, InferenceError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| invalid("no pending update"))?;
        let revision = pending.update.revision();
        let accepted = match result {
            Ok(observed) if observed == revision => true,
            Ok(_) => return Err(invalid("update receipt revision differs")),
            Err(TrainingError::Rejected { .. }) => false,
            Err(error) => return Err(error.into()),
        };
        // Prepare every fallible feedback transition before committing any host
        // state. Readback errors/cancellation retain the pending update for retry.
        let feedback = match &pending.feedback {
            Some(feedback) if accepted => Some(feedback.state.observe(
                feedback_loss.ok_or_else(|| invalid("missing accepted-step feedback loss"))?,
                feedback.learning_rate,
                self.state.epoch,
            )?),
            Some(feedback) => Some(feedback.state.clone()),
            None => None,
        };
        if accepted {
            self.state.accepted_updates += 1;
            self.state.learning_rate = pending.next_rate.clone();
        } else {
            self.state.rejected_updates += 1;
        }
        self.state.optimizer_feedback = feedback;
        self.pending = None;
        Ok(ResidentVisionStepOutcome {
            attempted_revision: revision,
            accepted,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn settle(&mut self) -> Result<ResidentVisionStepOutcome, InferenceError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| invalid("no pending update"))?;
        let (result, loss) = match &pending.feedback {
            Some(feedback) => match pending.update.snapshot_with_scalar(&feedback.loss)?.read() {
                Ok((revision, loss)) => (Ok(revision), Some(loss)),
                Err(error) => (Err(error), None),
            },
            None => (pending.update.snapshot()?.read(), None),
        };
        self.finish_settlement(result, loss)
    }

    /// Cancellation leaves the pending update intact; a caller may settle again.
    #[cfg(target_arch = "wasm32")]
    pub async fn settle_async(&mut self) -> Result<ResidentVisionStepOutcome, InferenceError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| invalid("no pending update"))?;
        let (result, loss) = match &pending.feedback {
            Some(feedback) => match pending
                .update
                .snapshot_with_scalar(&feedback.loss)?
                .read_async()
                .await
            {
                Ok((revision, loss)) => (Ok(revision), Some(loss)),
                Err(error) => (Err(error), None),
            },
            None => (pending.update.snapshot()?.read_async().await, None),
        };
        self.finish_settlement(result, loss)
    }

    pub fn checkpoint_snapshot(&self) -> Result<VisionTrainingCheckpointSnapshot, InferenceError> {
        self.ensure_settled()?;
        let input = self.loader.checkpoint(&self.dataset_sha256)?;
        self.state
            .validate(self.model.parameter_snapshot().revision(), &input)?;
        Ok(VisionTrainingCheckpointSnapshot {
            model: self.model.checkpoint_snapshot()?,
            input,
            trainer: self.state.clone(),
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
