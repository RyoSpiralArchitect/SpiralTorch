// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

use crate::dataset::DataLoader;
use crate::loss::{
    Loss, SoftmaxCrossEntropy, SparseClassificationDelta, SparseClassificationMetrics,
};
use crate::module::{
    fingerprint_bool, fingerprint_bytes, fingerprint_f32, fingerprint_usize, Module,
    ParameterMovementReport, ParameterTrainingFingerprint, StateFingerprint,
};
use crate::plan::RankPlanner;
use crate::roundtable::{
    simulate_proposal_locally, BlackcatModerator, DistConfig, GlobalProposal, HeurOpLog,
    MetaConductor, ModeratorMinutes, OutcomeBand, RoundtableNode,
};
use crate::schedule::{BandEnergy, GradientBands, RoundtableConfig, RoundtableSchedule};
use crate::{PureResult, Tensor, TensorError};
use st_core::backend::device_caps::DeviceCaps;
use st_core::backend::unison_heuristics::RankKind;
#[cfg(feature = "collapse")]
use st_core::engine::collapse_drive::{CollapseConfig, CollapseDrive, DriveCmd};
use st_core::runtime::autopilot::Autopilot;
use st_core::runtime::blackcat::{BlackCatRuntime, StepMetrics};
#[cfg(any(feature = "psi", feature = "psychoid"))]
use st_core::telemetry::hub;
#[cfg(feature = "psi")]
use st_core::telemetry::psi::{PsiConfig, PsiInput, PsiMeter, PsiReading};
#[cfg(feature = "psychoid")]
use st_core::telemetry::psychoid::{PsychoidConfig, PsychoidEvent, PsychoidMeter, PsychoidReading};
use st_tensor::pure::topos::OpenCartesianTopos;
use std::collections::HashMap;
use std::time::{Duration, Instant};

const FNV64_OFFSET: u64 = 0xcbf29ce484222325;

/// High-level orchestrator that keeps hypergrad, SpiralK, and module updates aligned.
pub struct ModuleTrainer {
    planner: RankPlanner,
    curvature: f32,
    hyper_learning_rate: f32,
    fallback_learning_rate: f32,
    blackcat: Option<BlackCatRuntime>,
    blackcat_moderator: Option<BlackcatModerator>,
    autopilot: Option<Autopilot>,
    band_weight_fn: Option<BandWeightFn>,
    max_grad_norm: Option<f32>,
    gradient_accumulation_steps: usize,
    injector_enabled: bool,
    distribution: Option<RoundtableNode>,
    meta_conductor: Option<MetaConductor>,
    heur_log: HeurOpLog,
    #[cfg(feature = "psi")]
    psi: Option<PsiMeter>,
    #[cfg(feature = "psychoid")]
    psychoid: Option<PsychoidMeter>,
    #[cfg(feature = "psychoid")]
    psychoid_log: bool,
    #[cfg(feature = "collapse")]
    collapse: Option<CollapseDrive>,
}

impl core::fmt::Debug for ModuleTrainer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ModuleTrainer(curv={},lr_h={},lr_f={})",
            self.curvature, self.hyper_learning_rate, self.fallback_learning_rate
        )
    }
}

/// Function pointer used to convert band energy into Above/Here/Beneath weights.
pub type BandWeightFn = fn(BandEnergy) -> (f32, f32, f32);

impl ModuleTrainer {
    /// Creates a new trainer with the provided device capabilities and learning rates.
    pub fn new(
        caps: DeviceCaps,
        curvature: f32,
        hyper_learning_rate: f32,
        fallback_learning_rate: f32,
    ) -> Self {
        Self {
            planner: RankPlanner::new(caps),
            curvature,
            hyper_learning_rate,
            fallback_learning_rate,
            blackcat: None,
            blackcat_moderator: None,
            autopilot: None,
            band_weight_fn: None,
            max_grad_norm: None,
            gradient_accumulation_steps: 1,
            injector_enabled: false,
            distribution: None,
            meta_conductor: None,
            heur_log: HeurOpLog::default(),
            #[cfg(feature = "psi")]
            psi: None,
            #[cfg(feature = "psychoid")]
            psychoid: None,
            #[cfg(feature = "psychoid")]
            psychoid_log: false,
            #[cfg(feature = "collapse")]
            collapse: None,
        }
    }

    /// Attaches the BlackCat runtime so contextual rewards update after each step.
    pub fn with_blackcat(mut self, runtime: BlackCatRuntime) -> Self {
        self.blackcat = Some(runtime);
        self
    }

    /// Installs a Blackcat moderator that seats between local and distributed consensus.
    pub fn install_blackcat_moderator(&mut self, threshold: f32, participants: usize) {
        self.blackcat_moderator = Some(BlackcatModerator::with_default_runtime(
            threshold.max(0.1),
            participants.max(1),
        ));
        self.meta_conductor = None;
    }

    /// Installs a moderator with a custom runtime configuration.
    pub fn install_blackcat_moderator_with_runtime(
        &mut self,
        runtime: BlackCatRuntime,
        threshold: f32,
        participants: usize,
    ) {
        self.blackcat_moderator = Some(BlackcatModerator::new(
            runtime,
            threshold.max(0.1),
            participants.max(1),
        ));
        self.meta_conductor = None;
    }

    /// Clears any configured moderator.
    pub fn clear_blackcat_moderator(&mut self) {
        self.blackcat_moderator = None;
    }

    /// Attaches an Autopilot runtime for contextual kernel selection.
    pub fn with_autopilot(mut self, autopilot: Autopilot) -> Self {
        self.autopilot = Some(autopilot);
        self
    }

    /// Enables per-band reweighting of the loss/gradient.
    pub fn set_band_weights(&mut self, weight_fn: BandWeightFn) {
        self.band_weight_fn = Some(weight_fn);
    }

    /// Connects the trainer to a distributed roundtable node.
    pub fn configure_distribution(&mut self, config: DistConfig) {
        self.distribution = Some(RoundtableNode::new(config));
    }

    /// Removes any configured distribution node.
    pub fn clear_distribution(&mut self) {
        if let Some(node) = self.distribution.as_mut() {
            node.drain();
        }
        self.distribution = None;
    }

    /// Installs a meta-layer conductor so this trainer can aggregate remote summaries.
    pub fn install_meta_conductor(&mut self, threshold: f32, participants: usize) {
        self.blackcat_moderator = None;
        self.meta_conductor = Some(MetaConductor::new(threshold.max(0.1), participants.max(1)));
    }

    /// Returns the current heuristics op-log.
    pub fn heuristics_log(&self) -> &HeurOpLog {
        &self.heur_log
    }

    /// Returns the latest moderator minutes captured by Blackcat.
    pub fn blackcat_minutes(&self) -> Vec<ModeratorMinutes> {
        self.blackcat_moderator
            .as_ref()
            .map(|m| m.minutes().to_vec())
            .unwrap_or_default()
    }

    /// Clears any registered band weighting rule.
    pub fn clear_band_weights(&mut self) {
        self.band_weight_fn = None;
    }

    /// Clips accumulated gradients to the provided global L2 norm before each step.
    ///
    /// Passing `None` disables clipping. The limit must be finite and strictly
    /// positive when present.
    pub fn set_max_grad_norm(&mut self, max_norm: Option<f32>) -> PureResult<()> {
        if let Some(limit) = max_norm {
            if limit <= 0.0 || !limit.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "max_grad_norm",
                    value: limit,
                });
            }
        }
        self.max_grad_norm = max_norm;
        Ok(())
    }

    /// Returns the configured global gradient clipping threshold.
    pub fn max_grad_norm(&self) -> Option<f32> {
        self.max_grad_norm
    }

    /// Sets how many mini-batches should accumulate before one optimizer step.
    ///
    /// A value of `1` preserves the classic one-step-per-batch behavior.
    pub fn set_gradient_accumulation_steps(&mut self, steps: usize) -> PureResult<()> {
        if steps == 0 {
            return Err(TensorError::IoError {
                message: "gradient_accumulation_steps must be positive".to_string(),
            });
        }
        self.gradient_accumulation_steps = steps;
        Ok(())
    }

    /// Returns the configured mini-batches per optimizer step.
    pub fn gradient_accumulation_steps(&self) -> usize {
        self.gradient_accumulation_steps
    }

    /// Captures the trainer-side state that affects optimizer/resume behavior.
    pub fn state_snapshot(&self) -> TrainerStateSnapshot {
        TrainerStateSnapshot {
            curvature: self.curvature,
            hyper_learning_rate: self.hyper_learning_rate,
            fallback_learning_rate: self.fallback_learning_rate,
            max_grad_norm: self.max_grad_norm,
            gradient_accumulation_steps: self.gradient_accumulation_steps,
            injector_enabled: self.injector_enabled,
            band_weights_enabled: self.band_weight_fn.is_some(),
            blackcat_enabled: self.blackcat.is_some(),
            blackcat_moderator_enabled: self.blackcat_moderator.is_some(),
            autopilot_enabled: self.autopilot.is_some(),
            distribution_enabled: self.distribution.is_some(),
            meta_conductor_enabled: self.meta_conductor.is_some(),
        }
    }

    /// Computes a stable digest of trainer configuration relevant to resume.
    pub fn state_fingerprint(&self) -> TrainerStateFingerprint {
        let snapshot = self.state_snapshot();
        let mut hash = FNV64_OFFSET;
        fingerprint_f32(&mut hash, snapshot.curvature);
        fingerprint_f32(&mut hash, snapshot.hyper_learning_rate);
        fingerprint_f32(&mut hash, snapshot.fallback_learning_rate);
        match snapshot.max_grad_norm {
            Some(value) => {
                fingerprint_bool(&mut hash, true);
                fingerprint_f32(&mut hash, value);
            }
            None => fingerprint_bool(&mut hash, false),
        }
        fingerprint_usize(&mut hash, snapshot.gradient_accumulation_steps);
        fingerprint_bool(&mut hash, snapshot.injector_enabled);
        fingerprint_bool(&mut hash, snapshot.band_weights_enabled);
        fingerprint_bool(&mut hash, snapshot.blackcat_enabled);
        fingerprint_bool(&mut hash, snapshot.blackcat_moderator_enabled);
        fingerprint_bool(&mut hash, snapshot.autopilot_enabled);
        fingerprint_bool(&mut hash, snapshot.distribution_enabled);
        fingerprint_bool(&mut hash, snapshot.meta_conductor_enabled);
        TrainerStateFingerprint {
            hash: format!("{hash:016x}"),
            gradient_accumulation_steps: snapshot.gradient_accumulation_steps,
            runtime_hooks: snapshot.runtime_hooks(),
        }
    }

    /// Captures the value, parameter-training, and trainer digests needed for
    /// an auditable fine-tuning resume boundary.
    pub fn resume_fingerprint<M: Module>(
        &self,
        module: &M,
    ) -> PureResult<TrainingResumeFingerprint> {
        let trainer = self.state_fingerprint();
        let parameters = module.state_fingerprint()?;
        let parameter_training = module.training_state_fingerprint()?;
        Ok(TrainingResumeFingerprint::from_parts(
            trainer,
            parameters,
            parameter_training,
        ))
    }

    /// Compares the current trainer/module pair with a previously captured
    /// resume fingerprint.
    pub fn audit_resume_fingerprint<M: Module>(
        &self,
        module: &M,
        expected: &TrainingResumeFingerprint,
    ) -> PureResult<TrainingResumeAudit> {
        let actual = self.resume_fingerprint(module)?;
        let trainer_matched = actual.trainer == expected.trainer;
        let parameters_matched = actual.parameters == expected.parameters;
        let parameter_training_matched = actual.parameter_training == expected.parameter_training;
        let matched = actual.hash == expected.hash
            && trainer_matched
            && parameters_matched
            && parameter_training_matched;
        Ok(TrainingResumeAudit {
            expected: expected.clone(),
            actual,
            trainer_matched,
            parameters_matched,
            parameter_training_matched,
            matched,
        })
    }

    /// Enables or disables the adaptive injector heuristics.
    pub fn enable_injector(&mut self, on: bool) {
        self.injector_enabled = on;
    }

    /// Returns the underlying planner.
    pub fn planner(&self) -> &RankPlanner {
        &self.planner
    }

    /// Produces a roundtable schedule for the provided output dimensions.
    pub fn roundtable(&self, rows: u32, cols: u32, config: RoundtableConfig) -> RoundtableSchedule {
        RoundtableSchedule::new(&self.planner, rows, cols, config)
    }

    /// Returns the fallback Euclidean learning rate.
    pub fn fallback_learning_rate(&self) -> f32 {
        self.fallback_learning_rate
    }

    /// Returns the curvature used for hypergrad preparation.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the learning rate used for hypergrad updates.
    pub fn hyper_learning_rate(&self) -> f32 {
        self.hyper_learning_rate
    }

    /// Attaches hypergrad tapes to all parameters of the provided module.
    pub fn prepare<M: Module>(&self, module: &mut M) -> PureResult<()> {
        module.attach_hypergrad(self.curvature, self.hyper_learning_rate)
    }

    /// Attaches hypergrad tapes with an explicit topos shared across parameters.
    pub fn prepare_with_topos<M: Module>(
        &self,
        module: &mut M,
        topos: OpenCartesianTopos,
    ) -> PureResult<()> {
        module.attach_hypergrad_with_topos(self.curvature, self.hyper_learning_rate, topos)
    }

    /// Clears accumulated gradients or hypergrad buffers.
    pub fn zero<M: Module>(&self, module: &mut M) -> PureResult<()> {
        module.zero_accumulators()
    }

    /// Applies the parameter updates using either the hypergrad tape or the fallback rate.
    pub fn step<M: Module>(&self, module: &mut M) -> PureResult<()> {
        module.apply_step(self.fallback_learning_rate)
    }

    /// Runs a full epoch over the provided iterator of `(input, target)` pairs.
    pub fn train_epoch<M, L, I>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        batches: I,
        schedule: &RoundtableSchedule,
    ) -> PureResult<EpochStats>
    where
        M: Module,
        L: Loss,
        I: IntoIterator,
        I::Item: IntoBatch,
    {
        self.zero(module)?;
        #[cfg(feature = "psi")]
        self.bootstrap_psi(schedule);
        #[cfg(feature = "psychoid")]
        self.bootstrap_psychoid(schedule);
        #[cfg(feature = "collapse")]
        self.bootstrap_collapse(schedule);
        let accumulation_steps = self.gradient_accumulation_steps.max(1);
        let accumulation_scale = 1.0 / accumulation_steps as f32;
        let mut accumulated_batches = 0usize;
        let mut total_loss = 0.0f32;
        let mut total_row_weighted_loss = 0.0f32;
        let mut rows = 0usize;
        let mut batches_seen = 0usize;
        let mut optimizer_steps = 0usize;
        for batch in batches.into_iter() {
            let (input, target) = batch.into_batch()?;
            let step_start = Instant::now();
            if let Some(rt) = self.blackcat.as_mut() {
                rt.begin_step();
            }
            let device_load = self.estimate_device_load();
            if let Some(ap) = self.autopilot.as_mut() {
                let (rows, cols) = input.shape();
                let depth = schedule.above().k + schedule.here().k + schedule.beneath().k;
                let context = ap.build_context(rows as u32, cols as u32, depth, device_load, &[]);
                let _ = ap.suggest(context);
            }
            let prediction = module.forward(&input)?;
            Self::ensure_tensor_finite("train_prediction", &prediction)?;
            let loss_value = loss.forward(&prediction, &target)?;
            Self::ensure_tensor_finite("train_loss", &loss_value)?;
            let row_count = loss.reduction_rows(&prediction, &target)?;
            let step_loss = loss_value.data().iter().copied().sum::<f32>();
            Self::ensure_finite("train_step_loss", step_loss)?;
            let grad_output = loss.backward(&prediction, &target)?;
            Self::ensure_tensor_finite("train_grad_output", &grad_output)?;
            let mut band_energy = schedule.band_energy(&grad_output)?;
            if let Some(rt) = self.blackcat.as_ref() {
                band_energy.drift = rt.frac_penalty() as f32;
            }
            let mut bands: GradientBands = schedule.split(&grad_output)?;
            let weights = self
                .band_weight_fn
                .map(|f| f(band_energy))
                .unwrap_or((1.0, 1.0, 1.0));
            bands.scale_inplace(
                weights.0 * accumulation_scale,
                weights.1 * accumulation_scale,
                weights.2 * accumulation_scale,
            );
            let weighted_loss = if self.band_weight_fn.is_some() {
                let effective = weights.1 + 0.5 * (weights.0 + weights.2);
                step_loss * effective.max(0.0)
            } else {
                step_loss
            };
            Self::ensure_finite("train_weighted_loss", weighted_loss)?;
            total_loss += weighted_loss;
            total_row_weighted_loss += weighted_loss * row_count as f32;
            rows += row_count;
            let mut extra = HashMap::new();
            let _ = module.backward_bands(&input, &bands)?;
            accumulated_batches += 1;
            let should_step = accumulated_batches >= accumulation_steps;
            #[cfg(feature = "psychoid")]
            let mut psychoid_events = 0usize;
            #[cfg(feature = "psi")]
            let mut psi_snapshot: Option<PsiReading> = None;
            #[cfg(feature = "psi")]
            {
                if let Some(meter) = self.psi.as_mut() {
                    let grad_l2 = Self::collect_grad_l2(module)?;
                    let act_drift = module.psi_probe().unwrap_or(0.0);
                    let input_snapshot = PsiInput {
                        loss: step_loss.abs(),
                        grad_l2,
                        update_ratio: 0.0,
                        act_drift,
                        attn_entropy: 0.0,
                        band_energy: band_energy.l1() + band_energy.drift.abs(),
                    };
                    let (reading, events) = meter.update(&input_snapshot);
                    psi_snapshot = Some(reading.clone());
                    hub::set_last_psi(&reading);
                    extra.insert("psi_total".to_string(), reading.total as f64);
                    for (component, value) in reading.breakdown.iter() {
                        let key = format!("psi_{}", component);
                        extra.insert(key, *value as f64);
                    }
                    extra.insert("psi_loss".to_string(), input_snapshot.loss as f64);
                    extra.insert("psi_grad_l2".to_string(), input_snapshot.grad_l2 as f64);
                    extra.insert(
                        "psi_update_ratio".to_string(),
                        input_snapshot.update_ratio as f64,
                    );
                    extra.insert("psi_act_drift".to_string(), input_snapshot.act_drift as f64);
                    extra.insert(
                        "psi_band_energy".to_string(),
                        input_snapshot.band_energy as f64,
                    );
                    extra.insert("psi_events".to_string(), events.len() as f64);
                }
            }
            #[cfg(feature = "psychoid")]
            {
                if let Some(meter) = self.psychoid.as_mut() {
                    if let Some(sample) = module.psychoid_sample(&input, &prediction) {
                        if let Some((reading, events)) = meter.observe(sample) {
                            hub::set_last_psychoid(&reading);
                            if self.psychoid_log {
                                Self::log_psychoid(&reading, &events);
                            }
                            extra.insert("psychoid_cti".to_string(), reading.cti as f64);
                            for (metric, value) in reading.raw.iter() {
                                extra.insert(
                                    format!("psychoid_raw_{}", metric.to_lowercase()),
                                    *value as f64,
                                );
                            }
                            for (metric, value) in reading.z_scores.iter() {
                                extra.insert(
                                    format!("psychoid_z_{}", metric.to_lowercase()),
                                    *value as f64,
                                );
                            }
                            psychoid_events = events.len();
                        }
                    }
                }
            }
            if let Some(node) = self.distribution.as_mut() {
                let outcome = OutcomeBand::from_weights(
                    band_energy.above,
                    band_energy.here,
                    band_energy.beneath,
                );
                let plan = match outcome {
                    OutcomeBand::Above => schedule.above(),
                    OutcomeBand::Here => schedule.here(),
                    OutcomeBand::Beneath => schedule.beneath(),
                };
                let signature = plan_signature(plan, outcome);
                let script_hint = plan.choice.to_unison_script(plan.kind).replace('\n', "; ");
                let psi_total = {
                    #[cfg(feature = "psi")]
                    {
                        psi_snapshot.as_ref().map(|reading| reading.total.max(0.0))
                    }
                    #[cfg(not(feature = "psi"))]
                    {
                        None
                    }
                };
                if let Some(summary) = node.record_decision(
                    signature,
                    script_hint,
                    plan.kind,
                    outcome,
                    (1.0 / (1.0 + weighted_loss.abs())).clamp(0.0, 1.0),
                    psi_total,
                    (band_energy.above, band_energy.here, band_energy.beneath),
                    band_energy.drift,
                ) {
                    if let Some(moderator) = self.blackcat_moderator.as_mut() {
                        let outcome = moderator.ingest(summary.clone());
                        if let Some(proposal) = outcome.proposal {
                            let (accepted, preview) =
                                simulate_proposal_locally(&proposal, &mut self.heur_log);
                            if accepted {
                                self.apply_proposal(&proposal, preview)?;
                            }
                        }
                    } else if let Some(conductor) = self.meta_conductor.as_mut() {
                        if let Some(proposal) = conductor.ingest(summary.clone()) {
                            let (accepted, preview) =
                                simulate_proposal_locally(&proposal, &mut self.heur_log);
                            if accepted {
                                self.apply_proposal(&proposal, preview)?;
                            }
                        }
                    }
                }
            }
            let mut applied_optimizer_step = false;
            if should_step {
                if let Some(limit) = self.max_grad_norm {
                    self.clip_grad_global_norm(module, limit)?;
                }
                #[cfg(feature = "collapse")]
                if let (Some(driver), Some(reading)) =
                    (self.collapse.as_mut(), psi_snapshot.as_ref())
                {
                    match driver.update(reading) {
                        DriveCmd::Collapse {
                            grad_scale,
                            max_norm,
                            lr_decay,
                        } => {
                            if grad_scale < 0.999 {
                                self.apply_grad_scale(module, grad_scale)?;
                            }
                            if let Some(limit) = max_norm {
                                self.clip_grad_global_norm(module, limit)?;
                            }
                            if let Some(decay) = lr_decay {
                                let factor = (1.0 - decay).clamp(0.1, 1.0);
                                self.optimizer_mul_lr(module, factor)?;
                            }
                        }
                        DriveCmd::Bloom { lr_mul } => {
                            if lr_mul > 1.0 {
                                self.optimizer_mul_lr(module, lr_mul)?;
                            }
                        }
                        DriveCmd::None => {}
                    }
                }
                self.step(module)?;
                self.zero(module)?;
                optimizer_steps += 1;
                accumulated_batches = 0;
                applied_optimizer_step = true;
            }
            batches_seen += 1;

            let elapsed_ms = if let Some(rt) = self.blackcat.as_ref() {
                rt.elapsed_since_begin()
                    .unwrap_or_else(|| Duration::from_secs_f64(0.0))
                    .as_secs_f64()
                    * 1_000.0
            } else {
                step_start.elapsed().as_secs_f64() * 1_000.0
            };
            extra.insert("band_above".to_string(), band_energy.above as f64);
            extra.insert("band_here".to_string(), band_energy.here as f64);
            extra.insert("band_beneath".to_string(), band_energy.beneath as f64);
            extra.insert("band_drift".to_string(), band_energy.drift as f64);
            extra.insert("step_loss".to_string(), step_loss as f64);
            extra.insert("loss_weighted".to_string(), weighted_loss as f64);
            extra.insert(
                "gradient_accumulation_steps".to_string(),
                accumulation_steps as f64,
            );
            extra.insert(
                "gradient_accumulation_position".to_string(),
                accumulated_batches as f64,
            );
            extra.insert(
                "optimizer_step".to_string(),
                if applied_optimizer_step { 1.0 } else { 0.0 },
            );
            #[cfg(feature = "psychoid")]
            {
                extra.insert("psychoid_events".to_string(), psychoid_events as f64);
            }
            let metrics = StepMetrics {
                step_time_ms: elapsed_ms,
                mem_peak_mb: 0.0,
                retry_rate: 0.0,
                extra,
            };
            if let Some(ap) = self.autopilot.as_mut() {
                ap.report(&metrics);
            }
            if let Some(rt) = self.blackcat.as_mut() {
                let reward = rt.post_step(&metrics);
                if reward > 0.0 {
                    let plan = schedule.above();
                    let script = plan
                        .choice
                        .to_unison_script(RankKind::TopK)
                        .replace('\n', "; ");
                    let _ = rt.try_adopt_soft(&script, 1, 1, 0.5);
                }
            }
        }
        if accumulated_batches > 0 {
            if accumulated_batches < accumulation_steps {
                let partial_scale = accumulation_steps as f32 / accumulated_batches as f32;
                self.apply_grad_scale(module, partial_scale)?;
            }
            if let Some(limit) = self.max_grad_norm {
                self.clip_grad_global_norm(module, limit)?;
            }
            self.step(module)?;
            self.zero(module)?;
            optimizer_steps += 1;
        }
        Ok(EpochStats {
            batches: batches_seen,
            optimizer_steps,
            rows,
            total_loss,
            total_row_weighted_loss,
            average_loss: if batches_seen == 0 {
                0.0
            } else {
                total_loss / batches_seen as f32
            },
            average_loss_per_row: if rows == 0 {
                0.0
            } else {
                total_row_weighted_loss / rows as f32
            },
        })
    }

    /// Evaluates a module over batches without mutating parameters or accumulators.
    ///
    /// The returned [`EpochStats`] uses the same batch and row-weighted loss
    /// accounting as [`Self::train_epoch`], making validation curves directly
    /// comparable to training histories.
    pub fn evaluate_epoch<M, L, I>(
        &self,
        module: &M,
        loss: &mut L,
        batches: I,
    ) -> PureResult<EpochStats>
    where
        M: Module,
        L: Loss,
        I: IntoIterator,
        I::Item: IntoBatch,
    {
        let mut total_loss = 0.0f32;
        let mut total_row_weighted_loss = 0.0f32;
        let mut rows = 0usize;
        let mut steps = 0usize;
        for batch in batches.into_iter() {
            let (input, target) = batch.into_batch()?;
            let prediction = module.forward(&input)?;
            Self::ensure_tensor_finite("eval_prediction", &prediction)?;
            let loss_value = loss.forward(&prediction, &target)?;
            Self::ensure_tensor_finite("eval_loss", &loss_value)?;
            let row_count = loss.reduction_rows(&prediction, &target)?;
            let step_loss = loss_value.data().iter().copied().sum::<f32>();
            Self::ensure_finite("eval_step_loss", step_loss)?;
            total_loss += step_loss;
            total_row_weighted_loss += step_loss * row_count as f32;
            rows += row_count;
            steps += 1;
        }
        Ok(EpochStats {
            batches: steps,
            optimizer_steps: 0,
            rows,
            total_loss,
            total_row_weighted_loss,
            average_loss: if steps == 0 {
                0.0
            } else {
                total_loss / steps as f32
            },
            average_loss_per_row: if rows == 0 {
                0.0
            } else {
                total_row_weighted_loss / rows as f32
            },
        })
    }

    /// Evaluates sparse class-index targets with active-row accuracy and perplexity.
    ///
    /// This mirrors [`Self::evaluate_epoch`] but keeps classification diagnostics
    /// attached to the same row mask used by [`SoftmaxCrossEntropy`].
    pub fn evaluate_sparse_classification_epoch<M, I>(
        &self,
        module: &M,
        loss: &SoftmaxCrossEntropy,
        batches: I,
    ) -> PureResult<SparseClassificationMetrics>
    where
        M: Module,
        I: IntoIterator,
        I::Item: IntoBatch,
    {
        let mut total_loss = 0.0f32;
        let mut active_rows = 0usize;
        let mut correct = 0usize;
        for batch in batches.into_iter() {
            let (input, target) = batch.into_batch()?;
            let prediction = module.forward(&input)?;
            let metrics = loss.sparse_metrics(&prediction, &target)?;
            Self::ensure_finite("sparse_eval_mean_loss", metrics.mean_loss)?;
            Self::ensure_finite("sparse_eval_accuracy", metrics.accuracy)?;
            Self::ensure_finite("sparse_eval_perplexity", metrics.perplexity)?;
            total_loss += metrics.mean_loss * metrics.active_rows as f32;
            Self::ensure_finite("sparse_eval_total_loss", total_loss)?;
            active_rows += metrics.active_rows;
            correct += metrics.correct;
        }
        SparseClassificationMetrics::from_totals(active_rows, correct, total_loss)
    }

    /// Runs multiple epochs over a cloneable [`DataLoader`] and returns per-epoch stats.
    ///
    /// This is intentionally a thin wrapper over [`Self::train_epoch`]: the
    /// loader is cloned for each epoch so deterministic shuffle/batch/prefetch
    /// settings remain stable while the caller receives a complete loss
    /// history for fine-tuning diagnostics.
    pub fn train_epochs<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
    ) -> PureResult<Vec<EpochStats>>
    where
        M: Module,
        L: Loss,
    {
        let mut history = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            history.push(self.train_epoch(module, loss, loader.clone(), schedule)?);
        }
        Ok(history)
    }

    /// Runs multiple epochs and captures both best-epoch and final state dicts.
    pub fn train_epochs_capture_best<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
    ) -> PureResult<EpochBestState>
    where
        M: Module,
        L: Loss,
    {
        let mut history = Vec::with_capacity(epochs);
        let mut best_state = module.state_dict()?;
        let mut best_loss = f32::INFINITY;
        for _ in 0..epochs {
            let stats = self.train_epoch(module, loss, loader.clone(), schedule)?;
            if stats.average_loss_per_row.is_finite() && stats.average_loss_per_row < best_loss {
                best_loss = stats.average_loss_per_row;
                best_state = module.state_dict()?;
            }
            history.push(stats);
        }
        let final_state = module.state_dict()?;
        let summary = summarize_epoch_history(&history);
        let best_fingerprint = crate::module::fingerprint_state_dict(&best_state);
        let final_fingerprint = crate::module::fingerprint_state_dict(&final_state);
        let best_differs_from_final = best_fingerprint != final_fingerprint;
        Ok(EpochBestState {
            history,
            summary,
            best_state,
            final_state,
            best_fingerprint,
            final_fingerprint,
            best_differs_from_final,
        })
    }

    /// Runs multiple epochs and restores the module to the best row-weighted epoch.
    pub fn train_epochs_restore_best<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
    ) -> PureResult<EpochBestState>
    where
        M: Module,
        L: Loss,
    {
        let captured = self.train_epochs_capture_best(module, loss, loader, schedule, epochs)?;
        if captured.summary.best_epoch.is_some() {
            let load = module.load_state_dict_checked(&captured.best_state)?;
            if !load.matched {
                return Err(TensorError::IoError {
                    message: "best epoch state failed checked restore".to_string(),
                });
            }
        }
        Ok(captured)
    }

    /// Runs multiple epochs and captures the best state using validation loss.
    pub fn train_epochs_capture_best_on_validation<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
    ) -> PureResult<EpochValidationBestState>
    where
        M: Module,
        L: Loss,
    {
        self.train_epochs_capture_best_on_validation_inner(
            module,
            loss,
            train_loader,
            validation_loader,
            schedule,
            epochs,
            ValidationTrainingControls::default(),
        )
    }

    /// Runs multiple epochs and captures the best validation state with early stopping.
    pub fn train_epochs_capture_best_on_validation_with_early_stopping<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        early_stopping: EarlyStoppingConfig,
    ) -> PureResult<EpochValidationBestState>
    where
        M: Module,
        L: Loss,
    {
        self.train_epochs_capture_best_on_validation_with_controls(
            module,
            loss,
            train_loader,
            validation_loader,
            schedule,
            epochs,
            ValidationTrainingControls::default().with_early_stopping(early_stopping.validate()?),
        )
    }

    /// Runs multiple epochs and captures the best validation state with FT controls.
    pub fn train_epochs_capture_best_on_validation_with_controls<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        controls: ValidationTrainingControls,
    ) -> PureResult<EpochValidationBestState>
    where
        M: Module,
        L: Loss,
    {
        self.train_epochs_capture_best_on_validation_inner(
            module,
            loss,
            train_loader,
            validation_loader,
            schedule,
            epochs,
            controls.validate()?,
        )
    }

    fn train_epochs_capture_best_on_validation_inner<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        controls: ValidationTrainingControls,
    ) -> PureResult<EpochValidationBestState>
    where
        M: Module,
        L: Loss,
    {
        let mut train_history = Vec::with_capacity(epochs);
        let mut validation_history = Vec::with_capacity(epochs);
        let mut best_state = module.state_dict()?;
        let mut best_loss = f32::INFINITY;
        let mut early_stop_reference = f32::INFINITY;
        let mut stale_epochs = 0usize;
        let mut plateau_reference = f32::INFINITY;
        let mut plateau_stale_epochs = 0usize;
        let mut lr_decay_steps = 0usize;
        let mut early_stopped = false;
        let mut stop_epoch = None;
        for epoch_idx in 0..epochs {
            let train_stats = self.train_epoch(module, loss, train_loader.clone(), schedule)?;
            let validation_stats = self.evaluate_epoch(module, loss, validation_loader.clone())?;
            let validation_loss = validation_stats.average_loss_per_row;
            if validation_loss.is_finite() && validation_loss < best_loss {
                best_loss = validation_loss;
                best_state = module.state_dict()?;
            }
            train_history.push(train_stats);
            validation_history.push(validation_stats);

            let mut should_early_stop = false;
            if let Some(config) = controls.early_stopping {
                if validation_loss.is_finite()
                    && validation_loss + config.min_delta < early_stop_reference
                {
                    early_stop_reference = validation_loss;
                    stale_epochs = 0;
                } else {
                    stale_epochs += 1;
                    if stale_epochs >= config.patience {
                        should_early_stop = true;
                    }
                }
            }
            if let Some(config) = controls.lr_plateau {
                if validation_loss.is_finite()
                    && validation_loss + config.min_delta < plateau_reference
                {
                    plateau_reference = validation_loss;
                    plateau_stale_epochs = 0;
                } else {
                    plateau_stale_epochs += 1;
                    let has_next_epoch = epoch_idx + 1 < epochs;
                    if plateau_stale_epochs >= config.patience
                        && has_next_epoch
                        && !should_early_stop
                    {
                        self.optimizer_mul_lr(module, config.factor)?;
                        lr_decay_steps += 1;
                        plateau_stale_epochs = 0;
                    }
                }
            }
            if should_early_stop {
                early_stopped = true;
                stop_epoch = Some(epoch_idx + 1);
                break;
            }
        }
        let final_state = module.state_dict()?;
        let train_summary = summarize_epoch_history(&train_history);
        let validation_summary = summarize_epoch_history(&validation_history);
        let best_fingerprint = crate::module::fingerprint_state_dict(&best_state);
        let final_fingerprint = crate::module::fingerprint_state_dict(&final_state);
        let best_differs_from_final = best_fingerprint != final_fingerprint;
        Ok(EpochValidationBestState {
            train_history,
            validation_history,
            train_summary,
            validation_summary,
            best_state,
            final_state,
            best_fingerprint,
            final_fingerprint,
            best_differs_from_final,
            epochs_requested: epochs,
            early_stopping: controls.early_stopping,
            early_stopped,
            stop_epoch,
            lr_plateau: controls.lr_plateau,
            lr_decay_steps,
            final_hyper_learning_rate: self.hyper_learning_rate,
            final_fallback_learning_rate: self.fallback_learning_rate,
        })
    }

    /// Runs multiple epochs and restores the module to the best validation epoch.
    pub fn train_epochs_restore_best_on_validation<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
    ) -> PureResult<EpochValidationBestState>
    where
        M: Module,
        L: Loss,
    {
        let captured = self.train_epochs_capture_best_on_validation(
            module,
            loss,
            train_loader,
            validation_loader,
            schedule,
            epochs,
        )?;
        if captured.validation_summary.best_epoch.is_some() {
            let load = module.load_state_dict_checked(&captured.best_state)?;
            if !load.matched {
                return Err(TensorError::IoError {
                    message: "best validation epoch state failed checked restore".to_string(),
                });
            }
        }
        Ok(captured)
    }

    /// Runs multiple validation-selected epochs with early stopping and restores the best state.
    pub fn train_epochs_restore_best_on_validation_with_early_stopping<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        early_stopping: EarlyStoppingConfig,
    ) -> PureResult<EpochValidationBestState>
    where
        M: Module,
        L: Loss,
    {
        self.train_epochs_restore_best_on_validation_with_controls(
            module,
            loss,
            train_loader,
            validation_loader,
            schedule,
            epochs,
            ValidationTrainingControls::default().with_early_stopping(early_stopping),
        )
    }

    /// Runs validation-selected epochs with FT controls and restores the best state.
    pub fn train_epochs_restore_best_on_validation_with_controls<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        controls: ValidationTrainingControls,
    ) -> PureResult<EpochValidationBestState>
    where
        M: Module,
        L: Loss,
    {
        let captured = self.train_epochs_capture_best_on_validation_with_controls(
            module,
            loss,
            train_loader,
            validation_loader,
            schedule,
            epochs,
            controls,
        )?;
        if captured.validation_summary.best_epoch.is_some() {
            let load = module.load_state_dict_checked(&captured.best_state)?;
            if !load.matched {
                return Err(TensorError::IoError {
                    message: "best validation epoch state failed checked restore".to_string(),
                });
            }
        }
        Ok(captured)
    }

    /// Runs fine-tuning epochs and selects the best validation state that
    /// stays within a retention-loss guard measured before training starts.
    pub fn train_epochs_capture_best_with_retention_guard<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: RetentionGuardConfig,
    ) -> PureResult<EpochRetentionBestState>
    where
        M: Module,
        L: Loss,
    {
        self.train_epochs_capture_best_with_retention_guard_and_controls(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            retention_guard,
            ValidationTrainingControls::default(),
        )
    }

    /// Runs fine-tuning epochs with validation controls and retention-aware
    /// best-state selection.
    pub fn train_epochs_capture_best_with_retention_guard_and_controls<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: RetentionGuardConfig,
        controls: ValidationTrainingControls,
    ) -> PureResult<EpochRetentionBestState>
    where
        M: Module,
        L: Loss,
    {
        self.train_epochs_capture_best_with_retention_guard_inner(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            retention_guard.validate()?,
            controls.validate()?,
        )
    }

    fn train_epochs_capture_best_with_retention_guard_inner<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: RetentionGuardConfig,
        controls: ValidationTrainingControls,
    ) -> PureResult<EpochRetentionBestState>
    where
        M: Module,
        L: Loss,
    {
        let initial_state = module.state_dict()?;
        let retention_baseline = self.evaluate_epoch(module, loss, retention_loader.clone())?;
        let baseline_retention_loss = retention_baseline.average_loss_per_row;
        Self::ensure_finite("retention_baseline_loss", baseline_retention_loss)?;
        let max_allowed_retention_loss_per_row =
            baseline_retention_loss + retention_guard.max_loss_increase;

        let mut train_history = Vec::with_capacity(epochs);
        let mut validation_history = Vec::with_capacity(epochs);
        let mut retention_history = Vec::with_capacity(epochs);
        let mut best_state = initial_state;
        let mut best_validation_loss = f32::INFINITY;
        let mut best_retention_loss = baseline_retention_loss;
        let mut guarded_best_epoch = None;
        let mut early_stop_reference = f32::INFINITY;
        let mut stale_epochs = 0usize;
        let mut plateau_reference = f32::INFINITY;
        let mut plateau_stale_epochs = 0usize;
        let mut lr_decay_steps = 0usize;
        let mut early_stopped = false;
        let mut stop_epoch = None;

        for epoch_idx in 0..epochs {
            let train_stats = self.train_epoch(module, loss, train_loader.clone(), schedule)?;
            let validation_stats = self.evaluate_epoch(module, loss, validation_loader.clone())?;
            let retention_stats = self.evaluate_epoch(module, loss, retention_loader.clone())?;
            let validation_loss = validation_stats.average_loss_per_row;
            let retention_loss = retention_stats.average_loss_per_row;
            let retention_ok =
                retention_loss.is_finite() && retention_loss <= max_allowed_retention_loss_per_row;
            let target_improved = validation_loss.is_finite()
                && validation_loss + retention_guard.target_min_delta < best_validation_loss;
            if retention_ok && target_improved {
                best_validation_loss = validation_loss;
                best_retention_loss = retention_loss;
                guarded_best_epoch = Some(epoch_idx + 1);
                best_state = module.state_dict()?;
            }
            train_history.push(train_stats);
            validation_history.push(validation_stats);
            retention_history.push(retention_stats);

            let mut should_early_stop = false;
            if let Some(config) = controls.early_stopping {
                if validation_loss.is_finite()
                    && validation_loss + config.min_delta < early_stop_reference
                {
                    early_stop_reference = validation_loss;
                    stale_epochs = 0;
                } else {
                    stale_epochs += 1;
                    if stale_epochs >= config.patience {
                        should_early_stop = true;
                    }
                }
            }
            if let Some(config) = controls.lr_plateau {
                if validation_loss.is_finite()
                    && validation_loss + config.min_delta < plateau_reference
                {
                    plateau_reference = validation_loss;
                    plateau_stale_epochs = 0;
                } else {
                    plateau_stale_epochs += 1;
                    let has_next_epoch = epoch_idx + 1 < epochs;
                    if plateau_stale_epochs >= config.patience
                        && has_next_epoch
                        && !should_early_stop
                    {
                        self.optimizer_mul_lr(module, config.factor)?;
                        lr_decay_steps += 1;
                        plateau_stale_epochs = 0;
                    }
                }
            }
            if should_early_stop {
                early_stopped = true;
                stop_epoch = Some(epoch_idx + 1);
                break;
            }
        }

        let final_state = module.state_dict()?;
        let train_summary = summarize_epoch_history(&train_history);
        let validation_summary = summarize_epoch_history(&validation_history);
        let retention_summary = summarize_epoch_history(&retention_history);
        let best_fingerprint = crate::module::fingerprint_state_dict(&best_state);
        let final_fingerprint = crate::module::fingerprint_state_dict(&final_state);
        let best_differs_from_final = best_fingerprint != final_fingerprint;
        Ok(EpochRetentionBestState {
            train_history,
            validation_history,
            retention_history,
            retention_baseline,
            train_summary,
            validation_summary,
            retention_summary,
            retention_guard,
            max_allowed_retention_loss_per_row,
            guarded_best_epoch,
            best_validation_loss_per_row: best_validation_loss,
            best_retention_loss_per_row: best_retention_loss,
            best_retention_loss_increase: best_retention_loss - baseline_retention_loss,
            best_state,
            final_state,
            best_fingerprint,
            final_fingerprint,
            best_differs_from_final,
            epochs_requested: epochs,
            early_stopping: controls.early_stopping,
            early_stopped,
            stop_epoch,
            lr_plateau: controls.lr_plateau,
            lr_decay_steps,
            final_hyper_learning_rate: self.hyper_learning_rate,
            final_fallback_learning_rate: self.fallback_learning_rate,
        })
    }

    /// Runs retention-guarded fine-tuning and restores the selected state.
    ///
    /// When no post-train epoch satisfies the guard, the selected state is the
    /// pre-fine-tune snapshot, preventing a forgetting-heavy run from being
    /// installed silently.
    pub fn train_epochs_restore_best_with_retention_guard<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: RetentionGuardConfig,
    ) -> PureResult<EpochRetentionBestState>
    where
        M: Module,
        L: Loss,
    {
        self.train_epochs_restore_best_with_retention_guard_and_controls(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            retention_guard,
            ValidationTrainingControls::default(),
        )
    }

    /// Runs retention-guarded fine-tuning with validation controls and restores
    /// the selected state.
    pub fn train_epochs_restore_best_with_retention_guard_and_controls<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: RetentionGuardConfig,
        controls: ValidationTrainingControls,
    ) -> PureResult<EpochRetentionBestState>
    where
        M: Module,
        L: Loss,
    {
        let captured = self.train_epochs_capture_best_with_retention_guard_and_controls(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            retention_guard,
            controls,
        )?;
        let load = module.load_state_dict_checked(&captured.best_state)?;
        if !load.matched {
            return Err(TensorError::IoError {
                message: "retention-guarded best state failed checked restore".to_string(),
            });
        }
        Ok(captured)
    }

    /// Runs sparse-classification fine-tuning and selects the best target
    /// validation state whose source-retention metrics stay inside the guard.
    pub fn train_epochs_capture_best_sparse_with_retention_guard<M>(
        &mut self,
        module: &mut M,
        loss: &mut SoftmaxCrossEntropy,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: SparseRetentionGuardConfig,
    ) -> PureResult<EpochSparseRetentionBestState>
    where
        M: Module,
    {
        self.train_epochs_capture_best_sparse_with_retention_guard_and_controls(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            retention_guard,
            ValidationTrainingControls::default(),
        )
    }

    /// Runs sparse-classification fine-tuning with validation controls and
    /// sparse source-retention best-state selection.
    pub fn train_epochs_capture_best_sparse_with_retention_guard_and_controls<M>(
        &mut self,
        module: &mut M,
        loss: &mut SoftmaxCrossEntropy,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: SparseRetentionGuardConfig,
        controls: ValidationTrainingControls,
    ) -> PureResult<EpochSparseRetentionBestState>
    where
        M: Module,
    {
        self.train_epochs_capture_best_sparse_with_retention_guard_inner(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            retention_guard.validate()?,
            controls.validate()?,
        )
    }

    fn train_epochs_capture_best_sparse_with_retention_guard_inner<M>(
        &mut self,
        module: &mut M,
        loss: &mut SoftmaxCrossEntropy,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: SparseRetentionGuardConfig,
        controls: ValidationTrainingControls,
    ) -> PureResult<EpochSparseRetentionBestState>
    where
        M: Module,
    {
        let initial_state = module.state_dict()?;
        let validation_baseline =
            self.evaluate_sparse_classification_epoch(module, loss, validation_loader.clone())?;
        Self::ensure_sparse_metrics("sparse_validation_baseline", &validation_baseline)?;
        let retention_baseline =
            self.evaluate_sparse_classification_epoch(module, loss, retention_loader.clone())?;
        Self::ensure_sparse_metrics("sparse_retention_baseline", &retention_baseline)?;
        let max_allowed_retention_loss =
            retention_baseline.mean_loss + retention_guard.max_loss_increase;
        Self::ensure_finite(
            "sparse_retention_max_allowed_loss",
            max_allowed_retention_loss,
        )?;
        let min_allowed_retention_accuracy =
            (retention_baseline.accuracy - retention_guard.max_accuracy_drop).max(0.0);
        let max_allowed_retention_perplexity =
            if let Some(increase) = retention_guard.max_perplexity_increase {
                let ceiling = retention_baseline.perplexity + increase;
                Self::ensure_finite("sparse_retention_max_allowed_perplexity", ceiling)?;
                Some(ceiling)
            } else {
                None
            };

        let mut train_history = Vec::with_capacity(epochs);
        let mut validation_history = Vec::with_capacity(epochs);
        let mut retention_history = Vec::with_capacity(epochs);
        let mut best_state = initial_state;
        let mut best_validation_metrics = validation_baseline;
        let mut best_retention_metrics = retention_baseline;
        let mut guarded_best_epoch = None;
        let mut early_stop_reference = validation_baseline.mean_loss;
        let mut stale_epochs = 0usize;
        let mut plateau_reference = validation_baseline.mean_loss;
        let mut plateau_stale_epochs = 0usize;
        let mut lr_decay_steps = 0usize;
        let mut early_stopped = false;
        let mut stop_epoch = None;
        let mut guard_accepted_epochs = 0usize;
        let mut guard_retention_rejected_epochs = 0usize;
        let mut guard_target_stale_epochs = 0usize;

        for epoch_idx in 0..epochs {
            let train_stats = self.train_epoch(module, loss, train_loader.clone(), schedule)?;
            let validation_metrics =
                self.evaluate_sparse_classification_epoch(module, loss, validation_loader.clone())?;
            Self::ensure_sparse_metrics("sparse_validation_epoch", &validation_metrics)?;
            let retention_metrics =
                self.evaluate_sparse_classification_epoch(module, loss, retention_loader.clone())?;
            Self::ensure_sparse_metrics("sparse_retention_epoch", &retention_metrics)?;

            let retention_ok = retention_metrics.mean_loss <= max_allowed_retention_loss
                && retention_metrics.accuracy + f32::EPSILON >= min_allowed_retention_accuracy
                && max_allowed_retention_perplexity
                    .map(|ceiling| retention_metrics.perplexity <= ceiling)
                    .unwrap_or(true);
            let target_improved = validation_metrics.mean_loss
                + retention_guard.target_min_loss_delta
                < best_validation_metrics.mean_loss;
            if retention_ok && target_improved {
                guard_accepted_epochs += 1;
                best_validation_metrics = validation_metrics;
                best_retention_metrics = retention_metrics;
                guarded_best_epoch = Some(epoch_idx + 1);
                best_state = module.state_dict()?;
            } else if !retention_ok {
                guard_retention_rejected_epochs += 1;
            } else {
                guard_target_stale_epochs += 1;
            }
            train_history.push(train_stats);
            validation_history.push(validation_metrics);
            retention_history.push(retention_metrics);

            let validation_loss = validation_metrics.mean_loss;
            let mut should_early_stop = false;
            if let Some(config) = controls.early_stopping {
                if validation_loss.is_finite()
                    && validation_loss + config.min_delta < early_stop_reference
                {
                    early_stop_reference = validation_loss;
                    stale_epochs = 0;
                } else {
                    stale_epochs += 1;
                    if stale_epochs >= config.patience {
                        should_early_stop = true;
                    }
                }
            }
            if let Some(config) = controls.lr_plateau {
                if validation_loss.is_finite()
                    && validation_loss + config.min_delta < plateau_reference
                {
                    plateau_reference = validation_loss;
                    plateau_stale_epochs = 0;
                } else {
                    plateau_stale_epochs += 1;
                    let has_next_epoch = epoch_idx + 1 < epochs;
                    if plateau_stale_epochs >= config.patience
                        && has_next_epoch
                        && !should_early_stop
                    {
                        self.optimizer_mul_lr(module, config.factor)?;
                        lr_decay_steps += 1;
                        plateau_stale_epochs = 0;
                    }
                }
            }
            if should_early_stop {
                early_stopped = true;
                stop_epoch = Some(epoch_idx + 1);
                break;
            }
        }

        let final_state = module.state_dict()?;
        let train_summary = summarize_epoch_history(&train_history);
        let best_fingerprint = crate::module::fingerprint_state_dict(&best_state);
        let final_fingerprint = crate::module::fingerprint_state_dict(&final_state);
        let best_differs_from_final = best_fingerprint != final_fingerprint;
        Ok(EpochSparseRetentionBestState {
            train_history,
            validation_history,
            retention_history,
            validation_baseline,
            retention_baseline,
            train_summary,
            retention_guard,
            max_allowed_retention_loss,
            min_allowed_retention_accuracy,
            max_allowed_retention_perplexity,
            guarded_best_epoch,
            guard_accepted_epochs,
            guard_retention_rejected_epochs,
            guard_target_stale_epochs,
            best_validation_metrics,
            best_retention_metrics,
            best_retention_loss_increase: best_retention_metrics.mean_loss
                - retention_baseline.mean_loss,
            best_retention_accuracy_drop: retention_baseline.accuracy
                - best_retention_metrics.accuracy,
            best_retention_perplexity_increase: best_retention_metrics.perplexity
                - retention_baseline.perplexity,
            best_state,
            final_state,
            best_fingerprint,
            final_fingerprint,
            best_differs_from_final,
            epochs_requested: epochs,
            early_stopping: controls.early_stopping,
            early_stopped,
            stop_epoch,
            lr_plateau: controls.lr_plateau,
            lr_decay_steps,
            final_hyper_learning_rate: self.hyper_learning_rate,
            final_fallback_learning_rate: self.fallback_learning_rate,
        })
    }

    /// Runs sparse-classification fine-tuning and restores the selected guarded state.
    pub fn train_epochs_restore_best_sparse_with_retention_guard<M>(
        &mut self,
        module: &mut M,
        loss: &mut SoftmaxCrossEntropy,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: SparseRetentionGuardConfig,
    ) -> PureResult<EpochSparseRetentionBestState>
    where
        M: Module,
    {
        self.train_epochs_restore_best_sparse_with_retention_guard_and_controls(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            retention_guard,
            ValidationTrainingControls::default(),
        )
    }

    /// Runs sparse-classification fine-tuning with validation controls and
    /// restores the selected guarded state.
    pub fn train_epochs_restore_best_sparse_with_retention_guard_and_controls<M>(
        &mut self,
        module: &mut M,
        loss: &mut SoftmaxCrossEntropy,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: SparseRetentionGuardConfig,
        controls: ValidationTrainingControls,
    ) -> PureResult<EpochSparseRetentionBestState>
    where
        M: Module,
    {
        let captured = self.train_epochs_capture_best_sparse_with_retention_guard_and_controls(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            retention_guard,
            controls,
        )?;
        let load = module.load_state_dict_checked(&captured.best_state)?;
        if !load.matched {
            return Err(TensorError::IoError {
                message: "sparse retention-guarded best state failed checked restore".to_string(),
            });
        }
        Ok(captured)
    }

    /// Runs sparse retention-guarded fine-tuning, restores the selected state,
    /// then audits target/retention deltas and parameter movement.
    pub fn train_epochs_restore_best_sparse_with_finetune_report<M>(
        &mut self,
        module: &mut M,
        loss: &mut SoftmaxCrossEntropy,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: SparseRetentionGuardConfig,
        movement_tolerance: f32,
    ) -> PureResult<SparseFineTuneReport>
    where
        M: Module,
    {
        self.train_epochs_restore_best_sparse_with_finetune_report_and_controls(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            retention_guard,
            movement_tolerance,
            ValidationTrainingControls::default(),
        )
    }

    /// Runs sparse retention-guarded fine-tuning with validation controls,
    /// restores the selected state, then audits target/retention deltas and
    /// parameter movement.
    pub fn train_epochs_restore_best_sparse_with_finetune_report_and_controls<M>(
        &mut self,
        module: &mut M,
        loss: &mut SoftmaxCrossEntropy,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        retention_loader: DataLoader,
        schedule: &RoundtableSchedule,
        epochs: usize,
        retention_guard: SparseRetentionGuardConfig,
        movement_tolerance: f32,
        controls: ValidationTrainingControls,
    ) -> PureResult<SparseFineTuneReport>
    where
        M: Module,
    {
        let before_state = module.state_dict()?;
        let resume_fingerprint = self.resume_fingerprint(module)?;
        let validation_after_loader = validation_loader.clone();
        let retention_after_loader = retention_loader.clone();
        let captured = self.train_epochs_restore_best_sparse_with_retention_guard_and_controls(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            retention_guard,
            controls,
        )?;
        let target_after =
            self.evaluate_sparse_classification_epoch(module, loss, validation_after_loader)?;
        Self::ensure_sparse_metrics("sparse_finetune_target_after", &target_after)?;
        let retention_after =
            self.evaluate_sparse_classification_epoch(module, loss, retention_after_loader)?;
        Self::ensure_sparse_metrics("sparse_finetune_retention_after", &retention_after)?;
        let movement = module.audit_parameter_movement(&before_state, movement_tolerance)?;
        Ok(SparseFineTuneReport {
            target_delta: captured.validation_baseline.delta_to(target_after),
            retention_delta: captured.retention_baseline.delta_to(retention_after),
            target_after,
            retention_after,
            movement_tolerance,
            movement,
            resume_fingerprint,
            accepted: captured.guarded_best_epoch.is_some(),
            captured,
        })
    }

    fn estimate_device_load(&self) -> f64 {
        let caps = self.planner.device_caps();
        caps.occupancy_score(caps.max_workgroup) as f64
    }

    fn ensure_finite(label: &'static str, value: f32) -> PureResult<()> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(TensorError::NonFiniteValue { label, value })
        }
    }

    fn ensure_tensor_finite(label: &'static str, tensor: &Tensor) -> PureResult<()> {
        for &value in tensor.data() {
            Self::ensure_finite(label, value)?;
        }
        Ok(())
    }

    fn ensure_sparse_metrics(
        label: &'static str,
        metrics: &SparseClassificationMetrics,
    ) -> PureResult<()> {
        if metrics.active_rows == 0 {
            return Err(TensorError::IoError {
                message: format!("{label} has no active sparse rows"),
            });
        }
        Self::ensure_finite(label, metrics.mean_loss)?;
        Self::ensure_finite(label, metrics.accuracy)?;
        Self::ensure_finite(label, metrics.perplexity)
    }

    #[cfg(feature = "psi")]
    fn bootstrap_psi(&mut self, schedule: &RoundtableSchedule) {
        if self.psi.is_some() || !schedule.psi_enabled() {
            return;
        }
        let cfg = PsiConfig::automated(schedule.psi_hint());
        self.psi = Some(PsiMeter::new(cfg));
    }

    #[cfg(feature = "psychoid")]
    fn bootstrap_psychoid(&mut self, schedule: &RoundtableSchedule) {
        if self.psychoid.is_some() || !schedule.psychoid_enabled() {
            return;
        }
        let cfg = PsychoidConfig::default();
        self.psychoid = Some(PsychoidMeter::new(cfg));
        self.psychoid_log = schedule.psychoid_log();
    }

    #[cfg(feature = "collapse")]
    fn bootstrap_collapse(&mut self, schedule: &RoundtableSchedule) {
        if self.collapse.is_some() || !schedule.collapse_enabled() {
            return;
        }
        let cfg = CollapseConfig::automated(schedule.psi_hint());
        self.collapse = Some(CollapseDrive::new(cfg));
    }

    fn apply_grad_scale<M: Module>(&self, module: &mut M, scale: f32) -> PureResult<()> {
        if (scale - 1.0).abs() <= f32::EPSILON {
            return Ok(());
        }
        module.visit_parameters_mut(&mut |param| {
            param.scale_accumulators(scale);
            Ok(())
        })
    }

    fn clip_grad_global_norm<M: Module>(&self, module: &mut M, max_norm: f32) -> PureResult<()> {
        if max_norm <= 0.0 {
            return Ok(());
        }
        let mut total = 0.0f64;
        module.visit_parameters(&mut |param| {
            total += param.accumulators_norm_sq();
            Ok(())
        })?;
        let norm = total.sqrt() as f32;
        if norm <= max_norm || norm <= f32::EPSILON {
            return Ok(());
        }
        let scale = (max_norm / norm).clamp(0.0, 1.0);
        self.apply_grad_scale(module, scale)
    }

    fn optimizer_mul_lr<M: Module>(&mut self, module: &mut M, factor: f32) -> PureResult<()> {
        if !factor.is_finite() || factor <= 0.0 {
            return Ok(());
        }
        self.fallback_learning_rate *= factor;
        self.hyper_learning_rate *= factor;
        module.visit_parameters_mut(&mut |param| {
            if let Some(tape) = param.hypergrad_mut() {
                tape.scale_learning_rate(factor);
            }
            Ok(())
        })
    }

    #[cfg(feature = "psi")]
    fn collect_grad_l2<M: Module>(module: &M) -> PureResult<f32> {
        let mut sum = 0.0f64;
        module.visit_parameters(&mut |param| {
            if let Some(tape) = param.hypergrad() {
                for &value in tape.gradient().iter() {
                    let v = value as f64;
                    sum += v * v;
                }
            } else if let Some(grad) = param.gradient() {
                for &value in grad.data().iter() {
                    let v = value as f64;
                    sum += v * v;
                }
            }
            Ok(())
        })?;
        Ok((sum).sqrt() as f32)
    }

    fn apply_proposal(
        &mut self,
        proposal: &GlobalProposal,
        preview_metrics: HashMap<String, f32>,
    ) -> PureResult<()> {
        let _ = preview_metrics;
        for op in &proposal.ops {
            self.heur_log.append(op.clone());
        }
        Ok(())
    }

    #[cfg(feature = "psychoid")]
    fn log_psychoid(reading: &PsychoidReading, events: &[PsychoidEvent]) {
        println!(
            "[psychoid] step={} cti={:.4} raw={{D:{:.3} S:{:.3} C:{:.3} K:{:.3} H:{:.3}}}",
            reading.step,
            reading.cti,
            reading.raw.get("D").copied().unwrap_or(0.0),
            reading.raw.get("S").copied().unwrap_or(0.0),
            reading.raw.get("C").copied().unwrap_or(0.0),
            reading.raw.get("K").copied().unwrap_or(0.0),
            reading.raw.get("H").copied().unwrap_or(0.0)
        );
        for event in events {
            match event {
                PsychoidEvent::DreamPass { step, cti } => {
                    println!("[psychoid-event] step={} dream-pass cti={:.4}", step, cti);
                }
                PsychoidEvent::DreamExport {
                    step,
                    diary,
                    symbols,
                } => {
                    println!(
                        "[psychoid-event] step={} dream-export symbols={:?} diary=\"{}\"",
                        step, symbols, diary
                    );
                }
            }
        }
    }
}

fn outcome_label(outcome: OutcomeBand) -> &'static str {
    match outcome {
        OutcomeBand::Above => "above",
        OutcomeBand::Here => "here",
        OutcomeBand::Beneath => "beneath",
    }
}

fn plan_signature(plan: &st_core::ops::rank_entry::RankPlan, outcome: OutcomeBand) -> String {
    format!(
        "{:?}:{}x{}:k{}:{}",
        plan.kind,
        plan.rows,
        plan.cols,
        plan.k,
        outcome_label(outcome)
    )
}

/// Trainer-side state that affects optimizer/resume behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainerStateSnapshot {
    pub curvature: f32,
    pub hyper_learning_rate: f32,
    pub fallback_learning_rate: f32,
    pub max_grad_norm: Option<f32>,
    pub gradient_accumulation_steps: usize,
    pub injector_enabled: bool,
    pub band_weights_enabled: bool,
    pub blackcat_enabled: bool,
    pub blackcat_moderator_enabled: bool,
    pub autopilot_enabled: bool,
    pub distribution_enabled: bool,
    pub meta_conductor_enabled: bool,
}

impl TrainerStateSnapshot {
    /// Counts optional runtime hooks that can change step behavior.
    pub fn runtime_hooks(&self) -> usize {
        [
            self.injector_enabled,
            self.band_weights_enabled,
            self.blackcat_enabled,
            self.blackcat_moderator_enabled,
            self.autopilot_enabled,
            self.distribution_enabled,
            self.meta_conductor_enabled,
        ]
        .into_iter()
        .filter(|enabled| *enabled)
        .count()
    }
}

/// Stable digest of trainer state used by resume audits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainerStateFingerprint {
    pub hash: String,
    pub gradient_accumulation_steps: usize,
    pub runtime_hooks: usize,
}

/// Combined trainer/module digest for auditable fine-tuning resume boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingResumeFingerprint {
    pub hash: String,
    pub trainer: TrainerStateFingerprint,
    pub parameters: StateFingerprint,
    pub parameter_training: ParameterTrainingFingerprint,
}

impl TrainingResumeFingerprint {
    pub fn from_parts(
        trainer: TrainerStateFingerprint,
        parameters: StateFingerprint,
        parameter_training: ParameterTrainingFingerprint,
    ) -> Self {
        let mut hash = FNV64_OFFSET;
        fingerprint_bytes(&mut hash, trainer.hash.as_bytes());
        fingerprint_bytes(&mut hash, parameters.hash.as_bytes());
        fingerprint_bytes(&mut hash, parameter_training.hash.as_bytes());
        Self {
            hash: format!("{hash:016x}"),
            trainer,
            parameters,
            parameter_training,
        }
    }
}

/// Comparison between an expected resume fingerprint and the current state.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingResumeAudit {
    pub expected: TrainingResumeFingerprint,
    pub actual: TrainingResumeFingerprint,
    pub trainer_matched: bool,
    pub parameters_matched: bool,
    pub parameter_training_matched: bool,
    pub matched: bool,
}

/// Metrics captured while running [`ModuleTrainer::train_epoch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochStats {
    /// Number of mini-batches processed.
    pub batches: usize,
    /// Number of optimizer updates applied.
    pub optimizer_steps: usize,
    /// Number of prediction rows observed across all batches.
    pub rows: usize,
    /// Sum of per-batch losses. Kept for backwards-compatible batch averaging.
    pub total_loss: f32,
    /// Sum of per-batch losses weighted by prediction row count.
    pub total_row_weighted_loss: f32,
    /// Average loss over optimizer steps / mini-batches.
    pub average_loss: f32,
    /// Average loss weighted by prediction row count, useful for sequence/token rows.
    pub average_loss_per_row: f32,
}

/// Validation early-stopping control for longer fine-tuning runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarlyStoppingConfig {
    /// Number of consecutive non-improving validation epochs allowed.
    pub patience: usize,
    /// Required row-weighted validation-loss drop to reset patience.
    pub min_delta: f32,
}

impl EarlyStoppingConfig {
    /// Creates a validation early-stopping config.
    pub fn new(patience: usize, min_delta: f32) -> PureResult<Self> {
        Self {
            patience,
            min_delta,
        }
        .validate()
    }

    fn validate(self) -> PureResult<Self> {
        if self.min_delta < 0.0 || !self.min_delta.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "early_stopping_min_delta",
                value: self.min_delta,
            });
        }
        Ok(self)
    }
}

/// Validation plateau learning-rate decay for longer fine-tuning runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrPlateauConfig {
    /// Consecutive validation epochs without meaningful improvement before decay.
    pub patience: usize,
    /// Multiplicative learning-rate decay factor.
    pub factor: f32,
    /// Required row-weighted validation-loss drop to reset plateau patience.
    pub min_delta: f32,
}

impl LrPlateauConfig {
    /// Creates a validation plateau LR-decay config.
    pub fn new(patience: usize, factor: f32, min_delta: f32) -> PureResult<Self> {
        Self {
            patience,
            factor,
            min_delta,
        }
        .validate()
    }

    fn validate(self) -> PureResult<Self> {
        if self.factor <= 0.0 || self.factor >= 1.0 || !self.factor.is_finite() {
            return Err(TensorError::IoError {
                message: "lr plateau factor must be finite and in (0, 1)".to_string(),
            });
        }
        if self.min_delta < 0.0 || !self.min_delta.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lr_plateau_min_delta",
                value: self.min_delta,
            });
        }
        Ok(self)
    }
}

/// Optional validation-driven controls for robust fine-tuning loops.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ValidationTrainingControls {
    /// Optional early stopping based on validation loss.
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Optional LR decay on validation plateau.
    pub lr_plateau: Option<LrPlateauConfig>,
}

impl ValidationTrainingControls {
    /// Enables validation early stopping.
    pub fn with_early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.early_stopping = Some(config);
        self
    }

    /// Enables validation plateau learning-rate decay.
    pub fn with_lr_plateau(mut self, config: LrPlateauConfig) -> Self {
        self.lr_plateau = Some(config);
        self
    }

    fn validate(self) -> PureResult<Self> {
        if let Some(config) = self.early_stopping {
            config.validate()?;
        }
        if let Some(config) = self.lr_plateau {
            config.validate()?;
        }
        Ok(self)
    }
}

/// Retention guard for fine-tuning runs that must preserve source behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetentionGuardConfig {
    /// Maximum allowed row-weighted retention-loss increase from the pre-FT baseline.
    pub max_loss_increase: f32,
    /// Required target-validation loss drop to replace the guarded best state.
    pub target_min_delta: f32,
}

impl RetentionGuardConfig {
    /// Creates a retention guard.
    pub fn new(max_loss_increase: f32, target_min_delta: f32) -> PureResult<Self> {
        Self {
            max_loss_increase,
            target_min_delta,
        }
        .validate()
    }

    /// Creates a guard with no target-validation delta threshold.
    pub fn allow_loss_increase(max_loss_increase: f32) -> PureResult<Self> {
        Self::new(max_loss_increase, 0.0)
    }

    fn validate(self) -> PureResult<Self> {
        if self.max_loss_increase < 0.0 || !self.max_loss_increase.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "retention_max_loss_increase",
                value: self.max_loss_increase,
            });
        }
        if self.target_min_delta < 0.0 || !self.target_min_delta.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "retention_target_min_delta",
                value: self.target_min_delta,
            });
        }
        Ok(self)
    }
}

/// Sparse-classification retention guard for tokenizerless/LM fine-tuning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseRetentionGuardConfig {
    /// Maximum allowed retention mean-loss increase from the pre-FT baseline.
    pub max_loss_increase: f32,
    /// Maximum allowed retention top-1 accuracy drop from the pre-FT baseline.
    pub max_accuracy_drop: f32,
    /// Optional maximum allowed retention perplexity increase from the pre-FT baseline.
    pub max_perplexity_increase: Option<f32>,
    /// Required target-validation mean-loss drop to replace the guarded best state.
    pub target_min_loss_delta: f32,
}

impl SparseRetentionGuardConfig {
    /// Creates a sparse retention guard with loss and accuracy constraints.
    pub fn new(max_loss_increase: f32, max_accuracy_drop: f32) -> PureResult<Self> {
        Self {
            max_loss_increase,
            max_accuracy_drop,
            max_perplexity_increase: None,
            target_min_loss_delta: 0.0,
        }
        .validate()
    }

    /// Adds a retention perplexity ceiling.
    pub fn with_max_perplexity_increase(
        mut self,
        max_perplexity_increase: f32,
    ) -> PureResult<Self> {
        self.max_perplexity_increase = Some(max_perplexity_increase);
        self.validate()
    }

    /// Requires a minimum target-validation mean-loss improvement before replacing best state.
    pub fn with_target_min_loss_delta(mut self, target_min_loss_delta: f32) -> PureResult<Self> {
        self.target_min_loss_delta = target_min_loss_delta;
        self.validate()
    }

    fn validate(self) -> PureResult<Self> {
        if self.max_loss_increase < 0.0 || !self.max_loss_increase.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "sparse_retention_max_loss_increase",
                value: self.max_loss_increase,
            });
        }
        if self.max_accuracy_drop < 0.0
            || self.max_accuracy_drop > 1.0
            || !self.max_accuracy_drop.is_finite()
        {
            return Err(TensorError::NonFiniteValue {
                label: "sparse_retention_max_accuracy_drop",
                value: self.max_accuracy_drop,
            });
        }
        if let Some(value) = self.max_perplexity_increase {
            if value < 0.0 || !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "sparse_retention_max_perplexity_increase",
                    value,
                });
            }
        }
        if self.target_min_loss_delta < 0.0 || !self.target_min_loss_delta.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "sparse_retention_target_min_loss_delta",
                value: self.target_min_loss_delta,
            });
        }
        Ok(self)
    }
}

/// Summary over a multi-epoch training history.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochHistory {
    /// Number of epochs summarized.
    pub epochs: usize,
    /// Total mini-batches across all epochs.
    pub batches: usize,
    /// Total optimizer updates applied across all epochs.
    pub optimizer_steps: usize,
    /// Total prediction rows observed across all epochs.
    pub rows: usize,
    /// First epoch's row-weighted loss.
    pub initial_loss_per_row: f32,
    /// Last epoch's row-weighted loss.
    pub final_loss_per_row: f32,
    /// Best epoch number using 1-based indexing.
    pub best_epoch: Option<usize>,
    /// Lowest row-weighted loss observed.
    pub best_loss_per_row: f32,
    /// Initial minus final row-weighted loss.
    pub final_improvement: f32,
    /// Initial minus best row-weighted loss.
    pub best_improvement: f32,
    /// True when the final epoch improved over the first.
    pub improved: bool,
    /// True when any epoch improved over the first.
    pub best_improved: bool,
}

/// Summarizes per-epoch stats into best/final improvement diagnostics.
pub fn summarize_epoch_history(history: &[EpochStats]) -> EpochHistory {
    if history.is_empty() {
        return EpochHistory {
            epochs: 0,
            batches: 0,
            optimizer_steps: 0,
            rows: 0,
            initial_loss_per_row: 0.0,
            final_loss_per_row: 0.0,
            best_epoch: None,
            best_loss_per_row: 0.0,
            final_improvement: 0.0,
            best_improvement: 0.0,
            improved: false,
            best_improved: false,
        };
    }

    let initial = history[0].average_loss_per_row;
    let final_loss = history
        .last()
        .map(|stats| stats.average_loss_per_row)
        .unwrap_or(initial);
    let (best_idx, best_loss) = history
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            left.average_loss_per_row
                .partial_cmp(&right.average_loss_per_row)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, stats)| (idx, stats.average_loss_per_row))
        .unwrap_or((0, initial));
    let batches = history.iter().map(|stats| stats.batches).sum();
    let optimizer_steps = history.iter().map(|stats| stats.optimizer_steps).sum();
    let rows = history.iter().map(|stats| stats.rows).sum();
    let final_improvement = initial - final_loss;
    let best_improvement = initial - best_loss;

    EpochHistory {
        epochs: history.len(),
        batches,
        optimizer_steps,
        rows,
        initial_loss_per_row: initial,
        final_loss_per_row: final_loss,
        best_epoch: Some(best_idx + 1),
        best_loss_per_row: best_loss,
        final_improvement,
        best_improvement,
        improved: final_improvement > 0.0 && final_improvement.is_finite(),
        best_improved: best_improvement > 0.0 && best_improvement.is_finite(),
    }
}

/// State snapshots captured around a multi-epoch run.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochBestState {
    /// Per-epoch training statistics.
    pub history: Vec<EpochStats>,
    /// Summary over the captured history.
    pub summary: EpochHistory,
    /// State dictionary captured at the best row-weighted epoch.
    pub best_state: HashMap<String, Tensor>,
    /// State dictionary captured after the final epoch.
    pub final_state: HashMap<String, Tensor>,
    /// Fingerprint for the best state dictionary.
    pub best_fingerprint: crate::module::StateFingerprint,
    /// Fingerprint for the final state dictionary.
    pub final_fingerprint: crate::module::StateFingerprint,
    /// True when the best epoch state differs from the final epoch state.
    pub best_differs_from_final: bool,
}

/// State snapshots captured while selecting the best epoch by validation loss.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochValidationBestState {
    /// Per-epoch training statistics.
    pub train_history: Vec<EpochStats>,
    /// Per-epoch validation statistics evaluated after each train epoch.
    pub validation_history: Vec<EpochStats>,
    /// Summary over the training history.
    pub train_summary: EpochHistory,
    /// Summary over the validation history used for best-state selection.
    pub validation_summary: EpochHistory,
    /// State dictionary captured at the best validation epoch.
    pub best_state: HashMap<String, Tensor>,
    /// State dictionary captured after the final training epoch.
    pub final_state: HashMap<String, Tensor>,
    /// Fingerprint for the best validation state dictionary.
    pub best_fingerprint: crate::module::StateFingerprint,
    /// Fingerprint for the final state dictionary.
    pub final_fingerprint: crate::module::StateFingerprint,
    /// True when the best validation epoch state differs from the final epoch state.
    pub best_differs_from_final: bool,
    /// Number of epochs the caller requested.
    pub epochs_requested: usize,
    /// Early-stopping configuration used for this run, when enabled.
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// True when training stopped before the requested epoch count.
    pub early_stopped: bool,
    /// One-based epoch index at which early stopping fired.
    pub stop_epoch: Option<usize>,
    /// LR decay-on-plateau configuration used for this run, when enabled.
    pub lr_plateau: Option<LrPlateauConfig>,
    /// Number of validation plateau LR decays applied during the run.
    pub lr_decay_steps: usize,
    /// Hypergrad learning rate after any validation plateau decays.
    pub final_hyper_learning_rate: f32,
    /// Fallback learning rate after any validation plateau decays.
    pub final_fallback_learning_rate: f32,
}

/// State snapshots captured while selecting a validation-best epoch subject to
/// a source-retention guard.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochRetentionBestState {
    /// Per-epoch training statistics.
    pub train_history: Vec<EpochStats>,
    /// Per-epoch target-validation statistics evaluated after each train epoch.
    pub validation_history: Vec<EpochStats>,
    /// Per-epoch source-retention statistics evaluated after each train epoch.
    pub retention_history: Vec<EpochStats>,
    /// Source-retention statistics evaluated before fine-tuning begins.
    pub retention_baseline: EpochStats,
    /// Summary over the training history.
    pub train_summary: EpochHistory,
    /// Summary over the target-validation history.
    pub validation_summary: EpochHistory,
    /// Summary over the source-retention history.
    pub retention_summary: EpochHistory,
    /// Guard used for best-state selection.
    pub retention_guard: RetentionGuardConfig,
    /// Absolute row-weighted retention-loss ceiling used by the guard.
    pub max_allowed_retention_loss_per_row: f32,
    /// One-based epoch index selected by the guard, or `None` when initial state won.
    pub guarded_best_epoch: Option<usize>,
    /// Target-validation loss at the selected guarded epoch.
    pub best_validation_loss_per_row: f32,
    /// Source-retention loss at the selected guarded epoch.
    pub best_retention_loss_per_row: f32,
    /// Selected retention loss minus the pre-FT retention baseline.
    pub best_retention_loss_increase: f32,
    /// State dictionary captured at the guarded best epoch or pre-FT snapshot.
    pub best_state: HashMap<String, Tensor>,
    /// State dictionary captured after the final training epoch.
    pub final_state: HashMap<String, Tensor>,
    /// Fingerprint for the selected state dictionary.
    pub best_fingerprint: crate::module::StateFingerprint,
    /// Fingerprint for the final state dictionary.
    pub final_fingerprint: crate::module::StateFingerprint,
    /// True when the selected state differs from the final epoch state.
    pub best_differs_from_final: bool,
    /// Number of epochs the caller requested.
    pub epochs_requested: usize,
    /// Early-stopping configuration used for this run, when enabled.
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// True when training stopped before the requested epoch count.
    pub early_stopped: bool,
    /// One-based epoch index at which early stopping fired.
    pub stop_epoch: Option<usize>,
    /// LR decay-on-plateau configuration used for this run, when enabled.
    pub lr_plateau: Option<LrPlateauConfig>,
    /// Number of validation plateau LR decays applied during the run.
    pub lr_decay_steps: usize,
    /// Hypergrad learning rate after any validation plateau decays.
    pub final_hyper_learning_rate: f32,
    /// Fallback learning rate after any validation plateau decays.
    pub final_fallback_learning_rate: f32,
}

/// State snapshots captured while selecting a sparse target-validation best
/// epoch subject to source-retention loss, accuracy, and optional perplexity guards.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochSparseRetentionBestState {
    /// Per-epoch training statistics.
    pub train_history: Vec<EpochStats>,
    /// Per-epoch sparse target-validation metrics evaluated after each train epoch.
    pub validation_history: Vec<SparseClassificationMetrics>,
    /// Per-epoch sparse source-retention metrics evaluated after each train epoch.
    pub retention_history: Vec<SparseClassificationMetrics>,
    /// Sparse target-validation metrics evaluated before fine-tuning begins.
    pub validation_baseline: SparseClassificationMetrics,
    /// Sparse source-retention metrics evaluated before fine-tuning begins.
    pub retention_baseline: SparseClassificationMetrics,
    /// Summary over the training history.
    pub train_summary: EpochHistory,
    /// Guard used for sparse best-state selection.
    pub retention_guard: SparseRetentionGuardConfig,
    /// Absolute retention mean-loss ceiling used by the guard.
    pub max_allowed_retention_loss: f32,
    /// Absolute retention top-1 accuracy floor used by the guard.
    pub min_allowed_retention_accuracy: f32,
    /// Optional absolute retention perplexity ceiling used by the guard.
    pub max_allowed_retention_perplexity: Option<f32>,
    /// One-based epoch index selected by the guard, or `None` when initial state won.
    pub guarded_best_epoch: Option<usize>,
    /// Number of post-train epochs installed as guarded best candidates.
    pub guard_accepted_epochs: usize,
    /// Number of post-train epochs rejected by the retention guard.
    pub guard_retention_rejected_epochs: usize,
    /// Number of retention-safe epochs that failed to improve the target enough.
    pub guard_target_stale_epochs: usize,
    /// Sparse target-validation metrics at the selected guarded epoch or pre-FT baseline.
    pub best_validation_metrics: SparseClassificationMetrics,
    /// Sparse source-retention metrics at the selected guarded epoch or pre-FT baseline.
    pub best_retention_metrics: SparseClassificationMetrics,
    /// Selected retention mean loss minus the pre-FT retention baseline.
    pub best_retention_loss_increase: f32,
    /// Pre-FT retention accuracy minus the selected retention accuracy.
    pub best_retention_accuracy_drop: f32,
    /// Selected retention perplexity minus the pre-FT retention baseline.
    pub best_retention_perplexity_increase: f32,
    /// State dictionary captured at the guarded best epoch or pre-FT snapshot.
    pub best_state: HashMap<String, Tensor>,
    /// State dictionary captured after the final training epoch.
    pub final_state: HashMap<String, Tensor>,
    /// Fingerprint for the selected state dictionary.
    pub best_fingerprint: crate::module::StateFingerprint,
    /// Fingerprint for the final state dictionary.
    pub final_fingerprint: crate::module::StateFingerprint,
    /// True when the selected state differs from the final epoch state.
    pub best_differs_from_final: bool,
    /// Number of epochs the caller requested.
    pub epochs_requested: usize,
    /// Early-stopping configuration used for this run, when enabled.
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// True when training stopped before the requested epoch count.
    pub early_stopped: bool,
    /// One-based epoch index at which early stopping fired.
    pub stop_epoch: Option<usize>,
    /// LR decay-on-plateau configuration used for this run, when enabled.
    pub lr_plateau: Option<LrPlateauConfig>,
    /// Number of validation plateau LR decays applied during the run.
    pub lr_decay_steps: usize,
    /// Hypergrad learning rate after any validation plateau decays.
    pub final_hyper_learning_rate: f32,
    /// Fallback learning rate after any validation plateau decays.
    pub final_fallback_learning_rate: f32,
}

/// Restored sparse fine-tuning report with target/retention deltas and
/// trainable/frozen parameter movement audited from the pre-FT snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseFineTuneReport {
    /// Underlying sparse retention-guarded best-state capture.
    pub captured: EpochSparseRetentionBestState,
    /// Target-validation metrics evaluated after restoring the selected state.
    pub target_after: SparseClassificationMetrics,
    /// Source-retention metrics evaluated after restoring the selected state.
    pub retention_after: SparseClassificationMetrics,
    /// Target-validation delta from the pre-FT baseline to the restored state.
    pub target_delta: SparseClassificationDelta,
    /// Source-retention delta from the pre-FT baseline to the restored state.
    pub retention_delta: SparseClassificationDelta,
    /// Parameter movement audit tolerance used for this report.
    pub movement_tolerance: f32,
    /// Parameter movement relative to the pre-FT state.
    pub movement: ParameterMovementReport,
    /// Resume fingerprint captured immediately before the FT loop starts.
    pub resume_fingerprint: TrainingResumeFingerprint,
    /// True when at least one post-train epoch satisfied the retention guard.
    pub accepted: bool,
}

/// Flat digest of a sparse fine-tuning report for experiment logs, CSV rows,
/// and lightweight cross-run comparisons.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseFineTuneReportSummary {
    pub status: &'static str,
    pub accepted: bool,
    pub target_loss_improved: bool,
    pub movement_ok: bool,
    pub guarded_best_epoch: Option<usize>,
    pub guard_epochs_run: usize,
    pub guard_accepted_epochs: usize,
    pub guard_retention_rejected_epochs: usize,
    pub guard_target_stale_epochs: usize,
    pub guard_acceptance_rate: f32,
    pub guard_retention_rejected_rate: f32,
    pub guard_target_stale_rate: f32,
    pub epochs_run: usize,
    pub train_rows: usize,
    pub train_batches: usize,
    pub optimizer_steps: usize,
    pub target_loss_delta: f32,
    pub target_accuracy_delta: f32,
    pub target_perplexity_delta: f32,
    pub retention_loss_delta: f32,
    pub retention_accuracy_delta: f32,
    pub retention_perplexity_delta: f32,
    /// Target loss improvement minus source-retention loss improvement.
    pub target_retention_gap: f32,
    /// Target/retention loss-improvement ratio when retention improvement is positive.
    pub target_retention_ratio: Option<f32>,
    pub best_retention_loss_increase: f32,
    pub best_retention_accuracy_drop: f32,
    pub best_retention_perplexity_increase: f32,
    pub retention_max_loss_increase: f32,
    pub retention_max_accuracy_drop: f32,
    pub retention_max_perplexity_increase: Option<f32>,
    pub target_min_loss_delta: f32,
    /// Target loss improvement beyond the configured minimum improvement.
    pub target_loss_margin: f32,
    /// Remaining source-retention loss-increase budget at the selected state.
    pub retention_loss_margin: f32,
    /// Remaining source-retention accuracy-drop budget at the selected state.
    pub retention_accuracy_margin: f32,
    /// Remaining source-retention perplexity budget when a ceiling is configured.
    pub retention_perplexity_margin: Option<f32>,
    pub max_allowed_retention_loss: f32,
    pub min_allowed_retention_accuracy: f32,
    pub max_allowed_retention_perplexity: Option<f32>,
    pub movement_status: &'static str,
    pub frozen_stable: bool,
    pub trainable_movement_observed: bool,
    pub movement_tolerance: f32,
    pub trainable_changed: usize,
    pub frozen_changed: usize,
    pub max_trainable_l2_delta: f32,
    pub max_frozen_l2_delta: f32,
    pub resume_hash: String,
    pub resume_trainer_hash: String,
    pub resume_parameter_hash: String,
    pub resume_parameter_training_hash: String,
    pub resume_trainable: usize,
    pub resume_frozen: usize,
    pub resume_hypergrad_tapes: usize,
    pub resume_gradient_accumulation_steps: usize,
    pub resume_runtime_hooks: usize,
    pub best_hash: String,
    pub final_hash: String,
    pub best_differs_from_final: bool,
}

/// Regression thresholds for comparing two sparse fine-tuning summaries.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SparseFineTuneRegressionLimits {
    /// Maximum allowed drop in `target_loss_delta` relative to a baseline run.
    pub max_target_loss_regression: Option<f32>,
    /// Maximum allowed drop in `retention_loss_delta` relative to a baseline run.
    pub max_retention_loss_regression: Option<f32>,
    /// Maximum allowed drop in `target_retention_gap` relative to a baseline run.
    pub max_target_retention_gap_regression: Option<f32>,
    /// Maximum allowed drop in `target_retention_ratio` relative to a baseline run.
    pub max_target_retention_ratio_regression: Option<f32>,
    /// Minimum current target-loss margin required for a passing comparison.
    pub min_target_loss_margin: Option<f32>,
    /// Minimum current target/retention loss-improvement ratio required.
    pub min_target_retention_ratio: Option<f32>,
    /// Minimum current source-retention loss margin required for a passing comparison.
    pub min_retention_loss_margin: Option<f32>,
    /// Minimum current source-retention accuracy margin required for a passing comparison.
    pub min_retention_accuracy_margin: Option<f32>,
    /// Minimum current source-retention perplexity margin required when configured.
    pub min_retention_perplexity_margin: Option<f32>,
    /// When true, the compact report status must stay unchanged.
    pub require_status_match: bool,
    /// When true, sparse fine-tune guard acceptance must stay unchanged.
    pub require_accepted_match: bool,
    /// When true, sparse retention guard settings must stay unchanged.
    pub require_guard_match: bool,
    /// When true, parameter movement audit tolerance must stay unchanged.
    pub require_movement_tolerance_match: bool,
    /// When true, FT-ready trainer/module resume fingerprints must stay unchanged.
    pub require_resume_match: bool,
}

/// Delta and gate outcome from comparing sparse fine-tuning summaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseFineTuneSummaryComparison {
    /// `current.target_loss_delta - baseline.target_loss_delta`.
    pub target_loss_delta_change: f32,
    /// `current.retention_loss_delta - baseline.retention_loss_delta`.
    pub retention_loss_delta_change: f32,
    /// `current.target_retention_gap - baseline.target_retention_gap`.
    pub target_retention_gap_change: f32,
    /// `current.target_retention_ratio - baseline.target_retention_ratio`.
    pub target_retention_ratio_change: Option<f32>,
    /// Positive amount by which target loss improvement worsened.
    pub target_loss_regression: f32,
    /// Positive amount by which retention loss improvement worsened.
    pub retention_loss_regression: f32,
    /// Positive amount by which target-vs-retention gap worsened.
    pub target_retention_gap_regression: f32,
    /// Positive amount by which target-vs-retention ratio worsened.
    pub target_retention_ratio_regression: Option<f32>,
    /// Positive amount by which target-loss margin missed the requested floor.
    pub target_loss_margin_shortfall: f32,
    /// Positive amount by which target-retention ratio missed the requested floor.
    pub target_retention_ratio_shortfall: Option<f32>,
    /// Positive amount by which retention-loss margin missed the requested floor.
    pub retention_loss_margin_shortfall: f32,
    /// Positive amount by which retention-accuracy margin missed the requested floor.
    pub retention_accuracy_margin_shortfall: f32,
    /// Positive amount by which retention-perplexity margin missed the requested floor.
    pub retention_perplexity_margin_shortfall: Option<f32>,
    /// True when compact report statuses differ.
    pub status_changed: bool,
    /// True when guard acceptance changed.
    pub accepted_changed: bool,
    /// True when sparse retention guard settings differ.
    pub guard_changed: bool,
    /// True when parameter movement audit tolerance differs.
    pub movement_tolerance_changed: bool,
    /// True when the FT-ready trainer/module resume fingerprint differs.
    pub resume_changed: bool,
    /// True when all requested regression gates pass.
    pub passed: bool,
}

impl SparseFineTuneRegressionLimits {
    /// Creates an informational comparison with no failing gates.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fails the comparison when target loss improvement regresses by more than
    /// `max_target_loss_regression`.
    pub fn with_max_target_loss_regression(
        mut self,
        max_target_loss_regression: f32,
    ) -> PureResult<Self> {
        validate_regression_limit(
            "sparse_finetune_max_target_loss_regression",
            max_target_loss_regression,
        )?;
        self.max_target_loss_regression = Some(max_target_loss_regression);
        Ok(self)
    }

    /// Fails the comparison when retention loss improvement regresses by more
    /// than `max_retention_loss_regression`.
    pub fn with_max_retention_loss_regression(
        mut self,
        max_retention_loss_regression: f32,
    ) -> PureResult<Self> {
        validate_regression_limit(
            "sparse_finetune_max_retention_loss_regression",
            max_retention_loss_regression,
        )?;
        self.max_retention_loss_regression = Some(max_retention_loss_regression);
        Ok(self)
    }

    /// Fails the comparison when target-vs-retention gap regresses by more
    /// than `max_target_retention_gap_regression`.
    pub fn with_max_target_retention_gap_regression(
        mut self,
        max_target_retention_gap_regression: f32,
    ) -> PureResult<Self> {
        validate_regression_limit(
            "sparse_finetune_max_target_retention_gap_regression",
            max_target_retention_gap_regression,
        )?;
        self.max_target_retention_gap_regression = Some(max_target_retention_gap_regression);
        Ok(self)
    }

    /// Fails the comparison when target-vs-retention ratio regresses by more
    /// than `max_target_retention_ratio_regression`.
    pub fn with_max_target_retention_ratio_regression(
        mut self,
        max_target_retention_ratio_regression: f32,
    ) -> PureResult<Self> {
        validate_regression_limit(
            "sparse_finetune_max_target_retention_ratio_regression",
            max_target_retention_ratio_regression,
        )?;
        self.max_target_retention_ratio_regression = Some(max_target_retention_ratio_regression);
        Ok(self)
    }

    /// Requires the current target-loss margin to stay at or above `min_margin`.
    pub fn with_min_target_loss_margin(mut self, min_margin: f32) -> PureResult<Self> {
        validate_regression_limit("sparse_finetune_min_target_loss_margin", min_margin)?;
        self.min_target_loss_margin = Some(min_margin);
        Ok(self)
    }

    /// Requires the current target/retention loss-improvement ratio to stay at
    /// or above `min_ratio`.
    pub fn with_min_target_retention_ratio(mut self, min_ratio: f32) -> PureResult<Self> {
        validate_regression_limit("sparse_finetune_min_target_retention_ratio", min_ratio)?;
        self.min_target_retention_ratio = Some(min_ratio);
        Ok(self)
    }

    /// Requires the current retention-loss margin to stay at or above `min_margin`.
    pub fn with_min_retention_loss_margin(mut self, min_margin: f32) -> PureResult<Self> {
        validate_regression_limit("sparse_finetune_min_retention_loss_margin", min_margin)?;
        self.min_retention_loss_margin = Some(min_margin);
        Ok(self)
    }

    /// Requires the current retention-accuracy margin to stay at or above `min_margin`.
    pub fn with_min_retention_accuracy_margin(mut self, min_margin: f32) -> PureResult<Self> {
        validate_regression_limit("sparse_finetune_min_retention_accuracy_margin", min_margin)?;
        self.min_retention_accuracy_margin = Some(min_margin);
        Ok(self)
    }

    /// Requires the current retention-perplexity margin to stay at or above `min_margin`.
    pub fn with_min_retention_perplexity_margin(mut self, min_margin: f32) -> PureResult<Self> {
        validate_regression_limit(
            "sparse_finetune_min_retention_perplexity_margin",
            min_margin,
        )?;
        self.min_retention_perplexity_margin = Some(min_margin);
        Ok(self)
    }

    /// Requires the compact report status to stay unchanged.
    pub fn with_status_match_required(mut self, require_status_match: bool) -> Self {
        self.require_status_match = require_status_match;
        self
    }

    /// Requires sparse fine-tune guard acceptance to stay unchanged.
    pub fn with_accepted_match_required(mut self, require_accepted_match: bool) -> Self {
        self.require_accepted_match = require_accepted_match;
        self
    }

    /// Requires sparse retention guard settings to stay unchanged.
    pub fn with_guard_match_required(mut self, require_guard_match: bool) -> Self {
        self.require_guard_match = require_guard_match;
        self
    }

    /// Requires parameter movement audit tolerance to stay unchanged.
    pub fn with_movement_tolerance_match_required(
        mut self,
        require_movement_tolerance_match: bool,
    ) -> Self {
        self.require_movement_tolerance_match = require_movement_tolerance_match;
        self
    }

    /// Requires FT-ready trainer/module resume fingerprints to stay unchanged.
    pub fn with_resume_match_required(mut self, require_resume_match: bool) -> Self {
        self.require_resume_match = require_resume_match;
        self
    }

    fn validate(self) -> PureResult<Self> {
        if let Some(value) = self.max_target_loss_regression {
            validate_regression_limit("sparse_finetune_max_target_loss_regression", value)?;
        }
        if let Some(value) = self.max_retention_loss_regression {
            validate_regression_limit("sparse_finetune_max_retention_loss_regression", value)?;
        }
        if let Some(value) = self.max_target_retention_gap_regression {
            validate_regression_limit(
                "sparse_finetune_max_target_retention_gap_regression",
                value,
            )?;
        }
        if let Some(value) = self.max_target_retention_ratio_regression {
            validate_regression_limit(
                "sparse_finetune_max_target_retention_ratio_regression",
                value,
            )?;
        }
        if let Some(value) = self.min_target_loss_margin {
            validate_regression_limit("sparse_finetune_min_target_loss_margin", value)?;
        }
        if let Some(value) = self.min_target_retention_ratio {
            validate_regression_limit("sparse_finetune_min_target_retention_ratio", value)?;
        }
        if let Some(value) = self.min_retention_loss_margin {
            validate_regression_limit("sparse_finetune_min_retention_loss_margin", value)?;
        }
        if let Some(value) = self.min_retention_accuracy_margin {
            validate_regression_limit("sparse_finetune_min_retention_accuracy_margin", value)?;
        }
        if let Some(value) = self.min_retention_perplexity_margin {
            validate_regression_limit("sparse_finetune_min_retention_perplexity_margin", value)?;
        }
        Ok(self)
    }
}

fn validate_regression_limit(label: &'static str, value: f32) -> PureResult<()> {
    if value < 0.0 || !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(())
}

fn summary_float_changed(current: f32, baseline: f32) -> bool {
    (current - baseline).abs() > f32::EPSILON
}

fn optional_summary_float_changed(current: Option<f32>, baseline: Option<f32>) -> bool {
    match (current, baseline) {
        (Some(current), Some(baseline)) => summary_float_changed(current, baseline),
        (None, None) => false,
        _ => true,
    }
}

fn optional_summary_float_change(current: Option<f32>, baseline: Option<f32>) -> Option<f32> {
    match (current, baseline) {
        (Some(current), Some(baseline)) => Some(current - baseline),
        _ => None,
    }
}

fn target_retention_ratio(target_loss_delta: f32, retention_loss_delta: f32) -> Option<f32> {
    if retention_loss_delta > 0.0 {
        let ratio = target_loss_delta / retention_loss_delta;
        if ratio.is_finite() {
            Some(ratio)
        } else {
            None
        }
    } else {
        None
    }
}

fn guard_epoch_rate(count: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        count as f32 / total as f32
    }
}

impl SparseFineTuneReportSummary {
    /// Compares this summary to a baseline using the same sign convention as
    /// sparse metric deltas: positive loss deltas are improvements, so a smaller
    /// current delta is a regression.
    pub fn compare_to(
        &self,
        baseline: &Self,
        limits: SparseFineTuneRegressionLimits,
    ) -> PureResult<SparseFineTuneSummaryComparison> {
        let limits = limits.validate()?;
        let target_loss_delta_change = self.target_loss_delta - baseline.target_loss_delta;
        let retention_loss_delta_change = self.retention_loss_delta - baseline.retention_loss_delta;
        let target_retention_gap_change = self.target_retention_gap - baseline.target_retention_gap;
        let target_retention_ratio_change = optional_summary_float_change(
            self.target_retention_ratio,
            baseline.target_retention_ratio,
        );
        if !target_loss_delta_change.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "sparse_finetune_target_loss_delta_change",
                value: target_loss_delta_change,
            });
        }
        if !retention_loss_delta_change.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "sparse_finetune_retention_loss_delta_change",
                value: retention_loss_delta_change,
            });
        }
        if !target_retention_gap_change.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "sparse_finetune_target_retention_gap_change",
                value: target_retention_gap_change,
            });
        }
        if let Some(value) = target_retention_ratio_change {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "sparse_finetune_target_retention_ratio_change",
                    value,
                });
            }
        }
        for (label, value) in [
            (
                "sparse_finetune_target_loss_margin",
                self.target_loss_margin,
            ),
            (
                "sparse_finetune_target_retention_gap",
                self.target_retention_gap,
            ),
            (
                "sparse_finetune_retention_loss_margin",
                self.retention_loss_margin,
            ),
            (
                "sparse_finetune_retention_accuracy_margin",
                self.retention_accuracy_margin,
            ),
        ] {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue { label, value });
            }
        }
        if let Some(value) = self.target_retention_ratio {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "sparse_finetune_target_retention_ratio",
                    value,
                });
            }
        }
        if let Some(value) = self.retention_perplexity_margin {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "sparse_finetune_retention_perplexity_margin",
                    value,
                });
            }
        }
        let target_loss_regression = (-target_loss_delta_change).max(0.0);
        let retention_loss_regression = (-retention_loss_delta_change).max(0.0);
        let target_retention_gap_regression = (-target_retention_gap_change).max(0.0);
        let target_retention_ratio_regression =
            target_retention_ratio_change.map(|change| (-change).max(0.0));
        let target_loss_margin_shortfall = limits
            .min_target_loss_margin
            .map_or(0.0, |floor| (floor - self.target_loss_margin).max(0.0));
        let target_retention_ratio_shortfall = limits.min_target_retention_ratio.map(|floor| {
            self.target_retention_ratio
                .map_or(floor, |ratio| (floor - ratio).max(0.0))
        });
        let retention_loss_margin_shortfall = limits
            .min_retention_loss_margin
            .map_or(0.0, |floor| (floor - self.retention_loss_margin).max(0.0));
        let retention_accuracy_margin_shortfall =
            limits.min_retention_accuracy_margin.map_or(0.0, |floor| {
                (floor - self.retention_accuracy_margin).max(0.0)
            });
        let retention_perplexity_margin_shortfall =
            limits.min_retention_perplexity_margin.map(|floor| {
                self.retention_perplexity_margin
                    .map_or(floor, |margin| (floor - margin).max(0.0))
            });
        let status_changed = self.status != baseline.status;
        let accepted_changed = self.accepted != baseline.accepted;
        let guard_changed =
            summary_float_changed(
                self.retention_max_loss_increase,
                baseline.retention_max_loss_increase,
            ) || summary_float_changed(
                self.retention_max_accuracy_drop,
                baseline.retention_max_accuracy_drop,
            ) || optional_summary_float_changed(
                self.retention_max_perplexity_increase,
                baseline.retention_max_perplexity_increase,
            ) || summary_float_changed(self.target_min_loss_delta, baseline.target_min_loss_delta);
        let movement_tolerance_changed =
            summary_float_changed(self.movement_tolerance, baseline.movement_tolerance);
        let resume_changed = self.resume_hash != baseline.resume_hash;
        let target_ok = limits
            .max_target_loss_regression
            .map_or(true, |limit| target_loss_regression <= limit);
        let retention_ok = limits
            .max_retention_loss_regression
            .map_or(true, |limit| retention_loss_regression <= limit);
        let target_retention_gap_ok = limits
            .max_target_retention_gap_regression
            .map_or(true, |limit| target_retention_gap_regression <= limit);
        let target_retention_ratio_ok =
            limits
                .max_target_retention_ratio_regression
                .map_or(true, |limit| {
                    target_retention_ratio_regression
                        .map_or(false, |regression| regression <= limit)
                });
        let target_margin_ok = limits
            .min_target_loss_margin
            .map_or(true, |floor| self.target_loss_margin >= floor);
        let target_retention_ratio_floor_ok =
            limits.min_target_retention_ratio.map_or(true, |floor| {
                self.target_retention_ratio
                    .map_or(false, |ratio| ratio >= floor)
            });
        let retention_loss_margin_ok = limits
            .min_retention_loss_margin
            .map_or(true, |floor| self.retention_loss_margin >= floor);
        let retention_accuracy_margin_ok = limits
            .min_retention_accuracy_margin
            .map_or(true, |floor| self.retention_accuracy_margin >= floor);
        let retention_perplexity_margin_ok =
            limits
                .min_retention_perplexity_margin
                .map_or(true, |floor| {
                    self.retention_perplexity_margin
                        .map_or(false, |margin| margin >= floor)
                });
        let status_ok = !limits.require_status_match || !status_changed;
        let accepted_ok = !limits.require_accepted_match || !accepted_changed;
        let guard_ok = !limits.require_guard_match || !guard_changed;
        let movement_tolerance_ok =
            !limits.require_movement_tolerance_match || !movement_tolerance_changed;
        let resume_ok = !limits.require_resume_match || !resume_changed;
        Ok(SparseFineTuneSummaryComparison {
            target_loss_delta_change,
            retention_loss_delta_change,
            target_retention_gap_change,
            target_retention_ratio_change,
            target_loss_regression,
            retention_loss_regression,
            target_retention_gap_regression,
            target_retention_ratio_regression,
            target_loss_margin_shortfall,
            target_retention_ratio_shortfall,
            retention_loss_margin_shortfall,
            retention_accuracy_margin_shortfall,
            retention_perplexity_margin_shortfall,
            status_changed,
            accepted_changed,
            guard_changed,
            movement_tolerance_changed,
            resume_changed,
            passed: target_ok
                && retention_ok
                && target_retention_gap_ok
                && target_retention_ratio_ok
                && target_margin_ok
                && target_retention_ratio_floor_ok
                && retention_loss_margin_ok
                && retention_accuracy_margin_ok
                && retention_perplexity_margin_ok
                && status_ok
                && accepted_ok
                && guard_ok
                && movement_tolerance_ok
                && resume_ok,
        })
    }
}

impl SparseFineTuneReport {
    /// Returns true when the guard installed a post-train epoch.
    pub fn accepted(&self) -> bool {
        self.accepted
    }

    /// Returns true when the restored state improved target mean loss.
    pub fn target_loss_improved(&self) -> bool {
        self.target_delta.loss_delta > 0.0
    }

    /// Returns true when frozen parameters stayed stable and trainable
    /// parameters moved after an accepted FT epoch.
    pub fn movement_ok(&self) -> bool {
        self.movement.frozen_stable()
            && (!self.accepted || self.movement.trainable_movement_observed())
    }

    /// Compact status label for FT harnesses and CI logs.
    pub fn status(&self) -> &'static str {
        if !self.accepted {
            "guard_rejected"
        } else if !self.movement.frozen_stable() {
            "frozen_changed"
        } else if !self.movement.trainable_movement_observed() {
            "no_trainable_movement"
        } else if !self.target_loss_improved() {
            "target_not_improved"
        } else {
            "ok"
        }
    }

    /// Returns a flat, persistence-friendly digest of the report.
    pub fn summary(&self) -> SparseFineTuneReportSummary {
        let guard_epochs_run = self.captured.train_summary.epochs;
        SparseFineTuneReportSummary {
            status: self.status(),
            accepted: self.accepted(),
            target_loss_improved: self.target_loss_improved(),
            movement_ok: self.movement_ok(),
            guarded_best_epoch: self.captured.guarded_best_epoch,
            guard_epochs_run,
            guard_accepted_epochs: self.captured.guard_accepted_epochs,
            guard_retention_rejected_epochs: self.captured.guard_retention_rejected_epochs,
            guard_target_stale_epochs: self.captured.guard_target_stale_epochs,
            guard_acceptance_rate: guard_epoch_rate(
                self.captured.guard_accepted_epochs,
                guard_epochs_run,
            ),
            guard_retention_rejected_rate: guard_epoch_rate(
                self.captured.guard_retention_rejected_epochs,
                guard_epochs_run,
            ),
            guard_target_stale_rate: guard_epoch_rate(
                self.captured.guard_target_stale_epochs,
                guard_epochs_run,
            ),
            epochs_run: self.captured.train_summary.epochs,
            train_rows: self.captured.train_summary.rows,
            train_batches: self.captured.train_summary.batches,
            optimizer_steps: self.captured.train_summary.optimizer_steps,
            target_loss_delta: self.target_delta.loss_delta,
            target_accuracy_delta: self.target_delta.accuracy_delta,
            target_perplexity_delta: self.target_delta.perplexity_delta,
            retention_loss_delta: self.retention_delta.loss_delta,
            retention_accuracy_delta: self.retention_delta.accuracy_delta,
            retention_perplexity_delta: self.retention_delta.perplexity_delta,
            target_retention_gap: self.target_delta.loss_delta - self.retention_delta.loss_delta,
            target_retention_ratio: target_retention_ratio(
                self.target_delta.loss_delta,
                self.retention_delta.loss_delta,
            ),
            best_retention_loss_increase: self.captured.best_retention_loss_increase,
            best_retention_accuracy_drop: self.captured.best_retention_accuracy_drop,
            best_retention_perplexity_increase: self.captured.best_retention_perplexity_increase,
            retention_max_loss_increase: self.captured.retention_guard.max_loss_increase,
            retention_max_accuracy_drop: self.captured.retention_guard.max_accuracy_drop,
            retention_max_perplexity_increase: self
                .captured
                .retention_guard
                .max_perplexity_increase,
            target_min_loss_delta: self.captured.retention_guard.target_min_loss_delta,
            target_loss_margin: self.target_delta.loss_delta
                - self.captured.retention_guard.target_min_loss_delta,
            retention_loss_margin: self.captured.retention_guard.max_loss_increase
                - self.captured.best_retention_loss_increase,
            retention_accuracy_margin: self.captured.retention_guard.max_accuracy_drop
                - self.captured.best_retention_accuracy_drop,
            retention_perplexity_margin: self
                .captured
                .retention_guard
                .max_perplexity_increase
                .map(|ceiling| ceiling - self.captured.best_retention_perplexity_increase),
            max_allowed_retention_loss: self.captured.max_allowed_retention_loss,
            min_allowed_retention_accuracy: self.captured.min_allowed_retention_accuracy,
            max_allowed_retention_perplexity: self.captured.max_allowed_retention_perplexity,
            movement_status: self.movement.status(),
            frozen_stable: self.movement.frozen_stable(),
            trainable_movement_observed: self.movement.trainable_movement_observed(),
            movement_tolerance: self.movement_tolerance,
            trainable_changed: self.movement.trainable_changed,
            frozen_changed: self.movement.frozen_changed,
            max_trainable_l2_delta: self.movement.max_trainable_l2_delta,
            max_frozen_l2_delta: self.movement.max_frozen_l2_delta,
            resume_hash: self.resume_fingerprint.hash.clone(),
            resume_trainer_hash: self.resume_fingerprint.trainer.hash.clone(),
            resume_parameter_hash: self.resume_fingerprint.parameters.hash.clone(),
            resume_parameter_training_hash: self.resume_fingerprint.parameter_training.hash.clone(),
            resume_trainable: self.resume_fingerprint.parameter_training.trainable,
            resume_frozen: self.resume_fingerprint.parameter_training.frozen,
            resume_hypergrad_tapes: self.resume_fingerprint.parameter_training.hypergrad_tapes,
            resume_gradient_accumulation_steps: self
                .resume_fingerprint
                .trainer
                .gradient_accumulation_steps,
            resume_runtime_hooks: self.resume_fingerprint.trainer.runtime_hooks,
            best_hash: self.captured.best_fingerprint.hash.clone(),
            final_hash: self.captured.final_fingerprint.hash.clone(),
            best_differs_from_final: self.captured.best_differs_from_final,
        }
    }
}

/// Helper trait that allows [`ModuleTrainer::train_epoch`] to accept both raw
/// `(Tensor, Tensor)` batches and fallible [`PureResult`] batches produced by
/// the [`dataset::DataLoader`] surface.
pub trait IntoBatch {
    fn into_batch(self) -> PureResult<(Tensor, Tensor)>;
}

impl IntoBatch for (Tensor, Tensor) {
    fn into_batch(self) -> PureResult<(Tensor, Tensor)> {
        Ok(self)
    }
}

impl IntoBatch for PureResult<(Tensor, Tensor)> {
    fn into_batch(self) -> PureResult<(Tensor, Tensor)> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use crate::layers::sequential::Sequential;
    use crate::layers::wave_gate::WaveGate;
    use crate::loss::{Loss, MeanSquaredError, SoftmaxCrossEntropy};
    use crate::schedule::RoundtableConfig;
    use st_tensor::pure::topos::OpenCartesianTopos;

    struct NonFiniteLoss;

    impl Loss for NonFiniteLoss {
        fn forward(&mut self, _prediction: &Tensor, _target: &Tensor) -> PureResult<Tensor> {
            Tensor::from_vec(1, 1, vec![f32::NAN])
        }

        fn backward(&mut self, prediction: &Tensor, _target: &Tensor) -> PureResult<Tensor> {
            Tensor::zeros(prediction.shape().0, prediction.shape().1)
        }
    }

    struct IdentityModule;

    impl Module for IdentityModule {
        fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
            Ok(input.clone())
        }

        fn backward(&mut self, input: &Tensor, _grad_output: &Tensor) -> PureResult<Tensor> {
            Tensor::zeros(input.shape().0, input.shape().1)
        }

        fn visit_parameters(
            &self,
            _visitor: &mut dyn FnMut(&crate::Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            Ok(())
        }

        fn visit_parameters_mut(
            &mut self,
            _visitor: &mut dyn FnMut(&mut crate::Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            Ok(())
        }
    }

    struct RowCountLoss;

    impl Loss for RowCountLoss {
        fn forward(&mut self, prediction: &Tensor, _target: &Tensor) -> PureResult<Tensor> {
            Tensor::from_vec(1, 1, vec![prediction.shape().0 as f32])
        }

        fn backward(&mut self, prediction: &Tensor, _target: &Tensor) -> PureResult<Tensor> {
            Tensor::zeros(prediction.shape().0, prediction.shape().1)
        }
    }

    struct ScriptedLoss {
        values: Vec<f32>,
        cursor: usize,
        grad_value: f32,
    }

    impl ScriptedLoss {
        fn new(values: Vec<f32>) -> Self {
            Self {
                values,
                cursor: 0,
                grad_value: 0.0,
            }
        }

        fn with_grad(values: Vec<f32>, grad_value: f32) -> Self {
            Self {
                values,
                cursor: 0,
                grad_value,
            }
        }
    }

    impl Loss for ScriptedLoss {
        fn forward(&mut self, _prediction: &Tensor, _target: &Tensor) -> PureResult<Tensor> {
            let Some(value) = self.values.get(self.cursor).copied() else {
                return Err(TensorError::IoError {
                    message: "scripted loss exhausted".to_string(),
                });
            };
            self.cursor += 1;
            Tensor::from_vec(1, 1, vec![value])
        }

        fn backward(&mut self, prediction: &Tensor, _target: &Tensor) -> PureResult<Tensor> {
            let (rows, cols) = prediction.shape();
            Tensor::from_vec(rows, cols, vec![self.grad_value; rows * cols])
        }
    }

    #[test]
    fn trainer_attaches_and_steps() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("fc", 2, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let input = crate::Tensor::from_vec(1, 2, vec![1.0, -1.0]).unwrap();
        let target = crate::Tensor::from_vec(1, 1, vec![0.5]).unwrap();
        let out = layer.forward(&input).unwrap();
        let grad = out.sub(&target).unwrap();
        let _ = layer.backward(&input, &grad).unwrap();
        trainer.step(&mut layer).unwrap();
        assert!(trainer.planner().topk(64, 128, 32).k > 0);
    }

    #[test]
    fn resume_fingerprint_detects_training_metadata_mismatch() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut source = Linear::new("resume", 2, 1).unwrap();
        trainer.prepare(&mut source).unwrap();
        source
            .set_parameters_trainable_by_suffix("::weight", false)
            .unwrap();
        source
            .scale_parameters_learning_rate_by_suffix("::bias", 1.1)
            .unwrap();
        source
            .set_parameters_weight_decay_by_suffix("::bias", 0.01)
            .unwrap();
        let state = source.state_dict().unwrap();
        let expected = trainer.resume_fingerprint(&source).unwrap();

        let mut target = Linear::new("resume", 2, 1).unwrap();
        trainer.prepare(&mut target).unwrap();
        let load = target.load_state_dict_checked(&state).unwrap();
        assert!(load.matched);
        let missing_metadata = trainer
            .audit_resume_fingerprint(&target, &expected)
            .unwrap();
        assert!(missing_metadata.parameters_matched);
        assert!(!missing_metadata.parameter_training_matched);
        assert!(!missing_metadata.matched);

        target
            .set_parameters_trainable_by_suffix("::weight", false)
            .unwrap();
        target
            .scale_parameters_learning_rate_by_suffix("::bias", 1.1)
            .unwrap();
        let missing_decay = trainer
            .audit_resume_fingerprint(&target, &expected)
            .unwrap();
        assert!(missing_decay.parameters_matched);
        assert!(!missing_decay.parameter_training_matched);
        assert!(!missing_decay.matched);

        target
            .set_parameters_weight_decay_by_suffix("::bias", 0.01)
            .unwrap();
        let restored = trainer
            .audit_resume_fingerprint(&target, &expected)
            .unwrap();
        assert!(restored.trainer_matched);
        assert!(restored.parameters_matched);
        assert!(restored.parameter_training_matched);
        assert!(restored.matched);
    }

    #[test]
    fn trainer_prepares_with_topos_for_wave_gate() {
        let caps = DeviceCaps::wgpu(64, true, 512);
        let trainer = ModuleTrainer::new(caps, -0.9, 0.06, 0.02);
        let encoder_curvature = trainer.curvature();
        let topos = OpenCartesianTopos::new(encoder_curvature, 1e-6, 1e4, 512, 16384).unwrap();
        let mut gate = WaveGate::with_topos(
            "wg",
            8,
            st_tensor::pure::LanguageWaveEncoder::new(encoder_curvature, 0.7).unwrap(),
            topos.clone(),
        )
        .unwrap();
        trainer.prepare_with_topos(&mut gate, topos).unwrap();
        let input =
            Tensor::from_vec(1, 8, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]).unwrap();
        let grad_out = gate.forward(&input).unwrap();
        let _ = gate.backward(&input, &grad_out).unwrap();
        trainer.step(&mut gate).unwrap();
        assert!(gate.gate().value().squared_l2_norm() > 0.0);
    }

    #[test]
    fn trainer_runs_epoch_with_roundtable_schedule() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.1, 0.05, 0.01);
        let mut model = Sequential::new();
        model.push(Linear::new("lin", 2, 1).unwrap());
        trainer.prepare(&mut model).unwrap();

        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![
            (
                Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
        ];

        let mut loss = MeanSquaredError::new();
        let stats = trainer
            .train_epoch(&mut model, &mut loss, dataset.clone(), &schedule)
            .unwrap();
        assert_eq!(stats.batches, dataset.len());
        assert_eq!(stats.optimizer_steps, dataset.len());
        assert_eq!(stats.rows, dataset.len());
        assert!(stats.total_loss.is_finite());
        assert!(stats.average_loss_per_row.is_finite());

        // Ensure the model parameters changed by running another batch and checking the outputs.
        let input = Tensor::from_vec(1, 2, vec![1.0, 1.0]).unwrap();
        let before = model.forward(&input).unwrap();
        trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .unwrap();
        let after = model.forward(&input).unwrap();
        assert_ne!(before.data(), after.data());
    }

    #[test]
    fn train_epoch_keeps_frozen_parameters_stable() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("ft", 2, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        layer.set_parameter_trainable("ft::weight", false).unwrap();

        let weight_before = layer.weight().value().clone();
        let bias_before = layer.bias().value().clone();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![(
            Tensor::from_vec(1, 2, vec![1.0, -1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.75]).unwrap(),
        )];
        let mut loss = MeanSquaredError::new();

        trainer
            .train_epoch(&mut layer, &mut loss, dataset, &schedule)
            .unwrap();

        assert_eq!(layer.weight().value(), &weight_before);
        assert_ne!(layer.bias().value(), &bias_before);
    }

    #[test]
    fn train_epoch_reports_row_weighted_loss() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("rows", 2, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let schedule = trainer.roundtable(3, 1, RoundtableConfig::default());
        let batches = vec![
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
            (
                Tensor::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap(),
                Tensor::from_vec(3, 1, vec![0.0, 0.0, 0.0]).unwrap(),
            ),
        ];
        let mut loss = RowCountLoss;
        let stats = trainer
            .train_epoch(&mut layer, &mut loss, batches, &schedule)
            .unwrap();

        assert_eq!(stats.batches, 2);
        assert_eq!(stats.optimizer_steps, 2);
        assert_eq!(stats.rows, 4);
        assert!((stats.total_loss - 4.0).abs() < 1e-6);
        assert!((stats.average_loss - 2.0).abs() < 1e-6);
        assert!((stats.total_row_weighted_loss - 10.0).abs() < 1e-6);
        assert!((stats.average_loss_per_row - 2.5).abs() < 1e-6);
    }

    #[test]
    fn train_epoch_uses_loss_reduction_rows_for_padding_masks() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("masked_ce", 2, 2).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let schedule = trainer.roundtable(2, 2, RoundtableConfig::default());
        let input = Tensor::from_vec(2, 2, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let target = Tensor::from_vec(2, 1, vec![0.0, -1.0]).unwrap();
        let mut loss = SoftmaxCrossEntropy::with_ignore_index(-1);

        let stats = trainer
            .train_epoch(&mut layer, &mut loss, vec![(input, target)], &schedule)
            .unwrap();

        assert_eq!(stats.rows, 1);
        assert!(stats.average_loss_per_row.is_finite());
        assert!((stats.average_loss - stats.average_loss_per_row).abs() < 1e-6);
    }

    #[test]
    fn train_epoch_accumulates_gradients_before_optimizer_step() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        trainer.set_gradient_accumulation_steps(2).unwrap();
        assert_eq!(trainer.gradient_accumulation_steps(), 2);
        let mut layer = Linear::new("accum", 2, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let before = layer.state_dict().unwrap();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![1.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.5]).unwrap(),
            ),
        ];
        let mut loss = MeanSquaredError::new();
        let stats = trainer
            .train_epoch(&mut layer, &mut loss, dataset, &schedule)
            .unwrap();

        assert_eq!(stats.batches, 3);
        assert_eq!(stats.optimizer_steps, 2);
        assert_eq!(stats.rows, 3);
        assert_ne!(layer.state_dict().unwrap(), before);
    }

    #[test]
    fn train_epoch_rescales_partial_gradient_accumulation_flush() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut single_step = ModuleTrainer::new(caps.clone(), -1.0, 0.0, 0.1);
        let mut accumulated = ModuleTrainer::new(caps, -1.0, 0.0, 0.1);
        accumulated.set_gradient_accumulation_steps(2).unwrap();
        let mut single_layer = Linear::new("partial", 2, 1).unwrap();
        let mut accumulated_layer = Linear::new("partial", 2, 1).unwrap();
        let schedule = single_step.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![(
            Tensor::from_vec(1, 2, vec![1.0, -0.5]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.25]).unwrap(),
        )];
        let mut single_loss = MeanSquaredError::new();
        let mut accumulated_loss = MeanSquaredError::new();

        let single_stats = single_step
            .train_epoch(
                &mut single_layer,
                &mut single_loss,
                dataset.clone(),
                &schedule,
            )
            .unwrap();
        let accumulated_stats = accumulated
            .train_epoch(
                &mut accumulated_layer,
                &mut accumulated_loss,
                dataset,
                &schedule,
            )
            .unwrap();

        assert_eq!(single_stats.optimizer_steps, 1);
        assert_eq!(accumulated_stats.optimizer_steps, 1);
        assert_eq!(
            single_layer.state_dict().unwrap(),
            accumulated_layer.state_dict().unwrap()
        );
    }

    #[test]
    fn gradient_accumulation_steps_rejects_zero() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let err = trainer.set_gradient_accumulation_steps(0).unwrap_err();
        assert!(matches!(err, TensorError::IoError { .. }));
        assert_eq!(trainer.gradient_accumulation_steps(), 1);
    }

    #[test]
    fn evaluate_epoch_reports_loss_without_updating_parameters() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("eval", 2, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let before = layer.state_dict().unwrap();
        let batches = vec![
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
            (
                Tensor::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap(),
                Tensor::from_vec(3, 1, vec![0.0, 0.0, 0.0]).unwrap(),
            ),
        ];
        let mut loss = RowCountLoss;
        let stats = trainer.evaluate_epoch(&layer, &mut loss, batches).unwrap();

        assert_eq!(stats.batches, 2);
        assert_eq!(stats.optimizer_steps, 0);
        assert_eq!(stats.rows, 4);
        assert!((stats.total_loss - 4.0).abs() < 1e-6);
        assert!((stats.average_loss - 2.0).abs() < 1e-6);
        assert!((stats.total_row_weighted_loss - 10.0).abs() < 1e-6);
        assert!((stats.average_loss_per_row - 2.5).abs() < 1e-6);
        assert_eq!(layer.state_dict().unwrap(), before);
    }

    #[test]
    fn evaluate_sparse_classification_epoch_reports_active_metrics() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let module = IdentityModule;
        let loss = SoftmaxCrossEntropy::with_ignore_index(-1);
        let batches = vec![
            (
                Tensor::from_vec(
                    3,
                    3,
                    vec![4.0, 1.0, 0.0, 0.0, 3.0, 1.0, f32::NAN, f32::NAN, f32::NAN],
                )
                .unwrap(),
                Tensor::from_vec(3, 1, vec![0.0, 2.0, -1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 3, vec![0.0, 5.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
        ];

        let metrics = trainer
            .evaluate_sparse_classification_epoch(&module, &loss, batches)
            .unwrap();

        assert_eq!(metrics.active_rows, 3);
        assert_eq!(metrics.correct, 2);
        assert!((metrics.accuracy - (2.0 / 3.0)).abs() < 1e-6);
        assert!(metrics.mean_loss.is_finite());
        assert!((metrics.perplexity - metrics.mean_loss.exp()).abs() < 1e-6);
    }

    #[test]
    fn train_epochs_reuses_loader_and_records_history() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("epochs", 2, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let schedule = trainer.roundtable(2, 1, RoundtableConfig::default());
        let loader = crate::dataset_from_vec(vec![
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
        ])
        .batched(2);
        let mut loss = MeanSquaredError::new();
        let history = trainer
            .train_epochs(&mut layer, &mut loss, loader, &schedule, 3)
            .unwrap();

        assert_eq!(history.len(), 3);
        for stats in &history {
            assert_eq!(stats.batches, 1);
            assert_eq!(stats.optimizer_steps, 1);
            assert_eq!(stats.rows, 2);
            assert!(stats.average_loss_per_row.is_finite());
        }
    }

    #[test]
    fn summarize_epoch_history_reports_best_and_improvement() {
        let history = vec![
            EpochStats {
                batches: 2,
                optimizer_steps: 2,
                rows: 8,
                total_loss: 4.0,
                total_row_weighted_loss: 32.0,
                average_loss: 2.0,
                average_loss_per_row: 4.0,
            },
            EpochStats {
                batches: 2,
                optimizer_steps: 2,
                rows: 8,
                total_loss: 2.0,
                total_row_weighted_loss: 16.0,
                average_loss: 1.0,
                average_loss_per_row: 2.0,
            },
            EpochStats {
                batches: 2,
                optimizer_steps: 2,
                rows: 8,
                total_loss: 3.0,
                total_row_weighted_loss: 24.0,
                average_loss: 1.5,
                average_loss_per_row: 3.0,
            },
        ];

        let summary = summarize_epoch_history(&history);
        assert_eq!(summary.epochs, 3);
        assert_eq!(summary.batches, 6);
        assert_eq!(summary.optimizer_steps, 6);
        assert_eq!(summary.rows, 24);
        assert_eq!(summary.best_epoch, Some(2));
        assert!((summary.initial_loss_per_row - 4.0).abs() < 1e-6);
        assert!((summary.final_loss_per_row - 3.0).abs() < 1e-6);
        assert!((summary.best_loss_per_row - 2.0).abs() < 1e-6);
        assert!((summary.final_improvement - 1.0).abs() < 1e-6);
        assert!((summary.best_improvement - 2.0).abs() < 1e-6);
        assert!(summary.improved);
        assert!(summary.best_improved);

        let empty = summarize_epoch_history(&[]);
        assert_eq!(empty.epochs, 0);
        assert_eq!(empty.best_epoch, None);
        assert!(!empty.improved);
    }

    #[test]
    fn train_epochs_restore_best_restores_best_state() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("best", 2, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let schedule = trainer.roundtable(2, 1, RoundtableConfig::default());
        let loader = crate::dataset_from_vec(vec![
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
        ])
        .batched(2);
        let mut loss = MeanSquaredError::new();
        let captured = trainer
            .train_epochs_restore_best(&mut layer, &mut loss, loader, &schedule, 3)
            .unwrap();
        let restored = layer.state_fingerprint().unwrap();

        assert_eq!(captured.history.len(), 3);
        assert_eq!(captured.summary.epochs, 3);
        assert_eq!(captured.summary.best_epoch.is_some(), true);
        assert_eq!(restored, captured.best_fingerprint);
        assert_eq!(
            crate::module::fingerprint_state_dict(&captured.best_state),
            captured.best_fingerprint
        );
        assert_eq!(
            crate::module::fingerprint_state_dict(&captured.final_state),
            captured.final_fingerprint
        );
    }

    #[test]
    fn validation_best_selection_uses_validation_history() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("validation_best", 1, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let train_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let validation_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let mut loss = ScriptedLoss::new(vec![10.0, 3.0, 2.0, 1.0, 1.0, 2.0]);

        let captured = trainer
            .train_epochs_restore_best_on_validation(
                &mut layer,
                &mut loss,
                train_loader,
                validation_loader,
                &schedule,
                3,
            )
            .unwrap();

        assert_eq!(captured.train_history.len(), 3);
        assert_eq!(captured.validation_history.len(), 3);
        assert_eq!(captured.train_summary.best_epoch, Some(3));
        assert_eq!(captured.validation_summary.best_epoch, Some(2));
        assert!((captured.train_summary.best_loss_per_row - 1.0).abs() < 1e-6);
        assert!((captured.validation_summary.best_loss_per_row - 1.0).abs() < 1e-6);
        assert_eq!(
            layer.state_fingerprint().unwrap(),
            captured.best_fingerprint
        );
    }

    #[test]
    fn retention_guard_selects_best_non_forgetting_epoch() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("retention_best", 1, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let train_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let validation_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let retention_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![0.5]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.5]).unwrap(),
        )])
        .batched(1);
        let mut loss = ScriptedLoss::new(vec![
            5.0,  // pre-FT retention baseline
            10.0, // epoch 1 train
            4.0,  // epoch 1 target validation
            5.05, // epoch 1 retention: guarded candidate
            9.0,  // epoch 2 train
            3.0,  // epoch 2 target validation: best target but forgets
            5.5,  // epoch 2 retention: rejected
            8.0,  // epoch 3 train
            3.5,  // epoch 3 target validation: next best target
            5.04, // epoch 3 retention: guarded winner
        ]);

        let captured = trainer
            .train_epochs_capture_best_with_retention_guard(
                &mut layer,
                &mut loss,
                train_loader,
                validation_loader,
                retention_loader,
                &schedule,
                3,
                RetentionGuardConfig::new(0.1, 0.0).unwrap(),
            )
            .unwrap();

        assert_eq!(captured.train_history.len(), 3);
        assert_eq!(captured.validation_history.len(), 3);
        assert_eq!(captured.retention_history.len(), 3);
        assert_eq!(captured.validation_summary.best_epoch, Some(2));
        assert_eq!(captured.guarded_best_epoch, Some(3));
        assert!((captured.retention_baseline.average_loss_per_row - 5.0).abs() < 1e-6);
        assert!((captured.max_allowed_retention_loss_per_row - 5.1).abs() < 1e-6);
        assert!((captured.best_validation_loss_per_row - 3.5).abs() < 1e-6);
        assert!((captured.best_retention_loss_per_row - 5.04).abs() < 1e-6);
        assert!((captured.best_retention_loss_increase - 0.04).abs() < 1e-6);
    }

    #[test]
    fn retention_guard_restores_initial_state_when_every_epoch_forgets() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("retention_initial", 1, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let initial_fingerprint = layer.state_fingerprint().unwrap();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let train_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let validation_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let retention_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![0.5]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.5]).unwrap(),
        )])
        .batched(1);
        let mut loss = ScriptedLoss::with_grad(
            vec![
                5.0,  // pre-FT retention baseline
                10.0, // epoch 1 train
                4.0,  // epoch 1 target validation
                5.2,  // epoch 1 retention: rejected
                9.0,  // epoch 2 train
                3.0,  // epoch 2 target validation
                5.3,  // epoch 2 retention: rejected
            ],
            1.0,
        );

        let captured = trainer
            .train_epochs_restore_best_with_retention_guard(
                &mut layer,
                &mut loss,
                train_loader,
                validation_loader,
                retention_loader,
                &schedule,
                2,
                RetentionGuardConfig::new(0.1, 0.0).unwrap(),
            )
            .unwrap();

        assert_eq!(captured.validation_summary.best_epoch, Some(2));
        assert_eq!(captured.guarded_best_epoch, None);
        assert_eq!(captured.best_fingerprint, initial_fingerprint);
        assert_eq!(layer.state_fingerprint().unwrap(), initial_fingerprint);
        assert!(captured.best_differs_from_final);
    }

    #[test]
    fn sparse_retention_guard_rejects_accuracy_forgetting() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.0, 1.0);
        let mut layer = Linear::new("sparse_retention", 1, 2).unwrap();
        layer
            .visit_parameters_mut(&mut |param| {
                for value in param.value_mut().data_mut() {
                    *value = 0.0;
                }
                Ok(())
            })
            .unwrap();
        let initial_fingerprint = layer.state_fingerprint().unwrap();
        let schedule = trainer.roundtable(1, 2, RoundtableConfig::default());
        let train_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let validation_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let retention_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
        )])
        .batched(1);
        let mut loss = SoftmaxCrossEntropy::new();

        let captured = trainer
            .train_epochs_restore_best_sparse_with_retention_guard(
                &mut layer,
                &mut loss,
                train_loader,
                validation_loader,
                retention_loader,
                &schedule,
                1,
                SparseRetentionGuardConfig::new(10.0, 0.0).unwrap(),
            )
            .unwrap();

        assert_eq!(captured.validation_baseline.accuracy, 0.0);
        assert_eq!(captured.retention_baseline.accuracy, 1.0);
        assert_eq!(captured.validation_history.len(), 1);
        assert_eq!(captured.retention_history.len(), 1);
        assert!(captured.validation_history[0].mean_loss < captured.validation_baseline.mean_loss);
        assert_eq!(captured.retention_history[0].accuracy, 0.0);
        assert_eq!(captured.guarded_best_epoch, None);
        assert_eq!(captured.guard_accepted_epochs, 0);
        assert_eq!(captured.guard_retention_rejected_epochs, 1);
        assert_eq!(captured.guard_target_stale_epochs, 0);
        assert_eq!(captured.best_fingerprint, initial_fingerprint);
        assert_eq!(layer.state_fingerprint().unwrap(), initial_fingerprint);
        assert!(captured.best_differs_from_final);
    }

    #[test]
    fn sparse_retention_guard_accepts_accuracy_within_floor() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.0, 1.0);
        let mut layer = Linear::new("sparse_accept", 1, 2).unwrap();
        layer
            .visit_parameters_mut(&mut |param| {
                for value in param.value_mut().data_mut() {
                    *value = 0.0;
                }
                Ok(())
            })
            .unwrap();
        let schedule = trainer.roundtable(1, 2, RoundtableConfig::default());
        let train_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let validation_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let retention_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
        )])
        .batched(1);
        let mut loss = SoftmaxCrossEntropy::new();

        let captured = trainer
            .train_epochs_restore_best_sparse_with_retention_guard(
                &mut layer,
                &mut loss,
                train_loader,
                validation_loader,
                retention_loader,
                &schedule,
                1,
                SparseRetentionGuardConfig::new(10.0, 1.0).unwrap(),
            )
            .unwrap();

        assert_eq!(captured.guarded_best_epoch, Some(1));
        assert_eq!(captured.guard_accepted_epochs, 1);
        assert_eq!(captured.guard_retention_rejected_epochs, 0);
        assert_eq!(captured.guard_target_stale_epochs, 0);
        assert_eq!(captured.best_validation_metrics.accuracy, 1.0);
        assert_eq!(captured.best_retention_metrics.accuracy, 0.0);
        assert!((captured.best_retention_accuracy_drop - 1.0).abs() < 1e-6);
        assert_eq!(
            layer.state_fingerprint().unwrap(),
            captured.best_fingerprint
        );
    }

    #[test]
    fn sparse_retention_guard_counts_target_stale_epochs() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.0, 1.0);
        let mut layer = Linear::new("sparse_stale", 1, 2).unwrap();
        layer
            .visit_parameters_mut(&mut |param| {
                for value in param.value_mut().data_mut() {
                    *value = 0.0;
                }
                Ok(())
            })
            .unwrap();
        let schedule = trainer.roundtable(1, 2, RoundtableConfig::default());
        let train_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let validation_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let retention_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
        )])
        .batched(1);
        let mut loss = SoftmaxCrossEntropy::new();
        let retention_guard = SparseRetentionGuardConfig::new(10.0, 1.0)
            .unwrap()
            .with_target_min_loss_delta(100.0)
            .unwrap();

        let captured = trainer
            .train_epochs_restore_best_sparse_with_retention_guard(
                &mut layer,
                &mut loss,
                train_loader,
                validation_loader,
                retention_loader,
                &schedule,
                1,
                retention_guard,
            )
            .unwrap();

        assert_eq!(captured.guarded_best_epoch, None);
        assert_eq!(captured.guard_accepted_epochs, 0);
        assert_eq!(captured.guard_retention_rejected_epochs, 0);
        assert_eq!(captured.guard_target_stale_epochs, 1);
    }

    #[test]
    fn sparse_finetune_report_audits_restored_deltas_and_movement() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.0, 1.0);
        let mut layer = Linear::new("sparse_report", 1, 2).unwrap();
        layer
            .visit_parameters_mut(&mut |param| {
                for value in param.value_mut().data_mut() {
                    *value = 0.0;
                }
                Ok(())
            })
            .unwrap();
        layer
            .set_parameters_trainable_by_suffix("::weight", false)
            .unwrap();
        let schedule = trainer.roundtable(1, 2, RoundtableConfig::default());
        let train_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let validation_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let retention_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
        )])
        .batched(1);
        let mut loss = SoftmaxCrossEntropy::new();

        let report = trainer
            .train_epochs_restore_best_sparse_with_finetune_report(
                &mut layer,
                &mut loss,
                train_loader,
                validation_loader,
                retention_loader,
                &schedule,
                1,
                SparseRetentionGuardConfig::new(10.0, 1.0).unwrap(),
                1e-8,
            )
            .unwrap();

        assert!(report.accepted());
        assert_eq!(report.status(), "ok");
        assert!(report.target_loss_improved());
        assert!(report.movement_ok());
        assert!(report.target_delta.loss_delta > 0.0);
        assert_eq!(report.target_after, report.captured.best_validation_metrics);
        assert_eq!(
            report.retention_after,
            report.captured.best_retention_metrics
        );
        assert!(report.movement.frozen_stable());
        assert!(report.movement.trainable_movement_observed());
        assert_eq!(
            layer.state_fingerprint().unwrap(),
            report.captured.best_fingerprint
        );
        let summary = report.summary();
        assert_eq!(summary.status, "ok");
        assert!(summary.accepted);
        assert!(summary.target_loss_improved);
        assert!(summary.movement_ok);
        assert_eq!(summary.guarded_best_epoch, Some(1));
        assert_eq!(summary.guard_epochs_run, 1);
        assert_eq!(summary.guard_accepted_epochs, 1);
        assert_eq!(summary.guard_retention_rejected_epochs, 0);
        assert_eq!(summary.guard_target_stale_epochs, 0);
        assert_eq!(summary.guard_acceptance_rate, 1.0);
        assert_eq!(summary.guard_retention_rejected_rate, 0.0);
        assert_eq!(summary.guard_target_stale_rate, 0.0);
        assert_eq!(summary.train_rows, 1);
        assert_eq!(summary.optimizer_steps, 1);
        assert_eq!(summary.target_loss_delta, report.target_delta.loss_delta);
        assert_eq!(
            summary.retention_loss_delta,
            report.retention_delta.loss_delta
        );
        assert!(
            (summary.target_retention_gap
                - (summary.target_loss_delta - summary.retention_loss_delta))
                .abs()
                < 1e-6
        );
        if summary.retention_loss_delta > 0.0 {
            assert_eq!(
                summary.target_retention_ratio,
                Some(summary.target_loss_delta / summary.retention_loss_delta)
            );
        } else {
            assert_eq!(summary.target_retention_ratio, None);
        }
        assert_eq!(summary.retention_max_loss_increase, 10.0);
        assert_eq!(summary.retention_max_accuracy_drop, 1.0);
        assert_eq!(summary.retention_max_perplexity_increase, None);
        assert_eq!(summary.target_min_loss_delta, 0.0);
        assert!((summary.target_loss_margin - summary.target_loss_delta).abs() < 1e-6);
        assert!(
            (summary.retention_loss_margin
                - (summary.retention_max_loss_increase - summary.best_retention_loss_increase))
                .abs()
                < 1e-6
        );
        assert!(
            (summary.retention_accuracy_margin
                - (summary.retention_max_accuracy_drop - summary.best_retention_accuracy_drop))
                .abs()
                < 1e-6
        );
        assert_eq!(summary.retention_perplexity_margin, None);
        assert_eq!(summary.movement_status, "ok");
        assert!(summary.frozen_stable);
        assert!(summary.trainable_movement_observed);
        assert_eq!(summary.movement_tolerance, 1e-8);
        assert_eq!(summary.resume_hash, report.resume_fingerprint.hash);
        assert_eq!(
            summary.resume_trainer_hash,
            report.resume_fingerprint.trainer.hash
        );
        assert_eq!(
            summary.resume_parameter_training_hash,
            report.resume_fingerprint.parameter_training.hash
        );
        assert_eq!(summary.resume_trainable, 1);
        assert_eq!(summary.resume_frozen, 1);
        assert_eq!(
            summary.resume_hypergrad_tapes,
            report.resume_fingerprint.parameter_training.hypergrad_tapes
        );
        assert_eq!(summary.best_hash, report.captured.best_fingerprint.hash);
        assert_eq!(summary.final_hash, report.captured.final_fingerprint.hash);

        let strict_limits = SparseFineTuneRegressionLimits::new()
            .with_max_target_loss_regression(0.0)
            .unwrap()
            .with_max_retention_loss_regression(0.0)
            .unwrap()
            .with_max_target_retention_gap_regression(0.0)
            .unwrap()
            .with_status_match_required(true)
            .with_accepted_match_required(true)
            .with_guard_match_required(true)
            .with_movement_tolerance_match_required(true)
            .with_resume_match_required(true);
        let stable_compare = summary.compare_to(&summary, strict_limits).unwrap();
        assert!(stable_compare.passed);
        assert_eq!(stable_compare.target_loss_delta_change, 0.0);
        assert_eq!(stable_compare.retention_loss_delta_change, 0.0);
        assert_eq!(stable_compare.target_retention_gap_change, 0.0);
        assert_eq!(stable_compare.target_retention_ratio_change, None);
        assert_eq!(stable_compare.target_retention_gap_regression, 0.0);
        assert_eq!(stable_compare.target_retention_ratio_regression, None);
        assert_eq!(stable_compare.target_loss_margin_shortfall, 0.0);
        assert_eq!(stable_compare.target_retention_ratio_shortfall, None);
        assert_eq!(stable_compare.retention_loss_margin_shortfall, 0.0);
        assert_eq!(stable_compare.retention_accuracy_margin_shortfall, 0.0);
        assert_eq!(stable_compare.retention_perplexity_margin_shortfall, None);
        assert!(!stable_compare.status_changed);
        assert!(!stable_compare.accepted_changed);
        assert!(!stable_compare.guard_changed);
        assert!(!stable_compare.movement_tolerance_changed);
        assert!(!stable_compare.resume_changed);

        let margin_limits = SparseFineTuneRegressionLimits::new()
            .with_min_target_loss_margin(summary.target_loss_margin + 0.25)
            .unwrap()
            .with_min_retention_loss_margin(summary.retention_loss_margin + 0.125)
            .unwrap()
            .with_min_retention_accuracy_margin(summary.retention_accuracy_margin + 0.0625)
            .unwrap();
        let margin_compare = summary.compare_to(&summary, margin_limits).unwrap();
        assert!(!margin_compare.passed);
        assert!((margin_compare.target_loss_margin_shortfall - 0.25).abs() < 1e-6);
        assert!((margin_compare.retention_loss_margin_shortfall - 0.125).abs() < 1e-6);
        assert!((margin_compare.retention_accuracy_margin_shortfall - 0.0625).abs() < 1e-6);

        let missing_perplexity_limit = SparseFineTuneRegressionLimits::new()
            .with_min_retention_perplexity_margin(0.0)
            .unwrap();
        let missing_perplexity_compare = summary
            .compare_to(&summary, missing_perplexity_limit)
            .unwrap();
        assert!(!missing_perplexity_compare.passed);
        assert_eq!(
            missing_perplexity_compare.retention_perplexity_margin_shortfall,
            Some(0.0)
        );

        let mut ratio_baseline = summary.clone();
        ratio_baseline.retention_loss_delta = 0.4;
        ratio_baseline.target_loss_delta = 1.2;
        ratio_baseline.target_retention_gap = 0.8;
        ratio_baseline.target_retention_ratio = Some(3.0);
        let mut ratio_regressed = ratio_baseline.clone();
        ratio_regressed.target_loss_delta = 1.0;
        ratio_regressed.target_retention_gap = 0.6;
        ratio_regressed.target_retention_ratio = Some(2.5);
        let ratio_limits = SparseFineTuneRegressionLimits::new()
            .with_max_target_retention_gap_regression(0.1)
            .unwrap()
            .with_max_target_retention_ratio_regression(0.25)
            .unwrap()
            .with_min_target_retention_ratio(2.75)
            .unwrap();
        let ratio_compare = ratio_regressed
            .compare_to(&ratio_baseline, ratio_limits)
            .unwrap();
        assert!(!ratio_compare.passed);
        assert!((ratio_compare.target_retention_gap_change + 0.2).abs() < 1e-6);
        assert!((ratio_compare.target_retention_gap_regression - 0.2).abs() < 1e-6);
        assert_eq!(ratio_compare.target_retention_ratio_change, Some(-0.5));
        assert_eq!(ratio_compare.target_retention_ratio_regression, Some(0.5));
        assert_eq!(ratio_compare.target_retention_ratio_shortfall, Some(0.25));

        let mut accepted_only_changed = summary.clone();
        accepted_only_changed.accepted = !summary.accepted;
        let accepted_limits =
            SparseFineTuneRegressionLimits::new().with_accepted_match_required(true);
        let accepted_compare = accepted_only_changed
            .compare_to(&summary, accepted_limits)
            .unwrap();
        assert!(!accepted_compare.passed);
        assert!(!accepted_compare.status_changed);
        assert!(accepted_compare.accepted_changed);

        let mut regressed = summary.clone();
        regressed.target_loss_delta -= 0.25;
        regressed.retention_loss_delta -= 0.125;
        regressed.status = "guard_rejected";
        regressed.accepted = false;
        let regressed_compare = regressed.compare_to(&summary, strict_limits).unwrap();
        assert!(!regressed_compare.passed);
        assert!((regressed_compare.target_loss_delta_change + 0.25).abs() < 1e-6);
        assert!((regressed_compare.retention_loss_delta_change + 0.125).abs() < 1e-6);
        assert!((regressed_compare.target_loss_regression - 0.25).abs() < 1e-6);
        assert!((regressed_compare.retention_loss_regression - 0.125).abs() < 1e-6);
        assert!(regressed_compare.status_changed);
        assert!(regressed_compare.accepted_changed);
        assert!(!regressed_compare.guard_changed);
        assert!(!regressed_compare.movement_tolerance_changed);
        assert!(!regressed_compare.resume_changed);

        let mut guard_changed = summary.clone();
        guard_changed.retention_max_loss_increase += 0.1;
        let guard_compare = guard_changed.compare_to(&summary, strict_limits).unwrap();
        assert!(!guard_compare.passed);
        assert!(guard_compare.guard_changed);

        let mut movement_tolerance_changed = summary.clone();
        movement_tolerance_changed.movement_tolerance = 1e-6;
        let movement_compare = movement_tolerance_changed
            .compare_to(&summary, strict_limits)
            .unwrap();
        assert!(!movement_compare.passed);
        assert!(movement_compare.movement_tolerance_changed);

        let mut resume_changed = summary.clone();
        resume_changed.resume_hash = "changed".to_string();
        let resume_compare = resume_changed.compare_to(&summary, strict_limits).unwrap();
        assert!(!resume_compare.passed);
        assert!(resume_compare.resume_changed);
    }

    #[test]
    fn sparse_finetune_report_marks_guard_rejection_without_installing_forgetting() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.0, 1.0);
        let mut layer = Linear::new("sparse_report_reject", 1, 2).unwrap();
        layer
            .visit_parameters_mut(&mut |param| {
                for value in param.value_mut().data_mut() {
                    *value = 0.0;
                }
                Ok(())
            })
            .unwrap();
        let initial_fingerprint = layer.state_fingerprint().unwrap();
        let schedule = trainer.roundtable(1, 2, RoundtableConfig::default());
        let train_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let validation_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let retention_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
        )])
        .batched(1);
        let mut loss = SoftmaxCrossEntropy::new();

        let report = trainer
            .train_epochs_restore_best_sparse_with_finetune_report(
                &mut layer,
                &mut loss,
                train_loader,
                validation_loader,
                retention_loader,
                &schedule,
                1,
                SparseRetentionGuardConfig::new(10.0, 0.0).unwrap(),
                1e-8,
            )
            .unwrap();

        assert!(!report.accepted());
        assert_eq!(report.status(), "guard_rejected");
        assert_eq!(report.captured.guarded_best_epoch, None);
        assert_eq!(report.target_delta.loss_delta, 0.0);
        assert_eq!(report.retention_delta.loss_delta, 0.0);
        assert_eq!(layer.state_fingerprint().unwrap(), initial_fingerprint);
        assert_eq!(report.captured.best_fingerprint, initial_fingerprint);
        assert!(report.captured.best_differs_from_final);
        let summary = report.summary();
        assert_eq!(summary.status, "guard_rejected");
        assert!(!summary.accepted);
        assert!(!summary.target_loss_improved);
        assert!(summary.movement_ok);
        assert_eq!(summary.guarded_best_epoch, None);
        assert_eq!(summary.guard_epochs_run, 1);
        assert_eq!(summary.guard_accepted_epochs, 0);
        assert_eq!(summary.guard_retention_rejected_epochs, 1);
        assert_eq!(summary.guard_target_stale_epochs, 0);
        assert_eq!(summary.guard_acceptance_rate, 0.0);
        assert_eq!(summary.guard_retention_rejected_rate, 1.0);
        assert_eq!(summary.guard_target_stale_rate, 0.0);
        assert_eq!(summary.target_loss_delta, 0.0);
        assert_eq!(summary.retention_loss_delta, 0.0);
        assert_eq!(summary.target_min_loss_delta, 0.0);
        assert_eq!(summary.target_loss_margin, 0.0);
        assert_eq!(summary.retention_loss_margin, 10.0);
        assert_eq!(summary.retention_accuracy_margin, 0.0);
        assert_eq!(summary.movement_tolerance, 1e-8);
        assert_eq!(summary.best_hash, initial_fingerprint.hash);
        assert!(summary.best_differs_from_final);
    }

    #[test]
    fn validation_early_stopping_respects_patience_and_restores_best() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("early_stop", 1, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let train_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let validation_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let mut loss = ScriptedLoss::new(vec![10.0, 5.0, 9.0, 4.0, 8.0, 4.1, 7.0, 4.2, 6.0, 4.3]);

        let captured = trainer
            .train_epochs_restore_best_on_validation_with_early_stopping(
                &mut layer,
                &mut loss,
                train_loader,
                validation_loader,
                &schedule,
                5,
                EarlyStoppingConfig::new(2, 0.0).unwrap(),
            )
            .unwrap();

        assert_eq!(captured.epochs_requested, 5);
        assert_eq!(captured.train_history.len(), 4);
        assert_eq!(captured.validation_history.len(), 4);
        assert!(captured.early_stopped);
        assert_eq!(captured.stop_epoch, Some(4));
        assert_eq!(captured.early_stopping.unwrap().patience, 2);
        assert_eq!(captured.validation_summary.best_epoch, Some(2));
        assert!((captured.validation_summary.best_loss_per_row - 4.0).abs() < 1e-6);
        assert_eq!(
            layer.state_fingerprint().unwrap(),
            captured.best_fingerprint
        );
    }

    #[test]
    fn early_stopping_rejects_negative_min_delta() {
        let err = EarlyStoppingConfig::new(1, -0.001).unwrap_err();
        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "early_stopping_min_delta",
                ..
            }
        ));
    }

    #[test]
    fn validation_lr_plateau_decays_learning_rates_without_stopping() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.08, 0.02);
        let mut layer = Linear::new("plateau", 1, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let train_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let validation_loader = crate::dataset_from_vec(vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )])
        .batched(1);
        let mut loss = ScriptedLoss::new(vec![10.0, 5.0, 9.0, 5.1, 8.0, 5.2, 7.0, 4.9]);

        let captured = trainer
            .train_epochs_capture_best_on_validation_with_controls(
                &mut layer,
                &mut loss,
                train_loader,
                validation_loader,
                &schedule,
                4,
                ValidationTrainingControls::default()
                    .with_lr_plateau(LrPlateauConfig::new(1, 0.5, 0.0).unwrap()),
            )
            .unwrap();

        assert_eq!(captured.train_history.len(), 4);
        assert!(!captured.early_stopped);
        assert_eq!(captured.lr_decay_steps, 2);
        assert_eq!(captured.lr_plateau.unwrap().patience, 1);
        assert!((trainer.hyper_learning_rate() - 0.02).abs() < 1e-6);
        assert!((trainer.fallback_learning_rate() - 0.005).abs() < 1e-6);
        assert!((captured.final_hyper_learning_rate - 0.02).abs() < 1e-6);
        assert!((captured.final_fallback_learning_rate - 0.005).abs() < 1e-6);
    }

    #[test]
    fn optimizer_lr_decay_scales_fallback_only_once() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.1);
        let mut layer = Linear::new("fallback_decay", 1, 1).unwrap();
        let bias_before = layer.bias().value().data()[0];

        trainer.optimizer_mul_lr(&mut layer, 0.5).unwrap();
        assert!((trainer.fallback_learning_rate() - 0.05).abs() < 1e-6);
        assert!((layer.bias().learning_rate_scale() - 1.0).abs() < 1e-6);

        layer
            .backward(
                &Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
                &Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            )
            .unwrap();
        trainer.step(&mut layer).unwrap();
        let bias_delta = layer.bias().value().data()[0] - bias_before;
        assert!((bias_delta + 0.05).abs() < 1e-6);
    }

    #[test]
    fn lr_plateau_rejects_invalid_factor() {
        let err = LrPlateauConfig::new(1, 1.0, 0.0).unwrap_err();
        assert!(matches!(err, TensorError::IoError { .. }));
    }

    #[test]
    fn trainer_clips_gradients_without_collapse_feature() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("clip", 2, 1).unwrap();
        let input = Tensor::from_vec(1, 2, vec![10.0, -10.0]).unwrap();
        let grad = Tensor::from_vec(1, 1, vec![100.0]).unwrap();
        layer.backward(&input, &grad).unwrap();

        let before = layer.weight().gradient().unwrap().squared_l2_norm().sqrt();
        assert!(before > 1.0);
        trainer.clip_grad_global_norm(&mut layer, 1.0).unwrap();
        let clipped = layer.weight().gradient().unwrap().squared_l2_norm().sqrt();
        assert!(clipped <= 1.0 + 1e-5);
    }

    #[test]
    fn train_epoch_rejects_non_finite_loss_before_update() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        trainer.set_max_grad_norm(Some(1.0)).unwrap();
        let mut layer = Linear::new("finite", 2, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let before = layer.state_dict().unwrap();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![(
            Tensor::from_vec(1, 2, vec![1.0, -1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.75]).unwrap(),
        )];
        let mut loss = NonFiniteLoss;

        let err = trainer
            .train_epoch(&mut layer, &mut loss, dataset, &schedule)
            .unwrap_err();
        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "train_loss",
                ..
            }
        ));
        assert_eq!(layer.state_dict().unwrap(), before);
    }
}
