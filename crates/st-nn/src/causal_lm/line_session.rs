//! Fixed-order, replayable SGD on complete terminated answers. No document
//! windowing, sampler, optimizer moments or client-side parameter arithmetic.
use super::*;
use sha2::{Digest, Sha256};

mod snapshot;

const PLAN_SCHEMA: &str = "spiraltorch.native_causal_lm_line_plan.v1";
const PLAN_V2: &str = "spiraltorch.native_causal_lm_line_plan.v2";
const PLAN_V3: &str = "spiraltorch.native_causal_lm_line_plan.v3";

#[derive(Debug, Eq, PartialEq)]
pub(super) struct LinePromptId(sha2::digest::Output<Sha256>);

impl LinePromptId {
    pub(super) fn new(tokens: &[usize]) -> Self {
        let mut h = Sha256::new();
        h.update(b"spiraltorch.native_causal_lm.line_prompt.v1\0");
        h.update((tokens.len() as u64).to_le_bytes());
        for &id in tokens {
            h.update((id as u64).to_le_bytes());
        }
        Self(h.finalize())
    }
}

impl std::fmt::Display for LinePromptId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sha256:{:x}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CausalLmLineObjective {
    Set {},
    Menu { gold_index: usize, mix: f64 },
    Subset { accepted_indices: Vec<usize> },
}

impl CausalLmLineObjective {
    pub const MAX_SPEC_BYTES: usize = 4096;
    fn validate(&self, count: usize) -> CausalLmResult<()> {
        match self {
            Self::Set {} => Ok(()),
            Self::Menu { gold_index, mix } => {
                NativeCausalLm::validate_ranking(count, *gold_index, *mix)
            }
            Self::Subset { accepted_indices } => {
                NativeCausalLm::validate_line_subset(count, accepted_indices)
            }
        }
    }

    /// Length-only preflight before a foreign client copies token IDs.
    pub fn training_shape(
        &self,
        model: &NativeCausalLm,
        prompt_len: usize,
        lengths: &[usize],
        stop: usize,
    ) -> CausalLmResult<[usize; 2]> {
        self.training_shape_for(
            model,
            prompt_len,
            lengths,
            stop,
            CausalLmAdaptationMode::Geometry,
        )
    }

    pub fn training_shape_for(
        &self,
        model: &NativeCausalLm,
        prompt_len: usize,
        lengths: &[usize],
        stop: usize,
        mode: CausalLmAdaptationMode,
    ) -> CausalLmResult<[usize; 2]> {
        self.validate(lengths.len())?;
        model.line_set_shape_for(prompt_len, lengths, stop, Some(mode))
    }

    pub fn gradients(
        &self,
        model: &NativeCausalLm,
        prompt: &[usize],
        lines: &[Vec<usize>],
        stop: usize,
        cap: Option<usize>,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        self.gradients_for(
            model,
            prompt,
            lines,
            stop,
            cap,
            CausalLmAdaptationMode::Geometry,
        )
    }

    /// Differentiate the same terminated-answer loss through all attached adapters.
    /// Only the requested parameter groups are returned; the base remains frozen.
    pub fn gradients_for(
        &self,
        model: &NativeCausalLm,
        prompt: &[usize],
        lines: &[Vec<usize>],
        stop: usize,
        cap: Option<usize>,
        mode: CausalLmAdaptationMode,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        use objective::PreparedCompetition as P;
        self.validate(lines.len())?;
        let tile = model.line_readout_tile_rows(cap)?;
        let (mut batch, support) = model.prepare_line_set(prompt, lines, stop, Some(mode))?;
        batch.competition = Some(match self {
            Self::Set {} => P::LineSet(support),
            Self::Menu { gold_index, mix } => {
                P::LineMenu(line_set::LineMenu::new(support, lines, *gold_index, *mix))
            }
            Self::Subset { accepted_indices } => {
                P::LineSubset(line_set::LineSubset::new(support, lines, accepted_indices))
            }
        });
        let mut g = model.line_training(&batch, tile, mode)?;
        g.line_prompt = Some(LinePromptId::new(prompt));
        Ok(g)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Example {
    prompt_ids: Vec<usize>,
    candidates: Vec<Vec<usize>>,
    stop_id: usize,
    objective: CausalLmLineObjective,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    #[serde(default, deserialize_with = "present")]
    mode: Option<CausalLmAdaptationMode>,
    examples: Vec<Example>,
    epochs: u32,
    #[serde(default, deserialize_with = "present")]
    learning_rate: Option<f32>,
    #[serde(default, deserialize_with = "present")]
    update: Option<Update>,
    max_tile_values: Option<usize>,
}

fn present<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(d).map(Some)
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Update {
    Sgd {
        learning_rate: f32,
    },
    ParameterGradientL2 {
        target_l2: f64,
    },
    AnchorForwardKl {
        target_kl: f64,
        max_parameter_l2: f64,
        anchor_examples: Vec<usize>,
        #[serde(default)]
        criterion: CausalLmAnchorKlCriterion,
        #[serde(default)]
        direction: CausalLmAnchorDirection,
    },
}

impl Plan {
    fn anchor_prompts(&self, indices: &[usize]) -> CausalLmResult<Vec<&[usize]>> {
        if !(1..=CausalLmAnchorKlPolicy::MAX_ANCHORS).contains(&indices.len()) {
            return Err("invalid number of anchor example indices".into());
        }
        indices
            .iter()
            .map(|&i| {
                self.examples
                    .get(i)
                    .map(|e| e.prompt_ids.as_slice())
                    .ok_or_else(|| "anchor example index out of range".into())
            })
            .collect()
    }

    fn contract(
        &self,
        origin: &NativeCausalLm,
    ) -> CausalLmResult<(CausalLmAdaptationMode, Update)> {
        let (mode, update) = match (
            self.schema.as_str(),
            self.mode,
            self.learning_rate,
            self.update.as_ref(),
        ) {
            (PLAN_SCHEMA, None, Some(learning_rate), None) => (
                CausalLmAdaptationMode::Geometry,
                Update::Sgd { learning_rate },
            ),
            (PLAN_V2, Some(mode), Some(learning_rate), None) => {
                (mode, Update::Sgd { learning_rate })
            }
            (PLAN_V3, Some(mode), None, Some(policy)) => {
                if let Update::AnchorForwardKl {
                    anchor_examples,
                    target_kl,
                    max_parameter_l2,
                    criterion,
                    direction,
                } = policy
                {
                    let prompts = self.anchor_prompts(anchor_examples)?;
                    CausalLmAnchorKlPolicy {
                        target_kl: *target_kl,
                        max_parameter_l2: *max_parameter_l2,
                    }
                    .admit(origin, mode, &prompts, *criterion)?;
                    direction.admit(origin, mode, &prompts, *max_parameter_l2)?;
                    if direction.uses_loss_prompt()
                        && !self
                            .examples
                            .iter()
                            .all(|e| prompts.contains(&e.prompt_ids.as_slice()))
                    {
                        return Err(
                            "loss_prompt requires every scheduled prompt in the anchor bank".into(),
                        );
                    }
                }
                (mode, policy.clone())
            }
            _ => {
                return Err("invalid complete-line plan schema, owner mode or update policy".into())
            }
        };
        origin.validate_adaptation_mode(mode)?;
        match update {
            Update::Sgd { learning_rate } if !learning_rate.is_finite() || learning_rate <= 0. => {
                return Err("invalid line SGD rate".into())
            }
            Update::ParameterGradientL2 { target_l2 } => {
                // An impossible global f32 budget can be refused before any decoder work.
                origin.admit_adaptation_distance(mode, target_l2)?;
            }
            _ => {}
        }
        Ok((mode, update))
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", content = "scores", rename_all = "snake_case")]
pub enum CausalLmLineSessionReport {
    Set(CausalLmLineSetReport),
    Menu(CausalLmLineMenuReport),
    Subset(CausalLmLineSubsetReport),
}

#[derive(Debug, Clone, Serialize)]
pub struct CausalLmLineSessionStep {
    pub update: u32,
    pub epoch: u32,
    pub example_index: usize,
    pub plan_id: String,
    pub pre_geometry_id: String,
    pub post_geometry_id: String,
    pub target_tokens: usize,
    pub loss: f32,
    pub objective: CausalLmLineSessionReport,
    pub work: CausalLmLineTrainingWork,
}

/// Owner-aware receipt. A LoRA-only model need not have any geometry artifact.
#[derive(Debug, Clone, Serialize)]
pub struct CausalLmLineAdaptationStep {
    pub update: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_ray: Option<CausalLmAdaptationRayStep>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor_kl: Option<CausalLmAnchorKlMatch>,
    /// Zero-based epoch and example that produced this one-based update.
    pub epoch: u32,
    pub example_index: usize,
    pub plan_id: String,
    pub pre_geometry_id: Option<String>,
    pub post_geometry_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<CausalLmAdaptationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_lora_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_lora_id: Option<String>,
    pub target_tokens: usize,
    pub loss: f32,
    pub objective: CausalLmLineSessionReport,
    pub work: CausalLmLineTrainingWork,
}

/// Opt-in structured refusal; other decoder, validation and numerical errors stay errors.
#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CausalLmLineAttempt {
    Applied {
        step: Box<CausalLmLineAdaptationStep>,
    },
    Unmatched {
        plan_id: String,
        next_update: u32,
        example_index: usize,
        refusal: Box<CausalLmAnchorKlRefusal>,
    },
}

fn pin(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Clone)]
pub struct NativeCausalLmLineTrainingSession {
    model: NativeCausalLm,
    plan: Arc<Plan>,
    plan_id: String,
    origin_geometry_id: Option<String>,
    origin_lora_id: Option<String>,
    mode: CausalLmAdaptationMode,
    update_policy: Update,
    updates: u32,
    total: u32,
}

impl NativeCausalLmLineTrainingSession {
    pub const MAX_PLAN_BYTES: usize = 1 << 20;
    pub const MAX_EXAMPLES: usize = 4096;
    pub const MAX_IDS: usize = 131_072;
    pub const MAX_UPDATES: u32 = 1 << 20;

    /// The exact UTF-8 plan bytes bind every prompt, ordered candidate, stop,
    /// objective, owner selection, step policy, epoch count and vocabulary tile policy.
    pub fn new(origin: NativeCausalLm, bytes: &[u8]) -> CausalLmResult<Self> {
        if bytes.len() > Self::MAX_PLAN_BYTES {
            return Err("line plan exceeds byte budget".into());
        }
        let plan: Plan = serde_json::from_slice(bytes)?;
        if !(1..=Self::MAX_EXAMPLES).contains(&plan.examples.len()) {
            return Err("invalid complete-line plan example count".into());
        }
        let (mode, update_policy) = plan.contract(&origin)?;
        let total = (plan.examples.len() as u32)
            .checked_mul(plan.epochs)
            .filter(|n| (1..=Self::MAX_UPDATES).contains(n))
            .ok_or("invalid line plan update count")?;
        origin.line_readout_tile_rows(plan.max_tile_values)?;
        let mut ids = 0usize;
        for example in &plan.examples {
            ids += example.prompt_ids.len()
                + example
                    .candidates
                    .iter()
                    .map(|line| line.len() + 1)
                    .sum::<usize>();
            if ids > Self::MAX_IDS {
                return Err("line plan exceeds total token budget".into());
            }
            example.objective.validate(example.candidates.len())?;
            origin.admit_line_support(
                &example.prompt_ids,
                &example.candidates,
                example.stop_id,
                Some(mode),
            )?;
        }
        Ok(Self {
            origin_geometry_id: origin.geometry_artifact_id().map(str::to_owned),
            origin_lora_id: origin.lora_artifact_id().map(str::to_owned),
            mode,
            update_policy,
            model: origin,
            plan: Arc::new(plan),
            plan_id: pin(bytes),
            updates: 0,
            total,
        })
    }

    pub fn model(&self) -> &NativeCausalLm {
        &self.model
    }
    pub fn plan_id(&self) -> &str {
        &self.plan_id
    }
    pub fn completed_updates(&self) -> u32 {
        self.updates
    }
    pub fn total_updates(&self) -> u32 {
        self.total
    }
    pub fn exhausted(&self) -> bool {
        self.updates == self.total
    }

    /// Compute from the pre-update snapshot. Neither owner nor cursor changes
    /// on a gradient, validation, or SGD error.
    pub fn step(&mut self) -> CausalLmResult<Option<CausalLmLineSessionStep>> {
        self.require_geometry_plan()?;
        self.next_step(|r| {
            Ok(CausalLmLineSessionStep {
                update: r.update,
                epoch: r.epoch,
                example_index: r.example_index,
                plan_id: r.plan_id,
                pre_geometry_id: r.pre_geometry_id.ok_or("missing geometry owner")?,
                post_geometry_id: r.post_geometry_id.ok_or("missing geometry owner")?,
                target_tokens: r.target_tokens,
                loss: r.loss,
                objective: r.objective,
                work: r.work,
            })
        })
    }

    fn require_geometry_plan(&self) -> CausalLmResult<()> {
        if self.plan.mode.is_some() {
            return Err("use the owner-aware step for an explicit-mode line plan".into());
        }
        Ok(())
    }

    /// Version-aware step, including v1 geometry plans. Choosing owners never changes the loss.
    pub fn step_adaptation(&mut self) -> CausalLmResult<Option<CausalLmLineAdaptationStep>> {
        self.next_step(Ok)
    }

    /// Like step_adaptation, but a bounded KL miss returns its measured trials
    /// and leaves the cursor and both owners at their entry state. Never retries.
    pub fn attempt_adaptation_step(&mut self) -> CausalLmResult<Option<CausalLmLineAttempt>> {
        match self.step_adaptation() {
            Ok(step) => Ok(step.map(|step| CausalLmLineAttempt::Applied {
                step: Box::new(step),
            })),
            Err(e) => {
                let refusal = e.downcast::<CausalLmAnchorKlRefusal>()?;
                Ok(Some(CausalLmLineAttempt::Unmatched {
                    plan_id: self.plan_id.clone(),
                    next_update: self.updates + 1,
                    example_index: self.updates as usize % self.plan.examples.len(),
                    refusal,
                }))
            }
        }
    }

    fn next_step<R>(
        &mut self,
        finish: impl FnOnce(CausalLmLineAdaptationStep) -> CausalLmResult<R>,
    ) -> CausalLmResult<Option<R>> {
        if self.exhausted() {
            return Ok(None);
        }
        let n = self.plan.examples.len();
        let i = self.updates as usize % n;
        let e = &self.plan.examples[i];
        let g = e.objective.gradients_for(
            &self.model,
            &e.prompt_ids,
            &e.candidates,
            e.stop_id,
            self.plan.max_tile_values,
            self.mode,
        )?;
        let (next, parameter_ray, anchor_kl) = match &self.update_policy {
            Update::Sgd { learning_rate } => (
                self.model
                    .step_adaptation_sgd(&g, self.mode, *learning_rate)?,
                None,
                None,
            ),
            Update::ParameterGradientL2 { target_l2 } => {
                let (next, ray) = self
                    .model
                    .step_adaptation_gradient_at_distance(&g, self.mode, *target_l2)?;
                (next, Some(ray), None)
            }
            Update::AnchorForwardKl {
                target_kl,
                max_parameter_l2,
                anchor_examples,
                criterion,
                direction,
            } => {
                let prompts = self.plan.anchor_prompts(anchor_examples)?;
                let (next, ray, report) = self.model.step_adaptation_at_anchor_kl_with_direction(
                    &g,
                    self.mode,
                    &prompts,
                    CausalLmAnchorKlPolicy {
                        target_kl: *target_kl,
                        max_parameter_l2: *max_parameter_l2,
                    },
                    *criterion,
                    *direction,
                )?;
                (next, Some(ray), Some(report))
            }
        };
        let objective = match &e.objective {
            CausalLmLineObjective::Set {} => CausalLmLineSessionReport::Set(
                g.line_set_report().ok_or("missing set report")?.clone(),
            ),
            CausalLmLineObjective::Menu { .. } => CausalLmLineSessionReport::Menu(
                g.line_menu_report().ok_or("missing menu report")?.clone(),
            ),
            CausalLmLineObjective::Subset { .. } => CausalLmLineSessionReport::Subset(
                g.line_subset_report()
                    .ok_or("missing subset report")?
                    .clone(),
            ),
        };
        let report = CausalLmLineAdaptationStep {
            parameter_ray,
            anchor_kl,
            update: self.updates + 1,
            epoch: self.updates / n as u32,
            example_index: i,
            plan_id: self.plan_id.clone(),
            pre_geometry_id: self.model.geometry_artifact_id().map(str::to_owned),
            post_geometry_id: next.geometry_artifact_id().map(str::to_owned),
            mode: self.plan.mode,
            pre_lora_id: self
                .plan
                .mode
                .and(self.model.lora_artifact_id())
                .map(str::to_owned),
            post_lora_id: self
                .plan
                .mode
                .and(next.lora_artifact_id())
                .map(str::to_owned),
            target_tokens: g.target_tokens(),
            loss: g.loss(),
            objective,
            work: g
                .line_training_work()
                .ok_or("missing readout work")?
                .clone(),
        };
        let update = report.update;
        let report = finish(report)?;
        self.model = next;
        self.updates = update;
        Ok(Some(report))
    }
}
