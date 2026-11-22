//! Fixed-order, replayable SGD on complete terminated answers. No document
//! windowing, sampler, optimizer moments or client-side parameter arithmetic.
use super::*;
use sha2::{Digest, Sha256};

mod snapshot;

const PLAN_SCHEMA: &str = "spiraltorch.native_causal_lm_line_plan.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CausalLmLineObjective {
    Set {},
    Menu { gold_index: usize, mix: f64 },
    Subset { accepted_indices: Vec<usize> },
}

impl CausalLmLineObjective {
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
        self.validate(lengths.len())?;
        model.line_set_shape(prompt_len, lengths, stop, true)
    }

    pub fn gradients(
        &self,
        model: &NativeCausalLm,
        prompt: &[usize],
        lines: &[Vec<usize>],
        stop: usize,
        cap: Option<usize>,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        match self {
            Self::Set {} => {
                model.geometry_line_set_gradients_with_tile_limit(prompt, lines, stop, cap)
            }
            Self::Menu { gold_index, mix } => model.geometry_line_menu_gradients_with_tile_limit(
                prompt,
                lines,
                *gold_index,
                stop,
                *mix,
                cap,
            ),
            Self::Subset { accepted_indices } => model
                .geometry_line_subset_gradients_with_tile_limit(
                    prompt,
                    lines,
                    accepted_indices,
                    stop,
                    cap,
                ),
        }
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
    examples: Vec<Example>,
    epochs: u32,
    learning_rate: f32,
    max_tile_values: Option<usize>,
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
    /// Zero-based epoch and example that produced this one-based update.
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

fn pin(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Clone)]
pub struct NativeCausalLmLineTrainingSession {
    model: NativeCausalLm,
    plan: Arc<Plan>,
    plan_id: String,
    origin_geometry_id: String,
    updates: u32,
    total: u32,
}

impl NativeCausalLmLineTrainingSession {
    pub const MAX_PLAN_BYTES: usize = 1 << 20;
    pub const MAX_EXAMPLES: usize = 4096;
    pub const MAX_IDS: usize = 131_072;
    pub const MAX_UPDATES: u32 = 1 << 20;

    /// The exact UTF-8 plan bytes bind every prompt, ordered candidate, stop,
    /// objective, learning rate, epoch count and vocabulary tile policy.
    pub fn new(origin: NativeCausalLm, bytes: &[u8]) -> CausalLmResult<Self> {
        if bytes.len() > Self::MAX_PLAN_BYTES {
            return Err("line plan exceeds byte budget".into());
        }
        origin.validate_adaptation_mode(CausalLmAdaptationMode::Geometry)?;
        let plan: Plan = serde_json::from_slice(bytes)?;
        if plan.schema != PLAN_SCHEMA
            || !(1..=Self::MAX_EXAMPLES).contains(&plan.examples.len())
            || !plan.learning_rate.is_finite()
            || plan.learning_rate <= 0.
        {
            return Err("invalid complete-line plan schema, case count or learning rate".into());
        }
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
                true,
            )?;
        }
        Ok(Self {
            origin_geometry_id: origin
                .geometry_artifact_id()
                .expect("admitted geometry")
                .into(),
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
        if self.exhausted() {
            return Ok(None);
        }
        let n = self.plan.examples.len();
        let i = self.updates as usize % n;
        let e = &self.plan.examples[i];
        let g = e.objective.gradients(
            &self.model,
            &e.prompt_ids,
            &e.candidates,
            e.stop_id,
            self.plan.max_tile_values,
        )?;
        let next = self.model.step_adaptation_sgd(
            &g,
            CausalLmAdaptationMode::Geometry,
            self.plan.learning_rate,
        )?;
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
        let report = CausalLmLineSessionStep {
            update: self.updates + 1,
            epoch: self.updates / n as u32,
            example_index: i,
            plan_id: self.plan_id.clone(),
            pre_geometry_id: g
                .geometry_artifact_id()
                .ok_or("missing gradient owner")?
                .into(),
            post_geometry_id: next
                .geometry_artifact_id()
                .ok_or("missing geometry owner")?
                .into(),
            target_tokens: g.target_tokens(),
            loss: g.loss(),
            objective,
            work: g
                .line_training_work()
                .ok_or("missing readout work")?
                .clone(),
        };
        self.model = next;
        self.updates = report.update;
        Ok(Some(report))
    }
}
