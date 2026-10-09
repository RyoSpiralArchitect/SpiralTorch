//! Bounded, paired corpus studies with request-bound, all-case resume.
//! Native and browser clients share model math, data order and acceptance gates.
use crate::{
    resident::{
        AttentionInferencePlan, AttentionMask, ByteDecoderCheckpoint, ByteDecoderGeometryPlan,
        ByteDecoderPlan, ByteLmBatch, InferenceOp, InferencePlan, ResidentByteDecoder,
        ResidualAttentionPlan, ToposResonatorKernel,
    },
    Tensor,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use st_backend_wgpu::{
    resident_tensor::ResidentTensor, resident_training::parameters::ResidentParameterUpdate,
    runtime::WgpuRuntime,
};
use st_kernel_contracts::classification::{ClassReduction, CrossEntropySpec};
use st_tensor::NdLayout;
use std::collections::BTreeSet;

/// Errors retain the underlying model, JSON, GPU or study validation error.
pub type ByteCorpusStudyResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Result<T> = ByteCorpusStudyResult<T>;
/// Input and checkpoint JSON are bounded before deserialization.
pub const BYTE_CORPUS_STUDY_MAX_BYTES: usize = 64 * 1024 * 1024;
const CHECKPOINT_SCHEMA: &str = "spiraltorch.byte_corpus.checkpoint.v1";

/// A fixed request: documents, initial models, SGD rate and all selected windows.
/// This is the bounded paired-study protocol, not an arbitrary training scheduler.
pub struct ByteCorpusStudy {
    request: Request,
    request_sha256: String,
}

/// All cases at the same accepted update boundary. The full request's exact bytes
/// must match on resume. This identity guard is not a signature or attestation of
/// past computation. Model JSON remains opaque through browser clients.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ByteCorpusStudyCheckpoint {
    schema: String,
    request_sha256: String,
    completed_updates: u32,
    cases: Vec<CaseCheckpoint>,
}

/// A completed or partial report and a checkpoint usable by a fresh runtime.
pub struct ByteCorpusStudySegment {
    pub report: Value,
    pub checkpoint: ByteCorpusStudyCheckpoint,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseCheckpoint {
    name: String,
    model_json: String,
    training: Vec<TrainingPoint>,
    validation: Vec<EvaluationPoint>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingPoint {
    revision: u32,
    ce: f32,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluationPoint {
    revision: u32,
    batch_losses: Vec<f32>,
}

impl EvaluationPoint {
    fn report(&self, r: &Request) -> Value {
        // Recompute f64 summaries from the original f32 readbacks on both paths.
        let mean = self.batch_losses.iter().map(|&v| f64::from(v)).sum::<f64>()
            / self.batch_losses.len() as f64;
        json!({"revision":self.revision,"batch_losses":self.batch_losses,
            "mean_ce":mean,"bits_per_byte":mean / std::f64::consts::LN_2,
            "target_bytes":r.config.batch * r.config.steps * self.batch_losses.len()})
    }
}

impl ByteCorpusStudyCheckpoint {
    pub fn completed_updates(&self) -> usize {
        self.completed_updates as usize
    }

    pub fn request_sha256(&self) -> &str {
        &self.request_sha256
    }

    pub fn to_json(&self) -> Result<String> {
        let json = serde_json::to_string(self)?;
        if json.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
            return Err("study checkpoint exceeds 64 MiB".into());
        }
        Ok(json)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    batch: usize,
    steps: usize,
    width: usize,
    hidden: usize,
    heads: usize,
    blocks: Vec<bool>,
    geometry_cols: usize,
    curvature: f32,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Parameter {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    seed: u64,
    geometry: bool,
    parameters: Vec<Parameter>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    config: Config,
    train_documents: Vec<Vec<u8>>,
    validation_documents: Vec<Vec<u8>>,
    train_batches: Vec<Vec<(usize, usize)>>,
    validation_batches: Vec<Vec<(usize, usize)>>,
    checkpoint_every: usize,
    rate: f32,
    cases: Vec<Case>,
}

fn parse(input: &[u8]) -> Result<Request> {
    if input.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
        return Err("study request exceeds 64 MiB".into());
    }
    let r: Request = serde_json::from_slice(input)?;
    let c = &r.config;
    // Bounds belong to this interactive study runner, not the public model API.
    if r.schema != "spiraltorch.byte_corpus.request.v1"
        || !(1..=8).contains(&c.batch)
        || !(2..=128).contains(&c.steps)
        || !(2..=128).contains(&c.width)
        || !(2..=256).contains(&c.hidden)
        || c.heads == 0
        || c.width % c.heads != 0
        || !(1..=4).contains(&c.blocks.len())
        || !(2..=32).contains(&c.geometry_cols)
        || c.geometry_cols % 2 != 0
        || !c.curvature.is_finite()
        || c.curvature >= 0.
        || r.rate <= 0.
        || !r.rate.is_finite()
        || !(1..=1024).contains(&r.train_batches.len())
        || !(1..=128).contains(&r.validation_batches.len())
        || !(1..=64).contains(&r.checkpoint_every)
        || !(2..=16).contains(&r.cases.len())
        || r.cases.len() % 2 != 0
    {
        return Err("invalid study configuration".into());
    }
    if r.train_documents.is_empty()
        || r.validation_documents.is_empty()
        || r.train_documents
            .iter()
            .any(|d| r.validation_documents.contains(d))
    {
        return Err("training and validation need disjoint, nonempty document sets".into());
    }
    for (docs, batches) in [
        (&r.train_documents, &r.train_batches),
        (&r.validation_documents, &r.validation_batches),
    ] {
        let documents: Vec<_> = docs.iter().map(Vec::as_slice).collect();
        for selections in batches {
            if selections.len() != c.batch {
                return Err("batch selection count differs".into());
            }
            ByteLmBatch::from_documents(&documents, selections, c.steps)?;
        }
    }
    let mut names = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    for case in &r.cases {
        if case.seed > (1u64 << 53) - 1 {
            return Err("seed must be exactly representable in browser JSON".into());
        }
        if !names.insert(&case.name) {
            return Err("duplicate case name".into());
        }
        seeds.insert(case.seed);
        plan(c, case)?;
    }
    for seed in seeds {
        let pair: Vec<_> = r.cases.iter().filter(|c| c.seed == seed).collect();
        if pair.len() != 2 || pair[0].geometry == pair[1].geometry {
            return Err("each seed needs one ordinary and one geometry case".into());
        }
        let core = |case: &Case| {
            case.parameters
                .iter()
                .filter(|p| !p.name.starts_with("geometry."))
                .map(|p| {
                    (
                        p.name.clone(),
                        p.shape.clone(),
                        p.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };
        if core(pair[0]) != core(pair[1]) {
            return Err("paired ordinary initial parameters differ".into());
        }
    }
    Ok(r)
}

struct Cursor<'a> {
    entries: std::slice::Iter<'a, Parameter>,
}

impl Cursor<'_> {
    fn values(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let p = self.entries.next().ok_or("missing model parameter")?;
        let count = shape
            .iter()
            .try_fold(1usize, |n, &v| n.checked_mul(v))
            .ok_or("parameter shape overflow")?;
        if p.name != name
            || p.shape != shape
            || p.values.len() != count
            || p.values.iter().any(|v| !v.is_finite())
        {
            return Err(format!("invalid parameter {name}").into());
        }
        Ok(p.values.clone())
    }
    fn tensor(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            if shape.len() == 1 { 1 } else { shape[0] },
            *shape.last().ok_or("empty shape")?,
            self.values(name, shape)?,
        )?)
    }
    fn norm(&mut self, prefix: &str, width: usize) -> Result<InferenceOp> {
        Ok(InferenceOp::LayerNorm {
            gain: self.tensor(&format!("{prefix}.gain"), &[width])?,
            bias: self.tensor(&format!("{prefix}.bias"), &[width])?,
            epsilon: 1e-5,
        })
    }
    fn linear(&mut self, prefix: &str, input: usize, output: usize) -> Result<InferenceOp> {
        Ok(InferenceOp::Linear {
            weight: self.tensor(&format!("{prefix}.weight"), &[input, output])?,
            bias: self.tensor(&format!("{prefix}.bias"), &[output])?,
        })
    }
}

fn plan(c: &Config, case: &Case) -> Result<ByteDecoderPlan> {
    let layout = NdLayout::contiguous(&[c.batch, c.steps, c.width])?;
    let mut p = Cursor {
        entries: case.parameters.iter(),
    };
    let token = p.tensor("token_embedding", &[256, c.width])?;
    let position = p.tensor("position_embedding", &[c.steps, c.width])?;
    let geometry = if case.geometry {
        let projection = InferencePlan::from_operations(
            layout.clone(),
            vec![p.linear("geometry.projection", c.width, c.geometry_cols)?],
        )?;
        let decay = p.values("geometry.raw_decay", &[c.geometry_cols / 2])?;
        let phase = p.values("geometry.raw_phase", &[c.geometry_cols / 2])?;
        let gains = (0..c.blocks.len())
            .map(|i| p.values(&format!("geometry.raw_gain.{i}"), &[c.heads]))
            .collect::<Result<Vec<_>>>()?;
        Some(ByteDecoderGeometryPlan::new(
            &projection,
            &decay,
            &phase,
            &gains,
            c.curvature,
        )?)
    } else {
        None
    };
    let mut blocks = Vec::new();
    for (i, &topos) in c.blocks.iter().enumerate() {
        let prefix = format!("block.{i}");
        let pre = InferencePlan::from_operations(
            layout.clone(),
            vec![p.norm(&format!("{prefix}.pre"), c.width)?],
        )?;
        let fused = p.values(&format!("{prefix}.qkv.weight"), &[c.width, 3 * c.width])?;
        let bias = p.values(&format!("{prefix}.qkv.bias"), &[3 * c.width])?;
        let mut projections = Vec::new();
        for slot in 0..3 {
            let weight = (0..c.width)
                .flat_map(|row| {
                    fused[row * 3 * c.width + slot * c.width
                        ..row * 3 * c.width + (slot + 1) * c.width]
                        .iter()
                        .copied()
                })
                .collect();
            projections.push((
                Tensor::from_vec(c.width, c.width, weight)?,
                Tensor::from_vec(
                    1,
                    c.width,
                    bias[slot * c.width..(slot + 1) * c.width].to_vec(),
                )?,
            ));
        }
        projections.push((
            p.tensor(&format!("{prefix}.output.weight"), &[c.width, c.width])?,
            p.tensor(&format!("{prefix}.output.bias"), &[c.width])?,
        ));
        let attention = AttentionInferencePlan::from_parameters(
            layout.clone(),
            c.heads,
            AttentionMask::Causal { query_offset: 0 },
            std::array::from_fn(|j| (&projections[j].0, &projections[j].1)),
        )?;
        let mut operations = vec![
            p.norm(&format!("{prefix}.feed"), c.width)?,
            p.linear(&format!("{prefix}.up"), c.width, c.hidden)?,
            InferenceOp::Gelu,
        ];
        if topos {
            operations.push(InferenceOp::ToposResonator {
                gate: p.tensor(&format!("{prefix}.topos_gate"), &[c.hidden])?,
                kernel: ToposResonatorKernel::new(0.2, 0.12, 0.3, 4)?,
                max_volume: c.batch * c.steps * c.hidden,
            });
        }
        operations.push(p.linear(&format!("{prefix}.down"), c.hidden, c.width)?);
        let feed = InferencePlan::from_operations(layout.clone(), operations)?;
        blocks.push(ResidualAttentionPlan::from_plans(&pre, &attention, &feed)?);
    }
    let head = InferencePlan::from_operations(
        layout,
        vec![
            p.norm("head", c.width)?,
            p.linear("head.output", c.width, 256)?,
        ],
    )?;
    if p.entries.next().is_some() {
        return Err("unexpected trailing model parameter".into());
    }
    let ordinary = ByteDecoderPlan::from_plans(&token, &position, &blocks, &head)?;
    Ok(match geometry {
        Some(g) => ordinary.with_causal_geometry(g)?,
        None => ordinary,
    })
}

async fn read_many(
    model: &ResidentByteDecoder,
    tensors: &[ResidentTensor],
) -> Result<Vec<Vec<f32>>> {
    let snapshot = model
        .tensor_device()
        .snapshot_many(&tensors.iter().collect::<Vec<_>>())?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    Ok(values)
}

async fn evaluate(model: &mut ResidentByteDecoder, r: &Request) -> Result<EvaluationPoint> {
    let docs: Vec<_> = r.validation_documents.iter().map(Vec::as_slice).collect();
    let mut losses = Vec::new();
    let spec = CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?;
    for selections in &r.validation_batches {
        let batch = ByteLmBatch::from_documents(&docs, selections, r.config.steps)?;
        let resident = model.prepare_batch(&batch)?;
        let tape = model.forward(&resident)?;
        losses.push(tape.next_byte_loss(spec)?.value().clone());
    }
    let values = read_many(model, &losses).await?;
    let losses: Vec<_> = values.iter().map(|v| v[0]).collect();
    if losses.iter().any(|v| !v.is_finite()) {
        return Err("non-finite validation loss".into());
    }
    Ok(EvaluationPoint {
        revision: model.parameter_snapshot().revision().try_into()?,
        batch_losses: losses,
    })
}

impl ByteCorpusStudy {
    /// CPU-only preflight. Exact input bytes (including whitespace) define the
    /// identity of a study, so retain the original request alongside checkpoints.
    pub fn from_json(input: &[u8]) -> Result<Self> {
        Ok(Self {
            request: parse(input)?,
            request_sha256: format!("{:x}", Sha256::digest(input)),
        })
    }

    pub fn total_updates(&self) -> usize {
        self.request.train_batches.len()
    }

    pub fn request_sha256(&self) -> &str {
        &self.request_sha256
    }

    /// Decode and validate every case before the caller requests a GPU.
    pub fn checkpoint_from_json(&self, input: &[u8]) -> Result<ByteCorpusStudyCheckpoint> {
        if input.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
            return Err("study checkpoint exceeds 64 MiB".into());
        }
        let checkpoint: ByteCorpusStudyCheckpoint = serde_json::from_slice(input)?;
        self.checked_models(&checkpoint)?;
        Ok(checkpoint)
    }

    /// Conservative bound through the final update, including escaped model
    /// JSON, case names and all scalar histories. Numeric text can grow after
    /// training even when the original request used compact literals like 1e-4.
    pub fn checkpoint_size_bound(&self) -> Result<usize> {
        let mut bound = serde_json::to_vec(&ByteCorpusStudyCheckpoint {
            schema: CHECKPOINT_SCHEMA.to_owned(),
            request_sha256: self.request_sha256.clone(),
            completed_updates: self.total_updates() as u32,
            cases: self
                .request
                .cases
                .iter()
                .map(|case| CaseCheckpoint {
                    name: case.name.clone(),
                    model_json: String::new(),
                    training: Vec::new(),
                    validation: Vec::new(),
                })
                .collect(),
        })?
        .len();
        let evaluations = (0..=self.total_updates())
            .filter(|&n| self.evaluation_boundary(n))
            .count();
        for case in &self.request.cases {
            let model_bytes = plan(&self.request.config, case)?
                .initial_checkpoint()
                .escaped_json_size_bound()?;
            let history_bytes = self.total_updates() * 64
                + evaluations * (64 + self.request.validation_batches.len() * 32);
            bound = bound
                .checked_add(model_bytes)
                .and_then(|n| n.checked_add(history_bytes))
                .ok_or("checkpoint size overflow")?;
        }
        Ok(bound)
    }

    fn evaluation_boundary(&self, revision: usize) -> bool {
        revision % self.request.checkpoint_every == 0 || revision == self.total_updates()
    }

    fn checked_models(
        &self,
        checkpoint: &ByteCorpusStudyCheckpoint,
    ) -> Result<Vec<ByteDecoderCheckpoint>> {
        let cursor = checkpoint.completed_updates();
        if checkpoint.schema != CHECKPOINT_SCHEMA
            || checkpoint.request_sha256 != self.request_sha256
            || cursor > self.total_updates()
            || checkpoint.cases.len() != self.request.cases.len()
        {
            return Err("checkpoint schema, request identity, cursor or case count differs".into());
        }
        let expected_evaluations: Vec<_> = (0..=cursor)
            .filter(|&n| self.evaluation_boundary(n))
            .collect();
        checkpoint
            .cases
            .iter()
            .zip(&self.request.cases)
            .map(|(saved, case)| {
                if saved.name != case.name
                    || saved.training.len() != cursor
                    || saved
                        .training
                        .iter()
                        .enumerate()
                        .any(|(i, point)| point.revision as usize != i + 1 || !point.ce.is_finite())
                    || saved.validation.len() != expected_evaluations.len()
                    || saved.validation.iter().zip(&expected_evaluations).any(
                        |(point, &revision)| {
                            point.revision as usize != revision
                                || point.batch_losses.len() != self.request.validation_batches.len()
                                || point.batch_losses.iter().any(|v| !v.is_finite())
                        },
                    )
                {
                    return Err("checkpoint case identity or history differs".into());
                }
                let model = ByteDecoderCheckpoint::from_json(&saved.model_json)?;
                let initial = plan(&self.request.config, case)?.initial_checkpoint();
                if model.attempted_revision() != cursor as u64
                    || model.topology_json()? != initial.topology_json()?
                    || (cursor == 0 && model.to_json()? != initial.to_json()?)
                {
                    return Err(
                        "checkpoint model revision, topology or initial values differ".into(),
                    );
                }
                Ok(model)
            })
            .collect()
    }

    /// Validate the absolute stop cursor and optional checkpoint without a GPU.
    pub fn validate_segment(
        &self,
        resume: Option<&ByteCorpusStudyCheckpoint>,
        stop_after: usize,
    ) -> Result<()> {
        if stop_after > self.total_updates()
            || resume.is_some_and(|c| c.completed_updates() > stop_after)
        {
            return Err("stop cursor must lie between saved cursor and total updates".into());
        }
        if self.checkpoint_size_bound()? > BYTE_CORPUS_STUDY_MAX_BYTES {
            return Err("study checkpoint worst-case size exceeds 64 MiB; reduce the study before requesting a GPU".into());
        }
        if let Some(checkpoint) = resume {
            self.checked_models(checkpoint)?;
        }
        Ok(())
    }

    /// Original uninterrupted study path; does not capture a model checkpoint.
    pub async fn run(&self, runtime: WgpuRuntime) -> Result<Value> {
        Ok(self
            .execute(runtime, None, self.total_updates(), false)
            .await?
            .0)
    }

    /// Advance all cases to an absolute cursor. Every update receipt must be
    /// accepted before a new checkpoint is returned. Failure/cancellation drops
    /// these local models, leaving the caller's last checkpoint unchanged.
    /// Pausing does not add evaluations to the request's original schedule.
    pub async fn advance(
        &self,
        runtime: WgpuRuntime,
        resume: Option<&ByteCorpusStudyCheckpoint>,
        stop_after: usize,
    ) -> Result<ByteCorpusStudySegment> {
        let (report, checkpoint) = self.execute(runtime, resume, stop_after, true).await?;
        Ok(ByteCorpusStudySegment {
            report,
            checkpoint: checkpoint.ok_or("missing captured checkpoint")?,
        })
    }

    async fn execute(
        &self,
        runtime: WgpuRuntime,
        resume: Option<&ByteCorpusStudyCheckpoint>,
        stop_after: usize,
        capture: bool,
    ) -> Result<(Value, Option<ByteCorpusStudyCheckpoint>)> {
        if capture {
            self.validate_segment(resume, stop_after)?;
        }
        let restored = resume.map(|c| self.checked_models(c)).transpose()?;
        if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
            return Err("real GPU required for this resident learning study".into());
        }
        let r = &self.request;
        let start = resume.map_or(0, ByteCorpusStudyCheckpoint::completed_updates);
        let documents: Vec<_> = r.train_documents.iter().map(Vec::as_slice).collect();
        let mut reports = Vec::new();
        let mut saved_cases = Vec::new();
        for (index, case) in r.cases.iter().enumerate() {
            let mut model = match &restored {
                Some(models) => models[index].restore_wgpu(runtime.clone())?,
                None => plan(&r.config, case)?.compile_training_wgpu(runtime.clone())?,
            };
            let (mut training, mut evaluations) = if let Some(checkpoint) = resume {
                let saved = &checkpoint.cases[index];
                (saved.training.clone(), saved.validation.clone())
            } else {
                let initial = model.parameter_snapshot();
                let initial_values = read_many(&model, initial.values()).await?;
                if initial_values
                    != case
                        .parameters
                        .iter()
                        .map(|p| p.values.clone())
                        .collect::<Vec<_>>()
                {
                    return Err("compiled parameter order or initial values differ".into());
                }
                (Vec::new(), vec![evaluate(&mut model, r).await?])
            };
            let mut pending: Vec<(ResidentParameterUpdate, ResidentTensor)> = Vec::new();
            for (step, selections) in r
                .train_batches
                .iter()
                .enumerate()
                .take(stop_after)
                .skip(start)
            {
                let batch = ByteLmBatch::from_documents(&documents, selections, r.config.steps)?;
                let resident = model.prepare_batch(&batch)?;
                let tape = model.forward(&resident)?;
                let loss =
                    tape.next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?)?;
                let gradients = model.backward(&tape, loss.prediction_gradient())?;
                let update = model.sgd(&gradients, r.rate)?;
                pending.push((update, loss.value().clone()));
                if self.evaluation_boundary(step + 1) || step + 1 == stop_after {
                    // Bounded bursts; all receipts are drained even at a pause
                    // that is not a scheduled evaluation boundary.
                    for (update, loss) in pending.drain(..) {
                        let receipt = update.snapshot_with_scalar(&loss)?;
                        #[cfg(not(target_arch = "wasm32"))]
                        let (revision, ce) = receipt.read()?;
                        #[cfg(target_arch = "wasm32")]
                        let (revision, ce) = receipt.read_async().await?;
                        if revision != training.len() as u64 + 1 || !ce.is_finite() {
                            return Err("unexpected revision or non-finite training loss".into());
                        }
                        training.push(TrainingPoint {
                            revision: revision.try_into()?,
                            ce,
                        });
                    }
                    if self.evaluation_boundary(step + 1) {
                        evaluations.push(evaluate(&mut model, r).await?);
                    }
                }
            }
            let final_values = read_many(&model, model.parameter_snapshot().values()).await?;
            reports.push(json!({"name":case.name,"seed":case.seed,"geometry":case.geometry,
                "parameter_tensors":case.parameters.len(),
                "parameter_scalars":case.parameters.iter().map(|p| p.values.len()).sum::<usize>(),
                "training":training,"validation":evaluations.iter().map(|e| e.report(r)).collect::<Vec<_>>(),
                "final_parameters":final_values}));
            if capture {
                let readback = model.checkpoint_snapshot()?;
                #[cfg(not(target_arch = "wasm32"))]
                let checkpoint = readback.read()?;
                #[cfg(target_arch = "wasm32")]
                let checkpoint = readback.read_async().await?;
                saved_cases.push(CaseCheckpoint {
                    name: case.name.clone(),
                    model_json: checkpoint.to_json()?,
                    training,
                    validation: evaluations,
                });
            }
        }
        let schema = if stop_after == self.total_updates() {
            "spiraltorch.byte_corpus.result.v1"
        } else {
            "spiraltorch.byte_corpus.partial.v1"
        };
        let report = json!({"schema":schema,"engine":"spiraltorch","request_sha256":self.request_sha256,
            "adapter":format!("{:?}",runtime.adapter_info()),"cases":reports,
            "scope":"Document-held-out byte corpus pilot; no language-quality superiority or speed claim"});
        let checkpoint = capture.then(|| ByteCorpusStudyCheckpoint {
            schema: CHECKPOINT_SCHEMA.to_owned(),
            request_sha256: self.request_sha256.clone(),
            completed_updates: stop_after as u32,
            cases: saved_cases,
        });
        Ok((report, checkpoint))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        let mut p = Vec::new();
        let mut add = |name: &str, shape: &[usize]| {
            p.push(json!({"name":name,"shape":shape,"values":vec![0.1;shape.iter().product()]}));
        };
        add("token_embedding", &[256, 2]);
        add("position_embedding", &[2, 2]);
        add("block.0.pre.gain", &[2]);
        add("block.0.pre.bias", &[2]);
        add("block.0.qkv.weight", &[2, 6]);
        add("block.0.qkv.bias", &[6]);
        add("block.0.output.weight", &[2, 2]);
        add("block.0.output.bias", &[2]);
        add("block.0.feed.gain", &[2]);
        add("block.0.feed.bias", &[2]);
        add("block.0.up.weight", &[2, 2]);
        add("block.0.up.bias", &[2]);
        add("block.0.down.weight", &[2, 2]);
        add("block.0.down.bias", &[2]);
        add("head.gain", &[2]);
        add("head.bias", &[2]);
        add("head.output.weight", &[2, 256]);
        add("head.output.bias", &[256]);
        let mut geometry = p[..2].to_vec();
        geometry.extend([
            json!({"name":"geometry.projection.weight","shape":[2,2],"values":vec![0.1;4]}),
            json!({"name":"geometry.projection.bias","shape":[2],"values":[0.1,0.1]}),
            json!({"name":"geometry.raw_decay","shape":[1],"values":[0.1]}),
            json!({"name":"geometry.raw_phase","shape":[1],"values":[0.1]}),
            json!({"name":"geometry.raw_gain.0","shape":[1],"values":[0.1]}),
        ]);
        geometry.extend_from_slice(&p[2..]);
        json!({"schema":"spiraltorch.byte_corpus.request.v1",
            "config":{"batch":1,"steps":2,"width":2,"hidden":2,"heads":1,
                "blocks":[false],"geometry_cols":2,"curvature":-0.75},
            "train_documents":[[1,2,3,4,5]],"validation_documents":[[7,8,9]],
            "train_batches":[[[0,0]],[[0,2]]],"validation_batches":[[[0,0]]],
            "checkpoint_every":2,"rate":0.05,
            "cases":[{"name":"plain","seed":7,"geometry":false,"parameters":p},
                {"name":"geometry","seed":7,"geometry":true,"parameters":geometry}]})
    }

    fn valid(value: &Value) -> bool {
        ByteCorpusStudy::from_json(&serde_json::to_vec(value).unwrap()).is_ok()
    }

    // Synthetic records exercise structural validation, not proof of training.
    fn saved(study: &ByteCorpusStudy, cursor: usize) -> ByteCorpusStudyCheckpoint {
        ByteCorpusStudyCheckpoint {
            schema: CHECKPOINT_SCHEMA.into(),
            request_sha256: study.request_sha256.clone(),
            completed_updates: cursor as u32,
            cases: study
                .request
                .cases
                .iter()
                .map(|case| {
                    let initial = plan(&study.request.config, case)
                        .unwrap()
                        .initial_checkpoint();
                    let mut record: Value =
                        serde_json::from_str(&initial.to_json().unwrap()).unwrap();
                    record["attempted_revision"] = json!(cursor.to_string());
                    CaseCheckpoint {
                        name: case.name.clone(),
                        model_json: record.to_string(),
                        training: (1..=cursor)
                            .map(|n| TrainingPoint {
                                revision: n as u32,
                                ce: 0.1,
                            })
                            .collect(),
                        validation: (0..=cursor)
                            .filter(|&n| study.evaluation_boundary(n))
                            .map(|n| EvaluationPoint {
                                revision: n as u32,
                                batch_losses: vec![0.1; study.request.validation_batches.len()],
                            })
                            .collect(),
                    }
                })
                .collect(),
        }
    }

    fn study() -> ByteCorpusStudy {
        ByteCorpusStudy::from_json(&serde_json::to_vec(&fixture()).unwrap()).unwrap()
    }

    #[test]
    fn resume_preflights_zero_mid_interval_and_completed_cursors() {
        let study = study();
        for cursor in 0..=2 {
            let checkpoint = saved(&study, cursor);
            let encoded = checkpoint.to_json().unwrap();
            let restored = study.checkpoint_from_json(encoded.as_bytes()).unwrap();
            assert_eq!(restored.to_json().unwrap(), encoded);
            assert_eq!(restored.completed_updates(), cursor);
            assert_eq!(restored.request_sha256(), study.request_sha256());
            assert_eq!(
                restored.cases[0].validation.len(),
                if cursor == 2 { 2 } else { 1 }
            );
            study.validate_segment(Some(&restored), cursor).unwrap();
            study.validate_segment(Some(&restored), 2).unwrap();
        }
        assert!(study.validate_segment(Some(&saved(&study, 1)), 0).is_err());
        assert!(study.validate_segment(None, 3).is_err());
    }

    #[test]
    fn resume_binds_exact_data_order_rate_and_request_bytes() {
        let study = study();
        let checkpoint = saved(&study, 1).to_json().unwrap();
        let mut variants = Vec::new();
        let mut value = fixture();
        value["rate"] = json!(0.04);
        variants.push(value);
        let mut value = fixture();
        value["train_documents"][0][0] = json!(6);
        variants.push(value);
        let mut value = fixture();
        value["train_batches"].as_array_mut().unwrap().reverse();
        variants.push(value);
        let mut value = fixture();
        value["cases"].as_array_mut().unwrap().reverse();
        variants.push(value);
        let mut value = fixture();
        value["checkpoint_every"] = json!(1);
        variants.push(value);
        for variant in variants {
            let changed =
                ByteCorpusStudy::from_json(&serde_json::to_vec(&variant).unwrap()).unwrap();
            assert!(changed.checkpoint_from_json(checkpoint.as_bytes()).is_err());
        }
        let mut whitespace = serde_json::to_vec(&fixture()).unwrap();
        whitespace.push(b'\n');
        assert!(ByteCorpusStudy::from_json(&whitespace)
            .unwrap()
            .checkpoint_from_json(checkpoint.as_bytes())
            .is_err());
    }

    #[test]
    fn resume_rejects_inconsistent_case_cursor_and_histories() {
        let study = study();
        let base = saved(&study, 1);
        let mut variants = Vec::new();
        let mut v = base.clone();
        v.schema.push('x');
        variants.push(v);
        let mut v = base.clone();
        v.completed_updates = 3;
        variants.push(v);
        let mut v = base.clone();
        v.completed_updates = 0;
        variants.push(v);
        let mut v = base.clone();
        v.cases.pop();
        variants.push(v);
        let mut v = base.clone();
        v.cases.swap(0, 1);
        variants.push(v);
        let mut v = base.clone();
        v.cases[0].training.clear();
        variants.push(v);
        let mut v = base.clone();
        v.cases[0].training[0].revision = 0;
        variants.push(v);
        let mut v = base.clone();
        v.cases[0].training[0].ce = f32::NAN;
        variants.push(v);
        let mut v = base.clone();
        v.cases[0].validation.clear();
        variants.push(v);
        let mut v = base.clone();
        v.cases[0].validation[0].revision = 1;
        variants.push(v);
        let mut v = base.clone();
        v.cases[0].validation[0].batch_losses.clear();
        variants.push(v);
        let mut v = base.clone();
        v.cases[0].validation[0].batch_losses[0] = f32::INFINITY;
        variants.push(v);
        let mut v = base.clone();
        v.cases[0].validation.push(EvaluationPoint {
            revision: 1,
            batch_losses: vec![0.1],
        });
        variants.push(v);
        for checkpoint in variants {
            assert!(study.checked_models(&checkpoint).is_err());
        }
        let mut unknown = serde_json::to_value(&base).unwrap();
        unknown["cases"][0]["training"][0]["accepted"] = json!(false);
        assert!(study
            .checkpoint_from_json(unknown.to_string().as_bytes())
            .is_err());
    }

    #[test]
    fn resume_rejects_foreign_topology_clock_and_changed_initial_values() {
        let study = study();
        let mut checkpoint = saved(&study, 1);
        let mut model: Value = serde_json::from_str(&checkpoint.cases[1].model_json).unwrap();
        model["model"]["geometry"]["curvature"] = json!(-0.5);
        checkpoint.cases[1].model_json = model.to_string();
        assert!(study.checked_models(&checkpoint).is_err());
        let mut checkpoint = saved(&study, 1);
        let mut model: Value = serde_json::from_str(&checkpoint.cases[0].model_json).unwrap();
        model["attempted_revision"] = json!("2");
        checkpoint.cases[0].model_json = model.to_string();
        assert!(study.checked_models(&checkpoint).is_err());
        let mut checkpoint = saved(&study, 0);
        let mut model: Value = serde_json::from_str(&checkpoint.cases[0].model_json).unwrap();
        model["model"]["token"]["values"][0] = json!(0.2);
        checkpoint.cases[0].model_json = model.to_string();
        assert!(study.checked_models(&checkpoint).is_err());
    }

    #[test]
    fn resume_json_is_bounded_before_decode() {
        assert!(study()
            .checkpoint_from_json(&vec![b' '; BYTE_CORPUS_STUDY_MAX_BYTES + 1])
            .is_err());
    }

    #[test]
    fn compact_large_request_is_rejected_for_resume_before_gpu() {
        let mut parameters = Vec::new();
        let mut add = |name: &str, shape: &[usize]| {
            parameters.push(json!({"name":name,"shape":shape,
                "values":vec![0.0001;shape.iter().product()]}));
        };
        add("token_embedding", &[256, 128]);
        add("position_embedding", &[128, 128]);
        for block in 0..4 {
            let prefix = format!("block.{block}");
            for (name, shape) in [
                ("pre.gain", vec![128]),
                ("pre.bias", vec![128]),
                ("qkv.weight", vec![128, 384]),
                ("qkv.bias", vec![384]),
                ("output.weight", vec![128, 128]),
                ("output.bias", vec![128]),
                ("feed.gain", vec![128]),
                ("feed.bias", vec![128]),
                ("up.weight", vec![128, 256]),
                ("up.bias", vec![256]),
                ("down.weight", vec![256, 128]),
                ("down.bias", vec![128]),
            ] {
                add(&format!("{prefix}.{name}"), &shape);
            }
        }
        add("head.gain", &[128]);
        add("head.bias", &[128]);
        add("head.output.weight", &[128, 256]);
        add("head.output.bias", &[256]);
        let mut geometry = parameters[..2].to_vec();
        for (name, shape) in [
            ("geometry.projection.weight", vec![128, 32]),
            ("geometry.projection.bias", vec![32]),
            ("geometry.raw_decay", vec![16]),
            ("geometry.raw_phase", vec![16]),
            ("geometry.raw_gain.0", vec![1]),
            ("geometry.raw_gain.1", vec![1]),
            ("geometry.raw_gain.2", vec![1]),
            ("geometry.raw_gain.3", vec![1]),
        ] {
            geometry.push(
                json!({"name":name,"values":vec![0.0001;shape.iter().product()],"shape":shape}),
            );
        }
        geometry.extend_from_slice(&parameters[2..]);
        let mut request = fixture();
        request["config"] = json!({"batch":1,"steps":128,"width":128,"hidden":256,
            "heads":1,"blocks":vec![false;4],"geometry_cols":32,"curvature":-0.75});
        request["train_documents"] = json!([vec![1; 129]]);
        request["validation_documents"] = json!([vec![2; 129]]);
        request["train_batches"] = json!([[[0, 0]]]);
        request["cases"] = json!((0..8).flat_map(|seed| [
            json!({"name":format!("plain-{seed}"),"seed":seed,"geometry":false,"parameters":parameters}),
            json!({"name":format!("geometry-{seed}"),"seed":seed,"geometry":true,"parameters":geometry}),
        ]).collect::<Vec<_>>());
        let input = request.to_string().replace("0.0001", "1e-4");
        drop(request);
        assert!(input.len() < BYTE_CORPUS_STUDY_MAX_BYTES);
        let study = ByteCorpusStudy::from_json(input.as_bytes()).unwrap();
        assert!(study.checkpoint_size_bound().unwrap() > BYTE_CORPUS_STUDY_MAX_BYTES);
        let error = study.validate_segment(None, 0).unwrap_err();
        assert!(error.to_string().contains("worst-case size"));
    }

    #[test]
    fn checkpoint_budget_covers_extreme_values_and_full_history() {
        let study = study();
        let mut checkpoint = saved(&study, study.total_updates());
        for case in &mut checkpoint.cases {
            for point in &mut case.training {
                point.ce = -f32::MAX;
            }
            for point in &mut case.validation {
                point.batch_losses.fill(f32::from_bits(1));
            }
        }
        assert!(checkpoint.to_json().unwrap().len() < study.checkpoint_size_bound().unwrap());
        assert!(study.checkpoint_size_bound().unwrap() < BYTE_CORPUS_STUDY_MAX_BYTES);
    }

    #[test]
    fn paired_document_study_preflights_without_a_gpu() {
        assert!(valid(&fixture()));
    }

    #[test]
    fn unequal_initial_backbones_and_parameter_shapes_are_rejected() {
        let mut v = fixture();
        v["cases"][1]["parameters"][0]["values"][0] = json!(0.2);
        assert!(!valid(&v));
        let mut v = fixture();
        v["cases"][1]["parameters"][2]["shape"] = json!([4, 1]);
        assert!(!valid(&v));
    }

    #[test]
    fn duplicate_modes_names_and_unknown_fields_are_rejected() {
        let mut v = fixture();
        v["cases"][1] = v["cases"][0].clone();
        v["cases"][1]["name"] = json!("second_plain");
        assert!(!valid(&v));
        let mut v = fixture();
        v["cases"][1]["name"] = json!("plain");
        assert!(!valid(&v));
        let mut v = fixture();
        v["ignore_errors"] = json!(true);
        assert!(!valid(&v));
    }

    #[test]
    fn split_leakage_and_invalid_final_selections_are_rejected() {
        let mut v = fixture();
        v["validation_documents"] = v["train_documents"].clone();
        assert!(!valid(&v));
        let mut v = fixture();
        v["train_batches"][1][0] = json!([0, 4]);
        assert!(!valid(&v));
        let mut v = fixture();
        v["validation_batches"][0] = json!([]);
        assert!(!valid(&v));
    }

    #[test]
    fn invalid_rates_and_unbounded_recipes_are_rejected() {
        for rate in [0., -0.1] {
            let mut v = fixture();
            v["rate"] = json!(rate);
            assert!(!valid(&v));
        }
        let mut v = fixture();
        v["config"]["steps"] = json!(129);
        assert!(!valid(&v));
        let mut v = fixture();
        v["config"]["heads"] = json!(0);
        assert!(!valid(&v));
        let mut v = fixture();
        v["cases"][0]["seed"] = json!(1u64 << 53);
        v["cases"][1]["seed"] = json!(1u64 << 53);
        assert!(!valid(&v));
    }
}
