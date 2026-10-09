//! Data-driven corpus learning, shared unchanged by native and browser clients.
//! This runner owns orchestration only; model math and updates are st-nn's.
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use st_backend_wgpu::{
    resident_tensor::ResidentTensor, resident_training::parameters::ResidentParameterUpdate,
    runtime::WgpuRuntime,
};
use st_kernel_contracts::classification::{ClassReduction, CrossEntropySpec};
use st_nn::{
    resident::{
        AttentionInferencePlan, AttentionMask, ByteDecoderGeometryPlan, ByteDecoderPlan,
        ByteLmBatch, InferenceOp, InferencePlan, ResidentByteDecoder, ResidualAttentionPlan,
        ToposResonatorKernel,
    },
    Tensor,
};
use st_tensor::NdLayout;
use std::collections::BTreeSet;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024;

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
    if input.len() > MAX_REQUEST_BYTES {
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

pub fn validate_request(input: &[u8]) -> Result<()> {
    parse(input).map(|_| ())
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

async fn evaluate(model: &mut ResidentByteDecoder, r: &Request) -> Result<Value> {
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
    let mean = losses.iter().map(|&v| f64::from(v)).sum::<f64>() / losses.len() as f64;
    Ok(
        json!({"revision":model.parameter_snapshot().revision(), "batch_losses":losses,
        "mean_ce":mean, "bits_per_byte":mean / std::f64::consts::LN_2,
        "target_bytes": r.config.batch * r.config.steps * losses.len()}),
    )
}

pub async fn run(runtime: WgpuRuntime, input: &[u8]) -> Result<Value> {
    let r = parse(input)?;
    if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
        return Err("real GPU required for this resident learning study".into());
    }
    let documents: Vec<_> = r.train_documents.iter().map(Vec::as_slice).collect();
    let mut reports = Vec::new();
    for case in &r.cases {
        let mut model = plan(&r.config, case)?.compile_training_wgpu(runtime.clone())?;
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
        let mut checkpoints = vec![evaluate(&mut model, &r).await?];
        let mut training = Vec::new();
        let mut pending: Vec<(ResidentParameterUpdate, ResidentTensor)> = Vec::new();
        for (step, selections) in r.train_batches.iter().enumerate() {
            let batch = ByteLmBatch::from_documents(&documents, selections, r.config.steps)?;
            let resident = model.prepare_batch(&batch)?;
            let tape = model.forward(&resident)?;
            let loss =
                tape.next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?)?;
            let gradients = model.backward(&tape, loss.prediction_gradient())?;
            let update = model.sgd(&gradients, r.rate)?;
            pending.push((update, loss.value().clone()));
            if (step + 1) % r.checkpoint_every == 0 || step + 1 == r.train_batches.len() {
                // Bounded resident bursts; only scalar loss and update receipts
                // cross the host boundary, never activations or gradients.
                for (update, loss) in pending.drain(..) {
                    let receipt = update.snapshot_with_scalar(&loss)?;
                    #[cfg(not(target_arch = "wasm32"))]
                    let (revision, ce) = receipt.read()?;
                    #[cfg(target_arch = "wasm32")]
                    let (revision, ce) = receipt.read_async().await?;
                    if revision != training.len() as u64 + 1 {
                        return Err("unexpected revision".into());
                    }
                    training.push(json!({"revision":revision,"ce":ce}));
                }
                checkpoints.push(evaluate(&mut model, &r).await?);
            }
        }
        let final_values = read_many(&model, model.parameter_snapshot().values()).await?;
        reports.push(
            json!({"name":case.name,"seed":case.seed,"geometry":case.geometry,
            "parameter_tensors":case.parameters.len(),
            "parameter_scalars":case.parameters.iter().map(|p| p.values.len()).sum::<usize>(),
            "training":training,"validation":checkpoints,"final_parameters":final_values}),
        );
    }
    Ok(
        json!({"schema":"spiraltorch.byte_corpus.result.v1","engine":"spiraltorch","request_sha256":format!("{:x}",Sha256::digest(input)),
        "adapter":format!("{:?}",runtime.adapter_info()),"cases":reports,
        "scope":"Document-held-out byte corpus pilot; no language-quality superiority or speed claim"}),
    )
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
        validate_request(&serde_json::to_vec(value).unwrap()).is_ok()
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
