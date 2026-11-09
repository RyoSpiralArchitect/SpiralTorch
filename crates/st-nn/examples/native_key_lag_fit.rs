//! Offline one-block QKV adaptation. CONFIG WEIGHTS LORA GEOMETRY, request JSON on stdin.
use serde::Deserialize;
use serde_json::json;
use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep};
use st_nn::{
    causal_lm::{
        CausalLmReadoutAccumulator, CausalLmResult, NativeCausalLm, NativeCausalLmQkvAdamW,
    },
    geometry_artifact::GeometryAdapterArtifact,
    weights::SafeTensorArchive,
};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{self, Read, Write},
    path::PathBuf,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    base: String,
    lora: String,
    geometry: String,
    layer: usize,
    start: Option<Pin>,
    action: Action,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    path: PathBuf,
    id: String,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Action {
    Fit {
        prompts: Vec<Vec<usize>>,
        gold: Vec<usize>,
        microbatch: usize,
        rate: f32,
        updates: u32,
        save_at: Vec<u32>,
        out: PathBuf,
    },
    Score {
        prompts: Vec<Vec<usize>>,
        prefix_len: usize,
    },
    Gradient {
        prompts: Vec<Vec<usize>>,
        gold: Vec<usize>,
        microbatch: usize,
    },
}

fn read(r: impl Read, cap: usize) -> CausalLmResult<Vec<u8>> {
    let mut v = Vec::new();
    r.take(cap as u64 + 1).read_to_end(&mut v)?;
    if v.len() > cap {
        return Err("input exceeds example limit".into());
    }
    Ok(v)
}
fn input(prompts: &[Vec<usize>]) -> CausalLmResult<(Vec<usize>, [usize; 2])> {
    let n = prompts.first().ok_or("empty prompt batch")?.len();
    if prompts.len() > 128 || !(1..=512).contains(&n) || prompts.iter().any(|p| p.len() != n) {
        return Err("expected bounded equal-length prefixes".into());
    }
    Ok((prompts.concat(), [prompts.len(), n]))
}
fn main() -> CausalLmResult<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 4 {
        return Err("expected CONFIG WEIGHTS LORA GEOMETRY".into());
    }
    let r: Request = serde_json::from_slice(&read(io::stdin().lock(), 1 << 20)?)?;
    let get = |i: usize, n| read(File::open(&args[i])?, n);
    let archive = SafeTensorArchive::from_owned_bytes(get(1, 32 << 20)?, Default::default())?;
    let m = NativeCausalLm::from_gpt2_json(&get(0, 1 << 20)?, &archive, Default::default())?
        .with_readout_accumulator(CausalLmReadoutAccumulator::F32WideBackward)
        .with_lora_json(std::str::from_utf8(&get(2, 1 << 20)?)?)?
        .with_geometry(GeometryAdapterArtifact::from_json(
            &get(3, 1 << 20)?,
            &r.geometry,
            &r.base,
        )?)?;
    if m.base_model_identity() != r.base
        || m.lora_artifact_id() != Some(r.lora.as_str())
        || m.geometry_artifact_id() != Some(r.geometry.as_str())
    {
        return Err("foreign decoder or adapters".into());
    }
    let mut train = match r.start {
        None => NativeCausalLmQkvAdamW::new(&m, r.layer, AdamWConfig::default())?,
        Some(p) => NativeCausalLmQkvAdamW::restore(
            &m,
            &read(
                File::open(p.path)?,
                NativeCausalLmQkvAdamW::MAX_CHECKPOINT_BYTES,
            )?,
            &p.id,
        )?,
    };
    if train.layer() != r.layer {
        return Err("unexpected checkpoint training layer".into());
    }
    let initial = train.serving_id();
    let data = match r.action {
        Action::Fit {
            prompts,
            gold,
            microbatch,
            rate,
            updates,
            save_at,
            out,
        } => {
            let (ids, shape) = input(&prompts)?;
            if rate <= 0.
                || updates == 0
                || updates > 512
                || save_at.len() > 8
                || save_at.windows(2).any(|p| p[0] >= p[1])
                || save_at.last().copied() != Some(updates)
                || save_at.contains(&0)
            {
                return Err("invalid fit schedule".into());
            }
            let step = AdamWStep::new(rate, 0.)?;
            std::fs::create_dir(&out)?;
            let mut losses = vec![];
            let mut pins = BTreeMap::new();
            for i in 1..=updates {
                losses.push(train.step_answers_microbatched(&ids, shape, &gold, microbatch, step)?);
                if save_at.contains(&i) {
                    let (bytes, pin) = train.checkpoint()?;
                    let mut f = OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .open(out.join(format!("{}.bin", train.steps())))?;
                    f.write_all(&bytes)?;
                    pins.insert(
                        train.steps(),
                        json!({"checkpoint":pin,"serving":train.serving_id()}),
                    );
                }
            }
            json!({"updates":losses,"pins":pins})
        }
        Action::Gradient {
            prompts,
            gold,
            microbatch,
        } => {
            if train.weights().len() > 1 << 16 {
                return Err("gradient export exceeds example budget".into());
            }
            let (ids, shape) = input(&prompts)?;
            let (loss, g) = train.answer_gradients_microbatched(&ids, shape, &gold, microbatch)?;
            json!({"loss":loss,"packed_bits":g.data().iter().map(|v|v.to_bits()).collect::<Vec<_>>()})
        }
        Action::Score {
            prompts,
            prefix_len,
        } => {
            let (_, shape) = input(&prompts)?;
            if prefix_len == 0
                || prefix_len >= shape[1]
                || shape[0].saturating_mul(m.config().vocabulary) > 1 << 19
            {
                return Err("invalid scoring budget or prefix".into());
            }
            let mut prefixes = BTreeMap::new();
            let mut rows = vec![];
            for q in prompts {
                let key = &q[..prefix_len];
                if !prefixes.contains_key(key) {
                    let count = m.context_cache_values(prefix_len, 1 << 20)?;
                    if (prefixes.len() + 1) * count > 1 << 20 {
                        return Err("prefix inventory exceeds cache budget".into());
                    }
                    prefixes.insert(key.to_vec(), train.prefill(key.to_vec(), 1 << 20)?);
                }
                let mut branch = prefixes[key].clone();
                for &token in &q[prefix_len..] {
                    branch = branch.branch(token)?;
                }
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                rows.push(json!({"ordinary":bits(&train.last_logits(&q)?),"cached":bits(branch.logits())}));
            }
            json!({"rows":rows,"prefixes":prefixes.len()})
        }
    };
    println!(
        "{}",
        json!({"base":r.base,"lora":r.lora,"geometry":r.geometry,"key_lag":m.key_lag(),"initial":initial,"end":train.serving_id(),"steps":train.steps(),"data":data})
    );
    Ok(())
}
