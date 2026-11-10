//! Synthetic language study only; see benchmarks/paired-prompt-contrast/PROJECTION.md.
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use st_nn::causal_lm::{
    CausalLmEvaluationConfig, CausalLmPairWeighting, CausalLmReadoutAccumulator,
    CausalLmResult as Result, DeclaredRivalRow, NativeCausalLm, TargetClassScore,
};
use st_nn::geometry_artifact::GeometryAdapterArtifact;
use st_nn::weights::SafeTensorArchive;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

mod evaluate;
mod train;

const FIXTURE: &str = "f0177d9c3c980737f35c9a1694bae9553fef9c319b72bedc0dc767979c2a753e";
const V: usize = 64;
const WIDTH: usize = 32;
const STOP: usize = 1;
const SITE: &str = "transformer.h.1.attn";
const ARMS: [&str; 4] = [
    "absolute_equal",
    "anchored_equal",
    "absolute_common",
    "anchored_common",
];
type Menus = BTreeMap<String, BTreeSet<Vec<usize>>>;

fn require(value: bool, message: &str) -> Result<()> {
    if !value {
        return Err(message.to_owned().into());
    }
    Ok(())
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut bytes)?;
    require(bytes.len() as u64 <= limit, "input exceeds study budget")?;
    Ok(bytes)
}

fn fresh(path: &Path) -> Result<File> {
    Ok(OpenOptions::new().write(true).create_new(true).open(path)?)
}

fn write(path: &Path, bytes: &[u8]) -> Result<String> {
    let mut f = fresh(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(sha(bytes))
}

fn put(path: &Path, v: &impl Serialize) -> Result<String> {
    write(path, &serde_json::to_vec(v)?)
}

fn event(f: &mut File, v: Value) -> Result<()> {
    serde_json::to_writer(&mut *f, &v)?;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}

struct Fixture<'a> {
    dir: &'a Path,
    pins: BTreeMap<String, String>,
}
impl<'a> Fixture<'a> {
    fn open(dir: &'a Path) -> Result<Self> {
        let raw = read(&dir.join("fixture.json"), 8192)?;
        require(sha(&raw) == FIXTURE, "not the sealed synthetic fixture")?;
        let manifest: Value = serde_json::from_slice(&raw)?;
        Ok(Self {
            dir,
            pins: serde_json::from_value(manifest["files"].clone())?,
        })
    }
    fn bytes(&self, name: &str, limit: u64) -> Result<Vec<u8>> {
        let b = read(&self.dir.join(name), limit)?;
        require(
            self.pins.get(name) == Some(&sha(&b)),
            "synthetic input digest differs",
        )?;
        Ok(b)
    }
    fn parsed<T: DeserializeOwned>(&self, name: &str) -> Result<T> {
        Ok(serde_json::from_slice(&self.bytes(name, 16_384)?)?)
    }
    fn origins(&self) -> Result<[NativeCausalLm; 2]> {
        let w = SafeTensorArchive::from_owned_bytes(
            self.bytes("model.safetensors", 100_000)?,
            Default::default(),
        )?;
        let base = NativeCausalLm::from_gpt2_json(
            &self.bytes("config.json", 2048)?,
            &w,
            Default::default(),
        )?
        .with_readout_accumulator(CausalLmReadoutAccumulator::F32WideBackward);
        let mut l: Value = serde_json::from_str(&base.with_lora(1, 2, 3.)?.lora_to_json()?)?;
        l["b"] = json!((0..WIDTH * 2)
            .map(|i| ((i % 9) as f32 - 4.) / 512.)
            .collect::<Vec<_>>());
        let base = base.with_lora_json(&serde_json::to_string(&l)?)?;
        let tensors=["gate","bias"].map(|k|{
            let values:Vec<f32>=(0..WIDTH).map(|i| if k=="gate"{0.5+((i%7) as f32-3.)/128.}else{((i%5) as f32-2.)/128.}).collect();
            json!({"name":k,"shape":[WIDTH],"role":"parameter","requires_grad":true,"values":values})
        });
        let a = GeometryAdapterArtifact::create_json(&serde_json::to_vec(&json!({
            "schema":"spiraltorch.geometry_adapter_artifact.v1","base_model_identity":base.base_model_identity(),
            "policy":"full_prefix_f32_no_cache.v1","training":true,"placements":[{
                "path":SITE,"training":true,"tensors":tensors,"recipe":{"schema":"spiraltorch.wave_stage_adapter.v1",
                "features":WIDTH,"strength":0.5,"max_values":4096,"stage":{"kind":"projection","curvature":-1.,"log_radius":0.}}}]
        }))?)?;
        Ok([
            base.with_geometry(a.clone())?,
            base.with_geometry(a.anchor_wave_projection(SITE)?)?,
        ])
    }
}

fn geom(m: &NativeCausalLm) -> Result<&GeometryAdapterArtifact> {
    m.geometry_artifact()
        .ok_or_else(|| "missing geometry".into())
}
fn save_geom(dir: &Path, name: &str, model: &NativeCausalLm) -> Result<()> {
    let a = geom(model)?;
    require(
        format!(
            "sha256:{}",
            write(
                &dir.join(format!("{name}.geometry.json")),
                a.to_json().as_bytes()
            )?
        ) == a.artifact_id(),
        "snapshot digest",
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    prompt: Vec<usize>,
    answer: Vec<usize>,
}
impl Sample {
    fn window(&self) -> (Vec<usize>, Vec<i64>) {
        let mut ids = self.prompt.clone();
        ids.extend(&self.answer);
        ids.push(STOP);
        let labels = ids
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if i < self.prompt.len() {
                    -100
                } else {
                    x as i64
                }
            })
            .collect();
        (ids, labels)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    field: String,
    short: Sample,
    long: Sample,
}

#[derive(Serialize, Deserialize)]
struct Endpoint {
    arm: String,
    steps: usize,
    owner: String,
    refusal: Option<String>,
}
#[derive(Serialize, Deserialize)]
struct Training {
    fixture: String,
    base: String,
    lora: String,
    origins: [String; 2],
    endpoints: Vec<Endpoint>,
    menus: String,
    journal: String,
    gradients: usize,
    updates: usize,
    measured_forwards: usize,
}

fn main() -> Result<()> {
    let argv: Vec<_> = std::env::args_os().skip(1).collect();
    match argv.as_slice(){
        [cmd,fixture,out] if cmd=="train"=>train::run(&Fixture::open(Path::new(fixture))?,Path::new(out)),
        [cmd,fixture,trained,out] if cmd=="score"=>evaluate::run(&Fixture::open(Path::new(fixture))?,Path::new(trained),Path::new(out)),
        _=>Err("usage: paired_projection_language train FIXTURE NEW_DIR | score FIXTURE TRAINED NEW_DIR".into()),
    }
}
