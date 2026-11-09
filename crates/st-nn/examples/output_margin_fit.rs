//! Fit several directions from one bounded, local training-only packet. Does not read check rows.
use serde::{Deserialize, Serialize};
use st_nn::causal_lm::{
    CausalLmOutputMarginBank, CausalLmOutputMarginRule as Rule, CausalLmReadoutAccumulator,
    CausalLmResult, NativeCausalLm,
};
use st_nn::geometry_artifact::GeometryAdapterArtifact;
use st_nn::weights::SafeTensorArchive;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::Path;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    geometry: String,
    left_bits: Vec<u64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Deck {
    base: String,
    lora: String,
    layer: usize,
    target: usize,
    alternative: usize,
    positive: Vec<Vec<usize>>,
    negative: Vec<Vec<usize>>,
    slot: Vec<usize>,
    edge_slot: Vec<usize>,
    states: BTreeMap<String, State>,
}
#[derive(Serialize)]
struct Fits<'a> {
    contrasts: &'a [[f64; 2]],
    pair_gradients: &'a [Vec<f64>],
    directions: BTreeMap<&'static str, st_nn::causal_lm::CausalLmOutputMarginDirection>,
}
fn read(p: &Path, limit: usize) -> CausalLmResult<Vec<u8>> {
    let mut b = Vec::new();
    File::open(p)?.take(limit as u64 + 1).read_to_end(&mut b)?;
    if b.len() > limit {
        return Err("example input too large".into());
    }
    Ok(b)
}
fn fits<'a>(
    m: &NativeCausalLm,
    d: &Deck,
    b: &'a CausalLmOutputMarginBank,
    compact: bool,
) -> CausalLmResult<Fits<'a>> {
    let mut directions = BTreeMap::new();
    for (name, labels, rule) in [
        ("mean", &d.slot, Rule::EqualMean),
        ("slot", &d.slot, Rule::CommonCosine),
        ("edge_slot", &d.edge_slot, Rule::CommonCosine),
    ] {
        let dir = b.direction(labels, rule)?;
        if compact && dir.coordinates().is_some() {
            dir.scalar_adapter(m)?;
        }
        directions.insert(name, dir);
    }
    Ok(Fits {
        contrasts: b.pair_contrasts(),
        pair_gradients: b.pair_gradients(),
        directions,
    })
}
fn main() -> CausalLmResult<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: output_margin_fit TRAINING_PACKET NEW_OUTPUT.jsonl".into());
    }
    let dir = Path::new(&args[0]);
    let d: Deck = serde_json::from_slice(&read(&dir.join("deck.json"), 1 << 20)?)?;
    if d.states.is_empty() || d.states.len() > 4 {
        return Err("expected one to four source states".into());
    }
    let raw = SafeTensorArchive::from_owned_bytes(
        read(&dir.join("model.safetensors"), 16 << 20)?,
        Default::default(),
    )?;
    let base = NativeCausalLm::from_gpt2_json(
        &read(&dir.join("config.json"), 1 << 20)?,
        &raw,
        Default::default(),
    )?
    .with_readout_accumulator(CausalLmReadoutAccumulator::F32WideBackward)
    .with_lora_json(std::str::from_utf8(&read(
        &dir.join("lora.json"),
        1 << 20,
    )?)?)?;
    if base.base_model_identity() != d.base || base.lora_artifact_id() != Some(d.lora.as_str()) {
        return Err("wrong base or LoRA".into());
    }
    // This example exports individual gradients as JSON; the library need not export them.
    let w = base.config().width;
    if [
        d.states.len(),
        d.positive.len().saturating_add(3),
        w,
        w.saturating_add(1),
    ]
    .into_iter()
    .try_fold(1usize, |a, b| a.checked_mul(b))
    .is_none_or(|n| n > 1 << 20)
    {
        return Err("example gradient export exceeds its coordinate budget".into());
    }
    let mut out = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[1])?,
    );
    for (state, s) in &d.states {
        let hex = s
            .geometry
            .strip_prefix("sha256:")
            .ok_or("invalid geometry identity")?;
        if hex.len() != 64 || !hex.bytes().all(|x| x.is_ascii_hexdigit()) {
            return Err("invalid geometry digest".into());
        }
        let model = base.with_geometry(GeometryAdapterArtifact::from_json(
            &read(&dir.join(format!("{hex}.json")), 1 << 20)?,
            &s.geometry,
            &d.base,
        )?)?;
        let u: Vec<_> = s.left_bits.iter().copied().map(f64::from_bits).collect();
        for (space, left) in [("dense", None), ("left", Some(u.as_slice()))] {
            let b = model.output_margin_bank(
                d.layer,
                &d.positive,
                &d.negative,
                d.target,
                d.alternative,
                left,
            )?;
            serde_json::to_writer(
                &mut out,
                &(state, space, fits(&model, &d, &b, left.is_some())?),
            )?;
            out.write_all(b"\n")?;
            out.flush()?;
        }
    }
    Ok(())
}
