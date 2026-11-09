//! Small offline driver for the native quartet learner. The fitting request contains no check deck.
//! Usage: native_quartet_projection CONFIG WEIGHTS LORA GEOMETRY < REQUEST.json
use serde::Deserialize;
use serde_json::json;
use st_nn::causal_lm::{
    CausalLmAnswerBinding, CausalLmAnswerGuard, CausalLmAnswerGuardStatus,
    CausalLmAnswerProtection, CausalLmQuartetObjective, CausalLmReadoutAccumulator, CausalLmResult,
    NativeCausalLm, NativeCausalLmProjectionPrefix, NativeCausalLmRankOneProjection,
};
use st_nn::geometry_artifact::GeometryAdapterArtifact;
use st_nn::weights::SafeTensorArchive;
use std::{
    fs::File,
    io::{self, Read},
};

fn bounded(r: impl Read, cap: usize) -> CausalLmResult<Vec<u8>> {
    let mut data = Vec::new();
    r.take(cap as u64 + 1).read_to_end(&mut data)?;
    if data.len() > cap {
        return Err("input exceeds example byte limit".into());
    }
    Ok(data)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    base: String,
    lora: String,
    geometry: String,
    layer: usize,
    site: String,
    left_bits: Vec<u64>,
    action: Action,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Branch {
    prefix: usize,
    suffix: Vec<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Probe {
    recipient: Vec<usize>,
    donor: Vec<usize>,
    position: usize,
    suffix: Vec<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AnswerBatch {
    prompts: Vec<Vec<usize>>,
    gold: Vec<usize>,
    #[serde(default)]
    groups: Vec<[usize; 4]>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AnswerMix {
    ids: [usize; 2],
    mix_bits: u64,
}
#[derive(Clone, Copy, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum AnswerUpdate {
    Observe,
    Descent {
        length: f64,
    },
    Guarded {
        config: CausalLmAnswerGuard,
        protect_other: bool,
    },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectionStart {
    owner: String,
    right_bits: Vec<u64>,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Action {
    Answers {
        batches: Vec<AnswerBatch>,
        order: Vec<usize>,
        update: AnswerUpdate,
        binding: Option<AnswerMix>,
        start: Option<ProjectionStart>,
    },
    Train {
        ids: [usize; 2],
        batches: Vec<Vec<Vec<usize>>>,
        groups: Vec<[usize; 4]>,
        order: Vec<usize>,
        objective: CausalLmQuartetObjective,
        length: f64,
    },
    Score {
        right_bits: Vec<u64>,
        prompts: Vec<Vec<usize>>,
    },
    Branches {
        right_bits: Vec<u64>,
        prefixes: Vec<Vec<usize>>,
        paths: Vec<Branch>,
    },
    KeyProbe {
        right_bits: Vec<u64>,
        cases: Vec<Probe>,
        layers: Vec<usize>,
    },
}

fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|x| x.to_bits()).collect()
}
fn follow(
    p: &NativeCausalLmProjectionPrefix,
    suffix: &[usize],
) -> CausalLmResult<NativeCausalLmProjectionPrefix> {
    if !(1..=64).contains(&suffix.len()) {
        return Err("invalid branch suffix length".into());
    }
    let mut next = p.branch(suffix[0])?;
    for &token in &suffix[1..] {
        next = next.branch(token)?;
    }
    Ok(next)
}

fn main() -> CausalLmResult<()> {
    let argv: Vec<_> = std::env::args().skip(1).collect();
    if argv.len() != 4 {
        return Err("expected CONFIG WEIGHTS LORA GEOMETRY and a JSON request on stdin".into());
    }
    let r: Request = serde_json::from_slice(&bounded(io::stdin().lock(), 1 << 20)?)?;
    let get = |i: usize, cap| bounded(File::open(&argv[i])?, cap);
    let archive = SafeTensorArchive::from_owned_bytes(get(1, 32 << 20)?, Default::default())?;
    let m = NativeCausalLm::from_gpt2_json(&get(0, 1 << 20)?, &archive, Default::default())?
        .with_readout_accumulator(CausalLmReadoutAccumulator::F32WideBackward)
        .with_lora_json(std::str::from_utf8(&get(2, 1 << 20)?)?)?;
    let m = m.with_geometry(GeometryAdapterArtifact::from_json(
        &get(3, 1 << 20)?,
        &r.geometry,
        &r.base,
    )?)?;
    if m.base_model_identity() != r.base
        || m.lora_artifact_id() != Some(r.lora.as_str())
        || m.geometry_artifact_id() != Some(r.geometry.as_str())
    {
        return Err("source identities do not match request".into());
    }
    let left: Vec<_> = r.left_bits.iter().copied().map(f64::from_bits).collect();
    let site = r.site.parse().map_err(|e: &str| e.to_owned())?;
    let restore = |right: Vec<u64>| {
        NativeCausalLmRankOneProjection::new(
            &m,
            r.layer,
            site,
            &left,
            &right.into_iter().map(f64::from_bits).collect::<Vec<_>>(),
        )
    };
    let cap = 1 << 20;
    let data = match r.action {
        Action::Answers {
            batches,
            order,
            update,
            binding,
            start,
        } => {
            if batches.is_empty()
                || batches.len() > 64
                || order.is_empty()
                || order.len() > 256
                || order.iter().any(|&i| i >= batches.len())
                || order.len().saturating_mul(left.len()).saturating_mul(4) > cap
                || (binding.is_none() && batches.iter().any(|b| !b.groups.is_empty()))
            {
                return Err("invalid answer batch schedule or output budget".into());
            }
            let mut a = if let Some(s) = start {
                let a = restore(s.right_bits)?;
                if a.projection_id() != s.owner {
                    return Err("answer run start differs from the pinned projection".into());
                }
                a
            } else {
                restore(vec![0; left.len()])?
            };
            let initial = a.projection_id();
            let mut steps = Vec::with_capacity(order.len());
            if let AnswerUpdate::Guarded {
                config,
                protect_other,
            } = update
            {
                config.validate()?;
                if binding.is_none() || (protect_other && batches.len() != 2) {
                    return Err("guard needs binding and aligned protection batches".into());
                }
            }
            let scheduled = order
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>();
            let mut held = std::collections::HashSet::new();
            let mut stalled = false;
            for i in order {
                let b = &batches[i];
                let mix = binding.as_ref().map(|v| CausalLmAnswerBinding {
                    ids: v.ids,
                    quartets: &b.groups,
                    mix: f64::from_bits(v.mix_bits),
                });
                let report = match update {
                    AnswerUpdate::Observe => {
                        serde_json::to_value(a.answer_gradients(&b.prompts, &b.gold, mix)?)?
                    }
                    AnswerUpdate::Descent { length } => {
                        serde_json::to_value(a.step_answers(&b.prompts, &b.gold, mix, length)?)?
                    }
                    AnswerUpdate::Guarded {
                        config,
                        protect_other,
                    } => {
                        let protect = protect_other.then(|| {
                            let p = &batches[1 - i];
                            CausalLmAnswerProtection {
                                prompts: &p.prompts,
                                gold: &p.gold,
                            }
                        });
                        let r = a.step_answers_guarded(
                            &b.prompts,
                            &b.gold,
                            mix.ok_or("missing binding")?,
                            protect,
                            config,
                        )?;
                        if r.status == CausalLmAnswerGuardStatus::Held {
                            held.insert(i);
                        } else {
                            held.clear();
                        }
                        stalled = held == scheduled;
                        serde_json::to_value(r)?
                    }
                };
                steps.push(json!({"batch":i,"report":report,"owner":a.projection_id(),"right_bits":a.right().iter().map(|x|x.to_bits()).collect::<Vec<_>>()}));
                if stalled {
                    break;
                }
            }
            let mut packet = json!({"initial":initial,"owner":a.projection_id(),"steps":steps});
            if matches!(update, AnswerUpdate::Guarded { .. }) {
                packet["stalled"] = json!(stalled);
            }
            packet
        }
        Action::Train {
            ids,
            batches,
            groups,
            order,
            objective,
            length,
        } => {
            if order.is_empty()
                || order.len() > 256
                || batches.is_empty()
                || batches.len() > 64
                || order.iter().any(|&i| i >= batches.len())
            {
                return Err("invalid training schedule".into());
            }
            let mut a = NativeCausalLmRankOneProjection::new(
                &m,
                r.layer,
                site,
                &left,
                &vec![0.; left.len()],
            )?;
            let mut steps = Vec::with_capacity(order.len());
            for i in order {
                let report = a.step_quartets(&batches[i], ids, &groups, objective, length)?;
                steps.push(json!({"batch":i,"report":report,"right_bits":a.right().iter().map(|x|x.to_bits()).collect::<Vec<_>>()}));
            }
            json!({"steps":steps})
        }
        Action::Score {
            right_bits,
            prompts,
        } => {
            if prompts.is_empty()
                || prompts.len() > 128
                || prompts.len().saturating_mul(m.config().vocabulary) > 1 << 20
            {
                return Err("invalid scoring size".into());
            }
            let a = NativeCausalLmRankOneProjection::new(
                &m,
                r.layer,
                site,
                &left,
                &right_bits
                    .into_iter()
                    .map(f64::from_bits)
                    .collect::<Vec<_>>(),
            )?;
            let rows = prompts
                .iter()
                .map(|q| {
                    Ok(a.last_logits(q)?
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>())
                })
                .collect::<CausalLmResult<Vec<_>>>()?;
            json!({"logit_bits":rows})
        }
        Action::Branches {
            right_bits,
            prefixes,
            paths,
        } => {
            if prefixes.is_empty()
                || prefixes.len() > 64
                || paths.is_empty()
                || paths.len() > 128
                || paths.len().saturating_mul(m.config().vocabulary) > cap / 2
                || paths.iter().any(|p| {
                    p.prefix >= prefixes.len() || p.suffix.is_empty() || p.suffix.len() > 64
                })
            {
                return Err("invalid prefix or branch roster".into());
            }
            let mut values = 0usize;
            for p in &prefixes {
                values = values
                    .checked_add(m.context_cache_values(p.len(), cap)?)
                    .ok_or("cache total overflow")?;
                if values > cap {
                    return Err("combined prefixes exceed example cache budget".into());
                }
            }
            let a = restore(right_bits)?;
            let states = prefixes
                .into_iter()
                .map(|p| a.prefill(p, cap))
                .collect::<CausalLmResult<Vec<_>>>()?;
            let mut rows = vec![];
            for path in paths {
                let z = follow(&states[path.prefix], &path.suffix)?;
                rows.push(json!({"cached":bits(z.logits()),"ordinary":bits(&a.last_logits(z.token_ids())?)}));
            }
            json!({"owner":a.projection_id(),"prefills":states.len(),"rows":rows})
        }
        Action::KeyProbe {
            right_bits,
            cases,
            layers,
        } => {
            if cases.is_empty()
                || cases.len() > 64
                || layers.is_empty()
                || layers.len() > 4
                || layers.iter().any(|&i| i >= m.config().layers)
                || cases.len().saturating_mul(
                    (1 + 2 * layers.len())
                        .saturating_mul(m.config().vocabulary)
                        .saturating_add(layers.len() * 2 * m.config().width),
                ) > cap
            {
                return Err("invalid cached-key probe roster".into());
            }
            let a = restore(right_bits)?;
            let mut rows = vec![];
            for case in cases {
                let recipient = a.prefill(case.recipient, cap)?;
                let donor = a.prefill(case.donor, cap)?;
                let factual = bits(follow(&recipient, &case.suffix)?.logits());
                let mut probes = vec![];
                for &layer in &layers {
                    probes.push(json!({"layer":layer,"recipient_key":bits(&recipient.key_row(layer,case.position)?),
                        "donor_key":bits(&donor.key_row(layer,case.position)?),
                        "self_logits":bits(&recipient.probe_key_from(&recipient,layer,case.position,&case.suffix)?),
                        "donor_logits":bits(&recipient.probe_key_from(&donor,layer,case.position,&case.suffix)?)}));
                }
                rows.push(json!({"factual":factual,"probes":probes}));
            }
            json!({"owner":a.projection_id(),"rows":rows})
        }
    };
    println!(
        "{}",
        json!({"base":r.base,"lora":r.lora,"geometry":r.geometry,"payload":data})
    );
    Ok(())
}
