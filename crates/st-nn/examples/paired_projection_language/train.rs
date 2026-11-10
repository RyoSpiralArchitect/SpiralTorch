use super::*;

const RATE: f32 = 0.035;
const MIX: f64 = 0.5;

fn fit(pairs: &[Pair]) -> Result<Menus> {
    let mut menus = Menus::new();
    let mut prompts = BTreeSet::new();
    require(pairs.len() == 18, "wrong pair denominator")?;
    for p in pairs {
        require(
            ["color", "shape", "material"].contains(&p.field.as_str())
                && p.short.answer.len() == 1
                && p.long.answer.len() == 2
                && p.long.answer[1] == p.short.answer[0]
                && p.long.answer[0] != p.short.answer[0],
            "not reciprocal short/long heads",
        )?;
        for s in [&p.short, &p.long] {
            require(
                s.prompt.len() == 5
                    && s.prompt
                        .iter()
                        .chain(&s.answer)
                        .all(|&id| id < V && id != STOP)
                    && prompts.insert(&s.prompt),
                "training support or duplicate prompt",
            )?;
            menus
                .entry(p.field.clone())
                .or_default()
                .insert(s.answer.clone());
        }
    }
    require(
        menus.len() == 3 && menus.values().all(|m| m.len() == 6),
        "menu support",
    )?;
    Ok(menus)
}

fn objective(m: &NativeCausalLm, s: &Sample, rival: usize) -> Result<(f32, Vec<Vec<u32>>)> {
    let (doc, _) = s.window();
    let x = &doc[..doc.len() - 1];
    let vals = m.forward(x, [1, x.len()], false)?.read_values()?;
    require(
        vals.len() == V * x.len(),
        "wrong independent vocabulary rows",
    )?;
    let mut rows = Vec::new();
    let mut loss = 0.;
    for (j, &target) in doc[s.prompt.len()..].iter().enumerate() {
        let start = (s.prompt.len() + j - 1) * V;
        let row = &vals[start..start + V];
        let ce = TargetClassScore::from_logits(row, target)?.nll;
        let value = if j == 0 {
            let d = f64::from(row[rival]) - f64::from(row[target]);
            (1. - MIX) * ce + MIX * (d.max(0.) + (-d.abs()).exp().ln_1p())
        } else {
            ce
        };
        loss += value;
        rows.push(row.iter().map(|v| v.to_bits()).collect());
    }
    Ok(((loss / rows.len() as f64) as f32, rows))
}

fn measured(
    m: &NativeCausalLm,
    s: &Sample,
    rival: usize,
    f: &mut File,
    who: &Value,
    phase: &str,
    n: &mut usize,
) -> Result<f32> {
    *n += 1;
    require(*n <= 288, "independent training forward cap")?;
    event(f, json!({"event":"start","kind":phase,"n":n,"who":who}))?;
    let (loss, rows) = objective(m, s, rival)?;
    event(
        f,
        json!({"event":"return","kind":phase,"n":n,"who":who,"owner":geom(m)?.artifact_id(),"loss":loss,"row_bits":rows}),
    )?;
    Ok(loss)
}

fn eligible_refusal(msg: &str) -> bool {
    matches!(
        msg,
        "requested geometry direction or distance cannot be represented accurately in f32"
            | "no representable strict common descent for this pair and step"
            | "no strict common descent for a zero gradient or zero distance"
            | "cannot assign a positive distance to a zero geometry direction"
    )
}

pub(super) fn run(fix: &Fixture, dir: &Path) -> Result<()> {
    fs::create_dir(dir)?;
    let pairs: Vec<Pair> = fix.parsed("train.json")?;
    let menus = fit(&pairs)?;
    let menus_pin = put(&dir.join("menus.json"), &menus)?;
    let initial = fix.origins()?;
    let ids = [
        geom(&initial[0])?.artifact_id().to_owned(),
        geom(&initial[1])?.artifact_id().to_owned(),
    ];
    for (j, k) in ["absolute_start", "anchored_start"].iter().enumerate() {
        save_geom(dir, k, &initial[j])?;
    }
    let base = initial[0].base_model_identity().to_owned();
    let lora = initial[0]
        .lora_artifact_id()
        .ok_or("missing LoRA")?
        .to_owned();
    let frozen = initial[0].lora_to_json()?;
    let mut arms = ARMS.map(|a| initial[usize::from(a.starts_with("anchored"))].clone());
    let mut ends: Vec<_> = ARMS
        .iter()
        .zip(&arms)
        .map(|(&a, m)| {
            Ok(Endpoint {
                arm: a.into(),
                steps: 0,
                owner: geom(m)?.artifact_id().into(),
                refusal: None,
            })
        })
        .collect::<Result<_>>()?;
    let (mut n, mut updates, mut probes) = (0, 0, 0);
    let mut f = fresh(&dir.join("events.jsonl"))?;
    for (i, pair) in pairs.iter().enumerate() {
        let mut radius = None;
        let mut initial_controls = None;
        for (j, (m, entry)) in arms.iter_mut().zip(&mut ends).enumerate() {
            if entry.refusal.is_some() {
                continue;
            }
            let pre = geom(m)?.artifact_id();
            let who = json!({"step":i+1,"arm":entry.arm,"pre":pre});
            require(entry.steps == i && entry.owner == pre, "stale model")?;
            let mut grads = Vec::new();
            let mut controls = Vec::new();
            for (side, (s, other)) in [(&pair.short, &pair.long), (&pair.long, &pair.short)]
                .into_iter()
                .enumerate()
            {
                let sw = json!({"step":i+1,"arm":entry.arm,"pre":pre,"side":side});
                let (doc, labels) = s.window();
                let rival = other.answer[0];
                let shape = [1, doc.len()];
                let plan = [DeclaredRivalRow {
                    position: s.prompt.len(),
                    target: s.answer[0],
                    rivals: vec![rival],
                }];
                n += 1;
                require(n <= 144, "gradient cap")?;
                event(
                    &mut f,
                    json!({"event":"start","kind":"gradient","n":n,"who":sw}),
                )?;
                let g =
                    m.geometry_declared_rival_gradients(&doc, shape, Some(&labels), &plan, MIX)?;
                require(
                    g.base_model_identity() == base
                        && g.lora_artifact_id() == Some(lora.as_str())
                        && g.geometry_artifact_id() == Some(pre)
                        && g.readout_accumulator() == m.readout_accumulator()
                        && g.window_shape() == shape
                        && g.target_tokens() == s.answer.len() + 1,
                    "gradient owner or mask",
                )?;
                let report = g
                    .declared_rival_report()
                    .ok_or("missing declared objective")?;
                require(
                    report.mix == MIX
                        && report.loss == g.loss()
                        && report.contrasted_tokens == 1
                        && report.rival_ids == 1,
                    "wrong reduced loss",
                )?;
                let params = g.parameters().get(SITE).ok_or("missing site")?;
                require(
                    g.parameters().len() == 1 && params.len() == 2,
                    "wrong trainable coordinates",
                )?;
                let mut bits = Vec::new();
                for key in ["bias", "gate"] {
                    let block = params.get(key).ok_or("missing channel")?;
                    require(
                        block.len() == WIDTH && block.iter().all(|x| x.is_finite()),
                        "invalid gradient",
                    )?;
                    bits.extend(block.iter().map(|x| x.to_bits()));
                }
                event(
                    &mut f,
                    json!({"event":"return","kind":"gradient","n":n,"who":sw,"bits":bits,"report":report}),
                )?;
                controls.push((bits, g.loss().to_bits()));
                let check = measured(m, s, rival, &mut f, &sw, "pre_loss", &mut probes)?;
                require(
                    (check - g.loss()).abs() <= 2e-6,
                    "independent pre-loss disagrees with training",
                )?;
                grads.push(g);
            }
            if i == 0 {
                if let Some(first) = &initial_controls {
                    require(
                        *first == controls,
                        "different starting gradients across charts",
                    )?;
                } else {
                    initial_controls = Some(controls);
                }
            }
            let policy = if j < 2 {
                CausalLmPairWeighting::Equal
            } else {
                CausalLmPairWeighting::CommonDescent
            };
            event(
                &mut f,
                json!({"event":"start","kind":"update","who":who,"requested":radius}),
            )?;
            let (next, r) =
                match m.step_paired_geometry_sgd(&grads[0], &grads[1], RATE, policy, radius) {
                    Ok(pair) => pair,
                    Err(err) if j != 0 && eligible_refusal(&err.to_string()) => {
                        let msg = err.to_string();
                        event(
                            &mut f,
                            json!({"event":"refused","kind":"update","who":who,"reason":msg}),
                        )?;
                        entry.refusal = Some(msg);
                        continue;
                    }
                    Err(e) => return Err(e),
                };
            let actual = geom(m)?.trainable_delta_l2(geom(&next)?)?;
            require(
                actual > 0.
                    && r.applied.l2 == actual
                    && r.trainable_values == 2 * WIDTH
                    && r.pre_geometry_id == pre
                    && r.post_geometry_id == geom(&next)?.artifact_id()
                    && next.base_model_identity() == base
                    && next.lora_to_json()? == frozen,
                "invalid accepted update",
            )?;
            if let Some(d) = radius {
                let matched = r.distance_match.as_ref().ok_or("missing step match")?;
                require(
                    matched.requested_l2 == d
                        && (actual / d - 1.).abs() <= 1e-4
                        && matched.relative_vector_error <= 1e-4,
                    "unmatched step",
                )?;
            } else {
                require(
                    j == 0 && r.distance_match.is_none(),
                    "missing absolute reference",
                )?;
                radius = Some(actual);
            }
            require(
                j < 2 || r.applied.both_descend,
                "unguarded accepted direction",
            )?;
            save_geom(dir, &format!("{}-{:02}", entry.arm, i + 1), &next)?;
            updates += 1;
            event(
                &mut f,
                json!({"event":"return","kind":"update","who":who,"receipt":r,"reference":radius}),
            )?;
            for (side, (s, other)) in [(&pair.short, &pair.long), (&pair.long, &pair.short)]
                .into_iter()
                .enumerate()
            {
                let sw = json!({"step":i+1,"arm":entry.arm,"pre":pre,"side":side});
                measured(
                    &next,
                    s,
                    other.answer[0],
                    &mut f,
                    &sw,
                    "post_loss",
                    &mut probes,
                )?;
            }
            entry.steps += 1;
            entry.owner = geom(&next)?.artifact_id().into();
            *m = next;
        }
    }
    f.sync_all()?;
    require(
        ends[0].steps == 18 && ends.iter().all(|x| (x.steps == 18) == x.refusal.is_none()),
        "incomplete arm without a refusal",
    )?;
    let journal = sha(&read(&dir.join("events.jsonl"), 8 << 20)?);
    put(
        &dir.join("training.json"),
        &Training {
            fixture: FIXTURE.into(),
            base,
            lora,
            origins: ids,
            endpoints: ends,
            menus: menus_pin,
            journal,
            gradients: n,
            updates,
            measured_forwards: probes,
        },
    )?;
    Ok(())
}
