use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    slot: usize,
    field: String,
    prompt: Vec<usize>,
}

pub(super) fn run(fix: &Fixture, trained: &Path, dir: &Path) -> Result<()> {
    fs::create_dir(dir)?;
    let train_bytes = read(&trained.join("training.json"), 16_384)?;
    let tr: Training = serde_json::from_slice(&train_bytes)?;
    let initial = fix.origins()?;
    require(
        tr.fixture == FIXTURE
            && tr.base == initial[0].base_model_identity()
            && initial[0].lora_artifact_id() == Some(tr.lora.as_str())
            && tr.endpoints.len() == 4,
        "foreign training index",
    )?;
    let mut states = vec![
        ("absolute_start", initial[0].clone()),
        ("anchored_start", initial[1].clone()),
    ];
    for (j, m) in initial.iter().enumerate() {
        require(
            tr.origins[j] == geom(m)?.artifact_id(),
            "start owner mismatch",
        )?;
    }
    for (j, e) in tr.endpoints.iter().enumerate() {
        require(
            e.arm == ARMS[j] && (e.steps == 18) == e.refusal.is_none() && e.steps <= 18,
            "unexpected endpoint",
        )?;
        if e.refusal.is_some() {
            continue;
        }
        let raw = read(
            &trained.join(format!("{}-18.geometry.json", ARMS[j])),
            32_768,
        )?;
        let restored = GeometryAdapterArtifact::from_json(&raw, &e.owner, &tr.base)?;
        let m = initial[j % 2].with_geometry(restored)?;
        require(
            m.lora_artifact_id() == Some(tr.lora.as_str()),
            "scoring with a different LoRA",
        )?;
        states.push((ARMS[j], m));
    }
    let menu_bytes = read(&trained.join("menus.json"), 8192)?;
    require(sha(&menu_bytes) == tr.menus, "training menu changed")?;
    let menus: Menus = serde_json::from_slice(&menu_bytes)?;
    let queries: Vec<Query> = fix.parsed("queries.json")?;
    require(
        queries.len() == 18 && menus.len() == 3,
        "wrong query or field count",
    )?;
    let mut docs = Vec::new();
    let mut masks = Vec::new();
    let mut roster = Vec::new();
    for (k, q) in queries.iter().enumerate() {
        require(
            q.slot == k && q.prompt.len() == 5 && q.prompt.iter().all(|&v| v < V && v != STOP),
            "invalid label-free query",
        )?;
        let menu = menus.get(&q.field).ok_or("field absent from training")?;
        require(menu.len() == 6, "wrong field support")?;
        for a in menu {
            require(
                (1..=2).contains(&a.len()) && a.iter().all(|&v| v < V && v != STOP),
                "invalid content",
            )?;
            let (d, labels) = Sample {
                prompt: q.prompt.clone(),
                answer: a.clone(),
            }
            .window();
            docs.push(d);
            masks.push(labels);
            roster.push((k, a.clone()));
        }
    }
    let refs: Vec<_> = states.iter().map(|(_, m)| m).collect();
    let mut log = fresh(&dir.join("calls.jsonl"))?;
    event(
        &mut log,
        json!({"event":"start","kind":"by_answer","documents":docs.len(),"owners":states.len()}),
    )?;
    let detail = NativeCausalLm::evaluate_variants_by_answer(
        &refs,
        &docs,
        Some(&masks),
        CausalLmEvaluationConfig {
            context_length: 12,
            microbatch_windows: 3,
        },
    )?;
    require(
        detail.document_count() == 108 && detail.answers().len() == 108,
        "missing native candidate answers",
    )?;
    let native = put(&dir.join("by-answer.json"), &detail)?;
    event(
        &mut log,
        json!({"event":"return","kind":"by_answer","sha":native}),
    )?;
    let owners: Vec<_> = states
        .iter()
        .map(|(name, m)| Ok(json!({"state":name,"owner":geom(m)?.artifact_id()})))
        .collect::<Result<_>>()?;
    let mut rows = 0;
    let mut cells = 0;
    let mut max_error = 0f64;
    let mut ledger = fresh(&dir.join("lines.jsonl"))?;
    for (k, q) in queries.iter().enumerate() {
        let menu = &menus[&q.field];
        let mut need = BTreeSet::new();
        for a in menu {
            for i in 0..=a.len() {
                need.insert(&a[..i]);
            }
        }
        require(need.len() == 10, "wrong distinct prefix count")?;
        for (j, (name, m)) in states.iter().enumerate() {
            let mut table = BTreeMap::new();
            for prefix in &need {
                rows += 1;
                require(rows <= 1080, "scoring prefix cap")?;
                let ident = json!({"slot":k,"state":name,"owner":geom(m)?.artifact_id(),"prefix":prefix,"n":rows});
                event(
                    &mut log,
                    json!({"event":"start","kind":"prefix","who":ident}),
                )?;
                let mut ids = q.prompt.clone();
                ids.extend_from_slice(prefix);
                let values = m.forward(&ids, [1, ids.len()], true)?.read_values()?;
                require(
                    values.len() == V && values.iter().all(|v| v.is_finite()),
                    "invalid scoring row",
                )?;
                event(
                    &mut log,
                    json!({"event":"return","kind":"prefix","who":ident,"row_bits":values.iter().map(|v|v.to_bits()).collect::<Vec<_>>()}),
                )?;
                table.insert(*prefix, values);
            }
            let mut lines = Vec::new();
            for (t, a) in menu.iter().enumerate() {
                let doc = k * 6 + t;
                let answer = &detail.answers()[doc];
                require(
                    roster[doc] == (k, a.clone())
                        && (answer.document, answer.start, answer.end)
                            == (doc, q.prompt.len(), docs[doc].len())
                        && answer.scores.len() == states.len(),
                    "native candidate alignment",
                )?;
                let parts: Vec<_> = a
                    .iter()
                    .chain([&STOP])
                    .enumerate()
                    .map(|(i, &v)| Ok(TargetClassScore::from_logits(&table[&a[..i]], v)?.nll))
                    .collect::<Result<_>>()?;
                let total = parts.iter().sum::<f64>();
                let native_total = answer.scores[j].nll * parts.len() as f64;
                max_error = max_error.max((total - native_total).abs());
                require(
                    (total - native_total).abs() <= 2e-5,
                    "prefix log likelihood differs from native answer scorer",
                )?;
                lines.push(
                    json!({"content":a,"parts":parts,"line":total,"native_line":native_total}),
                );
                cells += 1;
            }
            event(&mut ledger, json!({"slot":k,"state":name,"lines":lines}))?;
        }
    }
    ledger.sync_all()?;
    log.sync_all()?;
    require(
        rows == 180 * states.len() && cells == 108 * states.len(),
        "incomplete endpoint scoring",
    )?;
    put(
        &dir.join("scoring.json"),
        &json!({"fixture":FIXTURE,"training":sha(&train_bytes),"owners":owners,
        "documents":108,"prefix_rows":rows,"line_cells":cells,"max_abs_difference":max_error,"native":native,
        "journal":sha(&read(&dir.join("calls.jsonl"),4<<20)?),"lines":sha(&read(&dir.join("lines.jsonl"),1<<20)?)}),
    )?;
    Ok(())
}
