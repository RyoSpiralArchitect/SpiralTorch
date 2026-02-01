// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! LLM model-zoo: character-level fine-tuning using Z-space coherence scan
//! (no tokenizer / no BPE).

#[path = "_shared/backend.rs"]
mod backend;
#[path = "_shared/text_corpus.rs"]
mod text_corpus;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use st_core::plugin::{
    global_registry, PluginEvent, PluginEventJsonlWriter, PluginEventJsonlWriterConfig,
};
use st_nn::layers::ZSpaceSoftmax;
use st_nn::{
    load_json, save_json, CategoricalCrossEntropy, Embedding, Module, ModuleTrainer, PureResult,
    Relu, RoundtableConfig, Sequential, Tensor, TensorError, ZSpaceCoherenceScan,
};
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const FORMAT_ID: &str = "st-char-lm-coherence-v1";
const DEFAULT_UNK: char = '\u{FFFD}';
const RUN_SCHEMA: &str = "st.modelzoo.run.v1";

#[derive(Debug, Clone, Serialize)]
struct RunMeta {
    schema: String,
    arch: String,
    backend: String,
    device_caps: backend::DeviceCapsMeta,
    format: String,
    data_paths: Vec<String>,
    data_file_count: usize,
    data_files_manifest: String,
    events_path: Option<String>,
    weights_loaded_from: Option<String>,
    steps: usize,
    embed_dim: usize,
    hidden: usize,
    memory: usize,
    epochs: usize,
    batches_per_epoch: usize,
    batch: usize,
    learning_rate: f32,
    curvature: f32,
    temperature: f32,
    gen_len: usize,
    top_k: usize,
    seed: u64,
    prompt: String,
    vocab_size: usize,
}

#[derive(Debug, Clone, Serialize)]
struct EpochMetric {
    epoch: usize,
    batches: usize,
    average_loss: f32,
}

#[derive(Debug, Clone)]
struct Vocab {
    unk: char,
    symbols: Vec<char>,
    index: HashMap<char, usize>,
}

impl Vocab {
    fn from_symbols(unk: char, symbols: Vec<char>) -> Self {
        let mut index = HashMap::with_capacity(symbols.len());
        for (idx, ch) in symbols.iter().copied().enumerate() {
            index.insert(ch, idx);
        }
        Self {
            unk,
            symbols,
            index,
        }
    }

    fn build_from_text(text: &str, unk: char) -> Self {
        let mut set = BTreeSet::new();
        for ch in text.chars() {
            if ch != unk {
                set.insert(ch);
            }
        }
        let mut symbols = Vec::with_capacity(set.len() + 1);
        symbols.push(unk);
        symbols.extend(set.into_iter());
        Self::from_symbols(unk, symbols)
    }

    fn encode(&self, ch: char) -> usize {
        self.index.get(&ch).copied().unwrap_or(0)
    }

    fn decode(&self, idx: usize) -> char {
        self.symbols.get(idx).copied().unwrap_or(self.unk)
    }

    fn len(&self) -> usize {
        self.symbols.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CharLmMeta {
    format: String,
    steps: usize,
    embed_dim: usize,
    hidden: usize,
    memory: usize,
    curvature: f32,
    temperature: f32,
    unk: char,
    symbols: Vec<char>,
}

impl CharLmMeta {
    fn new(
        steps: usize,
        embed_dim: usize,
        hidden: usize,
        memory: usize,
        curvature: f32,
        temperature: f32,
        vocab: &Vocab,
    ) -> Self {
        Self {
            format: FORMAT_ID.to_string(),
            steps,
            embed_dim,
            hidden,
            memory,
            curvature,
            temperature,
            unk: vocab.unk,
            symbols: vocab.symbols.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct Args {
    data_paths: Vec<PathBuf>,
    load_weights: Option<PathBuf>,
    save_weights: Option<PathBuf>,
    run_dir: Option<PathBuf>,
    backend: String,
    events: Option<PathBuf>,
    steps: usize,
    embed_dim: usize,
    hidden: usize,
    memory: usize,
    epochs: usize,
    batches_per_epoch: usize,
    batch: usize,
    learning_rate: f32,
    curvature: f32,
    temperature: f32,
    gen_len: usize,
    top_k: usize,
    seed: u64,
    prompt: Option<String>,
}

impl Args {
    fn parse() -> Result<Self, TensorError> {
        let mut argv = env::args().skip(1).peekable();
        let mut data_args: Vec<String> = Vec::new();
        while let Some(arg) = argv.peek() {
            if arg.starts_with("--") {
                break;
            }
            data_args.push(argv.next().unwrap());
        }
        if data_args.is_empty() {
            return Err(TensorError::Generic(
                "usage: cargo run -p st-nn --example modelzoo_llm_char_coherence_scan -- <text_or_dir> [<text_or_dir> ...] [--load weights.json] [--save weights.json] [--run-dir PATH] [--backend auto|wgpu|cuda|hip|cpu] [--events PATH] [--steps N] [--embed-dim N] [--hidden N] [--memory N] [--epochs N] [--batches N] [--batch N] [--lr F] [--curvature F] [--temperature F] [--gen N] [--topk N] [--seed N] [--prompt STR]"
                    .to_string(),
            ));
        }

        let mut args = Self {
            data_paths: data_args.into_iter().map(PathBuf::from).collect(),
            load_weights: None,
            save_weights: None,
            run_dir: None,
            backend: "auto".to_string(),
            events: None,
            steps: 32,
            embed_dim: 32,
            hidden: 64,
            memory: 16,
            epochs: 6,
            batches_per_epoch: 24,
            batch: 8,
            learning_rate: 2e-2,
            curvature: -1.0,
            temperature: 1.0,
            gen_len: 200,
            top_k: 32,
            seed: 42,
            prompt: None,
        };

        while let Some(flag) = argv.next() {
            match flag.as_str() {
                "--load" => args.load_weights = Some(PathBuf::from(take_arg(&mut argv, "--load")?)),
                "--save" => args.save_weights = Some(PathBuf::from(take_arg(&mut argv, "--save")?)),
                "--run-dir" => {
                    args.run_dir = Some(PathBuf::from(take_arg(&mut argv, "--run-dir")?))
                }
                "--backend" => args.backend = take_arg(&mut argv, "--backend")?,
                "--events" => args.events = Some(PathBuf::from(take_arg(&mut argv, "--events")?)),
                "--steps" => args.steps = take_parse(&mut argv, "--steps")?,
                "--embed-dim" => args.embed_dim = take_parse(&mut argv, "--embed-dim")?,
                "--hidden" => args.hidden = take_parse(&mut argv, "--hidden")?,
                "--memory" => args.memory = take_parse(&mut argv, "--memory")?,
                "--epochs" => args.epochs = take_parse(&mut argv, "--epochs")?,
                "--batches" => args.batches_per_epoch = take_parse(&mut argv, "--batches")?,
                "--batch" => args.batch = take_parse(&mut argv, "--batch")?,
                "--lr" => args.learning_rate = take_parse(&mut argv, "--lr")?,
                "--curvature" => args.curvature = take_parse(&mut argv, "--curvature")?,
                "--temperature" => args.temperature = take_parse(&mut argv, "--temperature")?,
                "--gen" => args.gen_len = take_parse(&mut argv, "--gen")?,
                "--topk" => args.top_k = take_parse(&mut argv, "--topk")?,
                "--seed" => args.seed = take_parse(&mut argv, "--seed")?,
                "--prompt" => args.prompt = Some(take_arg(&mut argv, "--prompt")?),
                "--help" | "-h" => {
                    return Err(TensorError::Generic(
                        "usage: cargo run -p st-nn --example modelzoo_llm_char_coherence_scan -- <text_or_dir> [<text_or_dir> ...] [--load weights.json] [--save weights.json] [--run-dir PATH] [--backend auto|wgpu|cuda|hip|cpu] [--events PATH] [--steps N] [--embed-dim N] [--hidden N] [--memory N] [--epochs N] [--batches N] [--batch N] [--lr F] [--curvature F] [--temperature F] [--gen N] [--topk N] [--seed N] [--prompt STR]"
                            .to_string(),
                    ));
                }
                other => {
                    return Err(TensorError::Generic(format!(
                        "unknown flag: {other}. Try --help"
                    )));
                }
            }
        }

        if args.steps == 0
            || args.embed_dim == 0
            || args.hidden == 0
            || args.memory == 0
            || args.batch == 0
        {
            return Err(TensorError::InvalidValue {
                label: "char_lm_coherence_invalid_dims",
            });
        }
        if args.memory > args.steps {
            return Err(TensorError::InvalidDimensions {
                rows: args.memory,
                cols: args.steps,
            });
        }
        Ok(args)
    }
}

fn take_arg<I>(argv: &mut I, flag: &'static str) -> Result<String, TensorError>
where
    I: Iterator<Item = String>,
{
    argv.next()
        .ok_or_else(|| TensorError::Generic(format!("missing value for {flag}. Try --help")))
}

fn take_parse<I, T>(argv: &mut I, flag: &'static str) -> Result<T, TensorError>
where
    I: Iterator<Item = String>,
    T: std::str::FromStr,
{
    let raw = take_arg(argv, flag)?;
    raw.parse::<T>()
        .map_err(|_| TensorError::Generic(format!("invalid value for {flag}: {raw}")))
}

fn meta_path_for_weights(weights_path: &Path) -> PathBuf {
    let file_name = weights_path
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("char_lm.json");
    let meta_file = file_name
        .strip_suffix(".json")
        .map(|stem| format!("{stem}.meta.json"))
        .unwrap_or_else(|| format!("{file_name}.meta.json"));
    weights_path.with_file_name(meta_file)
}

fn read_meta(path: &Path) -> PureResult<CharLmMeta> {
    let reader = File::open(path).map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    serde_json::from_reader(reader).map_err(|err| TensorError::SerializationError {
        message: err.to_string(),
    })
}

fn write_meta(path: &Path, meta: &CharLmMeta) -> PureResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| TensorError::IoError {
            message: err.to_string(),
        })?;
    }
    let writer = File::create(path).map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    serde_json::to_writer_pretty(writer, meta).map_err(|err| TensorError::SerializationError {
        message: err.to_string(),
    })?;
    Ok(())
}

fn default_run_dir() -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    PathBuf::from("models/runs").join(format!("{}_{}", now.as_secs(), now.subsec_nanos()))
}

fn build_model(
    vocab_size: usize,
    steps: usize,
    embed_dim: usize,
    hidden: usize,
    memory: usize,
    curvature: f32,
    temperature: f32,
) -> PureResult<Sequential> {
    let mut model = Sequential::new();
    model.push(Embedding::new("embed", vocab_size, embed_dim)?);
    model.push(ZSpaceCoherenceScan::new(
        embed_dim,
        steps,
        memory,
        curvature,
        temperature,
    )?);
    model.push(st_nn::Linear::new("mix", embed_dim, hidden)?);
    model.push(Relu::new());
    model.push(st_nn::Linear::new("head", hidden, vocab_size)?);
    model.push(ZSpaceSoftmax::new(curvature, temperature)?);
    Ok(model)
}

fn encode_text(text: &str, vocab: &Vocab) -> Vec<usize> {
    text.chars().map(|ch| vocab.encode(ch)).collect()
}

fn encode_context_indices(context: &[usize]) -> PureResult<Tensor> {
    let mut data = Vec::with_capacity(context.len());
    for &idx in context {
        data.push(idx as f32);
    }
    Tensor::from_vec(1, context.len(), data)
}

fn build_random_batch(
    tokens: &[usize],
    vocab_size: usize,
    steps: usize,
    batch: usize,
    rng: &mut impl Rng,
) -> PureResult<(Tensor, Tensor)> {
    let max_start = tokens.len().saturating_sub(steps);
    if max_start == 0 {
        return Err(TensorError::EmptyInput("char_lm_tokens"));
    }
    let mut x = Tensor::zeros(batch, steps)?;
    let mut y = Tensor::zeros(batch, vocab_size)?;
    {
        let x_data = x.data_mut();
        let y_data = y.data_mut();
        for row in 0..batch {
            let start = rng.gen_range(0..max_start);
            for t in 0..steps {
                x_data[row * steps + t] = tokens[start + t] as f32;
            }
            let target = tokens[start + steps];
            if target < vocab_size {
                y_data[row * vocab_size + target] = 1.0;
            }
        }
    }
    Ok((x, y))
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &v) in values.iter().enumerate() {
        if v.is_finite() && v > best_val {
            best = idx;
            best_val = v;
        }
    }
    best
}

fn sample_from_probs(probs: &[f32], top_k: usize, rng: &mut impl Rng) -> usize {
    let vocab = probs.len();
    if vocab == 0 {
        return 0;
    }
    if top_k == 1 {
        return argmax(probs);
    }

    let mut candidates: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, p)| (idx, if p.is_finite() && p > 0.0 { p } else { 0.0 }))
        .collect();
    if top_k > 0 && top_k < candidates.len() {
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);
    }

    let total: f32 = candidates.iter().map(|(_, w)| *w).sum();
    if !total.is_finite() || total <= 0.0 {
        return argmax(probs);
    }
    let mut threshold = rng.gen_range(0.0..total);
    for (idx, weight) in candidates {
        threshold -= weight;
        if threshold <= 0.0 {
            return idx;
        }
    }
    argmax(probs)
}

fn generate_text(
    model: &Sequential,
    vocab: &Vocab,
    steps: usize,
    prompt: &str,
    gen_len: usize,
    top_k: usize,
    seed: u64,
) -> PureResult<String> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut context: Vec<usize> = prompt.chars().map(|ch| vocab.encode(ch)).collect();
    if context.len() < steps {
        let mut padded = vec![0usize; steps - context.len()];
        padded.extend(context);
        context = padded;
    } else if context.len() > steps {
        context = context[context.len() - steps..].to_vec();
    }

    let mut out = String::from(prompt);
    for _ in 0..gen_len {
        let x = encode_context_indices(&context)?;
        let probs = model.forward(&x)?;
        let row = &probs.data()[..vocab.len()];
        let next = sample_from_probs(row, top_k, &mut rng);
        out.push(vocab.decode(next));
        context.remove(0);
        context.push(next);
    }
    Ok(out)
}

fn main() -> PureResult<()> {
    let args = Args::parse()?;
    let backend_sel = backend::parse_backend(Some(args.backend.as_str()))?;
    let data_files = text_corpus::collect_text_files(&args.data_paths)?;
    if data_files.is_empty() {
        return Err(TensorError::EmptyInput("char_lm_text_files"));
    }
    let text = text_corpus::read_text_files_lossy(&data_files)?;
    if text.is_empty() {
        return Err(TensorError::EmptyInput("char_lm_text"));
    }

    let run_dir = args.run_dir.clone().unwrap_or_else(default_run_dir);
    std::fs::create_dir_all(&run_dir).map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    let samples_dir = run_dir.join("samples");
    std::fs::create_dir_all(&samples_dir).map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    std::fs::write(
        run_dir.join("command.txt"),
        env::args().collect::<Vec<_>>().join(" "),
    )
    .map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    let data_manifest_path = run_dir.join("data_files.txt");
    text_corpus::write_data_files_manifest(&data_manifest_path, &data_files)?;

    let _events_writer = if let Some(events_path) = args.events.as_ref() {
        if let Some(parent) = events_path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| TensorError::IoError {
                message: err.to_string(),
            })?;
        }
        Some(PluginEventJsonlWriter::subscribe(
            global_registry().event_bus().clone(),
            events_path,
            PluginEventJsonlWriterConfig::default(),
        )?)
    } else {
        None
    };
    global_registry()
        .event_bus()
        .publish(&PluginEvent::BackendChanged {
            backend: backend_sel.label.clone(),
        });

    let (steps, embed_dim, hidden, memory, curvature, temperature, vocab, mut model, loaded_from) =
        if let Some(ref weights_path) = args.load_weights {
            let meta_path = meta_path_for_weights(weights_path);
            let meta = read_meta(&meta_path)?;
            if meta.format != FORMAT_ID {
                return Err(TensorError::Generic(format!(
                    "unexpected meta format: {} (expected {FORMAT_ID})",
                    meta.format
                )));
            }
            let vocab = Vocab::from_symbols(meta.unk, meta.symbols);
            let model = build_model(
                vocab.len(),
                meta.steps,
                meta.embed_dim,
                meta.hidden,
                meta.memory,
                meta.curvature,
                meta.temperature,
            )?;
            (
                meta.steps,
                meta.embed_dim,
                meta.hidden,
                meta.memory,
                meta.curvature,
                meta.temperature,
                vocab,
                model,
                Some(weights_path.clone()),
            )
        } else {
            let vocab = Vocab::build_from_text(&text, DEFAULT_UNK);
            let model = build_model(
                vocab.len(),
                args.steps,
                args.embed_dim,
                args.hidden,
                args.memory,
                args.curvature,
                args.temperature,
            )?;
            (
                args.steps,
                args.embed_dim,
                args.hidden,
                args.memory,
                args.curvature,
                args.temperature,
                vocab,
                model,
                None,
            )
        };

    if let Some(weights_path) = loaded_from.as_ref() {
        load_json(&mut model, weights_path)?;
    }
    model.attach_hypergrad(curvature, args.learning_rate)?;

    let tokens = encode_text(&text, &vocab);
    if tokens.len() <= steps {
        return Err(TensorError::Generic(format!(
            "text too short for steps={steps}: len={}",
            tokens.len()
        )));
    }

    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| text.chars().take(steps).collect::<String>());
    let run_meta = RunMeta {
        schema: RUN_SCHEMA.to_string(),
        arch: "llm_char_coherence_scan".to_string(),
        backend: backend_sel.label.clone(),
        device_caps: backend_sel.caps.into(),
        format: FORMAT_ID.to_string(),
        data_paths: args
            .data_paths
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect(),
        data_file_count: data_files.len(),
        data_files_manifest: data_manifest_path.to_string_lossy().to_string(),
        events_path: args
            .events
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        weights_loaded_from: loaded_from
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        steps,
        embed_dim,
        hidden,
        memory,
        epochs: args.epochs,
        batches_per_epoch: args.batches_per_epoch,
        batch: args.batch,
        learning_rate: args.learning_rate,
        curvature,
        temperature,
        gen_len: args.gen_len,
        top_k: args.top_k,
        seed: args.seed,
        prompt: prompt.clone(),
        vocab_size: vocab.len(),
    };
    let run_writer =
        File::create(run_dir.join("run.json")).map_err(|err| TensorError::IoError {
            message: err.to_string(),
        })?;
    serde_json::to_writer_pretty(run_writer, &run_meta).map_err(|err| {
        TensorError::SerializationError {
            message: err.to_string(),
        }
    })?;
    let mut metrics_file =
        File::create(run_dir.join("metrics.jsonl")).map_err(|err| TensorError::IoError {
            message: err.to_string(),
        })?;

    let mut trainer = ModuleTrainer::new(
        backend_sel.caps,
        curvature,
        args.learning_rate,
        args.learning_rate,
    );
    let schedule = trainer.roundtable(
        args.batch as u32,
        vocab.len() as u32,
        RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5),
    );

    let mut loss = CategoricalCrossEntropy::new();

    println!(
        "arch=coherence_scan backend={} vocab={} files={} chars={} steps={} embed_dim={} hidden={} memory={} epochs={} batch={} lr={:.3e} curvature={} temp={} run_dir={}",
        backend_sel.label,
        vocab.len(),
        data_files.len(),
        text.chars().count(),
        steps,
        embed_dim,
        hidden,
        memory,
        args.epochs,
        args.batch,
        args.learning_rate,
        curvature,
        temperature,
        run_dir.display()
    );

    let mut last_sample: Option<String> = None;
    for epoch in 0..args.epochs {
        let mut rng = ChaCha8Rng::seed_from_u64(args.seed.wrapping_add(epoch as u64 * 10_000));
        let mut batches = Vec::with_capacity(args.batches_per_epoch);
        for _ in 0..args.batches_per_epoch {
            batches.push(build_random_batch(
                &tokens,
                vocab.len(),
                steps,
                args.batch,
                &mut rng,
            )?);
        }
        let stats = trainer.train_epoch(&mut model, &mut loss, batches, &schedule)?;
        let metric = EpochMetric {
            epoch,
            batches: stats.batches,
            average_loss: stats.average_loss,
        };
        writeln!(
            metrics_file,
            "{}",
            serde_json::to_string(&metric).map_err(|err| TensorError::SerializationError {
                message: err.to_string(),
            })?
        )
        .map_err(|err| TensorError::IoError {
            message: err.to_string(),
        })?;
        let sample = generate_text(
            &model,
            &vocab,
            steps,
            &prompt,
            args.gen_len,
            args.top_k,
            args.seed.wrapping_add(999).wrapping_add(epoch as u64),
        )?;
        std::fs::write(samples_dir.join(format!("epoch_{epoch:03}.txt")), &sample).map_err(
            |err| TensorError::IoError {
                message: err.to_string(),
            },
        )?;
        last_sample = Some(sample);
        println!(
            "epoch[{epoch}] batches={} avg_loss={:.6}",
            stats.batches, stats.average_loss
        );
    }

    let meta = CharLmMeta::new(
        steps,
        embed_dim,
        hidden,
        memory,
        curvature,
        temperature,
        &vocab,
    );
    let run_weights = run_dir.join("weights.json");
    save_json(&model, &run_weights)?;
    write_meta(&meta_path_for_weights(&run_weights), &meta)?;
    if let Some(save_weights) = args.save_weights.as_ref() {
        if let Some(parent) = save_weights.parent() {
            std::fs::create_dir_all(parent).map_err(|err| TensorError::IoError {
                message: err.to_string(),
            })?;
        }
        save_json(&model, save_weights)?;
        write_meta(&meta_path_for_weights(save_weights), &meta)?;
    }

    let sample = match last_sample {
        Some(sample) => sample,
        None => {
            let sample = generate_text(
                &model,
                &vocab,
                steps,
                &prompt,
                args.gen_len,
                args.top_k,
                args.seed.wrapping_add(999),
            )?;
            std::fs::write(samples_dir.join("init.txt"), &sample).map_err(|err| {
                TensorError::IoError {
                    message: err.to_string(),
                }
            })?;
            sample
        }
    };

    println!("--- sample (prompt + gen) ---");
    println!("{sample}");

    Ok(())
}
