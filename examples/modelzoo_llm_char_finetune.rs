// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! LLM model-zoo: character-level language model fine-tuning from raw text
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
use st_nn::layers::spiral_rnn::SpiralRnn;
use st_nn::layers::ZSpaceSoftmax;
use st_nn::{
    load_json, save_json, CategoricalCrossEntropy, Embedding, Module, ModuleTrainer, PureResult,
    RoundtableConfig, Sequential, Tensor, TensorError,
};
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const FORMAT_ID_V1: &str = "st-char-lm-v1";
const FORMAT_ID_V2: &str = "st-char-lm-v2";
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
    hidden: usize,
    embed_dim: Option<usize>,
    mode: String,
    epochs: usize,
    batches_per_epoch: usize,
    batch: usize,
    learning_rate: f32,
    curvature: f32,
    temperature: f32,
    gen_len: usize,
    top_k: usize,
    seed: u64,
    validation_fraction_requested: f32,
    validation_fraction_actual: f32,
    eval_samples: usize,
    train_tokens: usize,
    validation_tokens: usize,
    prompt: String,
    vocab_size: usize,
}

#[derive(Debug, Clone, Serialize)]
struct EpochMetric {
    epoch: usize,
    batches: usize,
    average_loss: f32,
    validation: Option<LanguageEvalMetric>,
}

#[derive(Debug, Clone, Serialize)]
struct LanguageEvalMetric {
    tokens: usize,
    windows: usize,
    mean_nll: f32,
    perplexity: Option<f32>,
    accuracy: f32,
    mean_target_probability: f32,
    mean_top_probability: f32,
}

#[derive(Debug, Clone, Serialize)]
struct TrainingSummary {
    initial_validation: Option<LanguageEvalMetric>,
    final_validation: Option<LanguageEvalMetric>,
    best_validation_epoch: Option<usize>,
    best_validation_mean_nll: Option<f32>,
    validation_nll_delta: Option<f32>,
    validation_accuracy_delta: Option<f32>,
}

#[derive(Debug, Clone)]
struct TokenSplit {
    train: Vec<usize>,
    validation: Vec<usize>,
    actual_validation_fraction: f32,
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
        symbols.extend(set);
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
    hidden: usize,
    curvature: f32,
    temperature: f32,
    #[serde(default)]
    embed_dim: Option<usize>,
    unk: char,
    symbols: Vec<char>,
}

impl CharLmMeta {
    fn new(
        steps: usize,
        hidden: usize,
        embed_dim: Option<usize>,
        curvature: f32,
        temperature: f32,
        vocab: &Vocab,
    ) -> Self {
        Self {
            format: if embed_dim.is_some() {
                FORMAT_ID_V2.to_string()
            } else {
                FORMAT_ID_V1.to_string()
            },
            steps,
            hidden,
            curvature,
            temperature,
            embed_dim,
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
    epochs: usize,
    batches_per_epoch: usize,
    batch: usize,
    learning_rate: f32,
    curvature: f32,
    temperature: f32,
    gen_len: usize,
    top_k: usize,
    seed: u64,
    validation_fraction: f32,
    eval_samples: usize,
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
                "usage: cargo run -p st-nn --example modelzoo_llm_char_finetune -- <text_or_dir> [<text_or_dir> ...] [--load weights.json] [--save weights.json] [--run-dir PATH] [--backend auto|wgpu|cuda|hip|cpu] [--events PATH] [--steps N] [--embed-dim N] [--hidden N] [--epochs N] [--batches N] [--batch N] [--lr F] [--curvature F] [--temperature F] [--gen N] [--topk N] [--seed N] [--val-fraction F] [--eval-samples N] [--prompt STR]"
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
            epochs: 6,
            batches_per_epoch: 24,
            batch: 8,
            learning_rate: 2e-2,
            curvature: -1.0,
            temperature: 1.0,
            gen_len: 200,
            top_k: 32,
            seed: 42,
            validation_fraction: 0.1,
            eval_samples: 256,
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
                "--epochs" => args.epochs = take_parse(&mut argv, "--epochs")?,
                "--batches" => args.batches_per_epoch = take_parse(&mut argv, "--batches")?,
                "--batch" => args.batch = take_parse(&mut argv, "--batch")?,
                "--lr" => args.learning_rate = take_parse(&mut argv, "--lr")?,
                "--curvature" => args.curvature = take_parse(&mut argv, "--curvature")?,
                "--temperature" => args.temperature = take_parse(&mut argv, "--temperature")?,
                "--gen" => args.gen_len = take_parse(&mut argv, "--gen")?,
                "--topk" => args.top_k = take_parse(&mut argv, "--topk")?,
                "--seed" => args.seed = take_parse(&mut argv, "--seed")?,
                "--val-fraction" => {
                    args.validation_fraction = take_parse(&mut argv, "--val-fraction")?
                }
                "--eval-samples" => args.eval_samples = take_parse(&mut argv, "--eval-samples")?,
                "--prompt" => args.prompt = Some(take_arg(&mut argv, "--prompt")?),
                "--help" | "-h" => {
                    return Err(TensorError::Generic(
                        "usage: cargo run -p st-nn --example modelzoo_llm_char_finetune -- <text_or_dir> [<text_or_dir> ...] [--load weights.json] [--save weights.json] [--run-dir PATH] [--backend auto|wgpu|cuda|hip|cpu] [--events PATH] [--steps N] [--embed-dim N] [--hidden N] [--epochs N] [--batches N] [--batch N] [--lr F] [--curvature F] [--temperature F] [--gen N] [--topk N] [--seed N] [--val-fraction F] [--eval-samples N] [--prompt STR]"
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

        if args.steps == 0 || args.embed_dim == 0 || args.hidden == 0 || args.batch == 0 {
            return Err(TensorError::InvalidValue {
                label: "char_lm_invalid_dims",
            });
        }
        if !args.validation_fraction.is_finite()
            || args.validation_fraction < 0.0
            || args.validation_fraction >= 1.0
        {
            return Err(TensorError::InvalidValue {
                label: "char_lm_validation_fraction",
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
    embed_dim: Option<usize>,
    steps: usize,
    hidden: usize,
    curvature: f32,
    temperature: f32,
) -> PureResult<Sequential> {
    let mut model = Sequential::new();
    if let Some(embed_dim) = embed_dim {
        model.push(Embedding::new("embed", vocab_size, embed_dim)?);
        model.push(SpiralRnn::new("char_rnn", embed_dim, hidden, steps)?);
    } else {
        model.push(SpiralRnn::new("char_rnn", vocab_size, hidden, steps)?);
    }
    model.push(st_nn::Linear::new("head", hidden, vocab_size)?);
    model.push(ZSpaceSoftmax::new(curvature, temperature)?);
    Ok(model)
}

fn encode_text(text: &str, vocab: &Vocab) -> Vec<usize> {
    text.chars().map(|ch| vocab.encode(ch)).collect()
}

fn split_train_validation_tokens(
    tokens: &[usize],
    steps: usize,
    validation_fraction: f32,
) -> TokenSplit {
    let min_window_tokens = steps + 1;
    if validation_fraction <= 0.0 || tokens.len() < min_window_tokens * 2 {
        return TokenSplit {
            train: tokens.to_vec(),
            validation: Vec::new(),
            actual_validation_fraction: 0.0,
        };
    }

    let requested = (tokens.len() as f32 * validation_fraction).round() as usize;
    let max_validation = tokens.len() - min_window_tokens;
    let validation_len = requested.clamp(min_window_tokens, max_validation);
    let split_at = tokens.len() - validation_len;
    let validation = tokens[split_at..].to_vec();
    let actual_validation_fraction = validation.len() as f32 / tokens.len() as f32;

    TokenSplit {
        train: tokens[..split_at].to_vec(),
        validation,
        actual_validation_fraction,
    }
}

fn encode_context_one_hot(context: &[usize], vocab_size: usize) -> PureResult<Tensor> {
    let steps = context.len();
    let cols = vocab_size * steps;
    let mut x = Tensor::zeros(1, cols)?;
    let data = x.data_mut();
    for (t, &idx) in context.iter().enumerate() {
        if idx < vocab_size {
            data[t * vocab_size + idx] = 1.0;
        }
    }
    Ok(x)
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
    embed_dim: Option<usize>,
    steps: usize,
    batch: usize,
    rng: &mut impl Rng,
) -> PureResult<(Tensor, Tensor)> {
    let max_start = tokens.len().saturating_sub(steps);
    if max_start == 0 {
        return Err(TensorError::EmptyInput("char_lm_tokens"));
    }
    let input_cols = vocab_size * steps;
    let mut x = Tensor::zeros(
        batch,
        if embed_dim.is_some() {
            steps
        } else {
            input_cols
        },
    )?;
    let mut y = Tensor::zeros(batch, vocab_size)?;
    {
        let x_data = x.data_mut();
        let y_data = y.data_mut();
        for row in 0..batch {
            let start = rng.gen_range(0..max_start);
            if embed_dim.is_some() {
                for t in 0..steps {
                    x_data[row * steps + t] = tokens[start + t] as f32;
                }
            } else {
                for t in 0..steps {
                    let idx = tokens[start + t];
                    if idx < vocab_size {
                        x_data[row * input_cols + t * vocab_size + idx] = 1.0;
                    }
                }
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

#[allow(clippy::too_many_arguments)]
fn generate_text(
    model: &Sequential,
    vocab: &Vocab,
    embed_dim: Option<usize>,
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
        let x = if embed_dim.is_some() {
            encode_context_indices(&context)?
        } else {
            encode_context_one_hot(&context, vocab.len())?
        };
        let probs = model.forward(&x)?;
        let row = &probs.data()[..vocab.len()];
        let next = sample_from_probs(row, top_k, &mut rng);
        out.push(vocab.decode(next));
        context.remove(0);
        context.push(next);
    }
    Ok(out)
}

fn evaluate_next_token(
    model: &mut Sequential,
    vocab_size: usize,
    embed_dim: Option<usize>,
    steps: usize,
    tokens: &[usize],
    max_samples: usize,
) -> PureResult<Option<LanguageEvalMetric>> {
    if tokens.len() <= steps {
        return Ok(None);
    }
    let available = tokens.len() - steps;
    let windows = if max_samples == 0 {
        available
    } else {
        available.min(max_samples)
    };
    if windows == 0 {
        return Ok(None);
    }

    model.eval()?;
    let result = (|| -> PureResult<LanguageEvalMetric> {
        let mut nll_sum = 0.0f32;
        let mut correct = 0usize;
        let mut target_probability_sum = 0.0f32;
        let mut top_probability_sum = 0.0f32;

        for sample_idx in 0..windows {
            let start = if windows <= 1 || available <= 1 {
                0
            } else {
                sample_idx * (available - 1) / (windows - 1)
            };
            let context = &tokens[start..start + steps];
            let target = tokens[start + steps];
            let x = if embed_dim.is_some() {
                encode_context_indices(context)?
            } else {
                encode_context_one_hot(context, vocab_size)?
            };
            let prediction = model.forward(&x)?;
            if prediction.shape() != (1, vocab_size) {
                return Err(TensorError::ShapeMismatch {
                    left: prediction.shape(),
                    right: (1, vocab_size),
                });
            }
            let row = &prediction.data()[..vocab_size];
            let top = argmax(row);
            let top_probability = row[top].clamp(0.0, 1.0);
            let target_probability = row.get(target).copied().unwrap_or(0.0).clamp(1.0e-9, 1.0);

            if top == target {
                correct += 1;
            }
            nll_sum += -target_probability.ln();
            target_probability_sum += target_probability;
            top_probability_sum += top_probability;
        }

        let mean_nll = nll_sum / windows as f32;
        let perplexity = if mean_nll.is_finite() && mean_nll < 80.0 {
            Some(mean_nll.exp())
        } else {
            None
        };
        Ok(LanguageEvalMetric {
            tokens: tokens.len(),
            windows,
            mean_nll,
            perplexity,
            accuracy: correct as f32 / windows as f32,
            mean_target_probability: target_probability_sum / windows as f32,
            mean_top_probability: top_probability_sum / windows as f32,
        })
    })();
    let restore = model.train();
    match (result, restore) {
        (Ok(metric), Ok(())) => Ok(Some(metric)),
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err),
    }
}

fn write_summary(path: &Path, summary: &TrainingSummary) -> PureResult<()> {
    let writer = File::create(path).map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    serde_json::to_writer_pretty(writer, summary).map_err(|err| {
        TensorError::SerializationError {
            message: err.to_string(),
        }
    })?;
    Ok(())
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

    let (steps, hidden, curvature, temperature, vocab, model, loaded_from) =
        if let Some(ref weights_path) = args.load_weights {
            let meta_path = meta_path_for_weights(weights_path);
            let meta = read_meta(&meta_path)?;
            if meta.format != FORMAT_ID_V1 && meta.format != FORMAT_ID_V2 {
                return Err(TensorError::Generic(format!(
                    "unexpected meta format: {} (expected {FORMAT_ID_V1} or {FORMAT_ID_V2})",
                    meta.format
                )));
            }
            if meta.format == FORMAT_ID_V2 && meta.embed_dim.is_none() {
                return Err(TensorError::Generic(
                    "meta format st-char-lm-v2 requires embed_dim".to_string(),
                ));
            }
            let vocab = Vocab::from_symbols(meta.unk, meta.symbols);
            let model = build_model(
                vocab.len(),
                meta.embed_dim,
                meta.steps,
                meta.hidden,
                meta.curvature,
                meta.temperature,
            )?;
            (
                meta.steps,
                meta.hidden,
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
                Some(args.embed_dim),
                args.steps,
                args.hidden,
                args.curvature,
                args.temperature,
            )?;
            (
                args.steps,
                args.hidden,
                args.curvature,
                args.temperature,
                vocab,
                model,
                None,
            )
        };

    let embed_dim = if let Some(ref weights_path) = loaded_from {
        let meta_path = meta_path_for_weights(weights_path);
        let meta = read_meta(&meta_path)?;
        meta.embed_dim
    } else {
        Some(args.embed_dim)
    };

    let mut model = model;
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
    let split = split_train_validation_tokens(&tokens, steps, args.validation_fraction);
    if split.train.len() <= steps {
        return Err(TensorError::Generic(format!(
            "training split too short for steps={steps}: len={}",
            split.train.len()
        )));
    }

    let mode = embed_dim
        .map(|dim| format!("embedding({dim})"))
        .unwrap_or_else(|| "one_hot".to_string());
    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| text.chars().take(steps).collect::<String>());
    let format = if embed_dim.is_some() {
        FORMAT_ID_V2
    } else {
        FORMAT_ID_V1
    };
    let run_meta = RunMeta {
        schema: RUN_SCHEMA.to_string(),
        arch: "llm_char_finetune".to_string(),
        backend: backend_sel.label.clone(),
        device_caps: backend_sel.caps.into(),
        format: format.to_string(),
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
        hidden,
        embed_dim,
        mode: mode.clone(),
        epochs: args.epochs,
        batches_per_epoch: args.batches_per_epoch,
        batch: args.batch,
        learning_rate: args.learning_rate,
        curvature,
        temperature,
        gen_len: args.gen_len,
        top_k: args.top_k,
        seed: args.seed,
        validation_fraction_requested: args.validation_fraction,
        validation_fraction_actual: split.actual_validation_fraction,
        eval_samples: args.eval_samples,
        train_tokens: split.train.len(),
        validation_tokens: split.validation.len(),
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
        "mode={mode} backend={} vocab={} files={} chars={} train_tokens={} validation_tokens={} steps={} hidden={} epochs={} batch={} lr={:.3e} curvature={} temp={} run_dir={}",
        backend_sel.label,
        vocab.len(),
        data_files.len(),
        text.chars().count(),
        split.train.len(),
        split.validation.len(),
        steps,
        hidden,
        args.epochs,
        args.batch,
        args.learning_rate,
        curvature,
        temperature,
        run_dir.display()
    );

    let initial_validation = evaluate_next_token(
        &mut model,
        vocab.len(),
        embed_dim,
        steps,
        &split.validation,
        args.eval_samples,
    )?;
    if let Some(validation) = initial_validation.as_ref() {
        println!(
            "validation[init] windows={} nll={:.6} ppl={} acc={:.2}%",
            validation.windows,
            validation.mean_nll,
            validation
                .perplexity
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "overflow".to_string()),
            validation.accuracy * 100.0
        );
    } else {
        println!(
            "validation[init] skipped: validation split is empty (increase data or --val-fraction)"
        );
    }

    let mut last_sample: Option<String> = None;
    let mut last_validation = initial_validation.clone();
    let mut best_validation_epoch = None;
    let mut best_validation_mean_nll = None;
    for epoch in 0..args.epochs {
        let mut rng = ChaCha8Rng::seed_from_u64(args.seed.wrapping_add(epoch as u64 * 10_000));
        let mut batches = Vec::with_capacity(args.batches_per_epoch);
        for _ in 0..args.batches_per_epoch {
            batches.push(build_random_batch(
                &split.train,
                vocab.len(),
                embed_dim,
                steps,
                args.batch,
                &mut rng,
            )?);
        }
        let stats = trainer.train_epoch(&mut model, &mut loss, batches, &schedule)?;
        let validation = evaluate_next_token(
            &mut model,
            vocab.len(),
            embed_dim,
            steps,
            &split.validation,
            args.eval_samples,
        )?;
        if let Some(validation_metric) = validation.as_ref() {
            if best_validation_mean_nll
                .map(|best| validation_metric.mean_nll < best)
                .unwrap_or(true)
            {
                best_validation_mean_nll = Some(validation_metric.mean_nll);
                best_validation_epoch = Some(epoch);
            }
        }
        let metric = EpochMetric {
            epoch,
            batches: stats.batches,
            average_loss: stats.average_loss,
            validation: validation.clone(),
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
            embed_dim,
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
        last_validation = validation.clone();
        if let Some(validation) = validation.as_ref() {
            println!(
                "epoch[{epoch}] batches={} avg_loss={:.6} val_nll={:.6} val_ppl={} val_acc={:.2}%",
                stats.batches,
                stats.average_loss,
                validation.mean_nll,
                validation
                    .perplexity
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "overflow".to_string()),
                validation.accuracy * 100.0
            );
        } else {
            println!(
                "epoch[{epoch}] batches={} avg_loss={:.6} val=skipped",
                stats.batches, stats.average_loss
            );
        }
    }

    let meta = CharLmMeta::new(steps, hidden, embed_dim, curvature, temperature, &vocab);
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
                embed_dim,
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

    let final_validation = last_validation;
    let validation_nll_delta = initial_validation
        .as_ref()
        .zip(final_validation.as_ref())
        .map(|(initial, final_metric)| final_metric.mean_nll - initial.mean_nll);
    let validation_accuracy_delta = initial_validation
        .as_ref()
        .zip(final_validation.as_ref())
        .map(|(initial, final_metric)| final_metric.accuracy - initial.accuracy);
    let summary = TrainingSummary {
        initial_validation,
        final_validation,
        best_validation_epoch,
        best_validation_mean_nll,
        validation_nll_delta,
        validation_accuracy_delta,
    };
    write_summary(&run_dir.join("summary.json"), &summary)?;

    println!("--- sample (prompt + gen) ---");
    println!("{sample}");

    Ok(())
}
