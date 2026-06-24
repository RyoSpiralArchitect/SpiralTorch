// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! LLM model-zoo: character-level language model fine-tuning from raw text
//! (no tokenizer / no BPE).

#[path = "_shared/backend.rs"]
mod backend;
#[path = "_shared/char_lm_eval.rs"]
mod char_lm_eval;
#[path = "_shared/text_corpus.rs"]
mod text_corpus;

use char_lm_eval::{
    capture_parameter_snapshot, evaluate_bigram_next_token, evaluate_next_token_with_unigram_lift,
    evaluate_unigram_next_token, head_prior_is_enabled, insert_head_prior_with_context,
    learning_rate_schedule_label, linear_with_weight_rms, push_char_embedding,
    residual_logit_scaler, scheduled_learning_rate, split_train_validation_tokens,
    summarize_learnability, validate_bigram_rank_guard, validate_bigram_rank_guard_band,
    validate_bigram_rank_guard_min_candidates, validate_bigram_soft_guard,
    validate_bigram_topk_guard, validate_char_feature, validate_head_prior,
    validate_head_residual_scale, validate_learning_rate_schedule, write_summary,
    BigramRankGuardCoverage, BigramTopKGuardTargets, BigramTopKGuardedCrossEntropy,
    CharLmInputMode, LanguageEvalMetric, LearnabilityMetric, TrainingSummary, CHAR_FEATURE_TOKEN,
    DEFAULT_BIGRAM_RANK_GUARD, DEFAULT_BIGRAM_RANK_GUARD_BAND, DEFAULT_BIGRAM_RANK_GUARD_MARGIN,
    DEFAULT_BIGRAM_RANK_GUARD_MIN_CANDIDATES, DEFAULT_BIGRAM_SOFT_GUARD, DEFAULT_BIGRAM_TOPK_GUARD,
    DEFAULT_BIGRAM_TOPK_GUARD_K, DEFAULT_CHAR_FEATURE, DEFAULT_HEAD_RESIDUAL_SCALE,
    HEAD_PRIOR_LEARNED_UNIGRAM,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use st_core::plugin::{
    global_registry, PluginEvent, PluginEventJsonlWriter, PluginEventJsonlWriterConfig,
};
use st_nn::layers::spiral_rnn::SpiralRnn;
use st_nn::layers::ZSpaceSoftmax;
use st_nn::{
    load_json, save_json, EpochTensorBackendStats, Lstm, Module, ModuleTrainer, Parameter,
    PureResult, RoundtableConfig, Sequential, Tensor, TensorError,
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
const DEFAULT_LINEAR_WEIGHT_RMS: f32 = 0.1;
const RECURRENT_SPIRAL: &str = "spiral";
const RECURRENT_LSTM: &str = "lstm";
const USAGE: &str = "usage: cargo run -p st-nn --example modelzoo_llm_char_finetune -- <text_or_dir> [<text_or_dir> ...] [--load weights.json] [--save weights.json] [--run-dir PATH] [--backend auto|wgpu|cuda|hip|cpu] [--events PATH] [--recurrent spiral|lstm] [--steps N] [--embed-dim N] [--char-feature token|token-bigram] [--hidden N] [--head-rms F] [--head-residual-scale F] [--head-prior none|unigram|learned-unigram|bigram|learned-bigram] [--bigram-topk-guard F] [--bigram-topk-guard-k N] [--bigram-rank-guard F] [--bigram-rank-guard-margin F] [--bigram-rank-guard-band F] [--bigram-rank-guard-min-candidates N] [--bigram-soft-guard F] [--epochs N] [--batches N] [--batch N] [--lr F] [--lr-warmup-epochs N] [--lr-final-scale F] [--curvature F] [--temperature F] [--gen N] [--topk N] [--seed N] [--val-fraction F] [--val-start-fraction F] [--eval-samples N] [--early-stop-patience N] [--restore-best-at-end] [--prompt STR]";

fn legacy_char_feature() -> String {
    CHAR_FEATURE_TOKEN.to_string()
}

fn legacy_recurrent() -> String {
    RECURRENT_SPIRAL.to_string()
}

#[derive(Debug, Clone, Serialize)]
struct RunMeta {
    schema: String,
    arch: String,
    backend: String,
    device_caps: backend::DeviceCapsMeta,
    backend_runtime: backend::BackendRuntimeMeta,
    tensor_policy: backend::TensorBackendPolicyMeta,
    roundtable_backend_audit: backend::RoundtableBackendAudit,
    format: String,
    data_paths: Vec<String>,
    data_file_count: usize,
    data_files_manifest: String,
    events_path: Option<String>,
    weights_loaded_from: Option<String>,
    steps: usize,
    hidden: usize,
    embed_dim: Option<usize>,
    recurrent: String,
    mode: String,
    char_feature: String,
    head_weight_rms: f32,
    head_residual_scale: Option<f32>,
    head_prior: String,
    bigram_topk_guard: f32,
    bigram_topk_guard_k: usize,
    bigram_rank_guard: f32,
    bigram_rank_guard_margin: f32,
    bigram_rank_guard_band: f32,
    bigram_rank_guard_min_candidates: usize,
    bigram_rank_guard_coverage: Option<BigramRankGuardCoverage>,
    bigram_soft_guard: f32,
    epochs: usize,
    batches_per_epoch: usize,
    batch: usize,
    learning_rate: f32,
    learning_rate_schedule: String,
    learning_rate_warmup_epochs: usize,
    learning_rate_final_scale: f32,
    curvature: f32,
    temperature: f32,
    gen_len: usize,
    top_k: usize,
    seed: u64,
    validation_fraction_requested: f32,
    validation_fraction_actual: f32,
    validation_start_fraction_requested: Option<f32>,
    validation_start_fraction_actual: Option<f32>,
    validation_start_token: Option<usize>,
    eval_samples: usize,
    early_stop_patience: usize,
    restore_best_at_end: bool,
    train_tokens: usize,
    validation_tokens: usize,
    prompt: String,
    vocab_size: usize,
}

#[derive(Debug, Clone, Serialize)]
struct EpochMetric {
    epoch: usize,
    batches: usize,
    learning_rate: f32,
    learning_rate_scale: f32,
    average_loss: f32,
    tensor_backend: EpochTensorBackendStats,
    validation: Option<LanguageEvalMetric>,
    learnability: LearnabilityMetric,
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
    #[serde(default = "legacy_char_feature")]
    char_feature: String,
    #[serde(default = "legacy_recurrent")]
    recurrent: String,
    #[serde(default)]
    head_prior: Option<String>,
    #[serde(default)]
    head_residual_scale: Option<f32>,
    unk: char,
    symbols: Vec<char>,
}

impl CharLmMeta {
    fn new(
        steps: usize,
        hidden: usize,
        embed_dim: Option<usize>,
        recurrent: String,
        char_feature: Option<String>,
        head_prior: Option<String>,
        head_residual_scale: Option<f32>,
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
            recurrent,
            char_feature: char_feature.unwrap_or_else(legacy_char_feature),
            head_prior,
            head_residual_scale,
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
    recurrent: String,
    steps: usize,
    embed_dim: usize,
    char_feature: String,
    hidden: usize,
    head_weight_rms: f32,
    head_residual_scale: f32,
    head_prior: String,
    bigram_topk_guard: f32,
    bigram_topk_guard_k: usize,
    bigram_rank_guard: f32,
    bigram_rank_guard_margin: f32,
    bigram_rank_guard_band: f32,
    bigram_rank_guard_min_candidates: usize,
    bigram_soft_guard: f32,
    epochs: usize,
    batches_per_epoch: usize,
    batch: usize,
    learning_rate: f32,
    learning_rate_warmup_epochs: usize,
    learning_rate_final_scale: f32,
    curvature: f32,
    temperature: f32,
    gen_len: usize,
    top_k: usize,
    seed: u64,
    validation_fraction: f32,
    validation_start_fraction: Option<f32>,
    eval_samples: usize,
    early_stop_patience: usize,
    restore_best_at_end: bool,
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
            return Err(TensorError::Generic(USAGE.to_string()));
        }

        let mut args = Self {
            data_paths: data_args.into_iter().map(PathBuf::from).collect(),
            load_weights: None,
            save_weights: None,
            run_dir: None,
            backend: "auto".to_string(),
            events: None,
            recurrent: RECURRENT_SPIRAL.to_string(),
            steps: 32,
            embed_dim: 32,
            char_feature: DEFAULT_CHAR_FEATURE.to_string(),
            hidden: 64,
            head_weight_rms: DEFAULT_LINEAR_WEIGHT_RMS,
            head_residual_scale: DEFAULT_HEAD_RESIDUAL_SCALE,
            head_prior: HEAD_PRIOR_LEARNED_UNIGRAM.to_string(),
            bigram_topk_guard: DEFAULT_BIGRAM_TOPK_GUARD,
            bigram_topk_guard_k: DEFAULT_BIGRAM_TOPK_GUARD_K,
            bigram_rank_guard: DEFAULT_BIGRAM_RANK_GUARD,
            bigram_rank_guard_margin: DEFAULT_BIGRAM_RANK_GUARD_MARGIN,
            bigram_rank_guard_band: DEFAULT_BIGRAM_RANK_GUARD_BAND,
            bigram_rank_guard_min_candidates: DEFAULT_BIGRAM_RANK_GUARD_MIN_CANDIDATES,
            bigram_soft_guard: DEFAULT_BIGRAM_SOFT_GUARD,
            epochs: 6,
            batches_per_epoch: 24,
            batch: 8,
            learning_rate: 2e-2,
            learning_rate_warmup_epochs: 0,
            learning_rate_final_scale: 1.0,
            curvature: -1.0,
            temperature: 1.0,
            gen_len: 200,
            top_k: 32,
            seed: 42,
            validation_fraction: 0.1,
            validation_start_fraction: None,
            eval_samples: 256,
            early_stop_patience: 0,
            restore_best_at_end: false,
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
                "--recurrent" => args.recurrent = take_arg(&mut argv, "--recurrent")?,
                "--steps" => args.steps = take_parse(&mut argv, "--steps")?,
                "--embed-dim" => args.embed_dim = take_parse(&mut argv, "--embed-dim")?,
                "--char-feature" => args.char_feature = take_arg(&mut argv, "--char-feature")?,
                "--hidden" => args.hidden = take_parse(&mut argv, "--hidden")?,
                "--head-rms" => args.head_weight_rms = take_parse(&mut argv, "--head-rms")?,
                "--head-residual-scale" => {
                    args.head_residual_scale = take_parse(&mut argv, "--head-residual-scale")?
                }
                "--head-prior" => args.head_prior = take_arg(&mut argv, "--head-prior")?,
                "--bigram-topk-guard" => {
                    args.bigram_topk_guard = take_parse(&mut argv, "--bigram-topk-guard")?
                }
                "--bigram-topk-guard-k" => {
                    args.bigram_topk_guard_k = take_parse(&mut argv, "--bigram-topk-guard-k")?
                }
                "--bigram-rank-guard" => {
                    args.bigram_rank_guard = take_parse(&mut argv, "--bigram-rank-guard")?
                }
                "--bigram-rank-guard-margin" => {
                    args.bigram_rank_guard_margin =
                        take_parse(&mut argv, "--bigram-rank-guard-margin")?
                }
                "--bigram-rank-guard-band" => {
                    args.bigram_rank_guard_band = take_parse(&mut argv, "--bigram-rank-guard-band")?
                }
                "--bigram-rank-guard-min-candidates" => {
                    args.bigram_rank_guard_min_candidates =
                        take_parse(&mut argv, "--bigram-rank-guard-min-candidates")?
                }
                "--bigram-soft-guard" => {
                    args.bigram_soft_guard = take_parse(&mut argv, "--bigram-soft-guard")?
                }
                "--epochs" => args.epochs = take_parse(&mut argv, "--epochs")?,
                "--batches" => args.batches_per_epoch = take_parse(&mut argv, "--batches")?,
                "--batch" => args.batch = take_parse(&mut argv, "--batch")?,
                "--lr" => args.learning_rate = take_parse(&mut argv, "--lr")?,
                "--lr-warmup-epochs" => {
                    args.learning_rate_warmup_epochs = take_parse(&mut argv, "--lr-warmup-epochs")?
                }
                "--lr-final-scale" => {
                    args.learning_rate_final_scale = take_parse(&mut argv, "--lr-final-scale")?
                }
                "--curvature" => args.curvature = take_parse(&mut argv, "--curvature")?,
                "--temperature" => args.temperature = take_parse(&mut argv, "--temperature")?,
                "--gen" => args.gen_len = take_parse(&mut argv, "--gen")?,
                "--topk" => args.top_k = take_parse(&mut argv, "--topk")?,
                "--seed" => args.seed = take_parse(&mut argv, "--seed")?,
                "--val-fraction" => {
                    args.validation_fraction = take_parse(&mut argv, "--val-fraction")?
                }
                "--val-start-fraction" => {
                    args.validation_start_fraction =
                        Some(take_parse(&mut argv, "--val-start-fraction")?)
                }
                "--eval-samples" => args.eval_samples = take_parse(&mut argv, "--eval-samples")?,
                "--early-stop-patience" => {
                    args.early_stop_patience = take_parse(&mut argv, "--early-stop-patience")?
                }
                "--restore-best-at-end" => args.restore_best_at_end = true,
                "--prompt" => args.prompt = Some(take_arg(&mut argv, "--prompt")?),
                "--help" | "-h" => {
                    return Err(TensorError::Generic(USAGE.to_string()));
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
        if args.head_weight_rms <= 0.0 || !args.head_weight_rms.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "char_lm_head_weight_rms",
                value: args.head_weight_rms,
            });
        }
        validate_char_feature(&args.char_feature)?;
        validate_recurrent(&args.recurrent)?;
        validate_head_prior(&args.head_prior)?;
        validate_head_residual_scale(args.head_residual_scale)?;
        validate_bigram_topk_guard(args.bigram_topk_guard, args.bigram_topk_guard_k)?;
        validate_bigram_rank_guard(
            args.bigram_rank_guard,
            args.bigram_rank_guard_margin,
            args.bigram_topk_guard_k,
        )?;
        validate_bigram_rank_guard_band(args.bigram_rank_guard_band)?;
        validate_bigram_rank_guard_min_candidates(
            args.bigram_rank_guard_min_candidates,
            args.bigram_topk_guard_k,
        )?;
        validate_bigram_soft_guard(args.bigram_soft_guard)?;
        validate_learning_rate_schedule(
            args.learning_rate,
            args.epochs,
            args.learning_rate_warmup_epochs,
            args.learning_rate_final_scale,
            "char_lm_finetune_learning_rate_schedule",
        )?;
        if !args.validation_fraction.is_finite()
            || args.validation_fraction < 0.0
            || args.validation_fraction >= 1.0
        {
            return Err(TensorError::InvalidValue {
                label: "char_lm_validation_fraction",
            });
        }
        if let Some(fraction) = args.validation_start_fraction {
            if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
                return Err(TensorError::InvalidValue {
                    label: "char_lm_validation_start_fraction",
                });
            }
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

fn validate_recurrent(value: &str) -> PureResult<()> {
    match value {
        RECURRENT_SPIRAL | RECURRENT_LSTM => Ok(()),
        _ => Err(TensorError::Generic(format!(
            "invalid --recurrent {value}; expected {RECURRENT_SPIRAL} or {RECURRENT_LSTM}"
        ))),
    }
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

#[derive(Debug)]
struct WindowedStatelessLstm {
    inner: Lstm,
    steps: usize,
    input_dim: usize,
    hidden: usize,
}

impl WindowedStatelessLstm {
    fn new(
        name: impl Into<String>,
        steps: usize,
        input_dim: usize,
        hidden: usize,
    ) -> PureResult<Self> {
        if steps == 0 || input_dim == 0 || hidden == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: steps,
                cols: input_dim.max(hidden),
            });
        }
        Ok(Self {
            inner: Lstm::new(name, input_dim, hidden)?,
            steps,
            input_dim,
            hidden,
        })
    }

    fn sequence_from_row(&self, input: &Tensor, row: usize) -> PureResult<Tensor> {
        let cols = self.steps * self.input_dim;
        let start = row * cols;
        let end = start + cols;
        Tensor::from_vec(
            self.steps,
            self.input_dim,
            input.data()[start..end].to_vec(),
        )
    }

    fn validate_input(&self, input: &Tensor) -> PureResult<()> {
        let (rows, cols) = input.shape();
        let expected_cols = self.steps * self.input_dim;
        if rows == 0 {
            return Err(TensorError::EmptyInput("char_lm_lstm_window"));
        }
        if cols != expected_cols {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (rows, expected_cols),
            });
        }
        Ok(())
    }
}

impl Module for WindowedStatelessLstm {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.validate_input(input)?;
        let (batch, _) = input.shape();
        let mut output = Vec::with_capacity(batch * self.hidden);
        for row in 0..batch {
            let sequence = self.sequence_from_row(input, row)?;
            self.inner.reset_state()?;
            let sequence_output = self.inner.forward(&sequence)?;
            let last_start = (self.steps - 1) * self.hidden;
            output.extend_from_slice(&sequence_output.data()[last_start..last_start + self.hidden]);
        }
        Tensor::from_vec(batch, self.hidden, output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.validate_input(input)?;
        let (batch, cols) = input.shape();
        if grad_output.shape() != (batch, self.hidden) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (batch, self.hidden),
            });
        }
        let mut grad_input = vec![0.0f32; batch * cols];
        for row in 0..batch {
            let sequence = self.sequence_from_row(input, row)?;
            self.inner.reset_state()?;
            let _ = self.inner.forward(&sequence)?;
            let mut grad_sequence = vec![0.0f32; self.steps * self.hidden];
            let grad_row = &grad_output.data()[row * self.hidden..(row + 1) * self.hidden];
            let last_start = (self.steps - 1) * self.hidden;
            grad_sequence[last_start..last_start + self.hidden].copy_from_slice(grad_row);
            let grad_sequence = Tensor::from_vec(self.steps, self.hidden, grad_sequence)?;
            let sequence_grad = self.inner.backward(&sequence, &grad_sequence)?;
            let output_start = row * cols;
            grad_input[output_start..output_start + cols].copy_from_slice(sequence_grad.data());
        }
        Tensor::from_vec(batch, cols, grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.inner.visit_parameters(visitor)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.inner.visit_parameters_mut(visitor)
    }
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
    recurrent: &str,
    char_feature: Option<&str>,
    steps: usize,
    hidden: usize,
    head_weight_rms: f32,
    head_residual_scale: Option<f32>,
    curvature: f32,
    temperature: f32,
) -> PureResult<Sequential> {
    validate_recurrent(recurrent)?;
    let mut model = Sequential::new();
    let recurrent_input_dim = if let Some(embed_dim) = embed_dim {
        push_char_embedding(
            &mut model,
            "embed",
            vocab_size,
            embed_dim,
            char_feature.unwrap_or(CHAR_FEATURE_TOKEN),
        )?;
        embed_dim
    } else {
        vocab_size
    };
    match recurrent {
        RECURRENT_SPIRAL => {
            model.push(SpiralRnn::new(
                "char_rnn",
                recurrent_input_dim,
                hidden,
                steps,
            )?);
        }
        RECURRENT_LSTM => {
            model.push(WindowedStatelessLstm::new(
                "char_lstm",
                steps,
                recurrent_input_dim,
                hidden,
            )?);
        }
        _ => unreachable!("validate_recurrent already checked recurrent kind"),
    }
    model.push(linear_with_weight_rms(
        "head",
        hidden,
        vocab_size,
        head_weight_rms,
    )?);
    if let Some(scale) = head_residual_scale {
        model.push(residual_logit_scaler(
            "head_residual_scale",
            vocab_size,
            scale,
        )?);
    }
    model.push(ZSpaceSoftmax::new(curvature, temperature)?);
    Ok(model)
}

fn encode_text(text: &str, vocab: &Vocab) -> Vec<usize> {
    text.chars().map(|ch| vocab.encode(ch)).collect()
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
    bigram_guard: Option<&BigramTopKGuardTargets>,
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
    let target_cols = bigram_guard
        .map(BigramTopKGuardTargets::target_cols)
        .unwrap_or(vocab_size);
    let mut y = Tensor::zeros(batch, target_cols)?;
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
            if let Some(guard) = bigram_guard {
                let previous = tokens[start + steps - 1];
                guard.write_target_row(y_data, row, previous, target);
            } else if target < vocab_size {
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

fn main() -> PureResult<()> {
    let args = Args::parse()?;
    let backend_sel = backend::parse_backend(Some(args.backend.as_str()))?;
    let backend_runtime = backend::prepare_backend_runtime(&backend_sel)?;
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

    let (
        steps,
        hidden,
        curvature,
        temperature,
        vocab,
        model,
        recurrent,
        loaded_from,
        loaded_head_prior,
        loaded_head_residual_scale,
    ) = if let Some(ref weights_path) = args.load_weights {
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
        if meta.embed_dim.is_some() {
            validate_char_feature(&meta.char_feature)?;
        }
        validate_recurrent(&meta.recurrent)?;
        let vocab = Vocab::from_symbols(meta.unk, meta.symbols);
        let mut model = build_model(
            vocab.len(),
            meta.embed_dim,
            meta.recurrent.as_str(),
            Some(meta.char_feature.as_str()),
            meta.steps,
            meta.hidden,
            args.head_weight_rms,
            meta.head_residual_scale,
            meta.curvature,
            meta.temperature,
        )?;
        if let Some(head_prior) = meta.head_prior.as_deref() {
            insert_head_prior_with_context(&mut model, head_prior, vocab.len(), meta.steps, None)?;
        }
        let head_prior = meta.head_prior.clone();
        let head_residual_scale = meta.head_residual_scale;
        (
            meta.steps,
            meta.hidden,
            meta.curvature,
            meta.temperature,
            vocab,
            model,
            meta.recurrent,
            Some(weights_path.clone()),
            head_prior,
            head_residual_scale,
        )
    } else {
        let vocab = Vocab::build_from_text(&text, DEFAULT_UNK);
        let model = build_model(
            vocab.len(),
            Some(args.embed_dim),
            args.recurrent.as_str(),
            Some(args.char_feature.as_str()),
            args.steps,
            args.hidden,
            args.head_weight_rms,
            Some(args.head_residual_scale),
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
            args.recurrent.clone(),
            None,
            None,
            Some(args.head_residual_scale),
        )
    };

    let embed_dim = if let Some(ref weights_path) = loaded_from {
        let meta_path = meta_path_for_weights(weights_path);
        let meta = read_meta(&meta_path)?;
        meta.embed_dim
    } else {
        Some(args.embed_dim)
    };
    let char_feature = if let Some(ref weights_path) = loaded_from {
        let meta_path = meta_path_for_weights(weights_path);
        let meta = read_meta(&meta_path)?;
        meta.embed_dim.map(|_| meta.char_feature)
    } else {
        Some(args.char_feature.clone())
    };
    let char_feature_label = char_feature
        .clone()
        .unwrap_or_else(|| "one_hot".to_string());

    let mut model = model;
    if let Some(weights_path) = loaded_from.as_ref() {
        load_json(&mut model, weights_path)?;
    }

    let tokens = encode_text(&text, &vocab);
    if tokens.len() <= steps {
        return Err(TensorError::Generic(format!(
            "text too short for steps={steps}: len={}",
            tokens.len()
        )));
    }
    let split = split_train_validation_tokens(
        &tokens,
        steps,
        args.validation_fraction,
        args.validation_start_fraction,
    );
    if split.train.len() <= steps {
        return Err(TensorError::Generic(format!(
            "training split too short for steps={steps}: len={}",
            split.train.len()
        )));
    }
    let head_prior = if loaded_from.is_some() {
        loaded_head_prior
            .as_ref()
            .map(|prior| format!("loaded:{prior}"))
            .unwrap_or_else(|| "loaded".to_string())
    } else {
        args.head_prior.clone()
    };
    let head_residual_scale = if loaded_from.is_some() {
        loaded_head_residual_scale
    } else {
        Some(args.head_residual_scale)
    };
    if loaded_from.is_none() {
        insert_head_prior_with_context(
            &mut model,
            &args.head_prior,
            vocab.len(),
            steps,
            Some(&split.train),
        )?;
    }
    let learning_rate_schedule = learning_rate_schedule_label(
        args.learning_rate_warmup_epochs,
        args.learning_rate_final_scale,
    )
    .to_string();
    model.attach_hypergrad(curvature, args.learning_rate)?;
    let bigram_guard = BigramTopKGuardTargets::new(
        vocab.len(),
        &split.train,
        args.bigram_topk_guard,
        args.bigram_topk_guard_k,
        args.bigram_rank_guard,
        args.bigram_rank_guard_margin,
        args.bigram_rank_guard_band,
        args.bigram_rank_guard_min_candidates,
        args.bigram_soft_guard,
    )?;
    let bigram_rank_guard_coverage = bigram_guard
        .as_ref()
        .and_then(|guard| guard.rank_coverage(&split.train));

    let mode = embed_dim
        .map(|dim| format!("embedding({dim},{char_feature_label})"))
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
    let arch = if recurrent == RECURRENT_LSTM {
        "llm_char_lstm"
    } else {
        "llm_char_finetune"
    };

    let mut trainer = ModuleTrainer::new(
        backend_sel.caps,
        curvature,
        args.learning_rate,
        args.learning_rate,
    );
    let roundtable_config = RoundtableConfig::default()
        .with_top_k(1)
        .with_mid_k(1)
        .with_bottom_k(1)
        .with_here_tolerance(1e-5);
    let schedule = trainer.roundtable(args.batch as u32, vocab.len() as u32, roundtable_config);
    let tensor_policy = backend::tensor_backend_policy_meta(backend_sel.caps);
    let roundtable_backend_audit = backend::roundtable_backend_audit(backend_sel.caps, &schedule);

    let run_meta = RunMeta {
        schema: RUN_SCHEMA.to_string(),
        arch: arch.to_string(),
        backend: backend_sel.label.clone(),
        device_caps: backend_sel.caps.into(),
        backend_runtime: backend_runtime.clone(),
        tensor_policy: tensor_policy.clone(),
        roundtable_backend_audit: roundtable_backend_audit.clone(),
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
        recurrent: recurrent.clone(),
        mode: mode.clone(),
        char_feature: char_feature_label.clone(),
        head_weight_rms: args.head_weight_rms,
        head_residual_scale,
        head_prior: head_prior.clone(),
        bigram_topk_guard: args.bigram_topk_guard,
        bigram_topk_guard_k: args.bigram_topk_guard_k,
        bigram_rank_guard: args.bigram_rank_guard,
        bigram_rank_guard_margin: args.bigram_rank_guard_margin,
        bigram_rank_guard_band: args.bigram_rank_guard_band,
        bigram_rank_guard_min_candidates: args.bigram_rank_guard_min_candidates,
        bigram_rank_guard_coverage,
        bigram_soft_guard: args.bigram_soft_guard,
        epochs: args.epochs,
        batches_per_epoch: args.batches_per_epoch,
        batch: args.batch,
        learning_rate: args.learning_rate,
        learning_rate_schedule: learning_rate_schedule.clone(),
        learning_rate_warmup_epochs: args.learning_rate_warmup_epochs,
        learning_rate_final_scale: args.learning_rate_final_scale,
        curvature,
        temperature,
        gen_len: args.gen_len,
        top_k: args.top_k,
        seed: args.seed,
        validation_fraction_requested: args.validation_fraction,
        validation_fraction_actual: split.actual_validation_fraction,
        validation_start_fraction_requested: args.validation_start_fraction,
        validation_start_fraction_actual: split.validation_start_fraction_actual,
        validation_start_token: split.validation_start_token,
        eval_samples: args.eval_samples,
        early_stop_patience: args.early_stop_patience,
        restore_best_at_end: args.restore_best_at_end,
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

    let mut loss = BigramTopKGuardedCrossEntropy::new(
        args.bigram_topk_guard,
        args.bigram_rank_guard,
        args.bigram_rank_guard_margin,
        args.bigram_soft_guard,
    )?;

    println!(
        "mode={mode} recurrent={} backend={} vocab={} files={} chars={} train_tokens={} validation_tokens={} steps={} hidden={} char_feature={} head_rms={} head_residual_scale={} head_prior={} bigram_topk_guard={} bigram_topk_guard_k={} bigram_rank_guard={} bigram_rank_guard_margin={} bigram_rank_guard_band={} bigram_rank_guard_min_candidates={} bigram_soft_guard={} epochs={} batch={} lr={:.3e} lr_schedule={} lr_warmup_epochs={} lr_final_scale={} curvature={} temp={} run_dir={}",
        recurrent,
        backend_sel.label,
        vocab.len(),
        data_files.len(),
        text.chars().count(),
        split.train.len(),
        split.validation.len(),
        steps,
        hidden,
        char_feature_label,
        args.head_weight_rms,
        head_residual_scale
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "none".to_string()),
        head_prior,
        args.bigram_topk_guard,
        args.bigram_topk_guard_k,
        args.bigram_rank_guard,
        args.bigram_rank_guard_margin,
        args.bigram_rank_guard_band,
        args.bigram_rank_guard_min_candidates,
        args.bigram_soft_guard,
        args.epochs,
        args.batch,
        args.learning_rate,
        learning_rate_schedule,
        args.learning_rate_warmup_epochs,
        args.learning_rate_final_scale,
        curvature,
        temperature,
        run_dir.display()
    );
    let roundtable_statuses = roundtable_backend_audit
        .bands
        .iter()
        .map(|band| format!("{}:{}", band.band, band.wgpu_exact_status))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "backend_policy matmul={} prepacked_matmul={} softmax={} tensor_util={} wgpu_rank_runtime_installed={} wgpu_rank_runtime_initialized={} roundtable_wgpu_exact_ready={} statuses={}",
        tensor_policy.matmul_backend,
        tensor_policy.prepacked_matmul_backend,
        tensor_policy.softmax_backend,
        tensor_policy.tensor_util_backend,
        backend_runtime.wgpu_rank_runtime_context_installed,
        backend_runtime.wgpu_rank_runtime_initialized,
        roundtable_backend_audit.any_wgpu_exact_runtime_ready,
        roundtable_statuses
    );

    let eval_input_mode = if embed_dim.is_some() {
        CharLmInputMode::TokenIndices
    } else {
        CharLmInputMode::OneHot
    };
    let initial_validation = evaluate_next_token_with_unigram_lift(
        &mut model,
        vocab.len(),
        eval_input_mode,
        steps,
        &split.train,
        &split.validation,
        args.eval_samples,
    )?;
    let unigram_validation = evaluate_unigram_next_token(
        vocab.len(),
        steps,
        &split.train,
        &split.validation,
        args.eval_samples,
    )?;
    let bigram_validation = evaluate_bigram_next_token(
        vocab.len(),
        steps,
        &split.train,
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
    if let Some(validation) = unigram_validation.as_ref() {
        println!(
            "validation[unigram] windows={} nll={:.6} ppl={} acc={:.2}% target_rank={:.2}",
            validation.windows,
            validation.mean_nll,
            validation
                .perplexity
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "overflow".to_string()),
            validation.accuracy * 100.0,
            validation.mean_target_rank
        );
    }
    if let Some(validation) = bigram_validation.as_ref() {
        println!(
            "validation[bigram] windows={} nll={:.6} ppl={} acc={:.2}% target_rank={:.2}",
            validation.windows,
            validation.mean_nll,
            validation
                .perplexity
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "overflow".to_string()),
            validation.accuracy * 100.0,
            validation.mean_target_rank
        );
    }

    let meta_head_prior = loaded_head_prior.clone().or_else(|| {
        (loaded_from.is_none() && head_prior_is_enabled(&args.head_prior))
            .then(|| args.head_prior.clone())
    });
    let meta_head_residual_scale = loaded_head_residual_scale
        .or_else(|| loaded_from.is_none().then_some(args.head_residual_scale));
    let meta = CharLmMeta::new(
        steps,
        hidden,
        embed_dim,
        recurrent.clone(),
        char_feature,
        meta_head_prior,
        meta_head_residual_scale,
        curvature,
        temperature,
        &vocab,
    );

    let mut last_sample: Option<String> = None;
    let mut last_validation = initial_validation.clone();
    let mut best_validation = None;
    let mut best_validation_epoch = None;
    let mut best_validation_mean_nll = None;
    let mut best_learning_rate = None;
    let mut best_checkpoint_path = None;
    let mut best_sample_path = None;
    let mut epochs_completed = 0usize;
    let mut epochs_without_validation_improvement = 0usize;
    let mut early_stopped_epoch = None;
    let mut active_learning_rate = args.learning_rate;
    for epoch in 0..args.epochs {
        let epoch_learning_rate = scheduled_learning_rate(
            args.learning_rate,
            epoch,
            args.epochs,
            args.learning_rate_warmup_epochs,
            args.learning_rate_final_scale,
        );
        let lr_scale = epoch_learning_rate / active_learning_rate;
        trainer.mul_learning_rate(&mut model, lr_scale)?;
        active_learning_rate = epoch_learning_rate;
        let mut rng = ChaCha8Rng::seed_from_u64(args.seed.wrapping_add(epoch as u64 * 10_000));
        let mut batches = Vec::with_capacity(args.batches_per_epoch);
        for _ in 0..args.batches_per_epoch {
            batches.push(build_random_batch(
                &split.train,
                vocab.len(),
                embed_dim,
                steps,
                args.batch,
                bigram_guard.as_ref(),
                &mut rng,
            )?);
        }
        let before_epoch = capture_parameter_snapshot(&model)?;
        let stats = trainer.train_epoch(&mut model, &mut loss, batches, &schedule)?;
        let validation = evaluate_next_token_with_unigram_lift(
            &mut model,
            vocab.len(),
            eval_input_mode,
            steps,
            &split.train,
            &split.validation,
            args.eval_samples,
        )?;
        let mut validation_is_best = false;
        if let Some(validation_metric) = validation.as_ref() {
            if best_validation_mean_nll
                .map(|best| validation_metric.mean_nll < best)
                .unwrap_or(true)
            {
                best_validation = Some(validation_metric.clone());
                best_validation_mean_nll = Some(validation_metric.mean_nll);
                best_validation_epoch = Some(epoch);
                best_learning_rate = Some(epoch_learning_rate);
                validation_is_best = true;
                epochs_without_validation_improvement = 0;
            } else {
                epochs_without_validation_improvement += 1;
            }
        }
        let learnability = summarize_learnability(
            &model,
            Some(&before_epoch),
            Some(stats.average_loss),
            validation.as_ref(),
        )?;
        let metric = EpochMetric {
            epoch,
            batches: stats.batches,
            learning_rate: epoch_learning_rate,
            learning_rate_scale: epoch_learning_rate / args.learning_rate,
            average_loss: stats.average_loss,
            tensor_backend: stats.tensor_backend,
            validation: validation.clone(),
            learnability: learnability.clone(),
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
        if validation_is_best {
            let best_weights = run_dir.join("best_weights.json");
            save_json(&model, &best_weights)?;
            write_meta(&meta_path_for_weights(&best_weights), &meta)?;
            let best_sample = samples_dir.join(format!("best_epoch_{epoch:03}.txt"));
            std::fs::write(&best_sample, &sample).map_err(|err| TensorError::IoError {
                message: err.to_string(),
            })?;
            best_checkpoint_path = Some(best_weights.to_string_lossy().to_string());
            best_sample_path = Some(best_sample.to_string_lossy().to_string());
        }
        last_sample = Some(sample);
        last_validation = validation.clone();
        if let Some(validation) = validation.as_ref() {
            println!(
                "epoch[{epoch}] batches={} lr={:.3e} avg_loss={:.6} val_nll={:.6} val_ppl={} val_acc={:.2}% update_l2={} update_ratio={}{}",
                stats.batches,
                epoch_learning_rate,
                stats.average_loss,
                validation.mean_nll,
                validation
                    .perplexity
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "overflow".to_string()),
                validation.accuracy * 100.0,
                learnability
                    .total_update_l2
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "-".to_string()),
                learnability
                    .mean_update_to_value_l2
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "-".to_string()),
                if validation_is_best { " best=checkpoint" } else { "" }
            );
        } else {
            println!(
                "epoch[{epoch}] batches={} lr={:.3e} avg_loss={:.6} val=skipped update_l2={} update_ratio={}",
                stats.batches,
                epoch_learning_rate,
                stats.average_loss,
                learnability
                    .total_update_l2
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "-".to_string()),
                learnability
                    .mean_update_to_value_l2
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "-".to_string())
            );
        }
        epochs_completed = epoch + 1;
        if args.early_stop_patience > 0
            && validation.is_some()
            && !validation_is_best
            && epochs_without_validation_improvement >= args.early_stop_patience
        {
            early_stopped_epoch = Some(epoch);
            println!(
                "early_stop[epoch={epoch}] patience={} best_epoch={} best_nll={:.6}",
                args.early_stop_patience,
                best_validation_epoch
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                best_validation_mean_nll.unwrap_or(f32::NAN)
            );
            break;
        }
    }

    let mut restored_best_at_end = false;
    let mut restored_best_checkpoint_path = None;
    if args.restore_best_at_end {
        if let Some(best_path) = best_checkpoint_path.clone() {
            load_json(&mut model, Path::new(&best_path))?;
            last_validation = best_validation.clone();
            let restored_sample = generate_text(
                &model,
                &vocab,
                embed_dim,
                steps,
                &prompt,
                args.gen_len,
                args.top_k,
                args.seed
                    .wrapping_add(999)
                    .wrapping_add(best_validation_epoch.unwrap_or(0) as u64),
            )?;
            std::fs::write(samples_dir.join("restored_best.txt"), &restored_sample).map_err(
                |err| TensorError::IoError {
                    message: err.to_string(),
                },
            )?;
            last_sample = Some(restored_sample);
            restored_best_at_end = true;
            restored_best_checkpoint_path = Some(best_path.clone());
            println!(
                "restore_best_at_end checkpoint={} best_epoch={} best_nll={:.6}",
                best_path,
                best_validation_epoch
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                best_validation_mean_nll.unwrap_or(f32::NAN)
            );
        } else {
            println!("restore_best_at_end skipped: no validation checkpoint was recorded");
        }
    }

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
    let final_vs_unigram_nll_delta = final_validation
        .as_ref()
        .zip(unigram_validation.as_ref())
        .map(|(final_metric, unigram)| final_metric.mean_nll - unigram.mean_nll);
    let final_vs_bigram_nll_delta = final_validation
        .as_ref()
        .zip(bigram_validation.as_ref())
        .map(|(final_metric, bigram)| final_metric.mean_nll - bigram.mean_nll);
    let best_validation_nll_delta = initial_validation
        .as_ref()
        .zip(best_validation.as_ref())
        .map(|(initial, best_metric)| best_metric.mean_nll - initial.mean_nll);
    let best_vs_unigram_nll_delta = best_validation
        .as_ref()
        .zip(unigram_validation.as_ref())
        .map(|(best_metric, unigram)| best_metric.mean_nll - unigram.mean_nll);
    let best_vs_bigram_nll_delta = best_validation
        .as_ref()
        .zip(bigram_validation.as_ref())
        .map(|(best_metric, bigram)| best_metric.mean_nll - bigram.mean_nll);
    let final_minus_best_validation_nll = final_validation
        .as_ref()
        .zip(best_validation.as_ref())
        .map(|(final_metric, best_metric)| final_metric.mean_nll - best_metric.mean_nll);
    let final_learning_rate = if restored_best_at_end {
        best_learning_rate.unwrap_or(active_learning_rate)
    } else {
        active_learning_rate
    };
    let summary = TrainingSummary {
        initial_validation,
        final_validation,
        unigram_validation,
        bigram_validation,
        best_validation,
        best_validation_epoch,
        best_validation_mean_nll,
        validation_nll_delta,
        validation_accuracy_delta,
        final_vs_unigram_nll_delta,
        final_vs_bigram_nll_delta,
        best_validation_nll_delta,
        best_vs_unigram_nll_delta,
        best_vs_bigram_nll_delta,
        final_minus_best_validation_nll,
        best_checkpoint_path,
        best_sample_path,
        restore_best_at_end: args.restore_best_at_end,
        restored_best_at_end,
        restored_best_checkpoint_path,
        learning_rate_schedule,
        learning_rate_warmup_epochs: args.learning_rate_warmup_epochs,
        learning_rate_final_scale: args.learning_rate_final_scale,
        best_learning_rate,
        final_learning_rate,
        epochs_completed,
        early_stopped_epoch,
    };
    write_summary(&run_dir.join("summary.json"), &summary)?;

    println!("--- sample (prompt + gen) ---");
    println!("{sample}");

    Ok(())
}
