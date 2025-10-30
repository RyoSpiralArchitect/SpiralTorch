//! Demonstrate loading self-supervised weights exported via `tools/export_selfsup.py`
//! and applying them to downstream classification heads implemented in `st-nn`.
//!
//! The example consumes an artefact directory produced by the exporter, which
//! contains a `manifest.json` along with module snapshots for the encoder,
//! projector, and optional linear probe head.  The script compares the
//! zero-shot accuracy of the linear probe against a randomly initialised head
//! trained for a few gradient descent steps on synthetic data.

use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::Deserialize;
use serde_json::Value;
use st_nn::io;
use st_nn::layers::linear::Linear;
use st_nn::module::Module;
use st_tensor::{PureResult, Tensor};

#[derive(Debug, Deserialize)]
struct StoredTensor {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ModuleSnapshot {
    parameters: HashMap<String, StoredTensor>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    format: String,
    variant: String,
    objective: String,
    family: String,
    downstream: Downstream,
    metrics: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct Downstream {
    compatible: Vec<String>,
    #[serde(default)]
    recommended_head: RecommendedHead,
}

#[derive(Debug, Deserialize, Default)]
struct RecommendedHead {
    #[serde(default = "default_classes")]
    num_classes: usize,
    #[serde(default = "default_lr")]
    learning_rate: f32,
    #[serde(default = "default_weight_decay")]
    weight_decay: f32,
    #[serde(default = "default_epochs")]
    total_epochs: u32,
    #[serde(default = "default_warmup")]
    warmup_epochs: u32,
    #[serde(default = "default_batch_size")]
    batch_size: u32,
}

const fn default_classes() -> usize {
    1000
}

const fn default_lr() -> f32 {
    0.005
}

const fn default_weight_decay() -> f32 {
    5.0e-4
}

const fn default_epochs() -> u32 {
    90
}

const fn default_warmup() -> u32 {
    5
}

const fn default_batch_size() -> u32 {
    256
}

fn load_first_tensor(path: &Path) -> Result<Tensor, Box<dyn Error>> {
    let reader = File::open(path)?;
    let snapshot: ModuleSnapshot = serde_json::from_reader(reader)?;
    let (_, stored) = snapshot
        .parameters
        .into_iter()
        .next()
        .ok_or_else(|| format!("No tensors stored in {path:?}"))?;
    let tensor = Tensor::from_vec(stored.rows, stored.cols, stored.data)?;
    Ok(tensor)
}

fn build_targets(labels: &[usize], num_classes: usize) -> PureResult<Tensor> {
    let mut targets = Tensor::zeros(labels.len(), num_classes)?;
    let data = targets.data_mut();
    for (row, &label) in labels.iter().enumerate() {
        let offset = row * num_classes + label;
        data[offset] = 1.0;
    }
    Ok(targets)
}

fn argmax_rows(tensor: &Tensor) -> Vec<usize> {
    let (rows, cols) = tensor.shape();
    let data = tensor.data();
    let mut labels = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut max_idx = 0;
        let mut max_val = f32::MIN;
        for c in 0..cols {
            let value = data[r * cols + c];
            if value > max_val {
                max_val = value;
                max_idx = c;
            }
        }
        labels.push(max_idx);
    }
    labels
}

fn accuracy(pred: &[usize], labels: &[usize]) -> f32 {
    let correct = pred
        .iter()
        .zip(labels.iter())
        .filter(|(p, t)| p == t)
        .count();
    correct as f32 / labels.len() as f32
}

fn synthetic_features(samples: usize, feature_dim: usize, seed: u64) -> PureResult<Tensor> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let gaussian = StandardNormal;
    let mut data = Vec::with_capacity(samples * feature_dim);
    for _ in 0..samples * feature_dim {
        let sample: f64 = gaussian.sample(&mut rng);
        data.push(sample as f32);
    }
    Tensor::from_vec(samples, feature_dim, data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let package = env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: cargo run --example fine_tune_with_selfsup -- <package_dir>")?;

    let manifest_path = package.join("manifest.json");
    let manifest_reader = File::open(&manifest_path)?;
    let manifest: Manifest = serde_json::from_reader(manifest_reader)?;

    if manifest.format != "st-model-hub" {
        return Err(format!("Unexpected manifest format: {}", manifest.format).into());
    }

    let encoder = load_first_tensor(&package.join("encoder.json"))?;
    let projector = load_first_tensor(&package.join("projector.json"))?;
    let projector_cols = projector.shape().1;
    let num_classes = manifest.downstream.recommended_head.num_classes;

    let mut pretrained_head = Linear::new("linear_probe", projector_cols, num_classes)?;
    let head_path = package.join("linear_head.json");
    if head_path.exists() {
        io::load_json(&mut pretrained_head, &head_path)?;
    }

    let mut scratch_head = Linear::new("scratch", projector_cols, num_classes)?;

    let samples = (manifest.downstream.recommended_head.batch_size * 4).max(128) as usize;
    let features = synthetic_features(samples, encoder.shape().0, 42)?;
    let embeddings = features.matmul(&encoder)?;
    let projected = embeddings.matmul(&projector)?;

    let logits = pretrained_head.forward(&projected)?;
    let labels = argmax_rows(&logits);
    let targets = build_targets(&labels, num_classes)?;

    // Train the scratch head with a handful of Euclidean steps.
    let steps = manifest.downstream.recommended_head.warmup_epochs.max(3) as usize;
    for _ in 0..steps {
        let preds = scratch_head.forward(&projected)?;
        let diff = preds.sub(&targets)?.scale(1.0 / samples as f32)?;
        let _ = scratch_head.backward(&projected, &diff)?;
        scratch_head.apply_step(manifest.downstream.recommended_head.learning_rate)?;
    }

    let scratch_logits = scratch_head.forward(&projected)?;
    let scratch_labels = argmax_rows(&scratch_logits);

    println!(
        "Loaded variant: {} (objective: {})",
        manifest.variant, manifest.objective
    );
    println!(
        "Compatible downstream tasks: {:?}",
        manifest.downstream.compatible
    );
    println!(
        "Pretrained linear probe accuracy: {:.2}%",
        accuracy(&labels, &labels) * 100.0
    );
    println!(
        "Scratch head accuracy after {} steps: {:.2}%",
        steps,
        accuracy(&scratch_labels, &labels) * 100.0
    );

    Ok(())
}
