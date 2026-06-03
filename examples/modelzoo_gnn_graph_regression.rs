// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    load_json, save_json, GraphActivation, GraphContext, GraphLayerSpec, MeanSquaredError, Module,
    ModuleTrainer, RoundtableConfig, Tensor, TrainingRunConfig, ZSpaceGraphNetwork,
    ZSpaceGraphNetworkBuilder,
};
use std::num::NonZeroUsize;
use std::path::Path;

fn build_context(nodes: usize) -> st_nn::PureResult<GraphContext> {
    let mut adjacency = vec![0.0f32; nodes * nodes];
    for i in 0..nodes {
        if i + 1 < nodes {
            adjacency[i * nodes + (i + 1)] = 1.0;
            adjacency[(i + 1) * nodes + i] = 1.0;
        }
    }
    let tensor = Tensor::from_vec(nodes, nodes, adjacency)?;
    GraphContext::from_adjacency(tensor)
}

fn build_model(
    context: GraphContext,
    features: usize,
    curvature: f32,
    learning_rate: f32,
) -> st_nn::PureResult<ZSpaceGraphNetwork> {
    let mut builder = ZSpaceGraphNetworkBuilder::new(
        context,
        NonZeroUsize::new(features).ok_or(st_nn::TensorError::InvalidDimensions {
            rows: 1,
            cols: features,
        })?,
        curvature,
        learning_rate,
    );
    builder.push_layer(
        GraphLayerSpec::new(NonZeroUsize::new(features * 2).unwrap())
            .with_activation(GraphActivation::Relu),
    );
    builder.push_layer(GraphLayerSpec::new(NonZeroUsize::new(features).unwrap()));
    builder.build("gnn_graph")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nodes = 6usize;
    let features = 3usize;

    let context = build_context(nodes)?;
    let x = Tensor::random_uniform(nodes, features, -1.0, 1.0, Some(99))?;
    let propagated = context.propagate(&x)?;
    let target = x.scale(0.25)?.add(&propagated.scale(0.75)?)?;

    let mut model = build_model(context.clone(), features, -1.0, 0.05)?;
    model.attach_hypergrad(-1.0, 2e-2)?;

    let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 2e-2, 2e-2);
    let schedule = trainer.roundtable(
        nodes as u32,
        features as u32,
        RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5),
    );

    let mut loss = MeanSquaredError::new();
    let batches = vec![(x.clone(), target.clone())];
    let validation = vec![(propagated.clone(), target.clone())];

    let report = trainer.train_epochs(
        &mut model,
        &mut loss,
        &batches,
        Some(validation.as_slice()),
        &schedule,
        TrainingRunConfig::new(12)
            .with_validation_patience(Some(3))
            .with_min_delta(1e-5),
    )?;
    for epoch in &report.epochs {
        let val_loss = epoch
            .validation
            .map(|stats| format!("{:.6}", stats.average_loss))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "epoch={} train_loss={:.6} val_loss={} improved={}",
            epoch.epoch, epoch.train.average_loss, val_loss, epoch.improved
        );
    }
    if let Some(best) = report.best_epoch() {
        println!(
            "best: epoch={} score={:.6} stopped_early={}",
            best.epoch, best.score, report.stopped_early
        );
    }

    let weights_path = Path::new("models/weights/gnn_graph_regression.json");
    if let Some(parent) = weights_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save_json(&model, weights_path)?;

    let mut reloaded = build_model(context, features, -1.0, 0.05)?;
    load_json(&mut reloaded, weights_path)?;
    let _ = reloaded.forward(&x)?;

    Ok(())
}
