// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    load_json, save_json, Dataset, GraphActivation, GraphContext, GraphLayerSpec, GraphReadout,
    MeanSquaredError, Module, ModuleTrainer, RoundtableConfig, Tensor, TrainingRunConfig,
    ZSpaceGraphBatchRegressor, ZSpaceGraphNetworkBuilder,
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
    nodes: usize,
    features: usize,
    curvature: f32,
    learning_rate: f32,
) -> st_nn::PureResult<ZSpaceGraphBatchRegressor> {
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
    let network = builder.build("gnn_graph")?;
    Ok(ZSpaceGraphBatchRegressor::new(
        network,
        GraphReadout::Mean,
        NonZeroUsize::new(nodes).ok_or(st_nn::TensorError::InvalidDimensions {
            rows: nodes,
            cols: features,
        })?,
    ))
}

fn graph_sample(
    context: &GraphContext,
    nodes: usize,
    features: usize,
    seed: u64,
) -> st_nn::PureResult<(Tensor, Tensor)> {
    let input = Tensor::random_uniform(nodes, features, -1.0, 1.0, Some(seed))?;
    let propagated = context.propagate(&input)?;
    let node_target = input.scale(0.25)?.add(&propagated.scale(0.75)?)?;
    let graph_target = GraphReadout::Mean.forward(&node_target)?;
    Ok((input, graph_target))
}

fn graph_samples(
    context: &GraphContext,
    nodes: usize,
    features: usize,
    seed: u64,
    count: usize,
) -> st_nn::PureResult<Vec<(Tensor, Tensor)>> {
    (0..count)
        .map(|idx| graph_sample(context, nodes, features, seed + idx as u64))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nodes = 6usize;
    let features = 3usize;

    let context = build_context(nodes)?;
    let train_samples = graph_samples(&context, nodes, features, 99, 16)?;
    let validation_samples = graph_samples(&context, nodes, features, 1_099, 4)?;
    let reload_probe = Tensor::cat_rows(
        &validation_samples
            .iter()
            .take(2)
            .map(|(input, _)| input.clone())
            .collect::<Vec<_>>(),
    )?;
    let train_loader = Dataset::from_vec(train_samples)
        .loader()
        .shuffle(99)
        .batched(4)
        .prefetch(2);
    let validation_loader = Dataset::from_vec(validation_samples).loader().batched(2);

    let mut model = build_model(context.clone(), nodes, features, -1.0, 0.05)?;
    model.attach_hypergrad(-1.0, 2e-2)?;

    let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 2e-2, 2e-2);
    let schedule = trainer.roundtable(
        4,
        features as u32,
        RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5),
    );

    let mut loss = MeanSquaredError::new();

    let report = trainer.train_epochs_loader(
        &mut model,
        &mut loss,
        &train_loader,
        Some(&validation_loader),
        &schedule,
        TrainingRunConfig::new(12)
            .with_validation_patience(Some(3))
            .with_min_delta(1e-5)
            .with_epoch_shuffle_seed(Some(99))
            .with_restore_best(true),
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
            "best: epoch={} score={:.6} stopped_early={} restored_best={}",
            best.epoch, best.score, report.stopped_early, report.restored_best
        );
    }

    let weights_path = Path::new("models/weights/gnn_graph_regression.json");
    if let Some(parent) = weights_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save_json(&model, weights_path)?;

    let mut reloaded = build_model(context, nodes, features, -1.0, 0.05)?;
    load_json(&mut reloaded, weights_path)?;
    let graph_prediction = reloaded.forward(&reload_probe)?;
    println!(
        "reloaded graph prediction shape={:?} readout={:?} nodes_per_graph={}",
        graph_prediction.shape(),
        reloaded.readout(),
        reloaded.nodes_per_graph().get()
    );

    Ok(())
}
