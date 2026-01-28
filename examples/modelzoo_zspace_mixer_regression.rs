// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use st_core::backend::device_caps::DeviceCaps;
use st_core::plugin::{global_registry, PluginEvent};
use st_nn::{
    load_json, save_json, EpochStats, Linear, MeanSquaredError, Module, ModuleTrainer, Relu,
    RoundtableConfig, Sequential, Tensor, ZSpaceMixer,
};
use std::path::Path;
use std::sync::Arc;

fn build_model(in_dim: usize, hidden: usize, out_dim: usize) -> st_nn::PureResult<Sequential> {
    let mut model = Sequential::new();
    model.push(Linear::new("l1", in_dim, hidden)?);
    model.push(Relu::new());
    model.push(ZSpaceMixer::new("mixer", hidden)?);
    model.push(Linear::new("l2", hidden, out_dim)?);
    Ok(model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optional: observe epoch-level progress via the plugin event bus.
    global_registry().event_bus().subscribe(
        "EpochEnd",
        Arc::new(|event| {
            if let PluginEvent::EpochEnd { epoch, loss } = event {
                println!("[epoch={epoch}] avg_loss={loss:.6}");
            }
        }),
    );

    let batch = 8u32;
    let in_dim = 4usize;
    let hidden = 16usize;
    let out_dim = 1u32;

    let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 1e-2, 1e-2);
    let schedule = trainer.roundtable(
        batch,
        out_dim,
        RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5),
    );

    let mut model = build_model(in_dim, hidden, out_dim as usize)?;
    model.attach_hypergrad(-1.0, 1e-2)?;

    let mut loss = MeanSquaredError::new();

    let x = Tensor::random_uniform(batch as usize, in_dim, -1.0, 1.0, Some(7))?;
    let y = Tensor::from_fn(batch as usize, 1, |r, _| {
        let offset = r * in_dim;
        let x0 = x.data()[offset];
        let x1 = x.data()[offset + 1];
        let x2 = x.data()[offset + 2];
        let x3 = x.data()[offset + 3];
        0.35 * x0 - 0.6 * x1 + 0.15 * x2 + 0.05 * x3
    })?;
    let batches = vec![(x.clone(), y.clone())];

    for _ in 0..4 {
        let EpochStats {
            batches,
            average_loss,
            ..
        } = trainer.train_epoch(&mut model, &mut loss, batches.clone(), &schedule)?;
        println!("stats: batches={batches} avg_loss={average_loss:.6}");
    }

    let weights_path = Path::new("models/weights/zspace_mixer_regression.json");
    if let Some(parent) = weights_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save_json(&model, weights_path)?;

    let mut reloaded = build_model(in_dim, hidden, out_dim as usize)?;
    load_json(&mut reloaded, weights_path)?;
    let _ = reloaded.forward(&x)?;

    Ok(())
}

