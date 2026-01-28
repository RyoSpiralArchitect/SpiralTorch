// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    load_json, save_json, EpochStats, Linear, MeanSquaredError, Module, ModuleTrainer, Relu,
    RoundtableConfig, Sequential, Tensor, WaveRnn,
};
use std::path::Path;

fn build_model(hidden: usize) -> st_nn::PureResult<Sequential> {
    let mut model = Sequential::new();
    model.push(WaveRnn::new("wrnn", 1, hidden, 3, 1, 1, -1.0, 0.5)?);
    model.push(Relu::new());
    model.push(Linear::new("head", hidden, 1)?);
    Ok(model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch = 6u32;
    let steps = 12usize;
    let hidden = 8usize;
    let out_dim = 1u32;

    let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 2e-2, 2e-2);
    let schedule = trainer.roundtable(
        batch,
        out_dim,
        RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5),
    );

    let mut model = build_model(hidden)?;
    model.attach_hypergrad(-1.0, 2e-2)?;

    let mut loss = MeanSquaredError::new();

    // Input is flattened 1D: (batch, in_channels * steps) with in_channels=1.
    let x = Tensor::random_uniform(batch as usize, steps, -1.0, 1.0, Some(123))?;
    let y = Tensor::from_fn(batch as usize, 1, |r, _| {
        let start = r * steps;
        let row = &x.data()[start..start + steps];
        let first = row[0];
        let last = row[steps - 1];
        let mean = row.iter().sum::<f32>() / steps as f32;
        0.6 * last - 0.4 * first + 0.1 * mean
    })?;
    let batches = vec![(x.clone(), y.clone())];

    for _ in 0..6 {
        let EpochStats {
            batches,
            average_loss,
            ..
        } = trainer.train_epoch(&mut model, &mut loss, batches.clone(), &schedule)?;
        println!("stats: batches={batches} avg_loss={average_loss:.6}");
    }

    let weights_path = Path::new("models/weights/wave_rnn_sequence.json");
    if let Some(parent) = weights_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save_json(&model, weights_path)?;

    let mut reloaded = build_model(hidden)?;
    load_json(&mut reloaded, weights_path)?;
    let _ = reloaded.forward(&x)?;

    Ok(())
}

