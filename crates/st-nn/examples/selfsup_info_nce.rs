#![cfg(feature = "selfsup")]

// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    InfoNCEConfig, LightningConfig, Linear, SelfSupBatch, SelfSupEpoch, SelfSupEpochTelemetry,
    SelfSupObjective, SelfSupStage, SpiralLightning, SpiralSession, Tensor,
};
use st_tensor::PureResult;

fn synthetic_pair(batch: usize, dim: usize, rng: &mut StdRng) -> PureResult<SelfSupBatch> {
    let mut anchors = Vec::with_capacity(batch * dim);
    let mut positives = Vec::with_capacity(batch * dim);
    for _ in 0..batch {
        for _ in 0..dim {
            anchors.push(rng.gen_range(-0.5f32..0.5f32));
        }
        for _ in 0..dim {
            positives.push(rng.gen_range(-0.5f32..0.5f32));
        }
    }
    let anchor_tensor = Tensor::from_vec(batch, dim, anchors)?;
    let positive_tensor = Tensor::from_vec(batch, dim, positives)?;
    SelfSupBatch::from_pairs(anchor_tensor, positive_tensor)
}

fn main() -> PureResult<()> {
    let caps = DeviceCaps::cpu();
    let session = SpiralSession::builder(caps).build()?;
    let mut lightning = SpiralLightning::new(session, 8, 4);

    let mut encoder = Linear::new("encoder", 4, 4)?;

    let mut rng = StdRng::seed_from_u64(0x5E1F_5u64);
    let mut batches = Vec::new();
    for _ in 0..4 {
        batches.push(synthetic_pair(4, 4, &mut rng)?);
    }
    let epoch = SelfSupEpoch::new(batches);

    let config = LightningConfig::new(8, 4);
    let objective = SelfSupObjective::InfoNCE(InfoNCEConfig::new(0.1, true));
    let stage = SelfSupStage::with_epochs(config, objective, vec![epoch]).with_label("warmup");

    let report = lightning.fit_selfsup_plan(&mut encoder, vec![stage])?;

    for (stage_idx, stage) in report.stages().iter().enumerate() {
        println!(
            "stage {stage_idx} → rows={} cols={}",
            stage.config().rows(),
            stage.config().cols()
        );
        for (epoch_idx, epoch) in stage.epochs().iter().enumerate() {
            match epoch.telemetry() {
                Some(SelfSupEpochTelemetry::InfoNCE(metrics)) => {
                    println!(
                        "  epoch {epoch_idx}: mean InfoNCE loss {:.6} across {} batches",
                        metrics.mean_loss, metrics.batches
                    );
                }
                None => println!("  epoch {epoch_idx}: telemetry unavailable"),
            }
        }
    }

    Ok(())
}
