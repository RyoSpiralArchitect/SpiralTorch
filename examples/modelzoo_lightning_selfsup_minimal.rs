// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Training model-zoo: SpiralLightning + self-supervised InfoNCE (minimal).

#[cfg(not(feature = "selfsup"))]
fn main() {
    eprintln!("This example requires building `st-nn` with the `selfsup` feature.");
}

#[cfg(feature = "selfsup")]
mod demo {
    use st_core::backend::device_caps::DeviceCaps;
    use st_nn::{
        load_json, save_json, InfoNCEConfig, LightningConfig, Linear, Module, Relu, RoundtableConfig,
        SelfSupBatch, SelfSupEpoch, SelfSupEpochTelemetry, SelfSupObjective, SelfSupStage,
        Sequential, SpiralLightning, SpiralSession, Tensor,
    };
    use std::path::Path;

    fn build_model(input_dim: usize, embed_dim: usize) -> st_nn::PureResult<Sequential> {
        let mut model = Sequential::new();
        model.push(Linear::new("enc1", input_dim, embed_dim)?);
        model.push(Relu::new());
        model.push(Linear::new("enc2", embed_dim, embed_dim)?);
        Ok(model)
    }

    fn build_epoch(
        pairs_per_epoch: usize,
        pair_batch: usize,
        input_dim: usize,
        seed: u64,
    ) -> st_nn::PureResult<SelfSupEpoch> {
        let mut batches = Vec::with_capacity(pairs_per_epoch);
        for idx in 0..pairs_per_epoch {
            let anchors = Tensor::random_uniform(
                pair_batch,
                input_dim,
                -1.0,
                1.0,
                Some(seed.wrapping_add(idx as u64)),
            )?;
            let jitter = Tensor::random_uniform(
                pair_batch,
                input_dim,
                -0.05,
                0.05,
                Some(seed.wrapping_add(10_000).wrapping_add(idx as u64)),
            )?;
            let positives = anchors.add(&jitter)?;
            batches.push(SelfSupBatch::from_pairs(anchors, positives)?);
        }
        Ok(SelfSupEpoch::new(batches))
    }

    pub fn run() -> st_nn::PureResult<()> {
        let pair_batch = 4usize;
        let input_dim = 8usize;
        let embed_dim = 6usize;
        let combined_rows = pair_batch * 2;

        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .with_hyper_learning_rate(2e-2)
            .with_fallback_learning_rate(2e-2)
            .build()?;

        let roundtable = RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5);

        let cfg = LightningConfig::new(combined_rows as u32, embed_dim as u32).with_roundtable(roundtable);
        let mut lightning = SpiralLightning::with_config(session, cfg.clone());

        let mut model = build_model(input_dim, embed_dim)?;

        let objective = SelfSupObjective::InfoNCE(InfoNCEConfig::new(0.2, true));
        let epochs = (0..6)
            .map(|idx| build_epoch(4, pair_batch, input_dim, 7_000 + idx as u64 * 100))
            .collect::<Result<Vec<_>, _>>()?;
        let stage = SelfSupStage::with_epochs(cfg.clone(), objective, epochs).with_label("selfsup.minimal");
        let report = lightning.fit_selfsup_plan(&mut model, [stage])?;

        for (stage_idx, stage) in report.stages().iter().enumerate() {
            let label = stage.label().unwrap_or("—");
            println!("stage[{stage_idx}] label={label}");
            for (epoch_idx, epoch) in stage.epochs().iter().enumerate() {
                let stats = epoch.stats();
                let mean_loss = epoch.telemetry().and_then(|telemetry| match telemetry {
                    SelfSupEpochTelemetry::InfoNCE(metrics) => Some(metrics.mean_loss),
                });
                println!(
                    "  epoch[{epoch_idx}] batches={} avg_loss={:.6} info_nce={}",
                    stats.batches,
                    stats.average_loss,
                    mean_loss
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "—".to_string())
                );
            }
        }

        let weights_path = Path::new("models/weights/lightning_selfsup_minimal.json");
        if let Some(parent) = weights_path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| st_nn::TensorError::IoError {
                message: err.to_string(),
            })?;
        }
        save_json(&model, weights_path)?;

        let mut reloaded = build_model(input_dim, embed_dim)?;
        load_json(&mut reloaded, weights_path)?;
        let sanity = Tensor::random_uniform(combined_rows, input_dim, -1.0, 1.0, Some(999))?;
        let _ = reloaded.forward(&sanity)?;

        Ok(())
    }
}

#[cfg(feature = "selfsup")]
fn main() -> st_nn::PureResult<()> {
    demo::run()
}
