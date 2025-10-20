// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
#![cfg(feature = "nerf")]

use rand::{rngs::StdRng, SeedableRng};
use st_logic::nerf_trainer::{NerfTrainer, NerfTrainingConfig};
use st_tensor::Tensor;
use st_vision::datasets::{MultiViewDatasetAdapter, MultiViewFrame};
use st_vision::nerf::{NerfField, NerfFieldConfig};

#[test]
fn synthetic_constant_color_regression() {
    let frame = MultiViewFrame::new(
        Tensor::from_vec(64, 3, vec![0.0; 64 * 3]).unwrap(),
        Tensor::from_vec(
            64,
            3,
            vec![0.0, 0.0, 1.0]
                .into_iter()
                .cycle()
                .take(64 * 3)
                .collect(),
        )
        .unwrap(),
        Tensor::from_vec(
            64,
            3,
            vec![0.7, 0.2, 0.1]
                .into_iter()
                .cycle()
                .take(64 * 3)
                .collect(),
        )
        .unwrap(),
        Tensor::from_vec(
            64,
            2,
            vec![0.0, 1.0].into_iter().cycle().take(64 * 2).collect(),
        )
        .unwrap(),
    )
    .unwrap();
    let dataset = MultiViewDatasetAdapter::new(vec![frame]).unwrap();

    let field = NerfField::new(NerfFieldConfig {
        hidden_width: 64,
        hidden_layers: 2,
        feature_dim: 16,
        color_layers: 1,
        color_hidden_width: 32,
        ..Default::default()
    })
    .unwrap();

    let mut trainer = NerfTrainer::new(
        field,
        NerfTrainingConfig {
            samples_per_ray: 8,
            batch_size: 32,
            learning_rate: 1e-3,
            steps_per_epoch: 5,
            stratified: true,
            seed: 7,
        },
    )
    .unwrap();

    let mut last_loss = f32::MAX;
    for _ in 0..4 {
        let stats = trainer.train_epoch(&dataset).unwrap();
        last_loss = stats.loss;
    }
    assert!(last_loss < 0.05, "loss should converge, got {last_loss}");

    let mut rng = StdRng::seed_from_u64(11);
    let eval_batch = dataset.sample_batch(&mut rng, 16).unwrap();
    let predictions = trainer.render_batch(&eval_batch).unwrap();
    let preds = predictions.data();
    for chunk in preds.chunks(3) {
        assert!((chunk[0] - 0.7).abs() < 0.1);
        assert!((chunk[1] - 0.2).abs() < 0.1);
        assert!((chunk[2] - 0.1).abs() < 0.1);
    }
}
