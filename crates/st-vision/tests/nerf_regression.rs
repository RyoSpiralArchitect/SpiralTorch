// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
#![cfg(feature = "nerf")]

use rand::{rngs::StdRng, SeedableRng};
use st_nn::module::Module;
use st_tensor::Tensor;

use st_vision::datasets::{MultiViewDatasetAdapter, MultiViewFrame};
use st_vision::nerf::{NerfField, NerfFieldConfig, NerfTrainer, NerfTrainingConfig};

fn synthetic_dataset() -> MultiViewDatasetAdapter {
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
    MultiViewDatasetAdapter::new(vec![frame]).unwrap()
}

fn field_config() -> NerfFieldConfig {
    NerfFieldConfig {
        hidden_width: 64,
        hidden_layers: 2,
        feature_dim: 16,
        color_layers: 1,
        color_hidden_width: 32,
        ..Default::default()
    }
}

fn training_config() -> NerfTrainingConfig {
    NerfTrainingConfig {
        samples_per_ray: 8,
        batch_size: 32,
        learning_rate: 1e-3,
        steps_per_epoch: 5,
        stratified: true,
        seed: 7,
    }
}

// Evaluate a parameter snapshot at fixed midpoints without advancing the
// training sampler or comparing two different stratified sample draws.
fn evaluation_loss(field: &NerfField, dataset: &MultiViewDatasetAdapter) -> f64 {
    let mut values = Vec::new();
    field
        .visit_parameters(&mut |p| {
            values.push(p.value().clone());
            Ok(())
        })
        .unwrap();
    let mut snapshot = NerfField::new(field_config()).unwrap();
    let mut values = values.into_iter();
    snapshot
        .visit_parameters_mut(&mut |p| {
            *p.value_mut() = values.next().unwrap();
            Ok(())
        })
        .unwrap();
    assert!(values.next().is_none());
    let mut config = training_config();
    config.stratified = false;
    let mut evaluator = NerfTrainer::new(snapshot, config.clone()).unwrap();
    let batch = dataset
        .sample_batch(&mut StdRng::seed_from_u64(11), config.batch_size)
        .unwrap();
    let predictions = evaluator.render_batch(&batch).unwrap();
    assert!(predictions.data().iter().all(|v| v.is_finite()));
    predictions
        .data()
        .iter()
        .zip(batch.colors.data())
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>()
        / predictions.data().len() as f64
}

#[test]
fn synthetic_constant_color_regression() {
    let dataset = synthetic_dataset();
    let field = NerfField::new(field_config()).unwrap();
    let trainer_config = training_config();
    let mut trainer = NerfTrainer::new(field, trainer_config.clone()).unwrap();

    let mut last_loss = f32::MAX;
    let mut best_loss = f32::MAX;
    for _ in 0..4 {
        let stats = trainer.train_epoch(&dataset).unwrap();
        last_loss = stats.loss;
        best_loss = best_loss.min(stats.loss);
    }
    assert!(
        best_loss.is_finite() && best_loss < 0.2,
        "loss should converge, got {best_loss}"
    );
    assert!(last_loss.is_finite());

    let mut rng = StdRng::seed_from_u64(11);
    let eval_batch = dataset
        .sample_batch(&mut rng, trainer_config.batch_size)
        .unwrap();
    let predictions = trainer.render_batch(&eval_batch).unwrap();
    assert!(predictions.data().iter().all(|value| value.is_finite()));
}

#[test]
fn seeded_training_makes_progress_on_fixed_evaluation() {
    let dataset = synthetic_dataset();
    let mut failures = Vec::new();
    for seed in [0, 1, 7, 13] {
        let field = NerfField::new_with_seed(field_config(), seed).unwrap();
        let mut trainer = NerfTrainer::new(field, training_config()).unwrap();
        let before = evaluation_loss(trainer.field(), &dataset);
        let mut losses = Vec::new();
        for _ in 0..4 {
            let stats = trainer.train_epoch(&dataset).unwrap();
            assert!(stats.loss.is_finite() && stats.avg_transmittance.is_finite());
            losses.push(stats.loss);
        }
        let after = evaluation_loss(trainer.field(), &dataset);
        eprintln!("NERF_TRAINING seed={seed} steps=20 before={before:.9} after={after:.9} epoch_losses={losses:?}");
        // Twenty small SGD steps are a progress check, not a convergence claim.
        // An absolute target-only bound also passes an entirely dead field.
        if after >= before - 1e-5 {
            failures.push((seed, before, after));
        }
    }
    assert!(failures.is_empty(), "fixed evaluation failed: {failures:?}");
}
