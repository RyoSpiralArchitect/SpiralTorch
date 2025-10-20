use rand::{rngs::StdRng, Rng, SeedableRng};
use spiral_selfsup::{contrastive, masked};

fn synthetic_batch(seed: u64, batch: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch)
        .map(|_| {
            (0..dim)
                .map(|_| rng.gen_range(-1.0f32..1.0f32))
                .collect::<Vec<f32>>()
        })
        .collect()
}

#[test]
fn info_nce_reproducible_across_seeds() {
    let anchors = synthetic_batch(42, 4, 8);
    let positives = synthetic_batch(1337, 4, 8);

    let first = contrastive::info_nce_loss(&anchors, &positives, 0.1, true).unwrap();
    let second = contrastive::info_nce_loss(&anchors, &positives, 0.1, true).unwrap();

    assert!((first.loss - second.loss).abs() < 1e-6);
    assert_eq!(first.logits, second.logits);
    assert_eq!(first.labels, second.labels);
}

#[test]
fn masked_mse_reproducible_with_identical_seed() {
    let preds = synthetic_batch(7, 3, 5);
    let targets = synthetic_batch(7, 3, 5); // identical seed => deterministic difference
    let mask = vec![vec![0, 1, 2], vec![1, 3], vec![0, 4]];

    let first = masked::masked_mse_loss(&preds, &targets, &mask).unwrap();
    let second = masked::masked_mse_loss(&preds, &targets, &mask).unwrap();

    assert!((first.loss - second.loss).abs() < 1e-6);
    assert_eq!(first.total_masked, second.total_masked);
    assert_eq!(first.per_example, second.per_example);
}
