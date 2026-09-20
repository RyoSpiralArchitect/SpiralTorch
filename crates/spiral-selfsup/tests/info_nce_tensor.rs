use rand::{rngs::StdRng, Rng, SeedableRng};
use spiral_selfsup::contrastive::{self, info_nce_loss_tensor};
use st_tensor::{Layout, Tensor};

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

fn tensor_from_batch(batch: &[Vec<f32>]) -> Tensor {
    let rows = batch.len();
    let cols = batch[0].len();
    let data = batch.iter().flat_map(|row| row.iter().copied()).collect();
    Tensor::from_vec(rows, cols, data).expect("valid tensor construction")
}

#[test]
fn tensor_layouts_preserve_logical_rows() {
    let anchors = synthetic_batch(31, 5, 7);
    let positives = synthetic_batch(59, 5, 7);
    for normalize in [false, true] {
        let expected = contrastive::info_nce_loss(&anchors, &positives, 0.3, normalize).unwrap();
        for anchor_layout in [Layout::RowMajor, Layout::ColMajor] {
            for positive_layout in [Layout::RowMajor, Layout::ColMajor] {
                let a = tensor_from_batch(&anchors)
                    .to_layout(anchor_layout)
                    .unwrap();
                let p = tensor_from_batch(&positives)
                    .to_layout(positive_layout)
                    .unwrap();
                let actual = info_nce_loss_tensor(&a, &p, 0.3, normalize).unwrap();
                assert!((actual.loss - expected.loss).abs() < 1e-5);
                for (&actual, &expected) in actual.logits.data().iter().zip(&expected.logits) {
                    assert!((actual - expected).abs() < 1e-5);
                }
                let as_result =
                    contrastive::info_nce_loss_tensor_as_result(&a, &p, 0.3, normalize).unwrap();
                assert_eq!(as_result, expected);
            }
        }
    }
}

#[test]
fn vector_result_rejects_empty_tensor_without_panicking() {
    for (rows, cols) in [(0, 3), (2, 0), (0, 0)] {
        let empty = Tensor::zeros(rows, cols).unwrap();
        assert!(contrastive::info_nce_loss_tensor_as_result(&empty, &empty, 0.3, true).is_err());
    }
}

#[test]
fn entry_points_reject_invalid_temperatures_and_shapes() {
    let a = tensor_from_batch(&synthetic_batch(7, 3, 5));
    let p = tensor_from_batch(&synthetic_batch(9, 3, 5));
    for temperature in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(info_nce_loss_tensor(&a, &p, temperature, true).is_err());
        assert!(contrastive::info_nce_loss_tensor_as_result(&a, &p, temperature, true).is_err());
    }
    for (rows, cols) in [(2, 5), (3, 4)] {
        let mismatched = Tensor::zeros(rows, cols).unwrap();
        assert!(info_nce_loss_tensor(&a, &mismatched, 0.3, true).is_err());
        assert!(contrastive::info_nce_loss_tensor_as_result(&a, &mismatched, 0.3, true).is_err());
    }
}

#[test]
fn tensor_wrapper_matches_vector_api() {
    let anchors = synthetic_batch(11, 4, 6);
    let positives = synthetic_batch(23, 4, 6);

    let tensor_result = info_nce_loss_tensor(
        &tensor_from_batch(&anchors),
        &tensor_from_batch(&positives),
        0.2,
        true,
    )
    .unwrap();

    let vector_result = contrastive::info_nce_loss(&anchors, &positives, 0.2, true).unwrap();

    assert!((tensor_result.loss - vector_result.loss).abs() < 1e-6);
    let tensor_logits = tensor_result.logits.data().to_vec();
    assert_eq!(tensor_logits, vector_result.logits);
    let tensor_labels: Vec<usize> = tensor_result
        .labels
        .data()
        .iter()
        .map(|&value| value.round() as usize)
        .collect();
    assert_eq!(tensor_labels, vector_result.labels);
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn cpu_backend_matches_reference() {
    let anchors = synthetic_batch(7, 3, 5);
    let positives = synthetic_batch(17, 3, 5);

    let tensor_result = info_nce_loss_tensor(
        &tensor_from_batch(&anchors),
        &tensor_from_batch(&positives),
        0.5,
        false,
    )
    .unwrap();

    let vector_result = contrastive::info_nce_loss(&anchors, &positives, 0.5, false).unwrap();

    assert!((tensor_result.loss - vector_result.loss).abs() < 1e-6);
    let tensor_logits = tensor_result.logits.data().to_vec();
    assert_eq!(tensor_logits, vector_result.logits);
    let tensor_labels: Vec<usize> = tensor_result
        .labels
        .data()
        .iter()
        .map(|&value| value.round() as usize)
        .collect();
    assert_eq!(tensor_labels, vector_result.labels);
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_feature_path_matches_reference() {
    let anchors = synthetic_batch(5, 2, 3);
    let positives = synthetic_batch(9, 2, 3);

    let tensor_result = info_nce_loss_tensor(
        &tensor_from_batch(&anchors),
        &tensor_from_batch(&positives),
        0.3,
        true,
    )
    .unwrap();

    let vector_result = contrastive::info_nce_loss(&anchors, &positives, 0.3, true).unwrap();

    assert!((tensor_result.loss - vector_result.loss).abs() < 1e-6);
    let tensor_logits = tensor_result.logits.data().to_vec();
    assert_eq!(tensor_logits, vector_result.logits);
    let tensor_labels: Vec<usize> = tensor_result
        .labels
        .data()
        .iter()
        .map(|&value| value.round() as usize)
        .collect();
    assert_eq!(tensor_labels, vector_result.labels);
}
