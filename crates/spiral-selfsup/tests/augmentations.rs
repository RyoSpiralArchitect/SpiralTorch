use spiral_selfsup::contrastive::{info_nce_loss, info_nce_loss_tensor};
use spiral_selfsup::dataset::{InfoNCEBatchConfig, InfoNCEBatcher, ViewPairSampler};
use st_tensor::Tensor;
use st_vision::datasets::{MultiViewDatasetAdapter, MultiViewFrame};
use st_vision::transforms::{
    CenterCrop, ImageTensor, Normalize, TransformOperation, TransformPipeline,
};

fn build_dataset() -> MultiViewDatasetAdapter {
    let origins = Tensor::from_vec(9, 3, (0..27).map(|v| v as f32 * 0.1).collect()).unwrap();
    let directions =
        Tensor::from_vec(9, 3, (0..27).map(|v| (v % 3) as f32 * 0.5).collect()).unwrap();
    let colors = Tensor::from_vec(9, 3, (0..27).map(|v| v as f32 / 10.0).collect()).unwrap();
    let bounds = Tensor::from_vec(
        9,
        2,
        vec![
            0.0, 1.0, 0.1, 1.1, 0.2, 1.2, 0.3, 1.3, 0.4, 1.4, 0.5, 1.5, 0.6, 1.6, 0.7, 1.7, 0.8,
            1.8,
        ],
    )
    .unwrap();
    let frame = MultiViewFrame::new(origins, directions, colors, bounds).unwrap();
    MultiViewDatasetAdapter::new(vec![frame]).unwrap()
}

fn colors_to_image(frame: &MultiViewFrame, height: usize, width: usize) -> ImageTensor {
    let colors = frame.colors.clone();
    let (rows, cols) = colors.shape();
    assert_eq!(rows, height * width);
    assert_eq!(cols, 3);
    let mut data = vec![0.0f32; rows * cols];
    let source = colors.data();
    for (index, pixel) in source.chunks_exact(cols).enumerate() {
        let y = index / width;
        let x = index % width;
        for channel in 0..cols {
            let offset = ((channel * height) + y) * width + x;
            data[offset] = pixel[channel];
        }
    }
    ImageTensor::new(3, height, width, data).unwrap()
}

fn augmentation_pipeline() -> TransformPipeline {
    let mut pipeline = TransformPipeline::with_seed(3);
    pipeline.add(TransformOperation::CenterCrop(
        CenterCrop::new(2, 2).unwrap(),
    ));
    pipeline.add(TransformOperation::Normalize(
        Normalize::new(vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]).unwrap(),
    ));
    pipeline
}

#[test]
fn augmentation_pipeline_matches_manual_transform() {
    let dataset = build_dataset();
    let pipeline = augmentation_pipeline();
    let mut sampler = ViewPairSampler::new(&dataset, 3, 3, Some(11), Some(&pipeline)).unwrap();
    let pair = sampler.sample_pair().unwrap();

    assert_eq!(pair.anchor.shape(), (3, 2, 2));
    assert_eq!(pair.positive.shape(), (3, 2, 2));

    let mut expected = colors_to_image(&dataset.frames()[0], 3, 3);
    let mut manual = pipeline.clone();
    manual.apply(&mut expected).unwrap();

    assert_eq!(pair.anchor, expected);
    assert_eq!(pair.positive, expected);
}

#[test]
fn info_nce_batcher_produces_tensor_batches() {
    let dataset = build_dataset();
    let pipeline = augmentation_pipeline();
    let config = InfoNCEBatchConfig {
        batch_size: 2,
        image_height: 3,
        image_width: 3,
        temperature: 0.25,
        normalize: true,
        seed: Some(5),
    };
    let mut batcher = InfoNCEBatcher::new(&dataset, Some(&pipeline), config.clone()).unwrap();
    let batch = batcher.sample_batch().unwrap();

    assert_eq!(batch.anchors.shape(), (2, 12));
    assert_eq!(batch.positives.shape(), (2, 12));

    let result = batcher.next_loss().unwrap();
    assert_eq!(result.batch, 2);
    assert_eq!(result.logits.shape(), (2, 2));
    assert_eq!(result.labels.shape(), (2, 1));
    assert!(result.loss.is_finite());

    // Ensure tensor and vector implementations agree for the sampled batch.
    let anchor_rows: Vec<Vec<f32>> = batch
        .anchors
        .data()
        .chunks_exact(12)
        .map(|chunk| chunk.to_vec())
        .collect();
    let positive_rows: Vec<Vec<f32>> = batch
        .positives
        .data()
        .chunks_exact(12)
        .map(|chunk| chunk.to_vec())
        .collect();
    let vector_loss = info_nce_loss(
        &anchor_rows,
        &positive_rows,
        config.temperature,
        config.normalize,
    )
    .unwrap();
    let tensor_loss = info_nce_loss_tensor(
        &batch.anchors,
        &batch.positives,
        config.temperature,
        config.normalize,
    )
    .unwrap();
    assert!((vector_loss.loss - tensor_loss.loss).abs() < 1e-5);
    assert_eq!(tensor_loss.batch, vector_loss.batch);
}

#[test]
fn info_nce_config_from_toml() {
    let data = r#"
        batch_size = 4
        image_height = 3
        image_width = 3
        temperature = 0.5
        normalize = false
        seed = 42
    "#;
    let config = InfoNCEBatchConfig::from_toml_str(data).unwrap();
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.image_height, 3);
    assert_eq!(config.image_width, 3);
    assert!(!config.normalize);
    assert_eq!(config.seed, Some(42));
}
