use super::*;

#[test]
fn normalization_rejects_invalid_statistics_channels_and_intermediates_atomically() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(Normalize::new(vec![bad], vec![1.]).is_err());
        assert!(Normalize::new(vec![0.], vec![bad]).is_err());
    }
    for bad in [0., -0., -1.] {
        assert!(Normalize::new(vec![0.], vec![bad]).is_err());
    }
    let mut image = ImageTensor::new(3, 1, 1, vec![1., 2., 3.]).unwrap();
    let before = image.clone();
    assert!(Normalize::new(vec![0.; 2], vec![1.; 2])
        .unwrap()
        .apply(&mut image)
        .is_err());
    assert_eq!(image, before);
    for (values, mean, std) in [
        (vec![0., f32::MAX], 0., 0.5),
        (vec![0., f32::MAX], -f32::MAX, f32::MAX),
        (vec![1., f32::INFINITY], 0., 1.),
    ] {
        let mut image = ImageTensor::new(1, 1, 2, values).unwrap();
        let before = image.clone();
        assert!(Normalize::new(vec![mean], vec![std])
            .unwrap()
            .apply(&mut image)
            .is_err());
        assert_eq!(image, before);
    }
    Normalize::new(vec![0., 1., 2.], vec![1., 2., 4.])
        .unwrap()
        .apply(&mut image)
        .unwrap();
    assert_eq!(image.as_slice(), &[1., 0.5, 0.25]);
}

#[test]
fn mixed_pipeline_rolls_back_image_and_rng_on_late_error() {
    let mut pipeline = TransformPipeline::with_seed(9);
    pipeline
        .add(TransformOperation::RandomHorizontalFlip(
            RandomHorizontalFlip::new(0.5).unwrap(),
        ))
        .add(TransformOperation::Normalize(
            Normalize::new(vec![0.], vec![0.5]).unwrap(),
        ))
        .add(TransformOperation::CenterCrop(
            CenterCrop::new(3, 3).unwrap(),
        ));
    let mut image = ImageTensor::new(1, 2, 2, vec![1., 2., 3., 4.]).unwrap();
    let before = image.clone();
    let rng = pipeline.rng.clone().gen::<u64>();
    assert!(pipeline.apply(&mut image).is_err());
    assert_eq!(image, before);
    assert_eq!(pipeline.rng.clone().gen::<u64>(), rng);
}

#[test]
fn dataloader_failed_later_sample_preserves_cursor_and_flip_rng() {
    let images = [vec![1.; 9], vec![f32::MAX; 9]];
    let dataset = TensorVisionDataset::from_samples(
        dataset_catalog()[0].clone(),
        images
            .into_iter()
            .map(|values| DatasetSample::new(ImageTensor::new(1, 3, 3, values).unwrap()))
            .collect(),
    )
    .unwrap();
    let mut pipeline = TransformPipeline::with_seed(51);
    pipeline
        .add(TransformOperation::RandomHorizontalFlip(
            RandomHorizontalFlip::new(0.5).unwrap(),
        ))
        .add(TransformOperation::Normalize(
            Normalize::new(vec![0.], vec![0.5]).unwrap(),
        ));
    let rng = pipeline.rng.clone().gen::<u64>();
    let mut loader = DataLoader::new(Arc::new(dataset), usize::MAX, Some(3))
        .unwrap()
        .with_pipeline(pipeline);
    for _ in 0..2 {
        assert!(loader.next_batch().is_err());
        assert_eq!(loader.position, 0);
        assert_eq!(
            loader.pipeline.as_ref().unwrap().rng.clone().gen::<u64>(),
            rng
        );
    }
    loader.pipeline = None;
    assert_eq!(loader.next_batch().unwrap().unwrap().len(), 2);
    assert!(loader.next_batch().unwrap().is_none());
}
