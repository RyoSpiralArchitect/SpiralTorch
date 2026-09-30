use super::*;

#[test]
fn global_pool_keeps_channels_and_uses_exact_loss_cotangent() {
    let mut pool = GlobalAveragePool2d::new(2, (2, 3)).unwrap();
    let input = Tensor::from_fn(2, 12, |r, c| (r * 12 + c) as f32).unwrap();
    let expected = [2.5, 8.5, 14.5, 20.5];
    for layout in [st_tensor::Layout::RowMajor, st_tensor::Layout::ColMajor] {
        let input = input.to_layout(layout).unwrap();
        let output = pool.forward(&input).unwrap();
        for (a, b) in output.data().iter().zip(expected) {
            assert!((a - b).abs() < 2e-6);
        }
        let seed = Tensor::from_vec(2, 2, vec![6., -12., 18., 0.])
            .unwrap()
            .to_layout(layout)
            .unwrap();
        let dx = pool.backward(&input, &seed).unwrap();
        for (actual, expected) in dx.data().chunks_exact(pool.area).zip([1., -2., 3., 0.]) {
            assert!(actual.iter().all(|&x| x == expected));
        }
    }
    assert!(pool.forward(&Tensor::zeros(2, 11).unwrap()).is_err());
    assert!(pool
        .backward(&input, &Tensor::zeros(1, 2).unwrap())
        .is_err());
    assert!(GlobalAveragePool2d::new(0, (2, 3)).is_err());
    assert!(GlobalAveragePool2d::new(2, (usize::MAX, 2)).is_err());
    let invalid = Tensor::from_vec(1, 12, vec![f32::NAN; 12]).unwrap();
    assert!(pool.forward(&invalid).is_err());
    assert!(pool
        .backward(&invalid, &Tensor::zeros(1, 2).unwrap())
        .is_err());
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
fn global_pool_resident_preserves_guards_without_unused_filter_overflow() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("global_pool.test").unwrap();
    let device = TensorDevice::new(runtime).unwrap();
    let mut pool = GlobalAveragePool2d::new(2, (2, 3)).unwrap();
    let input = Tensor::from_fn(3, 12, |r, c| (r * 12 + c) as f32 * 0.1).unwrap();
    let resident = device
        .upload(&[3, 2, 2, 3], input.data())
        .unwrap()
        .narrow(0, 1, 2)
        .unwrap();
    let host = Tensor::from_vec(2, 12, input.data()[12..].to_vec()).unwrap();
    let plan = pool.compile_resident(device.clone()).unwrap();
    let actual = plan
        .forward(&resident)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    for (a, b) in actual.iter().zip(pool.forward(&host).unwrap().data()) {
        assert!((a - b).abs() < 1e-5);
    }
    let seed = Tensor::from_vec(2, 2, vec![0.5, -0.1, 0.2, 0.8]).unwrap();
    let actual = plan
        .backward(&resident, &device.upload(&[2, 2], seed.data()).unwrap())
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap();
    for (a, b) in actual
        .iter()
        .zip(pool.backward(&host, &seed).unwrap().data())
    {
        assert!((a - b).abs() < 1e-6);
    }

    let pool = GlobalAveragePool2d::new(1, (1, 2))
        .unwrap()
        .compile_resident(device.clone())
        .unwrap();
    let huge = device.upload(&[2, 1, 1, 2], &[f32::MAX; 4]).unwrap();
    let seed = device.upload(&[2, 1], &[1e10; 2]).unwrap();
    assert_eq!(
        pool.forward(&huge)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        vec![f32::MAX; 2]
    );
    assert_eq!(
        pool.backward(&huge, &seed)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        vec![5e9; 4]
    );
    let invalid = device
        .upload(&[2, 1, 1, 2], &[-f32::MAX; 4])
        .unwrap()
        .mul(&device.upload(&[1], &[2.0]).unwrap())
        .unwrap()
        .relu()
        .unwrap();
    assert!(pool
        .forward(&invalid)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .is_err());
    assert!(pool
        .backward(&invalid, &seed)
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .is_err());
    assert_eq!(
        pool.backward(&huge, &seed)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        vec![5e9; 4]
    );
}
