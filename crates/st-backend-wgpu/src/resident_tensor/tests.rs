use super::*;

#[test]
fn shader_and_portable_addressing_are_checked() {
    let module = naga::front::wgsl::parse_str(&source()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let limits = wgpu::Limits::default();
    let view = NdLayout::contiguous(&[3, 4])
        .unwrap()
        .narrow(0, 2, 1)
        .unwrap();
    assert!(validate_view(&view, 12, &limits).is_ok());
    assert!(matches!(
        validate_view(&view, 4, &limits),
        Err(TensorError::StorageBounds)
    ));
    assert!(NdLayout::contiguous(&[u32::MAX as usize, 2])
        .map_err(TensorError::from)
        .and_then(|layout| validate_view(&layout, 0, &limits))
        .is_err());
    let small = wgpu::Limits {
        max_compute_workgroups_per_dimension: 2,
        ..limits
    };
    assert_eq!(grid(1024, &small).unwrap(), [2, 2, 4]);
    assert!(grid(1025, &small).is_err());
    let large = wgpu::Limits::default();
    let [x, y, count] = grid(u32::MAX as usize, &large).unwrap();
    assert!(u64::from(x) * u64::from(y) >= u64::from(count));
    assert!(u64::from(count - 1) * 256 + 255 <= u64::from(u32::MAX));
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn resident_views_broadcasts_chains_and_failures_on_real_gpu() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("tensor.nd.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let device = TensorDevice::new(runtime).unwrap();
    let data: Vec<_> = (0..24).map(|i| i as f32 / 8. - 1.5).collect();
    let root = device.upload(&[2, 3, 4], &data).unwrap();
    let view = root.permute(&[1, 0, 2]).unwrap().narrow(0, 1, 2).unwrap();
    assert!(view.shares_storage_with(&root));
    let bias = device.upload(&[4], &[0.1, 0.2, -0.3, -0.4]).unwrap();
    let gain = device.upload(&[], &[0.25]).unwrap();
    let mut output = view.clone();
    let mut expected: Vec<_> = (0..view.layout.len())
        .map(|i| data[view.layout.storage_index(i).unwrap()])
        .collect();
    for _ in 0..20 {
        output = output
            .add(&bias)
            .unwrap()
            .mul(&gain)
            .unwrap()
            .gelu()
            .unwrap();
        for (i, v) in expected.iter_mut().enumerate() {
            *v = ElementwiseOp::Gelu
                .apply((*v + [0.1, 0.2, -0.3, -0.4][i % 4]) * 0.25, 0.)
                .unwrap();
        }
    }
    let snapshot = output.snapshot().unwrap();
    drop(output);
    drop(view);
    for (a, b) in snapshot.read().unwrap().iter().zip(&expected) {
        assert!((a - b).abs() <= 1e-6);
    }
    assert_eq!(root.snapshot().unwrap().read().unwrap(), data);
    let empty = device
        .upload(&[1, 3], &[1.; 3])
        .unwrap()
        .broadcast_to(&[0, 3])
        .unwrap();
    assert!(empty
        .relu()
        .unwrap()
        .snapshot()
        .unwrap()
        .read()
        .unwrap()
        .is_empty());
    let scalar = device.upload(&[], &[-0.]).unwrap();
    assert_eq!(
        scalar.snapshot().unwrap().read().unwrap()[0].to_bits(),
        (-0f32).to_bits()
    );
    let invalid = device
        .upload(&[1], &[-f32::MAX])
        .unwrap()
        .mul(&device.upload(&[], &[2.]).unwrap())
        .unwrap();
    for out in [
        invalid.relu().unwrap(),
        invalid.broadcast_to(&[0]).unwrap().relu().unwrap(),
    ] {
        assert!(matches!(
            out.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    assert!(device.upload(&[1], &[f32::NAN]).is_err());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn tensor_snapshots_reuse_only_idle_storage_on_real_gpu() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("tensor.snapshot.cache").unwrap();
    let device = TensorDevice::new(runtime).unwrap();
    let root = device.upload(&[2, 2], &[1., 2., 3., 4.]).unwrap();
    let first = root.snapshot().unwrap();
    let id = first.staging.buffer().global_id();
    assert_eq!(first.read().unwrap(), [1., 2., 3., 4.]);
    let held = root.snapshot().unwrap();
    assert_eq!(held.staging.buffer().global_id(), id);
    let other = device.clone().upload(&[4], &[9.; 4]).unwrap();
    let different = other.snapshot().unwrap();
    assert_ne!(different.staging.buffer().global_id(), id);
    assert_eq!(different.read().unwrap(), [9.; 4]);
    assert_eq!(held.read().unwrap(), [1., 2., 3., 4.]);
    let old = root.permute(&[1, 0]).unwrap().snapshot().unwrap();
    for length in [0, 1, 4, 9, 4, 1, 0] {
        let current = device.upload(&[length], &vec![17.; length]).unwrap();
        drop(current.snapshot().unwrap());
        assert_eq!(
            current.snapshot().unwrap().read().unwrap(),
            vec![17.; length]
        );
    }
    let bad = device
        .upload(&[4], &[-f32::MAX; 4])
        .unwrap()
        .mul(&device.upload(&[], &[2.]).unwrap())
        .unwrap()
        .relu()
        .unwrap();
    let invalid = bad.snapshot().unwrap();
    assert_eq!(root.snapshot().unwrap().read().unwrap(), [1., 2., 3., 4.]);
    assert!(matches!(invalid.read(), Err(TensorError::NonFinite)));
    assert_eq!(root.snapshot().unwrap().read().unwrap(), [1., 2., 3., 4.]);
    drop(device);
    drop(root);
    drop(other);
    drop(bad);
    assert_eq!(old.read().unwrap(), [1., 3., 2., 4.]);
}
