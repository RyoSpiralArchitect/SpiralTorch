use super::*;

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("tensor.group.validity").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(TensorDevice::new(runtime).unwrap())
}

#[test]
fn grouped_validity_keeps_values_views_and_original_guards_immutable() {
    let Some(device) = device() else {
        return;
    };
    let input = device.upload(&[2, 3], &[1., -0., 3., 4., 5., 6.]).unwrap();
    let view = input.permute(&[1, 0]).unwrap();
    let healthy = device.upload(&[1], &[2.]).unwrap();
    let grouped = device.guard_together(&[&view, &healthy]).unwrap();
    assert!(grouped[0].shares_storage_with(&view));
    assert_eq!(grouped[0].layout(), view.layout());
    assert!(grouped[0]
        .snapshot()
        .unwrap()
        .read()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .eq([1.0f32, 4., -0., 5., 3., 6.].iter().map(|v| v.to_bits())));
    let huge = device.upload(&[1], &[f32::MAX]).unwrap();
    let bad = huge.mul(&huge).unwrap();
    let failed = device.guard_together(&[&grouped[0], &bad]).unwrap();
    assert!(failed[0].shares_storage_with(&input));
    for tensor in [
        failed[0].clone(),
        failed[0].relu().unwrap(),
        failed[0].contiguous().unwrap(),
        failed[0].narrow(0, 0, 0).unwrap(),
        failed[0].select(0, 1).unwrap(),
        device
            .guard_together(&[&failed[0], &healthy])
            .unwrap()
            .remove(0),
    ] {
        assert!(matches!(
            tensor.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    assert!(matches!(
        failed[0]
            .mean_squared_error(&view)
            .unwrap()
            .prediction_gradient()
            .snapshot()
            .unwrap()
            .read(),
        Err(TensorError::NonFinite)
    ));
    assert!(matches!(
        device
            .snapshot_many(&[&failed[0], &healthy])
            .unwrap()
            .read(),
        Err(TensorError::NonFinite)
    ));
    assert_eq!(
        grouped[0].snapshot().unwrap().read().unwrap(),
        [1., 4., -0., 5., 3., 6.]
    );
    assert_eq!(
        input.snapshot().unwrap().read().unwrap(),
        [1., -0., 3., 4., 5., 6.]
    );
    assert!(device.guard_together(&[]).unwrap().is_empty());
    assert!(device.guard_together(&[&view]).unwrap()[0].shares_storage_with(&view));
    assert!(matches!(
        device.guard_together(&[&failed[0]]).unwrap()[0]
            .snapshot()
            .unwrap()
            .read(),
        Err(TensorError::NonFinite)
    ));
}

#[test]
fn grouped_aliases_pin_capture_storage_and_cannot_be_recycled_as_original_guards() {
    let Some(device) = device() else {
        return;
    };
    let shape = NdLayout::contiguous(&[2]).unwrap();
    let mut original =
        capture::allocate_whole_outputs(&device, [&shape, &shape], wgpu::BufferUsages::empty())
            .unwrap();
    assert!(capture::whole_outputs_exclusively_owned(&mut original));
    let mut aliases = device
        .guard_together(&original.iter().collect::<Vec<_>>())
        .unwrap();
    assert!(!capture::whole_outputs_exclusively_owned(&mut original));
    drop(original);
    assert!(!aliases[0].exclusively_owned());
    assert!(!capture::whole_outputs_exclusively_owned(&mut aliases));
    let mut owned = device.upload(&[1], &[1.]).unwrap();
    assert!(owned.exclusively_owned());
    let healthy = device.upload(&[1], &[1.]).unwrap();
    let alias = device.guard_together(&[&owned, &healthy]).unwrap();
    assert!(!owned.exclusively_owned());
    drop(alias);
    assert!(owned.exclusively_owned());
}

#[test]
fn grouped_validity_rejects_foreign_device_even_for_a_single_tensor() {
    let Some(device) = device() else {
        return;
    };
    let other = pollster::block_on(WgpuRuntime::request_headless("tensor.group.foreign")).unwrap();
    assert!(!device
        .runtime()
        .context()
        .shares_handles_with(other.context()));
    let other = TensorDevice::new(other).unwrap();
    let foreign = other.upload(&[1], &[1.]).unwrap();
    let local = device.upload(&[1], &[1.]).unwrap();
    assert!(matches!(
        device.guard_together(&[&foreign]),
        Err(TensorError::DeviceMismatch)
    ));
    assert!(matches!(
        device.guard_together(&[&local, &foreign]),
        Err(TensorError::DeviceMismatch)
    ));
    let mut encoder = device
        .runtime()
        .context()
        .device()
        .create_command_encoder(&Default::default());
    assert!(matches!(
        device.guard_together_into(&[&foreign], &mut encoder),
        Err(TensorError::DeviceMismatch)
    ));
    assert!(matches!(
        device.guard_together_into(&[&local, &foreign], &mut encoder),
        Err(TensorError::DeviceMismatch)
    ));
    assert_eq!(local.snapshot().unwrap().read().unwrap(), [1.]);
}

#[test]
fn encoded_groups_capture_before_and_after_late_failure_without_mutating_earlier_aliases() {
    let Some(device) = device() else {
        return;
    };
    let original = device.upload(&[2], &[-0., 2.]).unwrap();
    let healthy = device.upload(&[1], &[3.]).unwrap();
    let huge = device.upload(&[1], &[f32::MAX]).unwrap();
    let bad = huge.mul(&huge).unwrap();
    let context = device.runtime().context();
    // Only this test's producer guard is mutable, to model a kernel detecting
    // failure between the two captures. The uploaded tensor remains immutable.
    let produced = ResidentTensor {
        validation: Some(Shared::new(
            runtime::upload_slice(
                context.device(),
                "tensor.group.test_producer_guard",
                &[0u32],
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            )
            .unwrap(),
        )),
        ..original.clone()
    };
    let mut encoder = context.device().create_command_encoder(&Default::default());
    assert!(device
        .guard_together_into(&[], &mut encoder)
        .unwrap()
        .is_empty());
    assert!(device
        .guard_together_into(&[&healthy], &mut encoder)
        .unwrap()[0]
        .shares_storage_with(&healthy));
    let before = device
        .guard_together_into(&[&produced, &healthy], &mut encoder)
        .unwrap();
    encoder.copy_buffer_to_buffer(bad.flags(), 0, produced.flags(), 0, 4);
    let after = device
        .guard_together_into(&[&before[0], &produced, &healthy], &mut encoder)
        .unwrap();
    context.queue().submit(Some(encoder.finish()));
    for tensor in [&original, &before[0]] {
        let values = tensor.snapshot().unwrap().read().unwrap();
        assert_eq!(values[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(values[1], 2.);
    }
    assert_eq!(before[1].snapshot().unwrap().read().unwrap(), [3.]);
    for tensor in after {
        assert!(matches!(
            tensor.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    assert_eq!(healthy.snapshot().unwrap().read().unwrap(), [3.]);
}
