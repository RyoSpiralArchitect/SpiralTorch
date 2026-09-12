use super::*;

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("tensor.loss.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(TensorDevice::new(runtime).unwrap())
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (a - b).abs() <= 2e-5 + 2e-4 * b.abs(),
            "{a} != {b}"
        );
    }
}

#[test]
fn mean_mse_reuses_training_kernels_for_views_empty_and_many_partials() {
    let Some(device) = device() else { return };
    for shape in [
        vec![],
        vec![2, 0, 3],
        vec![1],
        vec![257],
        vec![65537],
        vec![2, 3, 4, 5],
    ] {
        let n = shape.iter().product();
        let a: Vec<_> = (0..n).map(|i| ((i % 23) as f32 - 11.) / 16.).collect();
        let b: Vec<_> = (0..n).map(|i| ((i % 13) as f32 - 6.) / 32.).collect();
        let prediction = device.upload(&shape, &a).unwrap();
        let target = device.upload(&shape, &b).unwrap();
        let pair = prediction.mean_squared_error(&target).unwrap();
        assert_eq!(pair.value().layout().shape(), &[1, 1]);
        assert_eq!(pair.prediction_gradient().layout().shape(), shape);
        assert!(Shared::ptr_eq(
            &pair.value.storage.flags,
            &pair.prediction_gradient.storage.flags
        ));
        let diff: Vec<_> = a.iter().zip(&b).map(|(a, b)| a - b).collect();
        let expected = if n == 0 {
            0.
        } else {
            diff.iter().map(|v| v * v / n as f32).sum()
        };
        close(
            &pair.value().snapshot().unwrap().read().unwrap(),
            &[expected],
        );
        close(
            &pair
                .prediction_gradient()
                .snapshot()
                .unwrap()
                .read()
                .unwrap(),
            &diff.iter().map(|v| v * (2. / n as f32)).collect::<Vec<_>>(),
        );
    }
    let input = device
        .upload(
            &[3, 4, 5],
            &(0..60).map(|v| v as f32 / 32.).collect::<Vec<_>>(),
        )
        .unwrap()
        .narrow(1, 1, 2)
        .unwrap()
        .permute(&[2, 0, 1])
        .unwrap();
    let target = device
        .upload(&[], &[0.25])
        .unwrap()
        .broadcast_to(input.layout().shape())
        .unwrap();
    let a = input.snapshot().unwrap().read().unwrap();
    let pair = input.mean_squared_error(&target).unwrap();
    close(
        &pair.value().snapshot().unwrap().read().unwrap(),
        &[a.iter()
            .map(|v| (v - 0.25) * (v - 0.25) / a.len() as f32)
            .sum()],
    );
    close(
        &pair
            .prediction_gradient()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        &a.iter()
            .map(|v| (v - 0.25) * (2. / a.len() as f32))
            .collect::<Vec<_>>(),
    );
    assert!(device.0.mse.get().is_some());
}

#[test]
fn whole_loss_guards_retain_failures_without_poisoning_later_results() {
    let Some(device) = device() else { return };
    let prediction = device.upload(&[2, 2], &[1.; 4]).unwrap();
    let wrong = device.upload(&[4], &[0.; 4]).unwrap();
    assert!(matches!(
        prediction.mean_squared_error(&wrong),
        Err(TensorError::LossShape)
    ));
    assert!(device.0.mse.get().is_none());
    let target = device.upload(&[2, 2], &[0.; 4]).unwrap();
    let huge = device.upload(&[2, 2], &[1e20; 4]).unwrap();
    let overflow = huge.mean_squared_error(&target).unwrap();
    let maximum = device.upload(&[2, 2], &[f32::MAX; 4]).unwrap();
    let poisoned = maximum.add(&maximum).unwrap().mul(&target).unwrap();
    let bad_prediction = poisoned.mean_squared_error(&target).unwrap();
    let bad_target = prediction.mean_squared_error(&poisoned).unwrap();
    let valid = prediction.mean_squared_error(&target).unwrap();
    let held = valid.clone();
    for _ in 0..16 {
        drop(prediction.mean_squared_error(&target).unwrap());
    }
    drop((device, prediction, target, huge, maximum, poisoned, valid));
    for pair in [overflow, bad_prediction, bad_target] {
        for tensor in [pair.value(), pair.prediction_gradient()] {
            assert!(matches!(
                tensor.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
        }
    }
    close(&held.value().snapshot().unwrap().read().unwrap(), &[1.]);
    close(
        &held
            .prediction_gradient()
            .snapshot()
            .unwrap()
            .read()
            .unwrap(),
        &[0.5; 4],
    );
}
