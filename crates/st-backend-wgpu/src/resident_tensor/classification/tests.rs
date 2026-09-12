use super::*;
use st_kernel_contracts::classification::ClassReduction;

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("tensor.ce.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(TensorDevice::new(runtime).unwrap())
}
fn read(tensor: &ResidentTensor) -> Vec<f32> {
    tensor.snapshot().unwrap().read().unwrap()
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (a - b).abs() <= 2e-6 + 2e-5 * b.abs(),
            "{a} != {b}"
        );
    }
}
#[test]
fn shader_parses_and_validates_without_a_runtime() {
    let module =
        naga::front::wgsl::parse_str(include_str!("../shaders/cross_entropy.wgsl")).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}
#[test]
fn reductions_smoothed_ignore_views_and_small_tails_stay_resident() {
    let Some(device) = device() else { return };
    let x = device
        .upload(&[3, 2, 4], &[0.; 24])
        .unwrap()
        .permute(&[1, 0, 2])
        .unwrap();
    let y = device
        .upload(&[2, 3], &[0., 1., -100., 2., -100., 3.])
        .unwrap();
    for reduction in [
        ClassReduction::None,
        ClassReduction::Sum,
        ClassReduction::Mean,
    ] {
        let pair = x
            .cross_entropy_with_logits(&y, CrossEntropySpec::new(reduction, -100, 0.2).unwrap())
            .unwrap();
        let d = if reduction == ClassReduction::Mean {
            4.
        } else {
            1.
        };
        let rows: Vec<_> = [1., 1., 0., 1., 0., 1.]
            .iter()
            .map(|v| v * 4f32.ln())
            .collect();
        close(
            &read(pair.value()),
            &if reduction == ClassReduction::None {
                rows
            } else {
                vec![4. * 4f32.ln() / d]
            },
        );
        let expected: Vec<_> = [0, 1, -100, 2, -100, 3]
            .iter()
            .flat_map(|&label| {
                (0..4).map(move |c| {
                    if label == -100 {
                        0.
                    } else {
                        (0.25 - 0.05 - if c == label { 0.8 } else { 0. }) / d
                    }
                })
            })
            .collect();
        close(&read(pair.prediction_gradient()), &expected);
    }
    let x = device.upload(&[1, 2], &[80., 0.]).unwrap();
    let pair = x
        .cross_entropy_with_logits(
            &device.upload(&[1], &[0.]).unwrap(),
            CrossEntropySpec::new(ClassReduction::Mean, -100, 0.).unwrap(),
        )
        .unwrap();
    let tail = (-80f64).exp() as f32;
    for (a, b) in read(pair.value())
        .into_iter()
        .chain(read(pair.prediction_gradient()))
        .zip([tail, -tail, tail])
    {
        assert!((a / b - 1.).abs() < 2e-5, "tiny tail {a} != {b}");
    }
}
#[test]
fn wide_normalized_losses_tiny_smoothing_and_whole_loss_guards() {
    let Some(device) = device() else { return };
    let spec = |r, i, s| CrossEntropySpec::new(r, i, s).unwrap();
    let x = device
        .upload(&[2, 2], &[f32::MAX, -f32::MAX, 0., 0.])
        .unwrap();
    let y = device.upload(&[2], &[1., 0.]).unwrap();
    let pair = x
        .cross_entropy_with_logits(&y, spec(ClassReduction::Mean, -100, 0.))
        .unwrap();
    close(
        &pair
            .value()
            .snapshot()
            .unwrap()
            .read()
            .expect("wide mean value"),
        &[f32::MAX],
    );
    close(&read(pair.prediction_gradient()), &[0.5, -0.5, -0.25, 0.25]);
    let overflow = x
        .cross_entropy_with_logits(&y, spec(ClassReduction::Sum, -100, 0.))
        .unwrap();
    let x = device.upload(&[1, 2], &[f32::MAX, -f32::MAX]).unwrap();
    let y = device.upload(&[1], &[0.]).unwrap();
    let tiny = x
        .cross_entropy_with_logits(&y, spec(ClassReduction::Mean, -100, 1e-40))
        .unwrap();
    close(
        &tiny
            .value()
            .snapshot()
            .unwrap()
            .read()
            .expect("tiny smoothing value"),
        &[(f64::from(f32::MAX) * 1e-40) as f32],
    );
    let mut failures = vec![overflow];
    for (label, ignore) in [
        (0.5, -100),
        (-2., -100),
        (2., -100),
        (i64::MAX as f32, i64::MAX),
        (16777216., 16777217),
        (-100., -100),
    ] {
        let target = device.upload(&[1], &[label]).unwrap();
        failures.push(
            x.cross_entropy_with_logits(&target, spec(ClassReduction::Mean, ignore, 0.))
                .unwrap(),
        );
    }
    let ignored = x
        .cross_entropy_with_logits(
            &device.upload(&[1], &[i64::MIN as f32]).unwrap(),
            spec(ClassReduction::Sum, i64::MIN, 0.),
        )
        .unwrap();
    close(&read(ignored.value()), &[0.]);
    close(&read(ignored.prediction_gradient()), &[0., 0.]);
    let retained = x
        .cross_entropy_with_logits(&y, spec(ClassReduction::Sum, -100, 1.))
        .unwrap();
    drop((device, x, y));
    for pair in failures {
        for t in [pair.value(), pair.prediction_gradient()] {
            assert!(t.snapshot().unwrap().read().is_err());
        }
    }
    close(
        &retained
            .value()
            .snapshot()
            .unwrap()
            .read()
            .expect("retained uniform value"),
        &[f32::MAX],
    );
}
