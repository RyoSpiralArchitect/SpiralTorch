use super::*;

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) =
        runtime::ensure_default_runtime_blocking("tensor.concatenate.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    eprintln!("resident concatenate adapter: {:?}", runtime.adapter_info());
    Some(TensorDevice::new(runtime).unwrap())
}

fn read(t: &ResidentTensor) -> Vec<f32> {
    t.snapshot().unwrap().read().unwrap()
}

#[test]
fn shader_validates_without_a_device() {
    let source = shader_source();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn joins_logical_views_and_retains_values_after_inputs_drop() {
    let Some(device) = device() else {
        return;
    };
    let base = device
        .upload(&[2, 3, 4], &(0..24).map(|i| i as f32).collect::<Vec<_>>())
        .unwrap();
    let a = base.permute(&[1, 0, 2]).unwrap().narrow(0, 1, 2).unwrap();
    let b = device
        .upload(&[1, 1, 4], &[-0., -1., -2., -3.])
        .unwrap()
        .broadcast_to(&[2, 1, 4])
        .unwrap();
    let output = ResidentTensor::concatenate(&[&a, &b, &a], 1).unwrap();
    assert_eq!(output.layout.shape(), [2, 5, 4]);
    assert!(output.layout.is_contiguous());
    assert!(!output.shares_storage_with(&a));
    let av = read(&a);
    let bv = read(&b);
    let mut expected = Vec::new();
    for row in 0..2 {
        expected.extend_from_slice(&av[row * 8..row * 8 + 8]);
        expected.extend_from_slice(&bv[row * 4..row * 4 + 4]);
        expected.extend_from_slice(&av[row * 8..row * 8 + 8]);
    }
    drop((base, a, b));
    assert_eq!(
        read(&output)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    let singleton = ResidentTensor::concatenate(&[&output], 0).unwrap();
    assert!(!singleton.shares_storage_with(&output));
    assert_eq!(read(&singleton), expected);
}

#[test]
fn empty_inputs_cannot_erase_inherited_failures() {
    let Some(device) = device() else {
        return;
    };
    let finite = device.upload(&[1, 2], &[f32::MAX, 1.]).unwrap();
    let overflow = finite.mul(&finite).unwrap();
    let cropped = overflow.narrow(1, 1, 1).unwrap();
    let empty = overflow.narrow(0, 0, 0).unwrap();
    let valid = device.upload(&[1, 2], &[1., 2.]).unwrap();
    for result in [
        ResidentTensor::concatenate(&[&empty, &valid], 0).unwrap(),
        ResidentTensor::concatenate(&[&valid, &empty], 0).unwrap(),
        ResidentTensor::concatenate(&[&empty, &empty], 0).unwrap(),
        ResidentTensor::concatenate(&[&cropped, &cropped], 1).unwrap(),
    ] {
        assert!(matches!(
            result.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    let empty = device.upload(&[0, 2], &[]).unwrap();
    assert!(read(&ResidentTensor::concatenate(&[&empty, &empty], 0).unwrap()).is_empty());
    assert_eq!(
        read(&ResidentTensor::concatenate(&[&empty, &valid], 0).unwrap()),
        [1., 2.]
    );
    assert!(ResidentTensor::concatenate(&[], 0).is_err());
    assert!(ResidentTensor::concatenate(&[&valid], 2).is_err());
    assert!(ResidentTensor::concatenate(&[&valid, &empty], 1).is_err());
    let scalar = device.upload(&[], &[1.]).unwrap();
    assert!(ResidentTensor::concatenate(&[&scalar], 0).is_err());
}

#[test]
fn last_axis_join_crosses_workgroups_and_handles_empty_outer_dimensions() {
    let Some(device) = device() else {
        return;
    };
    let base = device
        .upload(
            &[3, 19, 17],
            &(0..969).map(|i| i as f32).collect::<Vec<_>>(),
        )
        .unwrap();
    let a = base.permute(&[1, 0, 2]).unwrap().narrow(0, 1, 17).unwrap();
    let b = device
        .upload(&[1, 1, 5], &[-0., -1., -2., -3., -4.])
        .unwrap()
        .broadcast_to(&[17, 3, 5])
        .unwrap();
    let joined = ResidentTensor::concatenate(&[&a, &b, &a], 2).unwrap();
    assert_eq!(joined.layout().shape(), [17, 3, 39]);
    let av = read(&a);
    let bv = read(&b);
    let mut expected = Vec::new();
    for row in 0..51 {
        expected.extend_from_slice(&av[row * 17..(row + 1) * 17]);
        expected.extend_from_slice(&bv[row * 5..(row + 1) * 5]);
        expected.extend_from_slice(&av[row * 17..(row + 1) * 17]);
    }
    assert!(read(&joined)
        .iter()
        .map(|v| v.to_bits())
        .eq(expected.iter().map(|v| v.to_bits())));

    let a = device.upload(&[0, 3], &[]).unwrap();
    let b = device.upload(&[0, 5], &[]).unwrap();
    let empty = ResidentTensor::concatenate(&[&a, &b], 1).unwrap();
    assert_eq!(empty.layout().shape(), [0, 8]);
    assert!(read(&empty).is_empty());
    let huge = device.upload(&[1, 3], &[f32::MAX; 3]).unwrap();
    let bad_empty = huge.mul(&huge).unwrap().narrow(0, 0, 0).unwrap();
    for inputs in [[&bad_empty, &b], [&b, &bad_empty]] {
        assert!(ResidentTensor::concatenate(&inputs, 1)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .is_err());
    }
}
