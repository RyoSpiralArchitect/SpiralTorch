use super::*;

#[test]
fn packed_capture_shader_validates() {
    let source = substitute_ops(include_str!("../../shaders/graph_capture.wgsl").replace(
        "CHECKED_ELEMENTWISE",
        include_str!("../../shaders/checked_elementwise.wgsl"),
    ));
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn packed_captures_own_values_and_whole_guards_on_real_gpu() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("tensor.capture.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let device = TensorDevice::new(runtime).unwrap();
    let gpu = device.runtime().context().device();
    let queue = device.runtime().context().queue();
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let expected: Vec<_> = (0..771).map(|i| (i % 17) as f32 * 0.25 - 2.).collect();
    let scalar = runtime::upload_slice(gpu, "capture.scalar", &[-0f32], usage).unwrap();
    let values = runtime::upload_slice(gpu, "capture.matrix", &expected, usage).unwrap();
    let empty = runtime::upload_slice(gpu, "capture.empty", &[f32::NAN], usage).unwrap();
    let guards = runtime::upload_slice(gpu, "capture.upstream", &[0u32; 5], usage).unwrap();
    let scalar_layout = NdLayout::contiguous(&[]).unwrap();
    let matrix = NdLayout::contiguous(&[3, 257]).unwrap();
    let empty_layout = NdLayout::contiguous(&[0, 7]).unwrap();
    assert!(PreparedCapture::new(&device, &[], &guards).is_err());
    assert!(PreparedCapture::new(
        &device,
        &[(&matrix.permute(&[1, 0]).unwrap(), &values)],
        &guards
    )
    .is_err());
    assert!(PreparedCapture::new(
        &device,
        &[(&matrix.narrow(0, 1, 2).unwrap(), &values)],
        &guards
    )
    .is_err());
    assert!(PreparedCapture::new(&device, &[(&matrix, &scalar)], &guards).is_err());
    let plan = PreparedCapture::new(
        &device,
        &[
            (&scalar_layout, &scalar),
            (&matrix, &values),
            (&empty_layout, &empty),
        ],
        &guards,
    )
    .unwrap();
    let capture = || {
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let result = plan.encode(&mut encoder).unwrap();
        queue.submit(Some(encoder.finish()));
        result
    };
    let original = capture();
    queue.write_buffer(&guards, 16, bytemuck::cast_slice(&[1u32]));
    let inherited = capture();
    queue.write_buffer(&guards, 16, bytemuck::cast_slice(&[0u32]));
    queue.write_buffer(&values, 770 * 4, bytemuck::cast_slice(&[f32::INFINITY]));
    let invalid = capture();
    queue.write_buffer(&values, 770 * 4, bytemuck::cast_slice(&[expected[770]]));
    let recovered = capture();
    assert!(Shared::ptr_eq(
        &original[0].storage.flags,
        &original[2].storage.flags
    ));
    assert!(!Shared::ptr_eq(
        &original[0].storage.flags,
        &recovered[0].storage.flags
    ));
    assert!(!original[0].shares_storage_with(&original[1]));
    assert!(!original[1].shares_storage_with(&recovered[1]));
    drop(plan);
    drop(values);
    drop(scalar);
    drop(empty);
    drop(guards);
    for result in [original, recovered] {
        assert_eq!(
            result[0].snapshot().unwrap().read().unwrap()[0].to_bits(),
            (-0f32).to_bits()
        );
        assert_eq!(result[1].snapshot().unwrap().read().unwrap(), expected);
        assert!(result[2].snapshot().unwrap().read().unwrap().is_empty());
    }
    for result in [inherited, invalid] {
        for tensor in result {
            assert!(matches!(
                tensor.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
        }
    }
}
