use super::*;

#[test]
fn capture_retention_accounts_for_all_values_padding_and_shared_guard() {
    assert_eq!(retention_limit([0, 1, 257].into_iter()), 4);
    let quarter = MAX_CAPTURE_BYTES as usize / 16;
    assert_eq!(retention_limit([quarter - 1].into_iter()), 4);
    assert_eq!(retention_limit([quarter].into_iter()), 3);
    assert_eq!(retention_limit([quarter * 4 - 1].into_iter()), 1);
    assert_eq!(retention_limit([quarter * 4].into_iter()), 0);
    assert_eq!(retention_limit([usize::MAX, usize::MAX].into_iter()), 0);
}

#[cfg(not(target_arch = "wasm32"))]
fn test_device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) =
        runtime::ensure_default_runtime_blocking("tensor.capture.reuse.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(TensorDevice::new(runtime).unwrap())
}

#[cfg(not(target_arch = "wasm32"))]
fn reusable_capture(plan: &mut PreparedCapture) -> Vec<ResidentTensor> {
    let context = plan.device.runtime().context().clone();
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let result = plan.encode_reusing(&mut encoder).unwrap();
    context.queue().submit(Some(encoder.finish()));
    result
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn capture_reuse_pins_whole_versions_and_resets_failed_guards() {
    let Some(device) = test_device() else {
        return;
    };
    let context = device.runtime().context();
    let gpu = context.device();
    let queue = context.queue();
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let expected: Vec<_> = (0..771).map(|i| (i % 17) as f32 * 0.25 - 2.).collect();
    let values = runtime::upload_slice(gpu, "capture.reuse.matrix", &expected, usage).unwrap();
    let scalar = runtime::upload_slice(gpu, "capture.reuse.scalar", &[-0f32], usage).unwrap();
    let guards = runtime::upload_slice(gpu, "capture.reuse.guards", &[0u32; 5], usage).unwrap();
    let matrix = NdLayout::contiguous(&[3, 257]).unwrap();
    let scalar_layout = NdLayout::contiguous(&[]).unwrap();
    let empty_layout = NdLayout::contiguous(&[0, 7]).unwrap();
    let mut plan = PreparedCapture::new(
        &device,
        &[
            (&matrix, &values),
            (&scalar_layout, &scalar),
            (&empty_layout, &scalar),
        ],
        &guards,
    )
    .unwrap();
    let original = reusable_capture(&mut plan);
    let snapshot = original[0].snapshot().unwrap();
    // Even a view of an unrelated member protects the common whole-VJP guard.
    let scalar_view = original[1].reshape(&[1]).unwrap();
    drop(original);
    queue.write_buffer(&guards, 16, bytemuck::cast_slice(&[1u32]));
    let bad = reusable_capture(&mut plan);
    let bad_reads: Vec<_> = bad.iter().map(|t| t.snapshot().unwrap()).collect();
    let bad_consumer = bad[0]
        .mul(&device.upload(&[], &[0.]).unwrap())
        .unwrap()
        .snapshot()
        .unwrap();
    drop(bad);
    queue.write_buffer(&guards, 16, bytemuck::cast_slice(&[0u32]));
    let recovered = reusable_capture(&mut plan);
    assert_eq!(
        plan.stats,
        CaptureStats {
            allocations: 2,
            reuses: 1
        }
    );
    assert_eq!(recovered[0].snapshot().unwrap().read().unwrap(), expected);
    assert!(recovered[2].snapshot().unwrap().read().unwrap().is_empty());
    drop(recovered);
    // Last-element nonfinite detection must also re-poison every member.
    queue.write_buffer(&values, 770 * 4, bytemuck::cast_slice(&[f32::INFINITY]));
    let nonfinite = reusable_capture(&mut plan);
    let nonfinite_reads: Vec<_> = nonfinite.iter().map(|t| t.snapshot().unwrap()).collect();
    drop(nonfinite);
    queue.write_buffer(&values, 770 * 4, bytemuck::cast_slice(&[expected[770]]));
    let last = reusable_capture(&mut plan);
    assert_eq!(
        plan.stats,
        CaptureStats {
            allocations: 2,
            reuses: 3
        }
    );
    drop(plan);
    assert_eq!(snapshot.read().unwrap(), expected);
    assert_eq!(
        scalar_view.snapshot().unwrap().read().unwrap()[0].to_bits(),
        (-0f32).to_bits()
    );
    assert_eq!(last[0].snapshot().unwrap().read().unwrap(), expected);
    for read in bad_reads
        .into_iter()
        .chain(nonfinite_reads)
        .chain([bad_consumer])
    {
        assert!(matches!(read.read(), Err(TensorError::NonFinite)));
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn capture_pool_spills_busy_versions_and_respects_weak_owners() {
    let Some(device) = test_device() else {
        return;
    };
    let context = device.runtime().context();
    let gpu = context.device();
    let queue = context.queue();
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let values = runtime::upload_slice(gpu, "capture.pool.values", &[1f32; 257], usage).unwrap();
    let guards = runtime::upload_slice(gpu, "capture.pool.guards", &[0u32], usage).unwrap();
    let layout = NdLayout::contiguous(&[257]).unwrap();
    let mut plan = PreparedCapture::new(&device, &[(&layout, &values)], &guards).unwrap();
    let mut retained = Vec::new();
    for i in 0..6 {
        queue.write_buffer(&values, 0, bytemuck::cast_slice(&[i as f32; 257]));
        retained.push(reusable_capture(&mut plan));
    }
    assert_eq!(plan.outputs.len(), MAX_CAPTURE_BATCHES);
    assert_eq!(
        plan.stats,
        CaptureStats {
            allocations: 6,
            reuses: 0
        }
    );
    let weak = Shared::downgrade(&retained[0][0].storage);
    drop(retained.remove(0));
    let spill = reusable_capture(&mut plan);
    assert_eq!(plan.stats.allocations, 7);
    assert_eq!(weak.upgrade().unwrap().values.size(), 257 * 4);
    drop(weak);
    let weak_guard = Shared::downgrade(&plan.outputs[0].outputs[0].storage.flags);
    drop(reusable_capture(&mut plan));
    assert_eq!(plan.stats.allocations, 8);
    drop(weak_guard);
    let reused = reusable_capture(&mut plan);
    assert_eq!(plan.stats.reuses, 1);
    drop(plan);
    for (i, batch) in retained.iter().enumerate() {
        assert_eq!(
            batch[0].snapshot().unwrap().read().unwrap(),
            vec![(i + 1) as f32; 257]
        );
    }
    for batch in [spill, reused] {
        assert_eq!(
            batch[0].snapshot().unwrap().read().unwrap(),
            vec![5f32; 257]
        );
    }
    // Exercise the no-retention branch without reserving an oversized test buffer.
    let mut unpooled = PreparedCapture::new(&device, &[(&layout, &values)], &guards).unwrap();
    unpooled.retention_limit = 0;
    for _ in 0..8 {
        drop(reusable_capture(&mut unpooled));
    }
    assert!(unpooled.outputs.is_empty());
    assert_eq!(
        unpooled.stats,
        CaptureStats {
            allocations: 8,
            reuses: 0
        }
    );
}

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
    let mut original = capture();
    assert!(
        original
            .iter_mut()
            .all(|tensor| !tensor.exclusively_owned()),
        "independent values still share one guard, so they cannot be recycled independently"
    );
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
    for mut result in [original, recovered] {
        assert_eq!(
            result[0].snapshot().unwrap().read().unwrap()[0].to_bits(),
            (-0f32).to_bits()
        );
        assert_eq!(result[1].snapshot().unwrap().read().unwrap(), expected);
        assert!(result[2].snapshot().unwrap().read().unwrap().is_empty());
        let mut singleton = result.remove(0);
        drop(result);
        assert!(singleton.exclusively_owned());
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
