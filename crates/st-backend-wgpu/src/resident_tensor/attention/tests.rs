use super::*;
use st_kernel_contracts::attention::{attention_reference, AttentionError};

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("tensor.attention.tests").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    eprintln!("resident attention adapter: {:?}", runtime.adapter_info());
    Some(TensorDevice::new(runtime).unwrap())
}

fn read(tensor: &ResidentTensor) -> Vec<f32> {
    tensor.snapshot().unwrap().read().unwrap()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 3e-6 + 3e-5 * b.abs(),
            "[{i}] {a} != {b}"
        );
    }
}

fn data(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (i as f32 * 0.17 + phase).sin() * 0.5)
        .collect()
}

fn merge_heads(shape: [usize; 4], values: &[f32]) -> Vec<f32> {
    let layout = NdLayout::contiguous(&shape)
        .unwrap()
        .permute(&[0, 2, 1, 3])
        .unwrap();
    (0..layout.len())
        .map(|i| values[layout.storage_index(i).unwrap()])
        .collect()
}

#[test]
fn shader_parses_and_validates_without_a_runtime() {
    let module = naga::front::wgsl::parse_str(SHADER_SOURCE).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(std::mem::size_of::<View>(), 32);
    assert_eq!(std::mem::size_of::<Params>(), 208);
    let params = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some("Params"))
        .unwrap()
        .1;
    let naga::TypeInner::Struct { members, span } = &params.inner else {
        panic!("Params must be a struct")
    };
    assert_eq!(*span as usize, std::mem::size_of::<Params>());
    assert_eq!(
        members
            .iter()
            .find(|m| m.name.as_deref() == Some("query"))
            .unwrap()
            .offset as usize,
        std::mem::offset_of!(Params, query)
    );
    assert_eq!(
        members
            .iter()
            .find(|m| m.name.as_deref() == Some("pair_bias"))
            .unwrap()
            .offset as usize,
        std::mem::offset_of!(Params, pair_bias)
    );
}

#[test]
fn descriptors_preserve_offsets_strides_broadcasts_and_storage_bounds() {
    let limits = wgpu::Limits::default();
    let packed = NdLayout::contiguous(&[2, 5, 3, 2, 7]).unwrap();
    let mut layouts: Vec<_> = (0..3)
        .map(|i| {
            packed
                .select(2, i)
                .unwrap()
                .permute(&[0, 2, 1, 3])
                .unwrap()
                .narrow(2, 1, 3)
                .unwrap()
        })
        .collect();
    layouts.push(
        NdLayout::contiguous(&[2, 2, 7, 5])
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap(),
    );
    layouts.push(
        NdLayout::contiguous(&[7])
            .unwrap()
            .narrow(0, 1, 5)
            .unwrap()
            .broadcast_to(&[2, 2, 5])
            .unwrap(),
    );
    layouts.push(
        NdLayout::contiguous(&[5, 3])
            .unwrap()
            .permute(&[1, 0])
            .unwrap()
            .broadcast_to(&[2, 2, 3, 5])
            .unwrap(),
    );
    for layout in layouts {
        let required = layout.required_storage_len().unwrap();
        let view = view_descriptor(&layout, required, &limits).unwrap();
        for logical in 0..layout.len() {
            let mut left = logical;
            let mut coordinates = vec![0; layout.rank()];
            for axis in (0..layout.rank()).rev() {
                coordinates[axis] = left % layout.shape()[axis];
                left /= layout.shape()[axis];
            }
            if layout.rank() == 3 {
                coordinates.insert(2, 0);
            }
            let address = view.offset as usize
                + coordinates
                    .iter()
                    .zip(view.strides)
                    .map(|(i, stride)| i * stride as usize)
                    .sum::<usize>();
            assert_eq!(Some(address), layout.storage_index(logical));
        }
        assert!(matches!(
            view_descriptor(&layout, required - 1, &limits),
            Err(TensorError::StorageBounds)
        ));
    }
    #[cfg(target_pointer_width = "64")]
    {
        let overflow = NdLayout::contiguous(&[1, 2, 1, u32::MAX as usize])
            .unwrap()
            .narrow(3, 0, 1)
            .unwrap();
        assert!(matches!(
            view_descriptor(&overflow, usize::MAX, &limits),
            Err(TensorError::Limit("layout address"))
        ));
    }
}

#[test]
fn capabilities_and_portable_grid_fail_before_pipeline_creation() {
    let shape = [1, 1, 1, 257];
    let spec = AttentionSpec::new(&shape, &shape, &shape, 1., AttentionMask::None).unwrap();
    let mut limits = wgpu::Limits::default();
    assert!(preflight(spec, &limits).is_err());
    let shape = [1, 1, 5, 3];
    let spec = AttentionSpec::new(&shape, &shape, &shape, 1., AttentionMask::None).unwrap();
    limits.max_compute_workgroups_per_dimension = 2;
    assert!(preflight(spec, &limits).is_err());
    limits.max_compute_workgroups_per_dimension = 3;
    assert_eq!(preflight(spec, &limits).unwrap(), [3, 2]);
    limits.max_bindings_per_bind_group = 7;
    assert!(preflight(spec, &limits).is_err());
    limits.max_bindings_per_bind_group = 8;
    limits.max_compute_workgroup_storage_size = (256 * 2 + 64 + 18) * 4 - 1;
    assert!(preflight(spec, &limits).is_err());
    limits.max_compute_workgroup_storage_size += 1;
    assert!(preflight(spec, &limits).is_ok());
    limits.max_uniform_buffer_binding_size = std::mem::size_of::<Params>() as u32 - 1;
    assert!(preflight(spec, &limits).is_err());
}

#[test]
fn plain_and_biased_attention_match_rust_across_heads_and_dimension_tails() {
    let Some(device) = device() else { return };
    for d in [1, 7, 16, 17, 32, 33, 64, 65, 127, 256] {
        let q_shape = [2, 2, 3, d];
        let k_shape = [2, 2, 5, d];
        let q = data(12 * d, 0.1);
        let k = data(20 * d, 1.2);
        let v = data(20 * d, -0.7);
        let z = data(20, 0.4);
        let pair = data(60, -0.5);
        let q_gpu = device.upload(&q_shape, &q).unwrap();
        let k_gpu = device.upload(&k_shape, &k).unwrap();
        let v_gpu = device.upload(&k_shape, &v).unwrap();
        let z_gpu = device.upload(&[2, 2, 5], &z).unwrap();
        let pair_gpu = device.upload(&[2, 2, 3, 5], &pair).unwrap();
        for mask in [
            AttentionMask::None,
            AttentionMask::Causal { query_offset: 0 },
            AttentionMask::Causal { query_offset: 2 },
        ] {
            for mode in 0..4 {
                let spec = AttentionSpec::new(&q_shape, &k_shape, &k_shape, 0.375, mask).unwrap();
                let expected = attention_reference(
                    spec,
                    &q,
                    &k,
                    &v,
                    (mode & 1 != 0).then_some(z.as_slice()),
                    (mode & 2 != 0).then_some(pair.as_slice()),
                )
                .unwrap();
                let output = q_gpu
                    .scaled_dot_attention(
                        &k_gpu,
                        &v_gpu,
                        0.375,
                        mask,
                        (mode & 1 != 0).then_some(&z_gpu),
                        (mode & 2 != 0).then_some(&pair_gpu),
                    )
                    .unwrap();
                close(&read(&output), &expected);
                assert_eq!(output.layout().shape(), q_shape);
                let merged = q_gpu
                    .scaled_dot_attention_merged_heads(
                        &k_gpu,
                        &v_gpu,
                        0.375,
                        mask,
                        (mode & 1 != 0).then_some(&z_gpu),
                        (mode & 2 != 0).then_some(&pair_gpu),
                    )
                    .unwrap();
                assert_eq!(merged.layout().shape(), spec.merged_output_shape().unwrap());
                assert!(merged.layout().is_contiguous());
                assert_eq!(merged.layout().offset(), 0);
                close(&read(&merged), &merge_heads(q_shape, &expected));
            }
        }
    }
}

#[test]
fn tiled_key_tails_cached_offsets_and_late_overflow_preserve_semantics() {
    let Some(device) = device() else { return };
    for count in (1..=9).chain(127..=137) {
        let qs = [1, 1, 1, 17];
        let ks = [1, 1, count, 17];
        let q = data(17, 0.2);
        let k = data(count * 17, 0.4);
        let v = data(count * 17, -0.6);
        let pair = data(count, -0.2);
        let q_gpu = device.upload(&qs, &q).unwrap();
        let k_gpu = device.upload(&ks, &k).unwrap();
        let v_gpu = device.upload(&ks, &v).unwrap();
        let pair_gpu = device.upload(&[1, 1, 1, count], &pair).unwrap();
        for offset in (0..count).filter(|&v| v < 9 || v + 1 == count) {
            let mask = AttentionMask::Causal {
                query_offset: offset,
            };
            let spec = AttentionSpec::new(&qs, &ks, &ks, -0.25, mask).unwrap();
            let expected = attention_reference(spec, &q, &k, &v, None, Some(&pair)).unwrap();
            let result = q_gpu
                .scaled_dot_attention(&k_gpu, &v_gpu, -0.25, mask, None, Some(&pair_gpu))
                .unwrap();
            close(&read(&result), &expected);
        }
    }
    let q = device.upload(&[1, 1, 1, 1], &[2.]).unwrap();
    let k = device
        .upload(&[1, 1, 5, 1], &[0., 0., 0., 0., f32::MAX])
        .unwrap();
    let v = device.upload(&[1, 1, 5, 1], &[1.; 5]).unwrap();
    let masked = q
        .scaled_dot_attention(
            &k,
            &v,
            1.,
            AttentionMask::Causal { query_offset: 3 },
            None,
            None,
        )
        .unwrap();
    close(&read(&masked), &[1.]);
    let unmasked = q
        .scaled_dot_attention(&k, &v, 1., AttentionMask::None, None, None)
        .unwrap();
    assert!(matches!(
        unmasked.snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
}

#[test]
fn tiled_normalization_preserves_flat_and_sharp_biased_distributions() {
    let Some(device) = device() else { return };
    let qs = [2, 2, 3, 17];
    let ks = [2, 2, 131, 17];
    let q = vec![0.; 12 * 17];
    let k = vec![0.; 4 * 131 * 17];
    let v = data(4 * 131 * 17, -0.6);
    let q_gpu = device.upload(&qs, &q).unwrap();
    let k_gpu = device.upload(&ks, &k).unwrap();
    let v_gpu = device.upload(&ks, &v).unwrap();
    for sharp in [false, true] {
        let z: Vec<_> = (0..4 * 131)
            .map(|i| if sharp && i % 131 == 3 { 1000. } else { -1000. })
            .collect();
        let pair: Vec<_> = (0..12 * 131)
            .map(|i| {
                if sharp && matches!(i % 131, 4 | 127 | 128 | 130) {
                    2000. + (i / 131) as f32 * 0.125
                } else {
                    0.
                }
            })
            .collect();
        let z_gpu = device.upload(&[2, 2, 131], &z).unwrap();
        let pair_gpu = device.upload(&[2, 2, 3, 131], &pair).unwrap();
        for mask in [
            AttentionMask::None,
            AttentionMask::Causal { query_offset: 126 },
            AttentionMask::Causal { query_offset: 128 },
        ] {
            let spec = AttentionSpec::new(&qs, &ks, &ks, 0.25, mask).unwrap();
            let expected = attention_reference(spec, &q, &k, &v, Some(&z), Some(&pair)).unwrap();
            for direct in [false, true] {
                let result = q_gpu
                    .attention_forward(
                        &k_gpu,
                        &v_gpu,
                        0.25,
                        mask,
                        Some(&z_gpu),
                        Some(&pair_gpu),
                        if direct {
                            OutputOrder::MergedHeads
                        } else {
                            OutputOrder::HeadMajor
                        },
                        &mut PassTimestampCursor::default(),
                    )
                    .unwrap();
                close(
                    &read(&result),
                    &if direct {
                        merge_heads(qs, &expected)
                    } else {
                        expected.clone()
                    },
                );
            }
        }
    }
}

#[test]
fn tiled_final_key_overflow_is_rejected_but_causal_tail_is_not_evaluated() {
    let Some(device) = device() else { return };
    let q = device.upload(&[1, 1, 1, 1], &[2.]).unwrap();
    let mut keys = vec![0.; 129];
    keys[128] = f32::MAX;
    let k = device.upload(&[1, 1, 129, 1], &keys).unwrap();
    let v = device.upload(&[1, 1, 129, 1], &[1.; 129]).unwrap();
    for direct in [false, true] {
        for mask in [
            AttentionMask::None,
            AttentionMask::Causal { query_offset: 127 },
        ] {
            let result = q
                .attention_forward(
                    &k,
                    &v,
                    1.,
                    mask,
                    None,
                    None,
                    if direct {
                        OutputOrder::MergedHeads
                    } else {
                        OutputOrder::HeadMajor
                    },
                    &mut PassTimestampCursor::default(),
                )
                .unwrap()
                .snapshot()
                .unwrap()
                .read();
            if mask == AttentionMask::None {
                assert!(matches!(result, Err(TensorError::NonFinite)));
            } else {
                close(&result.unwrap(), &[1.]);
            }
        }
    }
}

#[test]
fn tile_selection_preserves_short_sequences_and_unmeasured_wide_heads() {
    for (keys, dim, expected) in [
        (32, 16, 1),
        (127, 32, 1),
        (128, 32, 8),
        (129, 17, 8),
        (256, 33, 1),
        (256, 256, 1),
    ] {
        let q = [1, 1, 1, dim];
        let k = [1, 1, keys, dim];
        let spec = AttentionSpec::new(&q, &k, &k, 1., AttentionMask::None).unwrap();
        assert_eq!(key_tile(spec), expected);
    }
}

#[test]
fn strided_qkv_and_broadcast_bias_read_directly_and_outputs_are_owned() {
    let Some(device) = device() else { return };
    let q = device
        .upload(&[1, 3, 2, 3], &data(18, 0.1))
        .unwrap()
        .permute(&[0, 2, 1, 3])
        .unwrap()
        .narrow(2, 1, 2)
        .unwrap();
    let k = device
        .upload(&[1, 3, 2, 3], &data(18, 0.7))
        .unwrap()
        .permute(&[0, 2, 1, 3])
        .unwrap();
    let v = device
        .upload(&[1, 3, 2, 3], &data(18, 1.2))
        .unwrap()
        .permute(&[0, 2, 1, 3])
        .unwrap();
    let z = device
        .upload(&[3], &[0.2, -0.5, 0.1])
        .unwrap()
        .broadcast_to(&[1, 2, 3])
        .unwrap();
    let pair = device
        .upload(&[2, 3], &data(6, -0.3))
        .unwrap()
        .broadcast_to(&[1, 2, 2, 3])
        .unwrap();
    let mask = AttentionMask::Causal { query_offset: 1 };
    let spec = AttentionSpec::new(
        q.layout.shape(),
        k.layout.shape(),
        v.layout.shape(),
        0.5,
        mask,
    )
    .unwrap();
    let expected = attention_reference(
        spec,
        &read(&q),
        &read(&k),
        &read(&v),
        Some(&read(&z)),
        Some(&read(&pair)),
    )
    .unwrap();
    let output = q
        .scaled_dot_attention(&k, &v, 0.5, mask, Some(&z), Some(&pair))
        .unwrap();
    let merged = q
        .scaled_dot_attention_merged_heads(&k, &v, 0.5, mask, Some(&z), Some(&pair))
        .unwrap();
    let next = output.gelu().unwrap();
    let merged_next = merged.gelu().unwrap();
    drop((q, k, v, z, pair));
    close(&read(&output), &expected);
    close(&read(&merged), &merge_heads(spec.query_shape(), &expected));
    assert!(read(&merged_next).iter().all(|value| value.is_finite()));
    assert!(read(&next).iter().all(|value| value.is_finite()));
    close(&read(&output), &expected);
}

#[test]
fn strided_columns_offsets_and_independently_broadcast_inputs_match_reference() {
    let Some(device) = device() else { return };
    for count in [5, 129] {
        let d = 17;
        let q = device
            .upload(&[2, 2, d, 5], &data(4 * d * 5, 0.2))
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap()
            .narrow(2, 1, 3)
            .unwrap();
        let k = device
            .upload(&[2, 1, d, count + 2], &data(2 * d * (count + 2), 0.4))
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap()
            .narrow(2, 1, count)
            .unwrap()
            .broadcast_to(&[2, 2, count, d])
            .unwrap();
        let v = device
            .upload(&[1, 2, d, count + 3], &data(2 * d * (count + 3), -0.6))
            .unwrap()
            .permute(&[0, 1, 3, 2])
            .unwrap()
            .narrow(2, 2, count)
            .unwrap()
            .broadcast_to(&[2, 2, count, d])
            .unwrap();
        let z = device
            .upload(&[count + 2], &data(count + 2, 0.9))
            .unwrap()
            .narrow(0, 1, count)
            .unwrap()
            .broadcast_to(&[2, 2, count])
            .unwrap();
        let pair = device
            .upload(&[count, 3], &data(count * 3, -0.3))
            .unwrap()
            .permute(&[1, 0])
            .unwrap()
            .broadcast_to(&[2, 2, 3, count])
            .unwrap();
        for mask in [
            AttentionMask::None,
            AttentionMask::Causal {
                query_offset: count - 3,
            },
        ] {
            let spec = AttentionSpec::new(
                q.layout.shape(),
                k.layout.shape(),
                v.layout.shape(),
                0.375,
                mask,
            )
            .unwrap();
            let expected = attention_reference(
                spec,
                &read(&q),
                &read(&k),
                &read(&v),
                Some(&read(&z)),
                Some(&read(&pair)),
            )
            .unwrap();
            let output = q
                .scaled_dot_attention(&k, &v, 0.375, mask, Some(&z), Some(&pair))
                .unwrap();
            close(&read(&output), &expected);
            let merged = q
                .scaled_dot_attention_merged_heads(&k, &v, 0.375, mask, Some(&z), Some(&pair))
                .unwrap();
            close(&read(&merged), &merge_heads(spec.query_shape(), &expected));
        }
    }
}

#[test]
fn cropped_failed_storage_keeps_guards_for_every_direct_operand_and_empty_query() {
    let Some(device) = device() else { return };
    let failed = device
        .upload(&[2], &[0., f32::MAX])
        .unwrap()
        .mul(&device.upload(&[2], &[1., 2.]).unwrap())
        .unwrap()
        .narrow(0, 0, 1)
        .unwrap();
    let good = device.upload(&[1], &[0.]).unwrap();
    for slot in 0..5 {
        let input = |which, shape: &[usize]| {
            (if slot == which { &failed } else { &good })
                .broadcast_to(shape)
                .unwrap()
        };
        let output = input(0, &[2, 2, 1, 1])
            .scaled_dot_attention(
                &input(1, &[2, 2, 3, 1]),
                &input(2, &[2, 2, 3, 1]),
                1.,
                AttentionMask::Causal { query_offset: 0 },
                Some(&input(3, &[2, 2, 3])),
                Some(&input(4, &[2, 2, 1, 3])),
            )
            .unwrap();
        assert!(matches!(
            output.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
        assert!(matches!(
            output.gelu().unwrap().snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    let empty = device.upload(&[2, 2, 0, 1], &[]).unwrap();
    let failed = failed.broadcast_to(&[2, 2, 3, 1]).unwrap();
    let output = empty
        .scaled_dot_attention(&failed, &failed, 1., AttentionMask::None, None, None)
        .unwrap();
    assert!(matches!(
        output.snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
}

#[test]
fn merged_heads_keep_empty_shapes_and_all_guards() {
    let Some(device) = device() else { return };
    for shape in [[0, 2, 3, 7], [2, 0, 3, 7], [2, 2, 0, 7]] {
        let key_shape = [shape[0], shape[1], 5, shape[3]];
        let q = device.upload(&shape, &[]).unwrap();
        let kv = device
            .upload(&key_shape, &vec![0.; key_shape.iter().product()])
            .unwrap();
        let out = q
            .scaled_dot_attention_merged_heads(&kv, &kv, 1., AttentionMask::None, None, None)
            .unwrap();
        assert_eq!(
            out.layout().shape(),
            [shape[0], shape[2], shape[1] * shape[3]]
        );
        assert!(read(&out).is_empty());
    }
    let good = device.upload(&[1], &[0.]).unwrap();
    let failed = device
        .upload(&[2], &[0., f32::MAX])
        .unwrap()
        .mul(&device.upload(&[2], &[1., 2.]).unwrap())
        .unwrap()
        .narrow(0, 0, 1)
        .unwrap();
    for queries in [0, 1] {
        for slot in 0..5 {
            let input = |which, shape: &[usize]| {
                (if slot == which { &failed } else { &good })
                    .broadcast_to(shape)
                    .unwrap()
            };
            let output = input(0, &[2, 2, queries, 1])
                .scaled_dot_attention_merged_heads(
                    &input(1, &[2, 2, 3, 1]),
                    &input(2, &[2, 2, 3, 1]),
                    1.,
                    AttentionMask::Causal { query_offset: 0 },
                    Some(&input(3, &[2, 2, 3])),
                    Some(&input(4, &[2, 2, queries, 3])),
                )
                .unwrap();
            assert!(matches!(
                output.snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
            assert!(matches!(
                output.gelu().unwrap().snapshot().unwrap().read(),
                Err(TensorError::NonFinite)
            ));
        }
    }
    let q = device.upload(&[2, 2, 1, 1], &[f32::MAX; 4]).unwrap();
    let k = device.upload(&[2, 2, 1, 1], &[2.; 4]).unwrap();
    let overflow = q
        .scaled_dot_attention_merged_heads(&k, &k, 1., AttentionMask::None, None, None)
        .unwrap();
    assert!(matches!(
        overflow.snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
    assert!(q
        .scaled_dot_attention_merged_heads(&k, &k, f32::NAN, AttentionMask::None, None, None)
        .is_err());
    assert!(q
        .scaled_dot_attention_merged_heads(&k, &k, 1., AttentionMask::None, Some(&good), None)
        .is_err());
    assert!(q
        .scaled_dot_attention_merged_heads(
            &k,
            &k,
            1.,
            AttentionMask::Causal { query_offset: 1 },
            None,
            None
        )
        .is_err());
}

#[test]
fn future_bias_cannot_unmask_and_empty_query_has_no_dispatch() {
    let Some(device) = device() else { return };
    let q = device.upload(&[1, 1, 2, 1], &[0., 0.]).unwrap();
    let v = device.upload(&[1, 1, 2, 1], &[2., 10.]).unwrap();
    let z = device.upload(&[1, 1, 2], &[0., 3f32.ln()]).unwrap();
    let pair = device
        .upload(&[1, 1, 2, 2], &[0., f32::MAX, 0., 0.])
        .unwrap();
    let output = q
        .scaled_dot_attention(
            &q,
            &v,
            1.,
            AttentionMask::Causal { query_offset: 0 },
            Some(&z),
            Some(&pair),
        )
        .unwrap();
    close(&read(&output), &[2., 8.]);
    let empty = device.upload(&[1, 1, 0, 1], &[]).unwrap();
    let output = empty
        .scaled_dot_attention(
            &q,
            &v,
            1.,
            AttentionMask::Causal { query_offset: 2 },
            None,
            None,
        )
        .unwrap();
    assert!(read(&output).is_empty());
    assert!(matches!(
        q.scaled_dot_attention(
            &q,
            &v,
            1.,
            AttentionMask::Causal { query_offset: 1 },
            None,
            None
        ),
        Err(TensorError::Attention(AttentionError::QueryOffset))
    ));
}

#[test]
fn score_overflow_and_all_upstream_guards_propagate_even_through_masked_bias() {
    let Some(device) = device() else { return };
    let x = device.upload(&[1, 1, 1, 1], &[f32::MAX]).unwrap();
    let two = device.upload(&[1, 1, 1, 1], &[2.]).unwrap();
    let good = device.upload(&[1, 1, 1, 1], &[0.]).unwrap();
    let failed = x.mul(&two).unwrap();
    let score_overflow = x
        .scaled_dot_attention(&two, &good, 1., AttentionMask::None, None, None)
        .unwrap();
    let mut outputs = vec![score_overflow];
    for slot in 0..5 {
        let z = failed.reshape(&[1, 1, 1]).unwrap();
        let q = if slot == 0 { &failed } else { &good };
        let k = if slot == 1 { &failed } else { &good };
        let v = if slot == 2 { &failed } else { &good };
        outputs.push(
            q.scaled_dot_attention(
                k,
                v,
                1.,
                AttentionMask::None,
                (slot == 3).then_some(&z),
                (slot == 4).then_some(&failed),
            )
            .unwrap(),
        );
    }
    for output in outputs {
        assert!(matches!(
            output.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
        assert!(matches!(
            output.gelu().unwrap().snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
    }
    let kv = device.upload(&[1, 1, 2, 1], &[0., 0.]).unwrap();
    let z = device.upload(&[1, 1, 2], &[0., f32::MAX]).unwrap();
    let z = z
        .mul(&device.upload(&[1, 1, 2], &[1., 2.]).unwrap())
        .unwrap();
    let output = good
        .scaled_dot_attention(
            &kv,
            &kv,
            1.,
            AttentionMask::Causal { query_offset: 0 },
            Some(&z),
            None,
        )
        .unwrap();
    assert!(matches!(
        output.snapshot().unwrap().read(),
        Err(TensorError::NonFinite)
    ));
}

#[test]
fn device_ownership_shape_and_bias_admission_are_explicit() {
    let Some(device) = device() else { return };
    let q = device.upload(&[1, 1, 2, 3], &[0.; 6]).unwrap();
    let wrong_bias = device.upload(&[2], &[0.; 2]).unwrap();
    assert!(matches!(
        q.scaled_dot_attention(&q, &q, 1., AttentionMask::None, Some(&wrong_bias), None),
        Err(TensorError::Attention(AttentionError::BiasShape))
    ));
    assert!(matches!(
        q.scaled_dot_attention(&q, &q, f32::NAN, AttentionMask::None, None, None),
        Err(TensorError::Attention(AttentionError::Scale))
    ));
    let other_device = TensorDevice::new(
        pollster::block_on(WgpuRuntime::request_headless("attention.other_device")).unwrap(),
    )
    .unwrap();
    let other = other_device.upload(&[1, 1, 2, 3], &[0.; 6]).unwrap();
    assert!(matches!(
        q.scaled_dot_attention(&other, &q, 1., AttentionMask::None, None, None),
        Err(TensorError::DeviceMismatch)
    ));
    let shared_device = TensorDevice::new(device.runtime().clone()).unwrap();
    let shared = shared_device.upload(&[1, 1, 2, 3], &[1.; 6]).unwrap();
    close(
        &read(
            &q.scaled_dot_attention(&q, &shared, 1., AttentionMask::None, None, None)
                .unwrap(),
        ),
        &[1.; 6],
    );
}

#[test]
fn zero_bias_is_identity_and_large_finite_scores_remain_stable() {
    let Some(device) = device() else { return };
    let shape = [1, 1, 3, 1];
    let q = device.upload(&shape, &[1.; 3]).unwrap();
    let k = device.upload(&shape, &[-1e30, 0., 1e30]).unwrap();
    let v = device.upload(&shape, &[1., 2., 4.]).unwrap();
    let z = device.upload(&[1, 1, 3], &[0.; 3]).unwrap();
    let pair = device.upload(&[1, 1, 3, 3], &[0.; 9]).unwrap();
    for (scale, expected) in [(1., 4.), (-1., 1.), (0., 7. / 3.)] {
        let plain = q
            .scaled_dot_attention(&k, &v, scale, AttentionMask::None, None, None)
            .unwrap();
        let zero_biased = q
            .scaled_dot_attention(&k, &v, scale, AttentionMask::None, Some(&z), Some(&pair))
            .unwrap();
        close(&read(&plain), &[expected; 3]);
        close(&read(&zero_biased), &read(&plain));
    }
}

#[test]
fn pytorch_sdpa_fixture_matches_rust_and_resident_outputs() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/resident_attention_torch.json"
    ))
    .unwrap();
    assert_eq!(fixture["schema"], "spiraltorch.resident_attention_torch.v1");
    let device = device();
    let floats = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let shape = |v: &serde_json::Value| -> Vec<usize> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect()
    };
    for case in fixture["cases"].as_array().unwrap() {
        let q_shape = shape(&case["query_shape"]);
        let k_shape = shape(&case["key_shape"]);
        let q = floats(&case["query"]);
        let k = floats(&case["key"]);
        let v = floats(&case["value"]);
        let z = (!case["z_bias"].is_null()).then(|| floats(&case["z_bias"]));
        let pair = (!case["pair_bias"].is_null()).then(|| floats(&case["pair_bias"]));
        let expected = floats(&case["expected"]);
        let scale = case["scale"].as_f64().unwrap() as f32;
        let mask = case["query_offset"]
            .as_u64()
            .map_or(AttentionMask::None, |offset| AttentionMask::Causal {
                query_offset: offset as usize,
            });
        let spec = AttentionSpec::new(&q_shape, &k_shape, &k_shape, scale, mask).unwrap();
        let reference =
            attention_reference(spec, &q, &k, &v, z.as_deref(), pair.as_deref()).unwrap();
        close(&reference, &expected);
        if let Some(device) = &device {
            let q_gpu = device.upload(&q_shape, &q).unwrap();
            let k_gpu = device.upload(&k_shape, &k).unwrap();
            let v_gpu = device.upload(&k_shape, &v).unwrap();
            let z_gpu = z
                .as_ref()
                .map(|z| device.upload(&spec.z_bias_shape(), z).unwrap());
            let pair_gpu = pair
                .as_ref()
                .map(|p| device.upload(&spec.pair_bias_shape(), p).unwrap());
            let output = q_gpu
                .scaled_dot_attention(
                    &k_gpu,
                    &v_gpu,
                    scale,
                    mask,
                    z_gpu.as_ref(),
                    pair_gpu.as_ref(),
                )
                .unwrap();
            let actual = read(&output);
            close(&actual, &expected);
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("{} max_abs_error={max_error:e}", case["name"]);
        }
    }
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 20);
}
