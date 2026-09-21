use super::*;

#[test]
fn nerf_shaders_validate_and_preserve_legacy_struct_layouts() {
    for source in [SAMPLER, COMPOSITOR] {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        for (_, ty) in module.types.iter() {
            let (size, offsets): (u32, &[u32]) = match ty.name.as_deref() {
                Some("Ray") => (48, &[0, 16, 28, 32]),
                Some("SamplePoint") => (16, &[0, 12]),
                Some("FieldSample") => (32, &[0, 16]),
                _ => continue,
            };
            let naga::TypeInner::Struct { members, span } = &ty.inner else {
                panic!("expected struct");
            };
            assert_eq!(*span, size);
            assert_eq!(
                members.iter().map(|m| m.offset).collect::<Vec<_>>(),
                offsets
            );
        }
    }
    assert_eq!(std::mem::size_of::<GpuRay>(), 48);
    assert_eq!(std::mem::offset_of!(GpuRay, direction), 16);
    assert_eq!(std::mem::offset_of!(GpuRay, near), 28);
    assert_eq!(std::mem::offset_of!(GpuRay, far), 32);
}

#[test]
fn nerf_shapes_and_ray_values_are_checked_before_allocation() {
    let limits = wgpu::Limits::default();
    assert_eq!(dimensions(65, 257, &limits).unwrap(), 65 * 257);
    for (rays, samples) in [(0, 1), (1, 0), (usize::MAX, 2), (1, usize::MAX)] {
        assert!(dimensions(rays, samples, &limits).is_err());
    }
    let tiny = wgpu::Limits {
        max_compute_workgroups_per_dimension: 1,
        ..limits
    };
    assert!(dimensions(65, 1, &tiny).is_err());
    let ray = NerfRay {
        origin: [0.; 3],
        direction: [0., 0., 2.],
        near: -1.,
        far: 1.,
    };
    assert!(ray.checked(0).is_ok());
    assert!(NerfRay { far: -2., ..ray }.checked(1).is_err());
    assert!(NerfRay {
        near: -f32::MAX,
        far: f32::MAX,
        ..ray
    }
    .checked(2)
    .is_err());
    assert!(NerfRay {
        origin: [f32::NAN, 0., 0.],
        ..ray
    }
    .checked(3)
    .is_err());
    assert!(NerfRay {
        direction: [f32::INFINITY, 0., 0.],
        ..ray
    }
    .checked(4)
    .is_err());
}

#[cfg(not(target_arch = "wasm32"))]
mod gpu {
    use super::*;
    use crate::{
        resident_graph::ResidentGraph,
        resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
        runtime::WgpuRuntime,
    };
    use st_kernel_contracts::graph::{GraphDefinition, GraphParameter, GraphStage, ParameterRole};

    fn device() -> Option<TensorDevice> {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return None;
        }
        let (runtime, _) = runtime::ensure_default_runtime_blocking("nerf.resident.tests")
            .expect("explicit NeRF runtime tests require a WGPU device");
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        eprintln!("NeRF adapter: {:?}", runtime.adapter_info());
        Some(TensorDevice::new(runtime).unwrap())
    }

    fn assert_close(actual: &[f32], expected: &[f64], atol: f64, rtol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && (f64::from(a) - b).abs() <= atol + rtol * b.abs(),
                "index {i}: {a} vs {b}"
            );
        }
    }

    fn ray(span: f32) -> NerfRay {
        NerfRay {
            origin: [0.1, -0.2, 0.3],
            direction: [0.4, 0.5, -0.6],
            near: 0.,
            far: span,
        }
    }

    #[test]
    fn nerf_real_gpu_full_bins_jitter_thin_zero_and_workgroup_edges() {
        let Some(device) = device() else { return };
        let nerf = ResidentNerf::new(device.clone()).unwrap();
        for rows in [1, 2, 65] {
            let rays: Vec<_> = (0..rows).map(|i| ray([1., 0., 1e-7, 3.][i % 4])).collect();
            let resident = nerf.upload_rays(&rays).unwrap();
            for samples in [1, 8, 257] {
                for mode in [RaySampling::Midpoint, RaySampling::Stratified { seed: 17 }] {
                    let sampled = nerf.sample(&resident, samples, mode).unwrap();
                    let field = device
                        .upload(&[1, 1, 4], &[0.7, 0.2, 0.4, -0.3])
                        .unwrap()
                        .broadcast_to(&[rows, samples, 4])
                        .unwrap();
                    let rgba = nerf.composite(&sampled, &field).unwrap();
                    let expected: Vec<_> = rays
                        .iter()
                        .flat_map(|r| {
                            let opacity = -(-f64::from(0.7f32) * f64::from(r.far)).exp_m1();
                            [
                                f64::from(0.2f32) * opacity,
                                f64::from(0.4f32) * opacity,
                                f64::from(-0.3f32) * opacity,
                                opacity,
                            ]
                        })
                        .collect();
                    assert_close(
                        &rgba.snapshot().unwrap().read().unwrap(),
                        &expected,
                        3e-7,
                        3e-6,
                    );
                    let widths = sampled.widths().snapshot().unwrap().read().unwrap();
                    let points = sampled.points().snapshot().unwrap().read().unwrap();
                    for (i, r) in rays.iter().enumerate() {
                        let width = r.far / samples as f32;
                        for j in 0..samples {
                            assert_eq!(widths[i * samples + j], width);
                            let t = points[(i * samples + j) * 4 + 3];
                            assert!(
                                t >= j as f32 * width - 1e-6 && t <= (j + 1) as f32 * width + 1e-6
                            );
                            if mode == RaySampling::Midpoint {
                                assert!((t - (j as f32 + 0.5) * width).abs() <= 1e-6);
                            }
                            for d in 0..3 {
                                let actual = points[(i * samples + j) * 4 + d];
                                assert!(
                                    (actual - (r.origin[d] + r.direction[d] * t)).abs() <= 1e-6
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nerf_real_gpu_thin_opacity_opaque_tail_and_long_prefix() {
        let Some(device) = device() else { return };
        let nerf = ResidentNerf::new(device.clone()).unwrap();
        for (count, span, density, atol) in [(1, 1e-8, 1., 1e-14), (16_384, 1., 0.001, 2e-9)] {
            let rays = nerf.upload_rays(&[ray(span)]).unwrap();
            let samples = nerf.sample(&rays, count, RaySampling::Midpoint).unwrap();
            let field = device
                .upload(&[1, 1, 4], &[density, 0.25, -0.5, 1.])
                .unwrap()
                .broadcast_to(&[1, count, 4])
                .unwrap();
            let opacity = -(-f64::from(span) * f64::from(density)).exp_m1();
            let actual = nerf
                .composite(&samples, &field)
                .unwrap()
                .snapshot()
                .unwrap()
                .read()
                .unwrap();
            assert_close(
                &actual,
                &[opacity * 0.25, -opacity * 0.5, opacity, opacity],
                atol,
                3e-6,
            );
        }
        let rays = nerf.upload_rays(&[ray(2.)]).unwrap();
        let samples = nerf.sample(&rays, 2, RaySampling::Midpoint).unwrap();
        let field = device
            .upload(&[1, 2, 4], &[20., 0., 0., 0., 1., 1e8, 0., 0.])
            .unwrap();
        let actual = nerf
            .composite(&samples, &field)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap();
        let tail = (-20f64).exp() * -(-1f64).exp_m1() * 1e8;
        assert_close(&actual, &[tail, 0., 0., -(-21f64).exp_m1()], 1e-7, 3e-6);
    }

    #[test]
    fn nerf_real_gpu_nn_graph_chains_without_intermediate_readback() {
        let Some(device) = device() else { return };
        let rows = 3;
        let count = 17;
        let weights = [
            0.2f32, 0.1, -0.05, 0.3, -0.1, 0.25, 0.3, 0.02, 0.4, -0.2, 0.1, 0.15,
        ];
        let bias = [0.7f32, 0.2, 0.4, -0.1];
        let definition = GraphDefinition::new(
            NdLayout::contiguous(&[rows, count, 3]).unwrap(),
            vec![GraphStage::Linear {
                weight: 0,
                bias: 1,
                gelu: false,
            }],
            vec![
                GraphParameter {
                    role: ParameterRole::Weight,
                    shape: vec![3, 4],
                    values: weights.to_vec(),
                },
                GraphParameter {
                    role: ParameterRole::Bias,
                    shape: vec![4],
                    values: bias.to_vec(),
                },
            ],
        )
        .unwrap();
        let mut graph = ResidentGraph::new(
            device.runtime().clone(),
            definition,
            MatmulTile::default(),
            MatmulKernel::Scalar,
            MatmulAccumulation::Sequential,
        )
        .unwrap();
        let nerf = ResidentNerf::new(graph.tensor_device().clone()).unwrap();
        let rays = [ray(0.), ray(1.), ray(2.)];
        let resident = nerf.upload_rays(&rays).unwrap();
        let samples = nerf
            .sample(&resident, count, RaySampling::Midpoint)
            .unwrap();
        graph
            .set_input_tensor(&samples.positions().unwrap())
            .unwrap();
        graph.dispatch().unwrap();
        let field = graph.output_tensor().unwrap();
        let rgba = nerf.composite(&samples, &field).unwrap();
        let snapshot = rgba.snapshot().unwrap();
        let mut expected = Vec::new();
        for ray in rays {
            let width = f64::from(ray.far) / count as f64;
            let mut trans = 1f64;
            let mut result = [0f64; 4];
            for i in 0..count {
                let t = (i as f64 + 0.5) * width;
                let position = std::array::from_fn::<_, 3, _>(|d| {
                    (f64::from(ray.origin[d]) + f64::from(ray.direction[d]) * t) as f32
                });
                let mut value = bias;
                for d in 0..3 {
                    for c in 0..4 {
                        value[c] += position[d] * weights[d * 4 + c];
                    }
                }
                let tau = f64::from(value[0].max(0.)) * width;
                let weight = trans * -(-tau).exp_m1();
                for c in 0..3 {
                    result[c] += weight * f64::from(value[c + 1]);
                }
                result[3] += weight;
                trans *= (-tau).exp();
            }
            expected.extend(result);
        }
        let next = nerf
            .sample(&resident, count, RaySampling::Stratified { seed: 7 })
            .unwrap();
        graph.set_input_tensor(&next.positions().unwrap()).unwrap();
        graph.dispatch().unwrap();
        drop(graph);
        drop(nerf);
        drop(field);
        assert_close(&snapshot.read().unwrap(), &expected, 4e-7, 4e-6);
        assert_close(
            &rgba.snapshot().unwrap().read().unwrap(),
            &expected,
            4e-7,
            4e-6,
        );
    }

    #[test]
    fn nerf_real_gpu_rejects_invalid_shapes_devices_and_inherited_errors() {
        let Some(device) = device() else { return };
        let nerf = ResidentNerf::new(device.clone()).unwrap();
        let rays = nerf.upload_rays(&[ray(2.)]).unwrap();
        assert!(nerf.sample(&rays, 0, RaySampling::Midpoint).is_err());
        let samples = nerf.sample(&rays, 2, RaySampling::Midpoint).unwrap();
        let wrong = device.upload(&[2, 4], &[1.; 8]).unwrap();
        assert!(matches!(
            nerf.composite(&samples, &wrong),
            Err(NerfError::FieldShape)
        ));
        let invalid = device
            .upload(&[1, 2, 4], &[-f32::MAX; 8])
            .unwrap()
            .mul(&device.upload(&[], &[2.]).unwrap())
            .unwrap()
            .relu()
            .unwrap();
        let rejected = nerf.composite(&samples, &invalid).unwrap();
        assert!(matches!(
            rejected.snapshot().unwrap().read(),
            Err(TensorError::NonFinite)
        ));
        let overflowing = device
            .upload(&[1, 2, 4], &[f32::MAX, 1., 1., 1., f32::MAX, 1., 1., 1.])
            .unwrap();
        assert!(matches!(
            nerf.composite(&samples, &overflowing)
                .unwrap()
                .snapshot()
                .unwrap()
                .read(),
            Err(TensorError::NonFinite)
        ));
        let bad_rays = nerf
            .upload_rays(&[NerfRay {
                direction: [f32::MAX; 3],
                ..ray(8.)
            }])
            .unwrap();
        let bad_samples = nerf.sample(&bad_rays, 2, RaySampling::Midpoint).unwrap();
        let good = device.upload(&[1, 2, 4], &[1.; 8]).unwrap();
        assert!(matches!(
            nerf.composite(&bad_samples, &good)
                .unwrap()
                .snapshot()
                .unwrap()
                .read(),
            Err(TensorError::NonFinite)
        ));
        assert!(nerf
            .composite(&samples, &good)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .is_ok());
        let other =
            pollster::block_on(WgpuRuntime::request_headless("nerf.foreign.tests")).unwrap();
        let foreign = TensorDevice::new(other).unwrap();
        let field = foreign.upload(&[1, 2, 4], &[1.; 8]).unwrap();
        assert!(matches!(
            nerf.composite(&samples, &field),
            Err(NerfError::Tensor(TensorError::DeviceMismatch))
        ));
        let other_nerf = ResidentNerf::new(foreign).unwrap();
        assert!(matches!(
            other_nerf.sample(&rays, 2, RaySampling::Midpoint),
            Err(NerfError::Tensor(TensorError::DeviceMismatch))
        ));
    }

    #[test]
    fn nerf_real_gpu_counter_jitter_is_replayable_and_versioned() {
        let Some(device) = device() else { return };
        let nerf = ResidentNerf::new(device).unwrap();
        let rays = nerf.upload_rays(&[ray(1.), ray(1.)]).unwrap();
        let sample = |seed| {
            nerf.sample(&rays, 8, RaySampling::Stratified { seed })
                .unwrap()
        };
        let first = sample(7);
        let held = first.points().snapshot().unwrap();
        let repeat = sample(7).points().snapshot().unwrap().read().unwrap();
        let changed = sample(8).points().snapshot().unwrap().read().unwrap();
        let original = held.read().unwrap();
        assert_eq!(original, repeat);
        assert_ne!(original, changed);
        assert_ne!(&original[..32], &original[32..]);
    }
}
