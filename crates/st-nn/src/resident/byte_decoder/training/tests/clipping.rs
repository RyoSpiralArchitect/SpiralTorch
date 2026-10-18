use super::*;
use st_kernel_contracts::classification::ClassReduction;

fn capture(model: &ResidentByteDecoder) -> ByteDecoderCheckpoint {
    model.checkpoint_snapshot().unwrap().read().unwrap()
}

fn step(model: &mut ResidentByteDecoder, adamw: bool, frozen: bool) -> f32 {
    let before = capture(model);
    let host = ByteLmBatch::from_windows(&[b"abac", b"xyxz"]).unwrap();
    let batch = model.prepare_batch(&host).unwrap();
    let tape = model.forward(&batch).unwrap();
    let loss = tape
        .next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.).unwrap())
        .unwrap();
    let raw = model.backward(&tape, loss.prediction_gradient()).unwrap();
    let gradients: Vec<_> = raw
        .parameter_gradients()
        .iter()
        .map(|g| read(g).unwrap())
        .collect();
    let old: Vec<_> = model
        .parameter_snapshot()
        .values()
        .iter()
        .map(|p| read(p).unwrap())
        .collect();
    let slots = model.parameter_layout().geometry().unwrap().all();
    let rates: Vec<_> = (0..gradients.len())
        .map(|i| {
            if frozen && slots.contains(&i) {
                0.
            } else {
                0.005
            }
        })
        .collect();
    let squared: f64 = gradients
        .iter()
        .zip(&rates)
        .filter(|(_, r)| **r != 0.)
        .flat_map(|(g, _)| g)
        .map(|&v| f64::from(v).powi(2))
        .sum();
    let clip = GlobalNormClip::new(0.01).unwrap();
    assert!(
        squared.sqrt() > f64::from(clip.max_norm()),
        "clip must be exercised"
    );
    let factors = clip.factors(squared).unwrap();
    let scaled: Vec<Vec<f32>> = gradients
        .iter()
        .zip(&rates)
        .map(|(g, r)| {
            g.iter()
                .map(|&v| {
                    if *r != 0. {
                        factors.as_slice().iter().fold(v, |a, b| a * b)
                    } else {
                        v
                    }
                })
                .collect()
        })
        .collect();
    let steps: Vec<_> = rates
        .iter()
        .map(|&r| AdamWStep::new(r, 0.).unwrap())
        .collect();
    let update = if adamw {
        model.adamw_with_steps_clipped(&raw, &steps, clip).unwrap()
    } else {
        model.sgd_with_rates_clipped(&raw, &rates, clip).unwrap()
    };
    assert_eq!(
        update.snapshot().unwrap().read().unwrap(),
        before.attempted_revision() + 1
    );
    let after = capture(model);
    if frozen {
        assert!(after.geometry_parameters_match(&before).unwrap());
        if adamw {
            for i in slots.clone() {
                assert_eq!(
                    after.adamw_state().unwrap().slots[i],
                    before.adamw_state().unwrap().slots[i]
                );
            }
        }
    }
    for (i, tensor) in model.parameter_snapshot().values().iter().enumerate() {
        let actual = read(tensor).unwrap();
        for (j, value) in actual.iter().enumerate() {
            let expected = if adamw {
                let state = before.adamw_state().unwrap();
                let slot = &state.slots[i];
                steps[i]
                    .candidate(
                        state.config,
                        old[i][j],
                        scaled[i][j],
                        slot.first[j],
                        slot.second[j],
                        slot.steps,
                    )
                    .unwrap()
                    .value
            } else {
                old[i][j] - rates[i] * scaled[i][j]
            };
            assert!(
                (value - expected).abs() <= 2e-6 * expected.abs().max(0.01),
                "parameter {i} / {j}: {value} != {expected}"
            );
        }
        assert!(read(&raw.parameter_gradients()[i])
            .unwrap()
            .iter()
            .zip(&gradients[i])
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    assert!(model.backward(&tape, loss.prediction_gradient()).is_err());
    read(loss.value()).unwrap()[0]
}

#[test]
fn clipped_language_sgd_and_adamw_preserve_geometry_freeze_and_resume() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let runtime = st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.clip")
        .unwrap()
        .0;
    let plan = super::query_pullback::contextual_plan();
    for adamw in [false, true] {
        let mut model = plan.compile_training_wgpu(runtime.clone()).unwrap();
        if adamw {
            model.enable_adamw(AdamWConfig::default()).unwrap();
        }
        let initial = capture(&model);
        step(&mut model, adamw, false);
        assert!(!capture(&model).geometry_parameters_match(&initial).unwrap());
        step(&mut model, adamw, true);
        let saved = capture(&model);
        let decoded = ByteDecoderCheckpoint::from_json(&saved.to_json().unwrap()).unwrap();
        let mut resumed = decoded.restore_wgpu(runtime.clone()).unwrap();
        for frozen in [false, true, false] {
            assert_eq!(
                step(&mut model, adamw, frozen).to_bits(),
                step(&mut resumed, adamw, frozen).to_bits()
            );
            assert_eq!(
                capture(&model).to_json().unwrap(),
                capture(&resumed).to_json().unwrap()
            );
        }
    }
}
