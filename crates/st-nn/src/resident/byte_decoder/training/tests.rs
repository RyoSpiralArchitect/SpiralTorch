use super::*;

fn read(t: &ResidentTensor) -> Result<Vec<f32>, GpuTensorError> {
    t.snapshot()?.read()
}

#[test]
fn every_parameter_candidate_participates_in_one_atomic_update() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte_decoder.atomicity")
            .unwrap();
    let plan = super::super::tests::plan(AttentionMask::Causal { query_offset: 0 }, 2).unwrap();
    let host = ByteLmBatch::from_windows(&[b"abac", b"xyxz"]).unwrap();
    for fault in 0..plan.parameter_layout().len() {
        let mut model = plan.compile_training_wgpu(runtime.clone()).unwrap();
        let batch = model.prepare_batch(&host).unwrap();
        let forward = model.forward(&batch).unwrap();
        let seed = model
            .tensor_device()
            .upload(
                model.output_layout().shape(),
                &vec![0.001; model.output_layout().len()],
            )
            .unwrap();
        let mut gradient = model.backward(&forward, &seed).unwrap();
        let before = model.parameter_snapshot();
        gradient.parameters = before
            .values()
            .iter()
            .enumerate()
            .map(|(i, p)| {
                model
                    .tensor_device()
                    .upload(
                        p.layout().shape(),
                        &vec![if i == fault { 2. } else { 0.25 }; p.layout().len()],
                    )
                    .unwrap()
            })
            .collect();
        gradient.bound = before.bind_gradients(gradient.parameters.clone()).unwrap();
        let update = model.sgd(&gradient, f32::MAX).unwrap();
        assert!(
            matches!(update.snapshot().unwrap().read(), Err(TrainingError::Rejected { stage, .. }) if stage == fault)
        );
        let after = model.parameter_snapshot();
        assert_eq!(after.values().len(), 12);
        for (old, new) in before.values().iter().zip(after.values()) {
            let old = old.snapshot().unwrap().read().unwrap();
            let new = new.snapshot().unwrap().read().unwrap();
            assert_eq!(old.len(), new.len());
            assert!(old
                .iter()
                .map(|v| v.to_bits())
                .eq(new.iter().map(|v| v.to_bits())));
        }
        assert!(matches!(
            model.backward(&forward, &seed),
            Err(InferenceError::Training(TrainingError::StaleForward))
        ));
        let recovered = model.forward(&batch).unwrap();
        let old = forward.prediction().snapshot().unwrap().read().unwrap();
        let new = recovered.prediction().snapshot().unwrap().read().unwrap();
        assert_eq!(old, new);
    }
}

#[test]
fn late_token_or_position_accumulation_failure_guards_the_entire_vjp_family() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte_decoder.late_embedding")
            .unwrap();
    for token_failure in [true, false] {
        let shape = if token_failure { [1, 2, 4] } else { [2, 1, 4] };
        let plan = super::super::tests::plan_with_shape(
            shape,
            AttentionMask::Causal { query_offset: 0 },
            1,
        )
        .unwrap();
        let host = if token_failure {
            ByteLmBatch::from_windows(&[b"aaa"]).unwrap()
        } else {
            ByteLmBatch::from_windows(&[b"ab", b"cd"]).unwrap()
        };
        for rate in [0., 0.125] {
            let mut model = plan.compile_training_wgpu(runtime.clone()).unwrap();
            let batch = model.prepare_batch(&host).unwrap();
            let d = model.tensor_device();
            let z = d
                .upload(&[shape[0], 2, shape[1]], &vec![0.; shape[0] * 2 * shape[1]])
                .unwrap();
            let pair = d
                .upload(
                    &[shape[0], 2, shape[1], shape[1]],
                    &vec![0.; shape[0] * 2 * shape[1] * shape[1]],
                )
                .unwrap();
            let forward = model
                .forward_with_external_biases(
                    &batch,
                    &[ByteDecoderBias {
                        z_bias: Some(&z),
                        pair_bias: Some(&pair),
                    }],
                )
                .unwrap();
            let before = model.parameter_snapshot();
            let before_values: Vec<_> = before.values().iter().map(|p| read(p).unwrap()).collect();
            let seed = model
                .tensor_device()
                .upload(&shape, &vec![2e38; shape.iter().product()])
                .unwrap();
            // Private-boundary fault injection isolates the final scatter from
            // CE and earlier matrix reductions. It is not numerical VJP evidence.
            let token = read(&forward.token.backward(&seed).unwrap());
            let position = read(&forward.position.backward(&seed).unwrap());
            assert_eq!(token.is_ok(), !token_failure);
            assert_eq!(position.is_ok(), token_failure);
            let failed = if token_failure { token } else { position };
            assert!(matches!(failed, Err(GpuTensorError::NonFinite)));
            let trailing = before.values()[2..]
                .iter()
                .map(|p| {
                    model
                        .tensor_device()
                        .upload(p.layout().shape(), &vec![0.125; p.layout().len()])
                        .unwrap()
                })
                .collect();
            let gradient = model
                .bind_vjp(
                    &forward,
                    seed,
                    trailing,
                    vec![ByteDecoderBiasGradient {
                        z_bias: Some(z.clone()),
                        pair_bias: Some(pair.clone()),
                    }],
                )
                .unwrap();
            assert_eq!(gradient.parameter_gradients().len(), before.values().len());
            for g in gradient
                .parameter_gradients()
                .iter()
                .chain([gradient.embedding_output_gradient()])
                .chain(
                    gradient
                        .biases
                        .iter()
                        .flat_map(|b| b.z_bias.iter().chain(&b.pair_bias)),
                )
            {
                assert!(matches!(read(g), Err(GpuTensorError::NonFinite)));
            }
            let update = model.sgd(&gradient, rate).unwrap();
            assert!(
                matches!(update.snapshot().unwrap().read(), Err(TrainingError::Rejected { stage: 0, flags })
                if flags & st_backend_wgpu::resident_tensor::INVALID_TENSOR_FLAG != 0)
            );
            let after = model.parameter_snapshot();
            assert_eq!(before_values.len(), after.values().len());
            for (old, new) in before_values.iter().zip(after.values()) {
                let new = read(new).unwrap();
                assert_eq!(old.len(), new.len());
                assert!(old
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(new.iter().map(|v| v.to_bits())));
            }
            let recovered = model.forward(&batch).unwrap();
            assert_eq!(
                read(recovered.prediction()).unwrap(),
                read(forward.prediction()).unwrap()
            );
            let loss = recovered
                .next_byte_loss(
                    CrossEntropySpec::new(
                        st_kernel_contracts::classification::ClassReduction::Mean,
                        -100,
                        0.,
                    )
                    .unwrap(),
                )
                .unwrap();
            let valid = model
                .backward(&recovered, loss.prediction_gradient())
                .unwrap();
            assert_eq!(
                model
                    .sgd(&valid, 0.)
                    .unwrap()
                    .snapshot()
                    .unwrap()
                    .read()
                    .unwrap(),
                2
            );
        }
    }
}
