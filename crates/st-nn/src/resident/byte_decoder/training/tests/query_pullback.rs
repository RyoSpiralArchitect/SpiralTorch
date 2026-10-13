use super::*;

fn contextual_plan() -> ByteDecoderPlan {
    let base =
        super::super::super::tests::plan(AttentionMask::Causal { query_offset: 0 }, 2).unwrap();
    let values = |n: usize, scale: f32, offset: f32| {
        (0..n)
            .map(|i| offset + scale * (i as f32 * 0.37).sin())
            .collect::<Vec<_>>()
    };
    let branch =
        InferencePlan::from_operations(base.input_layout().clone(), vec![InferenceOp::Relu])
            .unwrap();
    let weight = Tensor::from_vec(
        4,
        4,
        (0..16)
            .map(|i| 0.2 * (i as f32 * 0.71).sin() + if i % 5 == 0 { 0.5 } else { 0. })
            .collect(),
    )
    .unwrap();
    let bias = Tensor::zeros(1, 4).unwrap();
    let attention = AttentionInferencePlan::from_parameters(
        base.input_layout().clone(),
        2,
        AttentionMask::Causal { query_offset: 0 },
        [(&weight, &bias); 4],
    )
    .unwrap();
    let block = ResidualAttentionPlan::from_plans(&branch, &attention, &branch).unwrap();
    let head = InferencePlan::from_operations(
        base.input_layout().clone(),
        vec![InferenceOp::Linear {
            weight: Tensor::from_vec(4, BYTE_LM_VOCAB, values(4 * BYTE_LM_VOCAB, 0.1, 0.)).unwrap(),
            bias: Tensor::zeros(1, BYTE_LM_VOCAB).unwrap(),
        }],
    )
    .unwrap();
    let plan = ByteDecoderPlan::from_plans(
        &Tensor::from_vec(BYTE_LM_VOCAB, 4, values(4 * BYTE_LM_VOCAB, 0.2, 0.2)).unwrap(),
        &Tensor::from_vec(5, 4, values(20, 0.03, 0.)).unwrap(),
        &vec![block; 2],
        &head,
    )
    .unwrap();
    use ByteDecoderPairMetric::*;
    let projection = InferencePlan::from_operations(
        plan.input_layout().clone(),
        vec![InferenceOp::Linear { weight, bias }],
    )
    .unwrap();
    let geometry = ByteDecoderGeometryPlan::new(
        &projection,
        &[-0.7, 0.4],
        &[-0.2, 0.3],
        &vec![vec![-0.6, 0.3]; 2],
        -0.75,
    )
    .unwrap()
    .with_mixture(
        &[
            PoincareSquared,
            EuclideanChordSquared,
            CategoricalFisherRaoSquared,
            ComplexProjectiveChordSquared,
        ],
        &vec![vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.5, -0.3]; 2],
        &vec![vec![1., 0.8, 24., 4., 1.2, 0.6, 48., 6.]; 2],
    )
    .unwrap();
    let query = InferencePlan::from_operations(
        geometry.projection().output_layout().clone(),
        vec![InferenceOp::Linear {
            weight: Tensor::from_vec(4, 8, values(32, 0.4, 0.)).unwrap(),
            bias: Tensor::zeros(1, 8).unwrap(),
        }],
    )
    .unwrap();
    plan.with_causal_geometry(geometry.with_query_projections(&vec![query; 2]).unwrap())
        .unwrap()
}

fn runtime() -> Option<WgpuRuntime> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    Some(
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte_decoder.query_pullback")
            .unwrap()
            .0,
    )
}

fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[test]
fn only_query_coordinate_edge_is_stopped_and_full_tapes_are_reusable() {
    let Some(runtime) = runtime() else { return };
    let plan = contextual_plan();
    let mut model = plan.compile_training_wgpu(runtime.clone()).unwrap();
    let host = ByteLmBatch::from_windows(&[b"abac", b"xyxz"]).unwrap();
    let batch = model.prepare_batch(&host).unwrap();
    let forward = model.forward(&batch).unwrap();
    let frozen = read(forward.prediction()).unwrap();
    let seed = model
        .tensor_device()
        .upload(
            model.output_layout().shape(),
            &(0..model.output_layout().len())
                .map(|i| (i as f32 * 0.17).sin() * 0.1)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let full = model.backward(&forward, &seed).unwrap();
    let embedding_full = read(full.embedding_output_gradient()).unwrap();
    let parameters_full: Vec<_> = full
        .parameter_gradients()
        .iter()
        .map(|p| read(p).unwrap())
        .collect();
    let detached = model
        .backward_with_query_pullback(&forward, &seed, ByteDecoderQueryPullback::StopCoordinates)
        .unwrap();
    let embedding_detached = read(detached.embedding_output_gradient()).unwrap();
    let parameters_detached: Vec<_> = detached
        .parameter_gradients()
        .iter()
        .map(|p| read(p).unwrap())
        .collect();
    assert!(same(
        &embedding_full,
        &read(full.embedding_output_gradient()).unwrap()
    ));
    for (old, retained) in parameters_full.iter().zip(full.parameter_gradients()) {
        assert!(same(old, &read(retained).unwrap()));
    }
    let again = model.backward(&forward, &seed).unwrap();
    assert_eq!(full.query_pullback(), ByteDecoderQueryPullback::Full);
    assert_eq!(
        detached.query_pullback(),
        ByteDecoderQueryPullback::StopCoordinates
    );
    assert!(same(&frozen, &read(forward.prediction()).unwrap()));
    assert!(same(
        &embedding_detached,
        &read(detached.embedding_output_gradient()).unwrap()
    ));
    assert!(same(
        &embedding_full,
        &read(again.embedding_output_gradient()).unwrap()
    ));
    for (old, retained) in parameters_detached
        .iter()
        .zip(detached.parameter_gradients())
    {
        assert!(same(old, &read(retained).unwrap()));
    }
    assert!(embedding_full
        .iter()
        .zip(&embedding_detached)
        .any(|(a, b)| (a - b).abs() > 1e-9));
    let g = plan.parameter_layout().geometry().unwrap();
    let affected = g.projection().start..g.raw_gains().start;
    let mut changed = 0;
    for (slot, ((full, detached), again)) in parameters_full
        .iter()
        .zip(&parameters_detached)
        .zip(again.parameter_gradients())
        .enumerate()
    {
        let (a, b, c) = (full.clone(), detached.clone(), read(again).unwrap());
        assert!(same(&a, &c));
        if slot < 2 || affected.contains(&slot) {
            changed += usize::from(!same(&a, &b));
        } else {
            assert!(same(&a, &b), "unrelated parameter slot {slot} changed");
        }
    }
    assert!(changed > 2);
    for range in g.query_projections() {
        assert!(full.parameter_gradients()[range.clone()]
            .iter()
            .flat_map(|p| read(p).unwrap())
            .any(|v| v.abs() > 1e-9));
    }
    let mut other = plan.compile_training_wgpu(runtime).unwrap();
    let batch = other.prepare_batch(&host).unwrap();
    let foreign = other.forward(&batch).unwrap();
    assert!(matches!(
        model.backward_with_query_pullback(
            &foreign,
            &seed,
            ByteDecoderQueryPullback::StopCoordinates
        ),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ));
    let update = model.sgd(&detached, 0.01).unwrap();
    assert_eq!(update.snapshot().unwrap().read().unwrap(), 1);
    assert!(matches!(
        model.backward_with_query_pullback(
            &forward,
            &seed,
            ByteDecoderQueryPullback::StopCoordinates
        ),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ));
    assert!(matches!(
        model.sgd(&full, 0.01),
        Err(InferenceError::Training(TrainingError::ParameterVersion))
    ));
}

#[test]
fn missing_router_is_rejected_without_poisoning_full_backward() {
    let Some(runtime) = runtime() else { return };
    let base =
        super::super::super::tests::plan(AttentionMask::Causal { query_offset: 0 }, 1).unwrap();
    for geometry in [false, true] {
        let plan = if geometry {
            base.clone()
                .with_causal_geometry(super::super::super::tests::geometry_for(&base))
                .unwrap()
        } else {
            base.clone()
        };
        let mut model = plan.compile_training_wgpu(runtime.clone()).unwrap();
        let batch = model
            .prepare_batch(&ByteLmBatch::from_windows(&[b"abac", b"xyxz"]).unwrap())
            .unwrap();
        let forward = model.forward(&batch).unwrap();
        let seed = model
            .tensor_device()
            .upload(
                model.output_layout().shape(),
                &vec![0.001; model.output_layout().len()],
            )
            .unwrap();
        assert!(matches!(
            model.backward_with_query_pullback(
                &forward,
                &seed,
                ByteDecoderQueryPullback::StopCoordinates
            ),
            Err(InferenceError::ByteDecoder(_))
        ));
        assert!(read(
            model
                .backward(&forward, &seed)
                .unwrap()
                .embedding_output_gradient()
        )
        .is_ok());
    }
}

#[test]
fn stopped_query_cotangent_failure_still_guards_every_parameter_and_atomic_update() {
    let Some(runtime) = runtime() else { return };
    let plan = contextual_plan();
    let mut model = plan.compile_training_wgpu(runtime).unwrap();
    let before: Vec<_> = model
        .parameter_snapshot()
        .values()
        .iter()
        .map(|p| read(p).unwrap())
        .collect();
    let batch = model
        .prepare_batch(&ByteLmBatch::from_windows(&[b"abac", b"xyxz"]).unwrap())
        .unwrap();
    let forward = model.forward(&batch).unwrap();
    let huge = model
        .tensor_device()
        .upload(
            model.output_layout().shape(),
            &vec![f32::MAX; model.output_layout().len()],
        )
        .unwrap();
    let bad = model
        .backward_with_query_pullback(
            &forward,
            &huge.mul(&huge).unwrap(),
            ByteDecoderQueryPullback::StopCoordinates,
        )
        .unwrap();
    for value in bad
        .parameter_gradients()
        .iter()
        .chain([bad.embedding_output_gradient()])
    {
        assert!(matches!(read(value), Err(GpuTensorError::NonFinite)));
    }
    assert!(matches!(
        model.sgd(&bad, 0.).unwrap().snapshot().unwrap().read(),
        Err(TrainingError::Rejected { stage: 0, .. })
    ));
    for (old, now) in before.iter().zip(model.parameter_snapshot().values()) {
        assert!(same(old, &read(now).unwrap()));
    }
}

#[test]
fn stopped_query_training_resumes_exactly_when_the_caller_reapplies_the_policy() {
    let Some(runtime) = runtime() else { return };
    let mut uninterrupted = contextual_plan()
        .compile_training_wgpu(runtime.clone())
        .unwrap();
    let host = ByteLmBatch::from_windows(&[b"abac", b"xyxz"]).unwrap();
    let spec = CrossEntropySpec::new(
        st_kernel_contracts::classification::ClassReduction::Mean,
        -100,
        0.,
    )
    .unwrap();
    let step = |model: &mut ResidentByteDecoder| {
        let batch = model.prepare_batch(&host).unwrap();
        let forward = model.forward(&batch).unwrap();
        let loss = forward.next_byte_loss(spec).unwrap();
        let default = model
            .backward(&forward, loss.prediction_gradient())
            .unwrap();
        assert_eq!(default.query_pullback(), ByteDecoderQueryPullback::Full);
        let stopped = model
            .backward_with_query_pullback(
                &forward,
                loss.prediction_gradient(),
                ByteDecoderQueryPullback::StopCoordinates,
            )
            .unwrap();
        assert!(read(loss.prediction_gradient())
            .unwrap()
            .iter()
            .any(|&v| v != 0.));
        model
            .sgd(&stopped, 0.01)
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap();
    };
    step(&mut uninterrupted);
    let encoded = uninterrupted
        .checkpoint_snapshot()
        .unwrap()
        .read()
        .unwrap()
        .to_json()
        .unwrap();
    let checkpoint = ByteDecoderCheckpoint::from_json(&encoded).unwrap();
    assert_eq!(checkpoint.to_json().unwrap(), encoded);
    let mut resumed = checkpoint.restore_wgpu(runtime).unwrap();
    for _ in 0..2 {
        step(&mut uninterrupted);
        step(&mut resumed);
        assert_eq!(
            uninterrupted
                .checkpoint_snapshot()
                .unwrap()
                .read()
                .unwrap()
                .to_json()
                .unwrap(),
            resumed
                .checkpoint_snapshot()
                .unwrap()
                .read()
                .unwrap()
                .to_json()
                .unwrap(),
        );
    }
}
