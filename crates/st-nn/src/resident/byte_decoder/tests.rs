use super::*;

pub(super) fn plan(
    mask: AttentionMask,
    block_count: usize,
) -> Result<ByteDecoderPlan, InferenceError> {
    plan_with_shape([2, 3, 4], mask, block_count)
}

pub(super) fn plan_with_shape(
    shape: [usize; 3],
    mask: AttentionMask,
    block_count: usize,
) -> Result<ByteDecoderPlan, InferenceError> {
    let layout = NdLayout::contiguous(&shape)?;
    let row = |n, value| Tensor::from_vec(1, n, vec![value; n]).unwrap();
    let branch = InferencePlan::from_operations(layout.clone(), vec![InferenceOp::Relu])?;
    let weight = Tensor::from_vec(4, 4, vec![0.01; 16])?;
    let bias = row(4, 0.);
    let attention =
        AttentionInferencePlan::from_parameters(layout.clone(), 2, mask, [(&weight, &bias); 4])?;
    let block = ResidualAttentionPlan::from_plans(&branch, &attention, &branch)?;
    let head = InferencePlan::from_operations(
        layout,
        vec![InferenceOp::Linear {
            weight: Tensor::from_vec(4, BYTE_LM_VOCAB, vec![0.02; 4 * BYTE_LM_VOCAB])?,
            bias: row(BYTE_LM_VOCAB, 0.),
        }],
    )?;
    ByteDecoderPlan::from_plans(
        &Tensor::from_vec(BYTE_LM_VOCAB, 4, vec![0.1; 4 * BYTE_LM_VOCAB])?,
        &Tensor::from_vec(shape[1] + 2, 4, vec![0.2; (shape[1] + 2) * 4])?,
        &vec![block; block_count],
        &head,
    )
}

#[test]
fn windows_shift_within_each_row_and_keep_every_byte_value() {
    let batch = ByteLmBatch::from_windows(&[&[0, 255, 128, 1], &[9, 8, 7, 6]]).unwrap();
    assert_eq!(batch.shape(), [2, 3]);
    assert_eq!(batch.input_bytes(), [0, 255, 128, 9, 8, 7]);
    assert_eq!(batch.target_bytes(), [255, 128, 1, 8, 7, 6]);
    for windows in [
        vec![],
        vec![&[][..]],
        vec![&[1][..]],
        vec![&[1, 2][..], &[1, 2, 3][..]],
    ] {
        assert!(ByteLmBatch::from_windows(&windows).is_err());
    }
}

#[test]
fn document_selection_cannot_invent_cross_document_transitions() {
    let docs: &[&[u8]] = &[b"AB", b"CD", &[0, 255, 128]];
    let batch = ByteLmBatch::from_documents(docs, &[(0, 0), (1, 0), (2, 1)], 1).unwrap();
    assert_eq!(batch.input_bytes(), [b'A', b'C', 255]);
    assert_eq!(batch.target_bytes(), [b'B', b'D', 128]);
    for (selection, steps) in [
        (vec![(0, 0)], 2),
        (vec![(3, 0)], 1),
        (vec![(0, 1)], 1),
        (vec![(0, usize::MAX)], 1),
        (vec![(0, 0)], usize::MAX),
        (vec![(0, 0)], 0),
        (vec![], 1),
    ] {
        assert!(ByteLmBatch::from_documents(docs, &selection, steps).is_err());
    }
}

#[test]
fn only_full_window_causal_blocks_are_accepted_and_parameter_groups_are_explicit() {
    assert!(plan(AttentionMask::None, 1).is_err());
    assert!(plan(AttentionMask::Causal { query_offset: 1 }, 1).is_err());
    assert!(plan(AttentionMask::Causal { query_offset: 0 }, 0).is_err());
    let plan = plan(AttentionMask::Causal { query_offset: 0 }, 2).unwrap();
    assert_eq!(plan.output_layout().shape(), [2, 3, BYTE_LM_VOCAB]);
    assert_eq!(plan.parameter_layout().blocks(), [2..6, 6..10]);
    assert_eq!(plan.parameter_layout().head(), 10..12);
    assert_eq!(plan.parameter_layout().len(), 12);
    assert_eq!(plan.block_count(), 2);
}

#[test]
fn tables_and_head_must_match_and_plans_freeze_tables() {
    let mut plan = plan(AttentionMask::Causal { query_offset: 0 }, 1).unwrap();
    let token = Tensor::from_vec(255, 4, vec![0.; 255 * 4]).unwrap();
    let position = Tensor::from_vec(5, 4, vec![0.; 20]).unwrap();
    assert!(ByteDecoderPlan::from_plans(&token, &position, &plan.blocks, &plan.head).is_err());
    let mut token = Tensor::from_vec(256, 4, vec![0.5; 1024]).unwrap();
    let short = Tensor::from_vec(2, 4, vec![0.; 8]).unwrap();
    assert!(ByteDecoderPlan::from_plans(&token, &short, &plan.blocks, &plan.head).is_err());
    let frozen = ByteDecoderPlan::from_plans(&token, &position, &plan.blocks, &plan.head).unwrap();
    token.data_mut().fill(9.);
    plan.token.values.fill(8.);
    assert!(frozen.token.values.iter().all(|&v| v == 0.5));
}

pub(super) fn geometry_for(plan: &ByteDecoderPlan) -> ByteDecoderGeometryPlan {
    let projection = InferencePlan::from_operations(
        plan.input_layout().clone(),
        vec![InferenceOp::Linear {
            weight: Tensor::from_vec(4, 4, vec![0.1; 16]).unwrap(),
            bias: Tensor::from_vec(1, 4, vec![0.01; 4]).unwrap(),
        }],
    )
    .unwrap();
    ByteDecoderGeometryPlan::new(
        &projection,
        &[-0.7, 0.4],
        &[-0.2, 0.3],
        &vec![vec![-0.6, 0.3]; plan.block_count()],
        -0.75,
    )
    .unwrap()
}

#[test]
fn headwise_geometry_is_frozen_shape_checked_and_can_be_reset_to_uniform() {
    use ByteDecoderPairMetric::*;
    let p = plan(AttentionMask::Causal { query_offset: 0 }, 2).unwrap();
    let g = geometry_for(&p);
    let mut routing = vec![
        vec![PoincareSquared, CategoricalFisherRaoSquared],
        vec![EuclideanChordSquared, PoincareSquared],
    ];
    let routed = g.clone().with_head_pair_metrics(&routing).unwrap();
    assert_eq!(routed.pair_metric(), None);
    assert_eq!(routed.head_pair_metrics().unwrap(), routing);
    routing[0][0] = EuclideanChordSquared;
    assert_eq!(routed.head_pair_metrics().unwrap()[0][0], PoincareSquared);
    assert_eq!(routed.raw_gains(), g.raw_gains());
    assert_eq!(routed.raw_decay(), g.raw_decay());
    assert_eq!(routed.raw_phase(), g.raw_phase());
    for bad in [
        vec![],
        vec![vec![PoincareSquared; 2]],
        vec![vec![]; 2],
        vec![vec![PoincareSquared], vec![PoincareSquared; 2]],
        vec![vec![PoincareSquared; 2], vec![PoincareSquared; 3]],
    ] {
        assert!(g.clone().with_head_pair_metrics(&bad).is_err());
    }
    for metric in [
        PoincareSquared,
        EuclideanChordSquared,
        CategoricalFisherRaoSquared,
    ] {
        let uniform = routed.clone().with_pair_metric(metric);
        assert_eq!(uniform.pair_metric(), Some(metric));
        assert_eq!(
            uniform.head_pair_metrics().unwrap(),
            vec![vec![metric; 2]; 2]
        );
        let by_grid = g
            .clone()
            .with_head_pair_metrics(&vec![vec![metric; 2]; 2])
            .unwrap();
        let a = p.clone().with_causal_geometry(uniform).unwrap();
        let b = p.clone().with_causal_geometry(by_grid).unwrap();
        assert_eq!(a.parameter_layout(), b.parameter_layout());
        assert_eq!(
            a.initial_checkpoint().to_json().unwrap(),
            b.initial_checkpoint().to_json().unwrap()
        );
    }
}

#[test]
fn causal_geometry_parameters_are_frozen_and_share_explicit_model_slots() {
    let p = plan(AttentionMask::Causal { query_offset: 0 }, 2).unwrap();
    let g = geometry_for(&p);
    let mut decay = vec![-0.7, 0.4];
    let mut gains = vec![vec![-0.6, 0.3]; 2];
    let frozen =
        ByteDecoderGeometryPlan::new(g.projection(), &decay, g.raw_phase(), &gains, -0.75).unwrap();
    decay.fill(9.);
    gains[0].fill(9.);
    assert_eq!(frozen.raw_decay(), [-0.7, 0.4]);
    assert_eq!(frozen.raw_gains()[0], [-0.6, 0.3]);
    let p = p.with_causal_geometry(frozen).unwrap();
    let slots = p.parameter_layout().geometry().unwrap();
    assert_eq!(slots.projection(), 2..4);
    assert_eq!((slots.raw_decay(), slots.raw_phase()), (4, 5));
    assert_eq!(slots.raw_gains(), 6..8);
    assert_eq!(p.parameter_layout().blocks(), [8..12, 12..16]);
    assert_eq!(p.parameter_layout().head(), 16..18);
    assert!(p.with_causal_geometry(g).is_err());
}

#[test]
fn mixture_has_one_router_owner_and_no_phantom_fixed_route() {
    use ByteDecoderPairMetric::*;
    let p = plan(AttentionMask::Causal { query_offset: 0 }, 2).unwrap();
    let g = geometry_for(&p);
    let metrics = [
        PoincareSquared,
        EuclideanChordSquared,
        CategoricalFisherRaoSquared,
    ];
    let mut logits = vec![vec![-0.4, 0.2, 0.6, 0.7, -0.1, 0.3]; 2];
    let scales = vec![vec![1., 0.8, 32., 1., 1.2, 48.]; 2];
    let mixed = g.clone().with_mixture(&metrics, &logits, &scales).unwrap();
    logits[0][0] = 9.;
    assert_eq!(mixed.mixture().unwrap().raw_logits()[0][0], -0.4);
    assert!(mixed.pair_metric().is_none());
    assert!(mixed.head_pair_metrics().is_none());
    let model = p.clone().with_causal_geometry(mixed.clone()).unwrap();
    let slots = model.parameter_layout().geometry().unwrap();
    assert_eq!(slots.raw_gains(), 6..8);
    assert_eq!(slots.mixture_logits(), Some(8..10));
    assert_eq!(slots.all(), 2..10);
    assert_eq!(model.parameter_layout().blocks(), [10..14, 14..18]);
    assert_eq!(model.parameter_layout().head(), 18..20);
    let fixed = mixed.clone().with_pair_metric(PoincareSquared);
    assert!(fixed.mixture().is_none());
    assert!(fixed.head_pair_metrics().is_some());
    let fixed = mixed
        .with_head_pair_metrics(&vec![vec![PoincareSquared, EuclideanChordSquared]; 2])
        .unwrap();
    assert!(fixed.mixture().is_none());
    assert_eq!(
        p.with_causal_geometry(fixed)
            .unwrap()
            .parameter_layout()
            .len(),
        18
    );
    for (metrics, logits, scales) in [
        (
            vec![PoincareSquared],
            vec![vec![0.; 2]; 2],
            vec![vec![1.; 2]; 2],
        ),
        (
            vec![PoincareSquared, PoincareSquared],
            vec![vec![0.; 4]; 2],
            vec![vec![1.; 4]; 2],
        ),
        (metrics.to_vec(), vec![vec![0.; 6]], scales.clone()),
        (metrics.to_vec(), vec![vec![0.; 5]; 2], scales.clone()),
        (metrics.to_vec(), vec![vec![f32::NAN; 6]; 2], scales.clone()),
        (metrics.to_vec(), vec![vec![0.; 6]; 2], vec![vec![0.; 6]; 2]),
        (
            metrics.to_vec(),
            vec![vec![0.; 6]; 2],
            vec![vec![-1.; 6]; 2],
        ),
        (
            metrics.to_vec(),
            vec![vec![0.; 6]; 2],
            vec![vec![f32::INFINITY; 6]; 2],
        ),
    ] {
        assert!(g.clone().with_mixture(&metrics, &logits, &scales).is_err());
    }
}

#[test]
fn mixture_packed_component_volume_is_rejected_before_gpu_allocation() {
    use ByteDecoderPairMetric::*;
    let projection = InferencePlan::from_operations(
        NdLayout::contiguous(&[1, 30000, 4]).unwrap(),
        vec![InferenceOp::Linear {
            weight: Tensor::from_vec(4, 4, vec![0.1; 16]).unwrap(),
            bias: Tensor::from_vec(1, 4, vec![0.; 4]).unwrap(),
        }],
    )
    .unwrap();
    let g = ByteDecoderGeometryPlan::new(&projection, &[0.; 2], &[0.; 2], &[vec![0.; 2]], -0.75)
        .unwrap();
    assert!(g
        .with_mixture(
            &[
                PoincareSquared,
                EuclideanChordSquared,
                CategoricalFisherRaoSquared
            ],
            &[vec![0.; 6]],
            &[vec![1.; 6]]
        )
        .is_err());
}

#[test]
fn geometry_rejects_wrong_shapes_curvature_and_nonfinite_parameters() {
    let p = plan(AttentionMask::Causal { query_offset: 0 }, 2).unwrap();
    let g = geometry_for(&p);
    for c in [0., 1., f32::NAN, f32::NEG_INFINITY] {
        assert!(ByteDecoderGeometryPlan::new(
            g.projection(),
            g.raw_decay(),
            g.raw_phase(),
            g.raw_gains(),
            c
        )
        .is_err());
    }
    for decay in [vec![0.], vec![0., f32::NAN]] {
        assert!(ByteDecoderGeometryPlan::new(
            g.projection(),
            &decay,
            g.raw_phase(),
            g.raw_gains(),
            -1.
        )
        .is_err());
    }
    for gain in [vec![], vec![vec![]], vec![vec![f32::INFINITY; 2]]] {
        assert!(ByteDecoderGeometryPlan::new(
            g.projection(),
            g.raw_decay(),
            g.raw_phase(),
            &gain,
            -1.
        )
        .is_err());
    }
    for gains in [vec![vec![0.; 2]], vec![vec![0.; 1]; 2]] {
        let wrong =
            ByteDecoderGeometryPlan::new(g.projection(), g.raw_decay(), g.raw_phase(), &gains, -1.)
                .unwrap();
        assert!(p.clone().with_causal_geometry(wrong).is_err());
    }
    let other = plan_with_shape([1, 3, 4], AttentionMask::Causal { query_offset: 0 }, 2).unwrap();
    assert!(other.with_causal_geometry(g).is_err());
    let odd = InferencePlan::from_operations(
        p.input_layout().clone(),
        vec![InferenceOp::Linear {
            weight: Tensor::from_vec(4, 3, vec![0.1; 12]).unwrap(),
            bias: Tensor::from_vec(1, 3, vec![0.; 3]).unwrap(),
        }],
    )
    .unwrap();
    assert!(
        ByteDecoderGeometryPlan::new(&odd, &[0.], &[0.], &[vec![0.; 2], vec![0.; 2]], -1.).is_err()
    );
}
