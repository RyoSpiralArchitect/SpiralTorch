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
