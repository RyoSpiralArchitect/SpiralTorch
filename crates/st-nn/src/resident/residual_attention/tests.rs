use super::*;

fn norm(shape: &[usize]) -> InferencePlan {
    let width = *shape.last().unwrap();
    InferencePlan::from_operations(
        NdLayout::contiguous(shape).unwrap(),
        vec![InferenceOp::LayerNorm {
            gain: Tensor::from_vec(1, width, vec![1.; width]).unwrap(),
            bias: Tensor::zeros(1, width).unwrap(),
            epsilon: 1e-5,
        }],
    )
    .unwrap()
}

fn attention(shape: &[usize], output: usize) -> AttentionInferencePlan {
    let width = *shape.last().unwrap();
    let projection = Tensor::zeros(width, width).unwrap();
    let out = Tensor::zeros(width, output).unwrap();
    let bias = Tensor::zeros(1, width).unwrap();
    let out_bias = Tensor::zeros(1, output).unwrap();
    AttentionInferencePlan::from_parameters(
        NdLayout::contiguous(shape).unwrap(),
        1,
        AttentionMask::Causal { query_offset: 0 },
        [
            (&projection, &bias),
            (&projection, &bias),
            (&projection, &bias),
            (&out, &out_bias),
        ],
    )
    .unwrap()
}

#[test]
fn composition_preserves_existing_graphs_and_checks_every_residual_edge() {
    let shape = [2, 3, 4];
    let pre = norm(&shape);
    let att = attention(&shape, 4);
    let post = InferencePlan::from_operations(
        NdLayout::contiguous(&shape).unwrap(),
        vec![
            InferenceOp::LayerNorm {
                gain: Tensor::from_vec(1, 4, vec![1.; 4]).unwrap(),
                bias: Tensor::zeros(1, 4).unwrap(),
                epsilon: 1e-5,
            },
            InferenceOp::ToposResonator {
                gate: Tensor::from_vec(1, 4, vec![0.2; 4]).unwrap(),
                kernel: ToposResonatorKernel::new(0.2, 1., 0.3, 3).unwrap(),
                max_volume: 24,
            },
        ],
    )
    .unwrap();
    let plan = ResidualAttentionPlan::from_plans(&pre, &att, &post).unwrap();
    assert_eq!(plan.input_layout(), plan.output_layout());
    assert_eq!(plan.pre.to_json().unwrap(), pre.to_json().unwrap());
    assert_eq!(
        plan.feed_forward.to_json().unwrap(),
        post.to_json().unwrap()
    );
    assert_eq!(
        plan.attention_spec().query_shape(),
        att.attention_spec().query_shape()
    );
    assert_eq!(
        plan.attention_spec().key_shape(),
        att.attention_spec().key_shape()
    );
    assert_eq!(
        plan.attention_spec().merged_output_shape().unwrap(),
        att.attention_spec().merged_output_shape().unwrap()
    );
    assert_eq!(plan.attention_spec().mask(), att.attention_spec().mask());
    assert_eq!(
        plan.attention_spec().scale().to_bits(),
        att.attention_spec().scale().to_bits()
    );
    assert!(ResidualAttentionPlan::from_plans(&norm(&[3, 2, 4]), &att, &post).is_err());
    assert!(ResidualAttentionPlan::from_plans(&pre, &att, &norm(&[3, 2, 4])).is_err());
    assert!(ResidualAttentionPlan::from_plans(&pre, &attention(&shape, 5), &post).is_err());
    let wrong_out = InferencePlan::from_operations(
        NdLayout::contiguous(&shape).unwrap(),
        vec![InferenceOp::Linear {
            weight: Tensor::zeros(4, 5).unwrap(),
            bias: Tensor::zeros(1, 5).unwrap(),
        }],
    )
    .unwrap();
    assert!(ResidualAttentionPlan::from_plans(&pre, &att, &wrong_out).is_err());
}
