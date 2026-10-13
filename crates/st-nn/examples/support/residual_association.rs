//! Distinguish a completed Linear projection from a reassociated residual add.
use super::*;

pub(super) async fn run(device: &TensorDevice) -> Result<Value> {
    let mut cases = Vec::new();
    for exponent in [24, 48] {
        for sign in [-1., 1.] {
            let large = sign * 2f32.powi(exponent);
            let layout = NdLayout::contiguous(&[1, 1, 1])?;
            let scalar = |v| Tensor::from_vec(1, 1, vec![v]);
            let zero = scalar(0.)?;
            let one = scalar(1.)?;
            let weight = scalar(large)?;
            let bias = scalar(-large)?;
            let pre = InferencePlan::from_operations(
                layout.clone(),
                vec![InferenceOp::Linear {
                    weight: one.clone(),
                    bias: zero.clone(),
                }],
            )?;
            let attention = AttentionInferencePlan::from_parameters(
                layout.clone(),
                1,
                AttentionMask::None,
                [
                    (&zero, &zero),
                    (&zero, &zero),
                    (&zero, &one),
                    (&weight, &bias),
                ],
            )?;
            let feed = InferencePlan::from_operations(
                layout,
                vec![InferenceOp::Linear {
                    weight: zero.clone(),
                    bias: zero,
                }],
            )?;
            let mut model = ResidualAttentionPlan::from_plans(&pre, &attention, &feed)?
                .compile_training_wgpu(device.runtime().clone())?;
            let input = device.upload(&[1, 1, 1], &[sign])?;
            let forward = model.forward(&input, None, None)?;
            if read(forward.prediction()).await? != [sign] {
                return Err("projection bias moved across residual boundary".into());
            }
            // Separate resident adds deliberately evaluate the other graph.
            let reassociated = input
                .add(&device.upload(&[1, 1, 1], &[large])?)?
                .add(&device.upload(&[1, 1, 1], &[-large])?)?;
            if read(&reassociated).await? != [0.] {
                return Err("residual association negative control is inert".into());
            }
            let gradient = model.backward(&forward, &device.upload(&[1, 1, 1], &[2.])?)?;
            if gradient.parameter_gradients().len() != 8
                || read(gradient.input_gradient()).await? != [2.]
                || read(&gradient.parameter_gradients()[5]).await? != [2.]
            {
                return Err("residual or projection-bias derivative lost".into());
            }
            cases.push(json!({"exponent":exponent,"sign":sign,"output":sign,
                "reassociated_control":0.,"input_gradient":2.,"output_bias_gradient":2.,"passed":true}));
        }
    }
    Ok(json!({"contract":"projection_bias_before_residual.v1","cases":cases}))
}
