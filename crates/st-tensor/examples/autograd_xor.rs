//! A bounded nonlinear learning fixture, not a language-model quality benchmark.
use st_tensor::{AutogradTensor, PureResult, Tensor};

fn predict(input: &AutogradTensor, parameters: &[AutogradTensor]) -> PureResult<AutogradTensor> {
    input
        .matmul(&parameters[0])?
        .add_row(&parameters[1])?
        .gelu()?
        .matmul(&parameters[2])?
        .add_row(&parameters[3])?
        .row_softmax()
}

fn main() -> PureResult<()> {
    let input = AutogradTensor::constant(Tensor::from_vec(
        4,
        2,
        vec![0., 0., 0., 1., 1., 0., 1., 1.],
    )?)?;
    let target = AutogradTensor::constant(Tensor::from_vec(
        4,
        2,
        vec![1., 0., 0., 1., 0., 1., 1., 0.],
    )?)?;
    let mut parameters = vec![
        AutogradTensor::variable(Tensor::random_normal(2, 8, 0., 0.6, Some(19))?)?,
        AutogradTensor::variable(Tensor::zeros(1, 8)?)?,
        AutogradTensor::variable(Tensor::random_normal(8, 2, 0., 0.6, Some(29))?)?,
        AutogradTensor::variable(Tensor::zeros(1, 2)?)?,
    ];
    let initial_loss = predict(&input, &parameters)?
        .mean_squared_error(&target)?
        .item_f32()?;
    for _ in 0..600 {
        let loss = predict(&input, &parameters)?.mean_squared_error(&target)?;
        let report = loss.backward()?;
        assert_eq!(report.leaf_gradient_count, 4);
        for parameter in &mut parameters {
            let gradient = parameter.grad().expect("trainable parameter gradient");
            let updated = parameter.value().sub(&gradient.scale(0.8)?)?;
            *parameter = AutogradTensor::variable(updated)?;
        }
    }
    let output = predict(&input, &parameters)?;
    let final_loss = output.mean_squared_error(&target)?.item_f32()?;
    let predicted: Vec<usize> = output
        .value()
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|row| usize::from(row[1] > row[0]))
        .collect();
    assert!(final_loss < 0.02 && final_loss < initial_loss / 10.0);
    assert_eq!(predicted, vec![0, 1, 1, 0]);
    println!(
        "{}",
        serde_json::json!({
            "fixture": "xor_nonlinear_autograd",
            "steps": 600,
            "initial_mse": initial_loss,
            "final_mse": final_loss,
            "predicted": predicted,
            "semantic_owner": "st-tensor",
            "evidence_scope": "four-example learning fixture; not an HF/FT benchmark"
        })
    );
    Ok(())
}
