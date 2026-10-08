use st_tensor::{AutogradTensor, PureResult, Tensor};

fn main() -> PureResult<()> {
    let x = AutogradTensor::variable(Tensor::from_vec(1, 2, vec![1.0, -2.0])?)?;
    x.hadamard(&x)?.sum()?.backward()?;
    let gradient = x.grad().expect("leaf gradient");
    assert_eq!(gradient.data(), &[2.0, -4.0]);
    println!("{:?}", gradient.data());
    Ok(())
}
