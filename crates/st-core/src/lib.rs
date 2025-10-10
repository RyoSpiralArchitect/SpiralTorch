pub mod device;
pub mod dtype;
pub mod error;
pub mod backend;
pub mod autograd;
pub mod tensor;
pub mod ops;

#[cfg(test)]
mod tests {
    use crate::{tensor::Tensor, ops::{matmul, reductions}};
    #[test]
    fn smoke() {
        let a = Tensor::ones(&[2,3]).requires_grad(true);
        let b = Tensor::ones(&[3,4]).requires_grad(true);
        let y = matmul::matmul2d(&a, &b).unwrap();
        reductions::sum_all(&y).unwrap().backward().unwrap();
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
    }
}
