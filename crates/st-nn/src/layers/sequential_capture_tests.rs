use super::*;
use crate::Dropout;

#[test]
fn backward_uses_the_mask_that_produced_the_loss() {
    let mut sequence = Sequential::new();
    sequence.push(Dropout::with_seed(0.5, Some(47)).unwrap());
    let input = Tensor::from_vec(4, 32, vec![1.0; 128]).unwrap();
    let output = sequence.forward(&input).unwrap();
    let upstream = Tensor::from_vec(4, 32, vec![1.0; 128]).unwrap();
    let actual = sequence.backward(&input, &upstream).unwrap();
    assert_eq!(actual.data(), output.data());
    let repeated = sequence.backward(&input, &upstream).unwrap();
    assert_eq!(repeated.data(), output.data());
}
