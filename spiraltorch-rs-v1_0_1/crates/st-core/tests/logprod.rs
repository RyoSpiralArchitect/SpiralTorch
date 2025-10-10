use st_core::{Tensor, ops};

#[test]
fn logprod_sign_and_grad() {
    let x = Tensor::from_array(ndarray::arr2(&[[1.0, -2.0, 4.0],[0.5, -1.0, 0.25]]).into_dyn()).requires_grad(true);
    let (logabs, sign) = ops::logprod(&x, 1, false, 1e-12, "propagate", "propagate").unwrap();
    assert_eq!(logabs.shape(), vec![2]);
    assert_eq!(sign.shape(), vec![2]);
    let s = ops::sum(&logabs).unwrap(); s.backward().unwrap();
    assert_eq!(x.grad().unwrap().shape(), x.shape());
}
