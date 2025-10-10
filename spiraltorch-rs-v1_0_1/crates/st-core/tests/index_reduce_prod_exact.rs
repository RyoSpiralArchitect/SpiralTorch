use st_core::{Tensor, ops};

#[test]
fn prod_backward_exact_base_zero_include_self() {
    let base = Tensor::from_array(ndarray::arr2(&[[0.0, 0.0],[2.0, 2.0]]).into_dyn()).requires_grad(true);
    let idx  = Tensor::from_i32(ndarray::arr1(&[0,1]).into_dyn());
    let src  = Tensor::from_array(ndarray::arr2(&[[3.0, 4.0],[5.0, 6.0]]).into_dyn()).requires_grad(true);
    let out = ops::index_reduce(&base, 0, &idx, &src, "prod", true).unwrap();
    let loss = ops::sum(&out).unwrap();
    loss.backward().unwrap();
    let gb = base.grad().unwrap();
    let gs = src.grad().unwrap();
    assert!((gb[[0,0]] - 12.0).abs() < 1e-5 && (gb[[0,1]] - 12.0).abs() < 1e-5);
    assert!((gs[[0,0]] - 0.0).abs() < 1e-5 && (gs[[0,1]] - 0.0).abs() < 1e-5);
    assert!((gb[[1,0]] - 30.0).abs() < 1e-5 && (gb[[1,1]] - 30.0).abs() < 1e-5);
    assert!((gs[[1,0]] - 12.0).abs() < 1e-5 && (gs[[1,1]] - 10.0).abs() < 1e-5);
}
