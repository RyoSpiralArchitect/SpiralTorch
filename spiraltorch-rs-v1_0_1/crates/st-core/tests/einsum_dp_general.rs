use st_core::{Tensor, ops};

#[test]
fn einsum_dp_general_with_batch() {
    let a = Tensor::from_array(ndarray::Array::from_shape_vec((2,3,4), (0..(2*3*4)).map(|v| v as f32 * 0.1).collect()).unwrap().into_dyn()).requires_grad(true);
    let b = Tensor::from_array(ndarray::Array::from_shape_vec((4,5), (0..(4*5)).map(|v| v as f32 * 0.01).collect()).unwrap().into_dyn()).requires_grad(true);
    let y_opt = ops::einsum_opt("bij,jk->bik", &[a.clone(), b.clone()], true).unwrap();
    let y_nv  = ops::einsum("bij,jk->bik", &[a.clone(), b.clone()]).unwrap();
    assert_eq!(y_opt.shape(), y_nv.shape());
    let s1 = ops::sum(&y_opt).unwrap(); s1.backward().unwrap();
    let s2 = ops::sum(&y_nv).unwrap(); s2.backward().unwrap();
    assert_eq!(a.grad().unwrap().shape(), vec![2,3,4]);
    assert_eq!(b.grad().unwrap().shape(), vec![4,5]);
}
