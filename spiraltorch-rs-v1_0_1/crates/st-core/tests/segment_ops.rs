use st_core::{Tensor, ops};

#[test]
fn segment_ops_shapes_and_grad() {
    let data = Tensor::from_array(ndarray::arr2(&[[1.0,2.0],[3.0,4.0],[5.0,6.0],[7.0,8.0]]).into_dyn()).requires_grad(true);
    let idx  = Tensor::from_i32(ndarray::arr1(&[0,1,1,2]).into_dyn());
    let ysum = ops::segment_sum(&data, &idx, 3).unwrap();
    let ymean = ops::segment_mean(&data, &idx, 3).unwrap();
    let ymax = ops::segment_max(&data, &idx, 3).unwrap();
    let ymin = ops::segment_min(&data, &idx, 3).unwrap();
    assert_eq!(ysum.shape(), vec![3,2]);
    assert_eq!(ymean.shape(), vec![3,2]);
    assert_eq!(ymax.shape(), vec![3,2]);
    assert_eq!(ymin.shape(), vec![3,2]);
    let l = ops::sum(&ysum).unwrap();
    l.backward().unwrap();
    assert_eq!(data.grad().unwrap().shape(), vec![4,2]);
}
