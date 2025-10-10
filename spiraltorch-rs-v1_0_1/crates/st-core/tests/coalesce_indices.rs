use st_core::{Tensor, ops};

#[test]
fn coalesce_basic() {
    let idx = Tensor::from_i32(ndarray::arr1(&[5,5,2,7,2]).into_dyn());
    let (uniq, remap, k) = ops::coalesce_indices(&idx).unwrap();
    assert_eq!(k, uniq.shape()[0]);
    assert_eq!(remap.shape()[0], 5);
}
