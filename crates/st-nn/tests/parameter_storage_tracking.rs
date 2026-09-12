use st_nn::{module::Parameter, Tensor};
use st_tensor::{PackedB, Tile};

#[test]
fn mutable_export_after_pack_creation_refreshes_parameter_packs() {
    let parameter = Parameter::new(
        "shared",
        Tensor::from_vec(3, 5, (0..15).map(|i| i as f32).collect()).unwrap(),
    );
    let before = parameter.ensure_matmul_pack().unwrap();
    let _transpose = parameter.ensure_matmul_transpose_pack().unwrap();
    let managed = parameter.value().to_dlpack().unwrap();
    let pointer = unsafe { (*managed).dl_tensor.data.cast::<f32>() };
    let _owner = unsafe { Tensor::from_dlpack(managed).unwrap() };
    // The producer write is serialized between calls, not concurrent with reads.
    unsafe { *pointer.add(14) = 99. };
    let after = parameter.ensure_matmul_pack().unwrap();
    let expected = PackedB::from_tensor(parameter.value(), Tile::col_major()).unwrap();
    assert_eq!(after.as_slice(), expected.as_slice());
    assert_eq!(before.as_slice()[14], 14.);
    let transposed = parameter.ensure_matmul_transpose_pack().unwrap();
    let expected = PackedB::from_tensor_transpose(parameter.value(), Tile::col_major()).unwrap();
    assert_eq!(transposed.as_slice(), expected.as_slice());
}
