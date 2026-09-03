use st_tensor::dlpack::{DlpackCopyPolicy, DlpackExportOptions};
use st_tensor::{MatmulBackend, PureResult, Tensor};

fn main() -> PureResult<()> {
    let source = Tensor::from_vec(2, 2, vec![1., 2., 3., 4.])?;
    let handle = source.export_dlpack(DlpackExportOptions::default())?;
    let mut imported = Tensor::from_managed_dlpack(handle)?;
    assert_eq!(source.data().as_ptr(), imported.data().as_ptr());

    imported.data_mut()[0] = 10.;
    assert_ne!(source.data().as_ptr(), imported.data().as_ptr());
    assert_eq!(source.data()[0], 1.);

    let copy = source.export_dlpack(DlpackExportOptions {
        copy: DlpackCopyPolicy::Always,
        ..Default::default()
    })?;
    let copied = Tensor::from_managed_dlpack(copy)?;
    assert_eq!(source.data(), copied.data());
    assert_ne!(source.data().as_ptr(), copied.data().as_ptr());

    let product = imported.matmul_with_backend(&copied, MatmulBackend::CpuNaive)?;
    assert_eq!(product.data(), &[16., 28., 15., 22.]);
    println!("DLPack: shared import, isolated mutation, explicit copy, and CPU matmul verified");
    Ok(())
}
