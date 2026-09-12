use super::*;

#[test]
fn finite_parameter_cache_tracks_mutation_failure_and_recovery() {
    let mut parameter = Parameter::new("finite", Tensor::from_vec(1, 3, vec![1.; 3]).unwrap());
    parameter.validate_finite("first").unwrap();
    assert!(parameter
        .finite_stamp
        .borrow()
        .as_ref()
        .unwrap()
        .matches(parameter.value()));
    let pointer = parameter.value().data().as_ptr();
    parameter.value_mut().data_mut()[2] = f32::NAN;
    assert_eq!(pointer, parameter.value().data().as_ptr());
    assert!(parameter.finite_stamp.borrow().is_none());
    assert!(
        matches!(parameter.validate_finite("changed"), Err(TensorError::NonFiniteValue { label: "changed", value }) if value.is_nan())
    );
    parameter.value_mut().data_mut()[2] = 5.;
    parameter.validate_finite("recovered").unwrap();
    assert!(parameter.finite_stamp.borrow().is_some());
}

#[test]
fn late_external_write_revokes_finite_cache_and_preserves_error_label() {
    let parameter = Parameter::new("external", Tensor::from_vec(1, 3, vec![1.; 3]).unwrap());
    parameter.validate_finite("first").unwrap();
    let managed = parameter.value().to_dlpack().unwrap();
    let pointer = unsafe { (*managed).dl_tensor.data.cast::<f32>() };
    let _owner = unsafe { Tensor::from_dlpack(managed).unwrap() };
    assert!(!parameter
        .finite_stamp
        .borrow()
        .as_ref()
        .unwrap()
        .matches(parameter.value()));
    unsafe {
        *pointer.add(2) = f32::INFINITY;
    }
    assert!(
        matches!(parameter.validate_finite("late"), Err(TensorError::NonFiniteValue { label: "late", value }) if value == f32::INFINITY)
    );
    assert!(parameter.finite_stamp.borrow().is_none());
    unsafe {
        *pointer.add(2) = 8.;
    }
    parameter.validate_finite("recovered").unwrap();
    assert!(parameter.finite_stamp.borrow().is_none());
    unsafe {
        *pointer = f32::NEG_INFINITY;
    }
    assert!(parameter.validate_finite("again").is_err());
}

#[test]
fn unchanged_packs_reuse_but_mutable_exports_are_never_cached() {
    let parameter = Parameter::new("pack", Tensor::from_vec(3, 5, vec![1.; 15]).unwrap());
    let first = parameter.ensure_matmul_pack().unwrap();
    let second = parameter.ensure_matmul_pack().unwrap();
    assert_eq!(first.as_slice().as_ptr(), second.as_slice().as_ptr());
    let owner = Tensor::from_managed_dlpack(
        parameter
            .value()
            .export_dlpack(st_tensor::dlpack::DlpackExportOptions {
                protocol: st_tensor::dlpack::DlpackProtocol::Versioned,
                copy: st_tensor::dlpack::DlpackCopyPolicy::Never,
            })
            .unwrap(),
    )
    .unwrap();
    parameter.ensure_matmul_pack().unwrap();
    parameter.ensure_matmul_transpose_pack().unwrap();
    assert!(parameter.packed_matmul.borrow().is_none());
    assert!(parameter.packed_matmul_transpose.borrow().is_none());
    drop(owner);
    parameter.ensure_matmul_pack().unwrap();
    assert!(parameter.packed_matmul.borrow().is_none());
}

#[test]
fn value_mut_releases_transpose_pack_before_mutation() {
    let mut parameter = Parameter::new("release", Tensor::from_vec(3, 5, vec![1.; 15]).unwrap());
    parameter.ensure_matmul_transpose_pack().unwrap();
    parameter.validate_finite("finite").unwrap();
    let pointer = parameter.value().data().as_ptr();
    parameter.value_mut().data_mut()[0] = 4.;
    assert_eq!(pointer, parameter.value().data().as_ptr());
    assert!(parameter.packed_matmul_transpose.borrow().is_none());
    assert!(parameter.finite_stamp.borrow().is_none());
}
