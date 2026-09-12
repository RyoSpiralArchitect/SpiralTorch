use super::*;

fn value() -> Tensor {
    Tensor::from_vec(2, 3, vec![-1., 2., 3., 4., 5., 6.]).unwrap()
}

#[test]
fn weak_stamp_does_not_retain_values_or_copy_single_owner_mutations() {
    let mut tensor = value();
    let pointer = tensor.data().as_ptr();
    let stamp = tensor.content_stamp().unwrap();
    assert_eq!(Arc::strong_count(&tensor.data), 1);
    assert!(stamp.matches(&tensor));
    tensor.data_mut()[5] = 99.;
    assert_eq!(tensor.data().as_ptr(), pointer);
    assert!(!stamp.matches(&tensor));
    assert!(stamp.storage.upgrade().is_none());
    let current = tensor.content_stamp().unwrap();
    assert!(current.matches(&tensor));
    drop(tensor);
    assert!(current.storage.upgrade().is_none());
    assert!(!current.matches(&value()));
}

#[test]
fn clone_mutation_detaches_without_revoking_unchanged_sibling() {
    let tensor = value();
    let stamp = tensor.content_stamp().unwrap();
    let mut sibling = tensor.clone();
    assert!(stamp.matches(&sibling));
    sibling.data_mut()[0] = 10.;
    assert!(stamp.matches(&tensor));
    assert!(!stamp.matches(&sibling));
    assert_eq!(tensor.data()[0], -1.);
    let managed = sibling.to_dlpack().unwrap();
    let pointer = unsafe { (*managed).dl_tensor.data.cast::<f32>() };
    let _owner = unsafe { Tensor::from_dlpack(managed).unwrap() };
    unsafe {
        *pointer = 20.;
    }
    assert!(stamp.matches(&tensor));
    assert_eq!(tensor.data()[0], -1.);
}

#[test]
fn owned_buffer_cow_clone_never_creates_untracked_mutable_aliases() {
    let tensor = value();
    let stamp = tensor.content_stamp().unwrap();
    // Exercise the internal clone boundary directly, before make_mut_slice.
    let sibling = Tensor {
        data: Arc::new((*tensor.data).clone()),
        rows: 2,
        cols: 3,
        layout: Layout::RowMajor,
    };
    assert_ne!(tensor.data().as_ptr(), sibling.data().as_ptr());
    let managed = sibling.to_dlpack().unwrap();
    let pointer = unsafe { (*managed).dl_tensor.data.cast::<f32>() };
    let _owner = unsafe { Tensor::from_dlpack(managed).unwrap() };
    unsafe {
        *pointer = 30.;
    }
    assert!(stamp.matches(&tensor));
    assert_eq!(tensor.data()[0], -1.);
}

#[test]
fn late_mutable_exports_revoke_existing_stamps_through_all_aliases() {
    for versioned in [false, true] {
        let tensor = value();
        let alias = tensor.clone();
        let stamp = tensor.content_stamp().unwrap();
        let managed = alias
            .export_dlpack(DlpackExportOptions {
                protocol: if versioned {
                    DlpackProtocol::Versioned
                } else {
                    DlpackProtocol::Legacy
                },
                copy: DlpackCopyPolicy::Never,
            })
            .unwrap();
        assert!(!stamp.matches(&tensor));
        assert!(tensor.content_stamp().is_none());
        assert!(alias.content_stamp().is_none());
        drop(managed);
        assert!(tensor.content_stamp().is_none());
    }
}

#[test]
fn copied_and_failed_exports_do_not_revoke_source_stamps() {
    let tensor = value();
    let stamp = tensor.content_stamp().unwrap();
    let managed = tensor.to_dlpack_versioned(true).unwrap();
    let pointer = unsafe { (*managed).dl_tensor.data.cast::<f32>() };
    let _owner = unsafe { Tensor::from_dlpack_versioned(managed).unwrap() };
    unsafe {
        *pointer = 99.;
    }
    assert!(stamp.matches(&tensor));
    assert_eq!(tensor.data()[0], -1.);
    let column_major = tensor.to_layout(Layout::ColMajor).unwrap();
    let stamp = column_major.content_stamp().unwrap();
    assert!(column_major.to_dlpack().is_err());
    assert!(stamp.matches(&column_major));
}

#[test]
fn snapshot_exports_and_foreign_readonly_imports_have_distinct_trust() {
    let mut snapshot = value().into_snapshot();
    let stamp = snapshot.content_stamp().unwrap();
    let foreign = Tensor::from_managed_dlpack(
        snapshot
            .export_dlpack(DlpackExportOptions {
                protocol: DlpackProtocol::Versioned,
                copy: DlpackCopyPolicy::Never,
            })
            .unwrap(),
    )
    .unwrap();
    assert!(stamp.matches(&snapshot));
    assert!(foreign.content_stamp().is_none());
    let protected = foreign.snapshot();
    assert!(protected.content_stamp().unwrap().matches(&protected));
    assert_ne!(protected.data().as_ptr(), foreign.data().as_ptr());
    let legacy_copy = unsafe { Tensor::from_dlpack(snapshot.to_dlpack().unwrap()).unwrap() };
    assert_ne!(legacy_copy.data().as_ptr(), snapshot.data().as_ptr());
    assert!(stamp.matches(&snapshot));
    assert!(snapshot
        .export_dlpack(DlpackExportOptions {
            protocol: DlpackProtocol::Legacy,
            copy: DlpackCopyPolicy::Never,
        })
        .is_err());
    assert!(stamp.matches(&snapshot));
    snapshot.data_mut()[0] = 42.;
    assert!(!stamp.matches(&snapshot));
    assert_eq!(foreign.data()[0], -1.);
    assert_eq!(protected.data()[0], -1.);
}

#[test]
fn rust_mutation_of_exported_storage_gets_a_private_tracked_copy() {
    let mut tensor = value();
    let managed = tensor.to_dlpack().unwrap();
    let pointer = unsafe { (*managed).dl_tensor.data.cast::<f32>() };
    let owner = unsafe { Tensor::from_dlpack(managed).unwrap() };
    assert!(tensor.content_stamp().is_none());
    tensor.data_mut()[0] = 77.;
    let stamp = tensor.content_stamp().unwrap();
    unsafe {
        *pointer = 99.;
    }
    assert!(stamp.matches(&tensor));
    assert_eq!(tensor.data()[0], 77.);
    assert_eq!(owner.data()[0], 99.);
}

#[test]
fn shape_layout_and_mutating_operations_invalidate_stamps() {
    let tensor = value();
    let stamp = tensor.content_stamp().unwrap();
    assert!(!stamp.matches(&tensor.view(3, 2).unwrap()));
    assert!(!stamp.matches(&tensor.to_layout(Layout::ColMajor).unwrap()));
    let mut metadata_only = tensor.clone();
    metadata_only.layout = Layout::ColMajor;
    assert!(!stamp.matches(&metadata_only));
    for operation in 0..4 {
        let mut tensor = value();
        let stamp = tensor.content_stamp().unwrap();
        match operation {
            0 => tensor
                .add_scaled_with_backend(&value(), 0.5, TensorUtilBackend::Cpu)
                .unwrap(),
            1 => tensor.relu_inplace(),
            2 => tensor.gelu_inplace(),
            _ => tensor
                .add_row_inplace_with_backend(&[1., 2., 3.], TensorUtilBackend::Cpu)
                .unwrap(),
        }
        assert!(!stamp.matches(&tensor), "operation={operation}");
        assert!(tensor.content_stamp().unwrap().matches(&tensor));
    }
}

#[test]
fn empty_allocations_never_match_by_shared_data_pointer_alone() {
    let a = Tensor::zeros(0, 3).unwrap();
    let stamp = a.content_stamp().unwrap();
    let b = Tensor::zeros(0, 3).unwrap();
    assert!(stamp.matches(&a));
    assert!(!stamp.matches(&b));
    drop(a);
    assert!(!stamp.matches(&Tensor::zeros(0, 3).unwrap()));
}
