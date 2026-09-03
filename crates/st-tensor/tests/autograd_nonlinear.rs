use st_tensor::dlpack::{
    DLManagedTensorVersioned, DlpackCopyPolicy, DlpackExportOptions, DlpackProtocol,
    DLPACK_FLAG_READ_ONLY,
};
use st_tensor::{AutogradTensor, Layout, Tensor};

fn tensor(rows: usize, cols: usize, values: &[f32]) -> Tensor {
    Tensor::from_vec(rows, cols, values.to_vec()).unwrap()
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected}"
        );
    }
}

#[test]
fn unique_native_leaf_reuses_storage_and_detach_shares_its_snapshot() {
    let source = tensor(1, 2, &[2.0, 3.0]);
    let pointer = source.data().as_ptr();
    let leaf = AutogradTensor::variable(source).unwrap();
    assert_eq!(pointer, leaf.value().data().as_ptr());
    assert!(leaf.value().is_snapshot());
    let detached = leaf.detach().unwrap();
    assert_eq!(pointer, detached.value().data().as_ptr());
    assert!(!detached.requires_grad());
    let mut exposed = leaf.value().clone();
    exposed.data_mut()[0] = 100.0;
    assert!(!exposed.is_snapshot());
    assert_eq!(leaf.value().data(), &[2.0, 3.0]);
}

#[test]
fn snapshots_isolate_existing_native_and_exported_aliases() {
    let mut source = tensor(1, 2, &[2.0, 3.0]);
    let snapshot = source.snapshot();
    assert_ne!(source.data().as_ptr(), snapshot.data().as_ptr());
    source.data_mut()[0] = 50.0;
    assert_eq!(snapshot.data(), &[2.0, 3.0]);

    let exported = source
        .export_dlpack(DlpackExportOptions {
            protocol: DlpackProtocol::Legacy,
            copy: DlpackCopyPolicy::Never,
        })
        .unwrap();
    let imported = Tensor::from_managed_dlpack(exported).unwrap();
    let old_pointer = imported.data().as_ptr();
    let snapshot = source.into_snapshot();
    assert_ne!(old_pointer, snapshot.data().as_ptr());
    let foreign_snapshot = imported.into_snapshot();
    assert_ne!(old_pointer, foreign_snapshot.data().as_ptr());
}

#[test]
fn snapshot_exports_obey_read_only_and_copy_negotiation() {
    let snapshot = tensor(1, 2, &[2.0, 3.0]).into_snapshot();
    let handle = snapshot
        .export_dlpack(DlpackExportOptions {
            protocol: DlpackProtocol::Versioned,
            copy: DlpackCopyPolicy::Never,
        })
        .unwrap();
    // SAFETY: the owned handle keeps the versioned header live for this borrow.
    let header = unsafe { &*handle.as_ptr().cast::<DLManagedTensorVersioned>() };
    assert_eq!(header.flags, DLPACK_FLAG_READ_ONLY);
    assert_eq!(
        header.dl_tensor.data.cast_const().cast::<f32>(),
        snapshot.data().as_ptr()
    );
    assert!(snapshot
        .export_dlpack(DlpackExportOptions {
            protocol: DlpackProtocol::Legacy,
            copy: DlpackCopyPolicy::Never,
        })
        .is_err());
    let legacy = snapshot
        .export_dlpack(DlpackExportOptions {
            protocol: DlpackProtocol::Legacy,
            copy: DlpackCopyPolicy::IfNeeded,
        })
        .unwrap();
    let restored = Tensor::from_managed_dlpack(legacy).unwrap();
    assert_ne!(restored.data().as_ptr(), snapshot.data().as_ptr());
    assert_eq!(restored.data(), snapshot.data());
}

#[test]
fn gradient_seeds_and_exposed_gradients_cannot_change_committed_state() {
    let x = AutogradTensor::variable(tensor(1, 2, &[2.0, 3.0])).unwrap();
    let mut seed = tensor(1, 2, &[4.0, 5.0]);
    x.backward_with_grad(&seed).unwrap();
    let previous = x.grad().unwrap();
    assert!(previous.is_snapshot());
    assert_ne!(previous.data().as_ptr(), seed.data().as_ptr());
    seed.data_mut()[0] = 100.0;
    let mut exposed = x.grad().unwrap();
    exposed.data_mut()[1] = 100.0;
    assert_eq!(x.grad().unwrap().data(), &[4.0, 5.0]);
    x.backward_with_grad(&tensor(1, 2, &[1.0, 2.0])).unwrap();
    assert_eq!(previous.data(), &[4.0, 5.0]);
    assert_eq!(x.grad().unwrap().data(), &[5.0, 7.0]);
}

#[test]
fn leaf_layout_is_normalized_before_nonlinear_kernels() {
    let row = tensor(2, 3, &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let col = row.to_layout(Layout::ColMajor).unwrap();
    let x = AutogradTensor::variable(col).unwrap();
    assert_eq!(x.value().layout(), Layout::RowMajor);
    assert_eq!(x.value().data(), row.data());
    x.relu().unwrap().sum().unwrap().backward().unwrap();
    assert_eq!(x.grad().unwrap().data(), &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn relu_zero_convention_and_bias_reduction_are_explicit() {
    let x = AutogradTensor::variable(tensor(2, 3, &[-1.0, 0.0, 2.0, 3.0, -2.0, 4.0])).unwrap();
    let bias = AutogradTensor::variable(tensor(1, 3, &[0.0, 0.0, 0.0])).unwrap();
    let y = x.add_row(&bias).unwrap().relu().unwrap();
    let seed = tensor(2, 3, &[2.0, 4.0, 6.0, 1.0, 3.0, 5.0]);
    let report = y.backward_with_grad(&seed).unwrap();
    assert_eq!(x.grad().unwrap().data(), &[0.0, 0.0, 6.0, 1.0, 0.0, 5.0]);
    assert_eq!(bias.grad().unwrap().data(), &[1.0, 0.0, 11.0]);
    assert_eq!(report.leaf_gradient_count, 2);
    assert!(x.add_row(&x).is_err());
}

#[test]
fn nonlinear_vjps_match_finite_differences_and_do_not_commit() {
    let values = [-2.0, -0.3, 0.8, 1.2, 2.0, -0.7];
    let seed = tensor(2, 3, &[0.2, -0.5, 1.3, 0.7, -1.0, 0.4]);
    for kind in ["relu", "gelu", "softmax"] {
        let operation = |input: &AutogradTensor| match kind {
            "relu" => input.relu().unwrap(),
            "gelu" => input.gelu().unwrap(),
            _ => input.row_softmax().unwrap(),
        };
        let x = AutogradTensor::variable(tensor(2, 3, &values)).unwrap();
        let y = operation(&x);
        let analytic = y.vector_jacobian_product(&x, &seed).unwrap();
        assert!(x.grad().is_none());
        let mut numeric = Vec::new();
        for index in 0..values.len() {
            let evaluate = |delta: f32| {
                let mut perturbed = values;
                perturbed[index] += delta;
                let p = AutogradTensor::constant(tensor(2, 3, &perturbed)).unwrap();
                operation(&p)
                    .value()
                    .data()
                    .iter()
                    .zip(seed.data())
                    .map(|(&value, &weight)| f64::from(value) * f64::from(weight))
                    .sum::<f64>()
            };
            numeric.push(((evaluate(0.001) - evaluate(-0.001)) / 0.002) as f32);
        }
        close(analytic.data(), &numeric, 0.002);
        y.backward_with_grad(&seed).unwrap();
        close(x.grad().unwrap().data(), analytic.data(), 1e-6);
    }
}

#[test]
fn softmax_vjp_couples_classes_and_handles_saturated_logits() {
    let x = AutogradTensor::variable(tensor(1, 3, &[0.0, 0.0, 0.0])).unwrap();
    let y = x.row_softmax().unwrap();
    y.backward_with_grad(&tensor(1, 3, &[1.0, 0.0, 0.0]))
        .unwrap();
    close(
        x.grad().unwrap().data(),
        &[2.0 / 9.0, -1.0 / 9.0, -1.0 / 9.0],
        1e-6,
    );
    let constant_seed = y
        .vector_jacobian_product(&x, &tensor(1, 3, &[f32::MAX; 3]))
        .unwrap();
    assert_eq!(constant_seed.data(), &[0.0, 0.0, 0.0]);
    let x = AutogradTensor::variable(tensor(1, 3, &[1e20, 0.0, -1e20])).unwrap();
    let y = x.row_softmax().unwrap();
    assert_eq!(y.value().data(), &[1.0, 0.0, 0.0]);
    let gradient = y
        .vector_jacobian_product(&x, &tensor(1, 3, &[f32::MAX; 3]))
        .unwrap();
    assert_eq!(gradient.data(), &[0.0, 0.0, 0.0]);
}

#[test]
fn nonlinear_empty_tensors_keep_shapes_and_zero_bias_gradients() {
    for (rows, cols) in [(0, 3), (2, 0), (0, 0)] {
        let x = AutogradTensor::variable(Tensor::zeros(rows, cols).unwrap()).unwrap();
        let bias = AutogradTensor::variable(Tensor::zeros(1, cols).unwrap()).unwrap();
        let y = x
            .add_row(&bias)
            .unwrap()
            .relu()
            .unwrap()
            .gelu()
            .unwrap()
            .row_softmax()
            .unwrap();
        assert_eq!(y.shape(), (rows, cols));
        y.sum().unwrap().backward().unwrap();
        assert_eq!(x.grad().unwrap().shape(), (rows, cols));
        assert_eq!(bias.grad().unwrap().data(), vec![0.0; cols]);
    }
}

#[test]
fn gelu_saturated_derivatives_remain_finite() {
    let x = AutogradTensor::variable(tensor(1, 4, &[-1e30, -10.0, 10.0, 1e30])).unwrap();
    x.gelu()
        .unwrap()
        .backward_with_grad(&tensor(1, 4, &[1.0; 4]))
        .unwrap();
    assert_eq!(x.grad().unwrap().data(), &[0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn nonlinear_backward_failure_preserves_existing_gradients() {
    let x = AutogradTensor::variable(tensor(1, 1, &[1e-38])).unwrap();
    x.sum().unwrap().backward().unwrap();
    let y = x.relu().unwrap().scale(f32::MAX).unwrap();
    assert!(y.backward_with_grad(&tensor(1, 1, &[2.0])).is_err());
    assert_eq!(x.grad().unwrap().data(), &[1.0]);
    assert!(y.grad().is_none());
}

#[test]
fn gelu_backward_rejects_non_finite_inputs_seeds_and_results() {
    use st_tensor::TensorUtilBackend;
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let finite = tensor(1, 1, &[1.0]);
        let bad = tensor(1, 1, &[invalid]);
        assert!(bad
            .gelu_backward_with_backend(&finite, TensorUtilBackend::Cpu)
            .is_err());
        assert!(finite
            .gelu_backward_with_backend(&bad, TensorUtilBackend::Cpu)
            .is_err());
    }
    assert!(tensor(1, 1, &[1.0])
        .gelu_backward_with_backend(&tensor(1, 1, &[f32::MAX]), TensorUtilBackend::Cpu,)
        .is_err());
}

#[test]
fn column_major_seeds_have_the_same_logical_vjp_as_row_major() {
    let values = tensor(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x = AutogradTensor::variable(values.clone()).unwrap();
    let output = x.hadamard(&x).unwrap();
    let seed = values.to_layout(Layout::ColMajor).unwrap();
    let vjp = output.vector_jacobian_product(&x, &seed).unwrap();
    assert_eq!(vjp.data(), &[2.0, 8.0, 18.0, 32.0, 50.0, 72.0]);
    output.backward_with_grad(&seed).unwrap();
    assert_eq!(x.grad().unwrap().data(), vjp.data());
}

#[test]
fn row_bias_gradient_cancels_large_finite_seeds_before_narrowing() {
    let x = AutogradTensor::variable(Tensor::zeros(3, 2).unwrap()).unwrap();
    let bias = AutogradTensor::variable(Tensor::zeros(1, 2).unwrap()).unwrap();
    let output = x.add_row(&bias).unwrap();
    let seed = tensor(
        3,
        2,
        &[
            f32::MAX,
            -f32::MAX,
            f32::MAX,
            -f32::MAX,
            -f32::MAX,
            f32::MAX,
        ],
    );
    let vjp = output.vector_jacobian_product(&bias, &seed).unwrap();
    assert_eq!(vjp.data(), &[f32::MAX, -f32::MAX]);
    assert!(bias.grad().is_none());
    output.backward_with_grad(&seed).unwrap();
    assert_eq!(bias.grad().unwrap().data(), vjp.data());
    assert_eq!(x.grad().unwrap().data(), seed.data());
}

#[test]
fn row_bias_gradient_rejects_final_overflow_without_partial_commit() {
    let x = AutogradTensor::variable(Tensor::zeros(2, 1).unwrap()).unwrap();
    let bias = AutogradTensor::variable(Tensor::zeros(1, 1).unwrap()).unwrap();
    bias.sum().unwrap().backward().unwrap();
    let output = x.add_row(&bias).unwrap();
    let seed = tensor(2, 1, &[f32::MAX, f32::MAX]);
    assert!(output.vector_jacobian_product(&bias, &seed).is_err());
    assert!(output.backward_with_grad(&seed).is_err());
    assert_eq!(bias.grad().unwrap().data(), &[1.0]);
    assert!(x.grad().is_none());
    assert!(output.grad().is_none());
}

#[test]
fn constant_row_bias_does_not_reduce_an_unused_gradient() {
    let x = AutogradTensor::variable(Tensor::zeros(2, 1).unwrap()).unwrap();
    let bias = AutogradTensor::constant(Tensor::zeros(1, 1).unwrap()).unwrap();
    let output = x.add_row(&bias).unwrap();
    let seed = tensor(2, 1, &[f32::MAX, f32::MAX]);
    assert_eq!(
        output.vector_jacobian_product(&x, &seed).unwrap().data(),
        seed.data()
    );
    output.backward_with_grad(&seed).unwrap();
    assert_eq!(x.grad().unwrap().data(), seed.data());
    assert!(bias.grad().is_none());
}
