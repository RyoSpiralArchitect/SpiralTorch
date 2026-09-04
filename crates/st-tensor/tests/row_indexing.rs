use st_tensor::{AutogradTensor, Layout, Tensor, TensorUtilBackend};

fn tensor(rows: usize, cols: usize, values: &[f32]) -> Tensor {
    Tensor::from_vec(rows, cols, values.to_vec()).unwrap()
}

#[test]
fn gather_scatter_duplicates_layout_and_adjoint() {
    let weights = tensor(3, 2, &[1., 2., 3., 4., 5., 6.]);
    let seed = tensor(3, 2, &[1., 2., 3., 4., 5., 6.]);
    let ids = [2, 0, 2];
    for layout in [Layout::RowMajor, Layout::ColMajor] {
        let gathered = weights
            .to_layout(layout)
            .unwrap()
            .gather_rows(&ids)
            .unwrap();
        assert_eq!(gathered.data(), &[5., 6., 1., 2., 5., 6.]);
        let scattered = seed
            .to_layout(layout)
            .unwrap()
            .scatter_add_rows(&ids, 3)
            .unwrap();
        assert_eq!(scattered.data(), &[3., 4., 0., 0., 6., 8.]);
        let lhs: f32 = gathered
            .data()
            .iter()
            .zip(seed.data())
            .map(|(a, b)| a * b)
            .sum();
        let rhs: f32 = weights
            .data()
            .iter()
            .zip(scattered.data())
            .map(|(a, b)| a * b)
            .sum();
        assert_eq!(lhs, rhs);
    }
}

#[test]
fn empty_invalid_ids_shapes_and_final_overflow() {
    let input = tensor(2, 0, &[]);
    assert_eq!(input.gather_rows(&[1, 0]).unwrap().shape(), (2, 0));
    assert!(input.gather_rows(&[2]).is_err());
    assert!(input.scatter_add_rows(&[0], 1).is_err());
    assert!(input.scatter_add_rows(&[0, 2], 2).is_err());
    let empty = tensor(0, 3, &[]);
    assert_eq!(empty.scatter_add_rows(&[], 2).unwrap().data(), &[0.; 6]);
    assert!(empty.scatter_add_rows(&[], usize::MAX).is_err());
    assert!(empty.gather_rows(&[0]).is_err());
    let large = tensor(4, 1, &[f32::MAX, f32::MAX, -f32::MAX, -f32::MAX]);
    assert_eq!(large.scatter_add_rows(&[0; 4], 1).unwrap().data(), &[0.]);
    let large = tensor(2, 1, &[f32::MAX; 2]);
    assert!(large.scatter_add_rows(&[0; 2], 1).is_err());
    assert_eq!(
        large
            .scatter_add_rows_scaled_with_backend(&[0; 2], 1, 0.5, TensorUtilBackend::Cpu)
            .unwrap()
            .data(),
        &[f32::MAX]
    );
    let invalid = tensor(1, 1, &[f32::NAN]);
    assert!(invalid.gather_rows(&[0]).is_err());
    assert!(invalid.scatter_add_rows(&[0], 1).is_err());
}

#[test]
fn graph_snapshots_indices_scatter_vjp_and_failure_atomicity() {
    let table = AutogradTensor::variable(tensor(3, 2, &[1., 2., 3., 4., 5., 6.])).unwrap();
    let mut ids = vec![2, 0, 2];
    let output = table.gather_rows(&ids).unwrap();
    ids.fill(1);
    output.sum().unwrap().backward().unwrap();
    assert_eq!(table.grad().unwrap().data(), &[1., 1., 0., 0., 2., 2.]);
    let previous = table.grad().unwrap();
    assert!(output
        .backward_with_grad(&tensor(3, 2, &[f32::MAX; 6]))
        .is_err());
    assert_eq!(table.grad().unwrap().data(), previous.data());
    assert_eq!(output.grad().unwrap().data(), &[1.; 6]);
    let leaf = AutogradTensor::variable(tensor(3, 2, &[1.; 6])).unwrap();
    let scatter = leaf.scatter_add_rows(&[2, 0, 2], 4).unwrap();
    scatter
        .backward_with_grad(&tensor(4, 2, &[1., 2., 3., 4., 5., 6., 7., 8.]))
        .unwrap();
    assert_eq!(leaf.grad().unwrap().data(), &[5., 6., 1., 2., 5., 6.]);
    let frozen = table.detach().unwrap().gather_rows(&[0]).unwrap();
    assert!(!frozen.requires_grad());
}

#[test]
fn tied_embedding_gradient_matches_finite_difference() {
    let values = vec![0.1, 0.3, -0.2, 0.5, 0.4, -0.1];
    let ids = [0, 2, 0];
    let table = AutogradTensor::variable(tensor(3, 2, &values)).unwrap();
    let output = table
        .gather_rows(&ids)
        .unwrap()
        .matmul(&table.transpose().unwrap())
        .unwrap();
    output.sum().unwrap().backward().unwrap();
    let objective = |values: &[f32]| {
        let table = tensor(3, 2, values);
        table
            .gather_rows(&ids)
            .unwrap()
            .matmul(&table.transpose())
            .unwrap()
            .data()
            .iter()
            .sum::<f32>()
    };
    for (index, &gradient) in table.grad().unwrap().data().iter().enumerate() {
        let mut plus = values.clone();
        let mut minus = values.clone();
        plus[index] += 1e-3;
        minus[index] -= 1e-3;
        let numerical = (objective(&plus) - objective(&minus)) / 2e-3;
        assert!(
            (gradient - numerical).abs() < 2e-4,
            "{index}: {gradient} != {numerical}"
        );
    }
}

#[test]
fn explicit_wgpu_gather_and_grouped_scatter() {
    #[cfg(feature = "wgpu_dense")]
    {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let table = tensor(3, 2, &[1., 2., 3., 4., 5., 6.]);
        let ids = [2, 0, 2];
        let gathered = table
            .gather_rows_with_backend(&ids, TensorUtilBackend::GpuWgpu)
            .unwrap();
        assert_eq!(gathered.data(), table.gather_rows(&ids).unwrap().data());
        let scattered = gathered
            .scatter_add_rows_with_backend(&ids, 3, TensorUtilBackend::GpuWgpu)
            .unwrap();
        assert_eq!(
            scattered.data(),
            gathered.scatter_add_rows(&ids, 3).unwrap().data()
        );
        // Work grows with actual token contributions, not vocab * token count.
        let ids: Vec<usize> = (0..2048).map(|i| (i * 17) % 4096).collect();
        let input = tensor(ids.len(), 4, &vec![0.25; ids.len() * 4]);
        let gpu = input
            .scatter_add_rows_with_backend(&ids, 4096, TensorUtilBackend::GpuWgpu)
            .unwrap();
        assert_eq!(
            gpu.data(),
            input.scatter_add_rows(&ids, 4096).unwrap().data()
        );
        let overflow = tensor(2, 1, &[f32::MAX; 2]);
        assert!(overflow
            .scatter_add_rows_with_backend(&[0, 0], 1, TensorUtilBackend::GpuWgpu)
            .is_err());
    }
    #[cfg(not(feature = "wgpu_dense"))]
    assert!(tensor(1, 1, &[1.])
        .gather_rows_with_backend(&[0], TensorUtilBackend::GpuWgpu)
        .is_err());
}

#[test]
#[cfg(feature = "wgpu_dense")]
fn wgpu_ids_above_float_integer_precision_remain_distinct() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let rows = (1 << 24) + 2;
    let mut table = vec![0.; rows];
    table[rows - 2] = 7.;
    table[rows - 1] = 9.;
    let output =
        st_tensor::wgpu_dense::gather_rows(&table, &[rows - 1, rows - 2], rows, 1).unwrap();
    assert_eq!(output, [9., 7.]);
}

#[test]
fn tied_embedding_learns_token_transitions() {
    use st_tensor::{AutogradSgd, CrossEntropyConfig};
    let variable =
        |rows, cols, values: &[f32]| AutogradTensor::variable(tensor(rows, cols, values)).unwrap();
    let mut optimizer = AutogradSgd::new(
        vec![
            variable(
                4,
                3,
                &[
                    0.2, -0.1, 0.3, -0.3, 0.2, 0.1, 0.1, 0.3, -0.2, -0.2, -0.3, 0.2,
                ],
            ),
            variable(3, 3, &[0.2, 0.1, -0.1, -0.2, 0.3, 0.1, 0.1, -0.2, 0.2]),
        ],
        0.15,
    )
    .unwrap();
    let gamma = AutogradTensor::constant(tensor(1, 3, &[1.; 3])).unwrap();
    let beta = AutogradTensor::constant(tensor(1, 3, &[0.; 3])).unwrap();
    let labels = [1, 2, 3, 0, 1, 2, 3, 0];
    let mut initial = 0.;
    for step in 0..=400 {
        let table = optimizer.parameter(0).unwrap();
        let logits = table
            .gather_rows(&[0, 1, 2, 3, 0, 1, 2, 3])
            .unwrap()
            .matmul(&optimizer.parameter(1).unwrap())
            .unwrap()
            .gelu()
            .unwrap()
            .layer_norm_affine(&gamma, &beta, 1e-5)
            .unwrap()
            .matmul(&table.transpose().unwrap())
            .unwrap();
        let loss = logits
            .cross_entropy_with_logits(&labels, CrossEntropyConfig::default())
            .unwrap();
        if step == 0 {
            initial = loss.item_f32().unwrap();
        }
        if step < 400 {
            loss.backward().unwrap();
            optimizer.step().unwrap();
        } else {
            let final_loss = loss.item_f32().unwrap();
            assert!(final_loss < initial * 0.1, "{initial} -> {final_loss}");
            for (row, &label) in logits.value().data().as_chunks::<4>().0.iter().zip(&labels) {
                let prediction = (0..4).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
                assert_eq!(prediction as i64, label);
            }
            println!("tied embedding: {initial} -> {final_loss}, 400 steps");
        }
    }
}
