use st_core::backend::device_caps::DeviceCaps;
use st_nn::{push_backend_policy, BackendPolicy, Linear, LoraLinear, Module, Tensor};
use st_tensor::Layout;
use std::collections::HashMap;

#[derive(Clone, Copy)]
enum Update {
    Sgd,
    Realgrad,
    Hypergrad,
}

fn attach(model: &mut impl Module, update: Update) {
    match update {
        Update::Sgd => (),
        Update::Realgrad => model.attach_realgrad(0.03).unwrap(),
        Update::Hypergrad => model.attach_hypergrad(-1.0, 0.03).unwrap(),
    }
}

#[track_caller]
fn close(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape(), expected.shape());
    let actual = actual.to_layout(Layout::RowMajor).unwrap();
    let expected = expected.to_layout(Layout::RowMajor).unwrap();
    for (index, (&a, &b)) in actual.data().iter().zip(expected.data()).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() < 2e-6,
            "logical element {index}: {a} != {b}"
        );
    }
}

fn compare_state(actual: &impl Module, expected: &impl Module) {
    let actual = actual.state_dict().unwrap();
    let expected = expected.state_dict().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (name, value) in actual {
        close(&value, &expected[&name]);
    }
}

fn seed_gradient(model: &mut impl Module, layout: Layout) {
    model
        .visit_parameters_mut(&mut |parameter| {
            let (rows, cols) = parameter.value().shape();
            let gradient =
                Tensor::from_fn(rows, cols, |r, c| ((r * cols + c) as f32 - 2.0) / 100.0)?;
            parameter.accumulate_euclidean(&gradient.to_layout(layout)?)
        })
        .unwrap();
}

fn train_pair<M: Module>(
    row: &mut M,
    column: &mut M,
    column_parameters: &[&str],
    update: Update,
    existing_gradient: bool,
) {
    column
        .visit_parameters_mut(&mut |parameter| {
            if column_parameters
                .iter()
                .any(|suffix| parameter.name().ends_with(suffix))
            {
                let value = parameter.value().to_layout(Layout::ColMajor)?;
                parameter.load_value(&value)?;
            }
            Ok(())
        })
        .unwrap();
    attach(row, update);
    attach(column, update);
    if existing_gradient {
        seed_gradient(row, Layout::RowMajor);
        seed_gradient(column, Layout::ColMajor);
    }
    let input = Tensor::from_vec(2, 3, vec![0.7, -0.4, 0.2, -0.2, 0.9, 0.5]).unwrap();
    let cotangent =
        Tensor::from_vec(2, 4, vec![0.3, -0.5, 0.2, 0.1, 0.8, 0.2, -0.1, -0.4]).unwrap();
    let initial = row.state_dict().unwrap();
    for _step in 0..2 {
        // Populate packed forward/transpose caches and accumulate microbatches
        // before updating, so layout repair must preserve both kinds of state.
        for scale in [1.0, 0.25] {
            let cotangent = cotangent.scale(scale).unwrap();
            close(
                &column.forward(&input).unwrap(),
                &row.forward(&input).unwrap(),
            );
            close(
                &column.backward(&input, &cotangent).unwrap(),
                &row.backward(&input, &cotangent).unwrap(),
            );
        }
        row.apply_step(0.03).unwrap();
        column.apply_step(0.03).unwrap();
        compare_state(column, row);
        close(
            &column.forward(&input).unwrap(),
            &row.forward(&input).unwrap(),
        );
    }
    assert_ne!(
        row.state_dict().unwrap(),
        initial,
        "updates must not be a no-op"
    );
}

fn lora() -> LoraLinear {
    let mut layer = LoraLinear::new("adapter", 3, 4, 2, 4.0).unwrap();
    layer
        .load_state_dict(&HashMap::from([
            (
                "adapter::lora_a".to_owned(),
                Tensor::from_vec(3, 2, vec![0.2, -0.1, 0.3, 0.1, -0.2, 0.25]).unwrap(),
            ),
            (
                "adapter::lora_b".to_owned(),
                Tensor::from_vec(2, 4, vec![0.1, -0.2, 0.15, 0.3, 0.2, 0.1, -0.1, 0.25]).unwrap(),
            ),
        ]))
        .unwrap();
    layer
}

fn check_lora(update: Update, existing_gradient: bool) {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    for column_parameters in [
        &["::lora_a"][..],
        &["::lora_b"][..],
        &["::lora_a", "::lora_b"][..],
    ] {
        let mut row = lora();
        let mut column = lora();
        let frozen = row.base_state_dict().unwrap();
        train_pair(
            &mut row,
            &mut column,
            column_parameters,
            update,
            existing_gradient,
        );
        assert_eq!(row.base_state_dict().unwrap(), frozen);
        assert_eq!(column.base_state_dict().unwrap(), frozen);
    }
}

fn check_linear(update: Update, existing_gradient: bool) {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let mut row = Linear::new("head", 3, 4).unwrap();
    let mut column = Linear::new("head", 3, 4).unwrap();
    train_pair(
        &mut row,
        &mut column,
        &["::weight", "::bias"],
        update,
        existing_gradient,
    );
}

#[test]
fn lora_sgd_is_parameter_layout_invariant() {
    check_lora(Update::Sgd, false);
}

#[test]
fn lora_realgrad_is_parameter_layout_invariant() {
    check_lora(Update::Realgrad, false);
}

#[test]
fn lora_hypergrad_is_parameter_layout_invariant() {
    check_lora(Update::Hypergrad, false);
}

#[test]
fn lora_preserves_existing_column_major_gradient() {
    check_lora(Update::Sgd, true);
}

#[test]
fn linear_sgd_is_parameter_layout_invariant() {
    check_linear(Update::Sgd, false);
}

#[test]
fn linear_realgrad_is_parameter_layout_invariant() {
    check_linear(Update::Realgrad, false);
}

#[test]
fn linear_hypergrad_is_parameter_layout_invariant() {
    check_linear(Update::Hypergrad, false);
}

#[test]
fn linear_preserves_existing_column_major_gradient() {
    check_linear(Update::Sgd, true);
}
