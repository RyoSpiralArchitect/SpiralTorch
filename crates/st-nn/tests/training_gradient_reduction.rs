use st_nn::{
    layers::{
        conv::{Conv1d, Conv2d, Conv3d, Conv4d, Conv6da},
        normalization::{BatchNorm1d, LayerNorm, ZSpaceBatchNorm1d, ZSpaceLayerNorm},
        Gelu,
    },
    loss::Loss,
    module::Module,
    Linear, LoraLinear, MeanSquaredError, Sequential, Tensor,
};
use std::collections::{BTreeMap, HashMap};

fn gradients(model: &impl Module) -> BTreeMap<String, Tensor> {
    let mut result = BTreeMap::new();
    model
        .visit_parameters(&mut |p| {
            result.insert(p.name().to_owned(), p.gradient().expect("gradient").clone());
            Ok(())
        })
        .unwrap();
    result
}

fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
            "element {i}: {a} != {b}"
        );
    }
}

fn mse(model: &impl Module, x: &Tensor, target: &Tensor) -> f64 {
    let prediction = model.forward(x).unwrap();
    prediction
        .data()
        .iter()
        .zip(target.data())
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / prediction.data().len() as f64
}

fn check_mse_vjp<M: Module>(model: M) {
    check_mse_vjp_with_tolerance(model, 2e-5);
}

fn check_mse_vjp_with_tolerance<M: Module>(mut model: M, tolerance: f32) {
    let x = Tensor::from_vec(3, 2, vec![0.7, -0.4, -0.2, 0.9, 0.5, 0.8]).unwrap();
    let target = Tensor::from_vec(3, 2, vec![0.4, -0.7, 0.3, -0.1, -0.2, 0.5]).unwrap();
    let seed = MeanSquaredError::new()
        .backward(&model.forward(&x).unwrap(), &target)
        .unwrap();
    let dx = model.backward(&x, &seed).unwrap();
    let analytic = gradients(&model);
    let state = model.state_dict().unwrap();
    let eps = 0.002;
    for (name, gradient) in &analytic {
        let parameter = &state[name];
        for (index, &original) in parameter.data().iter().enumerate() {
            let mut perturbed = state.clone();
            perturbed.get_mut(name).unwrap().data_mut()[index] = original + eps;
            model.load_state_dict(&perturbed).unwrap();
            let plus = mse(&model, &x, &target);
            perturbed.get_mut(name).unwrap().data_mut()[index] = original - eps;
            model.load_state_dict(&perturbed).unwrap();
            let minus = mse(&model, &x, &target);
            let numeric = ((plus - minus) / f64::from(2. * eps)) as f32;
            let actual = gradient.data()[index];
            assert!(
                actual.is_finite() && numeric.is_finite() && (actual - numeric).abs() <= tolerance,
                "{} {name}[{index}]: analytical={actual}, numerical={numeric}",
                std::any::type_name::<M>()
            );
        }
    }
    model.load_state_dict(&state).unwrap();
    for (index, &original) in x.data().iter().enumerate() {
        let mut perturbed = x.clone();
        perturbed.data_mut()[index] = original + eps;
        let plus = mse(&model, &perturbed, &target);
        perturbed.data_mut()[index] = original - eps;
        let minus = mse(&model, &perturbed, &target);
        close(
            &[dx.data()[index]],
            &[((plus - minus) / f64::from(2. * eps)) as f32],
            tolerance,
        );
    }
}

fn sequential() -> Sequential {
    let mut model = Sequential::new();
    model.push(Linear::new("up", 2, 3).unwrap());
    model.push(Gelu::new());
    model.push(Linear::new("down", 3, 2).unwrap());
    model
}

fn lora() -> LoraLinear {
    let mut model = LoraLinear::new("adapter", 2, 2, 2, 1.5).unwrap();
    let state = HashMap::from([
        (
            "adapter::lora_a".to_owned(),
            Tensor::from_vec(2, 2, vec![0.2, -0.3, 0.4, 0.1]).unwrap(),
        ),
        (
            "adapter::lora_b".to_owned(),
            Tensor::from_vec(2, 2, vec![0.5, -0.2, 0.3, 0.6]).unwrap(),
        ),
    ]);
    model.load_state_dict(&state).unwrap();
    model
}

#[test]
fn linear_mse_parameter_and_input_gradients_are_one_vjp() {
    check_mse_vjp(Linear::new("head", 2, 2).unwrap());
}

#[test]
fn lora_mse_parameter_and_input_gradients_are_one_vjp() {
    check_mse_vjp(lora());
}

#[test]
fn sequential_mse_parameter_and_input_gradients_are_one_vjp() {
    check_mse_vjp(sequential());
}

fn conv2d() -> Conv2d {
    Conv2d::new("conv", 1, 1, (1, 1), (1, 1), (0, 0), (1, 1), (1, 2)).unwrap()
}

fn conv1d() -> Conv1d {
    Conv1d::new("conv1", 1, 1, 1, 1, 0, 1).unwrap()
}

fn conv3d() -> Conv3d {
    Conv3d::new(
        "conv3",
        1,
        1,
        (1, 1, 1),
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
        (1, 1, 2),
    )
    .unwrap()
}

fn conv4d() -> Conv4d {
    Conv4d::new(
        "conv4",
        1,
        1,
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        (1, 1, 1, 2),
    )
    .unwrap()
}

fn conv6da() -> Conv6da {
    Conv6da::new("conv6", 1, 1, (1, 1, 2), 24, 0.0).unwrap()
}

fn layer_norm() -> LayerNorm {
    LayerNorm::new("norm", 2, -1.0, 1e-5).unwrap()
}

fn batch_norm() -> BatchNorm1d {
    BatchNorm1d::new("batch_norm", 2, 0.2, 1e-5).unwrap()
}

fn zspace_layer_norm() -> ZSpaceLayerNorm {
    ZSpaceLayerNorm::new("zspace_norm", 2, -0.9, 1e-5).unwrap()
}

fn zspace_batch_norm() -> ZSpaceBatchNorm1d {
    ZSpaceBatchNorm1d::new("zspace_batch_norm", 2, -0.75, 0.5, 1e-4)
        .unwrap()
        .with_projector_gain(0.6)
        .unwrap()
}

#[test]
fn convolution_family_mse_parameter_and_input_gradients_are_one_vjp() {
    check_mse_vjp(conv1d());
    check_mse_vjp(conv3d());
    check_mse_vjp(conv4d());
    check_mse_vjp(conv6da());
}

#[test]
fn conv2d_mse_parameter_and_input_gradients_are_one_vjp() {
    check_mse_vjp(conv2d());
}

#[test]
fn layer_norm_mse_parameter_and_input_gradients_are_one_vjp() {
    check_mse_vjp(layer_norm());
}

#[test]
fn normalization_family_mse_parameter_and_input_gradients_are_one_vjp() {
    check_mse_vjp_with_tolerance(batch_norm(), 1e-4);
    check_mse_vjp_with_tolerance(zspace_layer_norm(), 1e-4);
    check_mse_vjp_with_tolerance(zspace_batch_norm(), 1e-4);
}

fn check_convolution_overflow_is_rejected(mut model: impl Module) {
    let input = Tensor::from_vec(1, 2, vec![1.0e30; 2]).unwrap();
    let seed = Tensor::from_vec(1, 2, vec![1.0e30; 2]).unwrap();
    assert!(model.backward(&input, &seed).is_err());
    model
        .visit_parameters(&mut |parameter| {
            assert!(
                parameter.gradient().is_none(),
                "{} was updated",
                parameter.name()
            );
            Ok(())
        })
        .unwrap();
}

#[test]
fn overflowing_convolution_vjps_do_not_poison_parameter_gradients() {
    check_convolution_overflow_is_rejected(conv1d());
    check_convolution_overflow_is_rejected(conv2d());
    check_convolution_overflow_is_rejected(conv3d());
    check_convolution_overflow_is_rejected(conv4d());
    check_convolution_overflow_is_rejected(conv6da());
}

fn duplicated_batch_update<M: Module>(make: impl Fn() -> M) {
    let mut reference = None;
    for copies in [1, 2, 5] {
        let mut model = make();
        let x = Tensor::from_vec(2 * copies, 2, [0.7, -0.4, -0.2, 0.9].repeat(copies)).unwrap();
        let target =
            Tensor::from_vec(2 * copies, 2, [0.4, -0.7, 0.3, -0.1].repeat(copies)).unwrap();
        let seed = MeanSquaredError::new()
            .backward(&model.forward(&x).unwrap(), &target)
            .unwrap();
        model.backward(&x, &seed).unwrap();
        model.apply_step(0.125).unwrap();
        let state = model.state_dict().unwrap();
        if let Some(reference) = &reference {
            let reference: &HashMap<String, Tensor> = reference;
            for (name, tensor) in state {
                close(tensor.data(), reference[&name].data(), 1e-6);
            }
        } else {
            reference = Some(state);
        }
    }
}

#[test]
fn mean_loss_updates_are_invariant_to_batch_duplication() {
    duplicated_batch_update(|| Linear::new("head", 2, 2).unwrap());
    duplicated_batch_update(lora);
    duplicated_batch_update(sequential);
    duplicated_batch_update(conv2d);
    duplicated_batch_update(layer_norm);
    duplicated_batch_update(conv1d);
    duplicated_batch_update(conv3d);
    duplicated_batch_update(conv4d);
    duplicated_batch_update(conv6da);
    duplicated_batch_update(batch_norm);
    duplicated_batch_update(zspace_layer_norm);
    duplicated_batch_update(zspace_batch_norm);
}
