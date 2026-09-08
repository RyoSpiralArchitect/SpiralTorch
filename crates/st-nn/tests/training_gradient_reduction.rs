use st_nn::{
    layers::Gelu, loss::Loss, module::Module, Linear, LoraLinear, MeanSquaredError, Sequential,
    Tensor,
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

fn check_mse_vjp(mut model: impl Module) {
    let x = Tensor::from_vec(3, 2, vec![0.7, -0.4, -0.2, 0.9, 0.5, 0.8]).unwrap();
    let target = Tensor::from_vec(3, 2, vec![0.4, -0.7, 0.3, -0.1, -0.2, 0.5]).unwrap();
    let seed = MeanSquaredError::new()
        .backward(&model.forward(&x).unwrap(), &target)
        .unwrap();
    let dx = model.backward(&x, &seed).unwrap();
    let analytic = gradients(&model);
    let state = model.state_dict().unwrap();
    let eps = 0.002;
    for (name, parameter) in &state {
        for (index, &original) in parameter.data().iter().enumerate() {
            let mut perturbed = state.clone();
            perturbed.get_mut(name).unwrap().data_mut()[index] = original + eps;
            model.load_state_dict(&perturbed).unwrap();
            let plus = mse(&model, &x, &target);
            perturbed.get_mut(name).unwrap().data_mut()[index] = original - eps;
            model.load_state_dict(&perturbed).unwrap();
            let minus = mse(&model, &x, &target);
            let numeric = ((plus - minus) / f64::from(2. * eps)) as f32;
            close(&[analytic[name].data()[index]], &[numeric], 2e-5);
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
            2e-5,
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
}
