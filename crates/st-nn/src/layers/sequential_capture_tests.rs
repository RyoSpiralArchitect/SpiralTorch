use super::*;
use crate::layers::spiral_rnn::SpiralRnn;
use crate::layers::wave_scan::{WaveScan, WaveScanStack};
use crate::layers::Identity;
use crate::{
    BatchNorm1d, Dropout, Gelu, GradientBands, Linear, Lstm, ToposResonator, WaveRnn,
    ZSpaceCoherenceScan, ZSpaceCoherenceWaveBlock,
};
use std::cell::Cell;
use std::rc::Rc;

#[test]
fn backward_uses_the_mask_that_produced_the_loss() {
    let _guard = crate::test_global_state_lock();
    let mut sequence = Sequential::new();
    sequence.push(Dropout::with_seed(0.5, Some(47)).unwrap());
    let input = Tensor::from_vec(4, 32, vec![1.0; 128]).unwrap();
    let output = sequence.forward(&input).unwrap();
    let upstream = Tensor::from_vec(4, 32, vec![1.0; 128]).unwrap();
    let actual = sequence.backward(&input, &upstream).unwrap();
    assert_eq!(actual.data(), output.data());
    let repeated = sequence.backward(&input, &upstream).unwrap();
    assert_eq!(repeated.data(), output.data());
}

#[test]
fn batchnorm_pullbacks_do_not_update_running_statistics() {
    let _guard = crate::test_global_state_lock();
    let mut sequence = Sequential::new();
    sequence.push(BatchNorm1d::new("norm", 2, 0.1, 1e-5).unwrap());
    let input = Tensor::from_vec(3, 2, vec![1., -2., 3., 1., 0., 2.]).unwrap();
    let initial = sequence.state_dict().unwrap();
    sequence.forward(&input).unwrap();
    let after_forward = sequence.state_dict().unwrap();
    assert_ne!(
        initial["norm::running_mean"],
        after_forward["norm::running_mean"]
    );
    for _ in 0..2 {
        sequence.backward(&input, &input).unwrap();
        let after_backward = sequence.state_dict().unwrap();
        for name in ["norm::running_mean", "norm::running_var"] {
            assert_eq!(after_backward[name], after_forward[name]);
        }
    }
}

#[test]
fn lstm_sequence_matches_direct_layer_without_advancing_state_on_pullback() {
    let _guard = crate::test_global_state_lock();
    let mut reference = Lstm::new("recurrent", 2, 3).unwrap();
    let mut sequence = Sequential::new();
    sequence.push(Lstm::new("recurrent", 2, 3).unwrap());
    sequence
        .load_state_dict(&reference.state_dict().unwrap())
        .unwrap();
    let input = Tensor::from_vec(3, 2, vec![0.1, -0.2, 0.3, 0.1, 0., 0.2]).unwrap();
    let upstream = Tensor::from_vec(3, 3, vec![0.25; 9]).unwrap();
    for _ in 0..2 {
        assert_eq!(
            sequence.forward(&input).unwrap(),
            reference.forward(&input).unwrap()
        );
        for _ in 0..2 {
            assert_eq!(
                sequence.backward(&input, &upstream).unwrap(),
                reference.backward_retained(&input, &upstream).unwrap()
            );
        }
    }
}

fn consuming_layers() -> Vec<Box<dyn Module>> {
    vec![
        Box::new(Lstm::new("lstm", 6, 2).unwrap()),
        Box::new(SpiralRnn::new("spiral", 2, 2, 3).unwrap()),
        Box::new(WaveRnn::new("rnn", 2, 2, 1, 1, 0, -1., 1.).unwrap()),
        Box::new(WaveScan::new("scan", 2, 2, 1, 1, 0, 1, -1., 1.).unwrap()),
        Box::new(
            WaveScanStack::new(vec![
                WaveScan::new("a", 2, 2, 1, 1, 0, 1, -1., 1.).unwrap(),
                WaveScan::new("b", 2, 2, 1, 1, 0, 2, -1., 1.).unwrap(),
            ])
            .unwrap(),
        ),
        Box::new(ZSpaceCoherenceScan::new(2, 3, 3, -1., 1.).unwrap()),
        Box::new(ZSpaceCoherenceWaveBlock::new(2, 3, 3, -1., 1., 1, vec![1, 2]).unwrap()),
    ]
}

fn gradients(module: &dyn Module) -> HashMap<String, Tensor> {
    let mut result = HashMap::new();
    module
        .visit_parameters(&mut |p| {
            if let Some(g) = p.gradient() {
                result.insert(p.name().to_owned(), g.clone());
            }
            Ok(())
        })
        .unwrap();
    result
}

#[test]
fn every_consuming_layer_reuses_its_tape_in_nested_sequences_and_band_pullbacks() {
    let _guard = crate::test_global_state_lock();
    let input = Tensor::from_fn(2, 6, |r, c| (r * 6 + c) as f32 * 0.03 - 0.1).unwrap();
    let seed = Tensor::from_vec(2, 2, vec![0.25, -0.2, 0.125, 0.3]).unwrap();
    for (index, (mut reference, layer)) in consuming_layers()
        .into_iter()
        .zip(consuming_layers())
        .enumerate()
    {
        let mut child = Sequential::new();
        child.push_boxed(layer);
        let mut sequence = Sequential::new();
        sequence.push(child);
        sequence
            .load_state_dict(&reference.state_dict().unwrap())
            .unwrap();
        assert_eq!(
            sequence.forward(&input).unwrap(),
            reference.forward(&input).unwrap()
        );
        let expected = reference.backward(&input, &seed).unwrap();
        assert!(
            reference.backward(&input, &seed).is_err(),
            "one-shot contract, layer {index}"
        );
        let expected_parameters = gradients(reference.as_ref());
        for _ in 0..2 {
            sequence.zero_accumulators().unwrap();
            assert_eq!(
                sequence.backward(&input, &seed).unwrap(),
                expected,
                "layer {index}"
            );
            assert_eq!(gradients(&sequence), expected_parameters, "layer {index}");
        }
        sequence.zero_accumulators().unwrap();
        let bands =
            GradientBands::from_components(seed.clone(), seed.clone(), seed.clone()).unwrap();
        let actual = sequence.backward_bands(&input, &bands).unwrap();
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - 3. * expected).abs() < 1e-5, "layer {index}");
        }
        sequence.forward_untracked(&input).unwrap();
        assert!(sequence.backward(&input, &seed).is_err());
    }
}

#[test]
fn gradient_clearing_preserves_nested_topos_forward_but_parameter_updates_do_not() {
    let _guard = crate::test_global_state_lock();
    let mut child = Sequential::new();
    child.push(ToposResonator::new("topos", 2, 3).unwrap());
    let mut sequence = Sequential::new();
    sequence.push(child);
    let input = Tensor::from_vec(2, 3, vec![0.25; 6]).unwrap();
    sequence.forward(&input).unwrap();
    sequence.zero_accumulators().unwrap();
    let first = sequence.backward(&input, &input).unwrap();
    let first_parameters = gradients(&sequence);
    sequence.zero_accumulators().unwrap();
    assert_eq!(sequence.backward(&input, &input).unwrap(), first);
    assert_eq!(gradients(&sequence), first_parameters);
    sequence.apply_step(0.01).unwrap();
    assert!(sequence.backward(&input, &input).is_err());
}

#[test]
fn optimizer_rate_scaling_is_prevalidated_and_preserves_the_forward_tape() {
    let _guard = crate::test_global_state_lock();
    let mut child = Sequential::new();
    child.push(Linear::new("first", 1, 1).unwrap());
    child.push(ToposResonator::new("last", 1, 1).unwrap());
    let mut sequence = Sequential::new();
    sequence.push(child);
    sequence
        .visit_parameters_mut(&mut |p| {
            p.attach_realgrad(if p.name().starts_with("last") {
                f32::MAX
            } else {
                0.01
            })
        })
        .unwrap();
    let rates = |model: &Sequential| {
        let mut rates = Vec::new();
        model
            .visit_parameters(&mut |p| {
                rates.push(p.realgrad().unwrap().learning_rate());
                Ok(())
            })
            .unwrap();
        rates
    };
    let input = Tensor::from_vec(1, 1, vec![0.25]).unwrap();
    sequence.forward(&input).unwrap();
    let before = rates(&sequence);
    for factor in [2., 0., f32::NAN, f32::INFINITY] {
        assert!(sequence.scale_learning_rates(factor).is_err());
        assert_eq!(rates(&sequence), before);
    }
    sequence.scale_learning_rates(0.5).unwrap();
    assert_eq!(
        rates(&sequence),
        before.iter().map(|v| v * 0.5).collect::<Vec<_>>()
    );
    sequence.backward(&input, &input).unwrap();
}

struct Counted {
    forwards: Rc<Cell<usize>>,
    untracked: Rc<Cell<usize>>,
}

impl Module for Counted {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.forwards.set(self.forwards.get() + 1);
        if input.data().iter().any(|x| !x.is_finite()) {
            return Err(TensorError::InvalidValue {
                label: "test_forward_failure",
            });
        }
        Ok(input.clone())
    }

    fn forward_untracked_owned(&self, input: Tensor) -> PureResult<Tensor> {
        self.untracked.set(self.untracked.get() + 1);
        self.forward_owned(input)
    }

    fn backward(&mut self, _input: &Tensor, grad: &Tensor) -> PureResult<Tensor> {
        if grad.data().iter().any(|x| !x.is_finite()) {
            return Err(TensorError::InvalidValue {
                label: "test_backward_failure",
            });
        }
        Ok(grad.clone())
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

#[test]
fn nested_pullbacks_do_not_repeat_forward_and_untracked_propagates() {
    let _guard = crate::test_global_state_lock();
    let forwards = Rc::new(Cell::new(0));
    let untracked = Rc::new(Cell::new(0));
    let mut child = Sequential::new();
    child.push(Counted {
        forwards: forwards.clone(),
        untracked: untracked.clone(),
    });
    child.push(Dropout::with_seed(0.5, Some(53)).unwrap());
    let mut parent = Sequential::new();
    parent.push(child);
    parent.push(Gelu::new());
    let input = Tensor::from_vec(1, 64, vec![1.; 64]).unwrap();
    parent.forward(&input).unwrap();
    parent.backward(&input, &input).unwrap();
    parent.backward(&input, &input).unwrap();
    assert_eq!(forwards.get(), 1);
    parent.forward_untracked(&input).unwrap();
    assert_eq!(forwards.get(), 2);
    assert_eq!(untracked.get(), 1);
    assert!(parent.backward(&input, &input).is_err());
}

#[test]
fn eval_does_not_disable_backward_and_untracked_does_not_change_training() {
    let _guard = crate::test_global_state_lock();
    let mut sequence = Sequential::new();
    sequence.push(Dropout::with_seed(0.5, Some(7)).unwrap());
    let input = Tensor::from_vec(1, 128, vec![1.; 128]).unwrap();
    sequence.eval().unwrap();
    assert_eq!(sequence.forward(&input).unwrap().data(), input.data());
    assert_eq!(
        sequence.backward(&input, &input).unwrap().data(),
        input.data()
    );
    sequence.train().unwrap();
    assert!(sequence.backward(&input, &input).is_err());
    assert_ne!(
        sequence.forward_untracked(&input).unwrap().data(),
        input.data()
    );
}

#[test]
fn new_and_failed_forwards_invalidate_old_state_without_replaying() {
    let _guard = crate::test_global_state_lock();
    let forwards = Rc::new(Cell::new(0));
    let mut sequence = Sequential::new();
    sequence.push(Counted {
        forwards: forwards.clone(),
        untracked: Rc::new(Cell::new(0)),
    });
    let a = Tensor::from_vec(1, 2, vec![1., 2.]).unwrap();
    let b = Tensor::from_vec(1, 2, vec![3., 4.]).unwrap();
    assert!(sequence.backward(&a, &a).is_err());
    sequence.forward(&a).unwrap();
    sequence.forward(&b).unwrap();
    assert!(sequence.backward(&a, &a).is_err());
    sequence.backward(&b, &b).unwrap();
    let invalid = Tensor::from_vec(1, 2, vec![f32::NAN; 2]).unwrap();
    assert!(sequence.backward(&b, &invalid).is_err());
    assert!(sequence.backward(&b, &b).is_err());
    sequence.forward(&b).unwrap();
    assert!(sequence
        .forward(&Tensor::from_vec(1, 2, vec![f32::NAN; 2]).unwrap())
        .is_err());
    assert!(sequence.backward(&b, &b).is_err());
    assert_eq!(forwards.get(), 4);
}

#[test]
fn input_and_output_mutation_cannot_rewrite_saved_activations() {
    let _guard = crate::test_global_state_lock();
    let mut sequence = Sequential::new();
    sequence.push(Identity);
    sequence.push(Gelu::new());
    let input = Tensor::from_vec(2, 2, vec![-1., -0., 0.3, 1.]).unwrap();
    let expected = Gelu::new().backward(&input, &input).unwrap();
    let mut output = sequence.forward(&input).unwrap();
    output.data_mut().fill(99.);
    let bad_shape = Tensor::from_vec(1, 4, vec![1.; 4]).unwrap();
    assert!(sequence.backward(&input, &bad_shape).is_err());
    let mut changed = input.clone();
    changed.data_mut()[1] = 0.;
    assert!(sequence.backward(&changed, &input).is_err());
    assert_eq!(
        sequence
            .backward(&input.to_layout(Layout::ColMajor).unwrap(), &input)
            .unwrap()
            .data(),
        expected.data()
    );
    let exported = input.to_dlpack().unwrap();
    unsafe {
        *(*exported).dl_tensor.data.cast::<f32>() = 5.;
        (*exported).deleter.unwrap()(exported);
    }
    assert!(sequence.backward(&input, &input).is_err());
}

#[test]
fn mutation_boundaries_require_a_new_forward() {
    let _guard = crate::test_global_state_lock();
    let input = Tensor::from_vec(1, 2, vec![0.1, 0.2]).unwrap();
    let mut sequence = Sequential::new();
    sequence.push(Linear::new("linear", 2, 2).unwrap());
    for boundary in 0..5 {
        sequence.forward(&input).unwrap();
        match boundary {
            0 => sequence.visit_parameters_mut(&mut |_| Ok(())).unwrap(),
            1 => sequence
                .load_state_dict(&sequence.state_dict().unwrap())
                .unwrap(),
            2 => sequence.infuse_text("").unwrap(),
            3 => sequence.push(Identity),
            _ => sequence.insert(0, Identity).unwrap(),
        }
        assert!(sequence.backward(&input, &input).is_err());
    }
    sequence.forward(&input).unwrap();
    assert!(sequence.insert(usize::MAX, Identity).is_err());
    sequence.backward(&input, &input).unwrap();
}

#[test]
fn shared_parameter_exports_before_and_after_forward_cannot_change_pullback() {
    let _guard = crate::test_global_state_lock();
    for export_first in [false, true] {
        let mut sequence = Sequential::new();
        sequence.push(Linear::new("linear", 2, 2).unwrap());
        let input = Tensor::from_vec(1, 2, vec![0.1, 0.2]).unwrap();
        let export = |model: &Sequential| {
            let mut pointer = None;
            model
                .visit_parameters(&mut |p| {
                    if pointer.is_none() {
                        pointer = Some(p.value().to_dlpack()?);
                    }
                    Ok(())
                })
                .unwrap();
            pointer.unwrap()
        };
        let early = export_first.then(|| export(&sequence));
        sequence.forward(&input).unwrap();
        let pointer = early.unwrap_or_else(|| export(&sequence));
        unsafe {
            *(*pointer).dl_tensor.data.cast::<f32>() = f32::NAN;
            (*pointer).deleter.unwrap()(pointer);
        }
        assert!(sequence.backward(&input, &input).is_err());
        sequence
            .visit_parameters(&mut |p| {
                assert!(p.gradient().is_none());
                Ok(())
            })
            .unwrap();
    }
}
