use super::*;
use st_kernel_contracts::{
    elementwise::ElementwiseOp,
    pointwise::{PointwiseChain, PointwiseStep},
};

fn learner(runtime: WgpuRuntime) -> ResidentGraphLearner {
    let definition = GraphDefinition::new(
        NdLayout::contiguous(&[2, 2]).unwrap(),
        vec![GraphStage::Pointwise {
            chain: PointwiseChain::new(
                2,
                vec![PointwiseStep {
                    op: ElementwiseOp::Multiply,
                    rhs: Some(1),
                }],
            )
            .unwrap(),
            parameters: vec![0],
        }],
        vec![GraphParameter {
            role: ParameterRole::Gain,
            shape: vec![2],
            values: vec![1., 2.],
        }],
    )
    .unwrap();
    ResidentGraphLearner::new(
        runtime,
        definition,
        GraphGradientPolicy::Exact,
        Default::default(),
        MatmulKernel::Scalar,
        Default::default(),
    )
    .unwrap()
}

fn gradient(learner: &mut ResidentGraphLearner, input: &[f32]) -> GraphGradients {
    learner.upload(input).unwrap();
    let f = learner.forward().unwrap();
    let seed = learner.tensor_device().upload(&[2, 2], &[1.; 4]).unwrap();
    learner.backward(&f, &seed).unwrap()
}

fn values(tensors: Vec<ResidentTensor>) -> Vec<f32> {
    tensors
        .into_iter()
        .flat_map(|t| t.snapshot().unwrap().read().unwrap())
        .collect()
}

#[test]
fn microbatches_reuse_storage_and_enforce_parameter_not_input_identity() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("learner.microbatches").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let mut a = learner(runtime.clone());
    let mut foreign = learner(runtime);
    let mut sum = a.gradient_accumulator().unwrap();
    assert!(sum.is_empty());
    assert!(matches!(
        sum.parameter_gradients(),
        Err(TrainingError::EmptyAccumulator)
    ));
    assert!(matches!(
        a.sgd_accumulated(&sum, 0.),
        Err(TrainingError::EmptyAccumulator)
    ));
    let first = gradient(&mut a, &[1., 2., 3., 4.]);
    let second = gradient(&mut a, &[2., 1., 4., 3.]);
    assert!(matches!(
        a.sgd(&first, 0.),
        Err(TrainingError::StaleForward)
    ));
    let other = gradient(&mut foreign, &[1.; 4]);
    assert!(matches!(
        a.accumulate(&mut sum, &other, 1.),
        Err(TrainingError::AccumulatorState)
    ));
    assert!(matches!(
        foreign.zero_accumulator(&mut sum),
        Err(TrainingError::AccumulatorState)
    ));
    assert!(matches!(
        a.accumulate(&mut sum, &first, f32::NAN),
        Err(TrainingError::GradientWeight)
    ));
    for i in 0..300 {
        a.accumulate(&mut sum, if i % 2 == 0 { &first } else { &second }, 0.25)
            .unwrap();
    }
    assert_eq!(sum.len(), 300);
    let held = sum.parameter_gradients().unwrap();
    assert_eq!(values(sum.parameter_gradients().unwrap()), vec![375., 375.]);
    a.sgd_accumulated(&sum, 0.01).unwrap();
    let receipt = a.update_snapshot().unwrap();
    assert_eq!(receipt.input_generation(), 2);
    assert_eq!(receipt.read().unwrap(), 1);
    assert_eq!(
        a.parameter_snapshot().unwrap().read().unwrap().parameters()[0].values,
        vec![-2.75, -1.75]
    );
    assert!(matches!(
        a.sgd_accumulated(&sum, 0.),
        Err(TrainingError::AccumulatorState)
    ));
    assert!(matches!(
        a.accumulate(&mut sum, &first, 0.),
        Err(TrainingError::AccumulatorState)
    ));
    a.zero_accumulator(&mut sum).unwrap();
    assert_eq!(sum.parameter_generation(), 1);
    assert!(matches!(
        a.accumulate(&mut sum, &first, 1.),
        Err(TrainingError::AccumulatorState)
    ));
    let fresh = gradient(&mut a, &[1.; 4]);
    a.accumulate(&mut sum, &fresh, -0.5).unwrap();
    assert_eq!(values(sum.parameter_gradients().unwrap()), vec![-1., -1.]);
    sum.terms = u64::MAX;
    assert!(matches!(
        a.accumulate(&mut sum, &fresh, 1.),
        Err(TrainingError::Overflow)
    ));
    assert_eq!(values(sum.parameter_gradients().unwrap()), vec![-1., -1.]);
    drop((a, sum, first, second, fresh));
    assert_eq!(values(held), vec![375., 375.]);
}

#[test]
fn microbatch_guards_survive_zero_weights_overflow_cancellation_and_recover() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        runtime::ensure_default_runtime_blocking("learner.microbatch.guards").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let mut a = learner(runtime);
    let mut sum = a.gradient_accumulator().unwrap();
    let before = a.parameter_snapshot().unwrap().read().unwrap().parameters()[0]
        .values
        .clone();
    let good = gradient(&mut a, &[1.; 4]);
    let f = a.forward().unwrap();
    let huge = a.tensor_device().upload(&[2, 2], &[f32::MAX; 4]).unwrap();
    let poisoned = huge.add(&huge).unwrap();
    let bad = a.backward(&f, &poisoned).unwrap();
    a.accumulate(&mut sum, &bad, 0.).unwrap();
    a.accumulate(&mut sum, &good, 1.).unwrap();
    let held_bad = sum.parameter_gradients().unwrap();
    assert!(held_bad[0].snapshot().unwrap().read().is_err());
    a.sgd_accumulated(&sum, 0.).unwrap();
    assert!(a.update_snapshot().unwrap().read().is_err());
    assert_eq!(
        a.parameter_snapshot().unwrap().read().unwrap().parameters()[0].values,
        before
    );
    a.zero_accumulator(&mut sum).unwrap();
    let good = gradient(&mut a, &[1.; 4]);
    a.accumulate(&mut sum, &good, f32::MAX).unwrap();
    a.accumulate(&mut sum, &good, -f32::MAX).unwrap();
    a.sgd_accumulated(&sum, 0.).unwrap();
    assert!(a.update_snapshot().unwrap().read().is_err());
    assert_eq!(
        a.parameter_snapshot().unwrap().read().unwrap().parameters()[0].values,
        before
    );
    a.zero_accumulator(&mut sum).unwrap();
    let good = gradient(&mut a, &[1.; 4]);
    a.accumulate(&mut sum, &good, 0.5).unwrap();
    assert_eq!(values(sum.parameter_gradients().unwrap()), vec![1., 1.]);
    a.sgd_accumulated(&sum, 0.).unwrap();
    assert_eq!(a.update_snapshot().unwrap().read().unwrap(), 3);
    assert_eq!(
        a.parameter_snapshot().unwrap().read().unwrap().parameters()[0].values,
        before
    );
    assert!(held_bad[0].snapshot().unwrap().read().is_err());
    a.zero_accumulator(&mut sum).unwrap();
    let good = gradient(&mut a, &[1.; 4]);
    a.accumulate(&mut sum, &good, 1.).unwrap();
    a.sgd(&good, 0.).unwrap();
    assert!(matches!(
        a.sgd_accumulated(&sum, 0.),
        Err(TrainingError::AccumulatorState)
    ));
}
