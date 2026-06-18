use spiral_selfsup::trainer::DistributedDevice;
use st_core::backend::device_caps::DeviceCaps;
use st_nn::{Module, ModuleTrainer, Parameter, PureResult, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

static GROUP_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct FixedGradientModule {
    param: Parameter,
    gradient: Vec<f32>,
}

impl FixedGradientModule {
    fn new(gradient: Vec<f32>) -> Self {
        Self {
            param: Parameter::new(
                "distributed_weight",
                Tensor::zeros(1, gradient.len()).unwrap(),
            ),
            gradient,
        }
    }

    fn accumulate(&mut self) -> PureResult<()> {
        let update = Tensor::from_vec(1, self.gradient.len(), self.gradient.clone())?;
        self.param.accumulate_euclidean(&update)
    }

    fn values(&self) -> Vec<f32> {
        self.param.value().data().to_vec()
    }
}

impl Module for FixedGradientModule {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        Ok(input.clone())
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.accumulate()?;
        Ok(grad_output.clone())
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.param)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.param)
    }
}

fn one_local_step(gradient: Vec<f32>) -> Vec<f32> {
    let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 0.05, 0.1);
    let mut module = FixedGradientModule::new(gradient);
    module.accumulate().unwrap();
    trainer.step(&mut module).unwrap();
    module.values()
}

fn assert_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (actual, expected) in left.iter().zip(right.iter()) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn module_trainer_step_matches_averaged_distributed_accumulators() {
    let local_gradients = [vec![1.0, -1.0, 0.5], vec![3.0, 1.0, -0.5]];
    let expected = one_local_step(vec![2.0, 0.0, 0.0]);
    let group = format!(
        "module-trainer-accumulator-smoke-{}",
        GROUP_COUNTER.fetch_add(1, Ordering::Relaxed)
    );

    let devices = (0..local_gradients.len())
        .map(|rank| DistributedDevice::new(group.clone(), rank, local_gradients.len()).unwrap())
        .collect::<Vec<_>>();

    let handles = devices
        .into_iter()
        .zip(local_gradients)
        .map(|(device, gradient)| {
            thread::spawn(move || {
                let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 0.05, 0.1);
                trainer.set_training_device(device);
                let mut module = FixedGradientModule::new(gradient);
                module.accumulate().unwrap();
                trainer.step(&mut module).unwrap();
                (module.values(), trainer.last_accumulator_sync())
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        let (values, sync) = handle.join().unwrap();
        assert_close(&values, &expected);
        assert!(sync.enabled);
        assert_eq!(sync.world_size, 2);
        assert_eq!(sync.buffers, 1);
        assert_eq!(sync.values, 3);
    }
}
