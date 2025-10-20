use std::collections::HashMap;

use st_core::telemetry::xai_report::AttributionMetadata;
use st_nn::module::Module;
use st_tensor::{PureResult, Tensor};

use crate::xai::{
    AttributionOutput, GradCam, GradCamConfig, IntegratedGradients, IntegratedGradientsConfig,
};

pub struct ForwardAttributionHooks {
    grad_cam: HashMap<String, GradCamConfig>,
    activations: HashMap<String, Tensor>,
    gradients: HashMap<String, Tensor>,
}

impl ForwardAttributionHooks {
    pub fn new() -> Self {
        Self {
            grad_cam: HashMap::new(),
            activations: HashMap::new(),
            gradients: HashMap::new(),
        }
    }

    pub fn register_grad_cam(&mut self, layer: impl Into<String>, config: GradCamConfig) {
        self.grad_cam.insert(layer.into(), config);
    }

    pub fn record_activation(&mut self, layer: &str, activation: &Tensor) {
        self.activations
            .insert(layer.to_string(), activation.clone());
    }

    pub fn record_gradient(&mut self, layer: &str, gradient: &Tensor) {
        self.gradients.insert(layer.to_string(), gradient.clone());
    }

    pub fn clear(&mut self) {
        self.activations.clear();
        self.gradients.clear();
    }

    pub fn compute_grad_cam(&mut self, layer: &str) -> PureResult<Option<AttributionOutput>> {
        let Some(config) = self.grad_cam.get(layer).copied() else {
            return Ok(None);
        };
        let activation = match self.activations.remove(layer) {
            Some(value) => value,
            None => return Ok(None),
        };
        let gradient = match self.gradients.remove(layer) {
            Some(value) => value,
            None => {
                self.activations.insert(layer.to_string(), activation);
                return Ok(None);
            }
        };
        let heatmap = GradCam::attribute(&activation, &gradient, &config)?;
        let mut metadata = AttributionMetadata::for_algorithm("grad-cam");
        metadata.layer = Some(layer.to_string());
        metadata.insert_extra_number("height", config.height as f64);
        metadata.insert_extra_number("width", config.width as f64);
        metadata.insert_extra_flag("apply_relu", config.apply_relu);
        metadata.insert_extra_number("epsilon", config.epsilon as f64);
        Ok(Some(AttributionOutput::new(heatmap, metadata)))
    }
}

pub fn run_integrated_gradients<M: Module>(
    model: &mut M,
    input: &Tensor,
    baseline: &Tensor,
    config: IntegratedGradientsConfig,
    target_label: Option<&str>,
) -> PureResult<AttributionOutput> {
    let map = IntegratedGradients::attribute(model, input, baseline, &config)?;
    let mut metadata = AttributionMetadata::for_algorithm("integrated-gradients");
    metadata.steps = Some(config.steps);
    metadata.insert_extra_number("target_index", config.target_index as f64);
    match target_label {
        Some(label) => metadata.target = Some(label.to_string()),
        None => metadata.target = Some(config.target_index.to_string()),
    }
    Ok(AttributionOutput::new(map, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_tensor::Tensor;

    struct TestModule;

    impl Module for TestModule {
        fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
            Ok(input.clone())
        }

        fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
            Ok(grad_output.clone())
        }

        fn visit_parameters(
            &self,
            _visitor: &mut dyn FnMut(&st_nn::module::Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            Ok(())
        }

        fn visit_parameters_mut(
            &mut self,
            _visitor: &mut dyn FnMut(&mut st_nn::module::Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            Ok(())
        }
    }

    #[test]
    fn hooks_produce_grad_cam_reports() {
        let mut hooks = ForwardAttributionHooks::new();
        hooks.register_grad_cam(
            "conv1",
            GradCamConfig {
                height: 2,
                width: 2,
                apply_relu: true,
                epsilon: 1e-6,
            },
        );
        let activation =
            Tensor::from_vec(2, 4, vec![1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.5, 2.0]).unwrap();
        let gradient =
            Tensor::from_vec(2, 4, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.4]).unwrap();
        hooks.record_activation("conv1", &activation);
        hooks.record_gradient("conv1", &gradient);
        let attribution = hooks.compute_grad_cam("conv1").unwrap().unwrap();
        assert_eq!(attribution.metadata.algorithm, "grad-cam");
        assert_eq!(attribution.metadata.layer.as_deref(), Some("conv1"));
        let report = attribution.to_report();
        assert_eq!(report.shape(), (2, 2));
        assert_eq!(report.metadata.algorithm, "grad-cam");
        assert_eq!(report.metadata.layer.as_deref(), Some("conv1"));
    }

    #[test]
    fn integrated_gradients_hook_wraps_metadata() {
        let input = Tensor::from_vec(1, 2, vec![0.25, 0.75]).unwrap();
        let baseline = Tensor::zeros(1, 2).unwrap();
        let mut module = TestModule;
        let config = IntegratedGradientsConfig::new(8, 0);
        let attribution =
            run_integrated_gradients(&mut module, &input, &baseline, config, Some("class_a"))
                .unwrap();
        assert_eq!(attribution.metadata.algorithm, "integrated-gradients");
        assert_eq!(attribution.metadata.steps, Some(8));
        assert_eq!(attribution.metadata.target.as_deref(), Some("class_a"));
    }
}
