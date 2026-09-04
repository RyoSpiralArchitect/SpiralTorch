// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Ryo SpiralArchitect

use super::{emit_loss_backend_meta_with_backend, Loss};
use crate::execution::TensorUtilRoute;
use crate::{PureResult, Tensor};
use st_tensor::{class_indices_from_tensor, CrossEntropyConfig, TensorError, TensorUtilBackend};

/// CPU logits loss for `(samples, classes)` predictions and `(samples, 1)` integer targets.
/// All numerical semantics live in st-tensor. Unreduced backward uses an all-ones seed.
/// Strict accelerator execution is rejected until an accelerator kernel exists.
#[derive(Clone, Copy, Debug, Default)]
pub struct CrossEntropyWithLogits {
    config: CrossEntropyConfig,
}

impl CrossEntropyWithLogits {
    pub fn new(config: CrossEntropyConfig) -> PureResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> CrossEntropyConfig {
        self.config
    }

    fn validate_execution(&self, prediction: &Tensor) -> PureResult<TensorUtilRoute> {
        let route = crate::execution::current_tensor_util_route(prediction.len());
        if crate::execution::current_accelerator_fallback().is_strict()
            && matches!(route.selected_backend, TensorUtilBackend::GpuWgpu)
        {
            return Err(TensorError::BackendFailure {
                backend: "wgpu",
                message: "CrossEntropyWithLogits has a CPU kernel only; fallback disabled".into(),
            });
        }
        Ok(route)
    }

    fn emit_route(
        &self,
        name: &'static str,
        prediction: &Tensor,
        output: &Tensor,
        route: TensorUtilRoute,
    ) {
        let fallback = if matches!(route.requested_backend, TensorUtilBackend::GpuWgpu) {
            Some(
                if matches!(route.selected_backend, TensorUtilBackend::Cpu) {
                    "tensor utility size threshold selected CPU"
                } else {
                    "CrossEntropyWithLogits has a CPU kernel only"
                },
            )
        } else {
            None
        };
        emit_loss_backend_meta_with_backend(
            name,
            prediction,
            output.shape(),
            self.config.reduction.as_str(),
            "cpu",
            route.requested_backend_label(),
            "classification.st_tensor_cpu",
            fallback,
        );
    }
}

impl Loss for CrossEntropyWithLogits {
    fn forward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        let route = self.validate_execution(prediction)?;
        let output = prediction
            .cross_entropy_with_logits(&class_indices_from_tensor(target)?, self.config)?;
        self.emit_route(
            "cross_entropy_logits_loss_forward",
            prediction,
            &output,
            route,
        );
        Ok(output)
    }

    fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        let route = self.validate_execution(prediction)?;
        let labels = class_indices_from_tensor(target)?;
        let shape = self.config.output_shape(prediction.shape().0);
        let seed = Tensor::from_vec(shape.0, shape.1, vec![1.0; shape.0])?;
        let output = prediction.cross_entropy_with_logits_backward(&labels, self.config, &seed)?;
        self.emit_route(
            "cross_entropy_logits_loss_backward",
            prediction,
            &output,
            route,
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_tensor::LossReduction;

    #[test]
    fn adapter_matches_tensor_core_for_every_reduction() {
        let logits = Tensor::from_vec(3, 2, vec![2.0, -1.0, 0.5, 1.0, 7.0, -8.0]).unwrap();
        let target = Tensor::from_vec(3, 1, vec![0.0, 1.0, -100.0]).unwrap();
        for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
            let config = CrossEntropyConfig {
                reduction,
                label_smoothing: 0.2,
                ..Default::default()
            };
            let mut loss = CrossEntropyWithLogits::new(config).unwrap();
            let expected = logits
                .cross_entropy_with_logits(&[0, 1, -100], config)
                .unwrap();
            assert_eq!(
                loss.forward(&logits, &target).unwrap().data(),
                expected.data()
            );
            let shape = expected.shape();
            let seed = Tensor::from_vec(shape.0, shape.1, vec![1.0; expected.len()]).unwrap();
            let gradient = logits
                .cross_entropy_with_logits_backward(&[0, 1, -100], config, &seed)
                .unwrap();
            assert_eq!(
                loss.backward(&logits, &target).unwrap().data(),
                gradient.data()
            );
        }
    }

    #[test]
    fn adapter_rejects_fractional_or_mismatched_labels() {
        let logits = Tensor::zeros(2, 3).unwrap();
        let mut loss = CrossEntropyWithLogits::default();
        for target in [
            Tensor::from_vec(2, 1, vec![0.0, 1.5]).unwrap(),
            Tensor::zeros(1, 1).unwrap(),
            Tensor::zeros(2, 3).unwrap(),
        ] {
            assert!(loss.forward(&logits, &target).is_err());
            assert!(loss.backward(&logits, &target).is_err());
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn allowed_cpu_fallback_retains_the_requested_wgpu_route() {
        use crate::execution::{
            push_backend_policy, AcceleratorFallback, BackendPolicy, ExecutionConfig,
        };
        use st_core::backend::device_caps::DeviceCaps;
        use std::sync::{Arc, Mutex};
        for threshold in [0, 1024] {
            let _guard = push_backend_policy(BackendPolicy::from_device_caps_with_config(
                DeviceCaps::wgpu(32, true, 256),
                ExecutionConfig::new(AcceleratorFallback::Allow, threshold),
            ));
            let events = Arc::new(Mutex::new(Vec::new()));
            let captured = events.clone();
            let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
                if event.op_name.starts_with("cross_entropy_logits_loss_") {
                    captured.lock().unwrap().push(event.data.clone());
                }
            })));
            let logits = Tensor::zeros(2, 3).unwrap();
            let target = Tensor::from_vec(2, 1, vec![0.0, 1.0]).unwrap();
            let mut loss = CrossEntropyWithLogits::default();
            loss.forward(&logits, &target).unwrap();
            loss.backward(&logits, &target).unwrap();
            st_tensor::set_thread_meta_observer(previous);
            let events = events.lock().unwrap();
            assert_eq!(events.len(), 2);
            for event in events.iter() {
                assert_eq!(event["backend"], "cpu");
                assert_eq!(event["requested_backend"], "wgpu");
                assert_eq!(event["fallback"]["from"], "wgpu");
            }
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn cpu_only_loss_does_not_bypass_strict_accelerator_policy() {
        use crate::execution::{
            push_backend_policy, AcceleratorFallback, BackendPolicy, ExecutionConfig,
        };
        use st_core::backend::device_caps::DeviceCaps;
        let policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::wgpu(32, true, 256),
            ExecutionConfig::new(AcceleratorFallback::Forbid, 0),
        );
        let _guard = push_backend_policy(policy);
        let logits = Tensor::zeros(2, 3).unwrap();
        let target = Tensor::from_vec(2, 1, vec![0.0, 1.0]).unwrap();
        let mut loss = CrossEntropyWithLogits::default();
        assert!(matches!(
            loss.forward(&logits, &target),
            Err(TensorError::BackendFailure {
                backend: "wgpu",
                ..
            })
        ));
        assert!(matches!(
            loss.backward(&logits, &target),
            Err(TensorError::BackendFailure {
                backend: "wgpu",
                ..
            })
        ));
    }
}
