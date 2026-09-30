//! Inference adapter over the trainable Rust classifier, not a second model.
use crate::models::{ConvNeXtClassifier, ConvNeXtConfig};
use crate::{FeatureStage, ImageTensor, ModelKind, ModelMetadata, VisionModel, VisionTask};
use st_nn::module::Module;
use st_tensor::{PureResult, Tensor, TensorError};
use std::sync::{Arc, Mutex};

struct ConvNeXtVisionModel {
    metadata: ModelMetadata,
    count: usize,
    model: Mutex<ConvNeXtClassifier>,
}

impl ConvNeXtClassifier {
    /// Transfer this model, including learned weights, into the common inference
    /// interface. Native callers serialize cache access through a mutex; browser
    /// callers remain local to their WASM instance.
    #[cfg_attr(
        target_arch = "wasm32",
        expect(
            clippy::arc_with_non_send_sync,
            reason = "Preserve the common Arc model handle; browser GPU caches stay instance-local."
        )
    )]
    pub fn into_vision_model(self) -> PureResult<Arc<dyn VisionModel>> {
        let config = self.config();
        let metadata = ModelMetadata {
            kind: ModelKind::ConvNeXtTiny,
            name: if config == &ConvNeXtConfig::default() {
                "convnext_tiny"
            } else {
                "convnext_custom"
            },
            task: VisionTask::Classification,
            input_channels: config.input_channels,
            image_size: config.input_hw,
            num_classes: self.num_classes(),
            has_pretrained: false,
        };
        let mut count = 0usize;
        self.visit_parameters(&mut |p| {
            count = count
                .checked_add(p.value().data().len())
                .ok_or(TensorError::InvalidValue {
                    label: "vision_parameter_count",
                })?;
            Ok(())
        })?;
        Ok(Arc::new(ConvNeXtVisionModel {
            metadata,
            count,
            model: Mutex::new(self),
        }))
    }
}

impl ConvNeXtVisionModel {
    fn input(&self, images: &[ImageTensor]) -> PureResult<Tensor> {
        if images.is_empty() {
            return Err(TensorError::EmptyInput("vision_batch_forward"));
        }
        let expected = (
            self.metadata.input_channels,
            self.metadata.image_size.0,
            self.metadata.image_size.1,
        );
        let width = expected
            .0
            .checked_mul(expected.1)
            .and_then(|v| v.checked_mul(expected.2))
            .ok_or(TensorError::InvalidValue {
                label: "vision_input_size",
            })?;
        let len = width
            .checked_mul(images.len())
            .ok_or(TensorError::InvalidValue {
                label: "vision_batch_size",
            })?;
        for image in images {
            if image.shape() != expected {
                return Err(TensorError::InvalidValue {
                    label: "vision_image_shape",
                });
            }
            if let Some(&value) = image.as_slice().iter().find(|v| !v.is_finite()) {
                return Err(TensorError::NonFiniteValue {
                    label: "vision_image",
                    value,
                });
            }
        }
        let mut values = Vec::with_capacity(len);
        for image in images {
            values.extend_from_slice(image.as_slice());
        }
        Tensor::from_vec(images.len(), width, values)
    }

    fn lock(&self) -> PureResult<std::sync::MutexGuard<'_, ConvNeXtClassifier>> {
        self.model.lock().map_err(|_| TensorError::InvalidValue {
            label: "vision_model_poisoned",
        })
    }
}

impl VisionModel for ConvNeXtVisionModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    fn parameter_count(&self) -> Option<usize> {
        Some(self.count)
    }
    fn forward(&self, images: &[ImageTensor]) -> PureResult<Tensor> {
        self.lock()?.forward(&self.input(images)?)
    }
    fn extract_features(&self, stage: FeatureStage, image: &ImageTensor) -> PureResult<Tensor> {
        let input = self.input(std::slice::from_ref(image))?;
        let model = self.lock()?;
        match stage {
            FeatureStage::Stem => {
                let stem = model.stem_features(&input)?;
                let channels = model.config().stage_dims[0];
                Tensor::from_vec(channels, stem.data().len() / channels, stem.data().to_vec())
            }
            FeatureStage::Head => model.features(&input),
            FeatureStage::Logits => model.forward(&input),
        }
    }
}
