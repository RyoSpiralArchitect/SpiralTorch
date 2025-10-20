mod grad_cam;
mod integrated_gradients;
pub mod utils;

pub use grad_cam::{GradCam, GradCamConfig};
pub use integrated_gradients::{IntegratedGradients, IntegratedGradientsConfig};

use st_core::telemetry::xai_report::{AttributionMetadata, AttributionReport};
use st_tensor::Tensor;

#[derive(Debug, Clone)]
pub struct AttributionOutput {
    pub map: Tensor,
    pub metadata: AttributionMetadata,
}

impl AttributionOutput {
    pub fn new(map: Tensor, metadata: AttributionMetadata) -> Self {
        Self { map, metadata }
    }

    pub fn to_report(&self) -> AttributionReport {
        let (rows, cols) = self.map.shape();
        AttributionReport::new(self.metadata.clone(), rows, cols, self.map.data().to_vec())
    }
}
