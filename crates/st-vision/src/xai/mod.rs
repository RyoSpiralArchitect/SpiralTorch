mod grad_cam;
mod integrated_gradients;
pub mod utils;

pub use grad_cam::{GradCam, GradCamConfig};
pub use integrated_gradients::{IntegratedGradients, IntegratedGradientsConfig};

use st_core::telemetry::xai_report::{AttributionMetadata, AttributionReport};
use st_tensor::{PureResult, Tensor};

use self::utils::{blend_heatmap, box_blur};

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

    pub fn smoothed(&self, kernel_size: usize) -> PureResult<AttributionOutput> {
        let blurred = box_blur(&self.map, kernel_size)?;
        Ok(AttributionOutput::new(blurred, self.metadata.clone()))
    }

    pub fn overlay(&self, base: &Tensor, alpha: f32) -> PureResult<Tensor> {
        blend_heatmap(base, &self.map, alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothing_returns_blurred_map() {
        let metadata = AttributionMetadata::for_algorithm("grad-cam");
        let map = Tensor::from_vec(1, 4, vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        let output = AttributionOutput::new(map, metadata);
        let smoothed = output.smoothed(3).unwrap();
        assert_eq!(smoothed.metadata.algorithm, "grad-cam");
        assert!(smoothed.map.data()[1] < 1.0);
    }

    #[test]
    fn overlay_blends_with_base() {
        let metadata = AttributionMetadata::default();
        let map = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
        let base = Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap();
        let output = AttributionOutput::new(map, metadata);
        let blended = output.overlay(&base, 0.5).unwrap();
        let values = blended.data();
        assert!((values[0] - 0.5).abs() < 1e-6);
        assert!((values[1] - 0.5).abs() < 1e-6);
    }
}
