mod grad_cam;
mod integrated_gradients;
pub mod utils;

pub use grad_cam::{GradCam, GradCamConfig};
pub use integrated_gradients::{IntegratedGradients, IntegratedGradientsConfig};

use st_core::telemetry::xai_report::{AttributionMetadata, AttributionReport};
use st_tensor::{PureResult, Tensor};

use self::utils::{blend_heatmap, box_blur, clamp, mul, normalise_tensor, threshold_mask};

#[derive(Debug, Clone)]
pub struct AttributionOutput {
    pub map: Tensor,
    pub metadata: AttributionMetadata,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AttributionStatistics {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub entropy: f32,
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

    pub fn normalised(&self) -> PureResult<AttributionOutput> {
        let normalised = normalise_tensor(&self.map, 1e-6)?;
        Ok(AttributionOutput::new(normalised, self.metadata.clone()))
    }

    pub fn overlay(&self, base: &Tensor, alpha: f32) -> PureResult<Tensor> {
        blend_heatmap(base, &self.map, alpha)
    }

    pub fn gated_overlay(&self, base: &Tensor, threshold: f32, alpha: f32) -> PureResult<Tensor> {
        let mask = threshold_mask(&self.map, threshold)?;
        let emphasised = clamp(&self.map, 0.0, 1.0)?;
        let gated = mul(&self.map, &mask)?;
        let combined = blend_heatmap(base, &gated, alpha)?;
        blend_heatmap(&combined, &emphasised, alpha.min(1.0))
    }

    pub fn focus_mask(&self, threshold: f32) -> PureResult<Tensor> {
        threshold_mask(&self.map, threshold)
    }

    pub fn statistics(&self) -> AttributionStatistics {
        let data = self.map.data();
        if data.is_empty() {
            return AttributionStatistics::default();
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f32;
        for &value in data.iter() {
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
            sum += value;
        }
        let mean = sum / data.len() as f32;
        let mut entropy = 0.0f32;
        let mut normalised = data.to_vec();
        utils::normalise_unit_interval(&mut normalised, 1e-6);
        for value in normalised.iter() {
            if *value > f32::EPSILON {
                entropy -= *value * value.ln();
            }
        }
        AttributionStatistics {
            min,
            max,
            mean,
            entropy,
        }
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
    fn normalised_overlay_and_stats() {
        let metadata = AttributionMetadata::default();
        let map = Tensor::from_vec(1, 4, vec![0.0, 0.2, 0.8, 1.0]).unwrap();
        let base = Tensor::from_vec(1, 4, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let output = AttributionOutput::new(map, metadata);
        let normalised = output.normalised().unwrap();
        let gated = normalised.gated_overlay(&base, 0.5, 0.4).unwrap();
        let mask = normalised.focus_mask(0.5).unwrap();
        let stats = normalised.statistics();
        assert!(gated.data()[3] >= gated.data()[0]);
        assert_eq!(mask.data()[0], 0.0);
        assert!(stats.max <= 1.0);
        assert!(stats.entropy >= 0.0);
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
