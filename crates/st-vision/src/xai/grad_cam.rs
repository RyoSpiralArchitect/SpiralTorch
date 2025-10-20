use st_tensor::{PureResult, Tensor, TensorError};

use super::utils::{ensure_same_shape, normalise_unit_interval};

#[derive(Debug, Clone, Copy)]
pub struct GradCamConfig {
    pub height: usize,
    pub width: usize,
    pub apply_relu: bool,
    pub epsilon: f32,
}

impl GradCamConfig {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            apply_relu: true,
            epsilon: 1e-6,
        }
    }
}

pub struct GradCam;

impl GradCam {
    pub fn attribute(
        activations: &Tensor,
        gradients: &Tensor,
        config: &GradCamConfig,
    ) -> PureResult<Tensor> {
        ensure_same_shape(activations, gradients, "grad_cam_inputs")?;
        if config.height == 0 || config.width == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: config.height,
                cols: config.width,
            });
        }
        let spatial = config.height * config.width;
        let (channels, features) = activations.shape();
        if features != spatial {
            return Err(TensorError::ShapeMismatch {
                left: (channels, features),
                right: (channels, spatial),
            });
        }
        let mut weights = vec![0.0f32; channels];
        for channel in 0..channels {
            let mut sum = 0.0f32;
            for idx in 0..features {
                sum += gradients.data()[channel * features + idx];
            }
            weights[channel] = sum / features as f32;
        }
        let mut heatmap = vec![0.0f32; spatial];
        for idx in 0..features {
            let mut value = 0.0f32;
            for channel in 0..channels {
                let activation = activations.data()[channel * features + idx];
                value += weights[channel] * activation;
            }
            if config.apply_relu && value < 0.0 {
                value = 0.0;
            }
            heatmap[idx] = value;
        }
        normalise_unit_interval(&mut heatmap, config.epsilon);
        Tensor::from_vec(config.height, config.width, heatmap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grad_cam_normalises_heatmap() {
        let activations = Tensor::from_vec(
            2,
            4,
            vec![
                1.0, 1.0, 1.0, 1.0, // channel 0
                0.5, 1.0, 1.5, 2.0, // channel 1
            ],
        )
        .unwrap();
        let gradients = Tensor::from_vec(
            2,
            4,
            vec![
                1.0, 1.0, 1.0, 1.0, // channel 0
                0.0, 0.0, 0.0, 0.4, // channel 1
            ],
        )
        .unwrap();
        let config = GradCamConfig::new(2, 2);
        let heatmap = GradCam::attribute(&activations, &gradients, &config).unwrap();
        let expected = vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        for (value, expected) in heatmap.data().iter().zip(expected.iter()) {
            assert!((value - expected).abs() < 1e-6);
        }
        let (rows, cols) = heatmap.shape();
        assert_eq!((rows, cols), (2, 2));
    }
}
