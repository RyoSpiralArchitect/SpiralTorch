// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::nerf::encoding::PositionalEncoding;
use st_nn::layers::activation::Relu;
use st_nn::layers::linear::Linear;
use st_nn::layers::sequential::Sequential;
use st_nn::module::{Module, Parameter};
use st_tensor::{PureResult, Tensor, TensorError};

/// Describes how many raw coordinates the field expects for positions and directions.
#[derive(Clone, Copy, Debug)]
pub struct FieldSampleLayout {
    pub position_dims: usize,
    pub direction_dims: usize,
}

impl FieldSampleLayout {
    /// Returns the total number of scalar features that need to be supplied to
    /// the field per sample.
    pub fn total_dims(&self) -> usize {
        self.position_dims + self.direction_dims
    }
}

/// Configuration used to build a [`NerfField`].
#[derive(Clone, Debug)]
pub struct NerfFieldConfig {
    pub position_dims: usize,
    pub direction_dims: usize,
    pub position_frequencies: usize,
    pub direction_frequencies: usize,
    pub hidden_width: usize,
    pub hidden_layers: usize,
    pub feature_dim: usize,
    pub color_layers: usize,
    pub color_hidden_width: usize,
}

impl Default for NerfFieldConfig {
    fn default() -> Self {
        Self {
            position_dims: 3,
            direction_dims: 3,
            position_frequencies: 6,
            direction_frequencies: 4,
            hidden_width: 128,
            hidden_layers: 4,
            feature_dim: 32,
            color_layers: 1,
            color_hidden_width: 64,
        }
    }
}

/// NeRF style field backed by SpiralTorch linear layers and ReLU activations.
#[derive(Debug)]
pub struct NerfField {
    layout: FieldSampleLayout,
    position_encoding: PositionalEncoding,
    direction_encoding: Option<PositionalEncoding>,
    trunk: Sequential,
    density_head: Linear,
    feature_head: Linear,
    color_head: Sequential,
    feature_dim: usize,
    color_input_dim: usize,
}

impl NerfField {
    /// Constructs a field from the provided configuration.
    pub fn new(config: NerfFieldConfig) -> PureResult<Self> {
        if config.position_dims == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: config.position_dims,
                cols: config.hidden_width,
            });
        }
        let layout = FieldSampleLayout {
            position_dims: config.position_dims,
            direction_dims: config.direction_dims,
        };
        let position_encoding =
            PositionalEncoding::new(config.position_dims, config.position_frequencies)?;
        let direction_encoding = if config.direction_dims > 0 {
            let encoding =
                PositionalEncoding::new(config.direction_dims, config.direction_frequencies)?
                    .without_input();
            Some(encoding)
        } else {
            None
        };

        let mut trunk = Sequential::new();
        let mut last_dim = position_encoding.output_dims();
        if config.hidden_layers == 0 {
            last_dim = position_encoding.output_dims();
        }
        for idx in 0..config.hidden_layers {
            trunk.push(Linear::new(
                format!("trunk_fc{idx}"),
                last_dim,
                config.hidden_width,
            )?);
            trunk.push(Relu::new());
            last_dim = config.hidden_width;
        }

        let density_head = Linear::new("density", last_dim, 1)?;
        let feature_head = Linear::new("feature", last_dim, config.feature_dim)?;

        let mut color_head = Sequential::new();
        let dir_out = direction_encoding
            .as_ref()
            .map(|enc| enc.output_dims())
            .unwrap_or(0);
        let mut color_in = config.feature_dim + dir_out;
        for idx in 0..config.color_layers {
            color_head.push(Linear::new(
                format!("color_fc{idx}"),
                color_in,
                config.color_hidden_width,
            )?);
            color_head.push(Relu::new());
            color_in = config.color_hidden_width;
        }
        color_head.push(Linear::new("color_out", color_in, 3)?);

        Ok(Self {
            layout,
            position_encoding,
            direction_encoding,
            trunk,
            density_head,
            feature_head,
            color_head,
            feature_dim: config.feature_dim,
            color_input_dim: config.feature_dim + dir_out,
        })
    }

    /// Returns the raw sample layout expected by the field.
    pub fn sample_layout(&self) -> FieldSampleLayout {
        self.layout
    }

    /// Convenience helper that concatenates positions and directions into a
    /// single input tensor matching the field layout.
    pub fn assemble_input(
        &self,
        positions: &Tensor,
        directions: Option<&Tensor>,
    ) -> PureResult<Tensor> {
        let (rows, cols) = positions.shape();
        if cols != self.layout.position_dims {
            return Err(TensorError::ShapeMismatch {
                left: (rows, self.layout.position_dims),
                right: (rows, cols),
            });
        }
        let mut buffer = Vec::with_capacity(rows * self.layout.total_dims());
        let dir_dims = self.layout.direction_dims;
        let dir_tensor = match (dir_dims, directions) {
            (0, _) => None,
            (_, Some(dir)) => {
                let shape = dir.shape();
                if shape != (rows, dir_dims) {
                    return Err(TensorError::ShapeMismatch {
                        left: (rows, dir_dims),
                        right: shape,
                    });
                }
                Some(dir)
            }
            (_, None) => None,
        };
        let pos_data = positions.data();
        let dir_data = dir_tensor.map(|tensor| tensor.data());
        for row in 0..rows {
            let pos_off = row * cols;
            buffer.extend_from_slice(&pos_data[pos_off..pos_off + cols]);
            if let Some(dir_slice) = dir_data {
                let dir_off = row * dir_dims;
                buffer.extend_from_slice(&dir_slice[dir_off..dir_off + dir_dims]);
            } else if dir_dims > 0 {
                buffer.extend(std::iter::repeat(0.0).take(dir_dims));
            }
        }
        Tensor::from_vec(rows, self.layout.total_dims(), buffer)
    }

    fn split_inputs(&self, input: &Tensor) -> PureResult<(Tensor, Option<Tensor>)> {
        let (rows, cols) = input.shape();
        if cols != self.layout.total_dims() {
            return Err(TensorError::ShapeMismatch {
                left: (rows, self.layout.total_dims()),
                right: (rows, cols),
            });
        }
        let mut positions = Vec::with_capacity(rows * self.layout.position_dims);
        let mut directions = if self.layout.direction_dims > 0 {
            Some(Vec::with_capacity(rows * self.layout.direction_dims))
        } else {
            None
        };
        let data = input.data();
        for row in 0..rows {
            let offset = row * cols;
            positions.extend_from_slice(&data[offset..offset + self.layout.position_dims]);
            if let Some(dir_buffer) = directions.as_mut() {
                dir_buffer.extend_from_slice(
                    &data[offset + self.layout.position_dims..offset + self.layout.total_dims()],
                );
            }
        }
        let positions = Tensor::from_vec(rows, self.layout.position_dims, positions)?;
        let directions = match directions {
            Some(vec) => Some(Tensor::from_vec(rows, self.layout.direction_dims, vec)?),
            None => None,
        };
        Ok((positions, directions))
    }

    fn concat_features(&self, left: &Tensor, right: Option<Tensor>) -> PureResult<Tensor> {
        if let Some(right) = right {
            let (rows, lcols) = left.shape();
            let (rrows, rcols) = right.shape();
            if rows != rrows {
                return Err(TensorError::ShapeMismatch {
                    left: (rows, lcols),
                    right: (rrows, rcols),
                });
            }
            let mut data = Vec::with_capacity(rows * (lcols + rcols));
            let left_data = left.data();
            let right_data = right.data();
            for row in 0..rows {
                let loff = row * lcols;
                data.extend_from_slice(&left_data[loff..loff + lcols]);
                let roff = row * rcols;
                data.extend_from_slice(&right_data[roff..roff + rcols]);
            }
            Tensor::from_vec(rows, lcols + rcols, data)
        } else {
            Ok(left.clone())
        }
    }

    fn merge_outputs(&self, density: &Tensor, color: &Tensor) -> PureResult<Tensor> {
        let (rows, dcols) = density.shape();
        if dcols != 1 {
            return Err(TensorError::ShapeMismatch {
                left: (rows, 1),
                right: (rows, dcols),
            });
        }
        let (crow, ccols) = color.shape();
        if crow != rows || ccols != 3 {
            return Err(TensorError::ShapeMismatch {
                left: (rows, 3),
                right: (crow, ccols),
            });
        }
        let mut data = Vec::with_capacity(rows * 4);
        let density_data = density.data();
        let color_data = color.data();
        for row in 0..rows {
            data.push(density_data[row]);
            let offset = row * 3;
            data.extend_from_slice(&color_data[offset..offset + 3]);
        }
        Tensor::from_vec(rows, 4, data)
    }

    fn split_grad_output(&self, grad_output: &Tensor) -> PureResult<(Tensor, Tensor)> {
        let (rows, cols) = grad_output.shape();
        if cols != 4 {
            return Err(TensorError::ShapeMismatch {
                left: (rows, 4),
                right: (rows, cols),
            });
        }
        let mut density = Vec::with_capacity(rows);
        let mut color = Vec::with_capacity(rows * 3);
        let data = grad_output.data();
        for row in 0..rows {
            let offset = row * 4;
            density.push(data[offset]);
            color.extend_from_slice(&data[offset + 1..offset + 4]);
        }
        let density = Tensor::from_vec(rows, 1, density)?;
        let color = Tensor::from_vec(rows, 3, color)?;
        Ok((density, color))
    }
}

impl Module for NerfField {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (positions, directions) = self.split_inputs(input)?;
        let encoded_pos = self.position_encoding.encode(&positions)?;
        let trunk = self.trunk.forward(&encoded_pos)?;
        let density = self.density_head.forward(&trunk)?;
        let features = self.feature_head.forward(&trunk)?;
        let color_input = if let Some(dir_enc) = &self.direction_encoding {
            let dir_tensor = directions.as_ref().map_or_else(
                || Tensor::zeros(positions.shape().0, self.layout.direction_dims),
                |tensor| Ok(tensor.clone()),
            )?;
            let encoded_dir = dir_enc.encode(&dir_tensor)?;
            self.concat_features(&features, Some(encoded_dir))?
        } else {
            features.clone()
        };
        let color = self.color_head.forward(&color_input)?;
        self.merge_outputs(&density, &color)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (positions, directions) = self.split_inputs(input)?;
        let encoded_pos = self.position_encoding.encode(&positions)?;
        let trunk = self.trunk.forward(&encoded_pos)?;
        let density = self.density_head.forward(&trunk)?;
        let features = self.feature_head.forward(&trunk)?;
        let color_input = if let Some(dir_enc) = &self.direction_encoding {
            let dir_tensor = directions.as_ref().map_or_else(
                || Tensor::zeros(positions.shape().0, self.layout.direction_dims),
                |tensor| Ok(tensor.clone()),
            )?;
            let encoded_dir = dir_enc.encode(&dir_tensor)?;
            self.concat_features(&features, Some(encoded_dir))?
        } else {
            features.clone()
        };
        let (grad_density, grad_color) = self.split_grad_output(grad_output)?;
        let grad_color_input = self.color_head.backward(&color_input, &grad_color)?;

        let (grad_features_from_color, _grad_dir) = if self.color_input_dim > self.feature_dim {
            let dir_dim = self.color_input_dim - self.feature_dim;
            let (rows, cols) = grad_color_input.shape();
            let mut feat = Vec::with_capacity(rows * self.feature_dim);
            let mut dir = Vec::with_capacity(rows * dir_dim);
            let data = grad_color_input.data();
            for row in 0..rows {
                let offset = row * cols;
                feat.extend_from_slice(&data[offset..offset + self.feature_dim]);
                dir.extend_from_slice(&data[offset + self.feature_dim..offset + cols]);
            }
            (
                Tensor::from_vec(rows, self.feature_dim, feat)?,
                Some(Tensor::from_vec(rows, dir_dim, dir)?),
            )
        } else {
            (grad_color_input.clone(), None)
        };

        let grad_features = self
            .feature_head
            .backward(&trunk, &grad_features_from_color)?;
        let mut grad_trunk = self.density_head.backward(&trunk, &grad_density)?;
        grad_trunk.add_scaled(&grad_features, 1.0)?;
        let _ = self.trunk.backward(&encoded_pos, &grad_trunk)?;

        // The inputs are ray samples and should not receive gradients. Return
        // a zero tensor with matching shape to satisfy the Module contract.
        Tensor::zeros(input.shape().0, input.shape().1)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.trunk.visit_parameters(visitor)?;
        self.density_head.visit_parameters(visitor)?;
        self.feature_head.visit_parameters(visitor)?;
        self.color_head.visit_parameters(visitor)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.trunk.visit_parameters_mut(visitor)?;
        self.density_head.visit_parameters_mut(visitor)?;
        self.feature_head.visit_parameters_mut(visitor)?;
        self.color_head.visit_parameters_mut(visitor)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_forward_produces_density_and_color() {
        let field = NerfField::new(NerfFieldConfig::default()).unwrap();
        let layout = field.sample_layout();
        let input = Tensor::from_vec(
            2,
            layout.total_dims(),
            vec![0.0, 0.1, 0.2, 0.0, 0.0, 1.0, 0.5, -0.2, 0.3, 0.1, 0.9, -0.1],
        )
        .unwrap();
        let output = field.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 4));
    }

    #[test]
    fn backward_streams_gradients() {
        let mut field = NerfField::new(NerfFieldConfig::default()).unwrap();
        let layout = field.sample_layout();
        let input =
            Tensor::from_vec(1, layout.total_dims(), vec![0.2, -0.4, 0.1, 0.0, 0.0, 1.0]).unwrap();
        let output = field.forward(&input).unwrap();
        let grad = Tensor::from_vec(1, 4, vec![1.0, 0.5, -0.5, 0.25]).unwrap();
        let _ = field.backward(&input, &grad).unwrap();
        field.apply_step(1e-3).unwrap();
        let updated = field.forward(&input).unwrap();
        assert_ne!(output.data(), updated.data());
    }
}
