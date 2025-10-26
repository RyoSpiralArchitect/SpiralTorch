// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use st_core::theory::general_relativity::{ZRelativityModel, ZRelativityTensorBundle};
use st_tensor::{PureResult, Tensor, TensorError};

fn tensor_len(tensor: &Tensor) -> usize {
    tensor.len()
}

fn append_tensor_data(buffer: &mut Vec<f32>, tensor: &Tensor) {
    buffer.extend_from_slice(tensor.data());
}

fn slice_to_tensor(slice: &[f32], rows: usize, cols: usize) -> PureResult<Tensor> {
    Tensor::from_vec(rows, cols, slice.to_vec())
}

fn require_gradient_shape(grad: &Tensor, expected_cols: usize) -> PureResult<()> {
    let (rows, cols) = grad.shape();
    if rows != 1 || cols != expected_cols {
        return Err(TensorError::ShapeMismatch {
            left: (rows, cols),
            right: (1, expected_cols),
        });
    }
    Ok(())
}

/// Wraps a [`ZRelativityModel`] so its tensors participate in `nn.Module` training flows.
#[derive(Debug)]
pub struct ZRelativityModule {
    model: ZRelativityModel,
    block: Parameter,
    gauge: Option<Parameter>,
    gauge_static: Option<Tensor>,
    scalar: Option<Parameter>,
    scalar_static: Option<Tensor>,
    warp: Option<Parameter>,
    warp_static: Option<Tensor>,
    vector_length: usize,
}

impl ZRelativityModule {
    fn split_learnable(
        tensor: Tensor,
        learnable: bool,
        name: &str,
    ) -> (Option<Parameter>, Option<Tensor>) {
        if learnable {
            (Some(Parameter::new(name, tensor)), None)
        } else {
            (None, Some(tensor))
        }
    }

    fn vector_length(bundle: &ZRelativityTensorBundle) -> usize {
        let mut length = tensor_len(&bundle.block_metric)
            + tensor_len(&bundle.gauge_field)
            + tensor_len(&bundle.scalar_moduli);
        if let Some(warp) = &bundle.warp {
            length += tensor_len(warp);
        }
        length
    }

    fn aggregate_values(&self) -> PureResult<Vec<f32>> {
        let mut data = Vec::with_capacity(self.vector_length);
        append_tensor_data(&mut data, self.block.value());
        match (&self.gauge, &self.gauge_static) {
            (Some(param), _) => append_tensor_data(&mut data, param.value()),
            (None, Some(tensor)) => append_tensor_data(&mut data, tensor),
            _ => {}
        }
        match (&self.scalar, &self.scalar_static) {
            (Some(param), _) => append_tensor_data(&mut data, param.value()),
            (None, Some(tensor)) => append_tensor_data(&mut data, tensor),
            _ => {}
        }
        match (&self.warp, &self.warp_static) {
            (Some(param), _) => append_tensor_data(&mut data, param.value()),
            (None, Some(tensor)) => append_tensor_data(&mut data, tensor),
            _ => {}
        }
        Ok(data)
    }

    fn accumulate_optional(
        parameter: &mut Option<Parameter>,
        grad_slice: &[f32],
        rows: usize,
        cols: usize,
    ) -> PureResult<()> {
        if let Some(param) = parameter {
            let grad = slice_to_tensor(grad_slice, rows, cols)?;
            param.accumulate_euclidean(&grad)?;
        }
        Ok(())
    }

    /// Creates a module from an assembled relativity model.
    pub fn from_model(model: ZRelativityModel) -> PureResult<Self> {
        let bundle = model.tensor_bundle()?;
        let flags = model.learnable_flags();

        let block = Parameter::new("block_metric", bundle.block_metric.clone());
        let (gauge, gauge_static) =
            Self::split_learnable(bundle.gauge_field.clone(), flags.mixed, "gauge_field");
        let (scalar, scalar_static) = Self::split_learnable(
            bundle.scalar_moduli.clone(),
            flags.internal,
            "scalar_moduli",
        );
        let (warp, warp_static) = match bundle.warp.clone() {
            Some(tensor) => Self::split_learnable(tensor, flags.warp, "warp_factor"),
            None => (None, None),
        };

        let vector_length = Self::vector_length(&bundle);

        Ok(Self {
            model,
            block,
            gauge,
            gauge_static,
            scalar,
            scalar_static,
            warp,
            warp_static,
            vector_length,
        })
    }

    /// Returns the number of scalar elements exposed by [`forward`](Module::forward).
    pub fn parameter_dimension(&self) -> usize {
        self.vector_length
    }

    /// Provides an immutable reference to the wrapped model.
    pub fn model(&self) -> &ZRelativityModel {
        &self.model
    }
}

impl Module for ZRelativityModule {
    fn forward(&self, _input: &Tensor) -> PureResult<Tensor> {
        let data = self.aggregate_values()?;
        Tensor::from_vec(1, self.vector_length, data)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        require_gradient_shape(grad_output, self.vector_length)?;
        let mut offset = 0usize;
        let (block_rows, block_cols) = self.block.value().shape();
        let block_len = block_rows * block_cols;
        self.block.accumulate_euclidean(&slice_to_tensor(
            &grad_output.data()[offset..offset + block_len],
            block_rows,
            block_cols,
        )?)?;
        offset += block_len;

        if let Some(length) = self.gauge.as_ref().map(|param| {
            let (rows, cols) = param.value().shape();
            (rows, cols, rows * cols)
        }) {
            let (rows, cols, len) = length;
            let slice = &grad_output.data()[offset..offset + len];
            Self::accumulate_optional(&mut self.gauge, slice, rows, cols)?;
            offset += len;
        } else if let Some(tensor) = &self.gauge_static {
            offset += tensor_len(tensor);
        }

        if let Some(length) = self.scalar.as_ref().map(|param| {
            let (rows, cols) = param.value().shape();
            (rows, cols, rows * cols)
        }) {
            let (rows, cols, len) = length;
            let slice = &grad_output.data()[offset..offset + len];
            Self::accumulate_optional(&mut self.scalar, slice, rows, cols)?;
            offset += len;
        } else if let Some(tensor) = &self.scalar_static {
            offset += tensor_len(tensor);
        }

        if let Some(length) = self.warp.as_ref().map(|param| {
            let (rows, cols) = param.value().shape();
            (rows, cols, rows * cols)
        }) {
            let (rows, cols, len) = length;
            let slice = &grad_output.data()[offset..offset + len];
            Self::accumulate_optional(&mut self.warp, slice, rows, cols)?;
            offset += len;
        } else if let Some(tensor) = &self.warp_static {
            offset += tensor_len(tensor);
        }

        let _ = offset;
        Tensor::zeros(input.shape().0, input.shape().1)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.block)?;
        if let Some(param) = &self.gauge {
            visitor(param)?;
        }
        if let Some(param) = &self.scalar {
            visitor(param)?;
        }
        if let Some(param) = &self.warp {
            visitor(param)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.block)?;
        if let Some(param) = &mut self.gauge {
            visitor(param)?;
        }
        if let Some(param) = &mut self.scalar {
            visitor(param)?;
        }
        if let Some(param) = &mut self.warp {
            visitor(param)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, Matrix4, Vector4};
    use st_core::theory::general_relativity::{
        GeneralRelativityModel, InternalMetric, InternalPatch, InternalSpace, LorentzianMetric,
        MetricDerivatives, MetricSecondDerivatives, MixedBlock, PhysicalConstants, ProductGeometry,
        ProductMetric, SymmetryAnsatz, Topology, WarpFactor, ZManifold,
    };

    fn build_model() -> ZRelativityModel {
        let base_metric =
            LorentzianMetric::try_new(Matrix4::from_diagonal(&Vector4::new(-1.0, 1.0, 1.0, 1.0)))
                .unwrap();
        let spacetime = ZManifold::canonical();
        let base_model = GeneralRelativityModel::new(
            spacetime.clone(),
            base_metric.clone(),
            MetricDerivatives::zero(),
            MetricSecondDerivatives::zero(),
            SymmetryAnsatz::HomogeneousIsotropic,
            Topology::R4,
            vec![],
        );

        let internal = InternalMetric::try_new(DMatrix::identity(2, 2))
            .unwrap()
            .with_learnable(true);
        let mixed = MixedBlock::new(
            DMatrix::from_row_slice(
                4,
                internal.dimension(),
                &[0.0, 0.1, -0.1, 0.0, 0.05, -0.05, 0.02, -0.02],
            ),
            4,
            internal.dimension(),
        )
        .unwrap()
        .with_learnable(true);
        let warp = WarpFactor::from_multiplier(1.5)
            .unwrap()
            .with_learnable(true);

        let product_metric = ProductMetric::try_new(
            base_metric.clone(),
            internal.clone(),
            Some(mixed),
            Some(warp),
        )
        .unwrap();
        let internal_space =
            InternalSpace::new("compact Z", InternalPatch::new("torus", vec!["ψ", "χ"]));
        let geometry = ProductGeometry::new(spacetime, internal_space, product_metric);
        let constants = PhysicalConstants::new(6.67430e-11, 299_792_458.0);
        ZRelativityModel::assemble(geometry, base_model, constants, 1.0, 0.0).unwrap()
    }

    #[test]
    fn forward_exposes_parameter_vector() {
        let model = build_model();
        let module = ZRelativityModule::from_model(model).unwrap();
        let dummy = Tensor::zeros(1, 1).unwrap();
        let output = module.forward(&dummy).unwrap();
        assert_eq!(output.shape(), (1, module.parameter_dimension()));
    }

    #[test]
    fn backward_accumulates_gradients() {
        let model = build_model();
        let mut module = ZRelativityModule::from_model(model).unwrap();
        let dummy = Tensor::zeros(1, 1).unwrap();
        let dimension = module.parameter_dimension();
        let grad = Tensor::from_vec(1, dimension, vec![1.0; dimension]).unwrap();
        let back = module.backward(&dummy, &grad).unwrap();
        assert_eq!(back.shape(), (1, 1));
        assert!(module.block.gradient().is_some());
        if let Some(gauge) = &module.gauge {
            assert!(gauge.gradient().is_some());
        }
        if let Some(scalar) = &module.scalar {
            assert!(scalar.gradient().is_some());
        }
        if let Some(warp) = &module.warp {
            assert!(warp.gradient().is_some());
        }
    }
}
