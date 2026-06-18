// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use st_core::theory::general_relativity::{ZRelativityModel, ZRelativityTensorBundle};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, PureResult, Tensor, TensorError};

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

#[derive(Clone, Copy, Debug, Default)]
struct TensorAbsSummary {
    finite: usize,
    non_finite: usize,
    mean_abs: f32,
    max_abs: f32,
}

fn tensor_abs_summary(tensor: &Tensor) -> TensorAbsSummary {
    let mut finite = 0usize;
    let mut non_finite = 0usize;
    let mut total = 0.0f64;
    let mut max_abs = 0.0f32;
    for value in tensor.data() {
        if value.is_finite() {
            let abs = value.abs();
            finite += 1;
            total += abs as f64;
            max_abs = max_abs.max(abs);
        } else {
            non_finite += 1;
        }
    }
    TensorAbsSummary {
        finite,
        non_finite,
        mean_abs: if finite == 0 {
            0.0
        } else {
            (total / finite as f64) as f32
        },
        max_abs,
    }
}

fn optional_segment_len(parameter: &Option<Parameter>, static_tensor: &Option<Tensor>) -> usize {
    parameter
        .as_ref()
        .map(|param| tensor_len(param.value()))
        .or_else(|| static_tensor.as_ref().map(tensor_len))
        .unwrap_or(0)
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

    fn learnable_segments(&self) -> usize {
        1 + usize::from(self.gauge.is_some())
            + usize::from(self.scalar.is_some())
            + usize::from(self.warp.is_some())
    }

    fn emit_forward_meta(&self, input: &Tensor, output: &Tensor) {
        let input_shape = input.shape();
        let output_shape = output.shape();
        let flags = self.model.learnable_flags();
        let values = tensor_abs_summary(output);
        emit_tensor_op(
            "zrelativity_module_forward",
            &[input_shape.0, input_shape.1],
            &[output_shape.0, output_shape.1],
        );
        emit_tensor_op_meta("zrelativity_module_forward", || {
            serde_json::json!({
                "backend": "parameter_adapter",
                "requested_backend": "host",
                "kind": "st_nn_zrelativity_module_forward",
                "host_backend": "cpu",
                "input_rows": input_shape.0,
                "input_cols": input_shape.1,
                "output_rows": output_shape.0,
                "output_cols": output_shape.1,
                "parameter_dimension": self.vector_length,
                "total_dim": self.model.geometry.total_dimension(),
                "internal_dim": self.model.geometry.internal().dimension(),
                "has_warp": self.warp.is_some() || self.warp_static.is_some(),
                "learnable_segments": self.learnable_segments(),
                "learnable_warp": flags.warp,
                "learnable_mixed": flags.mixed,
                "learnable_internal": flags.internal,
                "block_len": tensor_len(self.block.value()),
                "gauge_len": optional_segment_len(&self.gauge, &self.gauge_static),
                "scalar_len": optional_segment_len(&self.scalar, &self.scalar_static),
                "warp_len": optional_segment_len(&self.warp, &self.warp_static),
                "value_finite": values.finite,
                "value_non_finite": values.non_finite,
                "value_abs_mean": values.mean_abs,
                "value_abs_max": values.max_abs,
            })
        });
    }

    fn emit_backward_meta(&self, input: &Tensor, grad_output: &Tensor, grad_input: &Tensor) {
        let input_shape = input.shape();
        let grad_output_shape = grad_output.shape();
        let grad_input_shape = grad_input.shape();
        let flags = self.model.learnable_flags();
        let values = tensor_abs_summary(grad_output);
        emit_tensor_op(
            "zrelativity_module_backward",
            &[grad_output_shape.0, grad_output_shape.1],
            &[grad_input_shape.0, grad_input_shape.1],
        );
        emit_tensor_op_meta("zrelativity_module_backward", || {
            serde_json::json!({
                "backend": "parameter_adapter",
                "requested_backend": "host",
                "kind": "st_nn_zrelativity_module_backward",
                "host_backend": "cpu",
                "input_rows": input_shape.0,
                "input_cols": input_shape.1,
                "grad_output_rows": grad_output_shape.0,
                "grad_output_cols": grad_output_shape.1,
                "grad_input_rows": grad_input_shape.0,
                "grad_input_cols": grad_input_shape.1,
                "parameter_dimension": self.vector_length,
                "total_dim": self.model.geometry.total_dimension(),
                "internal_dim": self.model.geometry.internal().dimension(),
                "learnable_segments": self.learnable_segments(),
                "learnable_warp": flags.warp,
                "learnable_mixed": flags.mixed,
                "learnable_internal": flags.internal,
                "block_len": tensor_len(self.block.value()),
                "gauge_len": optional_segment_len(&self.gauge, &self.gauge_static),
                "scalar_len": optional_segment_len(&self.scalar, &self.scalar_static),
                "warp_len": optional_segment_len(&self.warp, &self.warp_static),
                "grad_finite": values.finite,
                "grad_non_finite": values.non_finite,
                "grad_abs_mean": values.mean_abs,
                "grad_abs_max": values.max_abs,
            })
        });
    }
}

impl Module for ZRelativityModule {
    fn forward(&self, _input: &Tensor) -> PureResult<Tensor> {
        let data = self.aggregate_values()?;
        let output = Tensor::from_vec(1, self.vector_length, data)?;
        self.emit_forward_meta(_input, &output);
        Ok(output)
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
        let grad_input = Tensor::zeros(input.shape().0, input.shape().1)?;
        self.emit_backward_meta(input, grad_output, &grad_input);
        Ok(grad_input)
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
    use std::sync::{Arc, Mutex, OnceLock};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        static OBSERVER_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        OBSERVER_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("observer lock available")
    }

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

    #[test]
    fn forward_backward_emit_backend_meta() {
        let model = build_model();
        let mut module = ZRelativityModule::from_model(model).unwrap();
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let dummy = Tensor::zeros(1, 1).unwrap();
        let output = module.forward(&dummy).unwrap();
        let dimension = module.parameter_dimension();
        let grad = Tensor::from_vec(1, dimension, vec![1.0; dimension]).unwrap();
        let back = module.backward(&dummy, &grad).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(output.shape(), (1, dimension));
        assert_eq!(back.shape(), (1, 1));
        let events = events.lock().unwrap();
        let forward_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zrelativity_module_forward"
                    && data["kind"] == "st_nn_zrelativity_module_forward"
            })
            .expect("zrelativity_module_forward metadata event");
        assert_eq!(forward_meta.1["backend"], "parameter_adapter");
        assert_eq!(forward_meta.1["requested_backend"], "host");
        assert_eq!(forward_meta.1["host_backend"], "cpu");
        assert_eq!(forward_meta.1["parameter_dimension"], dimension);
        assert_eq!(forward_meta.1["total_dim"], 6);
        assert_eq!(forward_meta.1["internal_dim"], 2);
        assert_eq!(forward_meta.1["learnable_segments"], 4);
        assert_eq!(forward_meta.1["learnable_warp"], true);
        assert_eq!(forward_meta.1["learnable_mixed"], true);
        assert_eq!(forward_meta.1["learnable_internal"], true);
        assert_eq!(forward_meta.1["output_cols"], dimension);
        assert!(forward_meta.1["value_abs_max"].as_f64().unwrap_or(0.0) > 0.0);

        let backward_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zrelativity_module_backward"
                    && data["kind"] == "st_nn_zrelativity_module_backward"
            })
            .expect("zrelativity_module_backward metadata event");
        assert_eq!(backward_meta.1["backend"], "parameter_adapter");
        assert_eq!(backward_meta.1["requested_backend"], "host");
        assert_eq!(backward_meta.1["host_backend"], "cpu");
        assert_eq!(backward_meta.1["parameter_dimension"], dimension);
        assert_eq!(backward_meta.1["grad_output_cols"], dimension);
        assert_eq!(backward_meta.1["grad_input_rows"], 1);
        assert_eq!(backward_meta.1["grad_input_cols"], 1);
        assert_eq!(backward_meta.1["learnable_segments"], 4);
        assert!(backward_meta.1["grad_abs_mean"].as_f64().unwrap_or(0.0) > 0.0);
    }
}
