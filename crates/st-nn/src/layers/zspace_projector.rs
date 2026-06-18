// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::Module;
use crate::{PureResult, Tensor, TensorError};
use st_frac::mellin_types::{ComplexScalar, Scalar};
use st_frac::zspace::{
    evaluate_weighted_series_many, prepare_weighted_series, trapezoidal_weights,
};
use st_frac::FracBackend;
use st_tensor::{
    emit_tensor_op, emit_tensor_op_meta, LanguageWaveEncoder, OpenCartesianTopos, RewriteMonad,
    TensorBiome,
};
use thiserror::Error;

fn emit_zspace_projector_meta(
    op_name: &'static str,
    layer_backend: &'static str,
    rows: usize,
    cols: usize,
    backward: bool,
    curvature: f32,
    projection_backend: Option<String>,
    projection_gradient: bool,
    saturation_gradient: &'static str,
) {
    emit_tensor_op(op_name, &[rows, cols], &[rows, cols]);
    emit_tensor_op_meta(op_name, || {
        let values = rows.saturating_mul(cols);
        let projection_gradient_backend = if projection_gradient {
            Some("cpu")
        } else {
            None
        };
        let saturation_gradient_backend = if backward { Some("cpu") } else { None };
        serde_json::json!({
            "backend": layer_backend,
            "requested_backend": "auto",
            "kernel": "zspace_projector.rewrite",
            "kind": if backward { "zspace_projector_backward" } else { "zspace_projector_forward" },
            "rows": rows,
            "cols": cols,
            "values": values,
            "output_rows": rows,
            "output_cols": cols,
            "output_values": values,
            "curvature": curvature,
            "rewrite_backend": "cpu",
            "projection_backend": projection_backend,
            "projection_gradient": projection_gradient,
            "projection_gradient_backend": projection_gradient_backend,
            "saturation_gradient": saturation_gradient,
            "saturation_gradient_backend": saturation_gradient_backend,
            "estimated_rewrite_values": values,
            "estimated_projection_values": values,
            "estimated_projection_gradient_ops": if backward { values.saturating_mul(4) } else { 0 },
            "estimated_saturation_gradient_ops": if backward { values.saturating_mul(3) } else { 0 },
            "empty": rows == 0 || cols == 0,
        })
    });
}

fn porous_saturation_backward_factor(value: f32, saturation: f32, porosity: f32) -> f32 {
    if !value.is_finite() || saturation <= 0.0 {
        return 0.0;
    }
    let limit = saturation.abs();
    let magnitude = value.abs();
    if magnitude <= limit {
        return 1.0;
    }
    if porosity <= f32::EPSILON {
        return 0.0;
    }
    let absorb = (porosity * 0.25).min(1.0);
    let denom = magnitude + limit;
    if denom <= f32::EPSILON {
        return 0.0;
    }
    -2.0 * limit * limit * absorb / (denom * denom)
}

fn porous_saturation_backward(
    pre_saturation: &Tensor,
    grad_saturated: &Tensor,
    saturation: f32,
    porosity: f32,
) -> PureResult<Tensor> {
    if pre_saturation.shape() != grad_saturated.shape() {
        return Err(TensorError::ShapeMismatch {
            left: pre_saturation.shape(),
            right: grad_saturated.shape(),
        });
    }
    let (rows, cols) = pre_saturation.shape();
    let data = pre_saturation
        .data()
        .iter()
        .zip(grad_saturated.data().iter())
        .map(|(&value, &grad)| {
            grad * porous_saturation_backward_factor(value, saturation, porosity)
        })
        .collect();
    Tensor::from_vec(rows, cols, data)
}

fn poincare_projection_backward(
    preprojected: &Tensor,
    grad_projected: &Tensor,
    curvature: f32,
) -> PureResult<Tensor> {
    if preprojected.shape() != grad_projected.shape() {
        return Err(TensorError::ShapeMismatch {
            left: preprojected.shape(),
            right: grad_projected.shape(),
        });
    }
    if curvature >= 0.0 {
        return Err(TensorError::NonHyperbolicCurvature { curvature });
    }
    let (rows, cols) = preprojected.shape();
    let scale = (-curvature).sqrt();
    let mut data = vec![0.0f32; rows.saturating_mul(cols)];
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let x = &preprojected.data()[start..end];
        let grad = &grad_projected.data()[start..end];
        let norm = x.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= f32::EPSILON {
            let factor = if scale > 0.0 { 1.0 / scale } else { 1.0 };
            for col in 0..cols {
                data[start + col] = grad[col] * factor;
            }
            continue;
        }
        let tanh = (norm / scale).tanh();
        let factor = tanh / norm;
        let sech2 = 1.0 - tanh * tanh;
        let radial = ((sech2 * norm / scale) - tanh) / (norm * norm * norm);
        let dot = x
            .iter()
            .zip(grad.iter())
            .map(|(&value, &grad)| value * grad)
            .sum::<f32>();
        for col in 0..cols {
            data[start + col] = factor * grad[col] + radial * x[col] * dot;
        }
    }
    Tensor::from_vec(rows, cols, data)
}

/// Projects Euclidean activations back into the open-cartesian Z-space manifold.
#[derive(Clone, Debug)]
pub struct ZSpaceProjector {
    topos: OpenCartesianTopos,
    encoder: LanguageWaveEncoder,
}

impl ZSpaceProjector {
    /// Builds a projector with a guard topos and matching Z-space encoder.
    pub fn new(topos: OpenCartesianTopos, encoder: LanguageWaveEncoder) -> PureResult<Self> {
        if (topos.curvature() - encoder.curvature()).abs() > 1e-6 {
            return Err(crate::TensorError::CurvatureMismatch {
                expected: topos.curvature(),
                got: encoder.curvature(),
            });
        }
        Ok(Self { topos, encoder })
    }

    /// Returns the open-cartesian guard backing the projector.
    pub fn topos(&self) -> &OpenCartesianTopos {
        &self.topos
    }

    /// Returns the curvature enforced by the projector.
    pub fn curvature(&self) -> f32 {
        self.topos.curvature()
    }

    /// Returns the internal Z-space encoder for direct wave creation.
    pub fn encoder(&self) -> &LanguageWaveEncoder {
        &self.encoder
    }

    /// Applies a fractional regularisation penalty to the provided latent slice.
    pub fn regularize_frac(&self, z: &[f32], backend: &FracBackend) -> f32 {
        if z.len() < 2 {
            return 0.0;
        }
        let curvature = self.topos.curvature().abs().max(1e-6);
        let mut acc = 0.0f32;
        for window in z.windows(2) {
            let diff = window[1] - window[0];
            acc += diff.abs().powf(1.0 + curvature);
        }
        match backend {
            FracBackend::CpuRadix2 => acc,
            FracBackend::Wgpu { radix } => {
                let factor = (*radix as f32).max(2.0) / 2.0;
                acc * (1.0 + 0.25 * (factor - 1.0))
            }
        }
    }

    /// Encodes free-form text into a tensor already guarded by the projector.
    pub fn encode_text(&self, text: &str) -> PureResult<Tensor> {
        let tensor = self.encoder.encode_z_space(text)?;
        self.topos
            .guard_tensor("zspace_projector_encode", &tensor)?;
        Ok(tensor)
    }

    /// Collapses a tensor biome and projects the resulting canopy back into Z-space.
    pub fn reimport_biome(&self, biome: &TensorBiome) -> PureResult<Tensor> {
        if biome.is_empty() {
            return Err(crate::TensorError::EmptyInput("tensor_biome"));
        }
        if (biome.topos().curvature() - self.curvature()).abs() > 1e-6 {
            return Err(crate::TensorError::CurvatureMismatch {
                expected: self.curvature(),
                got: biome.topos().curvature(),
            });
        }
        let (rows, cols) = biome
            .shape()
            .ok_or(crate::TensorError::EmptyInput("tensor_biome"))?;
        let stacked_values = biome.len().saturating_mul(rows).saturating_mul(cols);
        let canopy_backend = current_tensor_util_backend_for_values(stacked_values);
        let canopy = biome.canopy_with_backend(canopy_backend)?;
        self.forward(&canopy)
    }
}

impl Module for ZSpaceProjector {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        self.topos
            .guard_tensor("zspace_projector_forward_in", input)?;
        let mut rewritten = input.clone();
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor("zspace_projector_forward_rewrite", &mut rewritten)?;
        let projection_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let projected = rewritten
            .project_to_poincare_with_backend(self.topos.curvature(), projection_backend)?;
        self.topos
            .guard_tensor("zspace_projector_forward_out", &projected)?;
        emit_zspace_projector_meta(
            "zspace_projector_forward",
            "composite",
            rows,
            cols,
            false,
            self.topos.curvature(),
            Some(projection_backend.to_string()),
            false,
            "forward_only",
        );
        Ok(projected)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, cols) = input.shape();
        self.topos
            .guard_tensor("zspace_projector_backward_in", grad_output)?;
        let monad = RewriteMonad::new(&self.topos);
        let mut preprojected = input.clone();
        monad.rewrite_tensor("zspace_projector_backward_rewrite", &mut preprojected)?;
        let grad_saturated =
            poincare_projection_backward(&preprojected, grad_output, self.topos.curvature())?;
        let grad = porous_saturation_backward(
            input,
            &grad_saturated,
            self.topos.saturation(),
            self.topos.porosity(),
        )?;
        emit_zspace_projector_meta(
            "zspace_projector_backward",
            "cpu",
            rows,
            cols,
            true,
            self.topos.curvature(),
            Some("cpu".to_string()),
            true,
            "porous_mix_exact",
        );
        Ok(grad)
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

// ===== SpiralTorch: Z-space projector (add-on) =====
#[derive(Debug, Error)]
pub enum ProjectorError {
    #[error("input samples must not be empty")]
    EmptySamples,
    #[error("z-points must not be empty")]
    EmptyZ,
    #[error(transparent)]
    Mellin(#[from] st_frac::mellin_types::MellinError),
    #[error(transparent)]
    ZSpace(#[from] st_frac::mellin_types::ZSpaceError),
}

pub struct StableZSpaceProjector {
    pub log_start: Scalar,
    pub log_step: Scalar,
    pub z_points: Vec<ComplexScalar>,
}

impl StableZSpaceProjector {
    pub fn new(
        log_start: Scalar,
        log_step: Scalar,
        z_points: Vec<ComplexScalar>,
    ) -> Result<Self, ProjectorError> {
        if z_points.is_empty() {
            return Err(ProjectorError::EmptyZ);
        }
        Ok(Self {
            log_start,
            log_step,
            z_points,
        })
    }

    /// 1系列を z_points へ射影（Horner＋非有限チェックは st-frac 側に内包）
    pub fn project_series(
        &self,
        samples: &[ComplexScalar],
    ) -> Result<Vec<ComplexScalar>, ProjectorError> {
        if samples.is_empty() {
            return Err(ProjectorError::EmptySamples);
        }
        let weights = trapezoidal_weights(samples.len())?;
        let weighted = prepare_weighted_series(samples, &weights)?;
        Ok(evaluate_weighted_series_many(&weighted, &self.z_points)?)
    }
}
// ===== end additions =====

#[cfg(test)]
mod tests {
    use super::*;
    use st_tensor::topos::OpenCartesianTopos;
    use std::sync::{Arc, Mutex, OnceLock};

    fn demo_topos() -> OpenCartesianTopos {
        OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap()
    }

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        static OBSERVER_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        OBSERVER_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("observer lock available")
    }

    #[test]
    fn projector_rewrites_forward_and_backward() {
        let topos = demo_topos();
        let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.5).unwrap();
        let mut module = ZSpaceProjector::new(topos, encoder).unwrap();
        let input =
            Tensor::from_vec(2, 4, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]).unwrap();
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        let grad_out =
            Tensor::from_vec(2, 4, vec![0.2, -0.1, 0.05, -0.3, 0.4, -0.2, 0.1, -0.05]).unwrap();
        let grad_in = module.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), grad_out.shape());
    }

    #[test]
    fn projector_forward_emits_poincare_projection_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let topos = demo_topos();
        let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.5).unwrap();
        let module = ZSpaceProjector::new(topos, encoder).unwrap();
        let input = Tensor::from_vec(1, 4, vec![0.1, -0.2, 0.3, -0.4]).unwrap();
        let output = module.forward(&input).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(output.shape(), input.shape());
        let events = events.lock().unwrap();
        let projection = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "project_to_poincare"
                    && data["rows"] == 1
                    && data["cols"] == 4
                    && data["kind"] == "hyperbolic_projection"
            })
            .expect("project_to_poincare metadata event");
        assert_eq!(projection.1["backend"], "cpu");
        assert_eq!(projection.1["requested_backend"], "auto");
        assert_eq!(projection.1["curvature"], -1.0);
        assert_eq!(projection.1["output_rows"], 1);
        assert_eq!(projection.1["output_cols"], 4);

        let projector = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_projector_forward"
                    && data["rows"] == 1
                    && data["cols"] == 4
                    && data["kind"] == "zspace_projector_forward"
            })
            .expect("zspace_projector_forward metadata event");
        assert_eq!(projector.1["backend"], "composite");
        assert_eq!(projector.1["rewrite_backend"], "cpu");
        assert_eq!(projector.1["projection_backend"], "auto");
        assert_eq!(projector.1["projection_gradient"], false);
        assert!(projector.1["projection_gradient_backend"].is_null());
        assert_eq!(projector.1["saturation_gradient"], "forward_only");
        assert!(projector.1["saturation_gradient_backend"].is_null());
        assert_eq!(projector.1["curvature"], -1.0);
    }

    #[test]
    fn projector_backward_emits_layer_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let topos = demo_topos();
        let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.5).unwrap();
        let mut module = ZSpaceProjector::new(topos, encoder).unwrap();
        let input = Tensor::from_vec(1, 4, vec![0.1, -0.2, 0.3, -0.4]).unwrap();
        let grad_out = Tensor::from_vec(1, 4, vec![0.2, -0.1, 0.05, -0.3]).unwrap();
        let grad_in = module.backward(&input, &grad_out).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(grad_in.shape(), grad_out.shape());
        let events = events.lock().unwrap();
        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_projector_backward"
                    && data["rows"] == 1
                    && data["cols"] == 4
                    && data["kind"] == "zspace_projector_backward"
            })
            .expect("zspace_projector_backward metadata event");
        assert_eq!(backward.1["backend"], "cpu");
        assert_eq!(backward.1["rewrite_backend"], "cpu");
        assert_eq!(backward.1["projection_backend"], "cpu");
        assert_eq!(backward.1["projection_gradient"], true);
        assert_eq!(backward.1["projection_gradient_backend"], "cpu");
        assert_eq!(backward.1["saturation_gradient"], "porous_mix_exact");
        assert_eq!(backward.1["saturation_gradient_backend"], "cpu");
        assert_eq!(backward.1["estimated_projection_values"], 4);
        assert_eq!(backward.1["estimated_projection_gradient_ops"], 16);
    }

    #[test]
    fn projector_backward_matches_projection_finite_difference_for_input() {
        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 0.5, 256, 8192)
            .unwrap()
            .with_porosity(0.8)
            .unwrap();
        let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.5).unwrap();
        let mut module = ZSpaceProjector::new(topos, encoder).unwrap();
        let input_values = vec![0.9, -0.35, 0.25];
        let input = Tensor::from_vec(1, 3, input_values.clone()).unwrap();
        let grad_out = Tensor::from_vec(1, 3, vec![0.4, -0.25, 0.3]).unwrap();

        let grad_input = module.backward(&input, &grad_out).unwrap();
        let analytic = grad_input.data()[0];

        let epsilon = 1.0e-3f32;
        let loss_at = |values: Vec<f32>| {
            let tensor = Tensor::from_vec(1, 3, values).unwrap();
            let out = module.forward(&tensor).unwrap();
            out.data()
                .iter()
                .zip(grad_out.data().iter())
                .map(|(&value, &grad)| value * grad)
                .sum::<f32>()
        };
        let mut plus = input_values.clone();
        plus[0] += epsilon;
        let mut minus = input_values;
        minus[0] -= epsilon;
        let finite_difference = (loss_at(plus) - loss_at(minus)) / (2.0 * epsilon);

        assert!(
            (analytic - finite_difference).abs() < 3.0e-3,
            "analytic={analytic} finite_difference={finite_difference}"
        );
    }

    #[test]
    fn projector_encodes_text_into_guarded_tensor() {
        let topos = demo_topos();
        let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.75).unwrap();
        let module = ZSpaceProjector::new(topos, encoder).unwrap();
        let encoded = module
            .encode_text("SpiralTorch expands the open topos")
            .unwrap();
        assert_eq!(encoded.shape().0, 1);
        assert!(encoded.shape().1 >= 2);
    }

    #[test]
    fn fractional_regularizer_scales_with_backend() {
        let topos = demo_topos();
        let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.5).unwrap();
        let module = ZSpaceProjector::new(topos, encoder).unwrap();
        let sample = vec![0.0, 0.5, -0.2, 0.8, -0.4];
        let cpu = module.regularize_frac(&sample, &FracBackend::CpuRadix2);
        let wgpu = module.regularize_frac(&sample, &FracBackend::Wgpu { radix: 4 });
        assert!(wgpu > cpu);
    }

    #[test]
    fn projector_reimports_biome() {
        let topos = demo_topos();
        let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.5).unwrap();
        let biome_topos = topos.clone();
        let mut biome = TensorBiome::new(biome_topos);
        biome
            .absorb(
                "projector_biome_a",
                Tensor::from_vec(1, 4, vec![0.2, -0.4, 0.6, -0.8]).unwrap(),
            )
            .unwrap();
        biome
            .absorb(
                "projector_biome_b",
                Tensor::from_vec(1, 4, vec![0.4, -0.2, 0.8, -0.6]).unwrap(),
            )
            .unwrap();

        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));
        let projector = ZSpaceProjector::new(topos, encoder).unwrap();
        let projected = projector.reimport_biome(&biome).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(projected.shape(), (1, 4));
        let events = events.lock().unwrap();
        let canopy = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "tensor_biome_canopy" && data["rows"] == 1 && data["cols"] == 4
            })
            .expect("tensor_biome_canopy metadata event");
        assert_eq!(canopy.1["backend"], "hybrid");
        assert_eq!(canopy.1["accumulation_backend"], "auto");
        assert_eq!(canopy.1["normalise_backend"], "auto");
        assert_eq!(canopy.1["rewrite_backend"], "topos_cpu");
        assert_eq!(canopy.1["kind"], "topos_biome_canopy");

        let projection = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "project_to_poincare" && data["rows"] == 1 && data["cols"] == 4
            })
            .expect("project_to_poincare metadata event");
        assert_eq!(projection.1["backend"], "cpu");
        assert_eq!(projection.1["kind"], "hyperbolic_projection");
    }
}
