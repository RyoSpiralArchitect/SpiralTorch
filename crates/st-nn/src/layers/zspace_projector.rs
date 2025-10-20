// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use st_frac::FracBackend;
use st_tensor::{LanguageWaveEncoder, OpenCartesianTopos, RewriteMonad, TensorBiome};

/// Projects Euclidean activations back into the open-cartesian Z-space manifold while
/// exposing trainable focus and spread controls for adaptive gating.
#[derive(Debug)]
pub struct ZSpaceProjector {
    topos: OpenCartesianTopos,
    encoder: LanguageWaveEncoder,
    focus: Parameter,
    spread: Parameter,
}

impl ZSpaceProjector {
    /// Builds a projector with a guard topos and matching Z-space encoder.
    pub fn new(topos: OpenCartesianTopos, encoder: LanguageWaveEncoder) -> PureResult<Self> {
        if (topos.curvature() - encoder.curvature()).abs() > 1e-6 {
            return Err(TensorError::CurvatureMismatch {
                expected: topos.curvature(),
                got: encoder.curvature(),
            });
        }
        let focus = Parameter::new("zspace_projector_focus", Tensor::from_vec(1, 1, vec![1.0])?);
        let spread = Parameter::new(
            "zspace_projector_spread",
            Tensor::from_vec(1, 1, vec![0.0])?,
        );
        Ok(Self {
            topos,
            encoder,
            focus,
            spread,
        })
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

    /// Returns the learnable focus scalar that contracts or amplifies each slice.
    pub fn focus(&self) -> &Parameter {
        &self.focus
    }

    /// Returns a mutable handle to the focus parameter.
    pub fn focus_mut(&mut self) -> &mut Parameter {
        &mut self.focus
    }

    /// Returns the learnable spread scalar that redistributes energy across a slice.
    pub fn spread(&self) -> &Parameter {
        &self.spread
    }

    /// Returns a mutable handle to the spread parameter.
    pub fn spread_mut(&mut self) -> &mut Parameter {
        &mut self.spread
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
            return Err(TensorError::EmptyInput("tensor_biome"));
        }
        if (biome.topos().curvature() - self.curvature()).abs() > 1e-6 {
            return Err(TensorError::CurvatureMismatch {
                expected: self.curvature(),
                got: biome.topos().curvature(),
            });
        }
        let canopy = biome.canopy()?;
        self.forward(&canopy)
    }
}

impl Module for ZSpaceProjector {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.topos
            .guard_tensor("zspace_projector_forward_in", input)?;
        let mut rewritten = input.clone();
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor("zspace_projector_forward_rewrite", &mut rewritten)?;
        let projected = rewritten.project_to_poincare(self.topos.curvature())?;
        self.topos
            .guard_tensor("zspace_projector_forward_out", &projected)?;

        let focus = self.focus.value().data()[0];
        let spread = self.spread.value().data()[0];
        let (rows, cols) = projected.shape();
        let mut data = projected.data().to_vec();
        for r in 0..rows {
            let offset = r * cols;
            let mut mean = 0.0f32;
            for c in 0..cols {
                mean += data[offset + c];
            }
            mean /= cols as f32;
            for c in 0..cols {
                data[offset + c] = data[offset + c] * focus + mean * spread;
            }
        }
        let output = Tensor::from_vec(rows, cols, data)?;
        self.topos
            .guard_tensor("zspace_projector_forward_focus", &output)?;
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.topos
            .guard_tensor("zspace_projector_backward_in", grad_output)?;
        let focus = self.focus.value().data()[0];
        let spread = self.spread.value().data()[0];

        let mut rewritten = input.clone();
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor("zspace_projector_backward_rewrite_input", &mut rewritten)?;
        let projected = rewritten.project_to_poincare(self.topos.curvature())?;

        let (rows, cols) = grad_output.shape();
        if projected.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: projected.shape(),
                right: grad_output.shape(),
            });
        }

        let mut grad_input_data = vec![0.0f32; rows * cols];
        let mut focus_grad = 0.0f32;
        let mut spread_grad = 0.0f32;

        for r in 0..rows {
            let offset = r * cols;
            let row_output = &grad_output.data()[offset..offset + cols];
            let row_projected = &projected.data()[offset..offset + cols];
            let sum_grad: f32 = row_output.iter().copied().sum();
            let mut mean_proj = 0.0f32;
            for value in row_projected {
                mean_proj += *value;
            }
            mean_proj /= cols as f32;

            for c in 0..cols {
                let grad_out = row_output[c];
                let proj = row_projected[c];
                grad_input_data[offset + c] += grad_out * focus + sum_grad * spread / cols as f32;
                focus_grad += grad_out * proj;
            }
            spread_grad += sum_grad * mean_proj;
        }

        let focus_update = Tensor::from_vec(1, 1, vec![focus_grad])?;
        let spread_update = Tensor::from_vec(1, 1, vec![spread_grad])?;
        self.focus.accumulate_euclidean(&focus_update)?;
        self.spread.accumulate_euclidean(&spread_update)?;

        let mut grad_tensor = Tensor::from_vec(rows, cols, grad_input_data)?;
        monad.rewrite_tensor("zspace_projector_backward_rewrite_out", &mut grad_tensor)?;
        Ok(grad_tensor)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.focus)?;
        visitor(&self.spread)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.focus)?;
        visitor(&mut self.spread)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_tensor::topos::OpenCartesianTopos;

    fn demo_topos() -> OpenCartesianTopos {
        OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap()
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
        let projector = ZSpaceProjector::new(topos, encoder).unwrap();
        let projected = projector.reimport_biome(&biome).unwrap();
        assert_eq!(projected.shape(), (1, 4));
    }

    #[test]
    fn projector_focus_and_spread_accumulate_gradients() {
        let topos = demo_topos();
        let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.6).unwrap();
        let mut module = ZSpaceProjector::new(topos, encoder).unwrap();
        let input = Tensor::from_vec(1, 4, vec![0.2, -0.1, 0.4, -0.3]).unwrap();
        let _ = module.forward(&input).unwrap();

        let grad_out = Tensor::from_vec(1, 4, vec![0.5, -0.25, 0.1, 0.2]).unwrap();
        let _ = module.backward(&input, &grad_out).unwrap();

        let focus_grad = module
            .focus()
            .gradient()
            .expect("focus gradient to accumulate");
        let spread_grad = module
            .spread()
            .gradient()
            .expect("spread gradient to accumulate");
        assert_eq!(focus_grad.shape(), (1, 1));
        assert_eq!(spread_grad.shape(), (1, 1));
    }
}
