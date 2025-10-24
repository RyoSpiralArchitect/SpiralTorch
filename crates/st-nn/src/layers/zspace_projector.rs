// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::{PureResult, Tensor};
use st_frac::FracBackend;
use st_tensor::{LanguageWaveEncoder, OpenCartesianTopos, RewriteMonad, TensorBiome};

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
        Ok(projected)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.topos
            .guard_tensor("zspace_projector_backward_in", grad_output)?;
        let mut grad = grad_output.clone();
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor("zspace_projector_backward_rewrite", &mut grad)?;
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
}
