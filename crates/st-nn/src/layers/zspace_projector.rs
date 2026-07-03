use crate::module::Module;
use crate::{PureResult, Tensor};
use st_frac::FracBackend;
use st_tensor::pure::{LanguageWaveEncoder, OpenCartesianTopos, RewriteMonad, TensorBiome};

/// Projects Euclidean activations back into the open-cartesian Z-space manifold.
#[derive(Clone, Debug)]
pub struct ZSpaceProjector {
    topos: OpenCartesianTopos,
    encoder: LanguageWaveEncoder,
    strength: f32,
}

impl ZSpaceProjector {
    /// Builds a projector with a guard topos and matching Z-space encoder.
    pub fn new(topos: OpenCartesianTopos, encoder: LanguageWaveEncoder) -> PureResult<Self> {
        Self::with_strength(topos, encoder, 1.0)
    }

    /// Builds a projector that blends input activations with the Z-space projection.
    ///
    /// `strength=1.0` preserves the original full projection path, while
    /// `strength=0.0` behaves as an identity pass-through. Intermediate values
    /// are useful for sweeps that compare how much Z-space routing helps a
    /// fine-tuning run.
    pub fn with_strength(
        topos: OpenCartesianTopos,
        encoder: LanguageWaveEncoder,
        strength: f32,
    ) -> PureResult<Self> {
        if (topos.curvature() - encoder.curvature()).abs() > 1e-6 {
            return Err(crate::TensorError::CurvatureMismatch {
                expected: topos.curvature(),
                got: encoder.curvature(),
            });
        }
        Self::validate_strength(strength)?;
        Ok(Self {
            topos,
            encoder,
            strength,
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

    /// Returns the input/projection blend strength.
    pub fn strength(&self) -> f32 {
        self.strength
    }

    fn validate_strength(strength: f32) -> PureResult<()> {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(crate::TensorError::NonFiniteValue {
                label: "zspace_projector_strength",
                value: strength,
            });
        }
        Ok(())
    }

    fn blend(&self, input: &Tensor, projected: &Tensor) -> PureResult<Tensor> {
        if self.strength <= f32::EPSILON {
            return Ok(input.clone());
        }
        if (self.strength - 1.0).abs() <= f32::EPSILON {
            return Ok(projected.clone());
        }
        let mut blended = input.scale(1.0 - self.strength)?;
        blended.add_scaled(projected, self.strength)?;
        Ok(blended)
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
        let projected = self.blend(input, &projected)?;
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
        self.blend(grad_output, &grad)
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
    use st_tensor::pure::topos::OpenCartesianTopos;

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
    fn projector_strength_zero_is_identity() {
        let topos = demo_topos();
        let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.5).unwrap();
        let mut module = ZSpaceProjector::with_strength(topos, encoder, 0.0).unwrap();
        let input = Tensor::from_vec(2, 3, vec![0.2, -0.4, 0.6, -0.3, 0.5, -0.7]).unwrap();
        let output = module.forward(&input).unwrap();
        assert_eq!(output, input);

        let grad = Tensor::from_vec(2, 3, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6]).unwrap();
        let grad_in = module.backward(&input, &grad).unwrap();
        assert_eq!(grad_in, grad);
    }

    #[test]
    fn projector_strength_blends_with_full_projection() {
        let topos = demo_topos();
        let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.5).unwrap();
        let input = Tensor::from_vec(1, 3, vec![1.2, -0.8, 0.4]).unwrap();
        let full = ZSpaceProjector::new(topos.clone(), encoder.clone())
            .unwrap()
            .forward(&input)
            .unwrap();
        let blended = ZSpaceProjector::with_strength(topos, encoder, 0.5)
            .unwrap()
            .forward(&input)
            .unwrap();
        let expected = input.scale(0.5).unwrap();
        let mut expected = expected;
        expected.add_scaled(&full, 0.5).unwrap();
        assert_eq!(blended.shape(), input.shape());
        for (got, want) in blended.data().iter().zip(expected.data().iter()) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn projector_rejects_out_of_range_strength() {
        let topos = demo_topos();
        let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.5).unwrap();
        let err = ZSpaceProjector::with_strength(topos, encoder, 1.5).unwrap_err();
        assert!(matches!(
            err,
            crate::TensorError::NonFiniteValue {
                label: "zspace_projector_strength",
                ..
            }
        ));
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
