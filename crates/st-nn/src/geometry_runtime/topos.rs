use super::{tensor, GeometryAdapterPlacement, NativeGeometryPullback, Result};
use crate::geometry_artifact::ToposRecipe;
use st_core::dynamics::topos_resonator::{
    ToposResonatorConfig, ToposResonatorLearningBatch, ToposResonatorOperator,
};
use st_tensor::{topos::OpenCartesianTopos, Tensor};
use std::collections::BTreeMap;

#[derive(Debug)]
pub(super) struct ToposRuntimeBatch {
    native: ToposResonatorLearningBatch,
    output: Tensor,
}

impl ToposRuntimeBatch {
    pub(super) fn forward(
        placement: &GeometryAdapterPlacement,
        recipe: &ToposRecipe,
        input: &Tensor,
    ) -> Result<Self> {
        let config = ToposResonatorConfig::new(recipe.coupling, recipe.iterations)?;
        let topos = OpenCartesianTopos::new(
            -1.,
            1e-6,
            recipe.saturation,
            recipe.iterations + 1,
            recipe.max_values,
        )?
        .with_porosity(recipe.porosity)?;
        let (rows, features) = input.shape();
        let native = ToposResonatorOperator::new(config, topos)?.capture_shared_rows(
            input.data(),
            tensor(placement, "gate"),
            rows,
            features,
        )?;
        let output = Tensor::from_vec(rows, features, native.output().to_vec())?;
        Ok(Self { native, output })
    }

    pub(super) fn output(&self) -> &Tensor {
        &self.output
    }

    pub(super) fn vjp(
        &self,
        upstream: &Tensor,
        need_input: bool,
    ) -> Result<NativeGeometryPullback> {
        let (rows, features) = self.output.shape();
        let (input, gate) = if need_input {
            let gradient = self.native.vjp(upstream.data())?;
            (
                Some(Tensor::from_vec(rows, features, gradient.grad_input)?),
                gradient.grad_gate,
            )
        } else {
            (None, self.native.vjp_gate(upstream.data())?)
        };
        Ok(NativeGeometryPullback {
            input,
            parameters: BTreeMap::from([("gate".into(), gate)]),
        })
    }
}
