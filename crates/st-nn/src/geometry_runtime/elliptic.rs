//! Native composition of the existing linear primitives and elliptic pullback.

use super::{
    tensor, GeometryAdapterPlacement, GeometryAdapterRecipe, NativeGeometryPullback, Result,
};
use crate::geometry_artifact::{EllipticRecipe, GEOMETRY_ADAPTER_MAX_VALUES};
use st_core::theory::microlocal::{
    EllipticAnchoredLearningBatch, EllipticCausalLearningBatch, EllipticGatedCausalLearningBatch,
    EllipticLearningBatch, EllipticWarp,
};
use st_tensor::{MatmulBackend, PackedB, Tensor, TensorUtilBackend, Tile};
use std::collections::BTreeMap;

const MAX_ROWS: usize = 65_536;
const MAX_PAIRS: usize = 1_048_576;
const MATMUL: MatmulBackend = MatmulBackend::CpuSimd;

pub(super) fn value_budget(features: usize) -> usize {
    GEOMETRY_ADAPTER_MAX_VALUES.min(features.saturating_mul(MAX_ROWS))
}

pub(super) fn sequence_values(features: usize, shape: [usize; 2]) -> Result<usize> {
    let rows = shape[0]
        .checked_mul(shape[1])
        .filter(|&n| n > 0 && n <= MAX_ROWS)
        .ok_or("native elliptic sequence exceeds nonempty row budget")?;
    rows.checked_mul(shape[1])
        .filter(|&n| n <= MAX_PAIRS)
        .ok_or("native elliptic sequence exceeds score-pair budget")?;
    rows.checked_mul(features)
        .filter(|&n| n <= GEOMETRY_ADAPTER_MAX_VALUES)
        .ok_or_else(|| "native elliptic sequence exceeds value budget".into())
}

fn finite(value: Tensor) -> Result<Tensor> {
    if value.data().iter().any(|v| !v.is_finite()) {
        return Err("native elliptic composition produced a non-finite tensor".into());
    }
    Ok(value)
}

#[derive(Debug)]
enum Map {
    Pointwise(EllipticLearningBatch),
    Anchored(EllipticAnchoredLearningBatch),
    Causal(EllipticCausalLearningBatch),
    Gated(EllipticGatedCausalLearningBatch),
}

#[derive(Debug)]
pub(super) struct EllipticRuntimeBatch {
    input: Tensor,
    orientation_weight: Tensor,
    readout_weight: Tensor,
    features: Tensor,
    map: Map,
    output: Tensor,
}

impl EllipticRuntimeBatch {
    pub(super) fn forward(
        placement: &GeometryAdapterPlacement,
        recipe: &EllipticRecipe,
        input: &Tensor,
        sequence: Option<[usize; 2]>,
    ) -> Result<Self> {
        let (rows, width) = input.shape();
        let orientation_weight =
            Tensor::from_vec(2, width, tensor(placement, "orientation.weight").to_vec())?;
        let readout_weight =
            Tensor::from_vec(width, 9, tensor(placement, "readout.weight").to_vec())?;
        // Artifact weights use torch-compatible [output, input] storage. Pack
        // their transpose directly rather than materializing another tensor.
        let coordinates = finite(input.matmul_prepacked_bias_with_backend(
            &PackedB::from_tensor_transpose(&orientation_weight, Tile::col_major())?,
            tensor(placement, "orientation.bias"),
            MATMUL,
        )?)?;
        let mut orientations = Vec::with_capacity(rows * 3);
        for pair in coordinates.data().chunks_exact(2) {
            orientations.extend_from_slice(&[1.0, pair[0], pair[1]]);
        }
        let warp = EllipticWarp::for_learning(
            recipe.curvature_radius,
            recipe.sheet_count,
            recipe.spin_harmonics,
        )?;
        let map = match placement.recipe {
            GeometryAdapterRecipe::EllipticAnchored { .. } => {
                Map::Anchored(warp.differentiate_anchored_batch(
                    &orientations,
                    tensor(placement, "raw_mix")[0],
                    MAX_ROWS,
                )?)
            }
            GeometryAdapterRecipe::EllipticCausal { .. } => {
                let [batch, steps] =
                    sequence.ok_or("causal elliptic execution requires explicit sequence axes")?;
                Map::Causal(warp.differentiate_causal_batch(
                    &orientations,
                    batch,
                    steps,
                    MAX_ROWS,
                    MAX_PAIRS,
                )?)
            }
            GeometryAdapterRecipe::EllipticGated { .. } => {
                let shape =
                    sequence.ok_or("gated elliptic execution requires explicit sequence axes")?;
                Map::Gated(warp.differentiate_gated_causal_batch(
                    &orientations,
                    shape,
                    tensor(placement, "raw_mix")[0],
                    MAX_ROWS,
                    MAX_PAIRS,
                )?)
            }
            GeometryAdapterRecipe::Elliptic { .. } => {
                Map::Pointwise(warp.differentiate_batch(&orientations, MAX_ROWS)?)
            }
            _ => return Err("invalid native elliptic recipe".into()),
        };
        let values = match &map {
            Map::Pointwise(batch) => batch.features(),
            Map::Anchored(batch) => batch.features(),
            Map::Causal(batch) => batch.features(),
            Map::Gated(batch) => batch.features(),
        };
        let features = Tensor::from_vec(rows, 9, values.to_vec())?;
        let output = finite(features.matmul_prepacked_with_backend(
            &PackedB::from_tensor_transpose(&readout_weight, Tile::col_major())?,
            MATMUL,
        )?)?;
        Ok(Self {
            input: input.clone(),
            orientation_weight,
            readout_weight,
            features,
            map,
            output,
        })
    }

    pub(super) fn output(&self) -> &Tensor {
        &self.output
    }

    pub(super) fn vjp(
        &self,
        upstream: &Tensor,
        need_input: bool,
    ) -> Result<NativeGeometryPullback> {
        let grad_readout = finite(upstream.matmul_lhs_transpose_scaled_with_backend(
            &self.features,
            1.0,
            MATMUL,
        )?)?;
        let grad_features = finite(upstream.matmul_with_backend(&self.readout_weight, MATMUL)?)?;
        let (orientations, raw_mix) = match &self.map {
            Map::Pointwise(batch) => (batch.vjp(grad_features.data())?, None),
            Map::Anchored(batch) => {
                let gradient = batch.vjp(grad_features.data())?;
                (gradient.orientations, Some(gradient.raw_mix))
            }
            Map::Causal(batch) => (batch.vjp(grad_features.data())?, None),
            Map::Gated(batch) => {
                let gradient = batch.vjp(grad_features.data())?;
                (gradient.orientations, Some(gradient.raw_mix))
            }
        };
        // Coordinate zero is the fixed chart anchor, not a trainable parameter.
        let grad_coordinates = Tensor::from_vec(
            self.input.shape().0,
            2,
            orientations
                .chunks_exact(3)
                .flat_map(|row| row[1..].iter().copied())
                .collect(),
        )?;
        let grad_orientation = finite(grad_coordinates.matmul_lhs_transpose_scaled_with_backend(
            &self.input,
            1.0,
            MATMUL,
        )?)?;
        let grad_bias =
            grad_coordinates.try_sum_axis0_scaled_with_backend(1.0, TensorUtilBackend::Cpu)?;
        let input = if need_input {
            Some(finite(
                grad_coordinates.matmul_with_backend(&self.orientation_weight, MATMUL)?,
            )?)
        } else {
            None
        };
        let mut parameters = BTreeMap::from([
            (
                "orientation.weight".into(),
                grad_orientation.data().to_vec(),
            ),
            ("orientation.bias".into(), grad_bias),
            ("readout.weight".into(), grad_readout.data().to_vec()),
        ]);
        if let Some(raw_mix) = raw_mix {
            parameters.insert("raw_mix".into(), vec![raw_mix]);
        }
        Ok(NativeGeometryPullback { input, parameters })
    }
}
