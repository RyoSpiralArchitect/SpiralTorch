//! Native gates and parameter charts around the existing bounded GL operator.

use super::{
    tensor, GeometryAdapterPlacement, GeometryAdapterRecipe, NativeGeometryPullback, Result,
};
use crate::geometry_artifact::FractionalRecipe;
use st_frac::learning::{
    FractionalGlAngleChart, FractionalGlGainLearningBatch, FractionalGlKernel,
    FractionalGlLearningBatch,
};
use st_tensor::{Tensor, TensorUtilBackend};
use std::collections::BTreeMap;

const CPU: TensorUtilBackend = TensorUtilBackend::Cpu;

pub(super) fn sequence_values(
    features: usize,
    shape: [usize; 2],
    recipe: &FractionalRecipe,
) -> Result<usize> {
    let count = shape[0]
        .checked_mul(shape[1])
        .and_then(|n| n.checked_mul(features))
        .filter(|&n| n > 0 && n <= recipe.max_values)
        .ok_or("native fractional sequence exceeds nonempty value budget")?;
    count
        .checked_mul(recipe.kernel_len)
        .filter(|&n| n <= recipe.max_products)
        .ok_or("native fractional sequence exceeds convolution product budget")?;
    Ok(count)
}

#[derive(Debug)]
enum Map {
    Plain(FractionalGlLearningBatch),
    Gain(FractionalGlGainLearningBatch),
}

#[derive(Debug)]
enum Order {
    Log(f32),
    Angle(FractionalGlAngleChart),
}

impl Order {
    fn alpha(&self) -> f32 {
        match self {
            Self::Log(alpha) => *alpha,
            Self::Angle(chart) => chart.alpha(),
        }
    }

    fn vjp(&self, gradient: f32) -> Result<(&'static str, f32)> {
        let result = match self {
            Self::Log(alpha) => ("log_alpha", gradient * alpha),
            Self::Angle(chart) => ("history_angle", chart.vjp(gradient)?),
        };
        if !result.1.is_finite() {
            return Err("native fractional order gradient is not finite".into());
        }
        Ok(result)
    }
}

#[derive(Debug)]
pub(super) struct FractionalRuntimeBatch {
    input: Tensor,
    signal: Tensor,
    gate: Vec<f32>,
    local_gate: Option<Vec<f32>>,
    order: Order,
    map: Map,
    output: Tensor,
}

fn gate_gradient(direction: &Tensor, signal: &Tensor, gate: &[f32]) -> Result<Vec<f32>> {
    let derivative: Vec<_> = gate.iter().map(|v| 1. - v * v).collect();
    Ok(direction
        .hadamard_with_backend(signal, CPU)?
        .mul_row_with_backend(&derivative, CPU)?
        .try_sum_axis0_with_backend(CPU)?)
}

impl FractionalRuntimeBatch {
    pub(super) fn forward(
        placement: &GeometryAdapterPlacement,
        recipe: &FractionalRecipe,
        input: &Tensor,
        shape: [usize; 2],
    ) -> Result<Self> {
        let (rows, features) = input.shape();
        let shape = [shape[0], shape[1], features];
        let kernel = FractionalGlKernel::new(
            recipe.kernel_len,
            recipe.step,
            recipe.max_values,
            recipe.max_products,
        )?;
        let order = if matches!(placement.recipe, GeometryAdapterRecipe::HistoryAngle { .. }) {
            Order::Angle(FractionalGlAngleChart::new(
                tensor(placement, "history_angle")[0],
            )?)
        } else {
            Order::Log(tensor(placement, "log_alpha")[0].exp())
        };
        let alpha = order.alpha();
        let map = match placement.recipe {
            GeometryAdapterRecipe::Fractional { .. } => {
                Map::Plain(kernel.forward(input.data(), &shape, 1, alpha)?)
            }
            GeometryAdapterRecipe::History { .. } => {
                Map::Plain(kernel.forward_history(input.data(), &shape, 1, alpha)?)
            }
            GeometryAdapterRecipe::HistoryL2 { gain, .. } => {
                Map::Plain(kernel.forward_history_l2(input.data(), &shape, 1, alpha, gain)?)
            }
            GeometryAdapterRecipe::HistoryGain { lag_window, .. }
            | GeometryAdapterRecipe::HistoryAngle { lag_window, .. } => {
                let gain = tensor(placement, "log_gain")[0];
                Map::Gain(if let Some([start, end]) = lag_window {
                    kernel.forward_history_log_gain_window(
                        input.data(),
                        &shape,
                        1,
                        alpha,
                        gain,
                        start..end,
                    )?
                } else {
                    kernel.forward_history_log_gain(input.data(), &shape, 1, alpha, gain)?
                })
            }
            _ => return Err("invalid native fractional recipe".into()),
        };
        let values = match &map {
            Map::Plain(batch) => batch.output(),
            Map::Gain(batch) => batch.output(),
        };
        let signal = Tensor::from_vec(rows, features, values.iter().copied().collect())?;
        let gate: Vec<_> = tensor(placement, "gate").iter().map(|v| v.tanh()).collect();
        let local_gate: Option<Vec<_>> =
            if matches!(placement.recipe, GeometryAdapterRecipe::Fractional { .. }) {
                None
            } else {
                Some(
                    tensor(placement, "local_gate")
                        .iter()
                        .map(|v| v.tanh())
                        .collect(),
                )
            };
        let history = signal.mul_row_with_backend(&gate, CPU)?;
        let output = if let Some(local) = &local_gate {
            input
                .mul_row_with_backend(local, CPU)?
                .add_with_backend(&history, CPU)?
        } else {
            history
        };
        Ok(Self {
            input: input.clone(),
            signal,
            gate,
            local_gate,
            order,
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
        let direction = upstream.mul_row_with_backend(&self.gate, CPU)?;
        let (input, alpha, log_gain) = match &self.map {
            Map::Plain(batch) if need_input => {
                let gradient = batch.vjp(direction.data())?;
                (Some(gradient.input), gradient.alpha, None)
            }
            Map::Gain(batch) if need_input => {
                let gradient = batch.vjp(direction.data())?;
                (
                    Some(gradient.input),
                    gradient.alpha,
                    Some(gradient.log_gain),
                )
            }
            Map::Plain(batch) => (None, batch.vjp_alpha(direction.data())?, None),
            Map::Gain(batch) => {
                let (alpha, gain) = batch.vjp_parameters(direction.data())?;
                (None, alpha, Some(gain))
            }
        };
        let (rows, features) = self.input.shape();
        let mut input = input
            .map(|data| Tensor::from_vec(rows, features, data))
            .transpose()?;
        let (name, order) = self.order.vjp(alpha)?;
        let mut parameters = BTreeMap::from([
            (
                "gate".into(),
                gate_gradient(upstream, &self.signal, &self.gate)?,
            ),
            (name.into(), vec![order]),
        ]);
        if let Some(gain) = log_gain {
            parameters.insert("log_gain".into(), vec![gain]);
        }
        if let Some(local) = &self.local_gate {
            if let Some(value) = &mut input {
                *value =
                    value.add_with_backend(&upstream.mul_row_with_backend(local, CPU)?, CPU)?;
            }
            parameters.insert(
                "local_gate".into(),
                gate_gradient(upstream, &self.input, local)?,
            );
        }
        Ok(NativeGeometryPullback { input, parameters })
    }
}
