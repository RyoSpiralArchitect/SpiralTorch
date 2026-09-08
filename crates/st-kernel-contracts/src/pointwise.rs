//! Shape-preserving pointwise chains. Every step evaluates the same logical
//! domain, so fusion cannot erase a nonempty intermediate by broadcasting to
//! an empty output. Shape-changing chains must use ordinary tensor operations.

use crate::{
    elementwise::ElementwiseOp,
    layout::{NdLayout, NdLayoutError},
};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointwiseStep {
    pub op: ElementwiseOp,
    /// Original input slot, not a previous step. Slot zero enables residuals.
    pub rhs: Option<usize>,
}

/// Execution policy, never permission to transfer devices or fall back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointwiseExecution {
    Sequential,
    Batched,
    Fused,
}

#[derive(Debug, Error)]
pub enum PointwiseError {
    #[error("pointwise forward, cotangent, derivative or gradient is non-finite")]
    NonFinite,
    #[error("pointwise chains require 1..=16 inputs and 1..=256 steps")]
    Budget,
    #[error("pointwise operand arity, index, or unused input slot")]
    Operands,
    #[error("pointwise input layouts differ from the prepared plan")]
    LayoutMismatch,
    #[error(transparent)]
    Layout(#[from] NdLayoutError),
}

#[derive(Clone, Debug)]
pub struct PointwiseChain {
    input_count: usize,
    steps: Vec<PointwiseStep>,
}

impl PointwiseChain {
    pub fn new(input_count: usize, steps: Vec<PointwiseStep>) -> Result<Self, PointwiseError> {
        if !(1..=16).contains(&input_count) || !(1..=256).contains(&steps.len()) {
            return Err(PointwiseError::Budget);
        }
        let mut used = vec![false; input_count];
        used[0] = true;
        for step in &steps {
            if step.op.is_binary() != step.rhs.is_some() {
                return Err(PointwiseError::Operands);
            }
            if let Some(rhs) = step.rhs {
                if rhs >= input_count {
                    return Err(PointwiseError::Operands);
                }
                used[rhs] = true;
            }
        }
        if used.contains(&false) {
            return Err(PointwiseError::Operands);
        }
        Ok(Self { input_count, steps })
    }
    pub fn input_count(&self) -> usize {
        self.input_count
    }
    pub fn steps(&self) -> &[PointwiseStep] {
        &self.steps
    }
    pub fn validate_layouts(&self, layouts: &[NdLayout]) -> Result<(), PointwiseError> {
        if layouts.len() != self.input_count {
            return Err(PointwiseError::Operands);
        }
        for layout in layouts {
            layout.broadcast_to(layouts[0].shape())?;
        }
        Ok(())
    }

    /// One logical element's exact reverse chain, before broadcast reduction.
    /// Repeated use of an original input slot accumulates all its contributions.
    pub fn vjp_scalar(&self, inputs: &[f32], cotangent: f32) -> Result<Vec<f32>, PointwiseError> {
        if inputs.len() != self.input_count {
            return Err(PointwiseError::Operands);
        }
        if !cotangent.is_finite() {
            return Err(PointwiseError::NonFinite);
        }
        let checked = |x: f32| x.is_finite().then_some(x).ok_or(PointwiseError::NonFinite);
        let mut tape = Vec::with_capacity(self.steps.len());
        let mut current = inputs[0];
        for step in &self.steps {
            tape.push(current);
            current = step
                .op
                .apply(current, step.rhs.map_or(0., |i| inputs[i]))
                .ok_or(PointwiseError::NonFinite)?;
        }
        let mut gradients = vec![0.; self.input_count];
        let mut delta = cotangent;
        for (step, &before) in self.steps.iter().zip(&tape).rev() {
            let (lhs, rhs) = step
                .op
                .partials(before, step.rhs.map_or(0., |i| inputs[i]))
                .ok_or(PointwiseError::NonFinite)?;
            if let Some(slot) = step.rhs {
                let contribution = checked(delta * rhs)?;
                gradients[slot] = checked(gradients[slot] + contribution)?;
            }
            delta = checked(delta * lhs)?;
        }
        gradients[0] = checked(gradients[0] + delta)?;
        Ok(gradients)
    }
}

/// Deterministic inverse of broadcasting. Indices refer to logical values, not
/// storage addresses; view/alias adjoints are a separate operation.
#[derive(Clone, Debug)]
pub struct BroadcastAdjoint {
    output: NdLayout,
    input: NdLayout,
    reduction: NdLayout,
}

impl BroadcastAdjoint {
    pub fn new(output_shape: &[usize], input_shape: &[usize]) -> Result<Self, PointwiseError> {
        let output = NdLayout::contiguous(output_shape)?;
        let input = NdLayout::contiguous(input_shape)?;
        input.broadcast_to(output_shape)?;
        let leading = output.rank() - input.rank();
        let reduction_shape: Vec<_> = output_shape
            .iter()
            .enumerate()
            .map(|(axis, &size)| {
                let input_dim = axis.checked_sub(leading).map_or(1, |i| input_shape[i]);
                if input_dim == 1 {
                    size
                } else {
                    1
                }
            })
            .collect();
        Ok(Self {
            output,
            input,
            reduction: NdLayout::contiguous(&reduction_shape)?,
        })
    }
    pub fn output_layout(&self) -> &NdLayout {
        &self.output
    }
    pub fn input_layout(&self) -> &NdLayout {
        &self.input
    }
    pub fn reduction_shape(&self) -> &[usize] {
        self.reduction.shape()
    }
    pub fn reduction_len(&self) -> usize {
        self.reduction.len()
    }
    pub fn source_index(&self, mut element: usize, mut reduction: usize) -> Option<usize> {
        if element >= self.input.len() || reduction >= self.reduction.len() {
            return None;
        }
        let leading = self.output.rank() - self.input.rank();
        let mut index = 0;
        for axis in (0..self.output.rank()).rev() {
            let size = axis
                .checked_sub(leading)
                .map_or(1, |i| self.input.shape()[i]);
            let input_coordinate = element % size;
            element /= size;
            let reduced = reduction % self.reduction.shape()[axis];
            reduction /= self.reduction.shape()[axis];
            index += (input_coordinate + reduced) * self.output.strides()[axis];
        }
        Some(index)
    }

    /// Same fixed 256-lane, two-stage addition tree as the GPU reduction.
    /// No-reduction inputs copy checked contributions without a sum.
    pub fn reduce(&self, contributions: &[f32]) -> Result<Vec<f32>, PointwiseError> {
        if contributions.len() != self.output.len() {
            return Err(PointwiseError::Operands);
        }
        let checked = |x: f32| x.is_finite().then_some(x).ok_or(PointwiseError::NonFinite);
        if self.reduction_len() == 1 {
            return contributions.iter().copied().map(checked).collect();
        }
        let tree = |mut lanes: [f32; 256]| -> Result<f32, PointwiseError> {
            for stride in [128, 64, 32, 16, 8, 4, 2, 1] {
                for lane in 0..stride {
                    lanes[lane] = checked(lanes[lane] + lanes[lane + stride])?;
                }
            }
            Ok(lanes[0])
        };
        let partials = self.reduction_len().div_ceil(256).max(1);
        let mut result = Vec::with_capacity(self.input.len());
        for element in 0..self.input.len() {
            let mut chunks = Vec::with_capacity(partials);
            for chunk in 0..partials {
                let mut lanes = [0.; 256];
                for (lane, value) in lanes.iter_mut().enumerate() {
                    if let Some(index) = self.source_index(element, chunk * 256 + lane) {
                        *value = checked(contributions[index])?;
                    }
                }
                chunks.push(tree(lanes)?);
            }
            if partials == 1 {
                result.push(chunks[0]);
            } else {
                let mut lanes = [0.; 256];
                for (i, &value) in chunks.iter().enumerate() {
                    lanes[i % 256] = checked(lanes[i % 256] + value)?;
                }
                result.push(tree(lanes)?);
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_operands_and_fixed_domain_are_explicit() {
        let add = PointwiseStep {
            op: ElementwiseOp::Add,
            rhs: Some(1),
        };
        let chain = PointwiseChain::new(2, vec![add]).unwrap();
        let scalar = NdLayout::contiguous(&[]).unwrap();
        let vector = NdLayout::contiguous(&[3]).unwrap();
        let empty = NdLayout::contiguous(&[0]).unwrap();
        assert!(chain
            .validate_layouts(&[vector.clone(), scalar.clone()])
            .is_ok());
        assert!(chain.validate_layouts(&[scalar.clone(), vector]).is_err());
        assert!(chain
            .validate_layouts(&[scalar.clone(), empty.clone()])
            .is_err());
        assert!(chain.validate_layouts(&[empty, scalar]).is_ok());
        assert!(PointwiseChain::new(3, vec![add]).is_err());
        assert!(PointwiseChain::new(1, vec![add]).is_err());
        assert!(PointwiseChain::new(2, vec![]).is_err());
        assert!(PointwiseChain::new(2, vec![add; 257]).is_err());
        assert!(PointwiseChain::new(
            2,
            vec![PointwiseStep {
                op: ElementwiseOp::Relu,
                rhs: Some(1)
            }]
        )
        .is_err());
    }

    #[test]
    fn reverse_chain_accumulates_original_input_residuals_and_checks_zero_seeds() {
        let chain = PointwiseChain::new(
            2,
            vec![
                PointwiseStep {
                    op: ElementwiseOp::Multiply,
                    rhs: Some(1),
                },
                PointwiseStep {
                    op: ElementwiseOp::Add,
                    rhs: Some(0),
                },
                PointwiseStep {
                    op: ElementwiseOp::Relu,
                    rhs: None,
                },
            ],
        )
        .unwrap();
        assert_eq!(chain.vjp_scalar(&[2., 3.], 0.5).unwrap(), vec![2., 1.]);
        assert_eq!(chain.vjp_scalar(&[-2., 3.], 0.5).unwrap(), vec![0., 0.]);
        assert!(chain.vjp_scalar(&[f32::MAX, 2.], 0.).is_err());
        assert_eq!(chain.vjp_scalar(&[0., 2.], f32::MAX).unwrap(), vec![0., 0.]);
        assert!(chain.vjp_scalar(&[1., 2.], f32::MAX).is_err());
    }

    #[test]
    fn broadcast_adjoint_partitions_each_logical_contribution_once() {
        for (shape, input) in [
            (vec![2, 3, 4], vec![4]),
            (vec![2, 3, 4], vec![1, 3, 1]),
            (vec![2, 3, 4], vec![]),
            (vec![2, 3, 4], vec![2, 3, 4]),
            (vec![0, 3], vec![1, 3]),
            (vec![], vec![]),
            (vec![513, 2], vec![2]),
        ] {
            let map = BroadcastAdjoint::new(&shape, &input).unwrap();
            let mut visited = vec![0; map.output.len()];
            for element in 0..map.input.len() {
                for reduced in 0..map.reduction_len() {
                    visited[map.source_index(element, reduced).unwrap()] += 1;
                }
            }
            assert!(visited.iter().all(|&n| n == 1));
            let values = vec![1.; map.output.len()];
            assert_eq!(
                map.reduce(&values).unwrap(),
                vec![map.reduction_len() as f32; map.input.len()]
            );
        }
        let map = BroadcastAdjoint::new(&[2], &[]).unwrap();
        assert!(map.reduce(&[f32::MAX, f32::MAX]).is_err());
        assert!(BroadcastAdjoint::new(&[2, 3], &[2]).is_err());
    }
}
