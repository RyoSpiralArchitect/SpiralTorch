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
}
