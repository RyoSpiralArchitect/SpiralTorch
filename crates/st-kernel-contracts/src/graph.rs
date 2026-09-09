//! Validated sequential graph and parameter ownership shared by all clients.
use crate::{
    layout::{NdLayout, NdLayoutError},
    pointwise::{PointwiseChain, PointwiseError},
};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterRole {
    Weight,
    Bias,
    Gain,
}

impl ParameterRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Weight => "weight",
            Self::Bias => "bias",
            Self::Gain => "gain",
        }
    }
}

/// Exact is a mathematical VJP. ModuleCompatible retains Scaler's legacy
/// extra row average without applying it to Linear weight/bias gradients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphGradientPolicy {
    Exact,
    ModuleCompatible,
}

impl GraphGradientPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ModuleCompatible => "module_compatible",
        }
    }
}

impl std::str::FromStr for GraphGradientPolicy {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "exact" => Ok(Self::Exact),
            "module_compatible" => Ok(Self::ModuleCompatible),
            _ => Err("gradient_policy must be 'exact' or 'module_compatible'"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GraphParameter {
    pub role: ParameterRole,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub enum GraphStage {
    Linear {
        weight: usize,
        bias: usize,
        gelu: bool,
    },
    Pointwise {
        chain: PointwiseChain,
        parameters: Vec<usize>,
    },
}

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("graph needs 1..=4096 stages and at most 8192 parameter slots")]
    Budget,
    #[error("graph input must be nonempty, contiguous, offset zero and rank >= 1")]
    Input,
    #[error("invalid graph parameter slot {0}")]
    Parameter(usize),
    #[error("invalid graph stage {0}")]
    Stage(usize),
    #[error("graph parameters must have exactly one owner; tying is not supported yet")]
    Ownership,
    #[error("graph shapes must fit the shared u32 address space")]
    AddressSpace,
    #[error(transparent)]
    Layout(#[from] NdLayoutError),
    #[error(transparent)]
    Pointwise(#[from] PointwiseError),
}

/// Construction validates all stages/values before a backend allocates resources.
#[derive(Clone, Debug)]
pub struct GraphDefinition {
    layouts: Vec<NdLayout>,
    stages: Vec<GraphStage>,
    parameters: Vec<GraphParameter>,
    owners: Vec<usize>,
}

fn portable(layout: &NdLayout) -> Result<(), GraphError> {
    for &n in layout
        .shape()
        .iter()
        .chain(layout.strides())
        .chain([&layout.len()])
    {
        u32::try_from(n).map_err(|_| GraphError::AddressSpace)?;
    }
    Ok(())
}

impl GraphDefinition {
    pub fn new(
        input: NdLayout,
        stages: Vec<GraphStage>,
        parameters: Vec<GraphParameter>,
    ) -> Result<Self, GraphError> {
        if input.is_empty() || input.rank() == 0 || !input.is_contiguous() || input.offset() != 0 {
            return Err(GraphError::Input);
        }
        if stages.is_empty() || stages.len() > 4096 || parameters.len() > 8192 {
            return Err(GraphError::Budget);
        }
        portable(&input)?;
        let mut parameter_layouts = Vec::new();
        for (id, p) in parameters.iter().enumerate() {
            let layout = NdLayout::contiguous(&p.shape)?;
            let rank = if p.role == ParameterRole::Weight {
                2
            } else {
                1
            };
            if layout.rank() != rank
                || layout.is_empty()
                || layout.len() != p.values.len()
                || p.values.iter().any(|v| !v.is_finite())
            {
                return Err(GraphError::Parameter(id));
            }
            portable(&layout)?;
            parameter_layouts.push(layout);
        }
        let mut owners = vec![usize::MAX; parameters.len()];
        let mut layouts = vec![input];
        for (index, stage) in stages.iter().enumerate() {
            let input = layouts.last().unwrap();
            let mut claim = |id: usize| -> Result<(), GraphError> {
                let owner = owners.get_mut(id).ok_or(GraphError::Parameter(id))?;
                if *owner != usize::MAX {
                    return Err(GraphError::Ownership);
                }
                *owner = index;
                Ok(())
            };
            let output = match stage {
                GraphStage::Linear { weight, bias, .. } => {
                    claim(*weight)?;
                    claim(*bias)?;
                    let w = &parameters[*weight];
                    let b = &parameters[*bias];
                    if w.role != ParameterRole::Weight
                        || b.role != ParameterRole::Bias
                        || w.shape[0] != *input.shape().last().unwrap()
                        || b.shape != [w.shape[1]]
                    {
                        return Err(GraphError::Stage(index));
                    }
                    let mut shape = input.shape().to_vec();
                    *shape.last_mut().unwrap() = w.shape[1];
                    NdLayout::contiguous(&shape)?
                }
                GraphStage::Pointwise {
                    chain,
                    parameters: ids,
                } => {
                    let mut operands = vec![input.clone()];
                    for &id in ids {
                        claim(id)?;
                        operands.push(parameter_layouts[id].clone());
                    }
                    chain.validate_layouts(&operands)?;
                    input.clone()
                }
            };
            portable(&output)?;
            layouts.push(output);
        }
        if owners.contains(&usize::MAX) {
            return Err(GraphError::Ownership);
        }
        Ok(Self {
            layouts,
            stages,
            parameters,
            owners,
        })
    }
    pub fn input_layout(&self) -> &NdLayout {
        &self.layouts[0]
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.layouts.last().unwrap()
    }
    pub fn layouts(&self) -> &[NdLayout] {
        &self.layouts
    }
    pub fn stages(&self) -> &[GraphStage] {
        &self.stages
    }
    pub fn parameters(&self) -> &[GraphParameter] {
        &self.parameters
    }
    pub fn parameter_owners(&self) -> &[usize] {
        &self.owners
    }
    pub fn with_values(&self, values: Vec<Vec<f32>>) -> Result<Self, GraphError> {
        if values.len() != self.parameters.len() {
            return Err(GraphError::Ownership);
        }
        let parameters = self
            .parameters
            .iter()
            .zip(values)
            .map(|(p, values)| GraphParameter {
                role: p.role,
                shape: p.shape.clone(),
                values,
            })
            .collect();
        Self::new(self.input_layout().clone(), self.stages.clone(), parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{elementwise::ElementwiseOp, pointwise::PointwiseStep};
    #[test]
    fn client_names_are_canonical_and_policy_is_explicit() {
        for policy in [
            GraphGradientPolicy::Exact,
            GraphGradientPolicy::ModuleCompatible,
        ] {
            assert_eq!(policy.as_str().parse(), Ok(policy));
        }
        for invalid in ["", "auto", "EXACT", " exact", "exact ", "module-compatible"] {
            assert!(invalid.parse::<GraphGradientPolicy>().is_err());
        }
        assert_eq!(ParameterRole::Weight.as_str(), "weight");
        assert_eq!(ParameterRole::Bias.as_str(), "bias");
        assert_eq!(ParameterRole::Gain.as_str(), "gain");
    }
    fn scaler() -> GraphStage {
        GraphStage::Pointwise {
            chain: PointwiseChain::new(
                2,
                vec![PointwiseStep {
                    op: ElementwiseOp::Multiply,
                    rhs: Some(1),
                }],
            )
            .unwrap(),
            parameters: vec![0],
        }
    }
    #[test]
    fn graph_owns_all_parameters_and_validates_before_allocation() {
        let shape = NdLayout::contiguous(&[2, 3, 4]).unwrap();
        let gain = GraphParameter {
            role: ParameterRole::Gain,
            shape: vec![4],
            values: vec![1.; 4],
        };
        let graph =
            GraphDefinition::new(shape.clone(), vec![scaler()], vec![gain.clone()]).unwrap();
        assert_eq!(graph.output_layout().shape(), &[2, 3, 4]);
        assert_eq!(graph.parameter_owners(), &[0]);
        assert!(
            GraphDefinition::new(shape.clone(), vec![scaler(), scaler()], vec![gain.clone()])
                .is_err()
        );
        assert!(
            GraphDefinition::new(shape.clone(), vec![scaler()], vec![gain.clone(), gain]).is_err()
        );
        assert!(graph.with_values(vec![vec![f32::NAN; 4]]).is_err());
        assert!(graph.with_values(vec![vec![1.; 3]]).is_err());
        assert_eq!(
            graph.with_values(vec![vec![2.; 4]]).unwrap().parameters()[0].values,
            vec![2.; 4]
        );
    }
}
