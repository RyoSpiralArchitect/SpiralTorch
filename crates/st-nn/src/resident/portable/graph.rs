//! Version 2 preserves every role, parameter ID and pointwise stage.
use super::*;
use crate::resident::{GraphDefinition, GraphParameter, GraphStage, ParameterRole};
use st_kernel_contracts::{
    elementwise::ElementwiseOp,
    graph::GraphError,
    pointwise::{PointwiseChain, PointwiseStep},
};

pub const GRAPH_PLAN_SCHEMA: &str = "spiraltorch.nn.inference_plan.v2";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema: String,
    input_shape: Vec<u32>,
    parameters: Vec<Parameter>,
    stages: Vec<Stage>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameter {
    role: Role,
    shape: Vec<u32>,
    values: Vec<f32>,
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    Weight,
    Bias,
    Gain,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Stage {
    Linear {
        weight: u32,
        bias: u32,
        gelu: bool,
    },
    Pointwise {
        parameters: Vec<u32>,
        steps: Vec<Step>,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Step {
    op: Op,
    rhs: Option<u32>,
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Op {
    Identity,
    Add,
    Multiply,
    Relu,
    Gelu,
}

pub(super) fn to_json(graph: &GraphDefinition) -> Result<String, InferenceError> {
    let record = Record {
        schema: GRAPH_PLAN_SCHEMA.to_owned(),
        input_shape: graph
            .input_layout()
            .shape()
            .iter()
            .copied()
            .map(portable_dim)
            .collect::<Result<_, _>>()?,
        parameters: graph
            .parameters()
            .iter()
            .map(|p| {
                Ok(Parameter {
                    role: match p.role {
                        ParameterRole::Weight => Role::Weight,
                        ParameterRole::Bias => Role::Bias,
                        ParameterRole::Gain => Role::Gain,
                    },
                    shape: p
                        .shape
                        .iter()
                        .copied()
                        .map(portable_dim)
                        .collect::<Result<_, _>>()?,
                    values: p.values.clone(),
                })
            })
            .collect::<Result<_, InferenceError>>()?,
        stages: graph
            .stages()
            .iter()
            .map(|s| {
                Ok(match s {
                    GraphStage::Linear { weight, bias, gelu } => Stage::Linear {
                        weight: portable_dim(*weight)?,
                        bias: portable_dim(*bias)?,
                        gelu: *gelu,
                    },
                    GraphStage::Pointwise { chain, parameters } => Stage::Pointwise {
                        parameters: parameters
                            .iter()
                            .copied()
                            .map(portable_dim)
                            .collect::<Result<_, _>>()?,
                        steps: chain
                            .steps()
                            .iter()
                            .map(|s| {
                                Ok(Step {
                                    op: match s.op {
                                        ElementwiseOp::Identity => Op::Identity,
                                        ElementwiseOp::Add => Op::Add,
                                        ElementwiseOp::Multiply => Op::Multiply,
                                        ElementwiseOp::Relu => Op::Relu,
                                        ElementwiseOp::Gelu => Op::Gelu,
                                    },
                                    rhs: s.rhs.map(portable_dim).transpose()?,
                                })
                            })
                            .collect::<Result<_, InferenceError>>()?,
                    },
                })
            })
            .collect::<Result<_, InferenceError>>()?,
    };
    Ok(serde_json::to_string(&record)?)
}

pub(super) fn from_json(payload: &str) -> Result<InferencePlan, InferenceError> {
    let record: Record = serde_json::from_str(payload)?;
    if record.schema != GRAPH_PLAN_SCHEMA {
        return Err(InferenceError::Schema(record.schema));
    }
    let shape = record
        .input_shape
        .iter()
        .map(|&v| v as usize)
        .collect::<Vec<_>>();
    let parameters = record
        .parameters
        .into_iter()
        .map(|p| GraphParameter {
            role: match p.role {
                Role::Weight => ParameterRole::Weight,
                Role::Bias => ParameterRole::Bias,
                Role::Gain => ParameterRole::Gain,
            },
            shape: p.shape.into_iter().map(|v| v as usize).collect(),
            values: p.values,
        })
        .collect();
    let stages = record
        .stages
        .into_iter()
        .map(|s| {
            Ok(match s {
                Stage::Linear { weight, bias, gelu } => GraphStage::Linear {
                    weight: weight as usize,
                    bias: bias as usize,
                    gelu,
                },
                Stage::Pointwise { parameters, steps } => GraphStage::Pointwise {
                    chain: PointwiseChain::new(
                        parameters.len() + 1,
                        steps
                            .into_iter()
                            .map(|s| PointwiseStep {
                                op: match s.op {
                                    Op::Identity => ElementwiseOp::Identity,
                                    Op::Add => ElementwiseOp::Add,
                                    Op::Multiply => ElementwiseOp::Multiply,
                                    Op::Relu => ElementwiseOp::Relu,
                                    Op::Gelu => ElementwiseOp::Gelu,
                                },
                                rhs: s.rhs.map(|v| v as usize),
                            })
                            .collect(),
                    )
                    .map_err(GraphError::from)?,
                    parameters: parameters.into_iter().map(|id| id as usize).collect(),
                },
            })
        })
        .collect::<Result<_, InferenceError>>()?;
    InferencePlan::from_graph_definition(GraphDefinition::new(
        NdLayout::contiguous(&shape)?,
        stages,
        parameters,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        layers::{Relu, Scaler},
        Linear, Sequential,
    };
    use serde_json::json;
    #[test]
    fn v2_roundtrip_and_invalid_graphs_fail_before_allocation() {
        let mut model = Sequential::new();
        model.push(Scaler::new("s", 2).unwrap());
        model.push(Linear::new("l", 2, 3).unwrap());
        model.push(Relu::new());
        let plan =
            InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 3, 2]).unwrap()).unwrap();
        let payload = plan.to_json().unwrap();
        assert!(payload.contains(GRAPH_PLAN_SCHEMA));
        assert_eq!(
            InferencePlan::from_json(&payload)
                .unwrap()
                .to_json()
                .unwrap(),
            payload
        );
        assert!(InferencePlan::from_json_with_limit(&payload, payload.len() - 1).is_err());
        let valid: serde_json::Value = serde_json::from_str(&payload).unwrap();
        for (pointer, value) in [
            ("/parameters/0/role", json!("weight")),
            ("/parameters/0/values", json!([1.])),
            ("/stages/0/parameters", json!([1])),
            ("/stages/0/steps/0/rhs", json!(0)),
            ("/stages/1/weight", json!(0)),
            ("/input_shape", json!([u32::MAX, 2])),
            ("/stages", json!([])),
        ] {
            let mut bad = valid.clone();
            *bad.pointer_mut(pointer).unwrap() = value;
            // rhs=0 is valid aliasing of the activation but leaves the gain unused;
            // the common chain contract must reject unused operand slots.
            assert!(
                InferencePlan::from_json(&bad.to_string()).is_err(),
                "{pointer}"
            );
        }
        let mut bad = valid.clone();
        bad["stages"][0]["unknown"] = json!(true);
        assert!(InferencePlan::from_json(&bad.to_string()).is_err());
    }
}
