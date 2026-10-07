//! v2 pointwise graphs, v3 LayerNorm, v4 subtract/divide, v5 shared Topos gates.
use super::*;
use crate::resident::{GraphDefinition, GraphParameter, GraphStage, ParameterRole};
use st_kernel_contracts::{
    elementwise::ElementwiseOp,
    graph::GraphError,
    pointwise::{PointwiseChain, PointwiseStep},
};

pub const GRAPH_PLAN_SCHEMA: &str = "spiraltorch.nn.inference_plan.v2";
pub const GRAPH_PLAN_SCHEMA_V3: &str = "spiraltorch.nn.inference_plan.v3";
pub const GRAPH_PLAN_SCHEMA_V4: &str = "spiraltorch.nn.inference_plan.v4";
pub const GRAPH_PLAN_SCHEMA_V5: &str = "spiraltorch.nn.inference_plan.v5";
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
    Gate,
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
    LayerNorm {
        gain: u32,
        bias: u32,
        epsilon: f32,
    },
    ToposResonator {
        gate: u32,
        coupling: f32,
        iterations: u32,
        saturation: f32,
        porosity: f32,
        max_volume: u32,
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
    Subtract,
    Divide,
    Relu,
    Gelu,
}

pub(super) fn to_json(graph: &GraphDefinition) -> Result<String, InferenceError> {
    let record = Record {
        schema: if graph
            .stages()
            .iter()
            .any(|stage| matches!(stage, GraphStage::ToposResonator { .. }))
        {
            GRAPH_PLAN_SCHEMA_V5
        } else if graph.stages().iter().any(|stage| {
            matches!(stage,
            GraphStage::Pointwise { chain, .. } if chain.steps().iter().any(|step|
                matches!(step.op, ElementwiseOp::Subtract | ElementwiseOp::Divide)))
        }) {
            GRAPH_PLAN_SCHEMA_V4
        } else if graph
            .stages()
            .iter()
            .any(|stage| matches!(stage, GraphStage::LayerNorm { .. }))
        {
            GRAPH_PLAN_SCHEMA_V3
        } else {
            GRAPH_PLAN_SCHEMA
        }
        .to_owned(),
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
                        ParameterRole::Gate => Role::Gate,
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
                                        ElementwiseOp::Subtract => Op::Subtract,
                                        ElementwiseOp::Divide => Op::Divide,
                                        ElementwiseOp::Relu => Op::Relu,
                                        ElementwiseOp::Gelu => Op::Gelu,
                                    },
                                    rhs: s.rhs.map(portable_dim).transpose()?,
                                })
                            })
                            .collect::<Result<_, InferenceError>>()?,
                    },
                    GraphStage::ToposResonator {
                        gate,
                        kernel,
                        max_volume,
                    } => Stage::ToposResonator {
                        gate: portable_dim(*gate)?,
                        coupling: kernel.coupling(),
                        iterations: portable_dim(kernel.iterations())?,
                        saturation: kernel.saturation(),
                        porosity: kernel.porosity(),
                        max_volume: portable_dim(*max_volume)?,
                    },
                    GraphStage::LayerNorm {
                        gain,
                        bias,
                        epsilon,
                    } => Stage::LayerNorm {
                        gain: portable_dim(*gain)?,
                        bias: portable_dim(*bias)?,
                        epsilon: *epsilon,
                    },
                })
            })
            .collect::<Result<_, InferenceError>>()?,
    };
    Ok(serde_json::to_string(&record)?)
}

pub(super) fn from_json(payload: &str) -> Result<InferencePlan, InferenceError> {
    let record: Record = serde_json::from_str(payload)?;
    if ![
        GRAPH_PLAN_SCHEMA,
        GRAPH_PLAN_SCHEMA_V3,
        GRAPH_PLAN_SCHEMA_V4,
        GRAPH_PLAN_SCHEMA_V5,
    ]
    .contains(&record.schema.as_str())
    {
        return Err(InferenceError::Schema(record.schema));
    }
    if record.schema != GRAPH_PLAN_SCHEMA_V5
        && (record
            .stages
            .iter()
            .any(|stage| matches!(stage, Stage::ToposResonator { .. }))
            || record
                .parameters
                .iter()
                .any(|p| matches!(p.role, Role::Gate)))
    {
        return Err(InferenceError::Schema(format!(
            "{} does not admit Topos gates",
            record.schema
        )));
    }
    if ![GRAPH_PLAN_SCHEMA_V4, GRAPH_PLAN_SCHEMA_V5].contains(&record.schema.as_str()) && record.stages.iter().any(|stage| matches!(stage,
        Stage::Pointwise { steps, .. } if steps.iter().any(|step| matches!(step.op, Op::Subtract | Op::Divide))))
    {
        return Err(InferenceError::Schema(format!("{} does not admit subtract/divide", record.schema)));
    }
    if record.schema == GRAPH_PLAN_SCHEMA
        && record
            .stages
            .iter()
            .any(|stage| matches!(stage, Stage::LayerNorm { .. }))
    {
        return Err(InferenceError::Schema(format!(
            "{} does not admit layer_norm",
            record.schema
        )));
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
                Role::Gate => ParameterRole::Gate,
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
                                    Op::Subtract => ElementwiseOp::Subtract,
                                    Op::Divide => ElementwiseOp::Divide,
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
                Stage::ToposResonator {
                    gate,
                    coupling,
                    iterations,
                    saturation,
                    porosity,
                    max_volume,
                } => GraphStage::ToposResonator {
                    gate: gate as usize,
                    kernel: crate::resident::ToposResonatorKernel::new(
                        coupling,
                        saturation,
                        porosity,
                        iterations as usize,
                    )
                    .map_err(GraphError::from)?,
                    max_volume: max_volume as usize,
                },
                Stage::LayerNorm {
                    gain,
                    bias,
                    epsilon,
                } => GraphStage::LayerNorm {
                    gain: gain as usize,
                    bias: bias as usize,
                    epsilon,
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
    fn v4_normalization_roundtrip_rejects_schema_downgrade() {
        let mut model = Sequential::new();
        model.push(Scaler::new("stats", 2).unwrap());
        let plan =
            InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 2]).unwrap()).unwrap();
        let mut record: serde_json::Value = serde_json::from_str(&plan.to_json().unwrap()).unwrap();
        record["schema"] = json!(GRAPH_PLAN_SCHEMA_V4);
        record["stages"][0]["steps"] = json!([
            {"op": "subtract", "rhs": 1}, {"op": "divide", "rhs": 1}
        ]);
        let payload = InferencePlan::from_json(&record.to_string())
            .unwrap()
            .to_json()
            .unwrap();
        assert!(payload.contains(GRAPH_PLAN_SCHEMA_V4));
        assert_eq!(
            InferencePlan::from_json(&payload)
                .unwrap()
                .to_json()
                .unwrap(),
            payload
        );
        for old in [GRAPH_PLAN_SCHEMA, GRAPH_PLAN_SCHEMA_V3] {
            assert!(InferencePlan::from_json(&payload.replace(GRAPH_PLAN_SCHEMA_V4, old)).is_err());
        }
    }

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
