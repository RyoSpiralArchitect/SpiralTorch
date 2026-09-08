//! Portable parameter snapshots; device allocation and execution are separate.

use super::{InferenceError, InferenceOp, InferencePlan};
use serde::{Deserialize, Serialize};
use st_tensor::{NdLayout, Tensor};

pub const INFERENCE_PLAN_SCHEMA: &str = "spiraltorch.nn.inference_plan.v1";
pub const DEFAULT_MAX_PLAN_JSON_BYTES: usize = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanRecord {
    schema: String,
    input_shape: Vec<u32>,
    stages: Vec<StageRecord>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StageRecord {
    inner: u32,
    cols: u32,
    weight: Vec<f32>,
    bias: Vec<f32>,
    gelu: bool,
}

fn portable_dim(value: usize) -> Result<u32, InferenceError> {
    value
        .try_into()
        .map_err(|_| InferenceError::PortableAddressSpace)
}

impl PlanRecord {
    fn validate_address_space(&self) -> Result<(), InferenceError> {
        if self.input_shape.is_empty() || self.input_shape.contains(&0) {
            return Err(InferenceError::InvalidLayout);
        }
        // Fixed-width arithmetic keeps native and wasm32 transport acceptance equal.
        let elements = self
            .input_shape
            .iter()
            .try_fold(1u32, |n, &d| n.checked_mul(d))
            .ok_or(InferenceError::PortableAddressSpace)?;
        let rows = elements / self.input_shape.last().unwrap();
        for stage in &self.stages {
            stage
                .inner
                .checked_mul(stage.cols)
                .ok_or(InferenceError::PortableAddressSpace)?;
            rows.checked_mul(stage.cols)
                .ok_or(InferenceError::PortableAddressSpace)?;
        }
        Ok(())
    }
}

impl InferencePlan {
    /// Export fixed f32 parameters, not a device image or a training checkpoint.
    pub fn to_json(&self) -> Result<String, InferenceError> {
        let record = PlanRecord {
            schema: INFERENCE_PLAN_SCHEMA.to_owned(),
            input_shape: self
                .input
                .shape()
                .iter()
                .copied()
                .map(portable_dim)
                .collect::<Result<_, _>>()?,
            stages: self
                .stages
                .iter()
                .map(|stage| {
                    Ok(StageRecord {
                        inner: portable_dim(stage.weight.shape().0)?,
                        cols: portable_dim(stage.weight.shape().1)?,
                        weight: stage.weight.data().to_vec(),
                        bias: stage.bias.data().to_vec(),
                        gelu: stage.gelu,
                    })
                })
                .collect::<Result<_, InferenceError>>()?,
        };
        record.validate_address_space()?;
        Ok(serde_json::to_string(&record)?)
    }

    pub fn from_json(payload: &str) -> Result<Self, InferenceError> {
        Self::from_json_with_limit(payload, DEFAULT_MAX_PLAN_JSON_BYTES)
    }

    /// The caller may explicitly raise the transport limit for larger models.
    /// Typed/local compilation is not restricted by this JSON transport budget.
    pub fn from_json_with_limit(payload: &str, max_bytes: usize) -> Result<Self, InferenceError> {
        if payload.len() > max_bytes {
            return Err(InferenceError::JsonLimit {
                actual: payload.len(),
                limit: max_bytes,
            });
        }
        let record: PlanRecord = serde_json::from_str(payload)?;
        if record.schema != INFERENCE_PLAN_SCHEMA {
            return Err(InferenceError::Schema(record.schema));
        }
        record.validate_address_space()?;
        let shape: Vec<_> = record.input_shape.iter().map(|&d| d as usize).collect();
        let input = NdLayout::contiguous(&shape)?;
        let mut operations = Vec::new();
        for (i, stage) in record.stages.into_iter().enumerate() {
            let inner = stage.inner as usize;
            let cols = stage.cols as usize;
            if inner.checked_mul(cols) != Some(stage.weight.len()) || stage.bias.len() != cols {
                return Err(InferenceError::Shape(i));
            }
            operations.push(InferenceOp::Linear {
                weight: Tensor::from_vec(inner, cols, stage.weight)?,
                bias: Tensor::from_vec(1, cols, stage.bias)?,
            });
            if stage.gelu {
                operations.push(InferenceOp::Gelu);
            }
        }
        Self::from_operations(input, operations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{layers::Gelu, Linear, Sequential};
    use serde_json::json;

    fn record() -> serde_json::Value {
        json!({"schema":INFERENCE_PLAN_SCHEMA,"input_shape":[2,3,2],"stages":[
            {"inner":2,"cols":2,"weight":[1.0,0.0,0.0,1.0],"bias":[0.0,0.0],"gelu":true}
        ]})
    }

    #[test]
    fn portable_roundtrip_uses_the_same_lowering() {
        let mut model = Sequential::new();
        model.push(Linear::new("up", 4, 7).unwrap());
        model.push(Gelu::new());
        model.push(Linear::new("down", 7, 3).unwrap());
        let plan =
            InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 5, 4]).unwrap()).unwrap();
        let payload = plan.to_json().unwrap();
        let restored = InferencePlan::from_json(&payload).unwrap();
        assert_eq!(restored.to_json().unwrap(), payload);
        assert_eq!(restored.input_layout(), plan.input_layout());
        assert_eq!(restored.output_layout(), plan.output_layout());
        assert_eq!(restored.source_operation_count(), 3);
        assert_eq!(restored.stage_count(), 2);
    }

    #[test]
    fn malformed_portable_plans_fail_before_device_allocation() {
        let valid = record();
        for (pointer, value) in [
            ("/schema", json!("unknown")),
            ("/input_shape", json!([])),
            ("/input_shape", json!([0, 2])),
            ("/input_shape", json!([true, 2])),
            ("/stages", json!([])),
            ("/stages/0/inner", json!(3)),
            ("/stages/0/cols", json!(usize::MAX)),
            ("/stages/0/gelu", json!("yes")),
            ("/stages/0/weight", json!([1.0])),
            ("/stages/0/bias", json!([0.0])),
        ] {
            let mut invalid = valid.clone();
            *invalid.pointer_mut(pointer).unwrap() = value;
            assert!(
                InferencePlan::from_json(&invalid.to_string()).is_err(),
                "{pointer}"
            );
        }
        let mut extra = valid.clone();
        extra["output_shape"] = json!([999]);
        assert!(InferencePlan::from_json(&extra.to_string()).is_err());
        let mut extra = valid.clone();
        extra["stages"][0]["unknown"] = json!(true);
        assert!(InferencePlan::from_json(&extra.to_string()).is_err());
        let nonfinite = valid.to_string().replace("1.0", "1e100");
        assert!(InferencePlan::from_json(&nonfinite).is_err());
    }

    #[test]
    fn json_budget_is_explicit_and_counts_bytes() {
        let payload = record().to_string();
        assert!(matches!(
            InferencePlan::from_json_with_limit(&payload, payload.len() - 1),
            Err(InferenceError::JsonLimit { .. })
        ));
        assert!(InferencePlan::from_json_with_limit(&payload, payload.len()).is_ok());
    }

    #[test]
    fn portable_address_space_matches_wasm32() {
        for shape in [json!([4_294_967_296u64, 2]), json!([u32::MAX, 2])] {
            let mut invalid = record();
            invalid["input_shape"] = shape;
            assert!(InferencePlan::from_json(&invalid.to_string()).is_err());
        }
        let mut invalid = record();
        invalid["input_shape"] = json!([2_147_483_647u32, 2]);
        invalid["stages"][0]["cols"] = json!(3);
        invalid["stages"][0]["weight"] = json!([1., 0., 0., 0., 1., 0.]);
        invalid["stages"][0]["bias"] = json!([0., 0., 0.]);
        assert!(matches!(
            InferencePlan::from_json(&invalid.to_string()),
            Err(InferenceError::PortableAddressSpace)
        ));

        #[cfg(target_pointer_width = "64")]
        {
            let layer = Linear::new("local", 2, 2).unwrap();
            let local = InferencePlan::from_module(
                &layer,
                NdLayout::contiguous(&[u32::MAX as usize, 2]).unwrap(),
            )
            .unwrap();
            assert!(matches!(
                local.to_json(),
                Err(InferenceError::PortableAddressSpace)
            ));
        }
    }

    #[test]
    fn parameter_bits_survive_portable_json() {
        let mut values = vec![
            0.0,
            -0.0,
            f32::MAX,
            -f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
        ];
        let mut bits = 17u32;
        for _ in 0..4096 {
            bits = bits.wrapping_mul(1664525).wrapping_add(1013904223);
            let value = f32::from_bits(bits);
            if value.is_finite() {
                values.push(value);
            }
        }
        let plan = InferencePlan::from_operations(
            NdLayout::contiguous(&[1]).unwrap(),
            vec![InferenceOp::Linear {
                weight: Tensor::from_vec(1, values.len(), values.clone()).unwrap(),
                bias: Tensor::zeros(1, values.len()).unwrap(),
            }],
        )
        .unwrap();
        let payload = plan.to_json().unwrap();
        let restored = InferencePlan::from_json(&payload).unwrap();
        let (weight, _) = restored.parameter_snapshots().next().unwrap();
        assert_eq!(
            weight
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(restored.to_json().unwrap(), payload);
    }
}
