use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use st_core::runtime::zspace_evidence::{
    summarize_zspace_polarity_evidence, validate_zspace_polarity_evidence_value,
    ZSpacePolarityEvidenceRequest,
};
use st_core::runtime::zspace_optimizer::{
    initialize_zspace_meta_optimizer, plan_zspace_parameter_trajectory,
    plan_zspace_parameter_trajectory_policy_from_value, restore_zspace_meta_optimizer,
    transition_zspace_meta_optimizer, validate_zspace_parameter_trajectory_policy_value,
    validate_zspace_parameter_trajectory_value, zspace_parameter_control_from_value,
    ZSpaceMetaOptimizerConfig, ZSpaceMetaOptimizerRestoreRequest, ZSpaceMetaOptimizerStepRequest,
    ZSpaceParameterTrajectoryPolicy, ZSpaceParameterTrajectoryRequest,
};
use st_core::runtime::zspace_optimizer_feedback::{
    control_zspace_optimizer_feedback, initialize_zspace_optimizer_feedback,
    observe_zspace_optimizer_feedback, restore_zspace_optimizer_feedback,
    ZSpaceOptimizerFeedbackConfig, ZSpaceOptimizerFeedbackControlRequest,
    ZSpaceOptimizerFeedbackObserveRequest, ZSpaceOptimizerFeedbackRestoreRequest,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn request_from_value<T: DeserializeOwned>(value: Value, label: &str) -> Result<T, String> {
    if !value.is_object() {
        return Err(format!("{label} must be an object"));
    }
    serde_json::from_value(value).map_err(|error| format!("invalid {label}: {error}"))
}

fn request_from_json<T: DeserializeOwned>(request_json: &str, label: &str) -> Result<T, String> {
    let value: Value = serde_json::from_str(request_json)
        .map_err(|error| format!("invalid {label} JSON: {error}"))?;
    request_from_value(value, label)
}

fn response_value<T: Serialize>(response: &T) -> Result<Value, String> {
    let mut value = serde_json::to_value(response)
        .map_err(|error| format!("Z-space meta-optimizer response encoding failed: {error}"))?;
    value
        .as_object_mut()
        .expect("Z-space meta-optimizer response is an object")
        .insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
    Ok(value)
}

fn semantic_report_value(mut report: Value, label: &str) -> Result<Value, String> {
    let object = report
        .as_object_mut()
        .ok_or_else(|| format!("{label} must be an object"))?;
    if let Some(client) = object.remove("execution_client") {
        if client != Value::String("wasm".to_owned()) {
            return Err(format!("{label} has an untrusted execution_client"));
        }
    }
    Ok(report)
}

fn trajectory_policy_from_name(name: &str) -> Result<ZSpaceParameterTrajectoryPolicy, String> {
    serde_json::from_value(Value::String(name.trim().to_lowercase()))
        .map_err(|error| format!("invalid Z-space parameter-trajectory policy: {error}"))
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

pub fn zspace_meta_optimizer_init_value(
    config: ZSpaceMetaOptimizerConfig,
) -> Result<Value, String> {
    let checkpoint = initialize_zspace_meta_optimizer(config).map_err(|error| error.to_string())?;
    response_value(&checkpoint)
}

pub fn zspace_meta_optimizer_restore_value(
    request: ZSpaceMetaOptimizerRestoreRequest,
) -> Result<Value, String> {
    let checkpoint = restore_zspace_meta_optimizer(request).map_err(|error| error.to_string())?;
    response_value(&checkpoint)
}

pub fn zspace_meta_optimizer_step_value(
    request: ZSpaceMetaOptimizerStepRequest,
) -> Result<Value, String> {
    let report = transition_zspace_meta_optimizer(request).map_err(|error| error.to_string())?;
    response_value(&report)
}

pub fn zspace_meta_optimizer_parameter_control_value(report: Value) -> Result<Value, String> {
    if !report.is_object() {
        return Err("Z-space meta-optimizer report must be an object".to_owned());
    }
    let control = zspace_parameter_control_from_value(report).map_err(|error| error.to_string())?;
    response_value(&control)
}

pub fn zspace_parameter_trajectory_value(
    request: ZSpaceParameterTrajectoryRequest,
) -> Result<Value, String> {
    let report = plan_zspace_parameter_trajectory(request).map_err(|error| error.to_string())?;
    response_value(&report)
}

pub fn zspace_parameter_trajectory_validate_value(report: Value) -> Result<Value, String> {
    let report = semantic_report_value(report, "Z-space parameter-trajectory report")?;
    let report =
        validate_zspace_parameter_trajectory_value(report).map_err(|error| error.to_string())?;
    response_value(&report)
}

pub fn zspace_parameter_trajectory_policy_value(
    source_report: Value,
    policy: ZSpaceParameterTrajectoryPolicy,
) -> Result<Value, String> {
    let source_report =
        semantic_report_value(source_report, "Z-space parameter-trajectory source report")?;
    let report = plan_zspace_parameter_trajectory_policy_from_value(source_report, policy)
        .map_err(|error| error.to_string())?;
    response_value(&report)
}

pub fn zspace_parameter_trajectory_policy_validate_value(report: Value) -> Result<Value, String> {
    let report = semantic_report_value(report, "Z-space parameter-trajectory policy report")?;
    let report = validate_zspace_parameter_trajectory_policy_value(report)
        .map_err(|error| error.to_string())?;
    response_value(&report)
}

pub fn zspace_polarity_evidence_value(
    request: ZSpacePolarityEvidenceRequest,
) -> Result<Value, String> {
    let report = summarize_zspace_polarity_evidence(request).map_err(|error| error.to_string())?;
    response_value(&report)
}

pub fn zspace_polarity_evidence_validate_value(report: Value) -> Result<Value, String> {
    let report = semantic_report_value(report, "Z-space polarity evidence report")?;
    let report =
        validate_zspace_polarity_evidence_value(report).map_err(|error| error.to_string())?;
    response_value(&report)
}

pub fn zspace_optimizer_feedback_init_value(
    config: ZSpaceOptimizerFeedbackConfig,
) -> Result<Value, String> {
    let checkpoint =
        initialize_zspace_optimizer_feedback(config).map_err(|error| error.to_string())?;
    response_value(&checkpoint)
}

pub fn zspace_optimizer_feedback_restore_value(
    request: ZSpaceOptimizerFeedbackRestoreRequest,
) -> Result<Value, String> {
    let checkpoint =
        restore_zspace_optimizer_feedback(request).map_err(|error| error.to_string())?;
    response_value(&checkpoint)
}

pub fn zspace_optimizer_feedback_observe_value(
    request: ZSpaceOptimizerFeedbackObserveRequest,
) -> Result<Value, String> {
    let report = observe_zspace_optimizer_feedback(request).map_err(|error| error.to_string())?;
    response_value(&report)
}

pub fn zspace_optimizer_feedback_control_value(
    request: ZSpaceOptimizerFeedbackControlRequest,
) -> Result<Value, String> {
    let report = control_zspace_optimizer_feedback(request).map_err(|error| error.to_string())?;
    response_value(&report)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerInitJson)]
pub fn zspace_meta_optimizer_init_json(config_json: &str) -> Result<String, JsValue> {
    let config =
        request_from_json(config_json, "Z-space meta-optimizer config").map_err(js_error)?;
    let payload = zspace_meta_optimizer_init_value(config).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerInitObject)]
pub fn zspace_meta_optimizer_init_object(config: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(config.clone()).map_err(js_error)?;
    let config = request_from_value(value, "Z-space meta-optimizer config").map_err(js_error)?;
    let payload = zspace_meta_optimizer_init_value(config).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerRestoreJson)]
pub fn zspace_meta_optimizer_restore_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json, "Z-space meta-optimizer restore request")
        .map_err(js_error)?;
    let payload = zspace_meta_optimizer_restore_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerRestoreObject)]
pub fn zspace_meta_optimizer_restore_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request =
        request_from_value(value, "Z-space meta-optimizer restore request").map_err(js_error)?;
    let payload = zspace_meta_optimizer_restore_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerStepJson)]
pub fn zspace_meta_optimizer_step_json(request_json: &str) -> Result<String, JsValue> {
    let request =
        request_from_json(request_json, "Z-space meta-optimizer step request").map_err(js_error)?;
    let payload = zspace_meta_optimizer_step_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerStepObject)]
pub fn zspace_meta_optimizer_step_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request =
        request_from_value(value, "Z-space meta-optimizer step request").map_err(js_error)?;
    let payload = zspace_meta_optimizer_step_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerParameterControlJson)]
pub fn zspace_meta_optimizer_parameter_control_json(report_json: &str) -> Result<String, JsValue> {
    let report: Value = serde_json::from_str(report_json).map_err(|error| {
        js_error(format!(
            "invalid Z-space meta-optimizer report JSON: {error}"
        ))
    })?;
    let payload = zspace_meta_optimizer_parameter_control_value(report).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerParameterControlObject)]
pub fn zspace_meta_optimizer_parameter_control_object(
    report: &JsValue,
) -> Result<JsValue, JsValue> {
    let report = serde_wasm_bindgen::from_value::<Value>(report.clone()).map_err(js_error)?;
    let payload = zspace_meta_optimizer_parameter_control_value(report).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceParameterTrajectoryJson)]
pub fn zspace_parameter_trajectory_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json, "Z-space parameter-trajectory request")
        .map_err(js_error)?;
    let payload = zspace_parameter_trajectory_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceParameterTrajectoryObject)]
pub fn zspace_parameter_trajectory_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request =
        request_from_value(value, "Z-space parameter-trajectory request").map_err(js_error)?;
    let payload = zspace_parameter_trajectory_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceParameterTrajectoryValidateJson)]
pub fn zspace_parameter_trajectory_validate_json(report_json: &str) -> Result<String, JsValue> {
    let report = serde_json::from_str(report_json).map_err(|error| {
        js_error(format!(
            "invalid Z-space parameter-trajectory report JSON: {error}"
        ))
    })?;
    let payload = zspace_parameter_trajectory_validate_value(report).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceParameterTrajectoryValidateObject)]
pub fn zspace_parameter_trajectory_validate_object(report: &JsValue) -> Result<JsValue, JsValue> {
    let report = serde_wasm_bindgen::from_value::<Value>(report.clone()).map_err(js_error)?;
    let payload = zspace_parameter_trajectory_validate_value(report).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceParameterTrajectoryPolicyJson)]
pub fn zspace_parameter_trajectory_policy_json(
    source_report_json: &str,
    policy: &str,
) -> Result<String, JsValue> {
    let source_report = serde_json::from_str(source_report_json).map_err(|error| {
        js_error(format!(
            "invalid Z-space parameter-trajectory source JSON: {error}"
        ))
    })?;
    let policy = trajectory_policy_from_name(policy).map_err(js_error)?;
    let payload =
        zspace_parameter_trajectory_policy_value(source_report, policy).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceParameterTrajectoryPolicyObject)]
pub fn zspace_parameter_trajectory_policy_object(
    source_report: &JsValue,
    policy: &str,
) -> Result<JsValue, JsValue> {
    let source_report =
        serde_wasm_bindgen::from_value::<Value>(source_report.clone()).map_err(js_error)?;
    let policy = trajectory_policy_from_name(policy).map_err(js_error)?;
    let payload =
        zspace_parameter_trajectory_policy_value(source_report, policy).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceParameterTrajectoryPolicyValidateJson)]
pub fn zspace_parameter_trajectory_policy_validate_json(
    report_json: &str,
) -> Result<String, JsValue> {
    let report = serde_json::from_str(report_json).map_err(|error| {
        js_error(format!(
            "invalid Z-space parameter-trajectory policy report JSON: {error}"
        ))
    })?;
    let payload = zspace_parameter_trajectory_policy_validate_value(report).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceParameterTrajectoryPolicyValidateObject)]
pub fn zspace_parameter_trajectory_policy_validate_object(
    report: &JsValue,
) -> Result<JsValue, JsValue> {
    let report = serde_wasm_bindgen::from_value::<Value>(report.clone()).map_err(js_error)?;
    let payload = zspace_parameter_trajectory_policy_validate_value(report).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePolarityEvidenceJson)]
pub fn zspace_polarity_evidence_json(request_json: &str) -> Result<String, JsValue> {
    let request =
        request_from_json(request_json, "Z-space polarity evidence request").map_err(js_error)?;
    let payload = zspace_polarity_evidence_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePolarityEvidenceObject)]
pub fn zspace_polarity_evidence_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request =
        request_from_value(value, "Z-space polarity evidence request").map_err(js_error)?;
    let payload = zspace_polarity_evidence_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePolarityEvidenceValidateJson)]
pub fn zspace_polarity_evidence_validate_json(report_json: &str) -> Result<String, JsValue> {
    let report = serde_json::from_str(report_json)
        .map_err(|error| js_error(format!("invalid Z-space polarity evidence JSON: {error}")))?;
    let payload = zspace_polarity_evidence_validate_value(report).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePolarityEvidenceValidateObject)]
pub fn zspace_polarity_evidence_validate_object(report: &JsValue) -> Result<JsValue, JsValue> {
    let report = serde_wasm_bindgen::from_value::<Value>(report.clone()).map_err(js_error)?;
    let payload = zspace_polarity_evidence_validate_value(report).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceOptimizerFeedbackInitJson)]
pub fn zspace_optimizer_feedback_init_json(config_json: &str) -> Result<String, JsValue> {
    let config =
        request_from_json(config_json, "Z-space optimizer feedback config").map_err(js_error)?;
    let payload = zspace_optimizer_feedback_init_value(config).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceOptimizerFeedbackInitObject)]
pub fn zspace_optimizer_feedback_init_object(config: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(config.clone()).map_err(js_error)?;
    let config =
        request_from_value(value, "Z-space optimizer feedback config").map_err(js_error)?;
    let payload = zspace_optimizer_feedback_init_value(config).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceOptimizerFeedbackRestoreJson)]
pub fn zspace_optimizer_feedback_restore_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json, "Z-space optimizer feedback restore request")
        .map_err(js_error)?;
    let payload = zspace_optimizer_feedback_restore_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceOptimizerFeedbackRestoreObject)]
pub fn zspace_optimizer_feedback_restore_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(value, "Z-space optimizer feedback restore request")
        .map_err(js_error)?;
    let payload = zspace_optimizer_feedback_restore_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceOptimizerFeedbackObserveJson)]
pub fn zspace_optimizer_feedback_observe_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(
        request_json,
        "Z-space optimizer feedback observation request",
    )
    .map_err(js_error)?;
    let payload = zspace_optimizer_feedback_observe_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceOptimizerFeedbackObserveObject)]
pub fn zspace_optimizer_feedback_observe_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(value, "Z-space optimizer feedback observation request")
        .map_err(js_error)?;
    let payload = zspace_optimizer_feedback_observe_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceOptimizerFeedbackControlJson)]
pub fn zspace_optimizer_feedback_control_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json, "Z-space optimizer feedback control request")
        .map_err(js_error)?;
    let payload = zspace_optimizer_feedback_control_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceOptimizerFeedbackControlObject)]
pub fn zspace_optimizer_feedback_control_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(value, "Z-space optimizer feedback control request")
        .map_err(js_error)?;
    let payload = zspace_optimizer_feedback_control_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::runtime::zspace_evidence::{
        ZSpacePolarityEvidenceRequest, ZSpacePolarityEvidenceRow,
    };
    use st_core::runtime::zspace_optimizer::{
        restore_zspace_meta_optimizer, transition_zspace_meta_optimizer,
        ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
    };
    use st_core::runtime::zspace_optimizer_feedback::{
        control_zspace_optimizer_feedback, initialize_zspace_optimizer_feedback,
        observe_zspace_optimizer_feedback, ZSpaceOptimizerFeedbackControlRequest,
        ZSpaceOptimizerFeedbackObservation, ZSpaceOptimizerFeedbackObserveRequest,
    };

    fn evidence_id(value: char) -> String {
        format!("sha256:{}", value.to_string().repeat(64))
    }

    #[test]
    fn wasm_step_matches_the_rust_contract_exactly() {
        let request: ZSpaceMetaOptimizerStepRequest = serde_json::from_value(json!({
            "config": {"dimension": 4},
            "state": {
                "z": [0.2, -0.1, 0.4, -0.3],
                "first_moment": [0.0, 0.0, 0.0, 0.0],
                "second_moment": [0.0, 0.0, 0.0, 0.0],
                "step": 0
            },
            "observation": {
                "speed": 0.8,
                "memory": 0.5,
                "stability": 0.6,
                "gradient": [0.1, -0.2, 0.3, -0.1],
                "telemetry": {
                    "topos.closure_pressure": 0.75,
                    "topos.training_hints.learning_rate_scale": 0.5
                }
            }
        }))
        .expect("valid request");
        let mut expected = serde_json::to_value(
            transition_zspace_meta_optimizer(request.clone()).expect("Rust transition"),
        )
        .expect("serializable report");
        expected.as_object_mut().expect("object").insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );

        let actual = zspace_meta_optimizer_step_value(request).expect("WASM transition");

        assert_eq!(actual, expected);
        assert_eq!(actual["semantic_backend"], "rust");
        assert_eq!(actual["execution_client"], "wasm");
        assert_eq!(actual["transition_validated"], true);
    }

    #[test]
    fn wasm_exposes_rust_validated_parameter_control_without_applying_parameters() {
        let request: ZSpaceMetaOptimizerStepRequest = serde_json::from_value(json!({
            "config": {"dimension": 2, "topos_control_gain": 1.0},
            "state": {
                "z": [0.0, 0.0],
                "first_moment": [0.0, 0.0],
                "second_moment": [0.0, 0.0],
                "step": 0
            },
            "observation": {
                "gradient": [0.1, -0.2],
                "telemetry": {
                    "topos.training_hints.learning_rate_scale": 0.5
                }
            }
        }))
        .expect("request");
        let report = transition_zspace_meta_optimizer(request).expect("Rust transition");

        let control = zspace_meta_optimizer_parameter_control_value(
            serde_json::to_value(report).expect("report value"),
        )
        .expect("parameter control");

        assert_eq!(
            control["contract_version"],
            "spiraltorch.zspace_parameter_control.v2"
        );
        assert_eq!(control["semantic_backend"], "rust");
        assert_eq!(control["absolute_learning_rate_scale"], 0.5);
        assert_eq!(control["source_step"], 1);
        assert_eq!(control["execution_client"], "wasm");
    }

    #[test]
    fn wasm_parameter_control_rejects_tampered_report() {
        let request: ZSpaceMetaOptimizerStepRequest = serde_json::from_value(json!({
            "config": {"dimension": 2},
            "state": {
                "z": [0.0, 0.0],
                "first_moment": [0.0, 0.0],
                "second_moment": [0.0, 0.0],
                "step": 0
            },
            "observation": {"gradient": [0.1, -0.2]}
        }))
        .expect("request");
        let report = transition_zspace_meta_optimizer(request).expect("Rust transition");
        let mut report = serde_json::to_value(report).expect("report value");
        report["adam"]["effective_learning_rate"] = json!(0.5);

        let error = zspace_meta_optimizer_parameter_control_value(report)
            .expect_err("tampered report must fail");

        assert!(error.contains("invariant"));
    }

    #[test]
    fn wasm_trajectory_policy_round_trips_through_the_rust_contract() {
        let request = ZSpaceParameterTrajectoryRequest {
            raw_learning_rate_scales: vec![1.2, 0.8, 0.5],
            nominal_learning_rates: vec![vec![0.01], vec![0.005], vec![0.0025]],
        };
        let source = zspace_parameter_trajectory_value(request).expect("source trajectory");
        assert_eq!(source["execution_client"], "wasm");

        let policy = zspace_parameter_trajectory_policy_value(
            source,
            ZSpaceParameterTrajectoryPolicy::DosePreservingComplement,
        )
        .expect("trajectory policy");
        assert_eq!(policy["semantic_backend"], "rust");
        assert_eq!(policy["execution_client"], "wasm");
        assert_eq!(policy["policy"], "dose_preserving_complement");
        let dose_ratio = policy["planned_dose_ratio"]
            .as_f64()
            .expect("numeric dose ratio");
        assert!((dose_ratio - 1.0).abs() <= 1.0e-12);

        let validated = zspace_parameter_trajectory_policy_validate_value(policy.clone())
            .expect("validated policy");
        assert_eq!(validated, policy);

        let mut tampered = policy;
        tampered["steps"][0]["planned_learning_rate_scale"] = json!(1.0);
        let error = zspace_parameter_trajectory_policy_validate_value(tampered)
            .expect_err("tampered policy must fail");
        assert!(error.contains("canonical Rust trajectory policy"));
    }

    #[test]
    fn wasm_polarity_evidence_round_trips_through_the_rust_contract() {
        let mut rows = Vec::new();
        for corpus in ['a', 'b', 'c'] {
            for seed in [13, 17, 23] {
                rows.push(ZSpacePolarityEvidenceRow {
                    corpus_id: evidence_id(corpus),
                    seed,
                    dose_normalized_shape_effect: 0.002,
                    complement_shape_effect: -0.001,
                    polarity_effect: -0.003,
                });
            }
        }
        let request = ZSpacePolarityEvidenceRequest {
            protocol_id: evidence_id('1'),
            runtime_identity_id: evidence_id('2'),
            trajectory_id: evidence_id('3'),
            trajectory_policy_id: evidence_id('4'),
            control_sequence_id: evidence_id('5'),
            nominal_schedule_sequence_id: evidence_id('6'),
            rows,
        };

        let report = zspace_polarity_evidence_value(request).expect("evidence aggregate");
        assert_eq!(report["semantic_backend"], "rust");
        assert_eq!(report["execution_client"], "wasm");
        assert_eq!(report["corpus_count"], 3);
        assert_eq!(report["seed_count_per_corpus"], 3);
        assert_eq!(report["bounded_polarity_improvement_observed"], true);

        let validated =
            zspace_polarity_evidence_validate_value(report.clone()).expect("validated evidence");
        assert_eq!(validated, report);

        let mut tampered = report;
        tampered["contrasts"]["polarity_effect"]["corpus_equal_weight_mean"] = json!(0.0);
        let error = zspace_polarity_evidence_validate_value(tampered)
            .expect_err("tampered evidence must fail");
        assert!(error.contains("canonical Rust evidence aggregate"));
    }

    #[test]
    fn wasm_restore_uses_rust_dimension_coercion() {
        let request: ZSpaceMetaOptimizerRestoreRequest = serde_json::from_value(json!({
            "config": {"dimension": 3},
            "state": {
                "z": [1.0],
                "first_moment": [2.0, 3.0, 4.0, 5.0],
                "second_moment": [],
                "step": 7
            },
            "strict": false
        }))
        .expect("valid request");
        let expected = restore_zspace_meta_optimizer(request.clone()).expect("Rust restore");
        let actual = zspace_meta_optimizer_restore_value(request).expect("WASM restore");

        assert_eq!(actual["state"]["z"], json!([1.0, 0.0, 0.0]));
        assert_eq!(actual["state"]["first_moment"], json!([2.0, 3.0, 4.0]));
        assert_eq!(actual["state"]["second_moment"], json!([0.0, 0.0, 0.0]));
        assert_eq!(
            actual["state"],
            serde_json::to_value(expected.state).unwrap()
        );
    }

    #[test]
    fn wasm_rejects_invalid_rust_state() {
        let request: ZSpaceMetaOptimizerRestoreRequest = serde_json::from_value(json!({
            "config": {"dimension": 2},
            "state": {
                "z": [0.0, 0.0],
                "first_moment": [0.0, 0.0],
                "second_moment": [0.0, -0.1],
                "step": 0
            },
            "strict": true
        }))
        .expect("syntactically valid request");
        let error = zspace_meta_optimizer_restore_value(request)
            .expect_err("negative second moment must fail closed");
        assert!(error.contains("second_moment[1]"));
    }

    #[test]
    fn wasm_requires_object_requests() {
        let error = request_from_value::<ZSpaceMetaOptimizerConfig>(
            json!([]),
            "Z-space meta-optimizer config",
        )
        .expect_err("array is not a config object");
        assert_eq!(error, "Z-space meta-optimizer config must be an object");
    }

    #[test]
    fn wasm_step_counter_stops_at_the_shared_exact_integer_limit() {
        let request: ZSpaceMetaOptimizerStepRequest = serde_json::from_value(json!({
            "config": {"dimension": 2},
            "state": {
                "z": [0.0, 0.0],
                "first_moment": [0.0, 0.0],
                "second_moment": [0.0, 0.0],
                "step": ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP
            },
            "observation": {"gradient": [0.1, -0.2]}
        }))
        .expect("exact cross-client step");

        let error = zspace_meta_optimizer_step_value(request)
            .expect_err("next step must exceed the shared exact integer limit");

        assert!(error.contains("cross-client maximum"));
    }

    #[test]
    fn wasm_feedback_observation_and_control_match_rust_exactly() {
        let config = ZSpaceOptimizerFeedbackConfig {
            warmup_observations: 0,
            ..ZSpaceOptimizerFeedbackConfig::default()
        };
        let checkpoint = initialize_zspace_optimizer_feedback(config.clone()).unwrap();
        let control_request = ZSpaceOptimizerFeedbackControlRequest {
            config: config.clone(),
            state: checkpoint.state,
            target_step: 1,
            proposed_learning_rate_scale: 0.5,
        };
        let control = control_zspace_optimizer_feedback(control_request).unwrap();
        let observe_request = ZSpaceOptimizerFeedbackObserveRequest {
            config,
            state: control.state_after,
            observation: ZSpaceOptimizerFeedbackObservation {
                step: 1,
                max_steps: Some(8),
                epoch: None,
                loss: 2.0,
                grad_norm: Some(1.0),
                learning_rate: Some(1.0e-4),
            },
        };
        let mut expected = serde_json::to_value(
            observe_zspace_optimizer_feedback(observe_request.clone()).unwrap(),
        )
        .unwrap();
        expected.as_object_mut().unwrap().insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );

        let actual = zspace_optimizer_feedback_observe_value(observe_request).unwrap();

        assert_eq!(actual, expected);
        assert_eq!(actual["semantic_backend"], "rust");
        assert_eq!(actual["projection"]["semantic_backend"], "rust");
    }

    #[test]
    fn wasm_feedback_rejects_out_of_order_control() {
        let checkpoint =
            initialize_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackConfig::default()).unwrap();
        let error =
            zspace_optimizer_feedback_control_value(ZSpaceOptimizerFeedbackControlRequest {
                config: checkpoint.config,
                state: checkpoint.state,
                target_step: 2,
                proposed_learning_rate_scale: 1.0,
            })
            .expect_err("out-of-order feedback control must fail");

        assert!(error.contains("next step"));
    }

    #[test]
    fn wasm_feedback_types_declare_the_complete_shared_surface() {
        let declarations = include_str!("../types/spiraltorch-wasm.d.ts");

        for symbol in [
            "ZSpaceOptimizerFeedbackConfigInput",
            "ZSpaceOptimizerFeedbackState",
            "ZSpaceOptimizerFeedbackObservationReport",
            "ZSpaceOptimizerFeedbackControlReport",
            "zspaceOptimizerFeedbackInitObject",
            "zspaceOptimizerFeedbackRestoreObject",
            "zspaceOptimizerFeedbackObserveObject",
            "zspaceOptimizerFeedbackControlObject",
            "ZSpacePolarityEvidenceRequest",
            "ZSpacePolarityEvidenceReport",
            "zspacePolarityEvidenceObject",
            "zspacePolarityEvidenceValidateObject",
        ] {
            assert!(declarations.contains(symbol), "missing {symbol}");
        }
        assert!(declarations.contains("st-core::runtime::zspace_optimizer_feedback"));
        assert!(declarations.contains("st-core::runtime::zspace_evidence"));
    }
}
