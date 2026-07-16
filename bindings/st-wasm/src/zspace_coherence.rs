use serde_json::Value;
use st_core::inference::zspace_coherence::{
    build_zspace_coherence_distribution_witness, project_zspace_coherence,
    validate_zspace_coherence_distribution_witness, ZSpaceCoherenceDistributionWitness,
    ZSpaceCoherenceProjectionError, ZSpaceCoherenceProjectionRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn add_execution_client(payload: &mut Value) {
    payload
        .as_object_mut()
        .expect("Z-space coherence projection payload is an object")
        .insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
}

fn request_from_value(value: Value) -> Result<ZSpaceCoherenceProjectionRequest, String> {
    let request = value
        .as_object()
        .ok_or_else(|| "Z-space coherence projection request must be an object".to_owned())?;
    if request
        .get("diagnostics")
        .is_some_and(|value| !value.is_object())
    {
        return Err("Z-space coherence projection 'diagnostics' must be an object".to_owned());
    }
    if request
        .get("coherence")
        .is_some_and(|value| !value.is_array())
    {
        return Err("Z-space coherence projection 'coherence' must be an array".to_owned());
    }
    if request
        .get("contour")
        .is_some_and(|value| !value.is_object() && !value.is_null())
    {
        return Err("Z-space coherence projection 'contour' must be an object".to_owned());
    }
    if request
        .get("config")
        .is_some_and(|value| !value.is_object())
    {
        return Err("Z-space coherence projection 'config' must be an object".to_owned());
    }
    if request
        .get("classification_policy")
        .is_some_and(|value| !value.is_object())
    {
        return Err(
            "Z-space coherence projection 'classification_policy' must be an object".to_owned(),
        );
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(input_json: &str) -> Result<ZSpaceCoherenceProjectionRequest, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

fn weights_from_value(value: Value) -> Result<Vec<f64>, String> {
    if !value.is_array() {
        return Err("Z-space coherence normalized weights must be an array".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn witness_from_value(value: Value) -> Result<ZSpaceCoherenceDistributionWitness, String> {
    if !value.is_object() {
        return Err("Z-space coherence distribution witness must be an object".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Project coherence diagnostics through the shared Rust contract for WASM clients.
pub fn zspace_coherence_project_value(
    request: ZSpaceCoherenceProjectionRequest,
) -> Result<Value, ZSpaceCoherenceProjectionError> {
    let mut payload = serde_json::to_value(project_zspace_coherence(request)?)
        .expect("Z-space coherence projection payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

/// Build complete coherence distribution evidence through the shared Rust contract.
pub fn zspace_coherence_distribution_witness_value(
    normalized_weights: Vec<f64>,
) -> Result<Value, ZSpaceCoherenceProjectionError> {
    let witness = build_zspace_coherence_distribution_witness(&normalized_weights)?;
    Ok(serde_json::to_value(witness)
        .expect("Z-space coherence distribution witness is serializable"))
}

/// Validate portable coherence distribution evidence and reconstruct its summary in Rust.
pub fn zspace_coherence_distribution_validate_value(
    witness: ZSpaceCoherenceDistributionWitness,
) -> Result<Value, ZSpaceCoherenceProjectionError> {
    let summary = validate_zspace_coherence_distribution_witness(&witness)?;
    Ok(serde_json::to_value(summary)
        .expect("Z-space coherence distribution summary is serializable"))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceCoherenceProjectJson)]
pub fn zspace_coherence_project_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = zspace_coherence_project_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceCoherenceProjectObject)]
pub fn zspace_coherence_project_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = zspace_coherence_project_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceCoherenceDistributionWitnessJson)]
pub fn zspace_coherence_distribution_witness_json(
    normalized_weights_json: &str,
) -> Result<String, JsValue> {
    let value = serde_json::from_str(normalized_weights_json).map_err(js_error)?;
    let weights = weights_from_value(value).map_err(js_error)?;
    let payload = zspace_coherence_distribution_witness_value(weights).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceCoherenceDistributionWitnessObject)]
pub fn zspace_coherence_distribution_witness_object(
    normalized_weights: &JsValue,
) -> Result<JsValue, JsValue> {
    let value =
        serde_wasm_bindgen::from_value::<Value>(normalized_weights.clone()).map_err(js_error)?;
    let weights = weights_from_value(value).map_err(js_error)?;
    let payload = zspace_coherence_distribution_witness_value(weights).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceCoherenceDistributionValidateJson)]
pub fn zspace_coherence_distribution_validate_json(witness_json: &str) -> Result<String, JsValue> {
    let value = serde_json::from_str(witness_json).map_err(js_error)?;
    let witness = witness_from_value(value).map_err(js_error)?;
    let payload = zspace_coherence_distribution_validate_value(witness).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceCoherenceDistributionValidateObject)]
pub fn zspace_coherence_distribution_validate_object(
    witness: &JsValue,
) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(witness.clone()).map_err(js_error)?;
    let witness = witness_from_value(value).map_err(js_error)?;
    let payload = zspace_coherence_distribution_validate_value(witness).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> ZSpaceCoherenceProjectionRequest {
        serde_json::from_value(json!({
            "diagnostics": {
                "mean_coherence": 0.3333333333333333,
                "coherence_entropy": 1.0296530140645737,
                "energy_ratio": 0.7,
                "z_bias": -0.12,
                "fractional_order": 0.4,
                "normalized_weights": [0.5, 0.3, 0.2],
                "preserved_channels": 3,
                "discarded_channels": 0,
                "dominant_channel": 0
            },
            "coherence": [0.6, 0.3, 0.1]
        }))
        .expect("valid coherence request")
    }

    #[test]
    fn wasm_projection_matches_rust_except_client_metadata() {
        let request = request();
        let expected = serde_json::to_value(
            project_zspace_coherence(request.clone()).expect("valid Rust projection"),
        )
        .expect("serializable Rust projection");
        let mut actual = zspace_coherence_project_value(request).expect("valid WASM projection");
        actual
            .as_object_mut()
            .expect("projection object")
            .remove("execution_client");

        assert_eq!(actual, expected);
        assert_eq!(actual["classification"]["label"], "cascade_imbalance");
        assert_eq!(
            actual["classification"]["reason"],
            "dominant_energy_ratio_at_or_above_cascade_min"
        );
        assert_eq!(actual["classification"]["semantic_backend"], "rust");
        assert_eq!(
            actual["contract_version"],
            "spiraltorch.zspace_coherence_projection.v2"
        );
        assert!(actual["evidence_validation_formula"]
            .as_str()
            .expect("evidence validation formula")
            .contains("coherence_entropy~=H(p)"));
        assert_eq!(
            actual["control"]["kind"],
            "spiraltorch.zspace_coherence_control"
        );
        assert_eq!(
            actual["control"]["contract_version"],
            "spiraltorch.zspace_coherence_control.v1"
        );
        assert_eq!(actual["control"]["semantic_backend"], "rust");
        assert!(actual["control"]["control_formula"].is_string());
        assert_eq!(
            actual["control"]["spectral_radius"],
            actual["derived"]["concentration"]
        );
        assert_eq!(
            actual["control"]["spectral_entropy"],
            actual["derived"]["normalized_entropy"]
        );
    }

    #[test]
    fn wasm_distribution_witness_is_the_same_rust_contract() {
        let payload = zspace_coherence_distribution_witness_value(vec![0.6, 0.3, 0.1]).unwrap();
        assert_eq!(
            payload["contract_version"],
            "spiraltorch.zspace_coherence_distribution_witness.v1"
        );
        assert_eq!(payload["semantic_backend"], "rust");
        let witness: ZSpaceCoherenceDistributionWitness =
            serde_json::from_value(payload).expect("WASM witness remains portable");
        let summary = zspace_coherence_distribution_validate_value(witness.clone()).unwrap();
        assert_eq!(summary["channels"], 3);
        assert!((summary["weight_mass"].as_f64().unwrap() - 1.0).abs() < 1.0e-12);

        let mut tampered = witness;
        tampered.semantic_backend = "wasm".to_owned();
        assert!(zspace_coherence_distribution_validate_value(tampered)
            .expect_err("client must not claim semantic ownership")
            .to_string()
            .contains("semantic_backend"));
        assert!(weights_from_value(json!({"weights": [1.0]}))
            .expect_err("weights object must fail")
            .contains("must be an array"));
    }

    #[test]
    fn wasm_ingress_is_strict_and_accepts_trace_entropy_alias() {
        let parsed = request_from_json(
            r#"{"diagnostics":{"mean_coherence":0.2,"entropy":0.3,"energy_ratio":0.6,"z_bias":0.1,"fractional_order":0.5}}"#,
        )
        .expect("trace alias");
        assert_eq!(parsed.diagnostics.coherence_entropy, 0.3);

        let custom = request_from_json(
            r#"{"diagnostics":{"mean_coherence":0.3333333333333333,"coherence_entropy":1.0296530140645737,"energy_ratio":0.7,"z_bias":0.1,"fractional_order":0.5,"normalized_weights":[0.5,0.3,0.2]},"classification_policy":{"background_energy_ratio_max":0.0001,"cascade_energy_ratio_min":0.8}}"#,
        )
        .expect("custom classification policy");
        let custom = zspace_coherence_project_value(custom).expect("custom projection");
        assert_eq!(custom["classification"]["label"], "diffuse_drift");
        assert_eq!(
            custom["classification"]["policy"]["cascade_energy_ratio_min"],
            0.8
        );

        assert!(request_from_json("[]")
            .expect_err("array request must fail")
            .contains("must be an object"));
        assert!(request_from_json(
            r#"{"diagnostics":{"mean_coherence":0.2,"coherence_entropy":0.3,"energy_ratio":0.6,"z_bias":0.1,"fractional_order":0.5},"unknown":true}"#,
        )
        .expect_err("unknown field must fail")
        .contains("unknown field"));
        assert!(request_from_json(
            r#"{"diagnostics":{"mean_coherence":0.2,"coherence_entropy":0.3,"energy_ratio":0.6,"z_bias":0.1,"fractional_order":0.5,"preserved_channels":1},"config":null}"#,
        )
        .expect_err("null config must fail")
        .contains("config' must be an object"));
        assert!(request_from_json(
            r#"{"diagnostics":{"mean_coherence":0.2,"coherence_entropy":0.3,"energy_ratio":0.6,"z_bias":0.1,"fractional_order":0.5},"classification_policy":null}"#,
        )
        .expect_err("null classification policy must fail")
        .contains("classification_policy' must be an object"));

        let inconsistent = request_from_json(
            r#"{"diagnostics":{"mean_coherence":0.3333333333333333,"coherence_entropy":0.5,"energy_ratio":0.7,"z_bias":0.1,"fractional_order":0.5,"normalized_weights":[0.5,0.3,0.2]},"coherence":[0.6,0.3,0.1]}"#,
        )
        .expect("structurally valid but contradictory evidence");
        assert!(zspace_coherence_project_value(inconsistent)
            .expect_err("Rust must reject contradictory entropy")
            .to_string()
            .contains("coherence_entropy"));
    }
}
