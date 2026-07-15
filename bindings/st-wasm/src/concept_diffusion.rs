use serde_json::Value;
use st_core::inference::concept_diffusion::{
    apply_concept_diffusion, ConceptDiffusionError, ConceptDiffusionRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn request_from_value(value: Value) -> Result<ConceptDiffusionRequest, String> {
    let request = value
        .as_object()
        .ok_or_else(|| "Z-space concept diffusion request must be an object".to_owned())?;
    for field in ["tags", "state", "affinity", "z_bias"] {
        if request.get(field).is_some_and(|value| !value.is_array()) {
            return Err(format!(
                "Z-space concept diffusion '{field}' must be an array"
            ));
        }
    }
    if request
        .get("diffusion_tensor")
        .is_some_and(|value| !value.is_array() && !value.is_null())
    {
        return Err("Z-space concept diffusion 'diffusion_tensor' must be an array".to_owned());
    }
    if request
        .get("observation")
        .is_some_and(|value| !value.is_object() && !value.is_null())
    {
        return Err("Z-space concept diffusion 'observation' must be an object".to_owned());
    }
    if request
        .get("config")
        .is_some_and(|value| !value.is_object())
    {
        return Err("Z-space concept diffusion 'config' must be an object".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(input_json: &str) -> Result<ConceptDiffusionRequest, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

pub fn zspace_concept_diffusion_value(
    request: ConceptDiffusionRequest,
) -> Result<Value, ConceptDiffusionError> {
    Ok(serde_json::to_value(apply_concept_diffusion(request)?)
        .expect("Z-space concept diffusion payload is serializable"))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceConceptDiffusionJson)]
pub fn zspace_concept_diffusion_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = zspace_concept_diffusion_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceConceptDiffusionObject)]
pub fn zspace_concept_diffusion_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = zspace_concept_diffusion_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wasm_concept_diffusion_matches_rust_exactly() {
        let request: ConceptDiffusionRequest = serde_json::from_value(json!({
            "tags": ["left", "right"],
            "state": [1.0, 0.0],
            "affinity": [[0.0, 1.0], [1.0, 0.0]],
            "config": {"timestep": 0.25}
        }))
        .expect("valid request");
        let expected = serde_json::to_value(
            apply_concept_diffusion(request.clone()).expect("valid Rust transition"),
        )
        .expect("serializable Rust transition");
        let actual =
            zspace_concept_diffusion_value(request).expect("valid WASM diffusion transition");

        assert_eq!(actual, expected);
        assert_eq!(actual["next_state"], json!([0.75, 0.25]));
        assert!(
            actual["effects"]["entropy_after_diffusion"]
                .as_f64()
                .expect("numeric entropy")
                > actual["effects"]["entropy_after_bias"]
                    .as_f64()
                    .expect("numeric entropy")
        );
    }

    #[test]
    fn wasm_concept_diffusion_ingress_fails_closed_on_contract_drift() {
        let unknown = request_from_json(r#"{"tags":["only"],"state":[1.0],"affinitty":[[0.0]]}"#)
            .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));

        let wrong_shape = request_from_json(r#"{"state":{}}"#).expect_err("state must be an array");
        assert_eq!(
            wrong_shape,
            "Z-space concept diffusion 'state' must be an array"
        );

        let null_config =
            request_from_json(r#"{"config":null}"#).expect_err("config must be a concrete object");
        assert_eq!(
            null_config,
            "Z-space concept diffusion 'config' must be an object"
        );

        let wrong_request = request_from_json(r#"[]"#).expect_err("request must be an object");
        assert_eq!(
            wrong_request,
            "Z-space concept diffusion request must be an object"
        );
    }

    #[test]
    fn wasm_concept_diffusion_preserves_rust_validation() {
        let request = request_from_json(
            r#"{"tags":["left","right"],"state":[0.5,0.5],"affinity":[[0.0,1.0],[0.5,0.0]]}"#,
        )
        .expect("syntactically valid request");
        let error = zspace_concept_diffusion_value(request)
            .expect_err("asymmetric affinity must fail closed");

        assert!(matches!(
            error,
            ConceptDiffusionError::AsymmetricMatrix {
                field: "affinity",
                ..
            }
        ));
    }
}
