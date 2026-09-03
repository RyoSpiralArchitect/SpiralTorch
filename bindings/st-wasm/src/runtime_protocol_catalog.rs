// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use st_core::runtime::zspace_runtime_protocol_catalog::{
    validate_zspace_runtime_protocol_catalog_value, zspace_runtime_protocol_catalog,
    ZSpaceRuntimeProtocolCatalogError, ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_BYTES,
    ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_DEPTH,
    ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_NODES,
};

use crate::utils::bounded_json_value_from_str;

#[cfg(target_arch = "wasm32")]
use js_sys::JsString;
#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::{bounded_json_string_from_js, js_error};

fn catalog_from_json(input_json: &str) -> Result<Value, String> {
    let value = bounded_json_value_from_str(
        input_json,
        ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_BYTES,
        ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_NODES,
        ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_DEPTH,
        "Z-space runtime protocol catalog",
    )?;
    if value.is_object() {
        Ok(value)
    } else {
        Err("Z-space runtime protocol catalog must be an object".to_owned())
    }
}

pub fn zspace_runtime_protocol_catalog_value() -> Result<Value, ZSpaceRuntimeProtocolCatalogError> {
    serde_json::to_value(zspace_runtime_protocol_catalog()?).map_err(|error| {
        ZSpaceRuntimeProtocolCatalogError::Encoding {
            message: error.to_string(),
        }
    })
}

pub fn validate_zspace_runtime_protocol_catalog_report_value(
    catalog: Value,
) -> Result<Value, ZSpaceRuntimeProtocolCatalogError> {
    serde_json::to_value(validate_zspace_runtime_protocol_catalog_value(catalog)?).map_err(
        |error| ZSpaceRuntimeProtocolCatalogError::Encoding {
            message: error.to_string(),
        },
    )
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceRuntimeProtocolCatalogJson)]
pub fn zspace_runtime_protocol_catalog_json() -> Result<String, JsValue> {
    let catalog = zspace_runtime_protocol_catalog_value().map_err(js_error)?;
    serde_json::to_string(&catalog).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceRuntimeProtocolCatalogObject)]
pub fn zspace_runtime_protocol_catalog_object() -> Result<JsValue, JsValue> {
    let catalog = zspace_runtime_protocol_catalog_value().map_err(js_error)?;
    to_json_compatible_js(&catalog)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceRuntimeProtocolCatalogJson)]
pub fn validate_zspace_runtime_protocol_catalog_json(
    catalog_json: &JsString,
) -> Result<String, JsValue> {
    let catalog_json = bounded_json_string_from_js(
        catalog_json,
        ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_BYTES,
        "Z-space runtime protocol catalog JSON",
    )?;
    let catalog = catalog_from_json(&catalog_json).map_err(js_error)?;
    let catalog =
        validate_zspace_runtime_protocol_catalog_report_value(catalog).map_err(js_error)?;
    serde_json::to_string(&catalog).map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wasm_catalog_matches_rust_exactly() {
        let expected = serde_json::to_value(zspace_runtime_protocol_catalog().expect("catalog"))
            .expect("catalog value");
        let actual = zspace_runtime_protocol_catalog_value().expect("WASM catalog");

        assert_eq!(actual, expected);
        assert_eq!(actual["semantic_backend"], "rust");
        assert_eq!(actual["protocol_count"], 6);
        assert_eq!(
            actual["protocol_order_rule"],
            "generation_evidence,periodicity,stochastic_schrodinger,stochastic_schrodinger_complex,repetition_unlikelihood,semantic_review"
        );
        for protocol in actual["protocols"].as_array().expect("protocol array") {
            let clients = protocol["clients"].as_array().expect("client array");
            assert_eq!(clients[0]["normal_admission"]["profile"], "typed_native");
            assert!(clients[0]["normal_admission"]["limits"].is_null());
            assert_eq!(
                clients[1]["normal_admission"]["profile"],
                "passive_json_containers"
            );
            assert_eq!(
                clients[2]["normal_admission"]["profile"],
                "bounded_json_string"
            );
            assert_eq!(
                clients[1]["normal_admission"]["limits"],
                clients[2]["normal_admission"]["limits"]
            );
        }
        assert_eq!(
            validate_zspace_runtime_protocol_catalog_report_value(actual.clone())
                .expect("catalog replay"),
            actual
        );
    }

    #[test]
    fn wasm_catalog_validator_rejects_tampering_and_unbounded_json() {
        let mut tampered = zspace_runtime_protocol_catalog_value().expect("catalog");
        tampered["protocols"][0]["clients"][2]["trusted_legacy_replay"] = json!(true);
        assert_eq!(
            validate_zspace_runtime_protocol_catalog_report_value(tampered),
            Err(ZSpaceRuntimeProtocolCatalogError::CatalogMismatch)
        );

        assert!(catalog_from_json("[]")
            .expect_err("catalog must be an object")
            .contains("must be an object"));
        assert!(bounded_json_value_from_str(
            r#"{"catalog_id":"a","catalog_id":"b"}"#,
            ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_BYTES,
            ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_NODES,
            ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_DEPTH,
            "catalog",
        )
        .expect_err("duplicate keys fail closed")
        .contains("duplicate JSON object key"));
    }

    #[test]
    fn catalogued_wasm_operations_are_declared_and_legacy_free() {
        let declarations = include_str!("../types/spiraltorch-wasm.d.ts");
        let catalog = zspace_runtime_protocol_catalog().expect("catalog");

        for protocol in catalog.protocols {
            let wasm = protocol
                .clients
                .into_iter()
                .find(|surface| surface.client == "wasm")
                .expect("WASM surface");
            assert!(!wasm.trusted_legacy_replay);
            for operation in wasm.operations {
                assert!(
                    declarations.contains(&format!("function {operation}(")),
                    "missing {operation}"
                );
                assert!(!operation.to_ascii_lowercase().contains("legacy"));
            }
        }
        for operation in [
            "zspaceRuntimeProtocolCatalogJson",
            "zspaceRuntimeProtocolCatalogObject",
            "validateZspaceRuntimeProtocolCatalogJson",
        ] {
            assert!(declarations.contains(&format!("function {operation}(")));
        }
        assert!(!declarations.contains("function validateZspaceRuntimeProtocolCatalogObject("));
    }
}
