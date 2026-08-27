// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use st_core::runtime::zspace_runtime_protocol_catalog::{
    validate_zspace_runtime_protocol_catalog_value, zspace_runtime_protocol_catalog,
    ZSpaceRuntimeProtocolCatalogError,
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

const WASM_PROTOCOL_CATALOG_MAX_INGRESS_BYTES: u64 = 1_024 * 1_024;
const WASM_PROTOCOL_CATALOG_MAX_INGRESS_NODES: u64 = 20_000;
const WASM_PROTOCOL_CATALOG_MAX_INGRESS_DEPTH: u32 = 16;

fn catalog_from_json(input_json: &str) -> Result<Value, String> {
    let value = bounded_json_value_from_str(
        input_json,
        WASM_PROTOCOL_CATALOG_MAX_INGRESS_BYTES,
        WASM_PROTOCOL_CATALOG_MAX_INGRESS_NODES,
        WASM_PROTOCOL_CATALOG_MAX_INGRESS_DEPTH,
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
        WASM_PROTOCOL_CATALOG_MAX_INGRESS_BYTES,
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
        assert_eq!(actual["protocol_count"], 3);
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
            WASM_PROTOCOL_CATALOG_MAX_INGRESS_BYTES,
            WASM_PROTOCOL_CATALOG_MAX_INGRESS_NODES,
            WASM_PROTOCOL_CATALOG_MAX_INGRESS_DEPTH,
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
