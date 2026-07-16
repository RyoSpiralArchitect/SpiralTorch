use serde::Deserialize;
use serde_json::Value;
use st_core::backend::device_caps::{BackendKind, DeviceCapsOverrides};
use st_core::backend::execution_plan::{AcceleratorFallback, ExecutionConfig};
use st_core::backend::runtime_probe::resolve_backend;
use st_core::backend::unison::RankKind;
use st_core::ops::rank_entry::try_plan_rank_with_config;

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankPlanRequest {
    kind: String,
    rows: u32,
    cols: u32,
    k: u32,
    #[serde(default = "default_backend")]
    backend: String,
    #[serde(default)]
    lane_width: Option<u32>,
    #[serde(default)]
    subgroup: Option<bool>,
    #[serde(default)]
    max_workgroup: Option<u32>,
    #[serde(default)]
    shared_mem_per_workgroup: Option<u32>,
    #[serde(default)]
    strict_accelerator: bool,
    #[serde(default)]
    tensor_util_wgpu_min_values: Option<usize>,
}

fn default_backend() -> String {
    "wgpu".to_owned()
}

fn request_from_value(value: Value) -> Result<RankPlanRequest, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(request_json: &str) -> Result<RankPlanRequest, String> {
    let value = serde_json::from_str(request_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

fn requested_backend(value: &str) -> Result<BackendKind, String> {
    if value.eq_ignore_ascii_case("auto") {
        Ok(BackendKind::Wgpu)
    } else {
        value
            .parse::<BackendKind>()
            .map_err(|error| error.to_string())
    }
}

fn add_client_fields(
    payload: &mut Value,
    requested_backend: BackendKind,
    effective_backend: BackendKind,
) {
    let object = payload
        .as_object_mut()
        .expect("rank-plan snapshot serializes as an object");
    object.insert("execution_client".to_owned(), "wasm".into());
    object.insert(
        "requested_backend".to_owned(),
        requested_backend.as_str().into(),
    );
    object.insert(
        "effective_backend".to_owned(),
        effective_backend.as_str().into(),
    );
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

fn rank_plan_value(request: RankPlanRequest) -> Result<Value, String> {
    let kind = request
        .kind
        .parse::<RankKind>()
        .map_err(|error| error.to_string())?;
    let requested_backend = requested_backend(&request.backend)?;
    let backend_resolution = resolve_backend(requested_backend);
    let effective_backend = backend_resolution.effective_backend;
    let caps = effective_backend
        .default_caps()
        .try_with_overrides(DeviceCapsOverrides {
            lane_width: request.lane_width,
            subgroup: request.subgroup,
            max_workgroup: request.max_workgroup,
            shared_mem_per_workgroup: request.shared_mem_per_workgroup,
        })
        .map_err(|error| error.to_string())?;
    let default_execution = ExecutionConfig::default();
    let execution_config = ExecutionConfig::new(
        if request.strict_accelerator {
            AcceleratorFallback::Forbid
        } else {
            AcceleratorFallback::Allow
        },
        request
            .tensor_util_wgpu_min_values
            .unwrap_or(default_execution.tensor_util_wgpu_min_values),
    );
    let plan = try_plan_rank_with_config(
        kind,
        request.rows,
        request.cols,
        request.k,
        caps,
        execution_config,
    )
    .map_err(|error| error.to_string())?;
    let mut payload =
        serde_json::to_value(plan.snapshot()).expect("rank-plan snapshot is serializable");
    add_client_fields(&mut payload, requested_backend, effective_backend);
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = rankPlanJson)]
pub fn rank_plan_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = rank_plan_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = rankPlanObject)]
pub fn rank_plan_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = rank_plan_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::backend::device_caps::DeviceCaps;
    use st_core::ops::rank_entry::{
        try_plan_rank_with_config, RANK_PLAN_CONTRACT_VERSION, RANK_PLAN_SEMANTIC_OWNER,
    };

    #[test]
    fn wasm_rank_plan_uses_the_rust_owned_contract() {
        let request = request_from_value(json!({
            "kind": "midk",
            "rows": 4,
            "cols": 128,
            "k": 8,
            "backend": "wgpu",
            "lane_width": 32,
            "subgroup": true,
            "max_workgroup": 256,
            "strict_accelerator": true,
            "tensor_util_wgpu_min_values": 4096
        }))
        .expect("valid rank-plan request");
        let payload = rank_plan_value(request).expect("valid rank plan");

        assert_eq!(payload["contract_version"], RANK_PLAN_CONTRACT_VERSION);
        assert_eq!(payload["semantic_owner"], RANK_PLAN_SEMANTIC_OWNER);
        assert_eq!(payload["semantic_backend"], "rust");
        assert_eq!(payload["execution_client"], "wasm");
        assert_eq!(payload["rank_kind"], "midk");
        assert_eq!(payload["input_elements"], 512);
        assert_eq!(payload["output_elements"], 32);
        assert_eq!(payload["device_caps"]["backend"], "wgpu");
        assert_eq!(payload["execution"]["accelerator_fallback"], "forbid");
        assert_eq!(payload["execution"]["tensor_util_wgpu_min_values"], 4096);
        assert!(payload["choice"]["workgroup"].as_u64().unwrap() > 0);
    }

    #[test]
    fn wasm_transport_matches_the_direct_rust_snapshot() {
        let request = request_from_value(json!({
            "kind": "midk",
            "rows": 4,
            "cols": 128,
            "k": 8,
            "backend": "wgpu",
            "strict_accelerator": true,
            "tensor_util_wgpu_min_values": 4096
        }))
        .expect("valid rank-plan request");
        let mut wasm = rank_plan_value(request).expect("valid WASM transport");
        let object = wasm.as_object_mut().expect("WASM payload object");
        object.remove("execution_client");
        object.remove("requested_backend");
        object.remove("effective_backend");

        let rust = try_plan_rank_with_config(
            RankKind::MidK,
            4,
            128,
            8,
            DeviceCaps::wgpu(32, true, 256),
            ExecutionConfig::new(AcceleratorFallback::Forbid, 4_096),
        )
        .expect("valid direct Rust plan");
        let rust = serde_json::to_value(rust.snapshot()).expect("serializable Rust snapshot");

        assert_eq!(wasm, rust);
    }

    #[test]
    fn wasm_rank_plan_rejects_invalid_shape_and_caps_in_rust() {
        let invalid_shape = request_from_value(json!({
            "kind": "topk",
            "rows": 2,
            "cols": 8,
            "k": 9
        }))
        .expect("shape reaches semantic validation");
        assert!(rank_plan_value(invalid_shape)
            .expect_err("k greater than cols must fail")
            .contains("exceeds cols"));

        let invalid_caps = request_from_value(json!({
            "kind": "topk",
            "rows": 2,
            "cols": 8,
            "k": 2,
            "lane_width": 0
        }))
        .expect("caps reach semantic validation");
        assert!(rank_plan_value(invalid_caps)
            .expect_err("zero lane width must fail")
            .contains("lane_width"));
    }

    #[test]
    fn wasm_rank_plan_ingress_rejects_unknown_fields() {
        let error = request_from_value(json!({
            "kind": "topk",
            "rows": 2,
            "cols": 8,
            "k": 2,
            "commander": "wasm"
        }))
        .expect_err("unknown request fields must fail closed");
        assert!(error.contains("unknown field"));
    }
}
