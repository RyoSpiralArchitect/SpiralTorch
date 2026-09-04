use serde::Deserialize;
use serde_json::Value;
use st_core::backend::device_caps::BackendKind;
use st_core::runtime::blackcat::bandit::SoftBanditMode;
use st_core::runtime::rank_adaptation::{
    RankAdaptationSession, RANK_ADAPTATION_MAX_SAFE_SELECTION_ID,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::rank_plan::{add_client_fields, resolve_rank_plan_value};
#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankAdaptationRequest {
    rank_plan: Value,
    scripts: Vec<String>,
    #[serde(default = "default_policy")]
    policy: String,
    #[serde(default)]
    seed: f64,
}

fn default_policy() -> String {
    "ucb".to_owned()
}

fn parse_policy(policy: &str) -> Result<SoftBanditMode, String> {
    match policy.trim().to_ascii_lowercase().as_str() {
        "ucb" | "upper_confidence_bound" => Ok(SoftBanditMode::UCB),
        "ts" | "thompson_sampling" => Ok(SoftBanditMode::TS),
        _ => Err("policy must be 'ucb' or 'thompson_sampling'".to_owned()),
    }
}

fn safe_u64(value: f64, field: &str) -> Result<u64, String> {
    if !value.is_finite()
        || value < 0.0
        || value.fract() != 0.0
        || value > RANK_ADAPTATION_MAX_SAFE_SELECTION_ID as f64
    {
        return Err(format!(
            "{field} must be a nonnegative JavaScript-safe integer"
        ));
    }
    Ok(value as u64)
}

struct ResolvedRankAdaptationSession {
    session: RankAdaptationSession,
    requested_backend: BackendKind,
    effective_backend: BackendKind,
}

fn session_from_value(value: Value) -> Result<ResolvedRankAdaptationSession, String> {
    let request = serde_json::from_value::<RankAdaptationRequest>(value)
        .map_err(|error| error.to_string())?;
    let resolved_plan = resolve_rank_plan_value(request.rank_plan)?;
    let session = RankAdaptationSession::try_from_spiralk(
        &resolved_plan.plan,
        &request.scripts,
        parse_policy(&request.policy)?,
        safe_u64(request.seed, "seed")?,
    )
    .map_err(|error| error.to_string())?;
    Ok(ResolvedRankAdaptationSession {
        session,
        requested_backend: resolved_plan.requested_backend,
        effective_backend: resolved_plan.effective_backend,
    })
}

fn add_execution_client(value: &mut Value) {
    value
        .as_object_mut()
        .expect("rank adaptation receipt serializes as an object")
        .insert("execution_client".to_owned(), "wasm".into());
}

fn decorate_plan(
    value: &mut Value,
    requested_backend: BackendKind,
    effective_backend: BackendKind,
) {
    add_client_fields(value, requested_backend, effective_backend);
}

fn decorate_selection(
    value: &mut Value,
    requested_backend: BackendKind,
    effective_backend: BackendKind,
) {
    add_execution_client(value);
    let plan = value
        .get_mut("plan")
        .expect("selection receipt contains a plan");
    decorate_plan(plan, requested_backend, effective_backend);
}

fn decorate_snapshot(
    value: &mut Value,
    requested_backend: BackendKind,
    effective_backend: BackendKind,
) {
    add_execution_client(value);
    let candidates = value
        .get_mut("candidates")
        .and_then(Value::as_array_mut)
        .expect("session snapshot contains candidates");
    for candidate in candidates {
        let plan = candidate
            .get_mut("plan")
            .expect("candidate snapshot contains a plan");
        decorate_plan(plan, requested_backend, effective_backend);
    }
}

#[cfg(target_arch = "wasm32")]
fn to_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = RankAdaptationSession)]
pub struct WasmRankAdaptationSession {
    inner: RankAdaptationSession,
    requested_backend: BackendKind,
    effective_backend: BackendKind,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_class = RankAdaptationSession)]
impl WasmRankAdaptationSession {
    #[wasm_bindgen(constructor)]
    pub fn new(request: &JsValue) -> Result<WasmRankAdaptationSession, JsValue> {
        let value = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
        let resolved = session_from_value(value).map_err(js_error)?;
        Ok(Self {
            inner: resolved.session,
            requested_backend: resolved.requested_backend,
            effective_backend: resolved.effective_backend,
        })
    }

    pub fn choose(&mut self) -> Result<JsValue, JsValue> {
        let mut next = self.inner.clone();
        let selection = next.try_choose().map_err(js_error)?;
        let mut value = serde_json::to_value(selection.receipt()).map_err(js_error)?;
        decorate_selection(&mut value, self.requested_backend, self.effective_backend);
        let value = to_js(&value)?;
        self.inner = next;
        Ok(value)
    }

    pub fn observe(
        &mut self,
        selection_id: f64,
        elapsed_ms: f64,
        correctness_passed: bool,
    ) -> Result<JsValue, JsValue> {
        let selection_id = safe_u64(selection_id, "selection_id").map_err(js_error)?;
        let mut next = self.inner.clone();
        let receipt = next
            .try_observe(selection_id, elapsed_ms, correctness_passed)
            .map_err(js_error)?;
        let mut value = serde_json::to_value(receipt).map_err(js_error)?;
        add_execution_client(&mut value);
        let value = to_js(&value)?;
        self.inner = next;
        Ok(value)
    }

    pub fn abandon(&mut self, selection_id: f64) -> Result<JsValue, JsValue> {
        let selection_id = safe_u64(selection_id, "selection_id").map_err(js_error)?;
        let mut next = self.inner.clone();
        let receipt = next.try_abandon(selection_id).map_err(js_error)?;
        let mut value = serde_json::to_value(receipt).map_err(js_error)?;
        add_execution_client(&mut value);
        let value = to_js(&value)?;
        self.inner = next;
        Ok(value)
    }

    pub fn snapshot(&self) -> Result<JsValue, JsValue> {
        let mut value = serde_json::to_value(self.inner.snapshot()).map_err(js_error)?;
        decorate_snapshot(&mut value, self.requested_backend, self.effective_backend);
        to_js(&value)
    }

    #[wasm_bindgen(js_name = pendingSelectionId)]
    pub fn pending_selection_id(&self) -> Option<f64> {
        self.inner.pending_selection_id().map(|value| value as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> Value {
        json!({
            "rank_plan": {
                "kind": "topk",
                "rows": 2,
                "cols": 256,
                "k": 8,
                "backend": "wgpu"
            },
            "scripts": ["u2: false;", "u2: true;"],
            "policy": "ucb",
            "seed": 17
        })
    }

    #[test]
    fn wasm_request_uses_the_rust_owned_session() {
        let mut resolved = session_from_value(request()).expect("session");
        let selection = resolved.session.try_choose().expect("selection");
        assert_eq!(selection.receipt().selection_id, 1);
        let receipt = resolved
            .session
            .try_observe(1, 4.0, true)
            .expect("observation");
        assert_eq!(receipt.reward, Some(0.2));
        assert_eq!(resolved.session.pending_selection_id(), None);
    }

    #[test]
    fn wasm_safe_integer_boundary_is_explicit() {
        assert_eq!(safe_u64(17.0, "seed"), Ok(17));
        assert!(safe_u64(f64::NAN, "seed").is_err());
        assert!(safe_u64(9_007_199_254_740_992.0, "seed").is_err());
    }

    #[test]
    fn typescript_declares_the_stateful_rank_contract() {
        let declarations = include_str!("../types/spiraltorch-wasm.d.ts");
        assert!(declarations.contains("export class RankAdaptationSession"));
        assert!(declarations.contains("RankAdaptationSelectionReceipt"));
        assert!(declarations.contains("selection_attempts: number"));
        assert!(declarations.contains("rng_stream_seed: string | null"));
        assert!(declarations.contains("execution_signature: string"));
        assert!(declarations.contains("pendingSelectionId(): number | undefined"));
    }
}
