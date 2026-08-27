use js_sys::JSON;
use std::fmt;
use wasm_bindgen::JsValue;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(inline_js = r#"
export function spiraltorchPreflightJsonCompatible(
    value,
    maximumBytes,
    maximumNodes,
    maximumDepth,
    context,
) {
    let bytes = 0;
    let nodes = 0;

    function fail(message) {
        throw new Error(message);
    }

    function charge(nextBytes, nextNodes) {
        bytes += nextBytes;
        nodes += nextNodes;
        if (bytes > maximumBytes) {
            fail(`${context} exceeds WASM ingress budget of ${maximumBytes} bytes`);
        }
        if (nodes > maximumNodes) {
            fail(`${context} exceeds WASM ingress budget of ${maximumNodes} JSON nodes`);
        }
    }

    function visit(current, depth) {
        if (depth > maximumDepth) {
            fail(`${context} exceeds WASM ingress depth limit ${maximumDepth}`);
        }
        charge(0, 1);

        if (current === null) {
            charge(4, 0);
            return;
        }
        const kind = typeof current;
        if (kind === "boolean") {
            charge(5, 0);
            return;
        }
        if (kind === "number") {
            if (!Number.isFinite(current)) {
                fail(`${context} must not contain non-finite numbers`);
            }
            charge(24, 0);
            return;
        }
        if (kind === "string") {
            charge(current.length * 6 + 2, 0);
            return;
        }
        if (Array.isArray(current)) {
            if (current.length > maximumNodes - nodes) {
                fail(`${context} exceeds WASM ingress budget of ${maximumNodes} JSON nodes`);
            }
            charge(Math.max(current.length - 1, 0) + 2, 0);
            for (let index = 0; index < current.length; index += 1) {
                visit(current[index], depth + 1);
            }
            return;
        }
        if (kind === "object") {
            charge(2, 0);
            let first = true;
            for (const key in current) {
                if (!Object.prototype.hasOwnProperty.call(current, key)) {
                    continue;
                }
                charge((first ? 0 : 1) + key.length * 6 + 3, 0);
                first = false;
                visit(current[key], depth + 1);
            }
            return;
        }
        fail(`${context} must contain only JSON-compatible values`);
    }

    visit(value, 0);
}
"#)]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(
        catch,
        js_name = spiraltorchPreflightJsonCompatible
    )]
    fn preflight_json_compatible_js_value_raw(
        value: &JsValue,
        maximum_bytes: f64,
        maximum_nodes: f64,
        maximum_depth: u32,
        context: &str,
    ) -> Result<(), JsValue>;
}

pub(crate) fn js_error(err: impl fmt::Display) -> JsValue {
    JsValue::from_str(&err.to_string())
}

pub(crate) fn js_value_to_string(value: &JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

pub(crate) fn json_to_js_value(json: &str) -> Result<JsValue, JsValue> {
    JSON::parse(json).map_err(|err| js_error(js_value_to_string(&err)))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn preflight_json_compatible_js_value(
    value: &JsValue,
    maximum_bytes: u64,
    maximum_nodes: u64,
    maximum_depth: u32,
    context: &str,
) -> Result<(), JsValue> {
    preflight_json_compatible_js_value_raw(
        value,
        maximum_bytes as f64,
        maximum_nodes as f64,
        maximum_depth,
        context,
    )
}
