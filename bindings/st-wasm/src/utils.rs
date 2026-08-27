#[cfg(target_arch = "wasm32")]
use js_sys::{JsString, JSON};
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};
use std::fmt;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;

struct JsonParseBudget<'a> {
    nodes: u64,
    maximum_nodes: u64,
    maximum_depth: u32,
    context: &'a str,
}

impl JsonParseBudget<'_> {
    fn charge<E: serde::de::Error>(&mut self, depth: u32) -> Result<(), E> {
        if depth > self.maximum_depth {
            return Err(E::custom(format!(
                "{} exceeds WASM ingress depth limit {}",
                self.context, self.maximum_depth
            )));
        }
        self.nodes = self.nodes.saturating_add(1);
        if self.nodes > self.maximum_nodes {
            return Err(E::custom(format!(
                "{} exceeds WASM ingress budget of {} JSON nodes",
                self.context, self.maximum_nodes
            )));
        }
        Ok(())
    }
}

struct BoundedJsonValueSeed<'budget, 'context> {
    budget: &'budget mut JsonParseBudget<'context>,
    depth: u32,
}

impl<'de> DeserializeSeed<'de> for BoundedJsonValueSeed<'_, '_> {
    type Value = JsonValue;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        self.budget.charge::<D::Error>(self.depth)?;
        deserializer.deserialize_any(BoundedJsonValueVisitor {
            budget: self.budget,
            depth: self.depth,
        })
    }
}

struct BoundedJsonValueVisitor<'budget, 'context> {
    budget: &'budget mut JsonParseBudget<'context>,
    depth: u32,
}

impl<'de> Visitor<'de> for BoundedJsonValueVisitor<'_, '_> {
    type Value = JsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value within the configured ingress budget")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(JsonValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(JsonValue::Number(JsonNumber::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(JsonValue::Number(JsonNumber::from(value)))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        JsonNumber::from_f64(value)
            .map(JsonValue::Number)
            .ok_or_else(|| E::custom("JSON numbers must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(JsonValue::String(value.to_owned()))
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E> {
        Ok(JsonValue::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(JsonValue::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(JsonValue::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(JsonValue::Null)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(BoundedJsonValueSeed {
            budget: self.budget,
            depth: self.depth.saturating_add(1),
        })? {
            values.push(value);
        }
        Ok(JsonValue::Array(values))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = JsonMap::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(<A::Error as serde::de::Error>::custom(
                    "duplicate JSON object key",
                ));
            }
            let value = object.next_value_seed(BoundedJsonValueSeed {
                budget: self.budget,
                depth: self.depth.saturating_add(1),
            })?;
            values.insert(key, value);
        }
        Ok(JsonValue::Object(values))
    }
}

pub(crate) fn bounded_json_value_from_str(
    input: &str,
    maximum_bytes: u64,
    maximum_nodes: u64,
    maximum_depth: u32,
    context: &str,
) -> Result<JsonValue, String> {
    if input.len() as u64 > maximum_bytes {
        return Err(format!(
            "{context} exceeds WASM ingress budget of {maximum_bytes} bytes"
        ));
    }
    let mut budget = JsonParseBudget {
        nodes: 0,
        maximum_nodes,
        maximum_depth,
        context,
    };
    let mut deserializer = serde_json::Deserializer::from_str(input);
    let value = BoundedJsonValueSeed {
        budget: &mut budget,
        depth: 0,
    }
    .deserialize(&mut deserializer)
    .map_err(|error| error.to_string())?;
    deserializer.end().map_err(|error| error.to_string())?;
    Ok(value)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(inline_js = r#"
function utf8StringBytes(value) {
    let bytes = 0;
    for (let index = 0; index < value.length; index += 1) {
        const unit = value.charCodeAt(index);
        if (unit <= 0x7f) {
            bytes += 1;
        } else if (unit <= 0x7ff) {
            bytes += 2;
        } else if (unit >= 0xd800 && unit <= 0xdbff
            && index + 1 < value.length
            && value.charCodeAt(index + 1) >= 0xdc00
            && value.charCodeAt(index + 1) <= 0xdfff) {
            bytes += 4;
            index += 1;
        } else {
            bytes += 3;
        }
    }
    return bytes;
}

function jsonStringBytes(value) {
    let bytes = 0;
    for (let index = 0; index < value.length; index += 1) {
        const unit = value.charCodeAt(index);
        if (unit === 0x22 || unit === 0x5c
            || unit === 0x08 || unit === 0x09 || unit === 0x0a
            || unit === 0x0c || unit === 0x0d) {
            bytes += 2;
        } else if (unit <= 0x1f) {
            bytes += 6;
        } else if (unit <= 0x7f) {
            bytes += 1;
        } else if (unit <= 0x7ff) {
            bytes += 2;
        } else if (unit >= 0xd800 && unit <= 0xdbff
            && index + 1 < value.length
            && value.charCodeAt(index + 1) >= 0xdc00
            && value.charCodeAt(index + 1) <= 0xdfff) {
            bytes += 4;
            index += 1;
        } else if (unit >= 0xd800 && unit <= 0xdfff) {
            bytes += 6;
        } else {
            bytes += 3;
        }
    }
    return bytes;
}

export function spiraltorchPreflightJsonString(value, maximumBytes, context) {
    if (typeof value !== "string") {
        throw new Error(`${context} must be a JSON string`);
    }
    if (utf8StringBytes(value) > maximumBytes) {
        throw new Error(`${context} exceeds WASM ingress budget of ${maximumBytes} bytes`);
    }
}

export function spiraltorchSnapshotJsonCompatible(
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
            return null;
        }
        const kind = typeof current;
        if (kind === "boolean") {
            charge(5, 0);
            return current;
        }
        if (kind === "number") {
            if (!Number.isFinite(current)) {
                fail(`${context} must not contain non-finite numbers`);
            }
            charge(24, 0);
            return current;
        }
        if (kind === "string") {
            charge(jsonStringBytes(current) + 2, 0);
            return current;
        }
        if (Array.isArray(current)) {
            const length = current.length;
            if (length > maximumNodes - nodes) {
                fail(`${context} exceeds WASM ingress budget of ${maximumNodes} JSON nodes`);
            }
            charge(Math.max(length - 1, 0) + 2, 0);
            const snapshot = new Array(length);
            for (let index = 0; index < length; index += 1) {
                snapshot[index] = visit(current[index], depth + 1);
            }
            return snapshot;
        }
        if (kind === "object") {
            charge(2, 0);
            const snapshot = Object.create(null);
            let first = true;
            for (const key in current) {
                if (!Object.prototype.hasOwnProperty.call(current, key)) {
                    continue;
                }
                charge((first ? 0 : 1) + jsonStringBytes(key) + 3, 0);
                first = false;
                snapshot[key] = visit(current[key], depth + 1);
            }
            return snapshot;
        }
        fail(`${context} must contain only JSON-compatible values`);
    }

    return visit(value, 0);
}
"#)]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(catch, js_name = spiraltorchPreflightJsonString)]
    fn preflight_json_string_raw(
        value: &JsString,
        maximum_bytes: f64,
        context: &str,
    ) -> Result<(), JsValue>;

    #[wasm_bindgen::prelude::wasm_bindgen(
        catch,
        js_name = spiraltorchSnapshotJsonCompatible
    )]
    fn snapshot_json_compatible_js_value_raw(
        value: &JsValue,
        maximum_bytes: f64,
        maximum_nodes: f64,
        maximum_depth: u32,
        context: &str,
    ) -> Result<JsValue, JsValue>;
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn js_error(err: impl fmt::Display) -> JsValue {
    JsValue::from_str(&err.to_string())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn js_value_to_string(value: &JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn json_to_js_value(json: &str) -> Result<JsValue, JsValue> {
    JSON::parse(json).map_err(|err| js_error(js_value_to_string(&err)))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn bounded_json_string_from_js(
    value: &JsString,
    maximum_bytes: u64,
    context: &str,
) -> Result<String, JsValue> {
    preflight_json_string_raw(value, maximum_bytes as f64, context)?;
    value
        .as_string()
        .ok_or_else(|| js_error(format!("{context} must be a JSON string")))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn snapshot_json_compatible_js_value(
    value: &JsValue,
    maximum_bytes: u64,
    maximum_nodes: u64,
    maximum_depth: u32,
    context: &str,
) -> Result<JsValue, JsValue> {
    snapshot_json_compatible_js_value_raw(
        value,
        maximum_bytes as f64,
        maximum_nodes as f64,
        maximum_depth,
        context,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bounded_json_parser_charges_nodes_and_depth_during_materialization() {
        assert_eq!(
            bounded_json_value_from_str(
                r#"{"rows":[1,true,null,"灯"]}"#,
                1_024,
                6,
                2,
                "test payload",
            )
            .expect("bounded JSON"),
            json!({"rows": [1, true, null, "灯"]})
        );

        let node_error = bounded_json_value_from_str("[0,0,0]", 1_024, 3, 4, "test payload")
            .expect_err("root plus three children exceed three nodes");
        assert!(node_error.contains("3 JSON nodes"));

        let depth_error =
            bounded_json_value_from_str(r#"{"outer":{"inner":0}}"#, 1_024, 8, 1, "test payload")
                .expect_err("nested value exceeds depth one");
        assert!(depth_error.contains("depth limit 1"));

        assert_eq!(
            bounded_json_value_from_str("{}", 1, 8, 1, "test payload")
                .expect_err("byte limit runs before parsing"),
            "test payload exceeds WASM ingress budget of 1 bytes"
        );
        assert!(
            bounded_json_value_from_str(r#"{"value":1,"value":2}"#, 1_024, 8, 1, "test")
                .expect_err("duplicate keys fail closed")
                .contains("duplicate JSON object key")
        );
    }
}
