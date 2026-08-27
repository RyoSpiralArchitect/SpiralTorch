// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::{Number, Value};

const MAX_EXACT_JSON_INTEGER: u64 = 9_007_199_254_740_991;

/// Compares canonical contract JSON while accepting JavaScript's integral float spelling.
///
/// Object keys, array order, scalar types, and numeric values must still match exactly.
/// The only relaxed case is an exactly representable integer such as `1` standing in for
/// the canonical floating-point spelling `1.0` after a JSON.stringify round trip.
pub(crate) fn values_equivalent(actual: &Value, canonical: &Value) -> bool {
    match (actual, canonical) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(actual), Value::Bool(canonical)) => actual == canonical,
        (Value::Number(actual), Value::Number(canonical)) => numbers_equivalent(actual, canonical),
        (Value::String(actual), Value::String(canonical)) => actual == canonical,
        (Value::Array(actual), Value::Array(canonical)) => {
            actual.len() == canonical.len()
                && actual
                    .iter()
                    .zip(canonical)
                    .all(|(actual, canonical)| values_equivalent(actual, canonical))
        }
        (Value::Object(actual), Value::Object(canonical)) => {
            actual.len() == canonical.len()
                && actual.iter().all(|(key, actual)| {
                    canonical
                        .get(key)
                        .is_some_and(|canonical| values_equivalent(actual, canonical))
                })
        }
        _ => false,
    }
}

fn numbers_equivalent(actual: &Number, canonical: &Number) -> bool {
    actual == canonical
        || exact_cross_client_f64(actual)
            .zip(exact_cross_client_f64(canonical))
            .is_some_and(|(actual, canonical)| actual == canonical)
}

fn exact_cross_client_f64(value: &Number) -> Option<f64> {
    if value.is_f64() {
        return value.as_f64().filter(|value| value.is_finite());
    }
    if let Some(value) = value.as_u64() {
        return (value <= MAX_EXACT_JSON_INTEGER).then_some(value as f64);
    }
    value
        .as_i64()
        .and_then(|value| (value.unsigned_abs() <= MAX_EXACT_JSON_INTEGER).then_some(value as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn accepts_javascript_integral_float_spelling_recursively() {
        let actual = json!({"ratio": 1, "nested": [0, 0.5]});
        let canonical = json!({"ratio": 1.0, "nested": [0.0, 0.5]});

        assert!(values_equivalent(&actual, &canonical));
    }

    #[test]
    fn rejects_changed_structure_and_numeric_values() {
        assert!(!values_equivalent(
            &json!({"ratio": 0}),
            &json!({"ratio": 1.0})
        ));
        assert!(!values_equivalent(
            &json!({"ratio": 1, "extra": true}),
            &json!({"ratio": 1.0})
        ));
        assert!(!values_equivalent(&json!([1, 2]), &json!([2, 1])));
    }

    #[test]
    fn never_aliases_unsafe_integers_through_f64_rounding() {
        assert!(!values_equivalent(
            &json!(9_007_199_254_740_993u64),
            &json!(9_007_199_254_740_992.0f64)
        ));
    }
}
