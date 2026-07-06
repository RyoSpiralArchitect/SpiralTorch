use serde_json::{json, Map, Value};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;
#[cfg(target_arch = "wasm32")]
use serde::Serialize;

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

fn finite_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64().filter(|value| value.is_finite()),
        _ => None,
    }
}

fn path<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    let mut cursor = value;
    for key in keys {
        cursor = cursor.get(*key)?;
    }
    Some(cursor)
}

fn number_at(value: &Value, keys: &[&str]) -> Option<f64> {
    path(value, keys).and_then(finite_f64)
}

fn bool_at(value: &Value, keys: &[&str]) -> bool {
    path(value, keys).and_then(Value::as_bool).unwrap_or(false)
}

fn string_at<'a>(value: &'a Value, keys: &[&str]) -> &'a str {
    path(value, keys).and_then(Value::as_str).unwrap_or("")
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn family(report: &Value) -> &'static str {
    let needle = format!(
        "{} {}",
        string_at(report, &["schema"]),
        string_at(report, &["kind"])
    )
    .to_lowercase();
    if needle.contains("mellin") {
        "mellin"
    } else if needle.contains("canvas") || needle.contains("hypertrain") {
        "canvas"
    } else {
        "unknown"
    }
}

fn report_label(index: usize, report: &Value) -> String {
    for keys in [
        &["label"][..],
        &["artifact_path"][..],
        &["artifactPath"][..],
        &["kind"][..],
    ] {
        let raw = string_at(report, keys);
        if !raw.is_empty() {
            return raw
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(raw)
                .trim_end_matches(".json")
                .to_string();
        }
    }
    format!("run_{index}")
}

fn loss_trace_stats(trace: Option<&Value>) -> Option<Value> {
    let rows = trace?.as_array()?;
    let values: Vec<f64> = rows
        .iter()
        .filter_map(|row| number_at(row, &["loss"]))
        .collect();
    if values.is_empty() {
        return None;
    }
    let first = values[0];
    let last = *values.last().unwrap_or(&first);
    let min = values
        .iter()
        .fold(f64::INFINITY, |acc, value| acc.min(*value));
    let max = values
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let absolute_improvement = first - last;
    let relative_improvement = if first == 0.0 {
        0.0
    } else {
        absolute_improvement / first.abs()
    };
    Some(json!({
        "count": values.len(),
        "first": first,
        "last": last,
        "min": min,
        "max": max,
        "mean": mean,
        "delta": last - first,
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "improved": last <= first,
    }))
}

fn summary_loss(report: &Value, family: &str) -> Option<f64> {
    if family == "mellin" {
        number_at(report, &["training", "finalLoss"])
            .or_else(|| {
                loss_trace_stats(path(report, &["training", "trace"]))
                    .and_then(|trace| number_at(&trace, &["last"]))
            })
            .or_else(|| number_at(report, &["inferenceProbe", "absDiffStats", "rms"]))
    } else if family == "canvas" {
        number_at(report, &["metrics", "last", "loss"])
            .or_else(|| number_at(report, &["metrics", "lossStats", "mean"]))
    } else {
        None
    }
}

fn summary_stability(report: &Value, family: &str, loss: Option<f64>) -> Option<f64> {
    if family == "canvas" {
        if let Some(value) = number_at(report, &["currentFrame", "desire", "stability"]) {
            return Some(value);
        }
    }
    loss.map(|value| 1.0 / (1.0 + value.max(0.0)))
}

fn runtime_readiness(report: &Value, family: &str) -> Value {
    let wasm_ready = bool_at(report, &["runtime", "wasm"]);
    let webgpu_available = bool_at(report, &["runtime", "webgpuAvailable"]);
    let webgpu_init_failed = bool_at(report, &["runtime", "webgpuInitFailed"]);
    let component_ready = [
        bool_at(report, &["runtime", "webgpuDeviceReady"]),
        bool_at(report, &["runtime", "webgpuTrainerReady"]),
        bool_at(report, &["runtime", "webgpuFftReady"]),
        bool_at(report, &["runtime", "webgpuTrailReady"]),
    ];
    let ready_count = component_ready.iter().filter(|ready| **ready).count() as u32;

    let mut score = 0.0;
    if wasm_ready {
        score += 0.35;
    }
    if webgpu_available {
        score += 0.20;
    }
    if ready_count > 0 {
        score += 0.35;
    } else if family == "mellin" && webgpu_available {
        score += 0.15;
    }
    if !webgpu_init_failed {
        score += 0.10;
    }

    let status = if ready_count > 0 {
        "webgpu_ready"
    } else if webgpu_available {
        "webgpu_available"
    } else if wasm_ready {
        "wasm_only"
    } else {
        "missing_runtime"
    };

    json!({
        "status": status,
        "score": clamp01(score),
        "wasm": wasm_ready,
        "webgpu_available": webgpu_available,
        "webgpu_device_ready": bool_at(report, &["runtime", "webgpuDeviceReady"]),
        "webgpu_component_ready_count": ready_count,
        "webgpu_init_failed": webgpu_init_failed,
    })
}

fn learning_progress(report: &Value, family: &str, loss: Option<f64>) -> Value {
    let trace = if family == "mellin" {
        loss_trace_stats(path(report, &["training", "trace"]))
    } else if family == "canvas" {
        loss_trace_stats(path(report, &["metrics", "tail"]))
    } else {
        None
    };

    let mut source = "missing";
    let mut first_loss = None;
    let mut last_loss = loss;
    let mut absolute_improvement = None;
    let mut relative_improvement = None;
    let mut improved = None;

    if let Some(trace) = trace {
        source = "trace";
        first_loss = number_at(&trace, &["first"]);
        if let Some(value) = number_at(&trace, &["last"]) {
            last_loss = Some(value);
        }
        absolute_improvement = number_at(&trace, &["absolute_improvement"]);
        relative_improvement = number_at(&trace, &["relative_improvement"]);
        improved = path(&trace, &["improved"]).and_then(Value::as_bool);
    } else if family == "canvas" {
        let stats = path(report, &["metrics", "lossStats"]);
        if let Some(stats) = stats {
            source = "loss_stats";
            first_loss = number_at(stats, &["max"]);
            last_loss = loss;
            if let (Some(first), Some(last)) = (first_loss, last_loss) {
                let absolute = first - last;
                absolute_improvement = Some(absolute);
                relative_improvement = Some(if first == 0.0 {
                    0.0
                } else {
                    absolute / first.abs()
                });
                improved = Some(absolute >= 0.0);
            }
        }
    }

    let progress_score = relative_improvement
        .map(clamp01)
        .or_else(|| improved.map(|value| if value { 1.0 } else { 0.0 }))
        .unwrap_or(0.5);

    json!({
        "source": source,
        "loss": loss,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "improved": improved,
        "progress_score": progress_score,
    })
}

fn audit_from_report(report: &Value) -> Value {
    let family = family(report);
    let loss = summary_loss(report, family);
    let stability = summary_stability(report, family, loss);
    let runtime = runtime_readiness(report, family);
    let progress = learning_progress(report, family, loss);
    let runtime_score = number_at(&runtime, &["score"]).unwrap_or(0.0);
    let progress_score = number_at(&progress, &["progress_score"]).unwrap_or(0.5);
    let loss_score = loss
        .map(|value| 1.0 / (1.0 + value.max(0.0)))
        .unwrap_or(0.5);
    let stability_score = stability.map(clamp01).unwrap_or(0.5);
    let readiness_score = clamp01(
        0.35 * runtime_score + 0.25 * stability_score + 0.25 * loss_score + 0.15 * progress_score,
    );

    let mut risk_flags = Vec::new();
    if !bool_at(&runtime, &["wasm"]) {
        risk_flags.push("wasm_runtime_missing");
    }
    if !bool_at(&runtime, &["webgpu_available"]) {
        risk_flags.push("webgpu_unavailable");
    }
    if bool_at(&runtime, &["webgpu_init_failed"]) {
        risk_flags.push("webgpu_init_failed");
    }
    if bool_at(&runtime, &["webgpu_available"])
        && number_at(&runtime, &["webgpu_component_ready_count"]).unwrap_or(0.0) == 0.0
        && family == "canvas"
    {
        risk_flags.push("canvas_webgpu_components_not_ready");
    }
    if loss.is_none() {
        risk_flags.push("loss_not_observed");
    }
    if path(&progress, &["improved"]).and_then(Value::as_bool) == Some(false) {
        risk_flags.push("loss_not_improved");
    }
    if bool_at(report, &["metrics", "truncated"]) {
        risk_flags.push("metrics_history_truncated");
    }

    let status = if readiness_score >= 0.78 && !risk_flags.contains(&"loss_not_improved") {
        "ready"
    } else if readiness_score >= 0.58 {
        "usable"
    } else {
        "needs_attention"
    };

    let mut recommendations = Vec::new();
    if !bool_at(&runtime, &["wasm"]) {
        recommendations.push("rerun the browser demo after the WASM package loads");
    }
    if bool_at(&runtime, &["webgpu_init_failed"]) {
        recommendations.push("inspect the browser WebGPU initialization failure");
    } else if !bool_at(&runtime, &["webgpu_available"]) {
        recommendations.push("capture the report in a browser with WebGPU available");
    }
    if path(&progress, &["improved"]).and_then(Value::as_bool) == Some(false) {
        recommendations.push("increase training steps or reduce the learning rate");
    }
    if bool_at(report, &["metrics", "truncated"]) {
        recommendations.push("download a fresh report before the metrics tail truncates");
    }
    if status == "ready" {
        recommendations.push("promote this report as a Z-space runtime context candidate");
    } else if status == "usable" {
        recommendations.push("compare against nearby runs before promotion");
    }

    json!({
        "kind": "spiraltorch.wasm_report_audit",
        "schema": string_at(report, &["schema"]),
        "report_kind": string_at(report, &["kind"]),
        "family": family,
        "artifact_path": path(report, &["artifact_path"])
            .or_else(|| path(report, &["artifactPath"]))
            .cloned()
            .unwrap_or(Value::Null),
        "status": status,
        "readiness_score": readiness_score,
        "runtime": runtime,
        "learning": progress,
        "loss_score": loss_score,
        "stability_score": stability_score,
        "risk_flags": risk_flags,
        "recommendations": recommendations,
    })
}

pub fn audit_wasm_report_value(report: &Value) -> Value {
    audit_from_report(report)
}

fn report_items(reports: &Value) -> Vec<(String, Value)> {
    if reports.get("schema").is_some() || reports.get("kind").is_some() {
        return vec![(report_label(0, reports), reports.clone())];
    }
    if let Some(rows) = reports.as_array() {
        return rows
            .iter()
            .enumerate()
            .map(|(index, report)| (report_label(index, report), report.clone()))
            .collect();
    }
    if let Some(map) = reports.as_object() {
        return map
            .iter()
            .map(|(label, report)| (label.clone(), report.clone()))
            .collect();
    }
    Vec::new()
}

pub fn compare_wasm_reports_value(reports: &Value) -> Value {
    let items = report_items(reports);
    let mut family_counts = Map::new();
    let mut rows = Vec::new();
    for (label, report) in items {
        let family = family(&report).to_string();
        let loss = summary_loss(&report, &family);
        let stability = summary_stability(&report, &family, loss);
        let audit = audit_wasm_report_value(&report);
        let count = family_counts
            .get(&family)
            .and_then(Value::as_u64)
            .unwrap_or(0)
            + 1;
        family_counts.insert(family.clone(), json!(count));
        rows.push(json!({
            "label": label,
            "schema": string_at(&audit, &["schema"]),
            "kind": string_at(&audit, &["report_kind"]),
            "family": family,
            "loss": loss,
            "stability": stability,
            "readiness_score": path(&audit, &["readiness_score"]).cloned().unwrap_or(Value::Null),
            "audit_status": string_at(&audit, &["status"]),
            "risk_flags": path(&audit, &["risk_flags"]).cloned().unwrap_or_else(|| json!([])),
            "audit": audit,
        }));
    }

    let best_loss = best_by(&rows, false, "loss");
    let best_stability = best_by(&rows, true, "stability");
    let best_readiness = best_by(&rows, true, "readiness_score");

    json!({
        "kind": "spiraltorch.wasm_report_comparison",
        "count": rows.len(),
        "families": family_counts,
        "best_loss": best_loss,
        "best_stability": best_stability,
        "best_readiness": best_readiness,
        "reports": rows,
    })
}

fn best_by(rows: &[Value], higher_is_better: bool, key: &str) -> Value {
    let mut best: Option<(&Value, f64)> = None;
    for row in rows {
        let Some(score) = number_at(row, &[key]) else {
            continue;
        };
        let replace = match best {
            None => true,
            Some((_, current)) if higher_is_better => score > current,
            Some((_, current)) => score < current,
        };
        if replace {
            best = Some((row, score));
        }
    }
    best.map(|(row, _)| row.clone()).unwrap_or(Value::Null)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = auditWasmReportJson)]
pub fn audit_wasm_report_json(report_json: &str) -> Result<String, JsValue> {
    let report = serde_json::from_str::<Value>(report_json).map_err(js_error)?;
    serde_json::to_string(&audit_wasm_report_value(&report)).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = auditWasmReportObject)]
pub fn audit_wasm_report_object(report: &JsValue) -> Result<JsValue, JsValue> {
    let report = serde_wasm_bindgen::from_value::<Value>(report.clone()).map_err(js_error)?;
    to_json_compatible_js(&audit_wasm_report_value(&report))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = compareWasmReportsJson)]
pub fn compare_wasm_reports_json(reports_json: &str) -> Result<String, JsValue> {
    let reports = serde_json::from_str::<Value>(reports_json).map_err(js_error)?;
    serde_json::to_string(&compare_wasm_reports_value(&reports)).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = compareWasmReportsObject)]
pub fn compare_wasm_reports_object(reports: &JsValue) -> Result<JsValue, JsValue> {
    let reports = serde_wasm_bindgen::from_value::<Value>(reports.clone()).map_err(js_error)?;
    to_json_compatible_js(&compare_wasm_reports_value(&reports))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canvas_report(last_loss: f64) -> Value {
        json!({
            "schema": "spiraltorch.wasm.canvas_hypertrain_report.v1",
            "kind": "canvas-hypertrain-training",
            "runtime": {
                "wasm": true,
                "webgpuAvailable": true,
                "webgpuDeviceReady": true,
                "webgpuTrainerReady": true
            },
            "currentFrame": {
                "desire": {"stability": 0.82}
            },
            "metrics": {
                "last": {"loss": last_loss},
                "lossStats": {"max": 0.2, "mean": 0.11},
                "tail": [
                    {"step": 0, "loss": 0.2},
                    {"step": 1, "loss": last_loss}
                ]
            }
        })
    }

    fn mellin_report(final_loss: f64) -> Value {
        json!({
            "schema": "spiraltorch.wasm.mellin_learning_report.v1",
            "kind": "mellin-log-grid-training",
            "runtime": {"wasm": true, "webgpuAvailable": true},
            "training": {
                "finalLoss": final_loss,
                "trace": [
                    {"step": 1, "loss": 0.5},
                    {"step": 2, "loss": final_loss}
                ]
            }
        })
    }

    #[test]
    fn audits_ready_canvas_report() {
        let audit = audit_wasm_report_value(&canvas_report(0.05));

        assert_eq!(audit["family"], "canvas");
        assert_eq!(audit["status"], "ready");
        assert_eq!(audit["runtime"]["status"], "webgpu_ready");
        assert_eq!(audit["learning"]["source"], "trace");
        assert!(audit["readiness_score"].as_f64().unwrap() > 0.8);
        assert_eq!(audit["risk_flags"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn compares_reports_by_readiness() {
        let comparison = compare_wasm_reports_value(&json!({
            "weak": mellin_report(0.2),
            "strong": canvas_report(0.02)
        }));

        assert_eq!(comparison["count"], 2);
        assert_eq!(comparison["families"]["mellin"], 1);
        assert_eq!(comparison["families"]["canvas"], 1);
        assert_eq!(comparison["best_readiness"]["label"], "strong");
        assert_eq!(comparison["best_loss"]["label"], "strong");
        let strong_row = comparison["reports"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["label"] == "strong")
            .unwrap();
        assert_eq!(strong_row["stability"], 0.82);
    }

    #[test]
    fn flags_missing_webgpu_and_bad_learning() {
        let mut report = canvas_report(0.3);
        report["runtime"]["webgpuAvailable"] = json!(false);
        report["runtime"]["webgpuDeviceReady"] = json!(false);
        report["runtime"]["webgpuTrainerReady"] = json!(false);

        let audit = audit_wasm_report_value(&report);
        let flags = audit["risk_flags"].as_array().unwrap();
        assert_eq!(audit["status"], "needs_attention");
        assert!(flags.contains(&json!("webgpu_unavailable")));
        assert!(flags.contains(&json!("loss_not_improved")));
    }
}
