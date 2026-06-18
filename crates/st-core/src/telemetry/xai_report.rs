use serde::{Deserialize, Serialize};
use serde_json::Value;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AttributionMetadata {
    pub algorithm: String,
    pub layer: Option<String>,
    pub target: Option<String>,
    pub steps: Option<usize>,
    pub extras: BTreeMap<String, Value>,
}

impl AttributionMetadata {
    pub fn for_algorithm(name: impl Into<String>) -> Self {
        Self {
            algorithm: name.into(),
            layer: None,
            target: None,
            steps: None,
            extras: BTreeMap::new(),
        }
    }

    pub fn insert_extra(&mut self, key: impl Into<String>, value: Value) {
        self.extras.insert(key.into(), value);
    }

    pub fn insert_extra_number(&mut self, key: impl Into<String>, value: f64) {
        self.insert_extra(key, Value::from(value));
    }

    pub fn insert_extra_flag(&mut self, key: impl Into<String>, value: bool) {
        self.insert_extra(key, Value::from(value));
    }

    pub fn insert_extra_text(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.insert_extra(key, Value::from(value.into()));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionReport {
    pub metadata: AttributionMetadata,
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

impl AttributionReport {
    pub fn new(metadata: AttributionMetadata, rows: usize, cols: usize, values: Vec<f32>) -> Self {
        assert_eq!(values.len(), rows * cols, "values do not match shape");
        emit_attribution_report_meta(&metadata, rows, cols, &values);
        Self {
            metadata,
            rows,
            cols,
            values,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    pub fn from_json_str(data: &str) -> serde_json::Result<Self> {
        serde_json::from_str(data)
    }
}

fn emit_attribution_report_meta(
    metadata: &AttributionMetadata,
    rows: usize,
    cols: usize,
    values: &[f32],
) {
    let mut finite = 0usize;
    let mut non_finite = 0usize;
    let mut positive = 0usize;
    let mut negative = 0usize;
    let mut zero = 0usize;
    let mut total = 0.0f64;
    let mut abs_total = 0.0f64;
    let mut max_abs = 0.0f32;
    for &value in values {
        if value.is_finite() {
            finite += 1;
            total += value as f64;
            abs_total += value.abs() as f64;
            max_abs = max_abs.max(value.abs());
            if value > 0.0 {
                positive += 1;
            } else if value < 0.0 {
                negative += 1;
            } else {
                zero += 1;
            }
        } else {
            non_finite += 1;
        }
    }
    emit_tensor_op("attribution_report_new", &[rows, cols], &[values.len()]);
    emit_tensor_op_meta("attribution_report_new", || {
        serde_json::json!({
            "backend": "cpu",
            "requested_backend": "auto",
            "kind": "st_core_attribution_report_new",
            "algorithm": metadata.algorithm.as_str(),
            "rows": rows,
            "cols": cols,
            "values": values.len(),
            "finite_values": finite,
            "non_finite_values": non_finite,
            "positive_values": positive,
            "negative_values": negative,
            "zero_values": zero,
            "mean_value": if finite == 0 { 0.0 } else { total / finite as f64 },
            "mean_abs_value": if finite == 0 { 0.0 } else { abs_total / finite as f64 },
            "max_abs_value": max_abs,
            "has_layer": metadata.layer.is_some(),
            "has_target": metadata.target.is_some(),
            "has_steps": metadata.steps.is_some(),
            "steps": metadata.steps.unwrap_or(0),
            "extras": metadata.extras.len(),
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn metadata_builders_store_fields() {
        let mut metadata = AttributionMetadata::for_algorithm("grad-cam");
        metadata.layer = Some("conv1".to_string());
        metadata.target = Some("class-1".to_string());
        metadata.steps = Some(16);
        metadata.insert_extra_number("height", 7.0);
        metadata.insert_extra_flag("smooth", true);
        metadata.insert_extra_text("note", "deterministic");
        assert_eq!(metadata.algorithm, "grad-cam");
        assert_eq!(metadata.layer.as_deref(), Some("conv1"));
        assert_eq!(metadata.target.as_deref(), Some("class-1"));
        assert_eq!(metadata.steps, Some(16));
        assert!(metadata.extras.contains_key("height"));
        assert!(metadata.extras.contains_key("smooth"));
        assert!(metadata.extras.contains_key("note"));
    }

    #[test]
    fn report_serialises_and_round_trips() {
        let metadata = AttributionMetadata::for_algorithm("integrated-gradients");
        let report = AttributionReport::new(metadata, 2, 2, vec![0.0, 0.25, 0.5, 1.0]);
        let json = report.to_json_string().unwrap();
        let decoded = AttributionReport::from_json_str(&json).unwrap();
        assert_eq!(decoded.shape(), (2, 2));
        assert_eq!(decoded.values.len(), 4);
        assert_eq!(decoded.metadata.algorithm, "integrated-gradients");
    }

    #[test]
    fn report_new_emits_backend_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut metadata = AttributionMetadata::for_algorithm("integrated-gradients");
        metadata.layer = Some("encoder.block.0".to_string());
        metadata.target = Some("token:7".to_string());
        metadata.steps = Some(12);
        metadata.insert_extra_number("loss", 0.25);
        let report = AttributionReport::new(metadata, 2, 3, vec![0.0, 0.25, -0.5, 1.0, 0.0, 0.75]);
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(report.shape(), (2, 3));
        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "attribution_report_new"
                    && data["algorithm"] == "integrated-gradients"
                    && data["rows"] == 2
                    && data["cols"] == 3
            })
            .expect("attribution_report_new metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["kind"], "st_core_attribution_report_new");
        assert_eq!(meta.1["finite_values"], 6);
        assert_eq!(meta.1["positive_values"], 3);
        assert_eq!(meta.1["negative_values"], 1);
        assert!(meta.1["has_layer"].as_bool().unwrap_or(false));
        assert!(meta.1["has_target"].as_bool().unwrap_or(false));
        assert_eq!(meta.1["steps"], 12);
        assert_eq!(meta.1["extras"], 1);
    }
}
