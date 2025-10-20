use serde::{Deserialize, Serialize};
use serde_json::Value;
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
