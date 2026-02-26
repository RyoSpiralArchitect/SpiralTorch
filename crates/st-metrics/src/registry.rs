use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

#[cfg(feature = "dashboard")]
use st_core::telemetry::dashboard::{DashboardFrame, DashboardMetric};

#[cfg(feature = "dashboard")]
use std::time::SystemTime;

type RegistryMap = HashMap<TypeId, Vec<Arc<dyn MetricComputation>>>;

static REGISTRY: OnceLock<RwLock<RegistryMap>> = OnceLock::new();

/// Unit metadata associated with a registered metric.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricUnit {
    /// Unitless scalar quantity.
    Unitless,
    /// Values expressed as probability in `[0, 1]`.
    Probability,
    /// Loss values where lower is better.
    Loss,
    /// Raw logits emitted by a contrastive head.
    Logit,
    /// Arbitrary unit represented by a custom label.
    Custom(&'static str),
}

/// Descriptor that summarises a registered metric.
#[derive(Clone, Debug)]
pub struct MetricDescriptor {
    pub name: &'static str,
    pub description: &'static str,
    pub unit: MetricUnit,
    pub tags: &'static [&'static str],
    pub higher_is_better: Option<bool>,
}

/// Concrete value returned by a metric evaluation.
#[derive(Clone, Debug)]
pub enum MetricValue {
    /// Scalar quantity.
    Scalar(f64),
    /// Histogram or batch of scalars.
    Distribution(Vec<f64>),
}

trait MetricComputation: Send + Sync {
    fn descriptor(&self) -> &MetricDescriptor;
    fn evaluate_any(&self, input: &dyn Any) -> Option<MetricValue>;
}

struct FnMetric<T, F>
where
    T: Any + Send + Sync + 'static,
    F: Fn(&T) -> Option<MetricValue> + Send + Sync + 'static,
{
    descriptor: MetricDescriptor,
    evaluator: F,
    _marker: std::marker::PhantomData<T>,
}

impl<T, F> MetricComputation for FnMetric<T, F>
where
    T: Any + Send + Sync + 'static,
    F: Fn(&T) -> Option<MetricValue> + Send + Sync + 'static,
{
    fn descriptor(&self) -> &MetricDescriptor {
        &self.descriptor
    }

    fn evaluate_any(&self, input: &dyn Any) -> Option<MetricValue> {
        input
            .downcast_ref::<T>()
            .and_then(|value| (self.evaluator)(value))
    }
}

fn registry() -> &'static RwLock<RegistryMap> {
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

fn insert_metric(type_id: TypeId, metric: Arc<dyn MetricComputation>) {
    let mut guard = registry()
        .write()
        .expect("metric registry poisoned while registering metric");
    let entry = guard.entry(type_id).or_default();
    if entry
        .iter()
        .any(|existing| existing.descriptor().name == metric.descriptor().name)
    {
        return;
    }
    entry.push(metric);
}

/// Registers a new metric evaluator for the provided type `T`.
pub fn register_metric<T, F>(descriptor: MetricDescriptor, evaluator: F)
where
    T: Any + Send + Sync + 'static,
    F: Fn(&T) -> Option<MetricValue> + Send + Sync + 'static,
{
    let type_id = TypeId::of::<T>();
    let metric: Arc<dyn MetricComputation> = Arc::new(FnMetric::<T, F> {
        descriptor,
        evaluator,
        _marker: std::marker::PhantomData,
    });
    insert_metric(type_id, metric);
}

/// Evaluates all registered metrics that match the provided value type.
pub fn evaluate<T>(value: &T) -> Vec<(MetricDescriptor, MetricValue)>
where
    T: Any + Send + Sync + 'static,
{
    let guard = registry()
        .read()
        .expect("metric registry poisoned while evaluating metrics");
    guard
        .get(&TypeId::of::<T>())
        .into_iter()
        .flat_map(|metrics| {
            metrics.iter().filter_map(|metric| {
                metric
                    .evaluate_any(value)
                    .map(|v| (metric.descriptor().clone(), v))
            })
        })
        .collect()
}

/// Returns the descriptors registered for the provided type without evaluating them.
pub fn descriptors_for<T>() -> Vec<MetricDescriptor>
where
    T: Any + Send + Sync + 'static,
{
    let guard = registry()
        .read()
        .expect("metric registry poisoned while enumerating descriptors");
    guard
        .get(&TypeId::of::<T>())
        .map(|metrics| {
            metrics
                .iter()
                .map(|metric| metric.descriptor().clone())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(feature = "json")]
fn unit_to_label(unit: MetricUnit) -> String {
    match unit {
        MetricUnit::Unitless => "unitless".to_string(),
        MetricUnit::Probability => "probability".to_string(),
        MetricUnit::Loss => "loss".to_string(),
        MetricUnit::Logit => "logit".to_string(),
        MetricUnit::Custom(label) => format!("custom:{label}"),
    }
}

#[cfg(feature = "json")]
fn value_to_json(value: &MetricValue) -> serde_json::Value {
    match value {
        MetricValue::Scalar(v) => serde_json::json!({
            "type": "scalar",
            "value": v,
        }),
        MetricValue::Distribution(values) => serde_json::json!({
            "type": "distribution",
            "value": values,
        }),
    }
}

#[cfg(feature = "json")]
fn descriptor_to_json(descriptor: &MetricDescriptor) -> serde_json::Value {
    serde_json::json!({
        "name": descriptor.name,
        "description": descriptor.description,
        "unit": unit_to_label(descriptor.unit),
        "tags": descriptor.tags,
        "higher_is_better": descriptor.higher_is_better,
    })
}

/// Converts evaluated metrics into a JSON array.
#[cfg(feature = "json")]
pub fn evaluation_to_json(results: &[(MetricDescriptor, MetricValue)]) -> serde_json::Value {
    serde_json::Value::Array(
        results
            .iter()
            .map(|(descriptor, value)| {
                serde_json::json!({
                    "descriptor": descriptor_to_json(descriptor),
                    "value": value_to_json(value),
                })
            })
            .collect(),
    )
}

/// Converts evaluated metrics into a scalar map for downstream dashboards.
///
/// - `Scalar` values are stored directly under the descriptor name.
/// - `Distribution` values emit derived scalars under suffixes:
///   `.count`, `.mean`, `.min`, `.max`.
pub fn evaluation_to_scalar_map(
    results: &[(MetricDescriptor, MetricValue)],
) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for (descriptor, metric_value) in results {
        match metric_value {
            MetricValue::Scalar(v) => {
                out.insert(descriptor.name.to_string(), *v);
            }
            MetricValue::Distribution(values) => {
                let count = values.len() as f64;
                out.insert(format!("{}.count", descriptor.name), count);
                if values.is_empty() {
                    continue;
                }
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                let mut sum = 0.0f64;
                for &value in values {
                    sum += value;
                    if value < min {
                        min = value;
                    }
                    if value > max {
                        max = value;
                    }
                }
                out.insert(format!("{}.mean", descriptor.name), sum / count.max(1.0));
                out.insert(format!("{}.min", descriptor.name), min);
                out.insert(format!("{}.max", descriptor.name), max);
            }
        }
    }
    out
}

/// Evaluates all registered metrics and returns a scalar map (see [`evaluation_to_scalar_map`]).
pub fn evaluate_scalar_map<T>(value: &T) -> HashMap<String, f64>
where
    T: Any + Send + Sync + 'static,
{
    evaluation_to_scalar_map(&evaluate(value))
}

#[cfg(feature = "dashboard")]
fn unit_to_dashboard_label(unit: MetricUnit) -> Option<String> {
    match unit {
        MetricUnit::Unitless => None,
        MetricUnit::Probability => Some("probability".to_string()),
        MetricUnit::Loss => Some("loss".to_string()),
        MetricUnit::Logit => Some("logit".to_string()),
        MetricUnit::Custom(label) => Some(label.to_string()),
    }
}

/// Converts evaluated metrics into a dashboard telemetry frame.
#[cfg(feature = "dashboard")]
pub fn evaluation_to_dashboard_frame(
    results: &[(MetricDescriptor, MetricValue)],
) -> DashboardFrame {
    let mut frame = DashboardFrame::new(SystemTime::now());
    for (descriptor, metric_value) in results {
        let unit = unit_to_dashboard_label(descriptor.unit);
        match metric_value {
            MetricValue::Scalar(v) => {
                let mut metric = DashboardMetric::new(descriptor.name, *v);
                if let Some(unit) = unit.clone() {
                    metric = metric.with_unit(unit);
                }
                frame.push_metric(metric);
            }
            MetricValue::Distribution(values) => {
                let count = values.len() as f64;
                frame.push_metric(
                    DashboardMetric::new(format!("{}.count", descriptor.name), count)
                        .with_unit("count"),
                );
                if values.is_empty() {
                    continue;
                }
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                let mut sum = 0.0f64;
                for &value in values {
                    sum += value;
                    if value < min {
                        min = value;
                    }
                    if value > max {
                        max = value;
                    }
                }
                let mean = sum / count.max(1.0);

                let mut mean_metric =
                    DashboardMetric::new(format!("{}.mean", descriptor.name), mean);
                let mut min_metric = DashboardMetric::new(format!("{}.min", descriptor.name), min);
                let mut max_metric = DashboardMetric::new(format!("{}.max", descriptor.name), max);
                if let Some(unit) = unit {
                    mean_metric = mean_metric.with_unit(unit.clone());
                    min_metric = min_metric.with_unit(unit.clone());
                    max_metric = max_metric.with_unit(unit);
                }
                frame.push_metric(mean_metric);
                frame.push_metric(min_metric);
                frame.push_metric(max_metric);
            }
        }
    }
    frame
}

/// Evaluates all registered metrics and returns a dashboard frame.
#[cfg(feature = "dashboard")]
pub fn evaluate_dashboard_frame<T>(value: &T) -> DashboardFrame
where
    T: Any + Send + Sync + 'static,
{
    evaluation_to_dashboard_frame(&evaluate(value))
}

/// Evaluates all registered metrics and publishes a dashboard frame to the global telemetry hub.
#[cfg(feature = "dashboard")]
pub fn push_dashboard_frame<T>(value: &T)
where
    T: Any + Send + Sync + 'static,
{
    let frame = evaluate_dashboard_frame(value);
    st_core::telemetry::hub::push_dashboard_frame(frame);
}

/// Evaluates all registered metrics and returns a JSON array.
#[cfg(feature = "json")]
pub fn evaluate_json<T>(value: &T) -> serde_json::Value
where
    T: Any + Send + Sync + 'static,
{
    evaluation_to_json(&evaluate(value))
}

/// Returns a JSON array of descriptors without running evaluation.
#[cfg(feature = "json")]
pub fn descriptors_json_for<T>() -> serde_json::Value
where
    T: Any + Send + Sync + 'static,
{
    serde_json::Value::Array(
        descriptors_for::<T>()
            .iter()
            .map(descriptor_to_json)
            .collect(),
    )
}

/// Emits evaluated metrics into `tracing` as `INFO` events.
#[cfg(feature = "tracing")]
pub fn emit_tracing<T>(value: &T)
where
    T: Any + Send + Sync + 'static,
{
    for (descriptor, metric_value) in evaluate(value) {
        match metric_value {
            MetricValue::Scalar(v) => tracing::event!(
                target: "spiraltorch::metrics",
                tracing::Level::INFO,
                metric = descriptor.name,
                unit = ?descriptor.unit,
                value = v,
            ),
            MetricValue::Distribution(values) => {
                let count = values.len();
                let mut mean = 0.0f64;
                if count > 0 {
                    mean = values.iter().copied().sum::<f64>() / (count as f64);
                }
                tracing::event!(
                    target: "spiraltorch::metrics",
                    tracing::Level::INFO,
                    metric = descriptor.name,
                    unit = ?descriptor.unit,
                    distribution_count = count,
                    distribution_mean = mean,
                );
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn clear_for_tests() {
    if let Some(lock) = REGISTRY.get() {
        lock.write()
            .expect("metric registry poisoned while clearing")
            .clear();
    }
}
