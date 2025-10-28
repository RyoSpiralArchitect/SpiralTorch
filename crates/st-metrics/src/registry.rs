use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

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

#[cfg(test)]
pub(crate) fn clear_for_tests() {
    if let Some(lock) = REGISTRY.get() {
        lock.write()
            .expect("metric registry poisoned while clearing")
            .clear();
    }
}
