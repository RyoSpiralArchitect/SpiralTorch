#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

type RegistryMap = HashMap<TypeId, Vec<Arc<dyn MetricComputation>>>;

static REGISTRY: OnceLock<RwLock<RegistryMap>> = OnceLock::new();

fn read_recover<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    match lock.read() {
        Ok(guard) => guard,
        Err(poisoned) => {
            lock.clear_poison();
            poisoned.into_inner()
        }
    }
}

fn write_recover<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    match lock.write() {
        Ok(guard) => guard,
        Err(poisoned) => {
            lock.clear_poison();
            poisoned.into_inner()
        }
    }
}

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    let payload = match payload.downcast::<String>() {
        Ok(message) => return *message,
        Err(payload) => payload,
    };
    let payload = match payload.downcast::<&'static str>() {
        Ok(message) => return (*message).to_string(),
        Err(payload) => payload,
    };

    if let Err(secondary_payload) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
        std::mem::forget(secondary_payload);
    }
    "non-string panic payload".to_string()
}

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
#[derive(Clone, Debug, PartialEq)]
pub enum MetricValue {
    /// Scalar quantity.
    Scalar(f64),
    /// Histogram or batch of scalars.
    Distribution(Vec<f64>),
}

impl MetricValue {
    /// Returns whether every scalar represented by this metric is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        match self {
            MetricValue::Scalar(value) => value.is_finite(),
            MetricValue::Distribution(values) => values.iter().all(|value| value.is_finite()),
        }
    }
}

/// A metric evaluator that panicked while processing an input value.
#[derive(Clone, Debug)]
pub struct MetricEvaluationFailure {
    pub descriptor: MetricDescriptor,
    pub message: String,
}

/// Values and isolated evaluator failures produced by one registry snapshot.
#[derive(Clone, Debug, Default)]
pub struct MetricEvaluationReport {
    pub values: Vec<(MetricDescriptor, MetricValue)>,
    pub failures: Vec<MetricEvaluationFailure>,
}

impl MetricEvaluationReport {
    /// Returns whether every matching evaluator completed without panicking.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failures.is_empty()
    }
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
    let mut guard = write_recover(registry());
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
///
/// Panicking evaluators are isolated and omitted. Use [`evaluate_report`] when failure details are
/// needed.
#[must_use]
pub fn evaluate<T>(value: &T) -> Vec<(MetricDescriptor, MetricValue)>
where
    T: Any + Send + Sync + 'static,
{
    evaluate_report(value).values
}

/// Evaluates a stable registry snapshot and reports panicking evaluators separately.
///
/// Evaluators run without a registry lock, so they may register metrics reentrantly. Metrics
/// registered during evaluation become visible on the next call.
#[must_use]
pub fn evaluate_report<T>(value: &T) -> MetricEvaluationReport
where
    T: Any + Send + Sync + 'static,
{
    let metrics: Vec<Arc<dyn MetricComputation>> = read_recover(registry())
        .get(&TypeId::of::<T>())
        .cloned()
        .unwrap_or_default();
    let mut report = MetricEvaluationReport::default();

    for metric in metrics {
        let descriptor = metric.descriptor().clone();
        match catch_unwind(AssertUnwindSafe(|| metric.evaluate_any(value))) {
            Ok(Some(metric_value)) => report.values.push((descriptor, metric_value)),
            Ok(None) => {}
            Err(payload) => report.failures.push(MetricEvaluationFailure {
                descriptor,
                message: panic_payload_message(payload),
            }),
        }
    }
    report
}

/// Returns the descriptors registered for the provided type without evaluating them.
#[must_use]
pub fn descriptors_for<T>() -> Vec<MetricDescriptor>
where
    T: Any + Send + Sync + 'static,
{
    let guard = read_recover(registry());
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
        write_recover(lock).clear();
    }
}

#[cfg(test)]
pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    static TEST_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    let lock = TEST_LOCK.get_or_init(|| std::sync::Mutex::new(()));
    match lock.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            lock.clear_poison();
            poisoned.into_inner()
        }
    }
}

#[cfg(test)]
pub(crate) fn write_available_for_tests() -> bool {
    registry().try_write().is_ok()
}

#[cfg(test)]
pub(crate) fn poison_for_tests() {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _guard = write_recover(registry());
        panic!("poison metric registry");
    }));
}
