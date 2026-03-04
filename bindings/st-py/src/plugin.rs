use pyo3::exceptions::{PyStopIteration, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyList, PyModule};
use pyo3::wrap_pyfunction;
use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::json::{json_to_py, py_to_json};
use crate::tensor::tensor_err_to_py;
use st_core::PureResult;
use st_core::TensorError;
use st_core::plugin::{
    global_registry, init_plugin_system, EventListener, Plugin, PluginCapability, PluginContext,
    PluginEvent, PluginEventJsonlWriter, PluginEventJsonlWriterConfig, PluginEventRecorder,
    PluginEventRecorderConfig, PluginMetadata,
};

fn plugin_event_to_py(py: Python<'_>, event: &PluginEvent) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    dict.set_item("ts", now.as_secs_f64())?;
    match event {
        PluginEvent::SystemInit => {
            dict.set_item("type", "SystemInit")?;
        }
        PluginEvent::SystemShutdown => {
            dict.set_item("type", "SystemShutdown")?;
        }
        PluginEvent::PluginLoaded { plugin_id } => {
            dict.set_item("type", "PluginLoaded")?;
            dict.set_item("plugin_id", plugin_id)?;
        }
        PluginEvent::PluginUnloaded { plugin_id } => {
            dict.set_item("type", "PluginUnloaded")?;
            dict.set_item("plugin_id", plugin_id)?;
        }
        PluginEvent::TensorOp {
            op_name,
            input_shape,
            output_shape,
        } => {
            dict.set_item("type", "TensorOp")?;
            dict.set_item("op_name", op_name)?;
            dict.set_item("input_shape", input_shape.clone())?;
            dict.set_item("output_shape", output_shape.clone())?;
        }
        PluginEvent::EpochStart { epoch } => {
            dict.set_item("type", "EpochStart")?;
            dict.set_item("epoch", *epoch)?;
        }
        PluginEvent::EpochEnd { epoch, loss } => {
            dict.set_item("type", "EpochEnd")?;
            dict.set_item("epoch", *epoch)?;
            dict.set_item("loss", *loss)?;
        }
        PluginEvent::BackendChanged { backend } => {
            dict.set_item("type", "BackendChanged")?;
            dict.set_item("backend", backend)?;
        }
        PluginEvent::Telemetry { data } => {
            dict.set_item("type", "Telemetry")?;
            dict.set_item("data", data.clone())?;
        }
        PluginEvent::Custom { event_type, .. } => {
            dict.set_item("type", event_type)?;
            dict.set_item("event_type", event_type)?;
            if let Some(payload) = event.downcast_data::<serde_json::Value>() {
                dict.set_item("payload", json_to_py(py, payload)?)?;
            } else if let Some(payload) = event.downcast_data::<String>() {
                dict.set_item("payload", payload.clone())?;
            } else {
                dict.set_item("payload", py.None())?;
            }
        }
    }
    Ok(dict.into_py(py))
}

fn plugin_metadata_to_py(py: Python<'_>, metadata: &PluginMetadata) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", metadata.id.clone())?;
    dict.set_item("version", metadata.version.clone())?;
    dict.set_item("name", metadata.name.clone())?;
    dict.set_item("description", metadata.description.clone())?;
    dict.set_item("author", metadata.author.clone())?;

    let dependencies = PyDict::new_bound(py);
    for (key, value) in &metadata.dependencies {
        dependencies.set_item(key, value)?;
    }
    dict.set_item("dependencies", dependencies)?;

    let capabilities = PyList::empty_bound(py);
    for cap in &metadata.capabilities {
        capabilities.append(cap.to_string())?;
    }
    dict.set_item("capabilities", capabilities)?;

    let extra = PyDict::new_bound(py);
    for (key, value) in &metadata.metadata {
        extra.set_item(key, value)?;
    }
    dict.set_item("metadata", extra)?;

    Ok(dict.into_py(py))
}

fn required_item<'py, T: FromPyObject<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    err: &str,
) -> PyResult<T> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(err.to_string()))?
        .extract()
}

fn optional_string_item(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract()?))
}

fn optional_string_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    let Ok(value) = obj.getattr(name) else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract()?))
}

fn parse_capability(label: &str) -> PluginCapability {
    let trimmed = label.trim();
    let lower = trimmed.to_ascii_lowercase();

    if let Some(inner) = trimmed
        .strip_prefix("Backend(")
        .and_then(|rest| rest.strip_suffix(')'))
        .or_else(|| {
            trimmed
                .strip_prefix("backend(")
                .and_then(|rest| rest.strip_suffix(')'))
        })
        .or_else(|| trimmed.strip_prefix("Backend:"))
        .or_else(|| trimmed.strip_prefix("backend:"))
    {
        return PluginCapability::Backend(inner.trim().to_string());
    }

    if let Some(inner) = trimmed
        .strip_prefix("Custom(")
        .and_then(|rest| rest.strip_suffix(')'))
        .or_else(|| {
            trimmed
                .strip_prefix("custom(")
                .and_then(|rest| rest.strip_suffix(')'))
        })
        .or_else(|| trimmed.strip_prefix("Custom:"))
        .or_else(|| trimmed.strip_prefix("custom:"))
    {
        return PluginCapability::Custom(inner.trim().to_string());
    }

    let normalized: String = lower
        .chars()
        .filter(|ch| !matches!(ch, '_' | '-' | ' ' | '\t' | '\n' | '\r'))
        .collect();

    match normalized.as_str() {
        "operators" => PluginCapability::Operators,
        "lossfunctions" => PluginCapability::LossFunctions,
        "optimizers" => PluginCapability::Optimizers,
        "dataloaders" => PluginCapability::DataLoaders,
        "visualization" => PluginCapability::Visualization,
        "telemetry" => PluginCapability::Telemetry,
        "language" => PluginCapability::Language,
        "vision" => PluginCapability::Vision,
        "reinforcementlearning" => PluginCapability::ReinforcementLearning,
        "graphneuralnetworks" => PluginCapability::GraphNeuralNetworks,
        "recommender" => PluginCapability::Recommender,
        _ => PluginCapability::Custom(trimmed.to_string()),
    }
}

fn collect_strings(
    values: &Bound<'_, PyAny>,
    name: &str,
    allow_empty: bool,
) -> PyResult<Vec<String>> {
    if let Ok(single) = values.extract::<String>() {
        if single.trim().is_empty() {
            return Err(PyValueError::new_err(format!("{name} must not be empty")));
        }
        return Ok(vec![single]);
    }

    let iter = PyIterator::from_bound_object(values).map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a string or iterable of strings"))
    })?;
    let mut out = Vec::new();
    for item in iter {
        let item = item?;
        let value: String = item
            .extract()
            .map_err(|_| PyTypeError::new_err(format!("{name} must contain only strings")))?;
        if value.trim().is_empty() {
            return Err(PyValueError::new_err(format!("{name} must not contain empty strings")));
        }
        out.push(value);
    }
    if out.is_empty() && !allow_empty {
        return Err(PyValueError::new_err(format!("{name} must not be empty")));
    }
    Ok(out)
}

fn plugin_metadata_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<PluginMetadata> {
    let id: String = required_item(dict, "id", "plugin metadata missing 'id'")?;
    let version: String = required_item(dict, "version", "plugin metadata missing 'version'")?;

    let mut meta = PluginMetadata::new(id, version);
    if let Some(value) = optional_string_item(dict, "name")? {
        meta = meta.with_name(value);
    }
    if let Some(value) = optional_string_item(dict, "description")? {
        meta = meta.with_description(value);
    }
    if let Some(value) = optional_string_item(dict, "author")? {
        meta = meta.with_author(value);
    }

    if let Some(deps_obj) = dict.get_item("dependencies")? {
        if !deps_obj.is_none() {
            let deps = deps_obj.downcast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("plugin metadata field 'dependencies' must be a mapping")
            })?;
            for (key, value) in deps.iter() {
                let key: String = key.extract().map_err(|_| {
                    PyTypeError::new_err("plugin metadata dependencies keys must be strings")
                })?;
                let value: String = value.extract().map_err(|_| {
                    PyTypeError::new_err("plugin metadata dependencies values must be strings")
                })?;
                meta = meta.with_dependency(key, value);
            }
        }
    }

    if let Some(cap_obj) = dict.get_item("capabilities")? {
        if !cap_obj.is_none() {
            for label in collect_strings(&cap_obj, "capabilities", true)? {
                meta = meta.with_capability(parse_capability(&label));
            }
        }
    }

    if let Some(extra_obj) = dict.get_item("metadata")? {
        if !extra_obj.is_none() {
            let extra = extra_obj.downcast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("plugin metadata field 'metadata' must be a mapping")
            })?;
            for (key, value) in extra.iter() {
                let key: String = key.extract().map_err(|_| {
                    PyTypeError::new_err("plugin metadata keys must be strings")
                })?;
                let value: String = value.extract().map_err(|_| {
                    PyTypeError::new_err("plugin metadata values must be strings")
                })?;
                meta = meta.with_metadata(key, value);
            }
        }
    }

    Ok(meta)
}

fn plugin_metadata_from_attrs(plugin: &Bound<'_, PyAny>) -> PyResult<PluginMetadata> {
    let id: String = plugin.getattr("id").map_err(|_| {
        PyValueError::new_err("python plugin must define metadata() or 'id' attribute")
    })?.extract()?;
    let version: String = plugin.getattr("version").map_err(|_| {
        PyValueError::new_err("python plugin must define metadata() or 'version' attribute")
    })?.extract()?;

    let mut meta = PluginMetadata::new(id, version);
    if let Some(value) = optional_string_attr(plugin, "name")? {
        meta = meta.with_name(value);
    }
    if let Some(value) = optional_string_attr(plugin, "description")? {
        meta = meta.with_description(value);
    }
    if let Some(value) = optional_string_attr(plugin, "author")? {
        meta = meta.with_author(value);
    }

    if let Ok(deps_obj) = plugin.getattr("dependencies") {
        if !deps_obj.is_none() {
            let deps = deps_obj.downcast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("python plugin attribute 'dependencies' must be a mapping")
            })?;
            for (key, value) in deps.iter() {
                let key: String = key.extract().map_err(|_| {
                    PyTypeError::new_err("python plugin dependencies keys must be strings")
                })?;
                let value: String = value.extract().map_err(|_| {
                    PyTypeError::new_err("python plugin dependencies values must be strings")
                })?;
                meta = meta.with_dependency(key, value);
            }
        }
    }

    if let Ok(cap_obj) = plugin.getattr("capabilities") {
        if !cap_obj.is_none() {
            for label in collect_strings(&cap_obj, "capabilities", true)? {
                meta = meta.with_capability(parse_capability(&label));
            }
        }
    }

    if let Ok(extra_obj) = plugin.getattr("metadata") {
        if !extra_obj.is_callable() && !extra_obj.is_none() {
            let extra = extra_obj.downcast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("python plugin attribute 'metadata' must be a mapping")
            })?;
            for (key, value) in extra.iter() {
                let key: String = key.extract().map_err(|_| {
                    PyTypeError::new_err("python plugin metadata keys must be strings")
                })?;
                let value: String = value.extract().map_err(|_| {
                    PyTypeError::new_err("python plugin metadata values must be strings")
                })?;
                meta = meta.with_metadata(key, value);
            }
        }
    }

    Ok(meta)
}

fn plugin_metadata_from_py(plugin: &Bound<'_, PyAny>) -> PyResult<PluginMetadata> {
    if let Ok(meta_attr) = plugin.getattr("metadata") {
        if meta_attr.is_callable() {
            let meta_obj = meta_attr.call0()?;
            if meta_obj.is_none() {
                return plugin_metadata_from_attrs(plugin);
            }
            let dict = meta_obj.downcast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("metadata() must return a dict-like mapping")
            })?;
            return plugin_metadata_from_dict(&dict);
        }

        if meta_attr.is_none() {
            return plugin_metadata_from_attrs(plugin);
        }

        if let Ok(dict) = meta_attr.downcast::<PyDict>() {
            let has_id = dict.get_item("id")?.is_some();
            let has_version = dict.get_item("version")?.is_some();
            if has_id && has_version {
                return plugin_metadata_from_dict(&dict);
            }
        }

        return plugin_metadata_from_attrs(plugin);
    }

    plugin_metadata_from_attrs(plugin)
}

fn plugin_event_types(plugin: &Bound<'_, PyAny>) -> PyResult<Option<Vec<String>>> {
    let Ok(event_types_attr) = plugin.getattr("event_types") else {
        return Ok(None);
    };
    let values = if event_types_attr.is_callable() {
        event_types_attr.call0()?
    } else {
        event_types_attr
    };
    if values.is_none() {
        return Ok(Some(Vec::new()));
    }
    Ok(Some(collect_strings(&values, "event_types", true)?))
}

fn plugin_auto_subscribe(plugin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let Ok(flag) = plugin.getattr("auto_subscribe") else {
        return Ok(true);
    };
    if flag.is_none() {
        return Ok(true);
    }
    flag.extract().map_err(|_| {
        PyTypeError::new_err("python plugin attribute 'auto_subscribe' must be a bool when set")
    })
}

fn python_plugin_is_callable(plugin: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    let Ok(attr) = plugin.getattr(name) else {
        return Ok(false);
    };
    if attr.is_none() {
        return Ok(false);
    }
    Ok(attr.is_callable())
}

struct PythonPlugin {
    metadata: PluginMetadata,
    plugin: Arc<Mutex<Py<PyAny>>>,
    subscriptions: Vec<(String, usize)>,
}

impl PythonPlugin {
    fn new(metadata: PluginMetadata, plugin: Py<PyAny>) -> Self {
        Self {
            metadata,
            plugin: Arc::new(Mutex::new(plugin)),
            subscriptions: Vec::new(),
        }
    }

    fn clone_plugin_ref(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let plugin = match self.plugin.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        Ok(plugin.clone_ref(py))
    }

    fn resolve_event_targets(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let plugin_obj = self.clone_plugin_ref(py)?;
        let plugin_obj = plugin_obj.bind(py);

        if !plugin_auto_subscribe(&plugin_obj)? {
            return Ok(Vec::new());
        }

        if !python_plugin_is_callable(&plugin_obj, "on_event")? {
            return Ok(Vec::new());
        }

        let mut targets = match plugin_event_types(&plugin_obj)? {
            Some(list) => list,
            None => vec!["*".to_string()],
        };
        targets.retain(|value| !value.trim().is_empty());
        Ok(targets)
    }

    fn subscribe_events(&mut self, ctx: &PluginContext, targets: Vec<String>) -> PureResult<()> {
        if targets.is_empty() {
            return Ok(());
        }

        let plugin = self.plugin.clone();
        let plugin_id = self.metadata.id.clone();
        for event_type in targets {
            let plugin = plugin.clone();
            let plugin_id = plugin_id.clone();
            let listener: EventListener = Arc::new(move |event| {
                Python::with_gil(|py| {
                    let payload = match plugin_event_to_py(py, event) {
                        Ok(payload) => payload,
                        Err(err) => {
                            err.print(py);
                            return;
                        }
                    };

                    let obj = match plugin.lock() {
                        Ok(guard) => guard.clone_ref(py),
                        Err(poisoned) => poisoned.into_inner().clone_ref(py),
                    };
                    let obj = obj.bind(py);
                    let on_event = match obj.getattr("on_event") {
                        Ok(method) => method,
                        Err(_) => return,
                    };
                    if !on_event.is_callable() {
                        return;
                    }
                    if let Err(err) = on_event.call1((payload,)) {
                        eprintln!("python plugin '{plugin_id}' on_event error: {err}");
                        err.print(py);
                    }
                });
            });
            let id = ctx.event_bus.subscribe(event_type.clone(), listener);
            self.subscriptions.push((event_type, id));
        }

        Ok(())
    }

    fn call_hook(&self, hook: &str) -> PureResult<()> {
        Python::with_gil(|py| -> PyResult<()> {
            let obj = self.clone_plugin_ref(py)?;
            let obj = obj.bind(py);
            let Ok(method) = obj.getattr(hook) else {
                return Ok(());
            };
            if method.is_none() {
                return Ok(());
            }
            if !method.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "python plugin attribute '{hook}' must be callable when set"
                )));
            }
            method.call0()?;
            Ok(())
        })
        .map_err(|err| {
            TensorError::Generic(format!(
                "python plugin '{}' {hook}() failed: {err}",
                self.metadata.id
            ))
        })?;
        Ok(())
    }
}

impl Plugin for PythonPlugin {
    fn metadata(&self) -> PluginMetadata {
        self.metadata.clone()
    }

    fn on_load(&mut self, ctx: &mut PluginContext) -> st_core::PureResult<()> {
        self.call_hook("on_load")?;

        let targets = Python::with_gil(|py| self.resolve_event_targets(py)).map_err(|err| {
            TensorError::Generic(format!(
                "python plugin '{}' event subscription lookup failed: {err}",
                self.metadata.id
            ))
        })?;

        self.subscribe_events(ctx, targets)?;

        Ok(())
    }

    fn on_unload(&mut self, ctx: &mut PluginContext) -> st_core::PureResult<()> {
        let hook_err = self.call_hook("on_unload").err();
        for (event_type, id) in self.subscriptions.drain(..) {
            ctx.event_bus.unsubscribe(&event_type, id);
        }
        if let Some(err) = hook_err {
            return Err(err);
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn validate_callback(py: Python<'_>, callback: &PyObject) -> PyResult<()> {
    let callback_any: &Bound<PyAny> = callback.bind(py);
    if !callback_any.is_callable() {
        return Err(PyTypeError::new_err(
            "callback must be a callable that accepts one dict argument",
        ));
    }
    Ok(())
}

fn listener_from_callback(callback: Arc<Mutex<Py<PyAny>>>) -> EventListener {
    Arc::new(move |event| {
        Python::with_gil(|py| {
            let payload = match plugin_event_to_py(py, event) {
                Ok(payload) => payload,
                Err(err) => {
                    err.print(py);
                    return;
                }
            };
            let callback = match callback.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            if let Err(err) = callback.call1(py, (payload,)) {
                err.print(py);
            }
        });
    })
}

#[pyfunction]
#[pyo3(signature = (event_type, callback))]
fn subscribe(py: Python<'_>, event_type: &str, callback: PyObject) -> PyResult<usize> {
    validate_callback(py, &callback)?;

    init_plugin_system().map_err(tensor_err_to_py)?;

    let callback = Arc::new(Mutex::new(callback));
    let listener = listener_from_callback(callback);

    Ok(global_registry()
        .event_bus()
        .subscribe(event_type.to_string(), listener))
}

#[pyfunction]
#[pyo3(signature = (event_type, subscription_id))]
fn unsubscribe(event_type: &str, subscription_id: usize) -> PyResult<bool> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    Ok(global_registry()
        .event_bus()
        .unsubscribe(event_type, subscription_id))
}

fn collect_event_types(event_types: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(single) = event_types.extract::<String>() {
        return Ok(vec![single]);
    }
    let iter = PyIterator::from_bound_object(event_types)
        .map_err(|_| PyTypeError::new_err("event_types must be a string or iterable of strings"))?;
    let mut out = Vec::new();
    for item in iter {
        let item = item?;
        let name: String = item
            .extract()
            .map_err(|_| PyTypeError::new_err("event_types must contain only strings"))?;
        out.push(name);
    }
    if out.is_empty() {
        return Err(PyValueError::new_err("event_types must not be empty"));
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (event_types, callback))]
fn subscribe_many(
    py: Python<'_>,
    event_types: &Bound<'_, PyAny>,
    callback: PyObject,
) -> PyResult<Vec<(String, usize)>> {
    validate_callback(py, &callback)?;
    init_plugin_system().map_err(tensor_err_to_py)?;

    let event_types = collect_event_types(event_types)?;
    let callback = Arc::new(Mutex::new(callback));
    let listener = listener_from_callback(callback);

    let mut out = Vec::with_capacity(event_types.len());
    for event_type in event_types {
        let id = global_registry()
            .event_bus()
            .subscribe(event_type.clone(), listener.clone());
        out.push((event_type, id));
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (subscriptions))]
fn unsubscribe_many(subscriptions: &Bound<'_, PyAny>) -> PyResult<usize> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    let iter = PyIterator::from_bound_object(subscriptions).map_err(|_| {
        PyTypeError::new_err("subscriptions must be an iterable of (event_type, id) pairs")
    })?;
    let mut removed = 0usize;
    for item in iter {
        let item = item?;
        let (event_type, id): (String, usize) = item.extract().map_err(|_| {
            PyTypeError::new_err("subscriptions must contain (event_type, id) pairs")
        })?;
        if global_registry().event_bus().unsubscribe(&event_type, id) {
            removed += 1;
        }
    }
    Ok(removed)
}

#[pyclass(module = "spiraltorch.plugin", name = "PluginQueue", unsendable)]
pub struct PyPluginQueue {
    event_type: String,
    subscription_id: usize,
    maxlen: usize,
    queue: Arc<Mutex<VecDeque<PyObject>>>,
    closed: bool,
}

impl PyPluginQueue {
    fn close_internal(&mut self) -> bool {
        if self.closed {
            return false;
        }
        self.closed = true;
        global_registry()
            .event_bus()
            .unsubscribe(&self.event_type, self.subscription_id)
    }
}

#[pymethods]
impl PyPluginQueue {
    #[getter]
    pub fn event_type(&self) -> &str {
        &self.event_type
    }

    #[getter]
    pub fn subscription_id(&self) -> usize {
        self.subscription_id
    }

    #[getter]
    pub fn maxlen(&self) -> usize {
        self.maxlen
    }

    pub fn poll(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| PyTypeError::new_err("event queue lock was poisoned"))?;
        Ok(queue.pop_front().map(|item| item.into_py(py)))
    }

    #[pyo3(signature = (max_items=None))]
    pub fn drain(&self, py: Python<'_>, max_items: Option<usize>) -> PyResult<Vec<PyObject>> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| PyTypeError::new_err("event queue lock was poisoned"))?;
        let take = max_items.unwrap_or(queue.len());
        let mut out = Vec::new();
        for _ in 0..take {
            if let Some(item) = queue.pop_front() {
                out.push(item.into_py(py));
            } else {
                break;
            }
        }
        Ok(out)
    }

    pub fn close(&mut self) -> PyResult<bool> {
        init_plugin_system().map_err(tensor_err_to_py)?;
        Ok(self.close_internal())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let _ = slf.close_internal();
        Ok(())
    }

    fn __len__(&self) -> PyResult<usize> {
        let queue = self
            .queue
            .lock()
            .map_err(|_| PyTypeError::new_err("event queue lock was poisoned"))?;
        Ok(queue.len())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| PyTypeError::new_err("event queue lock was poisoned"))?;
        if let Some(item) = queue.pop_front() {
            Ok(item.into_py(py))
        } else {
            Err(PyStopIteration::new_err(()))
        }
    }
}

impl Drop for PyPluginQueue {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.close_internal();
        }
    }
}

#[pyfunction]
#[pyo3(signature = (event_type="*", *, maxlen=1024))]
fn listen(event_type: &str, maxlen: usize) -> PyResult<PyPluginQueue> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    let maxlen = maxlen.max(1);
    let queue = Arc::new(Mutex::new(VecDeque::with_capacity(maxlen)));
    let queue_clone = queue.clone();
    let event_type_string = event_type.to_string();
    let subscription_id = global_registry().event_bus().subscribe(
        event_type_string.clone(),
        Arc::new(move |event| {
            Python::with_gil(|py| {
                let payload = match plugin_event_to_py(py, event) {
                    Ok(payload) => payload,
                    Err(err) => {
                        err.print(py);
                        return;
                    }
                };
                let mut queue = match queue_clone.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                if queue.len() >= maxlen {
                    queue.pop_front();
                }
                queue.push_back(payload);
            });
        }),
    );

    Ok(PyPluginQueue {
        event_type: event_type_string,
        subscription_id,
        maxlen,
        queue,
        closed: false,
    })
}

#[pyfunction]
fn event_types() -> HashMap<String, String> {
    [
        ("SystemInit", "System initialization event"),
        ("SystemShutdown", "System shutdown event"),
        ("PluginLoaded", "Plugin loaded event"),
        ("PluginUnloaded", "Plugin unloaded event"),
        ("TensorOp", "Tensor operation completed"),
        ("TensorOpMeta", "Tensor operation metadata emitted"),
        ("EpochStart", "Training epoch started"),
        ("EpochEnd", "Training epoch completed"),
        ("BackendChanged", "Backend changed"),
        ("Telemetry", "Telemetry data emitted"),
        ("ZSpaceTrace", "Z-space trace event"),
        ("*", "Wildcard subscription (all events)"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

#[pyfunction]
#[pyo3(signature = (plugin, *, replace=false))]
fn register_python_plugin(py: Python<'_>, plugin: PyObject, replace: bool) -> PyResult<String> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    let metadata = {
        let plugin_any = plugin.bind(py);
        plugin_metadata_from_py(&plugin_any)?
    };
    let plugin_id = metadata.id.clone();

    if replace && global_registry().get(&plugin_id).is_some() {
        global_registry()
            .unregister(&plugin_id)
            .map_err(tensor_err_to_py)?;
    }

    let wrapper = PythonPlugin::new(metadata, plugin);
    global_registry()
        .register(Box::new(wrapper))
        .map_err(tensor_err_to_py)?;

    Ok(plugin_id)
}

#[pyfunction]
#[pyo3(signature = (plugin_id))]
fn unregister_plugin(plugin_id: &str) -> PyResult<()> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    global_registry()
        .unregister(plugin_id)
        .map_err(tensor_err_to_py)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (plugin_id))]
fn plugin_metadata(py: Python<'_>, plugin_id: &str) -> PyResult<Option<PyObject>> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    let Some(handle) = global_registry().get(plugin_id) else {
        return Ok(None);
    };
    let meta = handle.metadata();
    Ok(Some(plugin_metadata_to_py(py, &meta)?))
}

#[pyfunction]
#[pyo3(signature = (capability))]
fn find_by_capability(capability: &str) -> PyResult<Vec<String>> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    let cap = parse_capability(capability);
    Ok(global_registry()
        .find_by_capability(&cap)
        .into_iter()
        .map(|handle| handle.metadata().id)
        .collect())
}

#[pyfunction]
#[pyo3(signature = (key))]
fn get_config(key: &str) -> PyResult<Option<String>> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    let ctx = global_registry().context();
    let ctx = ctx
        .lock()
        .map_err(|_| PyTypeError::new_err("plugin context lock was poisoned"))?;
    Ok(ctx.get_config(key))
}

#[pyfunction]
#[pyo3(signature = (key, value))]
fn set_config(key: &str, value: &str) -> PyResult<()> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    let ctx = global_registry().context();
    let ctx = ctx
        .lock()
        .map_err(|_| PyTypeError::new_err("plugin context lock was poisoned"))?;
    ctx.set_config(key, value);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (name, service))]
fn register_service(name: &str, service: PyObject) -> PyResult<()> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    let ctx = global_registry().context();
    let ctx = ctx
        .lock()
        .map_err(|_| PyTypeError::new_err("plugin context lock was poisoned"))?;
    ctx.register_service(name, Mutex::new(service));
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (name))]
fn get_service(py: Python<'_>, name: &str) -> PyResult<Option<PyObject>> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    let ctx = global_registry().context();
    let ctx = ctx
        .lock()
        .map_err(|_| PyTypeError::new_err("plugin context lock was poisoned"))?;
    let Some(service) = ctx.get_service::<Mutex<Py<PyAny>>>(name) else {
        return Ok(None);
    };
    let obj = match service.lock() {
        Ok(guard) => guard.clone_ref(py),
        Err(poisoned) => poisoned.into_inner().clone_ref(py),
    };
    Ok(Some(obj))
}

#[pyfunction]
fn list_services() -> PyResult<Vec<String>> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    let ctx = global_registry().context();
    let ctx = ctx
        .lock()
        .map_err(|_| PyTypeError::new_err("plugin context lock was poisoned"))?;
    Ok(ctx.list_services())
}

fn file_stem_module_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("plugin");
    }
    if out
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_digit())
    {
        out.insert(0, '_');
    }
    out
}

fn module_name_for_path(prefix: &str, path: &Path) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    let hash = hasher.finish();

    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(file_stem_module_name)
        .unwrap_or_else(|| "plugin".to_string());
    format!("{prefix}_{stem}_{hash:016x}")
}

fn collect_python_files(path: &Path, recursive: bool) -> PyResult<Vec<PathBuf>> {
    let mut out = Vec::new();

    if path.is_file() {
        if path.extension().and_then(|ext| ext.to_str()) != Some("py") {
            return Err(PyValueError::new_err(
                "load_path expects a .py file or directory containing .py files",
            ));
        }
        out.push(path.to_path_buf());
        return Ok(out);
    }

    if !path.is_dir() {
        return Err(PyValueError::new_err(
            "load_path expects a filesystem path pointing to a .py file or directory",
        ));
    }

    fn visit(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> PyResult<()> {
        let entries = std::fs::read_dir(dir).map_err(|err| {
            PyValueError::new_err(format!("failed to read directory '{}': {err}", dir.display()))
        })?;
        for entry in entries {
            let entry = entry.map_err(|err| {
                PyValueError::new_err(format!("failed to read directory entry: {err}"))
            })?;
            let path = entry.path();
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            if file_name.starts_with('.') {
                continue;
            }
            if path.is_dir() {
                if !recursive {
                    continue;
                }
                if file_name == "__pycache__" {
                    continue;
                }
                visit(&path, recursive, out)?;
                continue;
            }
            if !path.is_file() {
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some("py") {
                continue;
            }
            if file_name == "__init__.py" {
                continue;
            }
            out.push(path);
        }
        Ok(())
    }

    visit(path, recursive, &mut out)?;

    out.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    out.dedup();
    Ok(out)
}

fn load_module_from_file<'py>(
    py: Python<'py>,
    module_name: &str,
    file_path: &Path,
    reload: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let sys = PyModule::import_bound(py, "sys")?;
    let modules_any = sys.getattr("modules")?;
    let modules: &Bound<'py, PyDict> = modules_any.downcast()?;

    if !reload {
        if let Some(existing) = modules.get_item(module_name)? {
            return Ok(existing);
        }
    }

    let path_string = file_path.to_string_lossy().to_string();

    let importlib_util = PyModule::import_bound(py, "importlib.util")?;
    let spec = importlib_util.call_method1(
        "spec_from_file_location",
        (module_name, path_string.clone()),
    )?;
    if spec.is_none() {
        return Err(PyValueError::new_err(format!(
            "failed to build import spec for '{module_name}'"
        )));
    }

    let module = importlib_util.call_method1("module_from_spec", (spec.clone(),))?;
    modules.set_item(module_name, module.clone())?;

    if reload {
        // Avoid importing stale bytecode from __pycache__ when a file is edited in-place.
        // (On Windows this can happen when edits occur within the same 1-second timestamp window.)
        let importlib_machinery = PyModule::import_bound(py, "importlib.machinery")?;
        let source_loader = importlib_machinery
            .getattr("SourceFileLoader")?
            .call1((module_name, path_string.clone()))?;
        let source = source_loader.call_method1("get_source", (module_name,))?;
        if source.is_none() {
            return Err(PyValueError::new_err(format!(
                "failed to read plugin source for '{module_name}'"
            )));
        }

        let builtins = PyModule::import_bound(py, "builtins")?;
        let code = builtins
            .getattr("compile")?
            .call1((source, path_string, "exec"))?;
        let module_dict = module.getattr("__dict__")?;
        let module_dict_obj = module_dict.to_object(py);
        builtins.getattr("exec")?.call1((
            code,
            module_dict_obj.clone_ref(py),
            module_dict_obj,
        ))?;
    } else {
        let loader = spec.getattr("loader")?;
        if loader.is_none() {
            return Err(PyValueError::new_err(format!(
                "import spec for '{module_name}' has no loader"
            )));
        }
        loader.call_method1("exec_module", (module.clone(),))?;
    }
    Ok(module)
}

fn maybe_instantiate<'py>(
    py: Python<'py>,
    value: Bound<'py, PyAny>,
    instantiate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if !instantiate || !value.is_callable() {
        return Ok(value);
    }
    match value.call0() {
        Ok(instance) => Ok(instance),
        Err(err) => {
            if err.is_instance_of::<PyTypeError>(py) {
                Ok(value)
            } else {
                Err(err)
            }
        }
    }
}

fn collect_plugins_from_module<'py>(
    py: Python<'py>,
    module: &Bound<'py, PyAny>,
    instantiate: bool,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let candidates = [
        "__spiraltorch_plugins__",
        "spiraltorch_plugins",
        "plugins",
    ];
    for name in candidates {
        if let Ok(value) = module.getattr(name) {
            if value.is_none() {
                continue;
            }
            let iter = PyIterator::from_bound_object(&value).map_err(|_| {
                PyTypeError::new_err(format!(
                    "module attribute '{name}' must be an iterable of plugins"
                ))
            })?;
            let mut out = Vec::new();
            for item in iter {
                out.push(maybe_instantiate(py, item?, instantiate)?);
            }
            return Ok(out);
        }
    }

    let factories = ["create_plugins", "get_plugins"];
    for name in factories {
        if let Ok(factory) = module.getattr(name) {
            if factory.is_none() {
                continue;
            }
            if !factory.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "module attribute '{name}' must be callable"
                )));
            }
            let value = factory.call0()?;
            if value.is_none() {
                return Ok(Vec::new());
            }
            let iter = PyIterator::from_bound_object(&value).map_err(|_| {
                PyTypeError::new_err(format!(
                    "module factory '{name}()' must return an iterable of plugins"
                ))
            })?;
            let mut out = Vec::new();
            for item in iter {
                out.push(maybe_instantiate(py, item?, instantiate)?);
            }
            return Ok(out);
        }
    }

    let singles = ["create_plugin", "get_plugin", "plugin", "PLUGIN"];
    for name in singles {
        if let Ok(value) = module.getattr(name) {
            if value.is_none() {
                continue;
            }
            let plugin = maybe_instantiate(py, value, instantiate)?;
            return Ok(vec![plugin]);
        }
    }

    Ok(Vec::new())
}

#[pyfunction]
fn initialize_all() -> PyResult<()> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    global_registry()
        .initialize_all()
        .map_err(tensor_err_to_py)?;
    Ok(())
}

#[pyfunction]
fn shutdown() -> PyResult<()> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    global_registry().shutdown().map_err(tensor_err_to_py)?;
    Ok(())
}

#[pyclass(
    module = "spiraltorch.plugin",
    name = "PluginEventRecorder",
    unsendable
)]
pub struct PyPluginEventRecorder {
    inner: Option<PluginEventRecorder>,
}

impl PyPluginEventRecorder {
    fn inner(&self) -> PyResult<&PluginEventRecorder> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PluginEventRecorder is closed"))
    }
}

#[pymethods]
impl PyPluginEventRecorder {
    #[pyo3(signature = (capacity=2048))]
    #[new]
    fn new(capacity: usize) -> PyResult<Self> {
        init_plugin_system().map_err(tensor_err_to_py)?;
        let bus = global_registry().event_bus().clone();
        let inner = PluginEventRecorder::subscribe(
            bus,
            PluginEventRecorderConfig {
                capacity: capacity.max(1),
            },
        );
        Ok(Self { inner: Some(inner) })
    }

    pub fn close(&mut self) -> PyResult<bool> {
        init_plugin_system().map_err(tensor_err_to_py)?;
        Ok(self.inner.take().is_some())
    }

    pub fn elapsed_ms(&self) -> PyResult<u64> {
        Ok(self.inner()?.elapsed_ms())
    }

    pub fn clear(&self) -> PyResult<()> {
        self.inner()?.clear();
        Ok(())
    }

    pub fn snapshot_json(&self) -> PyResult<String> {
        let trace = self.inner()?.snapshot();
        serde_json::to_string_pretty(&trace)
            .map_err(|err| PyValueError::new_err(format!("failed to encode snapshot: {err}")))
    }

    pub fn snapshot(&self, py: Python<'_>) -> PyResult<PyObject> {
        let trace = self.inner()?.snapshot();
        let value = serde_json::to_value(trace)
            .map_err(|err| PyValueError::new_err(format!("failed to encode snapshot: {err}")))?;
        json_to_py(py, &value)
    }

    pub fn write_jsonl(&self, path: &str) -> PyResult<()> {
        self.inner()?.write_jsonl(path).map_err(tensor_err_to_py)?;
        Ok(())
    }

    #[pyo3(signature = (max_nodes=256))]
    pub fn to_mermaid(&self, max_nodes: usize) -> PyResult<String> {
        Ok(self.inner()?.to_mermaid_flowchart(max_nodes))
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let _ = slf.inner.take();
        Ok(())
    }
}

#[pyclass(
    module = "spiraltorch.plugin",
    name = "PluginEventJsonlWriter",
    unsendable
)]
pub struct PyPluginEventJsonlWriter {
    inner: Option<PluginEventJsonlWriter>,
}

impl PyPluginEventJsonlWriter {
    fn inner(&self) -> PyResult<&PluginEventJsonlWriter> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PluginEventJsonlWriter is closed"))
    }
}

#[pymethods]
impl PyPluginEventJsonlWriter {
    #[pyo3(signature = (path, *, capture_tensor_ops=false))]
    #[new]
    fn new(path: &str, capture_tensor_ops: bool) -> PyResult<Self> {
        init_plugin_system().map_err(tensor_err_to_py)?;
        let bus = global_registry().event_bus().clone();
        let inner = PluginEventJsonlWriter::subscribe(
            bus,
            path,
            PluginEventJsonlWriterConfig { capture_tensor_ops },
        )
        .map_err(tensor_err_to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    pub fn close(&mut self) -> PyResult<bool> {
        init_plugin_system().map_err(tensor_err_to_py)?;
        Ok(self.inner.take().is_some())
    }

    pub fn elapsed_ms(&self) -> PyResult<u64> {
        Ok(self.inner()?.elapsed_ms())
    }

    pub fn flush(&self) -> PyResult<()> {
        self.inner()?.flush().map_err(tensor_err_to_py)?;
        Ok(())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let _ = slf.inner.take();
        Ok(())
    }
}

#[pyfunction]
fn list_plugins() -> PyResult<Vec<String>> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    Ok(global_registry().list_plugins())
}

#[pyfunction]
#[pyo3(signature = (event_type, payload=None))]
fn publish(py: Python<'_>, event_type: &str, payload: Option<PyObject>) -> PyResult<()> {
    init_plugin_system().map_err(tensor_err_to_py)?;

    let event = match payload {
        None => PluginEvent::custom(event_type, serde_json::Value::Null),
        Some(payload) => {
            let payload = payload.bind(py);
            if payload.is_none() {
                PluginEvent::custom(event_type, serde_json::Value::Null)
            } else if let Ok(text) = payload.extract::<String>() {
                PluginEvent::custom(event_type, text)
            } else {
                PluginEvent::custom(event_type, py_to_json(&payload)?)
            }
        }
    };

    global_registry().event_bus().publish(&event);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (event_type))]
fn has_listeners(event_type: &str) -> PyResult<bool> {
    init_plugin_system().map_err(tensor_err_to_py)?;
    Ok(global_registry().event_bus().has_listeners(event_type))
}

#[pyfunction]
#[pyo3(signature = (group="spiraltorch.plugins", *, instantiate=true, replace=false))]
fn load_entrypoints(py: Python<'_>, group: &str, instantiate: bool, replace: bool) -> PyResult<Vec<String>> {
    init_plugin_system().map_err(tensor_err_to_py)?;

    let importlib_metadata = PyModule::import_bound(py, "importlib.metadata")?;
    let entry_points = importlib_metadata.getattr("entry_points")?.call0()?;

    let selected = if let Ok(select) = entry_points.getattr("select") {
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("group", group)?;
        select.call((), Some(&kwargs))?
    } else if let Ok(dict) = entry_points.downcast::<PyDict>() {
        match dict.get_item(group)? {
            Some(value) => value,
            None => py.None().into_bound(py),
        }
    } else {
        return Err(PyTypeError::new_err(
            "entry_points() returned an unsupported type (expected EntryPoints or dict)",
        ));
    };

    if selected.is_none() {
        return Ok(Vec::new());
    }

    let iter = PyIterator::from_bound_object(&selected).map_err(|_| {
        PyTypeError::new_err("entry_points() result is not iterable for the selected group")
    })?;

    let mut out = Vec::new();
    for item in iter {
        let item = item?;
        let loaded = item.getattr("load")?.call0()?;
        let plugin_obj = if instantiate && loaded.is_callable() {
            match loaded.call0() {
                Ok(instance) => instance,
                Err(err) => {
                    if err.is_instance_of::<PyTypeError>(py) {
                        loaded
                    } else {
                        return Err(err);
                    }
                }
            }
        } else {
            loaded
        };
        let plugin_id = register_python_plugin(py, plugin_obj.unbind(), replace)?;
        out.push(plugin_id);
    }

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (path, *, recursive=true, instantiate=true, strict=false, reload=false, replace=false, module_prefix="spiraltorch_path_plugin", add_sys_path=true))]
fn load_path(
    py: Python<'_>,
    path: &str,
    recursive: bool,
    instantiate: bool,
    strict: bool,
    reload: bool,
    replace: bool,
    module_prefix: &str,
    add_sys_path: bool,
) -> PyResult<Vec<String>> {
    init_plugin_system().map_err(tensor_err_to_py)?;

    let base = PathBuf::from(path);
    let files = collect_python_files(&base, recursive)?;
    if files.is_empty() {
        if strict {
            return Err(PyValueError::new_err(format!(
                "no python files found under '{}'. expected at least one .py file",
                base.display()
            )));
        }
        return Ok(Vec::new());
    }

    let sys = PyModule::import_bound(py, "sys")?;
    let sys_path_any = sys.getattr("path")?;
    let sys_path: &Bound<'_, PyList> = sys_path_any.downcast()?;
    if add_sys_path {
        let mut dirs: Vec<String> = files
            .iter()
            .filter_map(|p| p.parent())
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        dirs.sort();
        dirs.dedup();

        let mut existing = Vec::new();
        for item in sys_path.iter() {
            let Ok(text) = item.extract::<String>() else {
                continue;
            };
            existing.push(text);
        }

        for dir in dirs.into_iter().rev() {
            if existing.iter().any(|value| value == &dir) {
                continue;
            }
            sys_path.insert(0, &dir)?;
        }
    }

    if reload {
        PyModule::import_bound(py, "importlib")?
            .getattr("invalidate_caches")?
            .call0()?;
    }

    (|| -> PyResult<Vec<String>> {
        let mut ids = Vec::new();
        for file_path in files {
            let module_name = module_name_for_path(module_prefix, &file_path);
            let module = load_module_from_file(py, &module_name, &file_path, reload)?;
            let plugins = collect_plugins_from_module(py, &module, instantiate)?;
            if plugins.is_empty() {
                if strict {
                    return Err(PyValueError::new_err(format!(
                        "no plugins discovered in '{}'",
                        file_path.display()
                    )));
                }
                continue;
            }
            for plugin in plugins {
                let plugin_id = register_python_plugin(py, plugin.unbind(), replace)?;
                ids.push(plugin_id);
            }
        }
        Ok(ids)
    })()
}

#[pyfunction]
#[pyo3(signature = (path, *, recursive=true, instantiate=true, strict=false, module_prefix="spiraltorch_path_plugin", add_sys_path=true))]
fn reload_path(
    py: Python<'_>,
    path: &str,
    recursive: bool,
    instantiate: bool,
    strict: bool,
    module_prefix: &str,
    add_sys_path: bool,
) -> PyResult<Vec<String>> {
    load_path(
        py,
        path,
        recursive,
        instantiate,
        strict,
        true,
        true,
        module_prefix,
        add_sys_path,
    )
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "plugin")?;
    module.add("__doc__", "SpiralTorch plugin registry + event observability")?;
    module.add_class::<PyPluginQueue>()?;
    module.add_class::<PyPluginEventRecorder>()?;
    module.add_class::<PyPluginEventJsonlWriter>()?;
    module.add_function(wrap_pyfunction!(subscribe, &module)?)?;
    module.add_function(wrap_pyfunction!(subscribe_many, &module)?)?;
    module.add_function(wrap_pyfunction!(unsubscribe, &module)?)?;
    module.add_function(wrap_pyfunction!(unsubscribe_many, &module)?)?;
    module.add_function(wrap_pyfunction!(listen, &module)?)?;
    module.add_function(wrap_pyfunction!(event_types, &module)?)?;
    module.add_function(wrap_pyfunction!(list_plugins, &module)?)?;
    module.add_function(wrap_pyfunction!(register_python_plugin, &module)?)?;
    module.add_function(wrap_pyfunction!(unregister_plugin, &module)?)?;
    module.add_function(wrap_pyfunction!(plugin_metadata, &module)?)?;
    module.add_function(wrap_pyfunction!(find_by_capability, &module)?)?;
    module.add_function(wrap_pyfunction!(get_config, &module)?)?;
    module.add_function(wrap_pyfunction!(set_config, &module)?)?;
    module.add_function(wrap_pyfunction!(register_service, &module)?)?;
    module.add_function(wrap_pyfunction!(get_service, &module)?)?;
    module.add_function(wrap_pyfunction!(list_services, &module)?)?;
    module.add_function(wrap_pyfunction!(initialize_all, &module)?)?;
    module.add_function(wrap_pyfunction!(shutdown, &module)?)?;
    module.add_function(wrap_pyfunction!(publish, &module)?)?;
    module.add_function(wrap_pyfunction!(has_listeners, &module)?)?;
    module.add_function(wrap_pyfunction!(load_entrypoints, &module)?)?;
    module.add_function(wrap_pyfunction!(load_path, &module)?)?;
    module.add_function(wrap_pyfunction!(reload_path, &module)?)?;
    module.add(
        "__all__",
        vec![
            "PluginQueue",
            "PluginEventRecorder",
            "PluginEventJsonlWriter",
            "subscribe",
            "subscribe_many",
            "unsubscribe",
            "unsubscribe_many",
            "listen",
            "event_types",
            "list_plugins",
            "register_python_plugin",
            "unregister_plugin",
            "plugin_metadata",
            "find_by_capability",
            "get_config",
            "set_config",
            "register_service",
            "get_service",
            "list_services",
            "initialize_all",
            "shutdown",
            "publish",
            "has_listeners",
            "load_entrypoints",
            "load_path",
            "reload_path",
        ],
    )?;

    parent.add_submodule(&module)?;
    parent.add("plugin", module.to_object(py))?;
    Ok(())
}
