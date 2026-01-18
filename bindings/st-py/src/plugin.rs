use pyo3::exceptions::{PyStopIteration, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule};
use pyo3::wrap_pyfunction;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::tensor::tensor_err_to_py;
use st_core::plugin::{global_registry, init_plugin_system, EventListener, PluginEvent};

fn plugin_event_to_py(py: Python<'_>, event: &PluginEvent) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
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
        }
    }
    Ok(dict.into_py(py))
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

    Ok(global_registry().event_bus().subscribe(event_type.to_string(), listener))
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
    let iter = PyIterator::from_bound_object(event_types).map_err(|_| {
        PyTypeError::new_err("event_types must be a string or iterable of strings")
    })?;
    let mut out = Vec::new();
    for item in iter {
        let item = item?;
        let name: String = item.extract().map_err(|_| {
            PyTypeError::new_err("event_types must contain only strings")
        })?;
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
        let mut queue = self.queue.lock().map_err(|_| {
            PyTypeError::new_err("event queue lock was poisoned")
        })?;
        Ok(queue.pop_front().map(|item| item.into_py(py)))
    }

    #[pyo3(signature = (max_items=None))]
    pub fn drain(&self, py: Python<'_>, max_items: Option<usize>) -> PyResult<Vec<PyObject>> {
        let mut queue = self.queue.lock().map_err(|_| {
            PyTypeError::new_err("event queue lock was poisoned")
        })?;
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

    fn __len__(&self) -> PyResult<usize> {
        let queue = self.queue.lock().map_err(|_| {
            PyTypeError::new_err("event queue lock was poisoned")
        })?;
        Ok(queue.len())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut queue = self.queue.lock().map_err(|_| {
            PyTypeError::new_err("event queue lock was poisoned")
        })?;
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
        ("EpochStart", "Training epoch started"),
        ("EpochEnd", "Training epoch completed"),
        ("BackendChanged", "Backend changed"),
        ("Telemetry", "Telemetry data emitted"),
        ("*", "Wildcard subscription (all events)"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "plugin")?;
    module.add("__doc__", "SpiralTorch plugin/event observability")?;
    module.add_class::<PyPluginQueue>()?;
    module.add_function(wrap_pyfunction!(subscribe, &module)?)?;
    module.add_function(wrap_pyfunction!(subscribe_many, &module)?)?;
    module.add_function(wrap_pyfunction!(unsubscribe, &module)?)?;
    module.add_function(wrap_pyfunction!(unsubscribe_many, &module)?)?;
    module.add_function(wrap_pyfunction!(listen, &module)?)?;
    module.add_function(wrap_pyfunction!(event_types, &module)?)?;
    module.add(
        "__all__",
        vec![
            "PluginQueue",
            "subscribe",
            "subscribe_many",
            "unsubscribe",
            "unsubscribe_many",
            "listen",
            "event_types",
        ],
    )?;

    parent.add_submodule(&module)?;
    parent.add("plugin", module.to_object(py))?;
    Ok(())
}
