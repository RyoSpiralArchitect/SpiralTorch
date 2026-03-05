# SpiralTorch Plugin System

## Overview

The SpiralTorch plugin system provides a unified, type-safe mechanism for extending the framework with new capabilities. The system is designed to be:

- **Organic**: Plugins can discover and communicate with each other
- **Safe**: Type-safe interfaces with compile-time guarantees
- **Flexible**: Supports both static and dynamic loading
- **Observable**: Rich event system for monitoring and telemetry

## Architecture

The plugin system consists of several key components:

### 1. Plugin Trait

The core `Plugin` trait that all plugins must implement:

```rust
pub trait Plugin: Send + Sync {
    fn metadata(&self) -> PluginMetadata;
    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()>;
    fn on_unload(&mut self, ctx: &mut PluginContext) -> PureResult<()>;
    fn on_event(&mut self, event: &PluginEvent, ctx: &PluginContext) -> PureResult<()>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
```

### 2. Plugin Registry

The central registry that manages all loaded plugins:

```rust
pub struct PluginRegistry {
    plugins: RwLock<HashMap<String, PluginHandle>>,
    context: Arc<Mutex<PluginContext>>,
    event_bus: PluginEventBus,
}
```

### 3. Event System

A publish-subscribe event bus for inter-plugin communication:

```rust
pub enum PluginEvent {
    SystemInit,
    SystemShutdown,
    PluginLoaded { plugin_id: String },
    TensorOp { op_name: String, ... },
    EpochStart { epoch: usize },
    Custom { event_type: String, data: Arc<dyn Any> },
    // ...
}
```

### 4. Plugin Context

Provides plugins with access to configuration and services:

```rust
pub struct PluginContext {
    pub event_bus: PluginEventBus,
    config: Arc<Mutex<HashMap<String, String>>>,
    services: Arc<Mutex<HashMap<String, Arc<dyn Any>>>>,
}
```

## Creating a Plugin

### Basic Example

```rust
use st_core::plugin::{Plugin, PluginMetadata, PluginContext, PluginCapability};
use st_core::PureResult;
use std::any::Any;

struct MyOperatorPlugin {
    name: String,
}

impl Plugin for MyOperatorPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata::new("my_operator_plugin", "1.0.0")
            .with_name("My Operator Plugin")
            .with_description("Adds custom tensor operators")
            .with_author("Your Name")
            .with_capability(PluginCapability::Operators)
    }

    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        println!("Loading operator plugin: {}", self.name);
        
        // Register a service that other plugins can use
        ctx.register_service("my_operator", MyOperatorService::new());
        
        // Subscribe to tensor operation events
        ctx.subscribe("TensorOp", Arc::new(|event| {
            println!("Tensor op event: {:?}", event);
        }));
        
        Ok(())
    }

    fn on_unload(&mut self, _ctx: &mut PluginContext) -> PureResult<()> {
        println!("Unloading operator plugin");
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
```

### Registering Plugins

```rust
use st_core::plugin::{PluginRegistry, init_plugin_system};

// Initialize the global plugin system
init_plugin_system()?;

// Get the registry
let registry = global_registry();

// Register plugins
registry.register(Box::new(MyOperatorPlugin {
    name: "my_op".to_string(),
}))?;

// Initialize all plugins
registry.initialize_all()?;
```

## Plugin Capabilities

Plugins can declare what capabilities they provide:

```rust
pub enum PluginCapability {
    Backend(String),      // e.g., Backend("CUDA"), Backend("WGPU")
    Operators,            // Tensor operators
    LossFunctions,        // Loss functions for training
    Optimizers,           // Optimization algorithms
    DataLoaders,          // Data loading and preprocessing
    Visualization,        // Visualization and plotting
    Telemetry,            // Monitoring and metrics
    Language,             // NLP and language models
    Vision,               // Computer vision
    ReinforcementLearning, // RL environments and agents
    GraphNeuralNetworks,  // GNN components
    Recommender,          // Recommendation systems
    Custom(String),       // Custom capabilities
}
```

## Plugin Discovery

Plugins can be discovered in several ways:

### 1. Static Registration

Compile plugins into the binary and register them explicitly:

```rust
fn create_plugins() -> Vec<Box<dyn Plugin>> {
    vec![
        Box::new(MyPlugin1),
        Box::new(MyPlugin2),
        Box::new(MyPlugin3),
    ]
}

let loader = StaticPluginLoader::new(create_plugins);
loader.load_into(Path::new("."), registry)?;
```

### 2. Dynamic Loading

Load plugins from shared libraries at runtime (requires careful ABI management):

```rust
let loader = DynamicPluginLoader::new();
loader.load_into(Path::new("./plugins"), registry)?;
```

### 3. Python Plugins (SpiralTorch Python bindings)

When using the Python bindings (`pip install spiraltorch`), you can register and
discover **Python plugins** via `spiraltorch.plugin`.

```python
import spiraltorch as st

class DemoPlugin:
    def metadata(self) -> dict:
        return {"id": "demo_py_plugin", "version": "0.0.1", "capabilities": ["Telemetry"]}

    def on_load(self) -> None:
        st.plugin.set_config("demo_py_plugin.enabled", "1")
        st.plugin.register_service("demo_py_plugin.service", self)

    def on_event(self, event: dict) -> None:
        print("event:", event.get("type"))

st.plugin.register_python_plugin(DemoPlugin())
st.plugin.publish("HelloPlugin", {"ok": True})
```

Directory/module discovery helpers:

- `st.plugin.load_entrypoints(group="spiraltorch.plugins", instantiate=True, replace=False)` discovers plugins from Python entry points.
- `st.plugin.reload_entrypoints(group="spiraltorch.plugins", instantiate=True, strict=False)` reloads plugins by unloading the previously discovered entrypoint plugins for that group and then calling `load_entrypoints(..., replace=True)`.
- `st.plugin.unload_entrypoints(group="spiraltorch.plugins", strict=False)` unregisters plugins previously discovered via `load_entrypoints(...)` for that group.
- `st.plugin.load_path(path, recursive=True, strict=False, reload=False, replace=False, ...)` loads `.py` files from a directory (or a single file).
- `st.plugin.reload_path(path, ...)` is a convenience wrapper that does `reload=True, replace=True`.
- `st.plugin.watch_path(path, poll_interval=0.25, debounce=0.0, missing_grace=0.0, unload_on_stop=False, ...)` polls a directory/file and hot-reloads changed plugin files.
- `st.plugin.unload_path(path, recursive=True, strict=False)` unregisters plugins previously loaded from `load_path(...)` under a filesystem path.
- `st.plugin.unload_all(strict=False)` unregisters all currently registered plugins in reverse dependency order.
- `st.plugin.clear_services(prefix=None, strict=False)` unregisters services from the plugin context (optionally filtering by name prefix).

Entry point demo package: see `examples/python_entrypoint_plugin_demo/`.

Dependency-aware loading:

- `load_path(...)` registers plugins in dependency order when `metadata.dependencies` is present.
- With `replace=True`, SpiralTorch unregisters matched plugins in reverse dependency order before re-registering.
- `load_entrypoints(...)` uses the same dependency ordering logic when multiple plugins are discovered in a single call.

Loader-provided metadata:

- `load_path(...)` / `reload_path(...)` annotate `PluginMetadata.metadata` with `spiraltorch.source="path"`, plus `spiraltorch.source_path` (absolute `.py` file path) and `spiraltorch.source_module` (generated module name).
- `load_entrypoints(...)` annotates `PluginMetadata.metadata` with `spiraltorch.source="entrypoint"` and entry point fields (`spiraltorch.entrypoint_group`, `spiraltorch.entrypoint_name`, `spiraltorch.entrypoint_value`) when available.

Plugin discovery conventions for `load_path(...)`:

- A module can expose an iterable of plugin objects via `__spiraltorch_plugins__`, `spiraltorch_plugins`, or `plugins`.
- Or expose factories: `create_plugins()` / `get_plugins()` (return iterable).
- Or expose a single plugin via `create_plugin()` / `get_plugin()` / `plugin` / `PLUGIN`.

Python plugin event auto-subscription:

- If a plugin defines `on_event(self, event: dict)`, SpiralTorch auto-subscribes it to `"*"` (all events).
- Override by defining `event_types = ["TensorOp", ...]` (or an `event_types()` callable).
- Disable by setting `auto_subscribe = False` (or `event_types = []`).

## Event System

Plugins can communicate via the event bus:

### Publishing Events

```rust
// Publish a built-in event
ctx.event_bus.publish(&PluginEvent::EpochStart { epoch: 1 });

// Publish a custom event
#[derive(Clone)]
struct MyCustomData {
    value: i32,
}

let event = PluginEvent::custom("MyEvent", MyCustomData { value: 42 });
ctx.event_bus.publish(&event);
```

### Subscribing to Events

```rust
fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
    // Subscribe to specific events
    ctx.subscribe("EpochStart", Arc::new(|event| {
        if let PluginEvent::EpochStart { epoch } = event {
            println!("Epoch {} started", epoch);
        }
    }));
    
    // Subscribe to all events
    ctx.subscribe("*", Arc::new(|event| {
        println!("Event: {:?}", event);
    }));
    
    Ok(())
}
```

## Service Registry

Plugins can register services for other plugins to use:

### Registering a Service

```rust
trait MyService: Send + Sync {
    fn do_something(&self) -> String;
}

struct MyServiceImpl;

impl MyService for MyServiceImpl {
    fn do_something(&self) -> String {
        "Hello from service!".to_string()
    }
}

fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
    ctx.register_service("my_service", MyServiceImpl as Arc<dyn MyService>);
    Ok(())
}
```

### Using a Service

```rust
fn use_service(ctx: &PluginContext) {
    if let Some(service) = ctx.get_service::<Arc<dyn MyService>>("my_service") {
        println!("Service says: {}", service.do_something());
    }
}
```

### Unregistering a Service

```rust
fn on_unload(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
    ctx.unregister_service("my_service");
    Ok(())
}
```

Python bindings also expose `st.plugin.unregister_service("my_service")` (returns a bool indicating whether a service was removed) and `st.plugin.clear_services(prefix="my_prefix.")` (returns a list of removed service names).

## Plugin Dependencies

Plugins can declare dependencies on other plugins:

```rust
fn metadata(&self) -> PluginMetadata {
    PluginMetadata::new("my_plugin", "1.0.0")
        .with_dependency("base_plugin", ">=1.0.0")
        .with_dependency("utils_plugin", "^2.0.0")
}
```

The registry will validate that all dependencies are satisfied before loading a plugin.

## Best Practices

1. **Keep plugins focused**: Each plugin should have a single, well-defined purpose
2. **Use capabilities**: Declare what your plugin provides so others can discover it
3. **Handle errors gracefully**: Always return `PureResult` and handle errors properly
4. **Clean up resources**: Implement `on_unload` to properly clean up
5. **Document your plugin**: Provide clear metadata and documentation
6. **Version carefully**: Use semantic versioning for plugin versions
7. **Test thoroughly**: Write tests for your plugin's functionality

## Example: Complete Backend Plugin

```rust
use st_core::plugin::*;
use st_core::PureResult;

struct WgpuBackendPlugin {
    initialized: bool,
}

impl Plugin for WgpuBackendPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata::new("wgpu_backend", "1.0.0")
            .with_name("WGPU Backend")
            .with_description("WebGPU backend for portable GPU compute")
            .with_author("SpiralTorch Team")
            .with_capability(PluginCapability::Backend("WGPU".to_string()))
            .with_metadata("gpu_api", "WebGPU")
            .with_metadata("platforms", "Windows,Linux,macOS")
    }

    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        println!("Initializing WGPU backend...");
        
        // Initialize WGPU
        self.initialized = true;
        
        // Register backend service
        ctx.register_service("wgpu_device", create_wgpu_device()?);
        
        // Subscribe to backend change events
        ctx.subscribe("BackendChanged", Arc::new(|event| {
            if let PluginEvent::BackendChanged { backend } = event {
                println!("Backend changed to: {}", backend);
            }
        }));
        
        println!("WGPU backend ready");
        Ok(())
    }

    fn on_unload(&mut self, _ctx: &mut PluginContext) -> PureResult<()> {
        println!("Shutting down WGPU backend");
        self.initialized = false;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
```

## Integration with Existing Systems

The plugin system integrates seamlessly with SpiralTorch's existing features:

- **ZSpaceSequencer plugins**: Use `ZSpaceSequencerPlugin` for coherence sequencing
- **Module plugins**: Extend `st-nn` modules with custom layers
- **Backend plugins**: Add new compute backends (CUDA, ROCm, etc.)
- **Operator plugins**: Register custom tensor operations
- **Telemetry plugins**: Add monitoring and observability

## Future Extensions

- Hot-reloading support for development workflows (Python path plugins: `reload_path(...)` / `watch_path(...)` available; native hot reload is planned)
- Plugin sandboxing for security
- Dependency version resolution
- Plugin marketplace/registry
- Cross-language plugins (Python, Julia, Go)
