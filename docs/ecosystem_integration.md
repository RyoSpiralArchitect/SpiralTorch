# SpiralTorch Ecosystem Integration Guide

## Overview

SpiralTorch's ecosystem is designed to be **organically pluggable**, meaning components can be discovered, loaded, and composed dynamically at runtime. This guide explains how the various systems work together to create a flexible, extensible framework.

## Core Systems

### 1. Plugin System (`st-core::plugin`)

The foundation of ecosystem integration. Provides:

- **Plugin Registry**: Central registry for all plugins
- **Event Bus**: Pub/sub messaging between components
- **Service Registry**: Shared services across plugins
- **Lifecycle Management**: Load, initialize, and shutdown hooks

```rust
use st_core::plugin::{init_plugin_system, global_registry, Plugin};

// Initialize once at startup
init_plugin_system()?;

// Access anywhere
let registry = global_registry();
registry.register(my_plugin)?;
```

### 2. Module Discovery (`st-nn::discovery`)

Dynamic module registration and instantiation:

- **Module Metadata**: Describe module capabilities
- **Module Factories**: Create modules from configuration
- **Pipeline Builder**: Compose modules into sequential models

```rust
use st_nn::{ModuleDiscoveryRegistry, ModulePipelineBuilder};

let registry = ModuleDiscoveryRegistry::new();

// Register a module factory
registry.register(metadata, factory)?;

// Build a pipeline
let model = ModulePipelineBuilder::new(Arc::new(registry))
    .add_module("linear", config)?
    .add_module("relu", HashMap::new())?
    .build()?;
```

### 3. ZSpace Sequencer Integration

Bridge between ZSpace plugins and the unified system:

```rust
use st_nn::zspace_coherence::{ZSpacePluginAdapter, ZSpacePluginRegistry};

// Register a ZSpace sequencer plugin
ZSpacePluginRegistry::register(my_sequencer_plugin)?;

// Find all registered sequencers
let sequencers = ZSpacePluginRegistry::find_all_sequencers();
```

## Integration Patterns

### Pattern 1: Plugin with Service

Create a plugin that provides a service to other plugins:

```rust
struct MyServicePlugin {
    service: Arc<MyService>,
}

impl Plugin for MyServicePlugin {
    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        // Register service
        ctx.register_service("my_service", Arc::clone(&self.service));
        
        // Subscribe to events
        ctx.subscribe("SomeEvent", Arc::new(|event| {
            // Handle event
        }));
        
        Ok(())
    }
}
```

### Pattern 2: Module Registration Plugin

A plugin that registers neural network modules:

```rust
struct LayerPlugin {
    module_registry: Arc<ModuleDiscoveryRegistry>,
}

impl Plugin for LayerPlugin {
    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        // Register modules
        self.register_custom_layers()?;
        
        // Share registry as service
        ctx.register_service("module_registry", 
                            Arc::clone(&self.module_registry));
        
        Ok(())
    }
}
```

### Pattern 3: Event-Based Monitoring

Monitor ecosystem activity through events:

```rust
struct MonitorPlugin {
    metrics: Arc<Mutex<Metrics>>,
}

impl Plugin for MonitorPlugin {
    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        let metrics = Arc::clone(&self.metrics);
        
        ctx.subscribe("*", Arc::new(move |event| {
            metrics.lock().unwrap().record(event);
        }));
        
        Ok(())
    }
}
```

## Ecosystem Capabilities

Plugins can declare and discover capabilities:

```rust
pub enum PluginCapability {
    Backend(String),          // GPU/CPU backends
    Operators,                // Tensor operations
    LossFunctions,           // Training objectives
    Optimizers,              // Optimization algorithms
    DataLoaders,             // Data pipelines
    Visualization,           // Plotting and dashboards
    Telemetry,              // Monitoring and metrics
    Language,               // NLP/LLM components
    Vision,                 // Computer vision
    ReinforcementLearning,  // RL environments
    GraphNeuralNetworks,    // GNN layers
    Recommender,            // RecSys components
    Custom(String),         // Domain-specific
}
```

### Finding Plugins by Capability

```rust
// Find all optimizer plugins
let optimizers = registry.find_by_capability(
    &PluginCapability::Optimizers
);

// Find CUDA backend
let cuda_backends = registry.find_by_capability(
    &PluginCapability::Backend("CUDA".to_string())
);
```

## Configuration Management

Plugins can store and retrieve configuration:

```rust
fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
    // Set configuration
    ctx.set_config("learning_rate", "0.001");
    ctx.set_config("batch_size", "32");
    
    // Read configuration
    if let Some(lr) = ctx.get_config("learning_rate") {
        println!("Using learning rate: {}", lr);
    }
    
    Ok(())
}
```

## Event System

### Built-in Events

```rust
pub enum PluginEvent {
    SystemInit,
    SystemShutdown,
    PluginLoaded { plugin_id: String },
    PluginUnloaded { plugin_id: String },
    TensorOp { op_name: String, ... },
    EpochStart { epoch: usize },
    EpochEnd { epoch: usize, loss: f32 },
    BackendChanged { backend: String },
    Telemetry { data: HashMap<String, f32> },
    Custom { event_type: String, data: Arc<dyn Any> },
}
```

### Publishing Events

```rust
// Publish built-in event
event_bus.publish(&PluginEvent::EpochStart { epoch: 1 });

// Publish custom event
#[derive(Clone)]
struct MyData {
    value: i32,
}

event_bus.publish(&PluginEvent::custom("MyEvent", MyData { value: 42 }));
```

### Subscribing to Events

```rust
// Subscribe to specific event
ctx.subscribe("EpochStart", Arc::new(|event| {
    if let PluginEvent::EpochStart { epoch } = event {
        println!("Epoch {} started", epoch);
    }
}));

// Subscribe to all events
ctx.subscribe("*", Arc::new(|event| {
    println!("Event: {:?}", event);
}));
```

## Dynamic Module Composition

### Registering Module Factories

```rust
let registry = ModuleDiscoveryRegistry::new();

let metadata = ModuleMetadata {
    module_type: "custom_conv".to_string(),
    display_name: "Custom Convolution".to_string(),
    description: "Specialized convolution layer".to_string(),
    input_shape: Some(vec![3, 224, 224]),
    output_shape: Some(vec![64, 224, 224]),
    parameters: HashMap::new(),
    category: ModuleCategory::Convolution,
};

let factory = Arc::new(|config: &HashMap<String, String>| {
    let in_channels = config.get("in_channels")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    
    let out_channels = config.get("out_channels")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    
    // Create and return the module
    Ok(Box::new(MyCustomConv::new(in_channels, out_channels)?) 
       as Box<dyn Module>)
});

registry.register(metadata, factory)?;
```

### Building Pipelines

```rust
let pipeline = ModulePipelineBuilder::new(Arc::new(registry))
    .with_name("my_model")
    .add_module("custom_conv", hashmap! {
        "in_channels" => "3",
        "out_channels" => "64",
    })?
    .add_module("relu", HashMap::new())?
    .add_module("maxpool", hashmap! {
        "kernel_size" => "2",
    })?
    .build()?;
```

## Best Practices

### 1. Plugin Design

- **Single Responsibility**: Each plugin should have one clear purpose
- **Declare Dependencies**: Use `with_dependency()` for required plugins
- **Advertise Capabilities**: Let others discover what you provide
- **Clean Shutdown**: Release resources in `on_unload()`

### 2. Event Handling

- **Specific Subscriptions**: Subscribe to specific events when possible
- **Fast Handlers**: Keep event handlers lightweight
- **Error Handling**: Always handle errors gracefully
- **Thread Safety**: Event handlers must be `Send + Sync`

### 3. Service Registry

- **Type Safety**: Use strong types for services
- **Documentation**: Document service interfaces clearly
- **Version Compatibility**: Consider API versioning
- **Cleanup**: Remove services on unload

### 4. Module Registration

- **Rich Metadata**: Provide complete module descriptions
- **Validation**: Validate configuration in factories
- **Error Messages**: Give clear error messages
- **Examples**: Include usage examples in metadata

## Complete Example

See `examples/ecosystem_integration_demo.rs` for a complete working example that demonstrates:

- Plugin registration and discovery
- Module factory registration
- Dynamic pipeline composition
- Inter-plugin communication via events
- Service sharing between plugins
- Configuration management
- Lifecycle management

## Integration with Existing Features

### ZSpace Coherence Sequencer

```rust
use st_nn::zspace_coherence::ZSpacePluginRegistry;

// Register existing ZSpace plugins
ZSpacePluginRegistry::register(my_coherence_plugin)?;

// They're now discoverable through the unified system
let sequencers = global_registry().find_by_capability(
    &PluginCapability::Custom("ZSpaceSequencer".to_string())
);
```

### Backend Plugins

```rust
struct WgpuBackendPlugin;

impl Plugin for WgpuBackendPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata::new("wgpu", "1.0.0")
            .with_capability(PluginCapability::Backend("WGPU".to_string()))
    }
    
    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        // Initialize WGPU
        let device = create_wgpu_device()?;
        ctx.register_service("wgpu_device", device);
        Ok(())
    }
}
```

### Training Plugins

```rust
struct TrainingPlugin;

impl Plugin for TrainingPlugin {
    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        ctx.subscribe("EpochEnd", Arc::new(|event| {
            if let PluginEvent::EpochEnd { epoch, loss } = event {
                // Save checkpoint, update learning rate, etc.
            }
        }));
        Ok(())
    }
}
```

## Future Enhancements

- **Hot Reloading**: Dynamic plugin updates without restart
- **Plugin Marketplace**: Centralized plugin discovery
- **Cross-Language Plugins**: Python, Julia, Go plugin support
- **Plugin Sandboxing**: Security isolation
- **Dependency Resolution**: Automatic dependency management

## Resources

- [Plugin System Documentation](plugin_system.md)
- [Ecosystem Roadmap](ecosystem_roadmap.md)
- [API Reference](https://docs.rs/st-core/latest/st_core/plugin/)
- [Example Code](../examples/)
