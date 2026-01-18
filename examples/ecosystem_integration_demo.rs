// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Comprehensive ecosystem integration example.
//!
//! This demonstrates how the plugin system, module discovery, and dynamic
//! composition work together to create an organically pluggable ecosystem.

use st_core::plugin::{
    global_registry, init_plugin_system, Plugin, PluginCapability, PluginContext, PluginEvent,
    PluginLoader, PluginMetadata, StaticPluginLoader,
};
use st_core::PureResult;
use st_nn::{
    Linear, Module, ModuleCategory, ModuleDiscoveryRegistry, ModuleMetadata, Relu, Sequential,
};
use std::any::Any;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// A custom layer plugin that registers with both systems.
struct CustomLayerPlugin {
    registry: Arc<ModuleDiscoveryRegistry>,
}

impl CustomLayerPlugin {
    fn new() -> Self {
        Self {
            registry: Arc::new(ModuleDiscoveryRegistry::new()),
        }
    }

    fn register_custom_layers(&self) -> PureResult<()> {
        // Register a custom activation function
        let activation_metadata = ModuleMetadata {
            module_type: "swish".to_string(),
            display_name: "Swish Activation".to_string(),
            description: "x * sigmoid(x) activation function".to_string(),
            input_shape: None,
            output_shape: None,
            parameters: HashMap::new(),
            category: ModuleCategory::Activation,
        };

        let swish_factory = Arc::new(|_config: &HashMap<String, String>| {
            // In practice, would create actual Swish module
            println!("  🎯 Creating Swish activation module");
            Ok(Box::new(Relu::new()) as Box<dyn Module>)
        });

        self.registry.register(activation_metadata, swish_factory)?;

        println!("  ✅ Registered custom layers");
        Ok(())
    }
}

impl Plugin for CustomLayerPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata::new("custom_layer_plugin", "1.0.0")
            .with_name("Custom Layer Plugin")
            .with_description("Provides custom neural network layers")
            .with_capability(PluginCapability::Operators)
            .with_capability(PluginCapability::Custom("Layers".to_string()))
    }

    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        println!("🔧 Loading Custom Layer Plugin");
        
        // Register custom layers in the module discovery system
        self.register_custom_layers()?;
        
        // Make the module registry available as a service
        ctx.register_service("module_registry", Arc::clone(&self.registry));
        
        // Subscribe to module creation events
        ctx.subscribe("ModuleRegistered", Arc::new(|event| {
            if let PluginEvent::Custom { event_type, .. } = event {
                println!("  📢 Module registered: {}", event_type);
            }
        }));
        
        println!("✅ Custom Layer Plugin loaded");
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// An optimizer plugin that integrates with the ecosystem.
struct OptimizerPlugin;

impl Plugin for OptimizerPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata::new("optimizer_plugin", "1.0.0")
            .with_name("Advanced Optimizers")
            .with_description("Provides AdamW, Lion, and other modern optimizers")
            .with_capability(PluginCapability::Optimizers)
    }

    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        println!("🚀 Loading Optimizer Plugin");
        
        // In practice, would register optimizer factories
        ctx.set_config("default_optimizer", "adamw");
        ctx.set_config("default_lr", "0.001");
        
        println!("✅ Optimizer Plugin loaded");
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A monitoring plugin that tracks system activity.
struct MonitoringPlugin {
    event_count: usize,
}

impl MonitoringPlugin {
    fn new() -> Self {
        Self { event_count: 0 }
    }
}

impl Plugin for MonitoringPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata::new("monitoring_plugin", "1.0.0")
            .with_name("System Monitor")
            .with_description("Tracks and logs ecosystem activity")
            .with_capability(PluginCapability::Telemetry)
    }

    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        println!("📊 Loading Monitoring Plugin");
        
        // Subscribe to all events for monitoring
        ctx.subscribe("*", Arc::new(move |event| {
            match event {
                PluginEvent::EpochStart { epoch } => {
                    println!("  📈 Monitoring: Epoch {} started", epoch);
                }
                PluginEvent::EpochEnd { epoch, loss } => {
                    println!("  📉 Monitoring: Epoch {} completed with loss {:.4}", epoch, loss);
                }
                _ => {}
            }
        }));
        
        println!("✅ Monitoring Plugin loaded");
        Ok(())
    }

    fn on_event(&mut self, _event: &PluginEvent, _ctx: &PluginContext) -> PureResult<()> {
        self.event_count += 1;
        Ok(())
    }

    fn on_unload(&mut self, _ctx: &mut PluginContext) -> PureResult<()> {
        println!("  📊 Total events monitored: {}", self.event_count);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn create_ecosystem_plugins() -> Vec<Box<dyn Plugin>> {
    vec![
        Box::new(MonitoringPlugin::new()),
        Box::new(CustomLayerPlugin::new()),
        Box::new(OptimizerPlugin),
    ]
}

fn main() -> PureResult<()> {
    println!("🌀 SpiralTorch Ecosystem Integration Demo 🌀\n");
    println!("Demonstrating organic pluggability and dynamic composition\n");

    // Initialize the global plugin system
    init_plugin_system()?;
    let registry = global_registry();

    // Load all ecosystem plugins
    println!("📦 Loading ecosystem plugins...\n");
    let loader = StaticPluginLoader::new(create_ecosystem_plugins);
    let loaded_count = loader.load_into(Path::new("."), registry)?;
    
    println!("\n✅ Loaded {} plugins", loaded_count);
    
    // Initialize the ecosystem
    println!("\n🎬 Initializing ecosystem...\n");
    registry.initialize_all()?;

    // Demonstrate plugin discovery
    println!("\n🔍 Discovering plugins by capability...\n");
    
    let operator_plugins = registry.find_by_capability(&PluginCapability::Operators);
    println!("  Operator plugins: {}", operator_plugins.len());
    
    let optimizer_plugins = registry.find_by_capability(&PluginCapability::Optimizers);
    println!("  Optimizer plugins: {}", optimizer_plugins.len());
    
    let telemetry_plugins = registry.find_by_capability(&PluginCapability::Telemetry);
    println!("  Telemetry plugins: {}", telemetry_plugins.len());

    // Access the module registry service
    println!("\n🏗️  Building dynamic model pipeline...\n");
    
    let ctx = registry.context();
    let ctx_guard = ctx.lock().unwrap();
    
    if let Some(module_registry) = ctx_guard.get_service::<Arc<ModuleDiscoveryRegistry>>("module_registry") {
        println!("  📋 Available modules:");
        for module_type in module_registry.list_modules() {
            if let Some(meta) = module_registry.get_metadata(&module_type) {
                println!("    - {} ({})", meta.display_name, meta.module_type);
            }
        }
        
        // Build a pipeline dynamically
        println!("\n  🔨 Constructing pipeline from discovered modules...");
        
        let mut model = Sequential::new();
        model.push(Linear::new("input", 10, 20)?);
        model.push(Relu::new());
        model.push(Linear::new("hidden", 20, 10)?);
        
        println!("  ✅ Pipeline created with {} layers", model.len());
    }
    
    // Show configuration
    if let Some(optimizer) = ctx_guard.get_config("default_optimizer") {
        println!("\n⚙️  Default optimizer: {}", optimizer);
    }
    
    drop(ctx_guard);

    // Simulate training events
    println!("\n🏃 Simulating training workflow...\n");
    
    let event_bus = registry.event_bus();
    
    event_bus.publish(&PluginEvent::EpochStart { epoch: 1 });
    std::thread::sleep(std::time::Duration::from_millis(100));
    event_bus.publish(&PluginEvent::EpochEnd { epoch: 1, loss: 0.543 });
    
    event_bus.publish(&PluginEvent::EpochStart { epoch: 2 });
    std::thread::sleep(std::time::Duration::from_millis(100));
    event_bus.publish(&PluginEvent::EpochEnd { epoch: 2, loss: 0.421 });

    // Demonstrate inter-plugin communication
    println!("\n💬 Demonstrating inter-plugin communication...\n");
    
    #[derive(Clone)]
    struct CustomMetrics {
        accuracy: f32,
        precision: f32,
    }
    
    let metrics = CustomMetrics {
        accuracy: 0.92,
        precision: 0.89,
    };
    
    event_bus.publish(&PluginEvent::custom("CustomMetrics", metrics));
    
    // Shutdown
    println!("\n🛑 Shutting down ecosystem...\n");
    registry.shutdown()?;

    println!("\n✨ Ecosystem integration demo completed!\n");
    println!("Key features demonstrated:");
    println!("  ✓ Unified plugin registration and discovery");
    println!("  ✓ Dynamic module composition");
    println!("  ✓ Service registry for inter-plugin communication");
    println!("  ✓ Event-based monitoring and telemetry");
    println!("  ✓ Organic ecosystem integration\n");

    Ok(())
}
