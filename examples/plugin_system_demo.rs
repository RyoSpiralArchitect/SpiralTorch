// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Example demonstrating the SpiralTorch plugin system.
//!
//! This example shows how to create and register custom plugins that extend
//! the SpiralTorch ecosystem with new capabilities.

use st_core::plugin::{
    Plugin, PluginCapability, PluginContext, PluginEvent, PluginMetadata, PluginRegistry,
    PluginLoader, StaticPluginLoader,
};
use st_core::PureResult;
use std::any::Any;
use std::path::Path;
use std::sync::Arc;

/// A custom operator plugin that adds specialized tensor operations.
struct CustomOperatorPlugin {
    operation_count: usize,
}

impl CustomOperatorPlugin {
    fn new() -> Self {
        Self {
            operation_count: 0,
        }
    }

    fn increment_operations(&mut self) {
        self.operation_count += 1;
    }
}

impl Plugin for CustomOperatorPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata::new("custom_operator_plugin", "1.0.0")
            .with_name("Custom Operator Plugin")
            .with_description("Provides specialized tensor operations for domain-specific use cases")
            .with_author("SpiralTorch Example")
            .with_capability(PluginCapability::Operators)
            .with_metadata("operation_types", "fft,convolution,pooling")
    }

    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        println!("🔧 Loading Custom Operator Plugin");

        // Register a configuration value
        ctx.set_config("custom_op_enabled", "true");

        // Subscribe to tensor operation events
        ctx.subscribe(
            "TensorOp",
            Arc::new(|event| {
                if let PluginEvent::TensorOp { op_name, .. } = event {
                    println!("   📊 Tensor operation detected: {}", op_name);
                }
            }),
        );

        println!("✅ Custom Operator Plugin loaded successfully");
        Ok(())
    }

    fn on_event(&mut self, event: &PluginEvent, _ctx: &PluginContext) -> PureResult<()> {
        match event {
            PluginEvent::TensorOp { .. } => {
                self.increment_operations();
            }
            _ => {}
        }
        Ok(())
    }

    fn on_unload(&mut self, _ctx: &mut PluginContext) -> PureResult<()> {
        println!(
            "👋 Unloading Custom Operator Plugin (processed {} operations)",
            self.operation_count
        );
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A telemetry plugin that monitors system events.
struct TelemetryPlugin {
    events_received: usize,
}

impl TelemetryPlugin {
    fn new() -> Self {
        Self {
            events_received: 0,
        }
    }
}

impl Plugin for TelemetryPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata::new("telemetry_plugin", "1.0.0")
            .with_name("Telemetry Monitor")
            .with_description("Monitors and logs system events for observability")
            .with_capability(PluginCapability::Telemetry)
    }

    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        println!("📡 Loading Telemetry Plugin");

        // Subscribe to all events
        ctx.subscribe(
            "*",
            Arc::new(|event| {
                println!("   📨 Telemetry received event: {:?}", event);
            }),
        );

        println!("✅ Telemetry Plugin loaded");
        Ok(())
    }

    fn on_event(&mut self, _event: &PluginEvent, _ctx: &PluginContext) -> PureResult<()> {
        self.events_received += 1;
        Ok(())
    }

    fn on_unload(&mut self, _ctx: &mut PluginContext) -> PureResult<()> {
        println!(
            "👋 Unloading Telemetry Plugin (received {} events)",
            self.events_received
        );
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A visualization plugin that depends on the telemetry plugin.
struct VisualizationPlugin;

impl Plugin for VisualizationPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata::new("visualization_plugin", "1.0.0")
            .with_name("Visualization Plugin")
            .with_description("Provides visualization capabilities")
            .with_capability(PluginCapability::Visualization)
            .with_dependency("telemetry_plugin", "1.0.0")
    }

    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        println!("🎨 Loading Visualization Plugin");

        // Subscribe to epoch events
        ctx.subscribe(
            "EpochEnd",
            Arc::new(|event| {
                if let PluginEvent::EpochEnd { epoch, loss } = event {
                    println!("   📈 Visualizing epoch {}: loss = {:.4}", epoch, loss);
                }
            }),
        );

        println!("✅ Visualization Plugin loaded");
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Factory function to create all example plugins.
fn create_example_plugins() -> Vec<Box<dyn Plugin>> {
    vec![
        Box::new(TelemetryPlugin::new()),
        Box::new(CustomOperatorPlugin::new()),
        Box::new(VisualizationPlugin),
    ]
}

fn main() -> PureResult<()> {
    println!("🌀 SpiralTorch Plugin System Example 🌀\n");

    // Create a plugin registry
    let registry = PluginRegistry::new();

    // Use the static loader to load plugins
    let loader = StaticPluginLoader::new(create_example_plugins);
    let loaded_count = loader.load_into(Path::new("."), &registry)?;

    println!("\n📦 Loaded {} plugins", loaded_count);
    println!("   Registered plugins: {:?}\n", registry.list_plugins());

    // Initialize all plugins
    registry.initialize_all()?;

    println!("\n🎯 Simulating events...\n");

    // Emit some events to demonstrate the system
    let event_bus = registry.event_bus();

    // Simulate tensor operations
    event_bus.publish(&PluginEvent::TensorOp {
        op_name: "matmul".to_string(),
        input_shape: vec![32, 64],
        output_shape: vec![32, 128],
    });

    event_bus.publish(&PluginEvent::TensorOp {
        op_name: "softmax".to_string(),
        input_shape: vec![32, 128],
        output_shape: vec![32, 128],
    });

    // Simulate training
    event_bus.publish(&PluginEvent::EpochStart { epoch: 1 });
    event_bus.publish(&PluginEvent::EpochEnd {
        epoch: 1,
        loss: 0.453,
    });

    event_bus.publish(&PluginEvent::EpochStart { epoch: 2 });
    event_bus.publish(&PluginEvent::EpochEnd {
        epoch: 2,
        loss: 0.321,
    });

    // Demonstrate querying plugins by capability
    println!("\n🔍 Querying plugins by capability...\n");

    let operator_plugins = registry.find_by_capability(&PluginCapability::Operators);
    println!(
        "   Operator plugins: {}",
        operator_plugins
            .iter()
            .map(|h| h.metadata().id)
            .collect::<Vec<_>>()
            .join(", ")
    );

    let telemetry_plugins = registry.find_by_capability(&PluginCapability::Telemetry);
    println!(
        "   Telemetry plugins: {}",
        telemetry_plugins
            .iter()
            .map(|h| h.metadata().id)
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Access plugin context
    let ctx = registry.context();
    let ctx_guard = ctx.lock().unwrap();
    if let Some(enabled) = ctx_guard.get_config("custom_op_enabled") {
        println!("\n⚙️  Custom operators enabled: {}", enabled);
    }

    drop(ctx_guard);

    // Shutdown
    println!("\n🛑 Shutting down plugin system...\n");
    registry.shutdown()?;

    println!("✨ Plugin system example completed successfully!\n");

    Ok(())
}
