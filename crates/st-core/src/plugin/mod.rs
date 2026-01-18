// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Unified plugin/extension system for the SpiralTorch ecosystem.
//!
//! This module provides a pluggable architecture that allows components to be
//! dynamically discovered, loaded, and composed at runtime. The system supports:
//!
//! - **Plugin discovery**: Automatic registration of plugins via the registry
//! - **Lifecycle hooks**: Plugins can respond to initialization, events, and shutdown
//! - **Type-safe communication**: Event bus for inter-plugin messaging
//! - **Dependency resolution**: Plugins can declare dependencies on other plugins
//! - **Hot-reloading**: Support for dynamic plugin updates (where safe)
//!
//! # Examples
//!
//! ```rust,ignore
//! use st_core::plugin::{Plugin, PluginRegistry, PluginContext, PluginMetadata};
//! use st_core::PureResult;
//!
//! struct MyPlugin;
//!
//! impl Plugin for MyPlugin {
//!     fn metadata(&self) -> PluginMetadata {
//!         PluginMetadata::new("my_plugin", "1.0.0")
//!             .with_description("An example plugin")
//!     }
//!
//!     fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
//!         println!("MyPlugin loaded!");
//!         Ok(())
//!     }
//! }
//!
//! let mut registry = PluginRegistry::new();
//! registry.register(Box::new(MyPlugin))?;
//! registry.initialize_all()?;
//! ```

pub mod registry;
pub mod traits;
pub mod events;
pub mod loader;
pub mod context;

pub use registry::{PluginRegistry, PluginHandle};
pub use traits::{Plugin, PluginMetadata, PluginCapability};
pub use events::{PluginEvent, PluginEventBus, EventListener};
pub use loader::{DynamicPluginLoader, PluginLoader, StaticPluginLoader};
pub use context::{PluginContext, PluginDependency};

use crate::PureResult;
use std::sync::Arc;
use st_tensor::TensorOpEvent;
use st_tensor::set_tensor_op_observer;

/// Initialize the global plugin system.
///
/// This function should be called once at application startup to set up the
/// global plugin registry and event bus.
pub fn init_plugin_system() -> PureResult<()> {
    let _ = global_registry();
    Ok(())
}

/// Get a reference to the global plugin registry.
pub fn global_registry() -> &'static PluginRegistry {
    let registry = GLOBAL_REGISTRY.get_or_init(PluginRegistry::new);
    ensure_tensor_op_bridge(registry.event_bus().clone());
    registry
}

fn ensure_tensor_op_bridge(bus: PluginEventBus) {
    let _ = TENSOR_OP_BRIDGE.get_or_init(|| {
        let bus = bus.clone();
        set_tensor_op_observer(Some(Arc::new(move |event: &TensorOpEvent| {
            bus.publish(&PluginEvent::TensorOp {
                op_name: event.op_name.to_string(),
                input_shape: event.input_shape.clone(),
                output_shape: event.output_shape.clone(),
            });
        })));
    });
}

use std::sync::OnceLock;
static GLOBAL_REGISTRY: OnceLock<PluginRegistry> = OnceLock::new();
static TENSOR_OP_BRIDGE: OnceLock<()> = OnceLock::new();
