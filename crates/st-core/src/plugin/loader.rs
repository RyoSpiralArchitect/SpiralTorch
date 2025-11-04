// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Plugin loading mechanisms.

use super::registry::PluginRegistry;
use super::traits::Plugin;
use crate::PureResult;
use std::path::Path;

/// Plugin loader trait for discovering and instantiating plugins.
pub trait PluginLoader {
    /// Discover plugins from a given path.
    fn discover(&self, path: &Path) -> PureResult<Vec<Box<dyn Plugin>>>;

    /// Load all discovered plugins into a registry.
    fn load_into(&self, path: &Path, registry: &PluginRegistry) -> PureResult<usize> {
        let plugins = self.discover(path)?;
        let count = plugins.len();
        
        for plugin in plugins {
            registry.register(plugin)?;
        }
        
        Ok(count)
    }
}

/// Dynamic plugin loader for loading plugins from shared libraries.
///
/// Note: This requires the `libloading` crate and careful ABI management.
/// Currently implemented as a placeholder for static linking scenarios.
pub struct DynamicPluginLoader;

impl DynamicPluginLoader {
    /// Create a new dynamic plugin loader.
    pub fn new() -> Self {
        Self
    }
}

impl Default for DynamicPluginLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginLoader for DynamicPluginLoader {
    fn discover(&self, _path: &Path) -> PureResult<Vec<Box<dyn Plugin>>> {
        // Placeholder: Dynamic loading would require unsafe code and libloading
        // For now, plugins should be registered statically
        Ok(Vec::new())
    }
}

/// Registry-based plugin loader that discovers plugins from a factory function.
///
/// This is the recommended approach for most use cases, where plugins are
/// compiled into the binary and registered via a factory function.
pub struct StaticPluginLoader {
    factory: fn() -> Vec<Box<dyn Plugin>>,
}

impl StaticPluginLoader {
    /// Create a new static plugin loader with a factory function.
    pub fn new(factory: fn() -> Vec<Box<dyn Plugin>>) -> Self {
        Self { factory }
    }
}

impl PluginLoader for StaticPluginLoader {
    fn discover(&self, _path: &Path) -> PureResult<Vec<Box<dyn Plugin>>> {
        Ok((self.factory)())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::traits::{Plugin, PluginMetadata};
    use std::any::Any;

    struct TestPlugin;

    impl Plugin for TestPlugin {
        fn metadata(&self) -> PluginMetadata {
            PluginMetadata::new("test", "1.0.0")
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    fn test_factory() -> Vec<Box<dyn Plugin>> {
        vec![Box::new(TestPlugin)]
    }

    #[test]
    fn test_static_loader() {
        let loader = StaticPluginLoader::new(test_factory);
        let plugins = loader.discover(Path::new(".")).unwrap();
        assert_eq!(plugins.len(), 1);
    }

    #[test]
    fn test_load_into_registry() {
        let loader = StaticPluginLoader::new(test_factory);
        let registry = PluginRegistry::new();
        
        let count = loader.load_into(Path::new("."), &registry).unwrap();
        assert_eq!(count, 1);
        assert!(registry.get("test").is_some());
    }
}
