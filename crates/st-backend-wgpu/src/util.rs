// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::{
    borrow::Cow,
    collections::{hash_map::Entry, HashMap},
    fs,
    path::{Path, PathBuf},
};

use thiserror::Error;
use wgpu::{
    ComputePipeline, ComputePipelineDescriptor, Device, ShaderModuleDescriptor, ShaderSource,
};

/// Errors that may occur when loading WGSL shaders into a WGPU pipeline.
#[derive(Debug, Error)]
pub enum ShaderLoadError {
    /// The WGSL source file could not be read from disk.
    #[error("failed to read WGSL shader '{path}'")]
    Io {
        /// The underlying IO error raised by the filesystem.
        #[source]
        source: std::io::Error,
        /// The path that could not be read.
        path: PathBuf,
    },
    /// A requested pipeline override constant was not found in the WGSL source.
    #[error("shader override '{name}' not found in '{file}'")]
    OverrideNotFound {
        /// The override constant name that could not be located.
        name: String,
        /// The WGSL file path searched for the override.
        file: PathBuf,
    },
}

/// Read a WGSL shader relative to `shader_dir` and return the source string.
pub fn read_wgsl(shader_dir: impl AsRef<Path>, file: &str) -> Result<String, ShaderLoadError> {
    let path = shader_dir.as_ref().join(file);
    fs::read_to_string(&path).map_err(|source| ShaderLoadError::Io { source, path })
}

/// Lightweight cache for WGSL shader sources stored on disk.
///
/// Many of SpiralTorch's compute pipelines are composed of multiple entry
/// points that live in separate shader files. When bootstrapping the backend
/// we often need to read several of these files in quick succession. The cache
/// keeps the decoded UTF-8 source in memory so subsequent pipeline
/// construction avoids touching the filesystem again.
#[derive(Debug, Default)]
pub struct ShaderCache {
    shader_dir: PathBuf,
    sources: HashMap<PathBuf, String>,
}

impl ShaderCache {
    /// Create a cache rooted at `shader_dir`.
    pub fn new(shader_dir: impl Into<PathBuf>) -> Self {
        Self {
            shader_dir: shader_dir.into(),
            sources: HashMap::new(),
        }
    }

    /// Return the absolute directory containing cached shaders.
    pub fn shader_dir(&self) -> &Path {
        &self.shader_dir
    }

    /// Drop all cached shader sources.
    pub fn clear(&mut self) {
        self.sources.clear();
    }

    /// Retrieve the WGSL source for `file`, reading it from disk if necessary.
    pub fn source(&mut self, file: &str) -> Result<&str, ShaderLoadError> {
        let path = self.shader_dir.join(file);
        match self.sources.entry(path.clone()) {
            Entry::Occupied(entry) => Ok(entry.into_mut().as_str()),
            Entry::Vacant(entry) => {
                let source = fs::read_to_string(&path).map_err(|source| ShaderLoadError::Io {
                    source,
                    path: path.clone(),
                })?;
                Ok(entry.insert(source).as_str())
            }
        }
    }

    /// Load a compute pipeline by file name using the cached shader source.
    pub fn load_compute_pipeline(
        &mut self,
        device: &Device,
        file: &str,
        label: &str,
        entry_point: &str,
    ) -> Result<ComputePipeline, ShaderLoadError> {
        self.load_compute_pipeline_with_layout(device, file, label, entry_point, None)
    }

    /// Variant of [`ShaderCache::load_compute_pipeline`] that accepts an explicit layout.
    pub fn load_compute_pipeline_with_layout(
        &mut self,
        device: &Device,
        file: &str,
        label: &str,
        entry_point: &str,
        layout: Option<&wgpu::PipelineLayout>,
    ) -> Result<ComputePipeline, ShaderLoadError> {
        self.load_compute_pipeline_with_layout_and_overrides(
            device,
            file,
            label,
            entry_point,
            layout,
            &[],
        )
    }

    /// Variant of [`ShaderCache::load_compute_pipeline_with_layout`] that applies WGSL overrides.
    pub fn load_compute_pipeline_with_layout_and_overrides(
        &mut self,
        device: &Device,
        file: &str,
        label: &str,
        entry_point: &str,
        layout: Option<&wgpu::PipelineLayout>,
        overrides: &[(&str, u32)],
    ) -> Result<ComputePipeline, ShaderLoadError> {
        let specialized = if overrides.is_empty() {
            Cow::Borrowed(self.source(file)?)
        } else {
            Cow::Owned(apply_overrides(self.source(file)?, file, overrides)?)
        };
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(specialized),
        });
        Ok(device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout,
            module: &module,
            entry_point,
        }))
    }

    /// Ensure that all `files` are loaded into the cache.
    pub fn prefetch<I, S>(&mut self, files: I) -> Result<(), ShaderLoadError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for file in files {
            self.source(file.as_ref())?;
        }
        Ok(())
    }
}

/// Load a WGSL shader and build a compute pipeline for the provided entry point.
///
/// The helper keeps module creation consistent across SpiralTorch's WGPU kernels
/// and eliminates repetitive boilerplate from individual pipeline constructors.
pub fn load_compute_pipeline(
    device: &Device,
    shader_dir: impl AsRef<Path>,
    file: &str,
    label: &str,
    entry_point: &str,
) -> Result<ComputePipeline, ShaderLoadError> {
    load_compute_pipeline_with_layout(device, shader_dir, file, label, entry_point, None)
}

/// Variant of [`load_compute_pipeline`] that allows the caller to specify a
/// pipeline layout.
pub fn load_compute_pipeline_with_layout(
    device: &Device,
    shader_dir: impl AsRef<Path>,
    file: &str,
    label: &str,
    entry_point: &str,
    layout: Option<&wgpu::PipelineLayout>,
) -> Result<ComputePipeline, ShaderLoadError> {
    let source = read_wgsl(shader_dir, file)?;
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    });
    Ok(device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout,
        module: &module,
        entry_point,
    }))
}

fn apply_overrides(
    source: &str,
    file: &str,
    overrides: &[(&str, u32)],
) -> Result<String, ShaderLoadError> {
    let mut output = String::with_capacity(source.len() + overrides.len() * 16);
    let mut remaining: Vec<(&str, u32, bool)> = overrides
        .iter()
        .map(|(name, value)| (*name, *value, false))
        .collect();

    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("override ") {
            let mut replaced = false;
            for (name, value, seen) in remaining.iter_mut() {
                let target = format!("override {name} ");
                if trimmed.starts_with(&target) {
                    let prefix_len = line.len() - trimmed.len();
                    output.push_str(&line[..prefix_len]);
                    output.push_str(&format!("override {name} : u32 = {value}u;"));
                    replaced = true;
                    *seen = true;
                    break;
                }
            }
            if !replaced {
                output.push_str(line);
            }
        } else {
            output.push_str(line);
        }
        output.push('\n');
    }

    if let Some((name, _, _)) = remaining.iter().find(|(_, _, seen)| !*seen) {
        return Err(ShaderLoadError::OverrideNotFound {
            name: (*name).to_string(),
            file: PathBuf::from(file),
        });
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_DIR_ID: AtomicUsize = AtomicUsize::new(0);

    struct TempShaderDir {
        path: PathBuf,
    }

    impl TempShaderDir {
        fn new() -> Self {
            let mut path = std::env::temp_dir();
            let unique = NEXT_DIR_ID.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock before UNIX_EPOCH")
                .as_nanos();
            path.push(format!("spiraltorch-wgpu-cache-{unique}-{nanos}"));
            fs::create_dir(&path).expect("failed to create temporary shader directory");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }

        fn write(&self, name: &str, contents: &str) {
            fs::write(self.path.join(name), contents).expect("failed to write shader file");
        }
    }

    impl Drop for TempShaderDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn cache_reuses_source_without_retouching_disk() {
        let temp = TempShaderDir::new();
        temp.write("example.wgsl", "// first pass\n");

        let mut cache = ShaderCache::new(temp.path());
        let first = cache.source("example.wgsl").expect("initial read");
        assert_eq!(first, "// first pass\n");

        temp.write("example.wgsl", "// second pass\n");
        let cached = cache.source("example.wgsl").expect("cached read");
        assert_eq!(cached, "// first pass\n");
    }

    #[test]
    fn prefetch_warms_cache_for_multiple_files() {
        let temp = TempShaderDir::new();
        temp.write("a.wgsl", "// A\n");
        temp.write("b.wgsl", "// B\n");

        let mut cache = ShaderCache::new(temp.path());
        cache
            .prefetch(["a.wgsl", "b.wgsl"])
            .expect("prefetch should succeed");

        assert_eq!(cache.source("a.wgsl").unwrap(), "// A\n");
        assert_eq!(cache.source("b.wgsl").unwrap(), "// B\n");
    }

    #[test]
    fn apply_overrides_replaces_matching_constants() {
        let shader = "override WG_ROWS : u32 = 16u;\noverride WG_COLS : u32 = 16u;\n";
        let specialized = apply_overrides(shader, "test.wgsl", &[("WG_ROWS", 32), ("WG_COLS", 8)])
            .expect("override application");
        assert!(specialized.contains("override WG_ROWS : u32 = 32u;"));
        assert!(specialized.contains("override WG_COLS : u32 = 8u;"));
    }

    #[test]
    fn apply_overrides_errors_when_constant_missing() {
        let shader = "override WG_ROWS : u32 = 16u;\n";
        let err = apply_overrides(shader, "test.wgsl", &[("WG_COLS", 8)]).unwrap_err();
        match err {
            ShaderLoadError::OverrideNotFound { name, file } => {
                assert_eq!(name, "WG_COLS");
                assert_eq!(file, PathBuf::from("test.wgsl"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
