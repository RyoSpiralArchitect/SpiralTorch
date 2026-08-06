// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::{
    any::Any,
    borrow::Cow,
    collections::{hash_map::Entry, HashMap},
    fmt, fs,
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering as AtomicOrdering},
        Arc, RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

use thiserror::Error;
use wgpu::{
    ComputePipeline, ComputePipelineDescriptor, Device, ShaderModuleDescriptor, ShaderSource,
};

use crate::runtime::Shared;

fn read_recover<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    match lock.read() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let guard = poisoned.into_inner();
            lock.clear_poison();
            guard
        }
    }
}

fn write_recover<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    match lock.write() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let guard = poisoned.into_inner();
            lock.clear_poison();
            guard
        }
    }
}

pub(crate) fn device_supports_subgroup(device: &Device) -> bool {
    device.features().contains(wgpu::Features::SUBGROUP)
}

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
    /// WGSL failed to compile or validate when creating a shader module.
    #[error("failed to compile WGSL shader '{label}' ({context})")]
    Compile {
        /// Label assigned to the shader module.
        label: String,
        /// Additional context, such as the originating file path.
        context: String,
        /// Underlying error reported by WGPU.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The compute pipeline failed to validate when creating the pipeline object.
    #[error("failed to create compute pipeline '{label}' ({context})")]
    Pipeline {
        /// Label assigned to the compute pipeline.
        label: String,
        /// Additional context, such as the originating file path.
        context: String,
        /// Underlying error reported by WGPU.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

fn compile_module<F>(
    device: &Device,
    label: &str,
    source: F,
    context: Option<String>,
) -> Result<Shared<wgpu::ShaderModule>, ShaderLoadError>
where
    F: FnOnce() -> Result<Arc<str>, ShaderLoadError>,
{
    let context = context.unwrap_or_else(|| "inline".to_string());
    let source = source()?;
    let module = catch_unwind(AssertUnwindSafe(|| {
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(Cow::Borrowed(&source)),
        })
    }))
    .map_err(|payload| ShaderLoadError::Compile {
        label: label.to_string(),
        context,
        source: Box::new(ShaderCompileError(panic_payload_to_string(payload))),
    })?;
    Ok(Shared::new(module))
}

#[derive(Debug)]
struct ShaderCompileError(String);

impl fmt::Display for ShaderCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ShaderCompileError {}

#[derive(Debug)]
struct PipelineCreateError(String);

impl fmt::Display for PipelineCreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PipelineCreateError {}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

pub fn create_inline_module(
    device: &Device,
    label: &str,
    source: String,
) -> Result<Shared<wgpu::ShaderModule>, ShaderLoadError> {
    let source_arc: Arc<str> = Arc::from(source);
    compile_module(device, label, move || Ok(source_arc), None)
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
/// construction avoids touching the filesystem again. Device resources stay
/// owned by the dispatcher that constructs them, avoiding cross-device caches.
#[derive(Clone, Default)]
pub struct ShaderCache {
    inner: Arc<ShaderCacheInner>,
}

#[derive(Default)]
struct ShaderCacheInner {
    shader_dir: PathBuf,
    source_generation: AtomicU64,
    sources: RwLock<HashMap<PathBuf, Arc<str>>>,
}

impl ShaderCache {
    /// Create a cache rooted at `shader_dir`.
    pub fn new(shader_dir: impl Into<PathBuf>) -> Self {
        Self {
            inner: Arc::new(ShaderCacheInner {
                shader_dir: shader_dir.into(),
                source_generation: AtomicU64::new(0),
                sources: RwLock::new(HashMap::new()),
            }),
        }
    }

    /// Return the absolute directory containing cached shaders.
    pub fn shader_dir(&self) -> &Path {
        &self.inner.shader_dir
    }

    /// Drop all cached shader sources.
    pub fn clear(&self) {
        let sources = {
            let mut sources = write_recover(&self.inner.sources);
            let previous = std::mem::take(&mut *sources);
            self.inner
                .source_generation
                .fetch_add(1, AtomicOrdering::AcqRel);
            previous
        };
        drop(sources);
    }

    /// Retrieve the WGSL source for `file`, reading it from disk if necessary.
    pub fn source(&self, file: &str) -> Result<Arc<str>, ShaderLoadError> {
        let path = self.inner.shader_dir.join(file);

        self.source_with(path, |path| {
            let source = fs::read_to_string(path).map_err(|source| ShaderLoadError::Io {
                source,
                path: path.to_path_buf(),
            })?;
            Ok(Arc::from(source))
        })
    }

    fn source_with<F>(&self, path: PathBuf, loader: F) -> Result<Arc<str>, ShaderLoadError>
    where
        F: FnOnce(&Path) -> Result<Arc<str>, ShaderLoadError>,
    {
        let generation = self.inner.source_generation.load(AtomicOrdering::Acquire);
        if let Some(source) = read_recover(&self.inner.sources).get(&path) {
            return Ok(Arc::clone(source));
        }

        let source = loader(&path)?;
        let mut sources = write_recover(&self.inner.sources);
        if self.inner.source_generation.load(AtomicOrdering::Acquire) != generation {
            return Ok(source);
        }
        match sources.entry(path.clone()) {
            Entry::Occupied(entry) => Ok(Arc::clone(entry.get())),
            Entry::Vacant(entry) => Ok(Arc::clone(entry.insert(source))),
        }
    }

    /// Load a compute pipeline by file name using the cached shader source.
    pub fn load_compute_pipeline(
        &self,
        device: &Device,
        file: &str,
        label: &str,
        entry_point: &str,
    ) -> Result<Shared<ComputePipeline>, ShaderLoadError> {
        self.load_compute_pipeline_with_layout(device, file, label, entry_point, None)
    }

    /// Variant of [`ShaderCache::load_compute_pipeline`] that accepts an explicit layout.
    pub fn load_compute_pipeline_with_layout(
        &self,
        device: &Device,
        file: &str,
        label: &str,
        entry_point: &str,
        layout: Option<&wgpu::PipelineLayout>,
    ) -> Result<Shared<ComputePipeline>, ShaderLoadError> {
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
        &self,
        device: &Device,
        file: &str,
        label: &str,
        entry_point: &str,
        layout: Option<&wgpu::PipelineLayout>,
        overrides: &[(&str, u32)],
    ) -> Result<Shared<ComputePipeline>, ShaderLoadError> {
        let context = self.inner.shader_dir.join(file).display().to_string();
        let module = compile_module(
            device,
            label,
            || {
                let base = self.source(file)?;
                if overrides.is_empty() {
                    Ok(base)
                } else {
                    apply_overrides(&base, file, overrides)
                }
            },
            Some(context.clone()),
        )?;

        let pipeline = catch_unwind(AssertUnwindSafe(|| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout,
                module: module.as_ref(),
                entry_point,
                compilation_options: Default::default(),
            })
        }))
        .map_err(|payload| ShaderLoadError::Pipeline {
            label: label.to_string(),
            context,
            source: Box::new(PipelineCreateError(panic_payload_to_string(payload))),
        })?;
        Ok(Shared::new(pipeline))
    }

    /// Ensure that all `files` are loaded into the cache.
    pub fn prefetch<I, S>(&self, files: I) -> Result<(), ShaderLoadError>
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

impl fmt::Debug for ShaderCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sources = read_recover(&self.inner.sources)
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        f.debug_struct("ShaderCache")
            .field("shader_dir", &self.inner.shader_dir)
            .field("sources", &sources)
            .finish()
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
    let dir = shader_dir.as_ref();
    let context = dir.join(file).display().to_string();
    let module = compile_module(
        device,
        label,
        || {
            let source = read_wgsl(dir, file)?;
            Ok(Arc::<str>::from(source))
        },
        Some(context),
    )?;
    Ok(device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout,
        module: module.as_ref(),
        entry_point,
        compilation_options: Default::default(),
    }))
}

fn apply_overrides(
    source: &Arc<str>,
    file: &str,
    overrides: &[(&str, u32)],
) -> Result<Arc<str>, ShaderLoadError> {
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

    Ok(Arc::from(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
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

        let cache = ShaderCache::new(temp.path());
        let first = cache.source("example.wgsl").expect("initial read");
        assert_eq!(first.as_ref(), "// first pass\n");

        temp.write("example.wgsl", "// second pass\n");
        let cached = cache.source("example.wgsl").expect("cached read");
        assert_eq!(cached.as_ref(), "// first pass\n");
    }

    #[test]
    fn source_cache_recovers_after_poison() {
        let temp = TempShaderDir::new();
        temp.write("recovery.wgsl", "// recovered\n");
        let cache = ShaderCache::new(temp.path());
        let poisoned = catch_unwind(AssertUnwindSafe(|| {
            let _guard = cache
                .inner
                .sources
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            panic!("poison shader source cache");
        }));
        assert!(poisoned.is_err());
        assert!(cache.inner.sources.is_poisoned());

        let source = cache
            .source("recovery.wgsl")
            .expect("source cache recovered");

        assert_eq!(source.as_ref(), "// recovered\n");
        assert!(!cache.inner.sources.is_poisoned());
    }

    #[test]
    fn source_loader_runs_outside_cache_lock() {
        let cache = ShaderCache::new(".");
        let path = PathBuf::from("injected.wgsl");

        let source = cache
            .source_with(path, |_| {
                assert!(
                    cache.inner.sources.try_write().is_ok(),
                    "source loader ran while cache was locked"
                );
                Ok(Arc::from("// injected\n"))
            })
            .expect("injected source");

        assert_eq!(source.as_ref(), "// injected\n");
    }

    #[test]
    fn source_loader_panic_does_not_poison_cache() {
        let cache = ShaderCache::new(".");
        let result = catch_unwind(AssertUnwindSafe(|| {
            cache
                .source_with(
                    PathBuf::from("panicking-loader.wgsl"),
                    |_| -> Result<Arc<str>, ShaderLoadError> {
                        panic!("source loader failed");
                    },
                )
                .expect("loader should panic before returning");
        }));

        assert!(result.is_err());
        assert!(!cache.inner.sources.is_poisoned());
    }

    #[test]
    fn clear_prevents_inflight_source_from_repopulating_cache() {
        let cache = ShaderCache::new(".");
        let path = PathBuf::from("cleared-inflight.wgsl");

        let source = cache
            .source_with(path.clone(), |_| {
                cache.clear();
                Ok(Arc::from("// loaded before clear completed\n"))
            })
            .expect("inflight source result");

        assert_eq!(source.as_ref(), "// loaded before clear completed\n");
        assert!(!read_recover(&cache.inner.sources).contains_key(&path));
    }

    #[test]
    fn prefetch_warms_cache_for_multiple_files() {
        let temp = TempShaderDir::new();
        temp.write("a.wgsl", "// A\n");
        temp.write("b.wgsl", "// B\n");

        let cache = ShaderCache::new(temp.path());
        cache
            .prefetch(["a.wgsl", "b.wgsl"])
            .expect("prefetch should succeed");

        assert_eq!(cache.source("a.wgsl").unwrap().as_ref(), "// A\n");
        assert_eq!(cache.source("b.wgsl").unwrap().as_ref(), "// B\n");
    }

    #[test]
    fn apply_overrides_replaces_matching_constants() {
        let shader: Arc<str> =
            Arc::from("override WG_ROWS : u32 = 16u;\noverride WG_COLS : u32 = 16u;\n");
        let specialized = apply_overrides(&shader, "test.wgsl", &[("WG_ROWS", 32), ("WG_COLS", 8)])
            .expect("override application");
        assert!(specialized.contains("override WG_ROWS : u32 = 32u;"));
        assert!(specialized.contains("override WG_COLS : u32 = 8u;"));
    }

    #[test]
    fn apply_overrides_errors_when_constant_missing() {
        let shader: Arc<str> = Arc::from("override WG_ROWS : u32 = 16u;\n");
        let err = apply_overrides(&shader, "test.wgsl", &[("WG_COLS", 8)]).unwrap_err();
        match err {
            ShaderLoadError::OverrideNotFound { name, file } => {
                assert_eq!(name, "WG_COLS");
                assert_eq!(file, PathBuf::from("test.wgsl"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
