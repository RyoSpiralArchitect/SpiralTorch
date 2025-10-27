// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::{
    any::Any,
    borrow::Cow,
    collections::{hash_map::Entry, HashMap},
    fmt, fs,
    hash::{Hash, Hasher},
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
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
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ModuleCacheKey {
    device: usize,
    key: String,
}

impl ModuleCacheKey {
    fn new(device: &Device, key: impl Into<String>) -> Self {
        Self {
            device: device as *const Device as usize,
            key: key.into(),
        }
    }
}

static SHADER_MODULE_CACHE: OnceLock<Mutex<HashMap<ModuleCacheKey, Arc<wgpu::ShaderModule>>>> =
    OnceLock::new();

fn module_cache() -> &'static Mutex<HashMap<ModuleCacheKey, Arc<wgpu::ShaderModule>>> {
    SHADER_MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct PipelineCacheKey {
    module: ModuleCacheKey,
    entry_point: String,
    layout: Option<usize>,
}

impl fmt::Debug for PipelineCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineCacheKey")
            .field("module", &self.module)
            .field("entry_point", &self.entry_point)
            .field("layout", &self.layout)
            .finish()
    }
}

impl PipelineCacheKey {
    fn new(
        module: ModuleCacheKey,
        entry_point: impl Into<String>,
        layout: Option<&wgpu::PipelineLayout>,
    ) -> Self {
        Self {
            module,
            entry_point: entry_point.into(),
            layout: layout.map(|layout| layout as *const wgpu::PipelineLayout as usize),
        }
    }
}

fn cache_key_for_file(device: &Device, file: &str, overrides: &[(&str, u32)]) -> ModuleCacheKey {
    let mut key = String::from(file);
    if !overrides.is_empty() {
        key.push('?');
        for (index, (name, value)) in overrides.iter().enumerate() {
            if index > 0 {
                key.push('&');
            }
            key.push_str(name);
            key.push('=');
            key.push_str(&value.to_string());
        }
    }
    ModuleCacheKey::new(device, key)
}

fn cache_key_for_inline(device: &Device, label: &str, source: &str) -> ModuleCacheKey {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    let hash = hasher.finish();
    ModuleCacheKey::new(device, format!("{label}#{hash:x}"))
}

fn compile_cached_module(
    device: &Device,
    label: &str,
    key: ModuleCacheKey,
    source: String,
    context: Option<String>,
) -> Result<Arc<wgpu::ShaderModule>, ShaderLoadError> {
    {
        let cache = module_cache();
        if let Some(module) = cache.lock().unwrap().get(&key) {
            return Ok(Arc::clone(module));
        }
    }

    let module = catch_unwind(AssertUnwindSafe(|| {
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(Cow::Owned(source)),
        })
    }))
    .map_err(|payload| ShaderLoadError::Compile {
        label: label.to_string(),
        context: context.unwrap_or_else(|| "inline".to_string()),
        source: Box::new(ShaderCompileError(panic_payload_to_string(payload))),
    })?;

    let cache = module_cache();
    let module = Arc::new(module);
    cache.lock().unwrap().insert(key, Arc::clone(&module));
    Ok(module)
}

#[derive(Debug)]
struct ShaderCompileError(String);

impl fmt::Display for ShaderCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ShaderCompileError {}

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
) -> Result<Arc<wgpu::ShaderModule>, ShaderLoadError> {
    let key = cache_key_for_inline(device, label, &source);
    compile_cached_module(device, label, key, source, None)
}

/// Read a WGSL shader relative to `shader_dir` and return the source string.
pub fn read_wgsl(shader_dir: impl AsRef<Path>, file: &str) -> Result<String, ShaderLoadError> {
    let path = shader_dir.as_ref().join(file);
    fs::read_to_string(&path).map_err(|source| ShaderLoadError::Io { source, path })
}

/// Lightweight cache for WGSL shader sources stored on disk and the compute
/// pipelines derived from them.
///
/// Many of SpiralTorch's compute pipelines are composed of multiple entry
/// points that live in separate shader files. When bootstrapping the backend
/// we often need to read several of these files in quick succession. The cache
/// keeps the decoded UTF-8 source in memory so subsequent pipeline
/// construction avoids touching the filesystem again. Additionally, fully
/// constructed [`wgpu::ComputePipeline`] objects are memoized so repeated
/// kernel invocations avoid redundant pipeline creation work on the same
/// device and layout.
#[derive(Default)]
pub struct ShaderCache {
    shader_dir: PathBuf,
    sources: HashMap<PathBuf, String>,
    pipelines: HashMap<PipelineCacheKey, Arc<ComputePipeline>>,
}

impl ShaderCache {
    /// Create a cache rooted at `shader_dir`.
    pub fn new(shader_dir: impl Into<PathBuf>) -> Self {
        Self {
            shader_dir: shader_dir.into(),
            sources: HashMap::new(),
            pipelines: HashMap::new(),
        }
    }

    /// Return the absolute directory containing cached shaders.
    pub fn shader_dir(&self) -> &Path {
        &self.shader_dir
    }

    /// Drop all cached shader sources.
    pub fn clear(&mut self) {
        self.sources.clear();
        self.pipelines.clear();
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
    ) -> Result<Arc<ComputePipeline>, ShaderLoadError> {
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
    ) -> Result<Arc<ComputePipeline>, ShaderLoadError> {
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
    ) -> Result<Arc<ComputePipeline>, ShaderLoadError> {
        let module_key = cache_key_for_file(device, file, overrides);
        let pipeline_key = PipelineCacheKey::new(module_key.clone(), entry_point, layout);

        if let Some(pipeline) = self.pipelines.get(&pipeline_key) {
            return Ok(Arc::clone(pipeline));
        }

        let source = if overrides.is_empty() {
            self.source(file)?.to_string()
        } else {
            apply_overrides(self.source(file)?, file, overrides)?
        };
        let context = self.shader_dir.join(file).display().to_string();
        let module =
            compile_cached_module(device, label, module_key.clone(), source, Some(context))?;
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout,
            module: module.as_ref(),
            entry_point,
            compilation_options: Default::default(),
        });
        let pipeline = Arc::new(pipeline);
        self.pipelines.insert(pipeline_key, Arc::clone(&pipeline));
        Ok(pipeline)
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

impl fmt::Debug for ShaderCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShaderCache")
            .field("shader_dir", &self.shader_dir)
            .field("sources", &self.sources.keys().collect::<Vec<_>>())
            .field("pipelines", &self.pipelines.len())
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
    let source = read_wgsl(dir, file)?;
    let context = dir.join(file).display().to_string();
    let module = compile_cached_module(
        device,
        label,
        cache_key_for_file(device, file, &[]),
        source,
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
