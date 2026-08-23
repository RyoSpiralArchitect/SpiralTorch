//! Checked WGPU runtime primitives shared by backend kernels.
//!
//! This module owns device/queue lifetime and host/device buffer movement. It
//! deliberately does not choose a backend or define tensor semantics.

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
use std::mem::size_of;
use std::ops::Range;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{mpsc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use bytemuck::Pod;
use thiserror::Error;
use wgpu::util::DeviceExt;

#[cfg(not(target_arch = "wasm32"))]
const READBACK_TIMEOUT: Duration = Duration::from_secs(30);

/// Reference-counted ownership matching the WGPU target's threading model.
#[cfg(not(target_arch = "wasm32"))]
pub type Shared<T> = std::sync::Arc<T>;
/// Weak ownership matching native WGPU's multi-threaded handle model.
#[cfg(not(target_arch = "wasm32"))]
pub type WeakShared<T> = std::sync::Weak<T>;
/// Reference-counted ownership matching the browser's single-threaded WGPU handles.
#[cfg(target_arch = "wasm32")]
pub type Shared<T> = std::rc::Rc<T>;
/// Weak ownership matching browser WGPU's single-threaded handle model.
#[cfg(target_arch = "wasm32")]
pub type WeakShared<T> = std::rc::Weak<T>;

/// Device/queue pair kept alive for repeated backend dispatches.
#[derive(Clone, Debug)]
pub struct WgpuContext {
    device: Shared<wgpu::Device>,
    queue: Shared<wgpu::Queue>,
}

impl WgpuContext {
    pub fn new(device: Shared<wgpu::Device>, queue: Shared<wgpu::Queue>) -> Self {
        Self { device, queue }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn shared_device(&self) -> Shared<wgpu::Device> {
        self.device.clone()
    }

    pub fn shared_queue(&self) -> Shared<wgpu::Queue> {
        self.queue.clone()
    }

    /// Report whether two contexts own the exact same device and queue handles.
    pub fn shares_handles_with(&self, other: &Self) -> bool {
        Shared::ptr_eq(&self.device, &other.device) && Shared::ptr_eq(&self.queue, &other.queue)
    }
}

/// Adapter metadata and shared device state for one headless WGPU runtime.
#[derive(Clone, Debug)]
pub struct WgpuRuntime {
    context: WgpuContext,
    adapter_info: wgpu::AdapterInfo,
}

impl WgpuRuntime {
    pub fn new(context: WgpuContext, adapter_info: wgpu::AdapterInfo) -> Self {
        Self {
            context,
            adapter_info,
        }
    }

    pub fn context(&self) -> &WgpuContext {
        &self.context
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Request a headless adapter and enable SpiralTorch's supported optional features.
    pub async fn request_headless(label: &str) -> Result<Self, WgpuRuntimeError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let mut selected = None;
        for (power_preference, force_fallback_adapter) in [
            (wgpu::PowerPreference::HighPerformance, false),
            (wgpu::PowerPreference::LowPower, false),
            (wgpu::PowerPreference::LowPower, true),
        ] {
            let options = wgpu::RequestAdapterOptions {
                power_preference,
                compatible_surface: None,
                force_fallback_adapter,
            };
            if let Some(adapter) = instance.request_adapter(&options).await {
                selected = Some(adapter);
                break;
            }
        }
        let adapter = selected.ok_or(WgpuRuntimeError::NoAdapter)?;
        let adapter_info = adapter.get_info();
        let optional_features = wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16;
        let required_features = adapter.features() & optional_features;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some(label),
                    required_features,
                    required_limits: adapter.limits(),
                },
                None,
            )
            .await
            .map_err(|error| WgpuRuntimeError::DeviceRequest {
                message: error.to_string(),
            })?;
        Ok(Self::new(
            WgpuContext::new(Shared::new(device), Shared::new(queue)),
            adapter_info,
        ))
    }
}

/// Failures while sizing, allocating, or reading WGPU buffers.
#[non_exhaustive]
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum WgpuRuntimeError {
    #[error("no suitable headless WGPU adapter is available")]
    NoAdapter,
    #[error("WGPU device request failed: {message}")]
    DeviceRequest { message: String },
    #[error(
        "WGPU buffer '{resource}' byte count overflowed: elements={elements} element_size={element_size}"
    )]
    ByteCountOverflow {
        resource: String,
        elements: usize,
        element_size: usize,
    },
    #[error("WGPU buffer '{resource}' cannot be zero-sized")]
    ZeroSizedBuffer { resource: String },
    #[error(
        "WGPU device limit for '{resource}' requires {required} bytes, device provides {available}"
    )]
    DeviceLimit {
        resource: String,
        required: u64,
        available: u64,
    },
    #[error(
        "WGPU readback range for '{resource}' requires {required} bytes, buffer contains {available}"
    )]
    ReadbackRange {
        resource: String,
        required: u64,
        available: u64,
    },
    #[error("WGPU buffer '{resource}' is missing required usage {required}")]
    MissingUsage {
        resource: String,
        required: &'static str,
    },
    #[error("WGPU readback for '{resource}' must be aligned to 4 bytes, got {bytes}")]
    UnalignedReadback { resource: String, bytes: u64 },
    #[error("WGPU map failed for '{resource}': {message}")]
    Map { resource: String, message: String },
    #[error("WGPU map callback disconnected for '{resource}'")]
    MapCallbackDisconnected { resource: String },
    #[error("WGPU submission callback disconnected for '{operation}'")]
    SubmissionCallbackDisconnected { operation: &'static str },
    #[error("WGPU submission '{operation}' timed out after {timeout:?}")]
    SubmitTimeout {
        operation: &'static str,
        timeout: Duration,
    },
    #[error("WGPU map for '{resource}' timed out after {timeout:?}")]
    MapTimeout { resource: String, timeout: Duration },
    #[error(
        "WGPU map range for '{resource}' is invalid: start={start} end={end} buffer_size={available}"
    )]
    InvalidMapRange {
        resource: String,
        start: u64,
        end: u64,
        available: u64,
    },
    #[error(
        "WGPU map range for '{resource}' is not aligned: start={start} requires {start_alignment}, end={end} requires {end_alignment}"
    )]
    UnalignedMapRange {
        resource: String,
        start: u64,
        end: u64,
        start_alignment: u64,
        end_alignment: u64,
    },
    #[error("WGPU operation '{operation}' is not available on target '{target}'")]
    UnsupportedTarget {
        operation: &'static str,
        target: &'static str,
    },
}

/// Failure to replace the process or browser-thread default runtime.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum WgpuRuntimeInstallError {
    #[error("the default WGPU runtime is already installed")]
    AlreadyInstalled,
}

#[cfg(not(target_arch = "wasm32"))]
static DEFAULT_RUNTIME: OnceLock<Mutex<Option<WgpuRuntime>>> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn default_runtime_slot() -> &'static Mutex<Option<WgpuRuntime>> {
    DEFAULT_RUNTIME.get_or_init(|| Mutex::new(None))
}

#[cfg(not(target_arch = "wasm32"))]
fn lock_runtime_slot() -> MutexGuard<'static, Option<WgpuRuntime>> {
    match default_runtime_slot().lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let guard = poisoned.into_inner();
            default_runtime_slot().clear_poison();
            guard
        }
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static DEFAULT_RUNTIME: RefCell<Option<WgpuRuntime>> = const { RefCell::new(None) };
}

/// Return the installed default runtime without probing or creating an adapter.
#[cfg(not(target_arch = "wasm32"))]
pub fn default_runtime() -> Option<WgpuRuntime> {
    lock_runtime_slot().clone()
}

/// Return the browser-thread default runtime without probing or creating an adapter.
#[cfg(target_arch = "wasm32")]
pub fn default_runtime() -> Option<WgpuRuntime> {
    DEFAULT_RUNTIME.with(|slot| slot.borrow().clone())
}

/// Install an explicitly-created default runtime without silently replacing one.
#[cfg(not(target_arch = "wasm32"))]
pub fn install_default_runtime(runtime: WgpuRuntime) -> Result<(), WgpuRuntimeInstallError> {
    let mut slot = lock_runtime_slot();
    if slot.is_some() {
        return Err(WgpuRuntimeInstallError::AlreadyInstalled);
    }
    *slot = Some(runtime);
    Ok(())
}

/// Install an explicitly-created browser-thread runtime without replacing one.
#[cfg(target_arch = "wasm32")]
pub fn install_default_runtime(runtime: WgpuRuntime) -> Result<(), WgpuRuntimeInstallError> {
    DEFAULT_RUNTIME.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(WgpuRuntimeInstallError::AlreadyInstalled);
        }
        *slot = Some(runtime);
        Ok(())
    })
}

/// Return the native default runtime, creating it exactly once when absent.
#[cfg(not(target_arch = "wasm32"))]
pub fn ensure_default_runtime_blocking(
    label: &str,
) -> Result<(WgpuRuntime, bool), WgpuRuntimeError> {
    let mut slot = lock_runtime_slot();
    if let Some(runtime) = slot.as_ref() {
        return Ok((runtime.clone(), false));
    }
    let runtime = pollster::block_on(WgpuRuntime::request_headless(label))?;
    *slot = Some(runtime.clone());
    Ok((runtime, true))
}

/// Browser adapter acquisition must be awaited by the JavaScript event loop.
#[cfg(target_arch = "wasm32")]
pub fn ensure_default_runtime_blocking(
    _label: &str,
) -> Result<(WgpuRuntime, bool), WgpuRuntimeError> {
    Err(WgpuRuntimeError::UnsupportedTarget {
        operation: "synchronous default runtime acquisition",
        target: "wasm32",
    })
}

/// Reject APIs that require synchronously polling a browser-owned device.
#[cfg(not(target_arch = "wasm32"))]
pub fn ensure_blocking_readback_supported(
    _operation: &'static str,
) -> Result<(), WgpuRuntimeError> {
    Ok(())
}

/// Reject APIs that require synchronously polling a browser-owned device.
#[cfg(target_arch = "wasm32")]
pub fn ensure_blocking_readback_supported(operation: &'static str) -> Result<(), WgpuRuntimeError> {
    Err(blocking_readback_error(operation))
}

#[cfg(target_arch = "wasm32")]
fn blocking_readback_error(operation: &'static str) -> WgpuRuntimeError {
    WgpuRuntimeError::UnsupportedTarget {
        operation,
        target: "wasm32",
    }
}

/// Submit command buffers and bound host polling by an explicit timeout.
#[cfg(not(target_arch = "wasm32"))]
pub fn submit_with_timeout(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    command_buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
    timeout: Duration,
    operation: &'static str,
) -> Result<(), WgpuRuntimeError> {
    let (sender, receiver) = mpsc::channel();
    queue.submit(command_buffers);
    queue.on_submitted_work_done(move || {
        let _ = sender.send(());
    });

    let started = Instant::now();
    loop {
        device.poll(wgpu::Maintain::Poll);
        match receiver.try_recv() {
            Ok(()) => return Ok(()),
            Err(mpsc::TryRecvError::Empty) if started.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(mpsc::TryRecvError::Empty) => {
                return Err(WgpuRuntimeError::SubmitTimeout { operation, timeout });
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(WgpuRuntimeError::SubmissionCallbackDisconnected { operation });
            }
        }
    }
}

/// Browser submissions must complete through the JavaScript event loop.
#[cfg(target_arch = "wasm32")]
pub fn submit_with_timeout(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    _command_buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
    _timeout: Duration,
    _operation: &'static str,
) -> Result<(), WgpuRuntimeError> {
    Err(blocking_readback_error("blocking queue submission"))
}

/// Read an already MAP_READ-capable buffer range with bounded host polling.
#[cfg(not(target_arch = "wasm32"))]
pub fn map_read_bytes_with_timeout(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    range: Range<u64>,
    timeout: Duration,
    label: &str,
) -> Result<Vec<u8>, WgpuRuntimeError> {
    if !buffer.usage().contains(wgpu::BufferUsages::MAP_READ) {
        return Err(WgpuRuntimeError::MissingUsage {
            resource: label.to_owned(),
            required: "MAP_READ",
        });
    }
    if range.start > range.end || range.end > buffer.size() {
        return Err(WgpuRuntimeError::InvalidMapRange {
            resource: label.to_owned(),
            start: range.start,
            end: range.end,
            available: buffer.size(),
        });
    }
    if range.is_empty() {
        return Ok(Vec::new());
    }
    if !range.start.is_multiple_of(wgpu::MAP_ALIGNMENT)
        || !range.end.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
    {
        return Err(WgpuRuntimeError::UnalignedMapRange {
            resource: label.to_owned(),
            start: range.start,
            end: range.end,
            start_alignment: wgpu::MAP_ALIGNMENT,
            end_alignment: wgpu::COPY_BUFFER_ALIGNMENT,
        });
    }

    let slice = buffer.slice(range.clone());
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let started = Instant::now();
    loop {
        device.poll(wgpu::Maintain::Poll);
        match receiver.try_recv() {
            Ok(result) => {
                if let Err(error) = result {
                    buffer.unmap();
                    return Err(WgpuRuntimeError::Map {
                        resource: label.to_owned(),
                        message: error.to_string(),
                    });
                }
                break;
            }
            Err(mpsc::TryRecvError::Empty) if started.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(mpsc::TryRecvError::Empty) => {
                buffer.unmap();
                return Err(WgpuRuntimeError::MapTimeout {
                    resource: label.to_owned(),
                    timeout,
                });
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                buffer.unmap();
                return Err(WgpuRuntimeError::MapCallbackDisconnected {
                    resource: label.to_owned(),
                });
            }
        }
    }

    let mapped = slice.get_mapped_range();
    let output = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(output)
}

/// Browser mapping must complete through the JavaScript event loop.
#[cfg(target_arch = "wasm32")]
pub fn map_read_bytes_with_timeout(
    _device: &wgpu::Device,
    _buffer: &wgpu::Buffer,
    _range: Range<u64>,
    _timeout: Duration,
    _label: &str,
) -> Result<Vec<u8>, WgpuRuntimeError> {
    Err(blocking_readback_error("blocking buffer map"))
}

/// Compute an addressable byte count without allocating.
pub fn checked_byte_len<T>(
    resource: impl Into<String>,
    elements: usize,
) -> Result<u64, WgpuRuntimeError> {
    let resource = resource.into();
    let element_size = size_of::<T>();
    let bytes = elements
        .checked_mul(element_size)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| WgpuRuntimeError::ByteCountOverflow {
            resource: resource.clone(),
            elements,
            element_size,
        })?;
    Ok(bytes)
}

fn validate_buffer_size<T>(
    device: &wgpu::Device,
    resource: &str,
    elements: usize,
    usage: wgpu::BufferUsages,
) -> Result<u64, WgpuRuntimeError> {
    let bytes = checked_byte_len::<T>(resource, elements)?;
    if bytes == 0 {
        return Err(WgpuRuntimeError::ZeroSizedBuffer {
            resource: resource.to_owned(),
        });
    }

    let limits = device.limits();
    let mut available = limits.max_buffer_size;
    if usage.contains(wgpu::BufferUsages::STORAGE) {
        available = available.min(u64::from(limits.max_storage_buffer_binding_size));
    }
    if usage.contains(wgpu::BufferUsages::UNIFORM) {
        available = available.min(u64::from(limits.max_uniform_buffer_binding_size));
    }
    if bytes > available {
        return Err(WgpuRuntimeError::DeviceLimit {
            resource: resource.to_owned(),
            required: bytes,
            available,
        });
    }
    Ok(bytes)
}

/// Create a buffer initialized with POD data after validating device limits.
pub fn upload_slice<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> Result<wgpu::Buffer, WgpuRuntimeError> {
    validate_buffer_size::<T>(device, label, data.len(), usage)?;
    Ok(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        }),
    )
}

/// Allocate an unmapped POD buffer after validating its byte size and limits.
pub fn empty_buffer<T>(
    device: &wgpu::Device,
    label: &str,
    elements: usize,
    usage: wgpu::BufferUsages,
) -> Result<wgpu::Buffer, WgpuRuntimeError> {
    let size = validate_buffer_size::<T>(device, label, elements, usage)?;
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    }))
}

/// Copy a POD buffer to host memory with bounded map polling.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_buffer<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    elements: usize,
    label: &str,
) -> Result<Vec<T>, WgpuRuntimeError> {
    if elements == 0 {
        return Ok(Vec::new());
    }
    let size = validate_buffer_size::<T>(
        device,
        label,
        elements,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    )?;
    if !buffer.usage().contains(wgpu::BufferUsages::COPY_SRC) {
        return Err(WgpuRuntimeError::MissingUsage {
            resource: label.to_owned(),
            required: "COPY_SRC",
        });
    }
    if size % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
        return Err(WgpuRuntimeError::UnalignedReadback {
            resource: label.to_owned(),
            bytes: size,
        });
    }
    if size > buffer.size() {
        return Err(WgpuRuntimeError::ReadbackRange {
            resource: label.to_owned(),
            required: size,
            available: buffer.size(),
        });
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.backend.wgpu.readback.encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let mapped = map_read_bytes_with_timeout(device, &staging, 0..size, READBACK_TIMEOUT, label)?;
    // Stable Rust cannot use size_of::<T>() as an as_chunks const argument.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    let output = mapped
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    Ok(output)
}

/// Browser readback must be driven asynchronously by the JavaScript event loop.
#[cfg(target_arch = "wasm32")]
pub fn read_buffer<T: Pod>(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    _buffer: &wgpu::Buffer,
    _elements: usize,
    _label: &str,
) -> Result<Vec<T>, WgpuRuntimeError> {
    Err(blocking_readback_error("blocking buffer readback"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_byte_len_rejects_unaddressable_storage() {
        let elements = (isize::MAX as usize / size_of::<f32>()) + 1;
        assert!(matches!(
            checked_byte_len::<f32>("test", elements),
            Err(WgpuRuntimeError::ByteCountOverflow { .. })
        ));
    }

    #[test]
    fn checked_byte_len_preserves_exact_sizes() {
        assert_eq!(checked_byte_len::<u32>("test", 513).unwrap(), 2052);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn default_runtime_reuses_one_context_when_enabled() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let (first, _) = ensure_default_runtime_blocking("st.backend.runtime.test").unwrap();
        let (second, created) = ensure_default_runtime_blocking("st.backend.runtime.test").unwrap();
        assert!(!created);
        assert!(first.context().shares_handles_with(second.context()));
        assert_eq!(
            install_default_runtime(first),
            Err(WgpuRuntimeInstallError::AlreadyInstalled)
        );
    }
}
