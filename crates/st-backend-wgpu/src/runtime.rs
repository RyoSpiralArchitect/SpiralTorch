//! Checked WGPU runtime primitives shared by backend kernels.
//!
//! This module owns device/queue lifetime and host/device buffer movement. It
//! deliberately does not choose a backend or define tensor semantics.

use std::mem::size_of;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use bytemuck::Pod;
use thiserror::Error;
use wgpu::util::DeviceExt;

#[cfg(not(target_arch = "wasm32"))]
const READBACK_TIMEOUT: Duration = Duration::from_secs(30);

/// Reference-counted ownership matching the WGPU target's threading model.
#[cfg(not(target_arch = "wasm32"))]
pub type Shared<T> = std::sync::Arc<T>;
/// Reference-counted ownership matching the browser's single-threaded WGPU handles.
#[cfg(target_arch = "wasm32")]
pub type Shared<T> = std::rc::Rc<T>;

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
}

/// Failures while sizing, allocating, or reading WGPU buffers.
#[non_exhaustive]
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum WgpuRuntimeError {
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
    #[error("WGPU readback timed out after 30 seconds for '{resource}'")]
    ReadbackTimeout { resource: String },
    #[error("WGPU operation '{operation}' is not available on target '{target}'")]
    UnsupportedTarget {
        operation: &'static str,
        target: &'static str,
    },
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

    let slice = staging.slice(0..size);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let started = Instant::now();
    loop {
        device.poll(wgpu::Maintain::Poll);
        match rx.try_recv() {
            Ok(result) => {
                result.map_err(|error| WgpuRuntimeError::Map {
                    resource: label.to_owned(),
                    message: error.to_string(),
                })?;
                break;
            }
            Err(mpsc::TryRecvError::Empty) if started.elapsed() < READBACK_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(mpsc::TryRecvError::Empty) => {
                return Err(WgpuRuntimeError::ReadbackTimeout {
                    resource: label.to_owned(),
                });
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(WgpuRuntimeError::MapCallbackDisconnected {
                    resource: label.to_owned(),
                });
            }
        }
    }

    let mapped = slice.get_mapped_range();
    let output = mapped
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    drop(mapped);
    staging.unmap();
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
}
