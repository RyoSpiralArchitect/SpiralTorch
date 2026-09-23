//! Ordered owning snapshots of multiple POD buffers, packed into bounded maps.
use super::*;
use std::marker::PhantomData;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Segment {
    chunk: usize,
    bytes: Range<usize>,
}

#[derive(Debug, PartialEq, Eq)]
struct Packing {
    sizes: Vec<u64>,
    segments: Vec<Option<Segment>>,
}

fn packing(sizes: &[u64], limit: u64, label: &str) -> Result<Packing, WgpuRuntimeError> {
    let mut result = Packing {
        sizes: Vec::new(),
        segments: Vec::with_capacity(sizes.len()),
    };
    for &size in sizes {
        if size == 0 {
            result.segments.push(None);
            continue;
        }
        if !size.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
            return Err(WgpuRuntimeError::UnalignedReadback {
                resource: label.into(),
                bytes: size,
            });
        }
        let available = limit.min(isize::MAX as u64);
        if size > available {
            return Err(WgpuRuntimeError::DeviceLimit {
                resource: label.into(),
                required: size,
                available,
            });
        }
        if result
            .sizes
            .last()
            .is_none_or(|&used| used > available - size)
        {
            result.sizes.push(0);
        }
        let chunk = result.sizes.len() - 1;
        let start = result.sizes[chunk];
        result.sizes[chunk] += size;
        result.segments.push(Some(Segment {
            chunk,
            bytes: start as usize..(start + size) as usize,
        }));
    }
    Ok(result)
}

/// An owning snapshot of ordered buffer prefixes. Copies are submitted once;
/// mapping is explicit. Later writes to the sources cannot change the snapshot.
///
/// All sources must belong to the supplied context (enforced by WGPU). Prefixes
/// retain the same POD sizing, COPY_SRC and four-byte alignment requirements as
/// ordinary read_buffer. Empty prefixes are preserved without touching buffers.
/// Packing spills to another staging buffer at the device limit, so a combined
/// result larger than one buffer does not reject otherwise valid inputs.
pub struct ReadbackBatch<T: Pod> {
    chunks: Vec<ReadbackLease>,
    segments: Vec<Option<Segment>>,
    label: String,
    element: PhantomData<T>,
    // Keep the device alive until every pending map and staging buffer is gone.
    context: WgpuContext,
}

impl<T: Pod> ReadbackBatch<T> {
    pub fn copy(
        context: &WgpuContext,
        sources: &[(&wgpu::Buffer, usize)],
        label: &str,
    ) -> Result<Self, WgpuRuntimeError> {
        Self::copy_with_limit(
            context,
            sources,
            label,
            context.device().limits().max_buffer_size,
        )
    }

    fn copy_with_limit(
        context: &WgpuContext,
        sources: &[(&wgpu::Buffer, usize)],
        label: &str,
        staging_limit: u64,
    ) -> Result<Self, WgpuRuntimeError> {
        let spans: Vec<_> = sources
            .iter()
            .map(|&(source, elements)| (source, 0, elements))
            .collect();
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("st.backend.wgpu.readback_batch.encoder"),
                });
        let batch =
            Self::encode_spans_with_limit(context, &spans, label, staging_limit, &mut encoder)?;
        if !batch.chunks.is_empty() {
            context.queue().submit(Some(encoder.finish()));
        }
        Ok(batch)
    }

    /// Encode element-aligned source spans after the caller's GPU work. The
    /// caller submits the encoder once, before mapping this owning snapshot.
    pub(crate) fn encode_spans(
        context: &WgpuContext,
        spans: &[(&wgpu::Buffer, usize, usize)],
        label: &str,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<Self, WgpuRuntimeError> {
        Self::encode_spans_with_limit(
            context,
            spans,
            label,
            context.device().limits().max_buffer_size,
            encoder,
        )
    }

    fn encode_spans_with_limit(
        context: &WgpuContext,
        spans: &[(&wgpu::Buffer, usize, usize)],
        label: &str,
        staging_limit: u64,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<Self, WgpuRuntimeError> {
        let device = context.device();
        let sizes = spans
            .iter()
            .enumerate()
            .map(|(index, &(source, offset, elements))| {
                if elements == 0 {
                    return Ok(0);
                }
                let resource = format!("{label}[{index}]");
                let size = validate_buffer_size::<T>(
                    device,
                    &resource,
                    elements,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                )?;
                if !source.usage().contains(wgpu::BufferUsages::COPY_SRC) {
                    return Err(WgpuRuntimeError::MissingUsage {
                        resource,
                        required: "COPY_SRC",
                    });
                }
                let start = checked_byte_len::<T>(&resource, offset)?;
                if !start.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
                    return Err(WgpuRuntimeError::UnalignedReadback {
                        resource,
                        bytes: start,
                    });
                }
                let end =
                    start
                        .checked_add(size)
                        .ok_or_else(|| WgpuRuntimeError::ByteCountOverflow {
                            resource: resource.clone(),
                            elements,
                            element_size: size_of::<T>(),
                        })?;
                if end > source.size() {
                    return Err(WgpuRuntimeError::ReadbackRange {
                        resource,
                        required: end,
                        available: source.size(),
                    });
                }
                Ok(size)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Validate the complete request before allocating or submitting any copy.
        let packed = packing(
            &sizes,
            staging_limit.min(device.limits().max_buffer_size),
            label,
        )?;
        let chunks: Vec<_> = packed
            .sizes
            .iter()
            .map(|&size| {
                ReadbackLease::unpooled(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }))
            })
            .collect();
        for ((source, offset, _), segment) in spans.iter().zip(&packed.segments) {
            if let Some(segment) = segment {
                encoder.copy_buffer_to_buffer(
                    source,
                    checked_byte_len::<T>(label, *offset)?,
                    chunks[segment.chunk].buffer(),
                    segment.bytes.start as u64,
                    segment.bytes.len() as u64,
                );
            }
        }
        Ok(Self {
            chunks,
            segments: packed.segments,
            label: label.into(),
            element: PhantomData,
            context: context.clone(),
        })
    }

    /// Number of mapping operations, not the number of source buffers.
    pub fn staging_buffer_count(&self) -> usize {
        self.chunks.len()
    }

    /// One ordered, owning Vec per requested prefix. Native maps have the same
    /// bounded timeout as ordinary runtime readback.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(mut self) -> Result<Vec<Vec<T>>, WgpuRuntimeError> {
        let mut output = vec![Vec::new(); self.segments.len()];
        for (index, chunk) in self.chunks.iter_mut().enumerate() {
            let bytes = chunk.read(&self.context, READBACK_TIMEOUT, &self.label)?;
            decode(&self.segments, index, &bytes, &mut output);
        }
        Ok(output)
    }

    /// Synchronous mapping never blocks the browser event loop.
    #[cfg(target_arch = "wasm32")]
    pub fn read(self) -> Result<Vec<Vec<T>>, WgpuRuntimeError> {
        Err(blocking_readback_error("blocking batched readback"))
    }

    /// Browser completion uses the event loop. Dropping a pending future owns
    /// and cancels its maps through the same lease used by resident snapshots.
    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<Vec<Vec<T>>, WgpuRuntimeError> {
        let Self {
            chunks,
            segments,
            label,
            context,
            ..
        } = self;
        let mut output = vec![Vec::new(); segments.len()];
        for (index, chunk) in chunks.into_iter().enumerate() {
            let bytes = chunk.read_async(context.clone(), &label).await?;
            decode(&segments, index, &bytes, &mut output);
        }
        Ok(output)
    }
}

fn decode<T: Pod>(segments: &[Option<Segment>], index: usize, bytes: &[u8], output: &mut [Vec<T>]) {
    for (segment, destination) in segments.iter().zip(output) {
        if let Some(segment) = segment {
            if segment.chunk == index {
                // Generic POD width cannot be an as_chunks const argument.
                #[allow(clippy::chunks_exact_to_as_chunks)]
                let values = bytes[segment.bytes.clone()]
                    .chunks_exact(size_of::<T>())
                    .map(bytemuck::pod_read_unaligned)
                    .collect();
                *destination = values;
            }
        }
    }
}

/// Blocking convenience path, sharing the same packing/copy implementation.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_buffers<T: Pod>(
    context: &WgpuContext,
    sources: &[(&wgpu::Buffer, usize)],
    label: &str,
) -> Result<Vec<Vec<T>>, WgpuRuntimeError> {
    ReadbackBatch::copy(context, sources, label)?.read()
}

#[cfg(target_arch = "wasm32")]
pub fn read_buffers<T: Pod>(
    _context: &WgpuContext,
    _sources: &[(&wgpu::Buffer, usize)],
    _label: &str,
) -> Result<Vec<Vec<T>>, WgpuRuntimeError> {
    Err(blocking_readback_error("blocking batched readback"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn readback_batch_spill_maps_ordered_chunks_when_enabled() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let runtime = pollster::block_on(WgpuRuntime::request_headless("batch.spill")).unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let context = runtime.context();
        let buffer = upload_slice(
            context.device(),
            "batch.spill",
            &[1u32, 2],
            wgpu::BufferUsages::COPY_SRC,
        )
        .unwrap();
        // Exercise the real copy/map path with a small simulated staging limit.
        let batch = ReadbackBatch::<u32>::copy_with_limit(
            context,
            &[(&buffer, 2), (&buffer, 1), (&buffer, 1), (&buffer, 0)],
            "batch.spill",
            8,
        )
        .unwrap();
        assert_eq!(batch.staging_buffer_count(), 2);
        assert_eq!(
            batch.read().unwrap(),
            [vec![1, 2], vec![1], vec![1], vec![]]
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn encoded_spans_read_offsets_and_validate_ranges_when_enabled() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        let runtime = pollster::block_on(WgpuRuntime::request_headless("batch.spans")).unwrap();
        let context = runtime.context();
        let source = upload_slice(
            context.device(),
            "batch.spans",
            &[10u32, 20, 30, 40],
            wgpu::BufferUsages::COPY_SRC,
        )
        .unwrap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let batch = ReadbackBatch::<u32>::encode_spans_with_limit(
            context,
            &[(&source, 1, 2), (&source, 3, 1), (&source, 4, 0)],
            "batch.spans",
            8,
            &mut encoder,
        )
        .unwrap();
        assert_eq!(batch.staging_buffer_count(), 2);
        context.queue().submit(Some(encoder.finish()));
        assert_eq!(batch.read().unwrap(), [vec![20, 30], vec![40], vec![]]);

        let mut encoder = context.device().create_command_encoder(&Default::default());
        assert!(matches!(
            ReadbackBatch::<u32>::encode_spans(
                context,
                &[(&source, 3, 2)],
                "batch.spans.invalid",
                &mut encoder,
            ),
            Err(WgpuRuntimeError::ReadbackRange { .. })
        ));
        let halfwords = upload_slice(
            context.device(),
            "batch.spans.halfwords",
            &[1u16, 2, 3, 4],
            wgpu::BufferUsages::COPY_SRC,
        )
        .unwrap();
        assert!(matches!(
            ReadbackBatch::<u16>::encode_spans(
                context,
                &[(&halfwords, 1, 2)],
                "batch.spans.unaligned",
                &mut encoder,
            ),
            Err(WgpuRuntimeError::UnalignedReadback { .. })
        ));
    }

    #[test]
    fn packing_preserves_empty_prefixes_and_spills_without_overflow() {
        let result = packing(&[0, 8, 0, 8, 12], 16, "fixture").unwrap();
        assert_eq!(result.sizes, [16, 12]);
        assert_eq!(
            result.segments,
            [
                None,
                Some(Segment {
                    chunk: 0,
                    bytes: 0..8
                }),
                None,
                Some(Segment {
                    chunk: 0,
                    bytes: 8..16
                }),
                Some(Segment {
                    chunk: 1,
                    bytes: 0..12
                }),
            ]
        );
        assert!(packing(&[], 0, "empty").unwrap().sizes.is_empty());
        assert!(packing(&[0, 0], 0, "empty").unwrap().sizes.is_empty());
        for sizes in [&[3][..], &[20], &[u64::MAX - 3]] {
            assert!(packing(sizes, 16, "bad").is_err());
        }
    }

    #[test]
    fn pod_decode_is_ordered_and_preserves_float_bits() {
        let packed = packing(&[4, 0, 4, 4], 8, "fixture").unwrap();
        let values = [0x80000000u32, 0x7fc00001u32, 0x3f800000u32];
        let mut output = vec![Vec::<f32>::new(); 4];
        decode(
            &packed.segments,
            0,
            bytemuck::cast_slice(&values[..2]),
            &mut output,
        );
        decode(
            &packed.segments,
            1,
            bytemuck::cast_slice(&values[2..]),
            &mut output,
        );
        assert_eq!(
            output.iter().map(|v| v.len()).collect::<Vec<_>>(),
            [1, 0, 1, 1]
        );
        assert_eq!(
            output
                .into_iter()
                .flatten()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            values
        );
    }
}
