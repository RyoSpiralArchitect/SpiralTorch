//! The same ordered-snapshot checks run on native WGPU and browser WebGPU.
use st_backend_wgpu::runtime::{self, ReadbackBatch, WgpuRuntime, WgpuRuntimeError};
use wgpu::BufferUsages as Usage;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

async fn finish(batch: ReadbackBatch<u32>) -> Result<Vec<Vec<u32>>> {
    #[cfg(not(target_arch = "wasm32"))]
    let output = batch.read()?;
    #[cfg(target_arch = "wasm32")]
    let output = batch.read_async().await?;
    Ok(output)
}

pub async fn run(runtime: &WgpuRuntime, checkpoint: fn(&str)) -> Result<serde_json::Value> {
    checkpoint("ownership-start");
    let ctx = runtime.context();
    let gpu = ctx.device();
    let values = [0x80000000u32, 0x7fc00001, 0x3f800000, 0xffffffff];
    let a = runtime::upload_slice(gpu, "batch.a", &values, Usage::COPY_SRC | Usage::COPY_DST)?;
    let b = runtime::upload_slice(gpu, "batch.b", &[7u32, 8], Usage::COPY_SRC)?;
    let no_copy = runtime::upload_slice(gpu, "batch.no_copy", &[0u32], Usage::COPY_DST)?;
    let sources = [(&a, 4), (&no_copy, 0), (&a, 1), (&b, 2)];
    let old = ReadbackBatch::<u32>::copy(ctx, &sources, "batch.old")?;
    assert_eq!(old.staging_buffer_count(), 1);
    ctx.queue()
        .write_buffer(&a, 0, bytemuck::cast_slice(&[99u32; 4]));
    let new = ReadbackBatch::<u32>::copy(ctx, &sources, "batch.new")?;
    drop(ReadbackBatch::<u32>::copy(ctx, &sources, "batch.unread")?);
    checkpoint("ownership-copied");
    assert_eq!(
        finish(new).await?,
        [vec![99; 4], vec![], vec![99], vec![7, 8]]
    );
    drop(a);
    drop(b);
    checkpoint("ownership-new-read");
    assert_eq!(
        finish(old).await?,
        [values.to_vec(), vec![], vec![values[0]], vec![7, 8]]
    );
    for sources in [&[][..], &[(&no_copy, 0), (&no_copy, 0)][..]] {
        let empty = ReadbackBatch::<u32>::copy(ctx, sources, "batch.empty")?;
        assert_eq!(empty.staging_buffer_count(), 0);
        assert_eq!(finish(empty).await?, vec![Vec::<u32>::new(); sources.len()]);
    }
    checkpoint("ownership-old-and-empty-read");
    let valid = runtime::upload_slice(gpu, "batch.valid", &[1u32, 2, 3, 4], Usage::COPY_SRC)?;
    #[cfg(not(target_arch = "wasm32"))]
    assert_eq!(
        runtime::read_buffer::<u32>(gpu, ctx.queue(), &valid, 4, "batch.original")?,
        [1, 2, 3, 4]
    );
    assert!(matches!(
        ReadbackBatch::<u32>::copy(ctx, &[(&valid, 1), (&no_copy, 1)], "bad.usage"),
        Err(WgpuRuntimeError::MissingUsage { .. })
    ));
    assert!(matches!(
        ReadbackBatch::<u32>::copy(ctx, &[(&valid, 5)], "bad.range"),
        Err(WgpuRuntimeError::ReadbackRange { .. })
    ));
    assert!(matches!(
        ReadbackBatch::<u8>::copy(ctx, &[(&valid, 3)], "bad.alignment"),
        Err(WgpuRuntimeError::UnalignedReadback { .. })
    ));
    assert!(ReadbackBatch::<u32>::copy(ctx, &[(&valid, usize::MAX)], "bad.overflow").is_err());
    assert!(ReadbackBatch::<()>::copy(ctx, &[(&valid, 1)], "bad.zst").is_err());
    checkpoint("ownership-preflight");

    #[cfg(target_arch = "wasm32")]
    {
        use std::{future::Future, task::Poll};
        assert!(runtime::read_buffers::<u32>(ctx, &[], "blocked").is_err());
        let canceled = ReadbackBatch::<u32>::copy(ctx, &[(&valid, 4)], "batch.cancel")?;
        let mut future = Box::pin(canceled.read_async());
        let pending = std::future::poll_fn(|cx| {
            Poll::Ready(matches!(future.as_mut().poll(cx), Poll::Pending))
        })
        .await;
        assert!(pending, "browser map must yield before completion");
        drop(future);
        checkpoint("ownership-canceled");
    }
    assert_eq!(
        finish(ReadbackBatch::<u32>::copy(
            ctx,
            &[(&valid, 4)],
            "batch.after"
        )?)
        .await?,
        [vec![1, 2, 3, 4]]
    );
    Ok(
        serde_json::json!({"ordered_snapshot":true,"empty_prefixes":true,
        "preflight_errors":5,"unread_drop":true,"source_drop":true,
        "pending_map_cancellation":cfg!(target_arch="wasm32")}),
    )
}
