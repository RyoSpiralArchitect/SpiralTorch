//! Snapshot lifetime and cancellation checks shared by native and browser runs.
use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_tensor::{TensorDevice, TensorError, TensorReadback},
    runtime::WgpuRuntime,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

async fn read(snapshot: TensorReadback) -> std::result::Result<Vec<f32>, TensorError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        snapshot.read()
    }
    #[cfg(target_arch = "wasm32")]
    {
        snapshot.read_async().await
    }
}

fn exact(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("snapshot values or negative-zero bits changed".into());
    }
    Ok(())
}

pub async fn run(runtime: &WgpuRuntime) -> Result<Value> {
    let device = TensorDevice::new(runtime.clone())?;
    let mut held = Vec::new();
    for index in 0..24 {
        let n = index as f32;
        let (tensor, expected) = match index % 4 {
            0 => (
                device.upload(&[2, 2], &[n, 1., 2., 3.])?.permute(&[1, 0])?,
                vec![n, 2., 1., 3.],
            ),
            1 => (device.upload(&[], &[-0.])?, vec![-0.]),
            2 => (device.upload(&[0, 4], &[])?, vec![]),
            _ => (device.upload(&[3], &[n; 3])?.narrow(0, 1, 2)?, vec![n; 2]),
        };
        held.push((tensor.snapshot()?, expected));
    }
    let bad = device
        .upload(&[4], &[-f32::MAX; 4])?
        .mul(&device.upload(&[], &[2.])?)?
        .relu()?;
    let bad_snapshot = bad.snapshot()?;
    let empty_bad_snapshot = bad.narrow(0, 0, 0)?.snapshot()?;
    drop(bad);
    let valid = device.upload(&[4], &[7.; 4])?;
    for _ in 0..48 {
        drop(valid.snapshot()?);
        exact(&read(valid.snapshot()?).await?, &[7.; 4])?;
    }
    for (snapshot, expected) in held.into_iter().rev() {
        exact(&read(snapshot).await?, &expected)?;
    }
    for snapshot in [bad_snapshot, empty_bad_snapshot] {
        if !matches!(read(snapshot).await, Err(TensorError::NonFinite)) {
            return Err("recycling lost a held invalid-value guard".into());
        }
    }
    exact(&read(valid.snapshot()?).await?, &[7.; 4])?;

    #[cfg(not(target_arch = "wasm32"))]
    let cancelled_pending_maps = 0;
    #[cfg(target_arch = "wasm32")]
    let cancelled_pending_maps = {
        use std::future::Future;
        use std::task::{Context, Poll, Waker};
        for _ in 0..4 {
            let mut pending = Box::pin(valid.snapshot()?.read_async());
            let mut context = Context::from_waker(Waker::noop());
            if !matches!(pending.as_mut().poll(&mut context), Poll::Pending) {
                return Err("browser map did not enter a pending state".into());
            }
            drop(pending);
            exact(&read(valid.snapshot()?).await?, &[7.; 4])?;
        }
        4
    };
    let survivor = valid.snapshot()?;
    drop(valid);
    drop(device);
    exact(&read(survivor).await?, &[7.; 4])?;
    Ok(
        json!({"status":"passed", "held_snapshots":24, "discard_and_reuse_cycles":48,
        "retained_guard_failures":2, "cancelled_pending_maps":cancelled_pending_maps,
        "different_shapes_and_negative_zero":true, "survives_tensor_device_drop":true}),
    )
}
