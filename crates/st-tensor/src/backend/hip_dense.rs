// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "hip")]

use std::sync::atomic::{AtomicBool, Ordering};

use st_backend_hip as hip;

static HIP_READY: AtomicBool = AtomicBool::new(false);

fn ensure_runtime() -> Result<(), String> {
    if HIP_READY.load(Ordering::Relaxed) {
        if hip::runtime().is_some() {
            return Ok(());
        }
    }

    match hip::init() {
        Ok(_) => {
            HIP_READY.store(true, Ordering::Relaxed);
            Ok(())
        }
        Err(err) => Err(err.to_string()),
    }
}

pub fn is_available() -> bool {
    if HIP_READY.load(Ordering::Relaxed) && hip::runtime().is_some() {
        return true;
    }

    hip::init()
        .map(|runtime| {
            HIP_READY.store(true, Ordering::Relaxed);
            runtime.device_count() > 0
        })
        .unwrap_or(false)
}

pub fn should_use(rows: usize, inner: usize, cols: usize) -> bool {
    let volume = rows.saturating_mul(cols);
    volume >= 256 && inner >= 16
}

pub fn matmul_into(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), String> {
    ensure_runtime()?;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| "matmul rows*cols overflow".to_string())?;
    if out.len() != expected {
        return Err(format!(
            "output buffer length {} does not match rows*cols={}",
            out.len(),
            expected
        ));
    }
    hip::gemm_f32(rows, cols, inner, lhs, rhs, out).map_err(|err| err.to_string())
}

pub fn matmul(
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let mut out = vec![0.0f32; rows * cols];
    matmul_into(lhs, rhs, &mut out, rows, inner, cols)?;
    Ok(out)
}
