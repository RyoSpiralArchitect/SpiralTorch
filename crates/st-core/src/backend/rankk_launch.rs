// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Shared helpers that expose a lightweight launch-context bridge for the GPU
//! rank-k executors. The executors are written against `RankPlan` only, which
//! keeps the planning logic pure, while actual tensor storage lives in the
//! backend runtimes. Until the CUDA/HIP runtimes are fully wired this module
//! offers an ergonomic test-only hook that mirrors how the real launch API will
//! hand buffers to the GPU queues.
//!
//! The helpers intentionally keep the API surface small:
//!   * `LaunchBuffers` validates host slices (rows, cols, k) once.
//!   * `with_launch_buffers_{cuda,hip}` installs the buffers for the duration of
//!     a closure so the executor can borrow them.
//!   * `with_registered_buffers_{cuda,hip}` hands those slices to the software
//!     fallbacks that emulate the GPU kernels during tests.

use std::cell::RefCell;
use std::thread::LocalKey;

#[derive(Debug)]
pub struct LaunchBuffers<'a> {
    pub input: &'a [f32],
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
    pub out_vals: &'a mut [f32],
    pub out_idx: &'a mut [i32],
}

impl<'a> LaunchBuffers<'a> {
    pub fn new(
        input: &'a [f32],
        rows: u32,
        cols: u32,
        k: u32,
        out_vals: &'a mut [f32],
        out_idx: &'a mut [i32],
    ) -> Result<Self, String> {
        let rows_usize = rows as usize;
        let cols_usize = cols as usize;
        let k_usize = k as usize;

        if input.len() != rows_usize * cols_usize {
            return Err(format!(
                "input length {} does not match rows×cols {}",
                input.len(),
                rows_usize * cols_usize
            ));
        }
        if out_vals.len() != rows_usize * k_usize {
            return Err(format!(
                "output value length {} does not match rows×k {}",
                out_vals.len(),
                rows_usize * k_usize
            ));
        }
        if out_idx.len() != rows_usize * k_usize {
            return Err(format!(
                "output index length {} does not match rows×k {}",
                out_idx.len(),
                rows_usize * k_usize
            ));
        }

        Ok(Self {
            input,
            rows,
            cols,
            k,
            out_vals,
            out_idx,
        })
    }
}

#[derive(Clone, Copy)]
struct LaunchContext {
    input_ptr: *const f32,
    input_len: usize,
    out_vals_ptr: *mut f32,
    out_vals_len: usize,
    out_idx_ptr: *mut i32,
    out_idx_len: usize,
    rows: u32,
    cols: u32,
    k: u32,
}

impl<'a> From<&LaunchBuffers<'a>> for LaunchContext {
    fn from(buffers: &LaunchBuffers<'a>) -> Self {
        Self {
            input_ptr: buffers.input.as_ptr(),
            input_len: buffers.input.len(),
            out_vals_ptr: buffers.out_vals.as_ptr() as *mut f32,
            out_vals_len: buffers.out_vals.len(),
            out_idx_ptr: buffers.out_idx.as_ptr() as *mut i32,
            out_idx_len: buffers.out_idx.len(),
            rows: buffers.rows,
            cols: buffers.cols,
            k: buffers.k,
        }
    }
}

pub struct LaunchSlices<'a> {
    pub input: &'a [f32],
    pub out_vals: &'a mut [f32],
    pub out_idx: &'a mut [i32],
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
}

thread_local! {
    static CUDA_CTX: RefCell<Option<LaunchContext>> = RefCell::new(None);
    static HIP_CTX: RefCell<Option<LaunchContext>> = RefCell::new(None);
}

pub fn with_launch_buffers_cuda<'a, F, R>(buffers: LaunchBuffers<'a>, f: F) -> R
where
    F: FnOnce() -> R,
{
    with_launch_buffers_impl(&CUDA_CTX, buffers, f)
}

pub fn with_launch_buffers_hip<'a, F, R>(buffers: LaunchBuffers<'a>, f: F) -> R
where
    F: FnOnce() -> R,
{
    with_launch_buffers_impl(&HIP_CTX, buffers, f)
}

pub fn with_registered_buffers_cuda<F>(f: F) -> Result<(), String>
where
    F: FnOnce(LaunchSlices<'_>) -> Result<(), String>,
{
    with_registered_buffers_impl(&CUDA_CTX, f)
}

pub fn with_registered_buffers_hip<F>(f: F) -> Result<(), String>
where
    F: FnOnce(LaunchSlices<'_>) -> Result<(), String>,
{
    with_registered_buffers_impl(&HIP_CTX, f)
}

fn with_launch_buffers_impl<'a, F, R>(
    key: &'static LocalKey<RefCell<Option<LaunchContext>>>,
    buffers: LaunchBuffers<'a>,
    f: F,
) -> R
where
    F: FnOnce() -> R,
{
    struct Guard<'a> {
        key: &'static LocalKey<RefCell<Option<LaunchContext>>>,
        previous: Option<LaunchContext>,
        _buffers: LaunchBuffers<'a>,
    }

    impl<'a> Drop for Guard<'a> {
        fn drop(&mut self) {
            self.key.with(|cell| {
                *cell.borrow_mut() = self.previous.take();
            });
        }
    }

    let ctx = LaunchContext::from(&buffers);
    let mut buffers_slot = Some(buffers);

    key.with(|cell| {
        let previous = {
            let mut slot = cell.borrow_mut();
            std::mem::replace(&mut *slot, Some(ctx))
        };
        let guard = Guard {
            key,
            previous,
            _buffers: buffers_slot.take().expect("launch buffers present"),
        };
        let result = f();
        drop(guard);
        result
    })
}

fn with_registered_buffers_impl<F>(
    key: &'static LocalKey<RefCell<Option<LaunchContext>>>,
    f: F,
) -> Result<(), String>
where
    F: FnOnce(LaunchSlices<'_>) -> Result<(), String>,
{
    key.with(|cell| {
        let mut slot = cell.borrow_mut();
        let ctx = slot
            .as_mut()
            .ok_or_else(|| "no launch buffers registered for this thread".to_string())?;

        unsafe {
            let input = std::slice::from_raw_parts(ctx.input_ptr, ctx.input_len);
            let out_vals = std::slice::from_raw_parts_mut(ctx.out_vals_ptr, ctx.out_vals_len);
            let out_idx = std::slice::from_raw_parts_mut(ctx.out_idx_ptr, ctx.out_idx_len);
            f(LaunchSlices {
                input,
                out_vals,
                out_idx,
                rows: ctx.rows,
                cols: ctx.cols,
                k: ctx.k,
            })
        }
    })
}
