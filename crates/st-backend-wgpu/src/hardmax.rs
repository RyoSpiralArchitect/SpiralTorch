// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Convenience helpers that configure the row-wise softmax kernels to operate
//! in "hardmax" mode.
//!
//! The [`softmax`](crate::softmax) module exposes the low level pipeline
//! assembly logic that powers both the probability and argmax style reductions.
//! This wrapper layers a small amount of ergonomics so the WGPU backend offers
//! "soft/middle/hard" entry points mirroring the tensor API. The helpers mutate
//! the softmax uniforms to request the hardmax mask, optionally disable the
//! probability normalisation, and forward the pipeline construction utilities.

use crate::softmax;
use wgpu::{Buffer, Device, Queue};

/// Flag enabling hardmax-only execution in the WGSL kernels.
const FLAG_HARDMAX_ONLY: u32 = 1 << 1;
/// Flag requesting the hardmax mask to be written to the auxiliary buffer.
const FLAG_HARDMAX_MASK: u32 = 1 << 2;

/// Output mode requested from the hardmax helpers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    /// Emit both the softmax probabilities and the hardmax mask.
    SoftmaxAndMask,
    /// Only emit the hardmax mask, writing it into the primary output buffer.
    HardmaxOnly,
    /// Emit the hardmax mask into both the primary output buffer and the
    /// optional auxiliary mask binding.
    HardmaxOnlyWithMask,
}

impl Mode {
    fn wants_mask(self) -> bool {
        matches!(self, Mode::SoftmaxAndMask | Mode::HardmaxOnlyWithMask)
    }

    fn hardmax_only(self) -> bool {
        matches!(self, Mode::HardmaxOnly | Mode::HardmaxOnlyWithMask)
    }
}

/// Wrapper around [`softmax::Params`] that applies the hardmax configuration
/// bits while preserving all other layout information.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    inner: softmax::Params,
}

impl Params {
    /// Construct hardmax parameters from a softmax configuration and desired
    /// [`Mode`].
    pub fn new(base: softmax::Params, mode: Mode) -> Self {
        let mut params = Self { inner: base };
        params.apply_mode(mode);
        params
    }

    /// Update the hardmax output mode on the existing parameter payload.
    pub fn apply_mode(&mut self, mode: Mode) -> &mut Self {
        if mode.wants_mask() {
            self.inner.flags |= FLAG_HARDMAX_MASK;
        } else {
            self.inner.flags &= !FLAG_HARDMAX_MASK;
        }

        if mode.hardmax_only() {
            self.inner.flags |= FLAG_HARDMAX_ONLY;
        } else {
            self.inner.flags &= !FLAG_HARDMAX_ONLY;
        }

        self
    }

    /// Inspect the currently configured hardmax output mode.
    ///
    /// When both hardmax flags are cleared the helper returns
    /// [`Mode::HardmaxOnly`] to emphasise that the mask output is disabled even
    /// though the kernels will fall back to the vanilla softmax behaviour.
    pub fn mode(&self) -> Mode {
        let wants_mask = (self.inner.flags & FLAG_HARDMAX_MASK) != 0;
        let hardmax_only = (self.inner.flags & FLAG_HARDMAX_ONLY) != 0;

        match (hardmax_only, wants_mask) {
            (false, true) => Mode::SoftmaxAndMask,
            (true, false) => Mode::HardmaxOnly,
            (true, true) => Mode::HardmaxOnlyWithMask,
            (false, false) => Mode::HardmaxOnly,
        }
    }

    /// Borrow the inner [`softmax::Params`] structure.
    pub fn as_softmax(&self) -> &softmax::Params {
        &self.inner
    }

    /// Mutably borrow the inner [`softmax::Params`] structure.
    pub fn as_softmax_mut(&mut self) -> &mut softmax::Params {
        &mut self.inner
    }

    /// Consume the wrapper and recover the raw softmax configuration.
    pub fn into_softmax(self) -> softmax::Params {
        self.inner
    }

    /// Serialize the uniforms as bytes, matching [`softmax::Params::as_bytes`].
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }
}

impl From<(softmax::Params, Mode)> for Params {
    fn from((base, mode): (softmax::Params, Mode)) -> Self {
        Self::new(base, mode)
    }
}

impl From<Params> for softmax::Params {
    fn from(params: Params) -> Self {
        params.into_softmax()
    }
}

impl AsRef<softmax::Params> for Params {
    fn as_ref(&self) -> &softmax::Params {
        self.as_softmax()
    }
}

impl AsMut<softmax::Params> for Params {
    fn as_mut(&mut self) -> &mut softmax::Params {
        self.as_softmax_mut()
    }
}

/// Forward the softmax pipeline builder so hardmax stays in lockstep with the
/// probability kernels.
pub use softmax::{Builder, Dispatch, Pipelines};

/// Construct the compute pipelines used by the hardmax helpers.
pub fn create_pipelines(
    device: &Device,
    shader_dir: impl AsRef<std::path::Path>,
    supports_subgroup: bool,
) -> Result<Pipelines, crate::ShaderLoadError> {
    softmax::create_pipelines(device, shader_dir, supports_subgroup)
}

/// Upload the configured hardmax uniforms to a GPU buffer.
pub fn upload_params(device: &Device, queue: &Queue, params: &Params) -> Buffer {
    softmax::upload_params(device, queue, params.as_softmax())
}
