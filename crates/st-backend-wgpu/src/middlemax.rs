// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Convenience wrappers that focus the MidK/BottomK compaction family on the
//! `middlemax` reduction stage. The goal is to mirror the high-level ergonomics
//! provided by [`softmax`](crate::softmax) and the tensor-level [`hardmax`]
//! module so the Rank-K family exposes "soft/middle/hard" entry points with a
//! consistent surface.
//!
//! The underlying implementation reuses the compaction pipelines defined in
//! [`midk_bottomk`](crate::midk_bottomk). This module constrains the dispatch
//! arguments so the optional `out_middlemax` buffer becomes mandatory, forwards
//! the encode/dispatch helpers, and offers conversion utilities to safely adapt
//! a broader MidK/BottomK call-site when the middlemax reduction is required.

use std::{convert::TryFrom, fmt};

use crate::midk_bottomk;
use crate::ShaderLoadError;
use wgpu::{CommandBuffer, CommandEncoder, Device, Queue};

pub use crate::midk_bottomk::{
    ApplyStrategy, DispatchValidationError, ElementCounts, Kind, Pipelines,
};
pub use midk_bottomk::Builder;

/// Arguments required to launch the `middlemax` stage of the MidK/BottomK
/// compaction kernels.
pub struct DispatchArgs<'a> {
    /// Number of rows to process.
    pub rows: u32,
    /// Number of columns within each row.
    pub cols: u32,
    /// Stride between rows in the input/output buffers.
    pub row_stride: u32,
    /// Whether to run the kernels in MidK or BottomK mode.
    pub kind: Kind,
    /// Buffer containing the input values that will be compacted.
    pub values: &'a wgpu::Buffer,
    /// Buffer containing the predicate mask (non-zero entries are kept).
    pub mask: &'a wgpu::Buffer,
    /// Buffer that receives the per-row output counts written by the scan stage.
    pub out_positions: &'a wgpu::Buffer,
    /// Buffer that receives the compacted values.
    pub out_values: &'a wgpu::Buffer,
    /// Scratch buffer that stores per-tile prefix totals.
    pub prefix: &'a wgpu::Buffer,
    /// Buffer that receives the per-row middle-band maxima.
    pub out_middlemax: &'a wgpu::Buffer,
}

impl<'a> DispatchArgs<'a> {
    /// Ensure the dispatch arguments describe a valid tensor layout.
    pub fn validate(&self) -> Result<(), DispatchValidationError> {
        self.borrow_midk_args().validate()
    }

    /// Return `true` when there is no work to execute.
    pub fn is_empty(&self) -> bool {
        self.borrow_midk_args().is_empty()
    }

    /// Calculate the number of elements each buffer must contain.
    pub fn element_counts(&self) -> ElementCounts {
        self.borrow_midk_args().element_counts()
    }

    fn borrow_midk_args(&self) -> midk_bottomk::DispatchArgs<'a> {
        midk_bottomk::DispatchArgs {
            rows: self.rows,
            cols: self.cols,
            row_stride: self.row_stride,
            kind: self.kind,
            values: self.values,
            mask: self.mask,
            out_positions: self.out_positions,
            out_values: self.out_values,
            out_middlemax: Some(self.out_middlemax),
            prefix: self.prefix,
        }
    }

    fn into_midk_args(self) -> midk_bottomk::DispatchArgs<'a> {
        let DispatchArgs {
            rows,
            cols,
            row_stride,
            kind,
            values,
            mask,
            out_positions,
            out_values,
            prefix,
            out_middlemax,
        } = self;

        midk_bottomk::DispatchArgs {
            rows,
            cols,
            row_stride,
            kind,
            values,
            mask,
            out_positions,
            out_values,
            out_middlemax: Some(out_middlemax),
            prefix,
        }
    }
}

/// Error returned when trying to adapt generic MidK/BottomK arguments that did
/// not opt into the `middlemax` stage.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MissingMiddlemaxBufferError;

impl fmt::Display for MissingMiddlemaxBufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("middlemax dispatch requires `out_middlemax` to be provided")
    }
}

impl std::error::Error for MissingMiddlemaxBufferError {}

impl<'a> DispatchArgs<'a> {
    /// Attempt to adapt the broader MidK/BottomK dispatch arguments for the
    /// `middlemax` helpers.
    pub fn try_from_midk_args(
        args: midk_bottomk::DispatchArgs<'a>,
    ) -> Result<Self, MissingMiddlemaxBufferError> {
        Self::try_from(args)
    }
}

impl<'a> TryFrom<midk_bottomk::DispatchArgs<'a>> for DispatchArgs<'a> {
    type Error = MissingMiddlemaxBufferError;

    fn try_from(args: midk_bottomk::DispatchArgs<'a>) -> Result<Self, Self::Error> {
        let midk_bottomk::DispatchArgs {
            rows,
            cols,
            row_stride,
            kind,
            values,
            mask,
            out_positions,
            out_values,
            out_middlemax,
            prefix,
        } = args;

        let out_middlemax = out_middlemax.ok_or(MissingMiddlemaxBufferError)?;

        Ok(Self {
            rows,
            cols,
            row_stride,
            kind,
            values,
            mask,
            out_positions,
            out_values,
            prefix,
            out_middlemax,
        })
    }
}

/// Build the compaction pipelines required by the middlemax helpers.
pub fn create_pipelines(
    device: &Device,
    shader_dir: &str,
    supports_subgroup: bool,
    include_v1: bool,
    include_v2: bool,
) -> Result<Pipelines, ShaderLoadError> {
    midk_bottomk::create_pipelines(
        device,
        shader_dir,
        supports_subgroup,
        include_v1,
        include_v2,
    )
}

/// Dispatch the middlemax reduction along with the prerequisite compaction
/// stages.
pub fn dispatch(
    device: &Device,
    queue: &Queue,
    pipelines: &Pipelines,
    args: DispatchArgs<'_>,
    strategy: ApplyStrategy,
) -> Result<(), DispatchValidationError> {
    midk_bottomk::dispatch(device, queue, pipelines, args.into_midk_args(), strategy)
}

/// Encode the middlemax reduction into a fresh command buffer.
pub fn encode(
    device: &Device,
    pipelines: &Pipelines,
    args: DispatchArgs<'_>,
    strategy: ApplyStrategy,
) -> Result<Option<CommandBuffer>, DispatchValidationError> {
    midk_bottomk::encode(device, pipelines, args.into_midk_args(), strategy)
}

/// Encode the middlemax reduction using an existing command encoder.
pub fn encode_into(
    device: &Device,
    encoder: &mut CommandEncoder,
    pipelines: &Pipelines,
    args: DispatchArgs<'_>,
    strategy: ApplyStrategy,
) -> Result<bool, DispatchValidationError> {
    midk_bottomk::encode_into(device, encoder, pipelines, args.into_midk_args(), strategy)
}
