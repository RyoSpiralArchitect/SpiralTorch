// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Pure, backend-neutral kernel semantics for SpiralTorch.
//!
//! This crate owns validated data contracts and CPU reference oracles only. It
//! deliberately contains no runtime routing, device discovery, telemetry,
//! allocation, or foreign-function interfaces.

pub mod compaction;
