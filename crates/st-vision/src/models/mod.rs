// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Vision model backbones implemented on top of the SpiralTorch neural module stack.
//!
//! Each module implements [`st_nn::module::Module`], enabling downstream
//! consumers to attach optimisers, export checkpoints, and integrate with
//! SpiralTorch's training utilities.

mod utils;

pub mod convnext;
pub mod resnet;
pub mod vit;

pub use self::convnext::{ConvNeXtBackbone, ConvNeXtConfig};
pub use self::resnet::{ResNetBackbone, ResNetConfig};
pub use self::vit::{ViTBackbone, ViTConfig};
