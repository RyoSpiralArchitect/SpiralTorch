// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-core/src/backend/hip_loader.rs (skeleton)
#[cfg(feature="hip")]
pub struct HipModule {}
#[cfg(feature="hip")]
pub fn load_hsaco_module(_hsaco:&[u8]) -> Result<HipModule,String> { Ok(HipModule{}) }
