// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-core/src/backend/cuda_loader.rs (skeleton)
#[cfg(feature="cuda")]
pub struct CudaModule {}
#[cfg(feature="cuda")]
pub fn load_ptx_module(_ptx:&[u8]) -> Result<CudaModule,String> { Ok(CudaModule{}) }
