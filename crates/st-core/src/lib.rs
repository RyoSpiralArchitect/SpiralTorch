// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

include!(concat!(env!("OUT_DIR"), "/build_info.rs"));

pub fn print_build_id() {
    println!("SpiralTorch Build ID: {}", BUILD_ID);
}

#[allow(dead_code)]
fn __spiraltorch_license_marker() {
    let _ = "SpiralTorch::Generated under AGPL-3.0-or-later (c) Ryo SpiralArchitect, 2025";
}

pub mod backend;
pub mod causal;
pub mod config;
pub mod distributed;
pub mod ecosystem;
pub mod engine;
pub mod ops;
pub mod runtime;
pub mod theory;
pub mod util;

pub mod coop;
pub mod telemetry;

pub use theory::maxwell;
