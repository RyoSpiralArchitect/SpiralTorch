// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Helpers for turning SpiralK heuristic choices into concrete WGSL FFT kernels.
//!
//! The module provides a small bridge between the runtime heuristics (which
//! work in terms of `Choice`) and the browser/WASM path that expects ready-made
//! WGSL shaders.  The generated source focuses on clarity so integrators can
//! inspect or tweak the kernels before shipping them to end users.

use crate::backend::wgpu_heuristics::Choice;

/// Plan describing the FFT kernel that should be emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpiralKFftPlan {
    /// Preferred radix (2 or 4).
    pub radix: u32,
    /// Tile columns processed by each dispatch.
    pub tile_cols: u32,
    /// Number of ND segments folded inside the kernel.
    pub segments: u32,
    /// Whether the kernel targets subgroup execution.
    pub subgroup: bool,
}

impl SpiralKFftPlan {
    /// Construct a plan from a heuristic `Choice`.
    pub fn from_choice(choice: &Choice, subgroup: bool) -> Self {
        Self {
            radix: choice.radix.max(2).min(4),
            tile_cols: choice.tile_cols.max(1),
            segments: choice.segments.max(1),
            subgroup,
        }
    }

    /// Size of the compute workgroup.
    pub fn workgroup_size(&self) -> u32 {
        if self.subgroup {
            32
        } else {
            64
        }
    }

    /// Emit a WGSL kernel tuned for the current plan.  The shader keeps the
    /// structure intentionally simple so it can run inside WebGPU-enabled
    /// browsers without extra bindings.
    pub fn emit_wgsl(&self) -> String {
        let wg = self.workgroup_size();
        format!(
            "// Auto-generated WGSL kernel (radix {radix}, tile {tile}, segments {segments})\n\
             struct Complex {{ re: f32, im: f32 }};\n\
             @group(0) @binding(0) var<storage, read_write> data: array<Complex>;\n\
             fn twiddle(k: u32, stride: u32) -> Complex {{\n\
                 let angle = 6.2831855 * f32(k) / f32(stride);\n\
                 return Complex( cos(angle), -sin(angle) );\n\
             }}\n\
             fn mul(a: Complex, b: Complex) -> Complex {{\n\
                 return Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);\n\
             }}\n\
             fn add(a: Complex, b: Complex) -> Complex {{\n\
                 return Complex(a.re + b.re, a.im + b.im);\n\
             }}\n\
             fn sub(a: Complex, b: Complex) -> Complex {{\n\
                 return Complex(a.re - b.re, a.im - b.im);\n\
             }}\n\
             @compute @workgroup_size({wg})\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
                 if (gid.x >= {tile}u || gid.y >= {segments}u) {{\n\
                     return;\n\
                 }}\n\
                 let base = gid.y * {tile}u + gid.x;\n\
                 var stride = {radix}u;\n\
                 loop {{\n\
                     if (stride > {tile}u) {{ break; }}\n\
                     let tw = twiddle(gid.x % stride, stride);\n\
                     let a = data[base];\n\
                     let b = data[base + stride / {radix}u];\n\
                     let top = add(a, mul(b, tw));\n\
                     let bottom = sub(a, mul(b, tw));\n\
                     data[base] = top;\n\
                     data[base + stride / {radix}u] = bottom;\n\
                     stride = stride * {radix}u;\n\
                 }}\n\
             }}\n",
            radix = self.radix,
            tile = self.tile_cols,
            segments = self.segments,
            wg = wg,
        )
    }

    /// Emit a small SpiralK snippet that mirrors the generated kernel.  This is
    /// useful when the DSL needs to record the choice for future runs.
    pub fn emit_spiralk_hint(&self) -> String {
        format!(
            "soft (tile_cols, {tile}, 0.85, c >= {tile});\nsoft (radix, {radix}, 0.90, true);\nsoft (segments, {segments}, 0.75, true);",
            tile = self.tile_cols,
            radix = self.radix,
            segments = self.segments,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_includes_plan_information() {
        let choice = Choice {
            use_2ce: true,
            wg: 256,
            kl: 16,
            ch: 8192,
            algo_topk: 2,
            ctile: 1024,
            mode_midk: 2,
            mode_bottomk: 1,
            tile_cols: 2048,
            radix: 4,
            segments: 3,
        };
        let plan = SpiralKFftPlan::from_choice(&choice, true);
        let src = plan.emit_wgsl();
        assert!(src.contains("radix 4"));
        assert!(src.contains("@workgroup_size(32)"));
        assert!(src.contains("gid.y >= 3u"));
        let hint = plan.emit_spiralk_hint();
        assert!(hint.contains("tile_cols"));
    }
}
