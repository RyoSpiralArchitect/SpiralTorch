// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Small demo for KDSL explainability (structured evaluation trace).

use st_kdsl::{eval_program_with_trace, Ctx};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let script = r#"
        let base = r / 4;
        wg: base;
        soft(wg, base, 0.5, true);
        soft(radix, base, 1.0, false);

        for i in 0..4 {
            tile_cols: i + 1;
        }
    "#;

    let ctx = Ctx {
        r: 1024,
        c: 16384,
        k: 512,
        sg: false,
        sgc: 1,
        kc: 1,
        tile_cols: 64,
        radix: 4,
        segments: 1,
    };

    let (out, trace) = eval_program_with_trace(script, &ctx, 128)?;
    println!(
        "hard: wg={:?} tile_cols={:?} radix={:?} segments={:?}",
        out.hard.wg, out.hard.tile_cols, out.hard.radix, out.hard.segments
    );
    println!("soft_rules={}", out.soft.len());
    println!("{}", serde_json::to_string_pretty(&trace)?);
    Ok(())
}

