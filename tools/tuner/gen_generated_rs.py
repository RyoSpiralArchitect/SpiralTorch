# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#!/usr/bin/env python3
import json
import sys

def main():
    if len(sys.argv) < 2:
        print("usage: gen_generated_rs.py tuner_results.json", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        data = json.load(fh)

    json_blob = json.dumps(data, separators=(",", ":"))

    print("// Auto-generated from tuner_results.json")
    print("use crate::backend::device_caps::DeviceCaps as GenDeviceCaps;")
    print("use crate::backend::wasm_tuner::WasmTunerTable;")
    print("use crate::backend::wgpu_heuristics::Choice as GenChoice;")
    print("use std::sync::OnceLock;")
    print("")
    print("fn base_choice(rows: usize, cols: usize, k: usize, subgroup: bool) -> GenChoice {")
    print("    let max_wg = if subgroup { 256 } else { 128 };")
    print("    let caps = GenDeviceCaps::wgpu(32, subgroup, max_wg);")
    print("    let use_2ce = caps.prefers_two_stage_with_rows(rows as u32, cols as u32, k as u32);")
    print("    let wg = caps.recommended_workgroup(rows as u32);")
    print("    let kl = caps.recommended_kl(k as u32);")
    print("    let ch = caps.recommended_channel_stride(cols as u32);")
    print("    let ctile = caps.recommended_compaction_tile_default(cols as u32);")
    print("    GenChoice {")
    print("        use_2ce,")
    print("        wg,")
    print("        kl,")
    print("        ch,")
    print("        algo_topk: 0,")
    print("        ctile,")
    print("        mode_midk: 0,")
    print("        mode_bottomk: 0,")
    print("        tile_cols: ((cols.max(1) + 1023) / 1024) as u32 * 1024,")
    print("        radix: if k.is_power_of_two() { 4 } else { 2 },")
    print("        segments: if cols > 131_072 { 4 } else if cols > 32_768 { 2 } else { 1 },")
    print("    }")
    print("}")
    print("")
    print("fn table() -> &'static WasmTunerTable {")
    print("    static TABLE: OnceLock<WasmTunerTable> = OnceLock::new();")
    print("    TABLE.get_or_init(|| {")
    print("        WasmTunerTable::from_json_str(r#\"%s\"#).expect(\"invalid tuner table\")" % json_blob)
    print("    })")
    print("}")
    print("")
    print("pub fn choose(")
    print("    rows: usize,")
    print("    cols: usize,")
    print("    k: usize,")
    print("    subgroup: bool,")
    print(") -> Option<super::wgpu_heuristics::Choice> {")
    print("    let base = base_choice(rows, cols, k, subgroup);")
    print("    table().choose(base, rows, cols, k, subgroup)")
    print("}")

if __name__ == "__main__":
    main()
