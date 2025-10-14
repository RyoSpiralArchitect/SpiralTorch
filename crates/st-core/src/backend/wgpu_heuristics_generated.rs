// Auto-generated from tuner_results.json
use crate::backend::device_caps::DeviceCaps as GenDeviceCaps;
use crate::backend::wasm_tuner::WasmTunerTable;
use crate::backend::wgpu_heuristics::Choice as GenChoice;
use std::sync::OnceLock;

fn base_choice(rows: usize, cols: usize, k: usize, subgroup: bool) -> GenChoice {
    let max_wg = if subgroup { 256 } else { 128 };
    let caps = GenDeviceCaps::wgpu(32, subgroup, max_wg);
    let use_2ce = caps.prefers_two_stage_with_rows(rows as u32, cols as u32, k as u32);
    let wg = caps.recommended_workgroup(rows as u32);
    let kl = caps.recommended_kl(k as u32);
    let ch = caps.recommended_channel_stride(cols as u32);
    let ctile = caps.recommended_compaction_tile_default(cols as u32);
    GenChoice {
        use_2ce,
        wg,
        kl,
        ch,
        algo_topk: 0,
        ctile,
        mode_midk: 0,
        mode_bottomk: 0,
        tile_cols: ((cols.max(1) + 1023) / 1024) as u32 * 1024,
        radix: if k.is_power_of_two() { 4 } else { 2 },
        segments: if cols > 131_072 {
            4
        } else if cols > 32_768 {
            2
        } else {
            1
        },
    }
}

fn table() -> &'static WasmTunerTable {
    static TABLE: OnceLock<WasmTunerTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        WasmTunerTable::from_json_str(r#"[{"rows":256,"cols_min":0,"cols_max":4095,"k_max":128,"sg":true,"wg":128,"tile":512,"tile_cols":512,"radix":2,"segments":1},{"rows":512,"cols_min":4096,"cols_max":16383,"k_max":256,"sg":true,"wg":256,"tile":1024,"tile_cols":1024,"radix":4,"segments":2},{"rows":512,"cols_min":16384,"cols_max":65535,"k_max":2048,"sg":false,"wg":128,"tile":2048,"tile_cols":2048,"radix":4,"segments":4,"use_2ce":true},{"rows":1024,"cols_min":65536,"cols_max":262143,"k_max":4096,"sg":false,"wg":128,"tile":4096,"tile_cols":4096,"radix":4,"segments":4,"use_2ce":true,"mode_bottomk":2}]"#).expect("invalid tuner table")
    })
}

pub fn choose(
    rows: usize,
    cols: usize,
    k: usize,
    subgroup: bool,
) -> Option<super::wgpu_heuristics::Choice> {
    let base = base_choice(rows, cols, k, subgroup);
    table().choose(base, rows, cols, k, subgroup)
}
