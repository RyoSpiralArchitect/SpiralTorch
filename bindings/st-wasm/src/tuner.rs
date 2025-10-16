use js_sys::JSON;
use serde::Serialize;
use st_core::backend::device_caps::DeviceCaps;
use st_core::backend::wasm_tuner::{WasmTunerRecord, WasmTunerTable};
use st_core::backend::wgpu_heuristics::Choice;
use wasm_bindgen::prelude::*;

use crate::fft::WasmFftPlan;
use crate::utils::js_error;

#[wasm_bindgen]
pub struct WasmTuner {
    table: WasmTunerTable,
}

#[wasm_bindgen]
impl WasmTuner {
    /// Construct a tuner table. Pass `None` to start from an empty dataset or a JSON string to seed it.
    #[wasm_bindgen(constructor)]
    pub fn new(json: Option<String>) -> Result<WasmTuner, JsValue> {
        let table = match json {
            Some(json) => WasmTunerTable::from_json_str(&json).map_err(js_error)?,
            None => WasmTunerTable::new(),
        };
        Ok(Self { table })
    }

    /// Replace the table contents with the provided JSON blob.
    #[wasm_bindgen(js_name = loadJson)]
    pub fn load_json(&mut self, json: &str) -> Result<(), JsValue> {
        self.table = WasmTunerTable::from_json_str(json).map_err(js_error)?;
        Ok(())
    }

    /// Merge additional overrides from a JSON blob, keeping the ordering stable.
    #[wasm_bindgen(js_name = mergeJson)]
    pub fn merge_json(&mut self, json: &str) -> Result<(), JsValue> {
        let parsed = WasmTunerTable::from_json_str(json).map_err(js_error)?;
        let records = parsed.iter().cloned().collect::<Vec<_>>();
        self.table.extend_sorted(records);
        Ok(())
    }

    /// Number of records available in the table.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Returns `true` when no overrides are stored.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Clear every record from the table.
    pub fn clear(&mut self) {
        self.table.clear();
    }

    /// Append a new record represented as a JavaScript object.
    pub fn push(&mut self, record: JsValue) -> Result<(), JsValue> {
        let record = parse_record(record)?;
        self.table.push_sorted(record);
        Ok(())
    }

    /// Return the internal dataset as a JSON string.
    pub fn to_json(&self) -> Result<String, JsValue> {
        self.table.to_json().map_err(js_error)
    }

    /// Return the dataset as an array of JavaScript objects.
    pub fn records(&self) -> Result<JsValue, JsValue> {
        let records: Vec<WasmTunerRecord> = self.table.iter().cloned().collect();
        records_to_js(&records)
    }

    /// Query the tuner for a given workload. Returns `undefined` when no override matches.
    pub fn choose(&self, rows: u32, cols: u32, k: u32, subgroup: bool) -> Result<JsValue, JsValue> {
        let base = base_choice(rows as usize, cols as usize, k as usize, subgroup);
        let chosen = self
            .table
            .choose(base, rows as usize, cols as usize, k as usize, subgroup);
        match chosen {
            Some(choice) => choice_to_js(choice),
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Produce a tuned FFT plan for the provided workload if an override exists.
    #[wasm_bindgen(js_name = planFft)]
    pub fn plan_fft(&self, rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<WasmFftPlan> {
        let base = base_choice(rows as usize, cols as usize, k as usize, subgroup);
        let choice = self
            .table
            .choose(base, rows as usize, cols as usize, k as usize, subgroup)?;
        Some(WasmFftPlan::from_choice(choice, subgroup))
    }
}

/// Compute the baseline WGPU choice without consulting an override table.
#[wasm_bindgen(js_name = baseChoice)]
pub fn base_choice_js(rows: u32, cols: u32, k: u32, subgroup: bool) -> Result<JsValue, JsValue> {
    choice_to_js(base_choice(
        rows as usize,
        cols as usize,
        k as usize,
        subgroup,
    ))
}

fn base_choice(rows: usize, cols: usize, k: usize, subgroup: bool) -> Choice {
    let max_wg = if subgroup { 256 } else { 128 };
    let caps = DeviceCaps::wgpu(32, subgroup, max_wg);
    let rows = rows as u32;
    let cols = cols as u32;
    let k = k as u32;
    let use_2ce = caps.prefers_two_stage_with_rows(rows, cols, k);
    let wg = caps.recommended_workgroup(rows);
    let kl = caps.recommended_kl(k);
    let ch = caps.recommended_channel_stride(cols);
    let ctile = caps.recommended_compaction_tile_default(cols);
    Choice {
        use_2ce,
        wg,
        kl,
        ch,
        algo_topk: 0,
        ctile,
        mode_midk: 0,
        mode_bottomk: 0,
        tile_cols: ((cols.max(1) + 1023) / 1024) * 1024,
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

fn choice_to_js(choice: Choice) -> Result<JsValue, JsValue> {
    let json = serde_json::to_string(&ChoiceSerde::from(choice)).map_err(js_error)?;
    JSON::parse(&json).map_err(|err| js_error(js_value_to_string(&err)))
}

#[derive(Serialize)]
struct ChoiceSerde {
    use_2ce: bool,
    wg: u32,
    kl: u32,
    ch: u32,
    algo_topk: u8,
    ctile: u32,
    mode_midk: u8,
    mode_bottomk: u8,
    tile_cols: u32,
    radix: u32,
    segments: u32,
}

fn parse_record(value: JsValue) -> Result<WasmTunerRecord, JsValue> {
    let json = stringify_js_value(&value)?;
    serde_json::from_str(&json).map_err(js_error)
}

fn records_to_js(records: &[WasmTunerRecord]) -> Result<JsValue, JsValue> {
    let json = serde_json::to_string(records).map_err(js_error)?;
    JSON::parse(&json).map_err(|err| js_error(js_value_to_string(&err)))
}

fn stringify_js_value(value: &JsValue) -> Result<String, JsValue> {
    let json = JSON::stringify(value).map_err(|err| js_error(js_value_to_string(&err)))?;
    json.as_string()
        .ok_or_else(|| js_error("expected JSON string"))
}

fn js_value_to_string(value: &JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

impl From<Choice> for ChoiceSerde {
    fn from(choice: Choice) -> Self {
        Self {
            use_2ce: choice.use_2ce,
            wg: choice.wg,
            kl: choice.kl,
            ch: choice.ch,
            algo_topk: choice.algo_topk,
            ctile: choice.ctile,
            mode_midk: choice.mode_midk,
            mode_bottomk: choice.mode_bottomk,
            tile_cols: choice.tile_cols,
            radix: choice.radix,
            segments: choice.segments,
        }
    }
}
