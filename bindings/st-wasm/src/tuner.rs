use serde::{Deserialize, Serialize};
use st_core::backend::device_caps::DeviceCaps;
use st_core::backend::wasm_tuner::{WasmTunerRecord, WasmTunerTable};
use st_core::backend::wgpu_heuristics::{self, Choice};
use wasm_bindgen::prelude::*;

use crate::fft::{WasmFftPlan, WasmFftPlanSerde};
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

    /// Construct a tuner from a JavaScript array of records.
    #[wasm_bindgen(js_name = fromObject)]
    pub fn from_object(value: &JsValue) -> Result<WasmTuner, JsValue> {
        let records = parse_records(value)?;
        Ok(Self {
            table: WasmTunerTable::from_records(records),
        })
    }

    /// Replace the table contents with the provided JSON blob.
    #[wasm_bindgen(js_name = loadJson)]
    pub fn load_json(&mut self, json: &str) -> Result<(), JsValue> {
        self.table = WasmTunerTable::from_json_str(json).map_err(js_error)?;
        Ok(())
    }

    /// Replace the table contents with the provided array of records.
    #[wasm_bindgen(js_name = loadObject)]
    pub fn load_object(&mut self, value: &JsValue) -> Result<(), JsValue> {
        let records = parse_records(value)?;
        self.table = WasmTunerTable::from_records(records);
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

    /// Merge additional overrides from a JavaScript array of records.
    #[wasm_bindgen(js_name = mergeObject)]
    pub fn merge_object(&mut self, value: &JsValue) -> Result<(), JsValue> {
        let records = parse_records(value)?;
        self.table.extend_sorted(records);
        Ok(())
    }

    /// Number of records available in the table.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Retrieve a record by index. Returns `undefined` when the index is out of bounds.
    #[wasm_bindgen(js_name = recordAt)]
    pub fn record_at(&self, index: u32) -> Result<JsValue, JsValue> {
        match self.table.get(index as usize) {
            Some(record) => record_to_js(record),
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Locate the first record matching the provided workload. Returns `undefined` when nothing matches.
    #[wasm_bindgen(js_name = findRecord)]
    pub fn find_record(
        &self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> Result<JsValue, JsValue> {
        match self
            .table
            .find_record(rows as usize, cols as usize, k as usize, subgroup)
        {
            Some(record) => record_to_js(record),
            None => Ok(JsValue::UNDEFINED),
        }
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

    /// Replace the record at the provided index. Returns `true` when the index existed.
    #[wasm_bindgen(js_name = replaceIndex)]
    pub fn replace_index(&mut self, index: u32, record: JsValue) -> Result<bool, JsValue> {
        let record = parse_record(record)?;
        Ok(self.table.replace(index as usize, record))
    }

    /// Remove the record at the provided index and return it. Returns `undefined` when the index was invalid.
    #[wasm_bindgen(js_name = removeIndex)]
    pub fn remove_index(&mut self, index: u32) -> Result<JsValue, JsValue> {
        match self.table.remove(index as usize) {
            Some(record) => {
                let js = record_to_js(&record)?;
                Ok(js)
            }
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Remove the first record matching the workload. Returns `undefined` when nothing matched.
    #[wasm_bindgen(js_name = removeRecord)]
    pub fn remove_record(
        &mut self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> Result<JsValue, JsValue> {
        match self
            .table
            .remove_matching(rows as usize, cols as usize, k as usize, subgroup)
        {
            Some(record) => {
                let js = record_to_js(&record)?;
                Ok(js)
            }
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Return the internal dataset as a JSON string.
    pub fn to_json(&self) -> Result<String, JsValue> {
        self.table.to_json().map_err(js_error)
    }

    /// Return the dataset as an array of JavaScript objects.
    pub fn records(&self) -> Result<JsValue, JsValue> {
        self.to_object()
    }

    /// Export the dataset into a JavaScript array of records.
    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> Result<JsValue, JsValue> {
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

    /// Produce a tuned FFT plan encoded as JSON when an override exists.
    #[wasm_bindgen(js_name = planFftJson)]
    pub fn plan_fft_json(
        &self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> Result<Option<String>, JsValue> {
        match self.plan_fft(rows, cols, k, subgroup) {
            Some(plan) => Ok(Some(plan.to_json()?)),
            None => Ok(None),
        }
    }

    /// Produce a tuned FFT plan encoded as a plain JavaScript object when an override exists.
    #[wasm_bindgen(js_name = planFftObject)]
    pub fn plan_fft_object(
        &self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> Result<Option<JsValue>, JsValue> {
        match self.plan_fft(rows, cols, k, subgroup) {
            Some(plan) => Ok(Some(plan.to_object()?)),
            None => Ok(None),
        }
    }

    /// Resolve a tuned FFT plan. Falls back to heuristics (or base device caps)
    /// when no explicit override matches.
    #[wasm_bindgen(js_name = planFftWithFallback)]
    pub fn plan_fft_with_fallback(
        &self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> WasmFftPlan {
        let (choice, _, _) = self.resolve_fft_choice(rows, cols, k, subgroup);
        WasmFftPlan::from_choice(choice, subgroup)
    }

    /// Resolve a tuned FFT plan and encode it as JSON.
    #[wasm_bindgen(js_name = planFftWithFallbackJson)]
    pub fn plan_fft_with_fallback_json(
        &self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> Result<String, JsValue> {
        let plan = self.plan_fft_with_fallback(rows, cols, k, subgroup);
        plan.to_json()
    }

    /// Resolve a tuned FFT plan and encode it as a plain JavaScript object.
    #[wasm_bindgen(js_name = planFftWithFallbackObject)]
    pub fn plan_fft_with_fallback_object(
        &self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> Result<JsValue, JsValue> {
        let plan = self.plan_fft_with_fallback(rows, cols, k, subgroup);
        plan.to_object()
    }

    /// Resolve a tuned FFT plan and return metadata describing how the plan was
    /// assembled (override vs. heuristic vs. fallback).
    #[wasm_bindgen(js_name = planFftResolution)]
    pub fn plan_fft_resolution(
        &self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> ResolvedWasmFftPlan {
        let (choice, source, heuristic_used) = self.resolve_fft_choice(rows, cols, k, subgroup);
        let plan = WasmFftPlan::from_choice(choice, subgroup);
        ResolvedWasmFftPlan::new(plan, source, heuristic_used)
    }

    /// Resolve a tuned FFT plan and encode the metadata report as JSON.
    #[wasm_bindgen(js_name = planFftResolutionJson)]
    pub fn plan_fft_resolution_json(
        &self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> Result<String, JsValue> {
        let resolved = self.plan_fft_resolution(rows, cols, k, subgroup);
        resolved.to_json()
    }

    /// Resolve a tuned FFT plan and encode the metadata report as a plain JavaScript object.
    #[wasm_bindgen(js_name = planFftResolutionObject)]
    pub fn plan_fft_resolution_object(
        &self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> Result<JsValue, JsValue> {
        let resolved = self.plan_fft_resolution(rows, cols, k, subgroup);
        resolved.to_object()
    }

    /// Resolve a tuned FFT plan and return a JSON-ready report describing how the plan was
    /// assembled (override vs. heuristic vs. fallback).
    #[wasm_bindgen(js_name = planFftReport)]
    pub fn plan_fft_report(
        &self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> Result<JsValue, JsValue> {
        self.plan_fft_resolution_object(rows, cols, k, subgroup)
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

/// Emit the auto-generated WGSL kernel using the native heuristics pipeline.
#[wasm_bindgen]
pub fn auto_fft_wgsl(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<String> {
    wgpu_heuristics::auto_fft_wgsl(rows, cols, k, subgroup)
}

/// Emit the SpiralK hint associated with the generated WGSL kernel.
#[wasm_bindgen]
pub fn auto_fft_spiralk(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<String> {
    wgpu_heuristics::auto_fft_spiralk(rows, cols, k, subgroup)
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
    JsValue::from_serde(&ChoiceSerde::from(choice)).map_err(js_error)
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
    value.into_serde::<WasmTunerRecord>().map_err(js_error)
}

fn records_to_js(records: &[WasmTunerRecord]) -> Result<JsValue, JsValue> {
    JsValue::from_serde(records).map_err(js_error)
}

fn parse_records(value: &JsValue) -> Result<Vec<WasmTunerRecord>, JsValue> {
    value.into_serde::<Vec<WasmTunerRecord>>().map_err(js_error)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlanSource {
    Override,
    Heuristic,
    Fallback,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum PlanSourceSerde {
    Override,
    Heuristic,
    Fallback,
}

impl From<PlanSource> for PlanSourceSerde {
    fn from(source: PlanSource) -> Self {
        match source {
            PlanSource::Override => PlanSourceSerde::Override,
            PlanSource::Heuristic => PlanSourceSerde::Heuristic,
            PlanSource::Fallback => PlanSourceSerde::Fallback,
        }
    }
}

impl From<PlanSourceSerde> for PlanSource {
    fn from(source: PlanSourceSerde) -> Self {
        match source {
            PlanSourceSerde::Override => PlanSource::Override,
            PlanSourceSerde::Heuristic => PlanSource::Heuristic,
            PlanSourceSerde::Fallback => PlanSource::Fallback,
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmFftPlanSource {
    Override,
    Heuristic,
    Fallback,
}

impl From<PlanSource> for WasmFftPlanSource {
    fn from(source: PlanSource) -> Self {
        match source {
            PlanSource::Override => WasmFftPlanSource::Override,
            PlanSource::Heuristic => WasmFftPlanSource::Heuristic,
            PlanSource::Fallback => WasmFftPlanSource::Fallback,
        }
    }
}

impl From<WasmFftPlanSource> for PlanSource {
    fn from(source: WasmFftPlanSource) -> Self {
        match source {
            WasmFftPlanSource::Override => PlanSource::Override,
            WasmFftPlanSource::Heuristic => PlanSource::Heuristic,
            WasmFftPlanSource::Fallback => PlanSource::Fallback,
        }
    }
}

impl From<WasmFftPlanSource> for PlanSourceSerde {
    fn from(source: WasmFftPlanSource) -> Self {
        PlanSourceSerde::from(PlanSource::from(source))
    }
}

impl From<PlanSourceSerde> for WasmFftPlanSource {
    fn from(source: PlanSourceSerde) -> Self {
        WasmFftPlanSource::from(PlanSource::from(source))
    }
}

#[wasm_bindgen]
pub struct ResolvedWasmFftPlan {
    plan: WasmFftPlanSerde,
    override_applied: bool,
    heuristic_used: bool,
    source: PlanSource,
}

impl ResolvedWasmFftPlan {
    fn new(plan: WasmFftPlan, source: PlanSource, heuristic_used: bool) -> Self {
        Self {
            plan: plan.to_serde(),
            override_applied: matches!(source, PlanSource::Override),
            heuristic_used,
            source,
        }
    }

    fn from_serde(serde: ResolvedPlanSerde) -> Self {
        Self {
            plan: serde.plan,
            override_applied: serde.override_applied,
            heuristic_used: serde.heuristic_used,
            source: PlanSource::from(serde.source),
        }
    }

    fn to_serde(&self) -> ResolvedPlanSerde {
        ResolvedPlanSerde {
            plan: self.plan.clone(),
            override_applied: self.override_applied,
            heuristic_used: self.heuristic_used,
            source: PlanSourceSerde::from(self.source),
        }
    }
}

#[wasm_bindgen]
impl ResolvedWasmFftPlan {
    #[wasm_bindgen(getter)]
    pub fn plan(&self) -> WasmFftPlan {
        WasmFftPlan::from(self.plan.clone())
    }

    #[wasm_bindgen(getter, js_name = overrideApplied)]
    pub fn override_applied(&self) -> bool {
        self.override_applied
    }

    #[wasm_bindgen(getter, js_name = heuristicUsed)]
    pub fn heuristic_used(&self) -> bool {
        self.heuristic_used
    }

    #[wasm_bindgen(getter)]
    pub fn source(&self) -> WasmFftPlanSource {
        WasmFftPlanSource::from(self.source)
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.to_serde()).map_err(js_error)
    }

    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> Result<JsValue, JsValue> {
        JsValue::from_serde(&self.to_serde()).map_err(js_error)
    }

    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<ResolvedWasmFftPlan, JsValue> {
        let parsed = serde_json::from_str::<ResolvedPlanSerde>(json).map_err(js_error)?;
        Ok(Self::from_serde(parsed))
    }

    #[wasm_bindgen(js_name = fromObject)]
    pub fn from_object(value: &JsValue) -> Result<ResolvedWasmFftPlan, JsValue> {
        let parsed = value.into_serde::<ResolvedPlanSerde>().map_err(js_error)?;
        Ok(Self::from_serde(parsed))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ResolvedPlanSerde {
    plan: WasmFftPlanSerde,
    #[serde(rename = "overrideApplied")]
    override_applied: bool,
    #[serde(rename = "heuristicUsed")]
    heuristic_used: bool,
    source: PlanSourceSerde,
}

impl WasmTuner {
    fn resolve_fft_choice(
        &self,
        rows: u32,
        cols: u32,
        k: u32,
        subgroup: bool,
    ) -> (Choice, PlanSource, bool) {
        let fallback = base_choice(rows as usize, cols as usize, k as usize, subgroup);
        let (candidate, heuristic_used) =
            match wgpu_heuristics::choose_topk(rows, cols, k, subgroup) {
                Some(choice) => (choice, true),
                None => (fallback, false),
            };
        let mut source = if heuristic_used {
            PlanSource::Heuristic
        } else {
            PlanSource::Fallback
        };
        let resolved = match self.table.choose(
            candidate,
            rows as usize,
            cols as usize,
            k as usize,
            subgroup,
        ) {
            Some(choice) => {
                source = PlanSource::Override;
                choice
            }
            None => candidate,
        };
        (resolved, source, heuristic_used)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_tuner() -> WasmTuner {
        WasmTuner {
            table: WasmTunerTable::new(),
        }
    }

    fn override_record() -> WasmTunerRecord {
        WasmTunerRecord {
            rows_min: Some(256),
            rows_max: Some(1024),
            cols_min: 0,
            cols_max: 16_384,
            k_min: 0,
            k_max: 512,
            subgroup: Some(true),
            algo_topk: None,
            ctile: None,
            wg: Some(128),
            kl: None,
            ch: None,
            mode_midk: None,
            mode_bottomk: None,
            tile_cols: Some(2048),
            radix: Some(4),
            segments: Some(2),
            use_2ce: None,
        }
    }

    #[test]
    fn fallback_plan_prefers_override() {
        let mut tuner = empty_tuner();
        tuner.table.push_sorted(override_record());
        let plan = tuner.plan_fft_with_fallback(512, 4096, 128, true);
        assert_eq!(plan.tile_cols(), 2048);
        assert_eq!(plan.radix(), 4);
        let resolved = tuner.plan_fft_resolution(512, 4096, 128, true);
        assert!(resolved.override_applied());
        assert_eq!(resolved.plan().tile_cols(), 2048);
        assert_eq!(resolved.plan().radix(), 4);
        assert!(matches!(resolved.source(), WasmFftPlanSource::Override));
        let json = resolved.to_json().expect("json");
        let roundtrip = ResolvedWasmFftPlan::from_json(&json).expect("from json");
        assert!(roundtrip.override_applied());
    }

    #[test]
    fn fallback_plan_handles_missing_override() {
        let tuner = empty_tuner();
        let resolved = tuner.plan_fft_resolution(512, 4096, 128, true);
        assert!(!resolved.override_applied());
        assert!(matches!(
            resolved.source(),
            WasmFftPlanSource::Heuristic | WasmFftPlanSource::Fallback
        ));
        assert!(resolved.plan().tile_cols() >= 1);
        let object = resolved.to_object().expect("object");
        let roundtrip = ResolvedWasmFftPlan::from_object(&object).expect("from object");
        assert_eq!(roundtrip.override_applied(), resolved.override_applied());
    }

    #[test]
    fn plan_fft_json_serialises_override() {
        let mut tuner = empty_tuner();
        tuner.table.push_sorted(override_record());
        let json = tuner
            .plan_fft_json(512, 4096, 128, true)
            .expect("json result")
            .expect("override json");
        let parsed: WasmFftPlanSerde = serde_json::from_str(&json).expect("parse json");
        assert_eq!(parsed.tile_cols, 2048);
        assert_eq!(parsed.radix, 4);
    }

    #[test]
    fn plan_fft_json_absent_without_override() {
        let tuner = empty_tuner();
        let json = tuner
            .plan_fft_json(512, 4096, 128, true)
            .expect("json result");
        assert!(json.is_none());
    }

    #[test]
    fn plan_fft_with_fallback_json_matches_object() {
        let tuner = empty_tuner();
        let json = tuner
            .plan_fft_with_fallback_json(512, 4096, 128, true)
            .expect("json");
        let object = tuner
            .plan_fft_with_fallback_object(512, 4096, 128, true)
            .expect("object");
        let parsed_json: WasmFftPlanSerde = serde_json::from_str(&json).expect("parse json");
        let parsed_object: WasmFftPlanSerde = object.into_serde().expect("parse object");
        assert_eq!(parsed_json.tile_cols, parsed_object.tile_cols);
        assert_eq!(parsed_json.radix, parsed_object.radix);
    }

    #[test]
    fn plan_fft_resolution_json_matches_report() {
        let mut tuner = empty_tuner();
        tuner.table.push_sorted(override_record());
        let json = tuner
            .plan_fft_resolution_json(512, 4096, 128, true)
            .expect("json");
        let report = tuner
            .plan_fft_resolution_object(512, 4096, 128, true)
            .expect("object");
        let parsed_json: ResolvedPlanSerde = serde_json::from_str(&json).expect("parse json");
        let parsed_report: ResolvedPlanSerde = report.into_serde().expect("parse object");
        assert_eq!(parsed_json.plan.tile_cols, parsed_report.plan.tile_cols);
        assert_eq!(parsed_json.override_applied, parsed_report.override_applied);
    }
}
