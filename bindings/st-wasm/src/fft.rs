use js_sys::Float32Array;
use serde::{Deserialize, Serialize};
use st_core::backend::spiralk_fft::SpiralKFftPlan;
use st_core::backend::wgpu_heuristics::{self, Choice};
use st_frac::fft::{self, Complex32};
use wasm_bindgen::prelude::*;

use crate::utils::{js_error, json_to_js_value, stringify_js_value};

#[wasm_bindgen]
pub struct WasmFftPlan {
    plan: SpiralKFftPlan,
}

impl WasmFftPlan {
    pub(crate) fn from_choice(choice: Choice, subgroup: bool) -> Self {
        Self {
            plan: SpiralKFftPlan::from_choice(&choice, subgroup),
        }
    }

    pub(crate) fn from_plan(plan: SpiralKFftPlan) -> Self {
        Self { plan }
    }
}

#[wasm_bindgen]
impl WasmFftPlan {
    #[wasm_bindgen(constructor)]
    pub fn new(radix: u32, tile_cols: u32, segments: u32, subgroup: bool) -> WasmFftPlan {
        let plan = SpiralKFftPlan {
            radix: radix.max(2).min(4),
            tile_cols: tile_cols.max(1),
            segments: segments.max(1),
            subgroup,
        };
        Self::from_plan(plan)
    }

    #[wasm_bindgen(getter)]
    pub fn radix(&self) -> u32 {
        self.plan.radix
    }

    #[wasm_bindgen(getter, js_name = tileCols)]
    pub fn tile_cols(&self) -> u32 {
        self.plan.tile_cols
    }

    #[wasm_bindgen(getter)]
    pub fn segments(&self) -> u32 {
        self.plan.segments
    }

    #[wasm_bindgen(getter)]
    pub fn subgroup(&self) -> bool {
        self.plan.subgroup
    }

    #[wasm_bindgen(js_name = workgroupSize)]
    pub fn workgroup_size(&self) -> u32 {
        self.plan.workgroup_size()
    }

    pub fn wgsl(&self) -> String {
        self.plan.emit_wgsl()
    }

    #[wasm_bindgen(js_name = spiralkHint)]
    pub fn spiralk_hint(&self) -> String {
        self.plan.emit_spiralk_hint()
    }

    /// Serialise the plan into a JSON string so it can be persisted or sent over the network.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.plan).map_err(js_error)
    }

    /// Convert the plan into a plain JavaScript object with the same fields as [`toJson`].
    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> Result<JsValue, JsValue> {
        JsValue::from_serde(&self.plan).map_err(|err| js_error(err))
    }

    /// Rebuild a plan from a JSON string produced by [`toJson`].
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<WasmFftPlan, JsValue> {
        let plan = serde_json::from_str::<SpiralKFftPlan>(json).map_err(js_error)?;
        Ok(WasmFftPlan::from_plan(plan))
    }

    /// Rebuild a plan from a plain JavaScript object with the same fields as [`toObject`].
    #[wasm_bindgen(js_name = fromObject)]
    pub fn from_object(value: &JsValue) -> Result<WasmFftPlan, JsValue> {
        let plan = value
            .into_serde::<SpiralKFftPlan>()
            .map_err(|err| js_error(err))?;
        Ok(WasmFftPlan::from_plan(plan))
    }
}

pub(crate) fn auto_plan_internal(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
) -> Option<WasmFftPlan> {
    let choice = wgpu_heuristics::choose_topk(rows, cols, k, subgroup)?;
    Some(WasmFftPlan::from_choice(choice, subgroup))
}

#[wasm_bindgen(js_name = "auto_plan_fft")]
pub fn auto_plan_fft(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<WasmFftPlan> {
    auto_plan_internal(rows, cols, k, subgroup)
}

#[wasm_bindgen(js_name = "auto_fft_wgsl")]
pub fn auto_fft_wgsl(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<String> {
    auto_plan_internal(rows, cols, k, subgroup).map(|plan| plan.wgsl())
}

#[wasm_bindgen(js_name = "auto_fft_spiralk")]
pub fn auto_fft_spiralk(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<String> {
    auto_plan_internal(rows, cols, k, subgroup).map(|plan| plan.spiralk_hint())
}

#[wasm_bindgen(js_name = "auto_fft_plan_json")]
pub fn auto_fft_plan_json(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
) -> Result<Option<String>, JsValue> {
    match auto_plan_internal(rows, cols, k, subgroup) {
        Some(plan) => Ok(Some(plan.to_json()?)),
        None => Ok(None),
    }
}

#[wasm_bindgen(js_name = "auto_fft_plan_object")]
pub fn auto_fft_plan_object(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
) -> Result<Option<JsValue>, JsValue> {
    match auto_plan_internal(rows, cols, k, subgroup) {
        Some(plan) => Ok(Some(plan.to_object()?)),
        None => Ok(None),
    }
}

#[wasm_bindgen(js_name = "fft_forward")]
pub fn fft_forward(buffer: &Float32Array) -> Result<Float32Array, JsValue> {
    fft_transform(buffer, false).map(|updated| Float32Array::from(updated.as_slice()))
}

#[wasm_bindgen(js_name = "fft_inverse")]
pub fn fft_inverse(buffer: &Float32Array) -> Result<Float32Array, JsValue> {
    fft_transform(buffer, true).map(|updated| Float32Array::from(updated.as_slice()))
}

#[wasm_bindgen(js_name = "fft_forward_in_place")]
pub fn fft_forward_in_place(buffer: &Float32Array) -> Result<(), JsValue> {
    fft_transform(buffer, false).and_then(|updated| {
        let view = Float32Array::from(updated.as_slice());
        buffer.set(&view, 0)
    })
}

#[wasm_bindgen(js_name = "fft_inverse_in_place")]
pub fn fft_inverse_in_place(buffer: &Float32Array) -> Result<(), JsValue> {
    fft_transform(buffer, true).and_then(|updated| {
        let view = Float32Array::from(updated.as_slice());
        buffer.set(&view, 0)
    })
}

fn fft_transform(buffer: &Float32Array, inverse: bool) -> Result<Vec<f32>, JsValue> {
    let mut spectrum = typed_array_to_complex(buffer)?;
    fft::fft_inplace(&mut spectrum, inverse).map_err(js_error)?;
    Ok(complex_to_interleaved(&spectrum))
}

fn typed_array_to_complex(buffer: &Float32Array) -> Result<Vec<Complex32>, JsValue> {
    let len = buffer.length() as usize;
    if len % 2 != 0 {
        return Err(js_error(
            "FFT buffer must contain interleaved real/imag parts",
        ));
    }
    let mut host = vec![0.0f32; len];
    buffer.copy_to(&mut host);
    Ok(host
        .chunks_exact(2)
        .map(|chunk| Complex32::new(chunk[0], chunk[1]))
        .collect())
}

fn complex_to_interleaved(data: &[Complex32]) -> Vec<f32> {
    let mut host = Vec::with_capacity(data.len() * 2);
    for value in data {
        host.push(value.re);
        host.push(value.im);
    }
    host
}

#[derive(Serialize, Deserialize)]
struct WasmFftPlanSerde {
    radix: u32,
    #[serde(rename = "tileCols")]
    tile_cols: u32,
    segments: u32,
    subgroup: bool,
}

impl From<&SpiralKFftPlan> for WasmFftPlanSerde {
    fn from(plan: &SpiralKFftPlan) -> Self {
        Self {
            radix: plan.radix,
            tile_cols: plan.tile_cols,
            segments: plan.segments,
            subgroup: plan.subgroup,
        }
    }
}

impl From<&WasmFftPlan> for WasmFftPlanSerde {
    fn from(plan: &WasmFftPlan) -> Self {
        Self::from(&plan.plan)
    }
}

impl From<WasmFftPlanSerde> for SpiralKFftPlan {
    fn from(value: WasmFftPlanSerde) -> Self {
        SpiralKFftPlan {
            radix: value.radix.max(2).min(4),
            tile_cols: value.tile_cols.max(1),
            segments: value.segments.max(1),
            subgroup: value.subgroup,
        }
    }
}

impl From<WasmFftPlanSerde> for WasmFftPlan {
    fn from(value: WasmFftPlanSerde) -> Self {
        WasmFftPlan::from_plan(value.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_plan_matches_wgsl_helpers() {
        let plan = auto_plan_internal(512, 4096, 128, true).expect("plan expected");
        assert!(plan.wgsl().contains("@compute"));
        assert!(plan.spiralk_hint().contains("tile_cols"));
    }

    #[test]
    fn fft_roundtrip_restores_signal() {
        let mut host = vec![0.0f32; 8];
        host[0] = 1.0;
        let mut complex = host
            .chunks_exact(2)
            .map(|chunk| Complex32::new(chunk[0], chunk[1]))
            .collect::<Vec<_>>();
        fft::fft_inplace(&mut complex, false).unwrap();
        fft::fft_inplace(&mut complex, true).unwrap();
        let restored = complex_to_interleaved(&complex);
        assert!(restored[0] - 1.0 < 1e-5);
    }

    #[test]
    fn wasm_fft_plan_json_roundtrip() {
        let plan = WasmFftPlan::new(4, 2048, 3, true);
        let json = plan.to_json().expect("json serialisation");
        let parsed = WasmFftPlan::from_json(&json).expect("json parse");
        assert_eq!(parsed.radix(), 4);
        assert_eq!(parsed.tile_cols(), 2048);
        assert_eq!(parsed.segments(), 3);
        assert!(parsed.subgroup());
    }

    #[test]
    fn auto_plan_json_matches_structured_plan() {
        let plan = auto_plan_internal(256, 8192, 128, true).expect("plan expected");
        let json = auto_fft_plan_json(256, 8192, 128, true)
            .expect("result")
            .expect("json plan");
        let parsed = WasmFftPlan::from_json(&json).expect("parse json");
        assert_eq!(plan.radix(), parsed.radix());
        assert_eq!(plan.tile_cols(), parsed.tile_cols());
        assert_eq!(plan.segments(), parsed.segments());
    }
}
