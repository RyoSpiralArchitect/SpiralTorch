// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Validated SpiralK FFT plans and their executable WebGPU dispatch contract.
//!
//! One dispatch cannot synchronize independent WebGPU workgroups. The plan
//! therefore emits a bit-reversal pass followed by ordered, ping-pong
//! butterfly passes. Rust owns the stage sequence; Python and WASM only expose
//! the resulting contract and WGSL module.

use crate::backend::wgpu_heuristics::Choice;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

pub const SPIRALK_FFT_CONTRACT_VERSION: &str = "spiraltorch.fft_plan.v1";
pub const SPIRALK_FFT_CONTRACT_KIND: &str = "spiraltorch.fft_plan";
pub const SPIRALK_FFT_SEMANTIC_OWNER: &str = "st-core::backend::spiralk_fft";
pub const SPIRALK_FFT_MAX_SEGMENTS: u32 = 4;
/// Conservative portable ceiling: even the 32-lane path stays below WebGPU's
/// guaranteed 65,535 workgroups-per-dimension limit during bit reversal.
pub const SPIRALK_FFT_MAX_TILE_COLS: u32 = 1 << 20;

/// Convert a backend tile preference into the nearest covering radix tile.
/// Planner-generated hints use this helper; explicit user plans remain strict.
pub fn canonical_fft_tile_hint(hint: u32) -> u32 {
    hint.clamp(2, 8192).next_power_of_two()
}

/// Project a planner-generated radix hint onto the supported radix set.
/// A midpoint keeps the caller's already-valid fallback instead of inventing
/// a third radix. Explicit plans must use [`SpiralKFftPlan::try_new`] instead.
pub const fn canonical_fft_radix_hint(hint: u32, fallback: u32) -> u32 {
    match hint {
        0..=2 => 2,
        3 if fallback == 2 => 2,
        3 => 4,
        _ => 4,
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum SpiralKFftPlanError {
    #[error("FFT radix={radix} is unsupported; expected 2 or 4")]
    UnsupportedRadix { radix: u32 },
    #[error("FFT tile_cols={tile_cols} is too small; expected at least 2")]
    TileTooSmall { tile_cols: u32 },
    #[error("FFT tile_cols={tile_cols} must be a power of two")]
    TileNotPowerOfTwo { tile_cols: u32 },
    #[error(
        "FFT tile_cols={tile_cols} exceeds the portable WebGPU limit {max_tile_cols}",
        max_tile_cols = SPIRALK_FFT_MAX_TILE_COLS
    )]
    TileTooLarge { tile_cols: u32 },
    #[error(
        "FFT segments={segments} is invalid; expected 1..={max_segments}",
        max_segments = SPIRALK_FFT_MAX_SEGMENTS
    )]
    InvalidSegments { segments: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpiralKFftPlan {
    radix: u32,
    tile_cols: u32,
    segments: u32,
    subgroup: bool,
}

#[derive(Deserialize)]
struct SpiralKFftPlanRequest {
    radix: u32,
    tile_cols: u32,
    segments: u32,
    subgroup: bool,
}

impl<'de> Deserialize<'de> for SpiralKFftPlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let request = SpiralKFftPlanRequest::deserialize(deserializer)?;
        Self::try_new(
            request.radix,
            request.tile_cols,
            request.segments,
            request.subgroup,
        )
        .map_err(D::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpiralKFftBuffer {
    Primary,
    Scratch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpiralKFftStageKind {
    BitReverse,
    Butterfly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SpiralKFftDispatch {
    pub stage_index: u32,
    pub kind: SpiralKFftStageKind,
    pub entry_point: &'static str,
    pub radix: u32,
    pub span: u32,
    pub workgroups: [u32; 3],
    pub source: SpiralKFftBuffer,
    pub destination: SpiralKFftBuffer,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SpiralKFftPlanSnapshot {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub radix: u32,
    pub tile_cols: u32,
    pub segments: u32,
    pub subgroup: bool,
    pub workgroup_size: u32,
    pub dispatches: Vec<SpiralKFftDispatch>,
    pub output_buffer: SpiralKFftBuffer,
}

impl SpiralKFftPlan {
    pub fn try_new(
        radix: u32,
        tile_cols: u32,
        segments: u32,
        subgroup: bool,
    ) -> Result<Self, SpiralKFftPlanError> {
        if !matches!(radix, 2 | 4) {
            return Err(SpiralKFftPlanError::UnsupportedRadix { radix });
        }
        if tile_cols < 2 {
            return Err(SpiralKFftPlanError::TileTooSmall { tile_cols });
        }
        if !tile_cols.is_power_of_two() {
            return Err(SpiralKFftPlanError::TileNotPowerOfTwo { tile_cols });
        }
        if tile_cols > SPIRALK_FFT_MAX_TILE_COLS {
            return Err(SpiralKFftPlanError::TileTooLarge { tile_cols });
        }
        if !(1..=SPIRALK_FFT_MAX_SEGMENTS).contains(&segments) {
            return Err(SpiralKFftPlanError::InvalidSegments { segments });
        }
        Ok(Self {
            radix,
            tile_cols,
            segments,
            subgroup,
        })
    }

    pub fn from_choice(choice: &Choice, subgroup: bool) -> Result<Self, SpiralKFftPlanError> {
        Self::try_new(choice.radix, choice.tile_cols, choice.segments, subgroup)
    }

    pub const fn radix(&self) -> u32 {
        self.radix
    }

    pub const fn tile_cols(&self) -> u32 {
        self.tile_cols
    }

    pub const fn segments(&self) -> u32 {
        self.segments
    }

    pub const fn subgroup(&self) -> bool {
        self.subgroup
    }

    pub const fn workgroup_size(&self) -> u32 {
        if self.subgroup {
            32
        } else {
            64
        }
    }

    pub fn dispatches(&self) -> Vec<SpiralKFftDispatch> {
        let workgroup = self.workgroup_size();
        let mut dispatches = Vec::with_capacity(self.tile_cols.ilog2() as usize + 1);
        let mut source = SpiralKFftBuffer::Primary;
        let mut destination = SpiralKFftBuffer::Scratch;
        dispatches.push(SpiralKFftDispatch {
            stage_index: 0,
            kind: SpiralKFftStageKind::BitReverse,
            entry_point: "fft_bit_reverse",
            radix: 1,
            span: self.tile_cols,
            workgroups: [self.tile_cols.div_ceil(workgroup), self.segments, 1],
            source,
            destination,
        });
        std::mem::swap(&mut source, &mut destination);

        let log2_tile = self.tile_cols.ilog2();
        let mut stage_index = 1;
        if self.radix == 2 {
            let mut span = 2;
            while span <= self.tile_cols {
                push_butterfly_dispatch(
                    &mut dispatches,
                    stage_index,
                    2,
                    span,
                    self.tile_cols,
                    self.segments,
                    workgroup,
                    source,
                    destination,
                );
                stage_index += 1;
                std::mem::swap(&mut source, &mut destination);
                span = span.saturating_mul(2);
            }
        } else {
            let mut span = if log2_tile % 2 == 1 {
                push_butterfly_dispatch(
                    &mut dispatches,
                    stage_index,
                    2,
                    2,
                    self.tile_cols,
                    self.segments,
                    workgroup,
                    source,
                    destination,
                );
                stage_index += 1;
                std::mem::swap(&mut source, &mut destination);
                8
            } else {
                4
            };
            while span <= self.tile_cols {
                push_butterfly_dispatch(
                    &mut dispatches,
                    stage_index,
                    4,
                    span,
                    self.tile_cols,
                    self.segments,
                    workgroup,
                    source,
                    destination,
                );
                stage_index += 1;
                std::mem::swap(&mut source, &mut destination);
                span = span.saturating_mul(4);
            }
        }
        dispatches
    }

    pub fn output_buffer(&self) -> SpiralKFftBuffer {
        self.dispatches()
            .last()
            .map(|dispatch| dispatch.destination)
            .expect("validated FFT plans always contain at least two dispatches")
    }

    pub fn snapshot(&self) -> SpiralKFftPlanSnapshot {
        SpiralKFftPlanSnapshot {
            kind: SPIRALK_FFT_CONTRACT_KIND,
            contract_version: SPIRALK_FFT_CONTRACT_VERSION,
            semantic_owner: SPIRALK_FFT_SEMANTIC_OWNER,
            semantic_backend: "rust",
            radix: self.radix,
            tile_cols: self.tile_cols,
            segments: self.segments,
            subgroup: self.subgroup,
            workgroup_size: self.workgroup_size(),
            dispatches: self.dispatches(),
            output_buffer: self.output_buffer(),
        }
    }

    pub fn dispatch_manifest_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.snapshot())
    }

    /// Emit one WGSL module containing the entry points referenced by
    /// [`Self::dispatches`]. The host binds the selected source/destination
    /// buffers and updates `FftParams` before each ordered dispatch.
    pub fn emit_wgsl(&self) -> String {
        let workgroup = self.workgroup_size();
        let log2_tile = self.tile_cols.ilog2();
        format!(
            "// {contract} owned by {owner}\n\
             // Forward complex FFT: bit reversal + ordered ping-pong stages.\n\
             struct Complex {{ re: f32, im: f32, }};\n\
             struct FftParams {{ span: u32, radix: u32, tile_cols: u32, segments: u32, }};\n\
             @group(0) @binding(0) var<storage, read> fft_src: array<Complex>;\n\
             @group(0) @binding(1) var<storage, read_write> fft_dst: array<Complex>;\n\
             @group(0) @binding(2) var<uniform> fft_params: FftParams;\n\
             const FFT_TILE_COLS: u32 = {tile}u;\n\
             const FFT_SEGMENTS: u32 = {segments}u;\n\
             const FFT_LOG2_TILE: u32 = {log2_tile}u;\n\
             fn complex_add(a: Complex, b: Complex) -> Complex {{\n\
                 return Complex(a.re + b.re, a.im + b.im);\n\
             }}\n\
             fn complex_sub(a: Complex, b: Complex) -> Complex {{\n\
                 return Complex(a.re - b.re, a.im - b.im);\n\
             }}\n\
             fn complex_mul(a: Complex, b: Complex) -> Complex {{\n\
                 return Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);\n\
             }}\n\
             fn multiply_minus_i(value: Complex) -> Complex {{\n\
                 return Complex(value.im, -value.re);\n\
             }}\n\
             fn twiddle(exponent: u32, span: u32) -> Complex {{\n\
                 let angle = -6.283185307179586 * f32(exponent) / f32(span);\n\
                 return Complex(cos(angle), sin(angle));\n\
             }}\n\
             fn reverse_tile_bits(index: u32) -> u32 {{\n\
                 var value = index;\n\
                 var reversed = 0u;\n\
                 for (var bit = 0u; bit < FFT_LOG2_TILE; bit = bit + 1u) {{\n\
                     reversed = (reversed << 1u) | (value & 1u);\n\
                     value = value >> 1u;\n\
                 }}\n\
                 return reversed;\n\
             }}\n\
             fn params_match_plan() -> bool {{\n\
                 return fft_params.tile_cols == FFT_TILE_COLS && fft_params.segments == FFT_SEGMENTS;\n\
             }}\n\
             @compute @workgroup_size({workgroup})\n\
             fn fft_bit_reverse(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
                 if (!params_match_plan() || gid.x >= FFT_TILE_COLS || gid.y >= FFT_SEGMENTS) {{\n\
                     return;\n\
                 }}\n\
                 let base = gid.y * FFT_TILE_COLS;\n\
                 fft_dst[base + gid.x] = fft_src[base + reverse_tile_bits(gid.x)];\n\
             }}\n\
             @compute @workgroup_size({workgroup})\n\
             fn fft_stage(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
                 if (!params_match_plan() || gid.y >= FFT_SEGMENTS) {{\n\
                     return;\n\
                 }}\n\
                 let radix = fft_params.radix;\n\
                 let span = fft_params.span;\n\
                 if ((radix != 2u && radix != 4u) || span > FFT_TILE_COLS || span < radix) {{\n\
                     return;\n\
                 }}\n\
                 let butterfly_count = FFT_TILE_COLS / radix;\n\
                 if (gid.x >= butterfly_count) {{\n\
                     return;\n\
                 }}\n\
                 let segment_base = gid.y * FFT_TILE_COLS;\n\
                 if (radix == 2u) {{\n\
                     let half = span / 2u;\n\
                     let block = gid.x / half;\n\
                     let j = gid.x % half;\n\
                     let first = segment_base + block * span + j;\n\
                     let second = first + half;\n\
                     let even = fft_src[first];\n\
                     let odd = complex_mul(fft_src[second], twiddle(j, span));\n\
                     fft_dst[first] = complex_add(even, odd);\n\
                     fft_dst[second] = complex_sub(even, odd);\n\
                     return;\n\
                 }}\n\
                 let quarter = span / 4u;\n\
                 let block = gid.x / quarter;\n\
                 let j = gid.x % quarter;\n\
                 let first = segment_base + block * span + j;\n\
                 let i0 = first;\n\
                 let i1 = first + quarter;\n\
                 let i2 = first + 2u * quarter;\n\
                 let i3 = first + 3u * quarter;\n\
                 let w2 = twiddle(2u * j, span);\n\
                 let p0 = complex_add(fft_src[i0], complex_mul(fft_src[i1], w2));\n\
                 let p1 = complex_sub(fft_src[i0], complex_mul(fft_src[i1], w2));\n\
                 let p2 = complex_add(fft_src[i2], complex_mul(fft_src[i3], w2));\n\
                 let p3 = complex_sub(fft_src[i2], complex_mul(fft_src[i3], w2));\n\
                 let w1 = twiddle(j, span);\n\
                 let cross_even = complex_mul(p2, w1);\n\
                 let cross_odd = complex_mul(multiply_minus_i(p3), w1);\n\
                 fft_dst[i0] = complex_add(p0, cross_even);\n\
                 fft_dst[i2] = complex_sub(p0, cross_even);\n\
                 fft_dst[i1] = complex_add(p1, cross_odd);\n\
                 fft_dst[i3] = complex_sub(p1, cross_odd);\n\
             }}\n",
            contract = SPIRALK_FFT_CONTRACT_VERSION,
            owner = SPIRALK_FFT_SEMANTIC_OWNER,
            tile = self.tile_cols,
            segments = self.segments,
            log2_tile = log2_tile,
            workgroup = workgroup,
        )
    }

    pub fn emit_spiralk_hint(&self) -> String {
        format!(
            "soft (tile_cols, {tile}, 0.85, c >= {tile});\nsoft (radix, {radix}, 0.90, true);\nsoft (segments, {segments}, 0.75, true);",
            tile = self.tile_cols,
            radix = self.radix,
            segments = self.segments,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn push_butterfly_dispatch(
    dispatches: &mut Vec<SpiralKFftDispatch>,
    stage_index: u32,
    radix: u32,
    span: u32,
    tile_cols: u32,
    segments: u32,
    workgroup: u32,
    source: SpiralKFftBuffer,
    destination: SpiralKFftBuffer,
) {
    dispatches.push(SpiralKFftDispatch {
        stage_index,
        kind: SpiralKFftStageKind::Butterfly,
        entry_point: "fft_stage",
        radix,
        span,
        workgroups: [(tile_cols / radix).div_ceil(workgroup), segments, 1],
        source,
        destination,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;

    fn choice(radix: u32, tile_cols: u32) -> Choice {
        Choice {
            use_2ce: true,
            wg: 256,
            kl: 16,
            ch: 8192,
            algo_topk: 2,
            ctile: 1024,
            mode_midk: 2,
            mode_bottomk: 1,
            tile_cols,
            radix,
            segments: 3,
        }
    }

    #[test]
    fn plan_validation_rejects_semantically_invalid_shapes() {
        assert!(matches!(
            SpiralKFftPlan::try_new(3, 2048, 1, false),
            Err(SpiralKFftPlanError::UnsupportedRadix { radix: 3 })
        ));
        assert!(matches!(
            SpiralKFftPlan::try_new(2, 1536, 1, false),
            Err(SpiralKFftPlanError::TileNotPowerOfTwo { tile_cols: 1536 })
        ));
        assert!(matches!(
            SpiralKFftPlan::try_new(2, SPIRALK_FFT_MAX_TILE_COLS * 2, 1, true),
            Err(SpiralKFftPlanError::TileTooLarge { .. })
        ));
        assert!(serde_json::from_str::<SpiralKFftPlan>(
            r#"{"radix":4,"tile_cols":2048,"segments":9,"subgroup":true}"#
        )
        .is_err());
    }

    #[test]
    fn planner_radix_hints_never_create_a_third_runtime_radix() {
        assert_eq!(canonical_fft_radix_hint(3, 2), 2);
        assert_eq!(canonical_fft_radix_hint(3, 4), 4);
        assert_eq!(canonical_fft_radix_hint(99, 2), 4);
    }

    #[test]
    fn radix_four_dispatches_fuse_pairs_of_radix_two_stages() {
        let plan = SpiralKFftPlan::try_new(4, 2048, 3, true).unwrap();
        let dispatches = plan.dispatches();
        assert_eq!(dispatches[0].kind, SpiralKFftStageKind::BitReverse);
        assert_eq!(dispatches[1].radix, 2);
        assert_eq!(dispatches[1].span, 2);
        assert_eq!(dispatches[2].radix, 4);
        assert_eq!(dispatches[2].span, 8);
        assert_eq!(dispatches.last().unwrap().span, 2048);
        for pair in dispatches.windows(2) {
            assert_eq!(pair[0].destination, pair[1].source);
            assert_ne!(pair[0].source, pair[0].destination);
        }
    }

    #[test]
    fn emitted_wgsl_is_valid_and_matches_dispatch_entry_points() {
        let plan = SpiralKFftPlan::from_choice(&choice(4, 2048), true).unwrap();
        let source = plan.emit_wgsl();
        let module = naga::front::wgsl::parse_str(&source).expect("generated WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("generated WGSL validates");
        assert!(source.contains("fn fft_bit_reverse"));
        assert!(source.contains("fn fft_stage"));
        assert!(source.contains("multiply_minus_i"));
        assert_eq!(plan.snapshot().semantic_backend, "rust");
    }

    fn execute_dispatch_contract(plan: &SpiralKFftPlan, input: &[Complex32]) -> Vec<Complex32> {
        let n = plan.tile_cols() as usize;
        let mut primary = input.to_vec();
        let mut scratch = vec![Complex32::new(0.0, 0.0); n];
        for dispatch in plan.dispatches() {
            let (source, destination) = match (dispatch.source, dispatch.destination) {
                (SpiralKFftBuffer::Primary, SpiralKFftBuffer::Scratch) => {
                    (&primary[..], &mut scratch[..])
                }
                (SpiralKFftBuffer::Scratch, SpiralKFftBuffer::Primary) => {
                    (&scratch[..], &mut primary[..])
                }
                _ => unreachable!("dispatches always ping-pong"),
            };
            if dispatch.kind == SpiralKFftStageKind::BitReverse {
                let bits = plan.tile_cols().ilog2();
                for (index, out) in destination.iter_mut().enumerate() {
                    let reversed = (index as u32).reverse_bits() >> (u32::BITS - bits);
                    *out = source[reversed as usize];
                }
                continue;
            }
            let span = dispatch.span as usize;
            if dispatch.radix == 2 {
                let half = span / 2;
                for block in (0..n).step_by(span) {
                    for j in 0..half {
                        let angle = -std::f32::consts::TAU * j as f32 / span as f32;
                        let twiddle = Complex32::from_polar(1.0, angle);
                        let even = source[block + j];
                        let odd = source[block + j + half] * twiddle;
                        destination[block + j] = even + odd;
                        destination[block + j + half] = even - odd;
                    }
                }
            } else {
                let quarter = span / 4;
                for block in (0..n).step_by(span) {
                    for j in 0..quarter {
                        let w2 = Complex32::from_polar(
                            1.0,
                            -std::f32::consts::TAU * (2 * j) as f32 / span as f32,
                        );
                        let p0 = source[block + j] + source[block + j + quarter] * w2;
                        let p1 = source[block + j] - source[block + j + quarter] * w2;
                        let p2 =
                            source[block + j + 2 * quarter] + source[block + j + 3 * quarter] * w2;
                        let p3 =
                            source[block + j + 2 * quarter] - source[block + j + 3 * quarter] * w2;
                        let w1 = Complex32::from_polar(
                            1.0,
                            -std::f32::consts::TAU * j as f32 / span as f32,
                        );
                        let cross_even = p2 * w1;
                        let cross_odd = p3 * Complex32::new(0.0, -1.0) * w1;
                        destination[block + j] = p0 + cross_even;
                        destination[block + j + 2 * quarter] = p0 - cross_even;
                        destination[block + j + quarter] = p1 + cross_odd;
                        destination[block + j + 3 * quarter] = p1 - cross_odd;
                    }
                }
            }
        }
        match plan.output_buffer() {
            SpiralKFftBuffer::Primary => primary,
            SpiralKFftBuffer::Scratch => scratch,
        }
    }

    #[test]
    fn radix_two_and_four_contracts_match_rustfft() {
        for radix in [2, 4] {
            for size in [8, 16, 32] {
                let plan = SpiralKFftPlan::try_new(radix, size, 1, false).unwrap();
                let input = (0..size)
                    .map(|index| {
                        Complex32::new(
                            ((index * 7 + 3) % 11) as f32 / 11.0,
                            ((index * 5 + 1) % 13) as f32 / 13.0,
                        )
                    })
                    .collect::<Vec<_>>();
                let mut expected = input.clone();
                let mut planner = rustfft::FftPlanner::<f32>::new();
                planner
                    .plan_fft_forward(size as usize)
                    .process(&mut expected);
                let actual = execute_dispatch_contract(&plan, &input);
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).norm() < 1.0e-4);
                }
            }
        }
    }
}
