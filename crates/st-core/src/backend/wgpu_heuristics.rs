use super::consensus;
use super::device_caps::DeviceCaps;
use super::kdsl_bridge;
use super::spiralk_fft::SpiralKFftPlan;
use crate::backend::wgpu_heuristics_generated as gen;
#[cfg(feature = "logic")]
use st_logic::{solve_soft, Ctx as LCtx, SoftRule, SolveCfg as LCfg};

#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub kl: u32,
    pub ch: u32,
    pub algo_topk: u8,    // 0=auto, 1=heap, 2=bitonic, 3=kway
    pub ctile: u32,       // 0=auto
    pub mode_midk: u8,    // 0=auto, 1=1CE, 2=2CE
    pub mode_bottomk: u8, // 0=auto, 1=1CE, 2=2CE
    pub tile_cols: u32,   // column tiles for ND FFT/fractional kernels
    pub radix: u32,       // preferred FFT radix
    pub segments: u32,    // ND segment count for GPU kernels
}

fn fallback(_rows: u32, cols: u32, k: u32, subgroup: bool) -> Choice {
    let max_wg = if subgroup { 256 } else { 128 };
    let caps = DeviceCaps::wgpu(32, subgroup, max_wg);
    let use_2ce = caps.prefers_two_stage(cols, k);
    let wg = caps.recommended_workgroup();
    let kl = caps.recommended_kl(k);
    let ch = caps.recommended_channel_stride(cols);
    let ctile = caps.recommended_compaction_tile(cols);
    Choice {
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
        segments: if cols > 131_072 { 4 } else if cols > 32_768 { 2 } else { 1 },
    }
}

pub fn choose_topk(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    choose_kind(rows, cols, k, subgroup, "topk")
}
pub fn choose_midk(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    choose_kind(rows, cols, k, subgroup, "midk")
}
pub fn choose_bottomk(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    choose_kind(rows, cols, k, subgroup, "bottomk")
}

#[derive(Default, Clone, Copy)]
pub struct DslOverrides {
    pub algo_topk: u8,
    pub ctile: u32,
    pub mode_midk: u8,
    pub mode_bottomk: u8,
    pub tile_cols: u32,
    pub radix: u32,
    pub segments: u32,
}
fn overlay(c: &mut Choice, o: &DslOverrides) {
    if o.algo_topk != 0 {
        c.algo_topk = o.algo_topk;
    }
    if o.ctile != 0 {
        c.ctile = o.ctile;
    }
    if o.mode_midk != 0 {
        c.mode_midk = o.mode_midk;
    }
    if o.mode_bottomk != 0 {
        c.mode_bottomk = o.mode_bottomk;
    }
    if o.tile_cols != 0 {
        c.tile_cols = o.tile_cols;
    }
    if o.radix != 0 {
        c.radix = o.radix;
    }
    if o.segments != 0 {
        c.segments = o.segments;
    }
}

pub fn choose_kind(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    kind: &'static str,
) -> Option<Choice> {
    let (hard_dsl, soft_dsl, ov) =
        kdsl_bridge::parse_env_dsl_plus_kind(rows, cols, k, subgroup, kind);
    let soft_kv = consensus::kv_consensus_soft_rules(rows, cols, k, subgroup, kind);
    #[cfg(not(feature = "logic"))]
    let _ = (&soft_dsl, &soft_kv);

    #[cfg(feature = "logic")]
    {
        let mut all = soft_dsl.clone();
        all.extend(soft_kv);
        let use_soft = std::env::var("SPIRAL_HEUR_SOFT")
            .ok()
            .map(|v| v == "1")
            .unwrap_or(true);
        if use_soft {
            let ctx = LCtx {
                rows,
                cols,
                k,
                sg: subgroup,
            };
            let (c, score) = solve_soft(
                ctx,
                LCfg {
                    noise: 0.02,
                    seed: 0x5p1ral,
                },
                &all,
            );
            if score > 0.1 {
                let mut out = Choice {
                    use_2ce: c.use_2ce,
                    wg: c.wg,
                    kl: c.kl,
                    ch: c.ch,
                    ..fallback(rows, cols, k, subgroup)
                };
                overlay(&mut out, &ov);
                return Some(out);
            }
        }
    }
    if let Some(mut c) = hard_dsl {
        overlay(&mut c, &ov);
        return Some(c);
    }
    if let Some(mut c) = kdsl_bridge::choose_from_kv(rows, cols, k, subgroup) {
        overlay(&mut c, &ov);
        return Some(c);
    }
    if let Some(mut c) = gen::choose(rows as usize, cols as usize, k as usize, subgroup) {
        overlay(&mut c, &ov);
        return Some(c);
    }
    Some(fallback(rows, cols, k, subgroup))
}

/// Construct an FFT plan from the same heuristics used for TopK and emit the
/// auto-generated WGSL shader.  Returns `None` if the heuristic pipeline could
/// not find a suitable `Choice`.
pub fn auto_fft_wgsl(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<String> {
    let plan = auto_fft_plan(rows, cols, k, subgroup)?;
    Some(plan.emit_wgsl())
}

/// Produce the SpiralK hint snippet associated with the automatically emitted
/// WGSL kernel.
pub fn auto_fft_spiralk(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<String> {
    let plan = auto_fft_plan(rows, cols, k, subgroup)?;
    Some(plan.emit_spiralk_hint())
}

/// Internal helper that assembles the [`SpiralKFftPlan`] from the heuristic
/// `Choice`.
fn auto_fft_plan(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<SpiralKFftPlan> {
    let choice = choose_topk(rows, cols, k, subgroup)?;
    Some(SpiralKFftPlan::from_choice(&choice, subgroup))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_fft_kernel_and_hint() {
        let wgsl = auto_fft_wgsl(512, 4096, 128, true).expect("kernel expected");
        assert!(wgsl.contains("@workgroup_size"));
        let hint = auto_fft_spiralk(512, 4096, 128, true).unwrap();
        assert!(hint.contains("tile_cols"));
    }
}

include!("wgpu_heuristics_generated.rs");
