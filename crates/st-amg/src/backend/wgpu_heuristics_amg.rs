// crates/st-amg/src/backend/wgpu_heuristics_amg.rs
//! AMG Heuristics overlay: SpiralK/生成表 -> SoftRuleブレンド -> base_scoreに加算
//! 環境変数:
//!   SPIRAL_SOFT_MODE = {Sum|Normalize|Softmax|Prob}
//!   SPIRAL_BEAM_K = <usize>
//!   SPIRAL_SOFT_BANDIT_BLEND = <0..1>  (bandit重みの混合率)
use st_logic::{SoftMode, apply_softmode, SolveCfg};

#[cfg(feature = "learn_store")]
use st_logic::learn::{load, weight_from_bandit};

// 既存のChoice/initial_choice/base_score/soft_rules_from_spiralkは手元実装を利用してください。
#[derive(Clone, Debug)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub tile_cols: u32,
    pub jacobi_passes: u32,
    pub score: f32,
}

fn initial_choice(_rows: usize, _cols: usize, _nnz: usize, _subgroup: bool) -> Choice {
    Choice { use_2ce: true, wg: 256, tile_cols: 8192, jacobi_passes: 1, score: 0.0 }
}

// ダミー：スコアのベース（現場の実装で置換）
fn base_score_amg(c: &Choice) -> f32 {
    let mut s = 0.0f32;
    s += if c.use_2ce { 0.05 } else { 0.0 };
    s += (c.wg as f32).ln() * 0.1;
    s += (c.tile_cols as f32).ln() * 0.01;
    s -= (c.jacobi_passes as f32 - 1.0).abs() * 0.02;
    s
}

// ダミー：SpiralK/生成表からのSoftRule（現場の実装で置換）
fn soft_rules_from_spiralk(_rows: usize, _cols: usize, _nnz: usize, _sg: bool) -> Vec<st_logic::SoftRule> {
    vec![
        st_logic::SoftRule{ name:"wg=256", weight:0.6, score: 2.0 },
        st_logic::SoftRule{ name:"2ce",    weight:0.5, score: 1.2 },
        st_logic::SoftRule{ name:"tile=8k",weight:0.4, score: 0.8 },
    ]
}

fn neighbors_amg(c: &Choice) -> Vec<Choice> {
    let mut v = Vec::new();
    for &wg in &[128u32, 256, 512] {
        let mut n = c.clone(); n.wg = wg; v.push(n);
    }
    for &tc in &[4096u32, 8192, 16384] {
        let mut n = c.clone(); n.tile_cols = tc; v.push(n);
    }
    for &jp in &[1u32, 2] {
        let mut n = c.clone(); n.jacobi_passes = jp; v.push(n);
    }
    {
        let mut n = c.clone(); n.use_2ce = !c.use_2ce; v.push(n);
    }
    v
}

pub fn choose(rows: usize, cols: usize, nnz: usize, subgroup: bool) -> Choice {
    let soft_mode = match std::env::var("SPIRAL_SOFT_MODE").as_deref() {
        Ok("Normalize") => SoftMode::Normalize,
        Ok("Softmax")   => SoftMode::Softmax,
        Ok("Prob")      => SoftMode::Prob,
        _               => SoftMode::Sum,
    };
    let beam_k = std::env::var("SPIRAL_BEAM_K").ok().and_then(|s| s.parse::<usize>().ok());
    let cfg = SolveCfg { noise: 0.02, seed: 0x5u64, beam: beam_k, soft_mode: soft_mode };

    let mut soft = soft_rules_from_spiralk(rows, cols, nnz, subgroup);

    // bandit重みのブレンド（feature有効時のみ）
    #[cfg(feature = "learn_store")]
    {
        let sw = load();
        let lambda = std::env::var("SPIRAL_SOFT_BANDIT_BLEND")
            .ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.35);
        for r in &mut soft {
            let bw = weight_from_bandit(&sw, r.name);
            r.weight = (1.0 - lambda) * r.weight + lambda * bw;
        }
    }

    // 単発スコア（ビーム探索を使うなら beam_select を適用）
    let mut choice = initial_choice(rows, cols, nnz, subgroup);
    let score = base_score_amg(&choice) + apply_softmode(&soft, cfg.soft_mode);
    choice.score = score;

    if let Some(k) = cfg.beam {
        let max_depth = 3usize;
        choice = st_logic::beam_select(choice, |c| neighbors_amg(c),
            |c| base_score_amg(c) + apply_softmode(&soft, cfg.soft_mode),
            k, max_depth);
    }
    choice
}
