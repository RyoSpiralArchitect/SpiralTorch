// crates/st-amg/src/sr_learn.rs
//! A/Bの勝敗から Wilson 固定化と学習ストア更新を行うフック例
//! 既存の sr.rs から呼び出してください。

#[allow(unused_variables)]
pub fn maybe_append_soft(rule_expr: &str) {
    // 既存の heur.kdsl 追記処理に差し替えてください。
    // ここではダミー実装です。
    eprintln!("[sr] append soft => {}", rule_expr);
}

/// Wilsonスコア下限（正規近似, z: 1.96 相当で95%）
pub fn wilson_lower_bound(wins: u32, trials: u32, z: f32) -> f32 {
    if trials == 0 { return 0.0; }
    let n = trials as f32;
    let p = (wins as f32) / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = p + z2/(2.0*n);
    let margin = (p*(1.0-p)/n + z2/(4.0*n*n)).sqrt() * z;
    (center - margin) / denom
}

/// A/Bの勝敗が入ったときの更新フック
pub fn on_ab_outcome(winner_rules: &[&str], loser_rules: &[&str], wins: u32, trials: u32, z: f32, lb_thresh: f32) {
    let lb = wilson_lower_bound(wins, trials, z);
    if lb >= lb_thresh {
        // 代表的な勝者ルールを固定化（例）
        if let Some(&r0) = winner_rules.first() {
            maybe_append_soft(&format!("soft({},1.0)", r0));
        }
    }

    // 学習ストア（feature有効時のみ）
    #[cfg(feature = "learn_store")]
    {
        use st_logic::learn::{load, save, update_bandit};
        let mut sw = load();
        update_bandit(&mut sw, winner_rules, loser_rules);
        let _ = save(&sw);
    }
}
