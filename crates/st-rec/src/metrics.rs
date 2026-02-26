// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::HashSet;

/// Per-user ranking metrics computed over the first K recommendations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RecEvalRow {
    pub hits: usize,
    pub relevant: usize,
    pub precision: f32,
    pub recall: f32,
    pub hit_rate: f32,
    pub ndcg: f32,
    pub average_precision: f32,
}

/// Macro-average summary computed across multiple users.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RecEvalSummary {
    pub users: usize,
    pub k: usize,
    pub mean_precision: f32,
    pub mean_recall: f32,
    pub mean_hit_rate: f32,
    pub mean_ndcg: f32,
    pub mean_average_precision: f32,
}

fn log2(value: f32) -> f32 {
    value.ln() / std::f32::consts::LN_2
}

fn dcg_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    if k == 0 || recommended.is_empty() || relevant.is_empty() {
        return 0.0;
    }
    let mut dcg = 0.0f32;
    for (rank, item) in recommended.iter().take(k).enumerate() {
        if relevant.contains(item) {
            let denom = log2((rank + 2) as f32).max(1.0);
            dcg += 1.0 / denom;
        }
    }
    dcg
}

fn idcg_at_k(relevant_len: usize, k: usize) -> f32 {
    if relevant_len == 0 || k == 0 {
        return 0.0;
    }
    let take = relevant_len.min(k);
    let mut idcg = 0.0f32;
    for rank in 0..take {
        let denom = log2((rank + 2) as f32).max(1.0);
        idcg += 1.0 / denom;
    }
    idcg
}

fn average_precision_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    let relevant_len = relevant.len();
    if k == 0 || recommended.is_empty() || relevant_len == 0 {
        return 0.0;
    }

    let mut hits = 0usize;
    let mut ap = 0.0f32;
    for (idx, item) in recommended.iter().take(k).enumerate() {
        if relevant.contains(item) {
            hits += 1;
            ap += hits as f32 / (idx + 1) as f32;
        }
    }

    if hits == 0 {
        0.0
    } else {
        ap / (relevant_len as f32)
    }
}

/// Computes common top-K ranking metrics for a single user's recommendation list.
pub fn evaluate_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> RecEvalRow {
    let relevant_len = relevant.len();
    if k == 0 {
        return RecEvalRow {
            relevant: relevant_len,
            ..Default::default()
        };
    }

    let mut hits = 0usize;
    for item in recommended.iter().take(k) {
        if relevant.contains(item) {
            hits += 1;
        }
    }

    let denom_k = k as f32;
    let precision = if denom_k > 0.0 {
        hits as f32 / denom_k
    } else {
        0.0
    };
    let recall = if relevant_len > 0 {
        hits as f32 / relevant_len as f32
    } else {
        0.0
    };
    let hit_rate = if hits > 0 { 1.0 } else { 0.0 };
    let dcg = dcg_at_k(recommended, relevant, k);
    let idcg = idcg_at_k(relevant_len, k);
    let ndcg = if idcg > 0.0 {
        (dcg / idcg).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let average_precision = average_precision_at_k(recommended, relevant, k);

    RecEvalRow {
        hits,
        relevant: relevant_len,
        precision,
        recall,
        hit_rate,
        ndcg,
        average_precision,
    }
}

/// Macro-averages per-user evaluation rows.
pub fn summarize_at_k(rows: &[RecEvalRow], k: usize) -> RecEvalSummary {
    if rows.is_empty() {
        return RecEvalSummary::default();
    }
    let users = rows.len();
    let mut precision = 0.0f32;
    let mut recall = 0.0f32;
    let mut hit_rate = 0.0f32;
    let mut ndcg = 0.0f32;
    let mut ap = 0.0f32;
    for row in rows {
        precision += row.precision;
        recall += row.recall;
        hit_rate += row.hit_rate;
        ndcg += row.ndcg;
        ap += row.average_precision;
    }

    let denom = users as f32;
    RecEvalSummary {
        users,
        k,
        mean_precision: precision / denom,
        mean_recall: recall / denom,
        mean_hit_rate: hit_rate / denom,
        mean_ndcg: ndcg / denom,
        mean_average_precision: ap / denom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_match_simple_hit() {
        let relevant: HashSet<usize> = [2usize, 3].into_iter().collect();
        let recommended = vec![2usize, 1, 3, 4];
        let row = evaluate_at_k(&recommended, &relevant, 2);
        assert_eq!(row.hits, 1);
        assert_eq!(row.relevant, 2);
        assert!((row.precision - 0.5).abs() < 1e-6);
        assert!((row.recall - 0.5).abs() < 1e-6);
        assert_eq!(row.hit_rate, 1.0);
        assert!(row.ndcg > 0.0);
        assert!(row.average_precision > 0.0);
    }

    #[test]
    fn empty_relevant_returns_zeros() {
        let relevant: HashSet<usize> = HashSet::new();
        let recommended = vec![0usize, 1, 2];
        let row = evaluate_at_k(&recommended, &relevant, 3);
        assert_eq!(row.hits, 0);
        assert_eq!(row.relevant, 0);
        assert_eq!(row.precision, 0.0);
        assert_eq!(row.recall, 0.0);
        assert_eq!(row.hit_rate, 0.0);
        assert_eq!(row.ndcg, 0.0);
        assert_eq!(row.average_precision, 0.0);
    }

    #[test]
    fn summary_macro_averages() {
        let relevant_a: HashSet<usize> = [1usize].into_iter().collect();
        let relevant_b: HashSet<usize> = [4usize].into_iter().collect();
        let row_a = evaluate_at_k(&[1usize, 2, 3], &relevant_a, 3);
        let row_b = evaluate_at_k(&[0usize, 1, 2], &relevant_b, 3);
        let summary = summarize_at_k(&[row_a, row_b], 3);
        assert_eq!(summary.users, 2);
        assert!((summary.mean_hit_rate - 0.5).abs() < 1e-6);
    }
}
