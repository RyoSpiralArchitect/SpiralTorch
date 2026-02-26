// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! End-to-end demo: train a tiny recommender, score held-out items, and report top-K metrics.

use std::collections::{HashMap, HashSet};

use st_rec::{evaluate_at_k, summarize_at_k, RatingTriple, RecResult, SpiralRecommender};

fn main() -> RecResult<()> {
    let users = 3usize;
    let items = 7usize;
    let factors = 4usize;
    let k = 3usize;

    let mut recommender = SpiralRecommender::new(users, items, factors, 0.05, 0.002, -1.0)?;

    // Training interactions (explicit ratings).
    let train = vec![
        RatingTriple::new(0, 0, 5.0),
        RatingTriple::new(0, 1, 4.0),
        RatingTriple::new(0, 2, 3.5),
        RatingTriple::new(1, 2, 4.5),
        RatingTriple::new(1, 3, 4.0),
        RatingTriple::new(1, 4, 2.0),
        RatingTriple::new(2, 4, 4.8),
        RatingTriple::new(2, 5, 4.2),
    ];

    // Simple hold-out relevance sets (e.g. clicks/purchases in a test window).
    let relevant: Vec<HashSet<usize>> = vec![
        [3usize].into_iter().collect(),
        [5usize].into_iter().collect(),
        [6usize].into_iter().collect(),
    ];

    let mut exclude: HashMap<usize, Vec<usize>> = HashMap::new();
    for triple in &train {
        exclude.entry(triple.user).or_default().push(triple.item);
    }

    for _ in 0..12 {
        let report = recommender.train_epoch(&train)?;
        if report.rmse.is_nan() {
            break;
        }
    }

    let mut rows = Vec::with_capacity(users);
    for user in 0..users {
        let exclusions = exclude.get(&user).map(|items| items.as_slice());
        let recs = recommender.recommend_top_k(user, k, exclusions)?;
        let ranked: Vec<usize> = recs.into_iter().map(|rec| rec.item).collect();
        rows.push(evaluate_at_k(&ranked, &relevant[user], k));
    }

    let summary = summarize_at_k(&rows, k);
    println!(
        "users={} k={} precision@k={:.3} recall@k={:.3} hit@k={:.3} ndcg@k={:.3} map@k={:.3}",
        summary.users,
        summary.k,
        summary.mean_precision,
        summary.mean_recall,
        summary.mean_hit_rate,
        summary.mean_ndcg,
        summary.mean_average_precision,
    );
    Ok(())
}
