// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! End-to-end demo: train a recommender, rank a top-K slate, and let a bandit policy
//! learn directly from ranking metrics (e.g. NDCG@K).

use std::collections::{HashMap, HashSet};

use st_rec::{evaluate_at_k, summarize_at_k, RatingTriple, RecBanditController, SpiralRecommender};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let users = 3usize;
    let items = 7usize;
    let factors = 4usize;
    let k = 3usize;

    let mut recommender = SpiralRecommender::new(users, items, factors, 0.05, 0.002, -1.0)?;

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

    // Hold-out relevance sets (e.g. clicks/purchases in a test window).
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
        if !report.rmse.is_finite() {
            break;
        }
    }

    let mut baseline_rows = Vec::with_capacity(users);
    for user in 0..users {
        let exclusions = exclude.get(&user).map(|items| items.as_slice());
        let recs = recommender.recommend_top_k(user, k, exclusions)?;
        let ranked: Vec<usize> = recs.into_iter().map(|rec| rec.item).collect();
        baseline_rows.push(evaluate_at_k(&ranked, &relevant[user], k));
    }
    let baseline = summarize_at_k(&baseline_rows, k);
    println!(
        "baseline: precision@k={:.3} recall@k={:.3} hit@k={:.3} ndcg@k={:.3} map@k={:.3}",
        baseline.mean_precision,
        baseline.mean_recall,
        baseline.mean_hit_rate,
        baseline.mean_ndcg,
        baseline.mean_average_precision,
    );

    let mut controller = RecBanditController::new(&recommender, k, 0.02, 0.9)?;

    for epoch in 0..10 {
        controller.reset_episode();
        let mut rows = Vec::with_capacity(users);
        for user in 0..users {
            let exclusions = exclude.get(&user).map(|items| items.as_slice());
            let decision = controller.select_top_k(&recommender, user, k, exclusions)?;
            let ranked = decision.ranked_items_by_probability();
            let row = evaluate_at_k(&ranked, &relevant[user], k);
            controller.observe_reward(decision, row.ndcg)?;
            rows.push(row);
        }
        let report = controller.finish_episode()?;
        let summary = summarize_at_k(&rows, k);
        println!(
            "epoch={epoch:02} reward={:.3} policy.ndcg@k={:.3} policy.hit@k={:.3}",
            report.total_reward, summary.mean_ndcg, summary.mean_hit_rate
        );
    }

    Ok(())
}
