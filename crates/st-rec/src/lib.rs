// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use st_kdsl::query::QueryPlan;
use st_tensor::{topos::OpenCartesianTopos, Tensor, TensorError};

/// Errors surfaced by the recommender harness.
#[derive(Debug)]
pub enum SpiralRecError {
    Tensor(TensorError),
    /// Rating triples must live within the configured user/item bounds.
    OutOfBoundsRating {
        user: usize,
        item: usize,
    },
    /// Training requires at least one rating per epoch.
    EmptyBatch,
}

impl fmt::Display for SpiralRecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpiralRecError::Tensor(err) => write!(f, "{err}"),
            SpiralRecError::OutOfBoundsRating { user, item } => write!(
                f,
                "rating ({user}, {item}) falls outside the configured recommender bounds",
            ),
            SpiralRecError::EmptyBatch => write!(
                f,
                "at least one rating triple is required to run a training epoch",
            ),
        }
    }
}

impl std::error::Error for SpiralRecError {}

impl From<TensorError> for SpiralRecError {
    fn from(value: TensorError) -> Self {
        SpiralRecError::Tensor(value)
    }
}

/// Convenient result alias for recommendation helpers.
pub type RecResult<T> = Result<T, SpiralRecError>;

/// Lightweight rating tuple consumed by the trainer.
#[derive(Clone, Debug)]
pub struct RatingTriple {
    pub user: usize,
    pub item: usize,
    pub rating: f32,
}

impl RatingTriple {
    pub fn new(user: usize, item: usize, rating: f32) -> Self {
        Self { user, item, rating }
    }
}

/// Epoch report emitted after a recommender update.
#[derive(Clone, Debug)]
pub struct RecEpochReport {
    pub rmse: f32,
    pub samples: usize,
    pub regularization_penalty: f32,
}

/// Ranked item recommendation for a specific user.
#[derive(Clone, Debug, PartialEq)]
pub struct Recommendation {
    pub item: usize,
    pub score: f32,
}

/// Lightweight user-neighbourhood recommender derived from explicit ratings.
pub struct NeighborhoodModel {
    users: usize,
    items: usize,
    rating_matrix: Vec<f32>,
    smoothing: f32,
}

impl NeighborhoodModel {
    pub fn fit(
        users: usize,
        items: usize,
        ratings: &[RatingTriple],
        smoothing: f32,
    ) -> RecResult<Self> {
        if users == 0 || items == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: users,
                cols: items,
            }
            .into());
        }
        if smoothing < 0.0 {
            return Err(TensorError::NonPositiveTension { tension: smoothing }.into());
        }

        let mut matrix = vec![0.0f32; users * items];
        let mut counts = vec![0usize; users * items];
        for rating in ratings {
            if rating.user >= users || rating.item >= items {
                return Err(SpiralRecError::OutOfBoundsRating {
                    user: rating.user,
                    item: rating.item,
                });
            }
            let offset = rating.user * items + rating.item;
            matrix[offset] += rating.rating;
            counts[offset] += 1;
        }

        for idx in 0..matrix.len() {
            if counts[idx] > 0 {
                matrix[idx] /= counts[idx] as f32;
            }
        }

        Ok(Self {
            users,
            items,
            rating_matrix: matrix,
            smoothing: smoothing.max(1e-5),
        })
    }

    fn rating(&self, user: usize, item: usize) -> f32 {
        self.rating_matrix[user * self.items + item]
    }

    pub fn recommend(&self, user: usize, k: usize) -> RecResult<Vec<Recommendation>> {
        if user >= self.users {
            return Err(SpiralRecError::OutOfBoundsRating { user, item: 0 });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut similarities = vec![0.0f32; self.users];
        let target = &self.rating_matrix[user * self.items..(user + 1) * self.items];
        let mut target_norm = 0.0f32;
        for &value in target {
            target_norm += value * value;
        }
        target_norm = target_norm.sqrt() + self.smoothing;

        for other in 0..self.users {
            if other == user {
                continue;
            }
            let other_row = &self.rating_matrix[other * self.items..(other + 1) * self.items];
            let mut dot = 0.0f32;
            let mut other_norm = 0.0f32;
            for idx in 0..self.items {
                let lhs = target[idx];
                let rhs = other_row[idx];
                dot += lhs * rhs;
                other_norm += rhs * rhs;
            }
            similarities[other] =
                dot / ((other_norm.sqrt() + self.smoothing) * target_norm).max(self.smoothing);
        }

        let mut scores = vec![0.0f32; self.items];
        for other in 0..self.users {
            if other == user {
                continue;
            }
            let weight = similarities[other];
            if weight.abs() <= f32::EPSILON {
                continue;
            }
            for item in 0..self.items {
                scores[item] += weight * self.rating(other, item);
            }
        }

        let mut recs: Vec<Recommendation> = scores
            .into_iter()
            .enumerate()
            .map(|(item, score)| Recommendation { item, score })
            .collect();
        recs.sort_by(|a, b| b.score.total_cmp(&a.score));
        recs.truncate(k);
        Ok(recs)
    }
}

#[derive(Clone, Debug)]
pub enum KnowledgeEdge {
    UserItem {
        user: usize,
        item: usize,
        weight: f32,
    },
    ItemItem {
        source: usize,
        target: usize,
        weight: f32,
    },
}

pub struct KnowledgeGraphRecommender {
    users: usize,
    items: usize,
    damping: f32,
    user_to_item: Vec<Vec<(usize, f32)>>,
    item_to_item: Vec<Vec<(usize, f32)>>,
}

impl KnowledgeGraphRecommender {
    pub fn new(users: usize, items: usize, damping: f32) -> RecResult<Self> {
        if users == 0 || items == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: users,
                cols: items,
            }
            .into());
        }
        if !(0.0..=1.0).contains(&damping) {
            return Err(TensorError::NonPositiveTension { tension: damping }.into());
        }
        Ok(Self {
            users,
            items,
            damping,
            user_to_item: vec![Vec::new(); users],
            item_to_item: vec![Vec::new(); items],
        })
    }

    pub fn from_edges(
        users: usize,
        items: usize,
        damping: f32,
        edges: &[KnowledgeEdge],
    ) -> RecResult<Self> {
        let mut graph = Self::new(users, items, damping)?;
        for edge in edges {
            graph.add_edge(edge.clone())?;
        }
        Ok(graph)
    }

    pub fn add_edge(&mut self, edge: KnowledgeEdge) -> RecResult<()> {
        match edge {
            KnowledgeEdge::UserItem { user, item, weight } => {
                if user >= self.users || item >= self.items {
                    return Err(SpiralRecError::OutOfBoundsRating { user, item });
                }
                self.user_to_item[user].push((item, weight));
            }
            KnowledgeEdge::ItemItem {
                source,
                target,
                weight,
            } => {
                if source >= self.items || target >= self.items {
                    return Err(SpiralRecError::OutOfBoundsRating {
                        user: source.min(target),
                        item: source.max(target),
                    });
                }
                self.item_to_item[source].push((target, weight));
            }
        }
        Ok(())
    }

    fn propagate(&self, user: usize, hops: usize) -> Vec<f32> {
        let mut scores = vec![0.0f32; self.items];
        for &(item, weight) in &self.user_to_item[user] {
            scores[item] += weight;
        }
        let mut current = scores.clone();
        for _ in 0..hops {
            let mut next = current.clone();
            for item in 0..self.items {
                let score = current[item];
                if score.abs() <= f32::EPSILON {
                    continue;
                }
                for &(target, weight) in &self.item_to_item[item] {
                    next[target] += score * weight * self.damping;
                }
            }
            current = next;
        }
        current
    }

    pub fn recommend(&self, user: usize, k: usize, hops: usize) -> RecResult<Vec<Recommendation>> {
        if user >= self.users {
            return Err(SpiralRecError::OutOfBoundsRating { user, item: 0 });
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let scores = self.propagate(user, hops);
        let mut recs: Vec<Recommendation> = scores
            .into_iter()
            .enumerate()
            .map(|(item, score)| Recommendation { item, score })
            .collect();
        recs.sort_by(|a, b| b.score.total_cmp(&a.score));
        recs.truncate(k);
        Ok(recs)
    }

    pub fn item_degree(&self, item: usize) -> RecResult<usize> {
        if item >= self.items {
            return Err(SpiralRecError::OutOfBoundsRating { user: 0, item });
        }
        Ok(self.item_to_item[item].len())
    }
}

/// Matrix factorisation harness backed by SpiralTorch tensors and open topos guards.
pub struct SpiralRecommender {
    users: usize,
    items: usize,
    factors: usize,
    learning_rate: f32,
    regularization: f32,
    user_factors: Tensor,
    item_factors: Tensor,
    topos: OpenCartesianTopos,
}

impl SpiralRecommender {
    /// Builds a recommender with the supplied topology.
    pub fn new(
        users: usize,
        items: usize,
        factors: usize,
        learning_rate: f32,
        regularization: f32,
        curvature: f32,
    ) -> RecResult<Self> {
        if users == 0 || items == 0 || factors == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: users.max(1) * items.max(1),
                cols: factors,
            }
            .into());
        }
        if learning_rate <= 0.0 {
            return Err(TensorError::NonPositiveLearningRate {
                rate: learning_rate,
            }
            .into());
        }
        if regularization < 0.0 {
            return Err(TensorError::NonPositiveTension {
                tension: regularization,
            }
            .into());
        }

        let user_factors = Tensor::from_fn(users, factors, |u, f| {
            let phase = (u as f32 + 1.0) * (f as f32 + 1.0);
            (phase.sin() * 0.05).clamp(-0.25, 0.25)
        })?;
        let item_factors = Tensor::from_fn(items, factors, |i, f| {
            let phase = (i as f32 + 1.0) * (f as f32 + 1.0);
            (phase.cos() * 0.05).clamp(-0.25, 0.25)
        })?;
        let max_depth = users.saturating_add(items).max(8);
        let max_volume = users
            .saturating_mul(factors)
            .saturating_add(items.saturating_mul(factors))
            .max(8);
        let topos = OpenCartesianTopos::new(curvature, 1e-5, 64.0, max_depth, max_volume)?;

        Ok(Self {
            users,
            items,
            factors,
            learning_rate,
            regularization,
            user_factors,
            item_factors,
            topos,
        })
    }

    fn guard_indices(&self, user: usize, item: usize) -> RecResult<()> {
        if user >= self.users || item >= self.items {
            Err(SpiralRecError::OutOfBoundsRating { user, item })
        } else {
            Ok(())
        }
    }

    /// Predicts a score for the provided user/item pair.
    pub fn predict(&self, user: usize, item: usize) -> RecResult<f32> {
        self.guard_indices(user, item)?;
        let user_slice = &self.user_factors.data()[user * self.factors..(user + 1) * self.factors];
        let item_slice = &self.item_factors.data()[item * self.factors..(item + 1) * self.factors];
        Ok(user_slice
            .iter()
            .zip(item_slice.iter())
            .map(|(u, i)| u * i)
            .sum())
    }

    /// Extracts the user embedding as a row-major tensor.
    pub fn user_embedding(&self, user: usize) -> RecResult<Tensor> {
        self.guard_indices(user, 0)?;
        let slice = &self.user_factors.data()[user * self.factors..(user + 1) * self.factors];
        Tensor::from_vec(1, self.factors, slice.to_vec()).map_err(Into::into)
    }

    /// Extracts the item embedding as a row-major tensor.
    pub fn item_embedding(&self, item: usize) -> RecResult<Tensor> {
        self.guard_indices(0, item)?;
        let slice = &self.item_factors.data()[item * self.factors..(item + 1) * self.factors];
        Tensor::from_vec(1, self.factors, slice.to_vec()).map_err(Into::into)
    }

    /// Applies a single matrix-factorisation epoch over the provided ratings.
    pub fn train_epoch(&mut self, ratings: &[RatingTriple]) -> RecResult<RecEpochReport> {
        if ratings.is_empty() {
            return Err(SpiralRecError::EmptyBatch);
        }

        let mut squared_error = 0.0f32;
        let mut reg_penalty = 0.0f32;
        let mut user_buf = self.user_factors.data().to_vec();
        let mut item_buf = self.item_factors.data().to_vec();

        for rating in ratings {
            self.guard_indices(rating.user, rating.item)?;
            let user_offset = rating.user * self.factors;
            let item_offset = rating.item * self.factors;
            let (user_slice, item_slice) = (
                &user_buf[user_offset..user_offset + self.factors],
                &item_buf[item_offset..item_offset + self.factors],
            );
            let prediction: f32 = user_slice
                .iter()
                .zip(item_slice.iter())
                .map(|(u, i)| u * i)
                .sum();
            let diff = prediction - rating.rating;
            squared_error += diff * diff;

            for factor in 0..self.factors {
                let u_idx = user_offset + factor;
                let i_idx = item_offset + factor;
                let u_val = user_buf[u_idx];
                let i_val = item_buf[i_idx];
                let grad_u = diff * i_val + self.regularization * u_val;
                let grad_i = diff * u_val + self.regularization * i_val;
                user_buf[u_idx] = self.topos.saturate(u_val - self.learning_rate * grad_u);
                item_buf[i_idx] = self.topos.saturate(i_val - self.learning_rate * grad_i);
                reg_penalty += self.regularization * (u_val * u_val + i_val * i_val);
            }
        }

        self.topos.guard_slice("user_factors", &user_buf)?;
        self.topos.guard_slice("item_factors", &item_buf)?;
        self.user_factors = Tensor::from_vec(self.users, self.factors, user_buf)?;
        self.item_factors = Tensor::from_vec(self.items, self.factors, item_buf)?;

        let rmse = (squared_error / ratings.len() as f32).sqrt();

        Ok(RecEpochReport {
            rmse,
            samples: ratings.len(),
            regularization_penalty: reg_penalty / ratings.len() as f32,
        })
    }

    /// Returns the number of tracked users.
    pub fn users(&self) -> usize {
        self.users
    }

    /// Returns the number of tracked items.
    pub fn items(&self) -> usize {
        self.items
    }

    /// Returns the latent factor dimensionality.
    pub fn factors(&self) -> usize {
        self.factors
    }

    /// Produces the top-K ranked items for a user while respecting optional exclusions.
    pub fn recommend_top_k(
        &self,
        user: usize,
        k: usize,
        exclude: Option<&[usize]>,
    ) -> RecResult<Vec<Recommendation>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        self.guard_indices(user, 0)?;
        let user_slice = &self.user_factors.data()[user * self.factors..(user + 1) * self.factors];

        let skip = exclude.map(|items| {
            let mut set = HashSet::with_capacity(items.len());
            set.extend(items.iter().copied());
            set
        });

        let mut recommendations = Vec::with_capacity(self.items);
        for item in 0..self.items {
            if skip.as_ref().is_some_and(|set| set.contains(&item)) {
                continue;
            }

            let item_slice =
                &self.item_factors.data()[item * self.factors..(item + 1) * self.factors];
            let score = user_slice
                .iter()
                .zip(item_slice.iter())
                .map(|(u, i)| u * i)
                .sum();
            recommendations.push(Recommendation { item, score });
        }

        recommendations.sort_by(|a, b| b.score.total_cmp(&a.score));
        if recommendations.len() > k {
            recommendations.truncate(k);
        }

        Ok(recommendations)
    }

    /// Applies a KDsl query plan to the recommendation surface so callers can
    /// filter, order, and project the ranked items without leaving Rust.
    pub fn recommend_with_query(
        &self,
        user: usize,
        plan: &QueryPlan,
        exclude: Option<&[usize]>,
    ) -> RecResult<Vec<BTreeMap<String, f64>>> {
        let mut rows = Vec::new();
        let ranked = self.recommend_top_k(user, self.items, exclude)?;

        if ranked.is_empty() {
            return Ok(Vec::new());
        }

        let max_score = ranked
            .iter()
            .map(|rec| rec.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let normaliser: f32 = ranked.iter().map(|rec| (rec.score - max_score).exp()).sum();

        if normaliser.is_finite() && normaliser > f32::EPSILON {
            for (rank, rec) in ranked.iter().enumerate() {
                let mut row = BTreeMap::new();
                row.insert("user".to_string(), user as f64);
                row.insert("item".to_string(), rec.item as f64);
                row.insert("score".to_string(), rec.score as f64);
                let softmax = (rec.score - max_score).exp() / normaliser;
                row.insert("score_softmax".to_string(), softmax as f64);
                row.insert("rank".to_string(), (rank + 1) as f64);
                rows.push(row);
            }
        } else {
            for (rank, rec) in ranked.iter().enumerate() {
                let mut row = BTreeMap::new();
                row.insert("user".to_string(), user as f64);
                row.insert("item".to_string(), rec.item as f64);
                row.insert("score".to_string(), rec.score as f64);
                row.insert("score_softmax".to_string(), 0.0);
                row.insert("rank".to_string(), (rank + 1) as f64);
                rows.push(row);
            }
        }

        Ok(plan.execute(&rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_kdsl::compile_query;

    fn build_basic_recommender() -> SpiralRecommender {
        let mut rec = SpiralRecommender::new(2, 5, 3, 0.05, 0.01, -1.0).unwrap();
        let ratings = vec![
            RatingTriple::new(0, 0, 5.0),
            RatingTriple::new(0, 1, 3.0),
            RatingTriple::new(0, 2, 1.0),
            RatingTriple::new(1, 2, 4.0),
            RatingTriple::new(1, 3, 2.0),
        ];

        for _ in 0..8 {
            rec.train_epoch(&ratings).unwrap();
        }
        rec
    }

    #[test]
    fn recommend_top_k_orders_scores_descending() {
        let rec = build_basic_recommender();
        let results = rec.recommend_top_k(0, 3, None).unwrap();
        assert!(results.len() <= 3);
        for window in results.windows(2) {
            assert!(window[0].score >= window[1].score - f32::EPSILON);
        }
    }

    #[test]
    fn recommend_top_k_respects_exclusions_and_limits() {
        let rec = build_basic_recommender();
        let exclude = vec![0, 3];
        let results = rec.recommend_top_k(0, 10, Some(&exclude)).unwrap();
        assert!(results.iter().all(|rec| !exclude.contains(&rec.item)));
        assert!(results.len() <= rec.items() - exclude.len());
        assert!(rec.recommend_top_k(0, 0, None).unwrap().is_empty());
    }

    #[test]
    fn neighborhood_model_surfaces_peers() {
        let ratings = vec![
            RatingTriple::new(0, 0, 5.0),
            RatingTriple::new(0, 1, 1.0),
            RatingTriple::new(1, 0, 4.5),
            RatingTriple::new(1, 2, 4.0),
            RatingTriple::new(2, 2, 5.0),
        ];
        let model = NeighborhoodModel::fit(3, 3, &ratings, 1e-2).unwrap();
        let recs = model.recommend(0, 2).unwrap();
        assert!(!recs.is_empty());
        assert!(recs.iter().any(|rec| rec.item == 2));
    }

    #[test]
    fn knowledge_graph_propagates_scores() {
        let edges = vec![
            KnowledgeEdge::UserItem {
                user: 0,
                item: 0,
                weight: 1.0,
            },
            KnowledgeEdge::ItemItem {
                source: 0,
                target: 1,
                weight: 0.5,
            },
        ];
        let graph = KnowledgeGraphRecommender::from_edges(1, 3, 0.9, &edges).unwrap();
        let recs = graph.recommend(0, 2, 2).unwrap();
        assert!(recs.iter().any(|rec| rec.item == 1));
        assert!(graph.item_degree(0).unwrap() > 0);
    }

    #[test]
    fn recommend_with_query_filters_rows() {
        let rec = build_basic_recommender();
        let plan = compile_query(
            "SELECT item,score,score_softmax WHERE score_softmax > 0.2 ORDER BY score DESC LIMIT 2",
        )
        .unwrap();
        let exclude = vec![3];
        let rows = rec.recommend_with_query(0, &plan, Some(&exclude)).unwrap();
        assert!(rows.len() <= 2);
        for row in &rows {
            assert!(row.contains_key("item"));
            assert!(row.contains_key("score"));
            assert!(row.get("item").unwrap() != &3.0);
            assert!(row.get("score_softmax").unwrap() > &0.2);
        }
    }
}
