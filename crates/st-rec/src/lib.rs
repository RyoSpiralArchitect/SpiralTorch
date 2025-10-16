// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::fmt;

use st_tensor::pure::{topos::OpenCartesianTopos, Tensor, TensorError};

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
}
