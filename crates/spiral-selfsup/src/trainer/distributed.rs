// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex, Weak};
use thiserror::Error;

/// Minimal synchronous collective layer used by distributed trainers in tests.
pub(crate) mod st_distributed {
    use super::*;

    static GROUPS: Lazy<Mutex<HashMap<String, Weak<GroupState>>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));

    #[derive(Debug)]
    struct GroupState {
        inner: Mutex<GroupInner>,
        condvar: Condvar,
    }

    impl GroupState {
        fn new() -> Self {
            Self {
                inner: Mutex::new(GroupInner::default()),
                condvar: Condvar::new(),
            }
        }
    }

    #[derive(Debug, Default)]
    struct GroupInner {
        expected: usize,
        arrived: usize,
        generation: u64,
        buffer: Vec<f32>,
        result: Vec<f32>,
        ready_generation: u64,
        connected: HashSet<usize>,
        error: Option<DistributedError>,
    }

    /// Handle that represents a rendezvous session.
    #[derive(Debug)]
    pub struct RendezvousSession {
        group: Arc<GroupState>,
        group_id: String,
        rank: usize,
        world_size: usize,
    }

    impl RendezvousSession {
        pub fn rank(&self) -> usize {
            self.rank
        }

        pub fn world_size(&self) -> usize {
            self.world_size
        }

        fn group(&self) -> &Arc<GroupState> {
            &self.group
        }
    }

    impl Drop for RendezvousSession {
        fn drop(&mut self) {
            {
                let mut state = self.group.inner.lock().unwrap();
                state.connected.remove(&self.rank);
                if state.connected.is_empty() {
                    state.expected = 0;
                    state.arrived = 0;
                    state.buffer.clear();
                    state.result.clear();
                    state.ready_generation = 0;
                    state.error = None;
                }
            }

            let mut groups = GROUPS.lock().unwrap();
            let remove = groups
                .get(&self.group_id)
                .map(|weak| weak.upgrade().is_none())
                .unwrap_or(false);
            if remove {
                groups.remove(&self.group_id);
            }
        }
    }

    /// Errors produced by the in-memory rendezvous implementation.
    #[derive(Debug, Error, PartialEq, Eq, Clone)]
    pub enum DistributedError {
        #[error("world size must be positive, got {0}")]
        EmptyWorldSize(usize),
        #[error("rank {rank} is out of bounds for world size {world_size}")]
        RankOutOfBounds { rank: usize, world_size: usize },
        #[error("rendezvous group expects world size {expected}, got {got}")]
        WorldSizeMismatch { expected: usize, got: usize },
        #[error("rank {rank} already joined rendezvous group")]
        DuplicateRank { rank: usize },
        #[error("rank {rank} has not joined the rendezvous group")]
        UnknownRank { rank: usize },
        #[error("buffer length mismatch: expected {expected}, got {got}")]
        BufferLengthMismatch { expected: usize, got: usize },
    }

    /// Connects a worker to a rendezvous group.
    pub fn rendezvous(
        group: String,
        rank: usize,
        world_size: usize,
    ) -> Result<Arc<RendezvousSession>, DistributedError> {
        if world_size == 0 {
            return Err(DistributedError::EmptyWorldSize(world_size));
        }
        if rank >= world_size {
            return Err(DistributedError::RankOutOfBounds { rank, world_size });
        }

        let shared_group = {
            let mut guard = GROUPS.lock().unwrap();
            let entry = guard.entry(group.clone()).or_insert_with(Weak::new);
            if let Some(existing) = entry.upgrade() {
                existing
            } else {
                let created = Arc::new(GroupState::new());
                *entry = Arc::downgrade(&created);
                created
            }
        };

        {
            let mut state = shared_group.inner.lock().unwrap();
            if state.expected == 0 {
                state.expected = world_size;
            } else if state.expected != world_size {
                return Err(DistributedError::WorldSizeMismatch {
                    expected: state.expected,
                    got: world_size,
                });
            }
            if !state.connected.insert(rank) {
                return Err(DistributedError::DuplicateRank { rank });
            }
        }

        Ok(Arc::new(RendezvousSession {
            group: shared_group,
            group_id: group,
            rank,
            world_size,
        }))
    }

    /// Performs an all-reduce sum across the rendezvous group.
    pub fn all_reduce(
        session: &Arc<RendezvousSession>,
        buffer: &mut [f32],
    ) -> Result<(), DistributedError> {
        let group = session.group();
        let mut state = group.inner.lock().unwrap();

        if state.expected == 0 {
            state.expected = session.world_size;
        }
        if state.expected != session.world_size {
            return Err(DistributedError::WorldSizeMismatch {
                expected: state.expected,
                got: session.world_size,
            });
        }

        if !state.connected.contains(&session.rank) {
            return Err(DistributedError::UnknownRank {
                rank: session.rank(),
            });
        }

        if state.arrived == 0 {
            state.generation = state.generation.wrapping_add(1);
            state.buffer.clear();
            state.buffer.resize(buffer.len(), 0.0);
            state.ready_generation = 0;
            state.error = None;
        } else {
            if state.buffer.len() != buffer.len() {
                let error = DistributedError::BufferLengthMismatch {
                    expected: state.buffer.len(),
                    got: buffer.len(),
                };
                state.arrived = 0;
                state.ready_generation = state.generation;
                state.error = Some(error.clone());
                group.condvar.notify_all();
                return Err(error);
            }
        }

        for (dst, value) in state.buffer.iter_mut().zip(buffer.iter()) {
            *dst += *value;
        }

        state.arrived += 1;
        let current_generation = state.generation;

        if state.arrived == state.expected {
            state.result = state.buffer.clone();
            state.arrived = 0;
            state.ready_generation = current_generation;
            group.condvar.notify_all();
        } else {
            while state.ready_generation != current_generation {
                state = group.condvar.wait(state).unwrap();
            }
            if let Some(error) = state.error.clone() {
                return Err(error);
            }
        }

        if let Some(error) = state.error.clone() {
            return Err(error);
        }

        buffer.copy_from_slice(&state.result);
        Ok(())
    }
}
