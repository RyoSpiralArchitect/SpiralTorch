// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};

/// Minimal synchronous collective layer used by distributed trainers in tests.
pub(crate) mod st_distributed {
    use super::*;

    static GROUPS: Lazy<Mutex<HashMap<String, Arc<GroupState>>>> =
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
    }

    /// Handle that represents a rendezvous session.
    #[derive(Debug)]
    pub struct RendezvousSession {
        group: Arc<GroupState>,
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

    /// Connects a worker to a rendezvous group.
    pub fn rendezvous(group: String, rank: usize, world_size: usize) -> Arc<RendezvousSession> {
        let shared_group = {
            let mut guard = GROUPS.lock().unwrap();
            guard
                .entry(group)
                .or_insert_with(|| Arc::new(GroupState::new()))
                .clone()
        };

        {
            let mut state = shared_group.inner.lock().unwrap();
            if state.expected == 0 {
                state.expected = world_size;
            }
            assert_eq!(state.expected, world_size, "mismatched world size");
        }

        Arc::new(RendezvousSession {
            group: shared_group,
            rank,
            world_size,
        })
    }

    /// Performs an all-reduce sum across the rendezvous group.
    pub fn all_reduce(session: &Arc<RendezvousSession>, buffer: &mut [f32]) {
        let group = session.group();
        let mut state = group.inner.lock().unwrap();

        if state.expected == 0 {
            state.expected = session.world_size;
        }
        assert_eq!(state.expected, session.world_size, "mismatched world size");

        if state.arrived == 0 {
            state.generation = state.generation.wrapping_add(1);
            state.buffer.clear();
            state.buffer.resize(buffer.len(), 0.0);
            state.ready_generation = 0;
        } else {
            assert_eq!(
                state.buffer.len(),
                buffer.len(),
                "inconsistent tensor shape"
            );
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
        }

        buffer.copy_from_slice(&state.result);
    }
}
