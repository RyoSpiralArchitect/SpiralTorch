// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, WaitTimeoutResult, Weak};
use std::time::{Duration, Instant};
use thiserror::Error;

/// In-process synchronous collective layer used by distributed trainers.
pub(crate) mod st_distributed {
    use super::*;

    static GROUPS: Lazy<Mutex<HashMap<String, Weak<GroupState>>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));
    pub(crate) const DEFAULT_COLLECTIVE_TIMEOUT: Duration = Duration::from_secs(30);

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
        generation: u64,
        buffer: Vec<f32>,
        arrived: HashSet<usize>,
        connected: HashSet<usize>,
        completed: Option<CompletedRound>,
    }

    #[derive(Debug)]
    struct CompletedRound {
        generation: u64,
        outcome: Result<Vec<f32>, DistributedError>,
        pending: HashSet<usize>,
    }

    fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        match mutex.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                mutex.clear_poison();
                poisoned.into_inner()
            }
        }
    }

    fn wait_timeout_recover<'a, T>(
        condvar: &Condvar,
        mutex: &'a Mutex<T>,
        guard: MutexGuard<'a, T>,
        timeout: Duration,
    ) -> (MutexGuard<'a, T>, WaitTimeoutResult) {
        match condvar.wait_timeout(guard, timeout) {
            Ok(result) => result,
            Err(poisoned) => {
                mutex.clear_poison();
                poisoned.into_inner()
            }
        }
    }

    fn waited_millis(started: Instant) -> u64 {
        u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn complete_round(
        group: &GroupState,
        state: &mut GroupInner,
        outcome: Result<Vec<f32>, DistributedError>,
    ) {
        state.arrived.clear();
        state.buffer.clear();
        state.completed = Some(CompletedRound {
            generation: state.generation,
            outcome,
            pending: state.connected.clone(),
        });
        group.condvar.notify_all();
    }

    fn consume_completed(
        group: &GroupState,
        state: &mut GroupInner,
        rank: usize,
        buffer: &mut [f32],
    ) -> Option<Result<(), DistributedError>> {
        let completed = state.completed.as_mut()?;
        if !completed.pending.remove(&rank) {
            return None;
        }

        let outcome = match &completed.outcome {
            Ok(result) => {
                if result.len() != buffer.len() {
                    Err(DistributedError::BufferLengthMismatch {
                        expected: result.len(),
                        got: buffer.len(),
                    })
                } else {
                    buffer.copy_from_slice(result);
                    Ok(())
                }
            }
            Err(error) => Err(error.clone()),
        };
        let clear = completed.pending.is_empty();
        if clear {
            state.completed = None;
            group.condvar.notify_all();
        }
        Some(outcome)
    }

    /// Handle that represents a rendezvous session.
    #[derive(Debug)]
    pub struct RendezvousSession {
        group: Arc<GroupState>,
        group_id: String,
        rank: usize,
        world_size: usize,
        collective_in_flight: AtomicBool,
    }

    struct CollectiveCallGuard<'a> {
        in_flight: &'a AtomicBool,
    }

    impl Drop for CollectiveCallGuard<'_> {
        fn drop(&mut self) {
            self.in_flight.store(false, AtomicOrdering::Release);
        }
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
            let empty = {
                let mut state = lock_recover(&self.group.inner);
                state.connected.remove(&self.rank);

                if let Some(completed) = state.completed.as_mut() {
                    completed.pending.remove(&self.rank);
                    if completed.pending.is_empty() {
                        state.completed = None;
                        self.group.condvar.notify_all();
                    }
                } else if !state.arrived.is_empty() {
                    let error = DistributedError::ParticipantLeft {
                        rank: self.rank,
                        generation: state.generation,
                    };
                    complete_round(&self.group, &mut state, Err(error));
                }

                if state.connected.is_empty() {
                    state.expected = 0;
                    state.arrived.clear();
                    state.buffer.clear();
                    state.completed = None;
                    self.group.condvar.notify_all();
                }
                state.connected.is_empty()
            };

            if !empty {
                return;
            }

            let mut groups = lock_recover(&GROUPS);
            let remove = groups
                .get(&self.group_id)
                .map(|weak| Weak::ptr_eq(weak, &Arc::downgrade(&self.group)))
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
        #[error("rank {rank} contributed more than once to collective generation {generation}")]
        DuplicateContribution { rank: usize, generation: u64 },
        #[error("buffer length mismatch: expected {expected}, got {got}")]
        BufferLengthMismatch { expected: usize, got: usize },
        #[error("rank {rank} supplied a non-finite collective value at index {index}")]
        NonFiniteInput { rank: usize, index: usize },
        #[error("collective reduction became non-finite at index {index}")]
        NonFiniteReduction { index: usize },
        #[error("rank {rank} left during collective generation {generation}")]
        ParticipantLeft { rank: usize, generation: u64 },
        #[error("collective generation {generation} timed out after {waited_ms} ms")]
        CollectiveTimeout { generation: u64, waited_ms: u64 },
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
            let mut guard = lock_recover(&GROUPS);
            let entry = guard.entry(group.clone()).or_default();
            if let Some(existing) = entry.upgrade() {
                existing
            } else {
                let created = Arc::new(GroupState::new());
                *entry = Arc::downgrade(&created);
                created
            }
        };

        {
            let mut state = lock_recover(&shared_group.inner);
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
            collective_in_flight: AtomicBool::new(false),
        }))
    }

    /// Performs an all-reduce sum with an explicit rendezvous deadline.
    pub(crate) fn all_reduce_with_timeout(
        session: &Arc<RendezvousSession>,
        buffer: &mut [f32],
        timeout: Duration,
    ) -> Result<(), DistributedError> {
        let group = session.group();
        let mut state = lock_recover(&group.inner);

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

        let active_generation = state
            .completed
            .as_ref()
            .map(|completed| completed.generation)
            .unwrap_or(state.generation);
        if session
            .collective_in_flight
            .compare_exchange(false, true, AtomicOrdering::AcqRel, AtomicOrdering::Acquire)
            .is_err()
        {
            return Err(DistributedError::DuplicateContribution {
                rank: session.rank,
                generation: active_generation,
            });
        }
        let _call_guard = CollectiveCallGuard {
            in_flight: &session.collective_in_flight,
        };

        let started = Instant::now();
        let mut contributed_generation = None;

        loop {
            if state.completed.is_some() {
                if let Some(outcome) = consume_completed(group, &mut state, session.rank, buffer) {
                    return outcome;
                }
            } else if contributed_generation != Some(state.generation) {
                if state.arrived.contains(&session.rank) {
                    return Err(DistributedError::DuplicateContribution {
                        rank: session.rank,
                        generation: state.generation,
                    });
                }

                if state.arrived.is_empty() {
                    state.generation = state.generation.wrapping_add(1);
                    state.buffer.clear();
                    state.buffer.resize(buffer.len(), 0.0);
                } else if state.buffer.len() != buffer.len() {
                    let error = DistributedError::BufferLengthMismatch {
                        expected: state.buffer.len(),
                        got: buffer.len(),
                    };
                    state.arrived.insert(session.rank);
                    contributed_generation = Some(state.generation);
                    complete_round(group, &mut state, Err(error));
                    continue;
                }

                state.arrived.insert(session.rank);
                contributed_generation = Some(state.generation);
                if let Some(index) = buffer.iter().position(|value| !value.is_finite()) {
                    let error = DistributedError::NonFiniteInput {
                        rank: session.rank,
                        index,
                    };
                    complete_round(group, &mut state, Err(error));
                    continue;
                }

                let mut reduction_error = None;
                for (index, (dst, value)) in state.buffer.iter_mut().zip(buffer.iter()).enumerate()
                {
                    let sum = *dst + *value;
                    if !sum.is_finite() {
                        reduction_error = Some(DistributedError::NonFiniteReduction { index });
                        break;
                    }
                    *dst = sum;
                }

                if let Some(error) = reduction_error {
                    complete_round(group, &mut state, Err(error));
                    continue;
                }

                if state.arrived.len() == state.expected {
                    let result = std::mem::take(&mut state.buffer);
                    complete_round(group, &mut state, Ok(result));
                    continue;
                }
            }

            let elapsed = started.elapsed();
            if elapsed >= timeout {
                let generation = state
                    .completed
                    .as_ref()
                    .map(|completed| completed.generation)
                    .unwrap_or(state.generation);
                let timeout_error = DistributedError::CollectiveTimeout {
                    generation,
                    waited_ms: waited_millis(started),
                };
                if contributed_generation == Some(state.generation) && state.completed.is_none() {
                    complete_round(group, &mut state, Err(timeout_error));
                    continue;
                }
                return Err(timeout_error);
            }

            let remaining = timeout.saturating_sub(elapsed);
            let (next_state, _) =
                wait_timeout_recover(&group.condvar, &group.inner, state, remaining);
            state = next_state;
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        static GROUP_COUNTER: AtomicUsize = AtomicUsize::new(0);

        fn unique_group(prefix: &str) -> String {
            let id = GROUP_COUNTER.fetch_add(1, Ordering::SeqCst);
            format!("distributed-{prefix}-{id}")
        }

        #[test]
        fn repeated_collectives_do_not_overwrite_unconsumed_rounds() {
            let world_size = 4;
            let rounds = 64;
            let group = unique_group("repeated");
            let sessions = (0..world_size)
                .map(|rank| rendezvous(group.clone(), rank, world_size).unwrap())
                .collect::<Vec<_>>();

            let handles = sessions
                .into_iter()
                .enumerate()
                .map(|(rank, session)| {
                    thread::spawn(move || {
                        for round in 0..rounds {
                            let mut values = [round as f32 + rank as f32, 1.0];
                            all_reduce_with_timeout(&session, &mut values, Duration::from_secs(2))
                                .unwrap();
                            let rank_sum = (world_size * (world_size - 1) / 2) as f32;
                            assert_eq!(
                                values,
                                [
                                    world_size as f32 * round as f32 + rank_sum,
                                    world_size as f32
                                ]
                            );
                            if rank + 1 < world_size {
                                thread::sleep(Duration::from_millis(1));
                            }
                        }
                    })
                })
                .collect::<Vec<_>>();

            for handle in handles {
                handle.join().unwrap();
            }
        }

        #[test]
        fn non_finite_input_is_reported_to_every_connected_rank() {
            let group = unique_group("non-finite");
            let first = rendezvous(group.clone(), 0, 2).unwrap();
            let second = rendezvous(group, 1, 2).unwrap();
            let first_handle = thread::spawn(move || {
                let mut values = [f32::NAN];
                let result = all_reduce_with_timeout(&first, &mut values, Duration::from_secs(1));
                (result, values)
            });
            let second_handle = thread::spawn(move || {
                let mut values = [1.0];
                let result = all_reduce_with_timeout(&second, &mut values, Duration::from_secs(1));
                (result, values)
            });

            for handle in [first_handle, second_handle] {
                let (result, values) = handle.join().unwrap();
                assert_eq!(
                    result,
                    Err(DistributedError::NonFiniteInput { rank: 0, index: 0 })
                );
                assert!(values[0].is_nan() || values == [1.0]);
            }
        }

        #[test]
        fn non_finite_reduction_is_reported_to_every_connected_rank() {
            let group = unique_group("reduction-overflow");
            let sessions = [
                rendezvous(group.clone(), 0, 2).unwrap(),
                rendezvous(group, 1, 2).unwrap(),
            ];
            let handles = sessions.map(|session| {
                thread::spawn(move || {
                    let mut values = [f32::MAX];
                    let result =
                        all_reduce_with_timeout(&session, &mut values, Duration::from_secs(1));
                    (result, values)
                })
            });

            for handle in handles {
                let (result, values) = handle.join().unwrap();
                assert_eq!(
                    result,
                    Err(DistributedError::NonFiniteReduction { index: 0 })
                );
                assert_eq!(values, [f32::MAX]);
            }
        }

        #[test]
        fn missing_rank_times_out_without_mutating_the_caller_buffer() {
            let group = unique_group("timeout");
            let first = rendezvous(group.clone(), 0, 2).unwrap();
            let second = rendezvous(group.clone(), 1, 2).unwrap();
            let mut values = [3.0, -2.0];
            let started = Instant::now();

            let error = all_reduce_with_timeout(&first, &mut values, Duration::from_millis(20))
                .unwrap_err();

            assert!(matches!(
                error,
                DistributedError::CollectiveTimeout { generation: 1, .. }
            ));
            assert!(started.elapsed() < Duration::from_secs(1));
            assert_eq!(values, [3.0, -2.0]);
            drop(second);

            let replacement = rendezvous(group, 1, 2).unwrap();
            let first_worker = Arc::clone(&first);
            let first_handle = thread::spawn(move || {
                let mut values = [2.0];
                all_reduce_with_timeout(&first_worker, &mut values, Duration::from_secs(1))
                    .unwrap();
                values
            });
            let second_handle = thread::spawn(move || {
                let mut values = [5.0];
                all_reduce_with_timeout(&replacement, &mut values, Duration::from_secs(1)).unwrap();
                values
            });
            assert_eq!(first_handle.join().unwrap(), [7.0]);
            assert_eq!(second_handle.join().unwrap(), [7.0]);
        }

        #[test]
        fn rank_drop_aborts_an_active_collective() {
            let group = unique_group("drop");
            let first = rendezvous(group.clone(), 0, 2).unwrap();
            let second = rendezvous(group, 1, 2).unwrap();
            let worker_session = Arc::clone(&first);
            let worker = thread::spawn(move || {
                let mut values = [1.0];
                all_reduce_with_timeout(&worker_session, &mut values, Duration::from_secs(1))
            });

            let started = Instant::now();
            loop {
                let arrived = lock_recover(&first.group().inner).arrived.contains(&0);
                if arrived {
                    break;
                }
                assert!(started.elapsed() < Duration::from_secs(1));
                thread::sleep(Duration::from_millis(1));
            }
            drop(second);

            assert!(matches!(
                worker.join().unwrap(),
                Err(DistributedError::ParticipantLeft {
                    rank: 1,
                    generation: 1
                })
            ));
        }

        #[test]
        fn duplicate_contribution_is_rejected_without_aborting_the_round() {
            let group = unique_group("duplicate-contribution");
            let first = rendezvous(group.clone(), 0, 2).unwrap();
            let second = rendezvous(group, 1, 2).unwrap();
            let worker_session = Arc::clone(&first);
            let worker = thread::spawn(move || {
                let mut values = [1.0];
                let result =
                    all_reduce_with_timeout(&worker_session, &mut values, Duration::from_secs(1));
                (result, values)
            });

            let started = Instant::now();
            loop {
                let arrived = lock_recover(&first.group().inner).arrived.contains(&0);
                if arrived {
                    break;
                }
                assert!(started.elapsed() < Duration::from_secs(1));
                thread::sleep(Duration::from_millis(1));
            }

            let mut duplicate_values = [2.0];
            assert_eq!(
                all_reduce_with_timeout(&first, &mut duplicate_values, Duration::from_secs(1)),
                Err(DistributedError::DuplicateContribution {
                    rank: 0,
                    generation: 1
                })
            );
            let mut peer_values = [3.0];
            all_reduce_with_timeout(&second, &mut peer_values, Duration::from_secs(1)).unwrap();
            let (worker_result, worker_values) = worker.join().unwrap();
            worker_result.unwrap();
            assert_eq!(worker_values, [4.0]);
            assert_eq!(peer_values, [4.0]);
        }

        #[test]
        fn dropping_the_last_session_removes_the_group_registry_entry() {
            let group = unique_group("cleanup");
            let session = rendezvous(group.clone(), 0, 1).unwrap();
            assert!(lock_recover(&GROUPS).contains_key(&group));

            drop(session);

            assert!(!lock_recover(&GROUPS).contains_key(&group));
        }

        #[test]
        fn poisoned_group_lock_is_recovered_before_collective_execution() {
            let group = unique_group("poison-recovery");
            let session = rendezvous(group, 0, 1).unwrap();
            let shared_group = Arc::clone(session.group());
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = shared_group.inner.lock().unwrap();
                panic!("poison collective state");
            }));
            assert!(shared_group.inner.is_poisoned());

            let mut values = [2.0, -3.0];
            all_reduce_with_timeout(&session, &mut values, Duration::from_secs(1)).unwrap();

            assert_eq!(values, [2.0, -3.0]);
            assert!(!shared_group.inner.is_poisoned());
        }
    }
}
