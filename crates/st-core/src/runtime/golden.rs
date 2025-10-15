// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "golden")]
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

/// Shared ownership wrapper that mirrors `Arc` while leaving room for future
/// instrumentation.
#[derive(Debug, Clone)]
pub struct SpiralArc<T>(Arc<T>);

impl<T> SpiralArc<T> {
    pub fn new(value: T) -> Self {
        Self(Arc::new(value))
    }

    pub fn strong_count(this: &Self) -> usize {
        Arc::strong_count(&this.0)
    }

    pub fn get_ref(this: &Self) -> &T {
        &this.0
    }
}

impl<T> From<Arc<T>> for SpiralArc<T> {
    fn from(value: Arc<T>) -> Self {
        Self(value)
    }
}

impl<T> From<SpiralArc<T>> for Arc<T> {
    fn from(value: SpiralArc<T>) -> Self {
        value.0
    }
}

/// Mutex wrapper that avoids poisoning the lock and keeps the ergonomics inline
/// with the rest of SpiralTorch.
#[derive(Debug)]
pub struct SpiralMutex<T> {
    inner: Arc<Mutex<T>>,
}

impl<T> SpiralMutex<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(value)),
        }
    }

    pub fn lock(&self) -> SpiralMutexGuard<'_, T> {
        match self.inner.lock() {
            Ok(guard) => SpiralMutexGuard { guard },
            Err(poisoned) => SpiralMutexGuard {
                guard: poisoned.into_inner(),
            },
        }
    }
}

impl<T> Clone for SpiralMutex<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

pub struct SpiralMutexGuard<'a, T> {
    guard: MutexGuard<'a, T>,
}

impl<'a, T> Deref for SpiralMutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<'a, T> DerefMut for SpiralMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

#[derive(Debug, Clone)]
pub struct GoldenRuntimeError(pub String);

impl core::fmt::Display for GoldenRuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for GoldenRuntimeError {}

#[derive(Debug, Clone)]
pub struct GoldenRuntimeConfig {
    pub worker_threads: usize,
    pub thread_name: Option<String>,
}

impl Default for GoldenRuntimeConfig {
    fn default() -> Self {
        let default_workers = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            worker_threads: default_workers,
            thread_name: Some("golden".into()),
        }
    }
}

/// Lightweight runtime that mimics Tokio/Rayon ergonomics using the standard
/// library. Tasks are spawned onto named threads and can be joined later, while
/// reduction helpers keep aggregation deterministic.
#[derive(Clone)]
pub struct GoldenRuntime {
    workers: usize,
    name: Arc<String>,
    counter: Arc<AtomicUsize>,
}

impl GoldenRuntime {
    pub fn new(config: GoldenRuntimeConfig) -> Result<Self, GoldenRuntimeError> {
        let workers = config.worker_threads.max(1);
        let name = config.thread_name.unwrap_or_else(|| "golden".into());
        Ok(Self {
            workers,
            name: Arc::new(name),
            counter: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn worker_count(&self) -> usize {
        self.workers
    }

    pub fn spawn_blocking<F, R>(&self, func: F) -> Result<thread::JoinHandle<R>, GoldenRuntimeError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed);
        let label = format!("{}-{}", self.name, idx);
        thread::Builder::new()
            .name(label)
            .spawn(func)
            .map_err(|err| GoldenRuntimeError(format!("failed to spawn golden worker: {err}")))
    }

    pub fn reduce<T, R, Map, Fold>(&self, data: &[T], map: Map, fold: Fold, identity: R) -> R
    where
        T: Sync,
        R: Send + Sync + Clone,
        Map: Fn(&T) -> R + Send + Sync,
        Fold: Fn(R, R) -> R + Send + Sync,
    {
        if self.workers <= 1 || data.len() < 2 {
            return data
                .iter()
                .map(&map)
                .fold(identity.clone(), |acc, item| fold(acc, item));
        }
        let chunk = (data.len() + self.workers - 1) / self.workers;
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk_items in data.chunks(chunk) {
                let identity_clone = identity.clone();
                let map_ref = &map;
                let fold_ref = &fold;
                handles.push(scope.spawn(move || {
                    chunk_items
                        .iter()
                        .map(map_ref)
                        .fold(identity_clone, |acc, item| fold_ref(acc, item))
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap_or_else(|_| identity.clone()))
                .fold(identity.clone(), |acc, item| fold(acc, item))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_spawns_threads_and_reduces() {
        let runtime = GoldenRuntime::new(GoldenRuntimeConfig::default()).expect("runtime");
        let mut handles = Vec::new();
        for idx in 0..runtime.worker_count() {
            handles.push(runtime.spawn_blocking(move || idx * 2).expect("spawn"));
        }
        let mut total = 0usize;
        for handle in handles {
            total += handle.join().expect("join");
        }
        assert!(total > 0);

        let numbers: Vec<u32> = (0..32).collect();
        let reduced = runtime.reduce(&numbers, |value| *value as usize, |a, b| a + b, 0usize);
        assert_eq!(
            reduced,
            numbers.iter().copied().map(|v| v as usize).sum::<usize>()
        );
    }
}
