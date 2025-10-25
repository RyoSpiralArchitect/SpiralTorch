// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

#![cfg(feature = "golden")]
use std::convert::Infallible;
use std::ops::{Deref, DerefMut};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use st_tensor::{Tensor, TensorError};
use thiserror::Error;

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

#[derive(Debug)]
pub enum GoldenTaskError<E> {
    Runtime(GoldenRuntimeError),
    Task(E),
    Panic,
}

impl<E: core::fmt::Display> core::fmt::Display for GoldenTaskError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GoldenTaskError::Runtime(err) => write!(f, "{err}"),
            GoldenTaskError::Task(err) => write!(f, "{err}"),
            GoldenTaskError::Panic => write!(f, "golden runtime task panicked"),
        }
    }
}

impl<E> std::error::Error for GoldenTaskError<E> where E: std::error::Error + 'static {}

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

type ThreadResult<R, E> = Result<R, GoldenTaskError<E>>;

enum GoldenTaskMessage {
    Run(Box<dyn FnOnce() + Send + 'static>),
    Shutdown,
}

struct GoldenRuntimeInner {
    workers: usize,
    name: String,
    sender: Sender<GoldenTaskMessage>,
    handles: Mutex<Vec<thread::JoinHandle<()>>>,
    shutdown: AtomicBool,
}

impl GoldenRuntimeInner {
    fn spawn_workers(
        inner: &Arc<Self>,
        receiver: Receiver<GoldenTaskMessage>,
    ) -> Result<(), GoldenRuntimeError> {
        let shared_receiver = Arc::new(receiver);
        let mut handles = inner
            .handles
            .lock()
            .expect("golden runtime worker handle mutex poisoned");
        for idx in 0..inner.workers {
            let name = format!("{}-{}", inner.name, idx);
            let worker_receiver = shared_receiver.clone();
            let handle = thread::Builder::new()
                .name(name)
                .spawn(move || worker_loop(worker_receiver))
                .map_err(|err| {
                    GoldenRuntimeError(format!("failed to spawn golden worker: {err}"))
                })?;
            handles.push(handle);
        }
        Ok(())
    }

    fn worker_count(&self) -> usize {
        self.workers
    }
}

impl Drop for GoldenRuntimeInner {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        for _ in 0..self.workers {
            let _ = self.sender.send(GoldenTaskMessage::Shutdown);
        }
        let mut handles = self
            .handles
            .lock()
            .expect("golden runtime worker handle mutex poisoned");
        while let Some(handle) = handles.pop() {
            let _ = handle.join();
        }
    }
}

fn worker_loop(receiver: Arc<Receiver<GoldenTaskMessage>>) {
    while let Ok(message) = receiver.recv() {
        match message {
            GoldenTaskMessage::Run(job) => job(),
            GoldenTaskMessage::Shutdown => break,
        }
    }
}

pub struct GoldenJoinHandle<R, E> {
    receiver: Option<Receiver<ThreadResult<R, E>>>,
}

impl<R, E> GoldenJoinHandle<R, E> {
    pub fn join(self) -> ThreadResult<R, E> {
        let receiver = self
            .receiver
            .expect("golden runtime join handle already consumed");
        match receiver.recv() {
            Ok(result) => result,
            Err(_) => Err(GoldenTaskError::Runtime(GoldenRuntimeError(
                "golden runtime worker dropped result".into(),
            ))),
        }
    }
}

/// Lightweight runtime that mimics Tokio/Rayon ergonomics using the standard
/// library. Tasks are scheduled on a small pool of reusable threads and can be
/// joined later, while reduction helpers keep aggregation deterministic.
pub struct GoldenRuntime {
    inner: Arc<GoldenRuntimeInner>,
}

impl Clone for GoldenRuntime {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl GoldenRuntime {
    pub fn new(config: GoldenRuntimeConfig) -> Result<Self, GoldenRuntimeError> {
        let workers = config.worker_threads.max(1);
        let name = config.thread_name.unwrap_or_else(|| "golden".into());
        let (sender, receiver) = unbounded::<GoldenTaskMessage>();
        let inner = Arc::new(GoldenRuntimeInner {
            workers,
            name,
            sender,
            handles: Mutex::new(Vec::with_capacity(workers)),
            shutdown: AtomicBool::new(false),
        });
        GoldenRuntimeInner::spawn_workers(&inner, receiver)?;
        Ok(Self { inner })
    }

    pub fn worker_count(&self) -> usize {
        self.inner.worker_count()
    }

    pub fn execute<F, R, E>(&self, func: F) -> Result<R, GoldenTaskError<E>>
    where
        F: FnOnce() -> Result<R, E> + Send + 'static,
        R: Send + 'static,
        E: Send + 'static,
    {
        let handle = self
            .spawn_blocking(func)
            .map_err(GoldenTaskError::Runtime)?;
        handle.join()
    }

    pub fn tensor_random_uniform(
        &self,
        rows: usize,
        cols: usize,
        min: f32,
        max: f32,
        seed: Option<u64>,
    ) -> Result<Tensor, GoldenTensorError> {
        self.execute(move || Tensor::random_uniform(rows, cols, min, max, seed))
            .map_err(GoldenTensorError::from)
    }

    pub fn tensor_random_normal(
        &self,
        rows: usize,
        cols: usize,
        mean: f32,
        std: f32,
        seed: Option<u64>,
    ) -> Result<Tensor, GoldenTensorError> {
        self.execute(move || Tensor::random_normal(rows, cols, mean, std, seed))
            .map_err(GoldenTensorError::from)
    }

    pub fn spawn_blocking<F, R, E>(
        &self,
        func: F,
    ) -> Result<GoldenJoinHandle<R, E>, GoldenRuntimeError>
    where
        F: FnOnce() -> Result<R, E> + Send + 'static,
        R: Send + 'static,
        E: Send + 'static,
    {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return Err(GoldenRuntimeError(
                "golden runtime is shutting down and cannot accept new tasks".into(),
            ));
        }
        let (result_tx, result_rx) = bounded::<ThreadResult<R, E>>(1);
        let job = Box::new(move || {
            let result = panic::catch_unwind(AssertUnwindSafe(func))
                .map_err(|_| GoldenTaskError::Panic)
                .and_then(|inner| inner.map_err(GoldenTaskError::Task));
            let _ = result_tx.send(result);
        });
        self.inner
            .sender
            .send(GoldenTaskMessage::Run(job))
            .map_err(|_| {
                GoldenRuntimeError(
                    "golden runtime worker queue rejected task (runtime shutting down)".into(),
                )
            })?;
        Ok(GoldenJoinHandle {
            receiver: Some(result_rx),
        })
    }

    pub fn reduce<T, R, Map, Fold>(
        &self,
        data: &[T],
        map: Map,
        fold: Fold,
        identity: R,
    ) -> Result<R, GoldenTaskError<Infallible>>
    where
        T: Sync + 'static,
        R: Send + Sync + Clone + 'static,
        Map: Fn(&T) -> R + Send + Sync + 'static,
        Fold: Fn(R, R) -> R + Send + Sync + 'static,
    {
        let workers = self.worker_count();
        if workers <= 1 || data.len() < 2 {
            return data
                .iter()
                .map(&map)
                .fold(identity.clone(), |acc, item| fold(acc, item));
        }
        let chunk = (data.len() + workers - 1) / workers;
        let mut partials = Vec::new();
        thread::scope(|scope| -> Result<(), GoldenTaskError<Infallible>> {
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
            for handle in handles {
                match handle.join() {
                    Ok(value) => partials.push(value),
                    Err(_) => return Err(GoldenTaskError::Panic),
                }
            }
            Ok(())
        })?;

        let mut acc = identity;
        for value in partials {
            acc = fold(acc, value);
        }

        Ok(acc)
    }
}

#[derive(Debug, Error)]
pub enum GoldenTensorError {
    #[error("{0}")]
    Runtime(#[from] GoldenRuntimeError),
    #[error("{0}")]
    Tensor(#[from] TensorError),
}

impl From<GoldenTaskError<TensorError>> for GoldenTensorError {
    fn from(err: GoldenTaskError<TensorError>) -> Self {
        match err {
            GoldenTaskError::Runtime(inner) => Self::Runtime(inner),
            GoldenTaskError::Task(inner) => Self::Tensor(inner),
            GoldenTaskError::Panic => {
                Self::Runtime(GoldenRuntimeError("golden runtime task panicked".into()))
            }
        }
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
            handles.push(
                runtime
                    .spawn_blocking(move || Ok::<usize, Infallible>(idx * 2))
                    .expect("spawn"),
            );
        }
        let mut total = 0usize;
        for handle in handles {
            total += handle.join().expect("join");
        }
        assert!(total > 0);

        let numbers: Vec<u32> = (0..32).collect();
        let reduced = runtime
            .reduce(&numbers, |value| *value as usize, |a, b| a + b, 0usize)
            .expect("reduce");
        assert_eq!(
            reduced,
            numbers.iter().copied().map(|v| v as usize).sum::<usize>()
        );
    }

    #[test]
    fn runtime_random_uniform_matches_tensor() {
        let runtime = GoldenRuntime::new(GoldenRuntimeConfig::default()).expect("runtime");
        let direct = Tensor::random_uniform(3, 4, -1.0, 1.0, Some(7)).expect("direct");
        let generated = runtime
            .tensor_random_uniform(3, 4, -1.0, 1.0, Some(7))
            .expect("runtime tensor");
        assert_eq!(direct, generated);
    }

    #[test]
    fn runtime_random_normal_matches_tensor() {
        let runtime = GoldenRuntime::new(GoldenRuntimeConfig::default()).expect("runtime");
        let direct = Tensor::random_normal(2, 5, 0.5, 1.25, Some(11)).expect("direct");
        let generated = runtime
            .tensor_random_normal(2, 5, 0.5, 1.25, Some(11))
            .expect("runtime tensor");
        assert_eq!(direct, generated);
    }

    #[test]
    fn runtime_random_uniform_propagates_tensor_errors() {
        let runtime = GoldenRuntime::new(GoldenRuntimeConfig::default()).expect("runtime");
        let err = runtime
            .tensor_random_uniform(0, 4, 0.0, 1.0, None)
            .expect_err("invalid dimensions should fail");
        if let GoldenTensorError::Tensor(inner) = err {
            let message = inner.to_string();
            assert!(
                message.contains("invalid tensor dimensions"),
                "unexpected error message: {message}"
            );
        } else {
            panic!("expected tensor error variant");
        }
    }

    #[test]
    fn runtime_random_normal_propagates_tensor_errors() {
        let runtime = GoldenRuntime::new(GoldenRuntimeConfig::default()).expect("runtime");
        let err = runtime
            .tensor_random_normal(3, 3, 0.0, 0.0, None)
            .expect_err("invalid std should fail");
        let message = err.to_string();
        assert!(message.contains("random_normal_std"));
    }
}
