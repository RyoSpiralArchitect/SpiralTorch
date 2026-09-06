//! Persistent exact rank storage with enqueue-only dispatch and owned snapshots.

use super::{binding, storage_buffer, DispatchError, Output, Pipelines, Plan};
use crate::resident_matmul::{MatmulError, ResidentMatmul};
use crate::runtime::timestamps::{
    PassTimestampRecorder, PassTimestamps, TimestampErrorScopes, TimestampReadback,
};
use crate::runtime::{self, WgpuContext, WgpuRuntime, WgpuRuntimeError};
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;
use wgpu::util::DeviceExt;

pub const MAX_REPETITIONS: u32 = 1024;
// Portable pass-level instrumentation uses many more command buffers than the
// ordinary single-pass path. Submit before Metal's command-buffer pool fills.
const PROFILE_REPETITIONS_PER_SUBMISSION: u32 = 256;

#[derive(Debug, Error)]
pub enum ResidentRankError {
    #[error("resident rank requires nonempty rows, columns and k")]
    EmptyWorkspace,
    #[error("upload input before dispatch")]
    MissingInput,
    #[error("dispatch current input before requesting a snapshot")]
    StaleOutput,
    #[error("matmul source must have current dispatched output")]
    StaleSource,
    #[error("matmul output must match rank rows/columns and share the same device and queue")]
    IncompatibleSource,
    #[error("rank repetitions must be in 1..={MAX_REPETITIONS}")]
    InvalidRepetitions,
    #[error("rank input generation counter exhausted")]
    GenerationOverflow,
    #[error("rank profiling requires a private device; use ResidentRank::request_profiled or request_profiled_blocking")]
    ProfileRequiresPrivateDevice,
    #[error(transparent)]
    Matmul(#[from] MatmulError),
    #[error(transparent)]
    Dispatch(#[from] DispatchError),
    #[error(transparent)]
    Runtime(#[from] WgpuRuntimeError),
}

/// One fixed rank shape/tile. Upload invalidates output; dispatch never reads it.
pub struct ResidentRank {
    plan: Plan,
    pipelines: Pipelines,
    input: wgpu::Buffer,
    values: wgpu::Buffer,
    indices: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    generation: u64,
    output_generation: Option<u64>,
    pending_profile: Option<runtime::Shared<AtomicBool>>,
    private_profile_device: bool,
    readback_pool: runtime::ReadbackPool,
    // The owning device must outlive every workspace buffer and pipeline.
    runtime: WgpuRuntime,
}

impl ResidentRank {
    /// Own a dedicated timestamp device whose handles never escape this workspace.
    /// Shareable runtimes remain usable for ordinary work, not scoped profiling.
    pub async fn request_profiled(plan: Plan) -> Result<Self, ResidentRankError> {
        if plan.is_empty() {
            return Err(ResidentRankError::EmptyWorkspace);
        }
        let runtime =
            WgpuRuntime::request_profiled_headless("resident.rank.private_profile").await?;
        let errors = TimestampErrorScopes::try_new(runtime.context().clone())?;
        let workspace = Self::new(runtime, plan);
        errors.finish().check().await?;
        let mut workspace = workspace?;
        workspace.private_profile_device = true;
        Ok(workspace)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn request_profiled_blocking(plan: Plan) -> Result<Self, ResidentRankError> {
        pollster::block_on(Self::request_profiled(plan))
    }

    /// Await browser validation before exposing a workspace as executable.
    #[cfg(target_arch = "wasm32")]
    pub async fn new_async(runtime: WgpuRuntime, plan: Plan) -> Result<Self, ResidentRankError> {
        let context = runtime.context().clone();
        context
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let workspace = Self::new(runtime, plan);
        let validation = context.device().pop_error_scope().await;
        if let Some(error) = validation {
            return Err(DispatchError::PipelineBuild(error.to_string()).into());
        }
        workspace
    }

    pub fn new(runtime: WgpuRuntime, plan: Plan) -> Result<Self, ResidentRankError> {
        if plan.is_empty() {
            return Err(ResidentRankError::EmptyWorkspace);
        }
        let device = runtime.context().device();
        plan.validate_device(device)?;
        let snapshot_bytes = u64::from(plan.output_elements()) * 8;
        if snapshot_bytes > device.limits().max_buffer_size {
            return Err(DispatchError::DeviceLimit {
                resource: "combined rank snapshot",
                required: snapshot_bytes,
                available: device.limits().max_buffer_size,
            }
            .into());
        }
        let pipelines = Pipelines::new(device)?;
        let input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident.rank.input"),
            size: u64::from(plan.input_elements()) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch_values = storage_buffer(
            device,
            "resident.rank.scratch_values",
            plan.scratch_elements(),
            false,
        );
        let scratch_indices = storage_buffer(
            device,
            "resident.rank.scratch_indices",
            plan.scratch_elements(),
            false,
        );
        let counts = storage_buffer(
            device,
            "resident.rank.counts",
            plan.tile_state_elements(),
            false,
        );
        let cursors = storage_buffer(
            device,
            "resident.rank.cursors",
            plan.tile_state_elements(),
            false,
        );
        let values = storage_buffer(device, "resident.rank.values", plan.output_elements(), true);
        let indices = storage_buffer(
            device,
            "resident.rank.indices",
            plan.output_elements(),
            true,
        );
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resident.rank.params"),
            contents: bytemuck::bytes_of(&plan.params()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // The bind group retains scratch and uniform buffers for this workspace.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident.rank.bind_group"),
            layout: &pipelines.layout,
            entries: &[
                binding(0, &input),
                binding(1, &scratch_values),
                binding(2, &scratch_indices),
                binding(3, &counts),
                binding(4, &cursors),
                binding(5, &values),
                binding(6, &indices),
                binding(7, &params),
            ],
        });
        let readback_pool = runtime::ReadbackPool::new::<u64>(
            runtime.context().clone(),
            plan.output_elements() as usize,
        )?;
        Ok(Self {
            runtime,
            plan,
            pipelines,
            input,
            values,
            indices,
            bind_group,
            generation: 0,
            output_generation: None,
            pending_profile: None,
            private_profile_device: false,
            readback_pool,
        })
    }

    pub fn plan(&self) -> Plan {
        self.plan
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn output_is_current(&self) -> bool {
        self.output_generation == Some(self.generation)
            && self
                .pending_profile
                .as_ref()
                .is_none_or(|status| status.load(Ordering::Acquire))
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.runtime.adapter_info()
    }

    pub fn upload(&mut self, input: &[f32]) -> Result<(), ResidentRankError> {
        if input.len() != self.plan.input_elements() as usize {
            return Err(DispatchError::InputLength {
                expected: self.plan.input_elements() as usize,
                actual: input.len(),
            }
            .into());
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(ResidentRankError::GenerationOverflow)?;
        self.runtime
            .context()
            .queue()
            .write_buffer(&self.input, 0, bytemuck::cast_slice(input));
        self.generation = generation;
        self.output_generation = None;
        self.pending_profile = None;
        Ok(())
    }

    /// Queue an owned device-to-device input copy, with no host readback or alias.
    /// A later source update/drop cannot change this input. Validation failures
    /// leave both the prior input generation and output freshness unchanged.
    pub fn set_input_from_matmul(
        &mut self,
        source: &ResidentMatmul,
    ) -> Result<(), ResidentRankError> {
        let (source_context, output) = source
            .current_output()
            .ok_or(ResidentRankError::StaleSource)?;
        let (rows, _, cols) = source.shape().dimensions();
        let context = self.runtime.context();
        if (rows, cols) != (self.plan.rows() as usize, self.plan.cols() as usize)
            || !context.shares_handles_with(source_context)
        {
            return Err(ResidentRankError::IncompatibleSource);
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(ResidentRankError::GenerationOverflow)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(output, 0, &self.input, 0, self.input.size());
        context.queue().submit(Some(encoder.finish()));
        self.generation = generation;
        self.output_generation = None;
        self.pending_profile = None;
        Ok(())
    }

    /// Enqueue complete matmul/copy/rank chains in one submission, without maps.
    ///
    /// Unlike the copy-only setter, this computes from uploaded source operands;
    /// source output need not be current. Repetitions reuse those same operands.
    /// One call creates one rank input generation, regardless of repetitions.
    /// Validation failures preserve both workspaces. Freshness is recorded only
    /// after submission, not GPU completion. The rank input is still an owned copy.
    pub fn dispatch_from_matmul(
        &mut self,
        source: &mut ResidentMatmul,
        repetitions: u32,
    ) -> Result<u64, ResidentRankError> {
        if repetitions == 0 || repetitions > MAX_REPETITIONS {
            return Err(ResidentRankError::InvalidRepetitions);
        }
        let (source_context, output) = source.prepare_dispatch(repetitions)?;
        let (rows, _, cols) = source.shape().dimensions();
        let context = self.runtime.context();
        if (rows, cols) != (self.plan.rows() as usize, self.plan.cols() as usize)
            || !context.shares_handles_with(source_context)
        {
            return Err(ResidentRankError::IncompatibleSource);
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(ResidentRankError::GenerationOverflow)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        for _ in 0..repetitions {
            source.encode_dispatch(&mut encoder, 1);
            encoder.copy_buffer_to_buffer(output, 0, &self.input, 0, self.input.size());
            self.encode_dispatch(&mut encoder, 1);
        }
        context.queue().submit(Some(encoder.finish()));
        source.mark_dispatched();
        self.generation = generation;
        self.output_generation = Some(generation);
        self.pending_profile = None;
        Ok(generation)
    }

    /// Enqueues ordered sort/merge dispatches in one compute pass and submission.
    pub fn dispatch(&mut self, repetitions: u32) -> Result<u64, ResidentRankError> {
        if repetitions == 0 || repetitions > MAX_REPETITIONS {
            return Err(ResidentRankError::InvalidRepetitions);
        }
        if self.generation == 0 {
            return Err(ResidentRankError::MissingInput);
        }
        let context = self.runtime.context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        self.encode_dispatch(&mut encoder, repetitions);
        context.queue().submit(Some(encoder.finish()));
        self.output_generation = Some(self.generation);
        self.pending_profile = None;
        Ok(self.generation)
    }

    fn encode_dispatch(&self, encoder: &mut wgpu::CommandEncoder, repetitions: u32) {
        let (merge_x, merge_y) = self.plan.merge_workgroups();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resident.rank.pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bind_group, &[]);
        for _ in 0..repetitions {
            // Each WebGPU dispatch is its own usage scope. wgpu inserts storage
            // dependencies between dispatches, including reuse of scratch buffers.
            pass.set_pipeline(&self.pipelines.tile_sort);
            pass.dispatch_workgroups(self.plan.tiles_x(), self.plan.rows(), 1);
            pass.set_pipeline(self.pipelines.merge_pipeline(self.plan));
            pass.dispatch_workgroups(merge_x, merge_y, 1);
        }
    }

    /// Profile the same kernels with separate passes for portable GPU timestamps.
    /// This diagnostic uses up to 256 repetitions per submission; native Metal
    /// waits between chunks to bound command-buffer pressure. It is not the
    /// ordinary single-pass dispatch, and its intervals are not fast-path costs.
    /// Only the private-device factory enables this path: caller-owned device
    /// operations cannot enter its error scopes. Ordinary dispatch gains no new lock.
    /// Profiled output stays stale until its readback succeeds. Uploads, newer
    /// profiles and ordinary dispatches detach this profile's publication token.
    pub fn dispatch_profiled(
        &mut self,
        repetitions: u32,
    ) -> Result<RankProfileReadback, ResidentRankError> {
        if repetitions == 0 || repetitions > MAX_REPETITIONS {
            return Err(ResidentRankError::InvalidRepetitions);
        }
        if self.generation == 0 {
            return Err(ResidentRankError::MissingInput);
        }
        if !self.timestamp_queries_enabled() {
            return Err(WgpuRuntimeError::TimestampQueriesUnavailable.into());
        }
        if !self.private_profile_device {
            return Err(ResidentRankError::ProfileRequiresPrivateDevice);
        }
        let context = self.runtime.context();
        let errors = TimestampErrorScopes::try_new(context.clone())?;
        let timestamps = PassTimestampRecorder::new(context.clone(), repetitions * 2)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        for start in (0..repetitions).step_by(PROFILE_REPETITIONS_PER_SUBMISSION as usize) {
            let end = (start + PROFILE_REPETITIONS_PER_SUBMISSION).min(repetitions);
            self.encode_profiled_dispatch(&mut encoder, start..end, &timestamps);
            if end < repetitions {
                let commands = [encoder.finish()];
                #[cfg(not(target_arch = "wasm32"))]
                if self.adapter_info().backend == wgpu::Backend::Metal {
                    runtime::submit_with_timeout(
                        context.device(),
                        context.queue(),
                        commands,
                        std::time::Duration::from_secs(30),
                        "resident.rank.profile.chunk",
                    )?;
                } else {
                    context.queue().submit(commands);
                }
                #[cfg(target_arch = "wasm32")]
                context.queue().submit(commands);
                encoder = context.device().create_command_encoder(&Default::default());
            }
        }
        let mut readback = timestamps.resolve(&mut encoder);
        context.queue().submit(Some(encoder.finish()));
        readback.validate(errors.finish());
        let validated = runtime::Shared::new(AtomicBool::new(false));
        self.output_generation = Some(self.generation);
        self.pending_profile = Some(validated.clone());
        Ok(RankProfileReadback {
            readback,
            validated,
            plan: self.plan,
            generation: self.generation,
            repetitions,
            host_paced_chunks: cfg!(not(target_arch = "wasm32"))
                && self.adapter_info().backend == wgpu::Backend::Metal
                && repetitions > PROFILE_REPETITIONS_PER_SUBMISSION,
        })
    }

    pub fn timestamp_queries_enabled(&self) -> bool {
        self.runtime
            .context()
            .device()
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    fn encode_profiled_dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        repetitions: std::ops::Range<u32>,
        timestamps: &PassTimestampRecorder,
    ) {
        let (merge_x, merge_y) = self.plan.merge_workgroups();
        for repetition in repetitions {
            for (stage, (pipeline, x, y)) in [
                (
                    &self.pipelines.tile_sort,
                    self.plan.tiles_x(),
                    self.plan.rows(),
                ),
                (self.pipelines.merge_pipeline(self.plan), merge_x, merge_y),
            ]
            .into_iter()
            .enumerate()
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("resident.rank.pass"),
                    timestamp_writes: Some(timestamps.writes(repetition * 2 + stage as u32)),
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
    }

    /// Copies values and indices now into one owned map-readable staging buffer.
    pub fn snapshot(&self) -> Result<RankReadback, ResidentRankError> {
        if !self.output_is_current() {
            return Err(ResidentRankError::StaleOutput);
        }
        let context = self.runtime.context();
        let staging = self.readback_pool.checkout("resident.rank.snapshot");
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let bytes = u64::from(self.plan.output_elements()) * 4;
        encoder.copy_buffer_to_buffer(&self.values, 0, staging.buffer(), 0, bytes);
        encoder.copy_buffer_to_buffer(&self.indices, 0, staging.buffer(), bytes, bytes);
        context.queue().submit(Some(encoder.finish()));
        Ok(RankReadback {
            context: context.clone(),
            staging,
            plan: self.plan,
            generation: self.generation,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn synchronize(&self) -> Result<(), ResidentRankError> {
        let context = self.runtime.context();
        runtime::submit_with_timeout(
            context.device(),
            context.queue(),
            [],
            std::time::Duration::from_secs(30),
            "resident.rank",
        )?;
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    pub fn synchronize_async(
        &self,
    ) -> Result<
        impl std::future::Future<Output = Result<(), ResidentRankError>> + 'static,
        ResidentRankError,
    > {
        let context = self.runtime.context().clone();
        let staging = runtime::empty_buffer::<u32>(
            context.device(),
            "resident.rank.fence",
            1,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.values, 0, &staging, 0, 4);
        context.queue().submit(Some(encoder.finish()));
        Ok(async move {
            map_bytes(context, runtime::ReadbackLease::unpooled(staging))
                .await
                .map(|_| ())
        })
    }
}

/// GPU query results remain tied to the dispatch generation after later uploads.
pub struct RankProfileReadback {
    readback: TimestampReadback,
    validated: runtime::Shared<AtomicBool>,
    plan: Plan,
    generation: u64,
    repetitions: u32,
    host_paced_chunks: bool,
}

impl RankProfileReadback {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<RankGpuProfile, ResidentRankError> {
        let timestamps = self.readback.read()?;
        self.validated.store(true, Ordering::Release);
        Ok(RankGpuProfile {
            plan: self.plan,
            generation: self.generation,
            repetitions: self.repetitions,
            host_paced_chunks: self.host_paced_chunks,
            timestamps,
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<RankGpuProfile, ResidentRankError> {
        let timestamps = self.readback.read_async().await?;
        self.validated.store(true, Ordering::Release);
        Ok(RankGpuProfile {
            plan: self.plan,
            generation: self.generation,
            repetitions: self.repetitions,
            host_paced_chunks: self.host_paced_chunks,
            timestamps,
        })
    }
}

#[derive(Clone, Debug)]
pub struct RankGpuProfile {
    plan: Plan,
    generation: u64,
    repetitions: u32,
    host_paced_chunks: bool,
    timestamps: PassTimestamps,
}

impl RankGpuProfile {
    pub fn plan(&self) -> Plan {
        self.plan
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn repetitions(&self) -> u32 {
        self.repetitions
    }
    pub fn timestamps(&self) -> &PassTimestamps {
        &self.timestamps
    }

    /// Shared client schema; absolute u64 clocks and generations are decimal strings.
    /// Timestamps perturb execution and may be quantized to zero by the browser.
    pub fn report(&self) -> serde_json::Value {
        let mut sort_ns = 0.0;
        let mut merge_ns = 0.0;
        let mut zero_intervals = 0;
        let passes: Vec<_> = self.timestamps.passes.iter().enumerate().map(|(index, interval)| {
            let sort = index % 2 == 0;
            if sort { sort_ns += interval.elapsed_ns; } else { merge_ns += interval.elapsed_ns; }
            zero_intervals += usize::from(interval.elapsed_ns == 0.0);
            serde_json::json!({"repetition": index / 2, "stage": if sort { "tile_sort" } else { "row_merge" },
                "start_tick": interval.start_tick.to_string(), "end_tick": interval.end_tick.to_string(), "elapsed_ns": interval.elapsed_ns})
        }).collect();
        let span = self
            .timestamps
            .passes
            .first()
            .zip(self.timestamps.passes.last())
            .and_then(|(first, last)| last.end_tick.checked_sub(first.start_tick))
            .map(|ticks| ticks as f64 * self.timestamps.timestamp_period_ns);
        serde_json::json!({
            "schema": "spiraltorch.rank_gpu_profile.v1", "instrumented": true,
            "boundary": "Diagnostic separate-pass GPU timestamps, not the ordinary single-pass path; excludes uploads, query resolve/readback and CPU encoding; GPU span includes inter-pass and inter-submission gaps; instrumentation may perturb execution",
            "compute_submissions": self.repetitions.div_ceil(PROFILE_REPETITIONS_PER_SUBMISSION),
            "host_paced_chunks": self.host_paced_chunks,
            "max_repetitions_per_submission": PROFILE_REPETITIONS_PER_SUBMISSION,
            "generation": self.generation.to_string(), "repetitions": self.repetitions,
            "kind": self.plan.kind().as_str(), "rows": self.plan.rows(), "cols": self.plan.cols(), "k": self.plan.k(), "tile_cols": self.plan.tile_cols(),
            "merge_entry_point": self.plan.merge_mode().entry_point(),
            "timestamp_period_ns": self.timestamps.timestamp_period_ns,
            "zero_intervals": zero_intervals, "gpu_span_ns": span,
            "tile_sort_total_ns": sort_ns, "row_merge_total_ns": merge_ns,
            "passes": passes,
        })
    }
}

/// A snapshot remains valid after later uploads, dispatches, or workspace drop.
pub struct RankReadback {
    staging: runtime::ReadbackLease,
    plan: Plan,
    generation: u64,
    context: WgpuContext,
}

impl RankReadback {
    pub fn plan(&self) -> Plan {
        self.plan
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(mut self) -> Result<Output, ResidentRankError> {
        let bytes = self.staging.read(
            &self.context,
            std::time::Duration::from_secs(30),
            "resident.rank.snapshot",
        )?;
        Ok(decode(&bytes))
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<Output, ResidentRankError> {
        let bytes = map_bytes(self.context, self.staging).await?;
        Ok(decode(&bytes))
    }
}

fn decode(bytes: &[u8]) -> Output {
    let (values, indices) = bytes.split_at(bytes.len() / 2);
    Output {
        values: values
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect(),
        indices: indices
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| i32::from_le_bytes(*v))
            .collect(),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod profiling_tests {
    use super::*;
    use crate::rankk_exact_2ce::Kind;

    #[test]
    fn private_profile_devices_retire_after_workspace_and_pending_reads() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS").is_none() {
            return;
        }
        for cycle in 0..96 {
            let plan = Plan::try_new(Kind::TopK, 1, 8, 2, 4).unwrap();
            let mut rank = ResidentRank::request_profiled_blocking(plan).unwrap();
            rank.upload(&[0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();
            rank.dispatch(1).unwrap();
            let snapshot = rank.snapshot().unwrap();
            let pending = rank.dispatch_profiled(1).unwrap();
            if cycle % 3 == 0 {
                pending.read().unwrap();
                assert_eq!(snapshot.read().unwrap().values, [7., 6.]);
                drop(rank);
            } else {
                drop(rank);
                if cycle % 3 == 1 {
                    pending.read().unwrap();
                    assert_eq!(snapshot.read().unwrap().values, [7., 6.]);
                } else {
                    drop(snapshot);
                    drop(pending);
                }
            }
        }
    }

    #[test]
    fn shared_timestamp_device_cannot_open_rank_profiling_scopes() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS").is_none() {
            return;
        }
        let runtime = WgpuRuntime::request_profiled_headless_blocking("profile.shared").unwrap();
        let mut rank =
            ResidentRank::new(runtime, Plan::try_new(Kind::TopK, 1, 8, 2, 4).unwrap()).unwrap();
        assert!(rank.timestamp_queries_enabled());
        rank.upload(&[0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();
        for already_dispatched in [false, true] {
            if already_dispatched {
                rank.dispatch(1).unwrap();
            }
            assert!(matches!(
                rank.dispatch_profiled(1),
                Err(ResidentRankError::ProfileRequiresPrivateDevice)
            ));
            assert_eq!(rank.output_is_current(), already_dispatched);
            assert_eq!(rank.generation(), 1);
        }
        assert_eq!(rank.snapshot().unwrap().read().unwrap().values, [7., 6.]);
    }

    #[test]
    fn private_profile_scopes_cannot_capture_ordinary_workspace_errors() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS").is_none() {
            return;
        }
        let plan = Plan::try_new(Kind::TopK, 1, 8, 2, 4).unwrap();
        let mut profile = ResidentRank::request_profiled_blocking(plan).unwrap();
        let ordinary_runtime =
            WgpuRuntime::request_profiled_headless_blocking("profile.peer").unwrap();
        assert!(!profile
            .runtime
            .context()
            .shares_handles_with(ordinary_runtime.context()));
        let scopes = TimestampErrorScopes::try_new(profile.runtime.context().clone()).unwrap();
        let worker = std::thread::spawn(move || {
            let context = ordinary_runtime.context().clone();
            let mut ordinary = ResidentRank::new(ordinary_runtime, plan).unwrap();
            ordinary.upload(&[0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();
            ordinary.dispatch(1).unwrap();
            assert_eq!(
                ordinary.snapshot().unwrap().read().unwrap().values,
                [7., 6.]
            );
            let own_scopes = TimestampErrorScopes::try_new(context.clone()).unwrap();
            let _invalid = context
                .device()
                .create_query_set(&wgpu::QuerySetDescriptor {
                    label: None,
                    ty: wgpu::QueryType::Timestamp,
                    count: wgpu::QUERY_SET_MAX_QUERIES + 1,
                });
            assert!(pollster::block_on(own_scopes.finish().check()).is_err());
        });
        worker.join().unwrap();
        assert!(pollster::block_on(scopes.finish().check()).is_ok());
        profile.upload(&[0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();
        profile.dispatch_profiled(1).unwrap().read().unwrap();
        assert_eq!(profile.snapshot().unwrap().read().unwrap().values, [7., 6.]);
    }

    #[test]
    fn profile_output_is_published_only_by_the_latest_successful_read() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS").is_none() {
            return;
        }
        let mut rank =
            ResidentRank::request_profiled_blocking(Plan::try_new(Kind::TopK, 1, 8, 2, 4).unwrap())
                .unwrap();
        rank.upload(&[0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();
        let first = rank.dispatch_profiled(1).unwrap();
        assert!(!rank.output_is_current());
        assert!(matches!(
            rank.snapshot(),
            Err(ResidentRankError::StaleOutput)
        ));
        let latest = rank.dispatch_profiled(1).unwrap();
        first.read().unwrap();
        assert!(!rank.output_is_current());
        latest.read().unwrap();
        assert!(rank.output_is_current());
        assert_eq!(rank.snapshot().unwrap().read().unwrap().values, [7., 6.]);
        let old_generation = rank.dispatch_profiled(1).unwrap();
        rank.upload(&[-1.; 8]).unwrap();
        old_generation.read().unwrap();
        assert!(!rank.output_is_current());
        drop(rank.dispatch_profiled(1).unwrap());
        assert!(!rank.output_is_current());
        rank.dispatch(1).unwrap();
        assert!(rank.output_is_current());
        assert_eq!(rank.snapshot().unwrap().read().unwrap().values, [-1., -1.]);
    }

    #[test]
    fn failed_profile_validation_cannot_publish_output_or_invalidate_later_dispatch() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS").is_none() {
            return;
        }
        let mut rank =
            ResidentRank::request_profiled_blocking(Plan::try_new(Kind::TopK, 1, 8, 2, 4).unwrap())
                .unwrap();
        let context = rank.runtime.context().clone();
        rank.upload(&[0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();
        for later_dispatch in [false, true] {
            let mut pending = rank.dispatch_profiled(1).unwrap();
            let scope = TimestampErrorScopes::try_new(context.clone()).unwrap();
            let _invalid = context
                .device()
                .create_query_set(&wgpu::QuerySetDescriptor {
                    label: None,
                    ty: wgpu::QueryType::Timestamp,
                    count: wgpu::QUERY_SET_MAX_QUERIES + 1,
                });
            pending.readback.validate(scope.finish());
            assert!(!rank.output_is_current());
            assert!(matches!(
                rank.snapshot(),
                Err(ResidentRankError::StaleOutput)
            ));
            if later_dispatch {
                rank.dispatch(1).unwrap();
            }
            assert!(pending.read().is_err());
            assert_eq!(rank.output_is_current(), later_dispatch);
        }
        assert_eq!(rank.snapshot().unwrap().read().unwrap().values, [7., 6.]);
    }

    #[test]
    fn internal_profile_scope_contention_preserves_state() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS").is_none() {
            return;
        }
        let mut rank =
            ResidentRank::request_profiled_blocking(Plan::try_new(Kind::TopK, 1, 8, 2, 4).unwrap())
                .unwrap();
        // Internal test access only: production callers cannot obtain this context.
        let context = rank.runtime.context().clone();
        rank.upload(&[0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();
        rank.dispatch(1).unwrap();
        let scope = TimestampErrorScopes::try_new(context).unwrap();
        assert!(matches!(
            rank.dispatch_profiled(1),
            Err(ResidentRankError::Runtime(
                WgpuRuntimeError::TimestampProfilingBusy
            ))
        ));
        assert!(rank.output_is_current());
        assert_eq!(rank.snapshot().unwrap().read().unwrap().values, [7., 6.]);
        rank.upload(&[-1.; 8]).unwrap();
        assert!(matches!(
            rank.dispatch_profiled(1),
            Err(ResidentRankError::Runtime(
                WgpuRuntimeError::TimestampProfilingBusy
            ))
        ));
        assert!(!rank.output_is_current());
        assert_eq!(rank.generation(), 2);
        rank.dispatch(1).unwrap();
        assert_eq!(rank.snapshot().unwrap().read().unwrap().values, [-1., -1.]);
        drop(scope);
        assert_eq!(
            rank.dispatch_profiled(1)
                .unwrap()
                .read()
                .unwrap()
                .generation(),
            2
        );
    }

    #[test]
    fn default_runtime_rejects_profiling_without_mutating_output() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let runtime =
            pollster::block_on(WgpuRuntime::request_headless("profile.disabled")).unwrap();
        let mut rank =
            ResidentRank::new(runtime, Plan::try_new(Kind::MidK, 1, 8, 2, 4).unwrap()).unwrap();
        assert!(!rank.timestamp_queries_enabled());
        rank.upload(&[0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();
        assert!(matches!(
            rank.dispatch_profiled(1),
            Err(ResidentRankError::Runtime(
                WgpuRuntimeError::TimestampQueriesUnavailable
            ))
        ));
        assert!(!rank.output_is_current());
        rank.dispatch(1).unwrap();
        assert_eq!(rank.snapshot().unwrap().read().unwrap().values, [3., 4.]);
    }

    #[test]
    fn timestamp_profiles_own_storage_and_survive_workspace_reuse_and_drop() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS").is_none() {
            return;
        }
        let plan = Plan::try_new(Kind::MidK, 1, 8193, 65, 256).unwrap();
        let mut rank = ResidentRank::request_profiled_blocking(plan).unwrap();
        assert!(rank.timestamp_queries_enabled());
        assert!(matches!(
            rank.dispatch_profiled(1),
            Err(ResidentRankError::MissingInput)
        ));
        rank.upload(&(0..8193).map(|v| v as f32).collect::<Vec<_>>())
            .unwrap();
        for invalid in [0, MAX_REPETITIONS + 1] {
            assert!(matches!(
                rank.dispatch_profiled(invalid),
                Err(ResidentRankError::InvalidRepetitions)
            ));
        }
        assert!(!rank.output_is_current());
        let first = rank.dispatch_profiled(1).unwrap();
        let second = rank.dispatch_profiled(16).unwrap();
        assert!(!rank.output_is_current());
        // A later ordinary dispatch can publish output without consuming either
        // owned profile; their eventual reads must not own this freshness state.
        rank.dispatch(1).unwrap();
        let snapshot = rank.snapshot().unwrap();
        rank.upload(&vec![-3.; 8193]).unwrap();
        rank.dispatch(1).unwrap();
        drop(rank.dispatch_profiled(1).unwrap());
        drop(rank);
        let output = snapshot.read().unwrap();
        assert_eq!(output.indices, (4064..4129).collect::<Vec<_>>());
        assert_eq!(
            output.values,
            (4064..4129).map(|v| v as f32).collect::<Vec<_>>()
        );
        for (pending, repetitions) in [(second, 16), (first, 1)] {
            let profile = pending.read().unwrap();
            assert_eq!(profile.plan(), plan);
            assert_eq!(profile.generation(), 1);
            assert_eq!(profile.repetitions(), repetitions);
            assert_eq!(
                profile.timestamps().passes.len(),
                (repetitions * 2) as usize
            );
            assert!(profile
                .timestamps()
                .passes
                .iter()
                .any(|p| p.elapsed_ns > 0.0));
            let report = profile.report();
            assert_eq!(report["generation"], "1");
            assert_eq!(
                report["merge_entry_point"],
                "rankk_exact_2ce_midk_tournament"
            );
            assert!(report["passes"][0]["start_tick"].is_string());
        }
    }

    #[test]
    fn maximum_profile_query_budget_is_executable() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS").is_none() {
            return;
        }
        let mut rank =
            ResidentRank::request_profiled_blocking(Plan::try_new(Kind::TopK, 1, 8, 2, 4).unwrap())
                .unwrap();
        rank.upload(&[0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();
        let profile = rank
            .dispatch_profiled(MAX_REPETITIONS)
            .unwrap()
            .read()
            .unwrap();
        assert_eq!(
            profile.timestamps().passes.len(),
            MAX_REPETITIONS as usize * 2
        );
        assert_eq!(rank.snapshot().unwrap().read().unwrap().values, [7., 6.]);
        assert_eq!(profile.report()["compute_submissions"], 4);
        assert_eq!(
            profile.report()["host_paced_chunks"],
            rank.adapter_info().backend == wgpu::Backend::Metal
        );
        assert!(profile
            .timestamps()
            .passes
            .iter()
            .all(|p| p.end_tick >= p.start_tick));
    }

    #[test]
    fn maximum_ordinary_repetitions_reuse_scratch_in_one_pass() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let runtime =
            pollster::block_on(WgpuRuntime::request_headless("rank.single_pass.budget")).unwrap();
        for kind in [Kind::TopK, Kind::MidK, Kind::BottomK] {
            let plan = Plan::try_new(kind, 2, 65, 7, 32).unwrap();
            let mut rank = ResidentRank::new(runtime.clone(), plan).unwrap();
            let input: Vec<_> = (0..130)
                .map(|i| {
                    if i % 13 == 0 {
                        f32::NAN
                    } else {
                        (i % 7) as f32
                    }
                })
                .collect();
            let expected = crate::rankk_exact_2ce::tests::cpu_reference(kind, 2, 65, 7, &input);
            rank.upload(&input).unwrap();
            rank.dispatch(MAX_REPETITIONS).unwrap();
            let output = rank.snapshot().unwrap().read().unwrap();
            assert_eq!(output.indices, expected.indices);
            assert_eq!(output.values, expected.values);
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn map_bytes(
    context: WgpuContext,
    staging: runtime::ReadbackLease,
) -> Result<Vec<u8>, ResidentRankError> {
    Ok(staging
        .read_async(context, "resident.rank.snapshot")
        .await?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::rankk_exact_2ce::{tests::cpu_reference, Kind};

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn midk_merge_destinations_and_padding_remain_disjoint_when_enabled() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let runtime =
            pollster::block_on(WgpuRuntime::request_headless("midk.tile.ownership")).unwrap();
        for (cols, tile) in [
            (1024, 32),
            (1025, 32),
            (1025, 256),
            (2049, 1025),
            (2047, 8),
            (2055, 8),
        ] {
            for k in [1, 7, cols] {
                let plan = Plan::try_new(Kind::MidK, 3, cols, k, tile).unwrap();
                let mut rank = ResidentRank::new(runtime.clone(), plan).unwrap();
                for iteration in 0..4 {
                    let mut input = vec![f32::NAN; 3 * cols as usize];
                    if iteration % 2 == 0 {
                        input[..cols as usize].fill(1.0);
                        input[cols as usize] = -0.0;
                        input[cols as usize * 2 - 2] = 0.0;
                        input[cols as usize * 2 - 1] = -3.0;
                    }
                    let expected = cpu_reference(Kind::MidK, 3, cols, k, &input);
                    rank.upload(&input).unwrap();
                    rank.dispatch(3).unwrap();
                    let result = rank.snapshot().unwrap().read().unwrap();
                    assert_eq!(
                        result.indices, expected.indices,
                        "cols={cols} tile={tile} k={k}"
                    );
                    for (actual, expected) in result.values.iter().zip(expected.values) {
                        assert!(
                            (actual.is_nan() && expected.is_nan())
                                || actual.to_bits() == expected.to_bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn combined_readback_preserves_float_bits_and_signed_indices() {
        let words = [(-0.0f32).to_bits(), f32::NAN.to_bits(), 7u32, u32::MAX];
        let output = decode(bytemuck::cast_slice(&words));
        assert_eq!(output.values[0].to_bits(), (-0.0f32).to_bits());
        assert!(output.values[1].is_nan());
        assert_eq!(output.indices, [7, -1]);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn fragmented_midk_seek_matches_total_order_when_enabled() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let runtime = pollster::block_on(WgpuRuntime::request_headless("midk.prefix.seek"))
            .expect("requested runtime test requires WGPU");
        let extremes = [
            -f32::MAX,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for (cols, tile) in [(257, 1), (1025, 8), (1025, 32), (4097, 128), (8193, 256)] {
            for k in [1, 7, cols - 126, cols - 128, cols] {
                let plan = Plan::try_new(Kind::MidK, 3, cols, k, tile).unwrap();
                let mut rank = ResidentRank::new(runtime.clone(), plan).unwrap();
                for pattern in 0..4 {
                    let input = (0..3 * cols)
                        .map(|i| {
                            let column = i % cols;
                            match pattern {
                                0 => ((column * 73 % 127) as f32 - 63.) / 8.,
                                1 => {
                                    if column % 3 == 0 {
                                        -0.0
                                    } else {
                                        0.0
                                    }
                                }
                                2 => extremes[(i as usize * 37) % extremes.len()],
                                _ => f32::NAN,
                            }
                        })
                        .collect::<Vec<_>>();
                    let expected = cpu_reference(Kind::MidK, 3, cols, k, &input);
                    rank.upload(&input).unwrap();
                    rank.dispatch(2).unwrap();
                    let result = rank.snapshot().unwrap().read().unwrap();
                    assert_eq!(
                        result.indices, expected.indices,
                        "cols={cols} tile={tile} k={k} pattern={pattern}"
                    );
                    for (a, b) in result.values.iter().zip(expected.values) {
                        assert!((a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn resident_rank_lifecycle_matches_reference_when_enabled() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let runtime = pollster::block_on(WgpuRuntime::request_headless("resident.rank.test"))
            .expect("requested runtime test requires a WGPU adapter");
        for kind in [Kind::TopK, Kind::MidK, Kind::BottomK] {
            for tile in [8, 70, 128, 257, 1024] {
                let plan = Plan::try_new(kind, 2, 513, 7, tile).unwrap();
                let mut input = (0..1026)
                    .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
                    .collect::<Vec<_>>();
                input[0] = -0.0;
                input[1] = 0.0;
                input[2] = f32::INFINITY;
                input[513..].fill(f32::NAN);
                input[514] = 1.0;
                let expected = cpu_reference(kind, 2, 513, 7, &input);
                let mut workspace = ResidentRank::new(runtime.clone(), plan).unwrap();
                assert!(matches!(
                    workspace.dispatch(1),
                    Err(ResidentRankError::MissingInput)
                ));
                assert!(matches!(
                    workspace.snapshot(),
                    Err(ResidentRankError::StaleOutput)
                ));
                workspace.upload(&input).unwrap();
                assert!(matches!(
                    workspace.dispatch(0),
                    Err(ResidentRankError::InvalidRepetitions)
                ));
                assert!(matches!(
                    workspace.dispatch(MAX_REPETITIONS + 1),
                    Err(ResidentRankError::InvalidRepetitions)
                ));
                workspace.dispatch(3).unwrap();
                let snapshot = workspace.snapshot().unwrap();
                assert_eq!(snapshot.generation(), 1);
                assert!(workspace.upload(&[]).is_err());
                assert!(workspace.output_is_current());
                workspace.upload(&vec![99.0; 1026]).unwrap();
                assert!(!workspace.output_is_current());
                assert!(matches!(
                    workspace.snapshot(),
                    Err(ResidentRankError::StaleOutput)
                ));
                workspace.dispatch(2).unwrap();
                drop(workspace);
                let actual = snapshot.read().unwrap();
                assert_eq!(actual.indices, expected.indices, "{kind:?} tile={tile}");
                for (a, b) in actual.values.iter().zip(&expected.values) {
                    assert!((a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits());
                }
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn matmul_input_copy_is_owned_and_validation_is_transactional_when_enabled() {
        use crate::resident_matmul::MatmulShape;
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let runtime = pollster::block_on(WgpuRuntime::request_headless("rank.matmul.copy.test"))
            .expect("requested runtime test requires WGPU");
        eprintln!("matmul-rank adapter: {:?}", runtime.adapter_info());
        let values = [
            3., 1., 3., -2., 7., 4., 0., 1., -1., 8., 0., 2., 2., 3., 9., -2.,
        ];
        for kind in [Kind::TopK, Kind::MidK, Kind::BottomK] {
            let mut rank =
                ResidentRank::new(runtime.clone(), Plan::try_new(kind, 2, 8, 3, 8).unwrap())
                    .unwrap();
            rank.upload(&[0.; 16]).unwrap();
            rank.dispatch(1).unwrap();
            let initial_snapshot = rank.snapshot().unwrap();
            let mut source =
                ResidentMatmul::new(runtime.clone(), MatmulShape::new(2, 2, 8).unwrap()).unwrap();
            assert!(matches!(
                rank.set_input_from_matmul(&source),
                Err(ResidentRankError::StaleSource)
            ));
            // Equal byte counts alone must not authorize a differently shaped source.
            let mut wrong =
                ResidentMatmul::new(runtime.clone(), MatmulShape::new(1, 1, 16).unwrap()).unwrap();
            wrong.upload(&[1.], &[5.; 16]).unwrap();
            wrong.dispatch(1).unwrap();
            assert!(matches!(
                rank.set_input_from_matmul(&wrong),
                Err(ResidentRankError::IncompatibleSource)
            ));
            assert_eq!(rank.generation(), 1);
            assert!(rank.output_is_current());

            source.upload(&[1., 0., 0., 1.], &values).unwrap();
            source.dispatch(1).unwrap();
            rank.set_input_from_matmul(&source).unwrap();
            assert_eq!(rank.generation(), 2);
            assert!(!rank.output_is_current());
            source.upload_rhs(&[99.; 16]).unwrap();
            assert!(matches!(
                rank.set_input_from_matmul(&source),
                Err(ResidentRankError::StaleSource)
            ));
            assert_eq!(rank.generation(), 2);
            source.dispatch(1).unwrap();
            drop(source);
            rank.dispatch(2).unwrap();
            let actual = rank.snapshot().unwrap().read().unwrap();
            assert_eq!(actual, cpu_reference(kind, 2, 8, 3, &values));
            assert_eq!(
                initial_snapshot.read().unwrap(),
                cpu_reference(kind, 2, 8, 3, &[0.; 16])
            );

            let mut source =
                ResidentMatmul::new(runtime.clone(), MatmulShape::new(2, 2, 8).unwrap()).unwrap();
            source.upload(&[1., 0., 0., 1.], &values).unwrap();
            source.dispatch(1).unwrap();
            rank.generation = u64::MAX;
            rank.output_generation = Some(u64::MAX);
            assert!(matches!(
                rank.set_input_from_matmul(&source),
                Err(ResidentRankError::GenerationOverflow)
            ));
            assert_eq!(rank.generation(), u64::MAX);
            assert!(rank.output_is_current());
            assert_eq!(rank.snapshot().unwrap().read().unwrap(), actual);
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn snapshot_copy_stages_match_output_when_enabled() {
        use std::time::{Duration, Instant};
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let runtime =
            pollster::block_on(WgpuRuntime::request_headless("rank.snapshot.stages")).unwrap();
        let context = runtime.context();
        let mut rank = ResidentRank::new(
            runtime.clone(),
            Plan::try_new(Kind::TopK, 2, 257, 7, 256).unwrap(),
        )
        .unwrap();
        let values = (0..514).map(|i| (i % 31) as f32).collect::<Vec<_>>();
        rank.upload(&values).unwrap();
        rank.dispatch(1).unwrap();
        let expected = cpu_reference(Kind::TopK, 2, 257, 7, &values);
        for anchored in [false, true] {
            let _anchor = anchored.then(|| {
                runtime::empty_buffer::<u8>(
                    context.device(),
                    "snapshot.anchor",
                    112,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                )
                .unwrap()
            });
            for _ in 0..14 {
                rank.synchronize().unwrap();
                let t0 = Instant::now();
                let staging = runtime::empty_buffer::<u64>(
                    context.device(),
                    "resident.rank.snapshot",
                    rank.plan.output_elements() as usize,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                )
                .unwrap();
                let t1 = Instant::now();
                let mut encoder = context.device().create_command_encoder(&Default::default());
                let t2 = Instant::now();
                let bytes = u64::from(rank.plan.output_elements()) * 4;
                encoder.copy_buffer_to_buffer(&rank.values, 0, &staging, 0, bytes);
                let t3 = Instant::now();
                encoder.copy_buffer_to_buffer(&rank.indices, 0, &staging, bytes, bytes);
                let t4 = Instant::now();
                let commands = encoder.finish();
                let t5 = Instant::now();
                context.queue().submit(Some(commands));
                let t6 = Instant::now();
                let mapped = runtime::map_read_bytes_with_timeout(
                    context.device(),
                    &staging,
                    0..staging.size(),
                    Duration::from_secs(30),
                    "snapshot.stage.test",
                )
                .unwrap();
                let t7 = Instant::now();
                assert_eq!(decode(&mapped), expected);
                eprintln!("snapshot stages anchored={anchored} us allocate={:.3} encoder={:.3} values_copy={:.3} indices_copy={:.3} finish={:.3} submit={:.3} map={:.3}",
                (t1-t0).as_secs_f64()*1e6, (t2-t1).as_secs_f64()*1e6,
                (t3-t2).as_secs_f64()*1e6, (t4-t3).as_secs_f64()*1e6,
                (t5-t4).as_secs_f64()*1e6, (t6-t5).as_secs_f64()*1e6, (t7-t6).as_secs_f64()*1e6);
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn matmul_chain_is_owned_and_transactional_when_enabled() {
        use crate::resident_matmul::MatmulShape;
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let runtime = pollster::block_on(WgpuRuntime::request_headless("rank.matmul.chain.test"))
            .expect("requested runtime test requires WGPU");
        eprintln!("matmul-rank chain adapter: {:?}", runtime.adapter_info());
        let values = [
            3., 1., 3., -2., 7., 4., 0., 1., -1., 8., 0., 2., 2., 3., 9., -2.,
        ];
        for kind in [Kind::TopK, Kind::MidK, Kind::BottomK] {
            let mut rank =
                ResidentRank::new(runtime.clone(), Plan::try_new(kind, 2, 8, 3, 8).unwrap())
                    .unwrap();
            let mut source =
                ResidentMatmul::new(runtime.clone(), MatmulShape::new(2, 2, 8).unwrap()).unwrap();
            for partial_rhs in [false, true] {
                if partial_rhs {
                    source.upload_rhs(&values).unwrap();
                }
                assert!(matches!(
                    rank.dispatch_from_matmul(&mut source, 1),
                    Err(ResidentRankError::Matmul(MatmulError::MissingInputs))
                ));
                assert!(!source.output_is_current());
                assert!(!rank.output_is_current());
                assert_eq!(rank.generation(), 0);
            }
            source.upload(&[1., 0., 0., 1.], &values).unwrap();
            let source_generation = source.generation();
            for repetitions in [0, MAX_REPETITIONS + 1] {
                assert!(matches!(
                    rank.dispatch_from_matmul(&mut source, repetitions),
                    Err(ResidentRankError::InvalidRepetitions)
                ));
                assert_eq!(source.generation(), source_generation);
                assert!(!source.output_is_current());
                assert_eq!(rank.generation(), 0);
            }
            let mut wrong =
                ResidentMatmul::new(runtime.clone(), MatmulShape::new(1, 1, 16).unwrap()).unwrap();
            wrong.upload(&[1.], &[5.; 16]).unwrap();
            assert!(matches!(
                rank.dispatch_from_matmul(&mut wrong, 1),
                Err(ResidentRankError::IncompatibleSource)
            ));
            assert!(!wrong.output_is_current());
            assert_eq!(rank.generation(), 0);

            assert_eq!(rank.dispatch_from_matmul(&mut source, 3).unwrap(), 1);
            assert_eq!(source.generation(), source_generation);
            assert!(source.output_is_current() && rank.output_is_current());
            assert_eq!(source.snapshot().unwrap().read().unwrap(), values);
            let first = rank.snapshot().unwrap();
            source.upload_rhs(&[99.; 16]).unwrap();
            assert!(rank.output_is_current());

            rank.generation = u64::MAX;
            rank.output_generation = Some(u64::MAX);
            assert!(matches!(
                rank.dispatch_from_matmul(&mut source, 1),
                Err(ResidentRankError::GenerationOverflow)
            ));
            assert!(!source.output_is_current());
            assert_eq!(rank.generation(), u64::MAX);
            assert!(rank.output_is_current());
            assert_eq!(
                rank.snapshot().unwrap().read().unwrap(),
                cpu_reference(kind, 2, 8, 3, &values)
            );
            rank.generation = 1;
            rank.output_generation = Some(1);

            assert_eq!(rank.dispatch_from_matmul(&mut source, 16).unwrap(), 2);
            assert!(source.output_is_current() && rank.output_is_current());
            let second = rank.snapshot().unwrap();
            source.upload_rhs(&[-9.; 16]).unwrap();
            source.dispatch(1).unwrap();
            drop(source);
            rank.dispatch(1).unwrap();
            assert_eq!(
                rank.snapshot().unwrap().read().unwrap(),
                cpu_reference(kind, 2, 8, 3, &[99.; 16])
            );
            drop(rank);
            assert_eq!(first.generation(), 1);
            assert_eq!(first.read().unwrap(), cpu_reference(kind, 2, 8, 3, &values));
            assert_eq!(second.generation(), 2);
            assert_eq!(
                second.read().unwrap(),
                cpu_reference(kind, 2, 8, 3, &[99.; 16])
            );
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn matmul_copy_rejects_different_device_handles_when_enabled() {
        use crate::resident_matmul::MatmulShape;
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let runtime =
            pollster::block_on(WgpuRuntime::request_headless("rank.copy.destination")).unwrap();
        let foreign =
            pollster::block_on(WgpuRuntime::request_headless("rank.copy.source")).unwrap();
        assert!(!runtime.context().shares_handles_with(foreign.context()));
        let mut source = ResidentMatmul::new(foreign, MatmulShape::new(1, 1, 4).unwrap()).unwrap();
        source.upload(&[1.], &[99.; 4]).unwrap();
        source.dispatch(1).unwrap();
        let mut rank =
            ResidentRank::new(runtime, Plan::try_new(Kind::TopK, 1, 4, 2, 4).unwrap()).unwrap();
        rank.upload(&[1., 5., 3., 2.]).unwrap();
        rank.dispatch(1).unwrap();
        assert!(matches!(
            rank.set_input_from_matmul(&source),
            Err(ResidentRankError::IncompatibleSource)
        ));
        source.upload_rhs(&[-9.; 4]).unwrap();
        assert!(matches!(
            rank.dispatch_from_matmul(&mut source, 2),
            Err(ResidentRankError::IncompatibleSource)
        ));
        assert!(!source.output_is_current());
        assert_eq!(rank.generation(), 1);
        assert!(rank.output_is_current());
        assert_eq!(rank.snapshot().unwrap().read().unwrap().values, [5., 3.]);
    }
}
