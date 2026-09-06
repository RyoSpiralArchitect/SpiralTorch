//! Persistent exact rank storage with enqueue-only dispatch and owned snapshots.

use super::{binding, storage_buffer, DispatchError, Output, Pipelines, Plan};
use crate::resident_matmul::{MatmulError, ResidentMatmul};
use crate::runtime::{self, WgpuContext, WgpuRuntime, WgpuRuntimeError};
use thiserror::Error;
use wgpu::util::DeviceExt;

pub const MAX_REPETITIONS: u32 = 1024;

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
    #[error(transparent)]
    Matmul(#[from] MatmulError),
    #[error(transparent)]
    Dispatch(#[from] DispatchError),
    #[error(transparent)]
    Runtime(#[from] WgpuRuntimeError),
}

/// One fixed rank shape/tile. Upload invalidates output; dispatch never reads it.
pub struct ResidentRank {
    runtime: WgpuRuntime,
    plan: Plan,
    pipelines: Pipelines,
    input: wgpu::Buffer,
    values: wgpu::Buffer,
    indices: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    generation: u64,
    output_generation: Option<u64>,
}

impl ResidentRank {
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
        Ok(generation)
    }

    /// Enqueues repetitions as pairs of ordered compute passes in one submission.
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
        Ok(self.generation)
    }

    fn encode_dispatch(&self, encoder: &mut wgpu::CommandEncoder, repetitions: u32) {
        for _ in 0..repetitions {
            // Separate passes provide the storage dependency between sort and merge.
            for (pipeline, x, y) in [
                (
                    &self.pipelines.tile_sort,
                    self.plan.tiles_x(),
                    self.plan.rows(),
                ),
                (&self.pipelines.row_merge, self.plan.rows(), 1),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("resident.rank.pass"),
                    timestamp_writes: None,
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
        let staging = runtime::empty_buffer::<u64>(
            context.device(),
            "resident.rank.snapshot",
            self.plan.output_elements() as usize,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let bytes = u64::from(self.plan.output_elements()) * 4;
        encoder.copy_buffer_to_buffer(&self.values, 0, &staging, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.indices, 0, &staging, bytes, bytes);
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
        Ok(async move { map_bytes(context, staging).await.map(|_| ()) })
    }
}

/// A snapshot remains valid after later uploads, dispatches, or workspace drop.
pub struct RankReadback {
    context: WgpuContext,
    staging: wgpu::Buffer,
    plan: Plan,
    generation: u64,
}

impl RankReadback {
    pub fn plan(&self) -> Plan {
        self.plan
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<Output, ResidentRankError> {
        let bytes = runtime::map_read_bytes_with_timeout(
            self.context.device(),
            &self.staging,
            0..self.staging.size(),
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

#[cfg(target_arch = "wasm32")]
async fn map_bytes(
    context: WgpuContext,
    staging: wgpu::Buffer,
) -> Result<Vec<u8>, ResidentRankError> {
    let slice = staging.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    receiver
        .await
        .map_err(|_| WgpuRuntimeError::MapCallbackDisconnected {
            resource: "resident.rank.snapshot".into(),
        })?
        .map_err(|error| WgpuRuntimeError::Map {
            resource: "resident.rank.snapshot".into(),
            message: error.to_string(),
        })?;
    let mapped = slice.get_mapped_range();
    let bytes = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    drop(context);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::rankk_exact_2ce::{tests::cpu_reference, Kind};

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
