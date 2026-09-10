//! Private-device profiling of the same scheduled training step, not a second optimizer.
use super::*;
use crate::runtime::timestamps::{
    PassTimestampRecorder, PassTimestamps, TimestampErrorScopes, TimestampReadback,
};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug)]
struct ProfilePass {
    phase: &'static str,
    node: Option<usize>,
    part: usize,
    dispatches: usize,
    operand: Option<usize>,
}

impl ResidentGraphTraining {
    fn profile_passes(&self) -> Vec<ProfilePass> {
        let backend = self.adapter_info().backend;
        let mut passes = Vec::new();
        let width = forward_dispatches_per_pass(backend, self.nodes.len());
        for (part, nodes) in self.nodes.chunks(width).enumerate() {
            passes.push(ProfilePass {
                phase: if nodes.len() > 1 {
                    "forward_mixed"
                } else {
                    match nodes[0] {
                        Node::Linear { .. } => "forward_dense",
                        Node::Pointwise { .. } => "forward_pointwise",
                    }
                },
                node: (nodes.len() == 1).then_some(part * width),
                part,
                dispatches: nodes.len(),
                operand: None,
            });
        }
        let append = |passes: &mut Vec<ProfilePass>, phase, node, scheduled: &[Pass]| {
            for (part, chunk) in scheduled
                .chunks(dispatches_per_pass(backend, scheduled.len()))
                .enumerate()
            {
                passes.push(ProfilePass {
                    phase,
                    node,
                    part,
                    dispatches: chunk.len(),
                    operand: None,
                });
            }
        };
        append(&mut passes, "loss", None, &self.loss_passes);
        for (node, stage) in self.nodes.iter().enumerate().rev() {
            match stage {
                Node::Linear { backward, .. } => {
                    append(&mut passes, "dense_backward", Some(node), backward)
                }
                Node::Pointwise {
                    plan, workspace, ..
                } => {
                    for (operand, part) in plan.profile_passes(workspace) {
                        passes.push(ProfilePass {
                            phase: if operand.is_some() {
                                "pointwise_unbroadcast"
                            } else {
                                "pointwise_vjp"
                            },
                            node: Some(node),
                            part,
                            dispatches: 1,
                            operand,
                        });
                    }
                }
            }
        }
        append(&mut passes, "update", None, &self.update_passes);
        passes
    }
}

/// Timestamp-enabled runtime handles never escape this workspace. An unread,
/// cancelled or invalid profile blocks reuse instead of permitting an implicit retry.
pub struct ProfiledGraphTraining {
    inner: ResidentGraphTraining,
    pending: Option<Shared<AtomicBool>>,
}

impl ProfiledGraphTraining {
    pub async fn request(
        definition: GraphDefinition,
        policy: GraphGradientPolicy,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, TrainingError> {
        let runtime =
            WgpuRuntime::request_profiled_headless("graph.training.private_profile").await?;
        let errors = TimestampErrorScopes::try_new(runtime.context().clone())?;
        let inner =
            ResidentGraphTraining::new(runtime, definition, policy, tile, kernel, accumulation);
        errors.finish().check().await?;
        Ok(Self {
            inner: inner?,
            pending: None,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn request_blocking(
        definition: GraphDefinition,
        policy: GraphGradientPolicy,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, TrainingError> {
        pollster::block_on(Self::request(
            definition,
            policy,
            tile,
            kernel,
            accumulation,
        ))
    }

    fn ready(&self) -> Result<(), TrainingError> {
        if self
            .pending
            .as_ref()
            .is_some_and(|p| !p.load(Ordering::Acquire))
        {
            Err(TrainingError::PendingProfile)
        } else {
            Ok(())
        }
    }

    pub fn input_layout(&self) -> &NdLayout {
        self.inner.input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.inner.output_layout()
    }
    pub fn gradient_policy(&self) -> GraphGradientPolicy {
        self.inner.gradient_policy()
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.inner.adapter_info()
    }
    pub fn submitted_steps(&self) -> u64 {
        self.inner.submitted_steps()
    }
    pub fn batch_generation(&self) -> u64 {
        self.inner.batch_generation()
    }

    pub fn upload_batch(&mut self, input: &[f32], target: &[f32]) -> Result<(), TrainingError> {
        self.ready()?;
        self.inner.upload_batch(input, target)
    }

    /// Uninstrumented control on the same private timestamp-capable device.
    /// This remains an enqueue operation; read a guarded snapshot for acceptance.
    pub fn step(&mut self, rate: f32) -> Result<u64, TrainingError> {
        self.ready()?;
        self.inner.step(rate)
    }

    pub fn loss_snapshot(&self) -> Result<StepReadback, TrainingError> {
        self.ready()?;
        self.inner.loss_snapshot()
    }
    pub fn state_snapshot(&self) -> Result<GraphStateReadback, TrainingError> {
        self.ready()?;
        self.inner.state_snapshot()
    }
    pub fn parameter_snapshot(&self) -> Result<GraphParameterReadback, TrainingError> {
        self.ready()?;
        self.inner.parameter_snapshot()
    }

    /// Instrument existing passes, without splitting, fusing or removing work.
    /// Returned times exclude query resolution/readback and do not prove acceptance
    /// until read() validates both the GPU scopes and this attempt's loss/flags.
    pub fn step_profiled(&mut self, rate: f32) -> Result<GraphProfileReadback, TrainingError> {
        self.ready()?;
        let attempt = self.inner.next_attempt(rate)?;
        let passes = self.inner.profile_passes();
        let context = self.inner.device.runtime().context();
        let errors = TimestampErrorScopes::try_new(context.clone())?;
        let recorder = PassTimestampRecorder::new(
            context.clone(),
            u32::try_from(passes.len()).map_err(|_| TrainingError::Overflow)?,
        )?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let mut cursor = PassTimestampCursor::new(&recorder);
        self.inner.encode_step(&mut encoder, &mut cursor);
        cursor.finish();
        let loss = StepReadback {
            raw: readback::capture_into(
                context,
                &self.inner.loss_pool,
                &[&self.inner.loss, &self.inner.validation],
                &mut encoder,
            )?,
            stages: self.inner.stage_count() + 2,
            step: attempt,
            batch_generation: self.inner.batch_generation,
        };
        let mut timestamps = recorder.resolve(&mut encoder);
        let checked = Shared::new(AtomicBool::new(false));
        self.pending = Some(checked.clone());
        self.inner.write_rate(rate);
        context.queue().submit(Some(encoder.finish()));
        timestamps.validate(errors.finish());
        let direct_copy_bytes = self
            .inner
            .nodes
            .iter()
            .map(|node| match node {
                Node::Linear { .. } => 0,
                Node::Pointwise { plan, .. } => plan.direct_copy_bytes(),
            })
            .sum();
        self.inner.mark_step(attempt);
        Ok(GraphProfileReadback {
            timestamps,
            loss,
            checked,
            passes,
            step: attempt,
            batch_generation: self.inner.batch_generation,
            direct_copy_bytes,
        })
    }
}

/// Owning result of exactly one attempt; safe after profiler drop.
pub struct GraphProfileReadback {
    timestamps: TimestampReadback,
    loss: StepReadback,
    checked: Shared<AtomicBool>,
    passes: Vec<ProfilePass>,
    step: u64,
    batch_generation: u64,
    direct_copy_bytes: u64,
}

fn release_after_guard(checked: &AtomicBool, loss: &Result<f32, TrainingError>) {
    // A decoded numerical rejection proves rollback and permits an explicit new
    // attempt. Mapping/validation failures leave the workspace quarantined.
    if loss.is_ok() || matches!(loss, Err(TrainingError::Rejected { .. })) {
        checked.store(true, Ordering::Release);
    }
}

impl GraphProfileReadback {
    pub fn submitted_step(&self) -> u64 {
        self.step
    }
    pub fn batch_generation(&self) -> u64 {
        self.batch_generation
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<GraphGpuProfile, TrainingError> {
        let timestamps = self.timestamps.read()?;
        let loss = self.loss.read();
        release_after_guard(&self.checked, &loss);
        Ok(GraphGpuProfile {
            timestamps,
            passes: self.passes,
            loss: loss?,
            step: self.step,
            batch_generation: self.batch_generation,
            direct_copy_bytes: self.direct_copy_bytes,
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<GraphGpuProfile, TrainingError> {
        let timestamps = self.timestamps.read_async().await?;
        let loss = self.loss.read_async().await;
        release_after_guard(&self.checked, &loss);
        Ok(GraphGpuProfile {
            timestamps,
            passes: self.passes,
            loss: loss?,
            step: self.step,
            batch_generation: self.batch_generation,
            direct_copy_bytes: self.direct_copy_bytes,
        })
    }
}

#[derive(Clone, Debug)]
pub struct GraphGpuProfile {
    timestamps: PassTimestamps,
    passes: Vec<ProfilePass>,
    loss: f32,
    step: u64,
    batch_generation: u64,
    direct_copy_bytes: u64,
}

impl GraphGpuProfile {
    pub fn loss(&self) -> f32 {
        self.loss
    }
    pub fn submitted_step(&self) -> u64 {
        self.step
    }
    pub fn batch_generation(&self) -> u64 {
        self.batch_generation
    }
    pub fn timestamps(&self) -> &PassTimestamps {
        &self.timestamps
    }

    /// Rust-owned client schema. Absolute ticks/counters are strings, not JS f64.
    pub fn report(&self) -> serde_json::Value {
        let mut totals = std::collections::BTreeMap::<&str, f64>::new();
        let mut zero_intervals = 0;
        let passes: Vec<_> = self.passes.iter().zip(&self.timestamps.passes).map(|(metadata, time)| {
            *totals.entry(metadata.phase).or_default() += time.elapsed_ns;
            zero_intervals += usize::from(time.elapsed_ns == 0.);
            serde_json::json!({"phase":metadata.phase,"node":metadata.node,"part":metadata.part,
                "dispatches":metadata.dispatches,"operand":metadata.operand,
                "start_tick":time.start_tick.to_string(),"end_tick":time.end_tick.to_string(),"elapsed_ns":time.elapsed_ns})
        }).collect();
        let span = self
            .timestamps
            .passes
            .first()
            .zip(self.timestamps.passes.last())
            .and_then(|(first, last)| last.end_tick.checked_sub(first.start_tick))
            .map(|ticks| ticks as f64 * self.timestamps.timestamp_period_ns);
        serde_json::json!({"schema":"spiraltorch.graph_training_gpu_profile.v1","instrumented":true,"accepted":true,
            "boundary":"Diagnostic timestamps at existing compute-pass boundaries on a private device. No pass splitting; forward/dense-backward/update may each contain multiple dispatches. Excludes uploads, CPU encoding, query resolve/readback and untimed copies from compute sums; GPU span includes inter-pass/copy gaps. Instrumentation may perturb execution and browser timestamps may quantize to zero.",
            "submitted_step":self.step.to_string(),"batch_generation":self.batch_generation.to_string(),"loss":self.loss,
            "timestamp_period_ns":self.timestamps.timestamp_period_ns,"zero_intervals":zero_intervals,"gpu_span_ns":span,
            "direct_vjp_copy_bytes":self.direct_copy_bytes.to_string(),"phase_totals_ns":totals,"passes":passes})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::timestamps::PassTimestamp;

    #[test]
    fn report_preserves_integer_ticks_zero_intervals_and_real_pass_groups() {
        let tick = (1u64 << 60) + 1;
        let profile = GraphGpuProfile {
            timestamps: PassTimestamps {
                timestamp_period_ns: 2.,
                passes: vec![
                    PassTimestamp {
                        start_tick: tick,
                        end_tick: tick,
                        elapsed_ns: 0.,
                    },
                    PassTimestamp {
                        start_tick: tick + 5,
                        end_tick: tick + 8,
                        elapsed_ns: 6.,
                    },
                ],
            },
            passes: vec![
                ProfilePass {
                    phase: "forward_mixed",
                    node: None,
                    part: 0,
                    dispatches: 5,
                    operand: None,
                },
                ProfilePass {
                    phase: "update",
                    node: None,
                    part: 0,
                    dispatches: 3,
                    operand: None,
                },
            ],
            loss: 0.5,
            step: u64::MAX,
            batch_generation: tick,
            direct_copy_bytes: 16,
        };
        let report = profile.report();
        assert_eq!(report["zero_intervals"], 1);
        assert_eq!(report["submitted_step"], u64::MAX.to_string());
        assert_eq!(report["batch_generation"], tick.to_string());
        assert_eq!(report["passes"][0]["start_tick"], tick.to_string());
        assert_eq!(report["passes"][0]["dispatches"], 5);
        assert!(report["passes"][0]["node"].is_null());
        assert_eq!(report["phase_totals_ns"]["forward_mixed"], 0.);
        assert_eq!(report["phase_totals_ns"]["update"], 6.);
        assert_eq!(report["gpu_span_ns"], 16.);
    }

    #[test]
    fn only_checked_acceptance_or_rollback_unblocks_a_profile() {
        for (loss, ready) in [
            (Ok(1.), true),
            (Err(TrainingError::Rejected { stage: 2, flags: 1 }), true),
            (Err(TrainingError::InvalidReadback), false),
            (
                Err(TrainingError::Runtime(
                    WgpuRuntimeError::InvalidTimestamps {
                        message: "failure".into(),
                    },
                )),
                false,
            ),
        ] {
            let checked = AtomicBool::new(false);
            release_after_guard(&checked, &loss);
            assert_eq!(checked.load(Ordering::Acquire), ready);
        }
    }
}
