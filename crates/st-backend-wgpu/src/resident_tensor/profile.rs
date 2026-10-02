//! Explicit, bounded convolution-VJP timestamps on the real tensor execution path.
//! No default device replacement, pass splitting, or mapping inside the capture.

use super::*;
use crate::runtime::timestamps::{
    PassTimestampRecorder, PassTimestamps, TimestampErrorScopes, TimestampReadback,
    TimestampValidation,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

const MAX_OPERATIONS: usize = 128;
const MAX_PENDING_CAPTURES: usize = 4;

#[derive(Clone, Debug, PartialEq)]
pub struct ConvolutionVjpGeometry {
    pub kind: &'static str,
    pub input: Vec<usize>,
    pub weights: Vec<usize>,
    pub upstream: Vec<usize>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

pub struct ConvolutionVjpTiming {
    pub geometry: ConvolutionVjpGeometry,
    /// Existing input, weight, and bias passes, in that order. Packing, forward,
    /// other backward kernels, SGD, query resolution and host time are excluded.
    pub timestamps: PassTimestamps,
}

pub struct ConvolutionVjpProfile {
    pub operations: Vec<ConvolutionVjpTiming>,
}

impl ConvolutionVjpProfile {
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": "spiraltorch.convolution_vjp_gpu_profile.v1",
            "scope": "Existing convolution VJP passes only; not whole-step throughput",
            "operations": self.operations.iter().map(|operation| {
                let g = &operation.geometry;
                let passes = ["input_vjp", "weight_vjp", "bias_vjp"].iter()
                    .zip(&operation.timestamps.passes).map(|(phase, time)| serde_json::json!({
                        "phase": phase, "start_tick": time.start_tick.to_string(),
                        "end_tick": time.end_tick.to_string(), "elapsed_ns": time.elapsed_ns,
                    })).collect::<Vec<_>>();
                serde_json::json!({
                    "kind": g.kind, "input": g.input, "weights": g.weights,
                    "upstream": g.upstream, "stride": g.stride,
                    "padding": g.padding, "dilation": g.dilation,
                    "timestamp_period_ns": operation.timestamps.timestamp_period_ns,
                    "passes": passes,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests;

struct CapturedVjp {
    geometry: ConvolutionVjpGeometry,
    guard: Shared<wgpu::Buffer>,
}

struct Capture {
    token: Shared<()>,
    recorder: Option<Shared<PassTimestampRecorder>>,
    #[cfg(not(target_arch = "wasm32"))]
    thread: std::thread::ThreadId,
    pending: usize,
    operations: Vec<CapturedVjp>,
}

#[derive(Default)]
pub(crate) struct ProfileSlot(Mutex<Option<Capture>>, AtomicUsize);

struct CapturePermit(Shared<ProfileSlot>);

impl CapturePermit {
    fn acquire(slot: Shared<ProfileSlot>) -> Result<Self, TensorError> {
        let mut count = slot.1.load(Ordering::Acquire);
        loop {
            if count >= MAX_PENDING_CAPTURES {
                return Err(TensorError::Limit("unread convolution profile budget"));
            }
            match slot.1.compare_exchange_weak(
                count,
                count + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => count = observed,
            }
        }
        Ok(Self(slot))
    }
}

impl Drop for CapturePermit {
    fn drop(&mut self) {
        self.0 .1.fetch_sub(1, Ordering::AcqRel);
    }
}

impl std::fmt::Debug for ProfileSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvolutionProfileSlot")
            .finish_non_exhaustive()
    }
}

pub(crate) struct OperationCapture {
    recorder: Shared<PassTimestampRecorder>,
    operation_index: usize,
    geometry: ConvolutionVjpGeometry,
}

impl OperationCapture {
    pub(crate) fn writes(&self, index: usize) -> wgpu::ComputePassTimestampWrites<'_> {
        self.recorder
            .writes((self.operation_index * 3 + index) as u32)
    }

    pub(crate) fn finish(self, output: &ResidentTensor) {
        let mut slot = output
            .device
            .profile_slot()
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let capture = slot
            .as_mut()
            .expect("synchronous capture ended during encoding");
        capture.pending -= 1;
        capture.operations.push(CapturedVjp {
            geometry: self.geometry,
            guard: output.storage.flags.clone(),
        });
    }
}

impl ProfileSlot {
    pub(crate) fn begin(
        &self,
        geometry: impl FnOnce() -> ConvolutionVjpGeometry,
    ) -> Result<Option<OperationCapture>, TensorError> {
        let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let Some(capture) = slot.as_mut() else {
            return Ok(None);
        };
        #[cfg(not(target_arch = "wasm32"))]
        if capture.thread != std::thread::current().id() {
            return Err(WgpuRuntimeError::TimestampProfilingBusy.into());
        }
        if capture.operations.len() + capture.pending >= MAX_OPERATIONS {
            return Err(TensorError::Limit("convolution profile operation budget"));
        }
        let operation_index = capture.operations.len() + capture.pending;
        let recorder = capture
            .recorder
            .as_ref()
            .ok_or(TensorError::Readback)?
            .clone();
        capture.pending += 1;
        Ok(Some(OperationCapture {
            recorder,
            operation_index,
            geometry: geometry(),
        }))
    }
}

struct CaptureScope<'a> {
    slot: &'a ProfileSlot,
    token: Shared<()>,
}

impl<'a> CaptureScope<'a> {
    fn new(slot: &'a ProfileSlot) -> Result<Self, TensorError> {
        let mut state = slot.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.is_some() {
            return Err(WgpuRuntimeError::TimestampProfilingBusy.into());
        }
        let token = Shared::new(());
        *state = Some(Capture {
            token: token.clone(),
            recorder: None,
            #[cfg(not(target_arch = "wasm32"))]
            thread: std::thread::current().id(),
            pending: 0,
            operations: Vec::new(),
        });
        Ok(Self { slot, token })
    }

    fn take(&self) -> Option<Capture> {
        let mut state = self.slot.0.lock().unwrap_or_else(|e| e.into_inner());
        if state
            .as_ref()
            .is_some_and(|s| Shared::ptr_eq(&s.token, &self.token))
        {
            state.take()
        } else {
            None
        }
    }

    fn finish(self) -> Result<(Vec<CapturedVjp>, PassTimestampRecorder), TensorError> {
        let captured = self.take().ok_or(TensorError::Readback)?;
        if captured.pending != 0 || captured.operations.is_empty() {
            return Err(TensorError::Limit(
                "incomplete or empty convolution profile",
            ));
        }
        let recorder = Shared::try_unwrap(captured.recorder.ok_or(TensorError::Readback)?)
            .map_err(|_| TensorError::Readback)?;
        Ok((captured.operations, recorder))
    }
}

impl Drop for CaptureScope<'_> {
    fn drop(&mut self) {
        self.take();
    }
}

impl TensorDevice {
    pub(crate) fn profile_slot(&self) -> &ProfileSlot {
        &self.runtime().context().tensor_profile
    }

    /// Capture up to 128 VJPs submitted synchronously through this runtime's
    /// cloned context, including TensorDevice wrappers constructed by NN graphs.
    /// A separately constructed WgpuContext is not a clone of this capture scope.
    /// Use a caller-owned timestamp-enabled runtime, not the shared default.
    /// The closure must exclusively own its execution context: unrelated work
    /// and manually managed error scopes must not run on that device meanwhile.
    /// Foreign-thread VJPs and nested captures fail before their VJP dispatch.
    /// At most four owning readbacks may be outstanding on one cloned context.
    /// Errors and unwinding detach the capture; this does not roll back training.
    pub fn profile_convolution_vjps<T, E>(
        &self,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Result<(T, ConvolutionVjpProfileReadback), E>
    where
        E: From<TensorError>,
    {
        let context = self.runtime().context();
        if !context
            .device()
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return Err(TensorError::from(WgpuRuntimeError::TimestampQueriesUnavailable).into());
        }
        let permit = CapturePermit::acquire(context.tensor_profile.clone())?;
        let scope = CaptureScope::new(self.profile_slot())?;
        let errors = TimestampErrorScopes::try_new(context.clone()).map_err(TensorError::from)?;
        // A shared query set avoids exhausting Metal counter-sample buffers.
        let recorder = PassTimestampRecorder::new(context.clone(), (MAX_OPERATIONS * 3) as u32)
            .map_err(TensorError::from)?;
        self.profile_slot()
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
            .expect("capture scope installed")
            .recorder = Some(Shared::new(recorder));
        let result = operation()?;
        let (operations, recorder) = scope.finish()?;
        Ok((
            result,
            ConvolutionVjpProfileReadback {
                context: context.clone(),
                operations,
                recorder,
                validation: errors.finish(),
                _permit: permit,
            },
        ))
    }
}

/// Query samples and immutable VJP guards outlive the device wrapper and capture.
/// Read only after submitting the complete step; capture itself adds no wait,
/// readback, query resolution, or compute-pass boundary to the original schedule.
pub struct ConvolutionVjpProfileReadback {
    context: WgpuContext,
    operations: Vec<CapturedVjp>,
    recorder: PassTimestampRecorder,
    validation: TimestampValidation,
    _permit: CapturePermit,
}

struct ResolvedProfile {
    guards: runtime::ReadbackBatch<u32>,
    geometry: Vec<ConvolutionVjpGeometry>,
    timestamps: TimestampReadback,
    validation: TimestampValidation,
}

fn resolve(
    context: &WgpuContext,
    operations: Vec<CapturedVjp>,
    recorder: PassTimestampRecorder,
) -> Result<ResolvedProfile, TensorError> {
    let errors = TimestampErrorScopes::try_new(context.clone())?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let spans = operations
        .iter()
        .map(|op| (op.guard.as_ref(), 0, 1))
        .collect::<Vec<_>>();
    let guards =
        runtime::ReadbackBatch::encode_spans(context, &spans, "conv.profile.guards", &mut encoder)?;
    let timestamps = recorder.resolve_prefix(&mut encoder, (operations.len() * 3) as u32)?;
    let geometry = operations.into_iter().map(|op| op.geometry).collect();
    context.queue().submit(Some(encoder.finish()));
    Ok(ResolvedProfile {
        guards,
        geometry,
        timestamps,
        validation: errors.finish(),
    })
}

fn validate_guards(guards: &[Vec<u32>], count: usize) -> Result<(), TensorError> {
    if guards.len() != count || guards.iter().any(|g| g.len() != 1) {
        return Err(TensorError::Readback);
    }
    if guards.iter().any(|g| g[0] != 0) {
        return Err(TensorError::NonFinite);
    }
    Ok(())
}

fn assemble(
    geometry: Vec<ConvolutionVjpGeometry>,
    times: PassTimestamps,
) -> Result<ConvolutionVjpProfile, TensorError> {
    if times.passes.len() != geometry.len() * 3 {
        return Err(TensorError::Readback);
    }
    let operations = geometry
        .into_iter()
        .zip(times.passes.as_chunks::<3>().0)
        .map(|(geometry, passes)| ConvolutionVjpTiming {
            geometry,
            timestamps: PassTimestamps {
                timestamp_period_ns: times.timestamp_period_ns,
                passes: passes.to_vec(),
            },
        })
        .collect();
    Ok(ConvolutionVjpProfile { operations })
}

impl ConvolutionVjpProfileReadback {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<ConvolutionVjpProfile, TensorError> {
        // As in graph profiling, Metal needs completion before query resolution;
        // resolving in the last compute command buffer can expose unwritten ticks.
        runtime::submit_with_timeout(
            self.context.device(),
            self.context.queue(),
            std::iter::empty(),
            std::time::Duration::from_secs(30),
            "conv.profile.completion",
        )?;
        pollster::block_on(self.validation.check())?;
        let resolved = resolve(&self.context, self.operations, self.recorder)?;
        pollster::block_on(resolved.validation.check())?;
        validate_guards(&resolved.guards.read()?, resolved.geometry.len())?;
        assemble(resolved.geometry, resolved.timestamps.read()?)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<ConvolutionVjpProfile, TensorError> {
        let (sender, receiver) = futures_channel::oneshot::channel();
        self.context.queue().on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        receiver
            .await
            .map_err(|_| WgpuRuntimeError::SubmissionCallbackDisconnected {
                operation: "conv.profile.completion",
            })?;
        self.validation.check().await?;
        let resolved = resolve(&self.context, self.operations, self.recorder)?;
        resolved.validation.check().await?;
        validate_guards(
            &resolved.guards.read_async().await?,
            resolved.geometry.len(),
        )?;
        assemble(resolved.geometry, resolved.timestamps.read_async().await?)
    }
}
