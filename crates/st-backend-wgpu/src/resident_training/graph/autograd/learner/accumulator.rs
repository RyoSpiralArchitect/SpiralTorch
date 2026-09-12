//! Fixed-storage microbatch accumulation at one immutable parameter revision.
use super::*;
use crate::resident_tensor::capture::PreparedCapture;

/// Ordered sum of weighted exact parameter VJPs, without retaining every VJP.
/// Input batches may differ, but all contributions must share the learner and
/// parameter revision. No implicit averaging, clipping or optimizer is applied.
pub struct GraphGradientAccumulator {
    state: ParameterState,
    device: TensorDevice,
    values: Vec<wgpu::Buffer>,
    flags: wgpu::Buffer,
    capture: Option<PreparedCapture>,
    terms: u64,
    last_source: Option<(u64, u64)>,
}

impl GraphGradientAccumulator {
    pub fn len(&self) -> u64 {
        self.terms
    }
    pub fn is_empty(&self) -> bool {
        self.terms == 0
    }
    /// Counts attempted parameter transactions, including numerically rejected ones.
    pub fn parameter_generation(&self) -> u64 {
        self.state.revision
    }
    /// Owning, whole-accumulation-guarded GPU snapshots in stable parameter order.
    /// Snapshot allocation is explicit; ordinary accumulate/reset reuse storage.
    pub fn parameter_gradients(&self) -> Result<Vec<ResidentTensor>, TrainingError> {
        if self.is_empty() {
            return Err(TrainingError::EmptyAccumulator);
        }
        let Some(capture) = &self.capture else {
            return Ok(Vec::new());
        };
        let context = self.device.runtime().context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let outputs = capture.encode(&mut encoder)?;
        context.queue().submit(Some(encoder.finish()));
        Ok(outputs)
    }
}

impl ResidentGraphLearner {
    pub fn gradient_accumulator(&self) -> Result<GraphGradientAccumulator, TrainingError> {
        let g = &self.autograd.graph;
        let gpu = g.device.runtime().context().device();
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let values = g
            .definition
            .parameters()
            .iter()
            .map(|parameter| {
                runtime::empty_buffer::<f32>(
                    gpu,
                    "learner.accumulator",
                    parameter.values.len(),
                    usage,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let flags = runtime::empty_buffer::<u32>(gpu, "learner.accumulator.guard", 1, usage)?;
        let layouts = g
            .definition
            .parameters()
            .iter()
            .map(|p| NdLayout::contiguous(&p.shape).map_err(TensorError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let sources = layouts.iter().zip(&values).collect::<Vec<_>>();
        let capture = if sources.is_empty() {
            None
        } else {
            Some(PreparedCapture::new(&g.device, &sources, &flags)?)
        };
        Ok(GraphGradientAccumulator {
            state: self.autograd.parameters.clone(),
            device: g.device.clone(),
            values,
            flags,
            capture,
            terms: 0,
            last_source: None,
        })
    }

    /// Explicitly start a new window on this learner's current parameter state.
    /// No buffer is reallocated or read back; the first add overwrites old sums.
    pub fn zero_accumulator(
        &self,
        accumulator: &mut GraphGradientAccumulator,
    ) -> Result<(), TrainingError> {
        if !Shared::ptr_eq(
            &self.autograd.parameters.workspace,
            &accumulator.state.workspace,
        ) {
            return Err(TrainingError::AccumulatorState);
        }
        accumulator.state = self.autograd.parameters.clone();
        accumulator.terms = 0;
        accumulator.last_source = None;
        Ok(())
    }

    /// Accumulate captured VJPs across inputs/forwards at one parameter state.
    /// Zero weight still carries source validity; overflow/cancellation cannot
    /// clear a failure. CPU-known errors leave the window and GPU queue untouched.
    pub fn accumulate(
        &mut self,
        accumulator: &mut GraphGradientAccumulator,
        gradients: &GraphGradients,
        weight: f32,
    ) -> Result<u64, TrainingError> {
        if !weight.is_finite() {
            return Err(TrainingError::GradientWeight);
        }
        if !self.autograd.parameters.matches(&accumulator.state)
            || !accumulator.state.matches(&gradients.forward.parameters)
        {
            return Err(TrainingError::AccumulatorState);
        }
        let terms = accumulator
            .terms
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let g = &self.autograd.graph;
        let context = g.device.runtime().context();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let weights = self.composition.encode_into(
            g,
            &[(gradients, weight)],
            &accumulator.values,
            (!accumulator.is_empty()).then_some(&accumulator.flags),
            &mut encoder,
        );
        self.composition
            .capture_flags(&mut encoder, &accumulator.flags);
        self.composition.write_weights(context.queue(), &weights);
        context.queue().submit(Some(encoder.finish()));
        accumulator.terms = terms;
        accumulator.last_source =
            Some((gradients.input_generation(), gradients.submitted_forward()));
        Ok(terms)
    }

    /// Apply the explicit gain policy and transactional SGD once to the sum.
    /// Every attempted update invalidates this window, even zero-rate or rejected
    /// updates; zero_accumulator is required before collecting fresh gradients.
    pub fn sgd_accumulated(
        &mut self,
        accumulator: &GraphGradientAccumulator,
        rate: f32,
    ) -> Result<u64, TrainingError> {
        if !rate.is_finite() || rate < 0. {
            return Err(TrainingError::LearningRate);
        }
        if !self.autograd.parameters.matches(&accumulator.state) {
            return Err(TrainingError::AccumulatorState);
        }
        let source = accumulator
            .last_source
            .ok_or(TrainingError::EmptyAccumulator)?;
        let attempt = self.updates.checked_add(1).ok_or(TrainingError::Overflow)?;
        let g = &self.autograd.graph;
        let mut encoder = g
            .device
            .runtime()
            .context()
            .device()
            .create_command_encoder(&Default::default());
        encoder.clear_buffer(&g.validation, 0, None);
        for (source, output) in accumulator.values.iter().zip(&g.raw_gradients) {
            encoder.copy_buffer_to_buffer(source, 0, output, 0, output.size());
        }
        encoder.copy_buffer_to_buffer(
            &accumulator.flags,
            0,
            &g.validation,
            (g.nodes.len() + 2) as u64 * 4,
            4,
        );
        self.submit_update(encoder, rate, attempt, source);
        Ok(attempt)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
