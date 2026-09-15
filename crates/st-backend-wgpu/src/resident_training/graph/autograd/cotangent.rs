//! Checked pointwise seeds written directly to the private graph tape.
use super::*;
use crate::resident_tensor::pointwise::PointwisePlan;

impl ResidentGraphAutograd {
    /// Evaluate a fused pointwise cotangent and the graph's exact VJP in one
    /// submission, without materializing/copying an intermediate seed tensor.
    ///
    /// This evaluates the plan as a seed, not as an objective to differentiate.
    /// Inputs must match the prepared layouts and owning device/queue; the seed
    /// shape must equal the graph output. Strides, offsets and broadcasts retain
    /// the ordinary pointwise contract. Host validation errors preserve the tape.
    /// GPU seed/forward failures guard every returned gradient, as in backward().
    pub fn backward_pointwise(
        &mut self,
        forward: &GraphForward,
        plan: &PointwisePlan,
        inputs: &[&ResidentTensor],
    ) -> Result<GraphGradients, TrainingError> {
        let submission = self.check_backward(forward)?;
        let g = &self.graph;
        let context = g.device.runtime().context();
        plan.require_output(context, g.output_layout())?;
        plan.validate_inputs(inputs)?;
        if self
            .cotangent_inherited
            .as_ref()
            .is_none_or(|buffer| buffer.size() < inputs.len() as u64 * 4)
        {
            self.cotangent_inherited = Some(runtime::empty_buffer::<u32>(
                context.device(),
                "graph.cotangent.inherited",
                inputs.len(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )?);
        }
        let inherited = self.cotangent_inherited.as_ref().unwrap();
        let mut encoder = self.begin_backward();
        // A later plan may use fewer slots. Never inherit an older seed failure.
        encoder.clear_buffer(inherited, 0, None);
        for (i, input) in inputs.iter().enumerate() {
            encoder.copy_buffer_to_buffer(input.flags(), 0, inherited, i as u64 * 4, 4);
        }
        plan.encode_into(
            &mut encoder,
            &inputs.iter().map(|t| t.values()).collect::<Vec<_>>(),
            g.gradients.last().unwrap(),
            inherited,
            &g.pointwise_flags,
        );
        encoder.copy_buffer_to_buffer(
            &g.pointwise_flags,
            0,
            &g.validation,
            g.nodes.len() as u64 * 4,
            4,
        );
        encoder.clear_buffer(&g.pointwise_flags, 0, None);
        self.submit_backward(forward, submission, encoder)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
