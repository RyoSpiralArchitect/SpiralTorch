//! Rebind parameter values from a resident model owner without host round trips.
use super::*;

impl ResidentGraphAutograd {
    /// Replace all weights with same-device, same-shape immutable GPU values.
    /// A successful submission invalidates the current forward. Host validation
    /// errors preserve it; inherited failures guard every subsequent forward/VJP.
    /// This is a GPU-to-GPU copy into the compiled graph's existing bindings,
    /// not optimizer-state transfer or proof of numerical acceptance.
    pub fn set_parameter_tensors(
        &mut self,
        values: &[ResidentTensor],
    ) -> Result<(), TrainingError> {
        let g = &self.graph;
        let context = g.device.runtime().context();
        if values.len() != g.parameters.len() {
            return Err(TrainingError::ParameterLayout);
        }
        for (value, parameter) in values.iter().zip(g.definition.parameters()) {
            value.require_context(context)?;
            if value.layout().shape() != parameter.shape {
                return Err(TrainingError::ParameterLayout);
            }
        }
        let revision = self
            .parameters
            .revision
            .checked_add(1)
            .ok_or(TrainingError::Overflow)?;
        let gpu = context.device();
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let guard = if values.is_empty() {
            None
        } else {
            let flags = runtime::empty_buffer::<u32>(
                gpu,
                "graph.parameter_guards",
                values.len(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )?;
            let guard = runtime::empty_buffer::<u32>(
                gpu,
                "graph.parameter_guard",
                1,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            )?;
            for (i, (value, destination)) in values.iter().zip(&g.parameters).enumerate() {
                let packed = value.contiguous_into(&mut encoder)?;
                encoder.copy_buffer_to_buffer(
                    packed.values(),
                    0,
                    destination,
                    0,
                    destination.size(),
                );
                encoder.copy_buffer_to_buffer(packed.flags(), 0, &flags, i as u64 * 4, 4);
            }
            let capture = self
                .parameter_capture
                .get_or_insert_with(|| GuardCapture::new(gpu));
            let binding = capture.bind(gpu, &flags, &guard);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                capture.encode_in_pass(&mut pass, &binding);
            }
            Some(guard)
        };
        context.queue().submit(Some(encoder.finish()));
        self.parameter_guard = guard;
        self.parameters.revision = revision;
        self.current = None;
        Ok(())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
