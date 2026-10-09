//! Compose whole-operation validity without copying or changing tensor values.
use super::*;

impl TensorDevice {
    /// Return zero-copy aliases whose validity is the conjunction of all inputs.
    /// Shapes may differ. Existing handles retain their original validity, while
    /// every returned view inherits any failure in the group. Flags are frozen
    /// on the owning queue; no value copy, host readback or arithmetic is added.
    pub fn guard_together(
        &self,
        tensors: &[&ResidentTensor],
    ) -> Result<Vec<ResidentTensor>, TensorError> {
        let context = self.runtime().context();
        for tensor in tensors {
            tensor.require_context(context)?;
        }
        if tensors.len() <= 1 {
            return Ok(tensors.iter().map(|tensor| (*tensor).clone()).collect());
        }
        let gpu = context.device();
        storage_limit(tensors.len(), &gpu.limits())?;
        let inherited = runtime::empty_buffer::<u32>(
            gpu,
            "tensor.group.inherited",
            tensors.len(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        let validation = Shared::new(runtime::empty_buffer::<u32>(
            gpu,
            "tensor.group.validity",
            1,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?);
        let capture = self
            .0
            .guard_capture
            .get_or_init(|| guard_capture::GuardCapture::new(gpu));
        let binding = capture.bind(gpu, &inherited, &validation);
        let mut encoder = gpu.create_command_encoder(&Default::default());
        for (i, tensor) in tensors.iter().enumerate() {
            encoder.copy_buffer_to_buffer(tensor.flags(), 0, &inherited, i as u64 * 4, 4);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            capture.encode_in_pass(&mut pass, &binding);
        }
        context.queue().submit(Some(encoder.finish()));
        Ok(tensors
            .iter()
            .map(|tensor| ResidentTensor {
                validation: Some(validation.clone()),
                ..(*tensor).clone()
            })
            .collect())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
