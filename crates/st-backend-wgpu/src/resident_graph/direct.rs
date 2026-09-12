//! Bounded owning output versions and prepared direct graph boundaries.
use super::*;
use crate::resident_tensor::TensorReadback;

const MAX_OUTPUT_SLOTS: usize = 4;
const MAX_OUTPUT_BYTES: u64 = 32 * 1024 * 1024;

/// Bindings retain buffers, not tensor ownership. Only this private slot may
/// recycle them, and only when no tensor/view can still observe that version.
pub(super) struct OutputSlot {
    tensor: ResidentTensor,
    last: Option<BoundaryBinding>,
    guard: wgpu::BindGroup,
}

#[cfg(test)]
#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub(super) struct DirectStats {
    pub allocations: usize,
    pub reuses: usize,
    pub input_bindings: usize,
}

fn retention_limit(elements: usize) -> usize {
    (elements.max(1) as u64)
        .checked_add(1)
        .and_then(|n| n.checked_mul(4))
        .map_or(0, |bytes| {
            (MAX_OUTPUT_BYTES / bytes).min(MAX_OUTPUT_SLOTS as u64) as usize
        })
}

impl ResidentGraph {
    fn checkout_output(&mut self) -> Result<OutputSlot, GraphInferenceError> {
        if let Some(index) = self
            .output_slots
            .iter_mut()
            .position(|slot| slot.tensor.exclusively_owned())
        {
            #[cfg(test)]
            {
                self.direct_stats.reuses += 1;
            }
            return Ok(self.output_slots.swap_remove(index));
        }
        let gpu = self.device.runtime().context().device();
        let tensor = self.device.allocate_output(self.output_layout())?;
        let last = self.nodes.len() - 1;
        let last =
            (last != 0).then(|| self.bind_boundary(last, &self.activations[last], tensor.values()));
        let guard = self
            .guard_capture
            .get_or_insert_with(|| GuardCapture::new(gpu))
            .bind(gpu, &self.validation, tensor.flags());
        #[cfg(test)]
        {
            self.direct_stats.allocations += 1;
        }
        Ok(OutputSlot {
            tensor,
            last,
            guard,
        })
    }

    /// Submit the graph first, then capture its owning output version.
    /// Reading after later forwards or graph destruction never re-reads the
    /// current output. This preserves the ordinary forward scheduling.
    pub fn forward_tensor_snapshot(
        &mut self,
        input: &ResidentTensor,
    ) -> Result<TensorReadback, GraphInferenceError> {
        Ok(self.forward_tensor(input)?.snapshot()?)
    }

    /// Read resident input directly and write the last stage into an owning
    /// output version. Only completely unobserved output storage is recycled,
    /// within a four-slot / 32 MiB per-graph output-data budget. Busy/oversized
    /// slots never cause aliasing, waiting or CPU fallback: allocate separately.
    /// View packing, the graph and the output guard share one queue submission.
    /// The graph and its final guard capture also share one compute pass.
    pub fn forward_tensor(
        &mut self,
        input: &ResidentTensor,
    ) -> Result<ResidentTensor, GraphInferenceError> {
        input.require_context(self.device.runtime().context())?;
        if input.layout().shape() != self.input_layout().shape() {
            return Err(GraphInferenceError::InputShape);
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(GraphInferenceError::CounterOverflow)?;
        let dispatch = self
            .submitted_dispatches
            .checked_add(1)
            .ok_or(GraphInferenceError::CounterOverflow)?;
        let context = self.device.runtime().context().clone();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let packed = input.contiguous_into(&mut encoder)?;
        let slot = self.checkout_output()?;
        let singleton = self.nodes.len() == 1;
        let single_binding = if singleton {
            Some(self.bind_boundary(0, packed.values(), slot.tensor.values()))
        } else {
            if self.input_binding.is_none()
                || !self
                    .input_source
                    .as_ref()
                    .is_some_and(|source| source.shares_storage_with(&packed))
            {
                self.input_binding =
                    Some(self.bind_boundary(0, packed.values(), &self.activations[1]));
                #[cfg(test)]
                {
                    self.direct_stats.input_bindings += 1;
                }
            }
            None
        };
        let first = if singleton {
            single_binding.as_ref()
        } else {
            self.input_binding.as_ref()
        };
        self.encode_graph(
            &mut encoder,
            Some(&packed),
            first,
            slot.last.as_ref(),
            Some((self.guard_capture.as_ref().unwrap(), &slot.guard)),
        );
        context.queue().submit(Some(encoder.finish()));
        let output = slot.tensor.clone();
        self.generation = generation;
        self.submitted_dispatches = dispatch;
        self.output_generation = Some(generation);
        self.input_source = Some(packed);
        self.input_direct = true;
        self.resident_output = Some(output.clone());
        if self.output_slots.len() < retention_limit(self.output_layout().len()) {
            self.output_slots.push(slot);
        }
        Ok(output)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
