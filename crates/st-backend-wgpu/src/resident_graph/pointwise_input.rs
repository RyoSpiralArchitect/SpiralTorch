//! Reuse the first pointwise program with immutable view addressing metadata.
use super::*;

impl ResidentGraph {
    pub(super) fn prepare_pointwise_input(
        &mut self,
        layout: &NdLayout,
    ) -> Result<(), GraphInferenceError> {
        if self
            .pointwise_input
            .as_ref()
            .is_none_or(|plan| plan.input_layout() != layout)
        {
            let Node::Pointwise { plan, .. } = &self.nodes[0] else {
                unreachable!("pointwise input requires a first pointwise stage");
            };
            self.pointwise_input = Some(runtime::Shared::new(plan.with_input_layout(layout)?));
        }
        Ok(())
    }

    pub(super) fn bind_pointwise_input(
        &self,
        input: &ResidentTensor,
        output: &wgpu::Buffer,
    ) -> BoundaryBinding {
        let GraphStage::Pointwise { parameters, .. } = &self.definition.stages()[0] else {
            unreachable!("pointwise input requires a first pointwise stage");
        };
        let plan = self.pointwise_input.as_ref().unwrap().clone();
        let inputs: Vec<_> = std::iter::once(input.values())
            .chain(parameters.iter().map(|&id| &self.parameters[id]))
            .collect();
        let binding = plan.bind_into(&inputs, output, &self.empty_flags, &self.validation);
        BoundaryBinding::ViewPointwise { plan, binding }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
