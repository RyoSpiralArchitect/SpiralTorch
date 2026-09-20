//! Versioned terminal producers, without a prediction-value capture/copy.
use super::*;
use crate::resident_tensor::{capture::retention_limit, guard_capture::GuardCapture};

struct Version {
    tensor: ResidentTensor,
    forward: ForwardBinding,
    guard: wgpu::BindGroup,
}

pub(super) struct PredictionOutputs {
    versions: Vec<Version>,
    limit: usize,
    guard: GuardCapture,
    #[cfg(test)]
    allocations: usize,
    #[cfg(test)]
    reuses: usize,
}

impl PredictionOutputs {
    pub(super) fn new(g: &ResidentGraphTraining) -> Self {
        Self {
            versions: Vec::new(),
            limit: retention_limit(std::iter::once(g.output_layout().len())),
            guard: GuardCapture::new(g.device.runtime().context().device()),
            #[cfg(test)]
            allocations: 0,
            #[cfg(test)]
            reuses: 0,
        }
    }

    fn allocate(&self, g: &ResidentGraphTraining) -> Result<Version, TrainingError> {
        let tensor = g.device.allocate_output(g.output_layout())?;
        let gpu = g.device.runtime().context().device();
        let resources = &g.stage_resources;
        let stage = g.nodes.len() - 1;
        let forward = match (&g.nodes[stage], &g.definition.stages()[stage]) {
            (
                Node::Linear {
                    forward,
                    preactivation,
                    ..
                },
                GraphStage::Linear { weight, bias, gelu },
            ) => {
                let shape = &g.definition.parameters()[*weight].shape;
                let rows = g.input_layout().len() / g.input_layout().shape().last().unwrap();
                ForwardBinding::Linear(matrix_pass(
                    gpu,
                    &resources.matrix_layout,
                    &MatrixPipeline {
                        pipeline: forward.pipeline.clone(),
                        kind: TrainingMatmulKind::Forward { gelu: *gelu },
                    },
                    MatrixPass {
                        shape: MatmulShape::new(rows, shape[0], shape[1])?,
                        tile: resources.tile,
                        stage: stage as u32,
                        operands: [
                            &g.activations[stage],
                            &g.parameters[*weight],
                            tensor.values(),
                            &g.parameters[*bias],
                            &resources.unused_read,
                            &resources.unused_read,
                        ],
                        validation: &g.validation,
                        tape: preactivation,
                    },
                )?)
            }
            (
                Node::Pointwise {
                    plan, parameters, ..
                },
                GraphStage::Pointwise { .. },
            ) => {
                let inputs: Vec<_> = std::iter::once(&g.activations[stage])
                    .chain(parameters.iter().map(|&id| &g.parameters[id]))
                    .collect();
                ForwardBinding::Pointwise(plan.forward().bind_into(
                    &inputs,
                    tensor.values(),
                    &resources.empty_flags,
                    &g.pointwise_flags,
                ))
            }
            _ => unreachable!("validated terminal stage"),
        };
        let guard = self.guard.bind(gpu, &g.validation, tensor.flags());
        Ok(Version {
            tensor,
            forward,
            guard,
        })
    }

    /// Retain the returned version until submission. Submitted consumers are
    /// queue-ordered; live tensor/view/weak owners prevent reuse entirely.
    pub(super) fn encode(
        &mut self,
        g: &ResidentGraphTraining,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<ResidentTensor, TrainingError> {
        let version = if let Some(index) = self
            .versions
            .iter_mut()
            .position(|v| v.tensor.exclusively_owned())
        {
            #[cfg(test)]
            {
                self.reuses += 1;
            }
            self.versions.swap_remove(index)
        } else {
            let version = self.allocate(g)?;
            #[cfg(test)]
            {
                self.allocations += 1;
            }
            version
        };
        g.encode_forward_to(encoder, &mut Default::default(), Some(&version.forward));
        encoder.copy_buffer_to_buffer(
            &g.pointwise_flags,
            0,
            &g.validation,
            (g.nodes.len() + 2) as u64 * 4,
            4,
        );
        {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("graph.autograd.prediction_guard"),
                timestamp_writes: None,
            });
            self.guard.encode_in_pass(&mut compute, &version.guard);
        }
        let tensor = version.tensor.clone();
        if self.versions.len() < self.limit {
            self.versions.push(version);
        }
        Ok(tensor)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
