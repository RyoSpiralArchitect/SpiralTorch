//! Owning VJP destinations; shared intermediate scratch never escapes the tape.
use super::*;
use crate::resident_tensor::{
    capture::{allocate_whole_outputs, retention_limit, whole_outputs_exclusively_owned},
    guard_capture::GuardCapture,
    pointwise::vjp::VjpOutputBindings,
};

enum OutputBindings {
    Linear {
        weight: Pass,
        bias: Pass,
        input: Option<Pass>,
    },
    Pointwise(VjpOutputBindings),
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

struct Version {
    tensors: Vec<ResidentTensor>,
    bindings: Vec<OutputBindings>,
    guard: wgpu::BindGroup,
}

pub(super) struct GradientOutputs {
    layouts: Vec<NdLayout>,
    versions: Vec<Version>,
    limit: usize,
    guard: GuardCapture,
    #[cfg(test)]
    allocations: usize,
    #[cfg(test)]
    reuses: usize,
}

impl GradientOutputs {
    pub(super) fn new(graph: &ResidentGraphTraining) -> Result<Self, TrainingError> {
        let layouts = std::iter::once(Ok(graph.input_layout().clone()))
            .chain(
                graph
                    .definition
                    .parameters()
                    .iter()
                    .map(|p| NdLayout::contiguous(&p.shape)),
            )
            .collect::<Result<Vec<_>, _>>()
            .map_err(TensorError::from)?;
        Ok(Self {
            limit: retention_limit(layouts.iter().map(NdLayout::len)),
            layouts,
            versions: Vec::new(),
            guard: GuardCapture::new(graph.device.runtime().context().device()),
            #[cfg(test)]
            allocations: 0,
            #[cfg(test)]
            reuses: 0,
        })
    }

    fn allocate(&self, g: &ResidentGraphTraining) -> Result<Version, TrainingError> {
        let tensors =
            allocate_whole_outputs(&g.device, &self.layouts, wgpu::BufferUsages::COPY_DST)?;
        let gpu = g.device.runtime().context().device();
        let resources = &g.backward_resources;
        let rows = g.input_layout().len() / g.input_layout().shape().last().unwrap();
        let mut bindings = Vec::with_capacity(g.nodes.len());
        for (i, (node, stage)) in g.nodes.iter().zip(g.definition.stages()).enumerate() {
            bindings.push(match (node, stage) {
                (
                    Node::Linear {
                        backward, delta, ..
                    },
                    GraphStage::Linear { weight, bias, .. },
                ) => {
                    let (w, b) = (*weight, *bias);
                    let shape = &g.definition.parameters()[w].shape;
                    let (k, n) = (shape[0], shape[1]);
                    let matrix = |index: usize, kind, shape, a, b, output| {
                        matrix_pass(
                            gpu,
                            &resources.matrix_layout,
                            &MatrixPipeline {
                                pipeline: backward[index].pipeline.clone(),
                                kind,
                            },
                            MatrixPass {
                                shape,
                                tile: resources.tile,
                                stage: i as u32,
                                operands: [
                                    a,
                                    b,
                                    output,
                                    &resources.unused_read,
                                    &resources.unused_read,
                                    &resources.unused_read,
                                ],
                                validation: &g.validation,
                                tape: &resources.unused_out,
                            },
                        )
                    };
                    let weight = matrix(
                        1,
                        TrainingMatmulKind::WeightGradient,
                        MatmulShape::new(k, rows, n)?,
                        &g.activations[i],
                        delta,
                        tensors[w + 1].values(),
                    )?;
                    let grid = backward[2].groups;
                    let uniform = runtime::upload_slice(
                        gpu,
                        "graph.vjp.bias.params",
                        &[Params {
                            rows: rows as u32,
                            cols: n as u32,
                            len: n as u32,
                            stage: i as u32,
                            stages: (g.nodes.len() + 2) as u32,
                            gelu: 0,
                            groups_x: grid[0],
                            partials: 1,
                        }],
                        wgpu::BufferUsages::UNIFORM,
                    )?;
                    let bias = Pass {
                        pipeline: backward[2].pipeline.clone(),
                        binding: binding(
                            gpu,
                            &resources.element_layout,
                            &[
                                delta,
                                &resources.unused_read,
                                &resources.unused_read,
                                &resources.unused_read,
                                tensors[b + 1].values(),
                                &resources.unused_aux,
                                &g.validation,
                                &uniform,
                                &g.step_config,
                            ],
                        ),
                        groups: grid,
                    };
                    let input = if i == 0 {
                        Some(matrix(
                            3,
                            TrainingMatmulKind::InputGradient,
                            MatmulShape::new(rows, n, k)?,
                            delta,
                            &g.parameters[w],
                            tensors[0].values(),
                        )?)
                    } else {
                        None
                    };
                    OutputBindings::Linear {
                        weight,
                        bias,
                        input,
                    }
                }
                (
                    Node::Pointwise {
                        plan,
                        workspace,
                        parameters,
                        ..
                    },
                    GraphStage::Pointwise { .. },
                ) => {
                    let destinations: Vec<_> = std::iter::once(if i == 0 {
                        tensors[0].values()
                    } else {
                        &g.gradients[i]
                    })
                    .chain(parameters.iter().map(|&id| tensors[id + 1].values()))
                    .collect();
                    OutputBindings::Pointwise(plan.bind_outputs(
                        workspace,
                        &destinations,
                        &g.pointwise_flags,
                    ))
                }
                _ => unreachable!("validated graph node kinds"),
            });
        }
        let guard = self.guard.bind(gpu, &g.validation, tensors[0].flags());
        Ok(Version {
            tensors,
            bindings,
            guard,
        })
    }

    /// The caller must retain the returned tensors until this encoder is submitted
    /// on the owning queue. No output version is reused while any member or view
    /// is observable. Oversized/busy versions spill without waiting or aliasing.
    pub(super) fn encode(
        &mut self,
        g: &ResidentGraphTraining,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<Vec<ResidentTensor>, TrainingError> {
        let version = if let Some(index) = self
            .versions
            .iter_mut()
            .position(|v| whole_outputs_exclusively_owned(&mut v.tensors))
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
        for (i, (node, outputs)) in g.nodes.iter().zip(&version.bindings).enumerate().rev() {
            match (node, outputs) {
                (
                    Node::Linear { backward, .. },
                    OutputBindings::Linear {
                        weight,
                        bias,
                        input,
                    },
                ) => {
                    let passes = [
                        &backward[0],
                        weight,
                        bias,
                        input.as_ref().unwrap_or(&backward[3]),
                    ];
                    for chunk in
                        passes.chunks(dispatches_per_pass(g.adapter_info().backend, passes.len()))
                    {
                        let mut compute =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("graph.autograd.backward"),
                                timestamp_writes: None,
                            });
                        for pass in chunk {
                            pass.encode(&mut compute);
                        }
                    }
                }
                (
                    Node::Pointwise {
                        plan,
                        workspace,
                        parameters,
                        ..
                    },
                    OutputBindings::Pointwise(outputs),
                ) => {
                    let destinations = std::iter::once(if i == 0 {
                        version.tensors[0].values()
                    } else {
                        &g.gradients[i]
                    })
                    .chain(
                        parameters
                            .iter()
                            .map(|&id| version.tensors[id + 1].values()),
                    );
                    plan.encode_prepared_to(encoder, workspace, outputs, destinations);
                }
                _ => unreachable!("prepared gradient output bindings"),
            }
        }
        // Every dense output and every pointwise contribution/reduction is
        // checked by its producer. Preserve forward and seed failures too, then
        // freeze one whole-VJP guard, without re-reading/copying the values.
        encoder.copy_buffer_to_buffer(
            &g.pointwise_flags,
            0,
            &g.validation,
            (g.nodes.len() + 3) as u64 * 4,
            4,
        );
        {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("graph.autograd.guard"),
                timestamp_writes: None,
            });
            self.guard.encode_in_pass(&mut compute, &version.guard);
        }
        let tensors = version.tensors.clone();
        if self.versions.len() < self.limit {
            self.versions.push(version);
        }
        Ok(tensors)
    }
}
