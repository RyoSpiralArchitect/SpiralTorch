use super::*;
use st_backend_wgpu::resident_tensor::{
    causal_wave::ResidentCausalWaveForward,
    euclidean::{ResidentEuclideanBiasForward, ResidentPairBiasVjp},
    fisher_rao::ResidentFisherRaoBiasForward,
    mixture::ResidentHeadwiseMixture,
    poincare::ResidentPoincareBiasForward,
};

enum MetricTape {
    Poincare(ResidentPoincareBiasForward),
    Euclidean(ResidentEuclideanBiasForward),
    FisherRao(ResidentFisherRaoBiasForward),
}
impl MetricTape {
    fn forward(
        coordinates: &ResidentTensor,
        gain: &ResidentTensor,
        metric: ByteDecoderPairMetric,
        curvature: f32,
    ) -> Result<Self, GpuTensorError> {
        match metric {
            ByteDecoderPairMetric::PoincareSquared => coordinates
                .causal_poincare_bias(gain, curvature)
                .map(Self::Poincare),
            ByteDecoderPairMetric::EuclideanChordSquared => coordinates
                .causal_euclidean_bias(gain, 4.)
                .map(Self::Euclidean),
            ByteDecoderPairMetric::CategoricalFisherRaoSquared => coordinates
                .causal_fisher_rao_bias(gain)
                .map(Self::FisherRao),
        }
    }
    fn scores(&self) -> &ResidentTensor {
        match self {
            Self::Poincare(f) => f.scores(),
            Self::Euclidean(f) => f.scores(),
            Self::FisherRao(f) => f.scores(),
        }
    }
    fn backward(&self, seed: &ResidentTensor) -> Result<ResidentPairBiasVjp, GpuTensorError> {
        match self {
            Self::Poincare(f) => f.backward(seed),
            Self::Euclidean(f) => f.backward(seed),
            Self::FisherRao(f) => f.backward(seed),
        }
    }
}

// Consecutive equal heads share one kernel. A uniform block keeps the exact
// existing path; heterogeneous heads never require host activation readback.
fn head_groups(metrics: &[ByteDecoderPairMetric]) -> Vec<(Range<usize>, ByteDecoderPairMetric)> {
    let mut groups: Vec<(Range<usize>, ByteDecoderPairMetric)> = Vec::new();
    for (head, &metric) in metrics.iter().enumerate() {
        match groups.last_mut() {
            Some((range, previous)) if *previous == metric => range.end += 1,
            _ => groups.push((head..head + 1, metric)),
        }
    }
    groups
}

struct BlockMetricTape {
    groups: Vec<MetricTape>,
    scores: ResidentTensor,
    mixture: Option<(ResidentHeadwiseMixture, Vec<ResidentTensor>)>,
}

impl BlockMetricTape {
    fn backward(
        &self,
        seed: &ResidentTensor,
    ) -> Result<(ResidentTensor, ResidentTensor, Option<ResidentTensor>), GpuTensorError> {
        if let Some((mixture, scales)) = &self.mixture {
            let shape = seed.layout().shape();
            let mixed = mixture.backward(&seed.contiguous()?.reshape(&[
                shape[0],
                shape[1],
                shape[2] * shape[3],
            ])?)?;
            let mut coordinates: Option<ResidentTensor> = None;
            let mut gains: Option<ResidentTensor> = None;
            for (k, (group, scale)) in self.groups.iter().zip(scales).enumerate() {
                let cotangent = mixed
                    .components()
                    .select(3, k)?
                    .contiguous()?
                    .reshape(shape)?
                    .mul(scale)?;
                let g = group.backward(&cotangent)?;
                coordinates = Some(match coordinates {
                    Some(sum) => sum.add(g.coordinates())?,
                    None => g.coordinates().clone(),
                });
                gains = Some(match gains {
                    Some(sum) => sum.add(g.raw_gain())?,
                    None => g.raw_gain().clone(),
                });
            }
            return Ok((
                coordinates.expect("validated mixture components"),
                gains.expect("validated mixture components"),
                Some(mixed.logits().clone()),
            ));
        }
        if self.groups.len() == 1 {
            let vjp = self.groups[0].backward(seed)?;
            return Ok((vjp.coordinates().clone(), vjp.raw_gain().clone(), None));
        }
        let mut offset = 0;
        let mut coordinates: Option<ResidentTensor> = None;
        let mut gains = Vec::new();
        for group in &self.groups {
            let heads = group.scores().layout().shape()[1];
            let vjp = group.backward(&seed.narrow(1, offset, heads)?)?;
            coordinates = Some(match coordinates {
                Some(sum) => sum.add(vjp.coordinates())?,
                None => vjp.coordinates().clone(),
            });
            gains.push(vjp.raw_gain().clone());
            offset += heads;
        }
        Ok((
            coordinates.expect("a checked block has at least one head"),
            ResidentTensor::concatenate(&gains.iter().collect::<Vec<_>>(), 0)?,
            None,
        ))
    }
}

pub(super) struct GeometryTape {
    projection: GraphForward,
    wave: ResidentCausalWaveForward,
    metrics: Vec<BlockMetricTape>,
}

impl GeometryTape {
    pub(super) fn bias(&self, block: usize) -> &ResidentTensor {
        &self.metrics[block].scores
    }
    pub(super) fn get_bias(&self, block: usize) -> Option<&ResidentTensor> {
        self.metrics.get(block).map(|m| &m.scores)
    }
    pub(super) fn mixture_weights(&self, block: usize) -> Option<&ResidentTensor> {
        Some(self.metrics.get(block)?.mixture.as_ref()?.0.weights())
    }
}

pub(super) struct GeometryVjp {
    pub input: ResidentTensor,
    pub parameters: Vec<ResidentTensor>,
}

pub(super) struct GeometryAutograd {
    projection: ResidentGraphAutograd,
    layout: ByteDecoderGeometryParameterLayout,
    zero_state: ResidentTensor,
    curvature: f32,
    head_groups: Vec<Vec<(Range<usize>, ByteDecoderPairMetric)>>,
    mixture: Option<MixtureAutograd>,
}

struct MixtureAutograd {
    metrics: Vec<ByteDecoderPairMetric>,
    scales: Vec<Vec<ResidentTensor>>,
}

impl GeometryAutograd {
    pub(super) fn new(
        plan: &ByteDecoderGeometryPlan,
        layout: ByteDecoderGeometryParameterLayout,
        device: &TensorDevice,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<Self, InferenceError> {
        let [batch, _, cols] = plan.wave.shape();
        Ok(Self {
            projection: ResidentGraphAutograd::new(
                device.runtime().clone(),
                plan.projection.graph_definition()?,
                tile,
                kernel,
                accumulation,
            )?,
            layout,
            zero_state: device.upload(&[batch, cols], &vec![0.; plan.wave.state_len()])?,
            curvature: plan.curvature(),
            head_groups: plan
                .head_pair_metrics()
                .unwrap_or(&[])
                .iter()
                .map(|m| head_groups(m))
                .collect(),
            mixture: plan
                .mixture()
                .map(|m| {
                    let count = m.metrics().len();
                    let scales = m
                        .scales()
                        .iter()
                        .zip(plan.raw_gains())
                        .map(|(scales, gains)| {
                            (0..count)
                                .map(|k| {
                                    let values: Vec<_> =
                                        (0..gains.len()).map(|h| scales[h * count + k]).collect();
                                    device.upload(&[1, gains.len(), 1, 1], &values)
                                })
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok::<_, GpuTensorError>(MixtureAutograd {
                        metrics: m.metrics().to_vec(),
                        scales,
                    })
                })
                .transpose()?,
        })
    }

    pub(super) fn set_parameters(
        &mut self,
        values: &[ResidentTensor],
    ) -> Result<(), InferenceError> {
        self.projection
            .set_parameter_tensors(&values[self.layout.projection()])?;
        Ok(())
    }

    pub(super) fn forward(
        &mut self,
        input: &ResidentTensor,
        values: &[ResidentTensor],
    ) -> Result<GeometryTape, InferenceError> {
        self.projection.set_input_tensor(input)?;
        let projection = self.projection.forward()?;
        let wave = projection.prediction().causal_zspace_wave(
            &values[self.layout.raw_decay()],
            &values[self.layout.raw_phase()],
            &self.zero_state,
            self.curvature,
        )?;
        let metrics = values[self.layout.raw_gains()]
            .iter()
            .enumerate()
            .map(|(block, gain)| {
                if let Some(mixture) = &self.mixture {
                    let groups = mixture
                        .metrics
                        .iter()
                        .map(|metric| {
                            MetricTape::forward(wave.features(), gain, *metric, self.curvature)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let shape = groups[0].scores().layout().shape();
                    let parts = groups
                        .iter()
                        .zip(&mixture.scales[block])
                        .map(|(group, scale)| {
                            group.scores().mul(scale)?.reshape(&[
                                shape[0],
                                shape[1],
                                shape[2] * shape[3],
                                1,
                            ])
                        })
                        .collect::<Result<Vec<_>, GpuTensorError>>()?;
                    let components =
                        ResidentTensor::concatenate(&parts.iter().collect::<Vec<_>>(), 3)?;
                    let router = &values[self
                        .layout
                        .mixture_logits()
                        .expect("mixture owns logits")
                        .start
                        + block];
                    let forward = components.headwise_mixture(router)?;
                    let scores = forward.output().reshape(shape)?;
                    return Ok(BlockMetricTape {
                        groups,
                        scores,
                        mixture: Some((forward, mixture.scales[block].clone())),
                    });
                }
                let routing = &self.head_groups[block];
                let groups = routing
                    .iter()
                    .map(|(heads, metric)| {
                        let selected = if routing.len() == 1 {
                            gain.clone()
                        } else {
                            gain.narrow(0, heads.start, heads.len())?
                        };
                        MetricTape::forward(wave.features(), &selected, *metric, self.curvature)
                    })
                    .collect::<Result<Vec<_>, GpuTensorError>>()?;
                let scores = if groups.len() == 1 {
                    groups[0].scores().clone()
                } else {
                    ResidentTensor::concatenate(
                        &groups.iter().map(MetricTape::scores).collect::<Vec<_>>(),
                        1,
                    )?
                };
                Ok(BlockMetricTape {
                    groups,
                    scores,
                    mixture: None,
                })
            })
            .collect::<Result<Vec<_>, GpuTensorError>>()?;
        Ok(GeometryTape {
            projection,
            wave,
            metrics,
        })
    }

    pub(super) fn backward(
        &mut self,
        tape: &GeometryTape,
        biases: &[ByteDecoderBiasGradient],
    ) -> Result<GeometryVjp, InferenceError> {
        if biases.len() != tape.metrics.len() {
            return Err(TrainingError::ParameterLayout.into());
        }
        let mut coordinates: Option<ResidentTensor> = None;
        let mut gains = Vec::new();
        let mut routers = Vec::new();
        for (metric, bias) in tape.metrics.iter().zip(biases) {
            let seed = bias.pair_bias().ok_or(TrainingError::ParameterLayout)?;
            let (coordinate, gain, router) = metric.backward(seed)?;
            coordinates = Some(match coordinates {
                Some(sum) => sum.add(&coordinate)?,
                None => coordinate,
            });
            gains.push(gain);
            if let Some(router) = router {
                routers.push(router);
            }
        }
        // All consuming blocks contribute before recurrence BPTT. This model
        // resets at the window boundary, so its terminal-state cotangent is zero.
        let wave = tape.wave.backward(
            &coordinates.ok_or(TrainingError::ParameterLayout)?,
            &self.zero_state,
        )?;
        let projection = self.projection.backward(&tape.projection, wave.drive())?;
        let mut parameters = projection.parameter_gradients().to_vec();
        parameters.extend([wave.raw_decay().clone(), wave.raw_phase().clone()]);
        parameters.extend(gains);
        parameters.extend(routers);
        Ok(GeometryVjp {
            input: projection.input_gradient().clone(),
            parameters,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_preserve_order_and_only_coalesce_adjacent_equal_heads() {
        use ByteDecoderPairMetric::*;
        assert_eq!(
            head_groups(&[PoincareSquared; 4]),
            vec![(0..4, PoincareSquared)]
        );
        assert_eq!(
            head_groups(&[
                PoincareSquared,
                PoincareSquared,
                CategoricalFisherRaoSquared,
                EuclideanChordSquared,
                PoincareSquared
            ]),
            vec![
                (0..2, PoincareSquared),
                (2..3, CategoricalFisherRaoSquared),
                (3..4, EuclideanChordSquared),
                (4..5, PoincareSquared)
            ]
        );
    }
}
