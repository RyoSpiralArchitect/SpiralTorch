use super::*;
use st_backend_wgpu::resident_tensor::{
    causal_wave::ResidentCausalWaveForward,
    euclidean::{ResidentEuclideanBiasForward, ResidentPairBiasVjp},
    poincare::ResidentPoincareBiasForward,
};

enum MetricTape {
    Poincare(ResidentPoincareBiasForward),
    Euclidean(ResidentEuclideanBiasForward),
}
impl MetricTape {
    fn scores(&self) -> &ResidentTensor {
        match self {
            Self::Poincare(f) => f.scores(),
            Self::Euclidean(f) => f.scores(),
        }
    }
    fn backward(&self, seed: &ResidentTensor) -> Result<ResidentPairBiasVjp, GpuTensorError> {
        match self {
            Self::Poincare(f) => f.backward(seed),
            Self::Euclidean(f) => f.backward(seed),
        }
    }
}

pub(super) struct GeometryTape {
    projection: GraphForward,
    wave: ResidentCausalWaveForward,
    metrics: Vec<MetricTape>,
}

impl GeometryTape {
    pub(super) fn bias(&self, block: usize) -> &ResidentTensor {
        self.metrics[block].scores()
    }
    pub(super) fn get_bias(&self, block: usize) -> Option<&ResidentTensor> {
        self.metrics.get(block).map(MetricTape::scores)
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
    metric: ByteDecoderPairMetric,
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
            metric: plan.pair_metric(),
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
            .map(|gain| match self.metric {
                ByteDecoderPairMetric::PoincareSquared => wave
                    .features()
                    .causal_poincare_bias(gain, self.curvature)
                    .map(MetricTape::Poincare),
                ByteDecoderPairMetric::EuclideanChordSquared => wave
                    .features()
                    .causal_euclidean_bias(gain, 4.)
                    .map(MetricTape::Euclidean),
            })
            .collect::<Result<Vec<_>, _>>()?;
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
        for (metric, bias) in tape.metrics.iter().zip(biases) {
            let seed = bias.pair_bias().ok_or(TrainingError::ParameterLayout)?;
            let vjp = metric.backward(seed)?;
            coordinates = Some(match coordinates {
                Some(sum) => sum.add(vjp.coordinates())?,
                None => vjp.coordinates().clone(),
            });
            gains.push(vjp.raw_gain().clone());
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
        Ok(GeometryVjp {
            input: projection.input_gradient().clone(),
            parameters,
        })
    }
}
