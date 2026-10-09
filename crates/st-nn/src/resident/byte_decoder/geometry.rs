//! A causal metric encoder, owned and updated by the complete byte model.
use super::*;
use st_kernel_contracts::{causal_wave::CausalWaveSpec, poincare::PoincareBiasSpec};

/// Pair metric only: projection, causal wave, chart and parameter ownership stay
/// identical. The flat control uses 4*||x-y||^2, the origin-local Poincare scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ByteDecoderPairMetric {
    #[serde(rename = "poincare_squared.v1")]
    PoincareSquared,
    #[serde(rename = "euclidean_chord_squared.v1")]
    EuclideanChordSquared,
}

/// Projection parameters, two wave vectors and one head-gain vector per block.
/// These are absolute slots in the byte model, not a separate optimizer owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteDecoderGeometryParameterLayout {
    projection: Range<usize>,
    raw_decay: usize,
    raw_phase: usize,
    raw_gains: Range<usize>,
}

impl ByteDecoderGeometryParameterLayout {
    pub fn projection(&self) -> Range<usize> {
        self.projection.clone()
    }
    pub fn raw_decay(&self) -> usize {
        self.raw_decay
    }
    pub fn raw_phase(&self) -> usize {
        self.raw_phase
    }
    pub fn raw_gains(&self) -> Range<usize> {
        self.raw_gains.clone()
    }
    pub fn all(&self) -> Range<usize> {
        self.projection.start..self.raw_gains.end
    }
}

/// Tokenwise projection -> causal wave -> bounded chart -> selected pair bias.
/// Each document/window starts at zero state. Curvature is frozen; decay,
/// phase and per-head softplus gains are learned. Raw gain zero is not "off".
#[derive(Clone, Debug)]
pub struct ByteDecoderGeometryPlan {
    pub(super) projection: InferencePlan,
    pub(super) wave: CausalWaveSpec,
    raw_decay: Vec<f32>,
    raw_phase: Vec<f32>,
    raw_gains: Vec<Vec<f32>>,
    metric: ByteDecoderPairMetric,
}

impl ByteDecoderGeometryPlan {
    pub fn new(
        projection: &InferencePlan,
        raw_decay: &[f32],
        raw_phase: &[f32],
        raw_gains: &[Vec<f32>],
        curvature: f32,
    ) -> Result<Self, InferenceError> {
        let shape = <[usize; 3]>::try_from(projection.output_layout().shape())
            .map_err(|_| InferenceError::InvalidLayout)?;
        let input = projection.input_layout().shape();
        if input.len() != 3 || input[..2] != shape[..2] {
            return Err(InferenceError::InvalidLayout);
        }
        // This graph contract admits only tokenwise stages, not global DFTs
        // or sequence-mixing reshapes that could leak the suffix into a prefix.
        projection.graph_definition()?;
        let wave = CausalWaveSpec::new(shape, curvature)?;
        wave.validate_lengths(
            wave.len(),
            raw_decay.len(),
            raw_phase.len(),
            wave.state_len(),
        )?;
        if raw_gains.is_empty()
            || raw_decay
                .iter()
                .chain(raw_phase)
                .chain(raw_gains.iter().flatten())
                .any(|v| !v.is_finite())
        {
            return Err(InferenceError::ByteDecoder(
                "geometry requires finite parameters and a gain vector per block",
            ));
        }
        for gain in raw_gains {
            PoincareBiasSpec::new(shape, gain.len(), curvature)?;
        }
        Ok(Self {
            projection: projection.clone(),
            wave,
            raw_decay: raw_decay.to_vec(),
            raw_phase: raw_phase.to_vec(),
            raw_gains: raw_gains.to_vec(),
            metric: ByteDecoderPairMetric::PoincareSquared,
        })
    }

    pub fn projection(&self) -> &InferencePlan {
        &self.projection
    }
    pub fn pair_metric(&self) -> ByteDecoderPairMetric {
        self.metric
    }
    /// Select a metric without changing initial values or trainable scalar count.
    /// The flat metric retains the nonlinear bounded chart, not a linear encoder.
    pub fn with_pair_metric(mut self, metric: ByteDecoderPairMetric) -> Self {
        self.metric = metric;
        self
    }
    pub fn curvature(&self) -> f32 {
        self.wave.curvature()
    }
    pub fn raw_decay(&self) -> &[f32] {
        &self.raw_decay
    }
    pub fn raw_phase(&self) -> &[f32] {
        &self.raw_phase
    }
    pub fn raw_gains(&self) -> &[Vec<f32>] {
        &self.raw_gains
    }

    fn replace_initial_raw_gains(&mut self, raw_gains: &[Vec<f32>]) -> Result<(), InferenceError> {
        if raw_gains.len() != self.raw_gains.len()
            || raw_gains
                .iter()
                .zip(&self.raw_gains)
                .any(|(new, old)| new.len() != old.len() || new.iter().any(|v| !v.is_finite()))
        {
            return Err(InferenceError::ByteDecoder(
                "initial geometry gains must preserve every block/head shape and be finite",
            ));
        }
        self.raw_gains = raw_gains.to_vec();
        Ok(())
    }

    #[cfg(feature = "wgpu")]
    pub(super) fn parameter_values(&self) -> Result<Vec<GraphParameter>, InferenceError> {
        let mut values = self.projection.graph_definition()?.parameters().to_vec();
        for (role, v) in [
            (ParameterRole::Gate, &self.raw_decay),
            (ParameterRole::Gate, &self.raw_phase),
        ]
        .into_iter()
        .chain(self.raw_gains.iter().map(|v| (ParameterRole::Gain, v)))
        {
            values.push(GraphParameter {
                role,
                shape: vec![v.len()],
                values: v.clone(),
            });
        }
        Ok(values)
    }
}

impl ByteDecoderPlan {
    pub fn causal_geometry(&self) -> Option<&ByteDecoderGeometryPlan> {
        self.geometry.as_ref()
    }

    /// Reinitialize gains in a frozen plan, e.g. after one-time bias calibration.
    /// Does not mutate a compiled owner or implement an optimizer update. All
    /// other parameters, metric, topology and ownership ranges stay unchanged.
    pub fn with_initial_geometry_raw_gains(
        mut self,
        raw_gains: &[Vec<f32>],
    ) -> Result<Self, InferenceError> {
        self.geometry
            .as_mut()
            .ok_or(InferenceError::ByteDecoder("model has no causal geometry"))?
            .replace_initial_raw_gains(raw_gains)?;
        Ok(self)
    }

    /// Add learned causal geometry without changing the ownership/SGD boundary.
    /// Omit this builder for the ordinary byte decoder; no fake zero-gain mode.
    pub fn with_causal_geometry(
        mut self,
        geometry: ByteDecoderGeometryPlan,
    ) -> Result<Self, InferenceError> {
        if self.geometry.is_some()
            || geometry.projection.input_layout() != self.input_layout()
            || geometry.raw_gains.len() != self.blocks.len()
            || self
                .blocks
                .iter()
                .zip(&geometry.raw_gains)
                .any(|(b, g)| b.attention_spec().query_shape()[1] != g.len())
        {
            return Err(InferenceError::ByteDecoder(
                "geometry input, block/head gains or existing configuration mismatch",
            ));
        }
        let decay = 2usize
            .checked_add(geometry.projection.graph_definition()?.parameters().len())
            .ok_or(InferenceError::PortableAddressSpace)?;
        let phase = decay
            .checked_add(1)
            .ok_or(InferenceError::PortableAddressSpace)?;
        let gain = phase
            .checked_add(1)
            .ok_or(InferenceError::PortableAddressSpace)?;
        let end = gain
            .checked_add(geometry.raw_gains.len())
            .ok_or(InferenceError::PortableAddressSpace)?;
        let shift = end - 2;
        for range in self
            .parameters
            .blocks
            .iter_mut()
            .chain([&mut self.parameters.head])
        {
            range.start = range
                .start
                .checked_add(shift)
                .ok_or(InferenceError::PortableAddressSpace)?;
            range.end = range
                .end
                .checked_add(shift)
                .ok_or(InferenceError::PortableAddressSpace)?;
        }
        self.parameters.geometry = Some(ByteDecoderGeometryParameterLayout {
            projection: 2..decay,
            raw_decay: decay,
            raw_phase: phase,
            raw_gains: gain..end,
        });
        self.geometry = Some(geometry);
        Ok(self)
    }
}
