use super::*;
use readback::{loss_and_flags, values, RawSnapshot};

#[derive(Debug)]
pub struct GraphState {
    pub loss: f32,
    pub submitted_step: u64,
    pub batch_generation: u64,
    pub gradient_policy: GraphGradientPolicy,
    pub prediction: Vec<f32>,
    pub input_gradient: Vec<f32>,
    /// Post-update graph with unchanged parameter IDs, roles and topology.
    pub graph: GraphDefinition,
    /// Mathematical VJPs, before any explicit module-compatible scaling.
    pub raw_gradients: Vec<Vec<f32>>,
    pub effective_gradients: Vec<Vec<f32>>,
}

pub struct GraphStateReadback {
    pub(super) raw: RawSnapshot,
    pub(super) definition: GraphDefinition,
    pub(super) policy: GraphGradientPolicy,
    pub(super) step: u64,
    pub(super) batch_generation: u64,
}
impl GraphStateReadback {
    pub fn input_layout(&self) -> &NdLayout {
        self.definition.input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.definition.output_layout()
    }
    pub fn submitted_step(&self) -> u64 {
        self.step
    }
    pub fn batch_generation(&self) -> u64 {
        self.batch_generation
    }
    pub fn gradient_policy(&self) -> GraphGradientPolicy {
        self.policy
    }

    fn decode(
        bytes: &[u8],
        definition: GraphDefinition,
        policy: GraphGradientPolicy,
        step: u64,
        batch_generation: u64,
    ) -> Result<GraphState, TrainingError> {
        let (loss, mut offset) = loss_and_flags(bytes, definition.stages().len() + 2)?;
        let prediction = values(bytes, &mut offset, definition.output_layout().len())?;
        let input_gradient = values(bytes, &mut offset, definition.input_layout().len())?;
        let mut parameters = Vec::new();
        let mut raw_gradients = Vec::new();
        let mut effective_gradients = Vec::new();
        for p in definition.parameters() {
            parameters.push(values(bytes, &mut offset, p.values.len())?);
            raw_gradients.push(values(bytes, &mut offset, p.values.len())?);
            effective_gradients.push(values(bytes, &mut offset, p.values.len())?);
        }
        if offset != bytes.len() {
            return Err(TrainingError::InvalidReadback);
        }
        Ok(GraphState {
            loss,
            submitted_step: step,
            batch_generation,
            gradient_policy: policy,
            prediction,
            input_gradient,
            graph: definition.with_values(parameters)?,
            raw_gradients,
            effective_gradients,
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<GraphState, TrainingError> {
        Self::decode(
            &self.raw.read()?,
            self.definition,
            self.policy,
            self.step,
            self.batch_generation,
        )
    }
    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<GraphState, TrainingError> {
        Self::decode(
            &self.raw.read_async().await?,
            self.definition,
            self.policy,
            self.step,
            self.batch_generation,
        )
    }
}

pub struct GraphParameterReadback {
    pub(super) raw: Option<RawSnapshot>,
    pub(super) definition: GraphDefinition,
}
impl GraphParameterReadback {
    fn decode(bytes: &[u8], definition: GraphDefinition) -> Result<GraphDefinition, TrainingError> {
        let mut offset = 0;
        let parameters = definition
            .parameters()
            .iter()
            .map(|p| values(bytes, &mut offset, p.values.len()))
            .collect::<Result<Vec<_>, _>>()?;
        if offset != bytes.len() {
            return Err(TrainingError::InvalidReadback);
        }
        Ok(definition.with_values(parameters)?)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<GraphDefinition, TrainingError> {
        let bytes = match self.raw {
            Some(raw) => raw.read()?,
            None => vec![],
        };
        Self::decode(&bytes, self.definition)
    }
    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<GraphDefinition, TrainingError> {
        let bytes = match self.raw {
            Some(raw) => raw.read_async().await?,
            None => vec![],
        };
        Self::decode(&bytes, self.definition)
    }
}
