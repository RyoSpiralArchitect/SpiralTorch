use super::*;
use st_kernel_contracts::adamw::{AdamWConfig, AdamWSlot};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AdamWRecord {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    pub(super) slots: Vec<SlotRecord>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SlotRecord {
    shape: Vec<u32>,
    pub(super) first: Vec<f32>,
    pub(super) second: Vec<f32>,
    steps: u32,
}

pub(super) fn explicit_optimizer<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<AdamWRecord>, D::Error> {
    AdamWRecord::deserialize(d).map(Some)
}

impl ByteDecoderPlan {
    pub(in crate::resident::byte_decoder) fn parameter_shapes(
        &self,
    ) -> Result<Vec<Vec<usize>>, InferenceError> {
        let mut shapes = vec![self.token.shape.clone(), self.position.shape.clone()];
        if let Some(g) = &self.geometry {
            shapes.extend(g.parameter_values()?.into_iter().map(|p| p.shape));
        }
        for b in &self.blocks {
            shapes.extend(b.parameter_values()?.into_iter().map(|p| p.shape));
        }
        shapes.extend(
            self.head
                .graph_definition()?
                .parameters()
                .iter()
                .map(|p| p.shape.clone()),
        );
        Ok(shapes)
    }
}

impl AdamWRecord {
    pub(super) fn from_state(state: &AdamWState) -> Result<Self, InferenceError> {
        let slots = state
            .slots
            .iter()
            .map(|s| {
                Ok(SlotRecord {
                    shape: s
                        .shape
                        .iter()
                        .map(|&d| {
                            u32::try_from(d).map_err(|_| InferenceError::PortableAddressSpace)
                        })
                        .collect::<Result<_, _>>()?,
                    first: s.first.clone(),
                    second: s.second.clone(),
                    steps: s.steps,
                })
            })
            .collect::<Result<_, InferenceError>>()?;
        Ok(Self {
            beta1: state.config.beta1(),
            beta2: state.config.beta2(),
            epsilon: state.config.epsilon(),
            slots,
        })
    }

    pub(super) fn into_state(
        self,
        plan: &ByteDecoderPlan,
        revision: u64,
    ) -> Result<AdamWState, InferenceError> {
        let state = AdamWState {
            config: AdamWConfig::new(self.beta1, self.beta2, self.epsilon)?,
            slots: self
                .slots
                .into_iter()
                .map(|s| AdamWSlot {
                    shape: s.shape.into_iter().map(|d| d as usize).collect(),
                    first: s.first,
                    second: s.second,
                    steps: s.steps,
                })
                .collect(),
        };
        state.validate(&plan.parameter_shapes()?, revision)?;
        Ok(state)
    }
}
