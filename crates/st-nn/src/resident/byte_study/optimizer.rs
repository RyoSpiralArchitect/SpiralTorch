//! Orthogonal optimizer recipes; geometry designs keep their existing contracts.
use super::*;
use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep};

pub(super) const REQUEST_SCHEMA: &str = "spiraltorch.byte_corpus.request.v9";

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub(super) enum Optimizer {
    #[serde(rename = "stateless_sgd.v1")]
    Sgd {},
    #[serde(rename = "adamw_zero_rate_freeze.v1")]
    AdamW {
        #[serde(deserialize_with = "recipe_scalar")]
        beta1: f32,
        #[serde(deserialize_with = "recipe_scalar")]
        beta2: f32,
        #[serde(deserialize_with = "recipe_scalar")]
        epsilon: f32,
        #[serde(deserialize_with = "recipe_scalar")]
        weight_decay: f32,
        #[serde(deserialize_with = "recipe_scalar")]
        geometry_weight_decay: f32,
    },
}

pub(super) fn recipe_scalar<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<f32, D::Error> {
    let value = f64::deserialize(d)?;
    let narrowed = value as f32;
    if !value.is_finite() || (value != 0. && !narrowed.is_normal()) {
        return Err(serde::de::Error::custom(
            "training recipe scalar must be zero or normal float32",
        ));
    }
    Ok(narrowed)
}

pub(super) fn explicit_optimizer<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<Optimizer>, D::Error> {
    Optimizer::deserialize(d).map(Some)
}

pub(super) fn explicit_design<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<String>, D::Error> {
    String::deserialize(d).map(Some)
}

impl Optimizer {
    pub(super) fn config(self) -> Result<Option<AdamWConfig>> {
        Ok(match self {
            Self::Sgd {} => None,
            Self::AdamW {
                beta1,
                beta2,
                epsilon,
                ..
            } => Some(AdamWConfig::new(beta1, beta2, epsilon)?),
        })
    }

    fn decay(self, name: &str) -> f32 {
        match self {
            Self::Sgd {} => 0.,
            Self::AdamW {
                weight_decay,
                geometry_weight_decay,
                ..
            } => {
                if name.starts_with("geometry.") {
                    geometry_weight_decay
                } else {
                    weight_decay
                }
            }
        }
    }

    #[cfg(feature = "wgpu")]
    pub(super) fn steps(self, case: &Case, rate: f32) -> Result<Vec<AdamWStep>> {
        case.parameters
            .iter()
            .map(|p| {
                Ok(AdamWStep::new(
                    if case.parameter_frozen(&p.name) {
                        0.
                    } else {
                        rate
                    },
                    self.decay(&p.name),
                )?)
            })
            .collect()
    }

    pub(super) fn validate_model(
        self,
        case: &Case,
        model: &ByteDecoderCheckpoint,
        cursor: u32,
    ) -> Result<()> {
        match (self.config()?, model.adamw_state()) {
            (None, None) => Ok(()),
            (Some(config), Some(state))
                if config == state.config && state.slots.len() == case.parameters.len() =>
            {
                for (slot, parameter) in state.slots.iter().zip(&case.parameters) {
                    let expected = if case.parameter_frozen(&parameter.name) {
                        0
                    } else {
                        cursor
                    };
                    if slot.steps != expected {
                        return Err(
                            "optimizer accepted clock differs from study/freeze policy".into()
                        );
                    }
                }
                Ok(())
            }
            _ => Err("checkpoint optimizer/configuration differs from request".into()),
        }
    }
}

impl Request {
    pub(super) fn design_schema(&self) -> &str {
        self.design_schema.as_deref().unwrap_or(&self.schema)
    }

    pub(super) fn optimizer(&self) -> Optimizer {
        self.optimizer.unwrap_or(Optimizer::Sgd {})
    }

    pub(super) fn initial_checkpoint(&self, case: &Case) -> Result<ByteDecoderCheckpoint> {
        let initial = plan(&self.config, case)?.initial_checkpoint();
        Ok(match self.optimizer().config()? {
            Some(config) => initial.with_zero_adamw(config)?,
            None => initial,
        })
    }
}

pub(super) fn validate_spec(r: &Request) -> Result<()> {
    if r.schema == REQUEST_SCHEMA || r.schema == clipping::REQUEST_SCHEMA {
        if r.optimizer.is_none()
            || !r.design_schema.as_deref().is_some_and(|s| {
                matches!(
                    s,
                    "spiraltorch.byte_corpus.request.v1"
                        | "spiraltorch.byte_corpus.request.v2"
                        | "spiraltorch.byte_corpus.request.v3"
                        | "spiraltorch.byte_corpus.request.v4"
                        | "spiraltorch.byte_corpus.request.v5"
                        | "spiraltorch.byte_corpus.request.v6"
                        | "spiraltorch.byte_corpus.request.v7"
                        | "spiraltorch.byte_corpus.request.v8"
                )
            })
        {
            return Err(
                "v9/v10 require an explicit optimizer and v1-v8 geometry design schema".into(),
            );
        }
    } else if r.optimizer.is_some() || r.design_schema.is_some() {
        return Err("optimizer/design metadata belongs to v9/v10 only".into());
    }
    if r.optimizer().config()?.is_some() {
        for name in ["backbone", "geometry.parameter"] {
            AdamWStep::new(r.rate, r.optimizer().decay(name))?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) mod tests;
